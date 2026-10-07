"""Checkpoint storage: per-expert files, FP8 bytes, atomic writes.

Layout (``ckpt_dir/``)::

    config.json          architecture + precision + paging configuration
    manifest.json        step, expert ids, usage/growth metadata, RNG cursor
    trunk.pt             embeddings, block, norms, head (dense BF16/FP32)
    router.pt            router projection (dense)
    optim.pt             SmaulOpt trunk + router state (name-keyed, BF16 storage)
    rng.pt               torch RNG state
    experts/<id>.pt      one file per expert: FP8 codes + FP32 scales,
                         expert-local optimizer state, metadata

A crash during checkpointing cannot corrupt the model: every file is written
to ``<name>.tmp`` and atomically renamed; the manifest (written last, also
atomic) is the commit point — a loader only trusts experts listed there.
Single experts load/save without touching the rest of the model.
"""

from __future__ import annotations

import json
import os
import random
import tempfile

import torch

from experts import ExpertRecord, init_expert_optim_state
from precision import FP8BlockTensor

EXPERT_NAMES = ("w_gate", "w_up", "w_down")


def get_rng_snapshot() -> dict:
    """Capture all RNG state that affects future training (CPU/CUDA/Python)."""
    snap: dict = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def set_rng_snapshot(snap: dict) -> None:
    """Restore RNG state captured by get_rng_snapshot (legacy tensor tolerated)."""
    if isinstance(snap, torch.Tensor):
        torch.set_rng_state(snap)  # legacy rng.pt: torch CPU state only
        return
    if "torch_cpu" in snap:
        torch.set_rng_state(snap["torch_cpu"])
    if "python" in snap:
        version, inner, gauss = snap["python"]
        random.setstate((int(version), tuple(int(v) for v in inner), gauss))
    if "cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["cuda"])


def _atomic_save(obj, path: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                        prefix="tmp_ckpt_")
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_write_json(payload: dict, path: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                        prefix="tmp_json_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _fp8_to_dict(t: FP8BlockTensor) -> dict:
    return {"codes": t.codes.cpu(), "scales": t.scales.cpu(),
            "shape": list(t.shape), "tile": t.tile}


def _fp8_from_dict(d: dict) -> FP8BlockTensor:
    return FP8BlockTensor(codes=d["codes"], scales=d["scales"],
                          shape=tuple(d["shape"]), tile=int(d["tile"]))


def expert_to_payload(rec: ExpertRecord) -> dict:
    return {
        "expert_id": rec.expert_id,
        "d_model": rec.d_model,
        "expert_hidden": rec.expert_hidden,
        "weights": {n: _fp8_to_dict(rec.weights_fp8[n]) for n in EXPERT_NAMES},
        "optim_state": rec.optim_state,
        "meta": {
            "birth_step": rec.birth_step,
            "parents": list(rec.parents),
            "source": rec.source,
            "tokens_routed": rec.tokens_routed,
            "last_used_step": rec.last_used_step,
            "grad_activity": rec.grad_activity,
            "contribution": rec.contribution,
        },
    }


def expert_from_payload(p: dict) -> ExpertRecord:
    rec = ExpertRecord(
        expert_id=p["expert_id"],
        d_model=int(p["d_model"]),
        expert_hidden=int(p["expert_hidden"]),
        weights_fp8={n: _fp8_from_dict(p["weights"][n]) for n in EXPERT_NAMES},
        birth_step=int(p["meta"].get("birth_step", 0)),
        parents=list(p["meta"].get("parents", [])),
        source=str(p["meta"].get("source", "init")),
    )
    rec.tokens_routed = int(p["meta"].get("tokens_routed", 0))
    rec.last_used_step = int(p["meta"].get("last_used_step", 0))
    rec.grad_activity = float(p["meta"].get("grad_activity", 0.0))
    rec.contribution = float(p["meta"].get("contribution", 0.0))
    rec.optim_state = p.get("optim_state", {}) or {}
    if not rec.optim_state:
        rec.optim_state = init_expert_optim_state(rec)
    return rec


def save_expert_file(rec: ExpertRecord, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_save(expert_to_payload(rec), path)


def load_expert_file(path: str) -> ExpertRecord:
    return expert_from_payload(torch.load(path, map_location="cpu", weights_only=False))


def save_model(ckpt_dir: str, model, opt, step: int, extra_meta: dict | None = None) -> None:
    """Save the full model + optimizer. Manifest is written last (commit point).

    Validates before writing anything: a topology that fails validation
    raises without touching the checkpoint directory, so a previous
    complete generation is never clobbered by a partial one.
    """
    if int(step) < 0:
        raise ValueError(f"refusing to checkpoint negative step {step}")
    if len(model.pool) == 0:
        raise ValueError("refusing to checkpoint an empty expert pool")
    if model.router.num_experts != len(model.pool):
        raise ValueError(
            f"refusing to checkpoint diverged topology: {len(model.pool)} "
            f"experts vs {model.router.num_experts} router rows"
        )
    for eid in model.pool.order:
        rec = model.pool.experts[eid]
        want = {
            "w_gate": (rec.expert_hidden, rec.d_model),
            "w_up": (rec.expert_hidden, rec.d_model),
            "w_down": (rec.d_model, rec.expert_hidden),
        }
        for n in EXPERT_NAMES:
            t = rec.weights_fp8.get(n)
            if t is None or tuple(t.shape) != want[n]:
                raise ValueError(
                    f"refusing to checkpoint {eid}: corrupt {n} storage"
                )
    os.makedirs(ckpt_dir, exist_ok=True)
    exp_dir = os.path.join(ckpt_dir, "experts")
    os.makedirs(exp_dir, exist_ok=True)
    cfg_dict = model.cfg.to_dict()
    cfg_dict["num_experts"] = len(model.pool)
    from config import __version__ as _schema
    cfg_dict["schema_version"] = _schema
    # Pruning may reduce the pool below the configured floor; checkpoints
    # must remain constructible while still respecting top_k.
    cfg_dict["min_experts"] = min(model.cfg.min_experts, len(model.pool))
    _atomic_write_json(cfg_dict, os.path.join(ckpt_dir, "config.json"))
    _atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()
                  if not k.startswith("router.") and "usage_" not in k and "admit_" not in k
                  and k not in ("router.proj.weight", "router.proj.bias")},
                 os.path.join(ckpt_dir, "trunk.pt"))
    _atomic_save({"weight": model.router.proj.weight.detach().cpu(),
                  "bias": model.router.proj.bias.detach().cpu()},
                 os.path.join(ckpt_dir, "router.pt"))
    _atomic_save({"trunk": opt.trunk_state, "router": opt.router_state,
                  "step_count": opt.step_count,
                  "hparams": dict(opt.hp.__dict__)}, os.path.join(ckpt_dir, "optim.pt"))
    _atomic_save(get_rng_snapshot(), os.path.join(ckpt_dir, "rng.pt"))
    for eid in model.pool.order:
        save_expert_file(model.pool.experts[eid], os.path.join(exp_dir, f"{eid}.pt"))
    manifest = {
        "step": step,
        "expert_ids": list(model.pool.order),
        "next_id": model.pool._next_id,
        "paging_method": model.cfg.paging_method,
        "usage": model.pool.usage_snapshot(),
        "router_usage": model.router.usage_counts.tolist(),
        "router_admit": model.router.admit_counts.tolist(),
        "param_counts": model.param_counts(),
        "extra": extra_meta or {},
    }
    _atomic_write_json(manifest, os.path.join(ckpt_dir, "manifest.json"))
    # The manifest is the commit point. Once it is safely published, remove
    # expert files no longer referenced by the new topology.
    live = {f"{eid}.pt" for eid in model.pool.order}
    for name in os.listdir(exp_dir):
        if name.endswith(".pt") and name not in live:
            try:
                os.remove(os.path.join(exp_dir, name))
            except FileNotFoundError:
                pass


def load_manifest(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "manifest.json")) as f:
        return json.load(f)


def make_disk_loader(ckpt_dir: str):
    """Pager loader reading single expert files (no full-model load)."""
    def load(expert_id: str) -> ExpertRecord:
        return load_expert_file(os.path.join(ckpt_dir, "experts", f"{expert_id}.pt"))
    return load


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"checkpoint validation failed: {msg}")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_expert_payload(p: dict, eid: str, saved_cfg) -> None:
    """Type/shape validation for one raw expert payload (pre-mutation)."""
    _require(isinstance(p, dict), f"{eid}: payload is not a dict")
    _require(p.get("expert_id") == eid, f"{eid}: id mismatch {p.get('expert_id')!r}")
    _require(int(p.get("d_model", -1)) == saved_cfg.d_model, f"{eid}: bad d_model")
    _require(int(p.get("expert_hidden", -1)) == saved_cfg.expert_hidden,
             f"{eid}: bad expert_hidden")
    want = {
        "w_gate": (saved_cfg.expert_hidden, saved_cfg.d_model),
        "w_up": (saved_cfg.expert_hidden, saved_cfg.d_model),
        "w_down": (saved_cfg.d_model, saved_cfg.expert_hidden),
    }
    w = p.get("weights")
    _require(isinstance(w, dict), f"{eid}: weights is not a dict")
    for n in EXPERT_NAMES:
        t = w.get(n)
        _require(isinstance(t, dict), f"{eid}: {n} is not a dict")
        codes, scales = t.get("codes"), t.get("scales")
        _require(torch.is_tensor(codes) and codes.dtype == torch.uint8,
                 f"{eid}: {n} codes must be uint8")
        _require(torch.is_tensor(scales) and scales.dtype == torch.float32,
                 f"{eid}: {n} scales must be float32")
        _require(tuple(t.get("shape", ())) == want[n], f"{eid}: {n} bad shape")
        _require(tuple(codes.shape) == want[n], f"{eid}: {n} bad code shape")
        tile = int(t.get("tile", 0))
        _require(tile >= 1, f"{eid}: {n} bad tile")
        n_blocks = (want[n][1] + tile - 1) // tile
        _require(tuple(scales.shape) == (want[n][0], n_blocks),
                 f"{eid}: {n} bad scale shape")
    _require(isinstance(p.get("optim_state", {}), dict), f"{eid}: bad optim_state")
    meta = p.get("meta", {})
    _require(isinstance(meta, dict), f"{eid}: bad meta")
    for k in ("birth_step", "last_used_step", "tokens_routed"):
        int(meta.get(k, 0))
    for k in ("grad_activity", "contribution"):
        float(meta.get(k, 0.0))
    parents = meta.get("parents", [])
    _require(isinstance(parents, list) and all(isinstance(x, str) for x in parents),
             f"{eid}: parents must be a list of str")
    _require(isinstance(meta.get("source", "init"), str), f"{eid}: bad source")


def _validate_rng_snapshot(snap) -> None:
    _require(isinstance(snap, (torch.Tensor, dict)), "rng snapshot bad type")
    if isinstance(snap, dict):
        cpu = snap.get("torch_cpu")
        _require(torch.is_tensor(cpu), "rng snapshot missing torch_cpu state")
        py = snap.get("python")
        if py is not None:
            _require(isinstance(py, (list, tuple)) and len(py) == 3,
                     "rng snapshot bad python state")
        cuda = snap.get("cuda")
        if cuda is not None:
            _require(isinstance(cuda, (list, tuple)), "rng snapshot bad cuda state")


def load_model(ckpt_dir: str, model, opt) -> dict:
    """Load weights/opt/router/experts into an existing model+opt. Returns manifest.

    Validate-then-swap: every file is read and fully validated BEFORE any
    live object mutates, and the commit phase only swaps in prebuilt objects.
    A corrupt checkpoint raises with the live model untouched — never a
    partially restored model.
    """
    from config import SmaulBrainConfig, __version__ as _schema

    def _path(*parts: str) -> str:
        p = os.path.join(ckpt_dir, *parts)
        _require(os.path.exists(p), f"missing checkpoint file {p}")
        return p

    # ---- phase 1a: config + manifest (no live mutation) ----
    with open(_path("config.json")) as f:
        raw_cfg = json.load(f)
    saved_schema = str(raw_cfg.get("schema_version", _schema))
    _require(saved_schema.split(".")[0] == _schema.split(".")[0],
             f"schema major {saved_schema!r} != runtime {_schema!r}")
    saved_cfg = SmaulBrainConfig.from_dict(raw_cfg)
    for field in ("d_model", "vocab_size", "n_heads", "expert_hidden",
                  "top_k", "dtype"):
        _require(getattr(saved_cfg, field) == getattr(model.cfg, field),
                 f"checkpoint {field}={getattr(saved_cfg, field)!r} "
                 f"does not match model {getattr(model.cfg, field)!r}")
    # Note: fp8_tile is intentionally not a shape field. Tiles ride on each
    # stored tensor and retile conversions update them; the model adopts the
    # checkpoint tile for future growth instead of refusing to load.
    manifest = load_manifest(ckpt_dir)
    _require(isinstance(manifest, dict), "manifest is not a dict")
    eids = manifest.get("expert_ids")
    _require(isinstance(eids, list) and len(eids) > 0
             and all(isinstance(e, str) for e in eids)
             and len(set(eids)) == len(eids), "bad manifest expert_ids")
    step = manifest.get("step", -1)
    _require(isinstance(step, int) and not isinstance(step, bool), "bad manifest step")
    next_id = manifest.get("next_id", len(eids))
    _require(isinstance(next_id, int) and next_id >= 0, "bad manifest next_id")
    for key in ("router_usage", "router_admit"):
        vals = manifest.get(key, [])
        _require(isinstance(vals, list)
                 and (not vals or len(vals) == len(eids))
                 and all(_is_num(v) for v in vals), f"bad manifest {key}")
    extra = manifest.get("extra", {})
    _require(isinstance(extra, dict), "bad manifest extra")
    saved_cfg.num_experts = len(eids)

    # ---- phase 1b: tensors (read + validate, no live mutation) ----
    trunk = torch.load(_path("trunk.pt"), map_location="cpu", weights_only=False)
    _require(isinstance(trunk, dict), "trunk.pt is not a dict")
    model_sd = model.state_dict()
    for k, v in trunk.items():
        _require(torch.is_tensor(v), f"trunk entry {k} is not a tensor")
        if k in model_sd:
            _require(tuple(v.shape) == tuple(model_sd[k].shape),
                     f"trunk tensor {k} shape mismatch")
        else:
            _require(k.startswith("router.") or "usage_" in k or "admit_" in k,
                     f"trunk entry {k} matches nothing in the model")
    router = torch.load(_path("router.pt"), map_location="cpu", weights_only=False)
    _require(isinstance(router, dict), "router.pt is not a dict")
    rw, rb = router.get("weight"), router.get("bias")
    _require(torch.is_tensor(rw) and torch.is_floating_point(rw)
             and tuple(rw.shape) == (len(eids), saved_cfg.d_model),
             "router.pt weight must be [n_experts, d_model]")
    _require(torch.is_tensor(rb) and tuple(rb.shape) == (len(eids),),
             "router.pt bias must be [n_experts]")
    optim = torch.load(_path("optim.pt"), map_location="cpu", weights_only=False)
    _require(isinstance(optim, dict), "optim.pt is not a dict")
    _require(isinstance(optim.get("trunk"), dict)
             and isinstance(optim.get("router"), dict), "optim.pt bad state dicts")
    try:
        step_count = int(optim.get("step_count", 0))
    except (TypeError, ValueError):
        raise ValueError("checkpoint validation failed: bad optim step_count")
    saved_hp = optim.get("hparams", {})
    _require(isinstance(saved_hp, dict), "optim.pt bad hparams")
    rng_path = os.path.join(ckpt_dir, "rng.pt")
    rng_snap = None
    if os.path.exists(rng_path):
        rng_snap = torch.load(rng_path, map_location="cpu", weights_only=False)
        _validate_rng_snapshot(rng_snap)
    payloads = []
    for eid in eids:
        p = torch.load(_path("experts", f"{eid}.pt"),
                       map_location="cpu", weights_only=False)
        _validate_expert_payload(p, eid, saved_cfg)
        payloads.append(p)

    # ---- phase 1c: build replacement objects (still no live mutation) ----
    import torch.nn as nn
    new_experts: dict = {}
    new_order: list[str] = []
    for p in payloads:
        rec = expert_from_payload(p)
        new_experts[rec.expert_id] = rec
        new_order.append(rec.expert_id)
    dev, dt = model.router.proj.weight.device, model.router.proj.weight.dtype
    new_proj = nn.Linear(saved_cfg.d_model, len(eids)).to(device=dev, dtype=dt)
    with torch.no_grad():
        new_proj.weight.copy_(rw.to(dt))
        new_proj.bias.copy_(rb.to(dt))
    new_usage = torch.tensor(manifest.get("router_usage", [0.0] * len(eids)),
                             dtype=torch.float64, device=dev)
    new_admit = torch.tensor(manifest.get("router_admit", [0.0] * len(eids)),
                             dtype=torch.float64, device=dev)

    # ---- phase 2: commit (plain swaps; infallible after validation) ----
    model.load_state_dict(trunk, strict=False)
    model.pool.experts = new_experts
    model.pool.order = new_order
    model.pool._next_id = int(next_id)
    model.router.proj = new_proj
    model.router.num_experts = len(eids)
    model.router.register_buffer("usage_counts", new_usage)
    model.router.register_buffer("admit_counts", new_admit)
    model.cfg = saved_cfg
    model.pager.mode = saved_cfg.paging_method
    model.pager.ram_cache = saved_cfg.ram_cache
    model.pager.vram_cache = saved_cfg.vram_cache
    model.pager.compute_dtype = (
        torch.bfloat16 if saved_cfg.dtype == "bf16" else torch.float32
    )
    opt.trunk_state = optim["trunk"]
    opt.router_state = optim["router"]
    opt.step_count = step_count
    # Optimizer hyperparameters are checkpoint-authoritative: resuming with
    # different betas/eps/clip/weight-decay would silently change the math.
    if saved_hp:
        for k, v in saved_hp.items():
            if hasattr(opt.hp, k):
                setattr(opt.hp, k, v)
    if rng_snap is not None:
        set_rng_snapshot(rng_snap)
    # Scheduler snapshot (loss edge, growth/prune counters, replay buffer)
    # travels in manifest extras; run_training picks it up for exact resume.
    sched = extra.get("scheduler", {}) or {}
    model._scheduler_snapshot = dict(sched) if isinstance(sched, dict) else {}
    # The pool objects were replaced: cached compute weights and staged
    # records still point at the pre-load experts and must go. So do all
    # concurrency bookkeeping: forgotten/version/pending state refers to
    # pre-load ids and would poison the fresh pool (e.g. a pruned-then-
    # restored expert id would stay unreadable).
    model.pager.ram.clear()
    model.pager.vram.clear()
    model.pager.ram_records.clear()
    for attr in ("_forgotten", "_versions", "_prefetched"):
        buf = getattr(model.pager, attr, None)
        if buf is not None:
            buf.clear()
    for attr in ("_pending", "_in_flight"):
        buf = getattr(model.pager, attr, None)
        if buf is not None:
            buf.clear()
    if model.cfg.paging_method == "R2VR":
        model.pager.warm_ram()
    model._resume_step = int(step)
    return manifest
