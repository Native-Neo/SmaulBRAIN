"""Checkpoint storage: per-expert files, FP8 bytes, atomic writes.

Layout (``ckpt_dir/``)::

    config.json          architecture + precision + paging configuration
    manifest.json        step, expert ids, usage/growth metadata, RNG cursor
    trunk.pt             embeddings, block, norms, head (dense BF16/FP32)
    router.pt            router projection (dense)
    trunk_opt.pt         SmaulOpt trunk state (name-keyed, BF16 storage)
    router_opt.pt        SmaulOpt router state
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
import tempfile

import torch

from .experts import ExpertRecord, init_expert_optim_state
from .precision import FP8BlockTensor

EXPERT_NAMES = ("w_gate", "w_up", "w_down")


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
    """Save the full model + optimizer. Manifest is written last (commit point)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    exp_dir = os.path.join(ckpt_dir, "experts")
    os.makedirs(exp_dir, exist_ok=True)
    _atomic_write_json(model.cfg.to_dict(), os.path.join(ckpt_dir, "config.json"))
    _atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()
                  if not k.startswith("router.") and "usage_" not in k and "admit_" not in k
                  and k not in ("router.proj.weight", "router.proj.bias")},
                 os.path.join(ckpt_dir, "trunk.pt"))
    _atomic_save({"weight": model.router.proj.weight.detach().cpu(),
                  "bias": model.router.proj.bias.detach().cpu()},
                 os.path.join(ckpt_dir, "router.pt"))
    _atomic_save({"trunk": opt.trunk_state, "router": opt.router_state,
                  "step_count": opt.step_count}, os.path.join(ckpt_dir, "optim.pt"))
    _atomic_save(torch.get_rng_state(), os.path.join(ckpt_dir, "rng.pt"))
    for eid in model.pool.order:
        save_expert_file(model.pool.experts[eid], os.path.join(exp_dir, f"{eid}.pt"))
    manifest = {
        "step": step,
        "expert_ids": list(model.pool.order),
        "next_id": model.pool._next_id,
        "paging_method": model.cfg.paging_method,
        "usage": model.pool.usage_snapshot(),
        "router_usage": model.router.usage_counts.tolist(),
        "param_counts": model.param_counts(),
        "extra": extra_meta or {},
    }
    _atomic_write_json(manifest, os.path.join(ckpt_dir, "manifest.json"))


def load_manifest(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "manifest.json")) as f:
        return json.load(f)


def make_disk_loader(ckpt_dir: str):
    """Pager loader reading single expert files (no full-model load)."""
    def load(expert_id: str) -> ExpertRecord:
        return load_expert_file(os.path.join(ckpt_dir, "experts", f"{expert_id}.pt"))
    return load


def load_model(ckpt_dir: str, model, opt) -> dict:
    """Load weights/opt/router/experts into an existing model+opt. Returns manifest."""
    from .config import SmaulBrainConfig

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        saved_cfg = SmaulBrainConfig.from_dict(json.load(f))
    assert saved_cfg.d_model == model.cfg.d_model, "d_model mismatch"
    assert saved_cfg.vocab_size == model.cfg.vocab_size, "vocab mismatch"
    trunk = torch.load(os.path.join(ckpt_dir, "trunk.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(trunk, strict=False)
    router = torch.load(os.path.join(ckpt_dir, "router.pt"), map_location="cpu", weights_only=False)
    with torch.no_grad():
        # Resize router if expert count changed (growth/pruning is checkpoint-safe).
        n_saved = router["weight"].shape[0]
        while model.router.num_experts < n_saved:
            model.router.add_expert_row()
        model.router.proj.weight.copy_(router["weight"].to(model.router.proj.weight.dtype))
        model.router.proj.bias.copy_(router["bias"].to(model.router.proj.bias.dtype))
    optim = torch.load(os.path.join(ckpt_dir, "optim.pt"), map_location="cpu", weights_only=False)
    opt.trunk_state = optim["trunk"]
    opt.router_state = optim["router"]
    opt.step_count = int(optim.get("step_count", 0))
    manifest = load_manifest(ckpt_dir)
    # Rebuild pool exactly in manifest order (index-consistent after pruning).
    model.pool.experts.clear()
    model.pool.order.clear()
    for eid in manifest["expert_ids"]:
        rec = load_expert_file(os.path.join(ckpt_dir, "experts", f"{eid}.pt"))
        model.pool.experts[eid] = rec
        model.pool.order.append(eid)
    model.pool._next_id = int(manifest.get("next_id", len(model.pool.order)))
    while model.router.num_experts < len(model.pool):
        model.router.add_expert_row()
    # Restore router usage statistics (resize-safe; stats are FP64 buffers).
    try:
        saved_usage = manifest.get("router_usage", [])
        if len(saved_usage) == model.router.num_experts:
            model.router.usage_counts.copy_(torch.tensor(saved_usage, dtype=torch.float64))
    except Exception:
        pass
    rng_path = os.path.join(ckpt_dir, "rng.pt")
    if os.path.exists(rng_path):
        torch.set_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    if model.cfg.paging_method == "R2VR":
        model.pager.warm_ram()
    return manifest
