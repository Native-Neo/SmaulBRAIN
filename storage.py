"""Checkpoint storage: multi-file safetensors, atomic writes.

Layout (``ckpt_dir/``)::

    config.json          architecture + precision + paging configuration
    manifest.json        step, expert ids, usage/growth metadata (commit point),
                         plus a SHA-256 map of every payload file for
                         cross-file consistency checks on load
    resume_config.json   floor (min_experts), topology, scheduler snapshot —
                         the standalone pruner reads its floor from here
    meta.json            optimizer hparams + step count, RNG python state
    trunk.safetensors    embeddings, block, norms, head (dense BF16/FP32)
    router.safetensors   router projection (dense)
    optim.safetensors    SmaulOpt trunk + router moments (BF16/FP32 storage)
    rng.safetensors      torch RNG state (CPU + CUDA byte tensors)
    experts/<id>.safetensors   one file per expert: FP8 codes + FP32 scales,
                               expert-local optimizer moments
    experts/<id>.json          per-expert sidecar: dims, tiles, shapes,
                               optimizer structure/steps, metadata

No pickle anywhere: every tensor blob is safetensors, every non-tensor is
JSON, so checkpoints are cleanly extractable (e.g. for GGUF converters).

A crash mid-save cannot silently corrupt the model: every file is written
to a ``tmp_`` sidecar and atomically renamed; the manifest (written last,
also atomic) is the commit point — a loader only trusts experts listed
there — and its SHA-256 file map lets the loader refuse a mixed generation
(old manifest over new payloads or vice versa) loudly instead of training
on it. Single experts load/save without touching the rest of the model.

Per-expert sidecar rule: ``save_expert_file(rec, path)`` writes tensors to
``path`` and JSON meta to the sidecar derived from it — ``path`` with a
``.safetensors`` suffix swapped for ``.json``, otherwise ``path + ".json"``.
``load_expert_file`` resolves the sidecar the same way.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import random
import shutil
import tempfile

import torch
from safetensors.torch import load_file as _sf_load
from safetensors.torch import save_file as _sf_save

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    _fcntl = None

from experts import ExpertRecord, init_expert_optim_state
from precision import FP8BlockTensor

EXPERT_NAMES = ("w_gate", "w_up", "w_down")
SF_SUFFIX = ".safetensors"

RESUME_SCHEMA = "1"


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


def _fsync_file(path: str) -> None:
    """Flush one regular file's data+metadata to stable storage."""
    try:
        with open(path, "rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except OSError:
        pass


def _fsync_dir(dirpath: str) -> None:
    """Flush a directory entry table so a rename survives a crash."""
    try:
        fd = os.open(dirpath or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)


def _sweep_stale_tmp(directory: str) -> None:
    """Remove leftover atomic-write sidecars from a prior crash.

    Matches this module's tmp prefixes, the quantizer's ``tmp_quant_``
    prefix, staged generation dirs (``tmp_ckpt_gen_*``), and converter
    sidecars that may share the directory; best-effort so a concurrent
    writer never fails the sweep. Directories are removed recursively.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if name.startswith(("tmp_ckpt_", "tmp_json_", "tmp_quant_")) or name.endswith(".convert_tmp"):
            p = os.path.join(directory, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except OSError:
                pass


def _ckpt_lock_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, ".ckpt.lock")


@contextlib.contextmanager
def _ckpt_locked(ckpt_dir: str, exclusive: bool):
    """Inter-process checkpoint lock (best-effort flock).

    Writers hold an exclusive lock for the whole stage+publish sequence;
    readers hold a shared lock for the whole load. On hosts without
    ``fcntl`` this is a no-op. The lock file itself is never part of the
    checkpoint format and is ignored by the loader.
    """
    if _fcntl is None:
        yield
        return
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        fd = os.open(_ckpt_lock_path(ckpt_dir), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        yield
        return
    try:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH)
        except OSError:
            pass
        yield
    finally:
        try:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:
                pass
        finally:
            os.close(fd)


def _with_ckpt_lock(exclusive: bool):
    """Decorator: hold the checkpoint lock across save/load calls."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(ckpt_dir: str, *args, **kwargs):
            with _ckpt_locked(ckpt_dir, exclusive=exclusive):
                return fn(ckpt_dir, *args, **kwargs)
        return wrapper
    return deco


_TMP_GEN_PREFIX = "tmp_ckpt_gen_"


def _checked_sf_load(path: str) -> dict:
    """safetensors load with truncated-file detection normalized to ValueError."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise
    if st.st_size == 0:
        raise ValueError(f"checkpoint validation failed: empty file {path}")
    try:
        return _sf_load(path)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated file {path}: {e}"
        ) from e


def _detached_cpu_map(tensors: dict) -> dict:
    """Detach + CPU + contiguous copy map; clones storage-sharing tensors.

    safetensors refuses tensors that share memory (it would duplicate them
    on disk and reload them divergent), so any tensor whose storage was
    already emitted is cloned. Returns fresh tensors; inputs untouched.
    """
    out: dict = {}
    seen: set[int] = set()
    for k, t in tensors.items():
        c = t.detach().cpu().contiguous()
        ptr = c.untyped_storage().data_ptr()
        if ptr in seen:
            c = c.clone()
            ptr = c.untyped_storage().data_ptr()
        seen.add(ptr)
        out[k] = c
    return out


def _atomic_save_tensors(tensors: dict, path: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                        prefix="tmp_ckpt_")
    os.close(tmp_fd)
    try:
        _sf_save(_detached_cpu_map(tensors), tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, path)
        _fsync_dir(os.path.dirname(path) or ".")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_write_json(payload: dict, path: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                        prefix="tmp_json_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        _fsync_file(tmp_path)
        os.replace(tmp_path, path)
        _fsync_dir(os.path.dirname(path) or ".")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _sha256_file(path: str) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks (never fully in RAM)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable file {path}: {e}"
        ) from e
    return h.hexdigest()


def _manifest_rels(expert_ids: list[str]) -> list[str]:
    """Manifest-tracked relative paths: every file a generation consists of."""
    rels = ["config.json", "trunk.safetensors", "router.safetensors",
            "optim.safetensors", "rng.safetensors", "meta.json",
            "resume_config.json"]
    for eid in expert_ids:
        rels.append(f"experts/{eid}.safetensors")
        rels.append(f"experts/{eid}.json")
    return rels


def refresh_manifest_hashes(ckpt_dir: str) -> dict:
    """Recompute and atomically store the manifest's per-file SHA-256 map.

    Used after in-place precision conversion (quantize.py), which rewrites
    payload files without going through save_model. Rebuilds the map from
    the manifest's expert list so old (hash-less) checkpoints gain coverage
    too. Raises loudly on missing/unreadable files instead of certifying
    a partial generation.
    """
    manifest = load_manifest(ckpt_dir)
    eids = manifest.get("expert_ids", [])
    if not isinstance(eids, list) or not all(isinstance(e, str) for e in eids):
        raise ValueError("checkpoint validation failed: bad manifest expert_ids")
    files: dict[str, str] = {}
    for rel in _manifest_rels(eids):
        p = os.path.join(ckpt_dir, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"cannot hash missing checkpoint file {p}")
        files[rel] = _sha256_file(p)
    manifest["files"] = files
    _atomic_write_json(manifest, os.path.join(ckpt_dir, "manifest.json"))
    return manifest


def _verify_manifest_hashes(ckpt_dir: str, manifest: dict) -> None:
    """Refuse mixed/corrupt generations: every tracked file must hash-match.

    A crash between per-file publishes can leave an old manifest over new
    payloads (or vice versa); per-file atomicity alone cannot see that, but
    the hash map can. Manifests without a ``files`` map (pre-hash
    checkpoints) skip verification for backward compatibility.
    """
    files_map = manifest.get("files", {})
    if not isinstance(files_map, dict) or not files_map:
        return
    for rel, want in files_map.items():
        if not isinstance(rel, str) or not isinstance(want, str):
            raise ValueError(
                "checkpoint validation failed: bad manifest files map"
            )
        p = os.path.join(ckpt_dir, rel)
        if not os.path.exists(p):
            raise ValueError(
                f"checkpoint validation failed: tracked file missing {p} "
                "(mixed generation; refusing)"
            )
        got = _sha256_file(p)
        if got != want:
            raise ValueError(
                f"checkpoint validation failed: {rel} hash mismatch "
                "(mixed/corrupt generation; refusing to load)"
            )


def _opt_kind(st: dict) -> str:
    if "v" in st:
        return "full"
    return "factored"


def expert_tensors(rec: ExpertRecord) -> dict:
    """All of one expert's tensors: FP8 weights + local optimizer moments."""
    out: dict = {}
    for n in EXPERT_NAMES:
        t = rec.weights_fp8[n]
        out[f"{n}.codes"] = t.codes
        out[f"{n}.scales"] = t.scales
    for n, st in (rec.optim_state or {}).items():
        out[f"optim.{n}.m"] = st["m"]
        if _opt_kind(st) == "full":
            out[f"optim.{n}.v"] = st["v"]
        else:
            out[f"optim.{n}.v_row"] = st["v_row"]
            out[f"optim.{n}.v_col"] = st["v_col"]
    return out


def expert_meta(rec: ExpertRecord) -> dict:
    """All of one expert's non-tensor state (JSON-serializable)."""
    return {
        "expert_id": rec.expert_id,
        "d_model": rec.d_model,
        "expert_hidden": rec.expert_hidden,
        "weights": {
            n: {"shape": list(rec.weights_fp8[n].shape),
                "tile": rec.weights_fp8[n].tile}
            for n in EXPERT_NAMES
        },
        "optim": {
            n: {"kind": _opt_kind(st), "step": int(st.get("step", 0))}
            for n, st in (rec.optim_state or {}).items()
        },
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


def expert_from_parts(tensors: dict, meta: dict) -> ExpertRecord:
    """Rebuild an ExpertRecord from safetensors tensors + JSON meta."""
    weights = {}
    for n in EXPERT_NAMES:
        wm = meta["weights"][n]
        weights[n] = FP8BlockTensor(codes=tensors[f"{n}.codes"],
                                    scales=tensors[f"{n}.scales"],
                                    shape=tuple(wm["shape"]),
                                    tile=int(wm["tile"]))
    rec = ExpertRecord(
        expert_id=meta["expert_id"],
        d_model=int(meta["d_model"]),
        expert_hidden=int(meta["expert_hidden"]),
        weights_fp8=weights,
        birth_step=int(meta["meta"].get("birth_step", 0)),
        parents=list(meta["meta"].get("parents", [])),
        source=str(meta["meta"].get("source", "init")),
    )
    rec.tokens_routed = int(meta["meta"].get("tokens_routed", 0))
    rec.last_used_step = int(meta["meta"].get("last_used_step", 0))
    rec.grad_activity = float(meta["meta"].get("grad_activity", 0.0))
    rec.contribution = float(meta["meta"].get("contribution", 0.0))
    ost: dict = {}
    for n, om in meta.get("optim", {}).items():
        st: dict = {"m": tensors[f"optim.{n}.m"],
                    "step": int(om.get("step", 0))}
        if om.get("kind", "full") == "full":
            st["v"] = tensors[f"optim.{n}.v"]
        else:
            st["v_row"] = tensors[f"optim.{n}.v_row"]
            st["v_col"] = tensors[f"optim.{n}.v_col"]
        ost[n] = st
    rec.optim_state = ost or init_expert_optim_state(rec)
    return rec


def _expert_sidecar(path: str) -> str:
    if path.endswith(SF_SUFFIX):
        return path[: -len(SF_SUFFIX)] + ".json"
    return path + ".json"


def save_expert_file(rec: ExpertRecord, path: str) -> None:
    """Atomically save one expert (tensors + JSON sidecar).

    Does not sweep the target directory: callers (save_model, the
    quantizer) sweep explicitly, and sweeping here would delete staged
    ``*.convert_tmp`` sidecars sharing the directory.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    sidecar = _expert_sidecar(path)
    tmp_fd, tmp_tensors = tempfile.mkstemp(dir=parent, prefix="tmp_ckpt_")
    os.close(tmp_fd)
    try:
        _sf_save(_detached_cpu_map(expert_tensors(rec)), tmp_tensors)
        _fsync_file(tmp_tensors)
        os.replace(tmp_tensors, path)
        _atomic_write_json(expert_meta(rec), sidecar)
        _fsync_dir(parent)
    finally:
        if os.path.exists(tmp_tensors):
            os.remove(tmp_tensors)


def load_expert_file(path: str) -> ExpertRecord:
    """Load one expert written by :func:`save_expert_file`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing expert file {path}")
    sidecar = _expert_sidecar(path)
    if not os.path.exists(sidecar):
        raise FileNotFoundError(f"missing expert sidecar {sidecar}")
    tensors = _checked_sf_load(path)
    try:
        with open(sidecar) as f:
            meta = json.load(f)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable sidecar {sidecar}: {e}"
        ) from e
    return expert_from_parts(tensors, meta)


def _flatten_opt_store(store: dict, prefix: str) -> tuple[dict, dict]:
    """Flatten a SmaulOpt state store to safetensors keys + structure map."""
    tensors: dict = {}
    struct: dict = {}
    for k, st in store.items():
        tensors[f"{prefix}.{k}.m"] = st["m"]
        if _opt_kind(st) == "full":
            tensors[f"{prefix}.{k}.v"] = st["v"]
        else:
            tensors[f"{prefix}.{k}.v_row"] = st["v_row"]
            tensors[f"{prefix}.{k}.v_col"] = st["v_col"]
        struct[k] = {"kind": _opt_kind(st), "step": int(st.get("step", 0))}
    return tensors, struct


def _unflatten_opt_store(tensors: dict, struct: dict, prefix: str) -> dict:
    store: dict = {}
    for k, sm in struct.items():
        st: dict = {"m": tensors[f"{prefix}.{k}.m"],
                    "step": int(sm.get("step", 0))}
        if sm.get("kind", "full") == "full":
            st["v"] = tensors[f"{prefix}.{k}.v"]
        else:
            st["v_row"] = tensors[f"{prefix}.{k}.v_row"]
            st["v_col"] = tensors[f"{prefix}.{k}.v_col"]
        store[k] = st
    return store


def save_model(ckpt_dir: str, model, opt, step: int, extra_meta: dict | None = None) -> None:
    """Save the full model + optimizer. Manifest is written last (commit point).

    Filesystem transaction: every new file is first fully written and
    fsynced inside a temporary generation dir
    (``ckpt_dir/tmp_ckpt_gen_*`` on the same filesystem), then published
    into place with manifest last. A crash during staging leaves the
    previous complete generation untouched; only the fast rename phase
    runs in the live directory, and it holds an exclusive inter-process
    lock so concurrent readers (shared lock in :func:`load_model`) see
    either the previous or the new complete generation. Stale generation
    dirs and tmp sidecars from a prior crash are swept before staging.

    Cross-file guarantee (exact scope): each file is individually atomic
    (tmp sidecar + fsync + rename), and the manifest carries a SHA-256 map
    of every payload file that :func:`load_model` re-verifies — so a crash
    *between* per-file publishes cannot silently yield a mixed generation
    (it is refused loudly on the next load), though the interrupted save
    itself is lost and the previous complete generation remains live.

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
    with _ckpt_locked(ckpt_dir, exclusive=True):
        # Crash leftovers (tmp_ckpt_*/tmp_json_*/tmp_quant_*/.convert_tmp
        # plus staged generation dirs) must not accumulate: sweep them now
        # so the next load/save sees a clean dir.
        _sweep_stale_tmp(ckpt_dir)
        _sweep_stale_tmp(exp_dir)
        cfg_dict = model.cfg.to_dict()
        cfg_dict["num_experts"] = len(model.pool)
        from config import __version__ as _schema
        cfg_dict["schema_version"] = _schema
        # Pruning may reduce the pool below the configured floor; checkpoints
        # must remain constructible while still respecting top_k.
        cfg_dict["min_experts"] = min(model.cfg.min_experts, len(model.pool))
        extra = extra_meta or {}
        manifest = {
            "step": step,
            "expert_ids": list(model.pool.order),
            "next_id": model.pool._next_id,
            "paging_method": model.cfg.paging_method,
            "usage": model.pool.usage_snapshot(),
            "router_usage": model.router.usage_counts.tolist(),
            "router_admit": model.router.admit_counts.tolist(),
            "param_counts": model.param_counts(),
            "extra": extra,
        }
        resume_cfg = {
            "resume_schema": RESUME_SCHEMA,
            "step": step,
            "min_experts": model.cfg.min_experts,
            "max_experts": model.cfg.max_experts,
            "num_experts": len(model.pool),
            "next_id": model.pool._next_id,
            "topology": {
                "d_model": model.cfg.d_model,
                "n_heads": model.cfg.n_heads,
                "vocab_size": model.cfg.vocab_size,
                "expert_hidden": model.cfg.expert_hidden,
                "top_k": model.cfg.top_k,
                "dtype": model.cfg.dtype,
                "fp8_tile": model.cfg.fp8_tile,
                "paging_method": model.cfg.paging_method,
            },
            "scheduler": extra.get("scheduler", {}),
        }
        snap = get_rng_snapshot()
        trunk_tensors = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("router.") and "usage_" not in k and "admit_" not in k
                         and k not in ("router.proj.weight", "router.proj.bias")}
        router_tensors = {"weight": model.router.proj.weight,
                          "bias": model.router.proj.bias}
        optim_trunk_t, optim_trunk_s = _flatten_opt_store(opt.trunk_state, "trunk")
        optim_router_t, optim_router_s = _flatten_opt_store(opt.router_state, "router")
        optim_tensors = {**optim_trunk_t, **optim_router_t}
        meta = {
            "optim_hparams": dict(opt.hp.__dict__),
            "optim_step_count": int(opt.step_count),
            "optim_structure": {"trunk": optim_trunk_s, "router": optim_router_s},
            "rng_python": [snap["python"][0], list(snap["python"][1]),
                           snap["python"][2]],
        }
        rng_tensors = {"torch_cpu": snap["torch_cpu"]}
        if "cuda" in snap:
            for i, t in enumerate(snap["cuda"]):
                rng_tensors[f"cuda.{i}"] = t
        staging = tempfile.mkdtemp(dir=ckpt_dir, prefix=_TMP_GEN_PREFIX)
        try:
            st_exp = os.path.join(staging, "experts")
            os.makedirs(st_exp, exist_ok=True)
            _atomic_write_json(cfg_dict, os.path.join(staging, "config.json"))
            _atomic_save_tensors(trunk_tensors, os.path.join(staging, "trunk.safetensors"))
            _atomic_save_tensors(router_tensors, os.path.join(staging, "router.safetensors"))
            _atomic_save_tensors(optim_tensors, os.path.join(staging, "optim.safetensors"))
            _atomic_save_tensors(rng_tensors, os.path.join(staging, "rng.safetensors"))
            _atomic_write_json(meta, os.path.join(staging, "meta.json"))
            _atomic_write_json(resume_cfg, os.path.join(staging, "resume_config.json"))
            for eid in model.pool.order:
                rec = model.pool.experts[eid]
                _atomic_save_tensors(expert_tensors(rec),
                                     os.path.join(st_exp, f"{eid}.safetensors"))
                _atomic_write_json(expert_meta(rec),
                                   os.path.join(st_exp, f"{eid}.json"))
            # Cross-file consistency: hash every staged payload into the
            # manifest (written next, still pre-publish). The loader
            # re-verifies these hashes, so a crash between per-file
            # publishes — old manifest over new payloads or vice versa —
            # is refused loudly instead of silently training on a mixed
            # generation. Keys use "/" separators (portable, not os.sep).
            manifest["files"] = {
                rel: _sha256_file(os.path.join(staging, *rel.split("/")))
                for rel in _manifest_rels(list(model.pool.order))
            }
            _atomic_write_json(manifest, os.path.join(staging, "manifest.json"))
            _fsync_dir(st_exp)
            _fsync_dir(staging)
            for name in ("config.json", "trunk.safetensors", "router.safetensors",
                         "optim.safetensors", "rng.safetensors", "meta.json",
                         "resume_config.json"):
                os.replace(os.path.join(staging, name), os.path.join(ckpt_dir, name))
            for eid in model.pool.order:
                os.replace(os.path.join(st_exp, f"{eid}.safetensors"),
                           os.path.join(exp_dir, f"{eid}.safetensors"))
                os.replace(os.path.join(st_exp, f"{eid}.json"),
                           os.path.join(exp_dir, f"{eid}.json"))
            _fsync_dir(exp_dir)
            _fsync_dir(ckpt_dir)
            os.replace(os.path.join(staging, "manifest.json"),
                       os.path.join(ckpt_dir, "manifest.json"))
            _fsync_dir(ckpt_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        # The manifest is the commit point. Once it is safely published, remove
        # expert files no longer referenced by the new topology.
        live = {f"{eid}.safetensors" for eid in model.pool.order}
        live |= {f"{eid}.json" for eid in model.pool.order}
        try:
            names = os.listdir(exp_dir)
        except FileNotFoundError:
            names = []
        for name in names:
            if (name.endswith(".safetensors") or name.endswith(".json")) and name not in live:
                try:
                    os.remove(os.path.join(exp_dir, name))
                except OSError:
                    pass
        _fsync_dir(exp_dir)
        _fsync_dir(ckpt_dir)


def load_manifest(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint file {path}")
    try:
        if os.stat(path).st_size == 0:
            raise ValueError(
                f"checkpoint validation failed: empty file {path}"
            )
        with open(path) as f:
            return json.load(f)
    except ValueError as e:
        if "validation failed" in str(e):
            raise
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated file {path}: {e}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated file {path}: {e}"
        ) from e


def load_resume_config(ckpt_dir: str) -> dict:
    """Read the pruner/resume floor + topology snapshot (resume_config.json)."""
    path = os.path.join(ckpt_dir, "resume_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint file {path}")
    try:
        if os.stat(path).st_size == 0:
            raise ValueError(
                f"checkpoint validation failed: empty file {path}"
            )
        with open(path) as f:
            cfg = json.load(f)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated file {path}: {e}"
        ) from e
    _require(isinstance(cfg, dict), "resume_config.json is not a dict")
    return cfg


def make_disk_loader(ckpt_dir: str):
    """Pager loader reading single expert files (no full-model load)."""
    def load(expert_id: str) -> ExpertRecord:
        return load_expert_file(os.path.join(ckpt_dir, "experts", f"{expert_id}.safetensors"))
    return load


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"checkpoint validation failed: {msg}")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_expert_parts(tensors: dict, meta: dict, eid: str, saved_cfg) -> None:
    """Type/shape validation for one raw expert (tensors + JSON meta)."""
    _require(isinstance(meta, dict), f"{eid}: meta is not a dict")
    _require(meta.get("expert_id") == eid, f"{eid}: id mismatch {meta.get('expert_id')!r}")
    try:
        d_model = int(meta.get("d_model", -1))
        expert_hidden = int(meta.get("expert_hidden", -1))
    except (TypeError, ValueError):
        raise ValueError(
            f"checkpoint validation failed: {eid}: bad d_model/expert_hidden"
        ) from None
    _require(d_model == saved_cfg.d_model, f"{eid}: bad d_model")
    _require(expert_hidden == saved_cfg.expert_hidden,
             f"{eid}: bad expert_hidden")
    want = {
        "w_gate": (saved_cfg.expert_hidden, saved_cfg.d_model),
        "w_up": (saved_cfg.expert_hidden, saved_cfg.d_model),
        "w_down": (saved_cfg.d_model, saved_cfg.expert_hidden),
    }
    wmeta = meta.get("weights")
    _require(isinstance(wmeta, dict), f"{eid}: weights meta is not a dict")
    for n in EXPERT_NAMES:
        wm = wmeta.get(n)
        _require(isinstance(wm, dict), f"{eid}: {n} meta is not a dict")
        _require(tuple(wm.get("shape", ())) == want[n], f"{eid}: {n} bad shape")
        try:
            tile = int(wm.get("tile", 0))
        except (TypeError, ValueError):
            raise ValueError(
                f"checkpoint validation failed: {eid}: {n} bad tile"
            ) from None
        _require(tile >= 1, f"{eid}: {n} bad tile")
        codes, scales = tensors.get(f"{n}.codes"), tensors.get(f"{n}.scales")
        _require(torch.is_tensor(codes) and codes.dtype == torch.uint8,
                 f"{eid}: {n} codes must be uint8")
        _require(torch.is_tensor(scales) and scales.dtype == torch.float32,
                 f"{eid}: {n} scales must be float32")
        _require(tuple(codes.shape) == want[n], f"{eid}: {n} bad code shape")
        n_blocks = (want[n][1] + tile - 1) // tile
        _require(tuple(scales.shape) == (want[n][0], n_blocks),
                 f"{eid}: {n} bad scale shape")
    ost = meta.get("optim", {})
    _require(isinstance(ost, dict), f"{eid}: bad optim meta")
    for n, om in ost.items():
        _require(n in EXPERT_NAMES, f"{eid}: unknown optim entry {n!r}")
        _require(om.get("kind") in ("full", "factored"),
                 f"{eid}: optim {n} bad kind")
        try:
            s = int(om.get("step", -1))
        except (TypeError, ValueError):
            raise ValueError(
                f"checkpoint validation failed: {eid}: optim {n} bad step"
            ) from None
        _require(s >= 0, f"{eid}: optim {n} bad step")
        fields = ("m", "v") if om.get("kind") == "full" else ("m", "v_row", "v_col")
        st = {f: tensors.get(f"optim.{n}.{f}") for f in fields}
        st["step"] = s
        _validate_opt_entry(st, want[n], f"{eid}: optim {n}")
    m = meta.get("meta", {})
    _require(isinstance(m, dict), f"{eid}: bad meta")
    for k in ("birth_step", "last_used_step", "tokens_routed"):
        try:
            int(m.get(k, 0))
        except (TypeError, ValueError):
            raise ValueError(
                f"checkpoint validation failed: {eid}: bad meta {k}"
            ) from None
    for k in ("grad_activity", "contribution"):
        try:
            float(m.get(k, 0.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"checkpoint validation failed: {eid}: bad meta {k}"
            ) from None
    parents = m.get("parents", [])
    _require(isinstance(parents, list) and all(isinstance(x, str) for x in parents),
             f"{eid}: parents must be a list of str")
    _require(isinstance(m.get("source", "init"), str), f"{eid}: bad source")


def _validate_opt_entry(st: dict, want_shape: tuple | None, label: str) -> None:
    """Validate one SmaulOpt per-tensor state (m/v/step, pre-mutation)."""
    _require(isinstance(st, dict), f"{label} is not a dict")
    m = st.get("m")
    _require(torch.is_tensor(m) and m.dtype in (torch.bfloat16, torch.float32),
             f"{label} bad m (must be bf16/fp32 tensor)")
    if want_shape is not None:
        _require(tuple(m.shape) == tuple(want_shape), f"{label} bad m shape")
    try:
        step = int(st.get("step", -1))
    except (TypeError, ValueError):
        raise ValueError(
            f"checkpoint validation failed: {label} bad step"
        ) from None
    _require(step >= 0, f"{label} bad step")
    has_v = "v" in st
    has_factored = "v_row" in st or "v_col" in st
    _require(has_v != has_factored, f"{label} bad v structure")
    if has_v:
        v = st["v"]
        _require(torch.is_tensor(v) and v.dtype in (torch.bfloat16, torch.float32),
                 f"{label} bad v")
        _require(tuple(v.shape) == tuple(m.shape), f"{label} bad v shape")
    else:
        vr, vc = st["v_row"], st["v_col"]
        _require(torch.is_tensor(vr) and vr.dtype in (torch.bfloat16, torch.float32),
                 f"{label} bad v_row")
        _require(torch.is_tensor(vc) and vc.dtype in (torch.bfloat16, torch.float32),
                 f"{label} bad v_col")
        _require(tuple(vr.shape) == (m.shape[0], 1), f"{label} bad v_row shape")
        _require(tuple(vc.shape) == (1, m.shape[1]) if len(m.shape) == 2
                 else tuple(vc.shape) == tuple(m.shape),
                 f"{label} bad v_col shape")


def _validate_optim_store(store: dict, expected: dict, label: str) -> None:
    """Deep validation for a dense (trunk/router) optimizer store (pre-mutation)."""
    _require(isinstance(store, dict), f"{label} bad state dict")
    for k, st in store.items():
        _require(isinstance(k, str), f"{label} bad key {k!r}")
        if expected:
            _require(k in expected, f"{label} unexpected entry {k!r}")
        _validate_opt_entry(st, expected.get(k), f"{label} {k}")


def _validate_opt_hparams(saved_hp: dict) -> None:
    _require(isinstance(saved_hp, dict), "meta.json bad hparams")
    for k in ("lr", "beta_m", "beta_v", "eps", "wd", "clip", "update_clip"):
        if k in saved_hp:
            _require(_is_num(saved_hp[k]), f"meta.json bad hparam {k}")
    if "state_dtype" in saved_hp:
        _require(saved_hp["state_dtype"] in ("bf16", "fp32"),
                 "meta.json bad hparam state_dtype")
    if "factor_v" in saved_hp:
        _require(isinstance(saved_hp["factor_v"], bool),
                 "meta.json bad hparam factor_v")


def _validate_rng_parts(tensors: dict, py_state) -> None:
    cpu = tensors.get("torch_cpu")
    _require(torch.is_tensor(cpu) and cpu.dtype == torch.uint8,
             "rng snapshot missing torch_cpu state")
    for k, t in tensors.items():
        _require(torch.is_tensor(t) and t.dtype == torch.uint8,
                 f"rng snapshot bad entry {k}")
    if py_state is not None:
        _require(isinstance(py_state, (list, tuple)) and len(py_state) == 3,
                 "rng snapshot bad python state")


@_with_ckpt_lock(exclusive=False)
def load_model(ckpt_dir: str, model, opt) -> dict:
    """Load weights/opt/router/experts into an existing model+opt. Returns manifest.

    Validate-then-swap: every file is read and fully validated BEFORE any
    live object mutates, and the commit phase only swaps in prebuilt objects.
    A corrupt checkpoint raises with the live model untouched — never a
    partially restored model.

    Holds a shared inter-process lock for the whole read so a concurrent
    :func:`save_model` (exclusive lock, manifest-last publish) cannot
    interleave a mixed generation underneath the read.
    """
    from config import SmaulBrainConfig, __version__ as _schema

    def _path(*parts: str) -> str:
        p = os.path.join(ckpt_dir, *parts)
        _require(os.path.exists(p), f"missing checkpoint file {p}")
        try:
            _require(os.stat(p).st_size > 0, f"empty file {p}")
        except OSError as e:
            raise ValueError(f"checkpoint validation failed: unreadable file {p}: {e}") from e
        return p

    try:
        with open(_path("config.json")) as f:
            raw_cfg = json.load(f)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated config.json: {e}"
        ) from e
    _require(isinstance(raw_cfg, dict), "config.json is not a dict")
    saved_schema = str(raw_cfg.get("schema_version", _schema))
    _require(saved_schema.split(".")[0] == _schema.split(".")[0],
             f"schema major {saved_schema!r} != runtime {_schema!r}")
    try:
        saved_cfg = SmaulBrainConfig.from_dict(raw_cfg)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: bad config {e}"
        ) from e
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
    # Cross-file consistency before touching anything live: a mixed
    # generation (crash between publishes, partial quantize) fails here,
    # never as silent wrong-moment training later.
    _verify_manifest_hashes(ckpt_dir, manifest)
    saved_cfg.num_experts = len(eids)

    trunk = _checked_sf_load(_path("trunk.safetensors"))
    model_sd = model.state_dict()
    expected_trunk = {k for k in model_sd
                      if not (k.startswith("router.") or "usage_" in k or "admit_" in k)}
    for k in expected_trunk:
        # Pre-conv checkpoints lack byte_conv keys: skip the missing-key
        # check when the checkpoint itself declares conv off. The live
        # module is dropped after load (see below). The reverse direction
        # (conv checkpoint into conv-less model) still refuses via the
        # extra-key branch below.
        if k.startswith("byte_conv.") and not saved_cfg.use_byte_conv:
            continue
        _require(k in trunk, f"trunk.safetensors missing entry {k}")
    for k, v in trunk.items():
        _require(torch.is_tensor(v), f"trunk entry {k} is not a tensor")
        if k in model_sd:
            _require(tuple(v.shape) == tuple(model_sd[k].shape),
                     f"trunk tensor {k} shape mismatch")
        else:
            _require(k.startswith("router.") or "usage_" in k or "admit_" in k,
                     f"trunk entry {k} matches nothing in the model")
    router = _checked_sf_load(_path("router.safetensors"))
    rw, rb = router.get("weight"), router.get("bias")
    _require(torch.is_tensor(rw) and torch.is_floating_point(rw)
             and tuple(rw.shape) == (len(eids), saved_cfg.d_model),
             "router.safetensors weight must be [n_experts, d_model]")
    _require(torch.is_tensor(rb) and tuple(rb.shape) == (len(eids),),
             "router.safetensors bias must be [n_experts]")
    optim_t = _checked_sf_load(_path("optim.safetensors"))
    try:
        with open(_path("meta.json")) as f:
            meta = json.load(f)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated meta.json: {e}"
        ) from e
    _require(isinstance(meta, dict), "meta.json is not a dict")
    saved_hp = meta.get("optim_hparams", {})
    _validate_opt_hparams(saved_hp)
    try:
        step_count = int(meta.get("optim_step_count", 0))
    except (TypeError, ValueError):
        raise ValueError("checkpoint validation failed: bad optim step_count")
    _require(step_count >= 0, "bad optim step_count")
    ostruct = meta.get("optim_structure", {})
    _require(isinstance(ostruct, dict)
             and isinstance(ostruct.get("trunk"), dict)
             and isinstance(ostruct.get("router"), dict),
             "meta.json bad optim structure")
    trunk_opt = _unflatten_opt_store(optim_t, ostruct["trunk"], "trunk")
    router_opt = _unflatten_opt_store(optim_t, ostruct["router"], "router")
    _trunk_shapes = {n: tuple(p.shape) for n, p in model._trunk_params()}
    _validate_optim_store(trunk_opt, _trunk_shapes, "optim trunk")
    # Router width is checkpoint-authoritative (growth may widen the pool
    # beyond a fresh live model): validate against saved topology, not live.
    _router_shapes = {"router.proj.weight": (len(eids), saved_cfg.d_model),
                      "router.proj.bias": (len(eids),)}
    _validate_optim_store(router_opt, _router_shapes, "optim router")
    rng_t = _checked_sf_load(_path("rng.safetensors"))
    _validate_rng_parts(rng_t, meta.get("rng_python"))
    parts = []
    for eid in eids:
        t = _checked_sf_load(_path("experts", f"{eid}.safetensors"))
        sidecar = os.path.join(ckpt_dir, "experts", f"{eid}.json")
        _require(os.path.exists(sidecar), f"missing checkpoint file {sidecar}")
        try:
            with open(sidecar) as f:
                m = json.load(f)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"checkpoint validation failed: unreadable sidecar {sidecar}: {e}"
            ) from e
        _validate_expert_parts(t, m, eid, saved_cfg)
        parts.append((t, m))
    try:
        on_disk = {n for n in os.listdir(os.path.join(ckpt_dir, "experts"))
                   if n.endswith(".safetensors")}
    except OSError as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable experts dir: {e}"
        ) from e
    _require(on_disk == {f"{eid}.safetensors" for eid in eids},
             f"experts dir mismatch: disk {sorted(on_disk)} vs manifest {eids}")

    import torch.nn as nn
    new_experts: dict = {}
    new_order: list[str] = []
    for t, m in parts:
        rec = expert_from_parts(t, m)
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

    model.load_state_dict(trunk, strict=False)
    model.pool.experts = new_experts
    model.pool.order = new_order
    model.pool._next_id = int(next_id)
    model.router.proj = new_proj
    model.router.num_experts = len(eids)
    model.router.register_buffer("usage_counts", new_usage)
    model.router.register_buffer("admit_counts", new_admit)
    model.cfg = saved_cfg
    # Presence follows the checkpoint: byte-conv became default-on after
    # checkpoints without it, so a conv-built model loading a pre-conv
    # checkpoint must drop the module (strict=False silently skipped the
    # missing keys, leaving random weights that forward keys off the
    # module, not the flag). The reverse direction (conv checkpoint into a
    # conv-less model) still refuses loudly at trunk validation above.
    if not saved_cfg.use_byte_conv and getattr(model, "byte_conv", None) is not None:
        model.byte_conv = None
        model._conv_hist = None
    model.pager.mode = saved_cfg.paging_method
    model.pager.ram_cache = saved_cfg.ram_cache
    model.pager.vram_cache = saved_cfg.vram_cache
    model.pager.compute_dtype = (
        torch.bfloat16 if saved_cfg.dtype == "bf16" else torch.float32
    )
    opt.trunk_state = trunk_opt
    opt.router_state = router_opt
    opt.step_count = step_count
    # Optimizer hyperparameters are checkpoint-authoritative: resuming with
    # different betas/eps/clip/weight-decay would silently change the math.
    if saved_hp:
        for k, v in saved_hp.items():
            if hasattr(opt.hp, k):
                setattr(opt.hp, k, v)
    _rng_snap: dict = {"torch_cpu": rng_t["torch_cpu"]}
    if meta.get("rng_python"):
        _rng_snap["python"] = tuple(meta["rng_python"])
    _cuda_keys = sorted(k for k in rng_t if k.startswith("cuda."))
    if _cuda_keys:
        _rng_snap["cuda"] = [rng_t[k] for k in _cuda_keys]
    set_rng_snapshot(_rng_snap)
    # Scheduler snapshot (loss edge, growth counters, replay buffer) travels
    # in manifest extras; run_training picks it up for exact resume.
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
