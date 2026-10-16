"""V2 checkpoint format (version smaulbrain-v2:1).

Directory layout:
  manifest.json   all metadata + scheduler state + pending/committed growth records
  params.npz      master parameters (incl. all experts)
  optim.npz       SmaulOPT state arrays

Publish: build in a temp dir next to the target, then swap it in. Replacing an
existing checkpoint moves the old tree to <dir>.bak for the swap; if the process
dies in between, the previous checkpoint is still loadable from <dir>.bak and
load_checkpoint() falls back to it. manifest records sha256 of the payloads.
"""
import hashlib
import json
import os
import tempfile

import numpy as np

from . import vocab as _vocab
from smaulbrain import CHECKPOINT_VERSION


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blank_scheduler_state(growth_every_steps=10000, growth_loss_nats=0.5,
                          loss_beta=0.98) -> dict:
    interval = int(growth_every_steps)
    nxt = interval if interval > 0 else None
    return {
        "global_step": 0,
        "growth_every_steps": interval,
        "next_periodic_step": nxt,
        "loss_threshold": growth_loss_nats,
        "loss_beta": float(loss_beta),
        "smooth_loss": None,
        "loss_armed": True,
        "event_count": 0,
        "last_growth_step": None,
        "history": [],
        "pending": None,
    }


def build_manifest(preset_name: str, arch: dict, extra: dict, sched: dict,
                   data_cursor: int, rng_state: dict, n_experts_current: int,
                   expert_registry: list, opt_meta: dict) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "preset": preset_name,
        "preset_expert_count_init": arch.get("experts_init"),
        "n_experts_current": n_experts_current,
        "expert_registry": expert_registry,
        "vocab": _vocab.vocab_meta(),
        "arch": arch,
        "config": extra,
        "scheduler": sched,
        "data_cursor": int(data_cursor),
        "rng_state": rng_state,
        "optimizer": opt_meta,
        "files": {},
    }


def _write_tree(target_dir: str, manifest: dict, params: dict, opt_state: dict) -> str:
    parent = os.path.dirname(os.path.abspath(target_dir))
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".ckpt.", dir=parent)
    try:
        params_path = os.path.join(tmp, "params.npz")
        np.savez(params_path, **{k: np.asarray(v) for k, v in params.items()})
        opt_flat = {}
        for k, s in opt_state.items():
            for sk, sv in s.items():
                opt_flat[f"{k}::{sk}"] = np.asarray(sv)
        opt_path = os.path.join(tmp, "optim.npz")
        np.savez(opt_path, **opt_flat)
        manifest = dict(manifest)
        manifest["files"] = {
            "params.npz": _sha256(params_path),
            "optim.npz": _sha256(opt_path),
        }
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=_json_default)
        _publish_dir(tmp, target_dir)
        return target_dir
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _publish_dir(tmp: str, target_dir: str) -> None:
    """Publish tmp as target_dir, keeping the previous version recoverable.

    A fresh target is published with a single atomic os.replace(). Replacing an
    existing tree needs two renames, so a crash in between leaves the previous
    checkpoint at <target_dir>.bak (load_checkpoint falls back to it); the
    reverse rename restores the target on a failed second step.
    """
    import shutil
    target_dir = os.path.normpath(target_dir)
    if not os.path.exists(target_dir):
        os.replace(tmp, target_dir)
        return
    bak = target_dir + ".bak"
    if os.path.exists(bak):
        shutil.rmtree(bak)
    os.rename(target_dir, bak)
    try:
        os.rename(tmp, target_dir)
    except BaseException:
        os.rename(bak, target_dir)
        raise
    shutil.rmtree(bak, ignore_errors=True)


def _resolve_checkpoint_dir(ckpt_dir: str) -> str:
    """Recover the previous checkpoint when a publish was interrupted mid-swap."""
    if os.path.exists(os.path.join(ckpt_dir, "manifest.json")):
        return ckpt_dir
    bak = os.path.normpath(ckpt_dir) + ".bak"
    if os.path.exists(os.path.join(bak, "manifest.json")):
        return bak
    return ckpt_dir


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"unserializable {type(o)}")


def save_checkpoint(target_dir: str, manifest: dict, params: dict,
                    opt_state: dict) -> str:
    for c in manifest.get("scheduler", {}).get("history", []):
        if not isinstance(c.get("id"), int):
            raise ValueError("committed history entries require integer ids")
    ids = [c["id"] for c in manifest.get("scheduler", {}).get("history", [])]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate committed event ids")
    pend = manifest.get("scheduler", {}).get("pending")
    if pend is not None and pend.get("id") in ids:
        raise ValueError("pending event id already committed")
    _check_consistency(manifest, params)
    return _write_tree(target_dir, manifest, params, opt_state)


def _check_consistency(manifest: dict, params: dict) -> None:
    n = manifest.get("n_experts_current")
    reg = manifest.get("expert_registry", [])
    if n is not None and reg and len(reg) != n:
        raise ValueError("expert registry length disagrees with n_experts_current")
    if n is not None and manifest.get("preset", "").startswith("moe"):
        eids = sorted(int(k.split(".")[1]) for k in params if k.startswith("expert.") and k.endswith(".Win"))
        if eids != list(range(len(eids))):
            raise ValueError("expert params are not a dense 0..E-1 registry")
        if len(eids) != n:
            raise ValueError("param expert count disagrees with manifest")


def load_checkpoint(ckpt_dir: str) -> dict:
    ckpt_dir = _resolve_checkpoint_dir(ckpt_dir)
    man_path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"no manifest at {man_path}")
    with open(man_path) as f:
        manifest = json.load(f)
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"incompatible checkpoint version {manifest.get('version')!r} "
                         f"(expected {CHECKPOINT_VERSION})")
    for name in ("params.npz", "optim.npz"):
        p = os.path.join(ckpt_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"checkpoint missing {name} (incomplete write?)")
        digest = _sha256(p)
        if manifest.get("files", {}).get(name) not in (None, digest):
            raise ValueError(f"integrity mismatch for {name}")
    params = {k: np.asarray(v) for k, v in np.load(os.path.join(ckpt_dir, "params.npz")).items()}
    opt_flat = np.load(os.path.join(ckpt_dir, "optim.npz"))
    opt_state = {}
    for k, v in opt_flat.items():
        name, _, sub = k.partition("::")
        opt_state.setdefault(name, {})[sub] = np.asarray(v)
    _check_consistency(manifest, params)
    return {"manifest": manifest, "params": params, "opt_state": opt_state,
            "dir": ckpt_dir}
