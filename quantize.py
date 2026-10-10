"""Quantization utilities: convert checkpoints without full-model residency.

Supported conversions: FP32 -> BF16, FP32 -> FP8, BF16 -> FP8. Experts are
converted one file at a time (streamed through a row-block window), so a
1B-parameter pool never needs to sit in RAM at once. Topology, expert IDs,
tensor shapes, FP8 scaling metadata, optimizer state, and manifest metadata
are preserved; only the targeted tensors change dtype/format.

Trunk conversion (``convert_checkpoint(..., to="bf16")``) rewrites
trunk.safetensors/router.safetensors via staged sidecars with a joint
publish (both swap only after both convert), then refreshes the manifest's
per-file hashes so the loader's consistency check accepts the result.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

from storage import EXPERT_NAMES, load_expert_file, save_expert_file
from storage import (  # single source for crash-safe IO helpers
    _atomic_write_json,
    _checked_sf_load,
    _fsync_dir,
    _fsync_file,
    _sweep_stale_tmp,
)


def _sidecar(path: str) -> str:
    if path.endswith(".safetensors"):
        return path[: -len(".safetensors")] + ".json"
    return path + ".json"


def convert_expert_file(src: str, dst: str, to: str = "fp8", tile: int = 64) -> dict:
    """Convert one expert file independently. Returns a small conversion report."""
    from precision import (
        FP8BlockTensor, dequantize_fp8_row_block, quantize_fp8_blockwise
    )

    if not os.path.exists(src):
        raise FileNotFoundError(f"missing expert file (refusing): {src}")
    rec = load_expert_file(src)
    before = {n: rec.weights_fp8[n].nbytes() for n in EXPERT_NAMES}
    if to == "fp8":
        # Any source precision -> canonical FP8 block storage (also re-tiles).
        max_err = 0.0
        for n in EXPERT_NAMES:
            stored = rec.weights_fp8[n]
            # Requantize in bounded row windows; never materialize the whole
            # expert as FP32 at once.
            row_block = 256
            code_parts = []
            scale_parts = []
            for row_start in range(0, stored.shape[0], row_block):
                row_end = min(row_start + row_block, stored.shape[0])
                full = dequantize_fp8_row_block(
                    stored, row_start, row_end, dtype=torch.float32
                )
                part = quantize_fp8_blockwise(full, tile=tile)
                recon = dequantize_fp8_row_block(
                    part, 0, row_end - row_start, dtype=torch.float32
                )
                max_err = max(max_err, float((recon - full).abs().max().item()))
                code_parts.append(part.codes)
                scale_parts.append(part.scales)
            rec.weights_fp8[n] = FP8BlockTensor(
                codes=torch.cat(code_parts, dim=0),
                scales=torch.cat(scale_parts, dim=0),
                shape=stored.shape,
                tile=tile,
            )
    elif to == "bf16":
        # Experts stay FP8 on disk per the precision policy (large
        # storage-heavy tensors); BF16 conversion applies to the trunk.
        # Refusing here is intentional, not a missing feature.
        raise ValueError("expert storage is FP8 by policy; use convert_checkpoint(..., to='bf16') for trunk BF16")
    else:
        raise ValueError(f"unknown target {to}")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    # Stage under a sidecar name, then publish tensors + JSON sidecar together.
    # (save_expert_file derives the JSON sidecar from dst, so dst must not
    # collide with a live expert path.)
    save_expert_file(rec, dst)
    _fsync_file(dst)
    _fsync_dir(os.path.dirname(dst) or ".")
    after = {n: rec.weights_fp8[n].nbytes() for n in EXPERT_NAMES}
    report = {"expert_id": rec.expert_id, "to": to,
              "bytes_before": sum(before.values()), "bytes_after": sum(after.values())}
    if to == "fp8":
        report["max_abs_err"] = max_err
    return report


def convert_checkpoint(ckpt_dir: str, to: str = "fp8", tile: int = 64) -> list[dict]:
    """Convert checkpoint precision without full-model residency.

    ``to="fp8"`` requantizes/retile every expert file independently (the trunk
    is untouched — FP8 trunk storage is refused per the precision policy).
    ``to="bf16"`` casts trunk.safetensors/router.safetensors to BF16 (experts stay FP8).

    Transactional per path: ``to="fp8"`` validates every input first,
    converts into a staging dir, and publishes only after all conversions
    succeed; ``to="bf16"`` converts both dense files before either swaps
    into place. Precision metadata (manifest + config tile) and the
    manifest's per-file hashes update last (commit point), so a crash
    never leaves a half-converted pool behind silently.
    Holds the checkpoint exclusive lock across staging+publish so a
    concurrent save/load cannot observe a mixed generation.
    """
    if to not in ("fp8", "bf16"):
        raise ValueError(f"unknown target {to}")
    import storage as _st
    with _st._ckpt_locked(ckpt_dir, exclusive=True):
        return _convert_checkpoint_locked(ckpt_dir, to=to, tile=tile)


def _convert_checkpoint_locked(ckpt_dir: str, to: str = "fp8", tile: int = 64) -> list[dict]:
    import storage as _st
    manifest = _st.load_manifest(ckpt_dir)
    man_path = os.path.join(ckpt_dir, "manifest.json")
    cfg_path = os.path.join(ckpt_dir, "config.json")
    exp_dir = os.path.join(ckpt_dir, "experts")
    # Drop crash leftovers before staging new sidecars.
    _sweep_stale_tmp(ckpt_dir)
    _sweep_stale_tmp(exp_dir)
    reports = []
    if to == "fp8":
        paths = [(eid, os.path.join(ckpt_dir, "experts", f"{eid}.safetensors"))
                 for eid in manifest["expert_ids"]]
        missing = [eid for eid, p in paths
                   if not (os.path.exists(p) and os.path.exists(_sidecar(p)))]
        if missing:
            raise FileNotFoundError(f"missing expert files (refusing): {missing}")
        empty = [eid for eid, p in paths if os.stat(p).st_size == 0]
        if empty:
            raise ValueError(
                f"checkpoint validation failed: empty expert files {empty}"
            )
        staged: list[tuple[str, str]] = []
        # Stage inside a dedicated subdir: save_expert_file sweeps *.convert_tmp
        # in its target dir, so staging next to the live files would nuke
        # previously converted experts.
        stage_dir = tempfile.mkdtemp(dir=exp_dir, prefix="tmp_quant_stage_")
        try:
            for eid, p in paths:
                tmp = os.path.join(stage_dir, f"{eid}.safetensors.convert_tmp")
                reports.append(convert_expert_file(p, tmp, to=to, tile=tile))
                staged.append((tmp, p))
            for tmp, p in staged:
                _fsync_file(tmp)
                _fsync_file(tmp + ".json")
                os.replace(tmp, p)
                os.replace(tmp + ".json", _sidecar(p))
            _fsync_dir(exp_dir)
        finally:
            for tmp, _ in staged:
                for side in (tmp, tmp + ".json"):
                    if os.path.exists(side):
                        try:
                            os.remove(side)
                        except OSError:
                            pass
            try:
                os.rmdir(stage_dir)
            except OSError:
                pass
        try:
            with open(cfg_path) as f:
                cfg_dict = json.load(f)
        except Exception as e:
            raise ValueError(
                f"checkpoint validation failed: unreadable config.json: {e}"
            ) from e
        cfg_dict["fp8_tile"] = tile
        _atomic_write_json(cfg_dict, cfg_path)
    # to == "bf16": experts intentionally stay FP8; only the trunk converts.
    if to == "bf16":
        import storage as _st2
        # Phase 1 (pure build): convert + validate every file before any
        # publish, so a bad input cannot commit a half-converted pair.
        converted: dict[str, dict] = {}
        for name in ("trunk.safetensors", "router.safetensors"):
            p = os.path.join(ckpt_dir, name)
            if not os.path.exists(p):
                continue
            if os.stat(p).st_size == 0:
                raise ValueError(
                    f"checkpoint validation failed: empty file {p}"
                )
            obj = _checked_sf_load(p)
            converted[name] = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
                               for k, v in obj.items()}
        # Phase 2 (joint publish): stage to sidecars, then swap both into
        # place. A crash between the swaps can never silently yield
        # trunk-bf16 + router-fp32: the manifest hashes refreshed below
        # make the next load refuse the mixed pair loudly.
        staged: list[tuple[str, str]] = []
        try:
            for name, obj in converted.items():
                tmp = os.path.join(ckpt_dir, name + ".convert_tmp")
                _st2._atomic_save_tensors(obj, tmp)
                staged.append((tmp, os.path.join(ckpt_dir, name)))
            for tmp, p in staged:
                _fsync_file(tmp)
                os.replace(tmp, p)
            _fsync_dir(ckpt_dir)
        finally:
            for tmp, _ in staged:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
    manifest["precision"] = {"format": to, "fp8_tile": tile}
    _atomic_write_json(manifest, man_path)
    # Re-hash every payload (both paths rewrite files in place): without
    # this the loader's cross-file consistency check would refuse our own
    # conversion on the next load.
    import storage as _st3
    _st3.refresh_manifest_hashes(ckpt_dir)
    return reports
