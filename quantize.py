"""Quantization utilities: convert checkpoints without full-model residency.

Supported conversions: FP32 -> BF16, FP32 -> FP8, BF16 -> FP8. Experts are
converted one file at a time (streamed through a row-block window), so a
1B-parameter pool never needs to sit in RAM at once. Topology, expert IDs,
tensor shapes, FP8 scaling metadata, optimizer state, and manifest metadata
are preserved; only the targeted tensors change dtype/format.

Trunk conversion (``convert_trunk``) rewrites trunk.pt/router.pt in place
within the checkpoint directory.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch

from storage import EXPERT_NAMES, expert_from_payload, expert_to_payload, load_manifest


def _fsync_file(path: str) -> None:
    try:
        with open(path, "rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except OSError:
        pass


def _fsync_dir(dirpath: str) -> None:
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
    """Remove leftover converter sidecars from a prior crash (best-effort).

    Unified with storage: matches ``tmp_quant_``, ``tmp_ckpt_``,
    ``tmp_json_``, staged generation dirs (``tmp_ckpt_gen_*``), and
    ``.convert_tmp`` sidecars. Directories are removed recursively.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if name.startswith(("tmp_quant_", "tmp_ckpt_", "tmp_json_")) or name.endswith(".convert_tmp"):
            p = os.path.join(directory, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except OSError:
                pass


def _checked_torch_load(path: str):
    try:
        if os.stat(path).st_size == 0:
            raise ValueError(
                f"checkpoint validation failed: empty file {path}"
            )
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except OSError as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable file {path}: {e}"
        ) from e
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable/truncated file {path}: {e}"
        ) from e


def _atomic_write_json(payload: dict, path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix="tmp_quant_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_dir(os.path.dirname(path) or ".")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def convert_expert_file(src: str, dst: str, to: str = "fp8", tile: int = 64) -> dict:
    """Convert one expert file independently. Returns a small conversion report."""
    from precision import (
        FP8BlockTensor, dequantize_fp8_row_block, quantize_fp8_blockwise
    )

    if not os.path.exists(src):
        raise FileNotFoundError(f"missing expert file (refusing): {src}")
    rec = expert_from_payload(_checked_torch_load(src))
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
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst) or ".", prefix="tmp_quant_")
    os.close(fd)
    try:
        torch.save(expert_to_payload(rec), tmp)
        _fsync_file(tmp)
        os.replace(tmp, dst)
        _fsync_dir(os.path.dirname(dst) or ".")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
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
    ``to="bf16"`` casts trunk.pt/router.pt to BF16 (experts stay FP8).

    Transactional: every input is validated first, converted outputs land
    in sidecar files, and the sidecars replace the originals only after all
    conversions succeed — a failure never leaves a half-converted pool.
    Precision metadata (manifest + config tile) updates last (commit point).
    Holds the checkpoint exclusive lock across staging+publish so a
    concurrent save/load cannot observe a mixed generation.
    """
    if to not in ("fp8", "bf16"):
        raise ValueError(f"unknown target {to}")
    import storage as _st
    with _st._ckpt_locked(ckpt_dir, exclusive=True):
        return _convert_checkpoint_locked(ckpt_dir, to=to, tile=tile)


def _convert_checkpoint_locked(ckpt_dir: str, to: str = "fp8", tile: int = 64) -> list[dict]:
    manifest = load_manifest(ckpt_dir)
    man_path = os.path.join(ckpt_dir, "manifest.json")
    cfg_path = os.path.join(ckpt_dir, "config.json")
    exp_dir = os.path.join(ckpt_dir, "experts")
    # Drop crash leftovers before staging new sidecars.
    _sweep_stale_tmp(ckpt_dir)
    _sweep_stale_tmp(exp_dir)
    reports = []
    if to == "fp8":
        paths = [(eid, os.path.join(ckpt_dir, "experts", f"{eid}.pt"))
                 for eid in manifest["expert_ids"]]
        missing = [eid for eid, p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"missing expert files (refusing): {missing}")
        empty = [eid for eid, p in paths if os.stat(p).st_size == 0]
        if empty:
            raise ValueError(
                f"checkpoint validation failed: empty expert files {empty}"
            )
        staged: list[tuple[str, str]] = []
        try:
            for eid, p in paths:
                tmp = p + ".convert_tmp"
                reports.append(convert_expert_file(p, tmp, to=to, tile=tile))
                staged.append((tmp, p))
            for tmp, p in staged:
                _fsync_file(tmp)
                os.replace(tmp, p)
            _fsync_dir(exp_dir)
        finally:
            for tmp, _ in staged:
                if os.path.exists(tmp):
                    os.remove(tmp)
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
        for name in ("trunk.pt", "router.pt"):
            p = os.path.join(ckpt_dir, name)
            if not os.path.exists(p):
                continue
            if os.stat(p).st_size == 0:
                raise ValueError(
                    f"checkpoint validation failed: empty file {p}"
                )
            obj = _checked_torch_load(p)
            if isinstance(obj, dict):
                obj = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in obj.items()}
                fd, tmp = tempfile.mkstemp(dir=ckpt_dir, prefix="tmp_quant_")
                os.close(fd)
                try:
                    torch.save(obj, tmp)
                    _fsync_file(tmp)
                    os.replace(tmp, p)
                    _fsync_dir(ckpt_dir)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
    manifest["precision"] = {"format": to, "fp8_tile": tile}
    _atomic_write_json(manifest, man_path)
    return reports
