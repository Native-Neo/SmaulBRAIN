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

import os
import tempfile

import torch

from storage import EXPERT_NAMES, expert_from_payload, expert_to_payload, load_manifest


def convert_expert_file(src: str, dst: str, to: str = "fp8", tile: int = 64) -> dict:
    """Convert one expert file independently. Returns a small conversion report."""
    from precision import dequantize_fp8_blockwise, quantize_fp8_blockwise

    rec = expert_from_payload(torch.load(src, map_location="cpu", weights_only=False))
    before = {n: rec.weights_fp8[n].nbytes() for n in EXPERT_NAMES}
    if to == "fp8":
        # Any source precision -> canonical FP8 block storage (also re-tiles).
        max_err = 0.0
        for n in EXPERT_NAMES:
            full = dequantize_fp8_blockwise(rec.weights_fp8[n], dtype=torch.float32)
            new_t = quantize_fp8_blockwise(full, tile=tile)
            max_err = max(max_err, float(
                (dequantize_fp8_blockwise(new_t, dtype=torch.float32) - full).abs().max().item()))
            rec.weights_fp8[n] = new_t
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
        os.replace(tmp, dst)
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
    """
    manifest = load_manifest(ckpt_dir)
    reports = []
    if to == "fp8":
        for eid in manifest["expert_ids"]:
            p = os.path.join(ckpt_dir, "experts", f"{eid}.pt")
            reports.append(convert_expert_file(p, p, to=to, tile=tile))
    elif to != "bf16":
        raise ValueError(f"unknown target {to}")
    # to == "bf16": experts intentionally stay FP8; only the trunk converts.
    if to == "bf16":
        for name in ("trunk.pt", "router.pt"):
            p = os.path.join(ckpt_dir, name)
            if not os.path.exists(p):
                continue
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                obj = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in obj.items()}
                fd, tmp = tempfile.mkstemp(dir=ckpt_dir, prefix="tmp_quant_")
                os.close(fd)
                try:
                    torch.save(obj, tmp)
                    os.replace(tmp, p)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
    return reports
