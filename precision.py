"""Precision policy and blockwise FP8 storage.

Policy (best precision per component, no giant hidden FP32 master copy):
  FP8  - expert weights, large projection weights (storage; blockwise scaled)
  BF16 - activations, optimizer-state storage
  FP32 - recurrent attention state
  FP32 - norm statistics, accumulators, loss, optimizer math, scales,
         routing/halting statistics

FP8 format: E4M3 with per-block (tile) FP32 scales. A stored tensor is
``(codes: uint8, scales: float32, shape)`` — genuinely FP8 bytes on disk/RAM,
not relabeled FP32. Dequantization happens per expert on activation (one
expert at a time), never as a full-model FP32 transient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FP8_MAX = 448.0  # max normal of E4M3
FP8_DTYPE = torch.float8_e4m3fn


@dataclass
class FP8BlockTensor:
    """One FP8-stored tensor: uint8 codes + per-row-block FP32 scales."""

    codes: torch.Tensor  # uint8, same shape as the logical tensor
    scales: torch.Tensor  # float32, one scale per (row, block)
    shape: tuple[int, ...]
    tile: int = 64

    def nbytes(self) -> int:
        return self.codes.nelement() + self.scales.nelement() * 4


def quantize_fp8_blockwise(w: torch.Tensor, tile: int = 64) -> FP8BlockTensor:
    """Quantize a float tensor to blockwise-scaled E4M3 FP8.

    Each row is split into ``tile``-wide blocks; every block gets its own
    FP32 scale (amax/448). Values pass through real E4M3 casting so the
    stored bytes are true FP8 code points, not relabeled higher precision.
    """
    w32 = w.detach().to(torch.float32)
    orig = tuple(w32.shape)
    flat = w32.reshape(-1, orig[-1])
    n_cols = flat.shape[1]
    n_blocks = (n_cols + tile - 1) // tile
    pad = n_blocks * tile - n_cols
    if pad:
        flat = torch.cat([flat, torch.zeros(flat.shape[0], pad)], dim=1)
    blocks = flat.reshape(-1, n_blocks, tile)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = (amax / FP8_MAX).reshape(-1, n_blocks).contiguous()  # fp32
    scaled = blocks / amax * FP8_MAX
    q_f8 = scaled.to(FP8_DTYPE)  # genuine E4M3 rounding
    flat_q = q_f8.reshape(flat.shape)[:, :n_cols].contiguous()
    codes_u8 = flat_q.view(torch.uint8).reshape(orig).clone()
    return FP8BlockTensor(codes=codes_u8, scales=scales, shape=orig, tile=tile)


def dequantize_fp8_blockwise(t: FP8BlockTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct a float tensor from blockwise FP8 storage (per-expert use)."""
    n_cols = t.shape[-1]
    n_blocks = t.scales.shape[-1]
    codes_f8 = t.codes.reshape(-1, n_cols).contiguous().view(FP8_DTYPE)
    flat = codes_f8.to(torch.float32)
    pad = n_blocks * t.tile - n_cols
    if pad:
        flat = torch.cat([flat, torch.zeros(flat.shape[0], pad)], dim=1)
    blocks = flat.reshape(-1, n_blocks, t.tile)
    # q ~= orig/amax*448 and scales = amax/448, so orig ~= q*scales.
    out = (blocks * t.scales.unsqueeze(-1)).reshape(flat.shape)[:, :n_cols]
    return out.reshape(t.shape).to(dtype)


def dequantize_fp8_row_block(
    t: FP8BlockTensor, row_start: int, row_end: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize only a row slice — avoids full-matrix transient for big experts."""
    sub = FP8BlockTensor(
        codes=t.codes[row_start:row_end],
        scales=t.scales[row_start:row_end],
        shape=(row_end - row_start, *t.shape[1:]),
        tile=t.tile,
    )
    return dequantize_fp8_blockwise(sub, dtype=dtype)


# --- dtype policy table -----------------------------------------------------

#: component -> (storage dtype, compute dtype)
PRECISION_POLICY: dict[str, tuple[str, str]] = {
    "expert_weights": ("fp8", "bf16"),
    "large_projections": ("fp8", "bf16"),
    "activations": ("bf16", "bf16"),
    "recurrent_state": ("fp32", "fp32"),
    "optimizer_state": ("bf16", "fp32"),
    "norm_stats": ("fp32", "fp32"),
    "loss": ("fp32", "fp32"),
    "accumulators": ("fp32", "fp32"),
    "scales": ("fp32", "fp32"),
    "routing_stats": ("fp32", "fp32"),
    "halting_stats": ("fp32", "fp32"),
    "embeddings": ("bf16", "bf16"),
}


def compute_dtype(name: str, fallback: torch.dtype = torch.bfloat16) -> torch.dtype:
    table = {"fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp32": torch.float32}
    entry = PRECISION_POLICY.get(name)
    if entry is None:
        return fallback
    return table[entry[1]]
