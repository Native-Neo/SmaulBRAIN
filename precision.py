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

Sparse update contract (audit 028 — expert / row-range / tile-block):
  Level 1 (expert): a step touching selected experts decodes/updates/
    requantizes only those experts' tensors. Unstepped experts keep stored
    bytes bit-identical (proven by expert-level tests elsewhere).
  Level 2 (row range): ``dequantize_fp8_row_block`` / ``update_fp8_row_block``
    touch only rows ``[row_start:row_end)``. Untouched rows' codes AND scales
    are preserved bit-for-bit; no full-matrix transient is materialized.
  Level 3 (tile block): ``dequantize_fp8_block`` / ``update_fp8_tile_block``
    touch only rows ``[row_start:row_end)`` × columns ``[col_start:col_end)``.
    Reads accept any column window (no full-row transient). Writes require a
    tile-aligned column window (``col_start % tile == 0`` and ``col_end % tile
    == 0`` or ``col_end == n_cols``), because one FP32 scale is shared by a
    whole ``tile``-wide block: rewriting a partial tile from partial data
    would silently re-derive the shared scale and corrupt sibling columns.
    Touched tiles are requantized from the new values alone, which is exactly
    what a full-row requant would have written there (amax is per
    (row, block); zero padding of a ragged tail never changes an amax), while
    untouched tiles stay bit-identical.

Optimizer-state semantics for sparse updates (Adam-style SmaulOpt):
  Frozen means frozen, NOT zero-gradient: rows/tiles outside the update range
    keep their stored moments (``m``, ``v``/``v_row``) bit-identical. No
    ``beta`` decay is applied to them (a dense step with ``g == 0`` on those
    rows WOULD decay ``m *= beta_m`` — sparse updates deliberately do not).
  Global clock still ticks: the per-tensor ``step`` counter increments even
    on a sparse update, so bias-correction denominators (``1 - beta**t``)
    advance for frozen rows too. Their *stored* moments are untouched, but
    their *effective* ``m_hat``/``v_hat`` drift with ``t``. Exact pause-time
    equivalence would need per-row step counters (optimizer-side decision —
    see NEEDS-OPTIMIZER-HUNK notes in the issue thread).
  Factored-``v`` coupling: ``v_col`` is shared across ALL rows. A sparse row
    update still folds the slice's column means into the global ``v_col``,
    so future updates of untouched rows are perturbed through it even though
    their stored bytes/moments are bit-identical. Fully isolated sparse
    steps would need ``v_col`` frozen on range updates (optimizer-side).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import native

FP8_MAX = 448.0  # max normal of E4M3
FP8_DTYPE = torch.float8_e4m3fn


@dataclass
class FP8BlockTensor:
    """One FP8-stored tensor: uint8 codes + per-row-block FP32 scales."""

    codes: torch.Tensor  # uint8, same shape as the logical tensor
    scales: torch.Tensor  # float32, one scale per (row, block)
    shape: tuple[int, ...]
    tile: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.tile, int) or isinstance(self.tile, bool) \
                or self.tile < 1:
            raise ValueError(f"fp8 tile must be a positive int, got {self.tile!r}")

    def nbytes(self) -> int:
        return self.codes.nelement() + self.scales.nelement() * 4


def quantize_fp8_blockwise(w: torch.Tensor, tile: int = 64) -> FP8BlockTensor:
    """Quantize a float tensor to blockwise-scaled E4M3 FP8.

    Each row is split into ``tile``-wide blocks; every block gets its own
    FP32 scale (amax/448). Values pass through real E4M3 casting so the
    stored bytes are true FP8 code points, not relabeled higher precision.
    """
    if not isinstance(tile, int) or isinstance(tile, bool) or tile < 1:
        raise ValueError(f"fp8 tile must be a positive int, got {tile!r}")
    w32 = w.detach().to(torch.float32)
    orig = tuple(w32.shape)
    flat = w32.reshape(-1, orig[-1])
    n_cols = flat.shape[1]
    n_blocks = (n_cols + tile - 1) // tile
    rows = flat.shape[0]
    # Native path is bit-exact (proven by tests/test_cpp_parity.py), so it
    # may auto-dispatch even with gradients enabled: stored bytes are
    # identical either way. It needs no padding: the kernel loops exact cols.
    # Output buffers are allocated lazily: the kernel needs CPU tensors, so
    # non-CPU inputs skip native without pointlessly allocating CPU outputs.
    if flat.device.type == "cpu":
        scales = torch.empty(rows, n_blocks, dtype=torch.float32)
        codes_flat = torch.empty(rows, n_cols, dtype=torch.uint8)
        if native.call_fp8_quant(flat, codes_flat, scales, rows, n_cols, tile):
            codes_u8 = codes_flat.reshape(orig).clone()
            return FP8BlockTensor(codes=codes_u8, scales=scales, shape=orig, tile=tile)
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
    rows = t.codes.reshape(-1, n_cols).shape[0]
    codes_2d = t.codes.reshape(-1, n_cols)
    scales_2d = t.scales.reshape(-1, n_blocks)
    # Lazily allocated on the codes device: non-CPU storage skips the
    # CPU-only kernel without a wasted output allocation.
    if codes_2d.device.type == "cpu" and scales_2d.device.type == "cpu":
        out = torch.empty(rows, n_cols, dtype=torch.float32, device=codes_2d.device)
        if native.call_fp8_dequant(codes_2d, scales_2d, out, rows, n_cols, t.tile):
            return out.reshape(t.shape).to(dtype)
    # Reference fallback (also used when native is disabled/unavailable).
    codes_f8 = codes_2d.contiguous().view(FP8_DTYPE)
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


def update_fp8_row_block(
    t: FP8BlockTensor, new_rows: torch.Tensor, row_start: int, row_end: int
) -> None:
    """Quantize and replace only row_start:row_end in an FP8BlockTensor in place.

    Untouched rows (codes and scales) are preserved bit-for-bit.
    ``new_rows`` must match the slice shape exactly:
    ``(row_end - row_start, *t.shape[1:])``.
    """
    if row_start < 0 or row_end > t.shape[0] or row_start >= row_end:
        raise ValueError(
            f"invalid row range [{row_start}, {row_end}) for tensor with {t.shape[0]} rows"
        )
    expect = (row_end - row_start, *t.shape[1:])
    if tuple(new_rows.shape) != tuple(expect):
        raise ValueError(
            f"new_rows shape {tuple(new_rows.shape)} != slice shape {tuple(expect)} "
            f"for row range [{row_start}, {row_end})"
        )
    sub = quantize_fp8_blockwise(new_rows, tile=t.tile)
    t.codes[row_start:row_end] = sub.codes
    t.scales[row_start:row_end] = sub.scales


def dequantize_fp8_block(
    t: FP8BlockTensor,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize only rows [row_start:row_end) × cols [col_start:col_end).

    Level-3 (tile-block) read: no full-matrix and no full-row transient —
    only the requested ``(row_end - row_start) × (col_end - col_start)``
    window is materialized. Any column window is accepted (partial tiles are
    fine for reads: each element keeps its own block's scale). The result is
    bit-exact vs. slicing ``dequantize_fp8_blockwise(t)`` (same elementwise
    ``f32(code) * scale`` math; native/fallback parity is proven elsewhere).

    Contract: 2-D tensors (rows × cols, the expert-weight case). Higher-dim
    storage must use the row-range helpers.
    """
    if t.codes.ndim != 2 or len(t.shape) != 2:
        raise ValueError(
            "tile-block reads require a 2-D (rows x cols) FP8BlockTensor, "
            f"got codes.ndim={t.codes.ndim} shape={t.shape}"
        )
    n_rows, n_cols = int(t.shape[0]), int(t.shape[-1])
    if not (0 <= row_start < row_end <= n_rows):
        raise ValueError(f"invalid row range [{row_start}, {row_end}) for {n_rows} rows")
    if not (0 <= col_start < col_end <= n_cols):
        raise ValueError(f"invalid col range [{col_start}, {col_end}) for {n_cols} cols")
    codes_2d = t.codes.reshape(-1, n_cols)
    scales_2d = t.scales.reshape(-1, t.scales.shape[-1])
    sub_codes = codes_2d[row_start:row_end, col_start:col_end].contiguous()
    block_idx = torch.arange(col_start, col_end, device=scales_2d.device) // t.tile
    sub_scales = scales_2d[row_start:row_end][:, block_idx]
    codes_f8 = sub_codes.view(FP8_DTYPE).to(torch.float32)
    out = codes_f8 * sub_scales.to(codes_f8.device)
    return out.to(dtype)


def update_fp8_tile_block(
    t: FP8BlockTensor,
    new_slice: torch.Tensor,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> None:
    """Quantize and replace only rows [row_start:row_end) × cols [col_start:col_end).

    Level-3 (tile-block) write, in place. Untouched tiles (codes AND scales)
    are preserved bit-for-bit — no full-row or full-matrix requant.

    Contract:
      * 2-D tensors (rows × cols, the expert-weight case).
      * Column window must be tile-aligned: ``col_start % tile == 0`` and
        (``col_end % tile == 0`` or ``col_end == n_cols``). A partial tile
        shares its FP32 scale with sibling columns outside the window, so a
        partial-tile rewrite would corrupt them; alignment is enforced loudly.
      * ``new_slice.shape`` must be exactly ``(row_end - row_start,
        col_end - col_start)``.
      * Storage equivalence: each touched (row, block) is requantized from
        the new values alone with the same per-block amax math as
        ``quantize_fp8_blockwise`` (zero padding of a ragged tail never
        changes an amax), so touched tiles equal what a full-row requant of
        the same new rows would have written there.

    Optimizer note: freezing untouched tiles' stored bytes does NOT freeze
    the optimizer clock — the per-tensor ``step`` still ticks (bias
    corrections drift) and factored ``v_col`` is still shared. See the module
    docstring; per-column optimizer ranges are an optimizer-side hunk.
    """
    if t.codes.ndim != 2 or len(t.shape) != 2:
        raise ValueError(
            "tile-block updates require a 2-D (rows x cols) FP8BlockTensor, "
            f"got codes.ndim={t.codes.ndim} shape={t.shape}"
        )
    n_rows, n_cols = int(t.shape[0]), int(t.shape[-1])
    if not (0 <= row_start < row_end <= n_rows):
        raise ValueError(f"invalid row range [{row_start}, {row_end}) for {n_rows} rows")
    if not (0 <= col_start < col_end <= n_cols):
        raise ValueError(f"invalid col range [{col_start}, {col_end}) for {n_cols} cols")
    if col_start % t.tile != 0 or (col_end % t.tile != 0 and col_end != n_cols):
        raise ValueError(
            f"column range [{col_start}, {col_end}) must be tile-aligned "
            f"(tile={t.tile}): col_start % tile == 0 and "
            f"(col_end % tile == 0 or col_end == {n_cols})"
        )
    expect = (row_end - row_start, col_end - col_start)
    if tuple(new_slice.shape) != tuple(expect):
        raise ValueError(
            f"new_slice shape {tuple(new_slice.shape)} != window shape {tuple(expect)} "
            f"for rows [{row_start}, {row_end}) cols [{col_start}, {col_end})"
        )
    # Reference per-tile math (proven bit-exact vs. native dispatch): amax per
    # (row, block) over the new values alone, then genuine E4M3 rounding.
    w32 = new_slice.detach().to(torch.float32)
    b0 = col_start // t.tile
    b1 = min((col_end + t.tile - 1) // t.tile, int(t.scales.shape[-1]))
    for b in range(b0, b1):
        bc0 = b * t.tile
        bw = min(t.tile, n_cols - bc0)  # ragged tail block is narrower
        src = w32[:, bc0 - col_start : bc0 - col_start + bw]
        amax = src.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (amax / FP8_MAX).reshape(-1)
        q = (src / amax * FP8_MAX).to(FP8_DTYPE)
        # Direct 2-D indexing: guaranteed in-place even if storage is a view.
        t.codes[row_start:row_end, bc0 : bc0 + bw] = q.view(torch.uint8)
        t.scales[row_start:row_end, b] = scale




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
