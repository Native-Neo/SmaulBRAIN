"""Precision policy, hybrid FP8 execution, and blockwise FP8 storage.

Policy (no duplicate full-model master copy):
  FP8       - expert weights, blockwise-scaled storage
  hybrid FP8 - dense projections: E4M3 forward operands, E5M2 backward gradients
  BF16      - activations and optimizer-state storage
  FP32      - dense-projection master parameters, recurrent state, norms,
              accumulators, loss, optimizer math, scales, routing/halting stats

Expert FP8 format: E4M3 with per-block (tile) FP32 scales. A stored tensor is
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

from collections import deque
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

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
        if (not isinstance(self.shape, (tuple, list)) or not self.shape
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                       for dim in self.shape)):
            raise ValueError(f"FP8 logical shape must contain positive integer dimensions, got {self.shape!r}")
        shape = tuple(self.shape)
        if not torch.is_tensor(self.codes) or self.codes.dtype != torch.uint8:
            raise ValueError("FP8 codes must be a uint8 tensor")
        if tuple(self.codes.shape) != shape:
            raise ValueError(
                f"FP8 codes shape {tuple(self.codes.shape)} does not match logical shape {shape}"
            )
        if not torch.is_tensor(self.scales) or self.scales.dtype != torch.float32:
            raise ValueError("FP8 scales must be a float32 tensor")
        expected_scales = (
            self.codes.numel() // shape[-1],
            (shape[-1] + self.tile - 1) // self.tile,
        )
        if tuple(self.scales.shape) != expected_scales:
            raise ValueError(
                f"FP8 scales shape {tuple(self.scales.shape)} does not match "
                f"expected {expected_scales} for shape {shape} and tile {self.tile}"
            )
        if self.scales.device != self.codes.device:
            raise ValueError("FP8 codes and scales must be on the same device")

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
        flat = torch.cat([
            flat,
            torch.zeros(flat.shape[0], pad, dtype=flat.dtype, device=flat.device),
        ], dim=1)
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
        flat = torch.cat([
            flat,
            torch.zeros(flat.shape[0], pad, dtype=flat.dtype, device=flat.device),
        ], dim=1)
    blocks = flat.reshape(-1, n_blocks, t.tile)
    # q ~= orig/amax*448 and scales = amax/448, so orig ~= q*scales.
    out = (blocks * t.scales.unsqueeze(-1)).reshape(flat.shape)[:, :n_cols]
    return out.reshape(t.shape).to(dtype)


def _validate_fp8_row_range(t: FP8BlockTensor, row_start: int, row_end: int) -> None:
    """Validate the requested row interval against an FP8 block tensor."""
    if t.codes.ndim != 2 or len(t.shape) != 2:
        raise ValueError(
            "row-block operations require a 2-D (rows x cols) FP8BlockTensor, "
            f"got codes.ndim={t.codes.ndim} shape={t.shape}"
        )
    if (isinstance(row_start, bool) or not isinstance(row_start, int)
            or isinstance(row_end, bool) or not isinstance(row_end, int)
            or not (0 <= row_start < row_end <= t.shape[0])):
        raise ValueError(
            f"invalid row range [{row_start}, {row_end}) for tensor with {t.shape[0]} rows"
        )


def dequantize_fp8_row_block(
    t: FP8BlockTensor, row_start: int, row_end: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize only a row slice — avoids full-matrix transient for big experts."""
    _validate_fp8_row_range(t, row_start, row_end)
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
    _validate_fp8_row_range(t, row_start, row_end)
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
    bounds = (row_start, row_end, col_start, col_end)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in bounds):
        raise ValueError(f"tile-block indices must be integers, got {bounds!r}")
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
    bounds = (row_start, row_end, col_start, col_end)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in bounds):
        raise ValueError(f"tile-block indices must be integers, got {bounds!r}")
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
    "large_projections": ("fp32", "hybrid_fp8"),
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
    """Resolve a supported compute dtype name with a safe fallback."""
    # hybrid_fp8 returns activations in BF16 by default; its arithmetic
    # itself uses E4M3 forward and E5M2 backward quantization.
    table = {"fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16,
             "fp32": torch.float32, "hybrid_fp8": torch.bfloat16}
    entry = PRECISION_POLICY.get(name)
    if entry is None:
        return fallback
    return table[entry[1]]


# Hybrid execution policy: E4M3 forward operands and E5M2 backward gradients.
E4M3_MAX = 448.0
E5M2_MAX = 57344.0
E4M3_DTYPE = torch.float8_e4m3fn
E5M2_DTYPE = torch.float8_e5m2


class DelayedScaler:
    """Per-tensor delayed scaling: scale tracks the amax history maximum.

    ``scale = fp8_max * margin / max(amax_history)`` so a single spike
    cannot rescale (and overflow) the next step. Non-positive or
    non-finite amax readings are ignored, never adopted.
    """

    def __init__(self, fp8_max: float, history_len: int = 16, margin: float = 0.9) -> None:
        self.fp8_max = float(fp8_max)
        self.history: deque[float] = deque(maxlen=max(1, history_len))
        self.margin = float(margin)
        self.scale = 1.0

    def update(self, amax: float) -> float:
        """Fold one amax reading in; return the scale to use this step."""
        if math.isfinite(amax) and amax > 0.0:
            self.history.append(float(amax))
        hist_max = max(self.history) if self.history else 1.0
        hist_max = max(hist_max, 1e-12)
        self.scale = self.fp8_max * self.margin / hist_max
        return self.scale

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"DelayedScaler(scale={self.scale:.4g}, hist={len(self.history)})"


def _amax(x: torch.Tensor) -> float:
    """Return the maximum absolute finite magnitude of a tensor."""
    with torch.no_grad():
        m = x.detach().float().abs().amax().item()
    return float(m)


def _quantize_to_fp8(x: torch.Tensor, scale: float, fp8_dtype: torch.dtype,
                     fp8_max: float) -> torch.Tensor:
    """Scale, clamp into FP8 range, cast (subnormals flush via the cast)."""
    scaled = x.float() * float(scale)
    # Clamp BEFORE the cast: out-of-range would otherwise become inf/NaN
    # codes; tiny values are left alone to flush through subnormals to zero.
    scaled = torch.clamp(scaled, -fp8_max, fp8_max)
    return scaled.to(fp8_dtype)


def _dequantize_from_fp8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Restore scaled floating-point values from FP8 codes."""
    return q.float() / max(float(scale), 1e-12)


def _try_scaled_mm(a_q: torch.Tensor, b_q: torch.Tensor, out_dtype: torch.dtype):
    """Opportunistic FP8 matmul via torch._scaled_mm (Hopper+). None = fall back.

    a_q: [M, K] fp8, b_q: [N, K] fp8 (row-major twin of B = [K, N]).
    Scales are folded as 1.0 here because _quantize_to_fp8 already scaled
    values into FP8 code space; callers divide by the product of scales.
    """
    try:
        fn = getattr(torch, "_scaled_mm", None)
        if fn is None or a_q.device.type != "cuda":
            return None
        if torch.cuda.get_device_capability(a_q.device) < (9, 0):
            return None
        one = torch.tensor(1.0, device=a_q.device)
        out = fn(a_q, b_q.t(), scale_a=one, scale_b=one, out_dtype=out_dtype)
        if isinstance(out, tuple):
            out = out[0]
        return out
    except Exception:
        return None


class _HybridFP8Matmul(torch.autograd.Function):
    """Causal, stateless FP8 operands with E5M2 row-scaled gradients.

    Activation scales are per token (the last dimension is features), and
    weight scales are derived from the current weight tensor. No scale depends
    on neighboring tokens or prior calls, so padding, chunking, checkpoint
    reload, and streaming cannot change an earlier token's quantization.
    """

    @staticmethod
    def forward(ctx, x, w, bias, in_scaler, wt_scaler, grad_scaler,
                out_dtype, e5m2_for_grads):
        x32, w32 = x.float(), w.float()
        ctx.out_dtype = out_dtype
        ctx.e5m2_for_grads = bool(e5m2_for_grads)
        ctx.has_bias = bias is not None
        ctx.x_dtype, ctx.w_dtype = x.dtype, w.dtype
        if not (torch.isfinite(x32).all() and torch.isfinite(w32).all()
                and (bias is None or torch.isfinite(bias.float()).all())):
            # Poisoned inputs bypass quantization rather than creating FP8 NaNs.
            ctx.fallback = True
            ctx.save_for_backward(x32, w32)
            out = x32 @ w32.t()
            if bias is not None:
                out = out + bias.float()
            return out.to(out_dtype)

        ctx.fallback = False
        # Quantize each token independently. A single scale over [B,T,K]
        # leaks information across tokens and breaks full-vs-streaming parity.
        flat_x = x32.reshape(-1, x32.shape[-1])
        x_amax = flat_x.abs().amax(dim=-1, keepdim=True)
        x_scale = torch.where(
            x_amax > 0,
            (E4M3_MAX * 0.9) / x_amax.clamp_min(1e-12),
            torch.ones_like(x_amax),
        )
        x_q = (flat_x * x_scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3_DTYPE)
        x_scale_saved = x_scale.reshape(*x32.shape[:-1], 1)

        # The weight scale is stateless and independent of the input batch.
        w_amax = w32.detach().abs().amax()
        w_scale = torch.where(
            w_amax > 0,
            (E4M3_MAX * 0.9) / w_amax.clamp_min(1e-12),
            torch.ones_like(w_amax),
        )
        w_q = (w32 * w_scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3_DTYPE)
        ctx.save_for_backward(x_q.reshape(x32.shape), w_q, x_scale_saved, w_scale)

        # Per-token activation scales differ. FP8 GEMM is still valid: the
        # quantized matrix product is rescaled per output row afterwards.
        # Older/non-Hopper devices use the portable dequantized FP32 fallback.
        native_out = _try_scaled_mm(x_q, w_q, torch.float32)
        if native_out is not None:
            out2d = native_out / (x_scale * w_scale)
        else:
            x_dq = x_q.float() / x_scale
            w_dq = w_q.float() / w_scale
            out2d = x_dq @ w_dq.t()
        out = out2d.reshape(*x32.shape[:-1], w32.shape[0])
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for inputs, weights, bias, and scaling parameters."""
        saved = ctx.saved_tensors
        g32 = grad_output.float()
        grad_input = grad_weight = grad_bias = None
        if ctx.fallback:
            x32, w32 = saved
            if not torch.isfinite(g32).all():
                grad_input = torch.zeros_like(x32, dtype=ctx.x_dtype)
                grad_weight = torch.zeros_like(w32, dtype=ctx.w_dtype)
                if ctx.has_bias:
                    grad_bias = torch.zeros(w32.shape[0], device=g32.device,
                                            dtype=ctx.w_dtype)
                return grad_input, grad_weight, grad_bias, None, None, None, None, None
            grad_input = (g32 @ w32).to(ctx.x_dtype)
            grad_weight = g32.reshape(-1, g32.shape[-1]).t() @ x32.reshape(-1, x32.shape[-1])
            if ctx.has_bias:
                grad_bias = g32.reshape(-1, g32.shape[-1]).sum(dim=0)
            return grad_input, grad_weight, grad_bias, None, None, None, None, None

        x_q, w_q, x_scale, w_scale = saved
        if not torch.isfinite(g32).all():
            grad_input = torch.zeros_like(x_q, dtype=ctx.x_dtype)
            grad_weight = torch.zeros_like(w_q, dtype=ctx.w_dtype)
            if ctx.has_bias:
                grad_bias = torch.zeros(w_q.shape[0], device=g32.device,
                                        dtype=ctx.w_dtype)
            return grad_input, grad_weight, grad_bias, None, None, None, None, None

        x_dq = x_q.float() / x_scale
        w_dq = w_q.float() / w_scale
        flat_g = g32.reshape(-1, g32.shape[-1])
        # Keep the E5M2 gradient operand format, but scale each token
        # independently so extra padded rows cannot alter valid-token grads.
        if ctx.e5m2_for_grads:
            g_amax = flat_g.abs().amax(dim=-1, keepdim=True)
            g_scale = torch.where(
                g_amax > 0,
                (E5M2_MAX * 0.9) / g_amax.clamp_min(1e-12),
                torch.ones_like(g_amax),
            )
            g_q = (flat_g * g_scale).clamp(-E5M2_MAX, E5M2_MAX).to(E5M2_DTYPE)
            g_dq = (g_q.float() / g_scale).reshape(g32.shape)
        else:  # pragma: no cover - ablation path
            g_dq = g32
        g2d = g_dq.reshape(-1, g_dq.shape[-1])
        x2d = x_dq.reshape(-1, x_dq.shape[-1])
        grad_input = (g2d @ w_dq).reshape(x_dq.shape).to(ctx.x_dtype)
        grad_weight = g2d.t() @ x2d
        if ctx.has_bias:
            grad_bias = g2d.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class HybridFP8Linear(nn.Module):
    """Drop-in linear layer: stateless E4M3 forward, E5M2 backward, FP32 masters.

    Args:
        in_features, out_features, bias: as nn.Linear.
        out_dtype: compute/output dtype for activations (bfloat16 default).
        history_len: accepted for API compatibility; scaling is stateless.
        e5m2_for_grads: False bypasses E5M2 gradient quantization (ablation).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 out_dtype: torch.dtype = torch.bfloat16,
                 history_len: int = 16, e5m2_for_grads: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.e5m2_for_grads = bool(e5m2_for_grads)
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        self.in_scaler = DelayedScaler(E4M3_MAX, history_len)
        self.wt_scaler = DelayedScaler(E4M3_MAX, history_len)
        self.grad_scaler = DelayedScaler(E5M2_MAX, history_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HybridFP8Matmul.apply(
            x, self.weight, self.bias,
            self.in_scaler, self.wt_scaler, self.grad_scaler,
            self.out_dtype, self.e5m2_for_grads,
        )

    def extra_repr(self) -> str:
        """Describe this layer's dimensions and bias configuration."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, out_dtype={self.out_dtype}")

def enable_hybrid_fp8_linears(module: nn.Module, out_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Replace every nn.Linear below module with hybrid E4M3/E5M2 execution.

    Parameter names and shapes remain unchanged for checkpoint compatibility.
    Master parameters stay FP32; activations retain the requested output dtype.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, HybridFP8Linear):
            continue
        if isinstance(child, nn.Linear):
            # Preserve the caller's RNG stream: replacement weights are copied
            # from the initialized layer, so throwaway initialization is noise.
            with torch.random.fork_rng(devices=[]):
                replacement = HybridFP8Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, out_dtype=out_dtype,
                )
            with torch.no_grad():
                replacement.weight.copy_(child.weight.detach().float())
                if child.bias is not None:
                    replacement.bias.copy_(child.bias.detach().float())
            setattr(module, name, replacement)
        else:
            enable_hybrid_fp8_linears(child, out_dtype=out_dtype)
    return module
