"""Hybrid FP8 mixed-precision execution (E4M3 forward, E5M2 backward).

Study-only module (like kernels.SparseTopKHead): not wired into the
default model path, which uses precision.py FP8 storage + FP32 compute.
Kept for the E4M3/E5M2 simulation study and its dedicated test
(tests/test_hybrid_fp8.py); do not import from training/inference code
unless opting into the experiment explicitly.

Design: forward activations/weights quantize to ``torch.float8_e4m3fn``
(highest forward precision); incoming gradients quantize to
``torch.float8_e5m2`` in backward (wide exponent range kills
underflow/overflow). Master weights and all accumulators stay FP32.

Execution is simulated-portable: tensors genuinely pass through FP8
code points (quantization effects are real), while the matmul itself runs
in higher precision after dequantization unless ``torch._scaled_mm`` is
available (CUDA compute capability >= 9). No code path requires FP8
matmul support, so CPU and older GPUs fall back safely.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn

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
    """Linear forward in E4M3, gradients in E5M2 (see module docstring)."""

    @staticmethod
    def forward(ctx, x, w, bias, in_scaler, wt_scaler, grad_scaler,
                out_dtype, e5m2_for_grads):
        x32, w32 = x.float(), w.float()
        ctx.out_dtype = out_dtype
        ctx.e5m2_for_grads = bool(e5m2_for_grads)
        ctx.in_scaler, ctx.wt_scaler, ctx.grad_scaler = in_scaler, wt_scaler, grad_scaler
        ctx.has_bias = bias is not None
        ctx.x_dtype, ctx.w_dtype = x.dtype, w.dtype
        if not (torch.isfinite(x32).all() and torch.isfinite(w32).all()
                and (bias is None or torch.isfinite(bias.float()).all())):
            # Poisoned inputs: bypass quantization entirely (loud-safe math).
            ctx.fallback = True
            ctx.save_for_backward(x32, w32)
            out = x32 @ w32.t()
            if bias is not None:
                out = out + bias.float()
            return out.to(out_dtype)
        ctx.fallback = False
        in_scale = in_scaler.update(_amax(x32))
        wt_scale = wt_scaler.update(_amax(w32))
        x_q = _quantize_to_fp8(x32, in_scale, E4M3_DTYPE, E4M3_MAX)
        w_q = _quantize_to_fp8(w32, wt_scale, E4M3_DTYPE, E4M3_MAX)
        ctx.save_for_backward(x_q, w_q,
                              torch.tensor(in_scale), torch.tensor(wt_scale))
        native = _try_scaled_mm(x_q, w_q, torch.float32)
        if native is not None:
            out = native / (in_scale * wt_scale)
        else:
            out = _dequantize_from_fp8(x_q, in_scale) @ _dequantize_from_fp8(w_q, wt_scale).t()
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_q, w_q, in_scale_t, wt_scale_t = ctx.saved_tensors
        in_scale, wt_scale = float(in_scale_t), float(wt_scale_t)
        g32 = grad_output.float()
        grad_input = grad_weight = grad_bias = None
        if ctx.fallback or not torch.isfinite(g32).all():
            # Fallback path, or poisoned grads: standard FP32 grads, or zeros
            # when the incoming gradient itself is non-finite (skip, loudly safe).
            if not torch.isfinite(g32).all():
                dev = grad_output.device
                x_shape = list(x_q.shape)
                w_shape = list(w_q.shape)
                grad_input = torch.zeros(x_shape, device=dev, dtype=ctx.x_dtype)
                grad_weight = torch.zeros(w_shape, device=dev, dtype=ctx.w_dtype)
                if ctx.has_bias:
                    grad_bias = torch.zeros(w_shape[0], device=dev, dtype=ctx.w_dtype)
                return grad_input, grad_weight, grad_bias, None, None, None, None, None
            x32, w32 = x_q, w_q  # fallback saved plain tensors under these names
            grad_input = (g32 @ w32).to(ctx.x_dtype)
            grad_weight = g32.reshape(-1, g32.shape[-1]).t() @ x32.reshape(-1, x32.shape[-1])
            if ctx.has_bias:
                grad_bias = g32.reshape(-1, g32.shape[-1]).sum(dim=0)
            return grad_input, grad_weight, grad_bias, None, None, None, None, None
        g_scale = ctx.grad_scaler.update(_amax(g32))
        if ctx.e5m2_for_grads:
            g_q = _quantize_to_fp8(g32, g_scale, E5M2_DTYPE, E5M2_MAX)
        else:  # pragma: no cover - opt-out path for ablation studies
            g_q = _quantize_to_fp8(g32, g_scale, E4M3_DTYPE, E4M3_MAX)
        g_dq = _dequantize_from_fp8(g_q, g_scale)
        x_dq = _dequantize_from_fp8(x_q, in_scale)
        w_dq = _dequantize_from_fp8(w_q, wt_scale)
        g2d = g_dq.reshape(-1, g_dq.shape[-1])
        x2d = x_dq.reshape(-1, x_dq.shape[-1])
        grad_input = (g2d @ w_dq).reshape(x_dq.shape).to(ctx.x_dtype)
        grad_weight = g2d.t() @ x2d
        if ctx.has_bias:
            grad_bias = g2d.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class HybridFP8Linear(nn.Module):
    """Drop-in linear layer: E4M3 forward, E5M2 backward, FP32 masters.

    Args:
        in_features, out_features, bias: as nn.Linear.
        out_dtype: compute/output dtype for activations (bfloat16 default).
        history_len: delayed-scaler amax window per tensor.
        e5m2_for_grads: False forces E4M3 also in backward (ablation).
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
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, out_dtype={self.out_dtype}")
