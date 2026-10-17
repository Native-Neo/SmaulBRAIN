"""PyTorch quantization API with an optional packed C++ CPU backend.

All trainable weights and activations remain torch.Tensor objects. The native
backend uses a narrow CPU buffer bridge for packed ternary matrix operations;
it never stores a dense dequantized weight matrix.
"""
import math

import torch
from . import _quant_native as _native

FP8_E4M3_MAX = _native.FP8_E4M3_MAX
TiledPacked = _native.TiledPacked
prepare_ternary_backend = _native.prepare_ternary_backend
ternary_backend_name = _native.ternary_backend_name
recurrent_scan = _native.recurrent_scan
recurrent_backward_scan = _native.recurrent_backward_scan


def _tensor(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor): return x.to(device=device or x.device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _np(x, dtype=None):
    a = x.detach().to("cpu").contiguous()
    if dtype is not None: a = a.to(dtype=dtype)
    return a.numpy()


def ternary_threshold(W):
    W = _tensor(W); return float(0.7 * W.abs().mean().item())


def ternary_scale(W, q):
    W = _tensor(W); q = _tensor(q, device=W.device, dtype=torch.int8)
    nz = W.abs()[q.reshape(-1) != 0] if bool((q != 0).any()) else W.abs().reshape(-1)
    scale = float(nz.double().mean().item()) if nz.numel() else 1.0
    return scale if scale > 0 else 1.0


def _pack_ternary_numpy(W):
    packed, scale, shape, th = _native._pack_ternary_numpy(_np(_tensor(W)))
    return packed, float(scale), shape, float(th)


def pack_ternary(W):
    a = _np(_tensor(W))
    packed, scale, shape, threshold = _native.pack_ternary(a)
    return torch.from_numpy(packed.copy()), float(scale), tuple(shape), float(threshold)


def pack_ternary_tiled(W):
    a = _np(_tensor(W))
    packed, scale, shape, threshold = _native.pack_ternary_tiled(a)
    if isinstance(packed, TiledPacked):
        packed = TiledPacked(packed.buf.copy())
    else:
        packed = torch.from_numpy(packed.copy())
    return packed, float(scale), tuple(shape), float(threshold)


def unpack_codes(packed, shape):
    if isinstance(packed, TiledPacked):
        # Native tiled layout is [column tile][input row][4 bitplane bytes]:
        # pos-low8, neg-low8, pos-high8, neg-high8. Each bit is one column.
        # Decode to row-major signed trits only for explicit unpack requests.
        din, dout = (int(shape[0]), int(shape[1]))
        tiles = (dout + 15) // 16
        buf = packed.buf
        expected = tiles * din * 4
        if din < 1 or dout < 1 or buf.size != expected:
            raise ValueError("invalid tiled packed buffer for declared shape")
        import numpy as np
        buf = np.asarray(buf, dtype=np.uint8).reshape(-1)
        out = np.empty((din, dout), dtype=np.int8)
        columns = np.arange(dout, dtype=np.int64)
        tile = columns // 16
        local = columns % 16
        block = (local // 8) * 2
        bit = local % 8
        row_offsets = tile * din * 4 + block
        for i in range(din):
            base = row_offsets + i * 4
            positive = (buf[base] >> bit) & 1
            negative = (buf[base + 1] >> bit) & 1
            out[i] = np.where(positive != 0, 1, np.where(negative != 0, -1, 0))
        return torch.from_numpy(out)
    p = packed
    if isinstance(p, torch.Tensor): p = _np(p, torch.uint8)
    return torch.from_numpy(_native.unpack_codes(p, shape).copy())


def unpack_ternary(packed, shape, scale):
    return unpack_codes(packed, shape).float() * float(scale)


def packed_matvec_scalar(x, packed, shape, scale):
    return packed_matmat_scalar(x, packed, shape, scale)


def packed_matmat_scalar(X, packed, shape, scale):
    x = _tensor(X).detach().cpu().contiguous().numpy()
    if isinstance(packed, TiledPacked):
        # The scalar reference only accepts row-major packed bytes. Preserve
        # correctness for tiled inputs by using the native tiled kernel.
        result = _native.packed_matmat(x, packed, shape, scale, False)
    else:
        p = _np(packed, torch.uint8) if isinstance(packed, torch.Tensor) else packed
        result = _native.packed_matmat_scalar(x, p, shape, scale)
    return torch.from_numpy(result.copy()).to(_tensor(X).device)


def packed_matmat(X, packed, shape, scale, allow_dense_reconstruction=False):
    if allow_dense_reconstruction:
        raise RuntimeError("forbidden path: dense ternary reconstruction is not allowed")
    x = _tensor(X)
    # Preserve TiledPacked as a tagged object: removing the tag makes aligned
    # tiled buffers indistinguishable from row-major packed buffers and yields
    # incorrect outputs. Until a CUDA packed kernel exists, accelerator inputs
    # use the direct packed CPU kernel and transfer only activations/results;
    # they never expand the full weight matrix to FP32.
    packed_arg = packed
    if not isinstance(packed, TiledPacked) and isinstance(packed, torch.Tensor):
        packed_arg = _np(packed, torch.uint8)
    y = _native.packed_matmat(_np(x), packed_arg, shape, scale, False)
    return torch.from_numpy(y.copy()).to(device=x.device, dtype=x.dtype)


def packed_matmat_transpose_scalar(dY, packed, shape, scale):
    dy = _np(_tensor(dY))
    if isinstance(packed, TiledPacked):
        out = _native.packed_matmat_transpose(dy, packed, shape, scale)
    else:
        p = _np(packed, torch.uint8) if isinstance(packed, torch.Tensor) else packed
        out = _native.packed_matmat_transpose_scalar(dy, p, shape, scale)
    return torch.from_numpy(out.copy()).to(_tensor(dY).device)


def packed_matmat_transpose(dY, packed, shape, scale):
    dy = _tensor(dY)
    packed_arg = packed
    if not isinstance(packed, TiledPacked) and isinstance(packed, torch.Tensor):
        packed_arg = _np(packed, torch.uint8)
    out = _native.packed_matmat_transpose(_np(dy), packed_arg, shape, scale)
    return torch.from_numpy(out.copy()).to(device=dy.device, dtype=dy.dtype)


def packed_matvec(x, packed, shape, scale, allow_dense_reconstruction=False):
    return packed_matmat(x, packed, shape, scale, allow_dense_reconstruction)


def fp8_e4m3_quantize(W, scale=None):
    W = _tensor(W)
    table = torch.as_tensor(_native._FP8_POSITIVE_VALUES.copy(), device=W.device, dtype=torch.float64)
    wf = W.double()
    if scale is None:
        finite = wf[torch.isfinite(wf)].abs()
        mx = float(finite.max().item()) if finite.numel() else 0.0
        scale = mx / FP8_E4M3_MAX if mx > 0 else 1.0
    requested_scale = float(scale)
    if not math.isfinite(requested_scale) or requested_scale < 0:
        raise ValueError("FP8 quantization scale must be finite and >= 0")
    scale = float(torch.tensor(requested_scale, dtype=torch.float32).item())
    if not math.isfinite(scale):
        raise ValueError("FP8 quantization scale is outside the float32 range")
    if scale == 0:
        # Preserve the documented legacy behavior for scale=0 and for a
        # positive value too small to survive float32 conversion.
        scale = 1.0
    norm = wf / scale
    finite = torch.isfinite(norm)
    mag = torch.where(finite, norm.abs().clamp(max=FP8_E4M3_MAX), torch.zeros_like(norm))
    hi = torch.searchsorted(table, mag.contiguous()).clamp(max=table.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    lower, upper = table[lo], table[hi]
    nearest = torch.where((upper - mag) < (mag - lower), upper, lower)
    sign = torch.where(norm < 0, -1.0, 1.0)
    out = torch.where(finite, sign * nearest, norm) * scale
    return out.to(torch.float32), scale


def fp8_error_bound(scale: float) -> float:
    return abs(float(scale)) * (FP8_E4M3_MAX / 16.0)
