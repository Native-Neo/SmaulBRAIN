"""PyTorch quantization API with an optional packed C++ CPU backend.

All trainable weights and activations remain torch.Tensor objects. The native
backend uses a narrow CPU buffer bridge for packed ternary matrix operations;
it never stores a dense dequantized weight matrix.
"""
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
    p = packed.buf if isinstance(packed, TiledPacked) else packed
    if isinstance(p, torch.Tensor): p = _np(p, torch.uint8)
    return torch.from_numpy(_native.unpack_codes(p, shape).copy())


def unpack_ternary(packed, shape, scale):
    return unpack_codes(packed, shape).float() * float(scale)


def packed_matvec_scalar(x, packed, shape, scale):
    return packed_matmat_scalar(x, packed, shape, scale)


def packed_matmat_scalar(X, packed, shape, scale):
    p = packed.buf if isinstance(packed, TiledPacked) else packed
    if isinstance(p, torch.Tensor): p = _np(p, torch.uint8)
    x = _tensor(X).detach().cpu().contiguous().numpy()
    return torch.from_numpy(_native.packed_matmat_scalar(x, p, shape, scale).copy()).to(_tensor(X).device)


def packed_matmat(X, packed, shape, scale, allow_dense_reconstruction=False):
    if allow_dense_reconstruction:
        raise RuntimeError("forbidden path: dense ternary reconstruction is not allowed")
    x = _tensor(X)
    p = packed.buf if isinstance(packed, TiledPacked) else packed
    if isinstance(p, torch.Tensor): p = _np(p, torch.uint8)
    if x.device.type == "cpu":
        y = _native.packed_matmat(_np(x), p, shape, scale, False)
        return torch.from_numpy(y.copy()).to(dtype=x.dtype)
    # Accelerator fallback: decode only to signed ternary codes for the matmul;
    # no latent FP32/FP8 master weight is reconstructed or cached.
    codes = unpack_codes(p, shape).to(device=x.device, dtype=torch.int8)
    return (x @ codes.to(x.dtype)) * float(scale)


def packed_matmat_transpose_scalar(dY, packed, shape, scale):
    p = packed.buf if isinstance(packed, TiledPacked) else packed
    if isinstance(p, torch.Tensor): p = _np(p, torch.uint8)
    dy = _np(_tensor(dY))
    out = _native.packed_matmat_transpose_scalar(dy, p, shape, scale)
    return torch.from_numpy(out.copy()).to(_tensor(dY).device)


def packed_matmat_transpose(dY, packed, shape, scale):
    dy = _tensor(dY)
    p = packed.buf if isinstance(packed, TiledPacked) else packed
    if isinstance(p, torch.Tensor): p = _np(p, torch.uint8)
    if dy.device.type == "cpu":
        out = _native.packed_matmat_transpose(_np(dy), p, shape, scale)
        return torch.from_numpy(out.copy()).to(dtype=dy.dtype)
    codes = unpack_codes(p, shape).to(device=dy.device, dtype=torch.int8)
    return (dy @ codes.to(dy.dtype).T) * float(scale)


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
    scale = float(torch.tensor(float(scale), dtype=torch.float32).item()) or 1.0
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
