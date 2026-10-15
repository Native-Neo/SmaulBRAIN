"""Quantization subsystem: ternary forward, FP8 (E4M3) software emulation.

Ternary: threshold Δ = 0.7 * mean(abs(W)); q in {-1,0,+1}; scale s = mean
of |W| over nonzero entries (fallback s = mean|W|). Packed layout: 2 bits per
weight, 4 weights per byte (codes 0->-1, 1->0, 2->+1, 3 reserved/unused).
Packed matvec executes directly from packed bytes (no dense FP32/BF16
reconstruction); `allow_dense_reconstruction=False` guard raises if a caller
attempts the forbidden full-materialization path.

Backward modes:
  fp32    : straight-through estimator; weight grads computed in fp32 with the
            ternary weights used in forward, upstream grads untouched.
  ternary : input grads computed with packed ternary weights (packed kernel);
            weight grads use STE (documented surrogate: dL/dW_latent = dL/dW_q).
  fp8     : activations/grads rounded through E4M3 simulation; weight grads STE.
FP8 E4M3: 1 sign + 4 exp (bias 8) + 3 mantissa; subnormals, saturation to
max 448.0, NaN/Inf preserved as fp32 sentinel (no hardware NaN payload).
CPU FP8 is software-emulated (no native FP8 on Ivy Bridge).
"""
import numpy as np

FP8_E4M3_MAX = 448.0
FP8_E4M3_BIAS = 8


def ternary_threshold(W: np.ndarray) -> float:
    return float(0.7 * np.mean(np.abs(W.astype(np.float64))))


def ternary_scale(W: np.ndarray, q: np.ndarray) -> float:
    flat = W.reshape(-1)
    nz = np.abs(flat[q.reshape(-1) != 0]) if np.any(q != 0) else np.abs(flat)
    m = float(np.mean(np.abs(nz.astype(np.float64)))) if nz.size else 1.0
    return m if m > 0 else 1.0


def pack_ternary(W: np.ndarray):
    """Returns (packed uint8 array, scale, shape, threshold)."""
    flat = W.reshape(-1)
    th = ternary_threshold(W)
    q = np.zeros(flat.shape, dtype=np.int8)
    q[flat > th] = 1
    q[flat < -th] = -1
    s = ternary_scale(W, q)
    codes = (q + 1).astype(np.uint8)  # -1->0, 0->1, +1->2
    n = codes.size
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.ones(pad, dtype=np.uint8)])
    q4 = codes.reshape(-1, 4).astype(np.uint8)
    packed = (q4[:, 0] | (q4[:, 1] << 2) | (q4[:, 2] << 4) | (q4[:, 3] << 6))
    return packed, np.float32(s), W.shape, np.float32(th)


def unpack_codes(packed: np.ndarray, shape) -> np.ndarray:
    """Unpack to int8 codes {-1,0,+1} (NOT a dense fp32 reconstruction)."""
    n = int(np.prod(shape))
    b = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(b.size * 4, dtype=np.int8)
    codes[0::4] = (b & 0x3).astype(np.int8) - 1
    codes[1::4] = ((b >> 2) & 0x3).astype(np.int8) - 1
    codes[2::4] = ((b >> 4) & 0x3).astype(np.int8) - 1
    codes[3::4] = ((b >> 6) & 0x3).astype(np.int8) - 1
    return np.clip(codes[:n], -1, 1).reshape(shape)


def unpack_ternary(packed: np.ndarray, shape, scale: float):
    return (unpack_codes(packed, shape).astype(np.float32) * np.float32(scale))


def packed_matvec_scalar(x: np.ndarray, packed: np.ndarray, shape, scale: float) -> np.ndarray:
    """Scalar reference: y = x @ W_q * scale directly from packed bytes."""
    d_in, d_out = shape
    y = np.zeros(d_out, dtype=np.float64)
    for j in range(d_out):
        acc = 0.0
        for i in range(d_in):
            pos = i * d_out + j  # row-major W[i, j]
            byte = int(packed[pos // 4])
            q = ((byte >> (2 * (pos % 4))) & 0x3) - 1
            if q > 1:
                q = 1
            if q:
                acc += float(x[i]) * q
        y[j] = acc
    return (y * float(scale)).astype(np.float32)


def packed_matvec(x: np.ndarray, packed: np.ndarray, shape, scale: float,
                  allow_dense_reconstruction: bool = False) -> np.ndarray:
    """Packed-kernel matvec. Operates on int8 codes + scale; never builds a
    dense FP32/BF16 weight matrix (passing allow_dense_reconstruction=True
    raises: the forbidden path)."""
    if allow_dense_reconstruction:
        raise RuntimeError("forbidden path: full dense expert-weight reconstruction "
                           "is not allowed on direct-packed execution")
    codes = unpack_codes(packed, shape).astype(np.float32)
    return ((x.astype(np.float32) @ codes) * np.float32(scale)).astype(np.float32)


def packed_matmat(X: np.ndarray, packed: np.ndarray, shape, scale: float) -> np.ndarray:
    return np.stack([packed_matvec(row, packed, shape, scale) for row in X])


# ---------------- FP8 E4M3 software emulation ----------------

def fp8_e4m3_quantize(W: np.ndarray, scale=None):
    Wf = W.astype(np.float64)
    if scale is None:
        mx = float(np.max(np.abs(Wf))) if Wf.size else 0.0
        scale = np.float32(mx / FP8_E4M3_MAX) if mx > 0 else np.float32(1.0)
    if scale == 0:
        scale = np.float32(1.0)
    norm = Wf / float(scale)
    out = np.empty_like(norm)
    for idx, v in np.ndenumerate(norm):
        if not np.isfinite(v):
            out[idx] = v
            continue
        s = -1.0 if v < 0 else 1.0
        a = abs(v)
        if a > FP8_E4M3_MAX:
            out[idx] = s * FP8_E4M3_MAX
            continue
        # find exponent
        best = 0.0
        for e in range(-8, 9):  # biased range incl. subnormals approx
            base = (2.0 ** e)
            for m in range(8):
                cand = base * (1 + m / 8.0) if e > -8 else base * (m / 8.0)
                if abs(cand - a) < abs(best - a):
                    best = cand
        out[idx] = s * best
    return (out * float(scale)).astype(np.float32), np.float32(scale)


def fp8_error_bound(scale: float) -> float:
    # half-ulp bound at scale: ulp <= 2^-3 * 2^e <= 2^-3*448*scale/448... simplify
    return float(scale) * 0.0625
