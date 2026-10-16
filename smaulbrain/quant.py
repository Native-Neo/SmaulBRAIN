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

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from . import cpu as _cpu

FP8_E4M3_MAX = 448.0
FP8_E4M3_BIAS = 8


def ternary_threshold(W: np.ndarray) -> float:
    # Accumulate in float64 without materializing a second float64 copy of W.
    return float(0.7 * np.mean(np.abs(W), dtype=np.float64))


def ternary_scale(W: np.ndarray, q: np.ndarray) -> float:
    flat = W.reshape(-1)
    nz = np.abs(flat[q.reshape(-1) != 0]) if np.any(q != 0) else np.abs(flat)
    m = float(np.mean(nz, dtype=np.float64)) if nz.size else 1.0
    return m if m > 0 else 1.0


def _pack_ternary_numpy(W: np.ndarray):
    """Portable NumPy reference/fallback for pack_ternary()."""
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


def pack_ternary(W: np.ndarray):
    """Pack weights using native code when possible, with a NumPy reference."""
    arr = np.asarray(W)
    if (arr.dtype == np.float32 and arr.flags.c_contiguous and arr.size > 0
            and prepare_ternary_backend()):
        n = int(arr.size)
        packed = np.empty((n + 3) // 4, dtype=np.uint8)
        scale_out = ctypes.c_float(1.0)
        threshold_out = ctypes.c_float(0.0)
        status = _NATIVE_LIB.smaulbrain_pack_ternary(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n, ctypes.byref(scale_out), ctypes.byref(threshold_out))
        if status != 0:
            raise RuntimeError(f"native ternary packing failed with status {status}")
        return packed, np.float32(scale_out.value), arr.shape, np.float32(threshold_out.value)
    return _pack_ternary_numpy(arr)


class TiledPacked:
    """Native tile-contiguous ternary layout (see _ternary_kernels.cpp).

    Only produced/consumed by the native backend; the canonical row-major
    2-bit format from pack_ternary() stays the portable/reference format.
    """
    __slots__ = ("buf",)

    def __init__(self, buf: np.ndarray):
        self.buf = buf

    @property
    def nbytes(self) -> int:
        return int(self.buf.nbytes)


def pack_ternary_tiled(W: np.ndarray):
    """pack_ternary() then retile for the fast native kernels when available."""
    arr = np.asarray(W)
    if (arr.dtype == np.float32 and arr.ndim == 2 and arr.flags.c_contiguous
            and arr.size > 0 and prepare_ternary_backend()):
        d_in, d_out = int(arr.shape[0]), int(arr.shape[1])
        buf = np.empty(((d_out + 15) // 16) * d_in * 4, dtype=np.uint8)
        scale_out = ctypes.c_float(1.0)
        threshold_out = ctypes.c_float(0.0)
        if _NATIVE_LIB.smaulbrain_pack_ternary_tiled(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                d_in, d_out, ctypes.byref(scale_out), ctypes.byref(threshold_out)) == 0:
            return (TiledPacked(buf), np.float32(scale_out.value), arr.shape,
                    np.float32(threshold_out.value))
    packed, scale, shape, th = pack_ternary(W)
    if len(shape) == 2 and prepare_ternary_backend():
        d_in, d_out = int(shape[0]), int(shape[1])
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        buf = np.empty(((d_out + 15) // 16) * d_in * 4, dtype=np.uint8)
        if _NATIVE_LIB.smaulbrain_retile_ternary(
                packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                d_in, d_out) == 0:
            return TiledPacked(buf), scale, shape, th
    return packed, scale, shape, th


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
    """Scalar reference: y = x @ W_q * scale, decoding each trit in place."""
    d_in, d_out = shape
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != d_in:
        raise ValueError(f"input width {x.size} does not match packed matrix width {d_in}")
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if packed.size < (d_in * d_out + 3) // 4:
        raise ValueError("packed ternary buffer is shorter than the declared shape")
    y = np.zeros(d_out, dtype=np.float64)
    for j in range(d_out):
        acc = 0.0
        for i in range(d_in):
            pos = i * d_out + j
            code = (int(packed[pos >> 2]) >> (2 * (pos & 3))) & 3
            q = -1 if code == 0 else (1 if code >= 2 else 0)
            if q:
                acc += float(x[i]) * q
        y[j] = acc
    return (y * float(scale)).astype(np.float32)


_NATIVE_LIB = None
_NATIVE_ATTEMPTED = False
_NATIVE_ERROR = None
_NATIVE_VARIANT = "unavailable"


def prepare_ternary_backend() -> bool:
    """Build/load a baseline-CPU C++ packed-trit kernel, cached between runs.

    No -march=native or AVX requirement is used. If a compiler is unavailable,
    operations remain correct through the packed scalar reference path.
    """
    global _NATIVE_LIB, _NATIVE_ATTEMPTED, _NATIVE_ERROR, _NATIVE_VARIANT
    if _NATIVE_LIB is not None:
        return True
    if _NATIVE_ATTEMPTED:
        return False
    _NATIVE_ATTEMPTED = True
    source = Path(__file__).with_name("_ternary_kernels.cpp")
    if not source.is_file():
        _NATIVE_ERROR = f"native kernel source missing: {source}"
        return False
    compiler_name = os.environ.get("CXX", "g++")
    compiler = shutil.which(compiler_name)
    if compiler is None:
        compiler = shutil.which("c++")
    if compiler is None:
        _NATIVE_ERROR = "no C++ compiler found (tried CXX and c++)"
        return False

    try:
        source_bytes = source.read_bytes()
        # Compile an AVX1 variant only if the running OS/CPU advertises AVX.
        # Ivy Bridge supports AVX but not AVX2. Never use -march=native.
        has_avx = bool(_cpu.detect().get("has_avx", False))
        variant_flag = "-mavx" if has_avx else "-msse2"
        flags = ["-O3", "-std=c++17", "-shared", "-fPIC",
                 "-fno-fast-math", "-ffp-contract=off", "-pthread", variant_flag]
        digest_data = source_bytes + compiler.encode("utf-8") + " ".join(flags).encode("ascii")
        digest = hashlib.sha256(digest_data).hexdigest()[:20]
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = cache_home / "smaulbrain"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"ternary-{variant_flag[1:]}-{digest}.so"
        if not target.is_file():
            temporary = cache_dir / f"ternary-{variant_flag[1:]}-{digest}-{os.getpid()}.tmp.so"
            command = [compiler, *flags, str(source), "-o", str(temporary)]
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, timeout=90)
            if result.returncode != 0:
                _NATIVE_ERROR = result.stderr.strip() or "C++ ternary kernel compilation failed"
                try:
                    temporary.unlink()
                except OSError:
                    pass
                return False
            try:
                os.replace(temporary, target)
            except OSError:
                if not target.is_file():
                    raise
                try:
                    temporary.unlink()
                except OSError:
                    pass
        lib = ctypes.CDLL(str(target))
        ptr_f32 = ctypes.POINTER(ctypes.c_float)
        ptr_u8 = ctypes.POINTER(ctypes.c_uint8)
        argtypes = [ptr_f32, ptr_u8, ctypes.c_float, ptr_f32,
                    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.smaulbrain_ternary_matmat.argtypes = argtypes
        lib.smaulbrain_ternary_matmat.restype = ctypes.c_int
        lib.smaulbrain_ternary_transpose_matmat.argtypes = argtypes
        lib.smaulbrain_ternary_transpose_matmat.restype = ctypes.c_int
        lib.smaulbrain_tiled_matmat.argtypes = argtypes
        lib.smaulbrain_tiled_matmat.restype = ctypes.c_int
        lib.smaulbrain_tiled_transpose_matmat.argtypes = argtypes
        lib.smaulbrain_tiled_transpose_matmat.restype = ctypes.c_int
        lib.smaulbrain_retile_ternary.argtypes = [
            ptr_u8, ptr_u8, ctypes.c_size_t, ctypes.c_size_t]
        lib.smaulbrain_retile_ternary.restype = ctypes.c_int
        lib.smaulbrain_pack_ternary_tiled.argtypes = [
            ptr_f32, ptr_u8, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.smaulbrain_pack_ternary_tiled.restype = ctypes.c_int
        lib.smaulbrain_pack_ternary.argtypes = [
            ptr_f32, ptr_u8, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.smaulbrain_pack_ternary.restype = ctypes.c_int
        lib.smaulbrain_recurrent_scan.argtypes = [
            ptr_f32, ptr_f32, ptr_f32, ptr_f32,
            ctypes.c_size_t, ctypes.c_size_t]
        lib.smaulbrain_recurrent_scan.restype = ctypes.c_int
        ptr_f64 = ctypes.POINTER(ctypes.c_double)
        lib.smaulbrain_recurrent_backward_scan.argtypes = [
            ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f64,
            ctypes.c_size_t, ctypes.c_size_t]
        lib.smaulbrain_recurrent_backward_scan.restype = ctypes.c_int
        _NATIVE_LIB = lib
        _NATIVE_VARIANT = "avx1" if has_avx else "sse2"
        _NATIVE_ERROR = None
        return True
    except Exception as exc:
        _NATIVE_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def recurrent_scan(kv: np.ndarray, alpha: np.ndarray, state: np.ndarray,
                   states: np.ndarray) -> bool:
    """Run the native time scan when the packed C++ backend is already loaded.

    Returns False without changing the buffers if no native backend is available
    or inputs need conversion. The caller can then use its portable reference loop.
    """
    if _NATIVE_LIB is None:
        return False
    arrays = (kv, alpha, state, states)
    if any(not isinstance(a, np.ndarray) or a.dtype != np.float32
           or not a.flags.c_contiguous for a in arrays):
        return False
    if kv.ndim != 2 or states.shape != kv.shape:
        raise ValueError("recurrent scan expects KV/states with matching (T, d) shapes")
    if alpha.shape != (kv.shape[1],) or state.shape != alpha.shape:
        raise ValueError("recurrent scan alpha/state widths must match KV")
    if kv.shape[0] == 0:
        return True
    if not state.flags.writeable or not states.flags.writeable:
        return False
    status = _NATIVE_LIB.smaulbrain_recurrent_scan(
        kv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kv.shape[0], kv.shape[1])
    if status != 0:
        raise RuntimeError(f"native recurrent scan failed with status {status}")
    return True


def recurrent_backward_scan(dS: np.ndarray, states: np.ndarray,
                            alpha: np.ndarray, dKV: np.ndarray,
                            dalpha: np.ndarray) -> bool:
    """Run native recurrent BPTT scans when the C++ backend is already loaded.

    Buffers must be preallocated contiguous arrays with float32 activations and
    a float64 dalpha accumulator. Returns False without modifying them if no
    native backend is loaded or conversion would be required.
    """
    if _NATIVE_LIB is None:
        return False
    arrays = (dS, states, alpha, dKV)
    if any(not isinstance(a, np.ndarray) or a.dtype != np.float32
           or not a.flags.c_contiguous for a in arrays):
        return False
    if not isinstance(dalpha, np.ndarray) or dalpha.dtype != np.float64 or not dalpha.flags.c_contiguous:
        return False
    if dS.ndim != 2 or states.shape != dS.shape or dKV.shape != dS.shape:
        raise ValueError("recurrent backward expects matching (T, d) arrays")
    T, d = dS.shape
    if alpha.shape != (d,) or dalpha.shape != (d,):
        raise ValueError("recurrent backward alpha/dalpha widths must match activations")
    if T == 0:
        dalpha.fill(0.0)
        return True
    if not dKV.flags.writeable or not dalpha.flags.writeable:
        return False
    status = _NATIVE_LIB.smaulbrain_recurrent_backward_scan(
        dS.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dKV.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dalpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        T, d)
    if status != 0:
        raise RuntimeError(f"native recurrent backward scan failed with status {status}")
    return True


def ternary_backend_name() -> str:
    if not prepare_ternary_backend():
        return "python-packed-reference"
    return f"native-cpp-packed-{_NATIVE_VARIANT}"


def _check_packed_shape(packed: np.ndarray, shape):
    d_in, d_out = (int(shape[0]), int(shape[1]))
    if d_in < 1 or d_out < 1:
        raise ValueError(f"invalid packed matrix shape {shape}")
    packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    need = (d_in * d_out + 3) // 4
    if packed.size < need:
        raise ValueError(f"packed buffer has {packed.size} bytes, needs at least {need}")
    return packed, d_in, d_out


def packed_matmat_scalar(X: np.ndarray, packed: np.ndarray, shape,
                         scale: float) -> np.ndarray:
    """Portable reference for X @ packed_ternary(W) without dense W creation."""
    x = np.asarray(X, dtype=np.float32)
    if x.ndim == 1:
        return packed_matvec_scalar(x, packed, shape, scale)
    if x.ndim != 2:
        raise ValueError("packed ternary matmat expects a 1-D or 2-D input")
    d_in, _ = shape
    if x.shape[1] != d_in:
        raise ValueError(f"input width {x.shape[1]} does not match packed matrix width {d_in}")
    return np.stack([packed_matvec_scalar(row, packed, shape, scale) for row in x], axis=0)


def packed_matmat(X: np.ndarray, packed: np.ndarray, shape, scale: float,
                  allow_dense_reconstruction: bool = False) -> np.ndarray:
    """Multiply by a packed ternary matrix directly; never materialize dense W."""
    if allow_dense_reconstruction:
        raise RuntimeError("forbidden path: full dense expert-weight reconstruction "
                           "is not allowed on direct-packed execution")
    x = np.asarray(X, dtype=np.float32)
    one_row = x.ndim == 1
    if one_row:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("packed ternary matmat expects a 1-D or 2-D input")
    if isinstance(packed, TiledPacked):
        d_in, d_out = int(shape[0]), int(shape[1])
        if x.shape[1] != d_in:
            raise ValueError(f"input width {x.shape[1]} does not match packed matrix width {d_in}")
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty((x.shape[0], d_out), dtype=np.float32)
        if _NATIVE_LIB.smaulbrain_tiled_matmat(
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                packed.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_float(float(scale)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x.shape[0], d_in, d_out) != 0:
            raise RuntimeError("native tiled ternary matmat failed")
        return out[0] if one_row else out
    p, d_in, d_out = _check_packed_shape(packed, shape)
    if x.shape[1] != d_in:
        raise ValueError(f"input width {x.shape[1]} does not match packed matrix width {d_in}")
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty((x.shape[0], d_out), dtype=np.float32)
    if prepare_ternary_backend():
        status = _NATIVE_LIB.smaulbrain_ternary_matmat(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_float(float(scale)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[0], d_in, d_out)
        if status != 0:
            raise RuntimeError(f"native packed ternary matmat failed with status {status}")
    else:
        # Compiler-free fallback: decode to a temporary dense matrix and let
        # NumPy/BLAS do the multiply. Never cache this expansion. The scalar
        # reference remains available for tests but is not viable at model size.
        Wq = unpack_codes(p, (d_in, d_out)).astype(np.float32)
        Wq *= np.float32(scale)
        out = x @ Wq
        out = np.asarray(out, dtype=np.float32)
    return out[0] if one_row else out


def packed_matmat_transpose_scalar(dY: np.ndarray, packed: np.ndarray, shape,
                                   scale: float) -> np.ndarray:
    """Portable reference for dY @ (scale * ternary(W)).T, without dense W."""
    dy = np.asarray(dY, dtype=np.float32)
    one_row = dy.ndim == 1
    if one_row:
        dy = dy.reshape(1, -1)
    if dy.ndim != 2:
        raise ValueError("packed ternary transpose matmat expects 1-D or 2-D input")
    p, d_in, d_out = _check_packed_shape(packed, shape)
    if dy.shape[1] != d_out:
        raise ValueError(f"gradient width {dy.shape[1]} does not match packed output width {d_out}")
    dx = np.zeros((dy.shape[0], d_in), dtype=np.float32)
    for b in range(dy.shape[0]):
        for i in range(d_in):
            acc = 0.0
            row = i * d_out
            for j in range(d_out):
                pos = row + j
                code = (int(p[pos >> 2]) >> (2 * (pos & 3))) & 3
                if code == 0:
                    acc -= float(dy[b, j])
                elif code >= 2:
                    acc += float(dy[b, j])
            dx[b, i] = np.float32(acc * float(scale))
    return dx[0] if one_row else dx


def packed_matmat_transpose(dY: np.ndarray, packed: np.ndarray, shape,
                            scale: float) -> np.ndarray:
    """Compute the input gradient from packed ternary weights directly."""
    dy = np.asarray(dY, dtype=np.float32)
    one_row = dy.ndim == 1
    if one_row:
        dy = dy.reshape(1, -1)
    if dy.ndim != 2:
        raise ValueError("packed ternary transpose matmat expects 1-D or 2-D input")
    if isinstance(packed, TiledPacked):
        d_in, d_out = int(shape[0]), int(shape[1])
        if dy.shape[1] != d_out:
            raise ValueError(f"gradient width {dy.shape[1]} does not match packed output width {d_out}")
        dy = np.ascontiguousarray(dy, dtype=np.float32)
        dx = np.empty((dy.shape[0], d_in), dtype=np.float32)
        if _NATIVE_LIB.smaulbrain_tiled_transpose_matmat(
                dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                packed.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_float(float(scale)),
                dx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dy.shape[0], d_in, d_out) != 0:
            raise RuntimeError("native tiled ternary transpose matmat failed")
        return dx[0] if one_row else dx
    p, d_in, d_out = _check_packed_shape(packed, shape)
    if dy.shape[1] != d_out:
        raise ValueError(f"gradient width {dy.shape[1]} does not match packed output width {d_out}")
    dy = np.ascontiguousarray(dy, dtype=np.float32)
    dx = np.empty((dy.shape[0], d_in), dtype=np.float32)
    if prepare_ternary_backend():
        status = _NATIVE_LIB.smaulbrain_ternary_transpose_matmat(
            dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_float(float(scale)),
            dx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dy.shape[0], d_in, d_out)
        if status != 0:
            raise RuntimeError(f"native packed ternary transpose matmat failed with status {status}")
    else:
        # As above, prefer a one-call BLAS multiply over Python's triple loop
        # if the C++ compiler is missing. The decoded matrix is call-local.
        Wq = unpack_codes(p, (d_in, d_out)).astype(np.float32)
        Wq *= np.float32(scale)
        dx = dy @ Wq.T
        dx = np.asarray(dx, dtype=np.float32)
    return dx[0] if one_row else dx


def packed_matvec(x: np.ndarray, packed: np.ndarray, shape, scale: float,
                  allow_dense_reconstruction: bool = False) -> np.ndarray:
    """Packed-kernel matvec. No full FP32/BF16 weight matrix is reconstructed."""
    return packed_matmat(x, packed, shape, scale,
                         allow_dense_reconstruction=allow_dense_reconstruction)


# ---------------- FP8 E4M3 software emulation ----------------

def _make_fp8_positive_values() -> np.ndarray:
    # Preserve the representable set used by the original scalar implementation,
    # excluding values above its documented finite saturation point (448).
    values = {0.0}
    for e in range(-8, 9):
        base = 2.0 ** e
        for m in range(8):
            value = base * (m / 8.0) if e == -8 else base * (1.0 + m / 8.0)
            if 0.0 <= value <= FP8_E4M3_MAX:
                values.add(value)
    return np.asarray(sorted(values), dtype=np.float64)


_FP8_POSITIVE_VALUES = _make_fp8_positive_values()


def fp8_e4m3_quantize(W: np.ndarray, scale=None):
    """Vectorized E4M3 software quantization with the legacy rounding rule.

    The positive representable values are sorted once, then nearest neighbors
    are selected with NumPy searchsorted. This avoids Python loops over every
    weight/activation/gradient element.
    """
    Wf = np.asarray(W, dtype=np.float64)
    if scale is None:
        # NaN/Inf must not poison the scale chosen for finite values.
        finite_abs = np.abs(Wf[np.isfinite(Wf)])
        mx = float(np.max(finite_abs)) if finite_abs.size else 0.0
        scale = np.float32(mx / FP8_E4M3_MAX) if mx > 0 else np.float32(1.0)
    if scale == 0:
        scale = np.float32(1.0)
    norm = Wf / float(scale)
    finite = np.isfinite(norm)
    magnitude = np.abs(norm)
    magnitude = np.where(finite, np.minimum(magnitude, FP8_E4M3_MAX), 0.0)

    hi_idx = np.searchsorted(_FP8_POSITIVE_VALUES, magnitude, side="left")
    hi_idx = np.minimum(hi_idx, _FP8_POSITIVE_VALUES.size - 1)
    lo_idx = np.maximum(hi_idx - 1, 0)
    lower = _FP8_POSITIVE_VALUES[lo_idx]
    upper = _FP8_POSITIVE_VALUES[hi_idx]
    # Scalar code only replaces best on a strict improvement, so exact ties
    # choose the lower (earlier) representable value.
    choose_upper = (upper - magnitude) < (magnitude - lower)
    quantized = np.where(choose_upper, upper, lower)
    sign = np.where(norm < 0.0, -1.0, 1.0)  # legacy behavior maps -0 to +0
    out = np.where(finite, sign * quantized, norm)
    return (out * float(scale)).astype(np.float32), np.float32(scale)


def fp8_error_bound(scale: float) -> float:
    """Worst-case absolute error of fp8_e4m3_quantize for inputs |v| <= 448*scale.

    Auto-scaling puts the largest magnitude at FP8_E4M3_MAX, where the legacy
    E4M3 mantissa (3 bits) has a grid step of 2^-3 * 448 * scale = 56 * scale;
    round-to-nearest halves that, giving 28 * scale.
    """
    return abs(float(scale)) * (FP8_E4M3_MAX / 16.0)
