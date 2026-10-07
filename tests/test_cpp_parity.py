"""C++ kernel numerics parity: native kernels must match the Python math.

Compiling the kernels never proved equivalence. This file builds each
``kernels_cpp`` unit as a shared object and compares, via ctypes, against
the torch reference paths (``rmsnorm_fn``, ``linear_attn_step``, and the
FP8 quantize/dequantize roundtrip). Deterministic seeds keep the exact
FP8-code comparison stable across runs.
"""

import ctypes
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from linear_attention import LinearAttnState, linear_attn_step
from precision import dequantize_fp8_blockwise, quantize_fp8_blockwise
from rmsnorm import rmsnorm_fn

ROOT = os.path.join(os.path.dirname(__file__), "..")
CPP = os.path.join(ROOT, "kernels_cpp")

_F32 = ctypes.POINTER(ctypes.c_float)
_U8 = ctypes.POINTER(ctypes.c_ubyte)
_SZ = ctypes.c_size_t


def _build(name: str, tmp_path) -> ctypes.CDLL:
    src = os.path.join(CPP, name)
    so = str(tmp_path / (name.replace(".cpp", "") + ".so"))
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC",
                        src, "-o", so], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return ctypes.CDLL(so)


def test_rmsnorm_matches_torch(tmp_path):
    lib = _build("rmsnorm.cpp", tmp_path)
    lib.smaul_rmsnorm_forward.argtypes = [_F32, _F32, _F32, _SZ, _SZ, ctypes.c_float]
    rng = np.random.default_rng(0)
    rows, cols = 4, 32
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 2
    w = (rng.standard_normal(cols).astype(np.float32) * 0.5 + 1.0)
    y = np.zeros_like(x)
    lib.smaul_rmsnorm_forward(x.ctypes.data_as(_F32), w.ctypes.data_as(_F32),
                              y.ctypes.data_as(_F32), rows, cols, 1e-6)
    ref = rmsnorm_fn(torch.from_numpy(x), torch.from_numpy(w), eps=1e-6).numpy()
    assert np.allclose(y, ref, atol=1e-6, rtol=1e-5)


def test_linear_attn_step_matches_torch(tmp_path):
    lib = _build("linear_attn.cpp", tmp_path)
    lib.smaul_linear_attn_step_buf.argtypes = [_F32, _F32, _F32, _F32, _F32,
                                               _F32, _F32, _SZ, ctypes.c_float]
    rng = np.random.default_rng(1)
    Dh = 8
    S = np.zeros((Dh, Dh), dtype=np.float32)
    z = np.zeros(Dh, dtype=np.float32)
    scratch = np.zeros(2 * Dh, dtype=np.float32)
    y = np.zeros(Dh, dtype=np.float32)
    st = LinearAttnState.zeros(1, 1, Dh)
    for _ in range(5):
        q = (rng.standard_normal(Dh).astype(np.float32))
        k = (rng.standard_normal(Dh).astype(np.float32))
        v = (rng.standard_normal(Dh).astype(np.float32))
        lib.smaul_linear_attn_step_buf(S.ctypes.data_as(_F32), z.ctypes.data_as(_F32),
                                       q.ctypes.data_as(_F32), k.ctypes.data_as(_F32),
                                       v.ctypes.data_as(_F32), y.ctypes.data_as(_F32),
                                       scratch.ctypes.data_as(_F32), Dh, 1e-6)
        ref, st = linear_attn_step(st, torch.from_numpy(q)[None, None, :],
                                   torch.from_numpy(k)[None, None, :],
                                   torch.from_numpy(v)[None, None, :])
        assert np.allclose(y, ref.numpy()[0, 0], atol=1e-5, rtol=1e-4)
    assert np.allclose(S, st.S.numpy()[0, 0], atol=1e-5, rtol=1e-4)
    assert np.allclose(z, st.z.numpy()[0, 0], atol=1e-5, rtol=1e-4)


def test_fp8_codes_match_torch_cast(tmp_path):
    lib = _build("fp8_quant.cpp", tmp_path)
    lib.smaul_fp8_quant_row_block.argtypes = [_F32, _U8, _F32, _SZ, _SZ, _SZ]
    lib.smaul_fp8_dequant_row_block.argtypes = [_U8, _F32, _F32, _SZ, _SZ, _SZ]
    rng = np.random.default_rng(2)
    rows, cols, tile = 16, 130, 64  # 130 forces the padding path
    nblocks = (cols + tile - 1) // tile
    w = (rng.standard_normal((rows, cols)).astype(np.float32)) * 0.5
    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales = np.zeros((rows, nblocks), dtype=np.float32)
    lib.smaul_fp8_quant_row_block(w.ctypes.data_as(_F32), codes.ctypes.data_as(_U8),
                                  scales.ctypes.data_as(_F32), rows, cols, tile)
    ref = quantize_fp8_blockwise(torch.from_numpy(w), tile=tile)
    assert np.array_equal(codes, ref.codes.numpy())
    assert np.allclose(scales, ref.scales.numpy(), atol=0, rtol=0)
    back = np.zeros_like(w)
    lib.smaul_fp8_dequant_row_block(codes.ctypes.data_as(_U8), scales.ctypes.data_as(_F32),
                                    back.ctypes.data_as(_F32), rows, cols, tile)
    ref_back = dequantize_fp8_blockwise(ref, dtype=torch.float32).numpy()
    assert np.array_equal(back, ref_back)
    assert float(np.abs(back - w).mean()) < 0.02


def test_kernels_survive_zero_and_null_sizes(tmp_path):
    """Fuzz the guards: empty sizes and NULL pointers are no-ops, not faults."""
    rn = _build("rmsnorm.cpp", tmp_path)
    rn.smaul_rmsnorm_forward.argtypes = [_F32, _F32, _F32, _SZ, _SZ, ctypes.c_float]
    rn.smaul_rmsnorm_forward(None, None, None, 0, 0, 1e-6)
    y = np.full((1, 4), -1.0, dtype=np.float32)
    x = np.ones((1, 4), dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    rn.smaul_rmsnorm_forward(x.ctypes.data_as(_F32), w.ctypes.data_as(_F32),
                             y.ctypes.data_as(_F32), 1, 0, 1e-6)
    assert (y == -1.0).all()  # cols=0: untouched
    rn.smaul_rmsnorm_forward(x.ctypes.data_as(_F32), w.ctypes.data_as(_F32),
                             y.ctypes.data_as(_F32), 1, 4, float("nan"))
    assert (y == -1.0).all()  # NaN eps: rejected, untouched

    la = _build("linear_attn.cpp", tmp_path)
    la.smaul_linear_attn_step_buf.argtypes = [_F32, _F32, _F32, _F32, _F32,
                                              _F32, _F32, _SZ, ctypes.c_float]
    la.smaul_linear_attn_step_buf(None, None, None, None, None, None, None, 0, 1e-6)

    fq = _build("fp8_quant.cpp", tmp_path)
    fq.smaul_fp8_quant_row_block.argtypes = [_F32, _U8, _F32, _SZ, _SZ, _SZ]
    fq.smaul_fp8_dequant_row_block.argtypes = [_U8, _F32, _F32, _SZ, _SZ, _SZ]
    codes = np.full((2, 8), 0xA5, dtype=np.uint8)
    wq = np.ones((2, 8), dtype=np.float32)
    sc = np.zeros((2, 2), dtype=np.float32)
    fq.smaul_fp8_quant_row_block(wq.ctypes.data_as(_F32), codes.ctypes.data_as(_U8),
                                 sc.ctypes.data_as(_F32), 2, 8, 0)  # tile=0
    assert (codes == 0xA5).all()  # untouched
    fq.smaul_fp8_dequant_row_block(None, None, None, 0, 0, 0)


@pytest.mark.parametrize("cols,tile", [(1, 1), (1, 64), (3, 2), (7, 3), (65, 64), (5, 16)])
def test_fp8_odd_shapes_match_reference(tmp_path, cols, tile):
    """Odd/undersized/ragged tiles must match the torch path bit-for-bit."""
    lib = _build("fp8_quant.cpp", tmp_path)
    lib.smaul_fp8_quant_row_block.argtypes = [_F32, _U8, _F32, _SZ, _SZ, _SZ]
    lib.smaul_fp8_dequant_row_block.argtypes = [_U8, _F32, _F32, _SZ, _SZ, _SZ]
    rng = np.random.default_rng(1000 + cols * 17 + tile)
    rows = 3
    nblocks = (cols + tile - 1) // tile
    w = (rng.standard_normal((rows, cols)).astype(np.float32)) * 0.5
    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales = np.zeros((rows, nblocks), dtype=np.float32)
    lib.smaul_fp8_quant_row_block(w.ctypes.data_as(_F32), codes.ctypes.data_as(_U8),
                                  scales.ctypes.data_as(_F32), rows, cols, tile)
    ref = quantize_fp8_blockwise(torch.from_numpy(w), tile=tile)
    assert np.array_equal(codes, ref.codes.numpy())
    assert np.allclose(scales, ref.scales.numpy(), atol=0, rtol=0)
    back = np.zeros_like(w)
    lib.smaul_fp8_dequant_row_block(codes.ctypes.data_as(_U8), scales.ctypes.data_as(_F32),
                                    back.ctypes.data_as(_F32), rows, cols, tile)
    assert np.array_equal(back, dequantize_fp8_blockwise(ref).numpy())


def test_fp8_nan_inf_never_emit_nan_codes(tmp_path):
    """Out-of-range inputs saturate to finite codes; NaN codes never emitted."""
    lib = _build("fp8_quant.cpp", tmp_path)
    lib.smaul_fp8_quant_row_block.argtypes = [_F32, _U8, _F32, _SZ, _SZ, _SZ]
    lib.smaul_fp8_dequant_row_block.argtypes = [_U8, _F32, _F32, _SZ, _SZ, _SZ]
    w = np.array([[float("nan"), float("inf"), float("-inf"), 0.0,
                   1e-30, 500.0, -500.0, 0.5]], dtype=np.float32)
    codes = np.zeros_like(w, dtype=np.uint8)
    scales = np.zeros((1, 2), dtype=np.float32)
    lib.smaul_fp8_quant_row_block(w.ctypes.data_as(_F32), codes.ctypes.data_as(_U8),
                                  scales.ctypes.data_as(_F32), 1, 8, 4)
    assert not ((codes == 0x7F) | (codes == 0xFF)).any()  # finite-only storage
    assert not np.isnan(scales).any()  # NaN ignored in amax (may saturate to inf)
    back = np.zeros_like(w)
    lib.smaul_fp8_dequant_row_block(codes.ctypes.data_as(_U8), scales.ctypes.data_as(_F32),
                                    back.ctypes.data_as(_F32), 1, 8, 4)
    # No-crash fuzz only here: mixed NaN/Inf inputs legitimately disagree
    # with the torch path (torch amax propagates NaN), so dispatch routes
    # non-finite inputs to the reference path (see test_native.py).


@pytest.mark.parametrize("cols", [1, 3, 7, 33])
def test_rmsnorm_odd_and_single_col_matches_torch(tmp_path, cols):
    lib = _build("rmsnorm.cpp", tmp_path)
    lib.smaul_rmsnorm_forward.argtypes = [_F32, _F32, _F32, _SZ, _SZ, ctypes.c_float]
    rng = np.random.default_rng(2000 + cols)
    rows = 5
    x = (rng.standard_normal((rows, cols)).astype(np.float32)) * 2
    w = (rng.standard_normal(cols).astype(np.float32)) * 0.5 + 1.0
    y = np.zeros_like(x)
    lib.smaul_rmsnorm_forward(x.ctypes.data_as(_F32), w.ctypes.data_as(_F32),
                              y.ctypes.data_as(_F32), rows, cols, 1e-6)
    ref = rmsnorm_fn(torch.from_numpy(x), torch.from_numpy(w), eps=1e-6).numpy()
    assert np.allclose(y, ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("Dh", [1, 2, 3])
def test_linear_attn_small_dh_matches_torch(tmp_path, Dh):
    lib = _build("linear_attn.cpp", tmp_path)
    lib.smaul_linear_attn_step_buf.argtypes = [_F32, _F32, _F32, _F32, _F32,
                                               _F32, _F32, _SZ, ctypes.c_float]
    rng = np.random.default_rng(3000 + Dh)
    S = np.zeros((Dh, Dh), dtype=np.float32)
    z = np.zeros(Dh, dtype=np.float32)
    scratch = np.zeros(2 * Dh, dtype=np.float32)
    y = np.zeros(Dh, dtype=np.float32)
    st = LinearAttnState.zeros(1, 1, Dh)
    for _ in range(3):
        q = rng.standard_normal(Dh).astype(np.float32)
        k = rng.standard_normal(Dh).astype(np.float32)
        v = rng.standard_normal(Dh).astype(np.float32)
        lib.smaul_linear_attn_step_buf(S.ctypes.data_as(_F32), z.ctypes.data_as(_F32),
                                       q.ctypes.data_as(_F32), k.ctypes.data_as(_F32),
                                       v.ctypes.data_as(_F32), y.ctypes.data_as(_F32),
                                       scratch.ctypes.data_as(_F32), Dh, 1e-6)
        ref, st = linear_attn_step(st, torch.from_numpy(q)[None, None, :],
                                   torch.from_numpy(k)[None, None, :],
                                   torch.from_numpy(v)[None, None, :])
        assert np.allclose(y, ref.numpy()[0, 0], atol=1e-5, rtol=1e-4)


def test_rmsnorm_misaligned_buffer_matches_torch(tmp_path):
    """Unaligned input pointers stay correct on x86-64 (no alignment copy)."""
    lib = _build("rmsnorm.cpp", tmp_path)
    lib.smaul_rmsnorm_forward.argtypes = [_F32, _F32, _F32, _SZ, _SZ, ctypes.c_float]
    rng = np.random.default_rng(42)
    vals = (rng.standard_normal(4).astype(np.float32)) * 2
    buf = bytearray(17)
    buf[1:17] = vals.tobytes()  # force 1-byte misalignment
    x = np.frombuffer(buf, dtype=np.uint8)[1:17].view(np.float32).reshape(1, 4)
    assert x.ctypes.data % 4 != 0
    w = np.ones(4, dtype=np.float32)
    y = np.zeros((1, 4), dtype=np.float32)
    lib.smaul_rmsnorm_forward(x.ctypes.data_as(_F32), w.ctypes.data_as(_F32),
                              y.ctypes.data_as(_F32), 1, 4, 1e-6)
    ref = rmsnorm_fn(torch.from_numpy(np.array(x, dtype=np.float32)),
                     torch.from_numpy(w), eps=1e-6).numpy()
    assert np.allclose(y, ref, atol=1e-6, rtol=1e-5)
