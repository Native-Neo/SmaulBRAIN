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
