"""Kernels + RMSNorm: numerics, CPU parity, native build, sparse verdict."""

import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import platform

import pytest
import torch
import native
from kernels import compare_heads, linear_attn_memory_bound, time_fn
from rmsnorm import RMSNorm, rmsnorm_fn

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_rmsnorm_matches_reference_math():
    torch.manual_seed(0)
    n = RMSNorm(16)
    x = torch.randn(4, 16)
    ref = (x.float() / (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt())
    assert torch.allclose(n(x).float(), ref, atol=1e-5)
    assert n(x).dtype == x.dtype


def test_rmsnorm_bf16_safe():
    n = RMSNorm(16)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    y = n(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


def test_cpp_kernels_compile():
    for src in ("rmsnorm.cpp", "linear_attn.cpp", "fp8_quant.cpp"):
        r = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-c",
             os.path.join(ROOT, "kernels_cpp", src), "-o", "/dev/null"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_cpu_execution_no_per_element_python():
    # Hot paths run as single tensor ops: time scales sub-linearly with batch.
    torch.manual_seed(0)
    n = RMSNorm(128)
    x1 = torch.randn(32, 128)
    x2 = torch.randn(1024, 128)
    t1 = time_fn(lambda: n(x1), repeat=10)["median_ms"]
    t2 = time_fn(lambda: n(x2), repeat=10)["median_ms"]
    assert t2 < t1 * 32  # 32x data must cost far less than 32x time


def test_sparse_vs_dense_measured_not_assumed():
    res = compare_heads(repeat=5)
    assert res["sparse"]["params"] < res["dense"]["params"]  # fewer params...
    assert res["winner"] in ("dense", "sparse")  # ...but winner is measured
    print(f"\nhead verdict: {res['winner']} "
          f"(dense {res['dense']['median_ms']:.2f}ms vs "
          f"sparse {res['sparse']['median_ms']:.2f}ms)")


def test_attn_memory_formula():
    assert linear_attn_memory_bound(8, 64) == (8 * 4096 + 8 * 64) * 4


def test_sparse_head_rejects_bad_fan_in_and_device():
    from kernels import SparseTopKHead
    with pytest.raises(ValueError):
        SparseTopKHead(16, 8, fan_in=9)  # wider than the model dim
    with pytest.raises(ValueError):
        SparseTopKHead(16, 8, fan_in=0)
    h = SparseTopKHead(16, 8, fan_in=4)
    out = h(torch.randn(5, 8))
    assert out.shape == (5, 16) and torch.isfinite(out).all()


def test_native_build_pins_baseline_isa():
    """Ivy Bridge portability: baseline flags, never -march=native/fast-math."""
    flags = native._build_flags()
    assert "-march=native" not in " ".join(flags)
    assert "-ffast-math" not in flags and "-fno-fast-math" in flags
    if platform.machine().lower() in ("x86_64", "amd64"):
        assert "-march=x86-64" in flags and "-mtune=generic" in flags


def test_native_build_with_baseline_flags_compiles(tmp_path):
    for src in ("rmsnorm.cpp", "linear_attn.cpp", "fp8_quant.cpp"):
        so = os.path.join(str(tmp_path), src.replace(".cpp", ".so"))
        r = subprocess.run(
            ["g++", *native._build_flags(),
             os.path.join(ROOT, "kernels_cpp", src), "-o", so],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(so)


def test_cpu_capability_label_sane():
    cap = native.cpu_capability()
    assert isinstance(cap, str) and cap.strip()
    assert "\n" not in cap  # single-line label, safe to log/cache on


def test_native_self_test_passes_after_build():
    native._libs.clear()
    native._build_failed = None
    try:
        assert native.ensure_native()  # builds with baseline flags if needed
        native._self_test()  # raises on any kernel mismatch
        assert set(native._libs) == {"rmsnorm", "linear_attn", "fp8_quant"}
    finally:
        pass  # keep the warm cache for the remaining tests


def test_bind_rejects_symbol_less_library(tmp_path):
    """ABI mismatch: a .so without our symbols must fail loudly, not bind."""
    import ctypes
    src = tmp_path / "empty.cpp"
    src.write_text("extern \"C\" void dummy() {}\n")
    so = str(tmp_path / "empty.so")
    r = subprocess.run(["g++", "-shared", "-fPIC", str(src), "-o", so],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lib = ctypes.CDLL(so)
    for name in ("rmsnorm", "linear_attn", "fp8_quant"):
        with pytest.raises(RuntimeError, match="ABI"):
            native._bind(lib, name)
