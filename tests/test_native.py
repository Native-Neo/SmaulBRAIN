"""Native dispatch: prove both paths run, fallbacks are explicit, results hold.

The issue demands the runtime actually dispatch (not merely compile native
code) and fall back only under observable conditions. These tests drive the
counters in ``native.py``: native calls must appear when enabled, fallback
counts plus reasons must appear when disabled, and outputs must agree.
"""

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import torch

import native
from linear_attention import LinearAttnState, linear_attn_step
from precision import dequantize_fp8_blockwise, quantize_fp8_blockwise
from rmsnorm import RMSNorm, rmsnorm_fn


@pytest.fixture()
def force_native(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    yield
    monkeypatch.undo()


@pytest.fixture()
def no_native(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    native.reset_counters()
    yield
    monkeypatch.undo()


def test_cpu_capability_and_mode_parsing(monkeypatch):
    assert isinstance(native.cpu_capability(), str) and native.cpu_capability()
    monkeypatch.setenv("SMAUL_NATIVE", "force")
    assert native.native_mode() == "force"
    monkeypatch.setenv("SMAUL_NATIVE", "off")
    assert native.native_mode() == "off"
    monkeypatch.setenv("SMAUL_NATIVE", "whatever")
    assert native.native_mode() == "auto"


def test_rmsnorm_dispatches_natively_when_forced(force_native):
    n = RMSNorm(32)
    x = torch.randn(5, 32)
    with torch.no_grad():
        y = n(x)
    assert native.COUNTERS["rmsnorm_native"] == 1
    assert torch.allclose(y, rmsnorm_fn(x, n.weight), atol=1e-6)


def test_rmsnorm_keeps_gradients_and_determinism(no_native):
    n = RMSNorm(32)
    x = torch.randn(5, 32, requires_grad=True)
    n(x).sum().backward()
    assert x.grad is not None  # reference path keeps exact autograd
    assert native.COUNTERS["rmsnorm_native"] == 0
    assert native.COUNTERS["rmsnorm_fallback"] == 0  # not even attempted


def test_fp8_auto_dispatch_bit_identical(force_native, monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(8, 130) * 0.5
    t = quantize_fp8_blockwise(w, tile=64)
    assert native.COUNTERS["fp8_quant_native"] == 1
    r = dequantize_fp8_blockwise(t)
    assert native.COUNTERS["fp8_dequant_native"] == 1
    native.reset_counters()
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    t2 = quantize_fp8_blockwise(w, tile=64)
    r2 = dequantize_fp8_blockwise(t2)
    assert native.COUNTERS["fp8_quant_native"] == 0
    assert native.COUNTERS["fp8_dequant_native"] == 0
    assert native.COUNTERS["fp8_quant_fallback"] == 1
    assert torch.equal(t.codes, t2.codes)  # identical bytes either way
    assert torch.equal(r, r2)


def test_attn_step_native_agrees_and_refuses_grad(force_native):
    torch.manual_seed(0)
    B, H, Dh = 2, 2, 8
    st = LinearAttnState.zeros(B, H, Dh)
    st2 = LinearAttnState.zeros(B, H, Dh)
    q, k, v = torch.randn(B, H, Dh), torch.randn(B, H, Dh), torch.randn(B, H, Dh)
    with torch.no_grad():
        y_ref, _ = linear_attn_step(st, q, k, v)
        y_nat, _ = linear_attn_step(st2, q, k, v, use_native=True)
    # Forced mode engages everywhere eligible: both calls went native.
    assert native.COUNTERS["attn_step_native"] == 2 * B * H
    assert torch.allclose(y_ref, y_nat, atol=1e-5)
    assert torch.allclose(st.S, st2.S, atol=1e-5)
    # Gradient mode must refuse the grad-less native kernel.
    native.reset_counters()
    xg = torch.randn(B, H, Dh, requires_grad=True)
    linear_attn_step(LinearAttnState.zeros(B, H, Dh), xg, k, v, use_native=True)
    assert native.COUNTERS["attn_step_native"] == 0


def test_forced_native_without_compiler_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    monkeypatch.setattr(native.shutil, "which", lambda *_a, **_k: None)
    native._libs.clear()
    native._build_failed = None
    try:
        with pytest.raises(RuntimeError):
            native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), 1e-6)
    finally:
        native._libs.clear()
        native._build_failed = None


def test_rmsnorm_strided_input_copied_not_rejected(force_native):
    base = torch.randn(6, 8)
    before = base.clone()
    x = base[:, ::2]  # strided, noncontiguous read-only input
    assert not x.is_contiguous()
    w = torch.randn(4)
    y = native.call_rmsnorm(x, w, 1e-6)
    assert y is not None and y.shape == (6, 4)
    assert torch.allclose(y, rmsnorm_fn(x.contiguous(), w), atol=1e-6)
    assert native.COUNTERS["rmsnorm_native"] == 1
    assert torch.equal(base, before)  # inputs only read, never written
    assert y.data_ptr() not in (x.data_ptr(), w.data_ptr())  # fresh output


def test_rmsnorm_rejects_bad_shapes_and_eps(force_native):
    assert native.call_rmsnorm(torch.randn(2, 4), torch.randn(5), 1e-6) is None
    assert native.call_rmsnorm(torch.randn(0, 4), torch.randn(4), 1e-6) is None
    assert native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), float("nan")) is None
    assert native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), -1.0) is None
    assert native.call_rmsnorm(torch.randn(4), torch.randn(4), 1e-6) is None
    assert native.COUNTERS["rmsnorm_native"] == 0
    assert native.COUNTERS["rmsnorm_fallback"] == 5
    assert native.LAST_FALLBACK["rmsnorm"] == "needs 2D [rows, cols] input"


def test_attn_step_rejects_strided_state_and_bad_shapes(force_native):
    Dh = 8
    S = torch.zeros(Dh, Dh)
    z, q, k, v = torch.zeros(Dh), torch.randn(Dh), torch.randn(Dh), torch.randn(Dh)
    y, scratch = torch.empty(Dh), torch.empty(2 * Dh)
    assert not native.call_attn_step(S.t(), z, q, k, v, y, scratch, Dh, 1e-6)
    assert not native.call_attn_step(S, z, q, k, v, y, scratch, 0, 1e-6)
    assert not native.call_attn_step(S, z, q, k, v, y, scratch[: Dh + 1], Dh, 1e-6)
    assert not native.call_attn_step(S, z, q, k, v, y, scratch, Dh, float("inf"))
    assert torch.equal(S, torch.zeros(Dh, Dh))  # refused calls never mutate
    assert native.COUNTERS["attn_step_native"] == 0
    assert native.COUNTERS["attn_step_fallback"] == 4


def test_fp8_rejects_zero_tile_and_mismatched_shapes(force_native):
    w = torch.randn(2, 8)
    codes = torch.empty(2, 8, dtype=torch.uint8)
    scales = torch.empty(2, 2, dtype=torch.float32)
    assert not native.call_fp8_quant(w, codes, scales, 2, 8, 0)
    assert not native.call_fp8_quant(w, codes, scales[:, :1], 2, 8, 4)
    out = torch.empty(2, 8, dtype=torch.float32)
    assert not native.call_fp8_dequant(codes, scales, out, 2, 8, 0)
    assert not native.call_fp8_dequant(codes, scales[:, :1], out, 2, 8, 4)
    assert native.COUNTERS["fp8_quant_native"] == 0
    assert native.COUNTERS["fp8_dequant_native"] == 0


def test_fp8_nan_input_falls_back_bit_exact(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    w = torch.randn(4, 130)
    w[0, 0] = float("nan")
    w[1, 1] = float("inf")
    t = quantize_fp8_blockwise(w, tile=64)
    assert native.COUNTERS["fp8_quant_native"] == 0  # NaN stays on reference
    assert native.COUNTERS["fp8_quant_fallback"] == 1
    assert "non-finite" in native.LAST_FALLBACK["fp8_quant"]
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    t2 = quantize_fp8_blockwise(w, tile=64)
    assert torch.equal(t.codes, t2.codes)  # same bytes as pure reference


def test_build_timeout_falls_back_with_reason(monkeypatch):
    import subprocess as _sp
    monkeypatch.setenv("SMAUL_NATIVE", "auto")
    native._libs.clear()
    native._build_failed = None
    def _slow(*a, **k):
        raise _sp.TimeoutExpired(cmd="g++", timeout=60)
    monkeypatch.setattr(native.subprocess, "run", _slow)
    try:
        assert not native.ensure_native()
        assert "timed out" in (native._build_failed or "")
        assert native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), 1e-6) is None
        assert native.COUNTERS["rmsnorm_fallback"] >= 1
        assert "timed out" in native.LAST_FALLBACK["rmsnorm"]
    finally:
        native._libs.clear()
        native._build_failed = None


def test_native_output_ownership_and_grad_safety(force_native):
    x = torch.randn(3, 8)
    w = torch.randn(8)
    y = native.call_rmsnorm(x, w, 1e-6)
    assert y is not None and not y.requires_grad
    assert y.data_ptr() not in (x.data_ptr(), w.data_ptr())
    native.reset_counters()
    xg = torch.randn(3, 8, requires_grad=True)
    with torch.enable_grad():
        assert native.call_rmsnorm(xg, w, 1e-6) is None
    assert native.COUNTERS["rmsnorm_native"] == 0
    assert "autograd" in native.LAST_FALLBACK["rmsnorm"]


def test_parse_cpu_flags_token_exact():
    """Issue 63: detection is token-exact — ``avx`` alone never reads as AVX2."""
    flags = native._parse_cpu_flags(
        "processor\t: 0\nflags\t\t: sse sse2 ssse3 sse4_1 sse4_2 avx\n")
    assert flags == frozenset({"sse", "sse2", "ssse3", "sse4_1", "sse4_2", "avx"})
    assert "avx2" not in flags and "avx512f" not in flags  # no substring bleed
    assert native._parse_cpu_flags("no flags here\n") == frozenset()
    two = "flags : sse2 avx\nflags : sse2 avx2\n"
    assert native._parse_cpu_flags(two) == frozenset({"sse2", "avx"})


def test_unsupported_host_falls_back_with_reason(monkeypatch):
    """Issue 63: forced-unsupported path — pre-SSE2 host falls back loudly."""
    monkeypatch.setattr(native, "host_supports_baseline", lambda: False)
    monkeypatch.setenv("SMAUL_NATIVE", "auto")
    native.reset_counters()
    assert native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), 1e-6) is None
    assert native.COUNTERS["rmsnorm_native"] == 0
    assert native.COUNTERS["rmsnorm_fallback"] == 1
    assert "lacks" in native.LAST_FALLBACK["rmsnorm"]
    assert native.REQUIRED_ISA in native.LAST_FALLBACK["rmsnorm"]


def test_unsupported_host_forced_raises(monkeypatch):
    """Issue 63: SMAUL_NATIVE=force on an unsupported host raises, not SIGILL."""
    monkeypatch.setattr(native, "host_supports_baseline", lambda: False)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    with pytest.raises(RuntimeError, match="forced but unsupported"):
        native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), 1e-6)


def test_stale_so_rebuilds_in_place(monkeypatch, tmp_path):
    """Issue 63: a stale/symbol-less cached .so is rebuilt, not just deleted."""
    if shutil.which("g++") is None:
        pytest.skip("no g++")
    monkeypatch.setattr(native, "_cache_dir", lambda: str(tmp_path))
    (tmp_path / "empty.cpp").write_text('extern "C" void dummy() {}\n')
    stale = str(tmp_path / "rmsnorm.so")
    r = subprocess.run(
        ["g++", "-shared", "-fPIC", str(tmp_path / "empty.cpp"), "-o", stale],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    native.reset_native_state()
    native.reset_counters()
    try:
        assert native.ensure_native()  # rebuilds stale .so in the same call
        assert set(native._libs) == {"rmsnorm", "linear_attn", "fp8_quant"}
        assert native._verify_baseline_isa(stale)
        # The file on disk is now a real kernel, not the stale placeholder.
        nm = subprocess.run(["nm", "-D", stale], capture_output=True,
                            text=True, timeout=60)
        assert "smaul_rmsnorm_forward" in nm.stdout
        # And the loaded library is functional (proves alias-load recovery).
        monkeypatch.setenv("SMAUL_NATIVE", "1")
        y = native.call_rmsnorm(torch.randn(2, 4), torch.randn(4), 1e-6)
        assert y is not None and y.shape == (2, 4)
        assert native.COUNTERS["rmsnorm_native"] == 1
    finally:
        native.reset_native_state()


def test_isa_verifier_rejects_avx_and_accepts_baseline(tmp_path):
    """Issue 63: compiled-ISA audit — VEX .so rejected, baseline .so accepted."""
    if shutil.which("g++") is None or shutil.which("objdump") is None:
        pytest.skip("need g++ + objdump")
    src = tmp_path / "avx_probe.cpp"
    src.write_text(
        "#include <immintrin.h>\n"
        "extern \"C\" void probe(float* x, float* y) {\n"
        "  __m256 v = _mm256_loadu_ps(x);\n"
        "  _mm256_storeu_ps(y, _mm256_add_ps(v, v)); }\n")
    so = str(tmp_path / "avx_probe.so")
    r = subprocess.run(
        ["g++", "-O2", "-mavx", "-shared", "-fPIC", str(src), "-o", so],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        pytest.skip("compiler refused -mavx")
    assert not native._verify_baseline_isa(so)  # VEX mnemonics detected
    native.reset_native_state()
    try:
        assert native.ensure_native()
        d = native._cache_dir()
        for name in native.LIBS:
            assert native._verify_baseline_isa(os.path.join(d, name + ".so"))
    finally:
        native.reset_native_state()


def test_portable_fallback_matches_native(monkeypatch):
    """Issue 63: safe portable fallback — off-mode results equal native math."""
    torch.manual_seed(7)
    x = torch.randn(4, 16)
    w = torch.randn(16)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    with torch.no_grad():
        y_nat = native.call_rmsnorm(x, w, 1e-6)
    assert y_nat is not None and torch.allclose(y_nat, rmsnorm_fn(x, w), atol=1e-6)
    fw = torch.randn(4, 130) * 0.5
    t_nat = quantize_fp8_blockwise(fw, tile=64)
    assert native.COUNTERS["fp8_quant_native"] == 1
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    t_ref = quantize_fp8_blockwise(fw, tile=64)
    assert torch.equal(t_nat.codes, t_ref.codes)
    assert torch.equal(t_nat.scales, t_ref.scales)
    B, H, Dh = 1, 1, 8
    q, k, v = torch.randn(B, H, Dh), torch.randn(B, H, Dh), torch.randn(B, H, Dh)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    with torch.no_grad():
        y_nat2, _ = linear_attn_step(
            LinearAttnState.zeros(B, H, Dh), q, k, v, use_native=True)
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    native.reset_counters()
    with torch.no_grad():
        y_off, _ = linear_attn_step(
            LinearAttnState.zeros(B, H, Dh), q, k, v, use_native=True)
    assert torch.allclose(y_nat2, y_off, atol=1e-5)
    assert native.COUNTERS["attn_step_native"] == 0
    assert native.COUNTERS["attn_step_fallback"] >= 1


def test_status_observable(monkeypatch):
    """Issue 63: fallback observability — one status() call shows everything."""
    monkeypatch.setenv("SMAUL_NATIVE", "auto")
    s = native.status()
    assert s["mode"] == "auto"
    assert s["required_isa"] == native.REQUIRED_ISA == native.required_isa()
    cap = s["cpu_capability"]
    assert isinstance(cap, str) and cap and "\n" not in cap
    assert s["host_ok"] is True  # this host is x86_64 with SSE2
    assert "-march=native" not in " ".join(s["build_flags"])
    assert "-march=x86-64" in s["build_flags"]
    assert set(s["counters"]) == set(native.COUNTERS)
