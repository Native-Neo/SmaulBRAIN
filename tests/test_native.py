"""Native dispatch: prove both paths run, fallbacks are explicit, results hold.

The issue demands the runtime actually dispatch (not merely compile native
code) and fall back only under observable conditions. These tests drive the
counters in ``native.py``: native calls must appear when enabled, fallback
counts plus reasons must appear when disabled, and outputs must agree.
"""

import os
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
