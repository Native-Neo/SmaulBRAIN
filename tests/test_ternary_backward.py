"""Correctness and dispatch tests for the native ternary-aware backward path."""

import torch
import torch.nn.functional as F
import pytest

import native
from experts import _TernaryLinear


def _ternary_matrix(rows: int, cols: int) -> torch.Tensor:
    values = torch.tensor([-0.75, 0.0, 0.5, 0.0, 0.0, 1.25], dtype=torch.float32)
    return values.repeat((rows * cols + values.numel() - 1) // values.numel())[: rows * cols].reshape(rows, cols)


def test_native_kernel_is_really_dispatched_and_matches_dense_gemm(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "force")
    native.reset_native_state()
    native.reset_counters()

    torch.manual_seed(41)
    grad_out = torch.randn(6, 5, dtype=torch.float32).contiguous()
    weight = _ternary_matrix(5, 9).contiguous()

    got = native.call_ternary_backward_input(grad_out, weight)
    expected = grad_out @ weight

    assert got is not None
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)
    assert native.COUNTERS["ternary_backward_native"] == 1
    assert native.COUNTERS["ternary_backward_fallback"] == 0


def test_native_kernel_handles_all_zero_and_single_row_weights(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "force")
    native.reset_native_state()
    native.reset_counters()

    grad_out = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
    weight = torch.zeros((2, 4), dtype=torch.float32)
    got = native.call_ternary_backward_input(grad_out, weight)

    assert got is not None
    assert torch.equal(got, torch.zeros((1, 4), dtype=torch.float32))
    assert native.COUNTERS["ternary_backward_native"] == 1


def test_unsupported_inputs_cleanly_fall_back(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "force")
    native.reset_counters()
    got = native.call_ternary_backward_input(
        torch.ones((2, 3), dtype=torch.bfloat16),
        torch.ones((3, 4), dtype=torch.bfloat16),
    )
    assert got is None
    assert native.COUNTERS["ternary_backward_fallback"] == 1

    monkeypatch.setenv("SMAUL_NATIVE", "off")
    native.reset_counters()
    got = native.call_ternary_backward_input(
        torch.ones((2, 3), dtype=torch.float32),
        torch.ones((3, 4), dtype=torch.float32),
    )
    assert got is None
    assert native.COUNTERS["ternary_backward_fallback"] == 1


def test_custom_linear_preserves_input_and_ste_weight_gradients(monkeypatch):
    monkeypatch.setenv("SMAUL_NATIVE", "force")
    native.reset_native_state()
    native.reset_counters()
    torch.manual_seed(17)

    x0 = torch.randn(4, 7, dtype=torch.float32)
    w0 = _ternary_matrix(5, 7)
    upstream = torch.randn(4, 5, dtype=torch.float32)

    x_custom = x0.clone().requires_grad_(True)
    w_custom = w0.clone().requires_grad_(True)
    y_custom = _TernaryLinear.apply(x_custom, w_custom, True)
    (y_custom * upstream).sum().backward()

    x_dense = x0.clone().requires_grad_(True)
    w_dense = w0.clone().requires_grad_(True)
    y_dense = F.linear(x_dense, w_dense)
    (y_dense * upstream).sum().backward()

    assert torch.allclose(y_custom, y_dense, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_custom.grad, x_dense.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(w_custom.grad, w_dense.grad, atol=1e-5, rtol=1e-5)
    assert native.COUNTERS["ternary_backward_native"] == 3
