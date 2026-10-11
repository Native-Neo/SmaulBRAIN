"""Regression tests for optimizer numeric guards."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smaulopt import global_grad_norm


def test_global_grad_norm_does_not_overflow_on_large_finite_fp32_gradients():
    grad = torch.tensor([1.0e20, 1.0e20], dtype=torch.float32)
    got = global_grad_norm([grad])
    expected = (2.0 ** 0.5) * 1.0e20
    assert torch.isfinite(torch.tensor(got, dtype=torch.float64))
    assert abs(got / expected - 1.0) < 1e-6


def test_global_grad_norm_preserves_tiny_finite_fp32_gradients():
    grad = torch.tensor([1.0e-30, 0.0], dtype=torch.float32)
    got = global_grad_norm([grad])
    assert got > 0.0
    assert abs(got / 1.0e-30 - 1.0) < 1e-6
