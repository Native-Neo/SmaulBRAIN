"""Regression tests for low-memory expert training paths."""

import os
import sys
from unittest.mock import mock_open

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SmaulBrainConfig
from experts import _swiglu_tensors, swiglu_forward
from model import SmaulBrainModel


def _tiny_model(dtype="bf16"):
    cfg = SmaulBrainConfig(
        d_model=16,
        n_heads=4,
        num_experts=2,
        top_k=1,
        max_experts=2,
        min_experts=1,
        expert_hidden=16,
        max_depth=2,
        context_length=8,
        ram_cache=1,
        vram_cache=1,
        use_byte_conv=False,
        dtype=dtype,
    )
    return SmaulBrainModel(cfg)


@pytest.mark.parametrize("dtype, available_kib", [("bf16", 48), ("fp32", 56)])
def test_training_budget_includes_retained_expert_gradients(
    monkeypatch, dtype, available_kib
):
    model = _tiny_model(dtype=dtype)
    try:
        ids = torch.zeros(1, 8, dtype=torch.long)
        # This headroom fits activations, leaves, and checkpoint inputs, but
        # cannot also hold the gradients retained by those leaves.
        monkeypatch.setattr(
            "builtins.open", mock_open(read_data=f"MemAvailable: {available_kib} kB\n")
        )
        with pytest.raises(
            ValueError, match="estimated training working set exceeds 65%"
        ) as exc:
            model._check_activation_budget(ids, training=True)
        assert "expert_grads=" in str(exc.value)
        model._check_activation_budget(ids, training=False)

        monkeypatch.setattr(
            "builtins.open", mock_open(read_data="MemAvailable: 1024 kB\n")
        )
        model._check_activation_budget(ids, training=True)
    finally:
        model.pager.close()


def test_training_reuses_one_expert_leaf_dictionary_across_depths():
    model = _tiny_model()
    try:
        expert_id = model.pool.order[0]
        first = model._train_provider(expert_id)
        second = model._train_provider(expert_id)
        assert first is second
        assert len(model._leaves) == 1
        assert all(t.is_leaf and t.requires_grad for t in first.values())
        first["w_gate"].grad = torch.ones_like(first["w_gate"])
        grads = model.take_expert_grads()
        assert grads[expert_id]["w_gate"].dtype == first["w_gate"].dtype
        assert model._leaves == {}
    finally:
        model.pager.close()


def test_checkpointed_swiglu_matches_direct_forward_and_gradients():
    torch.manual_seed(7)
    x0 = torch.randn(5, 4)
    w0 = {
        "w_gate": torch.randn(7, 4),
        "w_up": torch.randn(7, 4),
        "w_down": torch.randn(4, 7),
    }
    x_checkpoint = x0.clone().requires_grad_(True)
    w_checkpoint = {name: value.clone().requires_grad_(True)
                    for name, value in w0.items()}
    x_direct = x0.clone().requires_grad_(True)
    w_direct = {name: value.clone().requires_grad_(True)
                for name, value in w0.items()}

    got = swiglu_forward(x_checkpoint, w_checkpoint)
    expected = _swiglu_tensors(
        x_direct, w_direct["w_gate"], w_direct["w_up"], w_direct["w_down"]
    )
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)

    got.square().sum().backward()
    expected.square().sum().backward()
    assert torch.allclose(x_checkpoint.grad, x_direct.grad, atol=1e-6, rtol=1e-6)
    for name in w_checkpoint:
        assert torch.allclose(
            w_checkpoint[name].grad, w_direct[name].grad, atol=1e-6, rtol=1e-6
        )

def test_ternary_qat_backward_uses_ternary_weights_and_full_precision_ste_gradients():
    from experts import ternary_qat_weight

    torch.manual_seed(19)
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    w = torch.tensor(
        [[-1.0, -0.2, 0.4, 1.2],
         [0.1, -0.8, 0.7, 0.3],
         [0.9, -0.4, -0.1, 0.6]],
        dtype=torch.float32,
        requires_grad=True,
    )
    q = ternary_qat_weight(w, threshold_ratio=0.5)
    grad_output = torch.randn(3, 3, dtype=torch.float32)

    output = torch.nn.functional.linear(x, q)
    output.backward(grad_output)

    # Input gradients use the actual ternary forward weights, not latent w.
    expected_dx = grad_output @ q.detach()
    # STE preserves the dense weight-gradient estimate for the optimizer.
    expected_dw = grad_output.T @ x.detach()
    assert torch.allclose(x.grad, expected_dx, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w.grad, expected_dw, atol=1e-6, rtol=1e-6)
    assert not torch.equal(q.detach(), w.detach())

