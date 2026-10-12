"""Regression tests for low-memory expert training paths."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SmaulBrainConfig
from experts import _swiglu_tensors, swiglu_forward
from model import SmaulBrainModel


def _tiny_model():
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
    )
    return SmaulBrainModel(cfg)


def test_training_reuses_one_expert_leaf_dictionary_across_depths():
    model = _tiny_model()
    try:
        expert_id = model.pool.order[0]
        first = model._train_provider(expert_id)
        second = model._train_provider(expert_id)
        assert first is second
        assert len(model._leaves) == 1
        assert all(t.is_leaf and t.requires_grad for t in first.values())
        model.clear_expert_grads()
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
