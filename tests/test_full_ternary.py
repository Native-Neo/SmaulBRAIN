"""Small correctness tests for opt-in full-model ternary QAT."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from config import SmaulBrainConfig
from experts import ternary_qat_weight
from model import SmaulBrainModel
from precision import HybridFP8Linear
from rmsnorm import RMSNorm
from ternary import (
    TernaryQATConv1d,
    TernaryQATEmbedding,
    TernaryQATLinear,
    TernaryQATRMSNorm,
)


def _tiny_full_ternary_config():
    return SmaulBrainConfig(
        d_model=16,
        n_heads=4,
        num_experts=2,
        top_k=1,
        expert_hidden=8,
        min_depth=1,
        max_depth=1,
        context_length=4,
        dtype="fp32",
        use_byte_conv=True,
        full_ternary_qat=True,
    )


def test_full_ternary_qat_wraps_all_trainable_layer_types():
    model = SmaulBrainModel(_tiny_full_ternary_config())
    assert isinstance(model.embed, TernaryQATEmbedding)
    assert isinstance(model.block.qkv, TernaryQATLinear)
    assert isinstance(model.block.o_proj, TernaryQATLinear)
    assert isinstance(model.router.proj, TernaryQATLinear)
    assert isinstance(model.head, TernaryQATLinear)
    assert isinstance(model.block.n1, TernaryQATRMSNorm)
    assert isinstance(model.n_final, TernaryQATRMSNorm)
    assert model.byte_conv is not None
    assert isinstance(model.byte_conv.conv, TernaryQATConv1d)

    forbidden = (nn.Linear, nn.Embedding, nn.Conv1d, HybridFP8Linear, RMSNorm)
    assert not [(name, type(layer).__name__) for name, layer in model.named_modules()
                if isinstance(layer, forbidden)]


def test_full_ternary_qat_runs_tiny_forward_backward_with_finite_grads():
    torch.manual_seed(7)
    model = SmaulBrainModel(_tiny_full_ternary_config())
    ids = torch.randint(0, 256, (1, 4))
    targets = torch.randint(0, 256, (1, 4))

    result = model(ids, targets, step=0)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()

    trunk_and_router = model._trunk_params() + model._router_params()
    checked = 0
    for name, parameter in trunk_and_router:
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            checked += 1
    assert checked > 0

    expert_grads = model.take_expert_grads()
    assert expert_grads
    for expert_id, weights in expert_grads.items():
        for name, grad in weights.items():
            assert torch.isfinite(grad).all(), f"{expert_id}.{name}"


def test_ternary_qat_quantizes_each_matrix_row_to_three_levels():
    weight = torch.tensor(
        [[2.0, -1.0, 0.01, -0.01], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    q = ternary_qat_weight(weight)
    assert torch.equal(q[1], torch.zeros_like(q[1]))
    row = q[0].detach()
    scale = row.abs().max()
    levels = torch.unique(torch.round(row / scale, decimals=6))
    assert set(levels.tolist()).issubset({-1.0, 0.0, 1.0})
    q.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))
