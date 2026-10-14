"""Small correctness tests for opt-in full-model ternary QAT."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from config import SmaulBrainConfig
from experts import ExpertRecord, make_expert, ternary_qat_weight
from model import SmaulBrainModel
import native
import smaulopt as smaulopt_module
from smaulopt import SmaulOpt
from types import SimpleNamespace
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



def test_full_ternary_qat_config_roundtrip_and_type_validation():
    config = _tiny_full_ternary_config()
    restored = SmaulBrainConfig.from_dict(config.to_dict())
    assert restored.full_ternary_qat is True
    with pytest.raises(ValueError, match="full_ternary_qat must be a bool"):
        SmaulBrainConfig.from_dict({"full_ternary_qat": "yes"})


def test_full_ternary_expert_forward_backward_never_calls_dense_dequantize(monkeypatch):
    """Full ternary must fail the test if the old FP8->dense expert path is reached."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("full_ternary_qat must not dequantize an expert matrix")

    monkeypatch.setattr(ExpertRecord, "dequantize", forbidden)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    model = SmaulBrainModel(_tiny_full_ternary_config())
    ids = torch.randint(0, 256, (1, 4))
    targets = torch.randint(0, 256, (1, 4))
    result = model(ids, targets, step=0)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    grads = model.take_expert_grads()
    assert grads
    assert native.COUNTERS["fp8_to_ternary_native"] >= 1
    assert native.COUNTERS["ternary_linear_forward_native"] >= 3


def test_full_ternary_expert_optimizer_never_dense_dequantizes(monkeypatch):
    """The expert optimizer updates FP8 codes tile-by-tile with no dense weight decode."""
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    record = make_expert("expert_00000", d_model=8, expert_hidden=16, fp8_tile=8)
    compute = {
        "w_gate": SimpleNamespace(grad=torch.randn(16, 8).mul_(0.001).contiguous()),
        "w_up": SimpleNamespace(grad=torch.randn(16, 8).mul_(0.001).contiguous()),
        "w_down": SimpleNamespace(grad=torch.randn(8, 16).mul_(0.001).contiguous()),
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct expert optimizer must not materialize FP32 weights")

    monkeypatch.setattr(smaulopt_module, "dequantize_fp8_blockwise", forbidden)
    monkeypatch.setattr(smaulopt_module, "quantize_fp8_blockwise", forbidden)
    opt = SmaulOpt()
    opt.step_expert(record, compute, 1.0, 1e-3, ternary_direct=True)

    assert native.COUNTERS["fp8_factored_update_native"] == 3
    for name in ("w_gate", "w_up", "w_down"):
        assert record.optim_state[name]["step"] == 1
        assert compute[name].grad is None
        assert torch.isfinite(record.weights_fp8[name].scales).all()
    assert record.version == 1
