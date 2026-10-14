"""Small correctness tests for opt-in full-model ternary QAT."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from config import SmaulBrainConfig
import experts as experts_module
from experts import ExpertRecord, make_expert, ternary_qat_weight
from model import SmaulBrainModel
import native
import smaulopt as smaulopt_module
import growth as growth_module
from smaulopt import SmaulOpt, SmaulOptHParams
from types import SimpleNamespace
from copy import deepcopy
from precision import HybridFP8Linear
from rmsnorm import RMSNorm
from ternary import (
    TernaryQATConv1d,
    TernaryQATEmbedding,
    TernaryQATLinear,
    TernaryQATRMSNorm,
)


def _tiny_full_ternary_config(paging_method="D2R"):
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
        paging_method=paging_method,
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


@pytest.mark.parametrize("paging_method", ["D2R", "R2VR", "D2VR"])
def test_full_ternary_expert_forward_backward_never_calls_dense_dequantize(
    monkeypatch, paging_method,
):
    def forbidden_dense(*_args, **_kwargs):
        raise AssertionError("full_ternary_qat must not materialize dense ternary weight matrices")

    monkeypatch.setattr(experts_module, "_cached_ternary_values", forbidden_dense)
    monkeypatch.setattr(experts_module, "_cached_packed_ternary_values", forbidden_dense)
    """No paging mode may route full ternary through dense expert dequantization."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("full_ternary_qat must not dequantize an expert matrix")

    monkeypatch.setattr(ExpertRecord, "dequantize", forbidden)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    native.reset_counters()
    model = SmaulBrainModel(_tiny_full_ternary_config(paging_method))
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


def test_direct_no_dequant_optimizer_matches_fp8_reference_update(monkeypatch):
    """The scalar updater must preserve SmaulOpt math within FP8 quantization."""
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    torch.manual_seed(29)
    direct_record = make_expert(
        "expert_00007", d_model=8, expert_hidden=16, fp8_tile=8
    )
    reference_record = deepcopy(direct_record)
    grads = {
        "w_gate": torch.randn(16, 8).mul_(0.001).contiguous(),
        "w_up": torch.randn(16, 8).mul_(0.001).contiguous(),
        "w_down": torch.randn(8, 16).mul_(0.001).contiguous(),
    }
    direct_compute = {
        name: SimpleNamespace(grad=grad.clone()) for name, grad in grads.items()
    }
    reference_compute = {
        name: SimpleNamespace(grad=grad.clone()) for name, grad in grads.items()
    }
    hp = SmaulOptHParams(state_dtype="fp32")
    direct_opt, reference_opt = SmaulOpt(hp), SmaulOpt(deepcopy(hp))
    direct_opt.step_expert(
        direct_record, direct_compute, 1.0, 0.02, ternary_direct=True
    )
    reference_opt.step_expert(
        reference_record, reference_compute, 1.0, 0.02
    )

    from precision import dequantize_fp8_blockwise
    for name in ("w_gate", "w_up", "w_down"):
        direct = direct_record.weights_fp8[name]
        reference = reference_record.weights_fp8[name]
        direct_values = dequantize_fp8_blockwise(direct, dtype=torch.float32)
        reference_values = dequantize_fp8_blockwise(reference, dtype=torch.float32)
        assert torch.allclose(direct_values, reference_values, atol=2e-3, rtol=2e-3), name
        for moment in ("m", "v_row", "v_col"):
            assert torch.allclose(
                direct_record.optim_state[name][moment],
                reference_record.optim_state[name][moment],
                atol=2e-6, rtol=2e-4,
            ), f"{name}.{moment}"


def test_full_ternary_growth_clones_fp8_without_dense_dequant_and_keeps_router_qat(monkeypatch):
    """Training growth uses the direct FP8 clone kernel, never dense parent weights."""
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    model = SmaulBrainModel(_tiny_full_ternary_config())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full_ternary_qat growth must not dequantize expert weights")

    monkeypatch.setattr(ExpertRecord, "dequantize", forbidden)
    monkeypatch.setattr(growth_module, "dequantize_fp8_blockwise", forbidden)
    old_count = len(model.pool)
    new_ids = growth_module.grow_topk_clones(
        model.pool, model.router, model.cfg.d_model, model.cfg.expert_hidden,
        step=1, seed=42, k=1, fp8_tile=model.cfg.fp8_tile,
        max_experts=model.cfg.max_experts, direct_no_dequant=True,
    )
    assert len(new_ids) == 1
    assert len(model.pool) == old_count + 1
    assert model.router.num_experts == old_count + 1
    assert isinstance(model.router.proj, TernaryQATLinear)
    child = model.pool.experts[new_ids[0]]
    assert child.source == "clone-mutated"
    assert child.parents
    for name in ("w_gate", "w_up", "w_down"):
        assert child.weights_fp8[name].codes.dtype == torch.uint8
        assert torch.isfinite(child.weights_fp8[name].scales).all()
