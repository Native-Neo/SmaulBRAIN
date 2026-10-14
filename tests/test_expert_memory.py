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


def _tiny_model(dtype="bf16", ternary_qat=False):
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
        ternary_qat=ternary_qat,
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
        assert "expert_mlp_temp=" in str(exc.value)
        assert "expert_dequant_temp=" in str(exc.value)
        assert "expert_peak_temp=" in str(exc.value)
        assert "pager_cache=" in str(exc.value)
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

def test_ternary_linear_backward_matches_dense_autograd():
    from experts import _TernaryLinear

    torch.manual_seed(23)
    x0 = torch.randn(5, 4, dtype=torch.float32)
    weight0 = torch.tensor(
        [[1.0, 0.0, -0.5, 0.0],
         [0.0, 0.25, 0.0, -1.0],
         [-0.75, 0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    upstream = torch.randn(5, 3, dtype=torch.float32)

    x_native = x0.clone().requires_grad_(True)
    w_native = weight0.clone().requires_grad_(True)
    y_native = _TernaryLinear.apply(x_native, w_native, True)
    (y_native * upstream).sum().backward()

    x_dense = x0.clone().requires_grad_(True)
    w_dense = weight0.clone().requires_grad_(True)
    y_dense = torch.nn.functional.linear(x_dense, w_dense)
    (y_dense * upstream).sum().backward()

    assert torch.allclose(y_native, y_dense, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_native.grad, x_dense.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w_native.grad, w_dense.grad, atol=1e-6, rtol=1e-6)



def test_ternary_qat_custom_linear_preserves_ste_gradients_without_saving_cache():
    from experts import _TernaryQATLinear, ternary_qat_weight

    torch.manual_seed(29)
    x0 = torch.randn(5, 4, dtype=torch.float32)
    w0 = torch.randn(3, 4, dtype=torch.float32)
    upstream = torch.randn(5, 3, dtype=torch.float32)

    x_ref = x0.clone().requires_grad_(True)
    w_ref = w0.clone().requires_grad_(True)
    reference = torch.nn.functional.linear(x_ref, ternary_qat_weight(w_ref))
    (reference * upstream).sum().backward()

    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    saved_ptrs = []
    def pack_saved(tensor):
        saved_ptrs.append(tensor.data_ptr())
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack_saved, lambda tensor: tensor):
        result = _TernaryQATLinear.apply(x, w, False)
    (result * upstream).sum().backward()

    assert torch.allclose(result, reference, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w.grad, w_ref.grad, atol=1e-6, rtol=1e-6)
    # The custom op retains the input and latent weight, never a third
    # full-size tensor holding the ternary-transformed weight.
    assert saved_ptrs == [x.data_ptr(), w.data_ptr()]


def test_ternary_qat_checkpointed_swiglu_preserves_gradients():
    from experts import swiglu_forward, ternary_qat_weight

    torch.manual_seed(31)
    x0 = torch.randn(3, 4, dtype=torch.float32)
    values = {
        "w_gate": torch.randn(6, 4),
        "w_up": torch.randn(6, 4),
        "w_down": torch.randn(4, 6),
    }
    x_ref = x0.clone().requires_grad_(True)
    w_ref = {name: value.clone().requires_grad_(True) for name, value in values.items()}
    ref_gate = torch.nn.functional.linear(x_ref, ternary_qat_weight(w_ref["w_gate"]))
    ref_up = torch.nn.functional.linear(x_ref, ternary_qat_weight(w_ref["w_up"]))
    ref = torch.nn.functional.linear(
        torch.nn.functional.silu(ref_gate) * ref_up,
        ternary_qat_weight(w_ref["w_down"]),
    )
    x = x0.clone().requires_grad_(True)
    w = {name: value.clone().requires_grad_(True) for name, value in values.items()}
    got = swiglu_forward(x, w, ternary_qat=True)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(ref)
    (got * upstream).sum().backward()
    (ref * upstream).sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for name in w:
        assert torch.allclose(w[name].grad, w_ref[name].grad, atol=1e-5, rtol=1e-5)


def test_ternary_backward_reports_bf16_native_fallback():
    import native

    native.reset_counters()
    grad_out = torch.randn(2, 3, dtype=torch.bfloat16)
    weight = torch.randn(3, 4, dtype=torch.bfloat16)
    assert native.call_ternary_backward_input(grad_out, weight) is None
    assert native.COUNTERS["ternary_backward_fallback"] > 0
    reason = native.LAST_FALLBACK["ternary_backward"]
    assert "float32" in reason
    assert "bfloat16" in reason

def test_ternary_qat_projection_temporary_is_in_memory_budget(monkeypatch):
    model = _tiny_model(ternary_qat=True)
    try:
        ids = torch.zeros(1, 8, dtype=torch.long)
        monkeypatch.setattr(
            "builtins.open", mock_open(read_data="MemAvailable: 48 kB\\n")
        )
        with pytest.raises(ValueError, match="estimated training working set exceeds 65%") as exc:
            model._check_activation_budget(ids, training=True)
        assert "ternary_qat_projection_temp=" in str(exc.value)
        assert "ternary_qat_cache=" in str(exc.value)
        assert "expert_mlp_temp=" in str(exc.value)
    finally:
        model.pager.close()



def test_ternary_qat_reuses_cached_projection_until_leaf_changes(monkeypatch):
    import experts
    from experts import _TernaryQATLinear

    original = experts._ternary_values
    calls = 0

    def counted(weight, threshold_ratio=0.7):
        nonlocal calls
        calls += 1
        return original(weight, threshold_ratio)

    monkeypatch.setattr(experts, "_ternary_values", counted)
    torch.manual_seed(37)
    weight = torch.randn(5, 7, requires_grad=True)

    x1 = torch.randn(3, 7, requires_grad=True)
    _TernaryQATLinear.apply(x1, weight, False).square().sum().backward()
    assert calls == 1
    cached = getattr(weight, "_smaul_ternary_qat_cache")
    assert cached[0] == weight._version
    assert cached[1].shape == weight.shape
    assert not cached[1].requires_grad

    # Another recurrent-depth invocation and its backward reuse one Q matrix.
    x2 = torch.randn(4, 7, requires_grad=True)
    _TernaryQATLinear.apply(x2, weight, False).square().sum().backward()
    assert calls == 1
    assert getattr(weight, "_smaul_ternary_qat_cache")[1].data_ptr() == cached[1].data_ptr()

    # An in-place weight update increments Tensor._version; never return stale Q.
    with torch.no_grad():
        weight.add_(0.125)
    x3 = torch.randn(2, 7, requires_grad=True)
    _TernaryQATLinear.apply(x3, weight, False).sum().backward()
    assert calls == 2
    assert getattr(weight, "_smaul_ternary_qat_cache")[0] == weight._version



@pytest.mark.parametrize("dtype, atol", [
    (torch.float32, 5e-4),
    (torch.bfloat16, 3e-2),
])
def test_packed_native_ternary_qat_matches_reference_forward_and_gradients(dtype, atol):
    import native
    from experts import _TernaryQATLinear, ternary_qat_weight

    native.reset_counters()
    assert native.ensure_native(), native.status()

    torch.manual_seed(41)
    x0 = torch.randn(3, 7, dtype=dtype)
    w0 = torch.randn(5, 7, dtype=dtype)
    upstream = torch.randn(3, 5, dtype=dtype)

    x_ref = x0.clone().requires_grad_(True)
    w_ref = w0.clone().requires_grad_(True)
    y_ref = torch.nn.functional.linear(x_ref, ternary_qat_weight(w_ref))
    (y_ref * upstream).sum().backward()

    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    y = _TernaryQATLinear.apply(x, w, True)
    (y * upstream).sum().backward()

    assert torch.allclose(y, y_ref, atol=atol, rtol=atol)
    assert torch.allclose(x.grad, x_ref.grad, atol=atol, rtol=atol)
    assert torch.allclose(w.grad, w_ref.grad, atol=atol, rtol=atol)
    assert native.COUNTERS["ternary_linear_forward_native"] > 0
    assert native.COUNTERS["ternary_linear_backward_native"] > 0


def test_packed_ternary_codes_cross_row_boundaries_without_padding():
    from experts import _cached_packed_ternary_values, _ternary_values

    weight = torch.tensor([
        [2.0, 0.0, -2.0],
        [-1.0, 1.0, 0.0],
    ], dtype=torch.float32, requires_grad=True)
    packed, scales = _cached_packed_ternary_values(weight, threshold_ratio=0.0)
    # Decode enough of the packed stream to verify flattened row-major order.
    decoded = []
    for index in range(weight.numel()):
        decoded.append((int(packed[index // 4]) >> ((index % 4) * 2)) & 3)
    q = _ternary_values(weight, threshold_ratio=0.0)
    expected = [
        1 if value > 0 else 2 if value < 0 else 0
        for value in q.detach().reshape(-1).tolist()
    ]
    assert decoded == expected
    assert torch.allclose(scales, q.detach().abs().amax(dim=-1))
