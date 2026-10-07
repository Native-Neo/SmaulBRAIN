"""Hybrid FP8 linear: E4M3 forward / E5M2 backward correctness."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hybrid_fp8 import E5M2_MAX, HybridFP8Linear


def _pair(seed=0, with_bias=True):
    torch.manual_seed(seed)
    hy = HybridFP8Linear(32, 16, bias=with_bias)
    ref = torch.nn.Linear(32, 16, bias=with_bias).to(torch.bfloat16)
    with torch.no_grad():
        ref.weight.copy_(hy.weight.to(torch.bfloat16))
        if with_bias:
            ref.bias.copy_(hy.bias.to(torch.bfloat16))
    return hy, ref


def test_forward_matches_bf16_linear():
    hy, ref = _pair()
    x = torch.randn(8, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        got, want = hy(x).float(), ref(x).float()
    assert got.shape == want.shape
    assert torch.isfinite(got).all()
    assert (got - want).abs().max().item() < 0.15  # E4M3 quant noise envelope


def test_backward_flows_with_finite_grads():
    hy, _ = _pair()
    x = torch.randn(8, 32, dtype=torch.bfloat16, requires_grad=True)
    loss = hy(x).float().pow(2).mean()
    loss.backward()
    assert hy.weight.grad is not None and hy.bias.grad is not None
    assert torch.isfinite(hy.weight.grad).all()
    assert torch.isfinite(hy.bias.grad).all()
    assert torch.isfinite(x.grad).all()
    assert x.grad.dtype == torch.bfloat16  # grad matches input dtype
    assert hy.in_scaler.history and hy.grad_scaler.history  # streams ticked


def test_loss_converges_over_ten_steps():
    torch.manual_seed(0)
    hy = HybridFP8Linear(32, 16)
    opt = torch.optim.Adam(hy.parameters(), lr=1e-2)
    x = torch.randn(8, 32, dtype=torch.bfloat16)
    tgt = torch.randn(8, 16)
    first, last = None, None
    for _ in range(10):
        opt.zero_grad()
        loss = (hy(x).float() - tgt).pow(2).mean()
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step()
        last = float(loss.detach())
    assert last < 0.5 * first


def test_zero_and_huge_inputs_stay_finite():
    hy, _ = _pair(seed=1)
    with torch.no_grad():
        z = hy(torch.zeros(4, 32, dtype=torch.bfloat16))
        assert torch.isfinite(z).all()  # zero input -> bias only, still finite
        big = hy(torch.full((4, 32), 1e4, dtype=torch.bfloat16))
        assert torch.isfinite(big).all()  # clamped, never inf


def test_nonfinite_gradients_zero_loudly_safe():
    hy, _ = _pair(seed=2)
    x = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    out = hy(x)
    grad_out = torch.full_like(out, float("inf"))
    grads = torch.autograd.grad(out, (x, hy.weight), grad_outputs=grad_out,
                                allow_unused=True)
    for g in grads:
        assert g is not None and torch.isfinite(g).all() and bool((g == 0).all())


def test_no_bias_variant_trains():
    hy, _ = _pair(seed=3, with_bias=False)
    assert hy.bias is None
    opt = torch.optim.SGD(hy.parameters(), lr=1e-2)
    x = torch.randn(8, 32, dtype=torch.bfloat16)
    tgt = torch.randn(8, 16)
    opt.zero_grad()
    before = float(((hy(x).float() - tgt).pow(2).mean()).detach())
    for _ in range(5):
        opt.zero_grad()
        loss = (hy(x).float() - tgt).pow(2).mean()
        loss.backward()
        opt.step()
    after = float(((hy(x).float() - tgt).pow(2).mean()).detach())
    assert after < before


def test_e5m2_range_exceeds_e4m3():
    assert E5M2_MAX > 100 * 448.0  # grads get the wide exponent range
