"""Recurrence: shared block reuse, genuine state influence, depth config."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from linear_attention import LinearAttnState
from recurrent import RecurrentState, SharedRecurrentBlock


def _block(d=32, h=4):
    torch.manual_seed(0)
    return SharedRecurrentBlock(d_model=d, n_heads=h)


def _moe(x):
    return x, torch.tensor(0.0), torch.zeros(x.shape[0], dtype=torch.long)


def test_shared_block_reuse_same_params():
    blk = _block()
    before = [p.clone() for p in blk.parameters()]
    h = torch.randn(2, 8, 32)
    n_params = sum(p.nelement() for p in blk.parameters())
    assert n_params > 0
    # Apply 4 times: parameter count must not grow (shared, not stacked).
    attn = LinearAttnState.zeros(2, 4, 8)
    for _ in range(4):
        h, attn, _, _ = blk(h, attn, _moe)
    assert sum(p.nelement() for p in blk.parameters()) == n_params
    for a, b in zip(before, blk.parameters()):
        assert torch.equal(a, b)  # forward does not mutate params


def test_recurrent_state_influences_output():
    blk = _block()
    h = torch.randn(2, 8, 32)
    s0 = LinearAttnState.zeros(2, 4, 8)
    s1 = LinearAttnState(S=torch.ones(2, 4, 8, 8), z=torch.ones(2, 4, 8))
    o0, _, _, _ = blk(h, s0, _moe)
    o1, _, _, _ = blk(h, s1, _moe)
    assert (o0 - o1).abs().max().item() > 1e-6  # state is NOT thrown away


def test_second_application_differs_from_first():
    blk = _block()
    h = torch.randn(2, 8, 32)
    attn = LinearAttnState.zeros(2, 4, 8)
    o1, attn, _, _ = blk(h, attn, _moe)
    o2, _, _, _ = blk(o1, attn, _moe)
    assert (o1 - o2).abs().max().item() > 1e-6


def test_recurrent_state_clone_independent():
    h = torch.randn(1, 4, 32)
    attn = LinearAttnState.zeros(1, 4, 8)
    st = RecurrentState(h=h, attn=attn)
    c = st.clone()
    c.h += 1.0
    assert not torch.equal(st.h, c.h)


def test_grad_flows_through_recurrence():
    blk = _block()
    h = torch.randn(2, 6, 32, requires_grad=True)
    attn = LinearAttnState.zeros(2, 4, 8)
    for _ in range(3):
        h, attn, _, _ = blk(h, attn, _moe)
    h.sum().backward()
    assert blk.qkv.weight.grad is not None
