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
    c.attn.S += 1.0
    c.attn.z += 1.0
    assert torch.equal(st.attn.S, attn.S)  # clone is deep: no aliasing
    assert not torch.equal(st.attn.S, c.attn.S)


def test_grad_flows_through_recurrence():
    blk = _block()
    h = torch.randn(2, 6, 32, requires_grad=True)
    attn = LinearAttnState.zeros(2, 4, 8)
    for _ in range(3):
        h, attn, _, _ = blk(h, attn, _moe)
    h.sum().backward()
    assert blk.qkv.weight.grad is not None


def test_block_does_not_mutate_inputs():
    blk = _block()
    h = torch.randn(2, 6, 32)
    attn = LinearAttnState.zeros(2, 4, 8)
    h0, s0, z0 = h.clone(), attn.S.clone(), attn.z.clone()
    blk(h, attn, _moe)
    assert torch.equal(h, h0)
    assert torch.equal(attn.S, s0) and torch.equal(attn.z, z0)


def test_block_rejects_batch_and_chunk_mismatch():
    import pytest
    blk = _block()
    h = torch.randn(2, 6, 32)
    with pytest.raises(ValueError):
        blk(h, LinearAttnState.zeros(3, 4, 8), _moe)
    with pytest.raises(ValueError):
        blk(h, LinearAttnState.zeros(2, 4, 8), _moe, chunk_size=0)
    with pytest.raises(ValueError):
        blk(h, LinearAttnState.zeros(2, 4, 8), _moe,
            keep=torch.ones(2, 3, dtype=torch.bool))


def test_block_chunk_size_invariant():
    blk = _block()
    h = torch.randn(2, 8, 32)
    attn = LinearAttnState.zeros(2, 4, 8)
    o_full, s_full, _, _ = blk(h.clone(), attn.clone(), _moe, chunk_size=256)
    o_one, s_one, _, _ = blk(h.clone(), attn.clone(), _moe, chunk_size=1)
    assert torch.allclose(o_full, o_one, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s_full.S, s_one.S, atol=2e-5, rtol=2e-5)


def test_block_stateful_continue_matches_full():
    blk = _block()
    h = torch.randn(1, 9, 32)
    attn = LinearAttnState.zeros(1, 4, 8)
    full, full_state, _, _ = blk(h.clone(), attn.clone(), _moe, chunk_size=64)
    for split in (1, 4, 8):
        a, s, _, _ = blk(h[:, :split].clone(), LinearAttnState.zeros(1, 4, 8),
                          _moe, chunk_size=3)
        b, carried, _, _ = blk(h[:, split:].clone(), s, _moe, chunk_size=2)
        chunked = torch.cat([a, b], dim=1)
        assert torch.allclose(full, chunked, atol=2e-5, rtol=2e-5)
        assert torch.allclose(full_state.S, carried.S, atol=2e-5, rtol=2e-5)


def test_block_keep_pads_match_valid_prefix():
    blk = _block()
    h = torch.randn(2, 8, 32)
    attn = LinearAttnState.zeros(2, 4, 8)
    keep = torch.ones(2, 8, dtype=torch.bool)
    keep[:, 5:] = False
    _, kept_state, _, _ = blk(h.clone(), attn.clone(), _moe, keep=keep)
    _, prefix_state, _, _ = blk(h[:, :5].clone(), LinearAttnState.zeros(2, 4, 8), _moe)
    assert torch.allclose(kept_state.S, prefix_state.S, atol=2e-5, rtol=2e-5)
    assert torch.allclose(kept_state.z, prefix_state.z, atol=2e-5, rtol=2e-5)


def test_recurrent_state_zeros_and_per_sequence_reset():
    st = RecurrentState.zeros(2, 4, 32, 4)
    assert st.h.shape == (2, 4, 32) and st.attn.S.shape == (2, 4, 8, 8)
    st.h += 3.0
    st.attn.S += 2.0
    st.reset([0])
    assert st.h[0].abs().sum().item() == 0 and st.attn.S[0].abs().sum().item() == 0
    assert st.h[1].abs().sum().item() > 0 and st.attn.S[1].abs().sum().item() > 0
    st.reset()
    assert st.h.abs().sum().item() == 0 and st.attn.S.abs().sum().item() == 0


def test_halt_grad_does_not_flow_into_trunk():
    blk = _block()
    h = torch.randn(2, 4, 32, requires_grad=True)
    attn = LinearAttnState.zeros(2, 4, 8)
    _, _, halt_logit, _ = blk(h, attn, _moe)
    halt_logit.sum().backward()
    assert h.grad is None or h.grad.abs().sum().item() == 0
    assert blk.qkv.weight.grad is None  # trunk untouched by halt loss


def test_long_stream_finite_and_memory_constant():
    blk = _block(d=16, h=2)
    torch.manual_seed(1)
    h = torch.randn(1, 512, 16)
    attn = LinearAttnState.zeros(1, 2, 8)
    out, st, _, _ = blk(h, attn, _moe, chunk_size=64)
    assert torch.isfinite(out).all() and torch.isfinite(st.S).all()
    assert st.nbytes() == LinearAttnState.zeros(1, 2, 8).nbytes()
