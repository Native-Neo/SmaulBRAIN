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


def test_block_returns_fresh_state_without_aliasing():
    blk = _block()
    h = torch.randn(2, 6, 32)
    attn = LinearAttnState.zeros(2, 4, 8)
    out, nxt, _, _ = blk(h, attn, _moe)
    assert nxt is not attn  # update delivered exactly once, as a new object
    assert nxt.S.data_ptr() != attn.S.data_ptr()
    assert nxt.z.data_ptr() != attn.z.data_ptr()
    assert out.data_ptr() != h.data_ptr()
    # Mutating the returned state must not touch the caller-owned input.
    nxt.S += 100.0
    nxt.z += 100.0
    assert attn.S.abs().sum().item() == 0
    assert attn.z.abs().sum().item() == 0


def test_block_rejects_state_shape_and_dtype_mismatch():
    import pytest
    blk = _block()
    h = torch.randn(2, 6, 32)
    ok = LinearAttnState.zeros(2, 4, 8)
    assert ok.matches(2, 4, 8)
    with pytest.raises(ValueError):
        blk(h, LinearAttnState.zeros(2, 3, 8), _moe)  # head-count mismatch
    with pytest.raises(ValueError):
        blk(h, LinearAttnState.zeros(2, 4, 7), _moe)  # head-dim mismatch
    bad = LinearAttnState(S=torch.zeros(2, 4, 8, 8, dtype=torch.bfloat16),
                          z=torch.zeros(2, 4, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        blk(h, bad, _moe)  # accumulators must be fp32
    with pytest.raises(ValueError):
        blk(torch.randn(6, 32), ok, _moe)  # h must be [B, T, D]
    with pytest.raises(ValueError):
        blk(torch.randn(2, 6, 16), ok, _moe)  # h width must equal d_model


def test_block_empty_chunk_and_empty_mid_stream():
    blk = _block()
    h = torch.randn(1, 9, 32)
    attn = LinearAttnState.zeros(1, 4, 8)
    full, full_state, full_halt, _ = blk(h.clone(), attn.clone(), _moe)
    # Single empty pass: empty outputs, state value unchanged in a new object.
    e, e_state, e_halt, _ = blk(h[:, :0].clone(), attn.clone(), _moe)
    assert e.shape == (1, 0, 32) and e_halt.shape == (1, 0)
    assert torch.equal(e_state.S, attn.S) and torch.equal(e_state.z, attn.z)
    assert e_state is not attn
    # Empty chunk mid-stream is an identity: [prefix, empty, suffix] == full.
    s = LinearAttnState.zeros(1, 4, 8)
    a, s, ha, _ = blk(h[:, :4].clone(), s, _moe)
    m, s, hm, _ = blk(h[:, 4:4].clone(), s, _moe)
    b, s, hb, _ = blk(h[:, 4:].clone(), s, _moe)
    assert m.shape[1] == 0 and hm.shape[1] == 0
    assert torch.allclose(torch.cat([a, m, b], dim=1), full, atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.cat([ha, hm, hb], dim=1), full_halt, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.S, full_state.S, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.z, full_state.z, atol=2e-5, rtol=2e-5)


def test_block_token_stream_matches_full_with_halt():
    blk = _block()
    # Nontrivial halt head: with fresh init halt is constant bias, so make
    # halt input-dependent to prove halt logits stream correctly too.
    with torch.no_grad():
        blk.halt.weight.normal_(std=0.1)
    h = torch.randn(1, 9, 32)
    full, full_state, full_halt, _ = blk(h.clone(), LinearAttnState.zeros(1, 4, 8),
                                         _moe, chunk_size=64)
    s = LinearAttnState.zeros(1, 4, 8)
    outs, halts = [], []
    for t in range(9):
        o, s, hl, _ = blk(h[:, t:t + 1].clone(), s, _moe, chunk_size=2)
        outs.append(o)
        halts.append(hl)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.cat(halts, dim=1), full_halt, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.S, full_state.S, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.z, full_state.z, atol=2e-5, rtol=2e-5)


def test_block_variable_splits_and_repeated_reuse():
    blk = _block()
    with torch.no_grad():
        blk.halt.weight.normal_(std=0.1)
    torch.manual_seed(7)
    h = torch.randn(2, 11, 32)
    full, full_state, full_halt, _ = blk(h.clone(), LinearAttnState.zeros(2, 4, 8),
                                         _moe, chunk_size=64)
    # Arbitrary multi-way split with mixed chunk sizes == one causal stream.
    bounds, sizes = [0, 2, 3, 7, 11], [5, 1, 4, 2]
    s = LinearAttnState.zeros(2, 4, 8)
    outs, halts = [], []
    for (lo, hi), cs in zip(zip(bounds[:-1], bounds[1:]), sizes):
        o, s, hl, _ = blk(h[:, lo:hi].clone(), s, _moe, chunk_size=cs)
        outs.append(o)
        halts.append(hl)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.cat(halts, dim=1), full_halt, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.S, full_state.S, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.z, full_state.z, atol=2e-5, rtol=2e-5)
    # Repeated reuse: same caller-owned state twice is deterministic and pristine.
    a = LinearAttnState.zeros(2, 4, 8)
    o1, _, _, _ = blk(h.clone(), a, _moe)
    o2, _, _, _ = blk(h.clone(), a, _moe)
    assert torch.equal(o1, o2)
    assert a.S.abs().sum().item() == 0 and a.z.abs().sum().item() == 0
    # Reset-then-reuse equals a fresh stream; fresh zeros never alias.
    r1, r2 = RecurrentState.zeros(2, 4, 32, 4), RecurrentState.zeros(2, 4, 32, 4)
    assert r1.h.data_ptr() != r2.h.data_ptr()
    r1.h += 1.0
    assert r2.h.abs().sum().item() == 0
    st = RecurrentState.zeros(2, 4, 32, 4)
    st.h += 3.0
    st.attn.S += 2.0
    st.steps_taken = 5
    st.reset()
    fresh = RecurrentState.zeros(2, 4, 32, 4)
    assert torch.equal(st.h, fresh.h) and torch.equal(st.attn.S, fresh.attn.S)
    assert st.steps_taken == 0


def test_block_keep_carries_across_chunks():
    blk = _block()
    torch.manual_seed(3)
    h = torch.randn(2, 8, 32)
    keep = torch.ones(2, 8, dtype=torch.bool)
    keep[:, 5:] = False  # trailing pads are not data
    _, kept_state, _, _ = blk(h.clone(), LinearAttnState.zeros(2, 4, 8),
                              _moe, keep=keep)
    # Same masked stream fed as chunks with split masks: pads add nothing
    # across the boundary either.
    s = LinearAttnState.zeros(2, 4, 8)
    _, s, _, _ = blk(h[:, :3].clone(), s, _moe, keep=keep[:, :3])
    _, s, _, _ = blk(h[:, 3:].clone(), s, _moe, keep=keep[:, 3:])
    assert torch.allclose(s.S, kept_state.S, atol=2e-5, rtol=2e-5)
    assert torch.allclose(s.z, kept_state.z, atol=2e-5, rtol=2e-5)


def test_long_stream_finite_and_memory_constant():
    blk = _block(d=16, h=2)
    torch.manual_seed(1)
    h = torch.randn(1, 512, 16)
    attn = LinearAttnState.zeros(1, 2, 8)
    out, st, _, _ = blk(h, attn, _moe, chunk_size=64)
    assert torch.isfinite(out).all() and torch.isfinite(st.S).all()
    assert st.nbytes() == LinearAttnState.zeros(1, 2, 8).nbytes()
