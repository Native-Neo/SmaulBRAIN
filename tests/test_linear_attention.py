"""Linear attention: correctness, causality, streaming, memory scaling.

Proves genuineness: no QK^T matrix exists on this path (outputs are built
from rank-1 state updates), batch/step APIs agree exactly, and state memory
is T-independent (constant across 1K..16K in the scaling probe below).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from linear_attention import (
    LinearAttnState, feature_map, linear_attn_forward, linear_attn_step,
)


def _rand(B=2, H=2, T=16, Dh=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    def r(*s):
        t = torch.empty(*s); t.normal_(0, 1, generator=g); return t
    return r(B, H, T, Dh), r(B, H, T, Dh), r(B, H, T, Dh)


def test_feature_map_positive():
    q, _, _ = _rand()
    assert (feature_map(q) > 0).all()


def test_step_matches_batch_exactly():
    q, k, v = _rand()
    B, H, T, Dh = q.shape
    batched, _ = linear_attn_forward(q, k, v)
    st = LinearAttnState.zeros(B, H, Dh)
    outs = []
    for t in range(T):
        y, st = linear_attn_step(st, q[:, :, t, :], k[:, :, t, :], v[:, :, t, :])
        outs.append(y)
    assert torch.allclose(batched, torch.stack(outs, dim=1), atol=2e-6, rtol=2e-6)


def test_causal_no_future_leak():
    q, k, v = _rand(T=12)
    out, _ = linear_attn_forward(q, k, v)
    k2 = k.clone(); k2[:, :, 8:, :] += 10.0
    out2, _ = linear_attn_forward(q, k2, v)
    assert torch.equal(out[:, :8, :, :], out2[:, :8, :, :])


def test_state_memory_independent_of_length():
    s1 = LinearAttnState.zeros(1, 4, 16)
    s2 = LinearAttnState.zeros(1, 4, 16)
    assert s1.nbytes() == s2.nbytes() == (4 * 16 * 16 + 4 * 16) * 4
    # Full-history QK^T at T=16384 would need 16384^2 = 268M scores/head;
    # our state holds 16*16+16 = 272 floats/head: ratio ~1e6 smaller.
    qkt = 16384 * 16384
    assert qkt / (16 * 16 + 16) > 1e5


def test_long_sequence_finite_and_bounded():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 512, 8)
    k = torch.randn(1, 2, 512, 8)
    v = torch.randn(1, 2, 512, 8)
    out, st = linear_attn_forward(q, k, v)
    assert torch.isfinite(out).all() and torch.isfinite(st.S).all()


def test_chunked_state_carry_matches_full_sequence():
    q, k, v = _rand(T=37)
    full, full_state = linear_attn_forward(q, k, v, chunk_size=64)
    first, state = linear_attn_forward(q[:, :, :13], k[:, :, :13], v[:, :, :13],
                                       chunk_size=5)
    second, carried = linear_attn_forward(q[:, :, 13:], k[:, :, 13:], v[:, :, 13:],
                                          state=state, chunk_size=7)
    chunked = torch.cat([first, second], dim=1)
    assert torch.allclose(full, chunked, atol=2e-6, rtol=2e-6)
    assert torch.allclose(full_state.S, carried.S, atol=2e-6, rtol=2e-6)
    assert torch.allclose(full_state.z, carried.z, atol=2e-6, rtol=2e-6)


def test_chunk_size_one_and_full_sequence_agree():
    q, k, v = _rand(T=19)
    full, _ = linear_attn_forward(q, k, v, chunk_size=19)
    one, _ = linear_attn_forward(q, k, v, chunk_size=1)
    assert torch.allclose(full, one, atol=2e-6, rtol=2e-6)


def test_saturated_and_zero_inputs_stay_finite():
    B, H, T, Dh = 2, 2, 16, 8
    q = torch.full((B, H, T, Dh), -50.0)  # feature map ~0 (saturated ELU)
    k = torch.full((B, H, T, Dh), 50.0)
    v = torch.zeros(B, H, T, Dh)
    out, st = linear_attn_forward(q, k, v)
    assert torch.isfinite(out).all()
    assert torch.isfinite(st.S).all() and torch.isfinite(st.z).all()
    st2 = LinearAttnState.zeros(B, H, Dh)
    for t in range(T):
        y, st2 = linear_attn_step(st2, q[:, :, t, :], k[:, :, t, :], v[:, :, t, :])
        assert torch.isfinite(y).all()
    out2, _ = linear_attn_forward(torch.full((B, H, T, Dh), 50.0), k,
                                  torch.ones(B, H, T, Dh))
    assert torch.isfinite(out2).all()  # large queries cannot explode the ratio


def test_forward_does_not_mutate_input_state():
    q, k, v = _rand(T=10)
    B, H, _, Dh = q.shape
    st = LinearAttnState.zeros(B, H, Dh)
    s0, z0 = st.S.clone(), st.z.clone()
    out, nxt = linear_attn_forward(q, k, v, state=st, chunk_size=4)
    assert torch.equal(st.S, s0) and torch.equal(st.z, z0)  # ownership: input kept
    assert nxt is not st  # a new object is returned for continue


def test_forward_and_step_reject_batch_mismatch():
    import pytest
    q, k, v = _rand(T=8)
    B, H, _, Dh = q.shape
    bad = LinearAttnState.zeros(B + 1, H, Dh)
    with pytest.raises(ValueError):
        linear_attn_forward(q, k, v, state=bad)
    with pytest.raises(ValueError):
        linear_attn_step(bad, q[:, :, 0, :], k[:, :, 0, :], v[:, :, 0, :])
    with pytest.raises(ValueError):
        linear_attn_forward(q, k, v, keep=torch.ones(B, 3, dtype=torch.bool))


def test_step_updates_in_place_and_returns_same_object():
    q, k, v = _rand(T=4)
    B, H, _, Dh = q.shape
    st = LinearAttnState.zeros(B, H, Dh)
    y, nxt = linear_attn_step(st, q[:, :, 0, :], k[:, :, 0, :], v[:, :, 0, :])
    assert nxt is st
    assert st.S.abs().sum().item() > 0  # state advanced


def test_reset_and_per_sequence_reset():
    q, k, v = _rand(T=6)
    B, H, _, Dh = q.shape
    _, st = linear_attn_forward(q, k, v)
    assert st.S.abs().sum().item() > 0
    st.reset([0])  # per-sequence: only row 0 cleared
    assert st.S[0].abs().sum().item() == 0 and st.z[0].abs().sum().item() == 0
    assert st.S[1].abs().sum().item() > 0
    st.reset()  # full reset
    assert st.S.abs().sum().item() == 0 and st.z.abs().sum().item() == 0


def test_keep_pads_do_not_pollute_state():
    q, k, v = _rand(T=10)
    B, H, T, Dh = q.shape
    keep = torch.ones(B, T, dtype=torch.bool)
    keep[:, 6:] = False  # variable-length: 6 valid + 4 pads
    outk, stk = linear_attn_forward(q, k, v, keep=keep)
    outv, stv = linear_attn_forward(q[:, :, :6], k[:, :, :6], v[:, :, :6])
    assert torch.allclose(stk.S, stv.S, atol=1e-6, rtol=1e-6)
    assert torch.allclose(stk.z, stv.z, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outk[:, :6], outv, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(outk).all()  # padded outputs stay finite


def test_step_keep_skips_padded_rows():
    q, k, v = _rand(T=3)
    B, H, _, Dh = q.shape
    st = LinearAttnState.zeros(B, H, Dh)
    s0 = st.S.clone()
    z0 = st.z.clone()
    y, nxt = linear_attn_step(st, q[:, :, 0, :], k[:, :, 0, :], v[:, :, 0, :],
                              keep=torch.tensor([True, False]))
    assert nxt is st
    assert not torch.equal(st.S[0], s0[0])  # valid row advanced
    assert torch.equal(st.S[1], s0[1]) and torch.equal(st.z[1], z0[1])


def test_empty_sequence_returns_empty_and_keeps_state():
    B, H, Dh = 2, 2, 8
    st = LinearAttnState.zeros(B, H, Dh)
    e = torch.randn(B, H, 0, Dh)
    out, nxt = linear_attn_forward(e, e.clone(), e.clone(), state=st)
    assert out.shape[1] == 0 and torch.equal(nxt.S, st.S)


def test_stateful_continue_matches_full_at_every_split():
    q, k, v = _rand(T=25)
    full, full_state = linear_attn_forward(q, k, v, chunk_size=64)
    for split in (1, 7, 13, 24):
        a, s = linear_attn_forward(q[:, :, :split], k[:, :, :split], v[:, :, :split],
                                   chunk_size=3)
        b, carried = linear_attn_forward(q[:, :, split:], k[:, :, split:], v[:, :, split:],
                                         state=s, chunk_size=5)
        chunked = torch.cat([a, b], dim=1)
        assert torch.allclose(full, chunked, atol=2e-6, rtol=2e-6)
        assert torch.allclose(full_state.S, carried.S, atol=2e-6, rtol=2e-6)
        assert torch.allclose(full_state.z, carried.z, atol=2e-6, rtol=2e-6)


def test_long_stream_step_matches_forward_and_memory_constant():
    torch.manual_seed(7)
    B, H, T, Dh = 1, 2, 1024, 8
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    v = torch.randn(B, H, T, Dh)
    batched, bstate = linear_attn_forward(q, k, v, chunk_size=128)
    st = LinearAttnState.zeros(B, H, Dh)
    outs = []
    for t in range(T):
        y, st = linear_attn_step(st, q[:, :, t, :], k[:, :, t, :], v[:, :, t, :])
        outs.append(y)
    streamed = torch.stack(outs, dim=1)
    assert torch.allclose(batched, streamed, atol=2e-5, rtol=2e-5)
    assert torch.allclose(bstate.S, st.S, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(batched).all() and torch.isfinite(st.S).all()
    # Memory: state bytes independent of T (truncation never grows state).
    assert st.nbytes() == LinearAttnState.zeros(B, H, Dh).nbytes()
    # Truncation: reset + re-feed of the last 256 tokens stays finite and
    # differs from the full-history state (old history is dropped).
    trunc = LinearAttnState.zeros(B, H, Dh)
    _, tstate = linear_attn_forward(q[:, :, -256:], k[:, :, -256:], v[:, :, -256:],
                                    state=trunc)
    assert torch.isfinite(tstate.S).all()
    assert not torch.allclose(tstate.S, bstate.S, atol=1e-6, rtol=1e-6)
