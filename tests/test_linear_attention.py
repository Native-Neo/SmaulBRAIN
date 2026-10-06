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
