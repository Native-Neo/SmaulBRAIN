import math

import numpy as np

from smaulbrain import math_ref as R
from smaulbrain import ops


def test_rmsnorm_against_scalar():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16)
    g = 0.5 + np.abs(rng.standard_normal(16))
    ref = R.rmsnorm_ref(list(x), list(g))
    y, cache = ops.rmsnorm_fwd(x.astype(np.float32), g.astype(np.float32))
    assert np.allclose(y, ref, atol=1e-5)


def test_rmsnorm_backward_against_scalar():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8).astype(np.float32)
    g = (0.5 + np.abs(rng.standard_normal(8))).astype(np.float32)
    dy = rng.standard_normal(8).astype(np.float32)
    _, cache = ops.rmsnorm_fwd(x, g)
    dx, dg = ops.rmsnorm_bwd(dy, cache)
    rdx, rdg = R.rmsnorm_backward_ref(list(x), list(g), list(dy))
    assert np.allclose(dx, rdx, atol=1e-4)
    assert np.allclose(dg, rdg, atol=1e-4)


def test_rmsnorm_backward_finite_difference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(6).astype(np.float32)
    g = (0.5 + np.abs(rng.standard_normal(6))).astype(np.float32)
    y, cache = ops.rmsnorm_fwd(x, g)
    dy = rng.standard_normal(6).astype(np.float32)
    dx, dg = ops.rmsnorm_bwd(dy, cache)
    eps = 1e-3
    num = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        yp, _ = ops.rmsnorm_fwd(xp, g)
        ym, _ = ops.rmsnorm_fwd(xm, g)
        num[i] = float(np.dot(yp - ym, dy) / (2 * eps))
    assert np.allclose(dx, num, atol=2e-2)


def test_recurrent_against_scalar_reference():
    rng = np.random.default_rng(3)
    d, T = 6, 5
    X = [list(rng.standard_normal(d)) for _ in range(T)]
    Wr = [[float(v) for v in row] for row in rng.standard_normal((d, d)) * 0.3]
    Wk = [[float(v) for v in row] for row in rng.standard_normal((d, d)) * 0.3]
    Wv = [[float(v) for v in row] for row in rng.standard_normal((d, d)) * 0.3]
    Wo = [[float(v) for v in row] for row in rng.standard_normal((d, d)) * 0.3]
    w = list(rng.standard_normal(d) * 0.1)
    u = list(rng.standard_normal(d) * 0.1)
    Yref, sref = R.recurrent_forward_ref(X, Wr, Wk, Wv, Wo, w, u)
    Xn = np.asarray(X, dtype=np.float32)
    attn = ops.RecurrentAttention.__new__(ops.RecurrentAttention)
    attn.Wr, attn.Wk, attn.Wv, attn.Wo = (np.asarray(a, dtype=np.float32)
                                          for a in (Wr, Wk, Wv, Wo))
    attn.w = np.asarray(w, dtype=np.float32)
    attn.u = np.asarray(u, dtype=np.float32)
    O, (s1, _c) = attn.forward(Xn)
    assert np.allclose(O, np.asarray(Yref), atol=1e-4)
    assert np.allclose(s1, np.asarray(sref), atol=1e-4)


def test_chunked_and_restored_state_equivalence():
    rng = np.random.default_rng(4)
    d, T = 8, 9
    attn = ops.RecurrentAttention(d, rng)
    X = rng.standard_normal((T, d)).astype(np.float32)
    Ofull, (sfull, _) = attn.forward(X)
    # chunked
    s = np.zeros(d, dtype=np.float32)
    outs = []
    for a, b in [(0, 4), (4, T)]:
        O, (s, _) = attn.forward(X[a:b], s0=s)
        outs.append(O)
    assert np.allclose(np.concatenate(outs), Ofull, atol=1e-5)
    # restored state: recompute suffix from serialized mid-state
    _, (smid, _) = attn.forward(X[:4])
    O2, _ = attn.forward(X[4:], s0=smid.clone())
    assert np.allclose(O2, Ofull[4:], atol=1e-5)


def test_cross_entropy_units_are_nats():
    logits = [1.0, 2.0, 3.0]
    nats = R.cross_entropy_nats_ref(logits, 2)
    m = max(logits)
    assert abs(nats - (m + math.log(sum(math.exp(l - m) for l in logits)) - 3.0)) < 1e-12
