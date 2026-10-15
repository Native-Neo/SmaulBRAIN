"""Scalar reference mathematics for SmaulBRAIN V2.

Authoritative equations (also documented in docs/architecture.md):

RMSNorm: y_i = x_i / sqrt(mean(x^2) + eps) * g_i ; g init ones, eps=1e-6.

Recurrent linear attention (per layer, per position t, dim d):
  r_t = x_t @ W_r ; k_t = x_t @ W_k ; v_t = x_t @ W_v
  alpha = exp(-softplus(w)) in (0,1), elementwise, w init ~ N(0,0.1)
  s_t = alpha * s_{t-1} + k_t * v_t        (elementwise, s_0 = 0)
  g_t = sigmoid(r_t)
  y_t = g_t * (s_t + u * k_t * v_t)        (u: learned bonus, init 0)
  o_t = y_t @ W_o
State: s in R^d, fp32 accumulation. Memory O(d) per layer, independent of T.

Gated channel mixing (dense): a = x@W_a, b = x@W_b, h = silu(a)*b, y = h@W_c.
Expert MLP (MoE): h = gelu(x@W_in), y = h@W_out.
Final: RMSNorm then tied logits = h @ E^T.

All refs below are plain-Python scalar loops (no numpy vectorization) so
optimized kernels can be tested against them.
"""
import math

EPS = 1e-6


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def silu(x: float) -> float:
    return x * sigmoid(x)


def gelu_tanh(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def rmsnorm_ref(x: list, g: list, eps: float = EPS):
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    return [x[i] * inv * g[i] for i in range(n)]


def rmsnorm_backward_ref(x: list, g: list, dy: list, eps: float = EPS):
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = math.sqrt(ms + eps)
    xn = [v / inv for v in x]
    dg = [dy[i] * xn[i] for i in range(n)]
    d_xn = [dy[i] * g[i] for i in range(n)]
    dot = sum(d_xn[i] * x[i] for i in range(n))
    dx = [(d_xn[i] - x[i] * dot / (n * (ms + eps))) / inv for i in range(n)]
    return dx, dg


def matvec_ref(x: list, W: list) -> list:
    """x: (d_in,), W: rows d_in x cols d_out -> out (d_out,)."""
    d_in = len(x)
    d_out = len(W[0])
    out = [0.0] * d_out
    for j in range(d_out):
        s = 0.0
        for i in range(d_in):
            s += x[i] * W[i][j]
        out[j] = s
    return out


def recurrent_step_ref(s_prev: list, k: list, v: list, r: list,
                       alpha: list, u: list):
    d = len(s_prev)
    kv = [k[i] * v[i] for i in range(d)]
    s_new = [alpha[i] * s_prev[i] + kv[i] for i in range(d)]
    out = [sigmoid(r[i]) * (s_new[i] + u[i] * kv[i]) for i in range(d)]
    return s_new, out


def recurrent_forward_ref(X: list, Wr: list, Wk: list, Wv: list, Wo: list,
                          w: list, u: list, s0=None):
    T = len(X)
    d = len(w)
    alpha = [math.exp(-softplus(wi)) for wi in w]
    s = list(s0) if s0 else [0.0] * d
    Y = []
    for t in range(T):
        r = matvec_ref(X[t], Wr)
        k = matvec_ref(X[t], Wk)
        v = matvec_ref(X[t], Wv)
        s, y = recurrent_step_ref(s, k, v, r, alpha, u)
        Y.append(matvec_ref(y, Wo))
    return Y, s


def cross_entropy_nats_ref(logits: list, target: int) -> float:
    m = max(logits)
    logsum = m + math.log(sum(math.exp(l - m) for l in logits))
    return logsum - logits[target]


def ternary_quantize_ref(w: float, thresh: float) -> int:
    if w > thresh:
        return 1
    if w < -thresh:
        return -1
    return 0
