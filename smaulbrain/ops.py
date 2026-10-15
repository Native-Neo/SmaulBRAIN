"""Vectorized numpy operators with manual backward passes (fp32 accumulation)."""
import numpy as np

from . import math_ref as _ref

EPS = 1e-6
MAX_TMP_ELEMENTS = 1 << 26  # 64M fp32 ~= 256MB bound for single temporaries


def _check_tmp(n: int, what: str) -> None:
    if n > MAX_TMP_ELEMENTS:
        raise MemoryError(f"temporary buffer bound exceeded for {what}: {n} elements")


def rmsnorm_fwd(x: np.ndarray, g: np.ndarray, eps: float = EPS):
    # Activations and scale parameters are already fp32. Keep the large
    # elementwise temporaries in fp32; reductions can choose their accumulator
    # dtype without first widening the whole tensor.
    x32 = np.asarray(x, dtype=np.float32)
    g32 = np.asarray(g, dtype=np.float32)
    ms = np.mean(x32 * x32, axis=-1, keepdims=True, dtype=np.float32)
    inv = 1.0 / np.sqrt(ms + np.float32(eps))
    xn = x32 * inv
    y = xn * g32
    cached_ms = np.float32(ms.reshape(-1)[0]) if ms.size == 1 else ms
    cache = (x32, g32, xn, cached_ms)
    return y.astype(np.float32, copy=False), cache


def rmsnorm_bwd(dy: np.ndarray, cache, eps: float = EPS):
    x, g, xn, cached_ms = cache
    dy32 = np.asarray(dy, dtype=np.float32)
    n = x.shape[-1]
    # Reuse the forward statistics instead of recomputing RMS from a widened
    # copy of x. Only row reductions use fp64 accumulation for stability.
    ms = np.asarray(cached_ms, dtype=np.float32)
    denom = ms + np.float32(eps)
    inv = 1.0 / np.sqrt(denom)
    d_xn = dy32 * g
    dot = np.sum(d_xn * x, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    dx = (d_xn - x * (dot / (n * denom))) * inv
    axes = tuple(range(dy32.ndim - 1))
    dg = np.sum(dy32 * xn, axis=axes, dtype=np.float64).astype(np.float32)
    return dx.astype(np.float32, copy=False), dg


def sigmoid(x):
    # Evaluate only the numerically stable branch for each element; np.where
    # evaluates both expressions and used to compute several full-array exp's.
    xc = np.clip(x, -60, 60)
    out = np.empty_like(xc, dtype=np.result_type(xc.dtype, np.float32))
    positive = xc >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-xc[positive]))
    e = np.exp(xc[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def silu(x):
    return x * sigmoid(x)


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class RecurrentAttention:
    """RWKV-like recurrent attention with O(d) state per layer."""

    def __init__(self, d: int, rng: np.random.Generator):
        s = 1.0 / np.sqrt(d)
        self.Wr = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.Wk = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.Wv = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.Wo = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.w = (rng.standard_normal(d) * 0.1).astype(np.float32)
        self.u = np.zeros(d, dtype=np.float32)

    def params(self):
        return {"Wr": self.Wr, "Wk": self.Wk, "Wv": self.Wv,
                "Wo": self.Wo, "w": self.w, "u": self.u}

    def forward(self, X: np.ndarray, s0=None):
        T, d = X.shape
        _check_tmp(T * d * 8, "recurrent attention")
        alpha = np.exp(-softplus(self.w.astype(np.float64))).astype(np.float32)
        s = np.zeros(d, dtype=np.float32) if s0 is None else s0.astype(np.float32).copy()
        R = X @ self.Wr
        K = X @ self.Wk
        V = X @ self.Wv
        KV = K * V
        S = np.empty((T, d), dtype=np.float32)
        for t in range(T):
            s = alpha * s + KV[t]
            S[t] = s
        G = sigmoid(R)
        Y = G * (S + self.u * KV)
        O = Y @ self.Wo
        cache = (X, R, K, V, KV, S, G, alpha)
        return O, (s.copy(), cache)

    def backward(self, dO: np.ndarray, cache):
        X, R, K, V, KV, S, G, alpha = cache
        T, d = X.shape
        dY = dO @ self.Wo.T
        Ypre = sigmoid(R) * (S + self.u * KV)
        dWo = Y_outer(dO, Ypre)
        sig = sigmoid(R)
        dS_total = dY * sig
        dG = dY * (S + self.u * KV) * sig * (1 - sig)
        # Gradients through time for s_t = alpha*s_{t-1} + kv_t
        dKV_from_state = np.empty_like(KV)
        ds = np.zeros(d, dtype=np.float64)
        for t in range(T - 1, -1, -1):
            ds = ds * alpha.astype(np.float64) + dS_total[t].astype(np.float64)
            dKV_from_state[t] = ds.astype(np.float32)
        du = np.sum(dY * sig * KV, axis=0)
        dKV = dKV_from_state + (dY * sig * self.u)
        dK = dKV * V
        dV = dKV * K
        dR = dG
        dX = dR @ self.Wr.T + dK @ self.Wk.T + dV @ self.Wv.T
        dWr = X.T @ dR
        dWk = X.T @ dK
        dWv = X.T @ dV
        # dw via alpha: dalpha chain through all steps
        # ds_t/dalpha = s_{t-1} + alpha*ds_{t-1}/dalpha ; accumulate
        dalpha = np.zeros(d, dtype=np.float64)
        dsdalpha = np.zeros(d, dtype=np.float64)
        Aw = alpha.astype(np.float64)
        for t in range(T):
            s_prev = S[t - 1].astype(np.float64) if t > 0 else np.zeros(d)
            dsdalpha = s_prev + Aw * dsdalpha
            dalpha += dS_total[t].astype(np.float64) * dsdalpha
        # alpha = exp(-softplus(w)); dalpha/dw = -alpha*sigmoid(w)
        sigw = 1.0 / (1.0 + np.exp(-np.clip(self.w.astype(np.float64), -60, 60)))
        dw = (dalpha * (-Aw * sigw)).astype(np.float32)
        grads = {"Wr": dWr.astype(np.float32), "Wk": dWk.astype(np.float32),
                 "Wv": dWv.astype(np.float32), "Wo": dWo.astype(np.float32),
                 "w": dw, "u": du.astype(np.float32)}
        return dX.astype(np.float32), grads


def Y_outer(A: np.ndarray, B: np.ndarray):
    return A.T @ B


class GatedFFN:
    """Dense channel mixing: silu(x@Wa)* (x@Wb) @ Wc."""

    def __init__(self, d: int, h: int, rng: np.random.Generator):
        s = 1.0 / np.sqrt(d)
        self.Wa = (rng.standard_normal((d, h)) * s).astype(np.float32)
        self.Wb = (rng.standard_normal((d, h)) * s).astype(np.float32)
        self.Wc = (rng.standard_normal((h, d)) * (1.0 / np.sqrt(h))).astype(np.float32)

    def params(self):
        return {"Wa": self.Wa, "Wb": self.Wb, "Wc": self.Wc}

    def forward(self, X):
        A = X @ self.Wa
        B = X @ self.Wb
        H = silu(A) * B
        Y = H @ self.Wc
        return Y, (X, A, B, H)

    def backward(self, dY, cache):
        X, A, B, H = cache
        dH = dY @ self.Wc.T
        sigA = sigmoid(A)
        dA = dH * B * (sigA * (1 + A * (1 - sigA)))
        dB = dH * silu(A)
        dX = dA @ self.Wa.T + dB @ self.Wb.T
        return dX.astype(np.float32), {
            "Wa": (X.T @ dA).astype(np.float32),
            "Wb": (X.T @ dB).astype(np.float32),
            "Wc": (H.T @ dY).astype(np.float32)}


class ExpertMLP:
    """MoE expert: gelu(x@Win)@Wout. Exactly 2*d*h params."""

    def __init__(self, d: int, h: int, rng: np.random.Generator):
        s = 1.0 / np.sqrt(d)
        self.Win = (rng.standard_normal((d, h)) * s).astype(np.float32)
        self.Wout = (rng.standard_normal((h, d)) * (1.0 / np.sqrt(h))).astype(np.float32)

    def params(self):
        return {"Win": self.Win, "Wout": self.Wout}

    def forward(self, X):
        H = gelu_tanh(X @ self.Win)
        return H @ self.Wout, (X, H)

    def backward(self, dY, cache):
        X, H = cache
        dH = dY @ self.Wout.T
        Z = X @ self.Win
        t = np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * Z ** 3))
        dg = 0.5 * (1 + t) + 0.5 * Z * (1 - t ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * Z ** 2)
        dZ = dH * dg
        return (dZ @ self.Win.T).astype(np.float32), {
            "Win": (X.T @ dZ).astype(np.float32),
            "Wout": (H.T @ dY).astype(np.float32)}
