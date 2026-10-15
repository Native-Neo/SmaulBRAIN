"""MoE routing and expert execution (numpy).

Router: scores_t = x_t @ W_router  (d -> E). Top-k with deterministic
tie-break: sort by (-score, expert_id). Weights = softmax over selected k.
Capacity: per-expert cap = ceil(capacity_factor * T*k / E); overflow tokens
are dropped from that expert (documented policy) and counted.
Empty selection -> zero contribution. Non-finite scores -> ValueError.

Gradients flow to selected experts and to router logits (via softmax weights).
"""
import numpy as np

from . import ops as _ops


class Router:
    def __init__(self, d: int, n_experts: int, rng: np.random.Generator):
        self.W = (rng.standard_normal((d, n_experts)) * (1.0 / np.sqrt(d))).astype(np.float32)

    @property
    def n_experts(self):
        return self.W.shape[1]

    def route(self, X: np.ndarray, top_k: int, capacity_factor: float = 1.25,
                disabled=()):
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite router inputs")
        scores = X @ self.W
        if not np.all(np.isfinite(scores)):
            raise ValueError("non-finite router scores")
        if disabled:
            scores = scores.copy()
            scores[:, list(disabled)] = -np.inf
        T, E = scores.shape
        if top_k > E:
            raise ValueError(f"top_k={top_k} exceeds n_experts={E}")
        if top_k > E - len(set(disabled)):
            raise ValueError(f"top_k={top_k} exceeds enabled experts={E - len(set(disabled))}")
        # deterministic top-k: full stable argsort by (-score, id)
        order = np.argsort(-scores, axis=1, kind="stable")
        top_idx = order[:, :top_k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        # softmax over selected
        m = top_scores.max(axis=1, keepdims=True)
        ex = np.exp(top_scores - m)
        weights = ex / ex.sum(axis=1, keepdims=True)
        cap = int(np.ceil(capacity_factor * T * top_k / E))
        cap = max(cap, 1)
        # capacity enforcement in deterministic token order
        load = np.zeros(E, dtype=np.int64)
        keep = np.ones_like(weights, dtype=bool)
        overflow = 0
        for t in range(T):
            for j in range(top_k):
                e = int(top_idx[t, j])
                if load[e] >= cap:
                    keep[t, j] = False
                    overflow += 1
                else:
                    load[e] += 1
        # renormalize kept weights
        wsum = np.where(keep, weights, 0.0).sum(axis=1, keepdims=True)
        wsum = np.maximum(wsum, 1e-12)
        weights = np.where(keep, weights, 0.0) / wsum
        # rows where everything dropped -> zero row (expert pass-through = residual only)
        cache = (X, scores, top_idx, weights, keep)
        return top_idx, weights, {"overflow_dropped": overflow, "cap_per_expert": cap,
                                  "load": load}, cache

    def backward(self, dWeights: np.ndarray, cache):
        X, scores, top_idx, weights, keep = cache
        T, k = top_idx.shape
        # softmax jacobian over kept entries
        dTop = np.zeros_like(weights)
        for t in range(T):
            kept = keep[t]
            if not np.any(kept):
                continue
            w = weights[t][kept]
            dw = dWeights[t][kept]
            s = float(np.dot(w, dw))
            dTop[t][kept] = w * (dw - s)
        dScores = np.zeros_like(scores)
        np.add.at(dScores, (np.repeat(np.arange(T), k), top_idx.ravel()), dTop.ravel())
        dW = X.T @ dScores
        dX = dScores @ self.W.T
        return dX.astype(np.float32), dW.astype(np.float32)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


def balance_aux_loss(weights: np.ndarray, top_idx: np.ndarray, n_experts: int) -> float:
    T = weights.shape[0]
    frac = np.zeros(n_experts)
    gate = np.zeros(n_experts)
    for t in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            e = int(top_idx[t, j])
            frac[e] += 1.0 / T
            gate[e] += float(weights[t, j]) / T
    return float(n_experts * np.dot(frac, gate))


def expert_output_stats(top_idx: np.ndarray, weights: np.ndarray, n_experts: int) -> dict:
    T = top_idx.shape[0]
    counts = np.zeros(n_experts, dtype=np.int64)
    for e in top_idx.ravel():
        counts[int(e)] += 1
    return {
        "tokens_total": int(T),
        "per_expert_counts": counts.tolist(),
        "mean_load": float(counts.mean()),
        "min_load": int(counts.min()),
        "max_load": int(counts.max()),
        "unused_experts": [int(i) for i in np.where(counts == 0)[0].tolist()],
    }
