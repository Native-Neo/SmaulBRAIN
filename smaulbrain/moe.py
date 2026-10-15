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
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > E:
            raise ValueError(f"top_k={top_k} exceeds n_experts={E}")
        enabled_count = E - len(set(disabled))
        if top_k > enabled_count:
            raise ValueError(f"top_k={top_k} exceeds enabled experts={enabled_count}")

        # Partition finds the top-k score threshold in linear time. Sort only
        # the k selected candidates. Repair ties at the threshold using the
        # smallest expert IDs so behavior is identical to stable argsort on
        # (-score, expert_id), even when argpartition's boundary tie is arbitrary.
        part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        cutoff = part_scores.min(axis=1)
        top_idx = np.empty((T, top_k), dtype=np.int64)
        for t in range(T):
            above = np.flatnonzero(scores[t] > cutoff[t])
            equal = np.flatnonzero(scores[t] == cutoff[t])
            chosen = np.concatenate((above, equal[:top_k - above.size]))
            order = np.argsort(-scores[t, chosen], kind="stable")
            top_idx[t] = chosen[order]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        # softmax over selected
        m = top_scores.max(axis=1, keepdims=True)
        ex = np.exp(top_scores - m)
        weights = ex / ex.sum(axis=1, keepdims=True)
        cap = int(np.ceil(capacity_factor * T * top_k / E))
        cap = max(cap, 1)
        # Capacity keeps the first 'cap' occurrences of each expert in
        # flattened token/rank order, exactly matching the old nested loop.
        flat_idx = top_idx.reshape(-1)
        assigned = np.bincount(flat_idx, minlength=E).astype(np.int64, copy=False)
        load = np.minimum(assigned, cap)
        keep_flat = np.ones(flat_idx.size, dtype=bool)
        for eid in np.flatnonzero(assigned > cap):
            positions = np.flatnonzero(flat_idx == eid)
            keep_flat[positions[cap:]] = False
        keep = keep_flat.reshape(top_idx.shape)
        overflow = int(flat_idx.size - int(load.sum()))
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
        # Vectorized softmax Jacobian over the kept (already renormalized)
        # expert weights. Dropped routes have zero weight and zero gradient.
        row_dot = np.sum(weights * dWeights, axis=1, keepdims=True)
        dTop = np.where(keep, weights * (dWeights - row_dot), 0.0)
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
    if T == 0:
        return 0.0
    chosen = np.asarray(top_idx, dtype=np.int64).reshape(-1)
    routed_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    counts = np.bincount(chosen, minlength=n_experts)
    weight_sums = np.bincount(chosen, weights=routed_weights, minlength=n_experts)
    frac = counts.astype(np.float64) / T
    gate = weight_sums / T
    return float(n_experts * np.dot(frac, gate))


def expert_output_stats(top_idx: np.ndarray, weights: np.ndarray, n_experts: int) -> dict:
    T = top_idx.shape[0]
    counts = np.bincount(np.asarray(top_idx, dtype=np.int64).reshape(-1),
                         minlength=n_experts).astype(np.int64, copy=False)
    return {
        "tokens_total": int(T),
        "per_expert_counts": counts.tolist(),
        "mean_load": float(counts.mean()),
        "min_load": int(counts.min()),
        "max_load": int(counts.max()),
        "unused_experts": [int(i) for i in np.where(counts == 0)[0].tolist()],
    }
