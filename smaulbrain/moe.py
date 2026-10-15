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

        # Partition finds the top-k threshold in linear time. Resolve boundary
        # ties over the full row with cumulative masks, then sort only the k
        # selected candidates. np.nonzero returns expert IDs in ascending order,
        # so a stable score sort preserves the expert-ID tie break without any
        # per-token Python loop.
        part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        cutoff = part_scores.min(axis=1, keepdims=True)
        above = scores > cutoff
        equal = scores == cutoff
        need_equal = top_k - above.sum(axis=1)
        equal_rank = np.cumsum(equal, axis=1)
        selected = above | (equal & (equal_rank <= need_equal[:, None]))
        rows, cols = np.nonzero(selected)
        selected_idx = cols.reshape(T, top_k)
        selected_scores = scores[rows, cols].reshape(T, top_k)
        local_order = np.argsort(-selected_scores, axis=1, kind="stable")
        top_idx = np.take_along_axis(selected_idx, local_order, axis=1)
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        # softmax over selected
        m = top_scores.max(axis=1, keepdims=True)
        ex = np.exp(top_scores - m)
        weights = ex / ex.sum(axis=1, keepdims=True)
        cap = int(np.ceil(capacity_factor * T * top_k / E))
        cap = max(cap, 1)
        # Keep the first 'cap' assignments per expert in flattened
        # token/rank order. A single stable grouping sort yields each route's
        # within-expert rank without a Python loop over tokens or experts.
        flat_idx = top_idx.reshape(-1)
        assigned = np.bincount(flat_idx, minlength=E)
        load = np.minimum(assigned, cap).astype(np.int64, copy=False)
        keep_flat = np.empty(flat_idx.size, dtype=bool)
        if flat_idx.size:
            grouped_positions = np.argsort(flat_idx, kind="stable")
            grouped_eids = flat_idx[grouped_positions]
            group_starts = np.empty(grouped_eids.size, dtype=bool)
            group_starts[0] = True
            group_starts[1:] = grouped_eids[1:] != grouped_eids[:-1]
            group_base = np.maximum.accumulate(
                np.where(group_starts, np.arange(grouped_eids.size), 0))
            rank = np.arange(grouped_eids.size) - group_base
            keep_flat[grouped_positions] = rank < cap
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
