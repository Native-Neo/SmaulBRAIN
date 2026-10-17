"""PyTorch top-k router and mixture-of-experts diagnostics."""
import math
import torch


class Router:
    def __init__(self, d: int, n_experts: int, rng=None, device="cpu"):
        if isinstance(rng, torch.Generator):
            generator = rng
        else:
            seed = 0 if rng is None else int(rng.integers(0, 2**63 - 1)) if hasattr(rng, "integers") else int(rng)
            generator = torch.Generator(device="cpu").manual_seed(seed)
        self.W = (torch.randn(d, n_experts, generator=generator) / math.sqrt(d)).to(device)

    @property
    def n_experts(self):
        return self.W.shape[1]

    def route(self, X, top_k: int, capacity_factor: float = 1.25, disabled=()):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.W.device)
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"router inputs must have shape (tokens, {self.W.shape[0]}), got {tuple(X.shape)}"
            )
        if not torch.isfinite(X).all(): raise ValueError("non-finite router inputs")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k!r}")
        if not math.isfinite(float(capacity_factor)) or float(capacity_factor) <= 0:
            raise ValueError("capacity_factor must be finite and > 0")
        disabled = sorted(set(int(i) for i in disabled))
        enabled = self.n_experts - len(disabled)
        if top_k > enabled: raise ValueError(f"top_k={top_k} exceeds enabled experts={enabled}")
        scores = X @ self.W
        if not torch.isfinite(scores).all(): raise ValueError("non-finite router scores")
        if disabled:
            if any(i < 0 or i >= self.n_experts for i in disabled): raise ValueError("disabled expert index out of range")
            scores = scores.clone(); scores[:, disabled] = -torch.inf
        idx = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
        top_scores = scores.gather(1, idx)
        raw_weights = torch.softmax(top_scores, dim=1)
        T = X.shape[0]; cap = max(1, math.ceil(capacity_factor * T * top_k / self.n_experts))
        keep = torch.ones_like(idx, dtype=torch.bool)
        flat = idx.reshape(-1)
        flat_keep = keep.reshape(-1)
        flat_scores = top_scores.reshape(-1)
        for eid in torch.unique(flat, sorted=True).tolist():
            positions = torch.nonzero(flat == eid, as_tuple=False).flatten()
            if positions.numel() > cap:
                # Capacity is a competition among token-expert assignments:
                # keep the strongest router scores, not whichever tokens
                # happen to occur earliest in the sequence.
                assignment_scores = flat_scores[positions]
                order = torch.argsort(assignment_scores, descending=True, stable=True)
                flat_keep[positions[order[cap:]]] = False
        kept = raw_weights * keep.to(raw_weights.dtype)
        denom = kept.sum(dim=1, keepdim=True)
        weights = torch.where(denom > 0, kept / denom.clamp_min(1e-12), torch.zeros_like(kept))
        counts = torch.bincount(flat, minlength=self.n_experts)
        load = counts.clamp(max=cap)
        info = {"overflow_dropped": int(flat.numel() - load.sum().item()), "cap_per_expert": cap,
                "load": load.detach().cpu().tolist()}
        return idx, weights, info, (X, scores, idx, weights, keep)

    def backward(self, dWeights, cache):
        X, scores, idx, weights, keep = cache
        dWeights = torch.as_tensor(dWeights, dtype=weights.dtype, device=weights.device)
        row_dot = (weights * dWeights).sum(dim=1, keepdim=True)
        dtop = torch.where(keep, weights * (dWeights - row_dot), torch.zeros_like(weights))
        dScores = torch.zeros_like(scores).scatter(1, idx, dtop)
        return dScores @ self.W.T, X.T @ dScores


def softmax_rows(x):
    return torch.softmax(x, dim=1)


def balance_aux_loss(weights, top_idx, n_experts):
    if weights.shape[0] == 0: return 0.0
    counts = torch.bincount(top_idx.reshape(-1), minlength=n_experts).float()
    sums = torch.zeros(n_experts, device=weights.device).index_add_(0, top_idx.reshape(-1), weights.reshape(-1))
    return float(n_experts * ((counts / weights.shape[0]) * (sums / weights.shape[0])).sum().item())


def expert_output_stats(top_idx, weights, n_experts):
    counts = torch.bincount(top_idx.reshape(-1), minlength=n_experts)
    return {"tokens_total": int(top_idx.shape[0]), "per_expert_counts": counts.cpu().tolist(),
            "mean_load": float(counts.float().mean().item()), "min_load": int(counts.min().item()),
            "max_load": int(counts.max().item()), "unused_experts": torch.nonzero(counts == 0).flatten().cpu().tolist()}
