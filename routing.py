"""Sparse token routing: top-k MoE router with load balancing.

Each token scores every expert (``logits = x @ W_router``), keeps the top-k,
renormalizes, and dispatches only to those experts. Tokens beyond per-expert
capacity (``capacity_factor``) are dropped to the residual path and counted.

Routing statistics (usage shares, admission counts) are kept in FP64 counters
(spec: routing statistics stay high precision); they are statistics, not
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutePlan:
    """Per-token dispatch decisions for one MoE application."""

    top_ids: torch.Tensor  # [N, K] expert ids per token
    top_weights: torch.Tensor  # [N, K] renormalized weights, rows sum to 1
    dropped: torch.Tensor  # [N] bool: token dropped by capacity
    probs: torch.Tensor  # [N, E] full softmax probs (for balance loss)


class SparseRouter(nn.Module):
    """Top-k router: Linear(d -> n_experts) + softmax + top-k + capacity."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.proj = nn.Linear(d_model, num_experts)
        # High-precision routing statistics (buffers, not parameters).
        self.register_buffer("usage_counts", torch.zeros(num_experts, dtype=torch.float64))
        self.register_buffer("admit_counts", torch.zeros(num_experts, dtype=torch.float64))

    def route(self, x: torch.Tensor) -> RoutePlan:
        """Route N tokens to top-k experts. x: [N, D] (any float dtype)."""
        logits = self.proj(x.to(self.proj.weight.dtype)).float()  # [N, E]
        probs = F.softmax(logits, dim=-1)
        top_w, top_ids = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Capacity: max tokens per expert; overflow tokens are dropped.
        n_tokens = x.shape[0]
        cap = max(1, int(self.capacity_factor * n_tokens * self.top_k / self.num_experts))
        order = torch.argsort(top_w.max(dim=-1).values, descending=True)
        assigned = torch.zeros(self.num_experts, dtype=torch.long)
        dropped = torch.zeros(n_tokens, dtype=torch.bool)
        for idx in order.tolist():
            chosen = False
            for slot in range(self.top_k):
                e = int(top_ids[idx, slot].item())
                if assigned[e] < cap:
                    assigned[e] += 1
                    chosen = True
                    break
            if not chosen:
                dropped[idx] = True
        # Usage statistics (no grad).
        with torch.no_grad():
            kept = top_ids[~dropped].reshape(-1)
            if kept.numel():
                counts = torch.bincount(kept, minlength=self.num_experts)
                self.usage_counts += counts.to(torch.float64)
                self.admit_counts += counts.to(torch.float64)
        return RoutePlan(top_ids=top_ids, top_weights=top_w, dropped=dropped, probs=probs)

    def balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Switch-style auxiliary loss: E * sum_e (mean_prob_e * frac_e)."""
        top_ids = probs.argmax(dim=-1)
        onehot = F.one_hot(top_ids, num_classes=self.num_experts).float()
        frac = onehot.mean(dim=0)
        mean_prob = probs.mean(dim=0)
        return (self.num_experts * (frac * mean_prob).sum()).to(probs.dtype)

    def usage_share(self) -> torch.Tensor:
        total = self.usage_counts.sum().clamp_min(1.0)
        return (self.usage_counts / total).float()

    def reset_stats(self) -> None:
        self.usage_counts.zero_()
        self.admit_counts.zero_()

    # -- dynamic topology: router rows follow expert ids --
    def add_expert_row(self, init: torch.Tensor | None = None) -> int:
        """Append one router row; returns the new expert index."""
        new_index = self.num_experts
        new = nn.Linear(self.d_model, self.num_experts + 1).to(self.proj.weight.dtype)
        with torch.no_grad():
            new.weight[: self.num_experts] = self.proj.weight
            new.bias[: self.num_experts] = self.proj.bias
            if init is not None and init.numel() >= self.d_model:
                new.weight[new_index] = init[: self.d_model].float()
                new.bias[new_index] = 0.0
            else:
                nn.init.normal_(new.weight[new_index], std=0.02)
                new.bias[new_index] = -1.0  # newborn experts admit little traffic
        self.proj = new
        self.num_experts += 1
        self.register_buffer("usage_counts",
                             torch.cat([self.usage_counts, torch.zeros(1, dtype=torch.float64)]))
        self.register_buffer("admit_counts",
                             torch.cat([self.admit_counts, torch.zeros(1, dtype=torch.float64)]))
        return new_index

    def remove_expert_row(self, index: int) -> None:
        """Delete router row ``index`` (called after expert pruning)."""
        assert 0 <= index < self.num_experts and self.num_experts > 1
        keep = [i for i in range(self.num_experts) if i != index]
        new = nn.Linear(self.d_model, self.num_experts - 1).to(self.proj.weight.dtype)
        with torch.no_grad():
            new.weight[:] = self.proj.weight[keep]
            new.bias[:] = self.proj.bias[keep]
        self.proj = new
        self.num_experts -= 1
        mask = torch.ones(len(self.usage_counts), dtype=torch.bool)
        mask[index] = False
        self.register_buffer("usage_counts", self.usage_counts[mask].contiguous())
        self.register_buffer("admit_counts", self.admit_counts[mask].contiguous())
