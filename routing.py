"""Sparse token routing: top-k MoE router with load balancing.

Each token scores every expert (``logits = x @ W_router``), keeps the top-k,
renormalizes, and dispatches only to those experts. Tokens beyond per-expert
capacity (``capacity_factor``) are dropped to the residual path and counted.

Routing statistics (usage shares, admission counts) are kept in FP64 counters
(spec: routing statistics stay high precision); they are statistics, not
parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutePlan:
    """Per-token dispatch decisions for one MoE application."""

    top_ids: torch.Tensor  # [N, K] expert ids per token
    top_weights: torch.Tensor  # [N, K] admitted weights (live rows sum to 1,
                               # dropped rows sum to 0; non-admitted slots are 0)
    dropped: torch.Tensor  # [N] bool: token dropped by capacity
    probs: torch.Tensor  # [N, E] full softmax probs (for balance loss)

    def validate(self, num_experts: int, top_k: int) -> None:
        """Structural contract: shapes, id range, finiteness, device unity."""
        n = self.top_ids.shape[0]
        assert self.top_ids.shape == (n, top_k), "top_ids shape"
        assert self.top_weights.shape == (n, top_k), "top_weights shape"
        assert self.dropped.shape == (n,), "dropped shape"
        assert self.probs.shape == (n, num_experts), "probs shape"
        assert self.top_ids.dtype == torch.long, "top_ids dtype"
        assert self.dropped.dtype == torch.bool, "dropped dtype"
        dev = self.top_ids.device
        for name in ("top_weights", "dropped", "probs"):
            assert getattr(self, name).device == dev, f"{name} device"
        if n == 0:
            return
        assert bool(((self.top_ids >= 0) & (self.top_ids < num_experts)).all()), \
            "expert id out of range"
        assert bool(torch.isfinite(self.top_weights).all()), "nonfinite weight"
        assert bool(torch.isfinite(self.probs).all()), "nonfinite prob"
        for row in self.top_ids.tolist():
            assert len(set(row)) == len(row), "duplicate top-k expert id"


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
        # Zero-expert pools are transient (growth starts from empty): plain
        # nn.Linear(., 0) construction warns on zero-element init, so build
        # it quietly. Non-empty pools keep the standard noisy-on-failure init.
        if num_experts == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.proj = nn.Linear(d_model, num_experts)
        else:
            self.proj = nn.Linear(d_model, num_experts)
        # Runtime-only new-expert steering (never checkpointed: plain Python
        # attributes, not parameters/buffers, so state_dict is untouched).
        # At rest both are cleared (0.0/()) so inference (forward_infer,
        # generate, evaluate_loss -- all dispatch through route()) is NEVER
        # implicitly steered. train_step sets them immediately around the
        # mode-`new` training forward only (try/finally, cleared right
        # after, including the exception path) and forces them cleared for
        # every other mode, so training never depends on stale values. To
        # explicitly steer inference, set both attrs manually before
        # calling generate/forward_infer.
        # Honored by route() only (training forward, forward_infer, and
        # streaming inference all dispatch through route()).
        self.new_routing_bias: float = 0.0
        self.new_expert_idx: tuple = ()
        # High-precision routing statistics (buffers, not parameters).
        self.register_buffer("usage_counts", torch.zeros(num_experts, dtype=torch.float64))
        self.register_buffer("admit_counts", torch.zeros(num_experts, dtype=torch.float64))
        # Tokens sanitized for nonfinite router input (observable, not silent).
        self.register_buffer("sanitized_counts", torch.zeros((), dtype=torch.float64))

    def route(self, x: torch.Tensor, enforce_capacity: bool = True) -> RoutePlan:
        """Route N tokens to top-k experts. x: [N, D] (any float dtype).

        Empty batches return a well-formed empty plan. Nonfinite logits are
        sanitized (NaN->0, +/-Inf->-/+1e4) and counted so dispatch artifacts
        (ids, weights, statistics, balance loss) stay finite; upstream NaNs
        still propagate through expert compute into the loss.

        Capacity is a training throughput guard and is batch-size relative,
        so inference must not enforce it (``enforce_capacity=False`` admits
        every slot): otherwise chunked/streaming inference would admit
        different slots than a full pass over the same tokens.
        """
        n_tokens = x.shape[0]
        if n_tokens == 0:
            dev = x.device
            empty_ids = torch.zeros((0, self.top_k), dtype=torch.long, device=dev)
            empty_w = torch.zeros((0, self.top_k), device=dev)
            plan = RoutePlan(top_ids=empty_ids, top_weights=empty_w,
                             dropped=torch.zeros(0, dtype=torch.bool, device=dev),
                             probs=torch.zeros((0, self.num_experts), device=dev))
            plan.validate(self.num_experts, self.top_k)
            return plan
        logits = self.proj(x.to(self.proj.weight.dtype)).float()  # [N, E]
        bad = ~torch.isfinite(logits).all(dim=-1)
        if bool(bad.any()):
            with torch.no_grad():
                self.sanitized_counts += float(bad.sum().item())
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        # New-expert steering: additive logit bonus on the runtime-selected
        # rows only. Skipped entirely when the bias is 0.0 (the default), so
        # the off path is bit-identical to unsteered routing. Only routing
        # weights/choice change; expert and trunk weights are untouched.
        _bias = float(getattr(self, "new_routing_bias", 0.0) or 0.0)
        if _bias != 0.0:
            _idx = [int(i) for i in (getattr(self, "new_expert_idx", ()) or ())
                    if 0 <= int(i) < self.num_experts]
            if _idx:
                logits[:, _idx] = logits[:, _idx] + _bias
        probs = F.softmax(logits, dim=-1)
        top_w, top_ids = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        n = top_ids.shape[0]
        if not enforce_capacity:
            admit = torch.ones_like(top_ids, dtype=torch.bool)
            dropped = torch.zeros(n, dtype=torch.bool, device=top_ids.device)
        else:
            admit, dropped = self._admit_slots(top_ids, top_w, n)
            top_w = torch.where(admit, top_w, torch.zeros_like(top_w))
            top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Usage statistics count admitted slots only (no grad).
        with torch.no_grad():
            kept = top_ids[admit]
            if kept.numel():
                counts = torch.bincount(kept, minlength=self.num_experts)
                self.usage_counts += counts.to(torch.float64)
                self.admit_counts += counts.to(torch.float64)
        plan = RoutePlan(top_ids=top_ids, top_weights=top_w, dropped=dropped, probs=probs)
        plan.validate(self.num_experts, self.top_k)
        return plan

    def _admit_slots(self, top_ids: torch.Tensor, top_w: torch.Tensor,
                     n_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-slot capacity admission. See route() for the contract.

        A live token dispatches only to the subset of its top-k that fits,
        with weights renormalized over the admitted subset. Non-admitted
        slots read weight 0 so downstream dispatch (which multiplies
        mask x weight) cannot serve what capacity refused. A token with no
        admitted slot is dropped to the residual path.

        Vectorized greedy: tokens are stably priority-ordered (same key as
        the old sequential loop, so results are identical), then each
        expert's queue position is a cumsum rank over the flattened
        (token-major, slot-minor) order — exactly the order the sequential
        loop visited. Rank < cap admits. No Python token loop, no host
        transfers on the hot path.
        """
        cap = max(1, int(self.capacity_factor * n_tokens * self.top_k / self.num_experts))
        # Stable priority: equal-confidence tokens keep input order, so the
        # same batch always admits the same slots (deterministic routing).
        order = torch.argsort(top_w.max(dim=-1).values, descending=True, stable=True)
        sorted_ids = top_ids[order]  # [N, K] priority order
        flat = sorted_ids.reshape(-1)  # token-major, slot-minor: loop order
        admit_flat = torch.zeros(flat.shape[0], dtype=torch.bool, device=top_ids.device)
        for e in range(self.num_experts):
            is_e = flat == e
            # Queue rank among this expert's slots in visit order; slots
            # beyond capacity are refused. (All-False columns yield rank -1
            # but stay masked out by is_e, so no empty-check sync is needed.)
            rank = is_e.to(torch.int64).cumsum(0) - 1
            admit_flat |= is_e & (rank < cap)
        admit_sorted = admit_flat.reshape(sorted_ids.shape)
        admit = torch.empty_like(admit_sorted)
        admit[order] = admit_sorted
        dropped = ~admit.any(dim=-1)
        return admit, dropped

    def balance_loss(self, probs: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Switch-style auxiliary loss: E * sum_e (mean_prob_e * frac_e).

        Empty routing yields exactly 0 (never NaN): no token reached an
        expert, so there is nothing to balance. ``keep`` (bool [N]) restricts
        both means to scored positions so padding cannot dilute the balance
        signal; all-excluded also yields 0.
        """
        if probs.shape[0] == 0:
            return probs.new_zeros(())
        rows = probs if keep is None else probs[keep]
        if rows.shape[0] == 0:
            return probs.new_zeros(())
        top_ids = rows.argmax(dim=-1)
        onehot = F.one_hot(top_ids, num_classes=self.num_experts).float()
        frac = onehot.mean(dim=0)
        mean_prob = rows.mean(dim=0)
        return (self.num_experts * (frac * mean_prob).sum()).to(probs.dtype)

    def usage_share(self) -> torch.Tensor:
        total = self.usage_counts.sum().clamp_min(1.0)
        return (self.usage_counts / total).float()

    def reset_stats(self) -> None:
        self.usage_counts.zero_()
        self.admit_counts.zero_()
        self.sanitized_counts.zero_()

    # -- dynamic topology: router rows follow expert ids --
    def _proj_like(self, out_features: int) -> nn.Linear:
        """Fresh Linear on the router's device/dtype (topology must follow)."""
        if out_features == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return nn.Linear(self.d_model, out_features).to(
                    device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        return nn.Linear(self.d_model, out_features).to(
            device=self.proj.weight.device, dtype=self.proj.weight.dtype)

    def add_expert_row(self, init: torch.Tensor | None = None) -> int:
        """Append one router row; returns the new expert index."""
        new_index = self.num_experts
        new = self._proj_like(self.num_experts + 1)
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
        dev = self.usage_counts.device
        self.register_buffer("usage_counts",
                             torch.cat([self.usage_counts,
                                        torch.zeros(1, dtype=torch.float64, device=dev)]))
        self.register_buffer("admit_counts",
                             torch.cat([self.admit_counts,
                                        torch.zeros(1, dtype=torch.float64, device=dev)]))
        return new_index

    def remove_expert_row(self, index: int) -> None:
        """Delete router row ``index`` (called after expert pruning)."""
        assert 0 <= index < self.num_experts and self.num_experts > 1
        keep = [i for i in range(self.num_experts) if i != index]
        new = self._proj_like(self.num_experts - 1)
        with torch.no_grad():
            new.weight[:] = self.proj.weight[keep]
            new.bias[:] = self.proj.bias[keep]
        self.proj = new
        self.num_experts -= 1
        mask = torch.ones(len(self.usage_counts), dtype=torch.bool, device=self.usage_counts.device)
        mask[index] = False
        self.register_buffer("usage_counts", self.usage_counts[mask].contiguous())
        self.register_buffer("admit_counts", self.admit_counts[mask].contiguous())
