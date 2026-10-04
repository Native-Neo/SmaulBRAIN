"""Expert pruning with grace periods and hysteresis.

An expert is pruned only when it is simultaneously:
  * old enough (age >= survival_steps — grace period against churn),
  * unused recently (steps since last use >= survival_steps),
  * low-signal on every channel (usage share, gradient activity,
    contribution all below thresholds — hysteresis, not a single counter),
  * not load-bearing for capacity (pool stays >= min_experts).

Pruning removes weights, optimizer state, router row, and metadata together;
the pool order compacts so checkpoint loading stays index-consistent.
Redundancy (near-duplicate of a sibling) is an additional trigger, measured
by cosine similarity of dequantized gate weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .precision import dequantize_fp8_blockwise


def dying_score(
    tokens_routed: int,
    total_tokens: int,
    grad_activity: float,
    contribution: float,
    usage_threshold: float = 1e-4,
    grad_threshold: float = 1e-7,
    contrib_threshold: float = 1e-6,
) -> bool:
    """True when ALL vitality signals are below threshold (hysteresis AND)."""
    share = (tokens_routed / max(1, total_tokens))
    return (
        share < usage_threshold
        and grad_activity < grad_threshold
        and contribution < contrib_threshold
    )


def find_victims(
    pool,  # ExpertPool (duck-typed)
    step: int,
    survival_steps: int = 500,
    min_experts: int = 1,
    usage_threshold: float = 1e-4,
    redundancy_cos: float = 0.995,
) -> list[str]:
    """Rank prune candidates (stable id order). Never returns > len-min_experts."""
    total = sum(r.tokens_routed for r in pool.experts.values())
    victims: list[str] = []
    for eid in sorted(pool.experts):
        rec = pool.experts[eid]
        age = step - rec.birth_step
        idle = step - rec.last_used_step
        if age < survival_steps or idle < survival_steps:
            continue  # grace period
        if dying_score(rec.tokens_routed, total, rec.grad_activity,
                       rec.contribution, usage_threshold):
            victims.append(eid)
    # Redundancy pass: near-duplicates of a lower-id sibling.
    if redundancy_cos < 1.0:
        ids = sorted(pool.experts)
        seen = set(victims)
        for i, a in enumerate(ids):
            if a in seen:
                continue
            ga = dequantize_fp8_blockwise(pool.experts[a].weights_fp8["w_gate"]).float().flatten()
            for b in ids[:i]:
                if b in seen:
                    continue
                gb = dequantize_fp8_blockwise(pool.experts[b].weights_fp8["w_gate"]).float().flatten()
                cos = F.cosine_similarity(ga, gb, dim=0).item()
                if cos > redundancy_cos and (step - pool.experts[a].birth_step) >= survival_steps:
                    victims.append(a)
                    seen.add(a)
                    break
    victims.sort()
    keep = max(0, len(pool) - min_experts)
    return victims[:keep]


def prune_experts(
    pool,  # ExpertPool
    router,  # SparseRouter
    victims: list[str],
) -> list[str]:
    """Remove victims: weights + optimizer state + router row + metadata.

    Router rows are removed highest-index-first so surviving indices stay
    valid during the sweep. Returns pruned ids in pool-order.
    """
    ordered = sorted(victims, key=lambda eid: pool.index_of(eid), reverse=True)
    pruned: list[str] = []
    for eid in ordered:
        idx = pool.index_of(eid)
        rec = pool.remove(eid)  # drops weights + optim_state + metadata
        assert rec.expert_id == eid
        router.remove_expert_row(idx)
        pruned.append(eid)
    return sorted(pruned, key=lambda eid: eid)
