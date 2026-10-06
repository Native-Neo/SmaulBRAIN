"""Expert pruning with grace periods and hysteresis.

An expert is pruned only when it is simultaneously:
  * old enough (age >= survival_steps — grace period against churn),
  * unused recently (steps since last use >= survival_steps),
  * low-signal on every channel (usage share, gradient activity,
    contribution all below thresholds — hysteresis, not a single counter),
  * not load-bearing for capacity (pool stays >= min_experts).

Only the worst ``max_victims`` go per evaluation (worst-first ranking), so
one bad cycle can never wipe out the pool. Pruning removes weights,
optimizer state, router row, and metadata together; the pool order compacts
so checkpoint loading stays index-consistent.
Redundancy (near-duplicate of a sibling) is an additional trigger, measured
by cosine similarity of dequantized gate weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from precision import dequantize_fp8_blockwise


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
    redundancy_usage_mult: float = 10.0,
    max_victims: int | None = 4,
) -> list[str]:
    """Rank prune candidates worst-first. Never returns > len-min_experts.

    At most ``max_victims`` (None = uncapped) go per call so a single
    evaluation cannot collapse the pool. The redundancy pass uses a looser
    usage bar (``usage_threshold * mult``): near-duplicates that still carry
    real traffic are capacity, not waste, and must survive.
    """
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
        loose = usage_threshold * redundancy_usage_mult
        for i, a in enumerate(ids):
            if a in seen:
                continue
            rec_a = pool.experts[a]
            share_a = rec_a.tokens_routed / max(1, total)
            if share_a >= loose or (step - rec_a.birth_step) < survival_steps:
                continue  # load-bearing or young: twins are capacity, not waste
            ga = dequantize_fp8_blockwise(rec_a.weights_fp8["w_gate"]).float().flatten()
            for b in ids[:i]:
                if b in seen:
                    continue
                gb = dequantize_fp8_blockwise(pool.experts[b].weights_fp8["w_gate"]).float().flatten()
                cos = F.cosine_similarity(ga, gb, dim=0).item()
                if cos > redundancy_cos:
                    victims.append(a)
                    seen.add(a)
                    break
    def _badness(eid: str) -> tuple:
        rec = pool.experts[eid]
        return (rec.tokens_routed / max(1, total), rec.grad_activity,
                rec.contribution, eid)

    victims.sort(key=_badness)  # worst first; eid tiebreak keeps it deterministic
    keep = max(0, len(pool) - min_experts)
    victims = victims[:keep]
    if max_victims is not None and max_victims >= 0:
        victims = victims[:max_victims]
    return victims


def prune_experts(
    pool,  # ExpertPool
    router,  # SparseRouter
    victims: list[str],
    pager=None,  # ExpertPager (duck-typed): forgotten per victim, else caller must
) -> list[str]:
    """Remove victims: weights + optimizer state + router row + pager traces.

    Validate-first: every victim must exist and pool/router widths must
    match before anything is removed, so a bad victim list cannot commit a
    prefix and leave pool/router diverged. Router rows are removed
    highest-index-first so surviving indices stay valid during the sweep.
    Pager traces are forgotten inline (never served ghosts on direct calls);
    the call is idempotent, so callers may also forget defensively.
    Returns pruned ids in pool-order.
    """
    missing = [eid for eid in victims if eid not in pool.experts]
    if missing:
        raise ValueError(f"unknown victims (refusing to prune): {missing}")
    if router.num_experts != len(pool):
        raise ValueError(
            f"pool/router out of sync: {len(pool)} experts vs "
            f"{router.num_experts} router rows (refusing to mutate)"
        )
    ordered = sorted(victims, key=lambda eid: pool.index_of(eid), reverse=True)
    pruned: list[str] = []
    for eid in ordered:
        idx = pool.index_of(eid)
        rec = pool.remove(eid)  # drops weights + optim_state + metadata
        assert rec.expert_id == eid
        router.remove_expert_row(idx)
        if pager is not None:
            pager.forget(eid)
        pruned.append(eid)
    return sorted(pruned, key=lambda eid: eid)
