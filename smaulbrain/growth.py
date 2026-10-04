"""Expert growth: controlled recombination of useful existing experts.

A new expert is never pure noise (unless the pool is empty, the documented
fallback): its weights are a convex combination of parent experts selected
for high contribution/usage, plus small seeded perturbation for symmetry
breaking. Growth records parents, initializes the router row from the
parent-mean row, initializes optimizer state, and is reproducible from the
caller-supplied seed. Results are checkpoint-safe by construction (new
expert file + router row + metadata).
"""

from __future__ import annotations

import torch

from .experts import ExpertPool, init_expert_optim_state, make_expert
from .precision import dequantize_fp8_blockwise


def select_parents(pool: ExpertPool, k: int = 2) -> list[str]:
    """Pick up to k parents by contribution (ties -> usage, then id order).

    Deterministic: sorting is fully specified, no sampling here (sampling, if
    wanted, happens in the caller with an explicit seed).
    """
    ranked = sorted(
        pool.experts.values(),
        key=lambda r: (-r.contribution, -r.tokens_routed, r.expert_id),
    )
    return [r.expert_id for r in ranked[: max(1, k)]]


def recombine_weights(
    pool: ExpertPool,
    parents: list[str],
    weights: list[float] | None = None,
    noise_std: float = 0.005,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Convex combination of dequantized parent weights + seeded noise."""
    assert parents, "need at least one parent (empty pool uses make_expert fallback)"
    if weights is None:
        weights = [1.0 / len(parents)] * len(parents)
    assert abs(sum(weights) - 1.0) < 1e-6 and all(w >= 0 for w in weights)
    out: dict[str, torch.Tensor] | None = None
    for eid, w in zip(parents, weights):
        dec = {n: dequantize_fp8_blockwise(pool.experts[eid].weights_fp8[n], torch.float32)
               for n in ("w_gate", "w_up", "w_down")}
        if out is None:
            out = {n: dec[n] * w for n in dec}
        else:
            for n in out:
                out[n] += dec[n] * w
    assert out is not None
    if noise_std > 0:
        for n in out:
            noise = torch.empty_like(out[n])
            if generator is None:
                torch.nn.init.normal_(noise, std=noise_std)
            else:
                noise.normal_(0.0, noise_std, generator=generator)
            out[n] += noise
    return out


def grow_expert(
    pool: ExpertPool,
    router,  # SparseRouter (duck-typed)
    d_model: int,
    expert_hidden: int,
    step: int,
    seed: int = 0,
    n_parents: int = 2,
    noise_std: float = 0.005,
    fp8_tile: int = 64,
) -> str:
    """Add one expert by recombination. Returns the new stable expert id.

    Reproducible: all randomness derives from ``seed``.
    Fallback: empty pool -> fresh random expert (documented, tested).
    """
    gen = torch.Generator().manual_seed(seed)
    if len(pool) == 0:
        rec = make_expert(pool.fresh_id(), d_model, expert_hidden, birth_step=step,
                          source="init", fp8_tile=fp8_tile, generator=gen)
        pool.add(rec)
        router.add_expert_row()
        return rec.expert_id
    parents = select_parents(pool, k=min(n_parents, len(pool)))
    # Parent-mean router row gives the child a sane starting admission.
    with torch.no_grad():
        rows = torch.stack([router.proj.weight[pool.index_of(p)].float() for p in parents])
        init_row = rows.mean(dim=0)
    w = recombine_weights(pool, parents, noise_std=noise_std, generator=gen)
    rec = make_expert(pool.fresh_id(), d_model, expert_hidden, weights=w, birth_step=step,
                      parents=parents, source="recombine", fp8_tile=fp8_tile)
    rec.optim_state = init_expert_optim_state(rec)
    pool.add(rec)
    router.add_expert_row(init=init_row)
    return rec.expert_id
