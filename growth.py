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

from experts import ExpertPool, ExpertRecord, init_expert_optim_state, make_expert
from precision import FP8BlockTensor, dequantize_fp8_blockwise

_EXPERT_WEIGHTS = ("w_gate", "w_up", "w_down")


def _check_optim_width(optim_state: dict | None, expect: int) -> None:
    """Validate-first: every momentum buffer must already match pool width.

    Raises before mutating when a buffer's dim-0 disagrees with the pool,
    so padding cannot compound a prior mismatch into silent misalignment.
    Missing entries are left alone: step_dense initializes them at the
    post-growth width on next use.
    """
    if optim_state is None:
        return
    for st in optim_state.values():
        if not isinstance(st, dict):
            continue
        for key in ("m", "v_row", "v"):
            t = st.get(key)
            if torch.is_tensor(t) and t.shape[0] > 0 and t.shape[0] != expect:
                raise ValueError(
                    f"optim_state out of sync: buffer {key} has "
                    f"{t.shape[0]} rows vs {expect} experts (refusing to mutate)"
                )


def _snapshot_router(router) -> tuple:
    """Capture router topology for exact rollback (handles 0->1 undo)."""
    return (
        router.num_experts,
        router.proj.weight.detach().clone(),
        router.proj.bias.detach().clone(),
        router.usage_counts.detach().clone(),
        router.admit_counts.detach().clone(),
    )


def _restore_router(router, snap: tuple) -> None:
    """Restore a snapshot from _snapshot_router (no validation, rollback only)."""
    n, w, b, u, a = snap
    new = router._proj_like(n)
    with torch.no_grad():
        if n > 0:
            new.weight.copy_(w)
            new.bias.copy_(b)
    router.proj = new
    router.num_experts = n
    router.register_buffer("usage_counts", u.clone())
    router.register_buffer("admit_counts", a.clone())


def _pad_state_rows(state: dict, n_new: int) -> None:
    """Append zero rows to dim 0 of every stored momentum buffer.

    Newborn router rows start with zero momentum (same as fresh init_state);
    surviving rows keep theirs bit-identically. Missing entries are left
    alone: step_dense initializes them at the post-growth width on next use.
    Two-phase: all replacements are built before any assignment, so a
    build failure (e.g. OOM in torch.cat) leaves every buffer untouched.
    """
    # Phase 1 (pure build): compute every replacement before assigning any,
    # so a failure leaves all buffers untouched.
    pending: list[tuple] = []
    for st in state.values():
        if not isinstance(st, dict):
            continue
        for key in ("m", "v_row", "v"):
            t = st.get(key)
            if torch.is_tensor(t) and t.shape[0] > 0:
                pending.append((st, key, torch.cat([t, torch.zeros(
                    (n_new, *t.shape[1:]), dtype=t.dtype, device=t.device)])))
    # Phase 2 (commit): assignments only; infallible after phase-1 build.
    for st, key, new in pending:
        st[key] = new


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
    missing = [eid for eid in parents if eid not in pool.experts]
    if missing:
        raise ValueError(f"unknown parents (refusing to recombine): {missing}")
    if weights is None:
        weights = [1.0 / len(parents)] * len(parents)
    if len(weights) != len(parents):
        raise ValueError(
            f"{len(weights)} weights for {len(parents)} parents (zip would truncate)"
        )
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


def _check_topology_in_sync(pool: ExpertPool, router) -> None:
    """Fail before mutating when pool and router have already diverged."""
    if router.num_experts != len(pool):
        raise ValueError(
            f"pool/router out of sync: {len(pool)} experts vs "
            f"{router.num_experts} router rows (refusing to mutate)"
        )


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
    max_experts: int | None = None,
    optim_state: dict | None = None,
) -> str:
    """Add one expert by recombination. Returns the new stable expert id.

    Reproducible: all randomness derives from ``seed``.
    Fallback: empty pool -> fresh random expert (documented, tested).
    Capacity-safe: at ``max_experts`` (when given) raises before mutating.
    Two-phase: the record and router row are fully built before either the
    pool or the router is touched, so a build failure cannot leave them
    diverged.
    ``optim_state`` (e.g. SmaulOpt.router_state) gains one zero row per new
    router row so surviving rows keep their momentum instead of the whole
    table resetting on shape mismatch. Validate-first: topology, capacity,
    and optimizer widths are checked before anything mutates. Any commit
    failure (router append, momentum pad) rolls back the pool addition and
    restores the ID cursor, so the prior topology returns exactly with no
    consumed IDs.
    """
    _check_topology_in_sync(pool, router)
    if max_experts is not None and len(pool) >= max_experts:
        raise ValueError(f"at capacity: {len(pool)} >= max_experts={max_experts}")
    _check_optim_width(optim_state, len(pool))
    next_before = pool._next_id
    router_snap = _snapshot_router(router)
    gen = torch.Generator().manual_seed(seed)
    try:
        if len(pool) == 0:
            rec = make_expert(pool.fresh_id(), d_model, expert_hidden, birth_step=step,
                              source="init", fp8_tile=fp8_tile, generator=gen)
            init_row = None
        else:
            parents = select_parents(pool, k=min(n_parents, len(pool)))
            # Parent-mean router row gives the child a sane starting admission.
            with torch.no_grad():
                rows = torch.stack([router.proj.weight[pool.index_of(p)].float() for p in parents])
                init_row = rows.mean(dim=0)
            w = recombine_weights(pool, parents, noise_std=noise_std, generator=gen)
            rec = make_expert(pool.fresh_id(), d_model, expert_hidden, weights=w, birth_step=step,
                              parents=parents, source="recombine", fp8_tile=fp8_tile)
            rec.optim_state = init_expert_optim_state(rec)
    except Exception:
        pool._next_id = next_before
        raise
    pool_added = False
    router_added = False
    try:
        pool.add(rec)
        pool_added = True
        router.add_expert_row(init=init_row)
        router_added = True
        if optim_state is not None:
            _pad_state_rows(optim_state, 1)
    except Exception:
        # Exact rollback: pool entry out, router topology restored, ID
        # cursor restored. _pad is two-phase so optim_state is untouched
        # on pad-build failure; a pad that already committed cannot occur
        # because pad is the last step.
        try:
            if router_added:
                _restore_router(router, router_snap)
            if pool_added:
                try:
                    pool.remove(rec.expert_id)
                except KeyError:
                    pass
        finally:
            pool._next_id = next_before
        raise
    return rec.expert_id


def grow_topk_clones(
    pool: ExpertPool,
    router,  # SparseRouter (duck-typed)
    d_model: int,
    expert_hidden: int,
    step: int,
    seed: int = 0,
    k: int = 8,
    n_mutated: int = 2,
    noise_std: float = 0.0005,
    fp8_tile: int = 64,
    max_experts: int | None = None,
    optim_state: dict | None = None,
) -> list[str]:
    """Duplicate the top-k experts by contribution (exact copies + mutants).

    Each clone inherits its parent's router row, so traffic splits between
    twins — approximately output-preserving under top-k renormalization.
    ``n_mutated`` clones (seeded choice) receive small Gaussian perturbation
    for exploration; the rest are bit-identical consolidation with fresh
    (zero) optimizer state. Returns new ids in parent-rank order.
    Reproducible from ``seed``; birth steps grant prune grace periods.
    Capacity-safe: never grows past ``max_experts`` (when given); returns
    fewer (possibly zero) ids when room runs out.
    Two-phase: all records and router rows are built before the pool or
    the router is touched, so a build failure cannot commit a prefix and
    leave pool/router diverged. Validate-first: topology, capacity, and
    optimizer widths are checked before anything mutates. Any commit
    failure rolls back the committed prefix and restores the ID cursor,
    so the prior topology returns exactly with no consumed IDs.
    """
    _check_topology_in_sync(pool, router)
    ranked = sorted(
        pool.experts.values(),
        key=lambda r: (-r.contribution, -r.tokens_routed, r.expert_id),
    )
    want = max(1, min(k, len(ranked)))
    if max_experts is not None:
        room = max(0, max_experts - len(pool))
        want = min(want, room)
        if want <= 0:
            return []
    _check_optim_width(optim_state, len(pool))
    next_before = pool._next_id
    router_snap = _snapshot_router(router)
    parents = ranked[:want]
    gen = torch.Generator().manual_seed(seed)
    mutated = set(torch.randperm(len(parents), generator=gen)[: max(0, min(n_mutated, len(parents)))].tolist())
    # Phase 1 (pure build): records + router rows, no pool/router mutation.
    built: list[tuple] = []
    try:
        for i, par in enumerate(parents):
            eid = pool.fresh_id()
            if i in mutated and noise_std > 0:
                w = {n: dequantize_fp8_blockwise(par.weights_fp8[n], torch.float32)
                     for n in _EXPERT_WEIGHTS}
                for n in w:
                    noise = torch.empty_like(w[n])
                    noise.normal_(0.0, noise_std, generator=gen)
                    w[n] += noise
                rec = make_expert(eid, d_model, expert_hidden, weights=w, birth_step=step,
                                  parents=[par.expert_id], source="clone-mutated",
                                  fp8_tile=fp8_tile)
            else:
                rec = ExpertRecord(
                    expert_id=eid,
                    d_model=d_model,
                    expert_hidden=expert_hidden,
                    weights_fp8={n: FP8BlockTensor(
                        codes=par.weights_fp8[n].codes.clone(),
                        scales=par.weights_fp8[n].scales.clone(),
                        shape=par.weights_fp8[n].shape,
                        tile=par.weights_fp8[n].tile,
                    ) for n in _EXPERT_WEIGHTS},
                    birth_step=step,
                    parents=[par.expert_id],
                    source="clone",
                )
                rec.optim_state = init_expert_optim_state(rec)
            with torch.no_grad():
                parent_row = router.proj.weight[pool.index_of(par.expert_id)].detach().clone()
            built.append((rec, parent_row))
    except Exception:
        # Build touched only the ID cursor (fresh_id per iteration); pool,
        # router, and optimizer state are still pristine.
        pool._next_id = next_before
        raise
    # Phase 2 (commit): pool and router move in lockstep; any failure
    # rolls back the prefix so the prior topology returns exactly.
    new_ids: list[str] = []
    try:
        for rec, parent_row in built:
            pool.add(rec)
            router.add_expert_row(init=parent_row)
            new_ids.append(rec.expert_id)
        if optim_state is not None and new_ids:
            _pad_state_rows(optim_state, len(new_ids))
    except Exception:
        # Exact rollback: committed prefix out, router topology restored,
        # ID cursor restored. _pad is two-phase so optim_state is untouched
        # on pad-build failure (pad is last, so no partial optim prefix).
        try:
            _restore_router(router, router_snap)
            for eid in reversed(new_ids):
                try:
                    pool.remove(eid)
                except KeyError:
                    pass
        finally:
            pool._next_id = next_before
        raise
    return new_ids
