"""Offline expert pruner: standalone CLI, not a library.

Training never prunes; pool shrinkage happens here, explicitly::

    python pruning.py --ckpt checkpoints/smaulbrain --rm-worst 4

The floor comes from the checkpoint's ``resume_config.json``
(``min_experts``); removal never takes the pool below it. Selection is the
standing hysteresis policy: an expert dies only when old enough, long idle,
AND below threshold on every vitality signal (usage share, gradient
activity, contribution), plus a redundancy pass that retires near-duplicate
gate weights below a looser usage bar. Removal is atomic (weights +
optimizer state + router row + pager traces, highest index first) and the
checkpoint is re-saved in place, still resumable (scheduler snapshot and
RNG state preserved).

This file is a script, not an importable module: nothing in the training
path imports it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from precision import dequantize_fp8_blockwise

FLOOR_DEFAULT = 64


def _check_optim_width(optim_state: dict | None, expect: int) -> None:
    """Validate-first: every momentum buffer must already match pool width.

    Raises before mutating when a buffer's dim-0 disagrees with the pool,
    so row drops cannot silently misalign survivors. Missing entries are
    left alone (step_dense initializes them on next use).
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


def _drop_state_row(state: dict, index: int) -> None:
    """Delete dim-0 row ``index`` from every stored momentum buffer.

    Mirrors remove_expert_row so surviving rows keep their momentum aligned.
    ``v_col`` is per-column and untouched; the shared step counter is kept.
    Two-phase: all replacements are built before any assignment, so a
    failure leaves every buffer untouched instead of a half-dropped row.
    """
    # Phase 1 (pure build): compute every replacement before assigning any.
    pending: list[tuple] = []
    for st in state.values():
        if not isinstance(st, dict):
            continue
        for key in ("m", "v_row", "v"):
            t = st.get(key)
            if torch.is_tensor(t) and 0 <= index < t.shape[0]:
                pending.append((st, key, torch.cat([t[:index], t[index + 1:]])))
    for st, key, new in pending:
        st[key] = new


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
    optim_state: dict | None = None,  # e.g. SmaulOpt.router_state: rows migrate
) -> list[str]:
    """Remove victims: weights + optimizer state + router row + pager traces.

    Validate-first: every victim must exist and pool/router widths must
    match before anything is removed, so a bad victim list cannot commit a
    prefix and leave pool/router diverged. Optimizer widths are validated
    too, so a stale momentum table refuses before misaligning survivors.
    Router rows are removed highest-index-first so surviving indices stay
    valid during the sweep. Each momentum-row drop is two-phase (build
    then assign) so one victim's drop cannot half-apply. Pager traces are
    forgotten inline last (never served ghosts on direct calls); pool and
    router stay in lockstep even if a side channel fails mid-sweep.
    Optimizer momentum rows are deleted inline at the same indices, so
    surviving rows keep their momentum instead of resetting on mismatch.
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
    _check_optim_width(optim_state, len(pool))
    ordered = sorted(victims, key=lambda eid: pool.index_of(eid), reverse=True)
    pruned: list[str] = []
    for eid in ordered:
        idx = pool.index_of(eid)
        rec = pool.remove(eid)  # drops weights + optim_state + metadata
        assert rec.expert_id == eid
        router.remove_expert_row(idx)
        if optim_state is not None:
            _drop_state_row(optim_state, idx)
        if pager is not None:
            pager.forget(eid)
    return sorted(pruned, key=lambda eid: eid)


def build_parser() -> argparse.ArgumentParser:
    from config import SmaulBrainConfig
    p = argparse.ArgumentParser(prog="pruning",
                                description="Remove the N worst experts from a checkpoint (offline; training never prunes).")
    p.add_argument("--ckpt", type=str, default="checkpoints/smaulbrain",
                   help="Checkpoint directory.")
    p.add_argument("--rm-worst", type=int, required=True,
                   help="How many of the worst experts to remove (capped by the resume_config floor).")
    p.add_argument("--survival-steps", type=int, default=None,
                   help=f"Grace period before an expert may die (default: {SmaulBrainConfig.prune_survival_steps}).")
    p.add_argument("--min-usage", type=float, default=None,
                   help=f"Usage share below which an expert is dying (default: {SmaulBrainConfig.prune_min_usage}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Rank and report victims without modifying the checkpoint.")
    return p


def main(argv: list[str] | None = None) -> int:
    from config import SmaulBrainConfig
    from model import SmaulBrainModel
    from smaulopt import SmaulOpt
    from storage import load_manifest, load_model, load_resume_config, save_model

    args = build_parser().parse_args(argv)
    if args.rm_worst < 0:
        print(json.dumps({"error": "--rm-worst must be >= 0"}), file=sys.stderr)
        return 2
    ckpt = args.ckpt
    try:
        resume_cfg = load_resume_config(ckpt)
        floor = int(resume_cfg.get("min_experts", FLOOR_DEFAULT))
    except FileNotFoundError:
        print(f"warning: no resume_config.json in {ckpt}; using default floor {FLOOR_DEFAULT}",
              file=sys.stderr)
        floor = FLOOR_DEFAULT
    except (ValueError, TypeError) as e:
        print(json.dumps({"error": f"bad resume_config.json: {e}"}), file=sys.stderr)
        return 2
    cfg_path = os.path.join(ckpt, "config.json")
    if not os.path.exists(cfg_path):
        print(json.dumps({"error": f"no checkpoint at {ckpt}; train first"}), file=sys.stderr)
        return 2
    with open(cfg_path) as f:
        cfg = SmaulBrainConfig.from_dict(json.load(f))
    survival = args.survival_steps if args.survival_steps is not None else cfg.prune_survival_steps
    usage = args.min_usage if args.min_usage is not None else cfg.prune_min_usage
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt()
    try:
        manifest = load_model(ckpt, model, opt)
    except (ValueError, FileNotFoundError) as e:
        print(json.dumps({"error": f"cannot load checkpoint: {e}"}), file=sys.stderr)
        return 2
    step = int(manifest.get("step", 0))
    total_tokens = sum(r.tokens_routed for r in model.pool.experts.values())
    if total_tokens == 0:
        print("warning: checkpoint has zero routed tokens; every old expert "
              "looks dying (grace periods still apply)", file=sys.stderr)
    removable = max(0, len(model.pool) - floor)
    if args.rm_worst > removable:
        print(json.dumps({"error": f"requested {args.rm_worst} but only {removable} removable "
                                   f"(pool={len(model.pool)}, floor={floor}); nothing changed"}),
              file=sys.stderr)
        return 2
    victims = find_victims(model.pool, step, survival_steps=survival,
                           min_experts=floor, usage_threshold=usage,
                           max_victims=args.rm_worst)
    result = {"requested": args.rm_worst, "floor": floor,
              "pool_before": len(model.pool),
              "victims": victims, "removed": [], "dry_run": bool(args.dry_run)}
    if not args.dry_run:
        result["removed"] = prune_experts(model.pool, model.router, victims,
                                          pager=model.pager, optim_state=opt.router_state)
        model.cfg.num_experts = len(model.pool)
        extra = manifest.get("extra", {})
        save_model(ckpt, model, opt, step,
                   extra_meta=extra if isinstance(extra, dict) else {})
        result["removed"] = removed
    result["pool_after"] = len(model.pool)
    model.pager.close()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
