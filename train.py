"""Training loop: same core model for training / fine-tuning.

Fine-tuning is a training mode, not a separate implementation: ``mode``
selects which parameter groups move:

  entire   - trunk + router + all active experts
  trunk    - shared trunk + router only (experts frozen)
  experts  - router + all active experts (trunk frozen)
  selected - router + listed expert ids only
  new      - router + experts born at/after ``new_since_step``

Each step: forward (ponder-weighted loss) -> backward -> SmaulOpt updates
(trunk at trunk_lr, router at router_lr, experts at expert_lr) -> pager
invalidation for rewritten experts. Replay interleaving, growth/pruning
schedule, retention eval, and checkpointing are all driven here.
"""

from __future__ import annotations

import time

import torch

import growth as growth_mod
import pruning as pruning_mod
from continual import ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report


class _GradOnly:
    """Thin grad carrier: step_expert reads .grad; base weights come from FP8."""

    def __init__(self, grad: torch.Tensor) -> None:
        self.grad = grad
        self.shape = tuple(grad.shape)


_MODES = ("entire", "trunk", "experts", "selected", "new")


def train_step(model, opt, cfg, x: torch.Tensor, y: torch.Tensor, step: int,
               mode: str = "entire", selected: list[str] | None = None,
               new_since_step: int = 0) -> dict:
    """One optimizer step. Returns loss stats + stepped expert ids."""
    if mode not in _MODES:
        raise ValueError(f"unknown training mode {mode!r}; expected one of {_MODES}")
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x, y, step=step)
    loss = out["loss"]
    loss.backward()
    stepped: list[str] = []
    train_trunk = mode in ("entire", "trunk")
    train_router = mode in ("entire", "trunk", "experts", "selected", "new")
    if train_trunk:
        opt.step_trunk(model._trunk_params(), cfg.trunk_lr)
    else:
        for _, p in model._trunk_params():
            p.grad = None
    if train_router:
        opt.step_router(model._router_params(), cfg.router_lr)
    else:
        for _, p in model._router_params():
            p.grad = None
    expert_grads = model.take_expert_grads()
    if mode in ("entire", "experts"):
        targets = list(expert_grads)
    elif mode == "selected":
        want = set(selected or [])
        targets = [e for e in expert_grads if e in want]
    elif mode == "new":
        targets = [e for e in expert_grads
                   if model.pool.experts[e].birth_step >= new_since_step]
    else:  # trunk mode: experts frozen
        targets = []
    for eid in targets:
        grads = expert_grads[eid]
        if not grads:
            continue
        rec = model.pool.experts[eid]
        compute = {n: _GradOnly(g) for n, g in grads.items()}
        opt.step_expert(rec, compute, 1.0, cfg.expert_lr)
        model.pager.invalidate(eid)
        stepped.append(eid)
    return {
        "loss": float(loss.item()),
        "nll": float(out["nll"].item()),
        "ponder_kl": float(out["ponder_kl"].item()),
        "acc": float(out["acc"]),
        "mean_depth": float(out["mean_depth"]),
        "stepped_experts": stepped,
    }


def run_training(
    model,
    opt,
    cfg,
    train_seqs: list[list[int]],
    steps: int = 20,
    batch_size: int = 2,
    mode: str = "entire",
    selected: list[str] | None = None,
    new_since_step: int = 0,
    ckpt_dir: str | None = None,
    save_every: int = 0,
    replay: ReplayBuffer | None = None,
    replay_n: int = 0,
    old_seqs: list[list[int]] | None = None,
    grow_every: int = 0,
    prune_every: int = 0,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Small-driver training over in-memory byte sequences.

    ``train_seqs`` are byte-id lists; batches cycle deterministically.
    Returns history + optional retention report (old_seqs evaluated before
    and after) so continual-learning retention is measured, not claimed.
    """
    from storage import save_model

    old_before = None
    if old_seqs:
        old_before = evaluate_loss(model, batch_from_seqs(old_seqs, cfg.context_length), cfg.context_length)
    hist: list[dict] = []
    n = max(1, len(train_seqs))
    start_step = int(getattr(model, "_resume_step", -1)) + 1
    run_started = time.perf_counter()
    bytes_processed = 0
    for local_step in range(steps):
        step_started = time.perf_counter()
        step = start_step + local_step
        batch_seqs = [train_seqs[(step * batch_size + i) % n] for i in range(batch_size)]
        if replay is not None and replay_n > 0:
            for s in train_seqs[(step * batch_size) % n : (step * batch_size) % n + 1]:
                replay.add(list(s))
            batch_seqs = batch_seqs + replay.sample(replay_n)
        b = batch_from_seqs(batch_seqs, cfg.context_length)
        stats = train_step(model, opt, cfg, b[:, : cfg.context_length],
                           b[:, 1 : cfg.context_length + 1], step, mode=mode,
                           selected=selected, new_since_step=new_since_step)
        stats["step"] = step
        step_bytes = int(b[:, : cfg.context_length].numel())
        bytes_processed += step_bytes
        hist.append(stats)
        if grow_every and (step + 1) % grow_every == 0 and len(model.pool) < cfg.max_experts:
            room = cfg.max_experts - len(model.pool)
            count = min(cfg.max_new_experts, room)
            for growth_index in range(count):
                eid = growth_mod.grow_expert(
                    model.pool, model.router, cfg.d_model, cfg.expert_hidden, step,
                    seed=seed + step * max(1, cfg.max_new_experts) + growth_index,
                    fp8_tile=cfg.fp8_tile,
                )
                log_fn(f"[grow] step={step} new={eid} pool={len(model.pool)}")
        if prune_every and (step + 1) % prune_every == 0:
            victims = pruning_mod.find_victims(model.pool, step,
                                               survival_steps=cfg.prune_survival_steps,
                                               min_experts=cfg.min_experts,
                                               usage_threshold=cfg.prune_min_usage)
            if victims:
                pruned = pruning_mod.prune_experts(model.pool, model.router, victims)
                log_fn(f"[prune] step={step} removed={pruned} pool={len(model.pool)}")
        if ckpt_dir and save_every and (step + 1) % save_every == 0:
            save_model(ckpt_dir, model, opt, step)
            model._resume_step = step

        elapsed = time.perf_counter() - step_started
        total_elapsed = time.perf_counter() - run_started
        completed = local_step + 1
        avg_step = total_elapsed / completed
        eta = max(0.0, (steps - completed) * avg_step)
        step_bps = step_bytes / max(elapsed, 1e-12)
        avg_bps = bytes_processed / max(total_elapsed, 1e-12)
        stepped = ",".join(stats["stepped_experts"]) or "-"
        log_fn(
            f"[train] step {completed}/{steps} (global={step}) "
            f"loss={stats['loss']:.4f} nll={stats['nll']:.4f} "
            f"acc={stats['acc']:.2%} depth={stats['mean_depth']:.2f} "
            f"time={elapsed:.2f}s bytes/s={step_bps:.1f} "
            f"avg_bytes/s={avg_bps:.1f} ETA={eta:.1f}s "
            f"experts={len(model.pool)} updated={stepped} "
            f"lr(trunk/router/expert)={cfg.trunk_lr:.2g}/"
            f"{cfg.router_lr:.2g}/{cfg.expert_lr:.2g}"
        )
    total_elapsed = time.perf_counter() - run_started
    if hist:
        model._resume_step = int(hist[-1]["step"])
    report = None
    if old_seqs and old_before is not None:
        old_after = evaluate_loss(model, batch_from_seqs(old_seqs, cfg.context_length), cfg.context_length)
        report = retention_report(old_before, old_after)
    return {
        "history": hist,
        "retention": report,
        "final_loss": hist[-1]["loss"] if hist else None,
        "steps_completed": len(hist),
        "elapsed_seconds": total_elapsed,
        "bytes_processed": bytes_processed,
        "bytes_per_second": bytes_processed / max(total_elapsed, 1e-12),
    }
