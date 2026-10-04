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

import torch

import growth as growth_mod
import pruning as pruning_mod
from continual import ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report


class _GradOnly:
    """Thin grad carrier: step_expert reads .grad; base weights come from FP8."""

    def __init__(self, grad: torch.Tensor) -> None:
        self.grad = grad
        self.shape = tuple(grad.shape)


def train_step(model, opt, cfg, x: torch.Tensor, y: torch.Tensor, step: int,
               mode: str = "entire", selected: list[str] | None = None,
               new_since_step: int = 0) -> dict:
    """One optimizer step. Returns loss stats + stepped expert ids."""
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
    for step in range(steps):
        batch_seqs = [train_seqs[(step * batch_size + i) % n] for i in range(batch_size)]
        if replay is not None and replay_n > 0:
            for s in train_seqs[(step * batch_size) % n : (step * batch_size) % n + 1]:
                replay.add(list(s))
            batch_seqs = batch_seqs + replay.sample(replay_n)
        b = batch_from_seqs(batch_seqs, cfg.context_length)
        stats = train_step(model, opt, cfg, b[:, : cfg.context_length],
                           b[:, 1 : cfg.context_length + 1], step, mode=mode)
        stats["step"] = step
        hist.append(stats)
        if grow_every and (step + 1) % grow_every == 0 and len(model.pool) < cfg.max_experts:
            eid = growth_mod.grow_expert(model.pool, model.router, cfg.d_model,
                                         cfg.expert_hidden, step, seed=seed + step,
                                         fp8_tile=cfg.fp8_tile)
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
    report = None
    if old_seqs and old_before is not None:
        old_after = evaluate_loss(model, batch_from_seqs(old_seqs, cfg.context_length), cfg.context_length)
        report = retention_report(old_before, old_after)
    return {"history": hist, "retention": report,
            "final_loss": hist[-1]["loss"] if hist else None}
