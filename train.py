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

import random
import time
from collections import deque

import torch

import growth as growth_mod
import pruning as pruning_mod
from bytes import PAD_ID


class _GradOnly:
    """Thin grad carrier: step_expert reads .grad; base weights come from FP8."""

    def __init__(self, grad: torch.Tensor) -> None:
        self.grad = grad
        self.shape = tuple(grad.shape)


_MODES = ("entire", "trunk", "experts", "selected", "new")


class ReplayBuffer:
    """Reservoir of past batches (byte-id sequences) for interleaving."""

    def __init__(self, capacity: int = 512, seed: int = 0) -> None:
        self.capacity = capacity
        self.buf: deque[list[int]] = deque()
        self.rng = random.Random(seed)
        self.seen = 0

    def add(self, seq: list[int]) -> None:
        self.seen += 1
        if len(self.buf) < self.capacity:
            self.buf.append(seq)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.buf[j] = seq

    def sample(self, n: int) -> list[list[int]]:
        if not self.buf:
            return []
        return [list(self.rng.choice(self.buf)) for _ in range(min(n, len(self.buf)))]

    def __len__(self) -> int:
        return len(self.buf)


def batch_from_seqs(seqs: list[list[int]], context: int, pad_id: int = PAD_ID) -> torch.Tensor:
    """Pack variable-length byte seqs into a [B, T] batch (truncate/pad)."""
    rows = []
    for s in seqs:
        s = s[: context + 1]
        if len(s) < context + 1:
            s = s + [pad_id] * (context + 1 - len(s))
        rows.append(torch.tensor(s, dtype=torch.long))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def evaluate_loss(model, batch: torch.Tensor, context: int) -> dict:
    """Old/new-data evaluation: mean NLL + accuracy (no gradients)."""
    was_training = model.training
    model.eval()
    x = batch[:, :context]
    y = batch[:, 1 : context + 1]
    out = model.forward_infer(x)
    logits = out["logits"].float()
    mask = y.ne(model.cfg.pad_id)
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_size),
        y.reshape(-1),
        reduction="none",
    ).reshape_as(y)
    denom = mask.sum().clamp_min(1)
    loss = per_token.masked_fill(~mask, 0.0).sum().div(denom).item()
    acc = ((logits.argmax(-1) == y) & mask).float().sum().div(denom).item()
    if was_training:
        model.train()
    return {"loss": loss, "acc": acc, "mean_executed": float(out["n_executed"])}


def retention_report(before: dict, after: dict) -> dict:
    """Numerically report forgetting: old-data metrics before vs. after."""
    return {
        "old_loss_before": before["loss"],
        "old_loss_after": after["loss"],
        "old_loss_delta": after["loss"] - before["loss"],
        "old_acc_before": before["acc"],
        "old_acc_after": after["acc"],
        "old_acc_delta": after["acc"] - before["acc"],
        "retained": bool(after["loss"] <= before["loss"] * 1.5),
    }


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
    grow_loss_below: float | None = 0.75,
    growths_per_prune: int = 2,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Small-driver training over in-memory byte sequences.

    ``train_seqs`` are byte-id lists; batches cycle deterministically.
    Growth fires on schedule (``grow_every``) and whenever the step loss
    newly dips below ``grow_loss_below`` (falling edge; negative disables).
    After every ``growths_per_prune`` growth events one prune evaluation
    runs, on top of the ``prune_every`` schedule.
    Returns history + optional retention report (old_seqs evaluated before
    and after) so continual-learning retention is measured, not claimed.
    """
    if growths_per_prune < 1:
        raise ValueError(f"growths_per_prune must be >= 1, got {growths_per_prune!r}")
    from storage import save_model

    old_before = None
    if old_seqs:
        old_before = evaluate_loss(model, batch_from_seqs(old_seqs, cfg.context_length, cfg.pad_id), cfg.context_length)
    hist: list[dict] = []
    n = max(1, len(train_seqs))
    start_step = int(getattr(model, "_resume_step", -1)) + 1
    growth_events = 0
    prev_loss: float | None = None

    def grow_batch(step: int, salt: int) -> bool:
        """Clone the top-8 experts (exact copies + 2 mutants), room-capped."""
        room = cfg.max_experts - len(model.pool)
        if room <= 0:
            return False
        new_ids = growth_mod.grow_topk_clones(
            model.pool, model.router, cfg.d_model, cfg.expert_hidden, step,
            seed=seed + salt, k=min(8, room), n_mutated=2,
            fp8_tile=cfg.fp8_tile,
        )
        for eid in new_ids:
            log_fn(f"[grow] step={step} new={eid} pool={len(model.pool)}")
        model.cfg.num_experts = len(model.pool)
        return bool(new_ids)

    def prune_eval(step: int) -> None:
        victims = pruning_mod.find_victims(model.pool, step,
                                           survival_steps=cfg.prune_survival_steps,
                                           min_experts=cfg.min_experts,
                                           usage_threshold=cfg.prune_min_usage)
        if victims:
            pruned = pruning_mod.prune_experts(model.pool, model.router, victims)
            for eid in pruned:
                model.pager.forget(eid)
            model.cfg.num_experts = len(model.pool)
            log_fn(f"[prune] step={step} removed={pruned} pool={len(model.pool)}")
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
        b = batch_from_seqs(batch_seqs, cfg.context_length, cfg.pad_id)
        stats = train_step(model, opt, cfg, b[:, : cfg.context_length],
                           b[:, 1 : cfg.context_length + 1], step, mode=mode,
                           selected=selected, new_since_step=new_since_step)
        stats["step"] = step
        step_bytes = int(b[:, : cfg.context_length].numel())
        bytes_processed += step_bytes
        hist.append(stats)
        grew = False
        if grow_every and (step + 1) % grow_every == 0:
            grew = grow_batch(step, step * max(1, cfg.max_new_experts)) or grew
        # Loss-triggered growth on the falling edge below the threshold; the
        # salt offset keeps its seeds distinct from scheduled growth.
        loss_now = float(stats["loss"])
        if (grow_loss_below is not None and prev_loss is not None
                and prev_loss >= grow_loss_below > loss_now):
            grew = grow_batch(step, step * max(1, cfg.max_new_experts) + 7919) or grew
        prev_loss = loss_now
        if grew:
            growth_events += 1
            if growth_events % growths_per_prune == 0:
                prune_eval(step)
        if prune_every and (step + 1) % prune_every == 0:
            prune_eval(step)
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
        old_after = evaluate_loss(model, batch_from_seqs(old_seqs, cfg.context_length, cfg.pad_id), cfg.context_length)
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
