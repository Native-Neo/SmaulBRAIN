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
invalidation for rewritten experts. Replay interleaving, growth schedule,
retention eval, and checkpointing are all driven here. Pruning never happens
during training: pool shrinkage is a manual offline operation
(``python pruning.py --rm-worst N``).
"""

from __future__ import annotations

import random
import time
from collections import deque

import torch

import growth as growth_mod
from bytes import PAD_ID


class _GradOnly:
    """Thin grad carrier: step_expert reads .grad; base weights come from FP8."""

    def __init__(self, grad: torch.Tensor | None) -> None:
        self.grad = grad
        self.shape = tuple(grad.shape) if grad is not None else None



_MODES = ("entire", "trunk", "experts", "selected", "new")


class ReplayBuffer:
    """Reservoir of past batches (byte-id sequences) for interleaving.

    Deterministic: all randomness derives from the owned ``random.Random``
    instance (seeded at construction, state persisted via to_dict/from_dict).
    No global RNG, no hash-order iteration (storage is a deque, sampling uses
    indexed choice, persistence preserves order).

    Capacity boundaries (defined): ``capacity`` must be >= 0; ``0`` means
    disabled (stores nothing, every sample is ``[]``). Negative capacities
    raise. Once full, each new item is kept with probability
    ``capacity / seen`` (classic Algorithm R: ``j = randrange(seen)``,
    replace ``buf[j]`` iff ``j < capacity``), so every prefix of the stream
    is uniformly represented.
    """

    def __init__(self, capacity: int = 512, seed: int = 0) -> None:
        if int(capacity) < 0:
            raise ValueError(f"ReplayBuffer capacity must be >= 0, got {capacity!r}")
        self.capacity = int(capacity)
        self.buf: deque[list[int]] = deque()
        self.rng = random.Random(seed)
        self.seen = 0

    def add(self, seq: list[int]) -> None:
        self.seen += 1
        seq = list(seq)  # ingress copy: later caller mutation must not corrupt replay
        if len(self.buf) < self.capacity:
            self.buf.append(seq)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.buf[j] = seq

    def sample(self, n: int) -> list[list[int]]:
        """Draw up to ``n`` past seqs, with replacement.

        Semantics (defined): each draw is an independent
        ``rng.choice(buf)``; duplicates are possible even when
        ``n <= len(buf)``. Returns ``min(max(n, 0), len(buf))`` copies
        (empty buffer or ``n <= 0`` -> ``[]``); callers must not assume a
        fixed output length until the buffer holds ``>= n`` items. Egress
        values are copies: mutating them cannot corrupt replay.
        """
        if n <= 0 or not self.buf:
            return []
        return [list(self.rng.choice(self.buf)) for _ in range(min(n, len(self.buf)))]

    def __len__(self) -> int:
        return len(self.buf)

    def to_dict(self) -> dict:
        """JSON-safe snapshot: complete state (sequences, cursor, sampler RNG).

        Complete means: ``capacity`` + ordered ``buf`` contents + ``seen``
        (reservoir cursor) + full ``rng`` state. Restoring via from_dict
        continues the exact sampling trajectory; no hidden cursor lives
        outside this dict.
        """
        version, inner, gauss = self.rng.getstate()
        return {
            "capacity": self.capacity,
            "buf": [list(s) for s in self.buf],
            "seen": self.seen,
            "rng": [int(version), [int(v) for v in inner], gauss],
        }

    @classmethod
    def from_dict(cls, d: dict, seed: int = 0) -> "ReplayBuffer":
        """Rebuild a buffer from to_dict (exact RNG continuation)."""
        buf = cls(capacity=int(d.get("capacity", 512)), seed=seed)
        buf.buf = deque([list(s) for s in d.get("buf", [])])
        buf.seen = int(d.get("seen", 0))
        rng = d.get("rng")
        if rng is not None:
            version, inner, gauss = rng
            buf.rng.setstate((int(version), tuple(int(v) for v in inner), gauss))
        return buf


def batch_from_seqs(seqs: list[list[int]], context: int, pad_id: int = PAD_ID) -> torch.Tensor:
    """Pack variable-length byte seqs into a [B, T] batch (truncate/pad).

    Padding defaults to PAD_ID (a structural special, never a real byte),
    so padded positions are distinguishable from data and masked from the
    loss. Callers may pass an explicit byte pad only when the pad positions
    are genuinely meant to score as data. Empty input raises (a batch must
    hold >= 1 row; undersized replay is clipped by the sampler, never by
    silently stacking zero rows).
    """
    if not seqs:
        raise ValueError("batch_from_seqs needs >= 1 seq (got empty list)")
    rows = []
    for s in seqs:
        s = s[: context + 1]
        if len(s) < context + 1:
            s = s + [pad_id] * (context + 1 - len(s))
        rows.append(torch.tensor(s, dtype=torch.long))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def evaluate_loss(model, batch: torch.Tensor, context: int) -> dict:
    """Old/new-data evaluation: mean NLL + accuracy (no gradients).

    Padding (PAD_ID) is excluded from both metrics; means normalize over
    valid targets only so pad length cannot dilute the measurement.
    """
    was_training = model.training
    model.eval()
    x = batch[:, :context]
    y = batch[:, 1 : context + 1]
    out = model.forward_infer(x)
    logits = out["logits"].float()
    valid = (y != PAD_ID)
    n_valid = int(valid.sum().item())
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, model.cfg.vocab_size),
                                           y.reshape(-1), reduction="none",
                                           ignore_index=PAD_ID)
    loss = (ce.sum() / max(1, n_valid)).item()
    acc = (((logits.argmax(-1) == y) & valid).float().sum() / max(1, n_valid)).item()
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
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"x and y must share one [B, T] shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"empty batch {tuple(x.shape)} carries no gradient signal")
    model.train()
    model.zero_grad(set_to_none=True)
    try:
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
        # Deterministic update order: pool.order when available (never dict
        # activation/insertion order, never hash/set order), else sorted ids.
        # Routing activation order must not affect optimizer/pager sequencing.
        _order = getattr(getattr(model, "pool", None), "order", None)
        if isinstance(_order, list) and _order:
            _rank = {eid: i for i, eid in enumerate(_order)}
            _key = lambda eid: (_rank.get(eid, 1 << 30), eid)
        else:
            _key = lambda eid: ("", eid)
        if mode in ("entire", "experts"):
            targets = sorted(expert_grads, key=_key)
        elif mode == "selected":
            want = set(selected or [])
            targets = sorted((e for e in expert_grads if e in want), key=_key)
        elif mode == "new":
            targets = sorted(
                (e for e in expert_grads
                 if model.pool.experts[e].birth_step >= new_since_step),
                key=_key)
        else:  # trunk mode: experts frozen
            targets = []
        for eid in targets:
            grads = expert_grads[eid]
            if not grads:
                continue
            rec = model.pool.experts[eid]
            compute = {n: _GradOnly(g) for n, g in grads.items()}
            # grad_scale stays 1.0: routing weights, ponder mass, and repeated
            # depth applications already scale these grads through autograd;
            # any manual factor here would double-count them.
            opt.step_expert(rec, compute, 1.0, cfg.expert_lr)
            model.pager.invalidate(eid)
            stepped.append(eid)
        # Global optimizer steps: exactly one per train_step call, including
        # steps whose updates were skipped (see skipped-step semantics). The
        # per-tensor bias corrections use each tensor's own step counter.
        opt.step_count += 1
    except Exception:
        model.clear_expert_grads(set_to_none=True)
        raise

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
    grow_every: int = 20000,
    grow_loss_below: float | None = 1.0,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Small-driver training over in-memory byte sequences.

    ``train_seqs`` are byte-id lists; batches cycle deterministically via the
    global step: ``train_seqs[(step * batch_size + i) % len]``. Resuming from
    ``model._resume_step`` continues the exact sequence (no repeat/skip) when
    the dataset order/length and ``batch_size`` are unchanged.

    Replay (when ``replay`` is not None and ``replay_n > 0``): each step
    samples past data only (pre-ingress buffer), so the current fresh batch
    can never duplicate itself via replay; the full fresh batch (all
    ``batch_size`` seqs) is then added to the reservoir. Total batch size is
    ``batch_size + min(replay_n, len(buffer pre-step))``: early steps run
    smaller until the buffer fills (defined behavior). Sampling is with
    replacement (see ReplayBuffer.sample). Loss is the mean over valid
    (non-PAD) tokens only, so fresh and replay tokens carry equal per-token
    weight and padding length cannot dilute the measurement; throughput
    (``bytes_processed``) likewise counts valid data bytes only.

    Growth fires on schedule (``grow_every``, default every 20000 steps)
    and whenever the step loss newly dips below ``grow_loss_below``
    (default 1.0; falling edge; negative disables). Training never prunes:
    the pool only grows here; shrinking is a manual offline operation
    (``python pruning.py --rm-worst N``), which respects the floor stored
    in the checkpoint's ``resume_config.json``.

    Exact trigger semantics (global steps, 0-indexed):
      * scheduled growth fires when ``(global_step + 1) % grow_every == 0``
        (``grow_every <= 0`` disables). Evaluated on the global step so a
        resume continues the cadence (no repeat/skip).
      * loss-edge growth fires when ``prev_loss >= grow_loss_below >
        loss_now`` (strict falling edge). ``prev_loss`` is the previous
        step's loss; the first step of a fresh run never fires (no
        previous loss). ``grow_loss_below is None`` or negative disables
        the edge (callers pass a negative value to disable).
      * only successful growths (new experts actually added; at-cap
        attempts return no ids) increment ``growth_events``.
      * trigger RNG is deterministic: scheduled growth seeds
        ``seed + step * max(1, max_new_experts)``, loss-edge growth adds a
        +7919 salt offset. Same ``(seed, step, max_new_experts)`` reproduces
        the same children; resuming with a different ``seed`` or
        ``max_new_experts`` intentionally re-bases the stream (both are
        pinned in the scheduler snapshot). The growth stream uses an
        isolated ``torch.Generator`` per event and the replay sampler uses
        an owned ``random.Random``: dirtying the global torch/python RNG
        between save and resume never perturbs the trajectory, while
        the RNG snapshot still restores the global RNG state itself.
    Returns history + optional retention report (old_seqs evaluated before
    and after) so continual-learning retention is measured, not claimed.
    """
    if not train_seqs:
        raise ValueError("train_seqs is empty: nothing to train on")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if replay_n < 0:
        raise ValueError(f"replay_n must be >= 0, got {replay_n!r}")
    from storage import save_model

    old_before = None
    if old_seqs:
        old_before = evaluate_loss(model, batch_from_seqs(old_seqs, cfg.context_length), cfg.context_length)
    hist: list[dict] = []
    n = max(1, len(train_seqs))
    start_step = int(getattr(model, "_resume_step", -1)) + 1
    # Resume scheduler state stashed by storage.load_model (loss edge,
    # growth counter, replay buffer). Fresh runs start clean.
    _snap = getattr(model, "_scheduler_snapshot", {}) or {}
    if not isinstance(_snap, dict):
        _snap = {}
    try:
        growth_events = int(_snap.get("growth_events", 0))
    except (TypeError, ValueError):
        growth_events = 0
    try:
        prev_loss = None if _snap.get("prev_loss") is None else float(_snap["prev_loss"])
    except (TypeError, ValueError):
        prev_loss = None  # corrupt entry: lose edge memory, never crash here
    if replay is None and isinstance(_snap.get("replay"), dict):
        # Snapshot restore (checkpoint-authoritative): passing replay=None on
        # resume continues the exact sampling trajectory. Passing an explicit
        # ReplayBuffer intentionally re-bases the sampler (fresh trajectory);
        # bit-equality with an uninterrupted run requires the None path.
        replay = ReplayBuffer.from_dict(_snap["replay"])

    def scheduler_snapshot() -> dict:
        # Dataset/batch cursor for exact resume: manifest step carries the
        # global step; batch_size/dataset_len/next_step pin the batch formula
        # (resuming with different values intentionally re-bases the cursor).
        # Scheduler knobs ride along so a resumer can reuse them exactly:
        # grow_every/grow_loss_below/seed/replay_n/max_new_experts/
        # min_experts/max_experts otherwise silently re-base trigger
        # cadences, batch composition, and RNG streams. max_new_experts pins
        # the growth salt (seed + step*max(1,max_new_experts)); replay_n pins
        # the interleaving width (batch_size + min(replay_n, len(buf))).
        # (config.json stays checkpoint-authoritative via storage.)
        return {
            "prev_loss": prev_loss,
            "growth_events": growth_events,
            "replay": replay.to_dict() if replay is not None else None,
            "batch_size": batch_size,
            "dataset_len": n,
            "next_step": start_step + len(hist),
            "grow_every": grow_every,
            "grow_loss_below": grow_loss_below,
            "seed": seed,
            "replay_n": replay_n,
            "max_new_experts": int(cfg.max_new_experts),
            "min_experts": int(cfg.min_experts),
            "max_experts": int(cfg.max_experts),
        }

    def grow_batch(step: int, salt: int) -> list[str]:
        """Clone the top experts (callee enforces the max_experts cap).

        Returns the new expert ids (empty when at cap): callers use the
        non-emptiness as the success signal so failed at-cap attempts never
        advance the growth counter.
        """
        count = min(cfg.max_new_experts, cfg.max_experts - len(model.pool))
        if count <= 0:
            return []
        new_ids = growth_mod.grow_topk_clones(
            model.pool, model.router, cfg.d_model, cfg.expert_hidden, step,
            seed=seed + salt, k=count,
            fp8_tile=cfg.fp8_tile, max_experts=cfg.max_experts,
            optim_state=opt.router_state,
        )
        for eid in new_ids:
            log_fn(f"[grow] step={step} new={eid} pool={len(model.pool)}")
        model.cfg.num_experts = len(model.pool)
        return list(new_ids)

    run_started = time.perf_counter()
    bytes_processed = 0
    events: list[dict] = []  # machine-readable growth ledger
    for local_step in range(steps):
        step_started = time.perf_counter()
        step = start_step + local_step
        fresh_seqs = [train_seqs[(step * batch_size + i) % n] for i in range(batch_size)]
        if replay is not None and replay_n > 0:
            # Past-only sampling: sample BEFORE ingress so the current fresh
            # batch cannot echo itself inside the same step.
            replay_batch = replay.sample(replay_n)
            batch_seqs = fresh_seqs + replay_batch
            for s in fresh_seqs:
                replay.add(s)
        else:
            batch_seqs = fresh_seqs
        b = batch_from_seqs(batch_seqs, cfg.context_length)
        stats = train_step(model, opt, cfg, b[:, : cfg.context_length],
                           b[:, 1 : cfg.context_length + 1], step, mode=mode,
                           selected=selected, new_since_step=new_since_step)
        stats["step"] = step
        # Throughput counts valid data bytes only: padded positions are
        # structural (PAD_ID, masked from the loss), not processed data.
        step_bytes = int((b[:, : cfg.context_length] != PAD_ID).sum().item())
        bytes_processed += step_bytes
        hist.append(stats)
        # Growth only: training never prunes (pool shrinkage is the manual
        # offline pruner). Successful growths increment the counter.
        new_ids: list[str] = []
        if grow_every and (step + 1) % grow_every == 0:
            new_ids += grow_batch(step, step * max(1, cfg.max_new_experts))
        # Loss-triggered growth on the falling edge below the threshold; the
        # salt offset keeps its seeds distinct from scheduled growth.
        loss_now = float(stats["loss"])
        if (grow_loss_below is not None and prev_loss is not None
                and prev_loss >= grow_loss_below > loss_now):
            new_ids += grow_batch(step, step * max(1, cfg.max_new_experts) + 7919)
        prev_loss = loss_now
        grew = bool(new_ids)
        if new_ids:
            events.append({"step": step, "type": "grow", "ids": list(new_ids)})
        if grew:
            growth_events += 1
        # Machine-readable per-step ledger (history entries stay JSON-safe).
        stats["grew"] = grew
        stats["new_experts"] = list(new_ids)
        stats["pruned_experts"] = []
        if ckpt_dir and save_every and (step + 1) % save_every == 0:
            save_model(ckpt_dir, model, opt, step,
                       extra_meta={"scheduler": scheduler_snapshot()})
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
        # Scheduler snapshot for exact resume: whoever saves the final
        # checkpoint passes this as extra_meta (see cli train command).
        "scheduler": scheduler_snapshot(),
        # Machine-readable growth ledger: one entry per growth event in
        # execution order.
        "growth_prune_events": [dict(e) for e in events],
        "growth_events": growth_events,
    }
