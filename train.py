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

Machine-readable ledger (stable API): every per-step ``stats`` dict carries
``grew`` (bool), ``new_experts`` (ids added this step), and
``pruned_experts`` (always ``[]`` — training never prunes; the key exists
as a list by contract and is reserved for offline-pruner integration, so
consumers may rely on its presence). ``run_training`` additionally returns
``growth_prune_events`` (one entry per growth event, in execution order)
and ``growth_events`` (count of successful growths).
"""

from __future__ import annotations

import math
import random
import signal
import time
from collections import deque

import torch

import growth as growth_mod
from bytes import PAD_ID


def cosine_scale(step: int, horizon: int) -> float:
    """Cosine LR multiplier: 1.0 at step 0 decaying to ~0.0 at horizon.

    ``horizon <= 0`` disables (constant 1.0). Steps past the horizon pin
    at ~0.0 (never negative, never NaN).
    """
    if horizon <= 0:
        return 1.0
    t = min(max(int(step), 0), horizon) / horizon
    return 0.5 * (1.0 + math.cos(math.pi * t))


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
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError(
                f"ReplayBuffer capacity must be an int (not bool), got {capacity!r}"
            )
        if capacity < 0:
            raise ValueError(f"ReplayBuffer capacity must be >= 0, got {capacity!r}")
        self.capacity = capacity
        self.buf: deque[list[int]] = deque()
        self.rng = random.Random(seed)
        self.seen = 0

    def add(self, seq: list[int]) -> None:
        if not isinstance(seq, (list, tuple)):
            raise ValueError(f"ReplayBuffer sequence must be a list or tuple, got {type(seq).__name__}")
        for token in seq:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError(
                    f"ReplayBuffer token ids must be non-negative ints (not bool), got {token!r}"
                )
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
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"ReplayBuffer sample count must be an int (not bool), got {n!r}")
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
        """Rebuild a buffer from to_dict (exact RNG continuation).

        Strict input validation (mirrors storage._validate_rng_parts style,
        duplicated locally to avoid a storage import cycle): bool values
        never coerce (True -> 1 is rejected), inner state entries must be
        ints in [0, 2**32), and gauss must be None or a finite float.
        Violations raise ValueError.
        """
        if not isinstance(d, dict):
            raise ValueError(f"ReplayBuffer payload must be a dict, got {type(d).__name__}")
        cap_raw = d.get("capacity", 512)
        if isinstance(cap_raw, bool) or not isinstance(cap_raw, int):
            raise ValueError(f"ReplayBuffer capacity must be an int (not bool), got {cap_raw!r}")
        if cap_raw < 0:
            raise ValueError(f"ReplayBuffer capacity must be >= 0, got {cap_raw!r}")
        buf_raw = d.get("buf", [])
        if not isinstance(buf_raw, list):
            raise ValueError(f"ReplayBuffer buf must be a list, got {type(buf_raw).__name__}")
        for _s in buf_raw:
            if not isinstance(_s, (list, tuple)):
                raise ValueError(f"ReplayBuffer buf entries must be lists, got {_s!r}")
            for _v in _s:
                if isinstance(_v, bool) or not isinstance(_v, int) or _v < 0:
                    raise ValueError(
                        f"ReplayBuffer token ids must be non-negative ints (not bool), got {_v!r}"
                    )
        seen_raw = d.get("seen", 0)
        if isinstance(seen_raw, bool) or not isinstance(seen_raw, int):
            raise ValueError(f"ReplayBuffer seen must be an int (not bool), got {seen_raw!r}")
        if seen_raw < 0:
            raise ValueError(f"ReplayBuffer seen must be >= 0, got {seen_raw!r}")
        expected_len = min(seen_raw, cap_raw)
        if len(buf_raw) != expected_len:
            raise ValueError(
                "ReplayBuffer snapshot is inconsistent: "
                f"len(buf)={len(buf_raw)}, seen={seen_raw}, capacity={cap_raw}; "
                f"expected len(buf)={expected_len}"
            )
        rng = d.get("rng")
        if rng is not None:
            if not (isinstance(rng, (list, tuple)) and len(rng) == 3):
                raise ValueError(f"ReplayBuffer rng must be [version, inner, gauss], got {rng!r}")
            version, inner, gauss = rng
            if isinstance(version, bool) or not isinstance(version, int):
                raise ValueError(f"ReplayBuffer rng version must be an int (not bool), got {version!r}")
            if not (isinstance(inner, (list, tuple)) and len(inner) == 625):
                raise ValueError("ReplayBuffer rng inner must be 625 entries")
            for _v in inner:
                if isinstance(_v, bool) or not isinstance(_v, int) or not (0 <= _v < 2 ** 32):
                    raise ValueError(f"ReplayBuffer rng inner entries must be ints in [0, 2**32), got {_v!r}")
            if not (gauss is None or (isinstance(gauss, float) and not isinstance(gauss, bool) and math.isfinite(gauss))):
                raise ValueError(f"ReplayBuffer rng gauss must be None or a finite float, got {gauss!r}")
            try:
                rng_state = (int(version), tuple(int(v) for v in inner), gauss)
            except (TypeError, ValueError) as e:
                raise ValueError(f"ReplayBuffer rng has bad entries: {e}") from e
        buf = cls(capacity=cap_raw, seed=seed)
        buf.buf = deque([list(s) for s in buf_raw])
        buf.seen = seen_raw
        if rng is not None:
            buf.rng.setstate(rng_state)  # validated above; no bool coercion survives
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
    if isinstance(context, bool) or not isinstance(context, int) or context < 1:
        raise ValueError(f"context must be an int >= 1, got {context!r}")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
        raise ValueError(f"pad_id must be a non-negative int (not bool), got {pad_id!r}")
    if not isinstance(seqs, (list, tuple)) or not seqs:
        raise ValueError("batch_from_seqs needs >= 1 sequence (got empty or invalid input)")
    rows = []
    for s in seqs:
        if not isinstance(s, (list, tuple)):
            raise ValueError(f"each sequence must be a list or tuple, got {type(s).__name__}")
        for token in s:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError(
                    f"token IDs must be non-negative ints (not bool), got {token!r}"
                )
        s = list(s[: context + 1])
        if len(s) < context + 1:
            s = s + [pad_id] * (context + 1 - len(s))
        rows.append(torch.tensor(s, dtype=torch.long))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def evaluate_loss(model, batch: torch.Tensor, context: int) -> dict:
    """Old/new-data evaluation: mean NLL + accuracy (no gradients).

    Padding (PAD_ID) is excluded from both metrics; means normalize over
    valid targets only so pad length cannot dilute the measurement.
    Scores the inference readout (threshold-selected depth), not the
    ponder-mixed predictor training optimizes — deliberately: retention
    compares deployed behavior before vs. after, and both sides use this
    same readout so the delta is apples-to-apples.
    """
    was_training = model.training
    model.eval()
    try:
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
        return {"loss": loss, "acc": acc, "mean_executed": float(out["n_executed"])}
    finally:
        model.train(was_training)


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


def _new_cap_factor(old_delta: torch.Tensor, new_delta: torch.Tensor, cap: float) -> float:
    """Safety-ceiling factor for old-router-row deltas (pure, deterministic).

    Returns ``f in [0, 1]`` such that scaling ``old_delta`` by ``f`` keeps
    every aggregate old/new movement ratio (Frobenius, mean per-row L2,
    mean abs) at or below ``cap``. All three ratios are linear in the old
    block's scale, so a single factor fixes them together. Returns 1.0 when
    no cap is needed, and also when the new block did not move (ratio
    undefined -- exact-mult semantics are kept instead of freezing).
    """
    f = 1.0
    o_fro = float(old_delta.float().pow(2).sum().sqrt().item())
    n_fro = float(new_delta.float().pow(2).sum().sqrt().item())
    if o_fro > 0.0:
        if n_fro < 1e-12:
            return 1.0
        f = min(f, cap * n_fro / o_fro)
    if old_delta.ndim == 1:
        o_mean = float(old_delta.float().abs().mean().item())
        n_mean = float(new_delta.float().abs().mean().item())
    else:
        o_mean = float(old_delta.float().pow(2).sum(dim=1).sqrt().mean().item())
        n_mean = float(new_delta.float().pow(2).sum(dim=1).sqrt().mean().item())
    if o_mean > 0.0:
        if n_mean < 1e-12:
            return 1.0
        f = min(f, cap * n_mean / o_mean)
    o_l1 = float(old_delta.float().abs().mean().item())
    n_l1 = float(new_delta.float().abs().mean().item())
    if o_l1 > 0.0:
        if n_l1 < 1e-12:
            return 1.0
        f = min(f, cap * n_l1 / o_l1)
    return max(0.0, min(1.0, f))


def train_step(model, opt, cfg, x: torch.Tensor, y: torch.Tensor, step: int,
               mode: str = "entire", selected: list[str] | None = None,
               new_since_step: int = 0, profile: bool = False) -> dict:
    """One optimizer step. Returns loss stats + stepped expert ids.

    Stepped ledger semantics (pinned): ``stepped_experts`` lists exactly
    the experts whose ``record.version`` bumped during this call, i.e.
    whose optimizer moments moved (applied update, not a skip). Skipped
    updates (no grads, non-finite grads) bump nothing and are never listed.
    A zero effective LR (e.g. cosine decay pinned at ~0.0 past the horizon)
    still counts as stepped: moments decay on the grad and ``version``
    bumps, the pager entry is invalidated, and the id appears in
    ``stepped`` even though weights are numerically unmoved. This churn is
    intentional — version-delta is unambiguous where the EMA activity
    signal can be exactly 0.0 — and callers must not interpret ``stepped``
    as proof of weight movement.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown training mode {mode!r}; expected one of {_MODES}")
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"x and y must share one [B, T] shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"empty batch {tuple(x.shape)} carries no gradient signal")
    model.train()
    model._profile_enabled = bool(profile)
    model._profile_times = {"router_s": 0.0, "expert_s": 0.0, "balance_s": 0.0}
    model.zero_grad(set_to_none=True)
    # New-expert routing steering (runtime-only): in mode `new`, add
    # cfg.new_routing_bias to the router logits of experts born at/after
    # new_since_step so new-domain tokens reach new experts more often.
    # Plain router attributes only -- never checkpointed, manifested, or
    # snapshotted. Scoped strictly to the training forward below: set
    # immediately before, cleared in a finally immediately after, so at
    # rest (inference: forward_infer/generate/evaluate_loss, save/load)
    # the router is always unsteered (0.0/()). The try/finally also
    # covers the exception path, so a failed forward leaves no stale
    # steering behind. Steering changes routing weights/choice only; the
    # trunk/expert freeze below is untouched (old-expert weights cannot
    # move).
    try:
        _router = getattr(model, "router", None)
        if _router is not None:
            if mode == "new":
                _bias = float(getattr(cfg, "new_routing_bias", 0.0) or 0.0)
                _pool = getattr(model, "pool", None)
                _order = list(getattr(_pool, "order", None) or [])
                _experts = getattr(_pool, "experts", {}) or {}
                _router.new_routing_bias = _bias
                _router.new_expert_idx = tuple(
                    i for i, eid in enumerate(_order)
                    if eid in _experts
                    and _experts[eid].birth_step >= new_since_step)
            else:
                _router.new_routing_bias = 0.0
                _router.new_expert_idx = ()
        forward_started = time.perf_counter() if profile else 0.0
        try:
            out = model(x, y, step=step)
        finally:
            if profile:
                forward_s = time.perf_counter() - forward_started
            if _router is not None:
                _router.new_routing_bias = 0.0
                _router.new_expert_idx = ()
        loss = out["loss"]
        if not torch.isfinite(loss.detach()).all().item():
            raise ValueError("non-finite training loss; refusing to step on NaN/inf")
        backward_started = time.perf_counter() if profile else 0.0
        loss.backward()
        backward_s = time.perf_counter() - backward_started if profile else 0.0
        optimizer_started = time.perf_counter() if profile else 0.0
        stepped: list[str] = []
        train_trunk = mode in ("entire", "trunk")
        train_router = mode in ("entire", "trunk", "experts", "selected", "new")
        # Cosine decay (default-off): scale all three LRs by the schedule.
        lr_scale = cosine_scale(step, cfg.cosine_decay_steps)
        if train_trunk:
            opt.step_trunk(model._trunk_params(), cfg.trunk_lr * lr_scale)
        else:
            for _, p in model._trunk_params():
                p.grad = None
        if train_router:
            # Mode `new` only: old experts' router rows learn at
            # router_lr_mult_new while newborn rows (birth_step >= cutoff,
            # incl. rows appended by mid-run growth) train at full LR.
            # Post-step-only delta rescale: the optimizer steps on UNSCALED
            # grads (bit-identical clip norm, factored v_row/v_col moments
            # and new-row deltas to a full-LR run), then old rows' applied
            # delta is rescaled by mult. Grad pre-scaling is deliberately
            # avoided: SmaulOpt's u = m_hat/(v_hat+eps) is ~invariant to
            # uniform grad rescaling, and proj.weight's factored v_col is
            # shared across old+new rows while the global grad-norm clip
            # spans scaled grads -- so grad-scale + delta-rescale overshoots
            # mult by ~2x. Post-step-only is exact by construction:
            # delta_final = mult x full-delta on every touched old row.
            # Safety ceiling: after the mult rescale, old rows are further
            # capped so old/new movement stays <= _NEW_CAP (0.015, under the
            # 0.02 bar with headroom for bf16 storage rounding) on every
            # step. This handles shrinking-new-row inflation: as newborn
            # rows converge their updates shrink while slowed old rows keep
            # larger full-memory grads, so a fixed mult alone would let
            # old/new drift past the bar over steps. The cap binds only
            # when needed (fresh steps sit at ~mult << cap, so the
            # counterfactual own-full-step ratio stays ~= mult); when there
            # are no new rows (or they did not move) the ratio is undefined
            # and the cap is skipped, keeping exact-mult semantics.
            _NEW_CAP = 0.015
            _held: list[tuple] = []
            _held_mult = 1.0
            if mode == "new":
                mult = float(getattr(cfg, "router_lr_mult_new", 0.005))
                if mult != 1.0:
                    pool = getattr(model, "pool", None)
                    order = list(getattr(pool, "order", None) or [])
                    experts = getattr(pool, "experts", {}) or {}
                    old_idx = [i for i, eid in enumerate(order)
                               if eid in experts and experts[eid].birth_step < new_since_step]
                    if old_idx:
                        _held_mult = mult
                        with torch.no_grad():
                            for rname, rp in model._router_params():
                                is_row = (("proj.weight" in rname
                                           and rp.data.ndim == 2)
                                          or ("proj.bias" in rname
                                              and rp.data.ndim == 1))
                                if not is_row:
                                    continue
                                idx = [i for i in old_idx if i < rp.data.shape[0]]
                                if not idx:
                                    continue
                                iset = set(idx)
                                nidx = [i for i in range(rp.data.shape[0]) if i not in iset]
                                _held.append((rp, rp.data[idx].clone(), list(idx),
                                              rp.data[nidx].clone() if nidx else None, nidx))
            opt.step_router(model._router_params(), cfg.router_lr * lr_scale)
            if _held:
                with torch.no_grad():
                    for rp, snap_old, idx, snap_new, nidx in _held:
                        snap32 = snap_old.float()
                        total = (rp.data[idx].float() - snap32) * _held_mult
                        if snap_new is not None and nidx:
                            new_delta = rp.data[nidx].float() - snap_new.float()
                            f = _new_cap_factor(total, new_delta, _NEW_CAP)
                            if f != 1.0:
                                total = total * f
                        rp.data[idx] = (snap32 + total).to(rp.data.dtype)
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
            _ver_before = int(getattr(rec, "version", 0))
            opt.step_expert(rec, compute, 1.0, cfg.expert_lr * lr_scale)
            # Applied iff the version bumped: skipped updates (non-finite
            # grads, no grads) bump nothing and must not pose as stepped nor
            # evict good cache entries. Version-delta is unambiguous where
            # the EMA activity signal is not (EMA can be exactly 0.0).
            if int(getattr(rec, "version", 0)) != _ver_before:
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
        # True bits-per-byte: NLL (nats) / ln 2. Regularizers (ponder KL,
        # MoE balance, lookahead, boundary) shape training but are not
        # compression, so the optimized total loss must never pose as BPB.
        "bpb": float(out["nll"].item()) / math.log(2),
        "ponder_kl": float(out["ponder_kl"].item()),
        "acc": float(out["acc"]),
        "mean_depth": float(out["mean_depth"]),
        "stepped_experts": stepped,
        **({
            **model._profile_times,
            "forward_s": forward_s,
            "backward_s": backward_s,
            "optimizer_s": time.perf_counter() - optimizer_started,
        } if profile else {}),
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
    profile: bool = False,
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
    Checkpointing: with ``ckpt_dir`` set, ``save_every > 0`` writes a
    checkpoint every ``save_every`` steps and once more at the end when the
    final step missed that cadence. ``save_every == 0`` (or ``ckpt_dir=None``)
    leaves all saving to the caller. Ctrl+C never kills the run mid-step:
    the first SIGINT only *requests* a stop, so the in-flight step finishes,
    the checkpoint is written right after it, and the function returns with
    ``interrupted: True``; a second SIGINT aborts immediately.

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
    saved_step: int | None = None  # last step written to ckpt_dir (dedupe)
    events: list[dict] = []  # machine-readable growth ledger

    # Ctrl+C: the first press only *requests* a stop, so the in-flight step
    # finishes; the loop then checkpoints and returns (a second press aborts
    # immediately). The handler exists only around the loop and is always
    # restored, so no exit path leaks it into the caller (e.g. a test runner).
    stop_requested = False
    sigint_installed = False
    prev_sigint = None

    def _restore_sigint() -> None:
        nonlocal sigint_installed
        if sigint_installed:
            sigint_installed = False
            try:
                signal.signal(signal.SIGINT, prev_sigint)
            except (ValueError, OSError, AttributeError):
                pass

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            _restore_sigint()
            raise KeyboardInterrupt("second Ctrl+C: aborting now")
        stop_requested = True
        log_fn("[train] Ctrl+C: finishing the current step, then saving and "
               "exiting (press Ctrl+C again to abort immediately)")

    try:
        prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)
        sigint_installed = True
    except (ValueError, OSError, AttributeError):
        pass  # not the main thread: signals stay fatal, nothing to install
    try:
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
                               selected=selected, new_since_step=new_since_step,
                               profile=profile)
            stats["step"] = step
            # Throughput counts valid data bytes only: padded positions are
            # structural (PAD_ID, masked from the loss), not processed data.
            step_bytes = int((b[:, : cfg.context_length] != PAD_ID).sum().item())
            bytes_processed += step_bytes
            hist.append(stats)
            # Growth only: training never prunes (pool shrinkage is the manual
            # offline pruner). Successful growths increment the counter.
            new_ids: list[str] = []
            growth_started = time.perf_counter() if profile else 0.0
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
            if profile:
                stats["growth_s"] = time.perf_counter() - growth_started
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
                saved_step = step

            elapsed = time.perf_counter() - step_started
            total_elapsed = time.perf_counter() - run_started
            completed = local_step + 1
            avg_step = total_elapsed / completed
            eta = max(0.0, (steps - completed) * avg_step)
            step_bps = step_bytes / max(elapsed, 1e-12)
            avg_bps = bytes_processed / max(total_elapsed, 1e-12)
            # Compact progress display: a visual bar plus the metrics most useful
            # during a small CPU run. Keep the full metric history in the result.
            fraction = min(1.0, completed / max(steps, 1))
            bar_width = 18
            filled = int(fraction * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            eta_seconds = int(round(eta))
            eta_label = f"{eta_seconds // 60:02d}:{eta_seconds % 60:02d}"
            line = (
                f"[train] [{bar}] {fraction:5.1%} {completed:>4}/{steps} "
                f"| loss {stats['loss']:.4f} | bpb {stats['bpb']:.3f} "
                f"| acc {stats['acc']:.2%} | {step_bps:,.0f} B/s "
                f"| ETA {eta_label} | experts {len(model.pool)}"
            )
            if profile:
                line += (
                    f" | fwd {stats['forward_s']:.3f}s"
                    f" route {stats['router_s']:.3f}s"
                    f" expert {stats['expert_s']:.3f}s"
                    f" balance {stats['balance_s']:.3f}s"
                    f" bwd {stats['backward_s']:.3f}s"
                    f" opt {stats['optimizer_s']:.3f}s"
                    f" grow {stats['growth_s']:.3f}s"
                )
                rss_kib = avail_kib = None
                try:
                    with open("/proc/self/status", encoding="utf-8") as f:
                        for row in f:
                            if row.startswith("VmRSS:"):
                                rss_kib = int(row.split()[1])
                                break
                    with open("/proc/meminfo", encoding="utf-8") as f:
                        for row in f:
                            if row.startswith("MemAvailable:"):
                                avail_kib = int(row.split()[1])
                                break
                except (OSError, ValueError, IndexError):
                    pass
                if rss_kib is not None:
                    line += f" | RSS {rss_kib / 1024:.0f} MiB"
                if avail_kib is not None:
                    line += f" | MemAvailable {avail_kib / 1024:.0f} MiB"
            log_fn(line)
            if stop_requested:
                # First Ctrl+C landed during this step: it is now complete
                # (and grown/logged), so checkpoint right here and return.
                if ckpt_dir and saved_step != step:
                    save_model(ckpt_dir, model, opt, step,
                               extra_meta={"scheduler": scheduler_snapshot()})
                    model._resume_step = step
                    log_fn(f"[train] checkpoint saved at step {step} -> {ckpt_dir}")
                break
    finally:
        _restore_sigint()

    # End-of-run checkpoint when the final step missed the periodic cadence
    # (e.g. save_every=100 on a 1050-step run) or the run was interrupted.
    if ckpt_dir and hist and saved_step != hist[-1]["step"]:
        save_model(ckpt_dir, model, opt, int(hist[-1]["step"]),
                   extra_meta={"scheduler": scheduler_snapshot()})
        model._resume_step = int(hist[-1]["step"])
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
        "interrupted": stop_requested,
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
