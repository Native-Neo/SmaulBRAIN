"""Automatic dynamic expert growth (canonical two-phase protocol).

Triggers (logical OR, both enabled by default):
  periodic: every --growth-every-steps completed optimizer steps (0 disables).
  loss:     smoothed CE (nats/byte) crosses from above to at/below
            --growth-loss-nats ('none' disables). Edge-triggered with rearm.

Event lifecycle: NO_EVENT -> WAITING_FOR_RANKING -> PREPARED -> VALIDATED ->
COMMITTED. Committed only when the checkpoint with the grown model +
scheduler state is atomically published. (Phase A pending checkpoint, then
Phase B committed checkpoint.)

Quality metric (source selection AND prune ranking): leave-one-out held-out
loss contribution: score(e) = loss_masked(e) - loss_baseline over eval
tokens (>= min_eval_samples, else WAITING_FOR_RANKING/insufficient-evidence).
Deterministic tie-break by expert id. Usage counts are reported but never
the ranking key.
"""
import time

import numpy as np

from . import config as _cfg

GROWTH_FULL_ADD = 8
GROWTH_TINY_ADD = 2


class GrowthScheduler:
    """Pure trigger bookkeeping; state mirrors the checkpoint scheduler record."""

    def __init__(self, every_steps=10000, loss_nats=0.5, loss_beta=0.98,
                 min_eval_samples=256):
        self.every_steps = int(every_steps)
        self.loss_threshold = loss_nats  # None disables
        self.loss_beta = float(loss_beta)
        self.min_eval_samples = int(min_eval_samples)
        self.global_step = 0
        self.smooth_loss = None
        self.loss_armed = True
        self.next_periodic_step = every_steps if every_steps and every_steps > 0 else None
        self.event_count = 0
        self.last_growth_step = None
        self.history = []
        self.pending = None

    # -- persistence --
    def to_record(self) -> dict:
        return {
            "global_step": self.global_step,
            "growth_every_steps": self.every_steps,
            "next_periodic_step": self.next_periodic_step,
            "loss_threshold": self.loss_threshold,
            "loss_beta": self.loss_beta,
            "smooth_loss": self.smooth_loss,
            "loss_armed": self.loss_armed,
            "event_count": self.event_count,
            "last_growth_step": self.last_growth_step,
            "history": [dict(h) for h in self.history],
            "pending": dict(self.pending) if self.pending else None,
        }

    @classmethod
    def from_record(cls, rec: dict):
        s = cls(every_steps=rec.get("growth_every_steps", 10000),
                loss_nats=rec.get("loss_threshold", 0.5),
                loss_beta=rec.get("loss_beta", 0.98))
        s.global_step = int(rec.get("global_step", 0))
        s.smooth_loss = rec.get("smooth_loss")
        s.loss_armed = bool(rec.get("loss_armed", True))
        s.next_periodic_step = rec.get("next_periodic_step")
        s.event_count = int(rec.get("event_count", 0))
        s.last_growth_step = rec.get("last_growth_step")
        s.history = [dict(h) for h in rec.get("history", [])]
        s.pending = dict(rec["pending"]) if rec.get("pending") else None
        return s

    # -- per-step trigger evaluation (call AFTER incrementing global_step) --
    def note_optimizer_step(self, loss_nats):
        """Update smoothing + evaluate triggers. Returns reason or None."""
        self.global_step += 1
        if loss_nats is not None and np.isfinite(loss_nats):
            b = self.loss_beta
            self.smooth_loss = (float(loss_nats) if self.smooth_loss is None
                                else b * self.smooth_loss + (1 - b) * float(loss_nats))
        step_due = (self.every_steps > 0 and self.next_periodic_step is not None
                    and self.global_step >= self.next_periodic_step)
        loss_due = False
        if (self.loss_threshold is not None and self.loss_armed
                and self.smooth_loss is not None and np.isfinite(self.smooth_loss)):
            if self.smooth_loss <= float(self.loss_threshold):
                loss_due = True
        # rearm: loss strictly above threshold re-arms for a future crossing
        if (self.loss_threshold is not None and not self.loss_armed
                and self.smooth_loss is not None and np.isfinite(self.smooth_loss)
                and self.smooth_loss > float(self.loss_threshold)):
            self.loss_armed = True
        if step_due and loss_due:
            return "both"
        if step_due:
            return "step"
        if loss_due:
            return "loss"
        return None

    def create_pending(self, reason: str, smooth_snapshot=None) -> dict:
        eid = (self.history[-1]["id"] + 1) if self.history else 1
        if self.pending is not None:
            raise RuntimeError("growth event already pending")
        self.pending = {
            "id": eid,
            "step": self.global_step,
            "reason": reason,
            "loss_value": self.smooth_loss if smooth_snapshot is None else smooth_snapshot,
            "interval": self.every_steps,
            "threshold": self.loss_threshold,
            "armed_before": self.loss_armed,
            "ranking": None,
            "sources": None,
            "mutation": None,
            "status": "WAITING_FOR_RANKING",
        }
        return self.pending

    def commit_event(self, entry: dict) -> None:
        if any(h["id"] == entry["id"] for h in self.history):
            raise ValueError(f"duplicate committed event id {entry['id']}")
        self.history.append(entry)
        self.event_count += 1
        self.last_growth_step = entry["step"]
        self.pending = None
        # advance periodic schedule past the event step (exactly-once advancement)
        if self.every_steps > 0:
            base = entry["step"]
            k = base // self.every_steps + 1
            self.next_periodic_step = k * self.every_steps
        # disarm loss trigger only if it participated
        if entry["reason"] in ("loss", "both"):
            self.loss_armed = False


# ---------------- quality ranking ----------------

def eval_loss_on_tokens(model, tokens) -> float:
    logits, _ = model.forward(tokens, train=False)
    inp = np.asarray(tokens[:-1])
    tgt = np.asarray(tokens[1:])
    m = logits[:-1].max(axis=1, keepdims=True)
    logsum = m + np.log(np.exp(logits[:-1] - m).sum(axis=1, keepdims=True))
    nll = logsum.squeeze(1) - logits[:-1][np.arange(len(tgt)), tgt]
    return float(nll.mean())


def rank_experts(model, eval_batches, min_samples: int):
    """Returns (scores dict, total_samples) or (None, total_samples) if insufficient."""
    total = sum(max(len(b) - 1, 0) for b in eval_batches)
    if total < min_samples:
        return None, total
    base_losses = [eval_loss_on_tokens(model, b) for b in eval_batches]
    from . import model as _m
    eids = _m.expert_ids(model.p)
    scores = {}
    for eid in eids:
        masked = []
        for b in eval_batches:
            logits, _ = model.forward(b, train=False, disabled=(eid,))
            tgt = np.asarray(b[1:])
            lg = logits[:-1]
            m = lg.max(axis=1, keepdims=True)
            ls = m + np.log(np.exp(lg - m).sum(axis=1, keepdims=True))
            masked.append(float((ls.squeeze(1) - lg[np.arange(len(tgt)), tgt]).mean()))
        scores[eid] = float(np.mean(masked) - np.mean(base_losses))
    return scores, total


# ---------------- growth transaction ----------------

def _add_count_for(spec) -> int:
    return GROWTH_FULL_ADD if spec.name == "moe-full" else GROWTH_TINY_ADD


def prepare_sources(scores: dict, n_needed: int):
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(order) < n_needed:
        return None
    # require finite scores; unreliable (NaN) sources block the event
    if any(not np.isfinite(s) for _, s in order[:n_needed]):
        return None
    return [eid for eid, _ in order[:n_needed]]


def clone_with_mutation(params: dict, sources: list, new_ids: list,
                        noise_scale: float, seed: int, n_exact_unperturbed: int):
    """Element-wise additive Gaussian mutation. First `n_exact_unperturbed`
    clones (of the best-ranked sources) are exact copies."""
    rng = np.random.default_rng(seed)
    detail = []
    for rank, (src, new) in enumerate(zip(sources, new_ids)):
        for suffix in ("Win", "Wout"):
            skey, nkey = f"expert.{src}.{suffix}", f"expert.{new}.{suffix}"
            W = params[skey]
            if rank < n_exact_unperturbed:
                params[nkey] = W.copy()
                detail.append({"expert": new, "source": src, "tensor": suffix,
                               "mutated": False})
            else:
                rms = float(np.sqrt(np.mean(W.astype(np.float64) ** 2)))
                sigma = noise_scale * (rms if rms > 0 else 1.0)
                eps = rng.standard_normal(W.shape).astype(np.float64) * sigma
                params[nkey] = (W.astype(np.float64) + eps).astype(np.float32)
                detail.append({"expert": new, "source": src, "tensor": suffix,
                               "mutated": True, "sigma": sigma})
    return detail


def grow_candidate(params: dict, router_W: np.ndarray, sources: list, new_ids: list,
                   noise_scale: float, seed: int, n_exact_unperturbed: int,
                   fail_at=None):
    """Build grown params + grown router in isolated candidate dicts."""
    if fail_at == "during_clone":
        raise RuntimeError("injected failure during_clone")
    cand = {k: v.copy() for k, v in params.items()}
    if fail_at == "during_mutate":
        raise RuntimeError("injected failure during_mutate")
    detail = clone_with_mutation(cand, sources, new_ids, noise_scale, seed,
                                 n_exact_unperturbed)
    rng = np.random.default_rng(seed + 999)
    old_e = router_W.shape[1]
    new_e = old_e + len(new_ids)
    new_router = np.empty((router_W.shape[0], new_e), dtype=np.float32)
    new_router[:, :old_e] = router_W
    for i, src in enumerate(sources):
        col = router_W[:, src].astype(np.float64)
        col_rms = float(np.sqrt(np.mean(col ** 2)))
        pert = rng.standard_normal(col.shape) * (0.01 * (col_rms or 1.0))
        new_router[:, old_e + i] = (col + pert).astype(np.float32)
    cand["router"] = new_router
    return cand, detail


def validate_candidate(spec, cand_params, opt_names) -> None:
    from . import model as _m
    eids = sorted(int(k.split(".")[1]) for k in cand_params if k.endswith(".Win"))
    if eids != list(range(len(eids))):
        raise ValueError("candidate expert registry not dense")
    for eid in eids:
        n = int(cand_params[f"expert.{eid}.Win"].size + cand_params[f"expert.{eid}.Wout"].size)
        if n != _cfg.EXPECTED_EXPERT_PARAMS[spec.name]:
            raise ValueError(f"candidate expert {eid} has {n} params")
    r = cand_params["router"]
    if r.shape[1] != len(eids):
        raise ValueError("candidate router width disagrees with registry")
    if not all(np.all(np.isfinite(cand_params[k])) for k in cand_params):
        raise ValueError("candidate contains non-finite values")
