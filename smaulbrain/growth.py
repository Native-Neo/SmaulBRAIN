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
import math
import torch

from . import config as _cfg

GROWTH_FULL_ADD = 8
GROWTH_TINY_ADD = 2

def _add_count_for(spec):
    return GROWTH_FULL_ADD if spec.name == "moe-full" else GROWTH_TINY_ADD

def _as_torch(value):
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)


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
            "min_eval_samples": self.min_eval_samples,
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
                loss_beta=rec.get("loss_beta", 0.98),
                min_eval_samples=rec.get("min_eval_samples", 256))
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
        if loss_nats is not None and math.isfinite(loss_nats):
            b = self.loss_beta
            self.smooth_loss = (float(loss_nats) if self.smooth_loss is None
                                else b * self.smooth_loss + (1 - b) * float(loss_nats))
        step_due = (self.every_steps > 0 and self.next_periodic_step is not None
                    and self.global_step >= self.next_periodic_step)
        loss_due = False
        if (self.loss_threshold is not None and self.loss_armed
                and self.smooth_loss is not None and math.isfinite(self.smooth_loss)):
            if self.smooth_loss <= float(self.loss_threshold):
                loss_due = True
        # rearm: loss strictly above threshold re-arms for a future crossing
        if (self.loss_threshold is not None and not self.loss_armed
                and self.smooth_loss is not None and math.isfinite(self.smooth_loss)
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
    tgt = torch.as_tensor(tokens[1:], dtype=torch.long, device=logits.device)
    return float(torch.nn.functional.cross_entropy(logits[:-1], tgt).item())


def rank_experts(model, eval_batches, min_samples: int, max_batches=None):
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    batches = list(eval_batches if max_batches is None else eval_batches[:max_batches])
    # Evidence must be counted only from batches that actually contribute to
    # the scores. Otherwise max_batches=1 can pass the threshold using tokens
    # from ignored batches and rank experts on too little data.
    total = sum(max(len(b) - 1, 0) for b in batches)
    if total < min_samples:
        return None, total
    if not batches:
        return None, total
    base = [eval_loss_on_tokens(model, b) for b in batches]
    from . import model as _m
    scores = {}
    for eid in _m.expert_ids(model.p):
        masked = [eval_loss_on_tokens_disabled(model, b, eid) for b in batches]
        scores[eid] = float(sum(masked) / len(masked) - sum(base) / len(base))
    return scores, total


def eval_loss_on_tokens_disabled(model, tokens, disabled):
    logits, _ = model.forward(tokens, train=False, disabled=(disabled,))
    tgt = torch.as_tensor(tokens[1:], dtype=torch.long, device=logits.device)
    return float(torch.nn.functional.cross_entropy(logits[:-1], tgt).item())


def prepare_sources(scores: dict, n_needed: int):
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(order) < n_needed or any(not math.isfinite(float(s)) for _, s in order[:n_needed]): return None
    return [eid for eid, _ in order[:n_needed]]


def clone_with_mutation(params: dict, sources: list, new_ids: list, noise_scale: float, seed: int, n_exact_unperturbed: int):
    rng = torch.Generator(device="cpu").manual_seed(int(seed)); detail = []
    for rank, (src, new) in enumerate(zip(sources, new_ids)):
        for suffix in ("Win", "Wout"):
            skey, nkey = f"expert.{src}.{suffix}", f"expert.{new}.{suffix}"; W = _as_torch(params[skey])
            if rank < n_exact_unperturbed:
                params[nkey] = W.detach().clone()
                detail.append({"expert":new,"source":src,"tensor":suffix,"mutated":False})
            else:
                rms = float(W.detach().double().square().mean().sqrt().item())
                sigma = noise_scale * (rms if rms > 0 else 1.0)
                noise = torch.randn(tuple(W.shape), generator=rng, dtype=W.dtype).to(W.device)
                params[nkey] = W.detach() + noise * sigma
                detail.append({"expert":new,"source":src,"tensor":suffix,"mutated":True,"sigma":sigma})
    return detail


def grow_candidate(params: dict, router_W, sources: list, new_ids: list, noise_scale: float, seed: int,
                   n_exact_unperturbed: int, fail_at=None):
    if fail_at == "during_clone": raise RuntimeError("injected failure during_clone")
    cand = {k:_as_torch(v).detach().clone() for k,v in params.items()}
    router_W = _as_torch(router_W)
    if fail_at == "during_mutate": raise RuntimeError("injected failure during_mutate")
    detail = clone_with_mutation(cand,sources,new_ids,noise_scale,seed,n_exact_unperturbed)
    rng = torch.Generator(device="cpu").manual_seed(int(seed)+999); old_e=router_W.shape[1]; new_e=old_e+len(new_ids)
    router=torch.empty((router_W.shape[0],new_e),dtype=router_W.dtype,device=router_W.device)
    router[:,:old_e]=router_W.detach()
    for i,src in enumerate(sources):
        col=router_W[:,src].detach(); rms=float(col.double().square().mean().sqrt().item())
        noise=torch.randn(tuple(col.shape), generator=rng, dtype=col.dtype).to(col.device)
        router[:,old_e+i]=col+noise*(0.01*(rms or 1.0))
    cand["router"]=router
    return cand,detail


def validate_candidate(spec,cand_params,opt_names) -> None:
    eids=sorted(int(k.split(".")[1]) for k in cand_params if k.endswith(".Win"))
    if eids != list(range(len(eids))): raise ValueError("candidate expert registry not dense")
    for eid in eids:
        n=cand_params[f"expert.{eid}.Win"].numel()+cand_params[f"expert.{eid}.Wout"].numel()
        if n != _cfg.EXPECTED_EXPERT_PARAMS[spec.name]: raise ValueError(f"candidate expert {eid} has {n} params")
    if cand_params["router"].shape[1] != len(eids): raise ValueError("candidate router width disagrees with registry")
    if not all(torch.isfinite(v).all().item() for v in cand_params.values()): raise ValueError("candidate contains non-finite values")
