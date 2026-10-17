"""Training loop with canonical growth/checkpoint ordering.

After each successful optimizer update, in order:
 1. increment global step; 2. update smoothed loss; 3. evaluate triggers;
 4. create one pending event on fire (dedup 'both'); 5. resolve ranking
    (NO optimizer update while an event is pending); 6. prepare; 7. build
    candidate in isolation; 8. validate; 9. publish committed checkpoint;
  10. adopt candidate; 11. next update.
"""
import os
import time
import math
import torch

from . import checkpoint as _ckpt
from . import config as _cfg
from . import growth as _growth
from . import model as _model
from . import paging as _paging
from . import smaulopt as _opt


def _expert_dict(params: dict, eid: int) -> dict:
    return {"Win": params[f"expert.{eid}.Win"], "Wout": params[f"expert.{eid}.Wout"]}


class Trainer:
    def __init__(self, spec, args, stream, eval_stream=None):
        self.spec = spec
        self.args = args
        self.stream = stream
        self.eval_stream = eval_stream
        self.rng = torch.Generator(device="cpu").manual_seed(int(args.seed))
        n_exp = spec.n_experts_init if spec.kind == "moe" else 0
        params = _model.build_params(spec, self.rng, n_exp, device=getattr(args, "device", "cpu"))
        self.model = _model.Model(spec, params, forward_quant=args.forward_quant,
                                  backward_mode=args.backward_mode, paging_mode=args.paging,
                                  paging_store=None, capacity_factor=args.expert_capacity_factor,
                                  device=getattr(args, "device", "cpu"))
        if spec.kind == "moe":
            self.model.store = self._make_store(args.paging, n_exp, self.model.p, None)
        self.opt = _opt.SmaulOPT(lr=args.lr, beta1=args.beta1, beta2=args.beta2,
                                 eps=args.eps, weight_decay=args.weight_decay)
        self.opt.register(self.model.p)
        # Dynamic growth is an MoE-only feature. Disable its triggers for
        # dense models instead of repeatedly reporting a skipped event once a
        # threshold has been reached.
        self.sched = _growth.GrowthScheduler(
            every_steps=args.growth_every_steps if spec.kind == "moe" else 0,
            loss_nats=args.growth_loss_nats if spec.kind == "moe" else None,
            loss_beta=args.growth_loss_beta, min_eval_samples=args.growth_min_eval_samples)
        self.log = []
        self.last_metrics = None
        self.stop_requested = False
        self.last_checkpoint_step = None

    def request_stop(self):
        """Request a graceful stop after the current optimizer update."""
        self.stop_requested = True

    def _live_experts_dir(self) -> str:
        # Live per-expert disk mirrors are transient and must not live inside
        # the published checkpoint tree, which _publish_dir replaces wholesale.
        return os.path.normpath(self.args.checkpoint) + ".experts_live"

    def _make_store(self, mode, n_exp, params, disk_dir):
        if mode in ("R", "V", "R2V"):
            ram = {e: _expert_dict(params, e) for e in range(n_exp)}
            return _paging.ExpertStore(mode, n_exp, ram_weights=ram)
        dd = disk_dir or self._live_experts_dir()
        os.makedirs(dd, exist_ok=True)
        for e in range(n_exp):
            w = _expert_dict(params, e)
            torch.save({k:v.detach().cpu() for k,v in w.items()}, os.path.join(dd, f"expert_{e}.pt"))
        return _paging.ExpertStore(mode, n_exp, disk_dir=dd,
                                   ram_cache_max_experts=self.args.paging_cache_experts,
                                   staging_max_bytes=self.args.paging_staging_bytes)

    # -- checkpoint payload --
    def _payload(self):
        spec = self.spec
        n_exp = len(_model.expert_ids(self.model.p)) if spec.kind == "moe" else 0
        arch = _cfg.arch_report(spec, n_exp)
        extra = {"forward_quant": self.model.forward_quant,
                 "backward_mode": self.model.backward_mode,
                 "paging": self.args.paging, "seed": self.args.seed,
                 "lr": self.args.lr, "seq_len": self.args.seq_len,
                 "capacity_factor": self.model.capacity_factor}
        man = _ckpt.build_manifest(
            spec.name, arch, extra, self.sched.to_record(),
            self.stream.cursor, {"torch_state": self.rng.get_state().tolist()},
            n_exp, list(range(n_exp)),
            {"class": "SmaulOPT", "lr": self.opt.lr, "beta1": self.opt.beta1,
             "beta2": self.opt.beta2, "eps": self.opt.eps,
             "weight_decay": self.opt.weight_decay})
        man["data_stream"] = dict(self.stream.identity)
        # Checkpoint writing is synchronous, so the serializer can read live
        # arrays directly. Avoid duplicating the entire parameter dict and the
        # optimizer's potentially multi-gigabyte moment tensors before writing.
        opt_state = {k: {sk: sv for sk, sv in state.items() if sk != "factored"}
                     for k, state in self.opt.state.items()}
        man["optimizer"]["factored"] = {k: bool(state["factored"]) for k,state in self.opt.state.items()}
        return man, self.model.p, opt_state

    def save(self, pending_phase=False):
        man, params, opt_state = self._payload()
        _ckpt.save_checkpoint(self.args.checkpoint, man, params, opt_state)
        self.last_checkpoint_step = int(man["scheduler"]["global_step"])
        return man

    def load(self, ckpt_dir):
        blob = _ckpt.load_checkpoint(ckpt_dir)
        man = blob["manifest"]
        self.last_checkpoint_step = int(man.get("scheduler", {}).get("global_step", 0))
        if man["preset"] != self.spec.name:
            raise ValueError(f"checkpoint preset {man['preset']} != {self.spec.name}")
        self.model.p = {k: v for k, v in blob["params"].items()}
        self.model.packed.invalidate()
        self.model.fp8.invalidate()
        flags = man.get("optimizer", {}).get("factored", {})
        self.opt.load_state_dict(blob["opt_state"], flags)
        for state in self.opt.state.values():
            for key,value in list(state.items()):
                if isinstance(value, torch.Tensor): state[key] = value.to(self.model.device)
        self.opt.register(self.model.p)
        self.sched = _growth.GrowthScheduler.from_record(man["scheduler"])
        if self.spec.kind == "dense":
            if self.sched.pending is not None:
                raise ValueError("dense checkpoint cannot contain a pending MoE growth event")
            self.sched.every_steps = 0
            self.sched.next_periodic_step = None
            self.sched.loss_threshold = None
        rng_state = man.get("rng_state", {})
        if "torch_state" in rng_state:
            self.rng.set_state(torch.tensor(rng_state["torch_state"], dtype=torch.uint8))
        # Legacy NumPy RNG state is intentionally not replayed; resumed model and optimizer
        # tensors are exact, while newly introduced random experts use their recorded seed.
        self.stream.load_state_dict({"cursor": man["data_cursor"], "path": self.stream.path,
                                     "size": self.stream.size,
                                     "identity": man.get("data_stream")})
        # rebuild paging store
        if self.spec.kind == "moe":
            n_exp = man["n_experts_current"]
            if self.args.paging in ("R", "V", "R2V"):
                ram = {e: _expert_dict(self.model.p, e) for e in range(n_exp)}
                self.model.store = _paging.ExpertStore(self.args.paging, n_exp, ram_weights=ram)
            else:
                dd = self._live_experts_dir()
                os.makedirs(dd, exist_ok=True)
                for e in range(n_exp):
                    torch.save({k:v.detach().cpu() for k,v in _expert_dict(self.model.p, e).items()},
                               os.path.join(dd, f"expert_{e}.pt"))
                self.model.store = _paging.ExpertStore(
                    self.args.paging, n_exp, disk_dir=dd,
                    ram_cache_max_experts=self.args.paging_cache_experts,
                    staging_max_bytes=self.args.paging_staging_bytes)
        return man

    # -- growth resolution (no optimizer step may intervene) --
    def _eval_batches(self, n_batches=4):
        src = self.eval_stream
        if src is not None:
            return [src.next_tokens(self.args.seq_len) for _ in range(n_batches)]
        # No dedicated eval stream: replay from the training cursor so evidence
        # collection does not silently skip training tokens.
        src = self.stream
        cursor = src.cursor
        try:
            return [src.next_tokens(self.args.seq_len) for _ in range(n_batches)]
        finally:
            src.cursor = cursor

    def _resolve_pending(self, fail_at=None):
        """Complete self.sched.pending through commit. Returns history entry."""
        from . import growth as _g
        sched, spec = self.sched, self.spec
        pend = sched.pending
        assert pend is not None, "no pending growth event"
        if spec.kind != "moe":
            raise ValueError("MoE growth is disabled for dense models")
        n_add = _growth._add_count_for(spec)
        n_exact = 2 if spec.name == "moe-full" else 0
        eids = _model.expert_ids(self.model.p)
        new_ids = list(range(len(eids), len(eids) + n_add))
        # ranking
        if pend.get("sources") is None:
            batches = self._eval_batches()
            # Growth ranking is a rare but expensive operation: score every
            # expert on one representative batch instead of forwarding all
            # experts over every evidence batch. The full sample count still
            # gates eligibility; the source ordering is explicitly approximate.
            score_samples = max(len(batches[0]) - 1, 0) if batches else 0
            scores, nsamp = _growth.rank_experts(
                self.model, batches, sched.min_eval_samples, max_batches=1)
            if scores is None:
                pend["status"] = "WAITING_FOR_RANKING"
                pend["ranking"] = {"samples": nsamp, "insufficient": True}
                self.save()
                raise _NeedEvidence(f"insufficient expert-quality evidence "
                                    f"({nsamp} < {sched.min_eval_samples}); trigger retained")
            sources = _growth.prepare_sources(scores, n_add)
            if sources is None:
                pend["status"] = "WAITING_FOR_RANKING"
                pend["ranking"] = {"samples": nsamp, "insufficient": True}
                self.save()
                raise _NeedEvidence("no reliably ranked source experts; trigger retained")
            pend["ranking"] = {"scores": {str(k): v for k, v in scores.items()},
                               "samples": nsamp,
                               "score_samples": score_samples,
                               "approximate": True}
            pend["sources"] = sources
            pend["status"] = "PREPARED"
            pend["mutation"] = {"scale": self.args.growth_noise_scale,
                                "seed": int(self.args.growth_seed if self.args.growth_seed is not None
                                            else self.args.seed + pend["id"])}
            self.save()  # Phase A: pending checkpoint (pre-growth model)
            if fail_at in ("before_pending", "during_pending", "after_pending"):
                raise _Injected(fail_at)
        if fail_at == "before_candidate":
            raise _Injected(fail_at)
        # candidate built in isolation; active model untouched until commit
        cand, detail = _growth.grow_candidate(
            self.model.p, self.model.p["router"], pend["sources"], new_ids,
            pend["mutation"]["scale"], pend["mutation"]["seed"], n_exact,
            fail_at=fail_at)
        if fail_at == "after_validate":
            raise _Injected(fail_at)
        _growth.validate_candidate(spec, cand, None)
        pend["status"] = "VALIDATED"
        # smoke validation on an isolated model object (active model untouched)
        probe = _model.Model(spec, {k: v for k, v in cand.items()},
                             forward_quant=self.model.forward_quant,
                             backward_mode=self.model.backward_mode,
                             paging_mode="R", paging_store=None,
                             capacity_factor=self.model.capacity_factor)
        toks = self._eval_batches(n_batches=1)[0]
        logits, _ = probe.forward(toks, train=False)
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("candidate validation failed: non-finite logits")
        _loss, _grads, _ = probe.loss_and_backward(toks)
        if fail_at == "during_optinit":
            raise _Injected(fail_at)
        # commit: scheduler transition + atomic publish, then adopt
        t0 = time.perf_counter()
        entry = {"id": pend["id"], "step": pend["step"], "reason": pend["reason"],
                 "prev_count": len(eids), "new_count": len(eids) + n_add,
                 "sources": pend["sources"], "new_ids": new_ids,
                 "scores": pend["ranking"]["scores"],
                 "mutation": dict(pend["mutation"], n_exact_unperturbed=n_exact),
                 "detail": detail, "status": "committed"}
        sched.commit_event(entry)
        if fail_at == "during_commit":
            raise _Injected(fail_at)
        self._resize_router_opt_state(cand)  # part of the committed transition
        self._save_committed(cand, entry, new_ids)  # Phase B publish
        entry["duration_s"] = time.perf_counter() - t0
        if fail_at == "after_commit":
            raise _Injected(fail_at)
        # only now adopt the candidate for continued execution
        self._adopt_candidate(cand, new_ids)
        self.log.append({"event": "growth", **{k: entry[k] for k in
                         ("id", "step", "reason", "prev_count", "new_count", "sources")}})
        return entry

    def _save_committed(self, cand, entry, new_ids):
        """Phase B: atomically publish grown model + updated scheduler (no adopt yet)."""
        model, sched, opt = self.model, self.sched, self.opt
        n_exp = len(_model.expert_ids(cand))
        arch = _cfg.arch_report(self.spec, n_exp)
        extra = {"forward_quant": model.forward_quant, "backward_mode": model.backward_mode,
                 "paging": self.args.paging, "seed": self.args.seed, "lr": self.args.lr,
                 "seq_len": self.args.seq_len, "capacity_factor": model.capacity_factor}
        man = _ckpt.build_manifest(
            self.spec.name, arch, extra, sched.to_record(), self.stream.cursor,
            {"torch_state": self.rng.get_state().tolist()}, n_exp, list(range(n_exp)),
            {"class": "SmaulOPT", "lr": opt.lr, "beta1": opt.beta1, "beta2": opt.beta2,
             "eps": opt.eps, "weight_decay": opt.weight_decay,
             "factored": {k: bool(s["factored"]) for k, s in opt.state.items()}})
        man["data_stream"] = dict(self.stream.identity)
        # optimizer state carried by name; router state carried per-column below
        # _write_tree consumes array references synchronously; avoid a second
        # copy of every optimizer tensor at the growth commit checkpoint.
        opt_state = {k: {sk: sv for sk, sv in s.items() if not isinstance(sv, bool)}
                     for k, s in opt.state.items()}
        _ckpt.save_checkpoint(self.args.checkpoint, man, cand, opt_state)

    def _resize_router_opt_state(self, cand):
        opt = self.opt
        old = opt.state.get("router")
        new_router = cand["router"]
        if old is None:
            return
        old_cols = int(old["m"].shape[1])
        new_m = torch.zeros_like(new_router)
        new_m[:, :old_cols] = old["m"]
        new_C = torch.zeros(new_router.shape[1], dtype=new_router.dtype, device=new_router.device)
        new_C[:old_cols] = old["C"]
        opt.state["router"] = {"m": new_m, "R": old["R"].clone(), "C": new_C, "factored": True}

    def _adopt_candidate(self, cand, new_ids):
        model, opt = self.model, self.opt
        model.p = cand
        opt.register(cand)  # zero-init state for the new experts
        if model.store is not None:
            for nid in new_ids:
                model.store.register_expert(nid, _expert_dict(cand, nid))
            model.store.n_experts = len(_model.expert_ids(cand))
            model.store.invalidate_all()
        model.packed.invalidate()
        model.fp8.invalidate()
        model._assert_counts()

    def train(self, max_steps=None, log_every=10, eval_every=0, ckpt_every=0,
              grad_clip=1.0, fail_at=None, growth_fail_at=None):
        steps = 0
        last_saved_step = None
        while True:
            # A SIGINT between iterations should not make us start another
            # optimizer update. For a stop requested before the first call to
            # train(), allow one update to finish, matching request_stop's API.
            if self.stop_requested and steps > 0:
                # The final-save block below writes any pending transaction
                # once, without duplicating a potentially large checkpoint.
                break
            # Recovery transitions are not optimizer updates. Resolve a saved
            # pending growth event even when the requested global-step target
            # has already been reached; otherwise it can remain stuck forever
            # on repeated runs with the same --max-steps value.
            if self.sched.pending is not None:
                self._resolve_pending(fail_at=growth_fail_at)
                continue
            if max_steps is not None and self.sched.global_step >= max_steps:
                break
            toks = self.stream.next_tokens(self.args.seq_len)
            loss, grads, info = self.model.loss_and_backward(toks)
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {self.sched.global_step}")
            # gradient clipping (global norm)
            # Reduce each tensor to a scalar instead of materializing a full
            # float64 copy of every gradient (catastrophic for dense-full).
            total = 0.0
            for grad in grads.values():
                total = math.hypot(total, float(torch.linalg.vector_norm(grad).item()))
            if not math.isfinite(total):
                raise RuntimeError(f"non-finite gradient norm at step {self.sched.global_step}")
            if total > grad_clip and total > 0:
                scale = grad_clip / total
                # Gradients are detached tensors owned by this step, so in-place
                # clipping avoids allocating another model-sized gradient set.
                for grad in grads.values():
                    grad.mul_(scale)
            # Training never consumes per-parameter update norms; skip their
            # extra full-array reductions unless an explicit caller requests them.
            self.opt.step(self.model.p, grads, collect_norms=False)
            # Only parameters with gradients were updated in-place. Invalidate
            # their quantized representations, not every projection in the model.
            self.model.packed.invalidate(grads.keys())
            self.model.fp8.invalidate(grads.keys())
            if self.model.store is not None and self.spec.kind == "moe":
                changed_experts = set()
                for name in grads:
                    parts = name.split(".")
                    if len(parts) == 3 and parts[0] == "expert" and parts[2] in ("Win", "Wout"):
                        changed_experts.add(int(parts[1]))
                for eid in changed_experts:
                    self.model.store.write_back(eid, _expert_dict(self.model.p, eid))
            # 1-4: step counter, smoothing, triggers, single pending creation
            reason = self.sched.note_optimizer_step(loss)
            self.last_metrics = {"step": self.sched.global_step, "loss": float(loss),
                                 "smooth_loss": self.sched.smooth_loss,
                                 "overflow_dropped": int(info.get("overflow_dropped", 0))}
            if reason is not None and self.sched.pending is None:
                if self.spec.kind != "moe":
                    self.log.append({"event": "growth-skipped-dense",
                                     "step": self.sched.global_step})
                else:
                    self.sched.create_pending(reason)
            # resolve immediately (same step): ranking -> prepare -> commit
            if self.sched.pending is not None:
                try:
                    self._resolve_pending(fail_at=growth_fail_at)
                except _NeedEvidence as e:
                    # training updates are paused while evidence is collected
                    raise RuntimeError(f"growth blocked at step {self.sched.global_step}: {e}")
            steps += 1
            if log_every and steps % log_every == 0:
                self.log.append({"step": self.sched.global_step, "loss": loss,
                                 "smooth": self.sched.smooth_loss,
                                 "overflow_dropped": int(info.get("overflow_dropped", 0))})
            if ckpt_every and self.sched.global_step % ckpt_every == 0:
                self.save()
                last_saved_step = self.sched.global_step
            if self.stop_requested:
                # A SIGINT handler only sets this flag. Finish the optimizer
                # update and any growth transaction before publishing state.
                if last_saved_step != self.sched.global_step:
                    self.save()
                    last_saved_step = self.sched.global_step
                break
            if fail_at == "after_step":
                raise _Injected(fail_at)
        if (last_saved_step != self.sched.global_step
                and (ckpt_every or self.stop_requested)):
            # Save a non-periodic final step and ensure a stop request that
            # arrives at the loop boundary still leaves a recoverable state.
            # A zero-step run with periodic saves enabled also gets a checkpoint.
            self.save()
        return self.log


class _NeedEvidence(RuntimeError):
    pass


class _Injected(RuntimeError):
    pass
