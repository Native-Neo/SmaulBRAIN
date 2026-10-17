"""Evidence-based expert pruning with atomic checkpoint replacement."""
import torch

from . import checkpoint as _ckpt
from . import growth as _growth
from . import model as _model


def rank_for_prune(params: dict, scores: dict, n_remove: int, top_k: int, n_experts: int):
    if n_remove <= 0:
        raise ValueError("--rm-worst must be positive")
    if n_remove >= n_experts:
        raise ValueError(f"cannot remove {n_remove} of {n_experts} experts "
                         f"(routing requires top_k={top_k} <= remaining)")
    if n_experts - n_remove < top_k:
        raise ValueError(f"removal would leave {n_experts - n_remove} experts, "
                         f"violating top_k={top_k}")
    order = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    return [eid for eid, _ in order[:n_remove]]


def prune_checkpoint(ckpt_dir: str, rm_worst: int, eval_batches, min_samples: int,
                     dry_run=False, assume_yes=False):
    blob = _ckpt.load_checkpoint(ckpt_dir)
    man = blob["manifest"]
    if not man["preset"].startswith("moe"):
        raise ValueError("pruning applies to MoE checkpoints only")
    from . import config as _cfg
    spec = _cfg.PRESETS[man["preset"]]
    params = blob["params"]
    eids = _model.expert_ids(params)
    # quality evidence from current model
    config = man.get("config", {})
    forward_quant = config.get("forward_quant", "fp32")
    backward_mode = config.get("backward_mode", "fp32")
    capacity_factor = float(config.get("capacity_factor", 1.25))
    m = _model.Model(
        spec,
        params,
        forward_quant=forward_quant,
        backward_mode=backward_mode,
        capacity_factor=capacity_factor,
    )
    scores, nsamp = _growth.rank_experts(m, eval_batches, min_samples)
    if scores is None:
        raise RuntimeError(f"insufficient quality evidence ({nsamp} < {min_samples}); "
                           f"pruning refused")
    victims = rank_for_prune(params, scores, rm_worst, spec.top_k, len(eids))
    report = {"candidates": victims,
              "scores": {str(v): scores[v] for v in victims},
              "samples": nsamp,
              "capacity_factor": capacity_factor,
              "forward_quant": forward_quant,
              "backward_mode": backward_mode,
              "prev_count": len(eids), "new_count": len(eids) - len(victims)}
    if dry_run:
        return report
    if not assume_yes:
        raise RuntimeError(f"refusing destructive replacement without --yes; "
                           f"dry-run report: {report}")
    keep = [e for e in eids if e not in set(victims)]
    remap = {old: new for new, old in enumerate(keep)}
    new_params = {k: v for k, v in params.items() if not k.startswith("expert.")}
    for old in keep:
        for suffix in ("Win", "Wout"):
            new_params[f"expert.{remap[old]}.{suffix}"] = params[f"expert.{old}.{suffix}"]
    # shrink router rows? router is d x E: drop pruned columns
    R = params["router"]
    new_params["router"] = R[:, keep].clone()
    # optimizer state: drop pruned experts, rebuild router entry
    opt_state = blob["opt_state"]
    new_opt = {k: v for k, v in opt_state.items()
               if not (k.startswith("expert.") and int(k.split(".")[1]) in set(victims))}
    renamed = {}
    for k, v in new_opt.items():
        if k.startswith("expert."):
            parts = k.split(".")
            renamed[f"expert.{remap[int(parts[1])]}.{'.'.join(parts[2:])}"] = v
        elif k == "router":
            renamed[k] = v  # router state resized on load/adopt; mark below
        else:
            renamed[k] = v
    man2 = dict(man)
    man2["n_experts_current"] = len(keep)
    man2["expert_registry"] = list(range(len(keep)))
    man2["arch"] = _cfg.arch_report(spec, len(keep))
    man2.setdefault("prune_history", []).append(
        {"removed": victims, "scores": {str(v): scores[v] for v in victims},
         "prev_count": len(eids), "new_count": len(keep)})
    # router optimizer state is shape-bound: drop it so it re-initializes
    renamed.pop("router", None)
    # Validate the candidate before touching the on-disk checkpoint. Saving
    # first and smoke-testing afterward can destroy the last usable checkpoint
    # if a shape/configuration problem makes inference fail.
    candidate_model = _model.Model(
        spec,
        new_params,
        forward_quant=forward_quant,
        backward_mode=backward_mode,
        capacity_factor=capacity_factor,
    )
    logits, _ = candidate_model.forward([65, 66, 67, 68], train=False)
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("pruned candidate failed validation")
    _ckpt.save_checkpoint(ckpt_dir, man2, new_params, renamed)
    # Reload the published generation and confirm serialized state is usable.
    blob2 = _ckpt.load_checkpoint(ckpt_dir)
    m2 = _model.Model(
        spec,
        blob2["params"],
        forward_quant=forward_quant,
        backward_mode=backward_mode,
        capacity_factor=capacity_factor,
    )
    logits, _ = m2.forward([65, 66, 67, 68], train=False)
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("published pruned checkpoint failed validation")
    return report
