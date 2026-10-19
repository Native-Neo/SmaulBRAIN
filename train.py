#!/usr/bin/env python3
"""SmaulBRAIN v2.1.0 training entry point."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import data as _data
from smaulbrain import train_loop as _tl


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Train SmaulBRAIN v2.1.0 (byte-level recurrent LM)",
                                 allow_abbrev=False)
    _cli.add_common_args(ap, need_train=True)
    ap.add_argument("--resume", action="store_true",
                    help="require an existing checkpoint (auto-resume is the default)")
    ap.add_argument("--steps", type=int, default=None,
                    help="number of additional optimizer updates (useful with --resume)")
    args = ap.parse_args(raw_argv)
    if args.steps is not None and args.steps < 0:
        ap.error("--steps must be >= 0")

    checkpoint_exists = (
        os.path.exists(os.path.join(args.checkpoint, "manifest.json"))
        or os.path.exists(os.path.join(os.path.normpath(args.checkpoint) + ".bak",
                                      "manifest.json"))
    )
    if args.resume and not checkpoint_exists:
        print(f"error: nothing to resume at {args.checkpoint}", file=sys.stderr)
        return 2

    checkpoint_blob = None
    if checkpoint_exists:
        from smaulbrain import checkpoint as _ckpt
        try:
            checkpoint_blob = _ckpt.load_checkpoint(args.checkpoint)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            print(f"error: unable to load checkpoint at {args.checkpoint}: {exc}",
                  file=sys.stderr)
            return 2
        manifest = checkpoint_blob["manifest"]
        if (not _was_provided(raw_argv, "--preset")
                and not args.moe and not args.dense):
            args.preset = manifest.get("preset", args.preset)
        saved = manifest.get("config", {}) or {}
        optimizer = manifest.get("optimizer", {}) or {}
        scheduler = manifest.get("scheduler", {}) or {}
        for option, attr, value in (
            ("--forward-quant", "forward_quant", saved.get("forward_quant")),
            ("--backward-mode", "backward_mode", saved.get("backward_mode")),
            ("--paging", "paging", saved.get("paging")),
            ("--seed", "seed", saved.get("seed")),
            ("--seq-len", "seq_len", saved.get("seq_len")),
            ("--expert-capacity-factor", "expert_capacity_factor",
             saved.get("capacity_factor")),
            ("--lr", "lr", optimizer.get("lr", saved.get("lr"))),
            ("--beta1", "beta1", optimizer.get("beta1")),
            ("--beta2", "beta2", optimizer.get("beta2")),
            ("--eps", "eps", optimizer.get("eps")),
            ("--weight-decay", "weight_decay", optimizer.get("weight_decay")),
        ):
            if value is not None and not _was_provided(raw_argv, option):
                setattr(args, attr, value)
        if saved.get("synth_bytes") is not None and not _was_provided(raw_argv, "--synth-bytes"):
            args.synth_bytes = int(saved["synth_bytes"])
        if (not _was_provided(raw_argv, "--corpus")
                and saved.get("data_stream", {}).get("kind") == "file"):
            args.corpus = saved["data_stream"].get("path")
        if not _was_provided(raw_argv, "--growth-every-steps"):
            args.growth_every_steps = str(scheduler.get("growth_every_steps", 10000))
        if not _was_provided(raw_argv, "--growth-loss-nats"):
            threshold = scheduler.get("loss_threshold", 0.5)
            args.growth_loss_nats = "none" if threshold is None else str(threshold)
        if not _was_provided(raw_argv, "--growth-loss-beta"):
            args.growth_loss_beta = float(scheduler.get("loss_beta", 0.98))
        if not _was_provided(raw_argv, "--growth-min-eval-samples"):
            args.growth_min_eval_samples = int(scheduler.get("min_eval_samples", 256))
        if not _was_provided(raw_argv, "--ckpt-every"):
            args.ckpt_every = 100

    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))

    spec = _cfg.PRESETS[preset]
    if checkpoint_blob is not None and checkpoint_blob["manifest"].get("preset") != preset:
        print(f"error: checkpoint preset {checkpoint_blob['manifest'].get('preset')} != {preset}",
              file=sys.stderr)
        return 2

    if preset.startswith("dense") and (args.growth_every_steps or args.growth_loss_nats is not None):
        print("note: MoE growth is disabled for dense models; growth triggers ignored")
    if args.corpus:
        stream = _data.ByteStream(path=args.corpus)
    else:
        stream = _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed))
    if checkpoint_blob is not None:
        saved_manifest = checkpoint_blob["manifest"]
        stream.load_state_dict({"cursor": saved_manifest["data_cursor"],
                                "path": stream.path,
                                "size": stream.size})

    try:
        tr = _tl.Trainer(spec, args, stream)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: unable to initialize training: {exc}", file=sys.stderr)
        return 2

    if checkpoint_blob is not None:
        man = tr.load(args.checkpoint)
        print(f"auto-resumed {preset} at global_step={man['scheduler']['global_step']} "
              f"experts={man['n_experts_current']}")
        saved_config = man.get("config", {})
        for key, current in (("forward_quant", args.forward_quant),
                             ("backward_mode", args.backward_mode),
                             ("seq_len", args.seq_len),
                             ("capacity_factor", args.expert_capacity_factor),
                             ("paging", args.paging),
                             ("seed", args.seed)):
            previous = saved_config.get(key)
            if previous is not None and previous != current:
                print(f"warning: checkpoint {key}={previous!r}, requested {current!r}; "
                      "resuming with changed settings can disrupt training", file=sys.stderr)
    else:
        if os.path.exists(os.path.join(args.checkpoint, "manifest.json")):
            print(f"error: checkpoint exists at {args.checkpoint} (use --resume)", file=sys.stderr)
            return 2

    del checkpoint_blob

    rep = tr.model.report()
    print(json.dumps({"preset": preset, "total_params": rep["total_params"],
                      "per_expert": rep.get("per_expert_params"),
                      "growth_every_steps": args.growth_every_steps,
                      "growth_loss_nats": args.growth_loss_nats}, indent=2))
    start_step = tr.sched.global_step
    target_step = start_step + args.steps if args.steps is not None else args.max_steps
    if target_step <= start_step:
        print(f"warning: no training steps will run (global_step={start_step}, "
              f"max_steps={target_step}); use --steps N for additional updates", file=sys.stderr)
    log = tr.train(max_steps=target_step, log_every=args.log_every,
                   ckpt_every=args.ckpt_every, grad_clip=args.grad_clip)
    tr.save()
    from smaulbrain import model as _model
    n_exp = len(_model.expert_ids(tr.model.p)) if preset.startswith("moe") else 0
    metrics = tr.last_metrics or {}
    assignments = args.seq_len * spec.n_layers * spec.top_k if spec.kind == "moe" else 0
    dropped = metrics.get("overflow_dropped")
    print(json.dumps({"done": True, "global_step": tr.sched.global_step,
                      "steps_this_run": tr.sched.global_step - start_step,
                      "last_loss": metrics.get("loss"),
                      "smooth_loss": tr.sched.smooth_loss, "experts": n_exp,
                      "last_overflow_dropped": dropped,
                      "last_overflow_fraction": (dropped / assignments if assignments and dropped is not None else None)}, indent=2))
    for entry in log[-5:]:
        print(json.dumps(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
