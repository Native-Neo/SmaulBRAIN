#!/usr/bin/env python3
"""SmaulBRAIN V2 training entry point."""
import argparse
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import data as _data
from smaulbrain import train_loop as _tl


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _restore_saved_defaults(args, argv, manifest):
    """Use checkpoint settings unless the user explicitly overrides them."""
    config = manifest.get("config", {})
    optimizer = manifest.get("optimizer", {})
    scheduler = manifest.get("scheduler", {})
    identity = manifest.get("data_stream", {})

    settings = [
        ("--forward-quant", "forward_quant", config.get("forward_quant")),
        ("--backward-mode", "backward_mode", config.get("backward_mode")),
        ("--paging", "paging", config.get("paging")),
        ("--seed", "seed", config.get("seed")),
        ("--seq-len", "seq_len", config.get("seq_len")),
        ("--expert-capacity-factor", "expert_capacity_factor", config.get("capacity_factor")),
        ("--lr", "lr", optimizer.get("lr", config.get("lr"))),
        ("--beta1", "beta1", optimizer.get("beta1")),
        ("--beta2", "beta2", optimizer.get("beta2")),
        ("--eps", "eps", optimizer.get("eps")),
        ("--weight-decay", "weight_decay", optimizer.get("weight_decay")),
        ("--growth-noise-scale", "growth_noise_scale", config.get("growth_noise_scale")),
        ("--growth-seed", "growth_seed", config.get("growth_seed")),
        ("--grad-clip", "grad_clip", config.get("grad_clip")),
    ]
    for option, attr, saved_value in settings:
        if saved_value is not None and not _was_provided(argv, option):
            setattr(args, attr, saved_value)

    if (config.get("synth_bytes") is not None
            and not _was_provided(argv, "--synth-bytes")):
        args.synth_bytes = int(config["synth_bytes"])

    # Restore the file corpus path when it was omitted. Equivalent symlink or
    # absolute/relative spellings remain accepted by ByteStream's identity test.
    if not _was_provided(argv, "--corpus") and identity.get("kind") == "file":
        args.corpus = identity.get("path")

    # Scheduler state is authoritative on normal resume. Explicit trigger flags
    # are applied after loading the scheduler, below.
    if not _was_provided(argv, "--growth-every-steps"):
        args.growth_every_steps = str(scheduler.get("growth_every_steps", 10000))
    if not _was_provided(argv, "--growth-loss-nats"):
        threshold = scheduler.get("loss_threshold", 0.5)
        args.growth_loss_nats = "none" if threshold is None else str(threshold)
    if not _was_provided(argv, "--growth-loss-beta"):
        args.growth_loss_beta = float(scheduler.get("loss_beta", 0.98))
    if not _was_provided(argv, "--growth-min-eval-samples"):
        args.growth_min_eval_samples = int(scheduler.get("min_eval_samples", 256))


def _apply_explicit_growth_overrides(trainer, args, argv):
    """Apply only scheduler flags actually supplied on this resume invocation."""
    sched = trainer.sched
    if trainer.spec.kind != "moe":
        return
    if _was_provided(argv, "--growth-every-steps"):
        sched.every_steps = args.growth_every_steps
        interval = sched.every_steps
        sched.next_periodic_step = (
            (sched.global_step // interval + 1) * interval if interval > 0 else None
        )
    if _was_provided(argv, "--growth-loss-nats"):
        sched.loss_threshold = args.growth_loss_nats
        sched.loss_armed = True
    if _was_provided(argv, "--growth-loss-beta"):
        sched.loss_beta = args.growth_loss_beta
    if _was_provided(argv, "--growth-min-eval-samples"):
        sched.min_eval_samples = args.growth_min_eval_samples


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Train SmaulBRAIN V2 (byte-level recurrent LM)")
    _cli.add_common_args(ap, need_train=True)
    ap.add_argument(
        "--resume", action="store_true",
        help="require a checkpoint (existing checkpoints auto-resume by default)",
    )
    ap.add_argument(
        "--steps", type=int, default=None,
        help="number of additional optimizer updates (useful with --resume)",
    )
    args = ap.parse_args(raw_argv)
    if args.steps is not None and args.steps < 0:
        ap.error("--steps must be >= 0")

    checkpoint_exists = (
        os.path.exists(os.path.join(args.checkpoint, "manifest.json"))
        or os.path.exists(os.path.join(os.path.normpath(args.checkpoint) + ".bak", "manifest.json"))
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
            args.preset = manifest["preset"]
        _restore_saved_defaults(args, raw_argv, manifest)

    try:
        preset = _cli.validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    spec = _cfg.PRESETS[preset]
    if checkpoint_blob is not None and checkpoint_blob["manifest"].get("preset") != preset:
        print(
            f"error: checkpoint preset {checkpoint_blob['manifest'].get('preset')} != {preset}",
            file=sys.stderr,
        )
        return 2
    if preset.startswith("dense") and (args.growth_every_steps or args.growth_loss_nats is not None):
        print("note: MoE growth is disabled for dense models; growth triggers ignored")

    try:
        if args.corpus:
            stream = _data.ByteStream(path=args.corpus)
        else:
            stream = _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed))
        if checkpoint_blob is not None:
            saved = checkpoint_blob["manifest"]
            stream.load_state_dict({
                "cursor": saved["data_cursor"],
                "path": stream.path,
                "size": stream.size,
                "identity": saved.get("data_stream"),
            })
        tr = _tl.Trainer(spec, args, stream, checkpoint_blob=checkpoint_blob)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: unable to initialize training: {exc}", file=sys.stderr)
        return 2

    if checkpoint_blob is not None:
        man = checkpoint_blob["manifest"]
        action = "resumed" if args.resume else "auto-resumed"
        print(f"{action} {preset} at global_step={man['scheduler']['global_step']} "
              f"experts={man['n_experts_current']}")
        saved_config = man.get("config", {})
        for key, current in (
            ("forward_quant", args.forward_quant),
            ("backward_mode", args.backward_mode),
            ("seq_len", args.seq_len),
            ("capacity_factor", args.expert_capacity_factor),
            ("paging", args.paging),
            ("seed", args.seed),
            ("lr", args.lr),
        ):
            previous = saved_config.get(key)
            if previous is not None and previous != current:
                print(
                    f"warning: checkpoint {key}={previous!r}, requested {current!r}; "
                    "resuming with changed settings can disrupt training",
                    file=sys.stderr,
                )
        saved_optimizer = man.get("optimizer", {})
        for key, current in (
            ("beta1", args.beta1), ("beta2", args.beta2),
            ("eps", args.eps), ("weight_decay", args.weight_decay),
        ):
            previous = saved_optimizer.get(key)
            if previous is not None and previous != current:
                print(
                    f"warning: checkpoint optimizer {key}={previous!r}, requested {current!r}; "
                    "resuming with changed settings can disrupt training",
                    file=sys.stderr,
                )
        _apply_explicit_growth_overrides(tr, args, raw_argv)

    # The checkpoint blob is no longer needed as an owning dictionary. The
    # model/optimizer retain their tensor references (or device copies).
    del checkpoint_blob

    rep = tr.model.report()
    print(json.dumps({
        "preset": preset,
        "total_params": rep["total_params"],
        "per_expert": rep.get("per_expert_params"),
        "growth_every_steps": tr.sched.every_steps,
        "growth_loss_nats": tr.sched.loss_threshold,
    }, indent=2))

    start_step = tr.sched.global_step
    target_step = start_step + args.steps if args.steps is not None else args.max_steps
    if target_step <= start_step:
        print(
            f"warning: no training steps will run (global_step={start_step}, "
            f"max_steps={target_step}); use --steps N for additional updates",
            file=sys.stderr,
        )
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _request_graceful_stop(signum, frame):
        if not tr.stop_requested:
            print(
                "\nCtrl+C received; stopping after the current optimizer update and saving checkpoint...",
                file=sys.stderr, flush=True,
            )
        tr.request_stop()

    signal.signal(signal.SIGINT, _request_graceful_stop)
    try:
        log = tr.train(
            max_steps=target_step,
            log_every=args.log_every,
            ckpt_every=args.ckpt_every,
            grad_clip=args.grad_clip,
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    # Close the narrow SIGINT race where the signal lands after train() returns
    # but before the handler is restored.
    if tr.stop_requested:
        if tr.last_checkpoint_step != tr.sched.global_step:
            tr.save()
    elif not args.ckpt_every:
        tr.save()

    from smaulbrain import model as _model
    n_exp = len(_model.expert_ids(tr.model.p)) if preset.startswith("moe") else 0
    metrics = tr.last_metrics or {}
    # ByteStream returns seq_len+1 tokens for next-byte targets; the model
    # consumes the first seq_len positions, so every layer routes seq_len tokens.
    assignments = args.seq_len * spec.n_layers * spec.top_k if spec.kind == "moe" else 0
    dropped = metrics.get("overflow_dropped")
    print(json.dumps({
        "done": True,
        "global_step": tr.sched.global_step,
        "steps_this_run": tr.sched.global_step - start_step,
        "last_loss": metrics.get("loss"),
        "smooth_loss": tr.sched.smooth_loss,
        "experts": n_exp,
        "last_overflow_dropped": dropped,
        "last_overflow_fraction": (
            dropped / assignments if assignments and dropped is not None else None
        ),
    }, indent=2))
    for entry in log[-5:]:
        print(json.dumps(entry))
    return 130 if tr.stop_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
