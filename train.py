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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train SmaulBRAIN V2 (byte-level recurrent LM)")
    _cli.add_common_args(ap, need_train=True)
    ap.add_argument("--resume", action="store_true", help="require a checkpoint (existing checkpoints auto-resume by default)")
    ap.add_argument("--steps", type=int, default=None,
                    help="number of additional optimizer updates (useful with --resume)")
    args = ap.parse_args(argv)
    if args.steps is not None and args.steps < 0:
        ap.error("--steps must be >= 0")
    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    spec = _cfg.PRESETS[preset]
    if preset.startswith("dense") and (args.growth_every_steps or args.growth_loss_nats is not None):
        print("note: MoE growth is disabled for dense models; growth triggers ignored")
    if args.corpus:
        stream = _data.ByteStream(path=args.corpus)
    else:
        stream = _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed))
    tr = _tl.Trainer(spec, args, stream)
    checkpoint_exists = (
        os.path.exists(os.path.join(args.checkpoint, "manifest.json"))
        or os.path.exists(os.path.join(os.path.normpath(args.checkpoint) + ".bak", "manifest.json"))
    )
    if args.resume and not checkpoint_exists:
        print(f"error: nothing to resume at {args.checkpoint}", file=sys.stderr)
        return 2
    if checkpoint_exists:
        man = tr.load(args.checkpoint)
        action = "resumed" if args.resume else "auto-resumed"
        print(f"{action} {preset} at global_step={man['scheduler']['global_step']} "
              f"experts={man['n_experts_current']}")
        saved = man.get("config", {})
        for key, current in (("forward_quant", args.forward_quant),
                             ("backward_mode", args.backward_mode),
                             ("seq_len", args.seq_len),
                             ("capacity_factor", args.expert_capacity_factor)):
            previous = saved.get(key)
            if previous is not None and previous != current:
                print(f"warning: checkpoint {key}={previous!r}, requested {current!r}; "
                      "resuming with changed settings can disrupt training", file=sys.stderr)
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
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _request_graceful_stop(signum, frame):
        if not tr.stop_requested:
            print("\nCtrl+C received; finishing this optimizer update, then saving checkpoint...",
                  file=sys.stderr, flush=True)
        tr.request_stop()

    signal.signal(signal.SIGINT, _request_graceful_stop)
    try:
        log = tr.train(max_steps=target_step, log_every=args.log_every,
                       ckpt_every=args.ckpt_every, grad_clip=args.grad_clip)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    # Trainer.train() saves as part of graceful-stop handling. Avoid rewriting
    # large checkpoints a second time after Ctrl+C.
    if not tr.stop_requested:
        tr.save()
    from smaulbrain import model as _model
    n_exp = len(_model.expert_ids(tr.model.p)) if preset.startswith("moe") else 0
    metrics = tr.last_metrics or {}
    assignments = max(0, args.seq_len - 1) * spec.n_layers * spec.top_k if spec.kind == "moe" else 0
    dropped = metrics.get("overflow_dropped")
    print(json.dumps({"done": True, "global_step": tr.sched.global_step,
                      "steps_this_run": tr.sched.global_step - start_step,
                      "last_loss": metrics.get("loss"),
                      "smooth_loss": tr.sched.smooth_loss, "experts": n_exp,
                      "last_overflow_dropped": dropped,
                      "last_overflow_fraction": (dropped / assignments if assignments and dropped is not None else None)}, indent=2))
    for entry in log[-5:]:
        print(json.dumps(entry))
    return 130 if tr.stop_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
