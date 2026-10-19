#!/usr/bin/env python3
"""SmaulBRAIN v2.1.1 evaluation: cross-entropy (nats/byte) + expert stats."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import data as _data


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Evaluate SmaulBRAIN v2.1.1 checkpoint",
                                 allow_abbrev=False)
    _cli.add_common_args(ap)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--synth-bytes", type=int, default=1 << 16)
    ap.add_argument("--eval-batches", type=int, default=16)
    args = ap.parse_args(raw_argv)

    if args.eval_batches < 1:
        ap.error("--eval-batches must be at least 1")

    from smaulbrain import checkpoint as _ckpt
    try:
        blob = _ckpt.load_checkpoint(args.checkpoint)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: unable to load checkpoint at {args.checkpoint}: {exc}",
              file=sys.stderr)
        return 2

    manifest = blob["manifest"]
    saved = manifest.get("config", {}) or {}

    if (not _was_provided(raw_argv, "--preset")
            and not args.moe and not args.dense):
        args.preset = manifest.get("preset", args.preset)

    for option, attr, key in (
        ("--forward-quant", "forward_quant", "forward_quant"),
        ("--backward-mode", "backward_mode", "backward_mode"),
        ("--paging", "paging", "paging"),
        ("--seed", "seed", "seed"),
        ("--seq-len", "seq_len", "seq_len"),
        ("--expert-capacity-factor", "expert_capacity_factor", "capacity_factor"),
    ):
        value = saved.get(key)
        if value is not None and not _was_provided(raw_argv, option):
            setattr(args, attr, value)
    if saved.get("synth_bytes") is not None and not _was_provided(raw_argv, "--synth-bytes"):
        args.synth_bytes = int(saved["synth_bytes"])
    if args.device == "cpu" and args.paging in ("V", "D2V", "R2V"):
        args.paging = "R"

    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))

    if manifest.get("preset") != preset:
        print(f"error: checkpoint preset {manifest.get('preset')} != {preset}",
              file=sys.stderr)
        return 2

    spec = _cfg.PRESETS[preset]

    # Early data validation before checkpoint I/O when we know the requested
    # sequence length cannot be used.
    if args.corpus is None and args.synth_bytes < args.seq_len + 1:
        ap.error("--synth-bytes must be at least --seq-len + 1 when --corpus is not set")

    try:
        stream = (_data.ByteStream(path=args.corpus) if args.corpus
                  else _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed + 1)))
        batches = [stream.next_tokens(args.seq_len) for _ in range(args.eval_batches)]
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: unable to prepare evaluation data: {exc}", file=sys.stderr)
        return 2

    from smaulbrain import growth as _growth
    from smaulbrain import model as _model
    mtp_depth = int(saved.get("mtp_depth", 0))
    mtp_loss_weight = float(saved.get("mtp_loss_weight", 0.1))
    capacity_factor = float(args.expert_capacity_factor)

    m = _model.Model(spec, dict(blob["params"]), forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode, paging_mode=args.paging,
                     capacity_factor=capacity_factor, mtp_depth=mtp_depth,
                     mtp_loss_weight=mtp_loss_weight, device=args.device)
    del blob
    losses = [_growth.eval_loss_on_tokens(m, b) for b in batches]
    print(json.dumps({
        "preset": preset,
        "forward_quant": args.forward_quant,
        "backward_mode": args.backward_mode,
        "capacity_factor": capacity_factor,
        "nats_per_byte_mean": (sum(losses) / len(losses)) if losses else 0.0,
        "nats_per_byte_min": (min(losses) if losses else 0.0),
        "batches": len(losses),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
