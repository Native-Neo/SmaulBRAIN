#!/usr/bin/env python3
"""SmaulBRAIN V2 expert pruning: python prune.py --rm-worst 8."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import data as _data
from smaulbrain import prune_cmd as _pc


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Prune worst MoE experts (evidence-based)", allow_abbrev=False)
    _cli.add_common_args(ap)
    ap.add_argument("--rm-worst", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="confirm destructive checkpoint replacement")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--synth-bytes", type=int, default=1 << 16)
    ap.add_argument("--eval-batches", type=int, default=8)
    args = ap.parse_args(raw_argv)

    if args.eval_batches < 1:
        ap.error("--eval-batches must be >= 1")
    from smaulbrain import checkpoint as _ckpt
    try:
        blob = _ckpt.load_checkpoint(args.checkpoint)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: unable to load checkpoint at {args.checkpoint}: {exc}",
              file=sys.stderr)
        return 2

    manifest = blob["manifest"]
    saved = manifest.get("config", {})
    if (not _was_provided(raw_argv, "--preset")
            and not args.moe and not args.dense):
        args.preset = manifest["preset"]
    if not _was_provided(raw_argv, "--seq-len") and saved.get("seq_len") is not None:
        args.seq_len = int(saved["seq_len"])
    if not _was_provided(raw_argv, "--seed") and saved.get("seed") is not None:
        args.seed = int(saved["seed"])
    if not _was_provided(raw_argv, "--synth-bytes") and saved.get("synth_bytes") is not None:
        args.synth_bytes = int(saved["synth_bytes"])
    for option, attr, key in (
        ("--forward-quant", "forward_quant", "forward_quant"),
        ("--backward-mode", "backward_mode", "backward_mode"),
        ("--paging", "paging", "paging"),
        ("--expert-capacity-factor", "expert_capacity_factor", "capacity_factor"),
    ):
        value = saved.get(key)
        if value is not None and not _was_provided(raw_argv, option):
            setattr(args, attr, value)

    try:
        preset = _cli.validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    if manifest.get("preset") != preset:
        print(f"error: checkpoint preset {manifest.get('preset')} != {preset}",
              file=sys.stderr)
        return 2
    if not args.corpus and args.synth_bytes < args.seq_len + 1:
        ap.error("--synth-bytes must be at least --seq-len + 1 when --corpus is not set")
    del blob  # prune_cmd loads the authoritative checkpoint once for mutation.

    try:
        if args.corpus:
            stream = _data.ByteStream(path=args.corpus)
        else:
            # Keep the synthetic evidence deterministic but distinct from the
            # default training stream by using its next seed.
            stream = _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed + 1))
        batches = [stream.next_tokens(args.seq_len) for _ in range(args.eval_batches)]
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: unable to prepare pruning evidence: {exc}", file=sys.stderr)
        return 2

    try:
        report = _pc.prune_checkpoint(
            args.checkpoint, args.rm_worst, batches,
            min_samples=args.growth_min_eval_samples,
            dry_run=args.dry_run, assume_yes=args.yes)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
