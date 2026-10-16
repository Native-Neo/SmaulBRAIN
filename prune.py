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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prune worst MoE experts (evidence-based)")
    _cli.add_common_args(ap)
    ap.add_argument("--rm-worst", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="confirm destructive checkpoint replacement")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--synth-bytes", type=int, default=1 << 16)
    ap.add_argument("--eval-batches", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    if args.corpus:
        stream = _data.ByteStream(path=args.corpus)
    else:
        stream = _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed))
    batches = [stream.next_tokens(args.seq_len) for _ in range(args.eval_batches)]
    try:
        report = _pc.prune_checkpoint(
            args.checkpoint, args.rm_worst, batches,
            min_samples=args.growth_min_eval_samples,
            dry_run=args.dry_run, assume_yes=args.yes)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
