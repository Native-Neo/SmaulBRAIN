#!/usr/bin/env python3
"""SmaulBRAIN V2 evaluation: cross-entropy (nats/byte) + expert stats."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import data as _data


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate SmaulBRAIN V2 checkpoint")
    _cli.add_common_args(ap)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--synth-bytes", type=int, default=1 << 16)
    ap.add_argument("--eval-batches", type=int, default=16)
    args = ap.parse_args(argv)
    if args.eval_batches < 1:
        ap.error("--eval-batches must be >= 1")
    if args.synth_bytes < args.seq_len + 1 and not args.corpus:
        ap.error("--synth-bytes must be at least --seq-len + 1 when --corpus is not set")
    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    from smaulbrain import checkpoint as _ckpt
    from smaulbrain import growth as _growth
    from smaulbrain import model as _model
    blob = _ckpt.load_checkpoint(args.checkpoint)
    if blob["manifest"].get("preset") != preset:
        print(f"error: checkpoint preset {blob['manifest'].get('preset')} != {preset}",
              file=sys.stderr)
        return 2
    spec = _cfg.PRESETS[preset]
    m = _model.Model(spec, dict(blob["params"]), forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode, device=args.device)
    stream = (_data.ByteStream(path=args.corpus) if args.corpus
              else _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed + 1)))
    batches = [stream.next_tokens(args.seq_len) for _ in range(args.eval_batches)]
    losses = [_growth.eval_loss_on_tokens(m, b) for b in batches]
    print(json.dumps({
        "preset": preset,
        "nats_per_byte_mean": sum(losses) / len(losses) if losses else 0.0,
        "nats_per_byte_min": min(losses) if losses else 0.0,
        "batches": len(batches),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
