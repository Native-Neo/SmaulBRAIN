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
    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    from smaulbrain import checkpoint as _ckpt
    from smaulbrain import growth as _growth
    from smaulbrain import model as _model
    blob = _ckpt.load_checkpoint(args.checkpoint)
    spec = _cfg.PRESETS[preset]
    m = _model.Model(spec, dict(blob["params"]), forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode)
    stream = (_data.ByteStream(path=args.corpus, seed=args.seed) if args.corpus
              else _data.ByteStream(data=_data.synth_bytes(args.synth_bytes, args.seed + 1),
                                    seed=args.seed + 1))
    batches = [stream.next_tokens(args.seq_len) for _ in range(args.eval_batches)]
    losses = [_growth.eval_loss_on_tokens(m, b) for b in batches]
    import numpy as _np
    print(json.dumps({
        "preset": preset,
        "nats_per_byte_mean": float(_np.mean(losses)),
        "nats_per_byte_min": float(_np.min(losses)),
        "batches": len(batches),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
