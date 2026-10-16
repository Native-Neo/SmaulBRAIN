#!/usr/bin/env python3
"""SmaulBRAIN V2 training entry point."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import data as _data
from smaulbrain import train_loop as _tl


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train SmaulBRAIN V2 (byte-level recurrent LM)")
    _cli.add_common_args(ap, need_train=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
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
    if args.resume:
        if not os.path.exists(os.path.join(args.checkpoint, "manifest.json")):
            print(f"error: nothing to resume at {args.checkpoint}", file=sys.stderr)
            return 2
        man = tr.load(args.checkpoint)
        print(f"resumed {preset} at global_step={man['scheduler']['global_step']} "
              f"experts={man['n_experts_current']}")
    else:
        if os.path.exists(os.path.join(args.checkpoint, "manifest.json")):
            print(f"error: checkpoint exists at {args.checkpoint} (use --resume)", file=sys.stderr)
            return 2
    rep = tr.model.report()
    print(json.dumps({"preset": preset, "total_params": rep["total_params"],
                      "per_expert": rep.get("per_expert_params"),
                      "growth_every_steps": args.growth_every_steps,
                      "growth_loss_nats": args.growth_loss_nats}, indent=2))
    log = tr.train(max_steps=args.max_steps, log_every=args.log_every,
                   ckpt_every=args.ckpt_every, grad_clip=args.grad_clip)
    tr.save()
    from smaulbrain import model as _model
    n_exp = len(_model.expert_ids(tr.model.p)) if preset.startswith("moe") else 0
    print(json.dumps({"done": True, "global_step": tr.sched.global_step,
                      "smooth_loss": tr.sched.smooth_loss, "experts": n_exp}, indent=2))
    for entry in log[-5:]:
        print(json.dumps(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
