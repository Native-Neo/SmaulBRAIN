"""Command-line interface: train / infer / checkpoint / quantize / report.

Every flag maps onto ``SmaulBrainConfig``; ``--help`` documents each one.
Subcommands share one model build path and one checkpoint format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from bytes import decode_text, encode_text
from config import SmaulBrainConfig, expert_hidden_for_target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smaulbrain",
                                description="SmaulBRAIN recurrent byte-level MoE LM")
    # --- architecture ---
    p.add_argument("--d-model", type=int, default=128, help="Shared trunk width.")
    p.add_argument("--n-heads", type=int, default=4, help="Linear-attention heads.")
    p.add_argument("--experts", type=int, default=4, dest="num_experts",
                   help="Initial dynamic expert count.")
    p.add_argument("--expert-size", type=int, default=None, dest="expert_size",
                   help="Expert hidden dim. Omitted: 3328 at d-model 512 "
                        "(~=5.12M params/expert); 256 at the d-model 128 default "
                        "(test-scale, keeps CPU runs fast).")
    p.add_argument("--active-experts", type=int, default=2, dest="top_k",
                   help="Top-k routed experts per token (active working set).")
    p.add_argument("--max-experts", type=int, default=64, help="Expert pool ceiling.")
    p.add_argument("--min-experts", type=int, default=1, help="Expert pool floor.")
    # --- adaptive depth ---
    p.add_argument("--max-depth", type=int, default=4, help="Max recurrent applications.")
    p.add_argument("--min-depth", type=int, default=1, help="Min recurrent applications.")
    p.add_argument("--halting-threshold", type=float, default=0.9,
                   help="Cumulative halt prob that stops inference depth.")
    # --- paging ---
    p.add_argument("--pagingmthd", type=str, default="D2R",
                   choices=["D2R", "R2VR", "D2VR"],
                   help="D2R=disk->RAM, R2VR=RAM->VRAM (staged), D2VR=disk->VRAM direct.")
    p.add_argument("--ram-cache", type=int, default=8, help="Max experts in RAM cache.")
    p.add_argument("--vram-cache", type=int, default=4, help="Max experts in VRAM cache.")
    # --- optimization ---
    p.add_argument("--expert-lr", type=float, default=2e-4, help="Expert learning rate.")
    p.add_argument("--trunk-lr-mult", type=float, default=0.1,
                   help="Shared-trunk LR multiplier (slow trunk vs fast experts).")
    # --- runtime ---
    p.add_argument("--context-length", type=int, default=1024, help="Training context.")
    p.add_argument("--threads", type=int, default=2, help="Torch CPU threads.")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"],
                   help="Activation compute dtype.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--ckpt", type=str, default="checkpoints/smaulbrain",
                   help="Checkpoint directory.")
    sub = p.add_subparsers(dest="cmd", required=True)
    # train
    t = sub.add_parser("train", help="Train / fine-tune (same model, modes differ).")
    t.add_argument("--steps", type=int, default=20, help="Optimizer steps.")
    t.add_argument("--batch", type=int, default=2, help="Batch size (sequences).")
    t.add_argument("--mode", type=str, default="entire",
                   choices=["entire", "trunk", "experts", "selected", "new"],
                   help="Fine-tuning selector: which groups move.")
    t.add_argument("--selected", type=str, default=None,
                   help="Comma-separated expert ids for --mode selected.")
    t.add_argument("--new-since", type=int, default=0,
                   help="Birth-step cutoff for --mode new.")
    t.add_argument("--data", type=str, default=None,
                   help="Text file for training bytes (default: synthetic demo).")
    t.add_argument("--save-every", type=int, default=0, help="Checkpoint every N steps (0=off).")
    t.add_argument("--grow-every", type=int, default=0, help="Growth eval every N steps (0=off).")
    t.add_argument("--prune-every", type=int, default=0, help="Prune eval every N steps (0=off).")
    # infer
    i = sub.add_parser("infer", help="Generate bytes from a prompt.")
    i.add_argument("--prompt", type=str, default="hello", help="Prompt text.")
    i.add_argument("--max-new", type=int, default=32, help="Bytes to generate.")
    i.add_argument("--temperature", type=float, default=0.0, help="Sampling temp (0=greedy).")
    # report
    r = sub.add_parser("report", help="Print dynamic parameter counts + paging stats.")
    # quantize
    q = sub.add_parser("quantize", help="Convert checkpoint precision per expert.")
    q.add_argument("--to", type=str, default="fp8", choices=["fp8", "bf16"],
                   help="fp8 requantizes experts; bf16 converts trunk.")
    return p


def config_from_args(args: argparse.Namespace) -> SmaulBrainConfig:
    expert_hidden = args.expert_size
    if expert_hidden is None:
        expert_hidden = expert_hidden_for_target(args.d_model) if args.d_model != 128 else 256
    return SmaulBrainConfig(
        d_model=args.d_model, n_heads=args.n_heads, num_experts=args.num_experts,
        top_k=args.top_k, max_experts=args.max_experts, min_experts=args.min_experts,
        expert_hidden=expert_hidden, max_depth=args.max_depth, min_depth=args.min_depth,
        halting_threshold=args.halting_threshold, paging_method=args.pagingmthd,
        ram_cache=args.ram_cache, vram_cache=args.vram_cache, expert_lr=args.expert_lr,
        trunk_lr_mult=args.trunk_lr_mult, context_length=args.context_length,
        threads=args.threads, dtype=args.dtype, seed=args.seed,
    )


def _demo_seqs(n: int = 64, length: int = 40) -> list[list[int]]:
    import random
    rng = random.Random(0)
    return [[rng.randrange(256) for _ in range(length)] for _ in range(n)]


def main(argv: list[str] | None = None) -> int:
    from infer import generate
    from model import SmaulBrainModel
    from smaulopt import SmaulOpt, SmaulOptHParams
    from storage import load_model, save_model

    args = build_parser().parse_args(argv)
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = config_from_args(args)

    if args.cmd == "quantize":
        from quantize import convert_checkpoint
        reports = convert_checkpoint(args.ckpt, to=args.to)
        print(json.dumps({"converted_experts": len(reports),
                          "bytes_after": sum(r["bytes_after"] for r in reports)}, indent=2))
        return 0

    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, wd=cfg.weight_decay,
                                   beta_m=cfg.beta_m, beta_v=cfg.beta_v,
                                   eps=cfg.epsilon, clip=cfg.grad_clip,
                                   state_dtype=cfg.state_dtype))
    if os.path.exists(os.path.join(args.ckpt, "manifest.json")):
        load_model(args.ckpt, model, opt)

    if args.cmd == "train":
        from train import run_training
        if args.data and os.path.exists(args.data):
            with open(args.data, "rb") as f:
                raw = f.read()
            seqs = [list(raw[i:i + cfg.context_length + 1])
                    for i in range(0, len(raw) - 1, cfg.context_length + 1)][:512] or _demo_seqs()
        else:
            seqs = _demo_seqs()
        res = run_training(model, opt, cfg, seqs, steps=args.steps,
                           batch_size=args.batch, mode=args.mode,
                           selected=args.selected.split(",") if args.selected else None,
                           new_since_step=args.new_since,
                           ckpt_dir=args.ckpt if args.save_every else None,
                           save_every=args.save_every,
                           grow_every=args.grow_every,
                           prune_every=args.prune_every)
        print(json.dumps({"final_loss": res["final_loss"],
                          "retention": res["retention"],
                          "experts": len(model.pool)}, indent=2))
        if args.save_every == 0:
            save_model(args.ckpt, model, opt, args.steps)
    elif args.cmd == "infer":
        res = generate(model, encode_text(args.prompt), max_new=args.max_new,
                       temperature=args.temperature, context=cfg.context_length,
                       seed=args.seed)
        print(decode_text(res["ids"]))
    elif args.cmd == "report":
        counts = model.param_counts()
        counts["paging"] = model.pager.stats.to_dict()
        counts["resident"] = model.pager.resident_counts()
        print(json.dumps(counts, indent=2))
    model.pager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
