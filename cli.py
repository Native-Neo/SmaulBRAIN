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
from config import SmaulBrainConfig


# Tiny defaults are the parser defaults (fast CPU smoke runs: 246K params).
TINY_DEFAULTS = {
    "d_model": 64, "n_heads": 4, "num_experts": 8, "top_k": 2,
    "expert_hidden": 128, "max_experts": 64, "min_experts": 2,
    "max_depth": 3, "min_depth": 1, "context_length": 128,
    "halting_threshold": 0.9, "ram_cache": 8, "vram_cache": 4,
}
# --full preset: ~5.12M params/expert, 64 experts, top-8 routing.
FULL_DEFAULTS = {
    **TINY_DEFAULTS,
    "d_model": 512, "num_experts": 64, "top_k": 8, "expert_hidden": 3328,
    "max_experts": 128, "min_experts": 8, "context_length": 1024,
    "ram_cache": 32, "vram_cache": 16,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smaulbrain",
                                description="SmaulBRAIN recurrent byte-level MoE LM")
    # --- architecture (tiny defaults; see --full) ---
    p.add_argument("--d-model", type=int, default=None, help="Shared trunk width (default: tiny=64).")
    p.add_argument("--n-heads", type=int, default=None, help="Linear-attention heads (default: tiny=4).")
    p.add_argument("--experts", type=int, default=None, dest="num_experts",
                   help="Initial dynamic expert count (default: tiny=8).")
    p.add_argument("--expert-size", type=int, default=None, dest="expert_size",
                   help="Expert hidden dim. Omitted: 128, or 3328 with --full "
                        "(~=5.12M params/expert at d-model 512).")
    p.add_argument("--full", action="store_true",
                   help="Full-size preset: 64 experts, top-8 routing, ~5.12M "
                        "params/expert. Any architecture flag passed explicitly "
                        "overrides the preset.")
    p.add_argument("--active-experts", type=int, default=None, dest="top_k",
                   help="Top-k routed experts per token (default: tiny=2).")
    p.add_argument("--max-experts", type=int, default=None, help="Expert pool ceiling (default: tiny=64).")
    p.add_argument("--min-experts", type=int, default=None, help="Expert pool floor (default: tiny=2).")
    # --- adaptive depth ---
    p.add_argument("--max-depth", type=int, default=None, help="Max recurrent applications (default: tiny=3).")
    p.add_argument("--min-depth", type=int, default=None, help="Min recurrent applications (default: tiny=1).")
    p.add_argument("--halting-threshold", type=float, default=None,
                   help="Cumulative halt prob that stops inference depth (default: tiny=0.9).")
    # --- paging ---
    p.add_argument("--pagingmthd", "--paging-method", type=str, default="D2R",
                   dest="pagingmthd",
                   choices=["D2R", "R2VR", "D2VR", "d2r", "r2vr", "d2vr"],
                   help="D2R=disk->RAM, R2VR=RAM->VRAM (staged), D2VR=disk->VRAM direct (case-insensitive).")
    p.add_argument("--ram-cache", type=int, default=None, help="Max experts in RAM cache (default: tiny=8).")
    p.add_argument("--vram-cache", type=int, default=None, help="Max experts in VRAM cache (default: tiny=4).")
    # --- optimization ---
    p.add_argument("--expert-lr", type=float, default=2e-4, help="Expert learning rate.")
    p.add_argument("--trunk-lr-mult", type=float, default=0.1,
                   help="Shared-trunk LR multiplier (slow trunk vs fast experts).")
    # --- runtime ---
    p.add_argument("--context-length", type=int, default=None, help="Training context (default: tiny=128).")
    p.add_argument("--attention-chunk-size", type=int, default=256,
                   help="Causal linear-attention training chunk size.")
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
    t.add_argument("--grow-loss-below", type=float, default=0.75,
                   help="Grow whenever step loss newly dips below this (negative disables).")
    t.add_argument("--growths-per-prune", type=int, default=2,
                   help="One prune evaluation every N growth events.")
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
    preset = FULL_DEFAULTS if args.full else TINY_DEFAULTS

    def pick(cli_value, key: str):
        # Flags default to None, so an explicitly passed value (even one
        # equal to a tiny default) always wins; otherwise the preset applies.
        return cli_value if cli_value is not None else preset[key]

    expert_hidden = args.expert_size
    if expert_hidden is None:
        expert_hidden = preset["expert_hidden"]
    return SmaulBrainConfig(
        d_model=pick(args.d_model, "d_model"),
        n_heads=pick(args.n_heads, "n_heads"),
        num_experts=pick(args.num_experts, "num_experts"),
        top_k=pick(args.top_k, "top_k"),
        max_experts=pick(args.max_experts, "max_experts"),
        min_experts=pick(args.min_experts, "min_experts"),
        expert_hidden=expert_hidden,
        max_depth=pick(args.max_depth, "max_depth"),
        min_depth=pick(args.min_depth, "min_depth"),
        halting_threshold=pick(args.halting_threshold, "halting_threshold"),
        paging_method=args.pagingmthd.upper(),
        ram_cache=pick(args.ram_cache, "ram_cache"),
        vram_cache=pick(args.vram_cache, "vram_cache"),
        expert_lr=args.expert_lr,
        trunk_lr_mult=args.trunk_lr_mult, context_length=pick(args.context_length, "context_length"),
        attention_chunk_size=args.attention_chunk_size,
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
        # The checkpoint is authoritative for all runtime configuration that
        # does not change tensor shapes; use it for the resumed run.
        cfg = model.cfg

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
                           grow_loss_below=args.grow_loss_below,
                           growths_per_prune=args.growths_per_prune,
                           prune_every=args.prune_every,
                           # Progress/grow/prune lines go to stderr: stdout
                           # stays pure JSON for scripting.
                           log_fn=lambda s: print(s, file=sys.stderr))
        print(json.dumps({"final_loss": res["final_loss"],
                          "retention": res["retention"],
                          "experts": len(model.pool),
                          "steps_completed": res["steps_completed"],
                          "elapsed_seconds": round(res["elapsed_seconds"], 3),
                          "bytes_processed": res["bytes_processed"],
                          "bytes_per_second": round(res["bytes_per_second"], 2)}, indent=2))
        if args.save_every == 0:
            save_model(args.ckpt, model, opt, int(getattr(model, "_resume_step", -1)),
                       extra_meta={"scheduler": res["scheduler"]})
    elif args.cmd == "infer":
        if os.path.exists(os.path.join(args.ckpt, "manifest.json")):
            # Serve future misses from the checkpoint files. load_model above
            # already cleared stale caches and staged R2VR RAM; wiring the
            # loader here must not repeat that work (double disk reads).
            from storage import make_disk_loader
            model.pager.load_from_disk = make_disk_loader(args.ckpt)
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
