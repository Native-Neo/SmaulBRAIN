"""Command-line interface: train / infer / checkpoint / quantize / report.

Every flag maps onto ``SmaulBrainConfig``; ``--help`` documents each one.
Subcommands share one model build path and one checkpoint format.

Resume semantics (checkpoint-authoritative):
- Shape fields (vocab_size, d_model, n_heads, expert_hidden, top_k,
  dtype) are validated BEFORE construction. An explicitly passed value
  that differs from the checkpoint aborts (exit 2) instead of building
  an incompatible model and swapping part of it.
- All other (runtime) fields are restored from the checkpoint. An
  explicitly passed runtime value that differs is ignored with a warning
  on stderr, except --threads/--seed which override as process knobs.
- Without a checkpoint, explicit flags win over the --full/tiny preset,
  otherwise the preset applies; non-preset knobs fall back to
  ``SmaulBrainConfig`` dataclass defaults.
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
    "expert_hidden": 128, "max_experts": 64, "min_experts": 64,
    "max_depth": 3, "min_depth": 1, "context_length": 128,
    "halting_threshold": 0.9, "ram_cache": 8, "vram_cache": 4,
}
# --full preset: ~5.12M params/expert, 64 experts, top-8 routing.
FULL_DEFAULTS = {
    **TINY_DEFAULTS,
    "d_model": 512, "num_experts": 64, "top_k": 8, "expert_hidden": 3328,
    "max_experts": 128, "min_experts": 64, "context_length": 1024,
    "ram_cache": 32, "vram_cache": 16,
}

# Topology fields validated BEFORE construction (mirrors
# storage.load_model's shape gate). Explicit mismatches abort resume.
SHAPE_FIELDS = ("vocab_size", "d_model", "n_heads", "expert_hidden",
                "top_k", "dtype")

# Maps SmaulBrainConfig field -> argparse dest holding the CLI override.
# `expert_hidden` is set via --expert-size (dest expert_size).
ARG_TO_CONFIG = {
    "d_model": "d_model",
    "n_heads": "n_heads",
    "num_experts": "num_experts",
    "expert_hidden": "expert_size",
    "top_k": "top_k",
    "max_experts": "max_experts",
    "min_experts": "min_experts",
    "capacity_factor": "capacity_factor",
    "max_depth": "max_depth",
    "min_depth": "min_depth",
    "halting_threshold": "halting_threshold",
    "halt_prior": "halt_prior",
    "ponder_beta": "ponder_beta",
    "moe_balance_weight": "moe_balance_weight",
    "context_length": "context_length",
    "attention_chunk_size": "attention_chunk_size",
    "paging_method": "paging_method",
    "ram_cache": "ram_cache",
    "vram_cache": "vram_cache",
    "expert_lr": "expert_lr",
    "trunk_lr_mult": "trunk_lr_mult",
    "router_lr_mult": "router_lr_mult",
    "router_lr_mult_new": "router_lr_mult_new",
    "new_routing_bias": "new_routing_bias",
    "use_byte_conv": "use_byte_conv",
    "lookahead_weight": "lookahead_weight",
    "boundary_weight": "boundary_weight",
    "cosine_decay_steps": "cosine_decay_steps",
    "weight_decay": "weight_decay",
    "grad_clip": "grad_clip",
    "beta_m": "beta_m",
    "beta_v": "beta_v",
    "epsilon": "epsilon",
    "dtype": "dtype",
    "fp8_tile": "fp8_tile",
    "state_dtype": "state_dtype",
    "threads": "threads",
    "seed": "seed",
    "rmsnorm_eps": "rmsnorm_eps",
    "vocab_size": "vocab_size",
    "max_new_experts": "max_new_experts",
    "grow_every": "config_grow_every",
}


def _cfg_default(name: str):
    """Dataclass default for a config field (single source of truth)."""
    return SmaulBrainConfig.__dataclass_fields__[name].default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smaulbrain",
                                description="SmaulBRAIN recurrent byte-level MoE LM")
    # --- vocabulary / specials (shape; validated before construction) ---
    p.add_argument("--vocab-size", type=int, default=None, dest="vocab_size",
                   help="Total ids (bytes + specials). Default 260 (256 bytes + "
                        "bos/eos/pad/sep). Padding is always PAD_ID (never byte 0). "
                        "On resume an explicit mismatch aborts; else checkpoint wins.")
    # --- architecture (tiny defaults; see --full) ---
    p.add_argument("--d-model", type=int, default=None, help="Shared trunk width (default: tiny=64, full=512).")
    p.add_argument("--n-heads", type=int, default=None, help="Linear-attention heads (default: tiny=4).")
    p.add_argument("--experts", type=int, default=None, dest="num_experts",
                    help="Initial dynamic expert count (default: tiny=8, full=64). "
                         "Dynamic after growth/manual pruning; checkpoint count wins on resume.")
    p.add_argument("--expert-size", type=int, default=None, dest="expert_size",
                   help="Expert hidden dim. Omitted: 128, or 3328 with --full "
                        "(~=5.12M params/expert at d-model 512). Shape-checked before build.")
    p.add_argument("--full", action="store_true",
                   help="Full-size preset: 64 experts, top-8 routing, ~5.12M "
                        "params/expert. Any architecture flag passed explicitly "
                        "overrides the preset. Ignored on resume (checkpoint wins).")
    p.add_argument("--active-experts", type=int, default=None, dest="top_k",
                   help="Top-k routed experts per token (default: tiny=2, full=8). Shape-checked.")
    p.add_argument("--max-experts", type=int, default=None, help="Expert pool ceiling (default: tiny=64, full=128). Checkpoint wins on resume.")
    p.add_argument("--min-experts", type=int, default=None, help="Expert pool floor (default: 64; hard floor for 64+ pools). Checkpoint wins on resume.")
    p.add_argument("--capacity-factor", type=float, default=None, dest="capacity_factor",
                   help="Per-expert routing capacity multiple (default: 1.5).")
    p.add_argument("--moe-balance-weight", type=float, default=None, dest="moe_balance_weight",
                   help="Aux routing-balance loss weight (default: 0.01).")
    # --- adaptive depth / pondering ---
    p.add_argument("--max-depth", type=int, default=None, help="Max recurrent applications (default: tiny=3).")
    p.add_argument("--min-depth", type=int, default=None, help="Min recurrent applications (default: tiny=1).")
    p.add_argument("--halting-threshold", type=float, default=None,
                   help="Cumulative halt prob that stops inference depth (default: tiny=0.9).")
    p.add_argument("--halt-prior", type=float, default=None, dest="halt_prior",
                   help="Geometric prior p for ponder KL (default: 0.1).")
    p.add_argument("--ponder-beta", type=float, default=None, dest="ponder_beta",
                   help="Weight of ponder KL regularizer (default: 0.01).")
    # --- paging ---
    p.add_argument("--paging-method", "--pagingmthd", type=str, default=None,
                   dest="paging_method",
                   choices=["D2R", "R2VR", "D2VR", "d2r", "r2vr", "d2vr"],
                   help="D2R=disk->RAM, R2VR=RAM->VRAM (staged), D2VR=disk->VRAM direct "
                        "(case-insensitive; default: D2R). Checkpoint wins on resume.")
    p.add_argument("--ram-cache", type=int, default=None, help="Max experts in RAM cache (default: tiny=8, full=32).")
    p.add_argument("--vram-cache", type=int, default=None, help="Max experts in VRAM cache (default: tiny=4, full=16).")
    # --- optimization (checkpoint optimizer hparams win on resume) ---
    p.add_argument("--expert-lr", type=float, default=None, help="Expert learning rate (default: 2e-4).")
    p.add_argument("--trunk-lr-mult", type=float, default=None, dest="trunk_lr_mult",
                   help="Shared-trunk LR multiplier (slow trunk vs fast experts; default: 0.1).")
    p.add_argument("--router-lr-mult", type=float, default=None, dest="router_lr_mult",
                   help="Router LR multiplier (default: 1.0).")
    p.add_argument("--router-lr-mult-new", type=float, default=None, dest="router_lr_mult_new",
                   help="Old-expert router-row LR multiplier in --mode new only (default: 0.005).")
    p.add_argument("--new-routing-bias", type=float, default=None, dest="new_routing_bias",
                   help="Additive router-logit bonus for new experts in --mode new "
                        "(default: 0.0 = off).")
    p.add_argument("--weight-decay", type=float, default=None, dest="weight_decay",
                   help="Decoupled weight decay (default: 0.01).")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip",
                   help="Global grad-norm cap; 0 disables (default: 1.0).")
    p.add_argument("--beta-m", type=float, default=None, dest="beta_m",
                   help="SmaulOpt first-moment decay (default: 0.9).")
    p.add_argument("--beta-v", type=float, default=None, dest="beta_v",
                   help="SmaulOpt second-moment decay (default: 0.999).")
    p.add_argument("--epsilon", type=float, default=None, dest="epsilon",
                   help="SmaulOpt denominator epsilon (default: 1e-8).")
    # --- precision policy ---
    p.add_argument("--fp8-tile", type=int, default=None, dest="fp8_tile",
                   help="Block size for FP8 per-block scaling (default: 64). "
                        "Adopted from checkpoint on resume (not a shape gate).")
    p.add_argument("--state-dtype", type=str, default=None, dest="state_dtype",
                   choices=["bf16", "fp32"],
                   help="Optimizer state storage dtype (default: bf16). Checkpoint wins on resume.")
    # --- runtime ---
    p.add_argument("--context-length", type=int, default=None, help="Training context (default: tiny=128, full=1024). Checkpoint wins on resume.")
    p.add_argument("--attention-chunk-size", type=int, default=None, dest="attention_chunk_size",
                   help="Causal linear-attention training chunk size (default: 256). "
                        "Runtime; checkpoint wins on resume.")
    p.add_argument("--threads", type=int, default=None, help="Torch CPU threads (default: 2). Explicit wins on resume.")
    p.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp32"],
                   help="Activation compute dtype (default: bf16). Shape-checked before build.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: 0). Explicit wins on resume.")
    p.add_argument("--rmsnorm-eps", type=float, default=None, dest="rmsnorm_eps",
                   help="RMSNorm epsilon (default: 1e-6).")
    # --- growth (config; schedule flag lives on train) ---
    p.add_argument("--grow-every-default", type=int, default=None, dest="config_grow_every",
                   help="Config growth interval (default: 20000). Train --grow-every "
                        "overrides per-run; omitted train flag falls back to this config value.")
    p.add_argument("--max-new-experts", type=int, default=None, dest="max_new_experts",
                   help="Cap per growth event (default: 8).")
    # --- BPB ports (conv default-on; lookahead/boundary/cosine default-off;
    # enabling any of them changes training math) ---
    p.add_argument("--use-byte-conv", action="store_true", default=None,
                   help="Causal byte-n-gram conv in the trunk (default: on).")
    p.add_argument("--no-byte-conv", action="store_false", dest="use_byte_conv",
                   default=None,
                   help="Disable the byte-n-gram conv (default: on; this flag turns it off).")
    p.add_argument("--lookahead-weight", type=float, default=None, dest="lookahead_weight",
                   help="Auxiliary t+2 prediction loss weight (default: 0.0 = off).")
    p.add_argument("--boundary-weight", type=float, default=None, dest="boundary_weight",
                   help="Auxiliary UTF-8-boundary loss weight (default: 0.0 = off).")
    p.add_argument("--cosine-decay-steps", type=int, default=None, dest="cosine_decay_steps",
                   help="Cosine LR decay horizon in steps (default: 0 = constant LR).")
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
    t.add_argument("--grow-every", type=int, default=None, help="Growth eval every N steps (0=off; omitted: config grow_every=20000).")
    t.add_argument("--grow-experts", type=int, default=0,
                   help="Grow N experts once at run start (before training) for a new skill; "
                        "birth step equals the run's start global step so --mode new "
                        "--new-since <same step> selects exactly these (0=off).")
    t.add_argument("--grow-loss-below", type=float, default=1.0,
                   help="Grow whenever step loss newly dips below this (default 1.0; negative disables).")
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
    preset = FULL_DEFAULTS if getattr(args, "full", False) else TINY_DEFAULTS

    def pick(cli_value, key: str):
        # Flags default to None, so an explicitly passed value (even one
        # equal to a tiny default) always wins; otherwise the preset applies.
        return cli_value if cli_value is not None else preset[key]

    def pick_cfg(cli_value, name: str):
        # Non-preset knobs (no --full variant): explicit or dataclass default.
        return cli_value if cli_value is not None else _cfg_default(name)

    expert_hidden = getattr(args, "expert_size", None)
    if expert_hidden is None:
        expert_hidden = preset["expert_hidden"]
    paging = getattr(args, "paging_method", None)
    paging = paging.upper() if paging is not None else _cfg_default("paging_method")
    return SmaulBrainConfig(
        vocab_size=pick_cfg(getattr(args, "vocab_size", None), "vocab_size"),
        d_model=pick(getattr(args, "d_model", None), "d_model"),
        n_heads=pick(getattr(args, "n_heads", None), "n_heads"),
        num_experts=pick(getattr(args, "num_experts", None), "num_experts"),
        top_k=pick(getattr(args, "top_k", None), "top_k"),
        max_experts=pick(getattr(args, "max_experts", None), "max_experts"),
        min_experts=pick(getattr(args, "min_experts", None), "min_experts"),
        expert_hidden=expert_hidden,
        max_depth=pick(getattr(args, "max_depth", None), "max_depth"),
        min_depth=pick(getattr(args, "min_depth", None), "min_depth"),
        capacity_factor=pick_cfg(getattr(args, "capacity_factor", None), "capacity_factor"),
        moe_balance_weight=pick_cfg(getattr(args, "moe_balance_weight", None), "moe_balance_weight"),
        halting_threshold=pick(getattr(args, "halting_threshold", None), "halting_threshold"),
        halt_prior=pick_cfg(getattr(args, "halt_prior", None), "halt_prior"),
        ponder_beta=pick_cfg(getattr(args, "ponder_beta", None), "ponder_beta"),
        paging_method=paging,
        ram_cache=pick(getattr(args, "ram_cache", None), "ram_cache"),
        vram_cache=pick(getattr(args, "vram_cache", None), "vram_cache"),
        expert_lr=pick_cfg(getattr(args, "expert_lr", None), "expert_lr"),
        trunk_lr_mult=pick_cfg(getattr(args, "trunk_lr_mult", None), "trunk_lr_mult"),
        router_lr_mult=pick_cfg(getattr(args, "router_lr_mult", None), "router_lr_mult"),
        router_lr_mult_new=pick_cfg(getattr(args, "router_lr_mult_new", None), "router_lr_mult_new"),
        new_routing_bias=pick_cfg(getattr(args, "new_routing_bias", None), "new_routing_bias"),
        use_byte_conv=pick_cfg(getattr(args, "use_byte_conv", None), "use_byte_conv"),
        lookahead_weight=pick_cfg(getattr(args, "lookahead_weight", None), "lookahead_weight"),
        boundary_weight=pick_cfg(getattr(args, "boundary_weight", None), "boundary_weight"),
        cosine_decay_steps=pick_cfg(getattr(args, "cosine_decay_steps", None), "cosine_decay_steps"),
        weight_decay=pick_cfg(getattr(args, "weight_decay", None), "weight_decay"),
        grad_clip=pick_cfg(getattr(args, "grad_clip", None), "grad_clip"),
        beta_m=pick_cfg(getattr(args, "beta_m", None), "beta_m"),
        beta_v=pick_cfg(getattr(args, "beta_v", None), "beta_v"),
        epsilon=pick_cfg(getattr(args, "epsilon", None), "epsilon"),
        dtype=pick_cfg(getattr(args, "dtype", None), "dtype"),
        fp8_tile=pick_cfg(getattr(args, "fp8_tile", None), "fp8_tile"),
        state_dtype=pick_cfg(getattr(args, "state_dtype", None), "state_dtype"),
        threads=pick_cfg(getattr(args, "threads", None), "threads"),
        seed=pick_cfg(getattr(args, "seed", None), "seed"),
        rmsnorm_eps=pick_cfg(getattr(args, "rmsnorm_eps", None), "rmsnorm_eps"),
        context_length=pick(getattr(args, "context_length", None), "context_length"),
        attention_chunk_size=pick_cfg(getattr(args, "attention_chunk_size", None), "attention_chunk_size"),
        grow_every=pick_cfg(getattr(args, "config_grow_every", None), "grow_every"),
        max_new_experts=pick_cfg(getattr(args, "max_new_experts", None), "max_new_experts"),
    )


def explicit_config_fields(args: argparse.Namespace) -> dict:
    """Config fields whose CLI flag was explicitly passed (value is not None)."""
    out: dict = {}
    for field, dest in ARG_TO_CONFIG.items():
        if field == "grow_every":
            continue  # schedule lives on train --grow-every; see resolve_train_grow_every
        val = getattr(args, dest, None)
        if val is not None:
            if dest == "paging_method":
                val = str(val).upper()
            out[field] = val
    # expert_hidden alias: dest is expert_size
    return out


def load_saved_config(ckpt_dir: str) -> SmaulBrainConfig | None:
    """Checkpoint config without building a model (None when no checkpoint)."""
    cfg_path = os.path.join(ckpt_dir, "config.json")
    man_path = os.path.join(ckpt_dir, "manifest.json")
    if not (os.path.exists(cfg_path) and os.path.exists(man_path)):
        return None
    with open(cfg_path) as f:
        raw = json.load(f)
    return SmaulBrainConfig.from_dict(raw)


def validate_resume_compatible(cli_cfg: SmaulBrainConfig,
                               saved_cfg: SmaulBrainConfig,
                               explicit: dict) -> None:
    """Abort on explicit shape mismatches BEFORE constructing any model."""
    bad = []
    for field in SHAPE_FIELDS:
        if field in explicit:
            cli_v = getattr(cli_cfg, field)
            ck_v = getattr(saved_cfg, field)
            if cli_v != ck_v:
                bad.append(f"{field}: CLI={cli_v!r} != checkpoint={ck_v!r}")
    if bad:
        raise ValueError("checkpoint topology mismatch ("
                         + "; ".join(bad)
                         + "); refusing to build an incompatible model")


def resolve_resume_config(args: argparse.Namespace,
                          cli_cfg: SmaulBrainConfig,
                          saved_cfg: SmaulBrainConfig | None) -> SmaulBrainConfig:
    """Checkpoint-authoritative merge with deterministic override semantics."""
    if saved_cfg is None:
        return cli_cfg
    explicit = explicit_config_fields(args)
    validate_resume_compatible(cli_cfg, saved_cfg, explicit)
    # Runtime fields: checkpoint wins; warn on explicit mismatches so the
    # ignore is never silent. Threads/seed are process knobs: explicit wins.
    for field, cli_v in explicit.items():
        if field in SHAPE_FIELDS or field in ("threads", "seed"):
            continue
        ck_v = getattr(saved_cfg, field, None)
        if cli_v != ck_v:
            print(f"warning: ignoring --{field.replace('_', '-')}={cli_v!r} "
                  f"(checkpoint has {ck_v!r})", file=sys.stderr)
    if getattr(args, "full", False):
        print("warning: ignoring --full preset on resume (checkpoint wins)",
              file=sys.stderr)
    cfg_dict = saved_cfg.to_dict()
    for knob in ("threads", "seed"):
        if getattr(args, knob, None) is not None:
            cfg_dict[knob] = getattr(args, knob)
    # num_experts tracks the live pool; storage stamps len(pool) on save and
    # restores len(manifest expert_ids) on load, so keep the checkpoint value.
    return SmaulBrainConfig.from_dict(cfg_dict)


def resolve_train_grow_every(args: argparse.Namespace, cfg: SmaulBrainConfig) -> int:
    """Train schedule: explicit --grow-every wins, else config.grow_every."""
    v = getattr(args, "grow_every", None)
    return int(v) if v is not None else int(cfg.grow_every)


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
    cli_cfg = config_from_args(args)

    if args.cmd == "quantize":
        from quantize import convert_checkpoint
        reports = convert_checkpoint(args.ckpt, to=args.to)
        print(json.dumps({"to": args.to,
                          "converted_experts": len(reports),
                          "bytes_before": sum(r.get("bytes_before", 0) for r in reports),
                          "bytes_after": sum(r.get("bytes_after", 0) for r in reports)}, indent=2))
        return 0

    # Validate topology BEFORE constructing any model: an explicit shape
    # mismatch aborts here instead of building an incompatible model and
    # swapping part of it. Runtime fields come from the checkpoint.
    saved_cfg = load_saved_config(args.ckpt)
    try:
        cfg = resolve_resume_config(args, cli_cfg, saved_cfg)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    has_ckpt = saved_cfg is not None

    import torch
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)

    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, wd=cfg.weight_decay,
                                   beta_m=cfg.beta_m, beta_v=cfg.beta_v,
                                   eps=cfg.epsilon, clip=cfg.grad_clip,
                                   state_dtype=cfg.state_dtype))
    if has_ckpt:
        load_model(args.ckpt, model, opt)
        # Checkpoint is authoritative for runtime config; re-apply the
        # process knobs whose explicit CLI values win over the checkpoint.
        for knob in ("threads", "seed"):
            if getattr(args, knob, None) is not None:
                setattr(model.cfg, knob, getattr(args, knob))
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
        grow_every = resolve_train_grow_every(args, cfg)
        cfg.grow_every = grow_every
        grow_n = int(getattr(args, "grow_experts", 0) or 0)
        if grow_n > 0:
            import growth as _growth_mod
            start_step = int(getattr(model, "_resume_step", -1)) + 1
            if start_step == 0:
                print("note: staged new-skill growth is a resume-flow operation "
                      "(train base, then --grow-experts on the checkpoint where "
                      "start >= 1 isolates cleanly); --new-since 0 will also "
                      "select init experts",
                      file=sys.stderr)
            room = int(cfg.max_experts) - len(model.pool)
            if room <= 0:
                print(f"warning: --grow-experts {grow_n} requested but pool at cap "
                      f"({len(model.pool)}/{cfg.max_experts}); adding 0",
                      file=sys.stderr)
            else:
                want = min(grow_n, room)
                grown: list[str] = []
                base_seed = int(cfg.seed)
                it = 0
                while len(grown) < want:
                    room_now = int(cfg.max_experts) - len(model.pool)
                    if room_now <= 0:
                        break
                    need = want - len(grown)
                    new_ids = _growth_mod.grow_topk_clones(
                        model.pool, model.router, cfg.d_model, cfg.expert_hidden,
                        start_step, seed=base_seed + it, k=need,
                        fp8_tile=cfg.fp8_tile, max_experts=cfg.max_experts,
                        optim_state=opt.router_state)
                    if not new_ids:
                        break
                    grown.extend(new_ids)
                    it += 1
                model.cfg.num_experts = len(model.pool)
                cfg.num_experts = len(model.pool)
                for eid in grown:
                    print(f"[grow] step={start_step} new={eid} pool={len(model.pool)}",
                          file=sys.stderr)
                print(f"[grow] requested={grow_n} added={len(grown)} pool={len(model.pool)}",
                      file=sys.stderr)
        res = run_training(model, opt, cfg, seqs, steps=args.steps,
                           batch_size=args.batch, mode=args.mode,
                           selected=args.selected.split(",") if args.selected else None,
                           new_since_step=args.new_since,
                           ckpt_dir=args.ckpt if args.save_every else None,
                           save_every=args.save_every,
                           grow_every=grow_every,
                           grow_loss_below=args.grow_loss_below,
                           # Progress/grow lines go to stderr: stdout
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
        # Inference is continuation-only: generating from a freshly
        # initialized model would emit confidently meaningless bytes.
        if not os.path.exists(os.path.join(args.ckpt, "manifest.json")):
            print(f"error: no checkpoint at {args.ckpt}; train first",
                  file=sys.stderr)
            return 2
        # Serve future misses from the checkpoint files. load_model above
        # already cleared stale caches and staged R2VR RAM; wiring the
        # loader here must not repeat that work (double disk reads).
        from storage import make_disk_loader
        model.pager.load_from_disk = make_disk_loader(args.ckpt)
        res = generate(model, encode_text(args.prompt), max_new=args.max_new,
                       temperature=args.temperature, context=cfg.context_length,
                       seed=cfg.seed)
        print(decode_text(res["ids"]))
    elif args.cmd == "report":
        counts = model.param_counts()
        counts["paging"] = model.pager.stats.to_dict()
        counts["resident"] = model.pager.resident_counts()
        # Unambiguous count categories (see SmaulBrainConfig.describe_counts):
        # logical == unique == total here (distinct experts, shared block once).
        counts["logical_params"] = counts["total_params"]
        counts["unique_params"] = counts["total_params"]
        print(json.dumps(counts, indent=2))
    model.pager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
