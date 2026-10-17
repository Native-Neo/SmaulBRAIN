"""Centralized CLI: every entry point uses these definitions + validation."""
import argparse
import math

from . import config as _cfg


def add_common_args(ap: argparse.ArgumentParser, need_train: bool = False):
    ap.add_argument("--preset", default="moe-tiny",
                    help="full|tiny (with --moe/--dense) or moe-full|moe-tiny|dense-full|dense-tiny")
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--paging", default="R", choices=list(_cfg.PAGING_MODES))
    ap.add_argument("--paging-cache-experts", type=int, default=8)
    ap.add_argument("--paging-staging-bytes", type=int, default=1 << 26)
    ap.add_argument("--forward-quant", default="fp32", choices=list(_cfg.FORWARD_QUANT_CHOICES))
    ap.add_argument("--backward-mode", default="fp32", choices=list(_cfg.BACKWARD_MODE_CHOICES))
    ap.add_argument("--device", default="cpu", help="cpu (accelerator optional, must not break CPU)")
    ap.add_argument("--checkpoint", default="checkpoints/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--growth-every-steps", default="10000")
    ap.add_argument("--growth-loss-nats", default="0.5")
    ap.add_argument("--growth-loss-beta", type=float, default=0.98,
                    help="EMA decay for loss smoothing")
    ap.add_argument("--growth-min-eval-samples", type=int, default=256)
    ap.add_argument("--growth-noise-scale", type=float, default=0.01,
                    help="Gaussian mutation sigma as fraction of source RMS")
    ap.add_argument("--growth-seed", type=int, default=None)
    ap.add_argument("--expert-capacity-factor", type=float, default=1.25)
    if need_train:
        ap.add_argument("--max-steps", type=int, default=50)
        ap.add_argument("--corpus", default=None, help="path to raw byte corpus")
        ap.add_argument("--synth-bytes", type=int, default=1 << 16)
        ap.add_argument("--grad-clip", type=float, default=1.0)
        ap.add_argument("--log-every", type=int, default=10)
        ap.add_argument("--ckpt-every", type=int, default=100,
                            help="save a recovery checkpoint every N optimizer updates (0 disables periodic saves)")


def validate_args(args) -> str:
    preset = _cfg.resolve_preset(args.preset, args.moe, args.dense)
    _cfg.validate_quant_combo(args.forward_quant, args.backward_mode)
    if args.paging not in _cfg.PAGING_MODES:
        raise ValueError(f"--paging must be one of {_cfg.PAGING_MODES}")
    args.growth_every_steps = _cfg.parse_growth_every_steps(args.growth_every_steps)
    args.growth_loss_nats = _cfg.parse_growth_loss_nats(args.growth_loss_nats)
    if preset.startswith("dense") and (args.growth_every_steps or args.growth_loss_nats is not None):
        # Growth flags accepted but MoE growth is disabled for dense models.
        pass
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be a finite number > 0")
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        raise ValueError("--beta1 and --beta2 must be in [0, 1)")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be a finite number > 0")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be a finite number >= 0")
    if not math.isfinite(args.growth_loss_beta) or not 0 <= args.growth_loss_beta < 1:
        raise ValueError("--growth-loss-beta must be finite and in [0, 1)")
    if args.growth_min_eval_samples < 1:
        raise ValueError("--growth-min-eval-samples must be >= 1")
    if not math.isfinite(args.growth_noise_scale) or args.growth_noise_scale < 0:
        raise ValueError("--growth-noise-scale must be finite and >= 0")
    if args.paging_cache_experts < 1:
        raise ValueError("--paging-cache-experts must be >= 1")
    if args.paging_staging_bytes < 1:
        raise ValueError("--paging-staging-bytes must be >= 1")
    if not math.isfinite(args.expert_capacity_factor) or args.expert_capacity_factor <= 0:
        raise ValueError("--expert-capacity-factor must be finite and > 0")
    if hasattr(args, "max_steps"):
        if args.max_steps < 0:
            raise ValueError("--max-steps must be >= 0")
        if not args.corpus and args.synth_bytes < args.seq_len + 1:
            raise ValueError("--synth-bytes must be at least --seq-len + 1 when --corpus is not set")
        if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
            raise ValueError("--grad-clip must be a finite number > 0")
        if args.log_every < 0:
            raise ValueError("--log-every must be >= 0")
        if args.ckpt_every < 0:
            raise ValueError("--ckpt-every must be >= 0")
    return preset
