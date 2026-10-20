#!/usr/bin/env python3
"""Default SmaulBRAIN training entry point: NCM + SmaulOPT."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import data as _data
from ncm_train import main as _ncm_train


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Train the default SmaulBRAIN NCM model using SmaulOPT",
        allow_abbrev=False,
    )
    _cli.add_common_args(parser, need_train=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="require an existing NCM training checkpoint",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="number of additional updates per NCM stage (overrides --max-steps)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument(
        "--stage", choices=("tokenizer", "generator", "both"), default="both",
        help="NCM training stage; both is the normal end-to-end mode",
    )
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(raw_argv)

    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be >= 0")
    if args.max_steps < 0:
        parser.error("--max-steps must be >= 0")
    if min(args.batch_size, args.levels, args.codebook_size, args.save_every) < 1:
        parser.error("batch-size, levels, codebook-size and save-every must be positive")

    try:
        preset = _cli.validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    # The NCM runtime uses fixed in-memory expert pools and has no MTP or legacy
    # paging implementation yet. Reject explicit requests instead of silently lying.
    if args.paging != "R":
        parser.error("the default NCM runtime currently supports --paging R only")
    if _was_provided(raw_argv, "--mtp-depth") and args.mtp_depth != 0:
        parser.error("MTP is not implemented in the default NCM architecture; use --mtp-depth 0")
    if _was_provided(raw_argv, "--mtp-loss-weight") and args.mtp_loss_weight != 0:
        parser.error("MTP is not implemented in the default NCM architecture; use --mtp-loss-weight 0")

    checkpoint = Path(args.checkpoint)
    tokenizer_training = checkpoint / "tokenizer-training.pt"
    generator_training = checkpoint / "generator-training.pt"
    ncm_resume_exists = tokenizer_training.is_file() or generator_training.is_file()
    legacy_manifest = checkpoint / "manifest.json"
    if args.resume and not ncm_resume_exists:
        parser.error(f"no NCM training checkpoint to resume at {checkpoint}")
    if legacy_manifest.is_file() and not ncm_resume_exists:
        parser.error(
            f"{checkpoint} contains a legacy SmaulBRAIN manifest, not NCM checkpoints; "
            "choose a fresh --checkpoint directory for the default NCM architecture"
        )

    updates = args.steps if args.steps is not None else args.max_steps
    ncm_args = [
        "--output", str(checkpoint),
        "--stage", args.stage,
        "--steps", str(updates),
        "--save-every", str(args.save_every),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--preset", preset,
        "--levels", str(args.levels),
        "--codebook-size", str(args.codebook_size),
        "--forward-quant", args.forward_quant,
        "--backward-mode", args.backward_mode,
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--device", args.device,
        "--seed", str(args.seed),
    ]

    if args.corpus:
        ncm_args.extend(["--data", args.corpus])
        return _ncm_train(ncm_args)

    # Synthetic bytes stay small and temporary, matching the old entry point's
    # no-corpus smoke-training behavior without adding persistent files.
    checkpoint.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="smaulbrain-ncm-synth-", suffix=".bin",
            dir=checkpoint, delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(_data.synth_bytes(args.synth_bytes, args.seed))
        ncm_args.extend(["--data", temp_name])
        return _ncm_train(ncm_args)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
