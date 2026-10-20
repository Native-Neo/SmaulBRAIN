#!/usr/bin/env python3
"""Default SmaulBRAIN inference entry point for coarse-to-fine NCM."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from ncm import NCMConfig, NCMGenerator, ResidualVQTokenizer


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _select_inference_config(saved: NCMConfig, args, argv) -> NCMConfig:
    forward_quant = (
        args.forward_quant
        if _was_provided(argv, "--forward-quant")
        else saved.forward_quant
    )
    backward_mode = (
        args.backward_mode
        if _was_provided(argv, "--backward-mode")
        else saved.backward_mode
    )
    if _was_provided(argv, "--forward-quant") and not _was_provided(argv, "--backward-mode"):
        if forward_quant == "fp8":
            backward_mode = "fp8"
        elif forward_quant == "fp32":
            backward_mode = "fp32"
    if _was_provided(argv, "--backward-mode") and not _was_provided(argv, "--forward-quant"):
        if backward_mode == "fp8":
            forward_quant = "fp8"
        elif backward_mode == "ternary":
            forward_quant = "ternary"
        else:
            forward_quant = "fp32"
    return replace(saved, forward_quant=forward_quant, backward_mode=backward_mode)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run default SmaulBRAIN NCM inference",
        allow_abbrev=False,
    )
    _cli.add_common_args(parser)
    parser.set_defaults(seed=None)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prefill-chunk", type=int, default=0,
        help="legacy argument; NCM encodes the complete prefix at once",
    )
    parser.add_argument("--stop-eos", action="store_true")
    args = parser.parse_args(raw_argv)

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.prefill_chunk < 0:
        parser.error("--prefill-chunk must be >= 0")
    if args.prefill_chunk > 0:
        parser.error("--prefill-chunk is not supported by the NCM runtime")
    if args.temperature < 0 or not math.isfinite(args.temperature):
        parser.error("--temperature must be a finite number >= 0")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if not 0 < args.top_p <= 1 or not math.isfinite(args.top_p):
        parser.error("--top-p must be finite and in (0, 1]")
    if args.stop_eos:
        parser.error("NCM's byte tokenizer has no EOS token; --stop-eos is not supported")
    if args.paging != "R":
        parser.error("the default NCM runtime currently supports --paging R only")

    checkpoint_dir = Path(args.checkpoint)
    tokenizer_path = checkpoint_dir / "tokenizer.pt"
    generator_path = checkpoint_dir / "generator.pt"
    if not tokenizer_path.is_file() or not generator_path.is_file():
        if (checkpoint_dir / "manifest.json").is_file():
            parser.error(
                f"{checkpoint_dir} contains a legacy SmaulBRAIN checkpoint, not NCM checkpoints"
            )
        parser.error(
            f"expected NCM checkpoints at {tokenizer_path} and {generator_path}; "
            "train the default model first with train.py"
        )

    try:
        tokenizer_state = torch.load(
            tokenizer_path, map_location=args.device, weights_only=True
        )
        generator_state = torch.load(
            generator_path, map_location=args.device, weights_only=True
        )
        saved_config = NCMConfig(**tokenizer_state["config"])
        generator_config = NCMConfig(**generator_state["config"])
        if saved_config != generator_config:
            parser.error("tokenizer and generator checkpoints use different NCM configurations")
        config = _select_inference_config(saved_config, args, raw_argv)
        preset = saved_config.preset
        if _was_provided(raw_argv, "--preset") or args.moe or args.dense:
            requested_preset = _cfg.resolve_preset(args.preset, args.moe, args.dense)
            if preset not in {"custom", requested_preset}:
                parser.error(
                    f"checkpoint preset {preset!r} does not match requested preset "
                    f"{requested_preset!r}"
                )
            preset = requested_preset
        _cfg.validate_quant_combo(config.forward_quant, config.backward_mode)

        tokenizer = ResidualVQTokenizer(config).to(args.device)
        tokenizer.load_state_dict(tokenizer_state["model"])
        tokenizer.eval()
        generator = NCMGenerator(config, tokenizer).to(args.device)
        generator.load_state_dict(generator_state["model"])
        generator.eval()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(f"unable to load NCM checkpoint: {exc}")

    if args.prompt_file:
        with open(args.prompt_file, "rb") as handle:
            prompt_bytes = handle.read()
    else:
        prompt_bytes = args.prompt.encode("utf-8", errors="surrogateescape")
    if not prompt_bytes:
        parser.error("prompt must contain at least one byte")
    if len(prompt_bytes) > config.max_seq_len:
        parser.error(
            f"prompt has {len(prompt_bytes)} bytes, exceeding checkpoint limit "
            f"{config.max_seq_len}"
        )
    if args.max_new_tokens > config.max_seq_len:
        parser.error(
            f"--max-new-tokens cannot exceed the checkpoint limit {config.max_seq_len}"
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
    prefix = torch.tensor(
        [list(prompt_bytes)], dtype=torch.long, device=torch.device(args.device)
    )
    start = __import__("time").perf_counter()
    with torch.inference_mode():
        output = generator.generate(
            args.max_new_tokens,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            prefix_tokens=prefix,
        )
    elapsed = __import__("time").perf_counter() - start
    generated = bytes(output[0].tolist())
    print(json.dumps({
        "preset": preset,
        "model_kind": config.model_kind,
        "forward_quant": config.forward_quant,
        "backward_mode": config.backward_mode,
        "prompt_bytes": len(prompt_bytes),
        "generated_bytes": len(generated),
        "generation_s": elapsed,
        "bytes_per_s": len(generated) / elapsed if elapsed > 0 else 0.0,
        "text_utf8": generated.decode("utf-8", errors="replace"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
