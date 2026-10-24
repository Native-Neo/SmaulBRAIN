#!/usr/bin/env python3
"""Benchmark an experimental NCM checkpoint end to end, including byte decoding."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ncm import (
    NCMConfig, NCMGenerator, ResidualVQTokenizer, level_length,
    migrate_ternary_state_dict,
)
from ncm_checkpoint import load_checkpoint as load_ncm_checkpoint
from smaulbrain import npgrad as ng


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("ncm-checkpoints"))
    parser.add_argument("--prompt", default="Once upon a time, ")
    parser.add_argument("--new-bytes", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="cpu",
                        help="CPU-only compatibility option; tensors always run on CPU")
    args = parser.parse_args(argv)
    if args.new_bytes < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("new-bytes and runs must be positive; warmup must be >= 0")

    tokenizer_path = args.checkpoint_dir / "tokenizer.safetensors"
    generator_path = args.checkpoint_dir / "generator.safetensors"
    if not tokenizer_path.is_file() or not generator_path.is_file():
        if ((args.checkpoint_dir / "tokenizer.pt").is_file()
                or (args.checkpoint_dir / "generator.pt").is_file()):
            parser.error(
                "legacy PyTorch NCM checkpoints require "
                "tests/_reference/convert_torch_checkpoint.py"
            )
        parser.error(f"missing NCM checkpoints under {args.checkpoint_dir}")
    tokenizer_state = load_ncm_checkpoint(tokenizer_path)
    generator_state = load_ncm_checkpoint(generator_path)
    config = NCMConfig(**tokenizer_state["config"])
    tokenizer = ResidualVQTokenizer(config)
    tokenizer_model_state, _ = migrate_ternary_state_dict(
        tokenizer, tokenizer_state["model"]
    )
    tokenizer.load_state_dict(tokenizer_model_state)
    tokenizer.eval()
    generator = NCMGenerator(config, tokenizer)
    generator_model_state, _ = migrate_ternary_state_dict(
        generator, generator_state["model"]
    )
    generator.load_state_dict(generator_model_state)
    del tokenizer_state, generator_state, tokenizer_model_state, generator_model_state
    tokenizer_parameters = sum(parameter.numel() for parameter in tokenizer.parameters())
    generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
    generator.eval()
    prompt_ids = list(args.prompt.encode("utf-8"))
    if not prompt_ids:
        parser.error("prompt must not be empty")
    if len(prompt_ids) > config.max_seq_len:
        parser.error("prompt is longer than configured max_seq_len")
    prefix = ng.Tensor(np.asarray([prompt_ids], dtype=np.int64))

    def run():
        with ng.no_grad():
            return generator.generate(
                args.new_bytes, device="cpu", temperature=args.temperature,
                top_k=args.top_k, prefix_tokens=prefix,
            )

    for _ in range(args.warmup):
        run()
    timings = []
    output = None
    for _ in range(args.runs):
        start = time.perf_counter()
        output = run()
        timings.append(time.perf_counter() - start)
    elapsed = sum(timings) / len(timings)
    generated = bytes(output.data[0].tolist())
    print("device: cpu")
    print(f"tokenizer_parameters: {tokenizer_parameters:,}")
    print(f"generator_parameters: {generator_parameters:,}")
    print(f"total_architecture_parameters: {tokenizer_parameters + generator_parameters:,}")
    print(f"prompt_bytes: {len(prompt_ids)}")
    print(f"generated_bytes: {len(generated)}")
    level_lengths = [
        level_length(args.new_bytes, level, config.levels)
        for level in range(config.levels)
    ]
    print(f"hierarchy_levels: {config.levels}")
    print(f"code_positions_per_level: {level_lengths}")
    print(f"total_code_positions: {sum(level_lengths)}")
    print(f"code_positions_per_output_byte: {sum(level_lengths) / len(generated):.3f}")
    print(f"mean_latency_seconds: {elapsed:.6f}")
    print(f"generated_bytes_per_second: {len(generated) / elapsed:.2f}")
    print(f"latency_seconds_per_run: {[round(x, 6) for x in timings]}")
    print("--- sample (UTF-8 decode uses replacement for invalid byte sequences) ---")
    print(generated.decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
