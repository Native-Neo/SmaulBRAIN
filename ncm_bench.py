#!/usr/bin/env python3
"""Benchmark an experimental NCM checkpoint end to end, including byte decoding."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ncm import NCMConfig, NCMTransformer, ResidualVQTokenizer, level_length


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("ncm-checkpoints"))
    parser.add_argument("--prompt", default="Once upon a time, ")
    parser.add_argument("--new-bytes", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.new_bytes < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("new-bytes and runs must be positive; warmup must be >= 0")

    device = torch.device(args.device)
    tokenizer_state = torch.load(
        args.checkpoint_dir / "tokenizer.pt", map_location=device, weights_only=False
    )
    generator_state = torch.load(
        args.checkpoint_dir / "generator.pt", map_location=device, weights_only=False
    )
    config = NCMConfig(**tokenizer_state["config"])
    tokenizer = ResidualVQTokenizer(config).to(device)
    tokenizer.load_state_dict(tokenizer_state["model"])
    tokenizer.eval()
    generator = NCMTransformer(config, tokenizer).to(device)
    generator.load_state_dict(generator_state["model"])
    generator.eval()
    prompt_ids = list(args.prompt.encode("utf-8"))
    if not prompt_ids:
        parser.error("prompt must not be empty")
    if len(prompt_ids) > config.max_seq_len:
        parser.error("prompt is longer than configured max_seq_len")
    prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def run():
        with torch.inference_mode():
            return generator.generate(
                args.new_bytes, device=device, temperature=args.temperature,
                top_k=args.top_k, prefix_tokens=prefix,
            )

    for _ in range(args.warmup):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    output = None
    for _ in range(args.runs):
        start = time.perf_counter()
        output = run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    elapsed = sum(timings) / len(timings)
    generated = bytes(output[0].tolist())
    print(f"device: {device}")
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
    if device.type == "cuda":
        print(f"peak_allocated_mib: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f}")
    print("--- sample (UTF-8 decode uses replacement for invalid byte sequences) ---")
    print(generated.decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
