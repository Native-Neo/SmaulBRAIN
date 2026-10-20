#!/usr/bin/env python3
"""Train the experimental SmaulBRAIN NCM tokenizer and generator.

Accepts plain-text files/directories and Parquet files with textual columns.
Data is streamed in bounded chunks; the entire corpus is never loaded into RAM.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from ncm import NCMConfig, NCMGenerator, ResidualVQTokenizer, ncm_loss


TEXT_COLUMNS = (
    "text", "content", "response", "output", "completion", "answer",
    "prompt", "instruction", "messages", "conversation",
)


def corpus_chunks(path: Path):
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    paths = [p for p in paths if p.suffix.lower() in {".txt", ".md", ".text", ".jsonl", ".parquet"}]
    if not paths:
        raise ValueError(f"No supported text or Parquet files found under {path}")
    for file_path in paths:
        if file_path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Parquet input needs pyarrow: pip install pyarrow") from exc
            parquet = pq.ParquetFile(file_path)
            names = parquet.schema_arrow.names
            ordered = sorted(
                names,
                key=lambda name: (TEXT_COLUMNS.index(name.lower()) if name.lower() in TEXT_COLUMNS
                                  else len(TEXT_COLUMNS)),
            )
            preferred = [name for name in ordered if name.lower() in TEXT_COLUMNS]
            if preferred:
                ordered = preferred
            for batch in parquet.iter_batches(batch_size=128, columns=ordered):
                columns = {name.lower(): batch.column(name).to_pylist() for name in ordered}
                for row in range(batch.num_rows):
                    parts = []
                    for name in ordered:
                        value = columns[name.lower()][row]
                        if value is None:
                            continue
                        if isinstance(value, (str, bytes)):
                            text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
                        else:
                            text = str(value)
                        if text.strip():
                            parts.append(text)
                    if parts:
                        yield ("\n".join(parts) + "\n").encode("utf-8", errors="ignore")
        else:
            with file_path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    yield chunk


class ByteBatchStream:
    """Random fixed-length byte windows from a bounded streaming corpus buffer."""

    def __init__(self, path: Path, seq_len: int, batch_size: int, buffer_bytes: int):
        self.path = path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_bytes = max(buffer_bytes, seq_len * batch_size * 4)
        self.buffer = bytearray()
        self.pending = b""
        self.source = iter(corpus_chunks(path))
        self._fill(self.seq_len + 1)

    def _fill(self, minimum: int) -> None:
        while len(self.buffer) < minimum:
            if self.pending:
                chunk, self.pending = self.pending, b""
            else:
                try:
                    chunk = next(self.source)
                except StopIteration:
                    self.source = iter(corpus_chunks(self.path))
                    try:
                        chunk = next(self.source)
                    except StopIteration as exc:
                        raise ValueError("Corpus contains no readable text") from exc
            room = self.buffer_bytes - len(self.buffer)
            if room > 0:
                self.buffer.extend(chunk[:room])
                if len(chunk) > room:
                    self.pending = chunk[room:]
            if len(self.buffer) >= self.buffer_bytes:
                break
        if len(self.buffer) < self.seq_len + 1:
            raise ValueError("Corpus must contain at least seq_len + 1 bytes")

    def next(self, device: torch.device) -> Tensor:
        self._fill(self.buffer_bytes)
        max_start = len(self.buffer) - self.seq_len
        starts = [random.randrange(max_start + 1) for _ in range(self.batch_size)]
        batch = torch.tensor(
            [list(self.buffer[start:start + self.seq_len]) for start in starts],
            dtype=torch.long, device=device,
        )
        # Move through the stream over time instead of repeatedly sampling one fixed buffer.
        advance = min(self.batch_size * self.seq_len, max(1, len(self.buffer) // 4))
        del self.buffer[:advance]
        return batch


def make_adamw(parameters, lr: float) -> torch.optim.AdamW:
    """AdamW, fused when this build supports it (CPU: 2.43 -> 0.61 ms/step measured)."""
    parameters = list(parameters)
    try:
        return torch.optim.AdamW(parameters, lr=lr, fused=True)
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(parameters, lr=lr)


def save_checkpoint(path: Path, model, optimizer, config, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)


def load_stage_checkpoint(path: Path, model, optimizer, device) -> int:
    if not path.exists():
        return 0
    state = torch.load(path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(state["model"])
    except RuntimeError as exc:
        if isinstance(model, ResidualVQTokenizer):
            reason = (
                "The tokenizer architecture changed; choose a fresh --output "
                "directory to start a new run."
            )
        else:
            reason = (
                "The recurrent NCM generator uses different weights from the "
                "older Transformer-based prototype; choose a fresh --output "
                "directory to start a new run."
            )
        raise RuntimeError(f"Checkpoint {path} is incompatible: {reason}") from exc
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data", required=True, type=Path, help="text/JSONL/Parquet file or directory")
    parser.add_argument("--output", type=Path, default=Path("ncm-checkpoints"))
    parser.add_argument("--stage", choices=("tokenizer", "generator", "both"), default="both")
    parser.add_argument("--steps", type=int, default=1000, help="optimizer steps per stage")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4, help="size of the shared expert pool")
    parser.add_argument("--top-k", type=int, default=2, help="experts activated per position")
    parser.add_argument(
        "--heads", type=int, default=4,
        help="legacy compatibility option; ignored by the recurrent NCM generator",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.steps, args.save_every, args.batch_size, args.seq_len, args.d_model,
           args.levels, args.codebook_size, args.layers, args.heads,
           args.experts, args.top_k) < 1:
        parser.error("steps, save-every, batch-size and model dimensions must be positive")
    if args.top_k > args.experts:
        parser.error("--top-k cannot exceed --experts")
    if args.seq_len < 2:
        parser.error("seq-len must be >= 2")
    if not args.data.exists():
        parser.error(f"data path does not exist: {args.data}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = NCMConfig(
        vocab_size=256, d_model=args.d_model, codebook_size=args.codebook_size,
        levels=args.levels, n_layers=args.layers, n_heads=args.heads,
        n_experts=args.experts, top_k=args.top_k,
        max_seq_len=args.seq_len, dropout=0.0,
    )
    stream = ByteBatchStream(args.data, args.seq_len, args.batch_size, 4 << 20)
    tokenizer_path = args.output / "tokenizer.pt"
    generator_path = args.output / "generator.pt"
    tokenizer_train_path = args.output / "tokenizer-training.pt"
    generator_train_path = args.output / "generator-training.pt"

    if args.stage in ("tokenizer", "both"):
        tokenizer = ResidualVQTokenizer(config).to(device)
        optimizer = make_adamw(tokenizer.parameters(), args.lr)
        start = load_stage_checkpoint(tokenizer_train_path, tokenizer, optimizer, device)
        tokenizer.train()
        start_time = time.perf_counter()
        for step in range(start, start + args.steps):
            batch = stream.next(device)
            _, loss, _ = tokenizer(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % args.save_every == 0 or step + 1 == start + args.steps:
                save_checkpoint(tokenizer_train_path, tokenizer, optimizer, config, step + 1)
                torch.save({"config": config.__dict__, "model": tokenizer.state_dict()}, tokenizer_path)
                components = tokenizer.last_loss_components
                print(
                    f"tokenizer step={step + 1} loss={loss.item():.4f} "
                    f"recon={components['reconstruction'].item():.4f} "
                    f"quant={components['quantization'].item():.4f} "
                    f"elapsed={time.perf_counter() - start_time:.1f}s",
                    flush=True,
                )
        if not tokenizer_path.exists():
            torch.save({"config": config.__dict__, "model": tokenizer.state_dict()}, tokenizer_path)
    else:
        if not tokenizer_path.exists():
            parser.error(f"missing tokenizer checkpoint: {tokenizer_path}")
        tokenizer_state = torch.load(tokenizer_path, map_location=device, weights_only=False)
        if tokenizer_state["config"] != config.__dict__:
            parser.error("CLI model settings do not match the saved tokenizer configuration")
        tokenizer = ResidualVQTokenizer(config).to(device)
        tokenizer.load_state_dict(tokenizer_state["model"])

    tokenizer_count = sum(parameter.numel() for parameter in tokenizer.parameters())
    print(f"tokenizer_parameters: {tokenizer_count:,}", flush=True)

    if args.stage in ("generator", "both"):
        if args.stage == "both":
            tokenizer_state = torch.load(tokenizer_path, map_location=device, weights_only=False)
            tokenizer.load_state_dict(tokenizer_state["model"])
        tokenizer.eval()
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        generator = NCMGenerator(config, tokenizer).to(device)
        generator_count = sum(parameter.numel() for parameter in generator.parameters())
        trainable_count = sum(
            parameter.numel() for parameter in generator.parameters()
            if parameter.requires_grad
        )
        print(
            f"generator_parameters: {generator_count:,} "
            f"trainable_generator_parameters: {trainable_count:,} "
            f"total_architecture_parameters: {tokenizer_count + generator_count:,}",
            flush=True,
        )
        optimizer = make_adamw(
            (p for p in generator.parameters() if p.requires_grad), args.lr
        )
        start = load_stage_checkpoint(generator_train_path, generator, optimizer, device)
        generator.train()
        start_time = time.perf_counter()
        for step in range(start, start + args.steps):
            batch = stream.next(device)
            # Train continuation: condition on the first half, predict the second half.
            split = max(1, batch.shape[1] // 2)
            prefix_tokens = batch[:, :split]
            target_tokens = batch[:, split:]
            with torch.no_grad():
                codes = tokenizer.encode(target_tokens)
            loss = ncm_loss(generator, codes, prefix_tokens=prefix_tokens)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % args.save_every == 0 or step + 1 == start + args.steps:
                save_checkpoint(generator_train_path, generator, optimizer, config, step + 1)
                torch.save({"config": config.__dict__, "model": generator.state_dict()}, generator_path)
                print(f"generator step={step + 1} loss={loss.item():.4f} "
                      f"elapsed={time.perf_counter() - start_time:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
