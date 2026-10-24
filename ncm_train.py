#!/usr/bin/env python3
"""Train the experimental SmaulBRAIN NCM tokenizer and generator.

Accepts plain-text files/directories and Parquet files with textual columns.
Data is streamed in bounded chunks; the entire corpus is never loaded into RAM.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from smaulbrain import npgrad as ng
from smaulbrain.npgrad import Tensor

from ncm import (
    NCMConfig, NCMGenerator, NCMLinear, ResidualVQTokenizer,
    apply_direct_ternary_updates, backward_ncm_loss_chunked,
    migrate_ternary_state_dict, ncm_loss,
)
from ncm_checkpoint import (
    load_checkpoint as load_ncm_checkpoint,
    save_model_checkpoint,
    save_training_checkpoint,
)
from smaulbrain import config as _preset_config


TEXT_COLUMNS = (
    "text", "content", "response", "output", "completion", "answer",
    "prompt", "instruction", "messages", "conversation",
)


def corpus_chunks(path: Path):
    supported = {".txt", ".md", ".text", ".jsonl", ".parquet", ".bin", ".dat", ".corpus"}
    if path.is_dir():
        paths = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        )
    else:
        # A directly supplied corpus file is raw byte input regardless of extension.
        paths = [path]
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

    def next(self, device=None) -> np.ndarray:
        self._fill(self.buffer_bytes)
        max_start = len(self.buffer) - self.seq_len
        starts = np.fromiter(
            (random.randrange(max_start + 1) for _ in range(self.batch_size)),
            dtype=np.int64,
            count=self.batch_size,
        )
        raw = np.frombuffer(self.buffer, dtype=np.uint8)
        offsets = starts[:, None] + np.arange(self.seq_len, dtype=np.int64)
        batch = raw[offsets].astype(np.int64)
        del raw  # release the buffer export before shrinking the bytearray
        # Move through the stream over time instead of repeatedly sampling one fixed buffer.
        advance = min(self.batch_size * self.seq_len, max(1, len(self.buffer) // 4))
        del self.buffer[:advance]
        return batch


class SmaulTrainingOptimizer:
    """Adapt SmaulOPT's named-tensor API to NumPy module training."""

    def __init__(self, model, lr: float, weight_decay: float = 0.0):
        from smaulbrain.smaulopt import SmaulOPT

        self.params = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.params:
            raise ValueError("SmaulOPT requires at least one trainable parameter")
        self.optimizer = SmaulOPT(lr=lr, weight_decay=weight_decay)
        self.optimizer.register(self.params)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.params.values():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad[...] = 0

    def step(self) -> None:
        gradients = {
            name: parameter.grad
            for name, parameter in self.params.items()
            if parameter.grad is not None
        }
        self.optimizer.step(self.params, gradients, collect_norms=False)

    def state_dict(self) -> dict:
        return {
            "type": "SmaulOPT",
            # Live tensors (no clone): the caller serializes immediately, and a clone would
            # duplicate every moment tensor in RAM at each checkpoint.
            "state": {
                name: {key: value for key, value in state.items() if key != "factored"}
                for name, state in self.optimizer.state.items()
            },
            "factored": {
                name: bool(state["factored"])
                for name, state in self.optimizer.state.items()
            },
            "n_updates": self.optimizer.n_updates,
            "hyperparameters": {
                "lr": self.optimizer.lr,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "eps": self.optimizer.eps,
                "weight_decay": self.optimizer.weight_decay,
            },
        }

    def load_state_dict(self, payload: dict) -> None:
        if payload.get("type") == "SmaulOPT":
            for name in ("lr", "beta1", "beta2", "eps", "weight_decay"):
                if name in payload.get("hyperparameters", {}):
                    setattr(
                        self.optimizer,
                        name,
                        float(payload["hyperparameters"][name]),
                    )
            self.optimizer.load_state_dict(
                payload.get("state", {}),
                payload.get("factored", {}),
                copy=False,
            )
            self.optimizer.n_updates = int(payload.get("n_updates", 0))
            self.optimizer.register(self.params)
            return

        # Older NCM checkpoints contain torch.optim.AdamW state. Its moments
        # cannot be converted faithfully to SmaulOPT's factored second moments.
        # Keep the model weights and training step, but start fresh optimizer state.
        if isinstance(payload, dict) and "state" in payload and "param_groups" in payload:
            print(
                "warning: checkpoint uses AdamW; preserving model weights and step "
                "but initializing fresh SmaulOPT optimizer state",
                file=sys.stderr,
                flush=True,
            )
            self.optimizer.register(self.params)
            return

        raise ValueError("Unrecognized optimizer checkpoint; expected SmaulOPT state")


def build_ncm_config(
    preset: str = "custom",
    *,
    seq_len: int = 128,
    levels: int = 4,
    codebook_size: int = 64,
    heads: int = 4,
    d_model: int | None = None,
    layers: int | None = None,
    experts: int | None = None,
    top_k: int | None = None,
    moe: bool = False,
    dense: bool = False,
    forward_quant: str = "fp32",
    backward_mode: str = "fp32",
) -> NCMConfig:
    """Build an NCM configuration from one of the four repository presets or custom sizing."""
    resolved = (
        _preset_config.resolve_preset(preset, moe, dense)
        if preset != "custom" else None
    )
    if preset == "custom" and (moe or dense):
        raise ValueError("--moe/--dense aliases require --preset tiny or full")
    _preset_config.validate_quant_combo(forward_quant, backward_mode)

    common = {
        "vocab_size": 256,
        "codebook_size": codebook_size,
        "levels": levels,
        "n_heads": heads,
        "max_seq_len": seq_len,
        "dropout": 0.0,
        "forward_quant": forward_quant,
        "backward_mode": backward_mode,
    }
    if resolved is None:
        dimension = 64 if d_model is None else d_model
        layer_count = 2 if layers is None else layers
        expert_count = 4 if experts is None else experts
        route_count = 2 if top_k is None else top_k
        if min(dimension, layer_count, expert_count, route_count) < 1:
            raise ValueError("custom dimensions and expert counts must be positive")
        if route_count > expert_count:
            raise ValueError("--top-k cannot exceed --experts")
        return NCMConfig(
            **common, d_model=dimension, n_layers=layer_count,
            n_experts=expert_count, top_k=route_count, preset="custom",
        )

    spec = _preset_config.PRESETS[resolved]
    dimension = spec.d_model if d_model is None else d_model
    layer_count = spec.n_layers if layers is None else layers
    if dimension < 1 or layer_count < 1:
        raise ValueError("model dimensions and layers must be positive")
    if spec.kind == "moe":
        expert_count = spec.n_experts_init if experts is None else experts
        route_count = spec.top_k if top_k is None else top_k
        if min(expert_count, route_count) < 1 or route_count > expert_count:
            raise ValueError("preset MoE requires positive expert count and top_k <= experts")
        return NCMConfig(
            **common, d_model=dimension, n_layers=layer_count,
            model_kind="moe", n_experts=expert_count, top_k=route_count,
            expert_hidden=spec.expert_hidden, preset=resolved,
        )
    if experts is not None or top_k is not None:
        raise ValueError("dense presets do not support --experts or --top-k")
    return NCMConfig(
        **common, d_model=dimension, n_layers=layer_count,
        model_kind="dense", n_experts=0, top_k=0,
        ffn_hidden=spec.ffn_hidden, preset=resolved,
    )


def save_checkpoint(path: Path, model, optimizer, config, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_training_checkpoint(path, config, model, optimizer, step)


def load_stage_checkpoint(path: Path, model, optimizer, device=None) -> int:
    if not path.exists():
        return 0
    state = load_ncm_checkpoint(path)
    model_state, migrated_ternary = migrate_ternary_state_dict(model, state["model"])
    try:
        model.load_state_dict(model_state)
    except (RuntimeError, ValueError) as exc:
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
    del model_state  # loaded weights are copied into the module; don't hold a second set
    if migrated_ternary or state["optimizer"] is None:
        if migrated_ternary:
            print(
                "warning: migrated FP32-master ternary weights to packed discrete "
                "codes; resetting optimizer state because the old full-precision "
                "weight moments do not apply to direct ternary updates",
                file=sys.stderr,
                flush=True,
            )
        optimizer.optimizer.register(optimizer.params)
    else:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0))


def configure_direct_ternary_updates(model, learning_rate: float) -> None:
    """Configure the sparse direct-discrete update rate for packed ternary matrices."""
    update_rate = min(3e-3, max(1e-5, float(learning_rate)))
    for module in model.modules():
        if isinstance(module, NCMLinear) and module.forward_quant == "ternary":
            module.ternary_update_rate = update_rate


def architecture_parameter_count(model, *, trainable_only: bool = False) -> int:
    """Count logical parameters, including matrix entries stored as packed ternary codes."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
    total += sum(
        module.in_features * module.out_features
        for module in model.modules()
        if isinstance(module, NCMLinear)
        and module.forward_quant == "ternary"
    )
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data", required=True, type=Path, help="text/JSONL/Parquet file or directory")
    parser.add_argument("--output", type=Path, default=Path("ncm-checkpoints"))
    parser.add_argument("--stage", choices=("tokenizer", "generator", "both"), default="both")
    parser.add_argument("--steps", type=int, default=None,
                        help="additional optimizer steps per stage (overrides --max-steps; default: 1000)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="target global step per stage")
    parser.add_argument("--save-every", type=int, default=100,
                        help="write checkpoints every N optimizer steps")
    parser.add_argument("--log-every", type=int, default=1,
                        help="print training metrics every N optimizer steps")
    parser.add_argument(
        "--loss-window", type=int, default=100,
        help="number of recent training batches used for rolling loss averages",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--preset", default="custom",
        choices=("custom", "tiny", "full", "dense-tiny", "dense-full", "moe-tiny", "moe-full"),
        help="use one of the four repository presets, or custom NCM dimensions",
    )
    parser.add_argument("--moe", action="store_true", help="interpret preset tiny/full as MoE")
    parser.add_argument("--dense", action="store_true", help="interpret preset tiny/full as dense")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None, help="override preset expert count")
    parser.add_argument("--top-k", type=int, default=None, help="override preset experts activated per position")
    parser.add_argument("--forward-quant", choices=("fp32", "fp8", "ternary"), default="fp32")
    parser.add_argument("--backward-mode", choices=("fp32", "fp8", "ternary"), default="fp32")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--grad-checkpoint", action="store_true",
        help="low-memory generator training: recompute blocks in backward and "
             "release each level's activations before building the next",
    )
    parser.add_argument(
        "--heads", type=int, default=4,
        help="legacy compatibility option; ignored by the recurrent NCM generator",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu",
                        help="CPU-only compatibility option; tensors always run on CPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.steps is None and args.max_steps is None:
        args.steps = 1000
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.steps is None and args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if min(args.save_every, args.log_every, args.loss_window, args.batch_size, args.seq_len,
           args.levels, args.codebook_size, args.heads) < 1:
        parser.error("save-every, log-every, batch-size, sequence length, levels and codebook size must be positive")
    if args.seq_len < 2:
        parser.error("seq-len must be >= 2")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")
    if not args.data.exists():
        parser.error(f"data path does not exist: {args.data}")
    try:
        config = build_ncm_config(
            args.preset, seq_len=args.seq_len, levels=args.levels,
            codebook_size=args.codebook_size, heads=args.heads,
            d_model=args.d_model, layers=args.layers, experts=args.experts,
            top_k=args.top_k, moe=args.moe, dense=args.dense,
            forward_quant=args.forward_quant, backward_mode=args.backward_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    ng.manual_seed(args.seed)
    device = "cpu"
    stream = ByteBatchStream(args.data, args.seq_len, args.batch_size, 4 << 20)
    tokenizer_path = args.output / "tokenizer.safetensors"
    generator_path = args.output / "generator.safetensors"
    tokenizer_train_path = args.output / "tokenizer-training.safetensors"
    generator_train_path = args.output / "generator-training.safetensors"

    tokenizer_steps_this_run = 0
    generator_steps_this_run = 0
    last_tokenizer_loss = None
    last_generator_loss = None
    tokenizer_loss_window = deque(maxlen=args.loss_window)
    tokenizer_recon_window = deque(maxlen=args.loss_window)
    tokenizer_quant_window = deque(maxlen=args.loss_window)
    generator_loss_window = deque(maxlen=args.loss_window)

    if args.stage == "tokenizer" or (args.stage == "both" and not generator_train_path.exists()):
        tokenizer = ResidualVQTokenizer(config)
        configure_direct_ternary_updates(tokenizer, args.lr)
        optimizer = SmaulTrainingOptimizer(tokenizer, args.lr, args.weight_decay)
        start = load_stage_checkpoint(tokenizer_train_path, tokenizer, optimizer, device)
        end = start + args.steps if args.steps is not None else max(start, args.max_steps)
        tokenizer.train()
        start_time = time.perf_counter()
        # Instantaneous batch losses can be noisy, especially with random byte
        # windows and VQ codebook assignments. Keep a bounded window so logs
        # show both the current batch and the recent trend without growing RAM.
        loss_window = deque(maxlen=100)
        reconstruction_window = deque(maxlen=100)
        quantization_window = deque(maxlen=100)
        for step in range(start, end):
            batch = ng.Tensor(stream.next(device))
            _, loss, _ = tokenizer(batch)
            components = tokenizer.last_loss_components
            loss_window.append(float(loss.detach()))
            reconstruction_window.append(float(components["reconstruction"]))
            quantization_window.append(float(components["quantization"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            apply_direct_ternary_updates(tokenizer)
            ng.clip_grad_norm_(tokenizer.parameters(), 1.0)
            optimizer.step()
            tokenizer_steps_this_run += 1
            last_tokenizer_loss = float(loss.detach())
            components = tokenizer.last_loss_components
            tokenizer_loss_window.append(last_tokenizer_loss)
            tokenizer_recon_window.append(float(components["reconstruction"]))
            tokenizer_quant_window.append(float(components["quantization"]))
            if (step + 1) % args.save_every == 0 or step + 1 == end:
                save_checkpoint(tokenizer_train_path, tokenizer, optimizer, config, step + 1)
                save_model_checkpoint(tokenizer_path, config, tokenizer)
            if (step + 1) % args.log_every == 0 or step + 1 == end:
                avg_loss = sum(tokenizer_loss_window) / len(tokenizer_loss_window)
                avg_recon = sum(tokenizer_recon_window) / len(tokenizer_recon_window)
                avg_quant = sum(tokenizer_quant_window) / len(tokenizer_quant_window)
                print(
                    f"tokenizer step={step + 1} "
                    f"loss={loss.item():.4f} loss_avg100={sum(loss_window) / len(loss_window):.4f} "
                    f"recon={components['reconstruction'].item():.4f} "
                    f"recon_avg100={sum(reconstruction_window) / len(reconstruction_window):.4f} "
                    f"quant={components['quantization'].item():.4f} "
                    f"quant_avg100={sum(quantization_window) / len(quantization_window):.4f} "
                    f"window={len(loss_window)} "
                    f"elapsed={time.perf_counter() - start_time:.1f}s",
                    flush=True,
                )
        if not tokenizer_path.exists():
            save_model_checkpoint(tokenizer_path, config, tokenizer)
    else:
        if not tokenizer_path.exists():
            parser.error(f"missing tokenizer checkpoint: {tokenizer_path}")
        tokenizer_state = load_ncm_checkpoint(tokenizer_path)
        try:
            saved_config = NCMConfig(**tokenizer_state["config"])
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid tokenizer checkpoint configuration: {exc}")
        if saved_config != config:
            parser.error("CLI model settings do not match the saved tokenizer configuration")
        tokenizer = ResidualVQTokenizer(config)
        tokenizer_model_state, migrated_ternary = migrate_ternary_state_dict(
            tokenizer, tokenizer_state["model"]
        )
        tokenizer.load_state_dict(tokenizer_model_state)
        del tokenizer_state, tokenizer_model_state
        if migrated_ternary:
            print(
                "Migrated tokenizer FP32-master ternary weights to packed discrete codes.",
                file=sys.stderr,
                flush=True,
            )

    tokenizer_count = architecture_parameter_count(tokenizer)
    print(f"tokenizer_parameters: {tokenizer_count:,}", flush=True)

    if args.stage in ("generator", "both"):
        if args.stage == "both":
            tokenizer_state = load_ncm_checkpoint(tokenizer_path)
            tokenizer.load_state_dict(tokenizer_state["model"])
            del tokenizer_state
        tokenizer.eval()
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        generator = NCMGenerator(
            config, tokenizer, gradient_checkpointing=args.grad_checkpoint
        )
        configure_direct_ternary_updates(generator, args.lr)
        generator_count = architecture_parameter_count(generator)
        trainable_count = architecture_parameter_count(generator, trainable_only=True)
        print(
            f"generator_parameters: {generator_count:,} "
            f"trainable_generator_parameters: {trainable_count:,} "
            f"total_architecture_parameters: {tokenizer_count + generator_count:,}",
            flush=True,
        )
        optimizer = None  # release the tokenizer-stage optimizer state before building the generator's
        optimizer = SmaulTrainingOptimizer(generator, args.lr, args.weight_decay)
        start = load_stage_checkpoint(generator_train_path, generator, optimizer, device)
        end = start + args.steps if args.steps is not None else max(start, args.max_steps)
        generator.train()
        start_time = time.perf_counter()
        for step in range(start, end):
            batch = stream.next(device)
            # Train continuation: condition on the first half, predict the second half.
            split = max(1, batch.shape[1] // 2)
            prefix_tokens = ng.Tensor(batch[:, :split])
            target_tokens = ng.Tensor(batch[:, split:])
            with ng.no_grad():
                codes = tokenizer.encode(target_tokens)
            optimizer.zero_grad(set_to_none=True)
            if args.grad_checkpoint:
                # Memory-first path: recompute blocks in backward and release
                # each level's activations before building the next one.
                loss = backward_ncm_loss_chunked(
                    generator, codes, prefix_tokens=prefix_tokens
                )
            else:
                loss = ncm_loss(generator, codes, prefix_tokens=prefix_tokens)
                loss.backward()
            apply_direct_ternary_updates(generator)
            ng.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            generator_steps_this_run += 1
            last_generator_loss = float(loss.detach())
            generator_loss_window.append(last_generator_loss)
            if (step + 1) % args.save_every == 0 or step + 1 == end:
                save_checkpoint(generator_train_path, generator, optimizer, config, step + 1)
                save_model_checkpoint(generator_path, config, generator)
            if (step + 1) % args.log_every == 0 or step + 1 == end:
                avg_loss = sum(generator_loss_window) / len(generator_loss_window)
                print(
                    f"generator step={step + 1} loss={loss.item():.4f} "
                    f"avg_loss={avg_loss:.4f} avg_n={len(generator_loss_window)} "
                    f"elapsed={time.perf_counter() - start_time:.1f}s",
                    flush=True,
                )
    print(json.dumps({
        "done": True,
        "preset": config.preset,
        "optimizer": "SmaulOPT",
        "tokenizer_steps_this_run": tokenizer_steps_this_run,
        "generator_steps_this_run": generator_steps_this_run,
        "steps_this_run": max(tokenizer_steps_this_run, generator_steps_this_run),
        "last_loss": (
            last_generator_loss
            if last_generator_loss is not None
            else last_tokenizer_loss
        ),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
