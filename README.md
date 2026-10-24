# SmaulBRAIN — NumPy port

A pure-NumPy port of the SmaulBRAIN byte-level recurrent language model. The
runtime uses a small tape-based autograd engine (`smaulbrain.npgrad`) from the
sibling `smaulbrain-core` package and runs on CPU in float32. The port retains
the dense and MoE presets, tied embedding/output weights, recurrent state,
stable top-k routing, expert growth/pruning, factored SmaulOPT, quantized
forward paths, checkpoint recovery, inference, evaluation, and benchmarking
tools.

## Install

From a checkout with `smaulbrain-core` as a sibling directory:

```bash
python -m pip install -e ../smaulbrain-core
python -m pip install -r requirements.txt
```

The editable install makes `smaulbrain` importable. If you prefer not to
install it, run commands with `PYTHONPATH=.:../smaulbrain-core`.

NumPy is the tensor/training runtime and safetensors is the checkpoint format.
Quantization and optimizer C++ kernels compile on demand if a working C++
compiler is available; the NumPy implementation remains the fallback. PyTorch
is only needed to run the archived parity oracles under `tests/_reference/`.

## Quick start

Train a tiny MoE model on reproducible synthetic bytes:

```bash
python train.py --preset moe-tiny --max-steps 50 --checkpoint checkpoints/demo
```

Continue to a final global step:

```bash
python train.py --preset moe-tiny --max-steps 100 --checkpoint checkpoints/demo --resume
```

Run a fixed number of additional optimizer updates after resuming:

```bash
python train.py --preset moe-tiny --steps 10 --checkpoint checkpoints/demo --resume
```

`--max-steps` is the target global step; `--steps` is the number of updates for
this invocation. The final summary reports raw loss separately from the
historical smoothed loss and reports the last routing overflow fraction.

Train on a raw corpus (files are read as bytes):

```bash
python train.py --preset dense-tiny --corpus ./data/corpus.bin --max-steps 50 --checkpoint checkpoints/dense
```

Generate text and evaluate:

```bash
python infer.py --preset moe-tiny --checkpoint checkpoints/demo --prompt "Hello" --max-new-tokens 64
python evaluate.py --preset moe-tiny --checkpoint checkpoints/legacy-core --eval-batches 16
python prune.py --preset moe-tiny --checkpoint checkpoints/legacy-core --rm-worst 1 --dry-run
```

`evaluate.py` and `prune.py` operate on the original core checkpoint format
(`manifest.json` plus safetensors payloads), not on the default NCM checkpoint
directory.

Run checks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
```

The default suite is NumPy-only and skips `tests/_reference/`. The archived
PyTorch parity oracle can be run separately when PyTorch is installed:

```bash
PYTHONPATH=.:../smaulbrain-core python -m pytest -q tests/_reference/test_ncm_parity.py
```

On small CPUs, limiting BLAS/OpenMP threads avoids thread-launch overhead for
the model's many small operations.

## Presets and runtime options

`--preset` accepts `dense-tiny`, `dense-full`, `moe-tiny`, and `moe-full`;
`--dense` / `--moe` aliases work with `--preset tiny|full`. Choose
`--forward-quant fp32|fp8|ternary` and `--backward-mode fp32|fp8|ternary`. The
port is CPU-only: `--device cpu` is the supported setting, and the default NCM
training and inference paths accept `--paging R` only. FP8 is software E4M3
emulation, not a promise of hardware FP8 arithmetic.

The original core runtime exposes `--paging R|V|D2R|D2V|R2V` and expert
growth controls (`--growth-every-steps`, `--growth-loss-nats`,
`--growth-loss-beta`, `--growth-min-eval-samples`, `--growth-noise-scale`,
`--growth-seed`). VRAM paging modes are not available in this CPU-only port.
The default NCM architecture uses a fixed in-memory expert pool, so NCM training
does not grow or prune experts.

## Project map

- `ncm.py`: NumPy NCM tokenizer, generator, ternary/FP8 linear paths, losses.
- `ncm_train.py`, `ncm_checkpoint.py`: NCM training loop and safetensors checkpoints.
- `train.py`, `infer.py`: default NCM training and inference entry points.
- `evaluate.py`, `prune.py`: evaluation and pruning for core-format checkpoints.
- `bench.py`, `bench_router.py`, `ncm_bench.py`: throughput/benchmark tools.
- `smaulbrain-core/smaulbrain/`: NumPy model, autograd, ops, routing, SmaulOPT,
  quantization, paging, growth, checkpointing, and inference runtime.
- `tests/`: NumPy runtime tests; `tests/_reference/`: archived PyTorch oracles.
- `docs/NCM.md`, `docs/systems.md`: NCM architecture, runtime settings, and known limitations.

## Default CLI architecture: NCM

`train.py` and `infer.py` use the coarse-to-fine Next-Concept Modeling
architecture by default. It combines a multiscale residual-VQ byte tokenizer
with a bidirectional recurrent generator, RMSNorm, residual connections, and
either a dense SwiGLU mixer or a shared sparse MoE pool. Positions within each
code level are predicted in parallel. This is an NCM-specific adaptation, not
the original causal next-byte model, and it has a separate checkpoint format.

All four repository presets are supported: `dense-tiny`, `dense-full`,
`moe-tiny`, and `moe-full`. They reuse the existing preset widths, layer
counts, dense FFN sizes, expert pool sizes, top-k values, and expert hidden
widths; tokenizer and hierarchy parameters mean NCM total parameter counts
differ from the original model's exact counts.

The default optimizer for floating-point parameters is **SmaulOPT**. NCM
linear layers support `fp32`, `fp8`, and packed `ternary` paths. When
`--forward-quant ternary` is selected, the linear matrix weights are stored as
packed 2-bit ternary codes; there is no persistent FP32 master-weight tensor or
dense accumulated weight-gradient tensor for those matrices. Backpropagation
computes a transient gradient signal and queues sparse gradient-sign updates
that modify the packed codes directly after the backward pass. Biases,
embeddings, codebooks, and normalization parameters remain floating point.

Direct ternary updates are an experimental discrete optimizer, not conventional
STE/QAT, and their convergence and generated-text quality must be measured.
Loading an older FP32-master ternary checkpoint converts its linear weights to
packed codes and resets that stage's optimizer state, which is incompatible with
the direct-discrete weight update path.

Long-sequence training normally retains activations for every generator block
and every hierarchy level, so peak memory grows with depth x sequence length x
levels. Pass `--grad-checkpoint` for the low-memory path: each block is
recomputed during backward and each level's activations are released before the
next level is built. Peak memory then tracks depth plus a single level instead
of their product. The recomputed path is mathematically identical to the plain
one (same loss and, up to float round-off, the same gradients), so it is safe to
resume a plain run with checkpointing enabled.

```bash
python train.py --preset dense-full --seq-len 8192 --batch-size 1 \
  --forward-quant ternary --backward-mode ternary --grad-checkpoint \
  --steps 100 --checkpoint checkpoints/ncm-long
```

Train from a raw-byte corpus, text files, JSONL, or Parquet:

```bash
python train.py --preset moe-tiny --corpus ./data/corpus.bin \
  --steps 100 --checkpoint checkpoints/ncm-run
python train.py --preset moe-tiny --corpus ./data/corpus.bin \
  --steps 100 --checkpoint checkpoints/ncm-ternary \
  --forward-quant ternary --backward-mode ternary
python infer.py --checkpoint checkpoints/ncm-run \
  --prompt "Once upon a time, " --max-new-tokens 128
```

Training writes model and training checkpoints as safetensors files:
`tokenizer.safetensors`, `generator.safetensors`,
`tokenizer-training.safetensors`, and `generator-training.safetensors`.
Small JSON metadata (config, step, optimizer hyperparameters) lives in the
safetensors header. Training resumes saved stages automatically. For Parquet
input, install `pyarrow`. The older `ncm_train.py` and `ncm_bench.py` commands
remain available for direct experiments. The prototype is smaller and simpler
than the paper's full recipe; validate held-out text quality and end-to-end
speed rather than treating the paper's reported speedups as guaranteed.

## Compatibility and limits

New NCM checkpoints are safetensors containers with integrity metadata and
atomic publication. The original core runtime writes a `manifest.json` plus
`params.safetensors` / `optim.safetensors`; legacy `smaulbrain-v2:1` NumPy
checkpoints and per-expert `.npz` files are still imported. Legacy PyTorch NCM
`.pt` files must be converted first with
`tests/_reference/convert_torch_checkpoint.py`. PyTorch RNG streams are not
reproduced: loaded weights are exact, but freshly seeded runs differ from the
original implementation. Full presets need substantial RAM; count checks do not
allocate full models. See `docs/NCM.md` and `docs/systems.md` before selecting
quantization settings. CPU regression tests do not certify convergence or
performance for full-size models.
