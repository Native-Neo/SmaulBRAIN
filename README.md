# SmaulBRAIN — PyTorch port

A PyTorch/autograd port of the SmaulBRAIN byte-level recurrent language model. The package retains the dense and MoE presets, tied embedding/output weights, recurrent state, stable top-k routing, expert growth/pruning, factored SmaulOPT, quantized forward paths, checkpoint recovery, inference, evaluation, and benchmarking tools.

## Install

```bash
pip install smaulbrain-core
```

For development from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch is the tensor/training runtime. NumPy remains a dependency for reading legacy `.npz` data/checkpoints and for the compatibility adapter used by the optional existing C++ kernels. Quantization and optimizer C++ kernels compile on demand if a working C++ compiler is available; the Torch implementation remains the fallback.

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

`--max-steps` is the target global step; `--steps` is the number of updates for this invocation. The final summary reports raw loss separately from the historical smoothed loss and reports the last routing overflow fraction.

Train on a raw corpus (files are read as bytes):

```bash
python train.py --preset dense-tiny --corpus ./data/corpus.bin --max-steps 50 --checkpoint checkpoints/dense
```

Generate text and evaluate:

```bash
python infer.py --preset moe-tiny --checkpoint checkpoints/demo --prompt "Hello" --max-new-tokens 64
python evaluate.py --preset moe-tiny --checkpoint checkpoints/demo --eval-batches 16
python prune.py --preset moe-tiny --checkpoint checkpoints/demo --rm-worst 1 --dry-run
```

Run checks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
```

On small CPUs, limiting BLAS/OpenMP threads avoids thread-launch overhead for the model's many small operations.

## Presets and runtime options

`--preset` accepts `dense-tiny`, `dense-full`, `moe-tiny`, and `moe-full`; `--dense` / `--moe` aliases work with `--preset tiny|full`. Choose `--forward-quant fp32|fp8|ternary`, `--backward-mode fp32|fp8|ternary`, and `--paging R|V|D2R|D2V|R2V`. VRAM modes require a supported CUDA device. FP8 is software E4M3 emulation, not a promise of hardware FP8 arithmetic.

Growth controls include `--growth-every-steps`, `--growth-loss-nats`, `--growth-loss-beta`, `--growth-min-eval-samples`, `--growth-noise-scale`, and `--growth-seed`. Use `0` to disable periodic growth and `none` to disable the loss trigger.

## Project map

- `smaulbrain/model.py`, `ops.py`, `moe.py`: PyTorch model, differentiable ops, routing.
- `smaulbrain/smaulopt.py`: custom factored optimizer and checkpointable state.
- `smaulbrain/quant.py`: FP8-emulation and packed ternary paths.
- `smaulbrain/paging.py`, `growth.py`, `prune_cmd.py`: expert storage/cache, growth, pruning.
- `smaulbrain/checkpoint.py`, `train_loop.py`: checkpointing, recovery, training loop.
- `smaulbrain/inference.py`, `data.py`, `vocab.py`: sampling, byte streams, encoding.
- `train.py`, `infer.py`, `evaluate.py`, `prune.py`, `bench.py`, `bench_router.py`: CLI tools.
- `tests/`: numerical references, hot-path checks, routing/quantization, optimizer, lifecycle and recovery tests.
- `docs/NCM.md`, `docs/systems.md`: NCM architecture, runtime settings, and known limitations.

## Default CLI architecture: NCM

`train.py` and `infer.py` now use the coarse-to-fine Next-Concept Modeling
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
packed codes and resets the old optimizer moments for those weights.

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

Training saves tokenizer and generator checkpoints separately and automatically
resumes saved training stages. For Parquet input, install `pyarrow`. The older
`ncm_train.py` and `ncm_bench.py` commands remain available for direct
experiments. The prototype is smaller and simpler than the paper's full recipe;
validate held-out text quality and end-to-end speed rather than treating the
paper's reported speedups as guaranteed.

## Compatibility and limits

New checkpoints use PyTorch `.pt` tensors with integrity checks and atomic publication. Legacy `smaulbrain-v2:1` NumPy checkpoints and per-expert `.npz` files are imported, but legacy NumPy RNG continuation is not bit-for-bit replayed. Full presets need substantial RAM; count checks do not allocate full models. See `docs/NCM.md` and `docs/systems.md` before selecting paging/quantization settings. CPU regression tests do not certify CUDA behavior or convergence/performance for full-size models.
