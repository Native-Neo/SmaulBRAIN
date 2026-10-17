# SmaulBRAIN — PyTorch port

A PyTorch/autograd port of the SmaulBRAIN byte-level recurrent language model. The package retains the dense and MoE presets, tied embedding/output weights, recurrent state, stable top-k routing, expert growth/pruning, factored SmaulOPT, quantized forward paths, checkpoint recovery, inference, evaluation, and benchmarking tools.

## Install

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
- `AUDIT.md`: conversion map, results, and known limitations.

## Compatibility and limits

New checkpoints use PyTorch `.pt` tensors with integrity checks and atomic publication. Legacy `smaulbrain-v2:1` NumPy checkpoints and per-expert `.npz` files are imported, but legacy NumPy RNG continuation is not bit-for-bit replayed. Full presets need substantial RAM; count checks do not allocate full models. See `AUDIT.md` and `docs/systems.md` before selecting paging/quantization settings. CPU regression tests do not certify CUDA behavior or convergence/performance for full-size models.
