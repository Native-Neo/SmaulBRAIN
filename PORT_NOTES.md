# Port notes — SmaulBRAIN

The runtime is pure NumPy plus the Python standard library. The NCM
implementation uses `smaulbrain.npgrad` and the safetensors reader/writer from
the sibling `smaulbrain-core` package. PyTorch remains only under
`tests/_reference/` as a development oracle.

## Parity results

`tests/_reference/test_ncm_parity.py` compares the original PyTorch NCM oracle
with this port using identical weights:

- Tokenizer forward: loss max abs error `9.9e-6`; logits max abs error
  `1.63e-4` (tolerance `rtol=1e-4`, `atol=3e-4`); codes are identical.
- Generator forward (MoE): logits max abs error `7.0e-5`; dense logits max abs
  error `3.6e-7`.
- Generator loss: max abs error `7.7e-6`.
- Full generator gradients: max abs error `8.8e-6` (tolerance `rtol=2e-4`,
  `atol=1e-4`).
- Chunked loss/backward parity also passes.

## Numeric deviations

- **float32 runtime.** Parameters, activations, gradients, and optimizer state
  are `float32`; reductions that the original accumulated in `float64` remain
  `float64`. Matrix-multiplication summation order differs between PyTorch and
  NumPy/BLAS, which accounts for the small forward differences above.
- **Recurrent scan.** The original NCM scan is replaced by the core chunked
  closed-form `npgrad.scan` with an explicit backward. The chunk size adapts to
  the decay range, so summation order differs from a token-by-token loop. The
  math and learned states are unchanged.
- **Interpolation.** `F.interpolate(..., mode="linear", align_corners=False)`
  and `area` / adaptive pooling are reimplemented in `npgrad` and covered by the
  parity suite. Half-pixel edge sampling follows PyTorch.
- **FP8.** E4M3 is software-emulated with explicit scaling, subnormals,
  saturation at 448, and NaN/Inf sentinels; it is not hardware FP8 arithmetic.
- **Ternary updates.** Packed ternary weights keep the original layout and
  direct-discrete update rule. Coordinate sampling consumes the NumPy RNG
  stream, so a seeded training run does not follow the same sampled update
  coordinates as the PyTorch run even when the starting weights match.
- **SmaulOPT.** The optimizer keeps the original factored state structure. The
  optional fused C++ update falls back to the NumPy path when no compiler or
  qualifying tensor is available; last-bit rounding may differ.
- **RNG streams.** NumPy PCG64 replaces the PyTorch RNG. Greedy generation is
  deterministic and does not use randomness; sampling and training runs do not
  reproduce PyTorch's random sequence. NumPy checkpoints restore the NumPy bit
  generator state, so NumPy runs resume exactly.

## Checkpoint changes

- New NCM checkpoints are safetensors files:
  `tokenizer.safetensors`, `generator.safetensors`,
  `tokenizer-training.safetensors`, and `generator-training.safetensors`.
  Model tensors are prefixed `model::`, optimizer tensors `optim::`, and small
  JSON metadata (config, step, optimizer hyperparameters) is stored in the
  safetensors header.
- Legacy PyTorch NCM `.pt` files are not read at runtime. Convert them with
  `tests/_reference/convert_torch_checkpoint.py` (development-only; it imports
  PyTorch).
- Loading an older NCM checkpoint that contains `torch.optim.AdamW` state keeps
  the model weights and training step but initializes fresh SmaulOPT state,
  because AdamW moments cannot be converted faithfully to SmaulOPT's factored
  second moments.
- The original core checkpoint format (`manifest.json` plus
  `params.safetensors` / `optim.safetensors`) is separate from NCM checkpoints;
  `evaluate.py` and `prune.py` target that core format.

## CLI and runtime behavior

- The build is CPU-only. `--device cuda` is accepted for CLI compatibility but
  NCM training and inference execute on CPU.
- The default NCM runtime supports `--paging R` only; other paging modes are
  rejected rather than silently ignored.
- `--stop-eos` is rejected because the NCM byte tokenizer has no EOS token.
- `--prefill-chunk` values above zero are rejected; the NCM encoder processes
  the complete prefix at once.
- `--mtp-depth` / `--mtp-loss-weight` above zero are rejected because the
  default NCM architecture does not implement MTP.
- `--heads` is accepted for command compatibility but has no effect on
  recurrent blocks.
- The default NCM architecture uses a fixed in-memory expert pool and does not
  grow or prune experts.

## Unsupported items

- `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` are not present
  in this repository; they were not invented during the port.
- CUDA execution, CUDA paging, and hardware FP8 are not supported.
- PyTorch RNG continuation and PyTorch-sampled ternary update trajectories are
  not reproduced.

## Test commands

```bash
PYTHONPATH=.:../smaulbrain-core python -m pytest -q tests
PYTHONPATH=.:../smaulbrain-core python -m pytest -q tests/_reference/test_ncm_parity.py
```
