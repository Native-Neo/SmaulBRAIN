# PyTorch port audit

## Scope

This package ports the original SmaulBRAIN Python model/runtime to PyTorch tensors and `torch.nn.Module`/autograd. Public model parameters and trainable state are Torch tensors; gradient propagation is handled by autograd, with a custom autograd operation for quantized linear layers. The original CLI surface, architecture families, presets, data flow, training loop, optimizer, quantization choices, routing, paging modes, growth/pruning, checkpointing/recovery, inference, evaluation, and benchmark entry points are retained.

The native C++ kernels are not rewritten as Python tensor code: their source is retained, and small adapters bridge Torch CPU storage to the existing C++ ABI. These adapters are optional and have Torch/Python fallbacks. NumPy is not used as the core model's forward/backward tensor runtime; it remains for legacy `.npz` import and compatibility with the existing native-kernel ABI.

## Conversion map

| Original subsystem | Port status | Notes |
|---|---|---|
| Model/forward/backward | Converted | `model.py` uses Torch parameters, ops and autograd; tied embeddings/output are retained. |
| RMSNorm, activations and recurrent attention | Converted | Torch operations preserve autograd; recurrent scan in model is a Python loop of tensor ops. |
| MoE/router and capacity policy | Converted | Torch scores, stable top-k routing and weighted expert outputs; overflow and disabled-expert paths are tested. |
| SmaulOPT | Converted | Torch state/update equations and checkpointable factored row/column moments. Optional fused native optimizer remains behind a bridge. |
| FP8 E4M3 | Converted as software emulation | Tensor conversion/scaling is Torch-based. This does not claim native FP8 hardware instructions. |
| Ternary quantization | Converted | Quantized codes are packed; forward path avoids reconstructing a full dense float weight matrix. The optional packed C++ kernel remains. |
| Data input / byte vocabulary | Converted | Training stream creates Torch batches from raw bytes; tokenizer remains byte-ID mapping. |
| Expert paging | Converted interface and tensor storage | Expert `.pt` files, cache and legacy `.npz` loading retained. The model still owns canonical trainable expert parameters, so paging is not a guarantee that all expert parameter RAM is eliminated. |
| Growth and pruning | Converted | Torch tensor mutation/copy, optimizer-state update, evidence gates and recovery lifecycle covered by tests. |
| Checkpointing/recovery | Converted | New `.pt` checkpoint format includes checksums, atomic publish/backup recovery and training state. Legacy `.npz` files can be imported. |
| Inference, evaluate, train CLI, benchmarks | Converted | Smoke-tested on CPU; generation uses a Torch generator and recurrent state. |
| C++ source kernels | Retained, not ported to Torch | This is deliberate to preserve optional packed-kernel/native optimizer paths. Python adapter files document this compatibility boundary. |

## Validation performed

Environment: Python 3.13.5, PyTorch 2.10.0 CPU build, no CUDA device available.

- `python -m compileall -q ...`: passed.
- `pytest -v tests`: **101 passed, 1 skipped** (CPU, OMP/MKL/OpenBLAS thread counts limited to 1).
- Resume CLI now supports `--steps N` for additional updates, reports raw and smoothed loss separately, reports the last overflow fraction, warns on quantization/sequence/capacity changes, and warns when `--max-steps` would do no work.
- Tiny MoE training to step 2: passed and checkpoint saved.
- Resume from step 2 through step 3: passed.
- Inference from saved checkpoint: passed (4 generated tokens in smoke test).
- Evaluation from saved checkpoint: passed (2 synthetic batches).
- Growth/prune/re-grow and fault-injected checkpoint recovery: included in passing lifecycle tests.
- Router benchmark smoke was run separately; results are environment-specific and are not a performance claim.
- Prune dry-run with fewer than the default minimum evidence samples was correctly refused, as designed.

## Known limits and residual risks

1. **CUDA/VRAM modes unverified.** This environment has CPU-only PyTorch. V/D2V/R2V paths validate accelerator requirements, but actual CUDA execution and memory behavior were not tested.
2. **Paged experts still have canonical trainable parameters in the model.** Disk cache operations and modes are converted, but the current optimizer/checkpoint ownership means paging alone does not eliminate the resident parameter pool. Treat it as an expert-store/cache path, not an out-of-core training guarantee.
3. **Recurrent model scan is autograd-native but not kernel-fused.** The model's recurrence is expressed as a Python loop of Torch tensor operations. Native recurrent scan/backward kernels exist as optional standalone/helper code and are covered against reference tests, but are not wired into the main model forward path. Long sequence throughput may therefore be slower than the former native implementation.
4. **FP8 is emulated.** The E4M3 representation is simulated with Torch operations; there is no hardware FP8 guarantee.
5. **Native C++ paths still cross an ABI adapter.** `smaulbrain/_quant_native.py` and `_native_opt_numpy.py` retain NumPy/ctypes boundary code. This avoids rebuilding established kernels, but those adapter files themselves are not a pure-Torch rewrite.
6. **Legacy RNG compatibility is not bit-exact.** Legacy weight/optimizer tensors are importable, but old NumPy RNG state is not replayed as a Torch generator stream. Future continuation is deterministic under the new Torch RNG path, not bitwise identical to a former NumPy run.
7. **Full presets are memory-heavy.** Their parameter-count tests are formula-based to avoid allocating hundreds of millions of parameters during testing. Full-size training convergence, peak memory and throughput were not measured here.
8. **Routing collapse is still possible.** The current model does not add a load-balancing auxiliary loss to the training objective. A high `last_overflow_fraction` means many expert assignments were dropped; investigate router/expert load before trusting training quality. The CLI now surfaces this metric but does not automatically rebalance routing.
9. **Numerical / training parity is bounded by tests.** Unit tests compare critical operations and lifecycle behaviors against references, but no full training corpus was available for convergence equivalence with the original project. Changing quantization mode on resume can sharply alter loss; the CLI now warns when the checkpoint configuration differs.

## Recommendation

Use `dense-tiny` or `moe-tiny` for local CPU training and validate on a real corpus before scaling. Treat full-preset training performance, CUDA paging, long-context speed, and exact legacy-run reproducibility as unverified. The automated suite passing is evidence for the covered behavior, not a blanket production certification.
