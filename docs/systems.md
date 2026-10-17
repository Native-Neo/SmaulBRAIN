# Runtime systems and limitations

## Checkpoint format

New checkpoints are published as an atomic directory containing a JSON manifest and PyTorch tensor payloads (`.pt`). The manifest records architecture, tokenizer metadata, quantization/paging settings, step/data cursor, RNG and optimizer/scheduler state, plus growth/prune history and integrity metadata. Publication uses a temporary sibling directory and backup recovery. The loader also imports the legacy `smaulbrain-v2:1` `params.npz` / `optim.npz` format; the next save writes the new Torch format. Legacy NumPy RNG state is accepted as metadata but is not converted into an identical Torch random stream.

## Paging

`--paging R|V|D2R|D2V|R2V` selects the expert store/cache mode. RAM and disk-backed modes are usable on CPU. VRAM modes require a supported accelerator and are not validated by the CPU-only test environment. Expert weights are stored in `.pt` files; old per-expert `.npz` files can be read. **Important:** the current `Model` keeps its canonical trainable expert parameters, so paging caches/stores do not currently guarantee that the entire expert parameter pool is removed from process RAM.

## Quantization

- `--forward-quant fp32|fp8|ternary`
- `--backward-mode fp32|fp8|ternary`

The supported combinations are `(fp32, fp32)`, `(fp8, fp8)`, `(ternary, fp32)`, `(ternary, ternary)`, and `(ternary, fp8)`. Ternary weights are 2-bit packed (four codes per byte); the main quantized forward path does not reconstruct dense floating-point weights. The straight-through estimator is a surrogate gradient, not an exact derivative of a discrete quantizer. FP8 is E4M3 software emulation with tensor scaling and saturation, not native FP8 arithmetic.

## SmaulOPT

The custom optimizer maintains a first moment and, for matrices, factored row/column statistics. Vectors/scalars use a full second-moment state. Missing gradients are skipped; optimizer state is checkpointed and resized/reset as experts are added or pruned. Optional C++ fused updates remain behind a CPU bridge.

## Dependencies and performance

Install with `pip install -r requirements.txt`. Torch is required; NumPy remains for legacy checkpoint import and optional ctypes/C++ bridges. A C++17 compiler is optional for native kernels. CPU-only behavior is tested. No full-model performance/convergence measurements are claimed for this port. The model recurrent scan currently uses a Python loop over Torch tensor ops, and full presets require substantial memory. Use tiny presets for CPU development and benchmark the actual machine/dataset before choosing larger settings.
