# Checkpoint format `smaulbrain-v2:1`

Directory: `manifest.json` (preset, arch, vocab, quant, paging meta, step,
data cursor, RNG, optimizer meta, scheduler record with `history[]` + `pending`,
growth/prune history, file hashes), `params.npz`, `optim.npz`,
`experts/` mirrors for disk-backed paging. Writes go to a sibling temp dir
then atomically published (`rename` swap with backup restore); readers only
see complete versions; hashes verified on load.

Invariants: no partial experts; model+scheduler serialized as one version;
committed ids unique; pending id never in history; registry/router/paging/
optimizer counts agree; pending resolved before the next optimizer step;
restoring committed never replays; failed writes never replace the last valid
version. Recovery cases (no pending / `WAITING_FOR_RANKING` / `PREPARED` /
`VALIDATED` / publish failure) are implemented in `Trainer.load` + `train`
and tested with fault injection.

# Paging

`--paging R|V|D2R|D2V|R2V` (`v2/paging.py`). R/D2R work on CPU-only hosts; V,
D2V, R2V require a compatible accelerator and raise actionable errors
otherwise (system RAM / iGPU / shared memory is never claimed as VRAM).
D2R/D2V use bounded LRU caches (`--paging-cache-experts`,
`--paging-staging-bytes`) with hit/miss/eviction/byte counters, pinning during
active matmuls, and invalidation on updates/restore/growth/prune.

# Quantization

`--forward-quant fp32|fp8|ternary`, `--backward-mode fp32|fp8|ternary`;
supported pairs: (fp32,fp32), (fp8,fp8), (ternary,fp32), (ternary,ternary),
(ternary,fp8). Ternary: threshold 0.7*mean|W|, scale = mean|W| over nonzeros,
2-bit packed (4/byte), packed kernels never build dense fp32 (guarded, tested).
Weight grads use the straight-through estimator (documented surrogate; no
claim of exact discrete derivatives). FP8 is E4M3 software emulation
(max 448, per-tensor scale, saturation, NaN/Inf preserved); no native FP8 is
claimed on Ivy Bridge. Packed caches are bounded and invalidated on any
parameter change.

# SmaulOPT

`v2/smaulopt.py`: `m <- b1*m + (1-b1)*g`; 2-D params use factored row/col
statistics `R (m), C (n)`, `V_hat = outer(R,C)/mean(R)`; vectors/scalars use a
full `v` (documented non-factored). No bias correction; `denom = sqrt(V)+eps`;
decoupled weight decay; fp32 state. Missing grads skipped; new experts
zero-initialized; pruned experts dropped; state checkpointed and resumed.

# Hardware, install, limitations

Target: i3-3220 (no AVX2/AVX-512), 8GB RAM, CPU-only. `pip install numpy
pytest`; no CUDA/Triton needed. Native C++ (`v2/kernels/cpu_kernels.cpp`) is
portable scalar only and optional: `c++ -O2 -std=c++11 -fPIC -shared
v2/kernels/cpu_kernels.cpp -o smaul_kernels.so`. `v2/cpu.py` detects features;
no mandatory path needs AVX2/512 (asserted in tests).

Measured on the target machine (i3-3220, numpy 2.4.6, seq 64):

| workload | forward | backward | opt step | decode tok/s | peak RSS |
|---|---|---|---|---|---|
| dense-tiny fp32 | 0.016s | 0.051s | 0.021s | ~211 | ~61MB |
| moe-tiny fp32 | 0.030s | 0.071s | 0.018s | ~402 | ~69MB |
| moe-tiny ternary-fwd/fp32-bwd | — | 0.17s | — | ~160 | ~74MB |

Weight footprints: dense-full 977MB, moe-full 1275MB (fp32). Full-preset
*training* (weights + grads + optimizer ≈ 3–4GB plus activations) is not
practical within 8GB alongside the OS; bottleneck is optimizer-state +
activation memory, reproducible via `python bench.py --preset dense-full`.
Practical on target: tiny presets for training; full presets for
forward/inference with `--paging D2R` and ternary forward; use
`--seq-len 16..64`, `--paging-cache-experts 2..8`. V/D2V/R2V are unverified
without an accelerator (they correctly refuse on this host).
