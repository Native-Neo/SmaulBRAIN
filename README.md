# SmaulBRAIN

**Byte-level Recurrent Adaptive Intelligent Network**

SmaulBRAIN is an experimental neural network that can learn continuously, use only part of its parameters at a time, and change its size while it trains.

The main idea is simple:

> **The model should be able to learn new things without having to keep the entire model in memory or throw away what it already knows.**

SmaulBRAIN combines ideas from SmaulNative and [mini-AGI](https://github.com/volotat/mini-AGI), along with its own changes.

## What does SmaulBRAIN do?

### Byte-level input and output

SmaulBRAIN works directly with bytes instead of using a normal subword tokenizer.

```text
Text / Data
    ↓
 Bytes
    ↓
SmaulBRAIN
    ↓
 Bytes
    ↓
Text / Data
```

This means it can work with any byte sequence without needing to build a large vocabulary first.

### Recurrent processing

SmaulBRAIN reuses its processing block multiple times instead of needing a separate set of layers for every step of computation.

The hidden state is updated each time the block runs.

```text
Input
  ↓
Block
  ↓
Block
  ↓
Block
  ↓
Output
```

The number of passes can also be changed depending on how much computation is needed.

### Sparse experts

SmaulBRAIN has a pool of experts.

For each piece of input, the router selects only some of them instead of running every expert.

```text
              Router
             /  |  \
            ↓   ↓   ↓
         Expert Expert Expert
            \   |   /
              Output
```

This allows the model to have many parameters without using all of them for every input.

### Experts can be added and removed

The expert pool does not have to stay the same size.

If the model needs more capacity, new experts can be added.

If some experts stop being useful, they can be removed.

For example:

```text
200M
 ↓
300M
 ↓
500M
 ↓
800M
 ↓
1B
 ↓
900M
```

The total parameter count can therefore change during training.

The individual experts can keep the same shape while the number of experts changes.

### Experts can be stored on disk

The whole expert pool does not need to stay in RAM or accelerator memory.

Inactive experts can stay on disk and be loaded when needed.

```text
        Expert Pool
            │
          Disk
            ↓
        RAM Cache
            ↓
      Active Experts
            ↓
         Compute
```

This means a model can have more total parameters than the amount of memory available for active computation.

The tradeoff is that loading experts from storage takes time.

### Continual learning

SmaulBRAIN is designed to keep learning after its initial training.

The shared parts of the model can use a smaller learning rate, while the selected experts can adapt more quickly.

The goal is to make learning new data less likely to damage older knowledge.

```text
Old knowledge
      +
New data
      ↓
Continued learning
```

### Mixed precision

SmaulBRAIN can use different numerical formats for different parts of the model.

The planned setup includes:

* FP8 for suitable weights
* BF16 for suitable activations and stored state
* FP32 where more precision is needed

The exact formats can change depending on the hardware and the part of the model.

## Architecture

The general structure looks like this:

```text
                 BYTES
                   │
                   ▼
               Embedding
                   │
                   ▼
              Shared State
                   │
                   ▼
          ┌─────────────────┐
          │ Recurrent Block │◄──────┐
          └────────┬────────┘       │
                   │                │
                   ▼                │
                 Router             │
                   │                │
             ┌─────┼─────┐          │
             ▼     ▼     ▼          │
          Expert Expert Expert      │
             └─────┼─────┘          │
                   │                │
                   ▼                │
             Updated State ─────────┘
                   │
                   ▼
              Byte Output
```

This is the basic idea rather than a promise that every part of the final implementation will look exactly like this.

## Main goals

SmaulBRAIN is being built to:

* Work directly with bytes
* Use sparse computation
* Reuse computation through recurrence
* Learn continuously
* Add and remove experts
* Keep inactive weights offloaded
* Use less memory than a dense model of the same total size
* Run on relatively weak hardware
* Stay simple enough to experiment with

## What SmaulBRAIN is not

SmaulBRAIN is not trying to be:

* A normal Transformer
* A giant dense model
* A model where every parameter is used for every input
* A model that needs its entire parameter set in RAM or VRAM
* A finished production system

It is an experimental architecture.

## Status

**Experimental / Research — implemented and measured (2026-10-04)**

79 tests pass (`python -m pytest tests/ -q`). Measured on CPU (torch 2.14,
4-core, 7.6 GB RAM) with tiny configs unless noted:

* 16K context forward: +61 MB delta RSS, ~6 s (d=32, 4 experts, top-1,
  depth 1); 1K→8K RSS ratios stay <1.5x per doubling (quadratic would be 4x).
* Training throughput (d=64, 8 experts, top-2, depth ≤3, ctx 128):
  D2R 248 tok/s, R2VR 248 tok/s, D2VR 318 tok/s; inference 12–22 tok/s.
* Continual-learning demo: new-data loss 5.60 → 2.88 over 12 steps with
  replay; old-data delta +0.06 (reported numerically, not claimed solved).
* Sparse-vs-dense byte head on CPU: dense 3.9 ms vs sparse 115 ms per
  1024-token batch — dense wins at 256-byte vocab (measured); sparsity is
  used for expert routing, not token connections.
* Expert compute BF16 vs FP32 on this CPU: 8.9 ms vs 1.1 ms — FP8 storage
  with FP32 compute is the faster default where BF16 lacks acceleration.
* Expert size is explicit (`--expert-size`); at d=512/h=3328 one expert is
  5,111,808 params (~5.12M).

Reference audit: SmaulNative's SmaulOpt (update equations, factored `v`,
BF16 state/FP32 math) and linear attention (ELU+1, S/z recurrence) were
verified in source and ported; its static MoE has no growth/pruning/paging
and its LR is constant. mini-AGI's depth recurrence, PonderNet halting,
paged disk/RAM/VRAM pool, recombination growth, per-expert moments, and
0.1x slow trunk were verified in source and adapted; its attention is
quadratic SDPA (replaced here with linear attention) and it has no FP8 and
no tests. Nothing was copied blindly; nothing was invented where the
reference could not be verified.

Breaking changes are still expected.

Tests, measurements, memory usage, and actual training results are more important than keeping the current design unchanged.

## Module guide

What each root-level module is and how to use it. Start with `config.py`:
every other module is configured through `SmaulBrainConfig`.

### What is `bytes.py`

The core input representation: raw bytes are the tokens (ids 0–255), so any
byte data works with no BPE/WordPiece vocabulary. Four structural specials
(`<bos>`, `<eos>`, `<pad>`, `<sep>`, ids 256+) may bracket sequences but are
skipped by byte decoding. Also owns `IncrementalByteDecoder`, which buffers
split UTF-8 tails during streaming generation.

### How to use `bytes.py`

```python
from bytes import encode_text, decode_text, decode_bytes, IncrementalByteDecoder

ids = encode_text("hi ✓")      # UTF-8 bytes -> ids, e.g. [104, 105, 32, 226, 156, 147]
text = decode_text(ids)        # ids -> str (invalid UTF-8 -> replacement char)
raw = decode_bytes(ids)        # ids -> bytes, specials (>=256) skipped

dec = IncrementalByteDecoder()
for chunk in [[104], [105]]:
    print(dec.feed(chunk), end="")  # emits decodable prefix, buffers split tails
print(dec.flush())
```

### What is `cli.py`

The command-line interface. Every flag maps onto `SmaulBrainConfig`, and the
`train` / `infer` / `report` / `quantize` subcommands share one model build
path and one checkpoint format. Training flags cover architecture (`--d-model`,
`--experts`, `--expert-size`, `--active-experts`, `--max/min-experts`,
`--max/min-depth`, `--halting-threshold`), paging (`--pagingmthd`,
`--ram-cache`, `--vram-cache`), optimization (`--expert-lr`,
`--trunk-lr-mult`), and runtime (`--context-length`, `--threads`, `--dtype`,
`--seed`, `--ckpt`).

### How to use `cli.py`

```bash
python main.py --d-model 32 --experts 2 --expert-size 64 --active-experts 1 \
  --max-depth 1 --context-length 64 --ckpt ckpt/demo train --steps 20 --batch 2

python main.py --ckpt ckpt/demo infer --prompt "hello" --max-new 32 --temperature 0.0
python main.py --ckpt ckpt/demo report            # dynamic parameter counts
python main.py --ckpt ckpt/demo quantize --to fp8 # requantize experts
```

Fine-tuning selectors: `--mode trunk|experts|selected|new`, with
`--selected expert_00001,expert_00002` or `--new-since <step>`, plus
`--grow-every N` / `--prune-every N` schedules.

### What is `config.py`

The single source of truth: `SmaulBrainConfig` holds topology (`d_model`,
`n_heads`, `vocab_size`), recurrent depth (`min/max_depth`), halting
(`halting_threshold`, `halt_prior`, `ponder_beta`), the dynamic MoE
(`num/top_k/max/min_experts`, explicit `expert_hidden`), paging
(`paging_method`, `ram/vram_cache`), learning rates (`expert_lr`,
`trunk/router_lr_mult`), precision (`dtype`, `fp8_tile`, `state_dtype`), and
growth/pruning knobs. `__post_init__` validates every field, and derived
helpers (`trunk_lr`, `per_expert_params`, `total/active_params`,
`to_dict`/`from_dict`, `describe_counts`) keep checkpoints and reports
consistent. `expert_hidden_for_target` sizes one expert to ~5.12M params.

### How to use `config.py`

```python
from config import SmaulBrainConfig, expert_hidden_for_target

cfg = SmaulBrainConfig(d_model=128, n_heads=4, num_experts=8, top_k=2,
                       max_depth=4, paging_method="D2R")
print(cfg.trunk_lr, cfg.per_expert_params)  # slow-trunk LR, params/expert
print(expert_hidden_for_target(512))        # 3328 -> ~5.12M params

d = cfg.to_dict()                 # what checkpoints persist in config.json
cfg2 = SmaulBrainConfig.from_dict(d)
print(cfg2.describe_counts())     # shared/router/expert/total/active splits
```

### What is `continual.py`

The continual-learning toolkit: a reservoir `ReplayBuffer` that interleaves
old byte sequences into training, `batch_from_seqs` for packing variable
length sequences into padded batches, `evaluate_loss` for gradient-free
old/new-data scoring, and `retention_report`, which reports forgetting as
numbers (loss/accuracy deltas plus a `retained` heuristic) instead of
claiming it is solved.

### How to use `continual.py`

```python
from continual import ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report

buf = ReplayBuffer(capacity=512, seed=0)
buf.add([104, 105])          # reservoir-kept past sequence
old = buf.sample(4)          # interleave into the next batch

b = batch_from_seqs(old_seqs, context=128)   # [B, T+1], padded/truncated
before = evaluate_loss(model, b, context=128)
# ... train on new data ...
after = evaluate_loss(model, b, context=128)
print(retention_report(before, after))  # numeric deltas, no "solved" claims
```

### What is `experts.py`

The dynamic expert pool. An `ExpertRecord` bundles FP8 block weights, its own
BF16 factored optimizer state, and metadata (stable `expert_NNNNN` id, birth
step, parents, source, usage/gradient/contribution statistics) — identity is
independent of router index and cache slot. `make_expert` builds records,
`ExpertPool` maps router index <-> expert id, and its `forward` dispatches
tokens batched per expert (one SwiGLU matmul per active expert) with routing
weights, tracking per-expert usage as it goes.

### How to use `experts.py`

```python
from experts import ExpertPool, make_expert

pool = ExpertPool()
pool.add(make_expert(pool.fresh_id(), d_model=128, expert_hidden=256))
print(pool.order)                       # ['expert_00000', ...] (stable ids)
print(pool.experts['expert_00000'].param_count)

# Weighted combination over a routing plan (provider = paging layer):
out = pool.forward(x, plan.top_ids, plan.top_weights, plan.dropped,
                   provider=pager.provider, step=step)
print(pool.usage_snapshot())            # per-expert tokens/activity/parents
```

### What is `growth.py`

Controlled expert growth — never pure noise without a documented reason.
`select_parents` ranks experts deterministically by contribution (then usage,
then id); `recombine_weights` builds a convex parent-mean plus small seeded
perturbation; `grow_expert` assigns the next stable id, derives the new
router row from the parent-mean row, initializes optimizer state, records
parents/birth step, and stays reproducible from the caller seed. An empty
pool falls back to a fresh random expert.

### How to use `growth.py`

```python
from growth import grow_expert, select_parents

print(select_parents(pool, k=2))   # deterministic parent ranking
eid = grow_expert(pool, router, d_model=128, expert_hidden=256,
                  step=100, seed=123, n_parents=2)
# Same seed + same pool -> byte-identical child (verified in tests).
```

In training, schedule it with `run_training(..., grow_every=200)` or the
CLI `--grow-every 200` (capped by `max_experts`).

### What is `infer.py`

Inference on the same core model and checkpoint format — no separate
architecture. `generate` runs autoregressive byte generation through
`forward_infer` (early halting, no gradients), sampling with temperature /
top-k / top-p, streaming UTF-8 decode, and returns paging counters alongside
ids/text/depths. It preserves the caller's train/eval mode and runs on the
model's device.

### How to use `infer.py`

```python
from infer import generate
from bytes import encode_text, decode_text

res = generate(model, encode_text("hello"), max_new=32,
               temperature=0.0, top_k=0, top_p=1.0,
               context=1024, seed=0)
print(res["text"])     # streamed byte decode
print(res["depths"])   # per-token executed depth (adaptive halting)
print(res["paging"])   # disk reads / cache hits / evictions
```

`temperature=0.0` is greedy; `top_k`/`top_p` shape sampling otherwise.

### What is `kernels.py`

The vectorized CPU hot paths plus the sparse-vs-dense study. `time_fn`
benchmarks any callable (median/mean ms). `SparseTopKHead` is a fixed fan-in
sparse alternative to the dense byte head, and `compare_heads` times both on
CPU and reports the measured winner — dense wins ~30x at 256-byte vocab, so
sparsity lives in expert routing, not token connections.
`linear_attn_memory_bound` states the T-independent attention memory formula.

### How to use `kernels.py`

```python
from kernels import compare_heads, time_fn, linear_attn_memory_bound

print(compare_heads()["winner"])   # 'dense' on CPU at 256-byte vocab
print(time_fn(lambda: model.forward_infer(x))["median_ms"])
print(linear_attn_memory_bound(heads=4, head_dim=8))  # bytes, T-independent
```

Native C++ equivalents live in `kernels_cpp/` (`g++ -O2 -std=c++17 -c`).

### What is `linear_attention.py`

Genuine linear attention — no QKᵀ matrix, ever. With the ELU+1 feature map,
each step does a rank-1 outer-product update into the recurrent state
(`S += φ(k)ᵀv`, `z += φ(k)`) and reads `y = φ(q)ᵀS / (φ(q)ᵀz + eps)`: O(T·Dh²)
time and O(Dh²) memory, causal by construction. `LinearAttnState` is the
streamable (S, z) pair; `linear_attn_forward` folds a chunk, `linear_attn_step`
advances one token, and both agree exactly (tested).

### How to use `linear_attention.py`

```python
from linear_attention import LinearAttnState, linear_attn_forward, linear_attn_step

out, state = linear_attn_forward(q, k, v)  # [B, H, T, Dh] + terminal state
print(state.nbytes())                      # constant in T (no quadratic growth)

st = LinearAttnState.zeros(B, H, Dh, device=q.device)
for t in range(T):                         # incremental inference, O(1)/step
    y, st = linear_attn_step(st, q[:, :, t], k[:, :, t], v[:, :, t])
```

### What is `main.py`

The six-line program entry point. It imports `main()` from `cli.py` and runs
it, so `python main.py ...` is the way to reach every subcommand without
installing the project as a package.

## License

SmaulBRAIN is distributed under the **PolyForm Noncommercial License 1.0.0**.

See [`LICENSE`](LICENSE) for the full license.

Commercial use requires separate permission from the copyright holder.

## Why SmaulBRAIN?

The project started with a simple question:

> **What if a neural network could change its own size as it learns?**

Instead of making one large model where everything is always loaded and active, SmaulBRAIN tries to keep a larger pool of knowledge while only using the parts that are needed.

The model can therefore:

* learn new things,
* keep older knowledge,
* add capacity,
* remove unused capacity,
* and keep inactive weights outside active memory.

## Acknowledgements

SmaulBRAIN is influenced by both **SmaulNative** and **mini-AGI**.

In particular, [mini-AGI](https://github.com/volotat/mini-AGI) influenced several parts of the design, including:

* Byte-level language modeling
* Continual learning
* Recurrent processing
* Sparse expert routing
* Adaptive computation
* Growing and pruning experts
* Storing inactive experts outside active memory
* Keeping optimizer state with individual experts
* Using a smaller learning rate for shared model parts

SmaulBRAIN is **not a fork of mini-AGI**. It is a separate implementation that combines ideas from mini-AGI with ideas developed in SmaulNative, including sparse recurrent computation, linear attention, and mixed-precision execution.

The goal is to experiment with these ideas in a different architecture and make something that can run on limited hardware.

### Related project

**mini-AGI — A Continually Learning Byte-Level Language Model**

https://github.com/volotat/mini-AGI

Citation:

```bibtex
@software{Borsky_mini_AGI_2026,
  author = {Borsky, Alexey},
  month = {9},
  title = {{mini-AGI: A Continually Learning Byte-Level Language Model}},
  url = {https://github.com/volotat/mini-AGI},
  version = {1.0.0},
  year = {2026}
}
```

Thanks to the authors and contributors of mini-AGI and the other projects and research that inspired this work.
