# SmaulBRAIN

**Byte-level Recurrent Adaptive Intelligent Network**

SmaulBRAIN is an experimental neural network that can learn continuously, use only part of its parameters at a time, and change its size while it trains.

The main idea is simple:

> **The model should be able to learn new things without having to keep the entire model in memory or throw away what it already knows.**

SmaulBRAIN combines ideas from SmaulNative and [mini-AGI](https://github.com/volotat/mini-AGI), along with its own changes.

> **License note:** SmaulBRAIN is under the **PolyForm Noncommercial
> License 1.0.0** — free for research, learning, and noncommercial use,
> but commercial use needs a separate license. That deliberately limits
> adoption: if you want to use this commercially or contribute with
> downstream commercial use in mind, talk to the maintainer first (see
> [License](#license)).

## Requirements and quickstart

* Python 3.10+ and `pip install -r requirements.txt`
  (`torch>=2.2`, `numpy`, `pytest`, `safetensors`). Developed and measured
  with Python 3.14 + torch 2.14 (CPU); CUDA works wherever torch does.
* No tokenizer files, no downloads, no build step — clone and run.

```bash
pip install -r requirements.txt
python cli.py --ckpt ckpt/demo train --steps 20 --batch 2   # tiny 246K config
python cli.py --ckpt ckpt/demo infer --prompt "hello" --max-new 32
python cli.py --ckpt ckpt/demo report                        # param counts
python pruning.py --ckpt ckpt/demo --rm-worst 0 --dry-run    # pool + floor preview
python -m pytest tests/ -q                                   # 421 tests
```

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

**Experimental / Research**

### Measured

379 tests pass (`python -m pytest tests/ -q`). Measured on CPU (torch 2.14,
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

### Verified against reference

SmaulNative's SmaulOpt (update equations, factored `v`,
BF16 state/FP32 math) and linear attention (ELU+1, S/z recurrence) were
verified in source and ported; its static MoE has no growth/pruning/paging
and its LR is constant. mini-AGI's depth recurrence, PonderNet halting,
paged disk/RAM/VRAM pool, recombination growth, per-expert moments, and
0.1x slow trunk were verified in source and adapted; its attention is
quadratic SDPA (replaced here with linear attention) and it has no FP8 and
no tests. Nothing was copied blindly; nothing was invented where the
reference could not be verified.

### Known open

* Breaking changes are still expected — checkpoint schema, CLI flags, and
  the growth/pruning policy have all changed before and may change again.
* Checkpoints are safetensors (no pickle), but there is **no GGUF
  converter yet** — tensors are extractable, the conversion script is not
  written.
* Numbers above are tiny-config CPU measurements; the `--full` preset
  (328M params) has no published throughput/quality numbers yet.
* The offline pruner refuses below the 64-expert floor rather than
  compensating — pools parked exactly at the floor can only grow, never
  shrink, until the policy says otherwise.

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
`--max/min-depth`, `--halting-threshold`), paging (`--paging-method`,
`--ram-cache`, `--vram-cache`), optimization (`--expert-lr`,
`--trunk-lr-mult`), and runtime (`--context-length`, `--threads`, `--dtype`,
`--seed`, `--ckpt`).

### How to use `cli.py`

Bare defaults are the tiny 246K config (fast CPU smoke runs). `--full`
switches to the full preset: 64 experts, top-8 routing, ~5.12M params/expert
(328.5M total, 42.2M active). Any architecture flag passed explicitly
overrides the preset.

```bash
python cli.py --ckpt ckpt/demo train --steps 20 --batch 2
python cli.py --full --ckpt ckpt/big train --steps 20 --batch 2

python cli.py --ckpt ckpt/demo infer --prompt "hello" --max-new 32 --temperature 0.0
python cli.py --ckpt ckpt/demo report            # dynamic parameter counts
python cli.py --ckpt ckpt/demo quantize --to fp8 # requantize experts
```

Fine-tuning selectors: `--mode trunk|experts|selected|new`, with
`--selected expert_00001,expert_00002` or `--new-since <step>`, plus the
`--grow-every N` schedule. Training never prunes; shrink the pool offline
with `python pruning.py --ckpt <dir> --rm-worst N`.

BPB ports (byte-conv default-on; lookahead/boundary/cosine default-off;
enabling any of them changes training math):
`--use-byte-conv` / `--no-byte-conv` (causal depthwise n-gram conv over
byte embeddings),
`--lookahead-weight` / `--boundary-weight` (auxiliary t+2 prediction +
UTF-8-boundary losses), `--cosine-decay-steps` (cosine LR decay horizon,
0 = constant). Every step logs `bpb` (NLL nats/ln2 — pure compression;
ponder/balance/aux regularizers shape training but are excluded) alongside loss.

### What is `config.py`

The single source of truth: `SmaulBrainConfig` holds topology (`d_model`,
`n_heads`, `vocab_size`), recurrent depth (`min/max_depth`), halting
(`halting_threshold`, `halt_prior`, `ponder_beta`), the dynamic MoE
(`num/top_k/max/min_experts`, explicit `expert_hidden`), paging
(`paging_method`, `ram/vram_cache`), learning rates (`expert_lr`,
`trunk/router_lr_mult`), precision (`dtype`, `fp8_tile`, `state_dtype`), and
growth knobs plus the offline pruner's grace/dying thresholds.
`__post_init__` validates every field, and derived
helpers (`trunk_lr`, `per_expert_params`, `total/active_params`,
`to_dict`/`from_dict`, `describe_counts`) keep checkpoints and reports
consistent. `expert_hidden_for_target` sizes one expert to ~5.12M params.

### How to use `config.py`

```python
from config import SmaulBrainConfig, expert_hidden_for_target, __version__

print(__version__)
cfg = SmaulBrainConfig(d_model=128, n_heads=4, num_experts=8, top_k=2,
                       max_depth=4, paging_method="D2R")
print(cfg.trunk_lr, cfg.per_expert_params)  # slow-trunk LR, params/expert
print(expert_hidden_for_target(512))        # 3328 -> ~5.12M params

d = cfg.to_dict()                 # what checkpoints persist in config.json
cfg2 = SmaulBrainConfig.from_dict(d)
print(cfg2.describe_counts())     # shared/router/expert/total/active splits
```

### What is replay/retention (in `train.py`)

The continual-learning toolkit, merged into the training loop: a reservoir
`ReplayBuffer` that interleaves old byte sequences into training,
`batch_from_seqs` for packing variable length sequences into padded batches,
`evaluate_loss` for gradient-free old/new-data scoring, and
`retention_report`, which reports forgetting as numbers (loss/accuracy
deltas plus a `retained` heuristic) instead of claiming it is solved.

### How to use replay/retention (in `train.py`)

```python
from train import ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report

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

Controlled expert growth — every clone mutated, never the parent.
`select_parents` ranks experts deterministically by contribution (then usage,
then id); `grow_topk_clones` clones the top-k, each with its own relative
perturbation uniform in [1%, 10%] (per-weight random signs, multiplicative,
seeded), so every clone differs from its parent by 1–10% while the parent
stays bit-identical; `grow_expert` assigns the next stable id, derives the new
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

In training, schedule it with `run_training(..., grow_every=20000)` or the
CLI `--grow-every 20000` (capped by `max_experts`), or let it fire whenever
the step loss newly dips below 1.0 (`grow_loss_below`).

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

### What is `model.py`

`SmaulBrainModel`: the shared trunk (byte embedding, init norm, one
`SharedRecurrentBlock`, final norm, byte head) plus the `SparseRouter`,
`ExpertPool`, and `ExpertPager`. The depth loop re-applies the shared block
(`min_depth..max_depth`) with PonderNet halting; training minimizes
halting-weighted cross-entropy plus ponder KL plus MoE balance loss.
Train/infer providers separate gradient-carrying expert leaves from plain
cached weights, and `param_counts` reports shared/router/expert/active plus
RAM/VRAM-resident splits as the topology changes.

### How to use `model.py`

```python
from config import SmaulBrainConfig
from model import SmaulBrainModel

model = SmaulBrainModel(SmaulBrainConfig(d_model=128, num_experts=8, top_k=2))
out = model(ids, targets, step=0)   # training: loss/nll/ponder_kl/acc/depths
print(out["loss"], out["mean_depth"])
out["loss"].backward()              # then step SmaulOpt groups (see train.py)

inf = model.forward_infer(ids)      # no grad, early halting
print(model.param_counts())         # dynamic counts incl. resident splits
```

### What is `paging.py`

`ExpertPager`: three genuinely different execution paths — D2R (disk→RAM
compute cache), R2VR (disk→RAM staging, then RAM→VRAM per use), D2VR
(disk→VRAM direct, RAM bypassed) — with LRU cache caps, per-path counters
that prove which route executed, thread-safe fetches, background `prefetch`
of predicted experts, and `invalidate` after optimizer rewrites. Optimizer
state lives on the record, so it follows the expert, never a cache slot.

### How to use `paging.py`

```python
from paging import ExpertPager

pager = ExpertPager(pool, mode="D2R", ram_cache=8, vram_cache=4)
if pager.mode == "R2VR":
    pager.warm_ram()                 # bulk disk->RAM staging at startup
w = pager.provider("expert_00003")   # mode-specific fetch of compute weights
pager.prefetch(["expert_00004"])     # async load while current expert computes
pager.await_prefetch()
pager.invalidate("expert_00003")     # after its optimizer step
print(pager.stats.to_dict())         # disk_reads/hits/evictions per path
```

### What is `precision.py`

The precision policy plus real FP8 storage. `FP8BlockTensor` holds uint8
E4M3 codes with per-row-block FP32 scales — genuine FP8 bytes, not relabeled
FP32 — dequantized one expert (or row slice) at a time, never as a full-model
transient. `PRECISION_POLICY` maps each component to its storage/compute
dtype (FP8 weights, BF16 activations/state, FP32 stats/math), and
`compute_dtype` resolves it in code.

### How to use `precision.py`

```python
from precision import quantize_fp8_blockwise, dequantize_fp8_blockwise, compute_dtype
import torch

t = quantize_fp8_blockwise(weight_fp32, tile=64)
print(t.codes.dtype, t.scales.shape)  # uint8 bytes + one FP32 scale per block
w = dequantize_fp8_blockwise(t, dtype=torch.bfloat16)
print(compute_dtype("activations"))   # torch.bfloat16 per PRECISION_POLICY
```

### What is `pruning.py`

The offline expert pruner — a standalone script, not a library (training
never prunes; nothing imports this file). Run it explicitly:

```bash
python pruning.py --ckpt ckpt/run1 --rm-worst 4          # remove 4 worst
python pruning.py --ckpt ckpt/run1 --rm-worst 4 --dry-run  # rank only
```

Selection is hysteresis with grace periods. An expert dies only when old
enough, long idle, AND below threshold on every vitality signal (usage
share, gradient activity, contribution) — one strong signal saves it. A
redundancy pass additionally retires near-duplicate gate weights, but only
below a looser usage bar, so load-bearing twins survive. The floor comes
from the checkpoint's `resume_config.json` (`min_experts`, default 64):
removal never takes the pool below it — asking for more than
`len(pool) - floor` refuses with exit 2 and changes nothing.
`prune_experts` removes weights, optimizer state, router row, and metadata
together, highest index first, keeping checkpoints index-consistent; the
checkpoint is re-saved in place, still resumable (scheduler snapshot and
RNG state preserved, so training continues on the next global step).

### What is `synth_data.py`

Synthetic training-text generator with no API keys and no paid models.
It shells out to the local `opencode run` CLI using only OpenCode Zen
**free** models (`--model`, default `big-pickle`; anything not on the free
allowlist is refused — see `--list-models`). First authenticate once with
`opencode auth login` (pick Zen). Built-in `--preset AB` generates two
disjoint everyday domains (syslog lines vs cooking steps) as
`<name>_train.txt` / `<name>_held.txt` plus a `manifest.json` with hashes;
train/held splits are value-disjoint by construction (disjoint topic pools,
dedup, overlap dropped loudly). Feed the files to training
(`cli.py --data ...`) or the gauntlet
(`retention_harness.py --data-a ... --data-b ...`).

```bash
python synth_data.py --list-models
python synth_data.py --preset AB --model big-pickle --out data/ab
python retention_harness.py --data-a data/ab/domainA_train.txt --data-b data/ab/domainB_train.txt
```

### What is `quantize.py`

Checkpoint conversion without full-model residency. `convert_expert_file`
requantizes/retiles one expert file independently and reports byte counts
plus the measured `max_abs_err`; `convert_checkpoint` applies that across a
checkpoint, or casts trunk/router to BF16. FP8 trunk storage is refused on
purpose (norms and accumulators stay precise per the precision policy);
experts stay FP8 on disk by the same policy.

### How to use `quantize.py`

```python
from quantize import convert_expert_file, convert_checkpoint

print(convert_expert_file("ckpt/experts/expert_00000.pt",
                          "ckpt/experts/expert_00000.pt",
                          to="fp8", tile=32))  # retile + error report
convert_checkpoint("ckpt", to="bf16")          # trunk/router -> BF16
```

Or via CLI: `python cli.py --ckpt ckpt quantize --to fp8`.

### What is `recurrent.py`

The shared recurrent block applied `min_depth..max_depth` times per forward
pass — one parameter set, reused, never stacked. Each application runs
RMSNorm → linear-attention projections (continuing the incoming S/z
accumulator, so state genuinely influences the next step) → RMSNorm →
residual → routed MoE → RMSNorm → residual, plus a halt head (bias −2.0, so
the model ponders before halting). `RecurrentState` bundles the hidden vector
with the attention state carried between applications.

### How to use `recurrent.py`

```python
from recurrent import SharedRecurrentBlock, RecurrentState
from linear_attention import LinearAttnState

blk = SharedRecurrentBlock(d_model=128, n_heads=4)
attn = LinearAttnState.zeros(B, 4, 32)
h, attn, halt_logit, aux = blk(h, attn, moe_fn)  # one shared application
# Repeat with the returned (h, attn): step n+1 reads step n's state.
```

`moe_fn` is injected: `(x_flat) -> (y_flat, aux_loss, usage_ids)`.

### What is `rmsnorm.py`

RMSNorm at every major recurrent/residual boundary, with the precision
contract the rest of the system relies on: statistics (mean of squares,
inverse square root) always run in FP32 even for BF16 inputs, then cast back.
`RMSNorm` is the module (FP32 weight), `rmsnorm_fn` the functional form;
both are pure vectorized tensor ops, numerically matched by
`kernels_cpp/rmsnorm.cpp`.

### How to use `rmsnorm.py`

```python
from rmsnorm import RMSNorm, rmsnorm_fn

n = RMSNorm(dim=128)   # weight stays FP32 by design
y = n(x)               # x: BF16 in -> BF16 out, FP32 statistics inside
y = rmsnorm_fn(x, n.weight)
```

### What is `routing.py`

The sparse top-k router. Each token scores every expert, keeps the top-k
(renormalized to sum to 1), and per-expert capacity drops overflow tokens to
the residual path. A Switch-style balance loss keeps traffic spread, FP64
usage/admission counters feed growth and offline pruning, and `add/remove_expert_row`
keep router rows aligned with pool ids (preserving dtype) as the topology
changes.

### How to use `routing.py`

```python
from routing import SparseRouter

router = SparseRouter(d_model=128, num_experts=8, top_k=2, capacity_factor=1.5)
plan = router.route(x)                    # [N, D] -> top_ids/weights/dropped/probs
aux = router.balance_loss(plan.probs)     # add moe_balance_weight * aux to loss
print(router.usage_share())               # FP64-backed traffic distribution
router.add_expert_row()                   # after growth (dtype preserved)
```

### What is `smaulopt.py`

SmaulOpt, ported equation-for-equation from the verified SmaulNative
implementation: momentum over `g`, second moment over `|g|` (not g²), bias
correction, normalized update `u = m̂/(v̂+eps)`, decoupled weight decay.
Two-D weights use factored row/col second moments (full matrix never
materialized); state stores BF16 with FP32 math; global FP64 grad clipping
with a non-finite skip. `SmaulOpt` splits LR groups (`step_trunk`,
`step_router`, `step_expert`), and expert states live on the records so they
follow experts across paging and die with pruned experts.

### How to use `smaulopt.py`

```python
from smaulopt import SmaulOpt, SmaulOptHParams

opt = SmaulOpt(SmaulOptHParams(lr=2e-4, wd=0.01, clip=1.0, state_dtype="bf16"))
opt.step_trunk(model._trunk_params(), cfg.trunk_lr)    # slow shared trunk
opt.step_router(model._router_params(), cfg.router_lr)
opt.step_expert(record, grad_carriers, 1.0, cfg.expert_lr)  # rewrites FP8
```

Dense groups update in place; `step_expert` decodes, updates, and
requantizes only that expert's blocks.

### What is `storage.py`

Atomic, multi-file safetensors checkpointing — no pickle anywhere, so every
tensor is cleanly extractable (e.g. for GGUF converters). Layout:
`config.json`, `manifest.json` (written last as the commit point),
`resume_config.json` (prune floor, topology, scheduler snapshot),
`meta.json` (optimizer hparams, RNG state), `trunk.safetensors`,
`router.safetensors`, `optim.safetensors`, `rng.safetensors`, and one
`experts/<id>.safetensors` + `experts/<id>.json` sidecar per expert (FP8
codes + scales, expert-local optimizer moments, metadata). A crash mid-save
leaves the previous checkpoint intact. Loading resizes the router both
ways, restores usage stats and RNG, clears stale pager caches, and rebuilds
the pool in manifest order. `make_disk_loader` reads single experts without
loading the model.

### How to use `storage.py`

```python
from storage import save_model, load_model, load_expert_file, make_disk_loader

save_model("ckpt/run1", model, opt, step=100)
manifest = load_model("ckpt/run1", fresh_model, fresh_opt)  # returns manifest
rec = load_expert_file("ckpt/run1/experts/expert_00002.safetensors")  # no full load
loader = make_disk_loader("ckpt/run1")                       # pager wiring
```

### What is `train.py`

One training loop for training and fine-tuning (a mode, not a separate
model). `train_step` runs forward → backward → mode-gated SmaulOpt updates:
`entire` moves everything, `trunk` freezes experts, `experts` freezes the
trunk, `selected` steps listed ids, `new` steps experts born after a cutoff.
`run_training` adds replay interleaving, the growth schedule (training
never prunes — pool shrinkage is the offline `pruning.py` script),
periodic checkpointing, and before/after retention measurement.

### How to use `train.py`

```python
from train import train_step, run_training

stats = train_step(model, opt, cfg, x, y, step=0, mode="entire")
print(stats["loss"], stats["mean_depth"], stats["stepped_experts"])

res = run_training(model, opt, cfg, train_seqs, steps=20, batch_size=2,
                   mode="selected", selected=["expert_00001"],
                   replay=ReplayBuffer(512), replay_n=2,
                   old_seqs=old_data, grow_every=20000,
                   ckpt_dir="ckpt/run1", save_every=50)
print(res["final_loss"], res["retention"])  # retention measured, not claimed
```

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
