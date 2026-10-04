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
