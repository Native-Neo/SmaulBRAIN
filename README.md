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

**Experimental / Research**

The architecture is still being developed.

Breaking changes are expected.

Tests, measurements, memory usage, and actual training results are more important than keeping the current design unchanged.

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
