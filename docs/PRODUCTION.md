# Production-readiness checklist

SmaulBRAIN is still an experimental research project. Passing CI verifies tested invariants, not model quality, crash recovery under power loss, production workload performance, or safe operation within a particular RAM budget.

## Before a long training run

1. Start with the tiny default configuration and a short run. Verify loss is finite and decreases, then save and reload the checkpoint and run inference.
2. Keep the checkpoint on a filesystem with enough free space. Use a separate copy of valuable checkpoints; atomic writes do not protect against disk failure or a damaged filesystem.
3. Use `--save-every` to bound lost work. Test resume on a disposable checkpoint before a long run.
4. Watch system memory and swap/zram during a representative training step, not just model construction. Peak memory includes activations, gradients, optimizer state, temporary dequantized expert weights, paging caches, and checkpoint serialization.
5. Do not assume disk paging makes training memory-bounded: active experts, staging buffers, gradients, and optimizer state can still occupy RAM.

## Allocation-free training preflight

Before allocating the model, validate the checkpoint configuration and training
input and inspect the exact parameter/configuration report:

```bash
python cli.py --ckpt checkpoints/smaulbrain train --preflight-only --data ./dataset
```

Omit `--data` to preflight the synthetic demo input. This checks configuration
and dataset loading without constructing the model. It is **not** a memory
estimator and does not replace a short real training run on the target machine.

## Safe benchmark workflow

The benchmark defaults to a small model and refuses configurations above 5 million estimated parameters before allocating the model:

```bash
python benchmarks/bench.py --steps 2
python benchmarks/bench.py --mode R2VR --steps 2 --d-model 64 --experts 8 --top-k 2 --expert-hidden 128 --context-length 128 --batch-size 1
```

The JSON report includes the exact benchmark configuration, measured training/inference throughput, process peak RSS, checkpoint size, paging counters, routing share, and expert-compute timings. Run it more than once and keep the output alongside the CPU model, RAM, OS, Python/PyTorch versions, paging mode, and configuration.

To test a larger model, raise the parameter cap deliberately, or use `--allow-large` after checking available RAM:

```bash
python benchmarks/bench.py --steps 1 --d-model 128 --experts 8 --top-k 2 --expert-hidden 256 --context-length 128 --batch-size 1 --max-params 10000000
```

The parameter guard is a tripwire, **not a RAM estimator**. Parameter count alone cannot predict peak training memory. If memory is tight, lower batch size, context length, model width, expert hidden size, and depth; increase them one at a time and record peak RSS. Do not run the full preset merely because its experts are paged.

## Target-hardware acceptance

For an 8 GB RAM system, production readiness requires a representative benchmark and training/resume smoke test on that actual machine. CI runners are not substitutes for target-hardware measurements. Record:
- available RAM before launch and peak RSS during training/checkpoint save;
- throughput after warm-up and over multiple steps;
- whether paging causes sustained disk thrashing;
- finite loss/gradients and successful checkpoint reload/resume;
- results for the intended context, batch, expert count, and precision.

No full-preset throughput or memory guarantee is made until those measurements exist. If the process is OOM-killed or swaps continuously, the configuration is not supported on that host.

## Release gates

- Full CI passes on the exact proposed commit.
- Checkpoint round-trip, interrupted/resumed training, and growth-after-reload invariants pass.
- Benchmark results are reproducible and identify hardware/software/configuration.
- Known constraints and experimental features are documented.
- The maintainer explicitly reviews and merges the PR; CI automation must not merge it.
