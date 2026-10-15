# SmaulBRAIN V2

Byte-level recurrent linear-attention language modeling with Mixture-of-Experts,
quantization, paging, factored optimization, and automatic expert growth —
built from scratch for CPU-first execution.

## Quickstart

```bash
pip install numpy pytest          # only hard dependencies
python train.py --moe --preset tiny --max-steps 50
python train.py --dense --preset tiny --max-steps 50
python train.py --preset moe-full --max-steps 100   # alias form
python infer.py --preset moe-tiny --checkpoint checkpoints/run --prompt "Hello"
python evaluate.py --preset moe-tiny --checkpoint checkpoints/run
python prune.py --rm-worst 8 --checkpoint checkpoints/run --dry-run
python bench.py --preset dense-tiny
pytest tests/ -q
```

Preset flags: `--moe/--dense` are mutually exclusive; `--preset` accepts
`full|tiny` (with `--moe/--dense`) or full aliases
`moe-full|moe-tiny|dense-full|dense-tiny`.

## Layout

- `train.py / infer.py / prune.py / evaluate.py / bench.py` — entry points
- `v2/` — framework: `vocab, config, cli, cpu, math_ref, ops, model, moe,
  paging, quant, kernels/, smaulopt, growth, checkpoint, train_loop,
  inference, data, prune_cmd`
- `tests/` — mathematical, preset, paging, quant, optimizer, growth,
  crash-recovery, and end-to-end tests
- `docs/` — architecture, presets, growth, checkpointing, hardware, limits

## Growth loss units

Growth monitors causal next-byte cross-entropy in **nats per predicted byte**.
Defaults: `--growth-every-steps 10000`, `--growth-loss-nats 0.5`.
`0` disables periodic growth; `none` disables the loss trigger.
Loss trigger is edge-triggered (fires once per downward crossing, re-arms only
after rising above threshold). See `docs/growth.md`.
