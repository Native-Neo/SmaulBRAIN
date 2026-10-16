# Growth scheduler

Automatic growth lives in the training loop (`smaulbrain/train_loop.py`),
driven by `smaulbrain/growth.py`. No manual command is needed; ordinary MoE
training grows.

## Triggers (logical OR)

- `--growth-every-steps 10000` (non-negative int; `0` disables). Fires at
  global optimizer steps 10000, 20000, … Only completed optimizer updates
  count. After restore at step 15000 with interval 10000, next fires at 20000.
- `--growth-loss-nats 0.5` (finite non-negative float or `none`; `none`
  disables). Fires when EMA-smoothed (`--growth-loss-beta`, default 0.98)
  training CE in **nats per predicted byte** crosses from above to at/below
  the threshold. Edge-triggered: after a loss-triggered event the trigger
  disarms until smoothed loss rises above the threshold again. Non-finite or
  missing losses never trigger.

If both fire on one step, exactly one event with reason `both` is created.

## Event

Full: +8 experts (best 8 sources by quality; clones of top-2 sources exact,
other 6 get element-wise additive Gaussian noise `N(0, sigma^2)`,
`sigma = --growth-noise-scale * RMS(source)` per tensor, default scale 0.01,
seed from `--growth-seed` or `seed + event_id`).
Tiny: +2 experts, both mutated.

Quality = leave-one-out held-out loss contribution
(`loss_masked(e) - loss_baseline`); usage counts are reported but never rank.
Minimum `--growth-min-eval-samples` (default 256) tokens or the trigger is
retained in `WAITING_FOR_RANKING` and training pauses — never random choice.
Ties break by expert id.

## Transaction (exactly-once committed effect)

`WAITING_FOR_RANKING -> PREPARED -> VALIDATED -> COMMITTED`, Phase A pending
checkpoint (pre-growth model + pending record) then Phase B committed
checkpoint (grown model + history + scheduler transition, atomically
published). Candidate built in isolation; adopted only after publication;
router optimizer columns carried; caches invalidated. Startup restores pending
or committed state per `docs/checkpoint.md`. Crash-recovery is fault-injection
tested (`tests/test_e2e_recovery.py`).

## Pruning

`python prune.py --rm-worst 8 --checkpoint CKPT [--dry-run] [--yes]`
ranks by the same quality metric, refuses on insufficient evidence or
top-k/registry violations, and atomically replaces the checkpoint only after
validating the pruned model (reload + inference smoke).
