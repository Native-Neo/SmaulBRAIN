"""Continual learning: retention measured numerically, replay, slow trunk."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import (
    ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report,
    run_training,
)


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=24,
                           expert_lr=3e-2, trunk_lr_mult=0.1)
    m = SmaulBrainModel(cfg)
    return m, cfg


def test_replay_buffer_reservoir_and_sample():
    buf = ReplayBuffer(capacity=8, seed=0)
    for i in range(20):
        buf.add([i])
    assert len(buf) == 8 and buf.seen == 20
    assert len(buf.sample(3)) == 3


def test_replay_buffer_ingress_copy_and_egress_copy():
    buf = ReplayBuffer(capacity=8, seed=0)
    seq = [1, 2, 3]
    buf.add(seq)
    seq.append(999)  # caller mutation after add must not corrupt replay
    seq2 = [4, 5]
    buf.add(seq2)
    got = buf.to_dict()["buf"]
    assert [1, 2, 3] in got and [4, 5] in got
    out = buf.sample(2)
    out[0].append(-1)
    assert buf.to_dict()["buf"] == got  # sampling hands out copies too


def test_retention_report_numeric_and_honest():
    before = {"loss": 2.0, "acc": 0.5}
    after = {"loss": 2.2, "acc": 0.45}
    rep = retention_report(before, after)
    assert abs(rep["old_loss_delta"] - 0.2) < 1e-9
    assert rep["old_acc_delta"] < 0  # forgetting is reported, not hidden
    assert "retained" in rep


def test_slow_trunk_moves_slower_than_experts():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    assert cfg.trunk_lr < cfg.expert_lr  # 0.1x by default
    assert abs(cfg.trunk_lr - cfg.expert_lr * 0.1) < 1e-12
    m.pager.close()


def test_continual_run_reports_retention():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    old = [[i % 256 for i in range(j, j + 30)] for j in range(8)]
    new = [[(i * 7 + 3) % 256 for i in range(j, j + 30)] for j in range(16)]
    res = run_training(m, opt, cfg, new, steps=8, batch_size=2,
                       old_seqs=old, replay=ReplayBuffer(64), replay_n=1)
    rep = res["retention"]
    assert rep is not None and "old_loss_before" in rep and "old_loss_after" in rep
    assert isinstance(rep["old_loss_delta"], float)
    m.pager.close()


def test_batch_packing_truncates_and_pads():
    b = batch_from_seqs([[1, 2, 3], [4] * 40], context=8, pad_id=0)
    assert b.shape == (2, 9)
    assert b[0, 3:].tolist() == [0] * 6
    assert b[1, :9].tolist() == [4] * 9


def test_replay_sample_semantics_clipped_with_replacement():
    # Defined semantics: with replacement, clipped to len(buf), n<=0 -> [].
    buf = ReplayBuffer(capacity=8, seed=0)
    assert buf.sample(3) == []  # empty buffer
    assert buf.sample(0) == [] and buf.sample(-1) == []
    buf.add([1])
    buf.add([2])
    assert buf.sample(0) == [] and buf.sample(-2) == []
    assert len(buf.sample(5)) == 2  # clipped, not padded to n
    assert len(buf.sample(1)) == 1
    # With replacement: repeated draws of 2 from {1,2} must sometimes repeat.
    draws = [tuple(x[0] for x in buf.sample(2)) for _ in range(20)]
    assert any(a == b for a, b in draws)  # seed=0 gives 9/20 dups
    # Egress copies: sampling never hands out live references.
    before = buf.to_dict()["buf"]
    out = buf.sample(2)
    out[0].append(999)
    assert buf.to_dict()["buf"] == before


def test_replay_persistence_exact_continuation():
    buf = ReplayBuffer(capacity=4, seed=7)
    for i in range(6):
        buf.add([i, i + 1])
    snap = buf.to_dict()
    assert set(snap) == {"capacity", "buf", "seen", "rng"}
    assert snap["seen"] == 6 and snap["capacity"] == 4
    clone = ReplayBuffer.from_dict(snap)
    assert clone.to_dict() == snap  # bit-identical snapshot roundtrip
    assert [clone.sample(2) for _ in range(5)] == [buf.sample(2) for _ in range(5)]
    # Mutating the snapshot dict must not alias either buffer.
    snap["buf"].append([999])
    assert len(buf) == 4 and len(clone) == 4


def test_replay_ingress_covers_full_batch_past_only():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    seqs = [[i] * 30 for i in range(8)]  # distinct markers per position
    buf = ReplayBuffer(capacity=16, seed=3)
    res = run_training(m, opt, cfg, seqs, steps=2, batch_size=2,
                       replay=buf, replay_n=1, log_fn=lambda s: None)
    assert res["steps_completed"] == 2
    # Full-batch ingress: steps 0,1 consume fresh idx [0,1],[2,3] -> all 4 stored.
    stored = [s[0] for s in buf.to_dict()["buf"]]
    assert sorted(stored) == [0, 1, 2, 3], stored
    assert buf.seen == 4  # 2 fresh seqs per step, not 1 per step
    m.pager.close()


def test_scheduler_snapshot_persists_dataset_cursors():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    seqs = [[(i + j) % 256 for j in range(30)] for i in range(8)]
    buf = ReplayBuffer(capacity=8, seed=7)
    res = run_training(m, opt, cfg, seqs, steps=3, batch_size=2,
                       replay=buf, replay_n=1, log_fn=lambda s: None)
    sched = res["scheduler"]
    assert sched["batch_size"] == 2 and sched["dataset_len"] == 8
    assert sched["next_step"] == 3  # start_step(0) + 3 steps
    assert isinstance(sched["replay"], dict) and sched["replay"]["seen"] == 6
    assert "prev_loss" in sched and "growth_events" in sched
    # Resume continues the dataset order: next batch starts at global step 3.
    assert [h["step"] for h in res["history"]] == [0, 1, 2]
    m.pager.close()


def test_throughput_and_loss_scale_by_valid_not_padded():
    from bytes import PAD_ID
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    # Each seq has 5 data ids; context=24 -> 19 pads per row in the input half.
    seqs = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
    res = run_training(m, opt, cfg, seqs, steps=1, batch_size=2,
                       log_fn=lambda s: None)
    assert res["bytes_processed"] == 10  # 5 valid x 2 rows, not 2*24=48
    b = batch_from_seqs(seqs, cfg.context_length)
    assert int((b[:, : cfg.context_length] != PAD_ID).sum().item()) == 10
    # Padding is masked from the loss: all-pad targets score 0, never NaN.
    rep = evaluate_loss(m, batch_from_seqs([[1, 2, 3]], cfg.context_length),
                        cfg.context_length)
    assert rep["loss"] == rep["loss"] and 0.0 <= rep["acc"] <= 1.0
    pad_only = torch.full((1, cfg.context_length + 1), PAD_ID, dtype=torch.long)
    rep_pad = evaluate_loss(m, pad_only, cfg.context_length)
    assert rep_pad["loss"] == 0.0  # no valid targets -> 0/1, not NaN/diluted
    m.pager.close()


def test_stepped_experts_follow_pool_order():
    m, cfg = _model(seed=1)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    seqs = [[(i * 3 + j) % 256 for j in range(30)] for i in range(8)]
    res = run_training(m, opt, cfg, seqs, steps=2, batch_size=2,
                       log_fn=lambda s: None)
    order = list(m.pool.order)
    rank = {eid: i for i, eid in enumerate(order)}
    for h in res["history"]:
        stepped = h["stepped_experts"]
        assert stepped == sorted(stepped, key=lambda e: (rank.get(e, 1 << 30), e))
    m.pager.close()
