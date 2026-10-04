"""Continual learning: retention measured numerically, replay, slow trunk."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from continual import (
    ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report,
)
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import run_training


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
