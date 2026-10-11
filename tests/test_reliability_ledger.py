"""m4 regression: zero-effective-LR ledger semantics are pinned, not fixed.

stepped := version-delta (moments moved), not weight movement. A zero
effective LR still bumps versions, still invalidates the pager, and still
appears in stepped even though weights are numerically unmoved. This file
pins that behavior so future readers do not "fix" the churn.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import batch_from_seqs, cosine_scale, train_step


def _tiny_zero_lr_cfg():
    return SmaulBrainConfig(
        d_model=32, n_heads=4, num_experts=2, top_k=1, expert_hidden=16,
        max_depth=2, min_depth=1, context_length=16,
        dtype="fp32", state_dtype="fp32", threads=1, seed=0, min_experts=1,
        cosine_decay_steps=5,
    )


def test_zero_effective_lr_still_steps_and_invalidates():
    cfg = _tiny_zero_lr_cfg()
    assert cosine_scale(5, cfg.cosine_decay_steps) == 0.0
    torch.manual_seed(0)
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, state_dtype=cfg.state_dtype))
    try:
        seqs = [[i % 256 for i in range(17)] for _ in range(4)]
        b = batch_from_seqs(seqs, cfg.context_length)
        vers_before = {eid: int(model.pool.experts[eid].version) for eid in model.pool.order}
        codes_before = {
            eid: model.pool.experts[eid].weights_fp8["w_gate"].codes.clone()
            for eid in model.pool.order
        }
        stats = train_step(model, opt, cfg, b[:, :cfg.context_length],
                           b[:, 1:cfg.context_length + 1], step=5, mode="entire")
        # Zero LR still counts as stepped (version-delta, moments moved).
        assert len(stats["stepped_experts"]) > 0
        for eid in stats["stepped_experts"]:
            assert int(model.pool.experts[eid].version) != vers_before[eid]
            # Weights numerically unmoved (FP8 codes bit-identical here).
            assert torch.equal(codes_before[eid],
                               model.pool.experts[eid].weights_fp8["w_gate"].codes)
    finally:
        model.pager.close()


def test_stepped_means_version_delta():
    """Normal-LR control: every stepped id bumped its version."""
    cfg = _tiny_zero_lr_cfg()
    cfg.cosine_decay_steps = 0  # constant LR
    torch.manual_seed(1)
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, state_dtype=cfg.state_dtype))
    try:
        seqs = [[(i + j) % 256 for j in range(17)] for i in range(4)]
        b = batch_from_seqs(seqs, cfg.context_length)
        vers_before = {eid: int(model.pool.experts[eid].version) for eid in model.pool.order}
        stats = train_step(model, opt, cfg, b[:, :cfg.context_length],
                           b[:, 1:cfg.context_length + 1], step=0, mode="entire")
        assert len(stats["stepped_experts"]) > 0
        for eid in stats["stepped_experts"]:
            assert int(model.pool.experts[eid].version) != vers_before[eid]
    finally:
        model.pager.close()
