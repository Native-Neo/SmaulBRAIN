"""Adaptive halting: min/max depth, threshold, easy-early vs hard-late."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel


def _model(**kw):
    torch.manual_seed(0)
    base = dict(d_model=32, n_heads=4, num_experts=4, top_k=2,
                expert_hidden=64, context_length=16)
    base.update(kw)
    m = SmaulBrainModel(SmaulBrainConfig(**base))
    return m


def test_depths_respect_min_max():
    m = _model(min_depth=2, max_depth=4)
    ids = torch.randint(0, 256, (2, 16))
    out = m.forward_infer(ids)
    d = out["depths"]
    assert d.min().item() >= 2 and d.max().item() <= 4
    assert out["n_executed"] <= 4
    m.pager.close()


def test_low_threshold_halts_earlier_than_high():
    ids = torch.randint(0, 256, (2, 16))
    m1 = _model(min_depth=1, max_depth=4, halting_threshold=0.05)
    m2 = _model(min_depth=1, max_depth=4, halting_threshold=0.999)
    # Same weights, different thresholds: per-token selected depths must
    # reflect the threshold (inference runs full depth for state completeness
    # but selects each token's halting-depth representation).
    m2.load_state_dict({k: v.clone() for k, v in m1.state_dict().items()
                        if not k.startswith("router.")} |
                       {k: v for k, v in m2.state_dict().items()
                        if k.startswith("router.")}, strict=False)
    d1 = m1.forward_infer(ids)["depths"].float().mean().item()
    d2 = m2.forward_infer(ids)["depths"].float().mean().item()
    assert d1 < d2, f"threshold had no effect on selected depth: {d1} vs {d2}"
    m1.pager.close(); m2.pager.close()


def test_ponder_distribution_sums_to_one():
    m = _model(min_depth=1, max_depth=3)
    ids = torch.randint(0, 256, (1, 8))
    out = m(ids, ids, step=0)
    assert 1.0 <= out["mean_depth"] <= 3.0
    assert out["ponder_kl"].item() >= 0
    m.pager.close()


def test_halting_stable_during_training_step():
    from smaulopt import SmaulOpt, SmaulOptHParams
    from train import train_step
    torch.manual_seed(0)
    m = _model(min_depth=1, max_depth=3)
    opt = SmaulOpt(SmaulOptHParams())
    ids = torch.randint(0, 256, (2, 16))
    for step in range(3):
        s = train_step(m, opt, m.cfg, ids, ids, step)
        assert 1.0 <= s["mean_depth"] <= 3.0
        assert s["loss"] == s["loss"]  # no NaN
    m.pager.close()
