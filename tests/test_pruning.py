"""Pruning: grace periods, hysteresis, full removal, post-prune checkpoints."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from smaulbrain.experts import ExpertPool, make_expert
from smaulbrain.pruning import find_victims, prune_experts
from smaulbrain.routing import SparseRouter


def _pool(n=4, seed=0):
    torch.manual_seed(seed)
    pool = ExpertPool()
    for _ in range(n):
        pool.add(make_expert(pool.fresh_id(), 16, 32, birth_step=0))
    return pool, SparseRouter(16, n, 2)


def test_grace_period_protects_young_experts():
    pool, r = _pool()
    assert find_victims(pool, step=100, survival_steps=500, min_experts=1) == []


def test_idle_low_signal_experts_are_victims_but_useful_survive():
    pool, r = _pool()
    u = pool.experts["expert_00000"]
    u.tokens_routed = 1000; u.grad_activity = 1e-3
    u.contribution = 1.0; u.last_used_step = 600
    v = find_victims(pool, step=600, survival_steps=500, min_experts=1)
    assert set(v) == {"expert_00001", "expert_00002", "expert_00003"}


def test_single_signal_not_enough_hysteresis():
    pool, r = _pool()
    u = pool.experts["expert_00001"]
    u.tokens_routed = 0  # unused...
    u.grad_activity = 1e-2  # ...but gradient-active -> survives
    u.last_used_step = 0
    v = find_victims(pool, step=600, survival_steps=500, min_experts=1)
    assert "expert_00001" not in v


def test_prune_removes_everything_router_included():
    pool, r = _pool()
    u = pool.experts["expert_00000"]
    u.tokens_routed = 500; u.grad_activity = 1e-3
    u.contribution = 0.5; u.last_used_step = 600
    victims = find_victims(pool, step=600, survival_steps=500, min_experts=1)
    opt_keys_before = set(pool.experts[victims[0]].optim_state)
    assert opt_keys_before == {"w_gate", "w_up", "w_down"}
    pruned = prune_experts(pool, r, victims)
    assert len(pruned) == 3 and len(pool) == 1
    assert r.num_experts == 1
    for eid in pruned:  # weights + optimizer state + metadata gone
        assert eid not in pool.experts
    assert pool.order == ["expert_00000"] and r.proj.weight.shape[0] == 1


def test_min_experts_floor_never_violated():
    pool, r = _pool(n=2)
    v = find_victims(pool, step=9999, survival_steps=10, min_experts=2)
    assert v == []


def test_checkpoint_correct_after_pruning(tmp_path):
    from smaulbrain.config import SmaulBrainConfig
    from smaulbrain.model import SmaulBrainModel
    from smaulbrain.smaulopt import SmaulOpt
    from smaulbrain.storage import save_model, load_model
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=16, n_heads=2, num_experts=3, top_k=1,
                           expert_hidden=32, max_depth=1)
    m = SmaulBrainModel(cfg); opt = SmaulOpt()
    keep = m.pool.order[0]
    m.pool.experts[keep].tokens_routed = 100
    m.pool.experts[keep].grad_activity = 1e-3
    m.pool.experts[keep].contribution = 1.0
    m.pool.experts[keep].last_used_step = 600
    victims = find_victims(m.pool, step=600, survival_steps=500, min_experts=1)
    prune_experts(m.pool, m.router, victims)
    d = str(tmp_path / "ckpt")
    save_model(d, m, opt, step=600)
    m2 = SmaulBrainModel(SmaulBrainConfig(d_model=16, n_heads=2, num_experts=1,
                                         top_k=1, expert_hidden=32, max_depth=1))
    man = load_model(d, m2, SmaulOpt())
    assert man["expert_ids"] == [keep] and len(m2.pool) == 1
    ids = torch.randint(0, 256, (1, 8))
    assert m2.forward_infer(ids)["logits"].shape == (1, 8, 256)
    m.pager.close(); m2.pager.close()
