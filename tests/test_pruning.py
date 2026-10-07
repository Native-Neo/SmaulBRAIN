"""Pruning: grace periods, hysteresis, full removal, post-prune checkpoints."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from experts import ExpertPool, make_expert
from pruning import find_victims, prune_experts
from routing import SparseRouter


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
    from config import SmaulBrainConfig
    from model import SmaulBrainModel
    from smaulopt import SmaulOpt
    from storage import save_model, load_model
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=16, n_heads=2, num_experts=3, top_k=1,
                           expert_hidden=32, max_depth=1, min_experts=1)
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
                                         top_k=1, expert_hidden=32, max_depth=1, min_experts=1))
    man = load_model(d, m2, SmaulOpt())
    assert man["expert_ids"] == [keep] and len(m2.pool) == 1
    ids = torch.randint(0, 256, (1, 8))
    assert m2.forward_infer(ids)["logits"].shape == (1, 8, m2.cfg.vocab_size)
    m.pager.close(); m2.pager.close()


def test_prune_unknown_victim_aborts_without_mutating():
    pool, r = _pool()
    order = list(pool.order)
    with pytest.raises(ValueError):
        prune_experts(pool, r, ["expert_00001", "expert_99999"])
    assert pool.order == order and len(pool) == 4 and r.num_experts == 4


def test_prune_diverged_topology_refuses():
    pool, r = _pool()
    r.add_expert_row()  # diverged: 4 experts vs 5 router rows
    with pytest.raises(ValueError):
        prune_experts(pool, r, ["expert_00001"])
    assert len(pool) == 4 and "expert_00001" in pool.experts


def test_prune_forgets_pager_caches_inline():
    from paging import ExpertPager
    pool, r = _pool()
    pg = ExpertPager(pool, mode="D2R", ram_cache=4,
                     load_from_disk=lambda eid: pool.experts[eid])
    eid = pool.order[0]
    pg.provider(eid)
    assert eid in pg.ram
    assert prune_experts(pool, r, [eid], pager=pg) == [eid]
    assert eid not in pg.ram and r.num_experts == 3
    with pytest.raises(KeyError):
        pg.provider(eid)  # no ghost served after removal
    pg.close()


def test_prune_keeps_survivor_momentum_aligned():
    from smaulopt import SmaulOpt, SmaulOptHParams
    pool, r = _pool()
    opt = SmaulOpt(SmaulOptHParams())
    d = r.proj.weight.shape[1]
    m = torch.stack([torch.full((d,), 100.0 + i) for i in range(4)])
    opt.router_state["w"] = {"m": m, "step": 7,
                             "v_row": torch.zeros(4, 1),
                             "v_col": torch.zeros(1, d)}
    assert prune_experts(pool, r, ["expert_00001"], optim_state=opt.router_state)
    got = opt.router_state["w"]["m"]
    assert got.shape == (3, d)
    for i, want in enumerate((100.0, 102.0, 103.0)):  # victim row excised
        assert torch.equal(got[i], torch.full((d,), want))
    assert opt.router_state["w"]["step"] == 7


def test_failing_side_channel_leaves_topology_consistent():
    """Rollback containment: if a side-channel (pager) blows up mid-sweep,
    the committed prefix still has pool/router in lockstep and the error
    surfaces instead of a silent divergence."""
    pool, r = _pool()

    class BoomPager:
        def forget(self, eid):
            raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError):
        prune_experts(pool, r, ["expert_00001", "expert_00002"],
                      pager=BoomPager())
    assert len(pool) == r.num_experts == 3  # committed prefix stays synced
    assert "expert_00002" not in pool.experts  # highest index goes first
    assert "expert_00001" in pool.experts  # uncommitted victim untouched


def test_prune_rejects_mismatched_optim_width_without_mutating():
    """Validate-first: stale momentum widths refuse before any removal."""
    from smaulopt import SmaulOpt, SmaulOptHParams
    pool, r = _pool()
    opt = SmaulOpt(SmaulOptHParams())
    d = r.proj.weight.shape[1]
    # Stale width: 2 rows vs 4 experts (e.g. leftover from a prior crash).
    opt.router_state["w"] = {"m": torch.zeros(2, d), "step": 3,
                             "v_row": torch.zeros(2, 1),
                             "v_col": torch.zeros(1, d)}
    order = list(pool.order)
    w_before = r.proj.weight.detach().clone()
    with pytest.raises(ValueError):
        prune_experts(pool, r, ["expert_00001"], optim_state=opt.router_state)
    assert pool.order == order and r.num_experts == 4  # nothing mutates
    assert torch.equal(r.proj.weight, w_before)
    assert opt.router_state["w"]["m"].shape == (2, d)  # stale table untouched


def test_prune_momentum_drop_is_two_phase_on_build_failure(monkeypatch):
    """Atomic drop: a cat failure mid-build leaves every buffer untouched."""
    import pruning as pruning_mod
    pool, r = _pool()
    from smaulopt import SmaulOpt, SmaulOptHParams
    opt = SmaulOpt(SmaulOptHParams())
    d = r.proj.weight.shape[1]
    m = torch.stack([torch.full((d,), 100.0 + i) for i in range(4)])
    opt.router_state["w"] = {"m": m.clone(), "step": 7,
                             "v_row": torch.zeros(4, 1),
                             "v_col": torch.zeros(1, d)}
    m_before = opt.router_state["w"]["m"].clone()
    vr_before = opt.router_state["w"]["v_row"].clone()
    real_cat = torch.cat
    calls = {"n": 0}

    def boom_once(tensors, *a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("cat on fire")
        return real_cat(tensors, *a, **k)
    monkeypatch.setattr(torch, "cat", boom_once)
    try:
        with pytest.raises(RuntimeError):
            pruning_mod._drop_state_row(opt.router_state, 1)
    finally:
        monkeypatch.setattr(torch, "cat", real_cat)
    # Two-phase build: no partial assignment happened.
    assert torch.equal(opt.router_state["w"]["m"], m_before)
    assert torch.equal(opt.router_state["w"]["v_row"], vr_before)
    assert opt.router_state["w"]["m"].shape == (4, d)
