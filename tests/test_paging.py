"""Paging: D2R / R2VR / D2VR execute genuinely different paths."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from experts import ExpertPool, make_expert
from paging import ExpertPager


def _pool(n=4, seed=0):
    torch.manual_seed(seed)
    pool = ExpertPool()
    for _ in range(n):
        pool.add(make_expert(pool.fresh_id(), 16, 32))
    return pool


def _counting(pool):
    def load(eid):
        return pool.experts[eid]
    return load


def test_d2r_disk_to_ram_never_vram():
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", ram_cache=2, load_from_disk=_counting(pool))
    e = pool.order
    pg.provider(e[0]); pg.provider(e[1]); pg.provider(e[0]); pg.provider(e[2])
    s = pg.stats
    assert (s.disk_reads, s.ram_hits, s.ram_evictions) == (3, 1, 1)
    assert s.vram_loads == 0 and len(pg.vram) == 0 and len(pg.ram) == 2
    pg.close()


def test_r2vr_stages_through_ram_never_direct_to_vram():
    pool = _pool()
    pg = ExpertPager(pool, mode="R2VR", vram_cache=2, load_from_disk=_counting(pool))
    pg.warm_ram()  # bulk staging: the only prefetch-free disk path at startup
    assert pg.stats.disk_reads == 4 and pg.stats.ram_loads == 4
    pg.provider(pool.order[0]); pg.provider(pool.order[0])
    assert pg.stats.vram_loads == 1 and pg.stats.vram_hits == 1
    # Post-growth expert (never warmed): staged disk -> RAM -> VRAM on demand.
    from experts import make_expert
    rec = make_expert("expert_00099", 16, 32)
    pool.add(rec)
    d0, r0 = pg.stats.disk_reads, pg.stats.ram_loads
    pg.provider("expert_00099")
    assert pg.stats.disk_reads == d0 + 1 and pg.stats.ram_loads == r0 + 1
    assert pg.stats.vram_loads == 2 and len(pg.ram) == 0  # RAM compute cache bypassed
    pg.close()


def test_d2vr_bypasses_ram():
    pool = _pool()
    pg = ExpertPager(pool, mode="D2VR", vram_cache=2, load_from_disk=_counting(pool))
    pg.provider(pool.order[0]); pg.provider(pool.order[0]); pg.provider(pool.order[1])
    assert len(pg.ram) == 0 and pg.stats.ram_loads == 0
    assert pg.stats.disk_reads == 2 and pg.stats.vram_hits == 1
    pg.close()


def test_modes_produce_identical_math():
    pools = [_pool() for _ in range(3)]
    for a, b in zip(pools[0].order, pools[1].order):
        pools[1].experts[b].weights_fp8 = pools[0].experts[a].weights_fp8
    for a, b in zip(pools[0].order, pools[2].order):
        pools[2].experts[b].weights_fp8 = pools[0].experts[a].weights_fp8
    pgs = [ExpertPager(pools[0], mode="D2R", load_from_disk=_counting(pools[0])),
           ExpertPager(pools[2], mode="D2VR", load_from_disk=_counting(pools[2]))]
    w0 = pgs[0].provider(pools[0].order[0])["w_gate"].float()
    w1 = pgs[1].provider(pools[2].order[0])["w_gate"].float()
    assert torch.allclose(w0.cpu(), w1.cpu(), atol=1e-3)
    for pg in pgs:
        pg.close()


def test_prefetch_loads_ahead_async():
    pool = _pool()
    pg = ExpertPager(pool, mode="D2VR", vram_cache=8, load_from_disk=_counting(pool))
    pg.prefetch([pool.order[2], pool.order[3]])
    pg.await_prefetch()
    assert pg.stats.prefetch_submitted == 2 and pg.stats.prefetch_hits == 2
    assert pool.order[2] in pg.vram and pool.order[3] in pg.vram
    pg.close()


def test_identity_independent_of_cache_slot_and_opt_state_follows():
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", ram_cache=1, load_from_disk=_counting(pool))
    e0, e1 = pool.order[0], pool.order[1]
    pool.experts[e0].optim_state["w_gate"]["m"].fill_(1.0)
    pg.provider(e0); pg.provider(e1); pg.provider(e0)  # evict + reload
    assert pool.experts[e0].optim_state["w_gate"]["m"].sum().item() > 0
    assert pool.order[0] == e0  # identity untouched by slot churn
    pg.close()


def test_prefetch_then_forget_leaves_no_ghost():
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    eid = pool.order[0]
    pg.prefetch([eid])
    pg.forget(eid)  # prune won the race: cancel/drop the pending load
    pg.await_prefetch()  # must not raise and must not count a ghost hit
    assert pg.stats.prefetch_hits == 0
    assert eid not in pg.ram and eid not in pg._pending
    pg.close()


def test_provider_joins_inflight_prefetch_without_reload():
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    eid = pool.order[0]
    pg.prefetch([eid])
    w = pg.provider(eid)  # joins the background fetch instead of loading twice
    assert pg.stats.disk_reads == 1
    assert eid in pg.ram and w is pg.ram[eid]
    pg.await_prefetch()  # already joined: nothing left to await or count
    assert pg.stats.prefetch_hits == 0
    pg.close()


def test_train_step_invalidates_only_stepped_experts():
    from config import SmaulBrainConfig
    from model import SmaulBrainModel
    from smaulopt import SmaulOpt, SmaulOptHParams
    from train import train_step
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=24,
                           expert_lr=3e-2)
    m = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    for eid in m.pool.order:
        m.pager.provider(eid)  # warm every compute cache
    assert len(m.pager.ram) == 4
    b = torch.randint(0, 256, (2, cfg.context_length + 1))
    s = train_step(m, opt, cfg, b[:, :cfg.context_length], b[:, 1:], step=0)
    assert s["stepped_experts"], "update path must run for this test"
    for eid in m.pool.order:
        if eid in s["stepped_experts"]:
            assert eid not in m.pager.ram  # rewritten: stale serve impossible
        else:
            assert eid in m.pager.ram  # untouched: cache hums along
    m.pager.close()
