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


def _stress_provider(mode):
    import random
    import threading
    pool = _pool()
    pg = ExpertPager(pool, mode=mode, load_from_disk=_counting(pool))
    ref = {}
    for eid in pool.order:
        ref[eid] = {k: v.clone() for k, v in pg.provider(eid).items()}
    pg.ram.clear(); pg.vram.clear(); pg.ram_records.clear()
    rng = random.Random(0)
    plans = [[pool.order[rng.randrange(len(pool))] for _ in range(20)] for _ in range(8)]
    barrier = threading.Barrier(len(plans) + 1)
    errors: list = []

    def worker(p):
        try:
            barrier.wait(timeout=30)
            for eid in p:
                got = pg.provider(eid)
                for k in got:
                    assert torch.equal(got[k], ref[eid][k])
        except Exception as e:  # noqa: BLE001 - collected, then asserted
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p,)) for p in plans]
    for t in threads:
        t.start()
    barrier.wait(timeout=30)
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert all(not t.is_alive() for t in threads)
    pg.close()


def test_concurrent_providers_match_sequential():
    _stress_provider("D2R")
    _stress_provider("R2VR")


def test_concurrent_duplicate_prefetch_collapses():
    import threading
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    barrier = threading.Barrier(4)

    def storm():
        barrier.wait(timeout=30)
        pg.prefetch([pool.order[0], pool.order[1], pool.order[0]])

    threads = [threading.Thread(target=storm) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert pg.stats.prefetch_submitted == 2  # collapsed despite the race
    pg.await_prefetch()
    pg.close()


def test_caches_stay_bounded_and_close_is_repeatable():
    pool = _pool()
    pg = ExpertPager(pool, mode="R2VR", ram_cache=2, vram_cache=2,
                     load_from_disk=_counting(pool))
    for _ in range(3):
        for eid in pool.order:
            pg.provider(eid)
        pg.prefetch(list(pool.order))
        pg.await_prefetch()
    counts = pg.resident_counts()
    assert counts["vram"] <= 2  # LRU caps hold under churn, no leak
    assert not pg._pending and not pg._prefetched  # nothing left in flight
    pg.close()
    pg.close()  # repeatable shutdown


def test_update_vs_prefetch_no_ghost_hit():
    import time
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    eid = pool.order[0]
    pg.prefetch([eid])
    import time as _t
    _t.sleep(0.05)
    rec = pool.experts[eid]
    from precision import dequantize_fp8_blockwise, quantize_fp8_blockwise
    base = dequantize_fp8_blockwise(rec.weights_fp8["w_gate"], dtype=torch.float32)
    rec.weights_fp8["w_gate"] = quantize_fp8_blockwise(
        base + 5.0, tile=rec.weights_fp8["w_gate"].tile)
    pg.invalidate(eid)  # update won the race: prior prefetch bytes are stale
    assert eid not in pg._prefetched  # invalidate discards the stale flag
    pg.await_prefetch()
    assert pg.stats.prefetch_hits == 0  # ghost hit must not be counted
    w = pg.provider(eid)  # fresh post-update bytes
    live = dequantize_fp8_blockwise(
        pool.experts[eid].weights_fp8["w_gate"], dtype=torch.float32)
    live = live.to(pg.compute_dtype).float()
    assert torch.allclose(w["w_gate"].float().cpu(), live.cpu(), atol=1e-3)
    pg.close()


def test_prune_vs_load_no_ghost():
    import threading
    import time
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    eid = pool.order[0]

    def slow(e):
        time.sleep(0.08)
        return pool.experts[e]
    pg.load_from_disk = slow
    t = threading.Thread(target=lambda: pg.prefetch([eid]))
    t.start()
    t.join(timeout=60)
    pg.forget(eid)  # prune won the race while the load was in flight
    try:
        pg.provider(eid)
        assert False, "pruned expert must raise"
    except KeyError:
        pass
    pg.await_prefetch()
    assert pg.stats.prefetch_hits == 0
    assert eid not in pg.ram and eid not in pg._pending
    pg.close()


def test_restore_vs_prefetch_readable():
    from experts import make_expert
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    eid = pool.order[0]
    pg.provider(eid)
    pg.forget(eid)
    try:
        pg.provider(eid)
        assert False, "forgotten id must stay unreadable until restore"
    except KeyError:
        pass
    rec = make_expert(eid, 16, 32)  # restored under the same stable id
    pool.experts[eid] = rec
    if eid not in pool.order:
        pool.order.append(eid)
    pg.restore(eid)
    w = pg.provider(eid)  # readable again, and tagged to the new object
    assert w is not None
    pg.prefetch([eid])  # post-restore prefetch collapses on the fresh entry
    pg.await_prefetch()
    assert eid in pg.ram
    pg.close()


def test_replace_without_invalidate_serves_fresh():
    from experts import make_expert
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", load_from_disk=_counting(pool))
    eid = pool.order[0]
    w0 = pg.provider(eid)
    pool.experts[eid] = make_expert(eid, 16, 32)  # incompatible replacement
    w1 = pg.provider(eid)  # tag (version+identity) must force a reload
    assert w1 is not w0
    pg.close()


def test_rapid_version_changes_converge():
    import threading
    import time
    from precision import dequantize_fp8_blockwise, quantize_fp8_blockwise
    for mode in ("D2R", "R2VR", "D2VR"):
        pool = _pool()
        pg = ExpertPager(pool, mode=mode, load_from_disk=_counting(pool))
        eid = pool.order[0]
        pg.provider(eid)
        errors: list = []

        def updater():
            try:
                for _ in range(15):
                    rec = pool.experts[eid]
                    base = dequantize_fp8_blockwise(
                        rec.weights_fp8["w_gate"], dtype=torch.float32)
                    rec.weights_fp8["w_gate"] = quantize_fp8_blockwise(
                        base + 1.0, tile=rec.weights_fp8["w_gate"].tile)
                    pg.invalidate(eid)
                    pg.prefetch([eid])
                    time.sleep(0.002)
            except Exception as e:  # noqa: BLE001 - collected
                errors.append(e)

        def reader():
            try:
                for _ in range(25):
                    try:
                        pg.provider(eid)
                    except KeyError:
                        pass
                    time.sleep(0.001)
            except Exception as e:  # noqa: BLE001 - collected
                errors.append(e)

        up = threading.Thread(target=updater)
        readers = [threading.Thread(target=reader) for _ in range(3)]
        up.start()
        for r in readers:
            r.start()
        up.join(timeout=60)
        for r in readers:
            r.join(timeout=60)
        assert not errors
        pg.await_prefetch()
        w = pg.provider(eid)
        live = dequantize_fp8_blockwise(
            pool.experts[eid].weights_fp8["w_gate"], dtype=torch.float32)
        live = live.to(pg.compute_dtype).float()
        if mode == "R2VR" or mode == "D2VR":
            live = live.to(pg.vram_device).float()
        assert torch.allclose(w["w_gate"].float().cpu(), live.cpu(), atol=1e-3)
        pg.close()


def test_stats_snapshot_thread_safe_under_contention():
    import threading
    pool = _pool()
    pg = ExpertPager(pool, mode="D2R", ram_cache=8, load_from_disk=_counting(pool))
    for eid in pool.order:
        pg.provider(eid)
    base = pg.snapshot_stats()["ram_hits"]
    n, m = 8, 50

    def hammer():
        for _ in range(m):
            for eid in pool.order:
                pg.provider(eid)

    threads = [threading.Thread(target=hammer) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    snap = pg.snapshot_stats()
    assert snap["ram_hits"] == base + n * m * len(pool.order)
    pg.close()
