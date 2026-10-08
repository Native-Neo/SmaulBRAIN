"""Experts: creation, stable IDs, FP8 roundtrip, optimizer state, growth."""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from experts import ExpertPool, make_expert
from growth import grow_expert, grow_topk_clones, recombine_weights, select_parents
from precision import dequantize_fp8_blockwise
from routing import SparseRouter


def _pool(n=3, d=16, h=32, seed=0):
    torch.manual_seed(seed)
    pool = ExpertPool()
    for _ in range(n):
        pool.add(make_expert(pool.fresh_id(), d, h))
    return pool


def test_expert_size_explicit_and_512_target():
    from config import SmaulBrainConfig, expert_hidden_for_target
    cfg = SmaulBrainConfig(d_model=512, expert_hidden=expert_hidden_for_target(512))
    assert abs(cfg.per_expert_params - 5_120_000) / 5_120_000 < 0.02


def test_ids_stable_unique_monotonic():
    pool = _pool()
    assert pool.order == ["expert_00000", "expert_00001", "expert_00002"]
    assert pool.fresh_id() == "expert_00003"


def test_fp8_storage_is_real_uint8_with_scales():
    pool = _pool(n=1)
    rec = pool.experts["expert_00000"]
    for n in ("w_gate", "w_up", "w_down"):
        t = rec.weights_fp8[n]
        assert t.codes.dtype == torch.uint8  # genuine FP8 bytes, not fp32 labels
        assert t.scales.dtype == torch.float32
    full = dequantize_fp8_blockwise(rec.weights_fp8["w_gate"]).float()
    assert full.shape == (32, 16)


def test_each_expert_owns_optimizer_state():
    pool = _pool(n=2)
    a, b = pool.experts["expert_00000"], pool.experts["expert_00001"]
    assert set(a.optim_state) == {"w_gate", "w_up", "w_down"}
    a.optim_state["w_gate"]["m"] += 1.0
    assert b.optim_state["w_gate"]["m"].abs().sum().item() == 0.0  # not shared


def test_growth_recombines_not_random():
    pool = _pool()
    pool.experts["expert_00001"].contribution = 9.0
    r = SparseRouter(16, 3, 2)
    eid = grow_expert(pool, r, 16, 32, step=10, seed=7)
    rec = pool.experts[eid]
    assert rec.source == "recombine" and rec.parents[0] == "expert_00001"
    assert rec.birth_step == 10 and rec.optim_state  # optimizer state initialized
    assert r.num_experts == 4  # router entry initialized


def test_growth_reproducible_and_empty_pool_fallback():
    pool = _pool()
    r = SparseRouter(16, 3, 2)
    pool_b, r_b = copy.deepcopy(pool), copy.deepcopy(r)
    a = grow_expert(pool, r, 16, 32, step=5, seed=123)
    b = grow_expert(pool_b, r_b, 16, 32, step=5, seed=123)
    assert a == b == "expert_00003"
    assert torch.equal(pool.experts[a].weights_fp8["w_gate"].codes,
                       pool_b.experts[b].weights_fp8["w_gate"].codes)
    empty, re_ = ExpertPool(), SparseRouter(16, 0, 1)
    eid = grow_expert(empty, re_, 16, 32, step=0, seed=0)
    assert empty.experts[eid].source == "init"  # documented fallback
    assert len(empty) == re_.num_experts == 1  # restart stays pool/router synced


def _synced(n=4):
    pool = _pool(n=n)
    return pool, SparseRouter(16, n, 2)


def test_grow_topk_clones_never_exceeds_max_experts():
    pool, r = _synced(n=4)
    ids = grow_topk_clones(pool, r, 16, 32, step=10, seed=3, k=8, max_experts=6)
    assert len(ids) == 2 and len(pool) == 6 and r.num_experts == 6
    assert grow_topk_clones(pool, r, 16, 32, step=11, seed=4, k=8,
                            max_experts=6) == []
    assert len(pool) == 6 and r.num_experts == 6  # at cap: nothing mutates


def test_grow_expert_at_capacity_raises_before_mutating():
    pool, r = _synced(n=2)
    with pytest.raises(ValueError):
        grow_expert(pool, r, 16, 32, step=1, seed=0, max_experts=2)
    assert len(pool) == 2 and r.num_experts == 2


def test_grow_refuses_diverged_topology_without_mutating():
    pool, _ = _synced(n=2)
    r = SparseRouter(16, 3, 2)  # diverged: 2 experts vs 3 router rows
    order = list(pool.order)
    with pytest.raises(ValueError):
        grow_expert(pool, r, 16, 32, step=1, seed=0)
    assert pool.order == order and len(pool) == 2 and r.num_experts == 3


def _marked_router_state(nrows, d=16):
    from smaulopt import SmaulOpt, SmaulOptHParams
    opt = SmaulOpt(SmaulOptHParams())
    m = torch.stack([torch.full((d,), 100.0 + i) for i in range(nrows)])
    opt.router_state["w"] = {"m": m, "step": 5,
                             "v_row": torch.zeros(nrows, 1),
                             "v_col": torch.zeros(1, d)}
    return opt


def test_growth_preserves_survivor_momentum():
    pool, r = _synced(n=3)
    opt = _marked_router_state(3)
    ids = grow_topk_clones(pool, r, 16, 32, step=1, seed=0, k=2,
                           optim_state=opt.router_state)
    assert len(ids) == 2 and r.num_experts == 5
    m = opt.router_state["w"]["m"]
    assert m.shape == (5, 16)
    for i in range(3):  # survivors bit-identical
        assert torch.equal(m[i], torch.full((16,), 100.0 + i))
    assert (m[3:] == 0).all()  # newborns start at zero momentum
    assert opt.router_state["w"]["step"] == 5


def test_recombine_is_parent_mean_plus_noise():
    pool = _pool()
    w = recombine_weights(pool, ["expert_00000", "expert_00001"],
                          weights=[0.5, 0.5], noise_std=0.0)
    from precision import dequantize_fp8_blockwise as dq
    pa = dq(pool.experts["expert_00000"].weights_fp8["w_gate"]).float()
    pb = dq(pool.experts["expert_00001"].weights_fp8["w_gate"]).float()
    assert torch.allclose(w["w_gate"], (pa + pb) / 2, atol=1e-5)


def test_parents_rank_by_contribution():
    pool = _pool()
    pool.experts["expert_00002"].contribution = 3.0
    pool.experts["expert_00000"].contribution = 1.0
    assert select_parents(pool, k=1) == ["expert_00002"]


def test_dispatch_matches_admitted_weighted_combination():
    from experts import swiglu_forward
    pool = _pool(n=3)
    torch.manual_seed(1)
    x = torch.randn(6, 16)
    # Fixed plan: token t uses experts (t%3, (t+1)%3) with weights (.7,.3);
    # token 5 is dropped and must contribute nothing.
    top_ids = torch.tensor([[i % 3, (i + 1) % 3] for i in range(6)])
    top_weights = torch.tensor([[0.7, 0.3]] * 6)
    dropped = torch.tensor([False] * 5 + [True])
    prov = {e: pool.experts[e].dequantize(torch.float32) for e in pool.order}
    out = pool.forward(x, top_ids, top_weights, dropped, lambda e: prov[e])
    for t in range(5):
        expect = torch.zeros(16)
        for slot, w in ((0, 0.7), (1, 0.3)):
            eid = pool.order[int(top_ids[t, slot])]
            expect += w * swiglu_forward(x[t : t + 1], prov[eid])[0]
        assert torch.allclose(out[t], expect, atol=1e-5), t
    assert torch.equal(out[5], torch.zeros(16))  # dropped token: pure residual
    assert pool.experts["expert_00000"].tokens_routed == 3  # live tokens only


def test_train_step_leaves_unstepped_experts_bit_identical():
    """Only routed (stepped) experts are decoded/updated/requantized.

    Unstepped experts' FP8 codes AND scales must survive a training step
    bit for bit — the expert-granularity half of the touched-blocks rule
    (intra-expert row-block granularity is audit 028's follow-up).
    """
    import random
    from config import SmaulBrainConfig
    from model import SmaulBrainModel
    from smaulopt import SmaulOpt, SmaulOptHParams
    from train import train_step

    torch.manual_seed(0)
    random.seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=24)
    m = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    before = {eid: {n: (rec.weights_fp8[n].codes.clone(),
                        rec.weights_fp8[n].scales.clone())
                    for n in ("w_gate", "w_up", "w_down")}
              for eid, rec in m.pool.experts.items()}
    b = torch.randint(0, 256, (2, cfg.context_length + 1))
    stats = train_step(m, opt, cfg, b[:, :cfg.context_length], b[:, 1:], step=0)
    assert len(stats["stepped_experts"]) > 0  # update path actually ran
    for eid, rec in m.pool.experts.items():
        for n in ("w_gate", "w_up", "w_down"):
            same_codes = torch.equal(rec.weights_fp8[n].codes, before[eid][n][0])
            same_scales = torch.equal(rec.weights_fp8[n].scales, before[eid][n][1])
            if eid in stats["stepped_experts"]:
                continue  # stepped experts may legitimately change
            assert same_codes and same_scales, (eid, n)  # untouched: bit identical
    m.pager.close()


def test_growth_build_failure_commits_nothing():
    """Two-phase rollback: a phase-1 build failure leaves pool/router exact."""
    pool, r = _synced(n=3)
    pool.experts["expert_00000"].weights_fp8["w_gate"] = None  # poison build
    order = list(pool.order)
    with pytest.raises(Exception):
        grow_topk_clones(pool, r, 16, 32, step=1, seed=0, k=3)
    assert pool.order == order and r.num_experts == 3


def test_expert_leaf_grads_keyed_by_stable_ids_and_routing_weights():
    from config import SmaulBrainConfig
    from model import SmaulBrainModel

    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=16)
    m = SmaulBrainModel(cfg)
    ids = torch.randint(0, 256, (2, 16))
    out = m(ids, ids, step=0)
    out["loss"].backward()

    grads = m.take_expert_grads()
    assert len(grads) > 0
    for eid, weight_grads in grads.items():
        # Keyed by stable IDs (expert_<NNNNN>), not arbitrary router indices
        assert eid in m.pool.experts
        assert eid.startswith("expert_")
        # Keyed by weight names with non-None gradients
        for wname, g in weight_grads.items():
            assert wname in ("w_gate", "w_up", "w_down")
            assert g is not None
            assert torch.isfinite(g).all()
            # Gradient must be non-zero since expert was routed with non-zero weight
            assert g.abs().sum().item() > 0.0
    m.pager.close()


def test_unconsumed_leaves_never_silently_discarded():
    from config import SmaulBrainConfig
    from model import SmaulBrainModel

    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=16)
    m = SmaulBrainModel(cfg)
    ids = torch.randint(0, 256, (2, 16))

    # First forward allocates active expert leaves
    m(ids, ids, step=0)
    assert len(m._leaves) > 0

    # Second forward without consuming or clearing must raise RuntimeError
    with pytest.raises(RuntimeError, match="unconsumed expert leaves"):
        m(ids, ids, step=1)

    # Calling zero_grad or clear_expert_grads safely clears the guard
    m.zero_grad(set_to_none=True)
    assert len(m._leaves) == 0

    # Now forward succeeds
    out = m(ids, ids, step=1)
    assert "loss" in out
    m.pager.close()


def test_leaf_cleanup_exception_safe():
    from config import SmaulBrainConfig
    from model import SmaulBrainModel

    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=16)
    m = SmaulBrainModel(cfg)
    ids = torch.randint(0, 256, (2, 16))

    # Simulate an exception mid-forward
    def boom(*args, **kwargs):
        raise RuntimeError("simulated depth explosion")

    orig_depth_loop = m._depth_loop
    m._depth_loop = boom
    with pytest.raises(RuntimeError, match="simulated depth explosion"):
        m(ids, ids, step=0)

    # Cleanup must be exception-safe: leaves cleared, no dangling state
    assert len(m._leaves) == 0

    # Restore and run properly
    m._depth_loop = orig_depth_loop
    out = m(ids, ids, step=0)
    assert "loss" in out
    m.take_expert_grads()
    m.pager.close()


def test_step_expert_avoids_optim_state_for_no_grad_leaves():
    from experts import make_expert
    from smaulopt import SmaulOpt, SmaulOptHParams
    from train import _GradOnly

    torch.manual_seed(0)
    rec = make_expert("expert_00000", 32, 64)
    # Clear eager optimizer state to test sparse/lazy allocation
    rec.optim_state = {}

    opt = SmaulOpt(SmaulOptHParams(lr=0.01))
    g = torch.randn(64, 32)
    # Only w_gate has gradients; w_up has no grad, w_down omitted
    compute = {"w_gate": _GradOnly(g), "w_up": _GradOnly(None)}

    opt.step_expert(rec, compute, 1.0, lr=0.01)

    # Only w_gate must have optimizer state; w_up and w_down must NOT
    assert "w_gate" in rec.optim_state
    assert "w_up" not in rec.optim_state
    assert "w_down" not in rec.optim_state


def test_release_expert_computation_graphs_and_clear_external_grads():
    from config import SmaulBrainConfig
    from model import SmaulBrainModel

    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=16)
    m = SmaulBrainModel(cfg)
    ids = torch.randint(0, 256, (2, 16))
    out = m(ids, ids, step=0)
    out["loss"].backward()

    # Before take_expert_grads, leaves exist and hold gradients
    leaf_tensors = [leaf for copies in m._leaves.values() for c in copies for leaf in c.values()]
    assert any(l.grad is not None for l in leaf_tensors)

    # Calling take_expert_grads clears leaf.grad and releases m._leaves
    grads = m.take_expert_grads()
    assert len(grads) > 0
    assert len(m._leaves) == 0
    for l in leaf_tensors:
        assert l.grad is None  # stale external gradients cleared
    m.pager.close()


def test_step_expert_row_range_preserves_untouched_fp8_bit_for_bit():
    from experts import make_expert
    from smaulopt import SmaulOpt, SmaulOptHParams
    from train import _GradOnly

    torch.manual_seed(0)
    rec = make_expert("expert_00000", 32, 64, fp8_tile=64)
    opt = SmaulOpt(SmaulOptHParams(lr=0.01))

    orig_codes = {n: rec.weights_fp8[n].codes.clone() for n in ("w_gate", "w_up", "w_down")}
    orig_scales = {n: rec.weights_fp8[n].scales.clone() for n in ("w_gate", "w_up", "w_down")}

    # Update only rows [8:16] of w_gate
    g = torch.randn(64, 32)
    compute = {"w_gate": _GradOnly(g)}
    opt.step_expert(rec, compute, 1.0, lr=0.01, row_range=(8, 16))

    # Touched slice changed
    assert not torch.equal(rec.weights_fp8["w_gate"].codes[8:16], orig_codes["w_gate"][8:16])

    # Untouched rows of w_gate are preserved bit-for-bit
    assert torch.equal(rec.weights_fp8["w_gate"].codes[:8], orig_codes["w_gate"][:8])
    assert torch.equal(rec.weights_fp8["w_gate"].codes[16:], orig_codes["w_gate"][16:])
    assert torch.equal(rec.weights_fp8["w_gate"].scales[:8], orig_scales["w_gate"][:8])
    assert torch.equal(rec.weights_fp8["w_gate"].scales[16:], orig_scales["w_gate"][16:])

    # Untouched weights (w_up, w_down) are completely preserved bit-for-bit
    assert torch.equal(rec.weights_fp8["w_up"].codes, orig_codes["w_up"])
    assert torch.equal(rec.weights_fp8["w_up"].scales, orig_scales["w_up"])
    assert torch.equal(rec.weights_fp8["w_down"].codes, orig_codes["w_down"])
    assert torch.equal(rec.weights_fp8["w_down"].scales, orig_scales["w_down"])


def test_grow_topk_build_failure_restores_id_cursor():
    """Growth rollback: phase-1 build failure consumes no IDs."""
    pool, r = _synced(n=3)
    pool.experts["expert_00000"].weights_fp8["w_gate"] = None  # poison build
    next_before = pool._next_id
    order = list(pool.order)
    w_before = r.proj.weight.detach().clone()
    with pytest.raises(Exception):
        grow_topk_clones(pool, r, 16, 32, step=1, seed=0, k=3)
    assert pool.order == order and r.num_experts == 3
    assert pool._next_id == next_before  # no consumed IDs
    assert pool.fresh_id() == "expert_00003"  # cursor reusable
    assert torch.equal(r.proj.weight, w_before)


def test_grow_expert_commit_failure_restores_exact_topology(monkeypatch):
    """Growth rollback: router-append failure undoes pool add + ID cursor."""
    pool, r = _synced(n=3)
    next_before = pool._next_id
    order = list(pool.order)
    w_before = r.proj.weight.detach().clone()
    b_before = r.proj.bias.detach().clone()

    def boom(init=None):
        raise RuntimeError("router on fire")
    monkeypatch.setattr(r, "add_expert_row", boom)
    with pytest.raises(RuntimeError):
        grow_expert(pool, r, 16, 32, step=1, seed=0)
    assert pool.order == order and len(pool) == 3 and r.num_experts == 3
    assert pool._next_id == next_before  # ID restored
    assert torch.equal(r.proj.weight, w_before)
    assert torch.equal(r.proj.bias, b_before)


def test_grow_topk_commit_failure_rolls_back_prefix_exactly(monkeypatch):
    """Growth rollback: mid-commit failure removes prefix, restores cursor."""
    pool, r = _synced(n=3)
    next_before = pool._next_id
    order = list(pool.order)
    w_before = r.proj.weight.detach().clone()
    from smaulopt import SmaulOpt, SmaulOptHParams
    opt = SmaulOpt(SmaulOptHParams())
    d = r.proj.weight.shape[1]
    m = torch.stack([torch.full((d,), 100.0 + i) for i in range(3)])
    opt.router_state["w"] = {"m": m.clone(), "step": 5,
                             "v_row": torch.zeros(3, 1),
                             "v_col": torch.zeros(1, d)}
    real_add = r.add_expert_row
    calls = {"n": 0}

    def flaky(init=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("router on fire mid-sweep")
        return real_add(init=init)
    monkeypatch.setattr(r, "add_expert_row", flaky)
    with pytest.raises(RuntimeError):
        grow_topk_clones(pool, r, 16, 32, step=1, seed=0, k=3,
                         optim_state=opt.router_state)
    # Exact prior topology: no prefix left behind, cursor restored.
    assert pool.order == order and len(pool) == 3 and r.num_experts == 3
    assert pool._next_id == next_before
    assert torch.equal(r.proj.weight, w_before)
    m_after = opt.router_state["w"]["m"]
    assert m_after.shape == (3, d)  # optim untouched (pad is last)
    for i in range(3):
        assert torch.equal(m_after[i], torch.full((d,), 100.0 + i))


def test_growth_rejects_mismatched_optim_width_without_mutating():
    """Validate-first: stale momentum widths refuse before any growth."""
    pool, r = _synced(n=3)
    from smaulopt import SmaulOpt, SmaulOptHParams
    opt = SmaulOpt(SmaulOptHParams())
    d = r.proj.weight.shape[1]
    opt.router_state["w"] = {"m": torch.zeros(2, d), "step": 1,
                             "v_row": torch.zeros(2, 1),
                             "v_col": torch.zeros(1, d)}
    order = list(pool.order)
    next_before = pool._next_id
    with pytest.raises(ValueError):
        grow_expert(pool, r, 16, 32, step=1, seed=0,
                    optim_state=opt.router_state)
    assert pool.order == order and r.num_experts == 3
    assert pool._next_id == next_before
    with pytest.raises(ValueError):
        grow_topk_clones(pool, r, 16, 32, step=1, seed=0, k=2,
                         optim_state=opt.router_state)
    assert pool.order == order and r.num_experts == 3
    assert pool._next_id == next_before


def test_grow_pad_failure_rolls_back_topology(monkeypatch):
    """Growth rollback: momentum-pad failure undoes pool + router + IDs."""
    import growth as growth_mod
    pool, r = _synced(n=3)
    from smaulopt import SmaulOpt, SmaulOptHParams
    opt = SmaulOpt(SmaulOptHParams())
    d = r.proj.weight.shape[1]
    m = torch.stack([torch.full((d,), 100.0 + i) for i in range(3)])
    opt.router_state["w"] = {"m": m.clone(), "step": 5,
                             "v_row": torch.zeros(3, 1),
                             "v_col": torch.zeros(1, d)}
    m_before = opt.router_state["w"]["m"].clone()
    next_before = pool._next_id
    order = list(pool.order)
    w_before = r.proj.weight.detach().clone()

    def boom(state, n_new):
        raise RuntimeError("pad on fire")
    monkeypatch.setattr(growth_mod, "_pad_state_rows", boom)
    with pytest.raises(RuntimeError):
        grow_expert(pool, r, 16, 32, step=1, seed=0,
                    optim_state=opt.router_state)
    assert pool.order == order and r.num_experts == 3
    assert pool._next_id == next_before
    assert torch.equal(r.proj.weight, w_before)
    assert torch.equal(opt.router_state["w"]["m"], m_before)


def test_clones_differ_from_parent_by_one_to_ten_percent():
    pool, r = _synced(n=4)
    names = ("w_gate", "w_up", "w_down")
    before = {eid: {n: rec.weights_fp8[n].codes.clone() for n in names}
              for eid, rec in pool.experts.items()}
    ids = grow_topk_clones(pool, r, 16, 32, step=1, seed=0, k=4)
    assert len(ids) == 4
    seen = set()
    for eid in ids:
        rec = pool.experts[eid]
        assert rec.source == "clone-mutated"  # every clone mutated, none exact
        par = rec.parents[0]
        rels = []
        for n in names:
            a = dequantize_fp8_blockwise(pool.experts[par].weights_fp8[n]).float()
            b = dequantize_fp8_blockwise(rec.weights_fp8[n]).float()
            rels.append(float((b - a).abs().mean()
                              / a.abs().mean().clamp_min(1e-12)))
        rel = sum(rels) / len(rels)
        assert 0.005 <= rel <= 0.15, (eid, rel)  # ~1-10% + FP8 epsilon
        key = tuple(rec.weights_fp8[n].codes.flatten().tolist() for n in names)
        assert key not in seen  # distinct clones
        seen.add(key)
    for eid, rec in pool.experts.items():
        if eid not in ids:  # parents never touched
            for n in names:
                assert torch.equal(rec.weights_fp8[n].codes, before[eid][n])

