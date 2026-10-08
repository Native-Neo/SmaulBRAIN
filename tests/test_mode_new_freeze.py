"""Mode-`new` freeze: old skills stay frozen while new experts train."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import train_step

NAMES = ("w_gate", "w_up", "w_down")
CUTOFF = 5


def _build(seed=0, mult=0.005, bias=0.0, num_experts=4, top_k=4, cutoff=CUTOFF):
    torch.manual_seed(seed)
    cfg = SmaulBrainConfig(
        d_model=32, n_heads=4, num_experts=num_experts, top_k=top_k,
        expert_hidden=32, max_depth=2, context_length=16,
        expert_lr=3e-2, router_lr_mult_new=mult, new_routing_bias=bias,
    )
    m = SmaulBrainModel(cfg)
    for i, eid in enumerate(m.pool.order):
        m.pool.experts[eid].birth_step = 0 if i < num_experts // 2 else cutoff
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    return m, opt, cfg


def _batch(cfg, seed=123):
    torch.manual_seed(seed)
    b = torch.randint(0, 256, (2, cfg.context_length + 1))
    return b[:, :cfg.context_length], b[:, 1:]


def _old_new_ids(m, cutoff=CUTOFF):
    old = [eid for eid in m.pool.order if m.pool.experts[eid].birth_step < cutoff]
    new = [eid for eid in m.pool.order if m.pool.experts[eid].birth_step >= cutoff]
    return old, new


def test_old_experts_fp8_frozen_new_move():
    m, opt, cfg = _build()
    try:
        old_ids, new_ids = _old_new_ids(m)
        before = {
            eid: {n: (rec.weights_fp8[n].codes.clone(), rec.weights_fp8[n].scales.clone())
                  for n in NAMES}
            for eid, rec in m.pool.experts.items()
        }
        x, y = _batch(cfg)
        for s in range(3):
            train_step(m, opt, cfg, x, y, step=s, mode="new", new_since_step=CUTOFF)
        for eid in old_ids:
            rec = m.pool.experts[eid]
            for n in NAMES:
                assert torch.equal(rec.weights_fp8[n].codes, before[eid][n][0]), (eid, n)
                assert torch.equal(rec.weights_fp8[n].scales, before[eid][n][1]), (eid, n)
        for eid in new_ids:
            rec = m.pool.experts[eid]
            moved = any(
                not torch.equal(rec.weights_fp8[n].codes, before[eid][n][0])
                for n in NAMES
            )
            assert moved, f"{eid} did not move"
    finally:
        m.pager.close()


def test_trunk_bit_identical():
    m, opt, cfg = _build()
    try:
        before = [(n, p.detach().clone()) for n, p in m._trunk_params()]
        x, y = _batch(cfg)
        for s in range(3):
            train_step(m, opt, cfg, x, y, step=s, mode="new", new_since_step=CUTOFF)
        after = m._trunk_params()
        assert len(after) == len(before)
        for (n0, p0), (n1, p1) in zip(before, after):
            assert n0 == n1
            assert torch.equal(p0, p1), n0
        # Liveness: one `entire` step must move the trunk (same objects,
        # proving the bit-identical comparison above resolves real movement).
        live_before = [(n, p.detach().clone()) for n, p in m._trunk_params()]
        train_step(m, opt, cfg, x, y, step=3, mode="entire")
        live_after = m._trunk_params()
        assert any(not torch.equal(p0, p1)
                   for (_, p0), (_, p1) in zip(live_before, live_after))
    finally:
        m.pager.close()


def test_old_optim_state_untouched_new_stepped():
    m, opt, cfg = _build()
    try:
        old_ids, new_ids = _old_new_ids(m)
        snap = {}
        for eid, rec in m.pool.experts.items():
            d = {}
            for wn, st in rec.optim_state.items():
                e = {"m": st["m"].clone(), "step": int(st["step"])}
                if "v_row" in st:
                    e["v_row"] = st["v_row"].clone()
                if "v_col" in st:
                    e["v_col"] = st["v_col"].clone()
                if "v" in st:
                    e["v"] = st["v"].clone()
                d[wn] = e
            snap[eid] = (d, int(rec.version), float(rec.grad_activity))
        x, y = _batch(cfg)
        for s in range(3):
            train_step(m, opt, cfg, x, y, step=s, mode="new", new_since_step=CUTOFF)
        for eid in old_ids:
            rec = m.pool.experts[eid]
            d0, v0, g0 = snap[eid]
            for wn, st in rec.optim_state.items():
                assert torch.equal(st["m"], d0[wn]["m"]), (eid, wn)
                if "v_row" in st:
                    assert torch.equal(st["v_row"], d0[wn]["v_row"]), (eid, wn)
                    assert torch.equal(st["v_col"], d0[wn]["v_col"]), (eid, wn)
                if "v" in st:
                    assert torch.equal(st["v"], d0[wn]["v"]), (eid, wn)
                assert int(st["step"]) == int(d0[wn]["step"]), (eid, wn)
            assert int(rec.version) == v0, eid
            assert float(rec.grad_activity) == g0, eid
        for eid in new_ids:
            rec = m.pool.experts[eid]
            d0, v0, g0 = snap[eid]
            assert int(rec.version) > v0, eid
            stepped = any(int(rec.optim_state[wn]["step"]) > int(d0[wn]["step"])
                          for wn in rec.optim_state)
            assert stepped, eid
    finally:
        m.pager.close()


def _router_row_means(m, before_w, after_w, old_idx, new_idx):
    with torch.no_grad():
        delta = after_w.float() - before_w.float()
        old_mean = float(delta[old_idx].abs().mean().item()) if old_idx else 0.0
        new_mean = float(delta[new_idx].abs().mean().item()) if new_idx else 0.0
    return old_mean, new_mean


def test_router_rows_bounded_and_stepped_only_new():
    m, opt, cfg = _build(mult=0.005)
    try:
        old_ids, new_ids = _old_new_ids(m)
        order = list(m.pool.order)
        old_idx = [order.index(e) for e in old_ids]
        new_idx = [order.index(e) for e in new_ids]
        w0 = m.router.proj.weight.detach().clone()
        b0 = m.router.proj.bias.detach().clone()
        x, y = _batch(cfg)
        stepped_all = []
        for s in range(3):
            st = train_step(m, opt, cfg, x, y, step=s, mode="new", new_since_step=CUTOFF)
            stepped_all.extend(st["stepped_experts"])
            assert set(st["stepped_experts"]) <= set(new_ids), st["stepped_experts"]
        assert len(stepped_all) > 0
        assert set(stepped_all) <= set(new_ids)
        w1 = m.router.proj.weight.detach().clone()
        b1 = m.router.proj.bias.detach().clone()
        old_mean, new_mean = _router_row_means(m, w0, w1, old_idx, new_idx)
        assert new_mean > 0.0, (old_mean, new_mean)
        assert old_mean > 0.0, (old_mean, new_mean)
        assert old_mean <= 0.05 * new_mean, (old_mean, new_mean)
        # Same two-sided bound for bias rows: catches a weight-only rescale
        # that leaves old bias at full LR.
        old_bmean, new_bmean = _router_row_means(m, b0, b1, old_idx, new_idx)
        assert new_bmean > 0.0, (old_bmean, new_bmean)
        assert old_bmean > 0.0, (old_bmean, new_bmean)
        assert old_bmean <= 0.05 * new_bmean, (old_bmean, new_bmean)
        # Control: mult=1.0 must resolve order-unity movement (sensitivity).
        m2, opt2, cfg2 = _build(seed=0, mult=1.0)
        try:
            order2 = list(m2.pool.order)
            old_idx2 = [order2.index(e) for e in old_ids]
            new_idx2 = [order2.index(e) for e in new_ids]
            w0b = m2.router.proj.weight.detach().clone()
            b0b = m2.router.proj.bias.detach().clone()
            for s in range(3):
                train_step(m2, opt2, cfg2, x, y, step=s, mode="new", new_since_step=CUTOFF)
            w1b = m2.router.proj.weight.detach().clone()
            b1b = m2.router.proj.bias.detach().clone()
            o2, n2 = _router_row_means(m2, w0b, w1b, old_idx2, new_idx2)
            assert n2 > 0.0 and o2 > 0.0, (o2, n2)
            ratio = o2 / max(n2, 1e-12)
            assert 0.2 < ratio < 5.0, ratio
            # Bias control: mult=1.0 must move both old and new bias rows.
            o2b, n2b = _router_row_means(m2, b0b, b1b, old_idx2, new_idx2)
            assert n2b > 0.0 and o2b > 0.0, (o2b, n2b)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_cutoff_boundary():
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(
        d_model=32, n_heads=4, num_experts=2, top_k=2,
        expert_hidden=32, max_depth=1, context_length=16, expert_lr=3e-2,
    )
    m = SmaulBrainModel(cfg)
    try:
        cutoff = 7
        e_old, e_new = m.pool.order[0], m.pool.order[1]
        m.pool.experts[e_old].birth_step = cutoff - 1
        m.pool.experts[e_new].birth_step = cutoff
        opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
        before = {eid: {n: (m.pool.experts[eid].weights_fp8[n].codes.clone(),
                             m.pool.experts[eid].weights_fp8[n].scales.clone())
                            for n in NAMES}
                  for eid in m.pool.order}
        torch.manual_seed(123)
        b = torch.randint(0, 256, (2, cfg.context_length + 1))
        st = train_step(m, opt, cfg, b[:, :cfg.context_length], b[:, 1:],
                        step=0, mode="new", new_since_step=cutoff)
        assert e_new in st["stepped_experts"], st["stepped_experts"]
        assert e_old not in st["stepped_experts"], st["stepped_experts"]
        for n in NAMES:
            assert torch.equal(m.pool.experts[e_old].weights_fp8[n].codes, before[e_old][n][0])
            assert torch.equal(m.pool.experts[e_old].weights_fp8[n].scales, before[e_old][n][1])
        assert any(not torch.equal(m.pool.experts[e_new].weights_fp8[n].codes, before[e_new][n][0])
                   for n in NAMES)
        assert any(not torch.equal(m.pool.experts[e_new].weights_fp8[n].scales, before[e_new][n][1])
                   for n in NAMES)
    finally:
        m.pager.close()


def test_steering_attrs_cleared():
    m, opt, cfg = _build(seed=0, bias=2.0)
    try:
        x, y = _batch(cfg)
        train_step(m, opt, cfg, x, y, step=0, mode="new", new_since_step=CUTOFF)
        assert float(m.router.new_routing_bias) == 0.0
        assert tuple(m.router.new_expert_idx) == ()
    finally:
        m.pager.close()


def test_shuffled_order_selects_by_birth_not_position():
    """Birth (not pool position) selects who steps in mode-`new`.

    Decorrelates order from age: order[0]=new, order[1]=old,
    order[2]=new, order[3]=old. A positional implementation (first
    half vs second half) would step the wrong set; only the
    birth-selected set {eid: birth_step >= cutoff} may step.
    """
    m, opt, cfg = _build(seed=0, num_experts=4, top_k=4, cutoff=CUTOFF)
    try:
        order = list(m.pool.order)
        assert len(order) == 4
        # Shuffle ages across positions: even slots new, odd slots old.
        m.pool.experts[order[0]].birth_step = CUTOFF
        m.pool.experts[order[1]].birth_step = 0
        m.pool.experts[order[2]].birth_step = CUTOFF
        m.pool.experts[order[3]].birth_step = 0
        expected_new = {order[0], order[2]}
        expected_old = {order[1], order[3]}
        birth_selected = {eid for eid in m.pool.order
                          if m.pool.experts[eid].birth_step >= CUTOFF}
        assert birth_selected == expected_new
        # Provably distinguishes: birth set differs from either positional half.
        first_half = set(order[:2])
        second_half = set(order[2:])
        assert birth_selected != first_half, (birth_selected, first_half)
        assert birth_selected != second_half, (birth_selected, second_half)
        before = {eid: {n: (m.pool.experts[eid].weights_fp8[n].codes.clone(),
                             m.pool.experts[eid].weights_fp8[n].scales.clone())
                            for n in NAMES}
                  for eid in order}
        x, y = _batch(cfg)
        st = train_step(m, opt, cfg, x, y, step=0, mode="new",
                        new_since_step=CUTOFF)
        # Exact set equality: birth-selected stepped, nothing else.
        assert set(st["stepped_experts"]) == birth_selected == expected_new
        assert not (set(st["stepped_experts"]) & expected_old)
        # Kill the alias in both directions: new-in-old-position moves,
        # old-in-new-position is bit-identical.
        for eid in expected_old:
            rec = m.pool.experts[eid]
            for n in NAMES:
                assert torch.equal(rec.weights_fp8[n].codes, before[eid][n][0]), (eid, n)
                assert torch.equal(rec.weights_fp8[n].scales, before[eid][n][1]), (eid, n)
        for eid in expected_new:
            rec = m.pool.experts[eid]
            moved = any(
                not torch.equal(rec.weights_fp8[n].codes, before[eid][n][0])
                for n in NAMES
            )
            assert moved, f"{eid} (new in old-position slot) did not move"
    finally:
        m.pager.close()
