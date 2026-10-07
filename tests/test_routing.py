"""Sparse routing: top-k, renormalization, capacity, balance, stats."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from routing import SparseRouter


def test_topk_rows_renormalized():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=6, top_k=2)
    plan = r.route(torch.randn(24, 16))
    assert plan.top_ids.shape == (24, 2)
    assert torch.allclose(plan.top_weights.sum(-1),
                          torch.ones(24), atol=1e-5)
    assert (plan.top_ids < 6).all()


def test_only_active_experts_dispatched():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=8, top_k=2)
    plan = r.route(torch.randn(32, 16))
    used = set(plan.top_ids.reshape(-1).tolist())
    assert len(used) <= 8 and all(0 <= u < 8 for u in used)


def test_capacity_drops_overflow_and_counts_them():
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=2, top_k=2, capacity_factor=0.05)
    plan = r.route(torch.randn(64, 8))
    assert plan.dropped.sum().item() > 0  # tiny capacity must drop


def test_balance_loss_bounded_and_finite():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=4, top_k=2)
    plan = r.route(torch.randn(40, 16))
    b = r.balance_loss(plan.probs)
    assert torch.isfinite(b) and 0 <= b.item() <= 4.0


def test_usage_stats_track_routing():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=4, top_k=1)
    before = r.usage_counts.sum().item()
    p1 = r.route(torch.randn(20, 16))
    p2 = r.route(torch.randn(20, 16))
    kept = int((~p1.dropped).sum()) + int((~p2.dropped).sum())
    assert r.usage_counts.sum().item() == before + kept
    share = r.usage_share()
    assert abs(share.sum().item() - 1.0) < 1e-6


def test_router_grows_and_shrinks_with_pool():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=3, top_k=1)
    i = r.add_expert_row()
    assert i == 3 and r.num_experts == 4
    plan = r.route(torch.randn(10, 16))
    assert plan.top_ids.max().item() < 4
    r.remove_expert_row(0)
    assert r.num_experts == 3
    plan = r.route(torch.randn(10, 16))
    assert plan.top_ids.max().item() < 3


def test_empty_batch_returns_empty_plan():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=4, top_k=2)
    plan = r.route(torch.zeros(0, 16))
    assert plan.top_ids.shape == (0, 2)
    assert plan.top_weights.shape == (0, 2)
    assert plan.dropped.shape == (0,)
    assert plan.probs.shape == (0, 4)
    assert float(r.balance_loss(plan.probs)) == 0.0


def test_nonfinite_input_sanitized_and_counted():
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=3, top_k=2)
    x = torch.randn(6, 8)
    x[0, :] = float("nan")
    x[1, 0] = float("inf")
    plan = r.route(x)
    assert torch.isfinite(plan.top_weights).all()
    assert ((plan.top_ids >= 0) & (plan.top_ids < 3)).all()
    assert torch.isfinite(plan.probs).all()
    assert torch.isfinite(r.balance_loss(plan.probs))
    assert float(r.sanitized_counts) == 2.0
    r.reset_stats()
    assert float(r.sanitized_counts) == 0.0


def test_routing_deterministic_for_same_input():
    torch.manual_seed(0)
    r = SparseRouter(16, num_experts=5, top_k=2)
    x = torch.randn(24, 16)
    a = r.route(x)
    b = r.route(x.clone())
    assert torch.equal(a.top_ids, b.top_ids)
    assert torch.equal(a.dropped, b.dropped)
    assert torch.allclose(a.top_weights, b.top_weights)


def test_all_dropped_keeps_balance_finite():
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=2, top_k=2, capacity_factor=0.05)
    plan = r.route(torch.randn(64, 8))
    assert plan.dropped.sum().item() > 0
    assert torch.isfinite(r.balance_loss(plan.probs))


def test_per_slot_admission_respects_capacity():
    torch.manual_seed(0)
    n, experts, k = 16, 4, 2
    r = SparseRouter(8, num_experts=experts, top_k=k, capacity_factor=0.5)
    cap = max(1, int(0.5 * n * k / experts))
    plan = r.route(torch.randn(n, 8))
    live = ~plan.dropped
    admitted_w = plan.top_weights[live]
    assert torch.allclose(admitted_w.sum(-1), torch.ones(live.sum()),
                          atol=1e-5)  # live rows renormalized over admitted subset
    if plan.dropped.any():
        assert (plan.top_weights[plan.dropped].sum(-1) == 0).all()
    per_expert = torch.bincount(plan.top_ids[plan.top_weights > 0],
                                minlength=experts)
    assert int(per_expert.max().item()) <= cap  # real slot load within cap
    assert int(r.usage_counts.sum().item()) == int(per_expert.sum().item())


def test_inference_routing_ignores_batch_size():
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=4, top_k=2, capacity_factor=0.05)
    x = torch.randn(8, 8)
    whole = r.route(x, enforce_capacity=False)
    assert not whole.dropped.any()
    parts = [r.route(x[i : i + 1], enforce_capacity=False) for i in range(8)]
    split_ids = torch.cat([p.top_ids for p in parts], dim=0)
    split_w = torch.cat([p.top_weights for p in parts], dim=0)
    assert torch.equal(whole.top_ids, split_ids)
    assert torch.allclose(whole.top_weights, split_w)


def test_identical_tokens_admit_in_input_order():
    # All rows tie on confidence: stable priority must admit earlier tokens
    # first, identically on every run (deterministic tie-breaking).
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=4, top_k=2, capacity_factor=0.5)
    row = torch.randn(1, 8)
    x = row.expand(8, 8).contiguous()
    first = r.route(x)
    assert first.dropped.tolist() == [False, False] + [True] * 6
    for _ in range(3):
        again = r.route(x)
        assert torch.equal(again.top_ids, first.top_ids)
        assert torch.equal(again.dropped, first.dropped)
        assert torch.allclose(again.top_weights, first.top_weights)


def test_topk_ids_unique_per_row_across_seeds():
    for seed in range(5):
        torch.manual_seed(seed)
        r = SparseRouter(16, num_experts=6, top_k=3)
        plan = r.route(torch.randn(32, 16))
        for row in plan.top_ids.tolist():
            assert len(set(row)) == 3  # topk never repeats an expert


def test_plan_validator_rejects_bad_plans():
    from routing import RoutePlan
    good_ids = torch.tensor([[0, 1], [2, 3]])
    good_w = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    RoutePlan(top_ids=good_ids, top_weights=good_w,
              dropped=torch.tensor([False, False]),
              probs=torch.zeros(2, 4)).validate(4, 2)
    dup = RoutePlan(top_ids=torch.tensor([[1, 1], [2, 3]]),
                    top_weights=good_w,
                    dropped=torch.tensor([False, False]),
                    probs=torch.zeros(2, 4))
    with pytest.raises(AssertionError):
        dup.validate(4, 2)
    oob = RoutePlan(top_ids=torch.tensor([[0, 9], [2, 3]]),
                    top_weights=good_w,
                    dropped=torch.tensor([False, False]),
                    probs=torch.zeros(2, 4))
    with pytest.raises(AssertionError):
        oob.validate(4, 2)


def test_plan_tensors_share_one_device():
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=4, top_k=2)
    plan = r.route(torch.randn(10, 8))
    devs = {plan.top_ids.device, plan.top_weights.device,
            plan.dropped.device, plan.probs.device}
    assert len(devs) == 1
    empty = r.route(torch.zeros(0, 8))
    assert empty.top_ids.device == empty.dropped.device == empty.probs.device


def test_audit027_empty_batch_leaves_stats_finite():
    # Empty batches: well-formed plan, stats untouched, zero loss, finite share.
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=4, top_k=2)
    r.route(torch.randn(10, 8))
    u_before = r.usage_counts.clone()
    a_before = r.admit_counts.clone()
    plan = r.route(torch.zeros(0, 8))
    assert plan.top_ids.shape == (0, 2)
    assert plan.top_weights.shape == (0, 2)
    assert plan.dropped.shape == (0,)
    assert plan.probs.shape == (0, 4)
    assert torch.equal(r.usage_counts, u_before)
    assert torch.equal(r.admit_counts, a_before)
    b = r.balance_loss(plan.probs)
    assert float(b) == 0.0 and bool(torch.isfinite(b))
    assert b.device == plan.probs.device
    share = r.usage_share()
    assert bool(torch.isfinite(share).all())
    # Fresh router with no traffic: share is all zeros but finite.
    fresh = SparseRouter(8, num_experts=4, top_k=2)
    s0 = fresh.usage_share()
    assert bool(torch.isfinite(s0).all())
    assert float(s0.sum()) == 0.0


def test_audit027_all_dropped_counts_admitted_and_stays_finite():
    # All-dropped assignments: stats count admitted slots only (< attempted
    # N*K), dropped rows sum to 0, balance/share stay finite.
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=2, top_k=2, capacity_factor=0.0)
    n = 16
    plan = r.route(torch.randn(n, 8))
    assert plan.dropped.sum().item() > 0
    # Admitted slots via positive weight (non-admitted slots read 0).
    admitted = plan.top_ids[plan.top_weights > 0]
    expected = torch.bincount(admitted, minlength=2).to(torch.float64) if admitted.numel() else torch.zeros(2, dtype=torch.float64)
    assert torch.equal(r.usage_counts, expected)
    assert torch.equal(r.admit_counts, expected)
    # Intentional duplicate: both counters track admitted traffic (see below).
    assert torch.equal(r.usage_counts, r.admit_counts)
    assert int(r.usage_counts.sum().item()) <= n * 2
    if plan.dropped.sum().item() > 0:
        assert int(r.usage_counts.sum().item()) < n * 2
        assert (plan.top_weights[plan.dropped].sum(-1) == 0).all()
    assert bool(torch.isfinite(r.balance_loss(plan.probs)))
    assert bool(torch.isfinite(r.usage_share()).all())


def test_audit027_inactive_expert_share_zero_finite():
    # Inactive experts: never-selected expert keeps 0 count/share, finite loss.
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=4, top_k=1)
    with torch.no_grad():
        r.proj.weight[3].zero_()
        r.proj.bias[3] = -1e9
    plan = r.route(torch.randn(32, 8))
    assert 3 not in set(plan.top_ids.reshape(-1).tolist())
    assert float(r.usage_counts[3]) == 0.0
    share = r.usage_share()
    assert float(share[3]) == 0.0
    assert bool(torch.isfinite(share).all())
    assert bool(torch.isfinite(r.balance_loss(plan.probs)))


def test_audit027_zero_denominators_never_nan():
    # Zero denominators: live rows renormalize to 1, dropped rows to 0,
    # share clamps at zero total; nothing becomes NaN/Inf.
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=4, top_k=2, capacity_factor=0.5)
    plan = r.route(torch.randn(16, 8))
    live = ~plan.dropped
    if int(live.sum()) > 0:
        assert torch.allclose(plan.top_weights[live].sum(-1),
                              torch.ones(int(live.sum())), atol=1e-5)
    if int(plan.dropped.sum()) > 0:
        assert (plan.top_weights[plan.dropped].sum(-1) == 0).all()
    assert bool(torch.isfinite(plan.top_weights).all())
    assert bool(torch.isfinite(plan.probs).all())
    fresh = SparseRouter(8, num_experts=4, top_k=2)
    assert bool(torch.isfinite(fresh.usage_share()).all())
    # keep=None vs keep=all-excluded both yield finite (0 for empty selection).
    b_all_out = r.balance_loss(plan.probs, keep=torch.zeros(plan.probs.shape[0], dtype=torch.bool))
    assert float(b_all_out) == 0.0 and bool(torch.isfinite(b_all_out))
    assert b_all_out.device == plan.probs.device


def test_audit027_nonfinite_triple_sanitized_and_counted():
    # NaN/Inf router inputs: NaN, +Inf, -Inf rows sanitized, counted, finite.
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=3, top_k=2)
    x = torch.randn(6, 8)
    x[0, :] = float("nan")
    x[1, 0] = float("inf")
    x[2, 0] = float("-inf")
    plan = r.route(x)
    assert float(r.sanitized_counts) == 3.0
    assert ((plan.top_ids >= 0) & (plan.top_ids < 3)).all()
    assert bool(torch.isfinite(plan.top_weights).all())
    assert bool(torch.isfinite(plan.probs).all())
    assert bool(torch.isfinite(r.balance_loss(plan.probs)))
    for row in plan.top_ids.tolist():
        assert len(set(row)) == len(row)


def test_audit027_variable_batch_counts_stable():
    # Variable batch/token counts: N=1..33 route without error, stats track
    # admitted slots, single-token batches never drop (cap >= 1).
    torch.manual_seed(0)
    for n in (1, 2, 7, 33):
        r = SparseRouter(8, num_experts=4, top_k=2)
        plan = r.route(torch.randn(n, 8))
        assert plan.top_ids.shape == (n, 2)
        assert plan.top_weights.shape == (n, 2)
        assert plan.dropped.shape == (n,)
        assert plan.probs.shape == (n, 4)
        assert bool(torch.isfinite(plan.top_weights).all())
        admitted = plan.top_ids[plan.top_weights > 0]
        assert int(r.usage_counts.sum().item()) == int(admitted.numel())
    torch.manual_seed(0)
    r1 = SparseRouter(8, num_experts=4, top_k=2)
    p1 = r1.route(torch.randn(1, 8))
    assert not p1.dropped.any()


def test_audit027_stats_represent_admitted_not_attempted():
    # Statistics must represent admitted traffic, not attempted top-k touches.
    # With forced dropping, admitted slots < N*K; both counters equal the
    # admitted bincount (kept identical for checkpoint compat).
    torch.manual_seed(0)
    r = SparseRouter(8, num_experts=2, top_k=2, capacity_factor=0.05)
    n = 32
    plan = r.route(torch.randn(n, 8))
    assert plan.dropped.sum().item() > 0
    attempted_total = n * 2
    admitted = plan.top_ids[plan.top_weights > 0]
    assert int(admitted.numel()) < attempted_total
    assert int(r.usage_counts.sum().item()) == int(admitted.numel())
    assert int(r.admit_counts.sum().item()) == int(admitted.numel())
    # usage_share is over admitted traffic, hence finite and sums to 1
    # once any slot was admitted.
    share = r.usage_share()
    assert bool(torch.isfinite(share).all())
    assert abs(float(share.sum()) - 1.0) < 1e-6
