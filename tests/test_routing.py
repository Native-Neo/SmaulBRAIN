"""Sparse routing: top-k, renormalization, capacity, balance, stats."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
