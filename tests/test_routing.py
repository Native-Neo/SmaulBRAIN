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
