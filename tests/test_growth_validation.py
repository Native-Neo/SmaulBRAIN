"""Regression tests for growth API argument validation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts import ExpertPool
from growth import grow_topk_clones
from routing import SparseRouter


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5, "2"])
def test_topk_growth_rejects_invalid_clone_count_without_mutation(bad_k):
    pool = ExpertPool()
    router = SparseRouter(d_model=8, num_experts=0, top_k=1)
    before = (list(pool.order), pool._next_id, router.num_experts)
    with pytest.raises(ValueError, match="k must be a positive int"):
        grow_topk_clones(pool, router, d_model=8, expert_hidden=16,
                         step=0, k=bad_k)
    assert (list(pool.order), pool._next_id, router.num_experts) == before


@pytest.mark.parametrize("bad_noise", [-1.0, float("nan"), float("inf"), True, "0.1"])
def test_growth_rejects_invalid_noise_before_mutation(bad_noise):
    from growth import grow_expert, recombine_weights

    pool = ExpertPool()
    router = SparseRouter(d_model=8, num_experts=0, top_k=1)
    with pytest.raises(ValueError, match="noise_std"):
        grow_expert(pool, router, d_model=8, expert_hidden=16,
                    step=0, noise_std=bad_noise)
    assert len(pool) == router.num_experts == 0
    with pytest.raises(ValueError, match="noise_std"):
        recombine_weights(pool, [], noise_std=bad_noise)


@pytest.mark.parametrize("bad_limit", [0, -1, True, 4.5, "5"])
def test_growth_rejects_invalid_capacity_limit_before_mutation(bad_limit):
    from growth import grow_expert

    pool = ExpertPool()
    router = SparseRouter(d_model=8, num_experts=0, top_k=1)
    before = (list(pool.order), pool._next_id, router.num_experts)
    with pytest.raises(ValueError, match="max_experts must be a positive int"):
        grow_expert(pool, router, d_model=8, expert_hidden=16,
                    step=0, max_experts=bad_limit)
    assert (list(pool.order), pool._next_id, router.num_experts) == before

    with pytest.raises(ValueError, match="max_experts must be a positive int"):
        grow_topk_clones(pool, router, d_model=8, expert_hidden=16,
                         step=0, max_experts=bad_limit)
    assert (list(pool.order), pool._next_id, router.num_experts) == before


def test_growth_respects_integer_capacity_limit():
    from growth import grow_expert

    pool = ExpertPool()
    router = SparseRouter(d_model=8, num_experts=0, top_k=1)
    grow_expert(pool, router, d_model=8, expert_hidden=16,
                step=0, max_experts=1)
    assert len(pool) == router.num_experts == 1
    with pytest.raises(ValueError, match="at capacity"):
        grow_expert(pool, router, d_model=8, expert_hidden=16,
                    step=1, max_experts=1)
    assert len(pool) == router.num_experts == 1


@pytest.mark.parametrize("bad_step", [-1, True, 1.5, "2"])
def test_growth_rejects_invalid_step_without_mutation(bad_step):
    from growth import grow_expert

    pool = ExpertPool()
    router = SparseRouter(d_model=8, num_experts=0, top_k=1)
    before = (list(pool.order), pool._next_id, router.num_experts)
    with pytest.raises(ValueError, match="step must be a non-negative int"):
        grow_expert(pool, router, d_model=8, expert_hidden=16, step=bad_step)
    with pytest.raises(ValueError, match="step must be a non-negative int"):
        grow_topk_clones(pool, router, d_model=8, expert_hidden=16, step=bad_step)
    assert (list(pool.order), pool._next_id, router.num_experts) == before


@pytest.mark.parametrize("bad_count", [0, -1, True, 1.5, "2"])
def test_growth_rejects_invalid_parent_count_without_mutation(bad_count):
    from growth import grow_expert

    pool = ExpertPool()
    router = SparseRouter(d_model=8, num_experts=0, top_k=1)
    with pytest.raises(ValueError, match="n_parents must be a positive int"):
        grow_expert(pool, router, d_model=8, expert_hidden=16,
                    step=0, n_parents=bad_count)
    assert len(pool) == router.num_experts == 0
