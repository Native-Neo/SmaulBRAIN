"""Regression tests for validate-first expert pruning."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts import ExpertPool, make_expert
from pruning import prune_experts
from routing import SparseRouter


def _pool_and_router():
    pool = ExpertPool()
    for _ in range(2):
        pool.add(make_expert(pool.fresh_id(), d_model=8, expert_hidden=16))
    router = SparseRouter(d_model=8, num_experts=2, top_k=1)
    return pool, router


def test_prune_rejects_duplicate_victims_without_partial_mutation():
    pool, router = _pool_and_router()
    before = list(pool.order)
    duplicate = before[0]
    with pytest.raises(ValueError, match="duplicate victims"):
        prune_experts(pool, router, [duplicate, duplicate])
    assert pool.order == before
    assert len(pool) == 2
    assert router.num_experts == 2
    assert router.proj.weight.shape[0] == 2
