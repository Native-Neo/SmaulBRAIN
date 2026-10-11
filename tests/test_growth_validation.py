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
