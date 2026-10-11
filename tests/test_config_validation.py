"""Regression tests for configuration dimensions rejected before model construction."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SmaulBrainConfig


@pytest.mark.parametrize("kwargs", [
    {"d_model": 0},
    {"d_model": -32},
    {"n_heads": 0},
    {"n_heads": -4},
])
def test_config_rejects_non_positive_model_dimensions(kwargs):
    with pytest.raises(AssertionError, match="d_model|n_heads"):
        SmaulBrainConfig(**kwargs)


def test_config_still_rejects_width_not_divisible_by_heads():
    with pytest.raises(AssertionError, match="split over heads"):
        SmaulBrainConfig(d_model=30, n_heads=4)

@pytest.mark.parametrize("bad", [2.5, "2", True, None])
def test_from_dict_rejects_non_integer_active_experts(bad):
    payload = SmaulBrainConfig().to_dict()
    payload["top_k"] = 2
    payload["active_experts"] = bad
    with pytest.raises(ValueError, match="active_experts"):
        SmaulBrainConfig.from_dict(payload)


def test_from_dict_rejects_active_experts_that_only_match_after_truncation():
    payload = SmaulBrainConfig().to_dict()
    payload["top_k"] = 2
    payload["active_experts"] = 2.5
    with pytest.raises(ValueError, match="active_experts"):
        SmaulBrainConfig.from_dict(payload)
