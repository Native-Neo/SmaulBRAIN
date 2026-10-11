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
