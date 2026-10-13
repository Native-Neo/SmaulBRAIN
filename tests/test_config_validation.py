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



def test_config_validation_survives_python_optimized_mode():
    """Invalid model dimensions remain rejected when Python disables assertions."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-O", "-c",
         "from config import SmaulBrainConfig; SmaulBrainConfig(d_model=0)"],
        cwd=root, text=True, capture_output=True, check=False, timeout=30,
    )
    assert result.returncode != 0
    assert "d_model must be >= 1" in result.stderr


@pytest.mark.parametrize("kwargs", [
    {"expert_lr": float("nan")},
    {"weight_decay": -1.0},
    {"ternary_backward": True},
    {"use_byte_conv": 1},
])
def test_config_rejects_invalid_runtime_values(kwargs):
    """Runtime settings reject non-finite values, invalid ranges, and flag conflicts."""
    with pytest.raises(AssertionError):
        SmaulBrainConfig(**kwargs)



def test_config_rejects_pruning_floor_above_maximum_pool():
    """The pruning floor cannot exceed the maximum expert pool size."""
    with pytest.raises(AssertionError, match="min_experts must be <= max_experts"):
        SmaulBrainConfig(num_experts=4, top_k=2, min_experts=65, max_experts=64)



def test_config_rejects_numeric_values_too_large_for_finite_float():
    """Construction and deserialization reject integers that overflow float checks."""
    huge_integer = 10 ** 1000
    with pytest.raises(AssertionError, match="expert_lr must be a finite number"):
        SmaulBrainConfig(expert_lr=huge_integer)

    payload = SmaulBrainConfig().to_dict()
    payload["expert_lr"] = huge_integer
    with pytest.raises(ValueError, match="expert_lr must be a finite number"):
        SmaulBrainConfig.from_dict(payload)


@pytest.mark.parametrize("maximum", [1, 8, 32, 64, 128])
def test_from_dict_caps_omitted_pruning_floor_for_legacy_configs(maximum):
    payload = {"num_experts": 1, "top_k": 1, "max_experts": maximum}

    config = SmaulBrainConfig.from_dict(payload)

    assert config.min_experts == min(SmaulBrainConfig.min_experts, maximum)
    assert "min_experts" not in payload


@pytest.mark.parametrize("minimum", [1, 8, 32])
def test_from_dict_preserves_explicit_pruning_floor(minimum):
    config = SmaulBrainConfig.from_dict(
        {"num_experts": 1, "top_k": 1, "max_experts": 32, "min_experts": minimum}
    )
    assert config.min_experts == minimum


@pytest.mark.parametrize("minimum", [0, 33, None, True])
def test_from_dict_rejects_invalid_explicit_pruning_floor(minimum):
    with pytest.raises(ValueError, match="min_experts"):
        SmaulBrainConfig.from_dict(
            {"num_experts": 1, "top_k": 1, "max_experts": 32, "min_experts": minimum}
        )


@pytest.mark.parametrize("maximum", [None, True, "32", 32.0])
def test_from_dict_keeps_maximum_type_validation_when_floor_is_omitted(maximum):
    with pytest.raises(ValueError, match="max_experts"):
        SmaulBrainConfig.from_dict({"max_experts": maximum})
