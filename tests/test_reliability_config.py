"""M1 regression: config float/bool/finite gaps."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest

from config import SmaulBrainConfig


def _valid_kwargs(**over):
    kw = dict(
        d_model=32, n_heads=4, num_experts=4, top_k=2, expert_hidden=32,
        max_depth=2, min_depth=1, context_length=16, dtype="fp32",
        state_dtype="fp32", threads=1, seed=0, min_experts=2,
    )
    kw.update(over)
    return kw


@pytest.mark.parametrize("field", [
    "halting_threshold", "halt_prior", "ponder_beta", "moe_balance_weight",
    "capacity_factor", "expert_lr", "trunk_lr_mult", "router_lr_mult",
    "router_lr_mult_new", "new_routing_bias", "weight_decay", "grad_clip",
    "beta_m", "beta_v", "epsilon", "rmsnorm_eps",
    "prune_min_usage", "lookahead_weight", "boundary_weight",
])
def test_float_fields_reject_bool(field):
    with pytest.raises((AssertionError, ValueError, TypeError)):
        SmaulBrainConfig(**_valid_kwargs(**{field: True}))


def test_expert_lr_true_refuses():
    with pytest.raises((AssertionError, ValueError)):
        SmaulBrainConfig(**_valid_kwargs(expert_lr=True))


def test_weight_decay_true_refuses():
    with pytest.raises((AssertionError, ValueError)):
        SmaulBrainConfig(**_valid_kwargs(weight_decay=True))


@pytest.mark.parametrize("bad", [1, 0, "yes", "true", 1.0, None])
def test_use_byte_conv_must_be_bool(bad):
    if bad is None:
        # None is rejected via from_dict null gate; direct ctor must also refuse
        with pytest.raises((AssertionError, ValueError, TypeError)):
            SmaulBrainConfig(**_valid_kwargs(use_byte_conv=bad))
    else:
        with pytest.raises((AssertionError, ValueError, TypeError)):
            SmaulBrainConfig(**_valid_kwargs(use_byte_conv=bad))


def test_use_byte_conv_bool_accepted():
    SmaulBrainConfig(**_valid_kwargs(use_byte_conv=True))
    SmaulBrainConfig(**_valid_kwargs(use_byte_conv=False))


@pytest.mark.parametrize("field,value", [
    ("expert_lr", float("inf")),
    ("expert_lr", float("-inf")),
    ("weight_decay", float("inf")),
    ("capacity_factor", float("inf")),
    ("capacity_factor", float("nan")),
    ("moe_balance_weight", float("inf")),
    ("moe_balance_weight", float("nan")),
    ("trunk_lr_mult", float("inf")),
    ("router_lr_mult", float("nan")),
    ("router_lr_mult_new", float("inf")),
    ("new_routing_bias", float("inf")),
    ("new_routing_bias", float("nan")),
    ("grad_clip", float("inf")),
    ("grad_clip", float("nan")),
    ("weight_decay", float("nan")),
    ("epsilon", float("inf")),
    ("rmsnorm_eps", float("inf")),
    ("prune_min_usage", float("inf")),
    ("lookahead_weight", float("nan")),
    ("boundary_weight", float("inf")),
    ("ponder_beta", float("inf")),
    ("ponder_beta", float("nan")),
    ("beta_m", float("nan")),
    ("beta_v", float("inf")),
])
def test_float_fields_reject_nonfinite(field, value):
    with pytest.raises((AssertionError, ValueError, TypeError)):
        SmaulBrainConfig(**_valid_kwargs(**{field: value}))


def test_from_dict_pollution_bool_and_str():
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"expert_lr": True})
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"weight_decay": True})
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"use_byte_conv": 1})
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"use_byte_conv": "yes"})
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"expert_lr": float("inf")})
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"weight_decay": float("nan")})
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"capacity_factor": True})


def test_from_dict_valid_roundtrip():
    c = SmaulBrainConfig(**_valid_kwargs())
    d = c.to_dict()
    c2 = SmaulBrainConfig.from_dict(d)
    assert c2.expert_lr == c.expert_lr
    assert c2.use_byte_conv is True or isinstance(c2.use_byte_conv, bool)
