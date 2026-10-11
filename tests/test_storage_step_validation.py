"""Regression tests for strict checkpoint step validation."""

from types import SimpleNamespace
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import storage


@pytest.mark.parametrize("step", [1.5, True, "2", None])
def test_save_model_rejects_non_integer_step_before_touching_checkpoint(tmp_path, step):
    model = SimpleNamespace(pool=[])
    with pytest.raises(ValueError, match="step"):
        storage.save_model(str(tmp_path / "checkpoint"), model, None, step=step)
    assert not (tmp_path / "checkpoint").exists()


def test_save_model_rejects_negative_step_before_touching_checkpoint(tmp_path):
    model = SimpleNamespace(pool=[])
    with pytest.raises(ValueError, match="negative step"):
        storage.save_model(str(tmp_path / "checkpoint"), model, None, step=-1)
    assert not (tmp_path / "checkpoint").exists()

def test_save_model_rejects_symlinked_checkpoint_root_before_cleanup(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    keep = target / "tmp_ckpt_keep"
    keep.write_text("must survive")
    link = tmp_path / "checkpoint-link"
    link.symlink_to(target, target_is_directory=True)

    model = SimpleNamespace(pool=[])
    with pytest.raises(ValueError, match="checkpoint path escapes|symlink"):
        storage.save_model(str(link), model, None, step=0)
    assert keep.read_text() == "must survive"


@pytest.mark.parametrize("step", [1.5, True, "2", None, -1])
def test_unflatten_optimizer_rejects_malformed_saved_step(step):
    with pytest.raises(ValueError, match="invalid step"):
        storage._unflatten_opt_store({}, {"weight": {"step": step}}, "trunk")


def test_unflatten_optimizer_rejects_non_dict_entry():
    with pytest.raises(ValueError, match="must be a dict"):
        storage._unflatten_opt_store({}, {"weight": 3}, "trunk")


def test_unflatten_optimizer_preserves_valid_integer_step():
    tensors = {
        "trunk.weight.m": torch.zeros(2, 2),
        "trunk.weight.v": torch.zeros(2, 2),
    }
    result = storage._unflatten_opt_store(
        tensors, {"weight": {"step": 2, "kind": "full"}}, "trunk"
    )
    assert result["weight"]["step"] == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("lr", -1.0),
        ("lr", float("inf")),
        ("beta_m", 1.0),
        ("beta_v", float("nan")),
        ("eps", 0.0),
        ("wd", -0.1),
        ("clip", -1.0),
        ("update_clip", 0.0),
        ("lr", True),
    ],
)
def test_checkpoint_optimizer_hparams_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match="bad hparam"):
        storage._validate_opt_hparams({name: value})


def test_checkpoint_optimizer_hparams_accept_valid_values():
    storage._validate_opt_hparams({
        "lr": 2e-4,
        "beta_m": 0.9,
        "beta_v": 0.999,
        "eps": 1e-8,
        "wd": 0.01,
        "clip": 1.0,
        "update_clip": 10.0,
        "state_dtype": "bf16",
        "factor_v": True,
    })


@pytest.mark.parametrize("kind", ["factored-ish", "", None, 1])
def test_unflatten_optimizer_rejects_unknown_state_kind(kind):
    with pytest.raises(ValueError, match="invalid kind"):
        storage._unflatten_opt_store(
            {}, {"weight": {"step": 0, "kind": kind}}, "trunk"
        )


def test_unflatten_optimizer_rejects_non_dict_structure():
    with pytest.raises(ValueError, match="structure must be a dict"):
        storage._unflatten_opt_store({}, [], "trunk")
