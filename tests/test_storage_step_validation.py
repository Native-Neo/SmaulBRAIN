"""Regression tests for strict checkpoint step validation."""

from types import SimpleNamespace
import os
import sys

import pytest

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
