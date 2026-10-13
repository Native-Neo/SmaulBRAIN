"""CLI preflight tests that must run before model-sized allocations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cli
from cli import main


def test_missing_training_data_fails_before_model_construction(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "does-not-exist"
    ckpt = tmp_path / "checkpoint"

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("model construction happened before data validation")

    monkeypatch.setattr(cli, "SmaulBrainModel", forbidden_constructor)
    result = main([
        "--allow-outside-ckpt", "--ckpt", str(ckpt),
        "train", "--steps", "1", "--data", str(missing),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "error: --data:" in captured.err
    assert "dataset path not found" in captured.err
    assert "model construction happened" not in captured.err
