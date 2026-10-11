"""Regression tests for CLI checkpoint config path validation."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli import load_saved_config


def test_cli_refuses_symlinked_config_before_parsing_it(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"d_model": 1_000_000}))
    (ckpt / "config.json").symlink_to(outside)
    (ckpt / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="symlink"):
        load_saved_config(str(ckpt))
