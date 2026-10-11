"""Regression tests for quantizer checkpoint path jail enforcement."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantize import convert_checkpoint


def test_quantizer_rejects_symlinked_config_before_staging(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "manifest.json").write_text(json.dumps({"expert_ids": []}))
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"fp8_tile": 8}))
    (ckpt / "config.json").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        convert_checkpoint(str(ckpt), to="fp8")
    assert outside.read_text() == json.dumps({"fp8_tile": 8})
