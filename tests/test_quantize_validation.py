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

def test_quantizer_validates_config_before_sweeping_or_converting(tmp_path):
    ckpt = tmp_path / "checkpoint"
    experts = ckpt / "experts"
    experts.mkdir(parents=True)
    (ckpt / "manifest.json").write_text(json.dumps({"expert_ids": ["expert_000"]}))
    (ckpt / "config.json").write_text("{ malformed json")
    root_tmp = ckpt / "tmp_ckpt_keep"
    expert_tmp = experts / "tmp_ckpt_keep"
    root_tmp.write_text("keep-root")
    expert_tmp.write_text("keep-expert")

    with pytest.raises(ValueError, match="config.json"):
        convert_checkpoint(str(ckpt), to="fp8")
    assert root_tmp.read_text() == "keep-root"
    assert expert_tmp.read_text() == "keep-expert"
