"""C1 regression: manifest files-map completeness (test-first).

SECURITY owns storage.py — this file only pins the contract. Pre-merge
two cases fail (truncated map accepted; empty/None map accepted via the
hash-less fallback); swapped-payload and missing-file already refuse.
Post-merge (security completeness fix) all must raise ValueError.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
import torch

from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from storage import load_model, save_model


def _tiny_cfg():
    return SmaulBrainConfig(
        d_model=32, n_heads=4, num_experts=2, top_k=1, expert_hidden=16,
        max_depth=2, min_depth=1, context_length=16,
        dtype="fp32", state_dtype="fp32", threads=1, seed=0, min_experts=1,
    )


def _save_tiny(tmp_path):
    d = str(tmp_path / "ckpt")
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, state_dtype=cfg.state_dtype))
    save_model(d, m, opt, 0)
    return d, cfg, m, opt


def _fresh_model(cfg):
    m2 = SmaulBrainModel(cfg)
    o2 = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, state_dtype=cfg.state_dtype))
    return m2, o2


def _rewrite_manifest(ckpt_dir, mutate):
    p = os.path.join(ckpt_dir, "manifest.json")
    with open(p) as f:
        man = json.load(f)
    mutate(man)
    with open(p, "w") as f:
        json.dump(man, f)
    return man


def test_truncated_files_map_refuses(tmp_path):
    """Dropping one entry from files map must refuse (completeness)."""
    d, cfg, m, opt = _save_tiny(tmp_path)
    try:
        def drop_one(man):
            files = dict(man["files"])
            files.pop(sorted(files.keys())[0])
            man["files"] = files
        _rewrite_manifest(d, drop_one)
        m2, o2 = _fresh_model(cfg)
        try:
            with pytest.raises(ValueError):
                load_model(d, m2, o2)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_swapped_payload_refuses(tmp_path):
    """Swapping two payload files must refuse via hash mismatch."""
    d, cfg, m, opt = _save_tiny(tmp_path)
    try:
        tp = os.path.join(d, "trunk.safetensors")
        rp = os.path.join(d, "router.safetensors")
        with open(tp, "rb") as f:
            tb = f.read()
        with open(rp, "rb") as f:
            rb = f.read()
        with open(tp, "wb") as f:
            f.write(rb)
        with open(rp, "wb") as f:
            f.write(tb)
        m2, o2 = _fresh_model(cfg)
        try:
            with pytest.raises(ValueError):
                load_model(d, m2, o2)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


@pytest.mark.parametrize("bad", [[], "x", 123, {}, None])
def test_wrong_type_files_map_refuses(tmp_path, bad):
    """Non-dict (or empty/None) files map must refuse with ValueError."""
    import warnings
    d, cfg, m, opt = _save_tiny(tmp_path)
    try:
        _rewrite_manifest(d, lambda man: man.__setitem__("files", bad))
        m2, o2 = _fresh_model(cfg)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError):
                    load_model(d, m2, o2)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_missing_tracked_file_refuses(tmp_path):
    """Deleting a manifest-tracked payload must refuse (mixed generation)."""
    d, cfg, m, opt = _save_tiny(tmp_path)
    try:
        os.remove(os.path.join(d, "trunk.safetensors"))
        m2, o2 = _fresh_model(cfg)
        try:
            with pytest.raises(ValueError):
                load_model(d, m2, o2)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()
