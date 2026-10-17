import json
import shutil

import pytest
import torch

from smaulbrain import checkpoint as CK
from smaulbrain import config as C
from smaulbrain import model as M


def _write_checkpoint(path):
    spec = C.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(0))
    manifest = CK.build_manifest(
        spec.name,
        C.arch_report(spec),
        {"forward_quant": "fp32", "backward_mode": "fp32"},
        CK.blank_scheduler_state(),
        0,
        {"torch_state": []},
        0,
        [],
        {"class": "SmaulOPT"},
    )
    CK.save_checkpoint(str(path), manifest, params, {})
    return path


def test_load_checkpoint_falls_back_when_primary_integrity_fails(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    backup = tmp_path / "checkpoint.bak"
    shutil.copytree(checkpoint, backup)

    params_path = checkpoint / "params.pt"
    damaged = bytearray(params_path.read_bytes())
    damaged[len(damaged) // 2] ^= 0xFF
    params_path.write_bytes(damaged)

    loaded = CK.load_checkpoint(str(checkpoint))
    assert loaded["dir"] == str(backup)
    assert loaded["manifest"]["preset"] == "dense-tiny"


def test_current_checkpoint_requires_hashes(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]["params.pt"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="no valid checkpoint found"):
        CK.load_checkpoint(str(checkpoint))


def test_checkpoint_rejects_router_width_mismatch(tmp_path):
    spec = C.PRESETS["moe-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(0))
    params["router"] = params["router"][:, :-1]
    manifest = CK.build_manifest(
        spec.name,
        C.arch_report(spec),
        {},
        CK.blank_scheduler_state(),
        0,
        {},
        spec.n_experts_init,
        list(range(spec.n_experts_init)),
        {},
    )

    with pytest.raises(ValueError, match="router shape disagrees"):
        CK.save_checkpoint(str(tmp_path / "invalid"), manifest, params, {})
