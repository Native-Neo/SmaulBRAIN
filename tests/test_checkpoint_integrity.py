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

    with pytest.raises(ValueError, match="checkpoint parameter router.*expected"):
        CK.save_checkpoint(str(tmp_path / "invalid"), manifest, params, {})


def test_checkpoint_publish_failure_preserves_existing_valid_backup(tmp_path, monkeypatch):
    target = _write_checkpoint(tmp_path / "checkpoint-failure")
    backup = target.parent / "checkpoint-failure.bak"
    shutil.copytree(target, backup)

    params_path = target / "params.pt"
    damaged = bytearray(params_path.read_bytes())
    damaged[len(damaged) // 2] ^= 0xFF
    params_path.write_bytes(damaged)

    new_tmp = tmp_path / ".new-checkpoint"
    new_tmp.mkdir()
    (new_tmp / "sentinel").write_text("new generation")
    real_replace = CK.os.replace
    failed = {"done": False}

    def fail_publish_once(src, dst):
        if str(src) == str(new_tmp) and str(dst) == str(target) and not failed["done"]:
            failed["done"] = True
            raise OSError("injected final publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(CK.os, "replace", fail_publish_once)
    with pytest.raises(OSError, match="injected final publish failure"):
        CK._publish_dir(str(new_tmp), str(target))

    assert (target / "params.pt").is_file()
    assert (backup / "manifest.json").is_file()
    loaded = CK.load_checkpoint(str(target))
    assert loaded["dir"] == str(backup)


def test_checkpoint_rejects_wrong_dense_projection_shape(tmp_path):
    spec = C.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(4))
    params["L0.attn.Wr"] = torch.zeros((spec.d_model + 1, spec.d_model))
    manifest = CK.build_manifest(
        spec.name,
        C.arch_report(spec),
        {},
        CK.blank_scheduler_state(),
        0,
        {},
        0,
        [],
        {},
    )
    with pytest.raises(ValueError, match="L0.attn.Wr.*expected"):
        CK.save_checkpoint(str(tmp_path / "bad-shape"), manifest, params, {})


def test_checkpoint_rejects_optimizer_moment_shape_mismatch(tmp_path):
    spec = C.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(5))
    bad_state = {
        "embed": {
            "m": torch.zeros((spec.vocab_size, spec.d_model + 1)),
            "v": torch.zeros((spec.vocab_size, spec.d_model + 1)),
        }
    }
    manifest = CK.build_manifest(
        spec.name,
        C.arch_report(spec),
        {},
        CK.blank_scheduler_state(),
        0,
        {},
        0,
        [],
        {},
    )
    with pytest.raises(ValueError, match="optimizer m shape for embed"):
        CK.save_checkpoint(str(tmp_path / "bad-optimizer"), manifest, params, bad_state)


def test_checkpoint_rejects_architecture_expert_count_disagreement(tmp_path):
    spec = C.PRESETS["moe-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(6))
    arch = C.arch_report(spec, spec.n_experts_init)
    arch["experts_current"] = spec.n_experts_init - 1
    manifest = CK.build_manifest(
        spec.name, arch, {}, CK.blank_scheduler_state(), 0, {},
        spec.n_experts_init, list(range(spec.n_experts_init)), {},
    )
    with pytest.raises(ValueError, match="experts_current disagrees"):
        CK.save_checkpoint(str(tmp_path / "arch-count"), manifest, params, {})


def test_checkpoint_rejects_mismatched_factored_optimizer_flag(tmp_path):
    spec = C.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(7))
    opt_state = {
        "embed": {
            "m": torch.zeros_like(params["embed"]),
            "R": torch.zeros(spec.vocab_size),
            "C": torch.zeros(spec.d_model),
        }
    }
    manifest = CK.build_manifest(
        spec.name, C.arch_report(spec), {}, CK.blank_scheduler_state(), 0, {},
        0, [], {"class": "SmaulOPT", "factored": {"embed": False}},
    )
    with pytest.raises(ValueError, match="factored flag for embed disagrees"):
        CK.save_checkpoint(str(tmp_path / "bad-factored-flag"), manifest, params, opt_state)


def test_checkpoint_rejects_nonfinite_parameter_values(tmp_path):
    spec = C.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(8))
    params["embed"][0, 0] = float("nan")
    manifest = CK.build_manifest(
        spec.name, C.arch_report(spec), {}, CK.blank_scheduler_state(), 0, {},
        0, [], {},
    )
    with pytest.raises(ValueError, match="embed contains non-finite"):
        CK.save_checkpoint(str(tmp_path / "nonfinite"), manifest, params, {})
