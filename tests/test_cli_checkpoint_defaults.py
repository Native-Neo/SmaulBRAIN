import json

import torch

import evaluate
import infer
from smaulbrain import checkpoint as CK
from smaulbrain import config as C
from smaulbrain import model as M


def _checkpoint(path):
    spec = C.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(17))
    config = {
        "forward_quant": "ternary",
        "backward_mode": "ternary",
        "paging": "R",
        "seed": 17,
        "seq_len": 12,
        "capacity_factor": 7.5,
        "lr": 0.0007,
    }
    manifest = CK.build_manifest(
        spec.name,
        C.arch_report(spec),
        config,
        CK.blank_scheduler_state(growth_every_steps=0, growth_loss_nats=None),
        0,
        {},
        0,
        [],
        {"class": "SmaulOPT"},
    )
    CK.save_checkpoint(str(path), manifest, params, {})
    return path


def test_inference_defaults_to_checkpoint_quantization_and_capacity(tmp_path, capsys, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "model")
    original_model = M.Model
    observed = {}

    def recording_model(*args, **kwargs):
        observed.update(kwargs)
        return original_model(*args, **kwargs)

    monkeypatch.setattr(M, "Model", recording_model)
    assert infer.main([
        "--checkpoint", str(checkpoint),
        "--prompt", "x",
        "--max-new-tokens", "0",
    ]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["preset"] == "dense-tiny"
    assert output["forward_quant"] == "ternary"
    assert output["backward_mode"] == "ternary"
    assert observed["forward_quant"] == "ternary"
    assert observed["backward_mode"] == "ternary"
    assert observed["capacity_factor"] == 7.5


def test_evaluation_defaults_to_checkpoint_quantization_and_capacity(tmp_path, capsys, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "model")
    original_model = M.Model
    observed = {}

    def recording_model(*args, **kwargs):
        observed.update(kwargs)
        return original_model(*args, **kwargs)

    monkeypatch.setattr(M, "Model", recording_model)
    assert evaluate.main([
        "--checkpoint", str(checkpoint),
        "--eval-batches", "1",
    ]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["preset"] == "dense-tiny"
    assert output["forward_quant"] == "ternary"
    assert output["backward_mode"] == "ternary"
    assert output["capacity_factor"] == 7.5
    assert observed["forward_quant"] == "ternary"
    assert observed["backward_mode"] == "ternary"
    assert observed["capacity_factor"] == 7.5
