import torch

from smaulbrain import checkpoint as CK
from smaulbrain import config as C
from smaulbrain import model as M
from smaulbrain import prune_cmd as PR


def test_pruning_uses_saved_capacity_and_quantization_settings(tmp_path, monkeypatch):
    spec = C.PRESETS["moe-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(0))
    config = {
        "forward_quant": "ternary",
        "backward_mode": "fp32",
        "capacity_factor": 3.75,
    }
    manifest = CK.build_manifest(
        spec.name,
        C.arch_report(spec),
        config,
        CK.blank_scheduler_state(growth_every_steps=0, growth_loss_nats=None),
        0,
        {},
        spec.n_experts_init,
        list(range(spec.n_experts_init)),
        {"class": "SmaulOPT"},
    )
    checkpoint = tmp_path / "checkpoint"
    CK.save_checkpoint(str(checkpoint), manifest, params, {})

    observed = {}

    def fake_rank(model, batches, min_samples):
        observed["capacity_factor"] = model.capacity_factor
        observed["forward_quant"] = model.forward_quant
        observed["backward_mode"] = model.backward_mode
        return {eid: float(eid) for eid in range(spec.n_experts_init)}, 100

    monkeypatch.setattr("smaulbrain.prune_cmd._growth.rank_experts", fake_rank)
    report = PR.prune_checkpoint(
        str(checkpoint),
        rm_worst=1,
        eval_batches=[[1, 2, 3]],
        min_samples=1,
        dry_run=True,
    )

    assert observed == {
        "capacity_factor": 3.75,
        "forward_quant": "ternary",
        "backward_mode": "fp32",
    }
    assert report["capacity_factor"] == 3.75
    assert report["forward_quant"] == "ternary"
