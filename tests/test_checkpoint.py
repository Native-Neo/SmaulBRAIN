"""Checkpoints: save/load roundtrip, optimizer restore, FP8 load, atomicity."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from storage import load_expert_file, load_model, save_model
from train import ReplayBuffer, run_training


def _resume_model(**kw):
    torch.manual_seed(0)
    base = dict(d_model=32, n_heads=4, num_experts=4, top_k=2, expert_hidden=64,
                max_depth=2, context_length=24, expert_lr=3e-2)
    base.update(kw)
    cfg = SmaulBrainConfig(**base)
    return SmaulBrainModel(cfg), SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr)), cfg


def _resume_seqs():
    import random
    rng = random.Random(11)
    return [[rng.randrange(256) for _ in range(30)] for _ in range(16)]


def _trained(tmp, seed=0):
    torch.manual_seed(seed)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=12)
    m = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams())
    ids = torch.randint(0, 256, (2, 12))
    out = m(ids, ids, step=0)
    out["loss"].backward()
    opt.step_trunk(m._trunk_params(), cfg.trunk_lr)
    save_model(tmp, m, opt, step=3)
    return m, opt


def test_save_load_roundtrip_bit_identical(tmp_path):
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    m2 = SmaulBrainModel(m.cfg)
    man = load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    assert man["step"] == 3 and man["expert_ids"] == m.pool.order
    assert torch.equal(m.embed.weight, m2.embed.weight)
    for eid in m.pool.order:
        a = m.pool.experts[eid].weights_fp8["w_gate"].codes
        b = m2.pool.experts[eid].weights_fp8["w_gate"].codes
        assert torch.equal(a, b)  # FP8 bytes survive the cycle
    m.pager.close(); m2.pager.close()


def test_optimizer_state_restored_by_expert_id(tmp_path):
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    eid = m.pool.order[0]
    m.pool.experts[eid].grad_activity = 0.5
    save_model(d, m, opt, step=4)
    m2 = SmaulBrainModel(m.cfg)
    load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    rec = m2.pool.experts[eid]
    assert rec.grad_activity == 0.5
    assert set(rec.optim_state) == {"w_gate", "w_up", "w_down"}
    m.pager.close(); m2.pager.close()


def test_single_expert_file_without_full_model(tmp_path):
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    rec = load_expert_file(os.path.join(d, "experts", "expert_00002.pt"))
    assert rec.expert_id == "expert_00002"
    assert rec.param_count == m.pool.experts["expert_00002"].param_count
    m.pager.close()


def test_no_temp_files_left_behind(tmp_path):
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    leftovers = [f for f in os.listdir(d) if f.startswith("tmp_")]
    assert leftovers == []
    exp = [f for f in os.listdir(os.path.join(d, "experts")) if f.startswith("tmp_")]
    assert exp == []
    m.pager.close()


def test_inference_identical_after_reload(tmp_path):
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    ids = torch.randint(0, 256, (1, 12))
    torch.manual_seed(0)
    a = m.forward_infer(ids)["logits"]
    m2 = SmaulBrainModel(m.cfg)
    load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    torch.manual_seed(0)
    b = m2.forward_infer(ids)["logits"]
    assert torch.equal(a, b)
    m.pager.close(); m2.pager.close()


def test_resume_step_and_saved_config_are_restored(tmp_path):
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=4, context_length=99)
    m2 = SmaulBrainModel(cfg)
    loaded = load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    assert loaded["step"] == 3
    assert m2._resume_step == 3
    assert m2.cfg.max_depth == 2
    assert m2.cfg.context_length == 12
    m.pager.close(); m2.pager.close()


def _losses(hist):
    return [(h["loss"], h["nll"], h["acc"], h["mean_depth"]) for h in hist]


def test_save_resume_matches_uninterrupted_run(tmp_path):
    """Mid-run checkpoint + resume == uninterrupted run, bit for bit.

    Exercises the full scheduler/RNG restore path: periodic save carries
    the loss edge, growth counter, replay buffer and RNG state; the
    resumed run (fresh objects, replay rebuilt from snapshot) must
    reproduce the uninterrupted trajectory exactly, including a growth
    event on each leg.
    """
    seqs = _resume_seqs()
    kw = dict(batch_size=2, replay_n=1, grow_every=2,
              grow_loss_below=-1.0, seed=5, log_fn=lambda s: None)

    m, opt, cfg = _resume_model()
    ref = run_training(m, opt, cfg, seqs, steps=4,
                       replay=ReplayBuffer(capacity=8, seed=7), **kw)
    ref_losses = _losses(ref["history"])
    ref_order = list(m.pool.order)
    ref_embed = m.embed.weight.detach().clone()

    d = str(tmp_path / "r")
    m1, opt1, cfg1 = _resume_model()
    leg1 = run_training(m1, opt1, cfg1, seqs, steps=2,
                        replay=ReplayBuffer(capacity=8, seed=7),
                        ckpt_dir=d, save_every=2, **kw)
    assert leg1["steps_completed"] == 2
    m1.pager.close()

    m2, opt2, _ = _resume_model()
    load_model(d, m2, opt2)
    assert m2._resume_step == 1
    assert m2._scheduler_snapshot.get("growth_events") == 1
    cfg2 = m2.cfg  # checkpoint is authoritative for runtime config
    leg2 = run_training(m2, opt2, cfg2, seqs, steps=2, batch_size=2,
                        replay_n=1, grow_every=2, grow_loss_below=-1.0,
                        seed=5, log_fn=lambda s: None)

    got = _losses(leg1["history"]) + _losses(leg2["history"])
    assert got == ref_losses
    assert list(m2.pool.order) == ref_order
    assert torch.equal(m2.embed.weight, ref_embed)
    m2.pager.close(); m.pager.close()


def test_save_refuses_diverged_topology_without_writing(tmp_path):
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    before = set(os.listdir(d))
    m.router.add_expert_row()  # diverged: router wider than pool
    with pytest.raises(ValueError):
        save_model(d, m, opt, step=9)
    assert set(os.listdir(d)) == before  # previous generation untouched
    m.pager.close()


def test_save_refuses_negative_step(tmp_path):
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    with pytest.raises(ValueError):
        save_model(d, m, opt, step=-1)
    m.pager.close()


def test_optimizer_hparams_roundtrip_authoritatively(tmp_path):
    from smaulopt import SmaulOptHParams
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    opt.hp.clip = 0.5
    opt.hp.wd = 0.03
    save_model(d, m, opt, step=5)
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())  # defaults differ from saved
    assert opt2.hp.clip != 0.5
    load_model(d, m2, opt2)
    assert opt2.hp.clip == 0.5 and opt2.hp.wd == 0.03  # checkpoint wins
    m.pager.close(); m2.pager.close()


def test_retile_updates_precision_metadata_and_reloads(tmp_path):
    import json
    from quantize import convert_checkpoint
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    reports = convert_checkpoint(d, to="fp8", tile=32)
    assert reports and all(r["to"] == "fp8" for r in reports)
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["precision"] == {"format": "fp8", "fp8_tile": 32}
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert cfg["fp8_tile"] == 32
    assert not [f for f in os.listdir(os.path.join(d, "experts"))
                if f.endswith(".convert_tmp")]  # no sidecars left behind
    m2 = SmaulBrainModel(m.cfg)
    load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    assert m2.cfg.fp8_tile == 32 and len(m2.pool) == len(m.pool)
    m.pager.close(); m2.pager.close()
