"""Checkpoints: save/load roundtrip, optimizer restore, FP8 load, atomicity."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from storage import load_expert_file, load_model, save_model


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
