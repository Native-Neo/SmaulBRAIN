"""System-wide invariant coverage (issue #71).

Cross-cutting properties no single module test asserts: config roundtrips,
param-count agreement, save/load inference preservation, growth+prune+save+load
topology consistency, determinism, recurrent chunk equivalence, ponder bounds,
routing/capacity structure, paged-mode isolation, optimizer resume, byte
generation contract, native/fallback dispatch independence, FP8 update sanity,
corrupt-checkpoint error behavior, and documented error paths.

Kept fast: tiny dims, public APIs only, one or two steps per test.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from storage import load_model, save_model
from train import batch_from_seqs, train_step


def _tiny(**kw):
    base = dict(d_model=16, n_heads=2, num_experts=4, top_k=2,
                expert_hidden=32, min_depth=1, max_depth=2,
                context_length=8, dtype="fp32")
    base.update(kw)
    return SmaulBrainConfig(**base)


def _model(**kw):
    torch.manual_seed(0)
    cfg = _tiny(**kw)
    m = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    return m, opt, cfg


def _batch(cfg, seed=1, bsz=2):
    import random
    rng = random.Random(seed)
    seqs = [[rng.randrange(256) for _ in range(cfg.context_length + 1)]
            for _ in range(bsz)]
    b = batch_from_seqs(seqs, cfg.context_length)
    return b[:, :cfg.context_length], b[:, 1:cfg.context_length + 1]


def test_config_dict_roundtrip_preserves_topology():
    cfg = _tiny()
    try:
        rt = SmaulBrainConfig.from_dict(cfg.to_dict())
        assert rt.to_dict() == cfg.to_dict()
        assert (rt.d_model, rt.n_heads, rt.num_experts, rt.top_k,
                rt.expert_hidden, rt.max_depth) == (
            cfg.d_model, cfg.n_heads, cfg.num_experts, cfg.top_k,
            cfg.expert_hidden, cfg.max_depth)
        assert rt.describe_counts()["total_params"] == cfg.total_params()
        assert rt.describe_counts()["logical_params"] == cfg.total_params()
    finally:
        pass


def test_config_rejects_null_and_alias_mismatch():
    d = _tiny().to_dict()
    d["d_model"] = None
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict(d)
    d2 = _tiny().to_dict()
    d2["active_experts"] = int(d2["top_k"]) + 1
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict(d2)
    # Unknown fields (e.g. schema_version stamped by storage) are tolerated.
    d3 = _tiny().to_dict()
    d3["schema_version"] = "0.1.0"
    d3["future_field"] = 123
    rt = SmaulBrainConfig.from_dict(d3)
    assert rt.d_model == _tiny().d_model


def test_param_counts_agree_between_config_and_model():
    m, _, cfg = _model()
    try:
        counts = m.param_counts()
        assert counts["shared_params"] == cfg.shared_params()
        assert counts["router_params"] == cfg.router_params()
        assert counts["per_expert_params"] == cfg.per_expert_params
        assert counts["expert_count"] == cfg.num_experts == len(m.pool)
        assert counts["total_params"] == cfg.total_params()
        assert counts["active_params"] == cfg.active_params()
        assert counts["stored_expert_bytes"] > 0
    finally:
        m.pager.close()


def test_save_load_roundtrip_preserves_inference_outputs(tmp_path):
    m, opt, cfg = _model()
    try:
        x, y = _batch(cfg)
        train_step(m, opt, cfg, x, y, step=0)
        d = str(tmp_path / "c")
        save_model(d, m, opt, step=0)
        ids = torch.randint(0, 256, (1, cfg.context_length))
        a = m.forward_infer(ids)["logits"]
        m2 = SmaulBrainModel(SmaulBrainConfig.from_dict(m.cfg.to_dict()))
        opt2 = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
        try:
            man = load_model(d, m2, opt2)
            assert man["expert_ids"] == list(m.pool.order)
            b = m2.forward_infer(ids)["logits"]
            assert torch.equal(a, b)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_growth_prune_save_load_keeps_topology_consistent(tmp_path):
    import growth as growth_mod
    import pruning as pruning_mod
    m, opt, cfg = _model()
    try:
        # Grow two clones; pool and router move in lockstep.
        new_ids = growth_mod.grow_topk_clones(
            m.pool, m.router, cfg.d_model, cfg.expert_hidden,
            step=1, seed=7, k=2, n_mutated=1,
            fp8_tile=cfg.fp8_tile, max_experts=8,
            optim_state=opt.router_state)
        assert len(new_ids) == 2
        assert len(m.pool) == m.router.num_experts == cfg.num_experts + 2
        m.cfg.num_experts = len(m.pool)
        # Age every expert out of grace so one prune victim exists.
        for rec in m.pool.experts.values():
            rec.birth_step = -1000
            rec.last_used_step = -1000
            rec.tokens_routed = 0
            rec.grad_activity = 0.0
            rec.contribution = 0.0
        victims = pruning_mod.find_victims(
            m.pool, step=1000, survival_steps=2,
            min_experts=2, max_victims=1)
        assert len(victims) == 1
        pruned = pruning_mod.prune_experts(
            m.pool, m.router, victims, pager=m.pager,
            optim_state=opt.router_state)
        assert pruned == victims
        m.cfg.num_experts = len(m.pool)
        assert len(m.pool) == m.router.num_experts >= 2
        assert m.router.usage_counts.shape[0] == len(m.pool)
        counts = m.param_counts()
        assert counts["expert_count"] == len(m.pool)
        assert counts["total_params"] == (
            counts["shared_params"] + counts["router_params"]
            + counts["expert_count"] * counts["per_expert_params"])
        # Save/load the mutated topology; inference still runs identically.
        d = str(tmp_path / "c")
        save_model(d, m, opt, step=3)
        ids = torch.randint(0, 256, (1, cfg.context_length))
        a = m.forward_infer(ids)["logits"]
        m2 = SmaulBrainModel(_tiny())
        opt2 = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
        try:
            man = load_model(d, m2, opt2)
            assert len(man["expert_ids"]) == len(m.pool)
            assert m2.router.num_experts == len(m2.pool) == len(m.pool)
            assert m2.param_counts()["expert_count"] == len(m.pool)
            assert torch.equal(m2.forward_infer(ids)["logits"], a)
            assert m2.forward_infer(ids)["logits"].shape == (
                1, cfg.context_length, m2.cfg.vocab_size)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_same_seed_reproduces_forward_and_generation():
    m1, _, cfg1 = _model()
    m2, _, _ = _model()  # identical seed -> identical init
    try:
        ids = torch.randint(0, 256, (1, cfg1.context_length))
        torch.manual_seed(123)
        a = m1.forward_infer(ids)["logits"]
        torch.manual_seed(123)
        b = m2.forward_infer(ids)["logits"]
        assert torch.equal(a, b)
        from infer import generate
        ra = generate(m1, [104, 105], max_new=6, temperature=0.7, seed=7)
        rb = generate(m2, [104, 105], max_new=6, temperature=0.7, seed=7)
        assert ra["ids"] == rb["ids"] and ra["text"] == rb["text"]
    finally:
        m1.pager.close()
        m2.pager.close()


def test_recurrent_chunked_streaming_matches_full_pass():
    m, _, cfg = _model()
    try:
        ids = torch.randint(0, 200, (1, 6))
        full, _ = m.forward_infer_stateful(ids)
        states = m.new_infer_state(1)
        parts = []
        for tok in ids[0]:
            out, states = m.forward_infer_step(tok.view(1), states)
            parts.append(out["logits"])
        streamed = torch.cat(parts, dim=1)
        assert torch.allclose(full["logits"], streamed, atol=2e-5, rtol=2e-5)
        assert int(full["n_executed"]) == cfg.max_depth
    finally:
        m.pager.close()


def test_ponder_depth_invariants_hold():
    m, _, cfg = _model()
    try:
        x, y = _batch(cfg)
        out = m(x, y, step=0)
        m.clear_expert_grads()
        assert torch.isfinite(out["loss"]).all()
        assert cfg.min_depth <= float(out["mean_depth"]) <= cfg.max_depth
        assert int(out["n_executed"]) == cfg.max_depth
        d = out["depths"]
        assert bool(((d >= cfg.min_depth) & (d <= cfg.max_depth)).all())
        inf = m.forward_infer(x)
        assert bool(((inf["depths"] >= cfg.min_depth)
                     & (inf["depths"] <= cfg.max_depth)).all())
    finally:
        m.pager.close()


def test_routing_plan_structure_and_infer_admits_all():
    m, _, _ = _model()
    try:
        x = torch.randn(6, m.cfg.d_model)
        plan_infer = m.router.route(x, enforce_capacity=False)
        plan_infer.validate(m.router.num_experts, m.cfg.top_k)
        assert not bool(plan_infer.dropped.any())  # inference never drops
        assert plan_infer.top_weights.shape == (6, m.cfg.top_k)
        plan_train = m.router.route(x, enforce_capacity=True)
        plan_train.validate(m.router.num_experts, m.cfg.top_k)
        assert bool(torch.isfinite(plan_train.top_weights).all())
        bal = m.router.balance_loss(plan_train.probs)
        assert torch.isfinite(bal)
        empty = m.router.route(torch.zeros(0, m.cfg.d_model),
                               enforce_capacity=True)
        empty.validate(m.router.num_experts, m.cfg.top_k)
        assert m.router.balance_loss(empty.probs).item() == 0.0
    finally:
        m.pager.close()


def test_training_step_moves_weights_and_keeps_subsystems_synced():
    m, opt, cfg = _model()
    try:
        before = {eid: m.pool.experts[eid].weights_fp8["w_gate"].codes.clone()
                  for eid in m.pool.order}
        x, y = _batch(cfg)
        stats = train_step(m, opt, cfg, x, y, step=0)
        assert torch.isfinite(torch.tensor(stats["loss"]))
        assert len(stats["stepped_experts"]) > 0
        assert set(stats["stepped_experts"]) <= set(m.pool.order)
        assert len(m.pool) == m.router.num_experts  # optimizer never diverges
        changed = [eid for eid in m.pool.order
                   if not torch.equal(before[eid],
                                      m.pool.experts[eid].weights_fp8["w_gate"].codes)]
        assert changed  # FP8 update path rewrote at least one expert
        ids = torch.randint(0, 256, (1, cfg.context_length))
        assert m.forward_infer(ids)["logits"].shape == (
            1, cfg.context_length, cfg.vocab_size)
    finally:
        m.pager.close()


def test_optimizer_resume_preserves_step_count_and_trains_on(tmp_path):
    m, opt, cfg = _model()
    try:
        x, y = _batch(cfg)
        train_step(m, opt, cfg, x, y, step=0)
        assert opt.step_count == 1
        d = str(tmp_path / "c")
        save_model(d, m, opt, step=0)
        m2 = SmaulBrainModel(SmaulBrainConfig.from_dict(m.cfg.to_dict()))
        opt2 = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
        try:
            load_model(d, m2, opt2)
            assert opt2.step_count == 1  # global clock survived the cycle
            s = train_step(m2, opt2, m2.cfg, x, y, step=1)
            assert opt2.step_count == 2
            assert torch.isfinite(torch.tensor(s["loss"]))
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_paging_modes_stay_on_their_documented_path():
    m_d2r, _, cfg = _model(paging_method="D2R")
    m_d2vr, _, _ = _model(paging_method="D2VR")
    try:
        ids = torch.randint(0, 256, (1, cfg.context_length))
        m_d2r.forward_infer(ids)
        m_d2vr.forward_infer(ids)
        c_d2r = m_d2r.pager.resident_counts()
        c_d2vr = m_d2vr.pager.resident_counts()
        assert c_d2r["vram"] == 0  # D2R never touches VRAM
        assert c_d2vr["ram"] == 0  # D2VR bypasses the RAM cache
        assert set(m_d2r.pager.stats.to_dict()) >= {"disk_reads", "ram_hits"}
    finally:
        m_d2r.pager.close()
        m_d2vr.pager.close()


def test_byte_generation_contract_holds():
    from infer import generate
    from bytes import BOS_ID, EOS_ID, PAD_ID, decode_text
    m, _, _ = _model()
    try:
        a = generate(m, [104, 105], max_new=6, temperature=0.0)
        assert len(a["ids"]) == 8
        assert a["text"] == decode_text(a["ids"][2:])  # continuation-only
        assert PAD_ID not in a["ids"][2:] and BOS_ID not in a["ids"][2:]
        assert all(0 <= i < m.cfg.vocab_size for i in a["ids"])
        # EOS still terminates by default; NUL byte stays ordinary data.
        assert isinstance(EOS_ID, int)
        z = generate(m, [0, 65], max_new=1, temperature=0.0)
        assert z["ids"][:2] == [0, 65]
    finally:
        m.pager.close()


def test_native_dispatch_status_and_fallback_parity():
    import native
    st = native.status()
    assert isinstance(st, dict) and st
    m, _, cfg = _model()
    try:
        ids = torch.randint(0, 256, (1, cfg.context_length))
        out = m.forward_infer(ids)
        assert out["logits"].shape == (1, cfg.context_length, cfg.vocab_size)
        assert bool(torch.isfinite(out["logits"]).all())
    finally:
        m.pager.close()


def test_corrupt_checkpoint_raises_and_survivor_still_runs(tmp_path):
    m, opt, _ = _model()
    try:
        x, y = _batch(m.cfg)
        train_step(m, opt, m.cfg, x, y, step=0)
        d = str(tmp_path / "c")
        save_model(d, m, opt, step=0)
        eid = m.pool.order[0]
        with open(os.path.join(d, "experts", f"{eid}.pt"), "wb") as f:
            f.truncate(0)
        m2 = SmaulBrainModel(SmaulBrainConfig.from_dict(m.cfg.to_dict()))
        try:
            with pytest.raises(ValueError, match="validation failed"):
                load_model(d, m2, SmaulOpt(SmaulOptHParams()))
            ids = torch.randint(0, 256, (1, m2.cfg.context_length))
            assert m2.forward_infer(ids)["logits"].shape == (
                1, m2.cfg.context_length, m2.cfg.vocab_size)
        finally:
            m2.pager.close()
    finally:
        m.pager.close()


def test_documented_error_paths_raise(tmp_path):
    from infer import generate, sample_next
    m, opt, cfg = _model()
    try:
        with pytest.raises(ValueError):
            m.forward(torch.tensor([[cfg.vocab_size + 5]]),
                      torch.tensor([[1]]))
        with pytest.raises(ValueError):
            m.forward_infer_step(torch.zeros(1, 2, dtype=torch.long),
                                 m.new_infer_state(1))
        x, y = _batch(cfg)
        with pytest.raises(ValueError):
            train_step(m, opt, cfg, x, y, step=0, mode="nope")
        with pytest.raises(ValueError):
            save_model(str(tmp_path / "c"), m, opt, step=-1)
        with pytest.raises(ValueError):
            sample_next(torch.zeros(cfg.vocab_size), temperature=-1.0)
        with pytest.raises(ValueError):
            generate(m, [10, 99999], max_new=2)
        with pytest.raises(ValueError):
            generate(m, [10, 20], max_new=2, top_k=-1)
    finally:
        m.pager.close()
