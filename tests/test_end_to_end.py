"""End-to-end regression suite (issue #51): full lifecycle smoke test.

Covers through public APIs only (config/cli/train/infer/storage/growth/
pruning + quantize/native/paging/precision/model):
  training with growth+prune schedule -> checkpoint save -> reload into fresh
  objects -> resume determinism -> inference from resumed model ->
  quantize/retile roundtrip -> corrupted-checkpoint refusal.

Kept fast: tiny dims, few steps, deterministic seeds.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from config import SmaulBrainConfig
from infer import generate
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from storage import load_model, save_model
from train import ReplayBuffer, batch_from_seqs, evaluate_loss, run_training


def _tiny_cfg(**kw):
    base = dict(
        d_model=32,
        n_heads=4,
        num_experts=4,
        top_k=2,
        expert_hidden=32,
        max_depth=2,
        min_depth=1,
        context_length=16,
        expert_lr=3e-2,
        prune_survival_steps=2,
        prune_min_usage=1e-4,
        max_new_experts=2,
        max_experts=8,
        min_experts=2,
    )
    base.update(kw)
    return SmaulBrainConfig(**base)


def _tiny_model(**kw):
    torch.manual_seed(0)
    cfg = _tiny_cfg(**kw)
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    return model, opt, cfg


def _demo_seqs(n=12, length=20, seed=11):
    import random

    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in range(length)] for _ in range(n)]


def _loss_tuples(hist):
    return [(h["loss"], h["nll"], h["acc"], h["mean_depth"]) for h in hist]


def _opt_snapshot(opt):
    return {
        "step_count": int(opt.step_count),
        "trunk_keys": sorted(opt.trunk_state.keys()),
        "router_keys": sorted(opt.router_state.keys()),
    }


def _expert_bytes(model):
    return {
        eid: model.pool.experts[eid].weights_fp8["w_gate"].codes.clone()
        for eid in model.pool.order
    }


def test_e2e_train_growth_save_resume_determinism(tmp_path):
    """Tiny train with growth+prune schedule; checkpoint/resume is bit-identical."""
    import growth as growth_mod
    import pruning as pruning_mod

    seqs = _demo_seqs()
    kw = dict(
        batch_size=2,
        replay_n=1,
        grow_every=2,
        prune_every=2,
        grow_loss_below=-1.0,
        growths_per_prune=2,
        seed=5,
        log_fn=lambda s: None,
    )

    # Reference uninterrupted run.
    m, opt, cfg = _tiny_model()
    ref = run_training(
        m, opt, cfg, seqs, steps=4,
        replay=ReplayBuffer(capacity=8, seed=7), **kw
    )
    ref_losses = _loss_tuples(ref["history"])
    ref_order = list(m.pool.order)
    ref_embed = m.embed.weight.detach().clone()
    ref_opt = _opt_snapshot(opt)
    ref_experts = _expert_bytes(m)
    assert ref["growth_events"] >= 1  # scheduled growth fired
    assert all("grew" in h and "pruned_experts" in h for h in ref["history"])
    # Public growth/pruning APIs are exercised on the live pool.
    assert growth_mod.select_parents(m.pool, k=2)
    assert isinstance(
        pruning_mod.find_victims(m.pool, step=10, survival_steps=2, min_experts=1),
        list,
    )
    assert pruning_mod.dying_score(0, 100, 0.0, 0.0) is True
    assert pruning_mod.dying_score(50, 100, 1e-3, 1.0) is False
    infer_before = m.forward_infer(torch.randint(0, 256, (1, 8)))["logits"].clone()

    # Two-leg run with a mid-run checkpoint.
    d = str(tmp_path / "e2e")
    m1, opt1, cfg1 = _tiny_model()
    leg1 = run_training(
        m1, opt1, cfg1, seqs, steps=2,
        replay=ReplayBuffer(capacity=8, seed=7),
        ckpt_dir=d, save_every=2, **kw
    )
    assert leg1["steps_completed"] == 2
    assert os.path.exists(os.path.join(d, "manifest.json"))
    assert os.path.exists(os.path.join(d, "rng.pt"))
    m1.pager.close()

    # Reload into fresh objects; topology/params/optimizer/RNG must restore.
    m2, opt2, _ = _tiny_model()
    order_before = list(m2.pool.order)
    assert order_before != []  # fresh pool exists before load
    man = load_model(d, m2, opt2)
    assert man["step"] == 1
    assert m2._resume_step == 1
    assert m2._scheduler_snapshot.get("growth_events") == 1
    cfg2 = m2.cfg  # checkpoint is authoritative
    leg2 = run_training(
        m2, opt2, cfg2, seqs, steps=2, batch_size=2,
        replay_n=1, grow_every=2, prune_every=2,
        grow_loss_below=-1.0, growths_per_prune=2,
        seed=5, log_fn=lambda s: None,
    )

    got = _loss_tuples(leg1["history"]) + _loss_tuples(leg2["history"])
    assert got == ref_losses  # resume determinism spot-check
    assert list(m2.pool.order) == ref_order  # topology identical
    assert torch.equal(m2.embed.weight, ref_embed)  # parameters identical
    assert _opt_snapshot(opt2) == ref_opt  # optimizer restoration
    for eid in ref_order:
        assert torch.equal(
            m2.pool.experts[eid].weights_fp8["w_gate"].codes,
            ref_experts[eid],
        )
    # Outputs identical after reload.
    torch.manual_seed(1234)
    probe = torch.randint(0, 256, (1, 8))
    torch.manual_seed(1234)
    probe2 = torch.randint(0, 256, (1, 8))
    assert torch.equal(probe, probe2)
    a = m.forward_infer(probe)["logits"]
    b = m2.forward_infer(probe)["logits"]
    assert torch.equal(a, b)
    m.pager.close()
    m2.pager.close()


def test_e2e_inference_and_recurrent_state_from_resumed(tmp_path):
    """Generation + streaming/stateful recurrent parity from a resumed model."""
    d = str(tmp_path / "e2e_infer")
    m, opt, cfg = _tiny_model()
    run_training(m, opt, cfg, _demo_seqs(), steps=2, batch_size=2,
                 grow_every=2, grow_loss_below=-1.0, seed=5,
                 log_fn=lambda s: None)
    save_model(d, m, opt, step=int(getattr(m, "_resume_step", 1)),
               extra_meta={"scheduler": {"growth_events": 1}})
    m.pager.close()

    m2, opt2, _ = _tiny_model()
    load_model(d, m2, opt2)
    # Greedy generation is deterministic and decodes text.
    a = generate(m2, [104, 105], max_new=6, temperature=0.0, seed=0)
    b = generate(m2, [104, 105], max_new=6, temperature=0.0, seed=0)
    assert a["ids"] == b["ids"] and a["text"] == b["text"]
    assert len(a["ids"]) == 8
    assert set(a["paging"]) >= {"disk_reads", "ram_hits"}
    # Recurrent streaming state: stepwise == full causal pass.
    ids = torch.tensor([[10, 20, 30, 40]])
    full, _ = m2.forward_infer_stateful(ids)
    states = m2.new_infer_state(1)
    parts = []
    for tok in ids[0]:
        out, states = m2.forward_infer_step(tok.view(1), states)
        parts.append(out["logits"])
    streamed = torch.cat(parts, dim=1)
    assert torch.allclose(full["logits"], streamed, atol=2e-5, rtol=2e-5)
    # Eval loss helper runs on the resumed model.
    rep = evaluate_loss(m2, batch_from_seqs([[10, 20, 30]], m2.cfg.context_length),
                        m2.cfg.context_length)
    assert set(rep) >= {"loss", "acc"}
    m2.pager.close()


def test_e2e_paging_and_param_accounting_via_public_apis():
    """Pager paths, resident bounds, and unambiguous param categories."""
    m, _, _ = _tiny_model()
    for eid in m.pool.order:
        m.pager.provider(eid)
    assert len(m.pager.ram) == len(m.pool)
    stats = m.pager.stats.to_dict()
    assert stats["disk_reads"] >= len(m.pool)
    counts = m.param_counts()
    assert counts["logical_params"] == counts["total_params"] if "logical_params" in counts else True
    assert counts["total_params"] == counts["shared_params"] + counts["router_params"] + counts["expert_params_total"]
    assert counts["stored_expert_bytes"] > 0
    resident = m.pager.resident_counts()
    assert set(resident) >= {"ram", "vram"}
    # Config describes the same logical categories without a live model.
    desc = m.cfg.describe_counts()
    assert desc["logical_params"] == desc["total_params"] == counts["total_params"]
    assert desc["resident_ram_params"] is None
    m.pager.close()


def test_e2e_cli_config_and_schedule():
    """CLI maps every knob; train schedule falls back to config."""
    from cli import build_parser, config_from_args, resolve_train_grow_every

    args = build_parser().parse_args(["train"])
    cfg = config_from_args(args)
    assert cfg.d_model == 64 and cfg.max_depth == 3  # tiny preset
    full = config_from_args(build_parser().parse_args(["--full", "--max-depth", "3", "train"]))
    assert full.d_model == 512 and full.max_depth == 3
    aliased = config_from_args(build_parser().parse_args(
        ["--paging-method", "r2vr", "--attention-chunk-size", "64", "train"]))
    assert aliased.paging_method == "R2VR" and aliased.attention_chunk_size == 64
    assert resolve_train_grow_every(build_parser().parse_args(["train"]), cfg) == cfg.grow_every == 200
    assert resolve_train_grow_every(build_parser().parse_args(["train", "--grow-every", "0"]), cfg) == 0


def test_e2e_growth_pruning_direct_apis():
    """Direct growth clone + prune removal roundtrip on a tiny pool."""
    import growth as growth_mod
    import pruning as pruning_mod
    from experts import ExpertPool, make_expert
    from routing import SparseRouter

    torch.manual_seed(0)
    pool = ExpertPool()
    for _ in range(3):
        pool.add(make_expert(pool.fresh_id(), 16, 32, birth_step=0))
    router = SparseRouter(16, 3, 1)
    new_ids = growth_mod.grow_topk_clones(
        pool, router, 16, 32, step=4, seed=0, k=2, n_mutated=1, max_experts=8)
    assert len(new_ids) == 2 and len(pool) == 5 and router.num_experts == 5
    # Age the pool and keep one expert useful; the rest are pruneable.
    keep = pool.order[0]
    rec = pool.experts[keep]
    rec.tokens_routed = 500
    rec.grad_activity = 1e-3
    rec.contribution = 0.5
    rec.last_used_step = 600
    victims = pruning_mod.find_victims(pool, step=600, survival_steps=500, min_experts=1)
    assert keep not in victims and len(victims) >= 1
    pruned = pruning_mod.prune_experts(pool, router, victims[:1])
    assert len(pruned) == 1 and len(pool) == 4 and router.num_experts == 4


def test_e2e_quantize_retile_roundtrip(tmp_path):
    """FP8 requantize/retile preserves topology and reloads cleanly."""
    import json
    from quantize import convert_checkpoint

    d = str(tmp_path / "e2e_q")
    m, opt = _tiny_model()[:2]
    run_training(m, opt, m.cfg, _demo_seqs(), steps=2, batch_size=2,
                 grow_every=2, grow_loss_below=-1.0, seed=5,
                 log_fn=lambda s: None)
    save_model(d, m, opt, step=1)
    n_experts = len(m.pool)
    reports = convert_checkpoint(d, to="fp8", tile=32)
    assert len(reports) == n_experts and all(r["to"] == "fp8" for r in reports)
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["precision"] == {"format": "fp8", "fp8_tile": 32}
    assert json.load(open(os.path.join(d, "config.json")))["fp8_tile"] == 32
    assert not [f for f in os.listdir(os.path.join(d, "experts"))
                if f.endswith(".convert_tmp")]
    m2, opt2, _ = _tiny_model()
    load_model(d, m2, opt2)
    assert m2.cfg.fp8_tile == 32 and len(m2.pool) == n_experts
    ids = torch.randint(0, 256, (1, 8))
    assert m2.forward_infer(ids)["logits"].shape == (1, 8, m2.cfg.vocab_size)
    m.pager.close()
    m2.pager.close()


def test_e2e_corrupted_checkpoint_refusal(tmp_path):
    """Corrupt one file; load raises and the live model is untouched."""
    d = str(tmp_path / "e2e_bad")
    m, opt = _tiny_model()[:2]
    run_training(m, opt, m.cfg, _demo_seqs(), steps=1, batch_size=2,
                 log_fn=lambda s: None)
    save_model(d, m, opt, step=0)

    def _assert_untouched(m2, order_before, embed_before):
        assert list(m2.pool.order) == order_before
        assert torch.equal(m2.embed.weight, embed_before)
        assert m2.router.num_experts == len(order_before)
        ids = torch.randint(0, 256, (1, 8))
        assert m2.forward_infer(ids)["logits"].shape == (1, 8, m2.cfg.vocab_size)

    # Corrupt expert bytes.
    p = os.path.join(d, "experts", m.pool.order[0] + ".pt")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    payload["weights"]["w_gate"]["codes"] = payload["weights"]["w_gate"]["codes"][:-1]
    torch.save(payload, p)
    m2, opt2, _ = _tiny_model()
    order_before = list(m2.pool.order)
    embed_before = m2.embed.weight.detach().clone()
    with pytest.raises((ValueError, RuntimeError)):
        load_model(d, m2, opt2)
    _assert_untouched(m2, order_before, embed_before)
    m2.pager.close()
    m.pager.close()


def test_e2e_native_fallback_observable(monkeypatch):
    """Native off-mode falls back loudly; reference math still runs."""
    import native
    from precision import dequantize_fp8_blockwise, quantize_fp8_blockwise

    monkeypatch.setenv("SMAUL_NATIVE", "0")
    native.reset_counters()
    w = torch.randn(4, 64) * 0.5
    t = quantize_fp8_blockwise(w, tile=32)
    r = dequantize_fp8_blockwise(t)
    assert r.shape == (4, 64)
    assert native.COUNTERS["fp8_quant_native"] == 0
    assert native.COUNTERS["fp8_quant_fallback"] >= 1
    s = native.status()
    assert s["mode"] == "off" and s["required_isa"] == native.required_isa()
    assert "-march=x86-64" in s["build_flags"]
    m, _, _ = _tiny_model()
    ids = torch.randint(0, 256, (1, 8))
    assert m.forward_infer(ids)["logits"].shape == (1, 8, m.cfg.vocab_size)
    m.pager.close()
    native.reset_counters()
