"""Continual learning: retention measured numerically, replay, slow trunk."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import (
    ReplayBuffer, batch_from_seqs, evaluate_loss, retention_report,
    run_training,
)


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=2, context_length=24,
                           expert_lr=3e-2, trunk_lr_mult=0.1)
    m = SmaulBrainModel(cfg)
    return m, cfg


def test_replay_buffer_reservoir_and_sample():
    buf = ReplayBuffer(capacity=8, seed=0)
    for i in range(20):
        buf.add([i])
    assert len(buf) == 8 and buf.seen == 20
    assert len(buf.sample(3)) == 3


def test_replay_buffer_ingress_copy_and_egress_copy():
    buf = ReplayBuffer(capacity=8, seed=0)
    seq = [1, 2, 3]
    buf.add(seq)
    seq.append(999)  # caller mutation after add must not corrupt replay
    seq2 = [4, 5]
    buf.add(seq2)
    got = buf.to_dict()["buf"]
    assert [1, 2, 3] in got and [4, 5] in got
    out = buf.sample(2)
    out[0].append(-1)
    assert buf.to_dict()["buf"] == got  # sampling hands out copies too


def test_retention_report_numeric_and_honest():
    before = {"loss": 2.0, "acc": 0.5}
    after = {"loss": 2.2, "acc": 0.45}
    rep = retention_report(before, after)
    assert abs(rep["old_loss_delta"] - 0.2) < 1e-9
    assert rep["old_acc_delta"] < 0  # forgetting is reported, not hidden
    assert "retained" in rep


def test_slow_trunk_moves_slower_than_experts():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    assert cfg.trunk_lr < cfg.expert_lr  # 0.1x by default
    assert abs(cfg.trunk_lr - cfg.expert_lr * 0.1) < 1e-12
    m.pager.close()


def test_continual_run_reports_retention():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    old = [[i % 256 for i in range(j, j + 30)] for j in range(8)]
    new = [[(i * 7 + 3) % 256 for i in range(j, j + 30)] for j in range(16)]
    res = run_training(m, opt, cfg, new, steps=8, batch_size=2,
                       old_seqs=old, replay=ReplayBuffer(64), replay_n=1)
    rep = res["retention"]
    assert rep is not None and "old_loss_before" in rep and "old_loss_after" in rep
    assert isinstance(rep["old_loss_delta"], float)
    m.pager.close()


def test_batch_packing_truncates_and_pads():
    b = batch_from_seqs([[1, 2, 3], [4] * 40], context=8, pad_id=0)
    assert b.shape == (2, 9)
    assert b[0, 3:].tolist() == [0] * 6
    assert b[1, :9].tolist() == [4] * 9


def test_replay_sample_semantics_clipped_with_replacement():
    # Defined semantics: with replacement, clipped to len(buf), n<=0 -> [].
    buf = ReplayBuffer(capacity=8, seed=0)
    assert buf.sample(3) == []  # empty buffer
    assert buf.sample(0) == [] and buf.sample(-1) == []
    buf.add([1])
    buf.add([2])
    assert buf.sample(0) == [] and buf.sample(-2) == []
    assert len(buf.sample(5)) == 2  # clipped, not padded to n
    assert len(buf.sample(1)) == 1
    # With replacement: repeated draws of 2 from {1,2} must sometimes repeat.
    draws = [tuple(x[0] for x in buf.sample(2)) for _ in range(20)]
    assert any(a == b for a, b in draws)  # seed=0 gives 9/20 dups
    # Egress copies: sampling never hands out live references.
    before = buf.to_dict()["buf"]
    out = buf.sample(2)
    out[0].append(999)
    assert buf.to_dict()["buf"] == before


def test_replay_persistence_exact_continuation():
    buf = ReplayBuffer(capacity=4, seed=7)
    for i in range(6):
        buf.add([i, i + 1])
    snap = buf.to_dict()
    assert set(snap) == {"capacity", "buf", "seen", "rng"}
    assert snap["seen"] == 6 and snap["capacity"] == 4
    clone = ReplayBuffer.from_dict(snap)
    assert clone.to_dict() == snap  # bit-identical snapshot roundtrip
    assert [clone.sample(2) for _ in range(5)] == [buf.sample(2) for _ in range(5)]
    # Mutating the snapshot dict must not alias either buffer.
    snap["buf"].append([999])
    assert len(buf) == 4 and len(clone) == 4


def test_replay_ingress_covers_full_batch_past_only():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    seqs = [[i] * 30 for i in range(8)]  # distinct markers per position
    buf = ReplayBuffer(capacity=16, seed=3)
    res = run_training(m, opt, cfg, seqs, steps=2, batch_size=2,
                       replay=buf, replay_n=1, log_fn=lambda s: None)
    assert res["steps_completed"] == 2
    # Full-batch ingress: steps 0,1 consume fresh idx [0,1],[2,3] -> all 4 stored.
    stored = [s[0] for s in buf.to_dict()["buf"]]
    assert sorted(stored) == [0, 1, 2, 3], stored
    assert buf.seen == 4  # 2 fresh seqs per step, not 1 per step
    m.pager.close()


def test_scheduler_snapshot_persists_dataset_cursors():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    seqs = [[(i + j) % 256 for j in range(30)] for i in range(8)]
    buf = ReplayBuffer(capacity=8, seed=7)
    res = run_training(m, opt, cfg, seqs, steps=3, batch_size=2,
                       replay=buf, replay_n=1, log_fn=lambda s: None)
    sched = res["scheduler"]
    assert sched["batch_size"] == 2 and sched["dataset_len"] == 8
    assert sched["next_step"] == 3  # start_step(0) + 3 steps
    assert isinstance(sched["replay"], dict) and sched["replay"]["seen"] == 6
    assert "prev_loss" in sched and "growth_events" in sched
    # Resume continues the dataset order: next batch starts at global step 3.
    assert [h["step"] for h in res["history"]] == [0, 1, 2]
    m.pager.close()


def test_throughput_and_loss_scale_by_valid_not_padded():
    from bytes import PAD_ID
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    # Each seq has 5 data ids; context=24 -> 19 pads per row in the input half.
    seqs = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
    res = run_training(m, opt, cfg, seqs, steps=1, batch_size=2,
                       log_fn=lambda s: None)
    assert res["bytes_processed"] == 10  # 5 valid x 2 rows, not 2*24=48
    b = batch_from_seqs(seqs, cfg.context_length)
    assert int((b[:, : cfg.context_length] != PAD_ID).sum().item()) == 10
    # Padding is masked from the loss: all-pad targets score 0, never NaN.
    rep = evaluate_loss(m, batch_from_seqs([[1, 2, 3]], cfg.context_length),
                        cfg.context_length)
    assert rep["loss"] == rep["loss"] and 0.0 <= rep["acc"] <= 1.0
    pad_only = torch.full((1, cfg.context_length + 1), PAD_ID, dtype=torch.long)
    rep_pad = evaluate_loss(m, pad_only, cfg.context_length)
    assert rep_pad["loss"] == 0.0  # no valid targets -> 0/1, not NaN/diluted
    m.pager.close()


def test_stepped_experts_follow_pool_order():
    m, cfg = _model(seed=1)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    seqs = [[(i * 3 + j) % 256 for j in range(30)] for i in range(8)]
    res = run_training(m, opt, cfg, seqs, steps=2, batch_size=2,
                       log_fn=lambda s: None)
    order = list(m.pool.order)
    rank = {eid: i for i, eid in enumerate(order)}
    for h in res["history"]:
        stepped = h["stepped_experts"]
        assert stepped == sorted(stepped, key=lambda e: (rank.get(e, 1 << 30), e))
    m.pager.close()


def _seqs(n=8):
    return [[(i * 3 + j) % 256 for j in range(30)] for i in range(n)]


def test_scheduler_global_step_trigger_boundaries():
    # Exact semantics: scheduled growth fires when (global_step+1) % grow_every == 0.
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    res = run_training(m, opt, cfg, _seqs(), steps=4, batch_size=2,
                       grow_every=2, grow_loss_below=None,
                       log_fn=lambda s: None)
    fired = [h["step"] for h in res["history"] if h["grew"]]
    assert fired == [1, 3]  # global steps 1 and 3, not local 0..3 pattern shift
    m.pager.close()


def test_scheduler_snapshot_persists_cadence_and_seed():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    res = run_training(m, opt, cfg, _seqs(), steps=2, batch_size=2,
                       grow_every=3, prune_every=5, grow_loss_below=0.5,
                       growths_per_prune=4, seed=11, log_fn=lambda s: None)
    sched = res["scheduler"]
    assert sched["growths_per_prune"] == 4
    assert sched["grow_every"] == 3 and sched["prune_every"] == 5
    assert sched["grow_loss_below"] == 0.5 and sched["seed"] == 11
    assert sched["growth_events"] == res["growth_events"]
    m.pager.close()


def test_loss_edge_first_step_never_spurious():
    # Fresh run: prev_loss is None so the first step cannot loss-fire,
    # even with an enormous threshold.
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    res = run_training(m, opt, cfg, _seqs(), steps=1, batch_size=2,
                       grow_every=0, grow_loss_below=1e9,
                       log_fn=lambda s: None)
    assert res["history"][0]["grew"] is False
    assert res["history"][0]["new_experts"] == []
    # Snapshot carries the edge memory for resume (no spurious re-fire).
    assert res["scheduler"]["prev_loss"] == res["history"][0]["loss"]
    m.pager.close()


def test_trigger_rng_streams_deterministic():
    import torch
    from precision import dequantize_fp8_blockwise

    def _run(seed):
        from config import SmaulBrainConfig
        from model import SmaulBrainModel
        torch.manual_seed(0)
        cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                               expert_hidden=64, max_depth=2, context_length=24)
        m = SmaulBrainModel(cfg)
        opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
        res = run_training(m, opt, cfg, _seqs(), steps=1, batch_size=2,
                           grow_every=1, grow_loss_below=None, seed=seed,
                           log_fn=lambda s: None)
        ids = list(res["history"][0]["new_experts"])
        sums = [float(dequantize_fp8_blockwise(
            m.pool.experts[e].weights_fp8["w_gate"]).sum().item()) for e in ids]
        parents = [tuple(m.pool.experts[e].parents) for e in ids]
        m.pager.close()
        return ids, parents, sums
    ids_a, par_a, sums_a = _run(9)
    ids_b, par_b, sums_b = _run(9)
    ids_c, par_c, sums_c = _run(10)
    assert ids_a == ids_b and par_a == par_b and sums_a == sums_b
    # Ids/parents derive from deterministic contribution ranking (seed-free);
    # the seeded stream controls mutation/noise, so weights must re-base.
    assert par_a == par_c and sums_a != sums_c


def test_prune_triggers_coalesce_to_one_eval_per_step():
    import train as train_mod
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    calls = []
    orig = train_mod.pruning_mod.find_victims

    def counting(*a, **k):
        calls.append(1)
        return []
    train_mod.pruning_mod.find_victims = counting
    try:
        run_training(m, opt, cfg, _seqs(), steps=3, batch_size=2,
                     grow_every=1, grow_loss_below=None,
                     growths_per_prune=1, prune_every=1, seed=0,
                     log_fn=lambda s: None)
    finally:
        train_mod.pruning_mod.find_victims = orig
    # Cadence (every growth) + schedule (every step) coincide: still 1/step.
    assert len(calls) == 3
    m.pager.close()


def test_trigger_while_at_cap_does_not_advance_cadence():
    m, cfg = _model()
    cfg.max_experts = len(m.pool)  # at cap before step 0
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    res = run_training(m, opt, cfg, _seqs(), steps=2, batch_size=2,
                       grow_every=1, grow_loss_below=None,
                       growths_per_prune=1, seed=0, log_fn=lambda s: None)
    assert res["scheduler"]["growth_events"] == 0
    assert res["growth_events"] == 0
    assert all(h["grew"] is False and h["new_experts"] == [] for h in res["history"])
    assert len(m.pool) == cfg.max_experts
    m.pager.close()


def test_machine_readable_growth_prune_ledger():
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    res = run_training(m, opt, cfg, _seqs(), steps=3, batch_size=2,
                       grow_every=1, grow_loss_below=None, seed=0,
                       log_fn=lambda s: None)
    for h in res["history"]:
        assert isinstance(h["grew"], bool)
        assert isinstance(h["new_experts"], list) and isinstance(h["pruned_experts"], list)
        assert h["grew"] == bool(h["new_experts"])
    evts = res["growth_prune_events"]
    assert all(set(e) == {"step", "type", "ids"} for e in evts)
    assert all(e["type"] in ("grow", "prune") for e in evts)
    # Same-step ordering: grow entries precede prune entries.
    for step in {e["step"] for e in evts}:
        kinds = [e["type"] for e in evts if e["step"] == step]
        assert kinds == sorted(kinds, key=lambda t: 0 if t == "grow" else 1)
    # Ledger matches per-step history.
    flat_grow = [eid for h in res["history"] for eid in h["new_experts"]]
    assert flat_grow == [eid for e in evts if e["type"] == "grow" for eid in e["ids"]]
    m.pager.close()


def test_parent_victim_selection_deterministic():
    from growth import select_parents
    from pruning import find_victims
    m, cfg = _model()
    try:
        assert select_parents(m.pool, k=2) == select_parents(m.pool, k=2)
        v1 = find_victims(m.pool, step=10_000, survival_steps=1,
                          min_experts=cfg.min_experts)
        v2 = find_victims(m.pool, step=10_000, survival_steps=1,
                          min_experts=cfg.min_experts)
        assert v1 == v2
        # min_experts floor: never proposes more than len - floor.
        assert len(v1) <= max(0, len(m.pool) - cfg.min_experts)
    finally:
        m.pager.close()


def test_resume_before_trigger_matches_uninterrupted():
    # Resuming immediately before a scheduled trigger must make the same
    # topology decision as running straight through (global-step cadence).
    from storage import save_model, load_model
    import tempfile
    seqs = _seqs()
    kw = dict(batch_size=2, grow_every=2, grow_loss_below=None,
              growths_per_prune=2, seed=0, log_fn=lambda s: None)
    m1, cfg1 = _model()
    o1 = SmaulOpt(SmaulOptHParams(lr=cfg1.expert_lr))
    full = run_training(m1, o1, cfg1, seqs, steps=4, **kw)
    full_ids = [list(h["new_experts"]) for h in full["history"]]
    m1.pager.close()
    # Split run: 1 step, checkpoint, resume for 3 more.
    m2, cfg2 = _model()
    o2 = SmaulOpt(SmaulOptHParams(lr=cfg2.expert_lr))
    part1 = run_training(m2, o2, cfg2, seqs, steps=1, **kw)
    assert [h["step"] for h in part1["history"]] == [0]
    with tempfile.TemporaryDirectory() as d:
        save_model(d, m2, o2, 0, extra_meta={"scheduler": part1["scheduler"]})
        m3, cfg3 = _model()
        o3 = SmaulOpt(SmaulOptHParams(lr=cfg3.expert_lr))
        load_model(d, m3, o3)
        cfg3 = m3.cfg
        part2 = run_training(m3, o3, cfg3, seqs, steps=3, **kw)
        resumed_ids = [list(h["new_experts"]) for h in part1["history"]] + \
            [list(h["new_experts"]) for h in part2["history"]]
        assert [h["step"] for h in part2["history"]] == [1, 2, 3]
        m2.pager.close()
        m3.pager.close()
    assert resumed_ids == full_ids


def test_scheduler_snapshot_pins_rng_cadence_knobs():
    # RNG stream splits + batch composition must ride along: replay_n pins
    # the interleaving width, max_new_experts pins the growth salt.
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    res = run_training(m, opt, cfg, _seqs(), steps=1, batch_size=2,
                       replay=ReplayBuffer(8, seed=7), replay_n=1,
                       grow_every=2, grow_loss_below=None, seed=5,
                       log_fn=lambda s: None)
    sched = res["scheduler"]
    assert sched["replay_n"] == 1
    assert sched["max_new_experts"] == int(cfg.max_new_experts)
    assert sched["seed"] == 5
    m.pager.close()


def test_global_rng_streams_isolated_and_restored():
    # Growth uses isolated Generators, replay an owned Random: dirtying the
    # global torch/python RNG around save/resume must not perturb the
    # trajectory, while rng.pt itself still restores the global states.
    import random
    from storage import save_model, load_model
    import tempfile
    seqs = _seqs()
    kw = dict(batch_size=2, replay_n=1, grow_every=2,
              grow_loss_below=-1.0, seed=5, log_fn=lambda s: None)
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    ref = run_training(m, opt, cfg, seqs, steps=4,
                       replay=ReplayBuffer(capacity=8, seed=7), **kw)
    ref_losses = [h["loss"] for h in ref["history"]]
    ref_embed = m.embed.weight.detach().clone()
    m.pager.close()
    m1, cfg1 = _model()
    o1 = SmaulOpt(SmaulOptHParams(lr=cfg1.expert_lr))
    leg1 = run_training(m1, o1, cfg1, seqs, steps=2,
                        replay=ReplayBuffer(capacity=8, seed=7), **kw)
    torch_cpu_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    with tempfile.TemporaryDirectory() as d:
        save_model(d, m1, o1, 1, extra_meta={"scheduler": leg1["scheduler"]})
        torch.randn(64)  # dirty globals between save and load
        [random.random() for _ in range(64)]
        m2, cfg2 = _model()
        o2 = SmaulOpt(SmaulOptHParams(lr=cfg2.expert_lr))
        torch.randn(32)
        load_model(d, m2, o2)
        # rng.pt restore: global states return to the checkpoint values.
        assert torch.equal(torch.get_rng_state(), torch_cpu_before)
        assert random.getstate()[1] == python_before[1]
        cfg2 = m2.cfg
        torch.randn(64)  # dirty again after load: training must not notice
        [random.random() for _ in range(64)]
        leg2 = run_training(m2, o2, cfg2, seqs, steps=2, batch_size=2,
                            replay_n=1, grow_every=2, grow_loss_below=-1.0,
                            seed=5, log_fn=lambda s: None)
        got = [h["loss"] for h in leg1["history"]] + \
            [h["loss"] for h in leg2["history"]]
        assert got == ref_losses
        assert torch.equal(m2.embed.weight, ref_embed)
        m1.pager.close()
        m2.pager.close()


def test_loss_edge_resume_matches_uninterrupted():
    # The falling-edge memory (prev_loss) crosses the checkpoint boundary:
    # a split run must fire the same loss-triggered growth as straight-through.
    from storage import save_model, load_model
    import tempfile
    seqs = _seqs()
    m0, cfg0 = _model()
    o0 = SmaulOpt(SmaulOptHParams(lr=cfg0.expert_lr))
    probe = run_training(m0, o0, cfg0, seqs, steps=3, batch_size=2,
                         grow_every=0, grow_loss_below=None,
                         log_fn=lambda s: None)
    m0.pager.close()
    losses = [h["loss"] for h in probe["history"]]
    thr = (losses[0] + losses[1]) / 2.0  # fires exactly at step 1
    assert losses[0] >= thr > losses[1]
    kw = dict(batch_size=2, grow_every=0, grow_loss_below=thr,
              growths_per_prune=2, seed=0, log_fn=lambda s: None)
    m1, cfg1 = _model()
    o1 = SmaulOpt(SmaulOptHParams(lr=cfg1.expert_lr))
    full = run_training(m1, o1, cfg1, seqs, steps=3, **kw)
    m1.pager.close()
    m2, cfg2 = _model()
    o2 = SmaulOpt(SmaulOptHParams(lr=cfg2.expert_lr))
    leg1 = run_training(m2, o2, cfg2, seqs, steps=1, **kw)
    assert leg1["scheduler"]["prev_loss"] == leg1["history"][0]["loss"]
    with tempfile.TemporaryDirectory() as d:
        save_model(d, m2, o2, 0, extra_meta={"scheduler": leg1["scheduler"]})
        m3, cfg3 = _model()
        o3 = SmaulOpt(SmaulOptHParams(lr=cfg3.expert_lr))
        load_model(d, m3, o3)
        cfg3 = m3.cfg
        leg2 = run_training(m3, o3, cfg3, seqs, steps=2, **kw)
        resumed_grew = [h["grew"] for h in leg1["history"]] + \
            [h["grew"] for h in leg2["history"]]
        resumed_loss = [h["loss"] for h in leg1["history"]] + \
            [h["loss"] for h in leg2["history"]]
        assert resumed_grew == [h["grew"] for h in full["history"]]
        assert resumed_loss == [h["loss"] for h in full["history"]]
        m2.pager.close()
        m3.pager.close()


def test_resume_replay_rebase_rule():
    # Cursor re-basing rule: replay=None on resume continues the snapshot
    # trajectory (bit-equal); an explicit fresh buffer intentionally re-bases
    # (deterministic but divergent). Both halves of the rule are pinned here.
    from storage import save_model, load_model
    import tempfile
    seqs = _seqs()
    kw = dict(batch_size=2, replay_n=1, grow_every=0,
              grow_loss_below=-1.0, seed=5, log_fn=lambda s: None)
    m, cfg = _model()
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    ref = run_training(m, opt, cfg, seqs, steps=4,
                       replay=ReplayBuffer(capacity=8, seed=7), **kw)
    ref_losses = [h["loss"] for h in ref["history"]]
    m.pager.close()
    m1, cfg1 = _model()
    o1 = SmaulOpt(SmaulOptHParams(lr=cfg1.expert_lr))
    leg1 = run_training(m1, o1, cfg1, seqs, steps=2,
                        replay=ReplayBuffer(capacity=8, seed=7), **kw)
    with tempfile.TemporaryDirectory() as d:
        save_model(d, m1, o1, 1, extra_meta={"scheduler": leg1["scheduler"]})
        # Snapshot path: replay=None -> exact continuation.
        m2, cfg2 = _model()
        o2 = SmaulOpt(SmaulOptHParams(lr=cfg2.expert_lr))
        load_model(d, m2, o2)
        leg2 = run_training(m2, o2, m2.cfg, seqs, steps=2, **kw)
        got = [h["loss"] for h in leg1["history"]] + \
            [h["loss"] for h in leg2["history"]]
        assert got == ref_losses
        m2.pager.close()
        # Re-base path: explicit fresh buffer restarts the sampler.
        m3, cfg3 = _model()
        o3 = SmaulOpt(SmaulOptHParams(lr=cfg3.expert_lr))
        load_model(d, m3, o3)
        leg3 = run_training(m3, o3, m3.cfg, seqs, steps=2,
                            replay=ReplayBuffer(capacity=8, seed=7), **kw)
        rebased = [h["loss"] for h in leg1["history"]] + \
            [h["loss"] for h in leg3["history"]]
        assert rebased != ref_losses  # intentional re-base, never silent equal
        m3.pager.close()
        m1.pager.close()
