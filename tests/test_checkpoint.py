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


def test_schema_major_mismatch_refuses(tmp_path):
    import json
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    p = os.path.join(d, "config.json")
    cfg = json.load(open(p))
    cfg["schema_version"] = "1.0.0"
    json.dump(cfg, open(p, "w"))
    m2 = SmaulBrainModel(m.cfg)
    with pytest.raises(ValueError):
        load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    m.pager.close(); m2.pager.close()


def test_config_rejects_nonsense_values():
    bad = dict(grad_clip=-1.0, capacity_factor=0.0, expert_lr=0.0,
               beta_m=1.5, fp8_tile=4, context_length=1, max_new_experts=0)
    for k, v in bad.items():
        with pytest.raises(AssertionError):
            SmaulBrainConfig(**{k: v})


def _tampered_load_fails_cleanly(tmp_path, tamper):
    """Corrupt one checkpoint file; load must raise with m2 bit-untouched."""
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    tamper(d)
    m2 = SmaulBrainModel(m.cfg)
    order_before = list(m2.pool.order)
    embed_before = m2.embed.weight.detach().clone()
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        load_model(d, m2, SmaulOpt(SmaulOptHParams()))
    assert list(m2.pool.order) == order_before  # pool never swapped
    assert torch.equal(m2.embed.weight, embed_before)  # trunk never copied
    assert m2.router.num_experts == len(order_before)  # router never resized
    ids = torch.randint(0, 256, (1, 12))  # survivor still runs
    assert m2.forward_infer(ids)["logits"].shape == (1, 12, m2.cfg.vocab_size)
    m.pager.close(); m2.pager.close()


def test_corrupt_expert_file_leaves_model_untouched(tmp_path):
    def tamper(d):
        import torch as _t
        p = os.path.join(d, "experts", "expert_00000.pt")
        payload = _t.load(p, map_location="cpu", weights_only=False)
        payload["weights"]["w_gate"]["codes"] = payload["weights"]["w_gate"]["codes"][:-1]
        _t.save(payload, p)
    _tampered_load_fails_cleanly(tmp_path, tamper)


def test_missing_expert_file_leaves_model_untouched(tmp_path):
    def tamper(d):
        os.remove(os.path.join(d, "experts", "expert_00001.pt"))
    _tampered_load_fails_cleanly(tmp_path, tamper)


def test_router_width_mismatch_leaves_model_untouched(tmp_path):
    def tamper(d):
        import torch as _t
        p = os.path.join(d, "router.pt")
        obj = _t.load(p, map_location="cpu", weights_only=False)
        obj["weight"] = obj["weight"][:2]
        obj["bias"] = obj["bias"][:2]
        _t.save(obj, p)
    _tampered_load_fails_cleanly(tmp_path, tamper)


def test_trunk_shape_mismatch_leaves_model_untouched(tmp_path):
    def tamper(d):
        import torch as _t
        p = os.path.join(d, "trunk.pt")
        obj = _t.load(p, map_location="cpu", weights_only=False)
        k = next(k for k in obj if isinstance(obj[k], _t.Tensor) and obj[k].ndim == 2)
        obj[k] = obj[k][:, :-1]
        _t.save(obj, p)
    _tampered_load_fails_cleanly(tmp_path, tamper)


def test_duplicate_manifest_ids_leaves_model_untouched(tmp_path):
    def tamper(d):
        import json as _j
        p = os.path.join(d, "manifest.json")
        man = _j.load(open(p))
        man["expert_ids"] = [man["expert_ids"][0], man["expert_ids"][0]]
        _j.dump(man, open(p, "w"))
    _tampered_load_fails_cleanly(tmp_path, tamper)


def test_garbage_rng_snapshot_leaves_model_untouched(tmp_path):
    def tamper(d):
        import torch as _t
        _t.save({"bogus": 1}, os.path.join(d, "rng.pt"))
    _tampered_load_fails_cleanly(tmp_path, tamper)


def test_stale_tmp_swept_on_save(tmp_path):
    """Crash leftovers (tmp_ckpt_/tmp_json_/.convert_tmp) are swept on save."""
    import storage as _st
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    exp_dir = os.path.join(d, "experts")
    # Plant stale sidecars imitating a crash mid-rename.
    for p in [os.path.join(d, "tmp_ckpt_crash"),
              os.path.join(d, "tmp_json_crash"),
              os.path.join(exp_dir, "tmp_ckpt_crash"),
              os.path.join(exp_dir, "stale.convert_tmp")]:
        with open(p, "w") as f:
            f.write("stale")
    save_model(d, m, opt, step=10)
    assert _st.load_manifest(d)["step"] == 10
    for root, _, files in os.walk(d):
        assert not [f for f in files if f.startswith(("tmp_ckpt_", "tmp_json_"))]
        assert not [f for f in files if f.endswith(".convert_tmp")]
    m.pager.close()


def test_truncated_expert_detected_as_validation_error(tmp_path):
    """Empty/truncated files raise ValueError (not raw EOF) with model untouched."""
    import storage as _st
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    eid = m.pool.order[0]
    p = os.path.join(d, "experts", f"{eid}.pt")
    with open(p, "wb") as f:
        f.truncate(0)
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    before = list(m2.pool.order)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    assert list(m2.pool.order) == before
    with pytest.raises(ValueError, match="validation failed"):
        _st.load_expert_file(p)
    # Truncated (non-empty) payload is also normalized to ValueError.
    with open(p, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="validation failed"):
        _st.load_expert_file(p)
    m.pager.close(); m2.pager.close()


def test_atomic_writes_fsync_before_publish(tmp_path, monkeypatch):
    """fsync ordering: file flushed before rename, dir flushed after."""
    import storage as _st
    calls = []
    real_fsync = os.fsync
    real_replace = os.replace

    def _recording_fsync(fd):
        calls.append(("fsync", fd))
        return real_fsync(fd)

    def _recording_replace(src, dst):
        calls.append(("replace", src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", _recording_fsync)
    monkeypatch.setattr(os, "replace", _recording_replace)
    d = str(tmp_path / "c")
    m, opt = _trained(d)
    # At least one file-fsync must precede a rename, and a dir-fsync follows.
    kinds = [c[0] for c in calls]
    assert "fsync" in kinds and "replace" in kinds
    first_replace = kinds.index("replace")
    assert "fsync" in kinds[:first_replace]
    m.pager.close()


def test_nested_dir_creation_is_race_safe(tmp_path):
    """Single-expert saves create missing parents; existing dirs are reused."""
    from storage import load_expert_file, save_expert_file
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    eid = m.pool.order[0]
    rec = m.pool.experts[eid]
    nested = os.path.join(d, "experts", "nested", "deep", f"{eid}.pt")
    save_expert_file(rec, nested)  # parents did not exist
    assert load_expert_file(nested).expert_id == eid
    save_expert_file(rec, nested)  # second save over existing dirs
    assert load_expert_file(nested).expert_id == eid
    m.pager.close()


def test_quantize_sweeps_stale_sidecars_and_refuses_empty(tmp_path):
    """Converter drops prior-crash sidecars; empty inputs fail loudly."""
    from quantize import convert_checkpoint
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    exp_dir = os.path.join(d, "experts")
    for p in [os.path.join(d, "tmp_quant_crash"),
              os.path.join(exp_dir, "stale.convert_tmp")]:
        with open(p, "w") as f:
            f.write("stale")
    reports = convert_checkpoint(d, to="fp8", tile=32)
    assert reports
    assert not [f for f in os.listdir(d) if f.startswith("tmp_quant_")]
    assert not [f for f in os.listdir(exp_dir) if f.endswith(".convert_tmp")]
    eid = m.pool.order[0]
    with open(os.path.join(exp_dir, f"{eid}.pt"), "wb") as f:
        f.truncate(0)
    with pytest.raises(ValueError, match="validation failed|empty"):
        convert_checkpoint(d, to="fp8", tile=32)
    m.pager.close()


def _snapshot_live(m, opt):
    """Full live-state snapshot: pool/router/trunk/pager/optimizer survivors."""
    import copy
    snap = {}
    snap["order"] = list(m.pool.order)
    snap["embed"] = m.embed.weight.detach().clone()
    snap["router_rows"] = m.router.num_experts
    snap["router_weight"] = m.router.proj.weight.detach().clone()
    snap["trunk_state"] = copy.deepcopy(opt.trunk_state)
    snap["router_state"] = copy.deepcopy(opt.router_state)
    snap["step_count"] = opt.step_count
    snap["pager_ram"] = set(m.pager.ram.keys())
    snap["pager_vram"] = set(m.pager.vram.keys())
    snap["pager_records"] = set(getattr(m.pager, "ram_records", {}).keys())
    return snap


def _assert_live_untouched(m, opt, snap):
    assert list(m.pool.order) == snap["order"]
    assert torch.equal(m.embed.weight, snap["embed"])
    assert m.router.num_experts == snap["router_rows"]
    assert torch.equal(m.router.proj.weight, snap["router_weight"])
    assert opt.trunk_state.keys() == snap["trunk_state"].keys()
    for k in snap["trunk_state"]:
        for sk in snap["trunk_state"][k]:
            a, b = snap["trunk_state"][k][sk], opt.trunk_state[k][sk]
            assert torch.equal(a, b) if torch.is_tensor(a) else a == b
    assert opt.router_state.keys() == snap["router_state"].keys()
    assert opt.step_count == snap["step_count"]
    assert set(m.pager.ram.keys()) == snap["pager_ram"]
    assert set(m.pager.vram.keys()) == snap["pager_vram"]
    assert set(getattr(m.pager, "ram_records", {}).keys()) == snap["pager_records"]
    ids = torch.randint(0, 256, (1, 12))
    assert m.forward_infer(ids)["logits"].shape == (1, 12, m.cfg.vocab_size)


def test_extra_expert_file_rejected_untouched(tmp_path):
    """Unlisted expert file on disk fails validation; live model untouched."""
    import shutil
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    src = os.path.join(d, "experts", m.pool.order[0] + ".pt")
    shutil.copy(src, os.path.join(d, "experts", "expert_99999.pt"))
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_invalid_config_json_type_rejected_untouched(tmp_path):
    """Non-numeric config value normalizes to ValueError; nothing mutates."""
    import json as _j
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    p = os.path.join(d, "config.json")
    cfg = _j.load(open(p))
    cfg["d_model"] = "not-a-number"
    _j.dump(cfg, open(p, "w"))
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_invalid_manifest_json_type_rejected_untouched(tmp_path):
    """String manifest step is an invalid JSON type; load fails cleanly."""
    import json as _j
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    p = os.path.join(d, "manifest.json")
    man = _j.load(open(p))
    man["step"] = "three"
    _j.dump(man, open(p, "w"))
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_incompatible_topology_rejected_untouched(tmp_path):
    """Saved expert_hidden != live topology fails before any mutation."""
    import json as _j
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    p = os.path.join(d, "config.json")
    cfg = _j.load(open(p))
    cfg["expert_hidden"] = int(cfg["expert_hidden"]) + 1000
    _j.dump(cfg, open(p, "w"))
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_trunk_missing_entry_rejected_untouched(tmp_path):
    """Trunk missing one key must fail (no silent strict=False partial)."""
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    p = os.path.join(d, "trunk.pt")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    obj.pop(next(iter(obj)))
    torch.save(obj, p)
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_corrupt_optimizer_state_rejected_untouched(tmp_path):
    """Non-tensor optimizer moment fails deep validation; opt untouched."""
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    p = os.path.join(d, "optim.pt")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    k = next(iter(obj["trunk"]))
    obj["trunk"][k]["m"] = "corrupted"
    torch.save(obj, p)
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_corrupt_expert_optim_shape_rejected_untouched(tmp_path):
    """Expert-local moment with wrong shape fails; pool/pager/opt untouched."""
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    eid = m.pool.order[0]
    p = os.path.join(d, "experts", f"{eid}.pt")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    obj["optim_state"]["w_gate"]["m"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    torch.save(obj, p)
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()


def test_corrupt_expert_meta_type_rejected_untouched(tmp_path):
    """Non-numeric expert meta normalizes to ValueError; nothing mutates."""
    d = str(tmp_path / "c")
    m, _ = _trained(d)
    eid = m.pool.order[0]
    p = os.path.join(d, "experts", f"{eid}.pt")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    obj["meta"]["tokens_routed"] = {"x": 1}
    torch.save(obj, p)
    m2 = SmaulBrainModel(m.cfg)
    opt2 = SmaulOpt(SmaulOptHParams())
    snap = _snapshot_live(m2, opt2)
    with pytest.raises(ValueError, match="validation failed"):
        load_model(d, m2, opt2)
    _assert_live_untouched(m2, opt2, snap)
    m.pager.close(); m2.pager.close()
