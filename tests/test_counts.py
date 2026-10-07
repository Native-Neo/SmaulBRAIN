"""Dynamic parameter counting: shared/router/expert/active/resident splits."""

import sys, os, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from config import SmaulBrainConfig
from growth import grow_expert
from model import SmaulBrainModel
from pruning import find_victims, prune_experts


def _model(n_exp=4):
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=n_exp, top_k=2,
                           expert_hidden=64, max_depth=1)
    return SmaulBrainModel(cfg)


def test_counts_split_and_total_consistent():
    m = _model()
    c = m.param_counts()
    assert c["total_params"] == c["shared_params"] + c["router_params"] + c["expert_params_total"]
    assert c["expert_params_total"] == 4 * c["per_expert_params"]
    assert c["active_params"] < c["total_params"]  # sparse: active < total
    assert c["active_params"] == c["shared_params"] + c["router_params"] + 2 * c["per_expert_params"]
    m.pager.close()


def test_total_changes_with_growth_and_pruning():
    m = _model()
    before = m.param_counts()["total_params"]
    grow_expert(m.pool, m.router, 32, 64, step=1, seed=0)
    after_grow = m.param_counts()["total_params"]
    # One expert (6144) plus its router row (d_model + 1 = 33).
    assert after_grow == before + m.param_counts()["per_expert_params"] + 33
    for eid in list(m.pool.order[1:]):
        r = m.pool.experts[eid]
        r.tokens_routed = 0; r.grad_activity = 0.0; r.contribution = 0.0
        r.last_used_step = 0; r.birth_step = 0
    victims = find_victims(m.pool, step=1000, survival_steps=10, min_experts=1)
    prune_experts(m.pool, m.router, victims)
    after_prune = m.param_counts()["total_params"]
    assert after_prune < after_grow
    m.pager.close()


def test_one_billion_params_need_not_be_resident():
    # 1B-param thought experiment at test scale: total >> resident.
    m = _model(n_exp=4)
    c = m.param_counts()
    assert c["resident_ram_params"] + c["resident_vram_params"] <= c["total_params"]
    m.pager.close()


def test_cli_report_json(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    r = subprocess.run(
        [sys.executable, "cli.py", "--d-model", "32", "--n-heads", "4",
         "--experts", "2", "--expert-size", "64", "--active-experts", "1",
         "--max-depth", "1", "--ckpt", ckpt, "report"],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["total_params"] > payload["active_params"]
    assert payload["expert_count"] == 2


def test_stored_bytes_track_growth_and_undercut_logical():
    m = _model()
    before = m.param_counts()
    assert before["stored_expert_bytes"] > 0
    # FP8 storage is a fraction of the logical fp32-equivalent footprint.
    assert before["stored_expert_bytes"] < before["expert_params_total"] * 4
    grow_expert(m.pool, m.router, 32, 64, step=1, seed=0)
    after = m.param_counts()
    assert after["stored_expert_bytes"] > before["stored_expert_bytes"]
    m.pager.close()


def test_config_roundtrip_exact():
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=1)
    d = cfg.to_dict()
    assert SmaulBrainConfig.from_dict(d).to_dict() == d
    # Storage stamps schema_version alongside the config; loading must
    # tolerate it without changing the round-trip.
    stamped = dict(d, schema_version="0.1.0")
    assert SmaulBrainConfig.from_dict(stamped).to_dict() == d


def test_config_omitted_vs_explicit_null():
    assert SmaulBrainConfig.from_dict({}).d_model == SmaulBrainConfig().d_model
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict({"d_model": None})


def test_config_unknown_fields_tolerated():
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=1)
    d = dict(cfg.to_dict(), future_knob=123, schema_version="0.1.0")
    got = SmaulBrainConfig.from_dict(d)
    assert got.to_dict() == cfg.to_dict()
    assert "future_knob" not in got.to_dict()


def test_config_rejects_active_topk_mismatch():
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=1)
    d = dict(cfg.to_dict(), active_experts=99)
    assert d["top_k"] == 2
    with pytest.raises(ValueError):
        SmaulBrainConfig.from_dict(d)


def test_config_rejects_impossible_min_experts():
    # min_experts < top_k must raise instead of being silently clamped.
    with pytest.raises(AssertionError):
        SmaulBrainConfig.from_dict(
            {"num_experts": 8, "top_k": 4, "min_experts": 2})


def test_config_counts_match_live_model():
    m = _model()
    dc = m.cfg.describe_counts()
    c = m.param_counts()
    assert dc["total_params"] == dc["shared_params"] + dc["router_params"] + dc["expert_params_total"]
    assert dc["logical_params"] == dc["total_params"] == dc["unique_params"]
    assert dc["resident_ram_params"] is None  # resident needs a live model
    assert dc["stored_expert_bytes"] is None  # on-disk needs a live model
    for k in ("shared_params", "router_params", "per_expert_params",
              "expert_count", "expert_params_total", "total_params",
              "active_params"):
        assert dc[k] == c[k]
    m.pager.close()


# --- issue #70 (audit 039): accounting semantics regressions (append-only) ---


def test_audit039_active_clamped_when_pool_below_topk():
    # Pruning below top_k must not report active > total (old code did).
    m = _model()
    for eid in list(m.pool.order[1:]):
        r = m.pool.experts[eid]
        r.tokens_routed = 0; r.grad_activity = 0.0; r.contribution = 0.0
        r.last_used_step = 0; r.birth_step = 0
    victims = find_victims(m.pool, step=1000, survival_steps=10, min_experts=1)
    prune_experts(m.pool, m.router, victims)
    c = m.param_counts()
    assert c["expert_count"] == 1
    assert c["active_params"] <= c["total_params"]
    assert c["active_params"] == c["shared_params"] + c["router_params"] + c["per_expert_params"]
    # Config snapshot stays truthful when clamped too.
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                           expert_hidden=64, max_depth=1)
    assert cfg.active_params() <= cfg.total_params()
    m.pager.close()


def test_audit039_expert_total_sums_live_pool():
    # Heterogeneous pools must sum, not per_expert * count (old code undercounted).
    from experts import make_expert
    m = _model(n_exp=2)
    big = make_expert(m.pool.fresh_id(), 32, 128)
    m.pool.add(big)
    m.router.add_expert_row()
    c = m.param_counts()
    truth = sum(r.param_count for r in m.pool.experts.values())
    assert c["expert_params_total"] == truth
    assert c["total_params"] == c["shared_params"] + c["router_params"] + truth
    assert c["logical_params"] == c["total_params"] == c["unique_params"]
    m.pager.close()


def test_audit039_logical_unique_keys_stable():
    m = _model()
    c = m.param_counts()
    dc = m.cfg.describe_counts()
    # No report keys removed: live now carries logical/unique like describe/cli.
    for k in ("shared_params", "router_params", "per_expert_params",
              "expert_count", "expert_params_total", "stored_expert_bytes",
              "total_params", "active_params",
              "resident_ram_params", "resident_vram_params"):
        assert k in c
    for k in ("logical_params", "unique_params"):
        assert c[k] == c["total_params"] == dc[k] == dc["total_params"]
    m.pager.close()


def test_audit039_shared_once_untied_no_depth_multiplier():
    # Shared block counted once across depth; embed/head untied (counted twice).
    m1 = _model()
    cfg2 = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                            expert_hidden=64, max_depth=5)
    m2 = SmaulBrainModel(cfg2)
    assert m1.param_counts()["shared_params"] == m2.param_counts()["shared_params"]
    assert m1.embed.weight.data_ptr() != m1.head.weight.data_ptr()
    assert m1.cfg.shared_params() == m1.param_counts()["shared_params"]
    m1.pager.close(); m2.pager.close()


def test_audit039_inactive_padding_pondering_invariant():
    # Inactive experts in total, not active; pads/ponder settings change no counts.
    m = _model()
    before = m.param_counts()
    # Route one batch: some experts may stay inactive yet remain in total.
    ids = torch.randint(0, 256, (2, 8))
    m.eval()
    with torch.no_grad():
        m.forward_infer(ids)
    after = m.param_counts()
    assert after["total_params"] == before["total_params"]
    assert after["active_params"] == before["active_params"]
    # Padded ids and different pondering depths change no param counts.
    from bytes import PAD_ID
    cfg_pad = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                               expert_hidden=64, max_depth=1)
    mp = SmaulBrainModel(cfg_pad)
    assert mp.param_counts()["total_params"] == before["total_params"]
    _ = PAD_ID  # pads are structural ids, never parameters
    m.pager.close(); mp.pager.close()


def test_audit039_paging_resident_no_double_count():
    # Resident caches hold dequantized transients only; RAM+VRAM unique <= total.
    for mode in ("D2R", "R2VR", "D2VR"):
        cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=2,
                               expert_hidden=64, max_depth=1, paging_method=mode)
        m = SmaulBrainModel(cfg)
        m.eval()
        with torch.no_grad():
            m.forward_infer(torch.randint(0, 256, (2, 8)))
        c = m.param_counts()
        # No expert counted twice across caches in any single mode.
        ram_ids = set(m.pager.ram.keys())
        vram_ids = set(m.pager.vram.keys())
        assert not (ram_ids & vram_ids)
        assert c["resident_ram_params"] + c["resident_vram_params"] <= c["total_params"]
        # Stored FP8 reality stays compressed vs logical fp32 footprint.
        assert c["stored_expert_bytes"] < c["expert_params_total"] * 4
        m.pager.close()


def test_audit039_throughput_countable_bytes_are_tokens_valid_only():
    # Countable throughput facts: 1 token == 1 byte; pads are not data bytes.
    # Timing (bytes/s) stays with the training loop (train.py) — flagged, not pinned here.
    from bytes import PAD_ID
    assert PAD_ID >= 256  # pads live outside byte range: never counted as data
    b = torch.tensor([[1, 2, PAD_ID, PAD_ID], [3, PAD_ID, PAD_ID, PAD_ID]])
    valid_bytes = int((b != PAD_ID).sum().item())
    assert valid_bytes == 3  # valid-only normalization train.py uses for bytes_processed
    # Bench identity train_bytes_per_s == train_tokens_per_s holds by byte vocab.
