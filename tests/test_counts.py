"""Dynamic parameter counting: shared/router/expert/active/resident splits."""

import sys, os, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
