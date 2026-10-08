"""Offline pruner script: floor, hysteresis, resumability (via CLI only).

Training never prunes, so these tests drive ``pruning.py`` as a subprocess
(``--rm-worst N``) against real checkpoints — the selection/removal logic
has no importable surface anymore.
"""

import json
import subprocess
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from storage import load_model, save_model

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _model(n_exp=6, min_experts=2, seed=0):
    torch.manual_seed(seed)
    cfg = SmaulBrainConfig(d_model=16, n_heads=2, num_experts=n_exp, top_k=1,
                           expert_hidden=32, max_depth=1, context_length=12,
                           expert_lr=3e-2, max_experts=64,
                           min_experts=min_experts)
    m = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    return m, opt, cfg


def _aged_ckpt(d, n_exp=6, min_experts=2, useful=None):
    """Checkpoint whose experts are all old/idle/zero-signal (prune-ready)."""
    import shutil
    m, opt, _ = _model(n_exp, min_experts)
    for rec in m.pool.experts.values():
        rec.birth_step = -600
        rec.last_used_step = -600
    if useful is not None:
        u = m.pool.experts[useful]
        u.tokens_routed = 1000
        u.grad_activity = 1e-3
        u.contribution = 1.0
        u.last_used_step = 600
    shutil.rmtree(d, ignore_errors=True)
    save_model(d, m, opt, step=600)
    m.pager.close()
    return d


def _prune(ckpt, *args):
    return subprocess.run(
        [sys.executable, "pruning.py", "--ckpt", ckpt, *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def _pool_size(ckpt):
    m, _, _ = _model()
    o = SmaulOpt(SmaulOptHParams())
    load_model(ckpt, m, o)
    n = len(m.pool)
    m.pager.close()
    return n


def test_pruner_removes_worst_and_stays_resumable(tmp_path):
    d = _aged_ckpt(str(tmp_path / "c"))
    r = _prune(d, "--rm-worst", "2")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["removed"] and len(out["removed"]) == 2
    assert out["pool_before"] == 6 and out["pool_after"] == 4
    assert out["floor"] == 2
    # Checkpoint loads cleanly with matching router/optimizer widths.
    m, _, _ = _model()
    o = SmaulOpt(SmaulOptHParams())
    man = load_model(d, m, o)
    assert man["step"] == 600
    assert len(m.pool) == m.router.num_experts == 4
    for st in o.router_state.values():
        for key in ("m", "v_row", "v"):
            t = st.get(key)
            if torch.is_tensor(t):
                assert t.shape[0] == 4
    m.pager.close()


def test_pruner_respects_resume_config_floor(tmp_path):
    d = _aged_ckpt(str(tmp_path / "c"))
    # Only 6 - 2 = 4 removable: asking for 5 refuses with nothing changed.
    r = _prune(d, "--rm-worst", "5")
    assert r.returncode == 2
    assert _pool_size(d) == 6
    # Pruning exactly to the floor succeeds.
    r = _prune(d, "--rm-worst", "4")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["pool_after"] == 2
    # At the floor nothing is removable anymore.
    r = _prune(d, "--rm-worst", "1")
    assert r.returncode == 2
    assert _pool_size(d) == 2


def test_pruner_dry_run_changes_nothing(tmp_path):
    d = _aged_ckpt(str(tmp_path / "c"))
    r = _prune(d, "--rm-worst", "2", "--dry-run")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["dry_run"] is True and out["removed"] == []
    assert len(out["victims"]) == 2
    assert _pool_size(d) == 6


def test_pruner_victims_deterministic(tmp_path):
    d = _aged_ckpt(str(tmp_path / "c"))
    a = json.loads(_prune(d, "--rm-worst", "3", "--dry-run").stdout)["victims"]
    b = json.loads(_prune(d, "--rm-worst", "3", "--dry-run").stdout)["victims"]
    assert a == b and len(a) == 3


def test_pruner_keeps_useful_expert_and_grace(tmp_path):
    d = _aged_ckpt(str(tmp_path / "c"), useful="expert_00000")
    out = json.loads(_prune(d, "--rm-worst", "4", "--dry-run").stdout)
    assert "expert_00000" not in out["victims"]  # high-signal survives
    assert len(out["victims"]) == 4
    # Grace: a freshly born pool proposes no victims at all.
    m, opt, _ = _model()
    import shutil
    d2 = str(tmp_path / "young")
    shutil.rmtree(d2, ignore_errors=True)
    save_model(d2, m, opt, step=0)
    m.pager.close()
    out = json.loads(_prune(d2, "--rm-worst", "2", "--dry-run").stdout)
    assert out["victims"] == []


def test_pruner_missing_checkpoint_errors(tmp_path):
    r = _prune(str(tmp_path / "nope"), "--rm-worst", "1")
    assert r.returncode == 2


def test_pruner_then_training_resumes(tmp_path):
    from train import run_training
    d = _aged_ckpt(str(tmp_path / "c"))
    r = _prune(d, "--rm-worst", "2")
    assert r.returncode == 0, r.stderr
    m, _, _ = _model()
    o = SmaulOpt(SmaulOptHParams())
    load_model(d, m, o)
    seqs = [[(i + j) % 200 for j in range(16)] for i in range(8)]
    res = run_training(m, o, m.cfg, seqs, steps=1, batch_size=2,
                       grow_every=0, grow_loss_below=None,
                       log_fn=lambda s: None)
    assert res["steps_completed"] == 1 and len(m.pool) == 4
    m.pager.close()
