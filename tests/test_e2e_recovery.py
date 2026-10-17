"""End-to-end lifecycle + crash-recovery (fault injection) tests."""
import types

import numpy as np
import torch
import pytest

from smaulbrain import checkpoint as CK
from smaulbrain import config as C
from smaulbrain import data as D
from smaulbrain import growth as G
from smaulbrain import inference as INF
from smaulbrain import model as M
from smaulbrain import prune_cmd as PR
from smaulbrain import train_loop as T


def _args(ckpt, **kw):
    d = dict(paging="R", paging_cache_experts=8, paging_staging_bytes=1 << 24,
             forward_quant="fp32", backward_mode="fp32", device="cpu",
             checkpoint=ckpt, seed=0, seq_len=16, lr=3e-4,
             beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.0,
             growth_every_steps=0, growth_loss_nats=None, growth_loss_beta=0.0,
             growth_min_eval_samples=32, growth_noise_scale=0.01, growth_seed=1,
             expert_capacity_factor=99.0)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _trainer(preset, ckpt, **kw):
    spec = C.PRESETS[preset]
    args = _args(ckpt, **kw)
    stream = D.ByteStream(data=D.synth_bytes(4096, 0))
    return T.Trainer(spec, args, stream)


def _expert_count(tr):
    return len(M.expert_ids(tr.model.p))


def test_periodic_growth_inside_training_loop():
    import tempfile, os
    ckpt = tempfile.mkdtemp() + "/ckpt"
    tr = _trainer("moe-tiny", ckpt, growth_every_steps=2)
    tr.train(max_steps=2)
    assert _expert_count(tr) == 34  # one event: +2 tiny experts
    assert tr.sched.history[0]["reason"] == "step"
    assert tr.sched.history[0]["id"] == 1


def test_loss_triggered_growth_inside_training_loop():
    import tempfile
    ckpt = tempfile.mkdtemp() + "/ckpt"
    # threshold above initial loss -> fires on first step's downward crossing
    tr = _trainer("moe-tiny", ckpt, growth_loss_nats=100.0)
    tr.train(max_steps=1)
    assert _expert_count(tr) == 34
    assert tr.sched.history[0]["reason"] == "loss"
    # further training below threshold must NOT repeat
    tr.train(max_steps=3)
    assert _expert_count(tr) == 34


def test_full_growth_counts_and_mutation_shape():
    import tempfile
    ckpt = tempfile.mkdtemp() + "/ckpt"
    tr = _trainer("moe-tiny", ckpt, growth_every_steps=1,
                  growth_min_eval_samples=16)
    tr.train(max_steps=1)
    e = tr.sched.history[0]
    assert e["prev_count"] == 32 and e["new_count"] == 34
    detail = e["detail"]
    mutated = [dd for dd in detail if dd["mutated"]]
    assert len(mutated) == 2 * 2  # both tiny clones mutated, 2 tensors each
    # new experts trainable: one more step changes them
    new_keys = [f"expert.{i}.{s}" for i in e["new_ids"] for s in ("Win", "Wout")]
    before = {k: tr.model.p[k].detach().clone() for k in new_keys}
    tr.train(max_steps=2)
    assert any(not torch.equal(tr.model.p[k], before[k]) for k in new_keys)


def test_growth_mutation_full_preset_counts():
    # verify +8 / 6-mutated / 2-exact arithmetic without training a full model
    rng = np.random.default_rng(0)
    spec = C.PRESETS["moe-full"]
    n0 = spec.n_experts_init
    assert n0 == 64
    params = {"router": rng.standard_normal((8, n0)).astype(np.float32)}
    for e in range(n0):
        params[f"expert.{e}.Win"] = rng.standard_normal((8, 4)).astype(np.float32)
        params[f"expert.{e}.Wout"] = rng.standard_normal((4, 8)).astype(np.float32)
    sources = list(range(8))
    new_ids = list(range(n0, n0 + 8))
    cand, detail = G.grow_candidate(params, params["router"], sources, new_ids,
                                    0.01, seed=3, n_exact_unperturbed=2)
    assert cand["router"].shape == (8, n0 + 8)
    mut = [d for d in detail if d["mutated"]]
    exact = [d for d in detail if not d["mutated"]]
    assert len(mut) == 6 * 2 and len(exact) == 2 * 2
    for d in exact:
        assert np.array_equal(cand[f"expert.{d['expert']}.{d['tensor']}"],
                              params[f"expert.{d['source']}.{d['tensor']}"])


def _run_with_fail(ckpt, fail_at, max_steps=2):
    tr = _trainer("moe-tiny", ckpt, growth_every_steps=1,
                  growth_min_eval_samples=16)
    try:
        tr.train(max_steps=max_steps, growth_fail_at=fail_at)
        return tr, None
    except T._Injected as e:
        return tr, str(e)
    except RuntimeError as e:
        if "injected" in str(e).lower() or "Inject" in type(e).__name__:
            return tr, str(e)
        raise


@pytest.mark.parametrize("fail_at", [
    "before_pending", "during_pending", "after_pending", "before_candidate",
    "during_clone", "during_mutate", "during_optinit", "after_validate",
    "during_commit", "after_commit",
])
def test_crash_recovery_exactly_once(tmp_path, fail_at):
    ckpt = str(tmp_path / "ckpt")
    tr, err = _run_with_fail(ckpt, fail_at)
    assert err is not None or fail_at == "after_commit"
    # latest valid checkpoint always loads
    blob = CK.load_checkpoint(ckpt)
    man = blob["manifest"]
    n_params = len([k for k in blob["params"] if k.endswith(".Win")])
    assert n_params == man["n_experts_current"]
    # resume in a FRESH process-equivalent (new Trainer) with no fault
    tr2 = _trainer("moe-tiny", ckpt, growth_every_steps=1,
                   growth_min_eval_samples=16)
    man2 = tr2.load(ckpt)
    # resolve any pending event, then suppress further triggers to isolate
    # exactly-once verification of the recovered event
    if tr2.sched.pending is not None:
        tr2._resolve_pending()
    tr2.sched.every_steps = 0
    tr2.sched.next_periodic_step = None
    step_before = tr2.sched.global_step
    tr2.train(max_steps=step_before + 1)
    # exactly-once: one history entry for event id 1, correct expert count
    ids = [h["id"] for h in tr2.sched.history]
    assert ids == [1]
    assert _expert_count(tr2) == 34
    final = CK.load_checkpoint(ckpt)
    assert final["manifest"]["scheduler"]["pending"] is None
    # model trains and infers after recovery
    eng = INF.InferenceEngine(tr2.model, seed=0, temperature=0.0)
    out = eng.generate([65, 66, 67, 68], max_new_tokens=4)
    assert len(out) == 4


def test_prune_lifecycle_and_regrow(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    tr = _trainer("moe-tiny", ckpt, growth_every_steps=1,
                  growth_min_eval_samples=16)
    tr.train(max_steps=1)
    assert _expert_count(tr) == 34
    stream = D.ByteStream(data=D.synth_bytes(4096, 9))
    batches = [stream.next_tokens(16) for _ in range(4)]
    rep = PR.prune_checkpoint(ckpt, 2, batches, min_samples=16,
                              dry_run=False, assume_yes=True)
    assert rep["new_count"] == 32
    tr2 = _trainer("moe-tiny", ckpt)
    man = tr2.load(ckpt)
    assert _expert_count(tr2) == 32
    assert man["arch"]["experts_current"] == 32
    logits, _ = tr2.model.forward([65, 66, 67, 68], train=False)
    assert torch.isfinite(logits).all().item()


def test_full_lifecycle_dense_and_moe(tmp_path):
    for preset in ("dense-tiny", "moe-tiny"):
        ckpt = str(tmp_path / preset)
        tr = _trainer(preset, ckpt)
        tr.train(max_steps=3)
        tr.save()
        # fresh process restore
        tr2 = _trainer(preset, ckpt)
        man = tr2.load(ckpt)
        assert man["scheduler"]["global_step"] == 3
        tr2.train(max_steps=5)
        eng = INF.InferenceEngine(tr2.model, seed=1, temperature=0.0)
        out = eng.generate([72, 101, 108, 108, 111], max_new_tokens=8)
        assert len(out) == 8
        # deterministic repeat
        eng2 = INF.InferenceEngine(tr2.model, seed=1, temperature=0.0)
        assert eng2.generate([72, 101, 108, 108, 111], max_new_tokens=8) == out


def test_graceful_stop_finishes_update_and_saves(tmp_path):
    ckpt = str(tmp_path / "graceful-stop")
    tr = _trainer("moe-tiny", ckpt)
    tr.request_stop()
    tr.train(max_steps=3)
    assert tr.sched.global_step == 1
    saved = CK.load_checkpoint(ckpt)
    assert saved["manifest"]["scheduler"]["global_step"] == 1
    # A fresh trainer can resume the saved state and continue normally.
    tr2 = _trainer("moe-tiny", ckpt)
    tr2.load(ckpt)
    tr2.train(max_steps=2)
    assert tr2.sched.global_step == 2


def test_stop_request_at_loop_boundary_still_saves_without_periodic_saves(tmp_path):
    ckpt = str(tmp_path / "stop-at-boundary")
    tr = _trainer("moe-tiny", ckpt)
    tr.request_stop()

    tr.train(max_steps=0, ckpt_every=0)

    saved = CK.load_checkpoint(ckpt)
    assert saved["manifest"]["scheduler"]["global_step"] == 0
