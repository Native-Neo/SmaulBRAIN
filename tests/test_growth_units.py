import numpy as np

from smaulbrain import growth as G


def test_periodic_schedule_exact_steps():
    s2 = G.GrowthScheduler(every_steps=10, loss_nats=None)
    got = []
    for i in range(25):
        r = s2.note_optimizer_step(5.0)
        if r == "step":
            got.append(i + 1)
            s2.commit_event({"id": len(s2.history) + 1, "step": s2.global_step,
                             "reason": "step"})
    assert got == [10, 20]


def test_loss_edge_trigger_no_repeat_and_rearm():
    s = G.GrowthScheduler(every_steps=0, loss_nats=0.5, loss_beta=0.0)
    assert s.note_optimizer_step(1.0) is None
    assert s.note_optimizer_step(0.4) == "loss"
    s.commit_event({"id": 1, "step": s.global_step, "reason": "loss"})
    assert s.note_optimizer_step(0.4) is None  # disarmed: no repeat below threshold
    assert s.note_optimizer_step(0.6) is None  # rises above -> rearm
    assert s.note_optimizer_step(0.4) == "loss"  # downward crossing fires again


def test_simultaneous_triggers_dedup_to_one():
    s = G.GrowthScheduler(every_steps=2, loss_nats=0.5, loss_beta=0.0)
    s.note_optimizer_step(1.0)
    r = s.note_optimizer_step(0.1)
    assert r == "both"


def test_disabled_triggers_and_bad_measurements():
    s = G.GrowthScheduler(every_steps=0, loss_nats=None)
    for v in (0.1, float("nan"), float("inf"), None):
        assert s.note_optimizer_step(v) is None
    s2 = G.GrowthScheduler(every_steps=5, loss_nats=None)
    assert s2.note_optimizer_step(float("nan")) is None
    got = []
    for _ in range(6):
        if s2.note_optimizer_step(3.0) == "step":
            got.append(s2.global_step)
            s2.commit_event({"id": len(s2.history) + 1, "step": s2.global_step,
                             "reason": "step"})
    assert got == [5]


def test_resume_schedule_after_restore():
    s = G.GrowthScheduler(every_steps=10, loss_nats=None)
    for _ in range(15):
        if s.note_optimizer_step(3.0) == "step":
            s.commit_event({"id": len(s.history) + 1, "step": s.global_step,
                            "reason": "step"})
    rec = s.to_record()
    assert rec["global_step"] == 15 and rec["next_periodic_step"] == 20
    s2 = G.GrowthScheduler.from_record(rec)
    got = []
    for _ in range(10):
        if s2.note_optimizer_step(3.0) == "step":
            got.append(s2.global_step)
            s2.commit_event({"id": len(s2.history) + 1, "step": s2.global_step,
                             "reason": "step"})
    assert got == [20]


def test_gaussian_mutation_exactness_and_seed():
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((4, 6)) * 2).astype(np.float32)
    params = {"expert.0.Win": W, "expert.0.Wout": W.copy()}
    p1 = dict(params)
    G.clone_with_mutation(p1, [0], [1], 0.01, seed=7, n_exact_unperturbed=0)
    p2 = dict(params)
    G.clone_with_mutation(p2, [0], [1], 0.01, seed=7, n_exact_unperturbed=0)
    assert np.array_equal(p1["expert.1.Win"], p2["expert.1.Win"])  # deterministic
    assert not np.array_equal(p1["expert.1.Win"], W)  # mutated
    p3 = dict(params)
    G.clone_with_mutation(p3, [0], [2], 0.01, seed=7, n_exact_unperturbed=1)
    assert np.array_equal(p3["expert.2.Win"], W)  # exact unperturbed copy
    diff = p1["expert.1.Win"].double().numpy() - W.astype(float)
    assert abs(diff.mean()) < 0.2 and diff.std() > 0  # zero-mean additive noise
