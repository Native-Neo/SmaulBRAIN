import numpy as np

from smaulbrain import smaulopt as O


def test_factored_state_shapes_and_memory():
    opt = O.SmaulOPT()
    W = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
    v = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    opt.register({"W": W, "v": v})
    assert opt.state["W"]["R"].shape == (6,)
    assert opt.state["W"]["C"].shape == (8,)
    assert opt.state["v"]["v"].shape == (8,)  # vectors documented non-factored
    mem = opt.state_memory_elements()
    full_equiv = 2 * (48 + 8)
    assert mem["factored_elements"] + mem["full_elements"] < full_equiv


def test_update_changes_params_and_handles_missing_grads():
    opt = O.SmaulOPT(lr=1e-3)
    W = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    W0 = W.copy()
    opt.register({"W": W})
    opt.step({"W": W}, {"W": np.ones_like(W)})
    assert not np.array_equal(W, W0)
    opt.step({"W": W}, {})  # missing grad: skipped safely
    opt.drop(["nope"])  # safe


def test_checkpoint_resume_parity():
    opt = O.SmaulOPT(lr=1e-3)
    W = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    opt.register({"W": W})
    g = np.random.default_rng(1).standard_normal((4, 5)).astype(np.float32)
    opt.step({"W": W}, {"W": g})
    sd = opt.state_dict()
    flags = {k: s["factored"] for k, s in opt.state.items()}
    opt2 = O.SmaulOPT(lr=1e-3)
    W2 = W.copy()
    opt2.register({"W": W2})
    opt2.load_state_dict(sd, flags)
    g2 = np.random.default_rng(2).standard_normal((4, 5)).astype(np.float32)
    Wa, Wb = W.copy(), W2.copy()
    opt.step({"W": Wa}, {"W": g2})
    opt2.step({"W": Wb}, {"W": g2})
    assert np.allclose(Wa, Wb)


def test_training_changes_params_and_routed_experts_update(tmp_path):
    import types
    from smaulbrain import config as C
    from smaulbrain import data as D
    from smaulbrain import model as M
    from smaulbrain import train_loop as T
    spec = C.PRESETS["moe-tiny"]
    args = types.SimpleNamespace(
        paging="R", paging_cache_experts=8, paging_staging_bytes=1 << 24,
        forward_quant="fp32", backward_mode="fp32", device="cpu",
        checkpoint=str(tmp_path / "ckpt"), seed=0, seq_len=16, lr=3e-4,
        beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.0,
        growth_every_steps=0, growth_loss_nats=None, growth_loss_beta=0.98,
        growth_min_eval_samples=64, growth_noise_scale=0.01, growth_seed=1,
        expert_capacity_factor=99.0)
    stream = D.ByteStream(data=D.synth_bytes(2048, 0), seed=0)
    tr = T.Trainer(spec, args, stream)
    before = {k: v.copy() for k, v in tr.model.p.items()}
    tr.train(max_steps=3, log_every=1)
    changed = [k for k in tr.model.p if not np.array_equal(tr.model.p[k], before[k])]
    assert len(changed) > 10
    # routed experts received updates (expert grads are non-trivial)
    exp_changed = [k for k in changed if k.startswith("expert.")]
    assert len(exp_changed) > 0
