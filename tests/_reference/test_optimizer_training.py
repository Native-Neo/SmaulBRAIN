import torch

from smaulbrain import smaulopt as O


def _rand(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def test_factored_state_shapes_and_memory():
    opt = O.SmaulOPT()
    W, v = _rand((6, 8), 0), _rand((8,), 1)
    opt.register({"W": W, "v": v})
    assert opt.state["W"]["R"].shape == (6,)
    assert opt.state["W"]["C"].shape == (8,)
    assert opt.state["v"]["v"].shape == (8,)
    mem = opt.state_memory_elements()
    assert mem["factored_elements"] + mem["full_elements"] < 2 * (48 + 8)


def test_update_changes_params_and_handles_missing_grads():
    opt = O.SmaulOPT(lr=1e-3)
    W = _rand((4, 5), 0); W0 = W.clone()
    opt.register({"W": W})
    opt.step({"W": W}, {"W": torch.ones_like(W)})
    assert not torch.equal(W, W0)
    opt.step({"W": W}, {})
    opt.drop(["nope"])


def test_checkpoint_resume_parity():
    opt = O.SmaulOPT(lr=1e-3)
    W = _rand((4, 5), 0); opt.register({"W": W})
    opt.step({"W": W}, {"W": _rand((4, 5), 1)})
    sd = opt.state_dict(); flags = {k: s["factored"] for k, s in opt.state.items()}
    opt2 = O.SmaulOPT(lr=1e-3); W2 = W.clone(); opt2.register({"W": W2})
    opt2.load_state_dict(sd, flags)
    g = _rand((4, 5), 2); Wa, Wb = W.clone(), W2.clone()
    opt.step({"W": Wa}, {"W": g}); opt2.step({"W": Wb}, {"W": g})
    assert torch.allclose(Wa, Wb)


def test_training_changes_params_and_routed_experts_update(tmp_path):
    import types
    from smaulbrain import config as C, data as D, train_loop as T
    args = types.SimpleNamespace(paging="R", paging_cache_experts=8, paging_staging_bytes=1 << 24,
        forward_quant="fp32", backward_mode="fp32", device="cpu", checkpoint=str(tmp_path / "ckpt"),
        seed=0, seq_len=16, lr=3e-4, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.0,
        growth_every_steps=0, growth_loss_nats=None, growth_loss_beta=0.98, growth_min_eval_samples=64,
        growth_noise_scale=0.01, growth_seed=1, expert_capacity_factor=99.0)
    tr = T.Trainer(C.PRESETS["moe-tiny"], args, D.ByteStream(data=D.synth_bytes(2048, 0)))
    before = {k: v.detach().clone() for k, v in tr.model.p.items()}
    tr.train(max_steps=3, log_every=1)
    changed = [k for k, v in tr.model.p.items() if not torch.equal(v, before[k])]
    assert len(changed) > 10
    assert any(k.startswith("expert.") for k in changed)
