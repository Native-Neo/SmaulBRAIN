"""Adaptive halting: min/max depth, threshold, easy-early vs hard-late."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from model import SmaulBrainModel


def _model(**kw):
    torch.manual_seed(0)
    base = dict(d_model=32, n_heads=4, num_experts=4, top_k=2,
                expert_hidden=64, context_length=16)
    base.update(kw)
    m = SmaulBrainModel(SmaulBrainConfig(**base))
    return m


def test_depths_respect_min_max():
    m = _model(min_depth=2, max_depth=4)
    ids = torch.randint(0, 256, (2, 16))
    out = m.forward_infer(ids)
    d = out["depths"]
    assert d.min().item() >= 2 and d.max().item() <= 4
    assert out["n_executed"] <= 4
    m.pager.close()


def test_low_threshold_halts_earlier_than_high():
    ids = torch.randint(0, 256, (2, 16))
    m1 = _model(min_depth=1, max_depth=4, halting_threshold=0.05)
    m2 = _model(min_depth=1, max_depth=4, halting_threshold=0.999)
    # Same weights, different thresholds: per-token selected depths must
    # reflect the threshold (inference runs full depth for state completeness
    # but selects each token's halting-depth representation).
    m2.load_state_dict({k: v.clone() for k, v in m1.state_dict().items()
                        if not k.startswith("router.")} |
                       {k: v for k, v in m2.state_dict().items()
                        if k.startswith("router.")}, strict=False)
    d1 = m1.forward_infer(ids)["depths"].float().mean().item()
    d2 = m2.forward_infer(ids)["depths"].float().mean().item()
    assert d1 < d2, f"threshold had no effect on selected depth: {d1} vs {d2}"
    m1.pager.close(); m2.pager.close()


def test_ponder_distribution_sums_to_one():
    m = _model(min_depth=1, max_depth=3)
    ids = torch.randint(0, 256, (1, 8))
    out = m(ids, ids, step=0)
    assert 1.0 <= out["mean_depth"] <= 3.0
    assert out["ponder_kl"].item() >= 0
    m.pager.close()


def test_halting_stable_during_training_step():
    from smaulopt import SmaulOpt, SmaulOptHParams
    from train import train_step
    torch.manual_seed(0)
    m = _model(min_depth=1, max_depth=3)
    opt = SmaulOpt(SmaulOptHParams())
    ids = torch.randint(0, 256, (2, 16))
    for step in range(3):
        s = train_step(m, opt, m.cfg, ids, ids, step)
        assert 1.0 <= s["mean_depth"] <= 3.0
        assert s["loss"] == s["loss"]  # no NaN
    m.pager.close()


def test_streamed_depths_match_full_pass():
    m = _model(min_depth=1, max_depth=3)
    ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
    full = m.forward_infer_stateful(ids)[0]["depths"]
    states = m.new_infer_state(1)
    parts = []
    for token in ids[0]:
        out, states = m.forward_infer_step(token.view(1), states)
        parts.append(out["depths"])
    assert torch.equal(torch.cat(parts, dim=1), full)  # halting is chunk-free
    m.pager.close()


def test_executed_work_is_always_full_depth():
    # Streaming design: every depth executes (states must observe all
    # tokens); adaptivity lives in readout. n_executed must say so.
    m = _model(min_depth=1, max_depth=3, halting_threshold=0.01)
    ids = torch.randint(0, 256, (2, 16))
    out = m.forward_infer(ids)
    assert out["n_executed"] == 3
    assert out["depths"].min().item() >= 1 and out["depths"].max().item() <= 3
    m.pager.close()


def test_saturated_halt_logits_stay_finite():
    m = _model(min_depth=1, max_depth=3)
    with torch.no_grad():
        m.block.halt.bias.fill_(50.0)  # force lam ~ 1 everywhere
    ids = torch.randint(0, 256, (2, 16))
    out = m(ids, ids, step=0)
    for key in ("loss", "nll", "ponder_kl", "balance"):
        assert torch.isfinite(out[key].detach()).all(), key
    m.pager.close()


def test_ponder_gated_by_min_depth():
    # P must put zero mass below min_depth; inference can never select
    # those depths, so the ponder weights must not score them either.
    m = _model(min_depth=2, max_depth=4, dtype="fp32")
    ids = torch.randint(0, 256, (2, 8))
    out = m(ids, ids, step=0)
    assert out["depths"].min().item() >= 2 and out["depths"].max().item() <= 4
    assert 2.0 <= out["mean_depth"] <= 4.0
    # Direct P check: depth-1 mass is exactly zero, rows still sum to 1.
    h = m.n_init(m.embed(ids).to(torch.float32))
    from linear_attention import LinearAttnState
    states = [LinearAttnState.zeros(2, 4, 8) for _ in range(4)]
    with torch.no_grad():
        hs, lams, _, _, _ = m._depth_loop(
            h, states, train=True, step=0,
            keep=torch.ones(2 * 8, dtype=torch.bool),
        )
        P, _, _ = m._ponder([l.float() for l in lams])
    assert torch.equal(P[0], torch.zeros_like(P[0]))
    assert torch.allclose(P.sum(dim=0), torch.ones_like(P.sum(dim=0)))
    m.clear_expert_grads()
    m.pager.close()


def test_train_depths_share_infer_threshold_rule():
    # Saturated halt: ponder expectation ~1, so both training (diagnostic)
    # and inference (readout) depths must select depth 1.
    m = _model(min_depth=1, max_depth=3, dtype="fp32")
    with torch.no_grad():
        m.block.halt.bias.fill_(50.0)
    ids = torch.randint(0, 256, (2, 8))
    train_depths = m(ids, ids, step=0)["depths"]
    infer_depths = m.forward_infer(ids)["depths"]
    assert torch.equal(train_depths, torch.ones_like(train_depths))
    assert torch.equal(train_depths, infer_depths)
    m.pager.close()


def test_mean_depth_is_valid_only_and_readout_is_ponder_mixed():
    from bytes import PAD_ID
    from train import batch_from_seqs
    torch.manual_seed(0)
    m = _model(min_depth=1, max_depth=3, dtype="fp32")
    m.eval()
    seq = [10, 20, 30, 40, 50, 60]
    b = batch_from_seqs([seq], context=8)
    x, y = b[:, :8], b[:, 1:]
    valid = (y != PAD_ID)
    with torch.no_grad():
        out = m(x, y, step=0)
    # Manual valid-only ponder expectation matches the diagnostic.
    _emb = m.embed(x).to(torch.float32)
    if getattr(m, "byte_conv", None) is not None:
        with torch.no_grad():
            _emb = m.byte_conv(_emb)
    h = m.n_init(_emb)
    from linear_attention import LinearAttnState
    states = [LinearAttnState.zeros(1, 4, 8) for _ in range(3)]
    with torch.no_grad():
        hs, lams, _, _, _ = m._depth_loop(
            h, states, train=True, step=0, keep=valid.reshape(-1))
        P, _, _ = m._ponder([l.float() for l in lams])
    m.clear_expert_grads()
    steps = torch.arange(1, 4).view(-1, 1, 1)
    expected = float((P * steps).sum(dim=0)[valid].sum().item()
                     / max(1, int(valid.sum().item())))
    assert out["mean_depth"] == expected
    # KL-vs-readout alignment: the returned logits are the ponder-mixed
    # predictor the loss scores (same P, same per-step logits).
    with torch.no_grad():
        manual = sum(P[n].unsqueeze(-1) * m.head(m.n_final(hs[n])).float()
                     for n in range(3))
    assert torch.allclose(out["logits"].detach(), manual, atol=1e-5)
    assert torch.allclose(P.sum(dim=0), torch.ones_like(P.sum(dim=0)))
    m.pager.close()


def test_ponder_single_depth_kl_is_zero():
    # max_depth=1: all mass force-stops at step 0 against prior mass 1
    # (truncated-geometric tail), so the ponder KL is exactly 0. The old
    # defective prior (p on the last step) charged -log(p) spuriously.
    torch.manual_seed(0)
    m = _model(min_depth=1, max_depth=1, dtype="fp32")
    ids = torch.randint(0, 256, (2, 8))
    out = m(ids, ids, step=0)
    assert out["ponder_kl"].item() == 0.0
    m.pager.close()


def test_ponder_prior_masses_sum_to_one():
    # Analytic check of the truncated-geometric prior the KL uses:
    # p*(1-p)^n before the last step, tail (1-p)^(D-1) on force-stop.
    for max_depth in (1, 2, 3, 5):
        torch.manual_seed(0)
        m = _model(min_depth=1, max_depth=max_depth, dtype="fp32")
        p = m.cfg.halt_prior
        D = max_depth
        masses = [p * (1 - p) ** n for n in range(D - 1)] + [(1 - p) ** (D - 1)]
        assert abs(sum(masses) - 1.0) < 1e-12
        m.pager.close()
