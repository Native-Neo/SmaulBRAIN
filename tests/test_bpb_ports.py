"""BPB ports (PR #72 rework): byte-conv, lookahead/boundary loss, cosine decay, BPB.

Byte-conv is default-on; lookahead/boundary/cosine stay default-off: with
those defaults the model is bit-identical to not having them (covered by
asserting absent modules + unchanged key sets).
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

from config import SmaulBrainConfig
from model import SmaulBrainModel
from recurrent import CausalByteConv
from smaulopt import SmaulOpt, SmaulOptHParams
from train import cosine_scale, train_step


def _cfg(**kw):
    base = dict(d_model=32, n_heads=4, num_experts=4, top_k=2,
                expert_hidden=64, max_depth=2, context_length=24,
                expert_lr=3e-2)
    base.update(kw)
    return SmaulBrainConfig(**base)


def test_conv_causal_shape_and_validation():
    torch.manual_seed(0)
    c = CausalByteConv(16, kernel_size=4)
    x = torch.randn(2, 10, 16)
    assert c(x).shape == (2, 10, 16)
    # output[t] sees only inputs[..t]: truncated prefix reproduces prefix.
    assert torch.equal(c(x)[:, :5, :], c(x[:, :5, :]))
    # Full future changes later outputs (context actually flows).
    assert not torch.equal(c(x)[:, 5:, :], c(torch.zeros_like(x))[:, 5:, :])
    with pytest.raises(AssertionError):
        CausalByteConv(16, kernel_size=0)
    with pytest.raises(ValueError):
        c(torch.randn(2, 4, 8))


def test_cosine_scale_boundaries():
    assert cosine_scale(0, 100) == 1.0
    assert cosine_scale(50, 100) == pytest.approx(0.5)
    assert abs(cosine_scale(100, 100)) < 1e-9
    assert cosine_scale(150, 100) == pytest.approx(0.0)
    assert cosine_scale(0, 0) == 1.0 and cosine_scale(7, -3) == 1.0


def test_config_validation_and_old_checkpoint_compat():
    with pytest.raises(AssertionError):
        _cfg(lookahead_weight=-0.1)
    with pytest.raises(AssertionError):
        _cfg(boundary_weight=-0.1)
    with pytest.raises(AssertionError):
        _cfg(cosine_decay_steps=-1)
    d = _cfg().to_dict()
    for k in ("use_byte_conv", "lookahead_weight", "boundary_weight",
              "cosine_decay_steps"):
        del d[k]  # old checkpoint without the keys
    cfg = SmaulBrainConfig.from_dict(d)
    assert (cfg.use_byte_conv, cfg.lookahead_weight,
            cfg.boundary_weight, cfg.cosine_decay_steps) == (True, 0.0, 0.0, 0)


def test_default_conv_on_and_explicit_off_keyset():
    torch.manual_seed(0)
    m = SmaulBrainModel(_cfg())
    assert m.byte_conv is not None and m.head_lookahead is None and m.head_boundary is None
    torch.manual_seed(0)
    m0 = SmaulBrainModel(_cfg(use_byte_conv=False, lookahead_weight=0.0,
                              boundary_weight=0.0, cosine_decay_steps=0))
    assert m0.byte_conv is None
    # Only the conv keys differ; every other trunk key is shared.
    assert (set(m.state_dict()) - set(m0.state_dict())
            == {"byte_conv.conv.weight", "byte_conv.conv.bias"})
    m.pager.close(); m0.pager.close()


def test_aux_losses_match_manual_math():
    torch.manual_seed(2)
    cfg = _cfg(lookahead_weight=0.15, boundary_weight=0.05)
    m = SmaulBrainModel(cfg)
    xb = torch.randint(0, 256, (2, 24))
    yb = torch.randint(0, 256, (2, 24))
    out = m(xb, yb, step=0)
    assert out["lookahead"].item() > 0 and out["boundary"].item() > 0
    # loss == nll + ponder + balance + alpha*la + beta*bd
    expect = (out["nll"] + cfg.ponder_beta * out["ponder_kl"]
              + cfg.moe_balance_weight * out["balance"]
              + 0.15 * out["lookahead"] + 0.05 * out["boundary"])
    assert abs(out["loss"].item() - expect.item()) < 1e-5
    m.pager.close()


def test_cosine_decay_changes_trajectory():
    def _run(horizon):
        torch.manual_seed(3)
        cfg = _cfg(cosine_decay_steps=horizon)
        m = SmaulBrainModel(cfg)
        opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
        xb = torch.randint(0, 256, (2, 24))
        yb = torch.randint(0, 256, (2, 24))
        losses = [train_step(m, opt, cfg, xb, yb, step=s, mode="entire")["loss"]
                  for s in range(3)]
        m.pager.close()
        return losses
    assert _run(0) != _run(4)  # schedule on vs off diverges


def test_bpb_key_equals_nll_over_ln2():
    torch.manual_seed(4)
    # Aux ports on: proves BPB is pure compression (NLL/ln2), not the
    # regularized total loss (ponder KL + balance + lookahead + boundary
    # shape training but are not bits per byte).
    m = SmaulBrainModel(_cfg(lookahead_weight=0.15, boundary_weight=0.05))
    opt = SmaulOpt(SmaulOptHParams(lr=3e-2))
    xb = torch.randint(0, 256, (2, 24))
    yb = torch.randint(0, 256, (2, 24))
    st = train_step(m, opt, m.cfg, xb, yb, step=0, mode="entire")
    assert abs(st["bpb"] - st["nll"] / math.log(2)) < 1e-9
    assert st["loss"] > st["nll"]  # regularizers active: loss/ln2 would lie
    m.pager.close()


def test_streaming_parity_with_conv_on():
    torch.manual_seed(5)
    m = SmaulBrainModel(_cfg(use_byte_conv=True))
    ids = torch.tensor([[10, 20, 30, 40]])
    full, _ = m.forward_infer_stateful(ids)
    states = m.new_infer_state(1)
    parts = []
    for tok in ids[0]:
        out, states = m.forward_infer_step(tok.view(1), states)
        parts.append(out["logits"])
    streamed = torch.cat(parts, dim=1)
    assert torch.allclose(full["logits"], streamed, atol=2e-5, rtol=2e-5)
    m.pager.close()


def test_cli_bpb_flags_map():
    from cli import build_parser, config_from_args
    args = build_parser().parse_args(
        ["--use-byte-conv", "--lookahead-weight", "0.15",
         "--boundary-weight", "0.05", "--cosine-decay-steps", "100", "train"])
    cfg = config_from_args(args)
    assert cfg.use_byte_conv is True
    assert (cfg.lookahead_weight, cfg.boundary_weight,
            cfg.cosine_decay_steps) == (0.15, 0.05, 100)
    cfg0 = config_from_args(build_parser().parse_args(["train"]))
    assert (cfg0.use_byte_conv, cfg0.lookahead_weight,
            cfg0.boundary_weight, cfg0.cosine_decay_steps) == (True, 0.0, 0.0, 0)
    cfg_off = config_from_args(build_parser().parse_args(["--no-byte-conv", "train"]))
    assert cfg_off.use_byte_conv is False
