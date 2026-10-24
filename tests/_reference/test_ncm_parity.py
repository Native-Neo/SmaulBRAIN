"""Development-only parity oracle: original PyTorch NCM vs NumPy NCM.

Run explicitly with:
    PYTHONPATH=.:../smaulbrain-core pytest -q tests/_reference/test_ncm_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_CORE = _ROOT.parent / "smaulbrain-core"
for path in (str(_CORE), str(_ROOT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ncm as np_ncm  # noqa: E402
import ncm_torch as ref  # noqa: E402
from smaulbrain import npgrad as ng  # noqa: E402


def _pair(model_kind="moe"):
    torch.manual_seed(1234)
    cfg_t = ref.NCMConfig(
        vocab_size=256, d_model=16, codebook_size=16, levels=3,
        n_layers=2, n_heads=4, max_seq_len=16, dropout=0.0,
        model_kind=model_kind, n_experts=4 if model_kind == "moe" else 0,
        top_k=2 if model_kind == "moe" else 0,
    )
    cfg_n = np_ncm.NCMConfig(**cfg_t.__dict__)
    tok_t = ref.ResidualVQTokenizer(cfg_t).eval()
    tok_n = np_ncm.ResidualVQTokenizer(cfg_n).eval()
    tok_n.load_state_dict({
        k: v.detach().cpu().numpy() for k, v in tok_t.state_dict().items()
    })
    gen_t = ref.NCMGenerator(cfg_t, tok_t).eval()
    gen_n = np_ncm.NCMGenerator(cfg_n, tok_n).eval()
    gen_n.load_state_dict({
        k: v.detach().cpu().numpy() for k, v in gen_t.state_dict().items()
    })
    return cfg_t, cfg_n, tok_t, tok_n, gen_t, gen_n


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_tokenizer_and_generator_parity(kind):
    _, _, tok_t, tok_n, gen_t, gen_n = _pair(kind)
    data_np = np.random.default_rng(7).integers(0, 256, (2, 9), dtype=np.int64)
    data_t = torch.from_numpy(data_np)
    with torch.no_grad():
        logits_t, loss_t, codes_t = tok_t(data_t)
        logits_n, loss_n, codes_n = tok_n(ng.Tensor(data_np))
    np.testing.assert_allclose(loss_n.data, float(loss_t), rtol=1e-4, atol=1e-5)
    for a, b in zip(logits_t, logits_n):
        np.testing.assert_allclose(b.data, a.detach().numpy(), rtol=1e-4, atol=3e-4)
    for a, b in zip(codes_t, codes_n):
        np.testing.assert_array_equal(a.numpy(), b.data)

    codes_np = [c.data.copy() for c in codes_n]
    codes_t = [torch.from_numpy(c.copy()) for c in codes_np]
    with torch.no_grad():
        gl_t = gen_t(codes_t, corrupt=False)
        gl_n = gen_n([ng.Tensor(c) for c in codes_np], corrupt=False)
    for a, b in zip(gl_t, gl_n):
        np.testing.assert_allclose(b.data, a.detach().numpy(), rtol=1e-4, atol=3e-4)

    gen_t.zero_grad()
    gen_n.zero_grad(set_to_none=True)
    loss_t = ref.ncm_loss(gen_t, codes_t)
    loss_t.backward()
    loss_n = np_ncm.ncm_loss(gen_n, [ng.Tensor(c) for c in codes_np])
    loss_n.backward()
    np.testing.assert_allclose(loss_n.data, float(loss_t.detach()), rtol=1e-4, atol=1e-5)
    gt = {k: v.grad.detach().numpy() for k, v in gen_t.named_parameters()
          if v.grad is not None}
    gn = {k: v.grad for k, v in gen_n.named_parameters() if v.grad is not None}
    assert set(gt) == set(gn)
    for key in gt:
        np.testing.assert_allclose(gn[key], gt[key], rtol=2e-4, atol=1e-4, err_msg=key)


def test_chunked_loss_parity():
    _, _, _, _, gen_t, gen_n = _pair("moe")
    data_np = np.random.default_rng(11).integers(0, 256, (2, 9), dtype=np.int64)
    with torch.no_grad():
        codes_t = [torch.from_numpy(c.copy()) for c in
                   [x.data for x in gen_n.tokenizer.encode(ng.Tensor(data_np))]]
    gen_t.zero_grad()
    gen_n.zero_grad(set_to_none=True)
    loss_t = ref.backward_ncm_loss_chunked(gen_t, codes_t)
    loss_n = np_ncm.backward_ncm_loss_chunked(
        gen_n, [ng.Tensor(c.numpy()) for c in codes_t]
    )
    np.testing.assert_allclose(loss_n.data, float(loss_t.detach()), rtol=1e-4, atol=1e-5)
    gt = {k: v.grad.detach().numpy() for k, v in gen_t.named_parameters()
          if v.grad is not None}
    gn = {k: v.grad for k, v in gen_n.named_parameters() if v.grad is not None}
    for key in gt:
        np.testing.assert_allclose(gn[key], gt[key], rtol=2e-4, atol=1e-4, err_msg=key)
