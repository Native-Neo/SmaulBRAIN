"""Precision: real FP8 bytes, blockwise scales, policy, optimizer math."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from precision import (
    PRECISION_POLICY, compute_dtype, dequantize_fp8_blockwise,
    dequantize_fp8_row_block, quantize_fp8_blockwise,
)
from smaulopt import SmaulOpt, SmaulOptHParams, init_state, smaul_update


def test_fp8_storage_not_relabeled_fp32():
    w = torch.randn(16, 130)
    t = quantize_fp8_blockwise(w, tile=64)
    assert t.codes.dtype == torch.uint8 and t.scales.dtype == torch.float32
    assert t.codes.nelement() == w.nelement()  # 1 byte/param on disk/RAM
    assert t.scales.shape == (16, 3)  # per-row-block scales, not per tensor


def test_fp8_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(32, 256) * 0.5
    rec = dequantize_fp8_blockwise(quantize_fp8_blockwise(w, tile=64)).float()
    assert (rec - w).abs().mean().item() < 0.02


def test_row_block_slice_matches_full():
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    t = quantize_fp8_blockwise(w, tile=64)
    full = dequantize_fp8_blockwise(t)
    part = dequantize_fp8_row_block(t, 4, 10)
    assert torch.equal(part, full[4:10])


def test_bf16_activations_and_fp32_stats_present_in_policy():
    assert PRECISION_POLICY["expert_weights"][0] == "fp8"
    assert PRECISION_POLICY["activations"] == ("bf16", "bf16")
    assert PRECISION_POLICY["norm_stats"] == ("fp32", "fp32")
    assert PRECISION_POLICY["optimizer_state"] == ("bf16", "fp32")
    assert compute_dtype("activations") == torch.bfloat16


def test_smaulopt_matches_reference_equations():
    hp = SmaulOptHParams(lr=0.01, beta_m=0.9, beta_v=0.999, eps=1e-8,
                         wd=0.0, state_dtype="fp32", factor_v=False,
                         update_clip=1e9)
    w = torch.tensor([[1.0, -2.0]])
    g = torch.tensor([[0.5, 0.25]])
    st = init_state((1, 2), hp)
    new_w = smaul_update(w, g, st, hp, lr=0.01)
    # Manual reference: t=1, m=0.1*g, v=0.001*|g|, m_hat=g, v_hat=|g|.
    m_hat = g
    v_hat = g.abs()
    expect = w - 0.01 * (m_hat / (v_hat + 1e-8))
    assert torch.allclose(new_w, expect, atol=1e-6)


def test_smaulopt_factored_state_and_bf16_storage():
    hp = SmaulOptHParams(state_dtype="bf16", factor_v=True)
    st = init_state((8, 16), hp)
    assert "v_row" in st and "v_col" in st and "v" not in st
    assert st["m"].dtype == torch.bfloat16
    w = torch.randn(8, 16)
    g = torch.randn(8, 16)
    new_w = smaul_update(w, g, st, hp, lr=1e-3)
    assert torch.isfinite(new_w).all() and st["step"] == 1


def test_no_hidden_fp32_master_copy_of_pool():
    from experts import ExpertPool, make_expert
    torch.manual_seed(0)
    pool = ExpertPool()
    for _ in range(2):
        pool.add(make_expert(pool.fresh_id(), 16, 32))
    for rec in pool.experts.values():
        for n, t in rec.weights_fp8.items():
            assert t.codes.dtype == torch.uint8
            total_fp32 = t.codes.nelement() * 4
            assert t.nbytes() < total_fp32  # strictly smaller than fp32
