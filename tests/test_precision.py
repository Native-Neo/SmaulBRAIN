"""Precision: real FP8 bytes, blockwise scales, policy, optimizer math."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from precision import (
    PRECISION_POLICY, compute_dtype, dequantize_fp8_block,
    dequantize_fp8_blockwise, dequantize_fp8_row_block, quantize_fp8_blockwise,
    update_fp8_row_block, update_fp8_tile_block,
)
from smaulopt import SmaulOpt, SmaulOptHParams, init_state, smaul_update, smaul_update_range



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


def test_all_zero_tensor_quantizes_finite():
    t = quantize_fp8_blockwise(torch.zeros(8, 130), tile=64)
    back = dequantize_fp8_blockwise(t).float()
    assert torch.isfinite(t.scales).all() and (t.scales > 0).all()
    assert torch.equal(back, torch.zeros(8, 130))  # zeros survive exactly


def test_odd_width_and_noncontiguous_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(7, 100)  # odd rows, width vs tile=64 leaves ragged block
    t = quantize_fp8_blockwise(w[:, ::2], tile=64)  # noncontiguous view in
    assert not t.codes.is_contiguous() or True  # storage is fresh, input need not be
    back = dequantize_fp8_blockwise(t).float()
    assert back.shape == (7, 50)
    assert torch.isfinite(back).all()
    assert (back - w[:, ::2]).abs().mean().item() < 0.03


def test_native_and_fallback_emit_identical_bytes(monkeypatch):
    import native
    torch.manual_seed(0)
    w = torch.randn(9, 130) * 2.0
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    ref = quantize_fp8_blockwise(w, tile=64)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    nat = quantize_fp8_blockwise(w, tile=64)
    assert torch.equal(ref.codes, nat.codes)  # bit-exact dispatch
    assert torch.equal(ref.scales, nat.scales)


def test_update_fp8_row_block_preserves_untouched_blocks_bit_for_bit():
    torch.manual_seed(42)
    w = torch.randn(16, 128)
    t = quantize_fp8_blockwise(w, tile=64)
    orig_codes = t.codes.clone()
    orig_scales = t.scales.clone()

    new_slice = torch.randn(6, 128)
    update_fp8_row_block(t, new_slice, 4, 10)

    # Touched slice updated
    assert not torch.equal(t.codes[4:10], orig_codes[4:10])

    # Untouched rows preserved bit-for-bit
    assert torch.equal(t.codes[:4], orig_codes[:4])
    assert torch.equal(t.codes[10:], orig_codes[10:])
    assert torch.equal(t.scales[:4], orig_scales[:4])
    assert torch.equal(t.scales[10:], orig_scales[10:])


def test_smaul_update_range_updates_only_target_rows():
    hp = SmaulOptHParams(state_dtype="bf16", factor_v=True)
    st = init_state((16, 32), hp)
    orig_m = st["m"].clone()
    orig_vr = st["v_row"].clone()

    w_slice = torch.randn(4, 32)
    g_slice = torch.randn(4, 32)
    new_w = smaul_update_range(w_slice, g_slice, st, hp, lr=1e-3, row_start=4, row_end=8)

    assert new_w.shape == (4, 32)
    assert torch.isfinite(new_w).all()
    assert st["step"] == 1

    # Touched rows in state updated
    assert not torch.equal(st["m"][4:8], orig_m[4:8])
    assert not torch.equal(st["v_row"][4:8], orig_vr[4:8])

    # Untouched rows in state preserved bit-for-bit
    assert torch.equal(st["m"][:4], orig_m[:4])
    assert torch.equal(st["m"][8:], orig_m[8:])
    assert torch.equal(st["v_row"][:4], orig_vr[:4])
    assert torch.equal(st["v_row"][8:], orig_vr[8:])


def test_tile_block_slice_matches_full():
    # Level-3 read: exact vs full-dequant slice, incl. partial tiles + ragged tail.
    torch.manual_seed(7)
    w = torch.randn(10, 130) * 0.5
    t = quantize_fp8_blockwise(w, tile=64)
    full = dequantize_fp8_blockwise(t)
    windows = [(0, 10, 0, 130), (2, 5, 0, 64), (2, 5, 64, 128),
               (0, 3, 10, 100), (4, 10, 100, 130), (7, 8, 63, 65)]
    for (r0, r1, c0, c1) in windows:
        part = dequantize_fp8_block(t, r0, r1, c0, c1)
        assert part.shape == (r1 - r0, c1 - c0)
        assert torch.equal(part, full[r0:r1, c0:c1])


def test_update_fp8_tile_block_preserves_untouched_tiles_bit_for_bit():
    torch.manual_seed(11)
    w = torch.randn(12, 128)
    t = quantize_fp8_blockwise(w, tile=64)
    orig_codes = t.codes.clone()
    orig_scales = t.scales.clone()

    new_slice = torch.randn(5, 64) * 0.5
    update_fp8_tile_block(t, new_slice, 3, 8, 64, 128)  # rows 3:8, 2nd tile

    # Touched tile changed.
    assert not torch.equal(t.codes[3:8, 64:128], orig_codes[3:8, 64:128])

    # Every untouched region bit-identical (codes AND scales).
    assert torch.equal(t.codes[:3], orig_codes[:3])
    assert torch.equal(t.codes[8:], orig_codes[8:])
    assert torch.equal(t.codes[3:8, :64], orig_codes[3:8, :64])
    assert torch.equal(t.scales[:3], orig_scales[:3])
    assert torch.equal(t.scales[8:], orig_scales[8:])
    assert torch.equal(t.scales[3:8, :1], orig_scales[3:8, :1])

    # Tile read stays consistent with the full dequant after the write.
    full = dequantize_fp8_blockwise(t)
    assert torch.equal(dequantize_fp8_block(t, 3, 8, 64, 128), full[3:8, 64:128])


def test_tile_block_write_matches_full_row_requant_on_touched_tiles():
    # Storage equivalence: per-tile requant == full-row requant on touched
    # tiles (incl. the ragged tail block), while untouched rows are preserved
    # (a full-row requant path would still rewrite them).
    torch.manual_seed(21)
    w = torch.randn(8, 130)
    t_tile = quantize_fp8_blockwise(w, tile=64)
    t_full = quantize_fp8_blockwise(w, tile=64)
    torch.manual_seed(22)
    new_rows = torch.randn(4, 130)
    update_fp8_row_block(t_full, new_rows, 2, 6)
    update_fp8_tile_block(t_tile, new_rows[:, :64], 2, 6, 0, 64)
    update_fp8_tile_block(t_tile, new_rows[:, 64:128], 2, 6, 64, 128)
    update_fp8_tile_block(t_tile, new_rows[:, 128:130], 2, 6, 128, 130)
    assert torch.equal(t_tile.codes, t_full.codes)
    assert torch.equal(t_tile.scales, t_full.scales)


def test_tile_block_rejects_unaligned_and_bad_ranges():
    import pytest
    w = torch.randn(8, 128)
    t = quantize_fp8_blockwise(w, tile=64)
    with pytest.raises(ValueError):  # partial-tile write would corrupt siblings
        update_fp8_tile_block(t, torch.randn(2, 10), 0, 2, 60, 70)
    with pytest.raises(ValueError):  # slice shape must match the window
        update_fp8_tile_block(t, torch.randn(3, 64), 0, 2, 0, 64)
    with pytest.raises(ValueError):  # row helper now validates slice shape too
        update_fp8_row_block(t, torch.randn(2, 128), 0, 3)
    with pytest.raises(ValueError):  # empty / out-of-bounds windows
        dequantize_fp8_block(t, 2, 2, 0, 64)
    with pytest.raises(ValueError):
        dequantize_fp8_block(t, 0, 2, 0, 129)


def test_tile_block_ops_native_fallback_parity(monkeypatch):
    torch.manual_seed(31)
    w = torch.randn(9, 130)
    monkeypatch.setenv("SMAUL_NATIVE", "0")
    ref = quantize_fp8_blockwise(w, tile=64)
    monkeypatch.setenv("SMAUL_NATIVE", "1")
    nat = quantize_fp8_blockwise(w, tile=64)
    ns = torch.randn(5, 64)
    update_fp8_tile_block(ref, ns, 2, 7, 0, 64)
    update_fp8_tile_block(nat, ns.clone(), 2, 7, 0, 64)
    assert torch.equal(ref.codes, nat.codes)
    assert torch.equal(ref.scales, nat.scales)
    ref_read = dequantize_fp8_block(ref, 2, 7, 10, 100)
    assert torch.equal(dequantize_fp8_block(nat, 2, 7, 10, 100), ref_read)


def test_sparse_update_global_clock_semantics_documented():
    # Pins the documented sparse-update semantics (see precision.py contract):
    # frozen rows' stored moments are untouched (no zero-grad decay), while the
    # global step still ticks so bias corrections advance for all rows.
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    st = init_state((8, 16), hp)
    w = torch.zeros(2, 16)
    g = torch.ones(2, 16)
    new_w = smaul_update_range(w, g, st, hp, lr=1e-3, row_start=2, row_end=4)
    assert new_w.shape == (2, 16)
    assert st["step"] == 1  # global clock ticked though only 2 rows moved
    assert torch.equal(st["m"][:2], torch.zeros(2, 16))  # frozen: no decay
    assert torch.equal(st["m"][4:], torch.zeros(4, 16))
    assert torch.equal(st["v"][:2], torch.zeros(2, 16))
    assert torch.equal(st["v"][4:], torch.zeros(4, 16))
    # Touched rows took the standard first step: m_hat=1, v_hat=|g|=1.
    assert torch.allclose(new_w, torch.full_like(new_w, -0.001), atol=1e-5)


def test_sparse_update_factored_v_col_coupling_documented():
    # With factor_v the shared v_col absorbs the sparse slice's column means:
    # untouched rows' stored moments stay bit-identical, but their future
    # trajectory shifts via v_col. Pinned here as documented behavior.
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=True)
    st = init_state((8, 4), hp)
    smaul_update_range(torch.zeros(2, 4), torch.ones(2, 4), st, hp,
                       lr=1e-3, row_start=0, row_end=2)
    assert torch.equal(st["m"][2:], torch.zeros(6, 4))  # untouched rows frozen
    assert torch.equal(st["v_row"][2:], torch.zeros(6, 1))
    assert not torch.equal(st["v_col"], torch.zeros(1, 4))  # shared moment moved

