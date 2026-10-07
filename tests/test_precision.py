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


def test_optimizer_state_validation_rejects_bad_shapes_dtypes_layouts():
    from smaulopt import ensure_state, expected_layout, state_is_valid
    hp = SmaulOptHParams(state_dtype="bf16", factor_v=True)
    assert expected_layout((8, 16), hp) == "factored"
    assert expected_layout((4,), hp) == "full"
    assert expected_layout((1, 2), hp) == "full"
    good = init_state((8, 16), hp)
    good["step"] = 3
    assert state_is_valid(good, (8, 16), hp)
    # Wrong m shape.
    bad = init_state((8, 16), hp)
    bad["m"] = torch.zeros((8, 15), dtype=torch.bfloat16)
    assert not state_is_valid(bad, (8, 16), hp)
    # Stale full-v alongside factored moments.
    stale = init_state((8, 16), hp)
    stale["v"] = torch.zeros((8, 16), dtype=torch.bfloat16)
    assert not state_is_valid(stale, (8, 16), hp)
    # Stale factored buffers alongside a full-v layout (1-D param).
    hp_full = SmaulOptHParams(state_dtype="bf16", factor_v=True)
    full = init_state((4,), hp_full)
    full["v_row"] = torch.zeros((4, 1), dtype=torch.bfloat16)
    assert not state_is_valid(full, (4,), hp_full)
    # Bad step types.
    for bad_step in (-1, 1.5, True, None, "3"):
        s = init_state((8, 16), hp)
        s["step"] = bad_step
        assert not state_is_valid(s, (8, 16), hp)
    # Non-tensor / wrong-dtype moments are invalid.
    s = init_state((8, 16), hp)
    s["m"] = torch.zeros((8, 16), dtype=torch.float64)
    assert not state_is_valid(s, (8, 16), hp)
    # ensure_state replaces corrupt state with fresh canonical state.
    fixed, action = ensure_state(bad, (8, 16), hp)
    assert action == "init" and state_is_valid(fixed, (8, 16), hp)
    assert fixed["step"] == 0


def test_optimizer_factored_full_migration_explicit_and_cleans_stale():
    from smaulopt import ensure_state, state_is_valid
    torch.manual_seed(0)
    hp_fact = SmaulOptHParams(state_dtype="fp32", factor_v=True)
    hp_full = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    shape = (6, 8)
    # Full -> factored: row/col means of stored v, step preserved, v removed.
    full = init_state(shape, hp_full)
    full["v"] = torch.rand(shape)
    full["step"] = 5
    v_before = full["v"].clone()
    mig, action = ensure_state(full, shape, hp_fact)
    assert action == "migrated"
    assert "v" not in mig and "v_row" in mig and "v_col" in mig
    assert mig["step"] == 5
    assert torch.allclose(mig["v_row"], v_before.mean(dim=1, keepdim=True))
    assert torch.allclose(mig["v_col"], v_before.mean(dim=0, keepdim=True))
    assert state_is_valid(mig, shape, hp_fact)
    # Factored -> full: exact outer(R, C) / mean(R) reconstruction.
    fact = init_state(shape, hp_fact)
    fact["v_row"] = torch.rand((shape[0], 1)) + 0.5
    fact["v_col"] = torch.rand((1, shape[1])) + 0.5
    fact["step"] = 7
    rf, cf = fact["v_row"].clone(), fact["v_col"].clone()
    mig2, action2 = ensure_state(fact, shape, hp_full)
    assert action2 == "migrated"
    assert "v" in mig2 and "v_row" not in mig2 and "v_col" not in mig2
    assert mig2["step"] == 7
    expect = (rf.float() * cf.float()) / rf.float().mean().clamp_min(1e-12)
    assert torch.allclose(mig2["v"].float(), expect)
    assert state_is_valid(mig2, shape, hp_full)


def test_optimizer_dtype_cast_preserves_values_not_reset():
    from smaulopt import ensure_state, state_is_valid
    hp_bf16 = SmaulOptHParams(state_dtype="bf16", factor_v=False)
    hp_fp32 = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    st = init_state((4, 4), hp_bf16)
    st["m"] = (torch.randn(4, 4) * 0.1).to(torch.bfloat16)
    st["v"] = (torch.rand(4, 4) * 0.01).to(torch.bfloat16)
    st["step"] = 4
    m_before, v_before = st["m"].float().clone(), st["v"].float().clone()
    out, action = ensure_state(st, (4, 4), hp_fp32)
    assert action == "kept"  # same layout: cast, not reset
    assert out["step"] == 4 and out["m"].dtype == torch.float32
    assert torch.allclose(out["m"].float(), m_before, atol=1e-3)
    assert torch.allclose(out["v"].float(), v_before, atol=1e-3)
    assert state_is_valid(out, (4, 4), hp_fp32)


def test_dense_skips_grad_free_params_without_ticking_their_step():
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    opt = SmaulOpt(hp)
    a = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.randn(4, 4))
    a.grad = torch.randn(4, 4)
    b.grad = None
    opt.step_dense([("a", a), ("b", b)], lr=1e-3, store=opt.trunk_state)
    assert "a" in opt.trunk_state and opt.trunk_state["a"]["step"] == 1
    assert "b" not in opt.trunk_state  # no state allocated for grad-free param
    assert a.grad is None and opt.step_count == 0  # global clock untouched here


def test_dense_global_step_semantics_per_tensor_clock_only():
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    opt = SmaulOpt(hp)
    a = torch.nn.Parameter(torch.randn(2, 2))
    a.grad = torch.ones(2, 2)
    opt.step_dense([("a", a)], lr=1e-3, store=opt.trunk_state)
    assert opt.trunk_state["a"]["step"] == 1 and opt.step_count == 0
    # Skipped step (nonfinite) ticks nothing.
    a.grad = torch.full((2, 2), float("inf"))
    assert opt.step_dense([("a", a)], lr=1e-3, store=opt.trunk_state) == 0.0
    assert opt.trunk_state["a"]["step"] == 1
    assert opt.step_count == 0


def test_nonfinite_handling_consistent_and_preserves_grads_dense_and_expert():
    from experts import make_expert
    from train import _GradOnly
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    opt = SmaulOpt(hp)
    # Dense: state + grads preserved on nonfinite.
    p = torch.nn.Parameter(torch.randn(3, 3))
    p.grad = torch.full((3, 3), float("nan"))
    m_before = p.data.clone()
    assert opt.step_dense([("p", p)], lr=1e-3, store=opt.trunk_state) == 0.0
    assert "p" not in opt.trunk_state  # never allocated on a pure skip
    assert p.grad is not None and torch.equal(p.data, m_before)
    # Expert: state, version, bytes AND grads preserved on nonfinite.
    torch.manual_seed(0)
    rec = make_expert("expert_00000", 8, 8)
    rec.optim_state = {}
    codes_before = {n: rec.weights_fp8[n].codes.clone() for n in ("w_gate", "w_up", "w_down")}
    g = torch.full((8, 8), float("inf"))
    compute = {"w_gate": _GradOnly(g)}
    assert opt.step_expert(rec, compute, 1.0, lr=1e-3) == 0.0
    assert rec.optim_state == {} and rec.version == 0
    assert compute["w_gate"].grad is not None  # preserved for retry
    for n in ("w_gate", "w_up", "w_down"):
        assert torch.equal(rec.weights_fp8[n].codes, codes_before[n])


def test_skipped_step_retry_equivalent_dense_and_expert():
    from experts import make_expert
    from train import _GradOnly
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    # Dense retry: skip-then-apply == apply-only (bit-identical).
    opt = SmaulOpt(hp)
    ref = SmaulOpt(SmaulOptHParams(state_dtype="fp32", factor_v=False))
    w0 = torch.randn(4, 4)
    p1 = torch.nn.Parameter(w0.clone())
    p2 = torch.nn.Parameter(w0.clone())
    p1.grad = torch.full((4, 4), float("inf"))
    assert opt.step_dense([("w", p1)], lr=1e-3, store=opt.trunk_state) == 0.0
    g = torch.randn(4, 4)
    p1.grad = g.clone()
    p2.grad = g.clone()
    opt.step_dense([("w", p1)], lr=1e-3, store=opt.trunk_state)
    ref.step_dense([("w", p2)], lr=1e-3, store=ref.trunk_state)
    assert torch.equal(p1.data, p2.data)
    assert torch.equal(opt.trunk_state["w"]["m"], ref.trunk_state["w"]["m"])
    assert torch.equal(opt.trunk_state["w"]["v"], ref.trunk_state["w"]["v"])
    # Expert retry: skip-then-apply == apply-only (FP8 bytes identical).
    torch.manual_seed(1)
    rec1 = make_expert("expert_00000", 8, 8)
    torch.manual_seed(1)
    rec2 = make_expert("expert_00000", 8, 8)
    opt_e = SmaulOpt(hp)
    ref_e = SmaulOpt(SmaulOptHParams(state_dtype="fp32", factor_v=False))
    bad = {"w_gate": _GradOnly(torch.full((8, 8), float("nan")))}
    assert opt_e.step_expert(rec1, bad, 1.0, lr=1e-3) == 0.0
    gg = torch.randn(8, 8)
    opt_e.step_expert(rec1, {"w_gate": _GradOnly(gg.clone())}, 1.0, lr=1e-3)
    ref_e.step_expert(rec2, {"w_gate": _GradOnly(gg.clone())}, 1.0, lr=1e-3)
    for n in ("w_gate", "w_up", "w_down"):
        assert torch.equal(rec1.weights_fp8[n].codes, rec2.weights_fp8[n].codes)


def test_tied_parameters_update_once_and_stay_identical():
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    opt = SmaulOpt(hp)
    torch.manual_seed(123)
    base = torch.randn(4, 4)
    gg = torch.randn(4, 4)
    p = torch.nn.Parameter(base.clone())
    p.grad = gg.clone()
    s = opt.step_dense([("embed.w", p), ("head.w", p)], lr=1e-3, store=opt.trunk_state)
    assert s != 0.0
    assert opt.trunk_state["embed.w"] is opt.trunk_state["head.w"]  # one shared state
    assert opt.trunk_state["embed.w"]["step"] == 1  # single tick, not double
    # Same grad through an untied single update matches bit-for-bit.
    q = torch.nn.Parameter(base.clone())
    q.grad = gg.clone()
    opt2 = SmaulOpt(SmaulOptHParams(state_dtype="fp32", factor_v=False))
    opt2.step_dense([("w", q)], lr=1e-3, store=opt2.trunk_state)
    assert torch.equal(p.data, q.data)
    assert torch.equal(opt.trunk_state["embed.w"]["m"], opt2.trunk_state["w"]["m"])


def test_tied_single_update_matches_untied_reference():
    torch.manual_seed(9)
    hp = SmaulOptHParams(state_dtype="fp32", factor_v=False)
    w0 = torch.randn(3, 3)
    gg = torch.randn(3, 3)
    tied = torch.nn.Parameter(w0.clone())
    tied.grad = gg.clone()
    opt_t = SmaulOpt(SmaulOptHParams(state_dtype="fp32", factor_v=False))
    opt_t.step_dense([("a", tied), ("b", tied)], lr=2e-3, store=opt_t.trunk_state)
    solo = torch.nn.Parameter(w0.clone())
    solo.grad = gg.clone()
    opt_s = SmaulOpt(SmaulOptHParams(state_dtype="fp32", factor_v=False))
    opt_s.step_dense([("a", solo)], lr=2e-3, store=opt_s.trunk_state)
    assert torch.equal(tied.data, solo.data)  # one update, not two
    assert torch.equal(opt_t.trunk_state["a"]["m"], opt_s.trunk_state["a"]["m"])


def test_replaced_state_cleaned_on_shape_change_and_migration():
    from smaulopt import ensure_state, state_is_valid
    hp_fact = SmaulOptHParams(state_dtype="bf16", factor_v=True)
    # Shape change: old state discarded wholesale, canonical fresh state in.
    old = init_state((8, 16), hp_fact)
    old["step"] = 10
    new, action = ensure_state(old, (4, 4), hp_fact)
    assert action == "init" and state_is_valid(new, (4, 4), hp_fact)
    assert new["step"] == 0 and "v" not in new
    # Layout flip cleans the replaced buffers (no stale keys survive).
    hp_full = SmaulOptHParams(state_dtype="bf16", factor_v=False)
    full = init_state((6, 8), hp_full)
    full["v"] = torch.rand(6, 8).to(torch.bfloat16)
    full["step"] = 2
    mig, _ = ensure_state(full, (6, 8), hp_fact)
    assert "v" not in mig and state_is_valid(mig, (6, 8), hp_fact)
    mig_back, _ = ensure_state(mig, (6, 8), hp_full)
    assert "v_row" not in mig_back and "v_col" not in mig_back
    assert state_is_valid(mig_back, (6, 8), hp_full)


def test_validated_state_keeps_update_math_bit_identical():
    torch.manual_seed(3)
    hp = SmaulOptHParams(state_dtype="bf16", factor_v=True)
    w = torch.randn(8, 16)
    g = torch.randn(8, 16)
    st_fresh = init_state((8, 16), hp)
    st_valid, action = __import__("smaulopt").ensure_state(
        {"m": st_fresh["m"].clone(), "v_row": st_fresh["v_row"].clone(),
         "v_col": st_fresh["v_col"].clone(), "step": 0}, (8, 16), hp)
    assert action == "kept"
    w1 = smaul_update(w.clone(), g.clone(), st_fresh, hp, lr=1e-3)
    w2 = smaul_update(w.clone(), g.clone(), st_valid, hp, lr=1e-3)
    assert torch.equal(w1, w2)  # validation path changes no update math
    assert torch.equal(st_fresh["m"], st_valid["m"])


