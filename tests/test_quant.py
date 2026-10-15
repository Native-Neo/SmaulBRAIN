import numpy as np
import pytest

from smaulbrain import config as C
from smaulbrain import model as M
from smaulbrain import moe as _moe
from smaulbrain import quant as Q


def _tiny_moe(quant="fp32", backward="fp32"):
    spec = C.PRESETS["moe-tiny"]
    p = M.build_params(spec, np.random.default_rng(0))
    return M.Model(spec, p, forward_quant=quant, backward_mode=backward)


def test_pack_unpack_roundtrip_and_scale():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 12)).astype(np.float32)
    packed, scale, shape, th = Q.pack_ternary(W)
    assert shape == (16, 12) and scale > 0 and th >= 0
    Wq = Q.unpack_ternary(packed, shape, scale)
    assert set(np.unique(np.round(Wq / scale)).tolist()) <= {-1.0, 0.0, 1.0}


def test_packed_matvec_matches_scalar_reference():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((9, 7)).astype(np.float32)
    packed, scale, shape, _ = Q.pack_ternary(W)
    x = rng.standard_normal(9).astype(np.float32)
    a = Q.packed_matvec_scalar(x, packed, shape, scale)
    b = Q.packed_matvec(x, packed, shape, scale)
    assert np.allclose(a, b, atol=1e-5)


def test_forbidden_dense_path_raises():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((4, 4)).astype(np.float32)
    packed, scale, shape, _ = Q.pack_ternary(W)
    with pytest.raises(RuntimeError):
        Q.packed_matvec(np.ones(4, dtype=np.float32), packed, shape, scale,
                        allow_dense_reconstruction=True)


@pytest.mark.parametrize("f,b", [("ternary", "fp32"), ("ternary", "ternary"),
                                 ("ternary", "fp8"), ("fp8", "fp8"), ("fp32", "fp32")])
def test_supported_quant_combos_run(f, b):
    m = _tiny_moe(f, b)
    toks = [10, 20, 30, 40, 50]
    logits, _ = m.forward(toks)
    assert np.all(np.isfinite(logits))
    loss, grads, _ = m.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert any(np.abs(g).sum() > 0 for g in grads.values())


def test_unsupported_combo_rejected():
    with pytest.raises(ValueError):
        C.validate_quant_combo("fp8", "fp32")
    with pytest.raises(ValueError):
        C.validate_quant_combo("fp32", "ternary")


def test_native_ternary_packer_matches_numpy_reference():
    rng = np.random.default_rng(237)
    for shape in ((12, 32), (3, 5), (17,), (4, 4)):
        W = rng.standard_normal(shape).astype(np.float32)
        expected = Q._pack_ternary_numpy(W)
        actual = Q.pack_ternary(W)
        assert np.array_equal(actual[0], expected[0])
        assert actual[2] == expected[2]
        assert np.isclose(actual[1], expected[1], rtol=2e-6, atol=1e-7)
        assert np.isclose(actual[3], expected[3], rtol=2e-6, atol=1e-7)

    # Non-finite inputs must not break the kernel or generate invalid codes.
    W = np.array([1.0, -3.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    expected = Q._pack_ternary_numpy(W)
    actual = Q.pack_ternary(W)
    assert np.array_equal(actual[0], expected[0])
    assert np.isclose(actual[1], expected[1], equal_nan=True)
    assert np.isclose(actual[3], expected[3], equal_nan=True)


def test_batch_one_packed_matmat_fast_path_matches_dense():
    rng = np.random.default_rng(701)
    W = rng.standard_normal((17, 64)).astype(np.float32)
    x = rng.standard_normal(17).astype(np.float32)
    packed, scale, shape, _ = Q.pack_ternary(W)
    dense_q = Q.unpack_ternary(packed, shape, scale)
    actual = Q.packed_matmat(x, packed, shape, scale)
    expected = x @ dense_q
    assert actual.shape == (64,)
    assert np.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_moe_forward_skips_unused_diagnostic_reductions(monkeypatch):
    m = _tiny_moe("ternary", "ternary")

    def forbidden(*args, **kwargs):
        raise AssertionError("unused MoE diagnostics ran in the forward hot path")

    monkeypatch.setattr(_moe, "balance_aux_loss", forbidden)
    monkeypatch.setattr(_moe, "expert_output_stats", forbidden)
    logits, _ = m.forward([10, 20, 30], train=False)
    assert np.all(np.isfinite(logits))


def test_packed_matmat_and_transpose_match_dense_reference():
    rng = np.random.default_rng(91)
    # Output width 32 exercises the AVX1/SSE2 packed kernel; batch 5
    # also exercises the odd final token in the two-row prefill tile.
    W = rng.standard_normal((12, 32)).astype(np.float32)
    X = rng.standard_normal((5, 12)).astype(np.float32)
    dY = rng.standard_normal((5, 32)).astype(np.float32)
    packed, scale, shape, _ = Q.pack_ternary(W)
    dense_q = Q.unpack_ternary(packed, shape, scale)
    expected_y = X @ dense_q
    expected_dx = dY @ dense_q.T
    actual_y = Q.packed_matmat(X, packed, shape, scale)
    actual_dx = Q.packed_matmat_transpose(dY, packed, shape, scale)
    assert np.allclose(actual_y, expected_y, atol=2e-5, rtol=2e-5)
    assert np.allclose(actual_dx, expected_dx, atol=2e-5, rtol=2e-5)


def test_model_ternary_path_never_unpacks_dense_weights(monkeypatch):
    m = _tiny_moe("ternary", "ternary")
    toks = [10, 20, 30, 40, 50]

    def forbidden(*args, **kwargs):
        raise AssertionError("model path attempted to unpack a packed ternary matrix")

    monkeypatch.setattr(Q, "unpack_codes", forbidden)
    logits, _ = m.forward(toks, train=False)
    assert np.all(np.isfinite(logits))
    loss, grads, _ = m.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(g)) for g in grads.values())


def _legacy_fp8_scalar_reference(W, scale=None):
    Wf = W.astype(np.float64)
    if scale is None:
        finite_abs = np.abs(Wf[np.isfinite(Wf)])
        mx = float(np.max(finite_abs)) if finite_abs.size else 0.0
        scale = np.float32(mx / Q.FP8_E4M3_MAX) if mx > 0 else np.float32(1.0)
    if scale == 0:
        scale = np.float32(1.0)
    norm = Wf / float(scale)
    out = np.empty_like(norm)
    values = []
    for e in range(-8, 9):
        base = 2.0 ** e
        for m in range(8):
            cand = base * (1 + m / 8.0) if e > -8 else base * (m / 8.0)
            values.append(cand)
    for idx, v in np.ndenumerate(norm):
        if not np.isfinite(v):
            out[idx] = v
            continue
        sign = -1.0 if v < 0 else 1.0
        a = abs(v)
        if a > Q.FP8_E4M3_MAX:
            out[idx] = sign * Q.FP8_E4M3_MAX
            continue
        best = 0.0
        for cand in values:
            if abs(cand - a) < abs(best - a):
                best = cand
        out[idx] = sign * best
    return (out * float(scale)).astype(np.float32), np.float32(scale)


def test_fp8_vectorized_conversion_matches_scalar_reference():
    rng = np.random.default_rng(42)
    W = np.concatenate([
        rng.standard_normal(257).astype(np.float32) * 12,
        np.array([0.0, -0.0, 1e-30, -1e-30, 1.0, -1.0,
                  448.0, -448.0, 500.0, -500.0, np.nan, np.inf, -np.inf],
                 dtype=np.float32),
    ])
    got, got_scale = Q.fp8_e4m3_quantize(W)
    expected, expected_scale = _legacy_fp8_scalar_reference(W)
    assert got_scale == expected_scale
    assert np.allclose(got, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_fp8_auto_scale_ignores_nonfinite_values():
    W = np.array([1.0, -3.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    Wq, scale = Q.fp8_e4m3_quantize(W)
    assert np.isfinite(scale) and scale > 0
    assert np.isfinite(Wq[0]) and np.isfinite(Wq[1])
    assert np.isnan(Wq[2])
    assert np.isposinf(Wq[3])
    assert np.isneginf(Wq[4])


def test_fp8_format_properties():
    W = np.array([0.0, 1.0, -1.0, 1000.0, -1000.0, np.nan], dtype=np.float32)
    Wq, scale = Q.fp8_e4m3_quantize(W)
    assert scale > 0
    assert abs(float(Wq[1]) - 1.0) / 1.0 < 0.1  # small values near-exact
    assert abs(float(Wq[3])) <= Q.FP8_E4M3_MAX * float(scale) + 1e-6  # saturation
    assert np.isnan(Wq[5])  # exceptional values preserved
    assert Q.fp8_error_bound(float(scale)) >= 0


def test_cache_reuse_and_invalidation():
    m = _tiny_moe("ternary", "fp32")
    toks = [1, 2, 3, 4]
    m.forward(toks)
    builds = m.packed.builds
    cache_keys = set(m.packed.d)
    m.forward(toks)  # same objects -> reuse, no rebuilds
    assert m.packed.builds == builds
    assert m.packed.reuse_hits > 0
    # Selective invalidation keeps packed projections whose params did not change.
    m.packed.invalidate(["L0.attn.Wr"])
    assert "L0.attn.Wr" not in m.packed.d
    assert cache_keys - {"L0.attn.Wr"} <= set(m.packed.d)
    m.packed.invalidate()
    m.forward(toks)
    assert m.packed.builds > builds


def test_fp8_weight_cache_reuses_and_selectively_invalidates_quantization():
    m = _tiny_moe("fp8", "fp8")
    toks = [1, 2, 3, 4]
    m.forward(toks)
    builds = m.fp8.builds
    assert builds > 0
    m.forward(toks)
    assert m.fp8.builds == builds
    assert m.fp8.reuse_hits > 0
    key = "L0.attn.Wr"
    m.fp8.invalidate([key])
    before = m.fp8.builds
    m.forward(toks)
    assert m.fp8.builds == before + 1
