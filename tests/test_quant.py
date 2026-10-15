import numpy as np
import pytest

from smaulbrain import config as C
from smaulbrain import model as M
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
    m.forward(toks)  # same objects -> reuse, no rebuilds
    assert m.packed.builds == builds
    assert m.packed.reuse_hits > 0
    m.packed.invalidate()
    m.forward(toks)
    assert m.packed.builds > builds
