import torch
import pytest

from smaulbrain import _native_opt, smaulopt


def _run(native, monkeypatch, shapes, wd):
    monkeypatch.setenv("SMAULBRAIN_NATIVE_OPT", "1" if native else "0")
    monkeypatch.setattr(_native_opt._backend, "_LIB", None)
    monkeypatch.setattr(_native_opt._backend, "_TRIED", False)
    rng = torch.Generator().manual_seed(3)
    params = {f"p{i}": torch.randn(s, generator=rng) for i, s in enumerate(shapes)}
    opt = smaulopt.SmaulOPT(lr=2e-3, weight_decay=wd); opt.register(params); norms = None
    for k in range(3):
        grads = {n: torch.randn(v.shape, generator=rng) * (k + 1) for n, v in params.items()}
        norms = opt.step(params, grads, collect_norms=True)
    return params, opt.state, norms


@pytest.mark.parametrize("wd", [0.0, 1e-3])
def test_native_factored_step_matches_torch(monkeypatch, wd):
    shapes = [(130, 301), (64, 64), (37, 41), (1, 2048), (2048, 1), (300,)]
    ref_p, ref_s, ref_n = _run(False, monkeypatch, shapes, wd)
    nat_p, nat_s, nat_n = _run(True, monkeypatch, shapes, wd)
    if _native_opt._backend._load() is None:
        pytest.skip("native optimizer kernel unavailable (no compiler)")
    for name in ref_p:
        assert torch.allclose(ref_p[name], nat_p[name], rtol=1e-5, atol=1e-6)
        for key in ("m", "R", "C", "v"):
            if key in ref_s[name]:
                assert torch.allclose(ref_s[name][key], nat_s[name][key], rtol=2e-4, atol=4e-8)
        assert ref_n[name] == pytest.approx(nat_n[name], rel=1e-4)


def test_native_factored_optimizer_declines_unsupported_weight_dtype():
    from smaulbrain import _native_opt

    opt = SmaulOPT()
    w = torch.zeros((32, 32), dtype=torch.bfloat16)
    st = {
        "m": torch.zeros_like(w),
        "R": torch.zeros(w.shape[0]),
        "C": torch.zeros(w.shape[1]),
    }
    g = torch.zeros(w.shape, dtype=torch.float32)
    assert _native_opt.factored_step(opt, w, g, st, collect_norms=False) is None
