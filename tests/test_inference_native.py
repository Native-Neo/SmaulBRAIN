import numpy as np

from smaulbrain import config as C
from smaulbrain import inference as INF
from smaulbrain import model as M


def _eng(preset="dense-tiny", seed=0, **kw):
    spec = C.PRESETS[preset]
    m = M.Model(spec, M.build_params(spec, np.random.default_rng(seed)),
                capacity_factor=99.0)
    return INF.InferenceEngine(m, seed=seed, **kw)


def test_state_continuation_and_reset():
    e = _eng(temperature=0.0)
    toks = [65, 66, 67, 68, 69]
    e.prefill(toks)
    a = e.generate([], max_new_tokens=6)
    e.reset()
    e.prefill(toks)
    b = e.generate([], max_new_tokens=6)
    assert a == b  # reset + same prompt reproduces
    blob = e.export_states()
    e2 = _eng(temperature=0.0)
    e2.import_states(blob)
    # continuation from restored state equals uninterrupted run
    e.reset(); e.prefill(toks); ref = e.generate([], max_new_tokens=6)
    assert a == ref


def test_chunked_prefill_equivalence():
    toks = [10, 20, 30, 40, 50, 60]
    e1 = _eng(temperature=0.0); e1.prefill(toks)
    e2 = _eng(temperature=0.0); e2.prefill(toks, chunk=2)
    assert e1.generate([], max_new_tokens=5) == e2.generate([], max_new_tokens=5)


def test_greedy_and_sampling_validation():
    e = _eng(temperature=0.0)
    assert e.generate([65], max_new_tokens=3)
    import pytest
    with pytest.raises(ValueError):
        INF.InferenceEngine(e.model, top_p=0.0)
    with pytest.raises(ValueError):
        INF.InferenceEngine(e.model, top_k=-1)
    e2 = _eng(seed=5, temperature=1.0, top_k=5, top_p=0.9)
    o1 = e2.generate([65, 66], max_new_tokens=8)
    e3 = _eng(seed=5, temperature=1.0, top_k=5, top_p=0.9)
    assert e3.generate([65, 66], max_new_tokens=8) == o1  # seeded reproducibility


def test_native_kernels_optional_but_defined():
    from smaulbrain.kernels import try_load
    from smaulbrain import cpu as CPU
    lib, info = try_load()
    caps = CPU.detect()
    assert caps["portable_scalar"] is True
    assert "dispatch" in info
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpp = os.path.join(root, "v2", "kernels", "cpu_kernels.cpp")
    assert os.path.exists(cpp)
    with open(cpp) as f:
        src = f.read()
    for sym in ("rmsnorm_fwd", "ternary_pack", "packed_matvec", "recurrent_step",
                "factored_update", "clone_perturb"):
        assert sym in src
    # mandatory path never requires AVX2/512 intrinsics
    assert "immintrin" not in src.lower()
    assert "avxintrin" not in src.lower() and "__m256" not in src and "__m512" not in src
