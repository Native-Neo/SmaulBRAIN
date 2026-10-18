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

    # V2 owns a focused packed-ternary kernel. V1's broad kernels.cpp path is
    # intentionally not required by this independent implementation.
    from pathlib import Path
    import smaulbrain
    cpp = Path(smaulbrain.__file__).resolve().parent / "_ternary_kernels.cpp"
    assert cpp.is_file()
    src = cpp.read_text()
    for sym in ("smaulbrain_ternary_matmat",
                "smaulbrain_ternary_transpose_matmat"):
        assert sym in src

    # The optional fast path is AVX1 only. AVX2/AVX-512 and -march=native
    # are forbidden; dispatch selects the AVX1 build only when the OS/CPU
    # advertises AVX, otherwise the SSE2/scalar build remains usable.
    assert "__AVX2__" not in src
    assert "__m256i" not in src and "__m512" not in src
    assert "_mm256_i32gather" not in src
