import numpy as np
import pytest

from smaulbrain import config as C
from smaulbrain import model as M
from smaulbrain import vocab as V


def test_byte_round_trip_all_values():
    raw = bytes(range(256))
    ids = V.encode_bytes(raw)
    assert ids == list(range(256))
    assert V.decode_ids(ids) == raw


def test_byte_round_trip_malformed_utf8():
    raw = b"\xff\xfehello\x80\xc3"
    assert V.decode_ids(V.encode_bytes(raw)) == raw


def test_vocab_metadata_in_checkpoint(tmp_path):
    import numpy as _np
    from smaulbrain import checkpoint as _ckpt
    man = _ckpt.build_manifest("dense-tiny", {}, {}, _ckpt.blank_scheduler_state(),
                               0, {"bit_generator": _np.random.default_rng(0).bit_generator.state},
                               0, [], {})
    assert man["vocab"]["vocab_size"] == 259
    assert man["vocab"]["bos_id"] == 256


@pytest.mark.parametrize("preset,expected", [
    ("dense-tiny", 256_000), ("dense-full", 256_000_000)])
def test_dense_exact_counts(preset, expected):
    spec = C.PRESETS[preset]
    rng = np.random.default_rng(0)
    if preset == "dense-full":
        pytest.skip("covered by test_dense_full_count_exact (memory-heavy)")
    p = M.build_params(spec, rng)
    assert M.count_params_dict(p) == expected
    M.Model(spec, p)  # init asserts


@pytest.mark.parametrize("preset,expected", [
    ("moe-tiny", 10_240), ("moe-full", 5_120_000)])
def test_expert_exact_counts(preset, expected):
    spec = C.PRESETS[preset]
    assert C.expert_param_count(spec) == expected
    assert spec.n_experts_init == (32 if preset == "moe-tiny" else 64)
    assert spec.top_k == (2 if preset == "moe-tiny" else 8)


def test_dense_full_count_exact():
    spec = C.PRESETS["dense-full"]
    assert C.dense_param_total(spec) == 256_000_000
    assert (spec.vocab_size, spec.d_model) == (259, 1000)


def test_moe_full_expert_count_exact():
    spec = C.PRESETS["moe-full"]
    assert C.expert_param_count(spec) == 5_120_000
    assert spec.n_experts_init == 64 and spec.top_k == 8


def test_invalid_presets_and_flags():
    with pytest.raises(ValueError):
        C.resolve_preset("full", False, False)
    with pytest.raises(ValueError):
        C.resolve_preset("moe-full", False, True)
    with pytest.raises(ValueError):
        C.resolve_preset("bogus", True, False)
    with pytest.raises(ValueError):
        C.resolve_preset("tiny", True, True)
    with pytest.raises(ValueError):
        C.parse_growth_every_steps(-1)
    with pytest.raises(ValueError):
        C.parse_growth_loss_nats("half")
    assert C.parse_growth_loss_nats("none") is None
    assert C.parse_growth_every_steps("0") == 0


def test_alias_forms():
    assert C.resolve_preset("full", True, False) == "moe-full"
    assert C.resolve_preset("tiny", False, True) == "dense-tiny"
    assert C.resolve_preset("moe-tiny", False, False) == "moe-tiny"
