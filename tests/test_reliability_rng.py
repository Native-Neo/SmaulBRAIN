"""M3 regression: set_rng_snapshot validation (test-first).

SECURITY owns storage.py — this file only pins the contract. Pre-merge,
gauss inf/nan/"x" and bool inner pass through (corrupting global RNG)
and -1 raises OverflowError instead of ValueError; version/empty/length
cases already refuse.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

import storage


def _good():
    return storage.get_rng_snapshot()


def _with_python(version=None, inner=None, gauss="keep"):
    snap = _good()
    v0, i0, g0 = snap["python"]
    v = v0 if version is None else version
    i = i0 if inner is None else inner
    g = g0 if gauss == "keep" else gauss
    out = dict(snap)
    out["python"] = (v, i, g)
    return out


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), "x", 1, True])
def test_gauss_rejects_nonfinite_or_nonstr(bad):
    with pytest.raises(ValueError):
        storage.set_rng_snapshot(_with_python(gauss=bad))


def test_gauss_none_accepted():
    # None is the only valid non-float gauss (cleared cache).
    snap = _with_python(gauss=None)
    storage.set_rng_snapshot(snap)


@pytest.mark.parametrize("n", [624, 626])
def test_inner_wrong_length_raises(n):
    with pytest.raises(ValueError):
        storage.set_rng_snapshot(_with_python(inner=tuple([1] * n)))


def test_inner_bool_rejects():
    with pytest.raises(ValueError):
        storage.set_rng_snapshot(_with_python(inner=tuple([True] * 625)))


def test_inner_out_of_range_rejects():
    with pytest.raises(ValueError):
        storage.set_rng_snapshot(_with_python(inner=tuple([2 ** 32] * 625)))
    with pytest.raises(ValueError):
        storage.set_rng_snapshot(_with_python(inner=tuple([-1] * 625)))


@pytest.mark.parametrize("bad", [True, "3", 1, 999])
def test_version_bool_or_str_rejects(bad):
    with pytest.raises(ValueError):
        storage.set_rng_snapshot(_with_python(version=bad))


def test_empty_dict_raises():
    with pytest.raises(ValueError):
        storage.set_rng_snapshot({})


def test_checkpoint_rng_parts_rejects_unknown_tensor_keys():
    snap = storage.get_rng_snapshot()
    tensors = {"torch_cpu": snap["torch_cpu"],
               "not_a_rng_state": torch.zeros(4, dtype=torch.uint8)}
    with pytest.raises(ValueError, match="bad entry name"):
        storage._validate_rng_parts(tensors, snap["python"])


def test_checkpoint_rng_parts_rejects_noncontiguous_cuda_states():
    snap = storage.get_rng_snapshot()
    tensors = {"torch_cpu": snap["torch_cpu"],
               "cuda.0": torch.zeros(4, dtype=torch.uint8),
               "cuda.2": torch.zeros(4, dtype=torch.uint8)}
    with pytest.raises(ValueError, match="contiguous cuda.0"):
        storage._validate_rng_parts(tensors, snap["python"])


def test_manifest_and_resume_config_refuse_symlink_leaves(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "manifest.json").symlink_to(outside)
    (ckpt / "resume_config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        storage.load_manifest(str(ckpt))
    with pytest.raises(ValueError, match="symlink"):
        storage.load_resume_config(str(ckpt))


def test_unsupported_python_rng_version_rejects_before_mutating_global_rng():
    snap = _with_python(version=999)
    torch_before = torch.get_rng_state().clone()
    python_before = __import__("random").getstate()
    with pytest.raises(ValueError, match="bad python RNG state"):
        storage.set_rng_snapshot(snap)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert __import__("random").getstate() == python_before


def test_invalid_torch_rng_payload_rejects_before_global_mutation():
    snap = _good()
    snap["torch_cpu"] = torch.zeros(4, dtype=torch.uint8)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="bad torch RNG state"):
        storage.set_rng_snapshot(snap)
    assert torch.equal(torch.get_rng_state(), before)


def test_invalid_python_rng_index_rejects_before_global_mutation():
    snap = _with_python()
    version, inner, gauss = snap["python"]
    invalid_inner = list(inner)
    invalid_inner[-1] = 999999
    snap["python"] = (version, invalid_inner, gauss)
    torch_before = torch.get_rng_state().clone()
    python_before = __import__("random").getstate()
    with pytest.raises(ValueError, match="bad python RNG state"):
        storage.set_rng_snapshot(snap)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert __import__("random").getstate() == python_before


def test_checkpoint_rng_parts_rejects_bad_torch_rng_payload():
    snap = storage.get_rng_snapshot()
    with pytest.raises(ValueError, match="bad torch RNG state"):
        storage._validate_rng_parts(
            {"torch_cpu": torch.zeros(4, dtype=torch.uint8)}, snap["python"]
        )
