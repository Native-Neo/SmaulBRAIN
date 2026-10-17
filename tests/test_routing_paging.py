import numpy as np
import pytest
import torch

from smaulbrain import moe as MOE
from smaulbrain import paging as P


def _router(d=8, e=6, seed=0):
    return MOE.Router(d, e, np.random.default_rng(seed))


def test_topk_deterministic_ties():
    r = _router()
    r.W[:] = 0.0  # all scores equal -> tie-break by expert id
    X = np.ones((4, 8), dtype=np.float32)
    idx, w, _info, _c = r.route(X, top_k=2, capacity_factor=99.0)
    assert (idx == np.array([[0, 1]] * 4)).all()
    assert np.allclose(w.sum(axis=1), 1.0)


def test_routing_batch_shapes_and_renorm():
    r = _router()
    for T in (1, 7, 33):
        X = np.random.default_rng(0).standard_normal((T, 8)).astype(np.float32)
        idx, w, _info, _c = r.route(X, top_k=3, capacity_factor=99.0)
        assert idx.shape == (T, 3) and w.shape == (T, 3)
        assert np.allclose(w.sum(axis=1), 1.0)


def test_capacity_overflow_policy():
    r = _router(d=4, e=2, seed=1)
    X = np.random.default_rng(2).standard_normal((16, 4)).astype(np.float32)
    _i, _w, info, _c = r.route(X, top_k=1, capacity_factor=0.5)
    assert info["overflow_dropped"] > 0
    assert info["cap_per_expert"] <= 8


def test_empty_and_invalid_routes():
    r = _router()
    X = np.zeros((3, 8), dtype=np.float32)
    idx, w, _info, _c = r.route(X, top_k=2)
    assert idx.shape == (3, 2)
    with pytest.raises(ValueError):
        r.route(X, top_k=99)
    bad = X.copy(); bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        r.route(bad if False else (bad * 0 + np.inf).astype(np.float32), top_k=2)


def test_router_gradient_finite_difference():
    rng = np.random.default_rng(5)
    r = MOE.Router(4, 3, rng)
    X = rng.standard_normal((5, 4)).astype(np.float32)
    idx, w, _info, cache = r.route(X, top_k=2, capacity_factor=99.0)
    dW = rng.standard_normal(w.shape).astype(np.float32)
    dX, dR = r.backward(dW, cache)
    eps = 1e-3
    num = np.zeros(tuple(r.W.shape), dtype=np.float32)
    for i in range(r.W.shape[0]):
        for j in range(r.W.shape[1]):
            W0 = r.W.detach().clone()
            with torch.no_grad(): r.W[i, j] = W0[i, j] + eps
            _, w1, _, _ = r.route(X, top_k=2, capacity_factor=99.0)
            with torch.no_grad(): r.W[i, j] = W0[i, j] - eps
            _, w0, _, _ = r.route(X, top_k=2, capacity_factor=99.0)
            with torch.no_grad(): r.W.copy_(W0)
            num[i, j] = float(((w1 - w0) * torch.from_numpy(dW)).sum() / (2 * eps))
    assert np.allclose(dR.numpy(), num, atol=5e-2)


def test_paging_r_and_d2r():
    ram = {0: {"Win": np.ones((4, 4), dtype=np.float32), "Wout": np.ones((4, 4), dtype=np.float32)}}
    s = P.ExpertStore("R", 1, ram_weights=ram)
    assert s.get(0)["Win"].shape == (4, 4)
    assert s.stats()["hits"] == 1


def test_paging_d2r_eviction_and_accounting(tmp_path):
    import os
    dd = str(tmp_path)
    for e in range(3):
        torch.save({"Win": torch.ones((2, 2)), "Wout": torch.ones((2, 2))}, os.path.join(dd, f"expert_{e}.pt"))
    s = P.ExpertStore("D2R", 3, disk_dir=dd, ram_cache_max_experts=1)
    s.get(0); s.get(1); s.get(0)
    st = s.stats()
    assert st["misses"] == 3 and st["hits"] == 0
    assert st["evictions"] >= 1 and st["bytes_transferred"] > 0
    s.write_back(0, {"Win": torch.zeros((2, 2)), "Wout": torch.zeros((2, 2))})
    assert s.get(0)["Win"].sum().item() == 0  # cache invalidated, reloaded fresh


def test_paging_v_modes_error_without_backend():
    ok, _reason = P.accelerator_available()
    if ok:
        pytest.skip("accelerator present; V-mode smoke would need device execution")
    for mode in ("V", "D2V", "R2V"):
        with pytest.raises(P.PagingError):
            P.ExpertStore(mode, 1, ram_weights={0: {}}, disk_dir="/tmp")


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_router_rejects_invalid_capacity_factor(capacity):
    r = _router()
    with pytest.raises(ValueError, match="capacity_factor"):
        r.route(np.zeros((2, 8), dtype=np.float32), top_k=2,
                capacity_factor=capacity)


def test_router_rejects_wrong_input_rank_or_width():
    r = _router()
    with pytest.raises(ValueError, match="shape"):
        r.route(np.zeros(8, dtype=np.float32), top_k=2)
    with pytest.raises(ValueError, match="shape"):
        r.route(np.zeros((2, 7), dtype=np.float32), top_k=2)
