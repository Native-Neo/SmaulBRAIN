import numpy as np

from smaulbrain import data as _data
from smaulbrain import growth as _growth
from smaulbrain import inference as _inference
from smaulbrain import moe as _moe
from smaulbrain import paging as _paging
from smaulbrain import quant as _quant
from smaulbrain import smaulopt as _smaulopt


def test_router_top_k_matches_stable_full_sort_including_cutoff_ties():
    router = _moe.Router(1, 6, np.random.default_rng(0))
    router.W[0] = np.array([3.0, 2.0, 2.0, 2.0, 1.0, 0.0], dtype=np.float32)
    X = np.ones((4, 1), dtype=np.float32)
    for disabled in ((), (0,)):
        top_idx, weights, info, cache = router.route(
            X, top_k=3, capacity_factor=100.0, disabled=disabled)
        scores = X @ router.W
        if disabled:
            scores[:, list(disabled)] = -np.inf
        expected = np.argsort(-scores, axis=1, kind="stable")[:, :3]
        assert np.array_equal(top_idx, expected)
        assert np.allclose(weights.sum(axis=1), 1.0)
        assert info["overflow_dropped"] == 0


def test_router_empty_token_batch_is_well_formed():
    router = _moe.Router(5, 7, np.random.default_rng(12))
    X = np.empty((0, 5), dtype=np.float32)
    top_idx, weights, info, cache = router.route(X, top_k=3)
    assert top_idx.shape == (0, 3)
    assert weights.shape == (0, 3)
    assert info["overflow_dropped"] == 0
    assert np.array_equal(info["load"], np.zeros(7, dtype=np.int64))
    assert cache[4].shape == (0, 3)


def test_router_vectorized_capacity_preserves_flat_token_order():
    router = _moe.Router(2, 4, np.random.default_rng(2))
    router.W[:] = np.array([[4, 3, 2, 1], [4, 3, 2, 1]], dtype=np.float32)
    X = np.ones((5, 2), dtype=np.float32)
    top_idx, weights, info, cache = router.route(X, 2, capacity_factor=0.2)
    scores = X @ router.W
    ref_idx = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    cap = max(int(np.ceil(0.2 * len(X) * 2 / 4)), 1)
    keep = np.ones_like(ref_idx, dtype=bool)
    load = np.zeros(4, dtype=np.int64)
    for t in range(len(X)):
        for j in range(2):
            e = int(ref_idx[t, j])
            if load[e] >= cap:
                keep[t, j] = False
            else:
                load[e] += 1
    assert np.array_equal(top_idx, ref_idx)
    assert np.array_equal(cache[4], keep)
    assert np.array_equal(info["load"], load)
    assert info["overflow_dropped"] == int((~keep).sum())
    assert np.allclose(weights, np.where(keep, np.exp(
        np.take_along_axis(scores, ref_idx, axis=1) -
        np.take_along_axis(scores, ref_idx, axis=1).max(axis=1, keepdims=True)), 0.0) /
        np.maximum(np.where(keep, np.exp(
            np.take_along_axis(scores, ref_idx, axis=1) -
            np.take_along_axis(scores, ref_idx, axis=1).max(axis=1, keepdims=True)), 0.0
        ).sum(axis=1, keepdims=True), 1e-12))


def test_router_backward_and_moe_diagnostics_vectorized_match_reference():
    rng = np.random.default_rng(13)
    router = _moe.Router(5, 7, rng)
    X = rng.standard_normal((9, 5)).astype(np.float32)
    top_idx, weights, info, cache = router.route(X, 3, capacity_factor=0.8)
    dweights = rng.standard_normal(weights.shape).astype(np.float32)
    dX, dW = router.backward(dweights, cache)

    _, scores, ids, w, keep = cache
    dtop = np.zeros_like(w)
    for t in range(w.shape[0]):
        kept = keep[t]
        if np.any(kept):
            wk, dw = w[t][kept], dweights[t][kept]
            dtop[t][kept] = wk * (dw - float(np.dot(wk, dw)))
    dscores = np.zeros_like(scores)
    np.add.at(dscores, (np.repeat(np.arange(len(X)), ids.shape[1]), ids.ravel()), dtop.ravel())
    assert np.allclose(dX, dscores @ router.W.T, atol=1e-6)
    assert np.allclose(dW, X.T @ dscores, atol=1e-6)

    ref_counts = np.zeros(router.n_experts, dtype=np.int64)
    ref_gate = np.zeros(router.n_experts, dtype=np.float64)
    for t in range(len(X)):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            ref_counts[e] += 1
            ref_gate[e] += float(w[t, j])
    expected_balance = router.n_experts * np.dot(
        ref_counts / len(X), ref_gate / len(X))
    assert np.isclose(_moe.balance_aux_loss(w, ids, router.n_experts), expected_balance)
    stats = _moe.expert_output_stats(ids, w, router.n_experts)
    assert stats["per_expert_counts"] == ref_counts.tolist()
    assert stats["tokens_total"] == len(X)


class _FakeModel:
    def __init__(self):
        self.seen = []

    def init_states(self):
        return [0]

    def serialize_states(self, states):
        return {"0": states[0]}

    def restore_states(self, blob):
        return [blob["0"]]

    def forward(self, tokens, states=None, train=False):
        if states is not None:
            states[0] += len(tokens)
        return None, states

    def step_token(self, last, states):
        self.seen.append(int(last))
        logits = np.zeros(259, dtype=np.float32)
        logits[(int(last) + 1) % 259] = 1.0
        return logits


def test_generation_keeps_last_token_scalar_and_handles_stop_iterable():
    model = _FakeModel()
    engine = _inference.InferenceEngine(model, temperature=0)
    generated = engine.generate([10, 11], max_new_tokens=5, stop_ids=iter([13]))
    assert generated == [12, 13]
    assert model.seen == [11, 12]


def test_stream_generation_keeps_last_token_scalar():
    model = _FakeModel()
    engine = _inference.InferenceEngine(model, temperature=0)
    generated = list(engine.generate_stream([], max_new_tokens=4))
    assert generated == [1, 2, 3, 4]
    assert model.seen == [0, 1, 2, 3]


def test_compiler_free_packed_matmul_fallback_uses_blas_not_scalar_loops(monkeypatch):
    rng = np.random.default_rng(31)
    W = rng.standard_normal((37, 72)).astype(np.float32)
    X = rng.standard_normal((3, 37)).astype(np.float32)
    dY = rng.standard_normal((3, 72)).astype(np.float32)
    packed, scale, shape, _ = _quant._pack_ternary_numpy(W)
    dense = _quant.unpack_ternary(packed, shape, scale)
    monkeypatch.setattr(_quant, "prepare_ternary_backend", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler-free fallback called Python scalar kernel")

    monkeypatch.setattr(_quant, "packed_matmat_scalar", forbidden)
    monkeypatch.setattr(_quant, "packed_matmat_transpose_scalar", forbidden)
    assert np.allclose(_quant.packed_matmat(X, packed, shape, scale), X @ dense, atol=2e-5)
    assert np.allclose(_quant.packed_matmat_transpose(dY, packed, shape, scale),
                       dY @ dense.T, atol=2e-5)


def test_dataset_stream_reuses_persistent_file_handle(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(bytes(range(32)))
    with _data.ByteStream(str(path)) as stream:
        fh = stream._fh
        assert fh is not None and not fh.closed
        assert stream.next_tokens(4) == [0, 1, 2, 3, 4]
        assert stream.next_tokens(4) == [5, 6, 7, 8, 9]
        assert stream._fh is fh
        assert not fh.closed
    assert fh.closed


def test_paged_expert_writeback_stays_cached_then_flushes_on_eviction(tmp_path):
    for eid in (0, 1):
        np.savez(tmp_path / f"expert_{eid}.npz",
                 Win=np.full((2, 2), eid, dtype=np.float32),
                 Wout=np.full((2, 2), eid + 1, dtype=np.float32))
    store = _paging.ExpertStore("D2R", 2, disk_dir=str(tmp_path),
                                ram_cache_max_experts=1)
    before = store.get(0)
    updated = {
        "Win": np.full((2, 2), 7, dtype=np.float32),
        "Wout": np.full((2, 2), 8, dtype=np.float32),
    }
    store.write_back(0, updated)
    assert np.array_equal(store.get(0)["Win"], updated["Win"])
    store.get(1)  # evicts expert 0 and must flush its dirty cache entry
    with np.load(tmp_path / "expert_0.npz") as saved:
        assert np.array_equal(saved["Win"], updated["Win"])
        assert np.array_equal(saved["Wout"], updated["Wout"])

def test_optimizer_inplace_update_buffers_preserve_norms_and_updates():
    rng = np.random.default_rng(41)
    params_a = {
        "matrix": rng.standard_normal((9, 7)).astype(np.float32),
        "vector": rng.standard_normal(11).astype(np.float32),
    }
    params_b = {k: v.copy() for k, v in params_a.items()}
    grads = {k: rng.standard_normal(v.shape).astype(np.float32)
             for k, v in params_a.items()}
    opt_a = _smaulopt.SmaulOPT(lr=2e-3, beta1=0.9, beta2=0.98,
                               eps=1e-8, weight_decay=1e-3)
    opt_b = _smaulopt.SmaulOPT(lr=2e-3, beta1=0.9, beta2=0.98,
                               eps=1e-8, weight_decay=1e-3)
    opt_a.register(params_a)
    opt_b.register(params_b)
    norms = opt_a.step(params_a, grads, collect_norms=True)
    no_norms = opt_b.step(params_b, grads, collect_norms=False)
    assert norms.keys() == params_a.keys()
    assert no_norms == {}
    for name in params_a:
        assert np.allclose(params_a[name], params_b[name], rtol=1e-6, atol=1e-7)
        for state_key in opt_a.state[name]:
            if state_key == "factored":
                assert opt_a.state[name][state_key] == opt_b.state[name][state_key]
            else:
                assert np.allclose(opt_a.state[name][state_key],
                                   opt_b.state[name][state_key], rtol=0, atol=0)

def test_growth_ranking_can_use_one_batch_without_changing_evidence_count():
    class FakeRankModel:
        def __init__(self):
            self.p = {
                "expert.0.Win": np.zeros((1, 1), dtype=np.float32),
                "expert.1.Win": np.zeros((1, 1), dtype=np.float32),
            }
            self.calls = 0

        def forward(self, tokens, train=False, disabled=()):
            self.calls += 1
            logits = np.zeros((len(tokens), 16), dtype=np.float32)
            # Make each ablation measurable while keeping all logits finite.
            if disabled:
                logits[:, int(disabled[0])] = 1.0
            return logits, None

    batches = [list(range(5)), list(range(5, 10))]
    model = FakeRankModel()
    scores, total_samples = _growth.rank_experts(
        model, batches, min_samples=8, max_batches=1)
    assert total_samples == 8
    assert set(scores) == {0, 1}
    assert model.calls == 3  # one baseline and one ablation per expert

