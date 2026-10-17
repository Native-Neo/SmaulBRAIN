import numpy as np
import torch

from smaulbrain import data as _data, growth as _growth, inference as _inference
from smaulbrain import moe as _moe, paging as _paging, quant as _quant, smaulopt as _smaulopt


def _np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def test_router_top_k_matches_stable_full_sort_including_cutoff_ties():
    router = _moe.Router(1, 6, 0)
    router.W[0] = torch.tensor([3.0, 2.0, 2.0, 2.0, 1.0, 0.0])
    X = torch.ones((4, 1))
    for disabled in ((), (0,)):
        idx, weights, info, _ = router.route(X, 3, capacity_factor=100.0, disabled=disabled)
        scores = X.numpy() @ router.W.numpy()
        if disabled: scores[:, list(disabled)] = -np.inf
        expected = np.argsort(-scores, axis=1, kind="stable")[:, :3]
        assert np.array_equal(_np(idx), expected)
        assert torch.allclose(weights.sum(1), torch.ones(4))
        assert info["overflow_dropped"] == 0


def test_router_empty_token_batch_is_well_formed():
    router = _moe.Router(5, 7, 12); X = torch.empty((0, 5))
    idx, weights, info, cache = router.route(X, 3)
    assert idx.shape == weights.shape == (0, 3)
    assert info["overflow_dropped"] == 0 and info["load"] == [0] * 7
    assert cache[4].shape == (0, 3)


def test_router_vectorized_capacity_preserves_flat_token_order():
    router = _moe.Router(2, 4, 2)
    router.W[:] = torch.tensor([[4, 3, 2, 1], [4, 3, 2, 1]], dtype=torch.float32)
    X = torch.ones((5, 2)); idx, weights, info, cache = router.route(X, 2, capacity_factor=0.2)
    scores = _np(X @ router.W); ids = _np(idx)
    ref_idx = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    cap = max(int(np.ceil(0.2 * len(X) * 2 / 4)), 1)
    keep = np.ones_like(ref_idx, dtype=bool); load = np.zeros(4, dtype=np.int64)
    for t in range(len(X)):
        for j in range(2):
            e = int(ref_idx[t, j])
            if load[e] >= cap: keep[t, j] = False
            else: load[e] += 1
    assert np.array_equal(ids, ref_idx)
    assert np.array_equal(_np(cache[4]), keep)
    assert info["load"] == load.tolist() and info["overflow_dropped"] == int((~keep).sum())
    top = np.take_along_axis(scores, ref_idx, axis=1)
    exp = np.exp(top - top.max(axis=1, keepdims=True))
    exp = np.where(keep, exp, 0.0); exp /= np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    assert np.allclose(_np(weights), exp)


def test_router_backward_and_moe_diagnostics_vectorized_match_reference():
    rng = np.random.default_rng(13); router = _moe.Router(5, 7, 13)
    X = rng.standard_normal((9, 5)).astype(np.float32)
    ids, weights, _, cache = router.route(X, 3, capacity_factor=0.8)
    dweights = torch.from_numpy(rng.standard_normal(tuple(weights.shape)).astype(np.float32))
    dX, dW = router.backward(dweights, cache)
    Xn, scores, idsn, w, keep = (X, *[None] * 4)
    _, scores_t, ids_t, w_t, keep_t = cache
    scores, ids, w, keep = map(_np, (scores_t, ids_t, w_t, keep_t))
    dw = _np(dweights); dtop = np.zeros_like(w)
    for t in range(w.shape[0]):
        kept = keep[t]
        if np.any(kept):
            wk, dwt = w[t][kept], dw[t][kept]
            dtop[t][kept] = wk * (dwt - float(np.dot(wk, dwt)))
    dscores = np.zeros_like(scores)
    np.add.at(dscores, (np.repeat(np.arange(len(Xn)), ids.shape[1]), ids.ravel()), dtop.ravel())
    assert np.allclose(_np(dX), dscores @ router.W.numpy().T, atol=1e-6)
    assert np.allclose(_np(dW), Xn.T @ dscores, atol=1e-6)
    counts = np.zeros(router.n_experts, dtype=np.int64); gate = np.zeros(router.n_experts)
    for t in range(len(Xn)):
        for j in range(ids.shape[1]): counts[ids[t, j]] += 1; gate[ids[t, j]] += float(w[t, j])
    expected = router.n_experts * np.dot(counts / len(Xn), gate / len(Xn))
    assert np.isclose(_moe.balance_aux_loss(w_t, ids_t, router.n_experts), expected)
    stats = _moe.expert_output_stats(ids_t, w_t, router.n_experts)
    assert stats["per_expert_counts"] == counts.tolist() and stats["tokens_total"] == len(Xn)


class _FakeModel:
    def __init__(self): self.seen = []
    def init_states(self): return [0]
    def serialize_states(self, states): return {"0": states[0]}
    def restore_states(self, blob): return [blob["0"]]
    def forward(self, tokens, states=None, train=False):
        if states is not None: states[0] += len(tokens)
        return None, states
    def step_token(self, last, states):
        self.seen.append(int(last)); logits = torch.zeros(259); logits[(int(last) + 1) % 259] = 1
        return logits


def test_generation_keeps_last_token_scalar_and_handles_stop_iterable():
    model = _FakeModel(); engine = _inference.InferenceEngine(model, temperature=0)
    assert engine.generate([10, 11], 5, iter([13])) == [12, 13]
    assert model.seen == [11, 12]


def test_stream_generation_keeps_last_token_scalar():
    model = _FakeModel(); engine = _inference.InferenceEngine(model, temperature=0)
    assert list(engine.generate_stream([], 4)) == [1, 2, 3, 4]
    assert model.seen == [0, 1, 2, 3]


def test_compiler_free_packed_matmul_fallback_uses_blas_not_scalar_loops(monkeypatch):
    rng = np.random.default_rng(31); W = rng.standard_normal((37, 72)).astype(np.float32)
    X = torch.from_numpy(rng.standard_normal((3, 37)).astype(np.float32))
    dY = torch.from_numpy(rng.standard_normal((3, 72)).astype(np.float32))
    packed, scale, shape, _ = _quant._pack_ternary_numpy(W)
    dense = _quant.unpack_ternary(packed, shape, scale)
    monkeypatch.setattr(_quant, "prepare_ternary_backend", lambda: False)
    monkeypatch.setattr(_quant, "packed_matmat_scalar", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(_quant, "packed_matmat_transpose_scalar", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert torch.allclose(_quant.packed_matmat(X, packed, shape, scale), X @ dense, atol=2e-5)
    assert torch.allclose(_quant.packed_matmat_transpose(dY, packed, shape, scale), dY @ dense.T, atol=2e-5)


def test_dataset_stream_reuses_persistent_file_handle(tmp_path):
    path = tmp_path / "bytes.bin"; path.write_bytes(bytes(range(32)))
    with _data.ByteStream(str(path)) as stream:
        fh = stream._fh; assert fh is not None and not fh.closed
        assert stream.next_tokens(4) == [0, 1, 2, 3, 4]
        assert stream.next_tokens(4) == [5, 6, 7, 8, 9]
        assert stream._fh is fh and not fh.closed
    assert fh.closed


def test_paged_expert_writeback_stays_cached_then_flushes_on_eviction(tmp_path):
    for eid in (0, 1):
        torch.save({"Win": torch.full((2, 2), eid), "Wout": torch.full((2, 2), eid + 1)}, tmp_path / f"expert_{eid}.pt")
    store = _paging.ExpertStore("D2R", 2, disk_dir=str(tmp_path), ram_cache_max_experts=1)
    store.get(0)
    updated = {"Win": torch.full((2, 2), 7), "Wout": torch.full((2, 2), 8)}
    store.write_back(0, updated)
    assert torch.equal(store.get(0)["Win"], updated["Win"])
    store.get(1)
    saved = torch.load(tmp_path / "expert_0.pt", weights_only=True)
    assert torch.equal(saved["Win"], updated["Win"]) and torch.equal(saved["Wout"], updated["Wout"])


def test_optimizer_inplace_update_buffers_preserve_norms_and_updates():
    gen = torch.Generator().manual_seed(41)
    params_a = {"matrix": torch.randn((9, 7), generator=gen), "vector": torch.randn(11, generator=gen)}
    params_b = {k: v.clone() for k, v in params_a.items()}
    grads = {k: torch.randn(v.shape, generator=gen) for k, v in params_a.items()}
    opt_a = _smaulopt.SmaulOPT(lr=2e-3, weight_decay=1e-3); opt_b = _smaulopt.SmaulOPT(lr=2e-3, weight_decay=1e-3)
    opt_a.register(params_a); opt_b.register(params_b)
    norms = opt_a.step(params_a, grads, collect_norms=True); no_norms = opt_b.step(params_b, grads, collect_norms=False)
    assert norms.keys() == params_a.keys() and no_norms == {}
    for name in params_a:
        assert torch.allclose(params_a[name], params_b[name], rtol=1e-6, atol=1e-7)
        for key, value in opt_a.state[name].items():
            if key == "factored": assert value == opt_b.state[name][key]
            else: assert torch.equal(value, opt_b.state[name][key])


def test_growth_ranking_can_use_one_batch_without_changing_evidence_count():
    class FakeRankModel:
        def __init__(self): self.p = {"expert.0.Win": torch.zeros((1, 1)), "expert.1.Win": torch.zeros((1, 1))}; self.calls = 0
        def forward(self, tokens, train=False, disabled=()):
            self.calls += 1; logits = torch.zeros((len(tokens), 16))
            if disabled: logits[:, int(disabled[0])] = 1.0
            return logits, None
    batches = [list(range(5)), list(range(5, 10))]; model = FakeRankModel()
    scores, samples = _growth.rank_experts(model, batches, min_samples=8, max_batches=1)
    assert samples == 8 and set(scores) == {0, 1} and model.calls == 3
