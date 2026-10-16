"""SmaulBRAIN V2 model: byte-level recurrent linear-attention LM (dense + MoE).

Layout per block: x -> RMSNorm(n1) -> recurrent-attn -> +residual ->
RMSNorm(n2) -> channel-mix (dense GatedFFN | shared MoE expert pool) -> +residual.
Head: RMSNorm(final) + tied embedding projection.

MoE uses a single shared expert pool + single shared router across all
blocks (documented choice: global expert registry; keeps per-expert counts
exact and memory bounded by one pool).
"""
import numpy as np
import weakref

from . import config as _cfg
from . import moe as _moe
from . import ops as _ops
from . import quant as _quant
from . import vocab as _vocab


def build_params(spec, rng: np.random.Generator, n_experts: int = -1):
    d = spec.d_model
    p = {}
    s = 1.0 / np.sqrt(d)
    p["embed"] = (rng.standard_normal((spec.vocab_size, d)) * s).astype(np.float32)
    p["final_norm"] = np.ones(d, dtype=np.float32)
    for i in range(spec.n_layers):
        px = f"L{i}."
        p[px + "attn.Wr"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
        p[px + "attn.Wk"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
        p[px + "attn.Wv"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
        p[px + "attn.Wo"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
        p[px + "attn.w"] = (rng.standard_normal(d) * 0.1).astype(np.float32)
        p[px + "attn.u"] = np.zeros(d, dtype=np.float32)
        p[px + "n1"] = np.ones(d, dtype=np.float32)
        p[px + "n2"] = np.ones(d, dtype=np.float32)
        if spec.kind == "dense":
            h = spec.ffn_hidden
            p[px + "ffn.Wa"] = (rng.standard_normal((d, h)) * s).astype(np.float32)
            p[px + "ffn.Wb"] = (rng.standard_normal((d, h)) * s).astype(np.float32)
            p[px + "ffn.Wc"] = (rng.standard_normal((h, d)) * (1.0 / np.sqrt(h))).astype(np.float32)
    if spec.kind == "moe":
        n = spec.n_experts_init if n_experts < 0 else n_experts
        p["router"] = (rng.standard_normal((d, n)) * s).astype(np.float32)
        for eid in range(n):
            p[f"expert.{eid}.Win"] = (rng.standard_normal((d, spec.expert_hidden)) * s).astype(np.float32)
            p[f"expert.{eid}.Wout"] = (
                rng.standard_normal((spec.expert_hidden, d)) * (1.0 / np.sqrt(spec.expert_hidden))).astype(np.float32)
    return p


def count_params_dict(p: dict) -> int:
    return int(sum(v.size for v in p.values()))


def expert_ids(p: dict) -> list:
    ids = set()
    for k in p:
        if k.startswith("expert."):
            ids.add(int(k.split(".")[1]))
    return sorted(ids)


class PackedCache:
    """Bounded cache of packed ternary weights. Key -> (packed, scale, shape)."""

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self.d = {}
        self.reuse_hits = 0
        self.builds = 0

    def get(self, key, W: np.ndarray):
        e = self.d.get(key)
        if (e is not None and e["ref"]() is W and e["id"] == id(W)
                and e["nbytes"] == W.nbytes):
            self.reuse_hits += 1
            return e["packed"], e["scale"], e["shape"]
        packed, scale, shape, _th = _quant.pack_ternary_tiled(W)
        if len(self.d) >= self.max_entries:
            self.d.pop(next(iter(self.d)))
        self.d[key] = {"packed": packed, "scale": scale, "shape": shape,
                       "id": id(W), "ref": weakref.ref(W), "nbytes": W.nbytes}
        self.builds += 1
        return packed, scale, shape

    def invalidate(self, keys=None):
        """Invalidate all cached weights, or only the named changed parameters."""
        if keys is None:
            self.d.clear()
            return
        for key in keys:
            self.d.pop(key, None)


class FP8Cache:
    """Bounded-by-model FP8 weight cache; entries are invalidated by parameter name."""

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self.d = {}
        self.builds = 0
        self.reuse_hits = 0

    def get(self, key, W: np.ndarray):
        entry = self.d.get(key)
        if (entry is not None and entry["ref"]() is W and entry["id"] == id(W)
                and entry["nbytes"] == W.nbytes):
            self.reuse_hits += 1
            return entry["weight"], entry["scale"]
        Wq, scale = _quant.fp8_e4m3_quantize(W)
        if key not in self.d and len(self.d) >= self.max_entries:
            self.d.pop(next(iter(self.d)))
        self.d[key] = {"weight": Wq, "scale": scale,
                       "id": id(W), "ref": weakref.ref(W), "nbytes": W.nbytes}
        self.builds += 1
        return Wq, scale

    def invalidate(self, keys=None):
        if keys is None:
            self.d.clear()
            return
        for key in keys:
            self.d.pop(key, None)


class Model:
    def __init__(self, spec, params: dict, forward_quant="fp32", backward_mode="fp32",
                 paging_mode="R", paging_store=None, loss_smooth_beta=0.99,
                 capacity_factor: float = 1.25):
        _cfg.validate_quant_combo(forward_quant, backward_mode)
        if forward_quant == "ternary":
            # Prepare the packed native backend before timing/runtime forward calls.
            # A packed scalar reference remains available when no compiler exists.
            _quant.prepare_ternary_backend()
        self.spec = spec
        self.p = params
        self.forward_quant = forward_quant
        self.backward_mode = backward_mode
        self.paging_mode = paging_mode
        self.store = paging_store
        self.packed = PackedCache()
        self.fp8 = FP8Cache()
        self.loss_smooth_beta = float(loss_smooth_beta)
        self.capacity_factor = float(capacity_factor)
        self._assert_counts()

    # -- counts --
    def _assert_counts(self):
        if self.spec.kind == "dense":
            total = count_params_dict(self.p)
            exp = _cfg.EXPECTED_DENSE_TOTAL[self.spec.name]
            if total != exp:
                raise ValueError(f"preset {self.spec.name}: got {total} params, expected {exp}")
        else:
            for eid in expert_ids(self.p):
                n = int(self.p[f"expert.{eid}.Win"].size + self.p[f"expert.{eid}.Wout"].size)
                exp = _cfg.EXPECTED_EXPERT_PARAMS[self.spec.name]
                if n != exp:
                    raise ValueError(f"preset {self.spec.name} expert {eid}: got {n}, expected {exp}")

    def report(self):
        n_exp = len(expert_ids(self.p)) if self.spec.kind == "moe" else 0
        rep = _cfg.arch_report(self.spec, n_exp)
        rep["forward_quant"] = self.forward_quant
        rep["backward_mode"] = self.backward_mode
        rep["paging_mode"] = self.paging_mode
        return rep

    # -- quantization-aware linear --
    def _qw(self, W: np.ndarray, key):
        """Return effective forward weight + backward handle per quant config."""
        if self.forward_quant == "fp32":
            return W, ("fp32", W)
        if self.forward_quant == "fp8":
            Wq, scale = self.fp8.get(key, W)
            return Wq, ("fp8", W, Wq, scale)
        packed, scale, shape = self.packed.get(key, W)
        return (packed, scale, shape), ("ternary", W, packed, scale, shape)

    def _lin_fwd(self, X: np.ndarray, W: np.ndarray, key):
        eff, handle = self._qw(W, key)
        if self.forward_quant == "ternary":
            packed, scale, shape = eff
            Y = _quant.packed_matmat(X, packed, shape, float(scale))
            return Y.astype(np.float32, copy=False), handle
        return (X @ eff).astype(np.float32), handle

    def _lin_bwd(self, dY: np.ndarray, X: np.ndarray, handle, codes=None):
        kind = handle[0]
        if kind == "fp32":
            W = handle[1]
            return (dY @ W.T).astype(np.float32), (X.T @ dY).astype(np.float32)
        if kind == "fp8":
            _, W, Wq, _scale = handle
            if self.backward_mode == "fp8":
                dYq, _ = _quant.fp8_e4m3_quantize(dY)
                Xq, _ = _quant.fp8_e4m3_quantize(X)
                return (dYq @ Wq.T).astype(np.float32), (Xq.T @ dYq).astype(np.float32)
            return (dY @ Wq.T).astype(np.float32), (X.T @ dY).astype(np.float32)

        # Use the packed ternary matrix directly for the input gradient. No
        # dense FP32 reconstruction is created or retained in the backward cache.
        _, W, packed, scale, shape = handle
        dX = _quant.packed_matmat_transpose(dY, packed, shape, float(scale))
        if self.backward_mode == "fp8":
            dYq, _ = _quant.fp8_e4m3_quantize(dY)
            Xq, _ = _quant.fp8_e4m3_quantize(X)
            dW = (Xq.T @ dYq).astype(np.float32)  # STE into latent master
        else:
            dW = (X.T @ dY).astype(np.float32)  # straight-through estimator
        return dX.astype(np.float32, copy=False), dW
    # -- recurrence (quant-aware projections) --
    def _attn_fwd(self, X, px, s0):
        p = self.p
        R, hR = self._lin_fwd(X, p[px + "attn.Wr"], px + "attn.Wr")
        K, hK = self._lin_fwd(X, p[px + "attn.Wk"], px + "attn.Wk")
        V, hV = self._lin_fwd(X, p[px + "attn.Wv"], px + "attn.Wv")
        w = p[px + "attn.w"]
        u = p[px + "attn.u"]
        alpha = np.exp(-_ops.softplus(w.astype(np.float64))).astype(np.float32)
        T, d = X.shape
        s = np.zeros(d, dtype=np.float32) if s0 is None else s0.astype(np.float32).copy()
        KV = K * V
        S = np.empty((T, d), dtype=np.float32)
        # Ternary mode already loaded the native C++ backend. Use its SIMD scan
        # to remove the per-token Python loop during prefill/training; keep the
        # numerically matching NumPy loop as the portable/non-native fallback.
        if not _quant.recurrent_scan(KV, alpha, s, S):
            for t in range(T):
                s = alpha * s + KV[t]
                S[t] = s
        G = _ops.sigmoid(R)
        Y = G * (S + u * KV)
        O, hO = self._lin_fwd(Y, p[px + "attn.Wo"], px + "attn.Wo")
        # Keep the projection input Y, but reconstruct KV during backward.
        # This drops one full [T,d] activation from every layer's train cache.
        cache = (X, R, K, V, S, Y, alpha, hR, hK, hV, hO, px)
        return O, (s.copy(), cache)

    def _attn_bwd(self, dO, cache):
        X, R, K, V, S, Y, alpha, hR, hK, hV, hO, px = cache
        T, d = X.shape
        p = self.p
        KV = K * V
        dY_full, dWo = self._lin_bwd(dO, Y, hO)
        dY = dY_full
        sig = _ops.sigmoid(R)
        dS_total = dY * sig
        dG = dY * (S + p[px + "attn.u"] * KV) * sig * (1 - sig)
        dKV_state = np.empty_like(KV)
        Aw = alpha.astype(np.float64)
        dalpha = np.zeros(d, dtype=np.float64)
        # The packed backend includes native time scans for both BPTT
        # recurrences. This removes two Python loops and repeated per-token
        # float64 row conversions. FP32-only/compiler-free runs keep the
        # straightforward reference implementation.
        if not _quant.recurrent_backward_scan(dS_total, S, alpha, dKV_state, dalpha):
            # Amortize dtype conversion instead of allocating a fresh float64
            # row on every step of both recurrent backward loops.
            dS64 = dS_total.astype(np.float64)
            S64 = S.astype(np.float64)
            ds = np.zeros(d, dtype=np.float64)
            for t in range(T - 1, -1, -1):
                ds = ds * Aw + dS64[t]
                dKV_state[t] = ds.astype(np.float32)
            dsdalpha = np.zeros(d, dtype=np.float64)
            for t in range(T):
                s_prev = S64[t - 1] if t > 0 else np.zeros(d)
                dsdalpha = s_prev + Aw * dsdalpha
                dalpha += dS64[t] * dsdalpha
        dKV = dKV_state + dY * sig * p[px + "attn.u"]
        dK = dKV * V
        dV = dKV * K
        dR = dG
        dXr, dWr = self._lin_bwd(dR, X, hR)
        dXk, dWk = self._lin_bwd(dK, X, hK)
        dXv, dWv = self._lin_bwd(dV, X, hV)
        dX = (dXr + dXk + dXv).astype(np.float32)
        sigw = 1.0 / (1.0 + np.exp(-np.clip(p[px + "attn.w"].astype(np.float64), -60, 60)))
        dw = (dalpha * (-Aw * sigw)).astype(np.float32)
        du = np.sum(dY * sig * KV, axis=0).astype(np.float32)
        grads = {px + "attn.Wr": dWr, px + "attn.Wk": dWk, px + "attn.Wv": dWv,
                 px + "attn.Wo": dWo, px + "attn.w": dw, px + "attn.u": du}
        return dX, grads

    # -- channel mixing --
    def _ffn_fwd(self, X, px):
        p = self.p
        A, hA = self._lin_fwd(X, p[px + "ffn.Wa"], px + "ffn.Wa")
        B, hB = self._lin_fwd(X, p[px + "ffn.Wb"], px + "ffn.Wb")
        H = _ops.silu(A) * B
        Y, hC = self._lin_fwd(H, p[px + "ffn.Wc"], px + "ffn.Wc")
        return Y, (X, A, B, H, hA, hB, hC, px)

    def _ffn_bwd(self, dY, cache):
        X, A, B, H, hA, hB, hC, px = cache
        dH, dWc = self._lin_bwd(dY, H, hC)
        sigA = _ops.sigmoid(A)
        dA = dH * B * (sigA * (1 + A * (1 - sigA)))
        dB = dH * _ops.silu(A)
        dXa, dWa = self._lin_bwd(dA, X, hA)
        dXb, dWb = self._lin_bwd(dB, X, hB)
        return (dXa + dXb).astype(np.float32), {px + "ffn.Wa": dWa, px + "ffn.Wb": dWb, px + "ffn.Wc": dWc}

    def _expert_weights(self, eid: int):
        if self.store is not None and self.spec.kind == "moe":
            try:
                return self.store.get(eid)
            except (KeyError, AttributeError):
                pass
        return {"Win": self.p[f"expert.{eid}.Win"], "Wout": self.p[f"expert.{eid}.Wout"]}

    def _expert_fwd(self, Xb: np.ndarray, eid: int):
        W = self._expert_weights(eid)
        Hpre, h1 = self._lin_fwd(Xb, W["Win"], f"expert.{eid}.Win")
        H = _ops.gelu_tanh(Hpre)
        Y, h2 = self._lin_fwd(H, W["Wout"], f"expert.{eid}.Wout")
        return Y, (Xb, Hpre, H, h1, h2, eid)

    def _expert_bwd(self, dYb, cache):
        Xb, Hpre, H, h1, h2, eid = cache
        dH, dWout = self._lin_bwd(dYb, H, h2)
        dZ = dH * _ops.gelu_tanh_grad(Hpre)
        dXb, dWin = self._lin_bwd(dZ, Xb, h1)
        return dXb, {f"expert.{eid}.Win": dWin, f"expert.{eid}.Wout": dWout}

    def _moe_fwd(self, X, router, top_k: int, disabled=()):
        top_idx, weights, info, rcache = router.route(X, top_k,
                                                      capacity_factor=self.capacity_factor,
                                                      disabled=disabled)
        T, d = X.shape
        Y = np.zeros_like(X)
        expert_caches = {}
        # Group token/rank assignments once rather than scanning top_idx with
        # np.where for every active expert. Stable sorting preserves the
        # original row-major order within each expert group.
        flat_eids = top_idx.reshape(-1)
        route_rows = np.repeat(np.arange(T, dtype=np.int64), top_idx.shape[1])
        route_cols = np.tile(np.arange(top_idx.shape[1], dtype=np.int64), T)
        grouped_order = np.argsort(flat_eids, kind="stable")
        grouped_eids, group_counts = np.unique(flat_eids, return_counts=True)
        group_starts = np.cumsum(group_counts, dtype=np.int64) - group_counts
        for eid, start, count in zip(grouped_eids, group_starts, group_counts):
            eid = int(eid)
            pos = grouped_order[int(start):int(start + count)]
            rows = route_rows[pos]
            cols = route_cols[pos]
            if self.store is not None:
                self.store.pin(eid)
            try:
                Xb = X[rows]
                Yb, ec = self._expert_fwd(Xb, eid)
                w = weights[rows, cols][:, None]
                Y[rows] += w * Yb
                # Cache the output used by router-gradient computation;
                # recomputing an expert forward in backward doubles its large
                # input/output projections.
                expert_caches[eid] = (rows, cols, ec, Yb)
            finally:
                if self.store is not None:
                    self.store.unpin(eid)
        cache = (X, top_idx, weights, rcache, expert_caches, router)
        # Only overflow is consumed by the public model path. Balance-loss and
        # per-expert reporting helpers are diagnostics, not part of the forward
        # result or loss, so do not run their Python loops on every decode token.
        aux = {"overflow_dropped": info["overflow_dropped"],
               "balance_aux": 0.0}
        return Y, (cache, aux)

    def _moe_bwd(self, dY, moe_cache):
        cache, _aux = moe_cache
        X, top_idx, weights, rcache, expert_caches, router = cache
        T, k = top_idx.shape
        dX = np.zeros_like(X)
        dWsum = np.zeros_like(weights)
        grads = {}
        for eid, (rows, cols, ec, Yb) in expert_caches.items():
            w = weights[rows, cols][:, None]
            dYb = dY[rows] * w
            dXb, g = self._expert_bwd(dYb, ec)
            dWsum[rows, cols] = np.sum(dY[rows] * Yb, axis=1)
            for kk, vv in g.items():
                grads[kk] = grads.get(kk, 0) + vv if isinstance(grads.get(kk), np.ndarray) else vv
            # Each token can select an expert at most once, so rows is
            # unique within one expert group; indexed addition is safe and
            # avoids np.add.at's general duplicate-index machinery.
            dX[rows] += dXb
        # router backward
        dXr, dRouter = router.backward(dWsum, rcache)
        dX = dX + dXr
        grads["router"] = dRouter
        return dX.astype(np.float32), grads

    # -- full sequence forward/backward --
    def forward(self, tokens, states=None, train=False, disabled=()):
        spec = self.spec
        T = len(tokens)
        X = self.p["embed"][np.asarray(tokens, dtype=np.int64)]
        layer_caches = []
        aux_all = {"overflow_dropped": 0, "balance_aux": 0.0}
        new_states: list = [] if states is None else states
        is_moe = spec.kind == "moe"
        if is_moe:
            router = _moe.Router.__new__(_moe.Router)
            router.W = self.p["router"]
        else:
            router = None
        for i in range(spec.n_layers):
            px = f"L{i}."
            s0 = None if states is None else states[i]
            h, c1 = _ops.rmsnorm_fwd(X, self.p[px + "n1"])
            o, (s1, c2) = self._attn_fwd(h, px, s0)
            X = (X + o).astype(np.float32)
            h2, c3 = _ops.rmsnorm_fwd(X, self.p[px + "n2"])
            if not is_moe:
                f, c4 = self._ffn_fwd(h2, px)
                aux = None
            else:
                assert router is not None
                f, (c4, aux) = self._moe_fwd(h2, router, spec.top_k, disabled=disabled)
                aux_all["overflow_dropped"] += aux["overflow_dropped"]
                aux_all["balance_aux"] += aux["balance_aux"]
            X = (X + f).astype(np.float32)
            if train:
                layer_caches.append((c1, c2, c3, c4, px))
            if states is None:
                new_states.append(s1)
            else:
                states[i] = s1
        H, cf = _ops.rmsnorm_fwd(X, self.p["final_norm"])
        logits = (H @ self.p["embed"].T).astype(np.float32)
        if train:
            return logits, (layer_caches, H, cf, aux_all)
        return logits, (states if states is not None else new_states)

    def loss_and_backward(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64)
        inp, tgt = tokens[:-1], tokens[1:]
        logits, cache = self.forward(inp, train=True)
        layer_caches, H, cf, aux = cache
        # softmax CE (nats per predicted byte)
        m = logits.max(axis=1, keepdims=True)
        logsum = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
        nll = (logsum.squeeze(1) - logits[np.arange(len(tgt)), tgt]).astype(np.float64)
        loss = float(nll.mean())
        dlogits = np.exp(logits - logsum).astype(np.float32)
        dlogits[np.arange(len(tgt)), tgt] -= 1.0
        dlogits /= len(tgt)
        grads: dict = {}
        dH = dlogits @ self.p["embed"]
        grads["embed"] = dlogits.T @ H
        dX, dgfinal = _ops.rmsnorm_bwd(dH, cf)
        grads["final_norm"] = dgfinal
        spec = self.spec
        for i in range(spec.n_layers - 1, -1, -1):
            px = f"L{i}."
            c1, attn_state_cache, c3n, mix_cache, _p = layer_caches[i]
            # X_out = X_mid + f  ->  dX_mid = dX (residual) ; df = dX
            dX_mid_resid = dX
            if spec.kind == "dense":
                dH2, gffn = self._ffn_bwd(dX, mix_cache)
                for kk, vv in gffn.items():
                    grads[kk] = vv
            else:
                dH2, gmoe = self._moe_bwd(dX, (mix_cache, None))
                for kk, vv in gmoe.items():
                    if isinstance(grads.get(kk), np.ndarray):
                        grads[kk] = grads[kk] + vv
                    else:
                        grads[kk] = vv
            dX_mid_norm, dgn2 = _ops.rmsnorm_bwd(dH2, c3n)
            grads[px + "n2"] = dgn2
            dX_mid = (dX_mid_resid + dX_mid_norm).astype(np.float32)
            # X_mid = X_in + o  ->  do = dX_mid
            dHo, gattn = self._attn_bwd(dX_mid, attn_state_cache)
            for kk, vv in gattn.items():
                grads[kk] = vv
            dX_in_norm, dgn1 = _ops.rmsnorm_bwd(dHo, c1)
            grads[px + "n1"] = dgn1
            dX = (dX_mid + dX_in_norm).astype(np.float32)
        # gradient wrt input embeddings accumulates into the tied embed matrix
        _ops.scatter_add_rows(grads["embed"], inp, dX)
        return loss, grads, aux

    # -- incremental inference --
    def init_states(self):
        return [np.zeros(self.spec.d_model, dtype=np.float32) for _ in range(self.spec.n_layers)]

    @staticmethod
    def serialize_states(states) -> dict:
        return {str(i): s.copy() for i, s in enumerate(states)}

    @staticmethod
    def restore_states(blob) -> list:
        return [np.asarray(blob[str(i)], dtype=np.float32) for i in range(len(blob))]

    def step_token(self, tok: int, states):
        logits, _ = self.forward([tok], states=states, train=False)
        return logits[0]
