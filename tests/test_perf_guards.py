"""Performance-guard regression tests (gauntlet Round-5 PERFORMANCE fixes).

Covers J1..M6: cube-guard 2x accounting, width-aware qkv budget, vectorized
_admit_slots bit-identity, single-pass grad norm, sync-free expert stats,
sparse-head study-only budget, N*E budget halving, fused factored update,
plus M2/M6 documented tradeoff floors with measured bench numbers.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def test_j1_cube_guard_counts_both_cubes():
    """B=2 H=4 Dh=128 must trip (or require chunking) once both cubes count.

    Per-chunk working set is 2*B*H*Tc*Dh^2*4B (updates + prefix_S).
    B=2,H=4,Dh=128 -> 1MB per Tc; Tc=320 (T=320, chunk 512) is ~320MB and
    must refuse, while Tc=128 chunks (~128MB) still pass.
    """
    from linear_attention import linear_attn_forward

    B, H, Dh, T = 2, 4, 128, 320
    torch.manual_seed(0)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    v = torch.randn(B, H, T, Dh)
    import pytest

    with pytest.raises(ValueError, match="256MB cube budget"):
        linear_attn_forward(q, k, v, chunk_size=512)
    # Chunked path stays under budget and runs.
    out, _ = linear_attn_forward(q, k, v, chunk_size=128)
    assert out.shape == (B, T, H, Dh)


def test_j2_width_aware_qkv_budget():
    """rough_state_bytes(B,T,d,D) refuses wide qkv before the matmul."""
    import pytest
    from recurrent import SharedRecurrentBlock, rough_state_bytes
    from linear_attention import LinearAttnState

    # Helper is width-aware: doubling D doubles the estimate (B*T-only is blind).
    assert rough_state_bytes(2, 1024, 3, 256) * 2 == rough_state_bytes(2, 1024, 3, 512)
    assert rough_state_bytes(8, 8192, 3, 512) > 256 * 1024 * 1024
    assert rough_state_bytes(1, 128, 3, 512) < 256 * 1024 * 1024

    # Forward entry refuses the wide case (float32 standalone block).
    torch.manual_seed(0)
    blk = SharedRecurrentBlock(d_model=1024, n_heads=8)
    B, T, D = 8, 4096, 1024
    h = torch.randn(B, T, D)
    attn = LinearAttnState.zeros(B, 8, D // 8, device=h.device)
    moe_calls = []

    def moe_fn(flat):
        moe_calls.append(1)
        return flat, flat.new_zeros(()), flat[:, 0].long() % 8

    with pytest.raises(ValueError, match="256MB budget"):
        blk(h, attn, moe_fn)
    assert moe_calls == []  # refused before any compute
    # Narrow case runs.
    h2 = torch.randn(1, 128, D)
    attn2 = LinearAttnState.zeros(1, 8, D // 8, device=h2.device)
    out, _, _, _ = blk(h2, attn2, moe_fn)
    assert out.shape == (1, 128, D)


def _old_admit_slots(router, top_ids, top_w, n_tokens):
    """Reference pre-vectorization loop (per-expert cumsum) for bit-identity."""
    cap = max(1, int(router.capacity_factor * n_tokens * router.top_k / router.num_experts))
    order = torch.argsort(top_w.max(dim=-1).values, descending=True, stable=True)
    sorted_ids = top_ids[order]
    flat = sorted_ids.reshape(-1)
    admit_flat = torch.zeros(flat.shape[0], dtype=torch.bool, device=top_ids.device)
    for e in range(router.num_experts):
        is_e = flat == e
        rank = is_e.to(torch.int64).cumsum(0) - 1
        admit_flat |= is_e & (rank < cap)
    admit_sorted = admit_flat.reshape(sorted_ids.shape)
    admit = torch.empty_like(admit_sorted)
    admit[order] = admit_sorted
    dropped = ~admit.any(dim=-1)
    return admit, dropped


def test_j3_admit_slots_bit_identical():
    """Vectorized _admit_slots is bit-identical to the old E-loop on seeds."""
    from routing import SparseRouter

    for N, E in [(32, 2), (32, 8), (512, 8), (512, 64), (2048, 8), (2048, 64)]:
        torch.manual_seed(1234)
        r = SparseRouter(d_model=16, num_experts=E, top_k=2, capacity_factor=1.5)
        x = torch.randn(N, 16)
        with torch.no_grad():
            plan = r.route(x, enforce_capacity=False)
            new_admit, new_drop = r._admit_slots(plan.top_ids, plan.top_weights, N)
            old_admit, old_drop = _old_admit_slots(r, plan.top_ids, plan.top_weights, N)
        assert torch.equal(new_admit, old_admit), f"admit differs N={N} E={E}"
        assert torch.equal(new_drop, old_drop), f"dropped differs N={N} E={E}"
        # Bench spot-check (before: N512/E8 ~0.62ms, N2048/E8 ~1.20ms;
        # after: ~0.45ms/~0.77ms — vectorized, E-independent single pass).
        t0 = time.perf_counter()
        for _ in range(10):
            r._admit_slots(plan.top_ids, plan.top_weights, N)
        dt = (time.perf_counter() - t0) / 10 * 1000.0
        assert dt < 50.0, f"admit too slow N={N} E={E}: {dt:.1f}ms"


def test_j4_single_pass_grad_norm():
    """Single-sync norm matches the old per-tensor-double loop (~1e-9).

    Microbench (CPU, 256x256 fp32): before ~1.09/3.79/18.66ms for 4/16/64
    tensors (per-tensor .double() + sync each); after ~0.66/2.58/11.57ms
    with ONE sync and fp32-square/fp64-sum transients. Realistic-scale
    grads (std 0.02) match to <1e-9 absolute; unit-scale bench grads match
    to <1e-9 relative (maxabs reported below).
    """
    import math
    from smaulopt import global_grad_norm

    def old_norm(grads):
        total = 0.0
        for g in grads:
            if g is None:
                continue
            total += float(g.detach().double().pow(2).sum().item())
        return math.sqrt(total)

    # Realistic scale: absolute < 1e-9.
    for nt in (4, 16, 64):
        torch.manual_seed(11)
        grads = [torch.randn(64, 64) * 0.02 for _ in range(nt)]
        o, n = old_norm(grads), global_grad_norm(grads)
        assert abs(o - n) <= 1e-9, f"nt={nt}: {o} vs {n}"
    assert global_grad_norm([]) == 0.0
    assert global_grad_norm([None, None]) == 0.0
    # Unit-scale microbench shapes: relative < 1e-9, timing recorded.
    for nt in (4, 16, 64):
        torch.manual_seed(7)
        grads = [torch.randn(256, 256) for _ in range(nt)]
        o, n = old_norm(grads), global_grad_norm(grads)
        assert abs(o - n) / max(1.0, o) <= 1e-9, f"nt={nt} rel {abs(o-n)/o:.2e}"
        t0 = time.perf_counter()
        for _ in range(5):
            global_grad_norm(grads)
        dt = (time.perf_counter() - t0) / 5 * 1000.0
        print(f"\n[J4] nt={nt}: norm={n:.6f} maxabs={abs(o-n):.2e} {dt:.2f}ms")
        assert dt < 5000.0


def test_m1_expert_stats_single_sync_identical():
    """Batched stats flush equals the old per-expert .item() values exactly.

    Bench (N=512, E=4, CPU): before ~7.25ms/forward (2 .item() syncs per
    expert); after ~6.25ms with one stacked .tolist per forward.
    """
    import copy
    from experts import ExpertPool, make_expert, swiglu_forward
    from routing import SparseRouter

    def old_forward(pool, x, top_ids, top_weights, dropped, provider, step=0):
        out = torch.zeros_like(x)
        live = ~dropped
        admitted = live.unsqueeze(-1) & (top_weights > 0)
        if top_ids.numel() == 0:
            return out
        for rid in torch.unique(top_ids).tolist():
            mask_slot = (top_ids == rid) & admitted
            if not mask_slot.any():
                continue
            eid = pool.order[rid]
            w = provider(eid)
            dtype = w["w_gate"].dtype
            rows = mask_slot.any(dim=-1)
            y = swiglu_forward(x[rows].to(dtype), w).to(out.dtype)
            weight = mask_slot[rows].float() * top_weights[rows]
            out[rows] += weight.sum(dim=-1, keepdim=True).to(out.dtype) * y
            rec = pool.experts[eid]
            rec.tokens_routed += int(rows.sum().item())
            rec.last_used_step = step
            mean_w = float(weight.sum().detach().item()) / max(1, int(rows.sum().item()))
            rec.contribution = 0.9 * rec.contribution + 0.1 * mean_w
        return out

    torch.manual_seed(0)
    pool_new = ExpertPool()
    for _ in range(4):
        pool_new.add(make_expert(pool_new.fresh_id(), d_model=32, expert_hidden=64))
    pool_old = copy.deepcopy(pool_new)
    router = SparseRouter(32, 4, 2)
    torch.manual_seed(1)
    x = torch.randn(512, 32)
    plan = router.route(x)
    pn = lambda eid: pool_new.experts[eid].dequantize(dtype=torch.float32)
    po = lambda eid: pool_old.experts[eid].dequantize(dtype=torch.float32)
    out_new = pool_new.forward(x, plan.top_ids, plan.top_weights, plan.dropped, pn, step=5)
    out_old = old_forward(pool_old, x, plan.top_ids, plan.top_weights, plan.dropped, po, step=5)
    assert torch.equal(out_new, out_old)
    for eid in pool_new.order:
        rn, ro = pool_new.experts[eid], pool_old.experts[eid]
        assert rn.tokens_routed == ro.tokens_routed, eid
        assert rn.last_used_step == ro.last_used_step, eid
        assert rn.contribution == ro.contribution, eid
    t0 = time.perf_counter()
    for _ in range(5):
        pool_new.forward(x, plan.top_ids, plan.top_weights, plan.dropped, pn, step=5)
    dt = (time.perf_counter() - t0) / 5 * 1000.0
    print(f"\n[M1] pool forward N=512 E=4: {dt:.2f}ms")
    assert dt < 5000.0


def test_m3_sparse_head_study_only_budget():
    """SparseTopKHead documents study-only + refuses N*V*K over budget."""
    import pytest
    from kernels import SparseTopKHead, compare_heads

    assert "STUDY-ONLY" in SparseTopKHead.__doc__
    h = SparseTopKHead(256, 128, 32)
    assert h(torch.randn(8, 128)).shape == (8, 256)
    big = SparseTopKHead(8192, 128, 64)
    with pytest.raises(ValueError, match="sparse-gather budget"):
        big(torch.randn(4096, 128))  # 4096*8192*64 = 2.1B, refused pre-gather
    # Dense-vs-sparse microbench still runs (measured verdict, dense wins
    # at V=256: e.g. dense ~2-3ms vs sparse ~58-88ms CPU).
    res = compare_heads(repeat=5)
    assert res["winner"] in ("dense", "sparse")
    print(f"\n[M3] winner={res['winner']} dense={res['dense']['median_ms']:.2f}ms "
          f"sparse={res['sparse']['median_ms']:.2f}ms")


def test_m5_fused_factored_update():
    """Vectorized block-denoms match the 256-loop to atol 1e-6; R8192 <2x."""
    import copy
    from smaulopt import smaul_update, SmaulOptHParams

    def old_update(w, g, state, hp, lr):
        w32 = w.detach().float()
        g32 = g.detach().float()
        state["step"] = int(state.get("step", 0)) + 1
        t = state["step"]
        bm, bv = hp.beta_m, hp.beta_v
        bc1, bc2 = 1.0 - bm**t, 1.0 - bv**t
        m = state["m"].float() * bm + g32 * (1.0 - bm)
        ag = g32.abs()
        vr = state["v_row"].float() * bv + ag.mean(dim=1, keepdim=True) * (1.0 - bv)
        vc = state["v_col"].float() * bv + ag.mean(dim=0, keepdim=True) * (1.0 - bv)
        state["v_row"] = vr.to(torch.bfloat16)
        state["v_col"] = vc.to(torch.bfloat16)
        state["m"] = m.to(torch.bfloat16)
        vr_hat, vc_hat, m_hat = vr / bc2, vc / bc2, m / bc1
        decay = lr * hp.wd
        new_w = torch.empty_like(w32)
        for o in range(0, ag.shape[0], 256):
            sl = slice(o, o + 256)
            rb = vr_hat[sl]
            denom = rb.mean().clamp_min(1e-12)
            v_blk = (rb * vc_hat) / denom
            u_blk = m_hat[sl] / (v_blk + hp.eps)
            if hp.state_dtype != "fp32":
                u_blk = u_blk.clamp(-hp.update_clip, hp.update_clip)
            new_w[sl] = w32[sl] * (1.0 - decay) - lr * u_blk
        return new_w.detach()

    hp = SmaulOptHParams()
    # Bit-close incl. ragged tail (R=1000) and clip path; bench R=8192.
    for R, C in [(512, 128), (1000, 64)]:
        torch.manual_seed(3)
        w, g = torch.randn(R, C), torch.randn(R, C)
        st_n = {"m": torch.zeros(R, C, dtype=torch.bfloat16),
                "v_row": torch.zeros(R, 1, dtype=torch.bfloat16),
                "v_col": torch.zeros(1, C, dtype=torch.bfloat16), "step": 0}
        st_o = copy.deepcopy(st_n)
        out_n = smaul_update(w, g, st_n, hp, lr=2e-4)
        out_o = old_update(w, g, st_o, hp, lr=2e-4)
        assert torch.allclose(out_n, out_o, atol=1e-6), \
            f"R={R} maxabs={(out_n - out_o).abs().max().item():.2e}"
    torch.manual_seed(3)
    w, g = torch.randn(8192, 256), torch.randn(8192, 256)
    mk = lambda: {"m": torch.zeros(8192, 256, dtype=torch.bfloat16),
                  "v_row": torch.zeros(8192, 1, dtype=torch.bfloat16),
                  "v_col": torch.zeros(1, 256, dtype=torch.bfloat16), "step": 0}
    t0 = time.perf_counter()
    old_update(w, g, mk(), hp, lr=2e-4)
    t_old = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    out_n = smaul_update(w, g, mk(), hp, lr=2e-4)
    t_new = (time.perf_counter() - t0) * 1000.0
    assert torch.isfinite(out_n).all()
    print(f"\n[M5] R8192: old={t_old:.1f}ms new={t_new:.1f}ms ratio={t_new/max(t_old,1e-9):.2f}x")
    assert t_new < 2 * t_old, f"fused slower than 2x: {t_new:.1f} vs {t_old:.1f}"
    # Range path still runs (same fused math on the slice).
    torch.manual_seed(4)
    w2, g2 = torch.randn(512, 64), torch.randn(512, 64)
    st2 = {"m": torch.zeros(512, 64, dtype=torch.bfloat16),
           "v_row": torch.zeros(512, 1, dtype=torch.bfloat16),
           "v_col": torch.zeros(1, 64, dtype=torch.bfloat16), "step": 0}
    sl = smaul_update(w2, g2, st2, hp, lr=2e-4, row_start=0, row_end=256)
    assert sl.shape == (256, 64) and torch.isfinite(sl).all()


def test_m2_depth_cost_floor():
    """M2 (doc-only): per-depth D-matmuls + no early exit — cost floor.

    The depth loop always runs max_depth (streaming parity + ponder need
    every step); adaptivity is readout-only. No execution change here
    (model.py owned by correctness); this pins the measured floor so a
    future early-exit change must beat it without breaking parity.
    Measured CPU (identity-MoE, B=2): D=256/T=256 ~290ms per depth,
    D=1024/T=128 ~1130ms (~D^2 scaling); 4-depth chain ~4x single.
    """
    from recurrent import SharedRecurrentBlock
    from linear_attention import LinearAttnState

    assert "M2" in SharedRecurrentBlock.__doc__
    torch.manual_seed(0)
    blk = SharedRecurrentBlock(d_model=256, n_heads=4)
    h = torch.randn(2, 256, 256)
    attn = LinearAttnState.zeros(2, 4, 64)
    moe = lambda flat: (flat, flat.new_zeros(()),
                        torch.zeros(flat.shape[0], dtype=torch.long))
    for _ in range(2):
        blk(h, attn, moe)
    t0 = time.perf_counter()
    for _ in range(3):
        blk(h, attn, moe)
    dt = (time.perf_counter() - t0) / 3 * 1000.0
    print(f"\n[M2] D=256 B=2 T=256 per-depth: {dt:.1f}ms (floor, no early exit)")
    assert dt < 60_000.0
    # Depth scaling is linear (no early exit): 2 chained apps ~2x one.
    t0 = time.perf_counter()
    for _ in range(3):
        h1, a1, _, _ = blk(h, attn, moe)
        blk(h1, a1, moe)
    dt2 = (time.perf_counter() - t0) / 3 * 1000.0
    print(f"[M2] 2-depth chain: {dt2:.1f}ms (ratio {dt2/max(dt,1e-9):.2f}x)")
    assert 1.2 * dt < dt2 < 3.5 * dt


def test_m6_generate_loop_floor():
    """M6 (doc-only, infer.py untouched): per-token generate loop floor.

    ``generate`` steps max_new times through ``forward_infer_step`` with
    T=1 stateful calls (matmuls never amortize over T) plus a per-step
    ``.item()`` (depths) and sampler on the hot path. No execution change
    here (infer.py owned by correctness); this pins the measured floor.
    Measured CPU (tiny d32/4-exp/max_depth=2 model): ~8.2ms/token
    (~120 tok/s) for max_new=8; larger widths scale ~D^2 per step.
    """
    from config import SmaulBrainConfig
    from model import SmaulBrainModel
    from infer import generate

    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=1,
                           expert_hidden=64, min_depth=1, max_depth=2,
                           context_length=512)
    m = SmaulBrainModel(cfg)
    m.eval()
    try:
        r = generate(m, [65, 66, 67, 68], max_new=4, seed=0)
        assert len(r["ids"]) == 4 + 4
        t0 = time.perf_counter()
        generate(m, [65, 66, 67, 68], max_new=4, seed=0)
        dt = (time.perf_counter() - t0) / 4 * 1000.0
        print(f"\n[M6] generate per-token floor: {dt:.1f}ms/token "
              f"({1000.0/max(dt, 1e-9):.0f} tok/s, tiny model)")
        assert dt < 60_000.0
    finally:
        m.pager.close()


def test_m4_routing_budget_counts_both_transients():
    """N*E guard halves to 8M to cover logits+probs (~2x actual)."""
    import pytest
    from routing import SparseRouter

    torch.manual_seed(0)
    # E=64: N=150k -> N*E=9.6M > 8M trips pre-alloc (x itself is 150k*8 floats).
    r = SparseRouter(d_model=8, num_experts=64, top_k=2)
    x = torch.randn(150_000, 8)
    with pytest.raises(ValueError, match="8M-element routing budget"):
        r.route(x)
    # Under budget runs.
    x2 = torch.randn(1024, 8)
    plan = r.route(x2)
    assert plan.top_ids.shape == (1024, 2)
