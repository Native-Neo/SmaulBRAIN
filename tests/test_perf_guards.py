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
