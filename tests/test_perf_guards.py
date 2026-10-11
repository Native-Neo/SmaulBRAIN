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
