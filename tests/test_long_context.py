"""Long context: measure peak RSS at 1K/2K/4K/8K, prove sub-quadratic growth.

A quadratic attention design needs ~16x memory for 4x length; this
architecture must stay far below that (linear activations + T-independent
recurrent state). Numbers are printed and asserted — 16K support is claimed
only for the lengths actually measured (see VALIDATION.md for 16K).
"""

import sys, os, resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from smaulbrain.config import SmaulBrainConfig
from smaulbrain.kernels import linear_attn_memory_bound
from smaulbrain.linear_attention import LinearAttnState
from smaulbrain.model import SmaulBrainModel


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _model():
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=32, n_heads=4, num_experts=4, top_k=1,
                           expert_hidden=64, min_depth=1, max_depth=1,
                           context_length=8192)
    return SmaulBrainModel(cfg)


def test_attention_state_bytes_constant_across_lengths():
    assert linear_attn_memory_bound(4, 8) == (4 * 64 + 4 * 8) * 4
    s = LinearAttnState.zeros(1, 4, 8)
    assert s.nbytes() == linear_attn_memory_bound(4, 8)


def test_rss_growth_subquadratic():
    m = _model()
    base = _rss_mb()
    rows = []
    for T in (1024, 2048, 4096, 8192):
        ids = torch.randint(0, 256, (1, T))
        out = m.forward_infer(ids)
        assert out["logits"].shape == (1, T, 256)
        peak = _rss_mb() - base
        rows.append((T, peak))
        print(f"\n[T={T}] delta-peak-RSS={peak:.1f}MB executed={out['n_executed']}")
    for (t0, p0), (t1, p1) in zip(rows, rows[1:]):
        ratio = (p1 + 1.0) / (p0 + 1.0)
        # Length doubles: quadratic => ~4x. Require clearly sub-quadratic (<3x).
        assert ratio < 3.0, f"{t0}->{t1}: {ratio:.2f}x"
    m.pager.close()
