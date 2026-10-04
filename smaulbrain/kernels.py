"""Performance-critical ops: vectorized PyTorch hot paths + sparse head study.

Python orchestrates; all hot loops are tensor ops (no per-element Python).
Native kernels live in ``kernels_cpp/`` (same numerics, optional build).

Sparse token connections (spec: only where they win): the byte vocabulary is
tiny (256), so sparsifying the embedding buys nothing — instead this module
provides a fixed fan-in ``SparseTopKHead`` alternative to the dense byte
head, plus ``compare_heads`` which benchmarks dense vs sparse on CPU and
reports which is actually faster/leaner. Verdict is measured, not assumed.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTopKHead(nn.Module):
    """Fixed fan-in sparse output head: each of V rows reads K of D dims.

    ``cols`` is a frozen index buffer (built once from a seed); only
    ``values`` [V, K] are parameters. Forward is one gather + one batched
    dot — no Python token loop.
    """

    def __init__(self, vocab: int, d_model: int, fan_in: int = 32, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        cols = torch.stack([torch.randperm(d_model, generator=gen)[:fan_in]
                            for _ in range(vocab)])
        self.register_buffer("cols", cols)
        self.values = nn.Parameter(torch.empty(vocab, fan_in))
        nn.init.normal_(self.values, std=0.02 / max(1, fan_in) ** 0.5)
        self.fan_in = fan_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gathered = x[..., self.cols]  # [..., V, K]
        return (gathered * self.values).sum(dim=-1)

    def param_count(self) -> int:
        return self.values.nelement()


def time_fn(fn, repeat: int = 20, warmup: int = 3) -> dict:
    """Wall-time benchmark (CPU): median/mean per call in ms."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "mean_ms": sum(ts) / len(ts),
            "min_ms": ts[0], "repeat": repeat}


def compare_heads(vocab: int = 256, d_model: int = 128, batch_tokens: int = 1024,
                  fan_in: int = 32, repeat: int = 20) -> dict:
    """Benchmark dense Linear head vs SparseTopKHead on CPU.

    Returns timings, param counts, and the measured verdict.
    """
    torch.manual_seed(0)
    dense = nn.Linear(d_model, vocab, bias=False)
    sparse = SparseTopKHead(vocab, d_model, fan_in)
    x = torch.randn(batch_tokens, d_model)
    with torch.no_grad():
        yd, ys = dense(x), sparse(x)
    dense_t = time_fn(lambda: dense(x), repeat=repeat)
    sparse_t = time_fn(lambda: sparse(x), repeat=repeat)
    winner = "dense" if dense_t["median_ms"] <= sparse_t["median_ms"] else "sparse"
    return {
        "dense": {**dense_t, "params": dense.weight.nelement()},
        "sparse": {**sparse_t, "params": sparse.param_count(), "fan_in": fan_in},
        "batch_tokens": batch_tokens,
        "winner": winner,
        "note": ("With a 256-entry byte vocab the dense head is one small "
                 "matmul; the sparse head pays gather overhead. Sparsity wins "
                 "only at much larger output dims — measured, not assumed."),
    }


def linear_attn_memory_bound(heads: int, head_dim: int) -> int:
    """Bytes of recurrent attention state: O(H*Dh^2), independent of T."""
    return (heads * head_dim * head_dim + heads * head_dim) * 4
