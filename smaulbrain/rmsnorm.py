"""RMSNorm used at every major recurrent/residual boundary.

Normalization math runs in FP32 even when inputs are BF16 (precision policy:
normalization statistics are sensitive), then casts back to the input dtype.
Implemented with vectorized tensor ops only — no Python per-element loops —
so the same code path is the CPU hot path (see kernels_cpp/rmsnorm.cpp for
the optional native kernel, numerically equivalent).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm without mean centering or bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps) * self.weight.to(torch.float32)
        return out.to(dtype)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"


def rmsnorm_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional RMSNorm with the same FP32-statistics contract."""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.float()).to(dtype)
