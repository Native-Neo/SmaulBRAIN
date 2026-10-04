"""Genuine linear attention with a recurrent/streamable state.

Form (per head): with a non-negative feature map phi (ELU+1, as verified in
the SmaulNative reference), the causal output is

    S_t = S_{t-1} + phi(k_t)^T v_t        # [Dh, Dh] accumulator matrix
    z_t = z_{t-1} + phi(k_t)              # [Dh] normalizer vector
    y_t = (phi(q_t)^T S_t) / (phi(q_t)^T z_t + eps)

Complexity is O(T * Dh^2) time and O(Dh^2) memory — linear in sequence length
T. There is no QK^T matrix anywhere: scores are never formed pairwise. The
(S, z) pair is the streamable state: ``step()`` advances one token with
bounded memory, and ``prefill()`` folds a chunk left-to-right.

Multi-head projections (qkv/o) intentionally live in the recurrent block
(``recurrent.py``); this module owns only the head-wise recurrence math so
incremental inference and batched training share one verified code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1: positive feature map enabling the kernel linearization."""
    return F.elu(x) + 1.0


@dataclass
class LinearAttnState:
    """Recurrent attention state. Memory is O(H*Dh^2), independent of T."""

    S: torch.Tensor  # [B, H, Dh, Dh] fp32 accumulator
    z: torch.Tensor  # [B, H, Dh] fp32 normalizer

    @classmethod
    def zeros(cls, batch: int, heads: int, head_dim: int, device=None) -> "LinearAttnState":
        return cls(
            S=torch.zeros(batch, heads, head_dim, head_dim),
            z=torch.zeros(batch, heads, head_dim),
        )

    def clone(self) -> "LinearAttnState":
        return LinearAttnState(S=self.S.clone(), z=self.z.clone())

    def nbytes(self) -> int:
        return self.S.nelement() * 4 + self.z.nelement() * 4


def linear_attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, LinearAttnState]:
    """Batched causal linear attention over [B, H, T, Dh] inputs.

    Returns outputs [B, H, T, Dh] plus the terminal recurrent state.
    The loop over T uses only rank-1 outer-product updates (no QK^T).
    Vectorized across B and H; the T loop is a genuine recurrence where
    step t reads state written by step t-1.
    """
    B, H, T, Dh = q.shape
    qf = feature_map(q.float())
    kf = feature_map(k.float())
    vf = v.float()
    # Per-step key normalization keeps the accumulator bounded (SmaulNative).
    kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    S = torch.zeros(B, H, Dh, Dh)
    z = torch.zeros(B, H, Dh)
    outs: list[torch.Tensor] = []
    for t in range(T):
        kt = kf[:, :, t, :]  # [B, H, Dh]
        vt = vf[:, :, t, :]
        qt = qf[:, :, t, :]
        S = S + kt.unsqueeze(-1) * vt.unsqueeze(-2)  # rank-1 outer update
        z = z + kt
        num = torch.einsum("bhd,bhde->bhe", qt, S)
        den = (qt * z).sum(dim=-1, keepdim=True).clamp_min(eps)
        outs.append((num / den).to(v.dtype))
    out = torch.stack(outs, dim=2)
    return out, LinearAttnState(S=S, z=z)


def linear_attn_step(
    state: LinearAttnState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, LinearAttnState]:
    """Advance the recurrent state by one token ([B, H, Dh] each).

    Returns the output for this token and the updated state (in place).
    This is the incremental-inference path with O(1) memory per step.
    """
    qf = feature_map(q.float())
    kf = feature_map(k.float())
    vf = v.float()
    kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    state.S += kf.unsqueeze(-1) * vf.unsqueeze(-2)
    state.z += kf
    num = torch.einsum("bhd,bhde->bhe", qf, state.S.float())
    den = (qf * state.z).sum(dim=-1, keepdim=True).clamp_min(eps)
    return (num / den).to(v.dtype), state
