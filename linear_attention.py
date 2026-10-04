"""Genuine linear attention with a recurrent/streamable state.

Form (per head): with a non-negative feature map phi (ELU+1, as verified in
the SmaulNative reference), the causal output is

    S_t = S_{t-1} + phi(k_t)^T v_t        # [Dh, Dh] accumulator matrix
    z_t = z_{t-1} + phi(k_t)              # [Dh] normalizer vector
    y_t = (phi(q_t)^T S_t) / (phi(q_t)^T z_t + eps)

Complexity is O(T * Dh^2) time and O(Dh^2) memory — linear in sequence length
T. There is no QK^T matrix anywhere: scores are never formed pairwise. The
(S, z) pair is the streamable state: ``linear_attn_step`` advances one token
with bounded memory, and ``linear_attn_forward`` folds a chunk left-to-right.

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
            S=torch.zeros(batch, heads, head_dim, head_dim, device=device),
            z=torch.zeros(batch, heads, head_dim, device=device),
        )

    def clone(self) -> "LinearAttnState":
        return LinearAttnState(S=self.S.clone(), z=self.z.clone())

    def nbytes(self) -> int:
        return self.S.nelement() * 4 + self.z.nelement() * 4


def linear_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    state: LinearAttnState | None = None,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, LinearAttnState]:
    """Causal linear attention with vectorized work inside each chunk.

    The additive state makes causal prefixes associative. Each chunk computes
    all of its token prefixes with cumsum; only the terminal state crosses the
    chunk boundary. There is no Python loop over individual tokens and no QK^T.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must all have shape [B, H, T, Dh]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    B, H, T, Dh = q.shape
    if state is None:
        state = LinearAttnState.zeros(B, H, Dh, device=q.device)
    S = state.S.float()
    z = state.z.float()
    qf = feature_map(q.float())
    kf = feature_map(k.float())
    vf = v.float()
    kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    chunks: list[torch.Tensor] = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        kc = kf[:, :, start:end, :]
        vc = vf[:, :, start:end, :]
        qc = qf[:, :, start:end, :]
        updates = kc.unsqueeze(-1) * vc.unsqueeze(-2)
        prefix_S = updates.cumsum(dim=2) + S.unsqueeze(2)
        prefix_z = kc.cumsum(dim=2) + z.unsqueeze(2)
        num = torch.einsum("bhcd,bhcde->bhce", qc, prefix_S)
        den = (qc * prefix_z).sum(dim=-1, keepdim=True).clamp_min(eps)
        chunks.append((num / den).to(v.dtype).transpose(1, 2))
        S = prefix_S[:, :, -1]
        z = prefix_z[:, :, -1]

    out = torch.cat(chunks, dim=1)
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
