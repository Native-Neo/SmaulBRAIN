"""Shared recurrent state and the shared recurrent block.

One ``SharedRecurrentBlock`` instance is applied min_depth..max_depth times per
forward pass (depth recurrence, as in mini-AGI's RecurCoder). The block owns:
RMSNorm -> linear-attention projections -> RMSNorm -> residual ->
sparse MoE (RMSNorm -> routing -> experts) -> RMSNorm -> residual,
plus a halt head for adaptive depth.

The recurrence is genuine: step n+1 reads both the hidden vector ``h`` and
the linear-attention accumulator ``(S, z)`` written by step n. Throwing the
state away would change the outputs (covered by tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_attention import LinearAttnState, linear_attn_forward
from rmsnorm import RMSNorm


@dataclass
class RecurrentState:
    """Carried across recurrent applications within one forward pass.

    Ownership: the block never mutates its inputs; ``forward`` returns new
    ``(h, attn)`` tensors. Reset starts from ``RecurrentState.zeros(...)``;
    continue passes the returned state back in. ``reset(indices)`` zeroes
    selected batch rows for per-sequence reuse.
    """

    h: torch.Tensor  # [B, T, D] hidden vectors (BF16 compute)
    attn: LinearAttnState  # [B, H, Dh, Dh] + [B, H, Dh] accumulators
    steps_taken: int = 0

    @classmethod
    def zeros(cls, batch: int, seq: int, d_model: int, n_heads: int, device=None,
              dtype: torch.dtype = torch.float32) -> "RecurrentState":
        """Fresh stream state: zero hidden + zero attention accumulators."""
        assert d_model % n_heads == 0
        return cls(
            h=torch.zeros(batch, seq, d_model, device=device, dtype=dtype),
            attn=LinearAttnState.zeros(batch, n_heads, d_model // n_heads, device=device),
            steps_taken=0,
        )

    def reset(self, indices=None) -> "RecurrentState":
        """Zero selected batch rows in place (per-sequence reset)."""
        if indices is None:
            self.h.zero_()
            self.attn.reset()
        else:
            idx = torch.as_tensor(indices, dtype=torch.long, device=self.h.device)
            self.h[idx] = 0.0
            self.attn.reset(idx)
        self.steps_taken = 0
        return self

    def clone(self) -> "RecurrentState":
        return RecurrentState(h=self.h.clone(), attn=self.attn.clone(), steps_taken=self.steps_taken)


#: MoE hook type: (x_flat [N, D]) -> (y_flat [N, D], aux_loss, usage_ids)
MoeFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class SharedRecurrentBlock(nn.Module):
    """The single shared block applied repeatedly by the depth loop."""

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n1 = RMSNorm(d_model, eps)
        self.n_attn = RMSNorm(d_model, eps)
        self.n2 = RMSNorm(d_model, eps)
        self.n3 = RMSNorm(d_model, eps)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # Halt head: bias starts negative so the model ponders before halting
        # (same rationale as mini-AGI's halt.bias = -2.0).
        self.halt = nn.Linear(d_model, 1)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        attn: LinearAttnState,
        moe_fn: MoeFn,
        chunk_size: int = 256,
        keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LinearAttnState, torch.Tensor, torch.Tensor]:
        """One recurrent application.

        Returns (h_next, attn_next, halt_logit [B, T], aux_loss).
        halt_logit is the raw (pre-sigmoid) per-token halting score.

        Ownership: inputs are never mutated; new tensors are returned.
        Reset with fresh zeros; continue by feeding back the returned
        state. ``keep`` (optional bool [B, T]) marks valid vs padded
        positions and is forwarded to linear attention so pads add
        nothing to the accumulators. Outputs are chunk-size invariant
        (chunking only changes summation order within float tolerance).
        The halt head reads ``h.detach()`` so halting gradients never
        flow into the trunk (halt learns to predict, not to steer).
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape [B, T, D], got {tuple(h.shape)}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        B, T, D = h.shape
        if D != self.d_model:
            raise ValueError(f"h width {D} != d_model {self.d_model}")
        if not attn.matches(B, self.n_heads, self.head_dim):
            raise ValueError(
                f"attn state mismatch: S={tuple(attn.S.shape)}/{attn.S.dtype} "
                f"z={tuple(attn.z.shape)}/{attn.z.dtype}, expected B={B} "
                f"H={self.n_heads} Dh={self.head_dim} fp32"
            )
        if keep is not None and keep.shape != (B, T):
            raise ValueError(f"keep must have shape [B, T]=[{B}, {T}], got {tuple(keep.shape)}")
        # -- linear attention branch (residual) --
        a = self.n1(h)
        qkv = self.qkv(a).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, Dh]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # Chunkwise causal linear attention. The incoming state is visible
        # to every token; within each chunk only earlier-token prefixes are used.
        # The input state is never mutated (linear_attn_forward clones).
        keep_mask = None
        if keep is not None:
            keep_mask = keep.to(dtype=torch.bool, device=h.device)
        y, attn_next = linear_attn_forward(
            q, k, v, eps=1e-6, state=attn, chunk_size=chunk_size, keep=keep_mask
        )
        y = y.reshape(B, T, D).to(h.dtype)
        h = h + self.o_proj(self.n_attn(y))
        # -- sparse MoE branch (Norm -> residual, injected routing+experts) --
        m = self.n2(h)
        flat = m.reshape(B * T, D)
        moe_out, aux_loss, _usage = moe_fn(flat)
        h = h + self.n3(moe_out.reshape(B, T, D).to(h.dtype))
        halt_logit = self.halt(h.detach()).squeeze(-1)  # [B, T], no halt-grad into trunk
        return h, attn_next, halt_logit, aux_loss
