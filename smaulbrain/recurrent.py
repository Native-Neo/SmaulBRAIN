"""Shared recurrent state and the shared recurrent block.

One ``SharedRecurrentBlock`` instance is applied min_depth..max_depth times per
forward pass (depth recurrence, as in mini-AGI's RecurCoder). The block owns:
RMSNorm -> linear-attention projections -> RMSNorm -> residual ->
MoE (injected) -> RMSNorm -> residual, plus a halt head for adaptive depth.

The recurrence is genuine: step n+1 reads both the hidden vector ``h`` and
the linear-attention accumulator ``(S, z)`` written by step n. Throwing the
state away would change the outputs (covered by tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .linear_attention import LinearAttnState
from .rmsnorm import RMSNorm


@dataclass
class RecurrentState:
    """Carried across recurrent applications within one forward pass."""

    h: torch.Tensor  # [B, T, D] hidden vectors (BF16 compute)
    attn: LinearAttnState  # [B, H, Dh, Dh] + [B, H, Dh] accumulators
    steps_taken: int = 0

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
    ) -> tuple[torch.Tensor, LinearAttnState, torch.Tensor, torch.Tensor]:
        """One recurrent application.

        Returns (h_next, attn_next, halt_logit [B, T], aux_loss).
        halt_logit is the raw (pre-sigmoid) per-token halting score.
        """
        B, T, D = h.shape
        # -- linear attention branch (residual) --
        a = self.n1(h)
        qkv = self.qkv(a).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, Dh]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # Continue from the incoming accumulator: genuine state influence.
        S, z = attn.S, attn.z
        qf = torch.nn.functional.elu(q.float()) + 1.0
        kf = torch.nn.functional.elu(k.float()) + 1.0
        kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vf = v.float()
        outs: list[torch.Tensor] = []
        for t in range(T):
            kt, vt, qt = kf[:, :, t, :], vf[:, :, t, :], qf[:, :, t, :]
            S = S + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            z = z + kt
            num = torch.einsum("bhd,bhde->bhe", qt, S)
            den = (qt * z).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            outs.append(num / den)
        y = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, T, D).to(h.dtype)
        h = h + self.o_proj(y)
        attn_next = LinearAttnState(S=S, z=z)
        # -- sparse MoE branch (residual, injected) --
        m = self.n2(h)
        flat = m.reshape(B * T, D)
        moe_out, aux_loss, _usage = moe_fn(flat)
        h = h + moe_out.reshape(B, T, D)
        h = self.n3(h)  # norm boundary: stabilizes the next recurrent application
        halt_logit = self.halt(h.detach()).squeeze(-1)  # [B, T], no halt-grad into trunk
        return h, attn_next, halt_logit, aux_loss
