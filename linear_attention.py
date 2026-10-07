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

import native


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1: positive feature map enabling the kernel linearization."""
    return F.elu(x) + 1.0


@dataclass
class LinearAttnState:
    """Recurrent attention state. Memory is O(H*Dh^2), independent of T.

    Ownership contract (explicit):

    * ``linear_attn_forward`` never mutates its input ``state``; it works
      on a private clone and returns a brand-new ``LinearAttnState``.
      Callers keep owning the input: reset (fresh zeros) starts a new
      stream, passing the returned state continues it.
    * ``linear_attn_step`` updates ``state`` in place (bounded O(1) memory)
      and returns ``(y, state)`` with the same object identity.
    * ``reset(indices)`` zeroes selected batch rows in place for
      per-sequence reset without reallocating.
    * Context truncation is a caller-level reset + re-feed of the last
      ``context`` tokens; the state itself never grows with T.
    """

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

    def reset(self, indices=None) -> "LinearAttnState":
        """Zero the accumulators in place (fresh stream).

        ``indices`` is None (reset whole batch) or batch indices to reset
        for per-sequence reuse of a pooled state. Returns self.
        """
        if indices is None:
            self.S.zero_()
            self.z.zero_()
            return self
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.S.device)
        self.S[idx] = 0.0
        self.z[idx] = 0.0
        return self

    def matches(self, batch: int, heads: int, head_dim: int) -> bool:
        """The one state contract: fp32 accumulators of exact expected shape."""
        return (
            self.S.shape == (batch, heads, head_dim, head_dim)
            and self.z.shape == (batch, heads, head_dim)
            and self.S.dtype == torch.float32
            and self.z.dtype == torch.float32
        )

    def nbytes(self) -> int:
        return self.S.nelement() * 4 + self.z.nelement() * 4


def linear_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    state: LinearAttnState | None = None,
    chunk_size: int = 256,
    keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LinearAttnState]:
    """Causal linear attention with vectorized work inside each chunk.

    The additive state makes causal prefixes associative. Each chunk computes
    all of its token prefixes with cumsum; only the terminal state crosses the
    chunk boundary. There is no Python loop over individual tokens and no QK^T.

    Ownership: the input ``state`` is never mutated; a private clone is
    folded and a new ``LinearAttnState`` is returned. Pass the returned
    state to continue a stream, or fresh zeros (or ``reset()``) to reset.

    ``keep`` (optional bool [B, T]): False marks padded positions, which
    contribute nothing to (S, z) so trailing/ragged pads cannot pollute a
    continued stream. Outputs at padded positions stay finite (ratio over
    the carried state). Variable-length batches should be padded and
    passed with their ``keep`` mask.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must all have shape [B, H, T, Dh]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    B, H, T, Dh = q.shape
    if state is None:
        state = LinearAttnState.zeros(B, H, Dh, device=q.device)
    if not state.matches(B, H, Dh):
        raise ValueError(
            f"state shape/dtype mismatch: S={tuple(state.S.shape)}/{state.S.dtype} "
            f"z={tuple(state.z.shape)}/{state.z.dtype}, "
            f"expected ([{B}, {H}, {Dh}, {Dh}]/fp32, [{B}, {H}, {Dh}]/fp32)"
        )
    if keep is not None:
        if keep.shape != (B, T):
            raise ValueError(f"keep must have shape [B, T]=[{B}, {T}], got {tuple(keep.shape)}")
        keep = keep.to(dtype=torch.bool, device=q.device)
    if T == 0:
        empty = torch.empty(B, 0, H, Dh, dtype=v.dtype, device=v.device)
        return empty, state.clone()
    # Private working copies: the caller's state must not alias the fold.
    S = state.S.clone().float()
    z = state.z.clone().float()
    qf = feature_map(q.float())
    kf = feature_map(k.float())
    vf = v.float()
    kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if keep is not None:
        m = keep.view(B, 1, T, 1).to(dtype=vf.dtype)
        kf = kf * m
        vf = vf * m

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
    use_native: bool = False,
    keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LinearAttnState]:
    """Advance the recurrent state by one token ([B, H, Dh] each).

    Returns the output for this token and the updated state (in place:
    the returned object IS ``state``). This is the incremental-inference
    path with O(1) memory per step.

    ``keep`` (optional bool [B]): False skips the state update for that
    batch row (padded step), returning the output over the carried state.

    ``use_native`` (or ``SMAUL_NATIVE=1``) routes through the C++ kernel.
    Native has no autograd rule, so it engages only with gradients disabled;
    results agree with the reference path up to float summation order.
    Masked steps (``keep`` given) always use the reference path so
    per-row skip semantics stay exact.
    """
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must all have shape [B, H, Dh]")
    B, H, Dh = q.shape
    if not state.matches(B, H, Dh):
        raise ValueError(
            f"state shape/dtype mismatch: S={tuple(state.S.shape)}/{state.S.dtype} "
            f"z={tuple(state.z.shape)}/{state.z.dtype}, "
            f"expected ([{B}, {H}, {Dh}, {Dh}]/fp32, [{B}, {H}, {Dh}]/fp32)"
        )
    keep_mask: torch.Tensor | None = None
    if keep is not None:
        keep_mask = torch.as_tensor(keep, dtype=torch.bool, device=q.device).reshape(B)
    if keep_mask is None and (use_native or native.native_mode() == "force") and not torch.is_grad_enabled():
        y_nat = _native_step(state, q, k, v, eps)
        if y_nat is not None:
            return y_nat, state
    qf = feature_map(q.float())
    kf = feature_map(k.float())
    vf = v.float()
    kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if keep_mask is not None:
        m = keep_mask.view(B, 1, 1).to(dtype=kf.dtype)
        kf = kf * m
        vf = vf * m
    state.S += kf.unsqueeze(-1) * vf.unsqueeze(-2)
    state.z += kf
    num = torch.einsum("bhd,bhde->bhe", qf, state.S.float())
    den = (qf * state.z).sum(dim=-1, keepdim=True).clamp_min(eps)
    return (num / den).to(v.dtype), state


def _native_step(
    state: LinearAttnState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """One native step over [B, H, Dh] inputs. None = caller must fall back.

    State tensors update in place, so they must already be contiguous: a
    defensive copy would silently discard the update.
    """
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        return None
    B, H, Dh = q.shape
    S, z = state.S, state.z
    if S.shape != (B, H, Dh, Dh) or z.shape != (B, H, Dh):
        return None
    # Snapshot first: per-head updates mutate state in place, so a mid-loop
    # failure must restore clean state — otherwise the caller's reference
    # fallback would double-apply the already-mutated heads.
    S0, z0 = S.clone(), z.clone()
    outs = torch.empty(B, H, Dh, dtype=torch.float32)
    ok = True
    for b in range(B):
        for h in range(H):
            scratch = torch.empty(2 * Dh, dtype=torch.float32)
            y = torch.empty(Dh, dtype=torch.float32)
            if not native.call_attn_step(
                S[b, h], z[b, h],
                q[b, h].to(torch.float32), k[b, h].to(torch.float32),
                v[b, h].to(torch.float32), y, scratch, Dh, eps,
            ):
                ok = False
                break
            outs[b, h] = y
        if not ok:
            break
    if not ok:
        S.copy_(S0)
        z.copy_(z0)
        return None
    return outs.to(v.dtype)
