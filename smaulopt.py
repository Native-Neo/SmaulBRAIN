"""SmaulOpt optimizer ported from the verified SmaulNative implementation.

Reference: SmaulNative ``model.py`` SmaulOpt v2.1 (default training optimizer
there; AdamW is NOT used in its training path). Ported behavior:

    m_t     = beta_m * m_{t-1} + (1 - beta_m) * g_t
    v_t     = beta_v * v_{t-1} + (1 - beta_v) * |g_t|      # |g|, not g^2
    m_hat   = m_t / (1 - beta_m^t);  v_hat = v_t / (1 - beta_v^t)
    u_t     = m_hat / (v_hat + eps)
    theta_t = theta_{t-1} - lr * u_t - lr * wd * theta_{t-1}  # decoupled WD

  * Factored second moment for 2-D weights: row/col marginal EMAs of |g|,
    reconstructed per row-block as outer(R_hat, C_hat) / mean(R_hat), so the
    full [R, C] moment matrix is never materialized (SmaulNative ``factor_v``).
  * State storage BF16 (default) with FP32 update math; ``update_clip``
    bounds |u| for narrow state; global grad-norm clip in float64 with
    non-finite guard that skips the step.
  * Expert-local states live on the ``ExpertRecord`` (``rec.optim_state``) so
    they follow the expert across disk/RAM/VRAM and die on pruning; trunk
    states are name-keyed in the optimizer.

What was NOT ported: nothing was invented — Lion (the other SmaulNative
optimizer) is intentionally omitted; SmaulBRAIN standardizes on SmaulOpt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from precision import (
    dequantize_fp8_blockwise,
    dequantize_fp8_row_block,
    quantize_fp8_blockwise,
    update_fp8_row_block,
)



@dataclass
class SmaulOptHParams:
    lr: float = 2e-4
    beta_m: float = 0.9
    beta_v: float = 0.999
    eps: float = 1e-8
    wd: float = 0.01
    clip: float = 1.0
    state_dtype: str = "bf16"  # bf16 | fp32
    factor_v: bool = True
    update_clip: float = 10.0


def _store_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float32


def _factored(shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and shape[0] >= 2 and shape[1] >= 2


def init_state(shape: tuple[int, ...], hparams: SmaulOptHParams) -> dict:
    """Zero state for one tensor: full m, full or factored v."""
    sd = _store_dtype(hparams.state_dtype)
    st: dict = {"m": torch.zeros(shape, dtype=sd), "step": 0}
    if hparams.factor_v and _factored(shape):
        st["v_row"] = torch.zeros((shape[0], 1), dtype=sd)
        st["v_col"] = torch.zeros((1, shape[1]), dtype=sd)
    else:
        st["v"] = torch.zeros(shape, dtype=sd)
    return st


def smaul_update(
    w: torch.Tensor,
    g: torch.Tensor,
    state: dict,
    hparams: SmaulOptHParams,
    lr: float,
    row_start: int | None = None,
    row_end: int | None = None,
) -> torch.Tensor:
    """One SmaulOpt update in FP32. Returns the new weight (fp32, detached).

    Supports optional block/range updates via row_start and row_end. When
    specified, only rows in [row_start:row_end] are updated in optimizer
    state, and the updated row block is returned.
    """
    if row_start is None and row_end is None:
        w32 = w.detach().float()
        g32 = g.detach().float()
        state["step"] = int(state.get("step", 0)) + 1
        t = state["step"]
        bm, bv = hparams.beta_m, hparams.beta_v
        bc1 = 1.0 - bm**t
        bc2 = 1.0 - bv**t

        m = state["m"].float() * bm + g32 * (1.0 - bm)
        ag = g32.abs()
        if "v" in state:
            v = state["v"].float() * bv + ag * (1.0 - bv)
            state["v"] = v.to(_store_dtype(hparams.state_dtype))
            v_hat = v / bc2
        else:
            vr = state["v_row"].float() * bv + ag.mean(dim=1, keepdim=True) * (1.0 - bv)
            vc = state["v_col"].float() * bv + ag.mean(dim=0, keepdim=True) * (1.0 - bv)
            state["v_row"] = vr.to(_store_dtype(hparams.state_dtype))
            state["v_col"] = vc.to(_store_dtype(hparams.state_dtype))
            vr_hat = vr / bc2
            vc_hat = vc / bc2
            # Reconstruct per row-block to bound transient memory (block=256).
            v_hat = torch.empty_like(ag)
            for o in range(0, ag.shape[0], 256):
                rb = vr_hat[o : o + 256]
                denom = rb.mean().clamp_min(1e-12)
                v_hat[o : o + 256] = (rb * vc_hat) / denom
        state["m"] = m.to(_store_dtype(hparams.state_dtype))
        m_hat = m / bc1
        u = m_hat / (v_hat + hparams.eps)
        if hparams.state_dtype != "fp32":
            u = u.clamp(-hparams.update_clip, hparams.update_clip)
        decay = lr * hparams.wd
        return (w32 * (1.0 - decay) - lr * u).detach()

    total_rows = state["m"].shape[0] if "m" in state else w.shape[0]
    r0 = 0 if row_start is None else int(row_start)
    r1 = total_rows if row_end is None else int(row_end)
    if r0 < 0 or r1 > total_rows or r0 >= r1:
        raise ValueError(f"invalid row range [{r0}, {r1}) for total rows {total_rows}")

    w_slice = w if w.shape[0] == (r1 - r0) else w[r0:r1]
    g_slice = g if g.shape[0] == (r1 - r0) else g[r0:r1]
    w32 = w_slice.detach().float()
    g32 = g_slice.detach().float()

    state["step"] = int(state.get("step", 0)) + 1
    t = state["step"]
    bm, bv = hparams.beta_m, hparams.beta_v
    bc1 = 1.0 - bm**t
    bc2 = 1.0 - bv**t

    m_slice = state["m"][r0:r1].float() * bm + g32 * (1.0 - bm)
    state["m"][r0:r1] = m_slice.to(_store_dtype(hparams.state_dtype))
    m_hat = m_slice / bc1
    ag = g32.abs()

    if "v" in state:
        v_slice = state["v"][r0:r1].float() * bv + ag * (1.0 - bv)
        state["v"][r0:r1] = v_slice.to(_store_dtype(hparams.state_dtype))
        v_hat = v_slice / bc2
    else:
        vr_slice = state["v_row"][r0:r1].float() * bv + ag.mean(dim=1, keepdim=True) * (1.0 - bv)
        state["v_row"][r0:r1] = vr_slice.to(_store_dtype(hparams.state_dtype))
        vc = state["v_col"].float() * bv + ag.mean(dim=0, keepdim=True) * (1.0 - bv)
        state["v_col"] = vc.to(_store_dtype(hparams.state_dtype))
        vr_hat = vr_slice / bc2
        vc_hat = vc / bc2
        v_hat = torch.empty_like(ag)
        for o in range(0, ag.shape[0], 256):
            rb = vr_hat[o : o + 256]
            denom = rb.mean().clamp_min(1e-12)
            v_hat[o : o + 256] = (rb * vc_hat) / denom

    u = m_hat / (v_hat + hparams.eps)
    if hparams.state_dtype != "fp32":
        u = u.clamp(-hparams.update_clip, hparams.update_clip)
    decay = lr * hparams.wd
    return (w32 * (1.0 - decay) - lr * u).detach()


def smaul_update_range(
    w: torch.Tensor,
    g: torch.Tensor,
    state: dict,
    hparams: SmaulOptHParams,
    lr: float,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Update only a row range [row_start:row_end] directly in optimizer state.

    Returns the updated row block (fp32).
    """
    return smaul_update(w, g, state, hparams, lr, row_start=row_start, row_end=row_end)



def global_grad_norm(grads: list[torch.Tensor]) -> float:
    """Global L2 grad norm accumulated in float64 (SmaulNative parity)."""
    total = 0.0
    for g in grads:
        if g is None:
            continue
        total += float(g.detach().double().pow(2).sum().item())
    return math.sqrt(total)


class SmaulOpt:
    """SmaulOpt with trunk/expert/router LR groups and expert-local states."""

    def __init__(self, hparams: SmaulOptHParams | None = None) -> None:
        self.hp = hparams or SmaulOptHParams()
        if self.hp.state_dtype not in ("bf16", "fp32"):
            raise ValueError(f"state_dtype must be bf16|fp32, got {self.hp.state_dtype!r}")
        self.trunk_state: dict[str, dict] = {}
        self.router_state: dict[str, dict] = {}
        self.step_count = 0

    # -- dense (trunk / router) --
    def step_dense(
        self,
        named_params: list[tuple[str, torch.Tensor]],
        lr: float,
        store: dict[str, dict],
    ) -> float:
        """Update dense params in place. Returns applied global scale (0 = skipped)."""
        grads = [p.grad for _, p in named_params if p.grad is not None]
        if not grads:
            return 0.0
        norm = global_grad_norm(grads)
        if not math.isfinite(norm):
            return 0.0
        scale = 1.0
        if self.hp.clip > 0 and norm > self.hp.clip:
            scale = self.hp.clip / (norm + 1e-12)
        eff_lr = lr * scale
        # NOTE: no step_count increment here. step_dense runs once per
        # parameter group (trunk, router, ...), so counting here would
        # double-count train steps. The single global increment lives in
        # train_step, which owns the once-per-step semantics.
        with torch.no_grad():
            for name, p in named_params:
                if p.grad is None:
                    continue
                st = store.get(name)
                if st is None or st["m"].shape != tuple(p.shape):
                    st = init_state(tuple(p.shape), self.hp)
                    store[name] = st
                new_w = smaul_update(p.data, p.grad * scale, st, self.hp, eff_lr)
                p.data.copy_(new_w.to(p.data.dtype))
                p.grad = None
        return scale

    def step_trunk(self, named_params: list[tuple[str, torch.Tensor]], trunk_lr: float) -> float:
        return self.step_dense(named_params, trunk_lr, self.trunk_state)

    def step_router(self, named_params: list[tuple[str, torch.Tensor]], router_lr: float) -> float:
        return self.step_dense(named_params, router_lr, self.router_state)

    # -- expert-local (state lives on the record; weights written back to FP8) --
    def step_expert(
        self,
        record,  # ExpertRecord (duck-typed to avoid a circular import)
        compute: dict[str, torch.Tensor],  # dequantized leaf weights w/ .grad
        grad_scale: float,
        lr: float,
        row_range: tuple[int, int] | None = None,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> float:
        """Apply grads to one expert and requantize its FP8 storage in place.

        Only this expert's blocks are decoded/updated/requantized — never the
        full model (avoids the old SmaulNative RQT full-matrix pathology).
        When row_range or row_start/row_end is specified, only that row block
        is updated, preserving untouched FP8 blocks bit-for-bit.
        Returns mean |update| as a gradient-activity signal (FP32 math).
        """

        names = [n for n in ("w_gate", "w_up", "w_down") if n in compute and compute[n].grad is not None]
        if not names:
            return 0.0
        norm = global_grad_norm([compute[n].grad for n in names])
        if not math.isfinite(norm):
            for n in names:
                compute[n].grad = None
            return 0.0
        scale = 1.0
        if self.hp.clip > 0 and norm > self.hp.clip:
            scale = self.hp.clip / (norm + 1e-12)
        activity = 0.0
        tile = record.weights_fp8["w_gate"].tile
        if row_range is not None:
            row_start, row_end = row_range
        with torch.no_grad():
            for n in names:
                grad_tensor = compute[n].grad
                shape = tuple(compute[n].shape)
                st = record.optim_state.get(n)
                if st is None or tuple(st["m"].shape) != shape:
                    st = init_state(shape, self.hp)
                    record.optim_state[n] = st

                if row_start is not None or row_end is not None:
                    total_rows = record.weights_fp8[n].shape[0]
                    r0 = 0 if row_start is None else row_start
                    r1 = total_rows if row_end is None else row_end
                    base_slice = dequantize_fp8_row_block(record.weights_fp8[n], r0, r1, dtype=torch.float32)
                    g_slice = grad_tensor if grad_tensor.shape[0] == (r1 - r0) else grad_tensor[r0:r1]
                    g = (g_slice * scale * grad_scale).float()
                    new_slice = smaul_update(base_slice, g, st, self.hp, lr * scale, row_start=r0, row_end=r1)
                    update_fp8_row_block(record.weights_fp8[n], new_slice, r0, r1)
                    activity += float((new_slice - base_slice).abs().mean().item())
                else:
                    g = (grad_tensor * scale * grad_scale).float()
                    base = dequantize_fp8_blockwise(record.weights_fp8[n], dtype=torch.float32)
                    new_w = smaul_update(base, g, st, self.hp, lr * scale)
                    record.weights_fp8[n] = quantize_fp8_blockwise(new_w, tile=tile)
                    activity += float((new_w - base).abs().mean().item())
                compute[n].grad = None
        record.grad_activity = 0.9 * record.grad_activity + 0.1 * (activity / max(1, len(names)))
        return record.grad_activity

