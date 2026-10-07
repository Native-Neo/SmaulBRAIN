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


_STORAGE_DTYPES = (torch.bfloat16, torch.float32)


def expected_layout(shape: tuple[int, ...], hparams: SmaulOptHParams) -> str:
    """Canonical second-moment layout for ``shape`` under ``hparams``.

    Returns ``"factored"`` iff ``factor_v`` is on and the shape is a
    factorizable 2-D matrix (>= 2 x 2); otherwise ``"full"``. This is the
    single source of truth for layout decisions (init, validation,
    migration); update math itself branches on stored keys so direct
    ``smaul_update`` calls stay bit-identical.
    """
    if hparams.factor_v and _factored(tuple(shape)):
        return "factored"
    return "full"


def _moment_ok(t, shape: tuple[int, ...]) -> bool:
    return (
        torch.is_tensor(t)
        and tuple(t.shape) == tuple(shape)
        and t.dtype in _STORAGE_DTYPES
    )


def state_is_valid(state: dict, shape: tuple[int, ...], hparams: SmaulOptHParams) -> bool:
    """True iff ``state`` exactly matches the canonical layout for ``shape``.

    Checks: ``m`` shape/dtype, integer ``step >= 0``, and the expected
    second-moment buffers with exact shapes — and no stale buffers from the
    other layout (a state carrying both ``v`` and ``v_row``/``v_col`` is
    invalid and must go through :func:`ensure_state` migration/cleanup,
    never silent mixed use). Either bf16 or fp32 storage counts as valid;
    dtype normalization to ``hparams.state_dtype`` is a cast (migration),
    not a reset.
    """
    if not isinstance(state, dict):
        return False
    shape = tuple(shape)
    if not _moment_ok(state.get("m"), shape):
        return False
    step = state.get("step", None)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return False
    if expected_layout(shape, hparams) == "factored":
        r, c = shape
        if not _moment_ok(state.get("v_row"), (r, 1)):
            return False
        if not _moment_ok(state.get("v_col"), (1, c)):
            return False
        if "v" in state:
            return False
    else:
        if not _moment_ok(state.get("v"), shape):
            return False
        if "v_row" in state or "v_col" in state:
            return False
    return True


def _cast_buffers(state: dict, hparams: SmaulOptHParams) -> bool:
    """Cast stored moment buffers to ``hparams.state_dtype`` in place.

    Preserves values (float round-trip) and device; returns True if any
    buffer was recast. Never touches ``step``.
    """
    sd = _store_dtype(hparams.state_dtype)
    changed = False
    for key in ("m", "v", "v_row", "v_col"):
        t = state.get(key)
        if torch.is_tensor(t) and t.dtype != sd:
            state[key] = t.float().to(sd)
            changed = True
    return changed


def migrate_state_layout(state: dict, shape: tuple[int, ...], hparams: SmaulOptHParams) -> dict:
    """Convert a shape/step-valid state to the canonical layout in place.

    Assumes ``m`` has the right shape and ``step`` is a valid int (checked
    by the caller); only the second moment is converted:

      * full ``v`` -> factored: ``v_row = mean_c(v)``, ``v_col = mean_r(v)``
        (row/col marginal means of the stored |g| EMA, matching how the
        factored buffers are accumulated in :func:`smaul_update`).
      * factored ``v_row``/``v_col`` -> full: ``v = outer(R, C) / mean(R)``
        (the exact reconstruction formula used in :func:`smaul_update`,
        with the same 1e-12 denominator floor).

    The replaced buffers are deleted so no stale moment survives alongside
    the migrated one. Values are computed in FP32 then stored in
    ``hparams.state_dtype``. Returns the same (mutated) dict.
    """
    sd = _store_dtype(hparams.state_dtype)
    shape = tuple(shape)
    if expected_layout(shape, hparams) == "factored":
        v = state.get("v")
        if torch.is_tensor(v) and tuple(v.shape) == shape:
            vf = v.detach().float()
            vr = vf.mean(dim=1, keepdim=True).to(sd)
            vc = vf.mean(dim=0, keepdim=True).to(sd)
            state["v_row"] = vr
            state["v_col"] = vc
        # Clean the replaced full moment (explicit; never keep both).
        state.pop("v", None)
    else:
        vr = state.get("v_row")
        vc = state.get("v_col")
        if (
            torch.is_tensor(vr)
            and torch.is_tensor(vc)
            and tuple(vr.shape) == (shape[0], 1)
            and tuple(vc.shape) == (1, shape[-1] if len(shape) == 2 else 1)
        ):
            rf = vr.detach().float()
            cf = vc.detach().float()
            denom = rf.mean().clamp_min(1e-12)
            state["v"] = ((rf * cf) / denom).to(sd)
        # Clean the replaced factored moments (explicit; never keep both).
        state.pop("v_row", None)
        state.pop("v_col", None)
    _cast_buffers(state, hparams)
    return state


def ensure_state(state: dict | None, shape: tuple[int, ...], hparams: SmaulOptHParams) -> tuple[dict, str]:
    """Validate/migrate ``state`` for ``shape``; fresh zero state if unusable.

    Returns ``(state, action)`` with action in ``{"kept", "migrated",
    "init"}``:

      * ``"kept"`` — canonical layout already; stale keys absent (or
        removed) and buffers recast to ``hparams.state_dtype`` as needed.
        ``step`` is preserved.
      * ``"migrated"`` — ``m``/``step`` were shape-valid but the second
        moment used the other layout; converted via
        :func:`migrate_state_layout` with ``step`` preserved and replaced
        buffers deleted.
      * ``"init"`` — ``m`` shape/dtype or ``step`` was unusable (or moment
        shapes corrupt beyond layout mismatch); replaced with a fresh
        :func:`init_state` (zero moments, ``step=0``). The caller's old dict
        is discarded, never merged, so no stale buffer survives.

    Never changes update math: migration only reshapes stored moments; the
    per-step ``smaul_update`` equations are untouched.
    """
    shape = tuple(shape)
    if (
        isinstance(state, dict)
        and torch.is_tensor(state.get("m"))
        and tuple(state["m"].shape) == shape
        and state["m"].dtype in _STORAGE_DTYPES
        and isinstance(state.get("step"), int)
        and not isinstance(state.get("step"), bool)
        and state["step"] >= 0
    ):
        if state_is_valid(state, shape, hparams):
            _cast_buffers(state, hparams)
            return state, "kept"
        # Shape/step-valid but layout-mismatched or moment-corrupt: try an
        # explicit layout migration when the source moments have usable
        # shapes; otherwise fall through to fresh init.
        layout = expected_layout(shape, hparams)
        if layout == "factored":
            v = state.get("v")
            vr, vc = state.get("v_row"), state.get("v_col")
            has_full = torch.is_tensor(v) and tuple(v.shape) == shape and v.dtype in _STORAGE_DTYPES
            has_fact = (
                torch.is_tensor(vr) and tuple(vr.shape) == (shape[0], 1) and vr.dtype in _STORAGE_DTYPES
                and torch.is_tensor(vc) and tuple(vc.shape) == (1, shape[1]) and vc.dtype in _STORAGE_DTYPES
            )
            if has_full and not has_fact:
                migrate_state_layout(state, shape, hparams)
                return state, "migrated"
            if has_fact and "v" in state:
                # Factored moments already correct; just clean the stale full
                # buffer left by a layout flip.
                state.pop("v", None)
                _cast_buffers(state, hparams)
                return state, "migrated"
        else:
            v = state.get("v")
            has_full = torch.is_tensor(v) and tuple(v.shape) == shape and v.dtype in _STORAGE_DTYPES
            vr, vc = state.get("v_row"), state.get("v_col")
            has_fact = torch.is_tensor(vr) or torch.is_tensor(vc)
            if has_fact and has_full and state_is_valid(
                {"m": state["m"], "step": state["step"], "v": v}, shape, hparams
            ):
                # Full moment already correct; drop stale factored buffers.
                state.pop("v_row", None)
                state.pop("v_col", None)
                _cast_buffers(state, hparams)
                return state, "migrated"
            if has_fact and not has_full:
                migrate_state_layout(state, shape, hparams)
                # Migration produced a full v only when source shapes were
                # usable; otherwise the state is still invalid -> fresh init.
                if state_is_valid(state, shape, hparams):
                    return state, "migrated"
        return init_state(shape, hparams), "init"
    return init_state(shape, hparams), "init"


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
        """Update dense params in place. Returns applied global scale (0 = skipped).

        Step semantics (defined): per-tensor ``step`` ticks only when that
        tensor's update is actually applied. A skipped step (no grads, or
        non-finite global norm) mutates no optimizer state and preserves
        grads, so a retry with the same grads is bit-equivalent to the
        first attempt. ``step_count`` (global train-step counter) is NOT
        touched here — it advances exactly once per ``train_step`` call
        (including skipped steps); see ``train.train_step``.

        Tied parameters (same tensor under several names) update exactly
        once: duplicates are deduped by tensor identity, the first name's
        state wins, and all alias names are pointed at the same state dict
        (divergent duplicate states are discarded) so tied weights stay
        bit-identical.
        """
        grads = []
        _seen_grad_ids: set[int] = set()
        for _, p in named_params:
            if p.grad is not None and id(p) not in _seen_grad_ids:
                # Dedupe tied tensors: the same gradient must contribute to
                # the global norm exactly once, or tied params would be
                # clipped differently from an untied single update.
                _seen_grad_ids.add(id(p))
                grads.append(p.grad)
        if not grads:
            return 0.0
        norm = global_grad_norm(grads)
        if not math.isfinite(norm):
            # Skipped step: leave state AND grads untouched for retry
            # equivalence (mirrors step_expert).
            return 0.0
        scale = 1.0
        if self.hp.clip > 0 and norm > self.hp.clip:
            scale = self.hp.clip / (norm + 1e-12)
        eff_lr = lr * scale
        # NOTE: no step_count increment here. step_dense runs once per
        # parameter group (trunk, router, ...), so counting here would
        # double-count train steps. The single global increment lives in
        # train_step, which owns the once-per-step semantics.
        # Dedupe tied parameters by identity (preserve tied identity: one
        # update, in-place copy keeps storage shared).
        by_id: dict[int, tuple[str, torch.Tensor]] = {}
        id_order: list[int] = []
        for name, p in named_params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in by_id:
                by_id[pid] = (name, p)
                id_order.append(pid)
        with torch.no_grad():
            for pid in id_order:
                name, p = by_id[pid]
                shape = tuple(p.shape)
                st = store.get(name)
                if st is None:
                    # Tied alias may hold the live state under another name.
                    for n2, p2 in named_params:
                        if id(p2) == pid and n2 in store and store[n2] is not None:
                            st = store[n2]
                            break
                st, action = ensure_state(st, shape, self.hp)
                if action == "init" or store.get(name) is not st:
                    store[name] = st
                # Point every alias name at the winning state object and
                # drop divergent duplicates (clean replaced state).
                for n2, p2 in named_params:
                    if id(p2) == pid and n2 != name and store.get(n2) is not st:
                        store[n2] = st
                new_w = smaul_update(p.data, p.grad * scale, st, self.hp, eff_lr)
                p.data.copy_(new_w.to(p.data.dtype))
                p.grad = None
            # Clear grads on tied duplicates too (they share storage with the
            # updated tensor; leaving a stale .grad would double-apply).
            for _, p in named_params:
                if id(p) in by_id and p.grad is not None:
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
        Only weights with actual (non-None) grads are touched: no state is
        allocated and no FP8 block is rewritten for grad-free weights.
        A skipped step (no grads, or non-finite global norm) mutates no
        optimizer state, bumps no version, touches no FP8 bytes, and
        preserves grads, so a retry with the same grads is bit-equivalent.
        Returns mean |update| as a gradient-activity signal (FP32 math).
        """

        names = [n for n in ("w_gate", "w_up", "w_down") if n in compute and compute[n].grad is not None]
        if not names:
            return 0.0
        norm = global_grad_norm([compute[n].grad for n in names])
        if not math.isfinite(norm):
            # Skipped step: leave state, version, FP8 bytes AND grads
            # untouched for retry equivalence (mirrors step_dense).
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
                st, action = ensure_state(st, shape, self.hp)
                if action == "init" or record.optim_state.get(n) is not st:
                    # Fresh or migrated state replaces the old entry; the
                    # old dict is discarded so no stale moment survives.
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
        record.version = getattr(record, "version", 0) + 1
        record.grad_activity = 0.9 * record.grad_activity + 0.1 * (activity / max(1, len(names)))
        return record.grad_activity


