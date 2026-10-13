"""Dynamic expert pool: FP8-stored SwiGLU experts with stable unique IDs.

Each expert targets ~5.12M parameters (explicit ``expert_hidden`` knob in the
config, not an accident of tensor dims). Experts are stored as FP8 block
tensors plus FP32 scales; compute weights are materialized per activation
through a caller-supplied provider (the paging layer), so this module never
forces the whole pool resident.

Every expert owns: weights, optimizer state (BF16 storage, FP32 math in the
optimizer), metadata (birth step, parent/source ids, usage/gradient/
contribution statistics). Identity (``expert_<NNNNN>``) is independent of
router index and cache slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from native import call_ternary_backward_input
from torch.utils.checkpoint import checkpoint

from precision import FP8BlockTensor, dequantize_fp8_blockwise, quantize_fp8_blockwise
from config import expert_param_count  # single source (config.py owns the formula)


def init_expert_weights(
    d_model: int, expert_hidden: int, generator: torch.Generator | None = None
) -> dict[str, torch.Tensor]:
    """Fresh SwiGLU weights (fp32): gate/up [H, D], down [D, H]."""
    def rand(*shape: int) -> torch.Tensor:
        t = torch.empty(*shape, dtype=torch.float32)
        if generator is None:
            torch.nn.init.normal_(t, std=0.02)
        else:
            t.normal_(0.0, 0.02, generator=generator)
        return t

    return {"w_gate": rand(expert_hidden, d_model),
            "w_up": rand(expert_hidden, d_model),
            "w_down": rand(d_model, expert_hidden)}


def _ternary_values(w: torch.Tensor, threshold_ratio: float = 0.7) -> torch.Tensor:
    """Build the scaled ternary values without an STE/autograd graph."""
    if w.ndim < 2:
        raise ValueError(f"ternary QAT expects a matrix or higher-rank tensor, got {tuple(w.shape)}")
    if not 0.0 <= threshold_ratio < 1.0:
        raise ValueError(f"threshold_ratio must be in [0, 1), got {threshold_ratio}")
    w32 = w.float()
    mean_abs = w32.abs().mean(dim=-1, keepdim=True)
    threshold = mean_abs * threshold_ratio
    trits = torch.where(w32 > threshold, 1.0,
                        torch.where(w32 < -threshold, -1.0, 0.0))
    nonzero = trits != 0
    count = nonzero.sum(dim=-1, keepdim=True).clamp_min(1)
    scale = (w32.abs() * nonzero).sum(dim=-1, keepdim=True) / count
    return (trits * scale).to(dtype=w.dtype)


def ternary_qat_weight(w: torch.Tensor, threshold_ratio: float = 0.7) -> torch.Tensor:
    """Per-output-row ternary QAT with a straight-through estimator.

    This helper is useful for inference and reference tests. Training uses
    _TernaryQATLinear so the ternary matrix is temporary and is recomputed
    during backward instead of being retained once per recurrent depth.
    """
    quantized = _ternary_values(w, threshold_ratio)
    return quantized.detach() + (w - w.detach())

class _TernaryLinear(torch.autograd.Function):
    """Linear autograd with an optional zero-skipping ternary dX kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, enabled: bool):
        ctx.enabled = bool(enabled)
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        grad_input = None
        if ctx.needs_input_grad[0]:
            if ctx.enabled:
                grad_input = call_ternary_backward_input(
                    grad_out_2d, weight.contiguous()
                )
            if grad_input is None:
                grad_input = grad_out_2d @ weight
            grad_input = grad_input.reshape_as(x)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = grad_out_2d.transpose(0, 1) @ x_2d
        return grad_input, grad_weight, None


class _TernaryQATLinear(torch.autograd.Function):
    """Ternary forward with STE weight gradients and recomputed ternary dX weights.

    Only the latent expert weight and input are saved for backward. The
    transformed ternary matrix exists only while each projection executes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, ternary_backward: bool):
        ctx.enabled = bool(ternary_backward)
        ctx.save_for_backward(x, weight)
        quantized = _ternary_values(weight)
        return F.linear(x, quantized)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # Recreate only this projection's ternary values. Do not attach
            # another quantization graph: the STE is the explicit grad_weight
            # formula below, while dX uses the true ternary forward operand.
            with torch.no_grad():
                quantized = _ternary_values(weight)
        if ctx.needs_input_grad[0]:
            if ctx.enabled:
                grad_input = call_ternary_backward_input(
                    grad_out_2d, quantized.contiguous()
                )
            if grad_input is None:
                grad_input = grad_out_2d @ quantized
            grad_input = grad_input.reshape_as(x)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_out_2d.transpose(0, 1) @ x_2d
        return grad_input, grad_weight, None


def _linear(x: torch.Tensor, weight: torch.Tensor, ternary_backward: bool):
    if ternary_backward:
        return _TernaryLinear.apply(x, weight, True)
    return F.linear(x, weight)


def _qat_linear(
    x: torch.Tensor, weight: torch.Tensor, ternary_backward: bool
) -> torch.Tensor:
    return _TernaryQATLinear.apply(x, weight, ternary_backward)


def _swiglu_tensors(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
    w_down: torch.Tensor, ternary_backward: bool = False,
    ternary_qat: bool = False,
) -> torch.Tensor:
    """Tensor-only SwiGLU core, suitable for activation checkpointing."""
    linear = _qat_linear if ternary_qat else _linear
    gate = linear(x, w_gate, ternary_backward)
    up = linear(x, w_up, ternary_backward)
    return linear(F.silu(gate) * up, w_down, ternary_backward)


def swiglu_forward(
    x: torch.Tensor, w: dict[str, torch.Tensor], ternary_backward: bool = False,
    ternary_qat: bool = False,
) -> torch.Tensor:
    """SwiGLU with checkpointed intermediates and memory-bounded ternary QAT.

    QAT linear layers save their latent weights and recompute ternary values
    for dX during backward, so quantized matrix copies are not retained
    across recurrent depths. The native dX kernel is CPU-FP32-only; other
    dtypes use the PyTorch matmul fallback.
    """
    args = (x, w["w_gate"], w["w_up"], w["w_down"],
            ternary_backward, ternary_qat)
    if torch.is_grad_enabled() and any(t.requires_grad for t in args[:4]):
        return checkpoint(_swiglu_tensors, *args, use_reentrant=False)
    return _swiglu_tensors(*args)


@dataclass
class ExpertRecord:
    """One expert: FP8 weights, optimizer state, metadata. ID is stable."""

    expert_id: str
    d_model: int
    expert_hidden: int
    weights_fp8: dict[str, FP8BlockTensor]
    birth_step: int = 0
    parents: list[str] = field(default_factory=list)
    source: str = "init"  # init | recombine | clone
    # Usage / contribution statistics (FP32, not parameters).
    tokens_routed: int = 0
    last_used_step: int = 0
    grad_activity: float = 0.0  # EMA of mean |grad|
    contribution: float = 0.0  # EMA of mean routing weight received
    # Expert-local optimizer state: {wname: {"m": bf16, "v_row": bf16, "v_col": bf16}}
    # (factored second moment, following the verified SmaulOpt layout).
    optim_state: dict = field(default_factory=dict)
    version: int = 0


    @property
    def param_count(self) -> int:
        return expert_param_count(self.d_model, self.expert_hidden)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        out = {}
        for name in ("w_gate", "w_up", "w_down"):
            out[name] = dequantize_fp8_blockwise(self.weights_fp8[name], dtype=dtype)
        return out

    def storage_bytes(self) -> int:
        return sum(t.nbytes() for t in self.weights_fp8.values())


def make_expert(
    expert_id: str,
    d_model: int,
    expert_hidden: int,
    weights: dict[str, torch.Tensor] | None = None,
    birth_step: int = 0,
    parents: list[str] | None = None,
    source: str = "init",
    fp8_tile: int = 64,
    generator: torch.Generator | None = None,
) -> ExpertRecord:
    w = weights if weights is not None else init_expert_weights(d_model, expert_hidden, generator)
    fp8 = {n: quantize_fp8_blockwise(w[n].float(), tile=fp8_tile) for n in ("w_gate", "w_up", "w_down")}
    rec = ExpertRecord(
        expert_id=expert_id,
        d_model=d_model,
        expert_hidden=expert_hidden,
        weights_fp8=fp8,
        birth_step=birth_step,
        parents=list(parents or []),
        source=source,
    )
    rec.optim_state = init_expert_optim_state(rec)
    return rec


def init_expert_optim_state(rec: ExpertRecord) -> dict:
    """Zero-initialized factored optimizer state (BF16 storage) per weight."""
    state: dict = {}
    for name in ("w_gate", "w_up", "w_down"):
        r, c = rec.weights_fp8[name].shape
        state[name] = {
            "m": torch.zeros(r, c, dtype=torch.bfloat16),
            "v_row": torch.zeros(r, 1, dtype=torch.bfloat16),
            "v_col": torch.zeros(1, c, dtype=torch.bfloat16),
            "step": 0,
        }
    return state


#: WeightProvider: expert_id -> compute-dtype SwiGLU weight dict (paging layer).
WeightProvider = Callable[[str], dict[str, torch.Tensor]]


class ExpertPool:
    """Ordered pool of experts. Router index <-> expert id mapping lives here."""

    def __init__(self) -> None:
        self.experts: dict[str, ExpertRecord] = {}
        self.order: list[str] = []  # router index -> expert_id
        self._next_id: int = 0

    # -- identity --
    def fresh_id(self) -> str:
        eid = f"expert_{self._next_id:05d}"
        self._next_id += 1
        return eid

    def add(self, rec: ExpertRecord) -> int:
        if rec.expert_id in self.experts:
            raise ValueError(f"duplicate {rec.expert_id} (refusing to overwrite)")
        self.experts[rec.expert_id] = rec
        self.order.append(rec.expert_id)
        try:
            n = int(rec.expert_id.split("_")[1])
            self._next_id = max(self._next_id, n + 1)
        except (IndexError, ValueError):
            pass
        return len(self.order) - 1

    def remove(self, expert_id: str) -> ExpertRecord:
        rec = self.experts.pop(expert_id)
        self.order.remove(expert_id)
        return rec

    def index_of(self, expert_id: str) -> int:
        return self.order.index(expert_id)

    def id_at(self, index: int) -> str:
        return self.order[index]

    def __len__(self) -> int:
        return len(self.order)

    # -- dispatch --
    def forward(
        self,
        x: torch.Tensor,
        top_ids: torch.Tensor,
        top_weights: torch.Tensor,
        dropped: torch.Tensor,
        provider: WeightProvider,
        step: int = 0,
        ternary_backward: bool = False,
        ternary_qat: bool = False,
    ) -> torch.Tensor:
        """Weighted expert combination. x: [N, D] -> [N, D].

        Dispatch is batched per expert (one SwiGLU matmul per active expert),
        never a Python loop over individual tokens. Only admitted slots
        (nonzero weight, live token) dispatch: capacity-refused slots carry
        weight 0 and contribute nothing, not even to routing statistics.
        """
        out = torch.zeros_like(x)
        live = ~dropped
        admitted = live.unsqueeze(-1) & (top_weights > 0)  # [N, K] actual dispatch
        if top_ids.numel() == 0:
            return out
        # Only experts actually addressed by this batch dispatch: O(active),
        # not O(pool). Router ids are pool indices (order[rid]).
        # Stats (tokens_routed/contribution) defer their host reads: per-expert
        # count/weight sums accumulate as tensors and flush in ONE host
        # transfer (single .tolist on a stacked [P, 2] fp64 pair) instead of
        # two .item() syncs per expert per depth. Values are bit-identical
        # to the old per-expert .item() path (int64 counts are exact in
        # fp64; fp32 sums cast exactly to fp64; same Python EMA formula).
        pending: list = []
        for rid in torch.unique(top_ids).tolist():
            if not 0 <= rid < len(self.order):
                raise ValueError(
                    f"router id {rid} out of range for pool of {len(self.order)} "
                    "(pool/router out of sync; refusing to mis-dispatch)"
                )
            mask_slot = (top_ids == rid) & admitted  # [N, K]
            if not mask_slot.any():
                continue
            eid = self.order[rid]
            w = provider(eid)
            # The pager caches on its own device (CPU RAM / simulated VRAM);
            # the trunk may live elsewhere (e.g. CUDA). Follow the
            # activations with a copy, never by mutating the cache entry.
            if w["w_gate"].device != x.device:
                w = {k: v.to(x.device) for k, v in w.items()}
            dtype = w["w_gate"].dtype
            rows = mask_slot.any(dim=-1)  # tokens touching this expert
            xs = x[rows].to(dtype)  # [M, D]
            y = swiglu_forward(
                xs, w, ternary_backward=ternary_backward,
                ternary_qat=ternary_qat,
            ).to(out.dtype)  # [M, D]
            weight = mask_slot[rows].float() * top_weights[rows]  # [M, K]
            out[rows] += weight.sum(dim=-1, keepdim=True).to(out.dtype) * y
            rec = self.experts[eid]
            pending.append((rec, rows.sum(), weight.sum().detach()))
        if pending:
            cnt_t = torch.stack([c for _, c, _ in pending])
            sum_t = torch.stack([s for _, _, s in pending])
            combined = torch.stack(
                [cnt_t.to(torch.float64), sum_t.to(torch.float64)], dim=1
            ).tolist()
            for (rec, _, _), (c_f, s_f) in zip(pending, combined):
                c, s = int(c_f), float(s_f)
                rec.tokens_routed += c
                rec.last_used_step = step
                rec.contribution = 0.9 * rec.contribution + 0.1 * (s / max(1, c))
        return out

    # -- stats --
    def usage_snapshot(self) -> dict[str, dict]:
        return {
            eid: {
                "tokens_routed": r.tokens_routed,
                "last_used_step": r.last_used_step,
                "grad_activity": r.grad_activity,
                "contribution": r.contribution,
                "birth_step": r.birth_step,
                "parents": list(r.parents),
                "source": r.source,
            }
            for eid, r in self.experts.items()
        }
