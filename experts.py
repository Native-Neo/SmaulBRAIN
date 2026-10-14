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

from native import (call_ternary_linear_forward, call_ternary_linear_input_grad,
                    call_fp8_to_packed_ternary, call_float_to_packed_ternary)
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


def _cached_ternary_values(w: torch.Tensor, threshold_ratio: float = 0.7) -> torch.Tensor:
    """Quantize once per immutable training leaf, reusing it across recurrent depth.

    A training leaf is not mutated until its complete backward has finished,
    and a fresh leaf is created for the next optimizer step. The cache is
    therefore attached to that leaf rather than held in a process-global map.
    Tensor versioning guards against accidental in-place changes reusing stale
    quantized values. This deliberately trades at most one compute-dtype copy
    per active expert weight for avoiding repeated full-matrix thresholding.
    """
    version = w._version
    cached = getattr(w, "_smaul_ternary_qat_cache", None)
    if cached is not None and cached[0] == version:
        return cached[1]
    with torch.no_grad():
        quantized = _ternary_values(w, threshold_ratio)
    w._smaul_ternary_qat_cache = (version, quantized)
    return quantized


def ternary_qat_weight(w: torch.Tensor, threshold_ratio: float = 0.7) -> torch.Tensor:
    """Per-output-row ternary QAT with a straight-through estimator.

    This helper is useful for inference and reference tests. Training uses
    _TernaryQATLinear to keep latent weights and activations in its explicit
    autograd save set; cached quantized values serve its ordinary matmul gradients.
    """
    quantized = _ternary_values(w, threshold_ratio)
    return quantized.detach() + (w - w.detach())

def _cached_packed_ternary_values(
    w: torch.Tensor, threshold_ratio: float = 0.7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 2-bit packed trits and rounded per-row scales, cached on a stable leaf.

    Four values share one byte: 00=zero, 01=positive, 10=negative.
    The packed stream is flattened across row boundaries, which keeps the
    format correct for any row width, including dimensions not divisible by 4.
    Quantization first follows the reference BF16/FP32 rounding exactly, then
    derives codes/scales from those representable ternary values.
    """
    version = w._version
    cached = getattr(w, "_smaul_packed_ternary_qat_cache", None)
    if cached is not None and cached[0] == version:
        return cached[1], cached[2]

    with torch.no_grad():
        quantized = _ternary_values(w, threshold_ratio)
        codes = torch.where(
            quantized > 0,
            torch.ones_like(quantized, dtype=torch.int16),
            torch.where(
                quantized < 0,
                torch.full_like(quantized, 2, dtype=torch.int16),
                torch.zeros_like(quantized, dtype=torch.int16),
            ),
        ).reshape(-1)
        padding = (-codes.numel()) % 4
        if padding:
            codes = F.pad(codes, (0, padding), value=0)
        groups = codes.reshape(-1, 4)
        packed = (
            groups[:, 0]
            | (groups[:, 1] << 2)
            | (groups[:, 2] << 4)
            | (groups[:, 3] << 6)
        ).to(torch.uint8).contiguous()
        # Q values are already rounded to the compute dtype by _ternary_values.
        # Recovering the magnitude here matches the reference for BF16 scales.
        scales = quantized.abs().amax(dim=-1).to(torch.float32).contiguous()

    w._smaul_packed_ternary_qat_cache = (version, packed, scales)
    return packed, scales


class _TernaryQATLinear(torch.autograd.Function):
    """Ternary-weight forward with ordinary PyTorch-style gradients.

    The CPU forward can consume packed trits directly. Backward deliberately
    uses regular tensor matrix multiplication only: no custom ternary dX
    kernel, zero-skipping gradient path, or extra backward dispatcher.
    The straight-through estimator is retained for the latent weight gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, require_packed: bool):
        ctx.save_for_backward(x, weight)
        ctx.require_packed = bool(require_packed)
        if (x.device.type == "cpu" and weight.device.type == "cpu"
                and x.dtype in (torch.float32, torch.bfloat16)
                and weight.dtype in (torch.float32, torch.bfloat16)
                and weight.is_contiguous()):
            # Convert each original weight value directly into its 2-bit code.
            # This avoids building the dense scaled-ternary matrix in Python.
            converted = call_float_to_packed_ternary(weight)
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            if converted is not None:
                packed, scales = converted
                native_output = call_ternary_linear_forward(x_2d, packed, scales)
                if native_output is not None:
                    return native_output.reshape(*x.shape[:-1], weight.shape[0])
        if ctx.require_packed:
            raise RuntimeError(
                "full_ternary_qat requires native packed ternary linear kernels; "
                "refusing to materialize a dense floating-point ternary weight matrix"
            )
        # Compatibility path for expert-only ternary-QAT when direct packing
        # is unavailable. Strict full-model mode never enters this fallback.
        quantized = _cached_ternary_values(weight)
        return F.linear(x, quantized)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
            # dX = dY @ Wq. Fold each ternary output-row scale into dY once,
            # then use packed signs for an add/sub-only inner loop.
            # The native path is CPU-only; unsupported cases keep dense fallback.
            grad_input_native = None
            if x.device.type == "cpu" and weight.device.type == "cpu":
                packed, scales = _cached_packed_ternary_values(weight)
                scaled_grad = (grad_out_2d.float() * scales.reshape(1, -1)).contiguous()
                grad_input_native = call_ternary_linear_input_grad(
                    scaled_grad, packed, int(weight.shape[1])
                )
            if grad_input_native is not None:
                grad_input = grad_input_native.reshape_as(x).to(dtype=x.dtype)
            else:
                quantized = _cached_ternary_values(weight)
                grad_input = (grad_out_2d.float() @ quantized.float()).reshape_as(x).to(dtype=x.dtype)
        if ctx.needs_input_grad[1]:
            grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            grad_weight = grad_out_2d.transpose(0, 1) @ x_2d
        return grad_input, grad_weight, None


@dataclass
class PackedTernaryMatrix:
    """Packed signed trits and one FP32 scale per output row; never a dense weight."""
    packed_codes: torch.Tensor
    row_scales: torch.Tensor
    shape: tuple[int, int]


@dataclass
class TernaryDirectExpertWeights:
    """Expert view whose effective weights remain in packed 2-bit form."""
    expert_id: str
    matrices: dict[str, PackedTernaryMatrix]
    compute_dtype: torch.dtype
    grad_sink: Callable[[str, str, torch.Tensor], None] | None = None
    tokens: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grad_sink is not None:
            # A scalar autograd edge ensures backward runs even when the trunk
            # input is frozen. Actual matrix gradients are sent to grad_sink.
            self.tokens = {
                name: torch.zeros((), dtype=torch.float32, requires_grad=True)
                for name in ("w_gate", "w_up", "w_down")
            }

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return self.compute_dtype


class _PackedTernaryLinear(torch.autograd.Function):
    """Linear operation over packed trits; it never builds a dense weight matrix."""

    @staticmethod
    def forward(ctx, x, token, packed_codes, row_scales,
                grad_sink, expert_id, weight_name, in_features, out_features):
        if x.device.type != "cpu" or x.dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError(
                "full ternary no-dequant expert kernels currently require CPU FP32/BF16 activations"
            )
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        output = call_ternary_linear_forward(x2, packed_codes, row_scales)
        if output is None:
            raise RuntimeError(
                "full ternary no-dequant path requires the native packed kernel; "
                "refusing to materialize dense FP32/BF16 expert weights as a fallback"
            )
        ctx.save_for_backward(x, token)
        ctx.packed_codes = packed_codes
        ctx.row_scales = row_scales
        ctx.grad_sink = grad_sink
        ctx.expert_id = expert_id
        ctx.weight_name = weight_name
        ctx.in_features = int(in_features)
        ctx.out_features = int(out_features)
        return output.reshape(*x.shape[:-1], int(out_features))

    @staticmethod
    def backward(ctx, grad_out):
        x, token = ctx.saved_tensors
        grad_input = None
        grad_out2 = grad_out.reshape(-1, ctx.out_features).contiguous()
        x2 = x.reshape(-1, ctx.in_features).contiguous()
        if ctx.needs_input_grad[0]:
            # Fold the per-row magnitude into dY; the native inner loop is
            # then only signed additions/subtractions and skips zero trits.
            grad_scaled = (
                grad_out2.float() * ctx.row_scales.reshape(1, -1)
            ).contiguous()
            grad_input2 = call_ternary_linear_input_grad(
                grad_scaled, ctx.packed_codes, ctx.in_features
            )
            if grad_input2 is None:
                raise RuntimeError(
                    "full ternary no-dequant backward requires the native packed input-gradient kernel; "
                    "refusing a dense-weight fallback"
                )
            grad_input = grad_input2.reshape_as(x).to(dtype=x.dtype)
        if ctx.grad_sink is not None:
            # STE weight gradient; only the gradient matrix is dense. The
            # effective weight itself remains packed and is never expanded.
            grad_weight = grad_out2.float().transpose(0, 1) @ x2.float()
            ctx.grad_sink(ctx.expert_id, ctx.weight_name, grad_weight)
        # token supplies a graph edge for weight-only expert training while
        # the exact matrix gradient travels through grad_sink.
        return grad_input, torch.zeros_like(token), None, None, None, None, None, None, None


def _direct_ternary_linear(
    x: torch.Tensor, weights: TernaryDirectExpertWeights, name: str
) -> torch.Tensor:
    matrix = weights.matrices[name]
    token = weights.tokens[name] if weights.grad_sink is not None else torch.zeros(())
    return _PackedTernaryLinear.apply(
        x, token, matrix.packed_codes, matrix.row_scales,
        weights.grad_sink, weights.expert_id, name,
        matrix.shape[1], matrix.shape[0],
    )


def _swiglu_direct_ternary(
    x: torch.Tensor, weights: TernaryDirectExpertWeights
) -> torch.Tensor:
    gate = _direct_ternary_linear(x, weights, "w_gate")
    up = _direct_ternary_linear(x, weights, "w_up")
    return _direct_ternary_linear(F.silu(gate) * up, weights, "w_down")


def _qat_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _TernaryQATLinear.apply(x, weight, False)


def _swiglu_tensors(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
    w_down: torch.Tensor, ternary_qat: bool = False,
) -> torch.Tensor:
    """Tensor-only SwiGLU core, suitable for activation checkpointing."""
    linear = _qat_linear if ternary_qat else F.linear
    gate = linear(x, w_gate)
    up = linear(x, w_up)
    return linear(F.silu(gate) * up, w_down)


def swiglu_forward(
    x: torch.Tensor,
    w: dict[str, torch.Tensor] | TernaryDirectExpertWeights,
    ternary_qat: bool = False,
) -> torch.Tensor:
    """SwiGLU over either ordinary tensors or directly packed ternary weights."""
    if isinstance(w, TernaryDirectExpertWeights):
        if torch.is_grad_enabled() and (x.requires_grad or w.grad_sink is not None):
            return checkpoint(_swiglu_direct_ternary, x, w, use_reentrant=False)
        return _swiglu_direct_ternary(x, w)
    args = (x, w["w_gate"], w["w_up"], w["w_down"], ternary_qat)
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


    def direct_ternary_weights(
        self,
        compute_dtype: torch.dtype,
        grad_sink: Callable[[str, str, torch.Tensor], None] | None = None,
        threshold_ratio: float = 0.7,
    ) -> TernaryDirectExpertWeights:
        """Prepare/reuse packed ternary codes directly from FP8 bytes.

        The conversion kernel decodes FP8 values scalarly inside native code
        and writes packed trits. It never materializes a decoded weight matrix.
        """
        round_bf16 = compute_dtype == torch.bfloat16
        cache = getattr(self, "_smaul_direct_ternary_cache", None)
        key = (self.version, round_bf16, float(threshold_ratio))
        if cache is None or cache[0] != key:
            matrices: dict[str, PackedTernaryMatrix] = {}
            for name in ("w_gate", "w_up", "w_down"):
                source = self.weights_fp8[name]
                result = call_fp8_to_packed_ternary(
                    source, threshold_ratio=threshold_ratio,
                    round_to_bf16=round_bf16,
                )
                if result is None:
                    raise RuntimeError(
                        "full ternary no-dequant mode requires the native FP8-to-trit "
                        "kernel; it will not decode weights into a dense FP32/BF16 tensor"
                    )
                packed, row_scales = result
                matrices[name] = PackedTernaryMatrix(
                    packed_codes=packed, row_scales=row_scales,
                    shape=(int(source.shape[0]), int(source.shape[1])),
                )
            self._smaul_direct_ternary_cache = (key, matrices)
        else:
            matrices = cache[1]
        return TernaryDirectExpertWeights(
            expert_id=self.expert_id, matrices=matrices,
            compute_dtype=compute_dtype, grad_sink=grad_sink,
        )

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
            if isinstance(w, TernaryDirectExpertWeights):
                if x.device.type != "cpu":
                    raise RuntimeError(
                        "full ternary no-dequant expert path currently supports CPU execution only"
                    )
                dtype = w.compute_dtype
            else:
                # The normal FP8 paging path caches compute tensors by device.
                if w["w_gate"].device != x.device:
                    w = {k: v.to(x.device) for k, v in w.items()}
                dtype = w["w_gate"].dtype
            rows = mask_slot.any(dim=-1)  # tokens touching this expert
            xs = x[rows].to(dtype)  # [M, D]
            y = swiglu_forward(xs, w, ternary_qat=ternary_qat).to(out.dtype)  # [M, D]
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
