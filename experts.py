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

from precision import FP8BlockTensor, dequantize_fp8_blockwise, quantize_fp8_blockwise


def expert_param_count(d_model: int, expert_hidden: int) -> int:
    return 3 * d_model * expert_hidden


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


def swiglu_forward(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """SwiGLU: down(silu(gate(x)) * up(x)). x: [N, D] -> [N, D]."""
    return F.linear(F.silu(F.linear(x, w["w_gate"])) * F.linear(x, w["w_up"]), w["w_down"])


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
            y = swiglu_forward(xs, w).to(out.dtype)  # [M, D]
            weight = mask_slot[rows].float() * top_weights[rows]  # [M, K]
            out[rows] += weight.sum(dim=-1, keepdim=True).to(out.dtype) * y
            rec = self.experts[eid]
            rec.tokens_routed += int(rows.sum().item())
            rec.last_used_step = step
            mean_weight = float(weight.sum().item()) / max(1, int(rows.sum().item()))
            rec.contribution = 0.9 * rec.contribution + 0.1 * mean_weight
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
