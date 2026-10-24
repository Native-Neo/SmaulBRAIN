"""Default coarse-to-fine Next-Concept Modeling runtime for SmaulBRAIN.

Train ResidualVQTokenizer first, freeze it, then train NCMGenerator against its
multiscale code hierarchies. The CLI entry points use this architecture by default.

This is the NumPy port of the original PyTorch module: parameters are
``smaulbrain.npgrad.Tensor`` objects and every layer has an explicit,
vectorized backward pass.  Public class names, method signatures and the
checkpoint key layout are unchanged except that tensors are NumPy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from smaulbrain import npgrad as ng
from smaulbrain import quant as _quant
from smaulbrain.npgrad import Parameter, Tensor


_TERNARY_GRADIENT_CHUNK_COLUMNS = 128


@dataclass(frozen=True)
class NCMConfig:
    vocab_size: int = 256
    d_model: int = 128
    codebook_size: int = 256
    levels: int = 4
    n_layers: int = 4
    n_heads: int = 4
    n_experts: int = 4
    top_k: int = 2
    max_seq_len: int = 1024
    dropout: float = 0.0
    commitment_weight: float = 0.25
    conditioning_corruption: float = 0.15
    level_loss_alpha: float = 1.0
    model_kind: str = "moe"
    ffn_hidden: int = 0
    expert_hidden: int = 0
    preset: str = "custom"
    forward_quant: str = "fp32"
    backward_mode: str = "fp32"

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.codebook_size < 2:
            raise ValueError("vocab_size and codebook_size must be >= 2")
        if min(self.d_model, self.levels, self.n_layers, self.n_heads) < 1:
            raise ValueError("model dimensions, levels, and heads must be positive")
        if self.model_kind not in {"dense", "moe"}:
            raise ValueError("model_kind must be 'dense' or 'moe'")
        if self.model_kind == "moe":
            if self.n_experts < 1 or not 1 <= self.top_k <= self.n_experts:
                raise ValueError("MoE top_k must be between 1 and n_experts")
            if self.expert_hidden < 0:
                raise ValueError("expert_hidden must be >= 0")
            if self.expert_hidden == 0:
                object.__setattr__(self, "expert_hidden", 4 * self.d_model)
        else:
            if self.n_experts != 0 or self.top_k != 0:
                raise ValueError("dense configurations must set n_experts=0 and top_k=0")
            if self.ffn_hidden < 0:
                raise ValueError("ffn_hidden must be >= 0")
            if self.ffn_hidden == 0:
                object.__setattr__(self, "ffn_hidden", 4 * self.d_model)
        if self.forward_quant not in {"fp32", "fp8", "ternary"}:
            raise ValueError("forward_quant must be fp32, fp8, or ternary")
        if self.backward_mode not in {"fp32", "fp8", "ternary"}:
            raise ValueError("backward_mode must be fp32, fp8, or ternary")
        supported = {
            ("fp32", "fp32"), ("fp8", "fp8"), ("ternary", "fp32"),
            ("ternary", "ternary"), ("ternary", "fp8"),
        }
        if (self.forward_quant, self.backward_mode) not in supported:
            raise ValueError(
                f"unsupported quantization combination: "
                f"{self.forward_quant}/{self.backward_mode}"
            )
        # n_heads remains accepted for old configs; recurrent NCM does not split heads.
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.conditioning_corruption <= 1:
            raise ValueError("conditioning_corruption must be in [0, 1]")
        if self.level_loss_alpha < 0:
            raise ValueError("level_loss_alpha must be >= 0")


def _arr(value):
    return value.data if isinstance(value, Tensor) else np.asarray(value)


def _as_tensor(value, dtype=np.float32):
    return ng.as_tensor(value, dtype=dtype)


def _linear_init(rng, out_features, in_features):
    bound = 1.0 / np.sqrt(in_features)
    return (rng.uniform(-bound, bound, (out_features, in_features))
            .astype(np.float32))


def _fp8_linear(x, weight_q, module):
    x = _as_tensor(x)
    original_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    out = x_flat.data @ weight_q.T
    if module.bias is not None:
        out = out + module.bias.data

    def bw(g):
        dy = g.reshape(-1, g.shape[-1])
        if module.backward_mode == "fp8":
            dy_q, _ = _quant.fp8_e4m3_quantize(dy)
            dx = dy_q @ weight_q
            x_q, _ = _quant.fp8_e4m3_quantize(x_flat.data)
            dweight = dy_q.T @ x_q
        else:
            dx = dy @ weight_q
            dweight = dy.T @ x_flat.data
        dbias = dy.sum(axis=0) if module.bias is not None else None
        return (dx.reshape(original_shape), dweight, dbias)

    inputs = (x, module.weight) if module.bias is None else (x, module.weight, module.bias)
    return ng.make_op(out.reshape(*original_shape[:-1], module.out_features), inputs, bw)


def _direct_ternary_linear(x, bias, module):
    x = _as_tensor(x)
    original_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    packed = module._packed_for_compute()
    scale = float(np.asarray(module.ternary_scale))
    shape = module._ternary_shape
    out = _quant.packed_matmat(x_flat.data, packed, shape, scale)
    if bias is not None:
        out = out + bias.data

    def bw(g):
        dy = g.reshape(-1, g.shape[-1])
        dx = _quant.packed_matmat_transpose(dy, packed, shape, scale)
        dbias = dy.sum(axis=0) if bias is not None else None
        if module.training and module.ternary_update_rate > 0:
            module._record_ternary_update_from_inputs(dy, x_flat.data)
        return (dx.reshape(original_shape),) if bias is None else (dx.reshape(original_shape), dbias)

    inputs = (x,) if bias is None else (x, bias)
    return ng.make_op(out.reshape(*original_shape[:-1], module.out_features), inputs, bw)


class _NCMQuantizedLinear:
    """Compatibility shim matching the former autograd.Function entry point."""

    @staticmethod
    def apply(x, weight, bias, forward_quant, backward_mode, packed, scale, packed_shape, fp8_weight, module=None):
        return _fp8_linear(x, fp8_weight, module)


class _DirectTernaryLinear:
    """Compatibility shim matching the former autograd.Function entry point."""

    @staticmethod
    def apply(x, bias, module):
        return _direct_ternary_linear(x, bias, module)


class NCMLinear(ng.Module):
    """Linear layer with optional FP32, FP8, or directly trained packed-ternary weights."""

    def __init__(self, in_features, out_features, bias=True,
                 forward_quant="fp32", backward_mode="fp32"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.forward_quant = forward_quant
        self.backward_mode = backward_mode
        self._tiled_packed = None
        self._cached_fp8_weight = None
        self.ternary_update_rate = 3e-4
        rng = ng.get_rng()
        self.weight = Parameter(_linear_init(rng, out_features, in_features))
        if bias:
            self.bias = Parameter(_linear_init(rng, out_features, 1).reshape(-1))
        else:
            self.bias = None

        if self.forward_quant == "ternary":
            packed, scale, shape, _ = _quant.pack_ternary(self.weight.data.T.copy())
            self.register_buffer("ternary_packed", np.asarray(packed, dtype=np.uint8))
            self.register_buffer("ternary_scale", np.asarray(float(scale), dtype=np.float32))
            self.register_buffer("ternary_updates", np.zeros((), dtype=np.int64))
            self._ternary_shape = tuple(shape)
            self._pending_ternary_indices = []
            self._pending_ternary_directions = []
            del self.weight
            self._cached_fp8_weight = None

    def _packed_for_compute(self):
        """Packed codes in the layout the native matmul kernels read fastest."""
        if self._tiled_packed is not None:
            return self._tiled_packed
        packed = None
        retile = getattr(_quant, "retile_ternary", None)
        if retile is not None:
            buf = retile(self.ternary_packed, self._ternary_shape)
            if buf is not None:
                packed = _quant.TiledPacked(buf)
        if packed is None:
            packed = self.ternary_packed
        self._tiled_packed = packed
        return packed

    def _invalidate_packed_cache(self) -> None:
        self._tiled_packed = None

    def _record_ternary_update(self, gradient_codes: Tensor) -> None:
        """Queue sparse gradient-sign updates until backward completes."""
        rate = float(self.ternary_update_rate)
        if rate <= 0:
            return
        gradient = np.ascontiguousarray(_arr(gradient_codes), dtype=np.float32)
        if gradient.shape != self._ternary_shape or not np.isfinite(gradient).all():
            return
        seed = int(ng.get_rng().integers(0, 2**63 - 1))
        sampler = getattr(_quant, "sample_ternary_gradient_updates", None)
        sampled = sampler(gradient, rate, seed) if sampler is not None else None
        if sampled is not None:
            indices, directions = sampled
            if np.asarray(indices).size:
                self._pending_ternary_indices.append(np.asarray(indices, dtype=np.int32))
                self._pending_ternary_directions.append(np.asarray(directions, dtype=np.int8))
            return
        importance = np.abs(gradient)
        mean_importance = float(importance.mean())
        if not mean_importance > 0:
            return
        probabilities = np.minimum(importance * (rate / mean_importance), 1.0)
        rng = np.random.default_rng(seed)
        selected = (rng.random(gradient.shape) < probabilities) & (importance > 0)
        indices = np.flatnonzero(selected.reshape(-1)).astype(np.int32)
        if indices.size == 0:
            return
        directions = (-np.sign(gradient.reshape(-1)[indices])).astype(np.int8)
        self._pending_ternary_indices.append(indices)
        self._pending_ternary_directions.append(directions)

    def _record_ternary_update_from_inputs(self, dy: Tensor, x_flat: Tensor) -> None:
        """Compute and sample ternary weight gradients without a full matrix."""
        rate = float(self.ternary_update_rate)
        if rate <= 0.0:
            return
        dy = np.asarray(_arr(dy), dtype=np.float32)
        x_flat = np.asarray(_arr(x_flat), dtype=np.float32)
        full_cols = int(self.out_features)
        full_rows = int(self.in_features)
        total = full_rows * full_cols
        if total > 2**31 - 1:
            raise ValueError("ternary update indices exceed int32 capacity")

        chunk_width = min(_TERNARY_GRADIENT_CHUNK_COLUMNS, full_cols)
        abs_sum = 0.0
        for start in range(0, full_cols, chunk_width):
            end = min(start + chunk_width, full_cols)
            grad_chunk = (dy[:, start:end].T @ x_flat).T
            if not np.isfinite(grad_chunk).all():
                return
            abs_sum += float(np.abs(grad_chunk.astype(np.float64)).sum())
        global_mean_abs = abs_sum / float(total)
        if not global_mean_abs > 0.0:
            return

        base_seed = int(ng.get_rng().integers(0, 2**63 - 1))
        chunk_sampler = getattr(_quant, "sample_ternary_gradient_update_chunk", None)
        pending_indices = []
        pending_directions = []

        for start in range(0, full_cols, chunk_width):
            end = min(start + chunk_width, full_cols)
            grad_chunk = (dy[:, start:end].T @ x_flat).T
            chunk_seed = (base_seed + start) % (2**63 - 1)
            if chunk_sampler is not None:
                indices, directions = chunk_sampler(
                    grad_chunk, rate, chunk_seed, global_mean_abs,
                    full_cols, start,
                )
                indices = np.asarray(indices, dtype=np.int32)
                directions = np.asarray(directions, dtype=np.int8)
            else:
                chunk_mean_abs = float(np.abs(grad_chunk.astype(np.float64)).sum()) / float(grad_chunk.size)
                if not chunk_mean_abs > 0.0:
                    continue
                adjusted_rate = rate * chunk_mean_abs / global_mean_abs
                sampler = getattr(_quant, "sample_ternary_gradient_updates", None)
                sampled = sampler(grad_chunk, adjusted_rate, chunk_seed) if sampler is not None else None
                if sampled is not None:
                    local_indices, directions = sampled
                    local_indices = np.asarray(local_indices, dtype=np.int64)
                    indices = ((local_indices // (end - start)) * full_cols
                               + start + local_indices % (end - start)).astype(np.int32)
                    directions = np.asarray(directions, dtype=np.int8)
                else:
                    importance = np.abs(grad_chunk)
                    probabilities = np.minimum(importance * (rate / global_mean_abs), 1.0)
                    rng = np.random.default_rng(int(chunk_seed))
                    selected = (rng.random(grad_chunk.shape) < probabilities) & (importance > 0)
                    local_indices = np.flatnonzero(selected.reshape(-1))
                    indices = ((local_indices // (end - start)) * full_cols
                               + start + local_indices % (end - start)).astype(np.int32)
                    directions = (-np.sign(grad_chunk.reshape(-1)[local_indices])).astype(np.int8)
            if indices.size:
                pending_indices.append(indices)
                pending_directions.append(directions)

        self._pending_ternary_indices.extend(pending_indices)
        self._pending_ternary_directions.extend(pending_directions)

    def _apply_pending_ternary_updates(self) -> None:
        """Apply queued sparse coordinate updates directly to packed 2-bit codes."""
        if not self._pending_ternary_indices:
            return

        applier = getattr(_quant, "apply_ternary_update_segments", None)
        if applier is not None:
            try:
                changed = applier(
                    self.ternary_packed, self._ternary_shape,
                    self._pending_ternary_indices, self._pending_ternary_directions,
                )
            except Exception:
                self._pending_ternary_indices.clear()
                self._pending_ternary_directions.clear()
                self._invalidate_packed_cache()
                raise
            if changed is not None:
                self._pending_ternary_indices.clear()
                self._pending_ternary_directions.clear()
                self._invalidate_packed_cache()
                if changed:
                    self.ternary_updates[...] += int(changed)
                return

        try:
            indices = np.concatenate([np.asarray(x, dtype=np.int64)
                                      for x in self._pending_ternary_indices])
            directions = np.concatenate([np.asarray(x, dtype=np.int16)
                                         for x in self._pending_ternary_directions])
        except Exception:
            self._pending_ternary_indices.clear()
            self._pending_ternary_directions.clear()
            raise
        self._pending_ternary_indices.clear()
        self._pending_ternary_directions.clear()
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        votes = np.zeros(unique_indices.size, dtype=np.int16)
        np.add.at(votes, inverse, directions)
        directions = np.sign(votes)
        active = directions != 0
        if not active.any():
            return
        unique_indices = unique_indices[active]
        directions = directions[active]

        packed = self.ternary_packed.reshape(-1)
        byte_indices = unique_indices // 4
        shifts = (unique_indices % 4 * 2).astype(np.int16)
        old_bytes = packed[byte_indices].astype(np.int16)
        old_raw = (old_bytes >> shifts) & 0x3
        new_raw = np.clip(old_raw + directions, 0, 2)
        changed = new_raw != old_raw
        if not changed.any():
            return
        byte_indices = byte_indices[changed]
        shifts = shifts[changed]
        new_raw = new_raw[changed]
        unique_bytes, byte_inverse = np.unique(byte_indices, return_inverse=True)
        masks = (0x3 << shifts).astype(np.int16)
        values = (new_raw << shifts).astype(np.int16)
        clear_masks = np.zeros(unique_bytes.size, dtype=np.int16)
        set_values = np.zeros(unique_bytes.size, dtype=np.int16)
        np.add.at(clear_masks, byte_inverse, masks)
        np.add.at(set_values, byte_inverse, values)
        old = packed[unique_bytes].astype(np.int16)
        updated = ((old & ~clear_masks) | set_values).astype(np.uint8)
        packed[unique_bytes] = updated
        self._invalidate_packed_cache()
        self.ternary_updates[...] += int(changed.sum())

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_quant == "fp32":
            return ng.linear(x, self.weight, self.bias)
        if self.forward_quant == "ternary":
            return _direct_ternary_linear(x, self.bias, self)
        fp8_weight, _ = _quant.fp8_e4m3_quantize(self.weight.data)
        self._cached_fp8_weight = fp8_weight
        return _fp8_linear(x, fp8_weight, self)


def migrate_ternary_state_dict(model: ng.Module, state_dict: dict) -> tuple[dict, bool]:
    """Convert legacy FP32-master ternary checkpoints to packed direct-ternary state."""
    converted = dict(state_dict)
    migrated = False
    for module_name, module in model.named_modules():
        if not isinstance(module, NCMLinear) or module.forward_quant != "ternary":
            continue
        prefix = f"{module_name}." if module_name else ""
        old_weight_key = prefix + "weight"
        packed_key = prefix + "ternary_packed"
        if old_weight_key not in converted or packed_key in converted:
            continue
        weight = np.asarray(_arr(converted.pop(old_weight_key)), dtype=np.float32)
        packed, scale, shape, _ = _quant.pack_ternary(weight.T.copy())
        if tuple(shape) != module._ternary_shape:
            raise ValueError(f"legacy ternary checkpoint shape mismatch at {old_weight_key}")
        converted[packed_key] = np.asarray(packed, dtype=np.uint8)
        converted[prefix + "ternary_scale"] = np.asarray(float(scale), dtype=np.float32)
        converted.setdefault(
            prefix + "ternary_updates", np.array(module.ternary_updates, copy=True)
        )
        migrated = True
    return converted, migrated


def apply_direct_ternary_updates(model: ng.Module) -> None:
    """Flush pending sparse ternary updates once per completed backward pass."""
    for module in model.modules():
        if isinstance(module, NCMLinear) and module.forward_quant == "ternary":
            module._apply_pending_ternary_updates()


def level_length(length: int, level: int, levels: int) -> int:
    """Length at level, from coarsest to finest."""
    divisor = 1 << (levels - level - 1)
    return max(1, (length + divisor - 1) // divisor)


def resize_sequence(x: Tensor, length: int, mode: str = "linear") -> Tensor:
    if x.shape[1] == length:
        return x
    transposed = ng.transpose(x, (0, 2, 1))
    if mode == "area":
        resized = ng.interpolate1d(transposed, length, "area")
    else:
        resized = ng.interpolate1d(transposed, length, "linear")
    return ng.transpose(resized, (0, 2, 1))


class RMSNorm(ng.Module):
    """SmaulBRAIN-style RMS normalization with epsilon 1e-6."""

    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dimension, dtype=np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = ng.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SmaulExpert(ng.Module):
    """One SmaulBRAIN-style GELU expert with bias-free projections."""

    def __init__(self, dimension: int, hidden_dimension: int,
                 forward_quant: str = "fp32", backward_mode: str = "fp32"):
        super().__init__()
        self.W_in = NCMLinear(
            dimension, hidden_dimension, bias=False,
            forward_quant=forward_quant, backward_mode=backward_mode,
        )
        self.W_out = NCMLinear(
            hidden_dimension, dimension, bias=False,
            forward_quant=forward_quant, backward_mode=backward_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.W_out(ng.gelu(self.W_in(x)))


class SmaulSharedMoE(ng.Module):
    """One sparse router and expert pool shared by every recurrent block."""

    def __init__(self, dimension: int, n_experts: int, top_k: int,
                 expert_hidden: int = 0, forward_quant: str = "fp32",
                 backward_mode: str = "fp32"):
        super().__init__()
        if n_experts < 1 or not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        if expert_hidden < 1:
            expert_hidden = 4 * dimension
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = NCMLinear(
            dimension, n_experts, bias=False,
            forward_quant=forward_quant, backward_mode=backward_mode,
        )
        self.experts = ng.ModuleList(
            SmaulExpert(dimension, expert_hidden, forward_quant, backward_mode)
            for _ in range(n_experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        route_logits = self.router(flat)
        chosen_experts = np.argsort(-route_logits.data, axis=-1, kind="stable")[:, :self.top_k]
        chosen_logits = ng.gather_last(route_logits, chosen_experts)
        route_weights = ng.softmax(chosen_logits, axis=-1)

        token_ids = np.arange(flat.shape[0])
        assignment_tokens = np.repeat(token_ids, self.top_k)
        assignment_experts = chosen_experts.reshape(-1)
        assignment_weights = route_weights.reshape(-1)
        order = np.argsort(assignment_experts, kind="stable")
        grouped_tokens = assignment_tokens[order]
        grouped_weights = assignment_weights[order]
        counts = np.bincount(assignment_experts, minlength=self.n_experts)

        output = Tensor(np.zeros_like(flat.data))
        start = 0
        for expert_id, count in enumerate(counts):
            end = start + int(count)
            if count:
                token_indices = grouped_tokens[start:end]
                expert_output = self.experts[expert_id](flat[token_indices])
                weights = grouped_weights[start:end].unsqueeze(-1)
                output = ng.index_add_rows(output, token_indices, expert_output * weights)
            start = end
        return output.reshape(original_shape)


class SmaulRecurrentBlock(ng.Module):
    """Bidirectional, parallel-scan adaptation of SmaulBRAIN's recurrent block.

    Attention uses the elementwise recurrent state s[t] = decay * s[t-1] +
    k[t] * v[t], with a reverse scan added so every position in an NCM level
    can use context from both sides while the whole level is predicted at once.
    """

    _SCAN_CHUNK = 128

    def __init__(self, dimension: int, forward_quant: str = "fp32",
                 backward_mode: str = "fp32"):
        super().__init__()
        self.norm_attention = RMSNorm(dimension)
        self.norm_channel = RMSNorm(dimension)
        linear_args = {
            "bias": False,
            "forward_quant": forward_quant,
            "backward_mode": backward_mode,
        }
        self.W_r = NCMLinear(dimension, dimension, **linear_args)
        self.W_k = NCMLinear(dimension, dimension, **linear_args)
        self.W_v = NCMLinear(dimension, dimension, **linear_args)
        self.W_o = NCMLinear(dimension, dimension, **linear_args)
        self.decay_logit = Parameter(np.full((dimension,), -4.0, dtype=np.float32))
        self.bonus = Parameter(np.ones(dimension, dtype=np.float32))

    def _scan(self, values: Tensor, decay_rate: Tensor) -> Tensor:
        """Chunked closed-form recurrent scan."""
        if values.shape[1] == 0:
            return values
        alpha = ng.exp(-decay_rate.float())
        return ng.scan(values, alpha)

    def forward(self, x: Tensor, segment_lengths: Sequence[int],
                channel_mixer: ng.Module) -> Tensor:
        if sum(segment_lengths) != x.shape[1] or any(length < 1 for length in segment_lengths):
            raise ValueError("segment_lengths must be positive and sum to the sequence length")

        normalized = self.norm_attention(x)
        read_gate = ng.sigmoid(self.W_r(normalized))
        keys = self.W_k(normalized)
        values = self.W_v(normalized)
        key_value = keys * values
        decay_rate = ng.clamp(ng.softplus(self.decay_logit), 1e-4, 0.1)

        contexts = []
        offset = 0
        for length in segment_lengths:
            segment = key_value[:, offset:offset + length]
            batch = segment.shape[0]
            both = self._scan(ng.concat((segment, ng.flip(segment, 1)), axis=0), decay_rate)
            contexts.append(0.5 * (both[:batch] + ng.flip(both[batch:], 1)))
            offset += length
        context = ng.concat(contexts, axis=1)
        attended = self.W_o(read_gate * (context + self.bonus * key_value))
        x = x + attended

        channel_input = self.norm_channel(x)
        return x + channel_mixer(channel_input)


class SmaulDenseFFN(ng.Module):
    """SwiGLU channel mixer used by the dense presets."""

    def __init__(self, dimension: int, hidden_dimension: int,
                 forward_quant: str = "fp32", backward_mode: str = "fp32"):
        super().__init__()
        self.W_a = NCMLinear(dimension, hidden_dimension, forward_quant=forward_quant,
                             backward_mode=backward_mode)
        self.W_b = NCMLinear(dimension, hidden_dimension, forward_quant=forward_quant,
                             backward_mode=backward_mode)
        self.W_c = NCMLinear(hidden_dimension, dimension, forward_quant=forward_quant,
                             backward_mode=backward_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_c(ng.silu(self.W_a(x)) * self.W_b(x))


class ResidualVQTokenizer(ng.Module):
    """Byte-level multiscale residual VQ tokenizer with a reconstruction loss."""

    def __init__(self, config: NCMConfig):
        super().__init__()
        self.config = config
        self.byte_embedding = ng.Embedding(config.vocab_size, config.d_model)
        self.encoder = ng.Sequential(
            NCMLinear(config.d_model, config.d_model,
                      forward_quant=config.forward_quant,
                      backward_mode=config.backward_mode),
            ng.GELU(),
            NCMLinear(config.d_model, config.d_model,
                      forward_quant=config.forward_quant,
                      backward_mode=config.backward_mode),
        )
        self.feature_norm = RMSNorm(config.d_model)
        self.codebooks = ng.ModuleList(
            ng.Embedding(config.codebook_size, config.d_model)
            for _ in range(config.levels)
        )
        self.register_buffer("_codebook_initialized",
                             np.zeros(config.levels, dtype=np.bool_))
        self.decoder_norm = RMSNorm(config.d_model)
        self.decoder = ng.Sequential(
            NCMLinear(config.d_model, config.d_model,
                      forward_quant=config.forward_quant,
                      backward_mode=config.backward_mode),
            ng.GELU(),
            NCMLinear(config.d_model, config.vocab_size,
                      forward_quant=config.forward_quant,
                      backward_mode=config.backward_mode),
        )
        self.last_loss_components: dict[str, Tensor] = {}

    def _quantize(self, residual: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        codebook = self.codebooks[level].weight
        if self.training and not bool(self._codebook_initialized[level]):
            samples = residual.detach().reshape(-1, residual.shape[-1]).data
            choices = ng.get_rng().integers(0, samples.shape[0], self.config.codebook_size)
            codebook.data[...] = samples[choices]
            self._codebook_initialized[level] = True
        flat = residual.reshape(-1, residual.shape[-1]).float()
        weights = codebook
        distances = (np.sum(flat.data ** 2, axis=1, keepdims=True)
                     - 2.0 * flat.data @ weights.data.T
                     + np.sum(weights.data ** 2, axis=1)[None, :])
        ids = np.argmin(distances, axis=-1).reshape(residual.shape[:2])
        ids_t = Tensor(ids.astype(np.int64))
        quantized = self.codebooks[level](ids_t)
        vq_loss = ng.mse_loss(quantized, residual.detach())
        commitment = ng.mse_loss(residual, quantized.detach())
        straight_through = residual + (quantized - residual).detach()
        return ids_t, straight_through, vq_loss + self.config.commitment_weight * commitment

    def _encode_levels(self, byte_ids: Tensor, features: Tensor):
        """Multiscale residual quantization.

        Returns (codes, cumulative reconstruction, vq loss). The decoder is
        not involved: encoding only needs the code ids.
        """
        cumulative = Tensor(np.zeros(features.shape, dtype=np.float32))
        codes: list[Tensor] = []
        vq_loss = Tensor(np.zeros((), dtype=np.float32))
        for level in range(self.config.levels):
            size = level_length(byte_ids.shape[1], level, self.config.levels)
            target = ng.transpose(ng.adaptive_avg_pool1d(
                ng.transpose(features, (0, 2, 1)), size
            ), (0, 2, 1))
            prior = resize_sequence(cumulative, size)
            ids, quantized, level_loss = self._quantize(target - prior, level)
            codes.append(ids)
            cumulative = cumulative + resize_sequence(quantized, byte_ids.shape[1])
            vq_loss = vq_loss + level_loss
        return codes, cumulative, vq_loss

    def forward(self, byte_ids: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time]")
        if not 1 <= byte_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("sequence length must be within [1, max_seq_len]")
        features = self.feature_norm(self.encoder(self.byte_embedding(byte_ids)))
        codes, cumulative, vq_loss = self._encode_levels(byte_ids, features)
        logits = self.decoder(self.decoder_norm(cumulative))
        reconstruction = ng.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), byte_ids.reshape(-1)
        )
        mean_vq_loss = vq_loss / self.config.levels
        total_loss = reconstruction + mean_vq_loss
        self.last_loss_components = {
            "reconstruction": reconstruction.detach(),
            "quantization": mean_vq_loss.detach(),
            "total": total_loss.detach(),
        }
        return logits, total_loss, codes

    def encode(self, byte_ids: Tensor) -> list[Tensor]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time]")
        if not 1 <= byte_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("sequence length must be within [1, max_seq_len]")
        was_training = self.training
        self.eval()
        with ng.no_grad():
            features = self.feature_norm(self.encoder(self.byte_embedding(byte_ids)))
            codes, _cumulative, _vq_loss = self._encode_levels(byte_ids, features)
        self.train(was_training)
        return codes

    def decode(self, codes: Sequence[Tensor], output_length: int) -> Tensor:
        if len(codes) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        if not 1 <= output_length <= self.config.max_seq_len:
            raise ValueError("output_length must be within [1, max_seq_len]")
        with ng.no_grad():
            batch = codes[0].shape[0]
            cumulative = Tensor(np.zeros((batch, output_length, self.config.d_model),
                                         dtype=np.float32))
            for level, ids in enumerate(codes):
                expected = level_length(output_length, level, self.config.levels)
                if ids.shape != (batch, expected):
                    raise ValueError(f"level {level} codes must have shape [batch, {expected}]")
                cumulative = cumulative + resize_sequence(
                    self.codebooks[level](ids), output_length)
            logits = self.decoder(self.decoder_norm(cumulative))
        return Tensor(np.argmax(logits.data, axis=-1).astype(np.int64))


def _checkpointed_block(block, hidden, segment_lengths, mixer):
    with ng.no_grad():
        output = block(hidden, segment_lengths, mixer)
    def bw(g):
        local = Tensor(hidden.data.copy(), requires_grad=hidden.requires_grad)
        with ng.enable_grad():
            recomputed = block(local, segment_lengths, mixer)
            recomputed.backward(g)
        return (local.grad,)
    return ng.make_op(output.data.copy(), (hidden,), bw)


class NCMGenerator(ng.Module):
    """Predict every code level in one training pass; generate one level per step."""

    def __init__(self, config: NCMConfig, tokenizer: ResidualVQTokenizer,
                 gradient_checkpointing: bool = False):
        super().__init__()
        if tokenizer.config != config:
            raise ValueError("tokenizer and generator must use the same NCMConfig")
        self.config = config
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._tokenizer = (tokenizer,)
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        tokenizer.eval()
        linear_args = {
            "forward_quant": config.forward_quant,
            "backward_mode": config.backward_mode,
        }
        self.input_projection = NCMLinear(config.d_model, config.d_model, **linear_args)
        self.prefix_projection = NCMLinear(config.d_model, config.d_model, **linear_args)
        self.level_embeddings = ng.Embedding(config.levels, config.d_model)
        self.position_embeddings = ng.Embedding(config.max_seq_len, config.d_model)
        if config.model_kind == "moe":
            self.channel_mixer = SmaulSharedMoE(
                config.d_model, config.n_experts, config.top_k,
                config.expert_hidden, config.forward_quant, config.backward_mode,
            )
            self.dense_mixers = None
        else:
            self.channel_mixer = None
            self.dense_mixers = ng.ModuleList(
                SmaulDenseFFN(config.d_model, config.ffn_hidden,
                              config.forward_quant, config.backward_mode)
                for _ in range(config.n_layers)
            )
        self.blocks = ng.ModuleList(
            SmaulRecurrentBlock(config.d_model, config.forward_quant, config.backward_mode)
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.output_heads = ng.ModuleList(
            NCMLinear(config.d_model, config.codebook_size, **linear_args)
            for _ in range(config.levels)
        )

    @property
    def tokenizer(self) -> ResidualVQTokenizer:
        return self._tokenizer[0]

    def _level_parts(self, codes: Sequence[Tensor], corrupt: bool) -> tuple[list[Tensor], list[int]]:
        if len(codes) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        batch = codes[0].shape[0]
        length = codes[-1].shape[1]
        if length > self.config.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")
        cumulative = Tensor(np.zeros((batch, length, self.config.d_model), dtype=np.float32))
        parts: list[Tensor] = []
        lengths: list[int] = []
        for level, ids in enumerate(codes):
            expected = level_length(length, level, self.config.levels)
            if ids.shape != (batch, expected):
                raise ValueError(f"level {level} codes must have shape [batch, {expected}]")
            lengths.append(expected)
            current = resize_sequence(cumulative, expected, mode="area")
            positions = np.arange(expected)
            part = self.input_projection(current)
            part = part + self.position_embeddings(positions).unsqueeze(0)
            part = part + self.level_embeddings.weight[level].reshape(1, 1, -1)
            parts.append(part)
            conditioning_ids = ids
            if corrupt and self.training and self.config.conditioning_corruption > 0:
                rng = ng.get_rng()
                replace = rng.random(ids.shape) < self.config.conditioning_corruption
                random_ids = rng.integers(0, self.config.codebook_size, ids.shape)
                conditioning_ids = Tensor(np.where(replace, random_ids, ids.data).astype(np.int64))
            quantized = self.tokenizer.codebooks[level](conditioning_ids)
            cumulative = cumulative + resize_sequence(quantized, length)
        return parts, lengths

    def _build_level_inputs(self, codes: Sequence[Tensor], corrupt: bool) -> tuple[Tensor, list[int]]:
        parts, lengths = self._level_parts(codes, corrupt)
        return ng.concat(parts, axis=1), lengths

    def _prefix_features(self, prefix_tokens: Tensor, batch: int) -> Tensor:
        """Frozen-tokenizer features for the prompt; carries no autograd graph."""
        if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != batch:
            raise ValueError("prefix_tokens must have shape [batch, prefix_length]")
        if not 1 <= prefix_tokens.shape[1] <= self.config.max_seq_len:
            raise ValueError("prefix length must be within [1, max_seq_len]")
        with ng.no_grad():
            return self.tokenizer.feature_norm(
                self.tokenizer.encoder(self.tokenizer.byte_embedding(prefix_tokens))
            )

    def _project_prefix(self, features: Tensor) -> tuple[Tensor, int]:
        positions = np.arange(features.shape[1])
        prefix = self.prefix_projection(features)
        prefix = prefix + self.position_embeddings(positions).unsqueeze(0)
        return prefix, prefix.shape[1]

    def _run_blocks(self, hidden: Tensor, segment_lengths: list[int]) -> Tensor:
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and ng.is_grad_enabled()
        )
        for layer_index, block in enumerate(self.blocks):
            mixer = (self.channel_mixer if self.config.model_kind == "moe"
                     else self.dense_mixers[layer_index])
            if use_checkpoint:
                hidden = _checkpointed_block(block, hidden, segment_lengths, mixer)
            else:
                hidden = block(hidden, segment_lengths, mixer)
        return hidden

    def iter_level_logits(self, codes: Sequence[Tensor], *, prefix_tokens: Tensor | None = None,
                          corrupt: bool = True):
        """Yield ``(logits, target_ids)`` one level at a time."""
        parts, _lengths = self._level_parts(codes, corrupt)
        features = None
        if prefix_tokens is not None:
            features = self._prefix_features(prefix_tokens, parts[0].shape[0])
        for level, (part, target_ids) in enumerate(zip(parts, codes)):
            if features is None:
                segment = part
                prefix_length = 0
            else:
                prefix, prefix_length = self._project_prefix(features)
                segment = ng.concat((prefix, part), axis=1)
            hidden = self.final_norm(self._run_blocks(segment, [segment.shape[1]]))
            if features is not None:
                hidden = hidden[:, prefix_length:]
            yield self.output_heads[level](hidden), target_ids

    def forward(self, codes: Sequence[Tensor], *, prefix_tokens: Tensor | None = None,
                corrupt: bool = True) -> list[Tensor]:
        inputs, lengths = self._build_level_inputs(codes, corrupt)
        level_inputs = ng.split(inputs, lengths, axis=1)
        prefix = None
        prefix_length = 0
        if prefix_tokens is not None:
            features = self._prefix_features(prefix_tokens, inputs.shape[0])
            prefix, prefix_length = self._project_prefix(features)

        segments = [
            ng.concat((prefix, part), axis=1) if prefix is not None else part
            for part in level_inputs
        ]
        segment_lengths = [part.shape[1] for part in segments]
        hidden = self.final_norm(
            self._run_blocks(ng.concat(segments, axis=1), segment_lengths)
        )

        encoded_segments = ng.split(hidden, segment_lengths, axis=1)
        logits: list[Tensor] = []
        for level, encoded in enumerate(encoded_segments):
            if prefix is not None:
                encoded = encoded[:, prefix_length:]
            logits.append(self.output_heads[level](encoded))
        return logits

    def predict_level(self, codes: Sequence[Tensor], level: int,
                      *, prefix_tokens: Tensor | None = None) -> Tensor:
        """Predict only one level, avoiding work on unrelated levels at inference."""
        if not 0 <= level < self.config.levels:
            raise ValueError("level is outside the configured hierarchy")
        if len(codes) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        batch = codes[0].shape[0]
        length = codes[-1].shape[1]
        if length > self.config.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")
        cumulative = Tensor(np.zeros((batch, length, self.config.d_model), dtype=np.float32))
        for prior_level in range(level):
            prior_length = level_length(length, prior_level, self.config.levels)
            ids = codes[prior_level]
            if ids.shape != (batch, prior_length):
                raise ValueError(
                    f"level {prior_level} codes must have shape [batch, {prior_length}]"
                )
            cumulative = cumulative + resize_sequence(
                self.tokenizer.codebooks[prior_level](ids), length
            )
        current = resize_sequence(cumulative, codes[level].shape[1], mode="area")
        current = self.input_projection(current)
        positions = np.arange(current.shape[1])
        current = current + self.position_embeddings(positions).unsqueeze(0)
        current = current + self.level_embeddings.weight[level].reshape(1, 1, -1)
        prefix_length = 0
        if prefix_tokens is None:
            hidden = current
        else:
            features = self._prefix_features(prefix_tokens, batch)
            prefix, prefix_length = self._project_prefix(features)
            hidden = ng.concat((prefix, current), axis=1)
        hidden = self.final_norm(self._run_blocks(hidden, [hidden.shape[1]]))
        if prefix_length:
            hidden = hidden[:, prefix_length:]
        return self.output_heads[level](hidden)

    def generate_codes(self, lengths: Sequence[int], device: str = "cpu",
                       temperature: float = 0.0, top_k: int = 0,
                       prefix_tokens: Tensor | None = None, top_p: float = 1.0) -> list[Tensor]:
        """Generate the hierarchy in K sequential calls, parallel within each level."""
        if len(lengths) != self.config.levels:
            raise ValueError("one length is required per level")
        if any(n < 1 or n > self.config.max_seq_len for n in lengths):
            raise ValueError("each level length must be within [1, max_seq_len]")
        if any(lengths[i] > lengths[i + 1] for i in range(len(lengths) - 1)):
            raise ValueError("level lengths must be nondecreasing")
        if temperature < 0 or top_k < 0:
            raise ValueError("temperature and top_k must be >= 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        batch_size = prefix_tokens.shape[0] if prefix_tokens is not None else 1
        codes = [Tensor(np.zeros((batch_size, n), dtype=np.int64)) for n in lengths]
        with ng.no_grad():
            for level in range(self.config.levels):
                logits = self.predict_level(codes, level, prefix_tokens=prefix_tokens)
                lg = logits.data
                if temperature == 0:
                    ids = np.argmax(lg, axis=-1)
                else:
                    if top_k:
                        k = min(top_k, lg.shape[-1])
                        threshold = np.partition(lg, -k, axis=-1)[..., -k, None]
                        lg = np.where(lg < threshold, -np.inf, lg)
                    probs = ng.softmax(Tensor(lg / temperature), axis=-1).data
                    if top_p < 1.0:
                        order = np.argsort(-probs, axis=-1, kind="stable")
                        sorted_probs = np.take_along_axis(probs, order, axis=-1)
                        remove = np.cumsum(sorted_probs, axis=-1) - sorted_probs >= top_p
                        sorted_probs = np.where(remove, 0.0, sorted_probs)
                        probs = np.zeros_like(probs)
                        np.put_along_axis(probs, order, sorted_probs, axis=-1)
                        probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
                    flat = probs.reshape(-1, probs.shape[-1])
                    u = ng.get_rng().random(flat.shape[0])
                    cdf = np.cumsum(flat, axis=-1)
                    ids = (cdf < u[:, None]).sum(axis=-1)
                    ids = np.minimum(ids, probs.shape[-1] - 1)
                    ids = ids.reshape(probs.shape[:-1])
                codes[level] = Tensor(ids.astype(np.int64))
        return codes

    def generate(self, output_length: int, device: str = "cpu",
                 temperature: float = 0.0, top_k: int = 0,
                 prefix_tokens: Tensor | None = None, top_p: float = 1.0) -> Tensor:
        lengths = [
            level_length(output_length, level, self.config.levels)
            for level in range(self.config.levels)
        ]
        codes = self.generate_codes(
            lengths, device, temperature, top_k, prefix_tokens, top_p=top_p
        )
        return self.tokenizer.decode(codes, output_length)


# Compatibility alias for existing scripts and user code that imported the old name.
NCMTransformer = NCMGenerator


def _level_weights(model: NCMGenerator, codes: Sequence[Tensor]) -> list[float]:
    alpha = model.config.level_loss_alpha
    weights = np.asarray(
        [max(1, target.shape[1]) ** (1.0 - alpha) for target in codes],
        dtype=np.float32,
    )
    weights = weights / weights.sum()
    return [float(w) for w in weights]


def ncm_loss(model: NCMGenerator, codes: Sequence[Tensor],
             *, prefix_tokens: Tensor | None = None) -> Tensor:
    """Length-normalized multilevel CE; alpha=1 weights each level equally."""
    logits = model(codes, prefix_tokens=prefix_tokens, corrupt=True)
    losses = [
        ng.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
        for pred, target in zip(logits, codes)
    ]
    weights = _level_weights(model, codes)
    return sum(weight * loss for weight, loss in zip(weights, losses))


def backward_ncm_loss_chunked(model: NCMGenerator, codes: Sequence[Tensor],
                              *, prefix_tokens: Tensor | None = None) -> Tensor:
    """Backpropagate one level at a time instead of holding every level."""
    weights = _level_weights(model, codes)
    total: Tensor | None = None
    for weight, (logits, target) in zip(
        weights,
        model.iter_level_logits(codes, prefix_tokens=prefix_tokens, corrupt=True),
    ):
        loss = ng.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        (weight * loss).backward()
        term = loss.detach() * weight
        total = term if total is None else total + term
    return total
