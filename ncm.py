"""Default coarse-to-fine Next-Concept Modeling runtime for SmaulBRAIN.

Train ResidualVQTokenizer first, freeze it, then train NCMGenerator against its
multiscale code hierarchies. The CLI entry points use this architecture by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from smaulbrain import quant as _quant


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


class _NCMQuantizedLinear(torch.autograd.Function):
    """Linear operation with packed ternary forward and no dense weight reconstruction."""

    @staticmethod
    def forward(ctx, x, weight, bias, forward_quant, backward_mode,
                packed, scale, packed_shape, fp8_weight):
        input_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        ctx.forward_quant = forward_quant
        ctx.backward_mode = backward_mode
        ctx.packed = packed
        ctx.scale = scale
        ctx.packed_shape = packed_shape
        ctx.has_bias = bias is not None
        ctx.input_shape = input_shape
        ctx.save_for_backward(x_flat, weight, fp8_weight)

        if forward_quant == "fp8":
            output = x_flat @ fp8_weight.t()
        elif forward_quant == "ternary":
            output = _quant.packed_matmat(x_flat, packed, packed_shape, scale)
        else:
            output = x_flat @ weight.t()
        if bias is not None:
            output = output + bias
        return output.reshape(*input_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, fp8_weight = ctx.saved_tensors
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        mode = ctx.backward_mode
        kind = ctx.forward_quant

        if kind == "ternary":
            dx = _quant.packed_matmat_transpose(
                dy, ctx.packed, ctx.packed_shape, ctx.scale
            )
        elif kind == "fp8":
            if mode == "fp8":
                dy_q, _ = _quant.fp8_e4m3_quantize(dy)
                dx = dy_q @ fp8_weight
            else:
                dx = dy @ fp8_weight
        else:
            dx = dy @ weight

        if kind != "fp32" and mode == "fp8":
            dy_for_weight, _ = _quant.fp8_e4m3_quantize(dy)
            x_for_weight, _ = _quant.fp8_e4m3_quantize(x)
        else:
            dy_for_weight, x_for_weight = dy, x
        dweight = dy_for_weight.t() @ x_for_weight
        dbias = dy.sum(dim=0) if ctx.has_bias else None
        return (
            dx.reshape(ctx.input_shape).to(dtype=x.dtype),
            dweight.to(dtype=weight.dtype),
            dbias,
            None, None, None, None, None, None,
        )


class _DirectTernaryLinear(torch.autograd.Function):
    """Linear autograd using packed ternary weights and direct discrete updates.

    No dense master-weight Parameter or dense weight-gradient buffer is retained.
    The backward pass computes a transient weight-gradient matrix, then applies
    sparse sign updates directly to the ternary codes and immediately repacks them.
    """

    @staticmethod
    def forward(ctx, x, bias, module):
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        packed = module.ternary_packed
        scale = float(module.ternary_scale.item())
        shape = (module.in_features, module.out_features)
        ctx.module = module
        ctx.packed = packed
        ctx.scale = scale
        ctx.packed_shape = shape
        ctx.input_shape = original_shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x_flat)
        output = _quant.packed_matmat(x_flat, packed, shape, scale)
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], module.out_features)

    @staticmethod
    def backward(ctx, grad_output):
        (x_flat,) = ctx.saved_tensors
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        dx = _quant.packed_matmat_transpose(
            dy, ctx.packed, ctx.packed_shape, ctx.scale
        )
        dbias = dy.sum(dim=0) if ctx.has_bias else None
        # Ternary codes are the trainable weights. Keep the dense gradient only
        # for this update; never attach it to a full-sized FP32 .grad tensor.
        if ctx.module.training and ctx.module.ternary_update_rate > 0:
            grad_codes = (dy.t() @ x_flat).t()
            ctx.module._update_ternary_codes(grad_codes)
        return dx.reshape(ctx.input_shape).to(dtype=x_flat.dtype), dbias, None


class NCMLinear(nn.Linear):
    """Linear layer with optional FP32, FP8, or directly trained packed-ternary weights."""

    def __init__(self, in_features, out_features, bias=True,
                 forward_quant="fp32", backward_mode="fp32"):
        super().__init__(in_features, out_features, bias=bias)
        self.forward_quant = forward_quant
        self.backward_mode = backward_mode
        self._cached_version = -1
        self._cached_packed = None
        self._cached_packed_scale = 1.0
        self._cached_packed_shape = None
        self._cached_fp8_weight = None
        self.ternary_update_rate = 3e-4

        if self.forward_quant == "ternary":
            # The initialized FP32 tensor exists only during this conversion.
            # Persistent trainable weight storage is the packed 2-bit code tensor.
            packed, scale, shape, _ = _quant.pack_ternary(
                self.weight.detach().t().contiguous()
            )
            self.register_buffer("ternary_packed", packed.to(dtype=torch.uint8))
            self.register_buffer(
                "ternary_scale", torch.tensor(float(scale), dtype=torch.float32)
            )
            self.register_buffer("ternary_updates", torch.zeros((), dtype=torch.long))
            self._ternary_shape = tuple(shape)
            del self._parameters["weight"]

    def _update_ternary_codes(self, gradient_codes: Tensor) -> None:
        """Apply gradient-directed stochastic coordinate updates without a master weight."""
        rate = float(self.ternary_update_rate)
        if rate <= 0:
            return

        gradient = gradient_codes.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if gradient.shape != self._ternary_shape or not torch.isfinite(gradient).all():
            return
        importance = gradient.abs()
        mean_importance = float(importance.mean())
        if not mean_importance > 0:
            return

        # Sample more often where the loss gradient is larger. Each selected
        # element moves by one ternary level, clamped to {-1, 0, +1}.
        probabilities = (importance * (rate / mean_importance)).clamp_(max=1.0)
        selected = torch.rand(importance.shape) < probabilities
        selected &= importance > 0
        if not bool(selected.any()):
            return

        codes = _quant.unpack_codes(self.ternary_packed, self._ternary_shape).to(torch.int8)
        direction = -gradient[selected].sign().to(torch.int8)
        updated = (
            codes[selected].to(torch.int16) + direction.to(torch.int16)
        ).clamp_(-1, 1).to(torch.int8)
        changed = updated != codes[selected]
        if not bool(changed.any()):
            return
        selected_indices = selected.nonzero(as_tuple=True)
        selected_codes = codes[selected_indices]
        selected_codes[changed] = updated[changed]
        codes[selected_indices] = selected_codes

        # Packing {-1, 0, +1} directly yields canonical 2-bit codes. Its
        # returned scale is one; preserve the learned-at-initialization layer scale.
        new_packed, _, new_shape, _ = _quant.pack_ternary(
            codes.to(dtype=torch.float32).contiguous()
        )
        if tuple(new_shape) != self._ternary_shape:
            raise RuntimeError("ternary weight shape changed during direct update")
        self.ternary_packed = new_packed.to(dtype=torch.uint8)
        self.ternary_updates.add_(changed.sum().to(dtype=self.ternary_updates.dtype))

    def _quantized_weight(self):
        version = self.weight._version
        if version == self._cached_version:
            return
        self._cached_version = version
        self._cached_fp8_weight = None
        self._cached_packed = None
        self._cached_packed_shape = None
        if self.forward_quant == "fp8":
            self._cached_fp8_weight, _ = _quant.fp8_e4m3_quantize(self.weight.detach())

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_quant == "fp32":
            return F.linear(x, self.weight, self.bias)
        if self.forward_quant == "ternary":
            return _DirectTernaryLinear.apply(x, self.bias, self)
        self._quantized_weight()
        fp8_weight = self._cached_fp8_weight
        return _NCMQuantizedLinear.apply(
            x, self.weight, self.bias, self.forward_quant, self.backward_mode,
            None, 1.0, None, fp8_weight,
        )


def migrate_ternary_state_dict(model: nn.Module, state_dict: dict) -> tuple[dict, bool]:
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
        weight = converted.pop(old_weight_key)
        packed, scale, shape, _ = _quant.pack_ternary(
            weight.detach().t().contiguous()
        )
        if tuple(shape) != module._ternary_shape:
            raise ValueError(f"legacy ternary checkpoint shape mismatch at {old_weight_key}")
        converted[packed_key] = packed.to(dtype=torch.uint8)
        converted[prefix + "ternary_scale"] = torch.tensor(
            float(scale), dtype=torch.float32
        )
        converted.setdefault(
            prefix + "ternary_updates", module.ternary_updates.detach().clone()
        )
        migrated = True
    return converted, migrated


def level_length(length: int, level: int, levels: int) -> int:
    """Length at level, from coarsest to finest."""
    divisor = 1 << (levels - level - 1)
    return max(1, (length + divisor - 1) // divisor)


def resize_sequence(x: Tensor, length: int, mode: str = "linear") -> Tensor:
    if x.shape[1] == length:
        return x
    if mode == "area":
        return F.interpolate(x.transpose(1, 2), size=length, mode="area").transpose(1, 2)
    return F.interpolate(
        x.transpose(1, 2), size=length, mode="linear", align_corners=False
    ).transpose(1, 2)


class RMSNorm(nn.Module):
    """SmaulBRAIN-style RMS normalization with epsilon 1e-6."""

    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class SmaulExpert(nn.Module):
    """One SmaulBRAIN-style GELU expert with bias-free projections."""

    def __init__(
        self, dimension: int, hidden_dimension: int,
        forward_quant: str = "fp32", backward_mode: str = "fp32",
    ):
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
        return self.W_out(F.gelu(self.W_in(x)))


class SmaulSharedMoE(nn.Module):
    """One sparse router and expert pool shared by every recurrent block."""

    def __init__(
        self, dimension: int, n_experts: int, top_k: int,
        expert_hidden: int = 0, forward_quant: str = "fp32",
        backward_mode: str = "fp32",
    ):
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
        self.experts = nn.ModuleList(
            SmaulExpert(
                dimension, expert_hidden, forward_quant, backward_mode
            ) for _ in range(n_experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        route_logits = self.router(flat)
        chosen_logits, chosen_experts = route_logits.topk(self.top_k, dim=-1)
        route_weights = torch.softmax(chosen_logits, dim=-1)
        output = torch.zeros_like(flat)
        # Sparse dispatch is faster than evaluating every expert in training and
        # inference (measured on CPU, 4 experts/top-2: 41.4 -> 25.6 ms per 128-byte
        # generation at d_model=128), so both modes share it.
        for expert_id, expert in enumerate(self.experts):
            token_indices, slots = torch.where(chosen_experts == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_indices))
            weights = route_weights[token_indices, slots].unsqueeze(-1)
            output = output.index_add(0, token_indices, expert_output * weights)
        return output.reshape(original_shape)


class SmaulRecurrentBlock(nn.Module):
    """Bidirectional, parallel-scan adaptation of SmaulBRAIN's recurrent block.

    Attention uses the elementwise recurrent state s[t] = decay * s[t-1] +
    k[t] * v[t], with a reverse scan added so every position in an NCM level
    can use context from both sides while the whole level is predicted at once.
    A chunked closed-form scan avoids a Python loop over every position and
    never constructs a time-by-time attention matrix. Channel mixing is handled
    by the single shared sparse expert pool; both sublayers use pre-RMSNorm.
    """

    # NCM's normal code levels are short (often <=128 positions). Larger
    # chunks reduce Python/Tensor-op launch overhead; long contexts still use
    # bounded chunks to keep the closed-form recurrence numerically stable.
    _SCAN_CHUNK = 128

    def __init__(
        self, dimension: int, forward_quant: str = "fp32",
        backward_mode: str = "fp32",
    ):
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
        # softplus(-4) starts near a 0.018 decay rate, retaining useful state.
        # The bounded rate keeps the vectorized scan numerically stable.
        self.decay_logit = nn.Parameter(torch.full((dimension,), -4.0))
        self.bonus = nn.Parameter(torch.ones(dimension))

    def _scan(self, values: Tensor, decay_rate: Tensor) -> Tensor:
        """Evaluate the recurrent state with stable, bounded-size vectorized scans."""
        if values.shape[1] == 0:
            return values
        values_fp32 = values.float()
        rate = decay_rate.float()
        state = values_fp32.new_zeros(values.shape[0], values.shape[2])
        output_chunks = []
        for start in range(0, values.shape[1], self._SCAN_CHUNK):
            chunk = values_fp32[:, start:start + self._SCAN_CHUNK]
            steps = torch.arange(
                1, chunk.shape[1] + 1, dtype=torch.float32, device=values.device
            ).unsqueeze(1)
            powers = torch.exp(-steps * rate.unsqueeze(0))
            weighted_prefix = (chunk / powers.unsqueeze(0)).cumsum(dim=1)
            scanned = weighted_prefix * powers.unsqueeze(0)
            scanned = scanned + powers.unsqueeze(0) * state.unsqueeze(1)
            state = scanned[:, -1, :]
            output_chunks.append(scanned)
        return torch.cat(output_chunks, dim=1).to(dtype=values.dtype)

    def forward(
        self, x: Tensor, segment_lengths: Sequence[int],
        channel_mixer: nn.Module,
    ) -> Tensor:
        if sum(segment_lengths) != x.shape[1] or any(length < 1 for length in segment_lengths):
            raise ValueError("segment_lengths must be positive and sum to the sequence length")

        normalized = self.norm_attention(x)
        read_gate = torch.sigmoid(self.W_r(normalized))
        keys = self.W_k(normalized)
        values = self.W_v(normalized)
        key_value = keys * values
        decay_rate = F.softplus(self.decay_logit).clamp(min=1e-4, max=0.1)

        # Reset recurrent state at each independent NCM level/prompt segment.
        contexts = []
        offset = 0
        for length in segment_lengths:
            segment = key_value[:, offset:offset + length]
            # One scan over [forward; reversed] halves tensor-op launches.
            batch = segment.shape[0]
            both = self._scan(torch.cat((segment, segment.flip(dims=(1,))), dim=0), decay_rate)
            contexts.append(0.5 * (both[:batch] + both[batch:].flip(dims=(1,))))
            offset += length
        context = torch.cat(contexts, dim=1)
        attended = self.W_o(read_gate * (context + self.bonus * key_value))
        x = x + attended

        channel_input = self.norm_channel(x)
        return x + channel_mixer(channel_input)


class SmaulDenseFFN(nn.Module):
    """SwiGLU channel mixer used by the dense presets."""

    def __init__(
        self, dimension: int, hidden_dimension: int,
        forward_quant: str = "fp32", backward_mode: str = "fp32",
    ):
        super().__init__()
        self.W_a = NCMLinear(
            dimension, hidden_dimension, forward_quant=forward_quant,
            backward_mode=backward_mode,
        )
        self.W_b = NCMLinear(
            dimension, hidden_dimension, forward_quant=forward_quant,
            backward_mode=backward_mode,
        )
        self.W_c = NCMLinear(
            hidden_dimension, dimension, forward_quant=forward_quant,
            backward_mode=backward_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.W_c(F.silu(self.W_a(x)) * self.W_b(x))


class ResidualVQTokenizer(nn.Module):
    """Byte-level multiscale residual VQ tokenizer with a reconstruction loss.

    Each level has a separate codebook and a longer sequence than the previous
    level. Code vectors are upsampled and accumulated before the next residual
    is quantized. This is a compact baseline, not the paper's full VQ-VAE.
    """

    def __init__(self, config: NCMConfig):
        super().__init__()
        self.config = config
        self.byte_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = nn.Sequential(
            NCMLinear(
                config.d_model, config.d_model,
                forward_quant=config.forward_quant,
                backward_mode=config.backward_mode,
            ),
            nn.GELU(),
            NCMLinear(
                config.d_model, config.d_model,
                forward_quant=config.forward_quant,
                backward_mode=config.backward_mode,
            ),
        )
        # Keep latent RMS bounded as the encoder learns; residual codebooks and
        # the reconstruction head should not see unbounded latent magnitudes.
        self.feature_norm = RMSNorm(config.d_model)
        self.codebooks = nn.ModuleList(
            nn.Embedding(config.codebook_size, config.d_model)
            for _ in range(config.levels)
        )
        self.register_buffer(
            "_codebook_initialized", torch.zeros(config.levels, dtype=torch.bool)
        )
        self.decoder_norm = RMSNorm(config.d_model)
        self.decoder = nn.Sequential(
            NCMLinear(
                config.d_model, config.d_model,
                forward_quant=config.forward_quant,
                backward_mode=config.backward_mode,
            ),
            nn.GELU(),
            NCMLinear(
                config.d_model, config.vocab_size,
                forward_quant=config.forward_quant,
                backward_mode=config.backward_mode,
            ),
        )
        self.last_loss_components: dict[str, Tensor] = {}

    def _quantize(self, residual: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        codebook = self.codebooks[level].weight
        if self.training and not bool(self._codebook_initialized[level]):
            # Seed code vectors from actual residuals to reduce early codebook collapse.
            with torch.no_grad():
                samples = residual.detach().reshape(-1, residual.shape[-1])
                choices = torch.randint(
                    samples.shape[0], (self.config.codebook_size,), device=samples.device
                )
                codebook.copy_(samples[choices].to(codebook.dtype))
                self._codebook_initialized[level] = True
        flat = residual.reshape(-1, residual.shape[-1]).float()
        weights = codebook.float()
        distances = (
            flat.square().sum(1, keepdim=True) - 2 * flat @ weights.t()
            + weights.square().sum(1).unsqueeze(0)
        )
        ids = distances.argmin(dim=-1).view(residual.shape[:2])
        quantized = self.codebooks[level](ids)
        vq_loss = F.mse_loss(quantized, residual.detach())
        commitment = F.mse_loss(residual, quantized.detach())
        straight_through = residual + (quantized - residual).detach()
        return ids, straight_through, vq_loss + self.config.commitment_weight * commitment

    def forward(self, byte_ids: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time]")
        if not 1 <= byte_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("sequence length must be within [1, max_seq_len]")
        features = self.feature_norm(self.encoder(self.byte_embedding(byte_ids)))
        cumulative = features.new_zeros(features.shape)
        codes: list[Tensor] = []
        vq_loss = features.new_zeros(())
        for level in range(self.config.levels):
            size = level_length(byte_ids.shape[1], level, self.config.levels)
            target = F.adaptive_avg_pool1d(
                features.transpose(1, 2), size
            ).transpose(1, 2)
            prior = resize_sequence(cumulative, size)
            ids, quantized, level_loss = self._quantize(target - prior, level)
            codes.append(ids)
            cumulative = cumulative + resize_sequence(quantized, byte_ids.shape[1])
            vq_loss = vq_loss + level_loss
        logits = self.decoder(self.decoder_norm(cumulative))
        reconstruction = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), byte_ids.reshape(-1)
        )
        mean_vq_loss = vq_loss / self.config.levels
        total_loss = reconstruction + mean_vq_loss
        # Detached diagnostics let the trainer report each term without holding
        # extra autograd graphs or synchronizing a GPU on every training step.
        self.last_loss_components = {
            "reconstruction": reconstruction.detach(),
            "quantization": mean_vq_loss.detach(),
            "total": total_loss.detach(),
        }
        return logits, total_loss, codes

    @torch.no_grad()
    def encode(self, byte_ids: Tensor) -> list[Tensor]:
        was_training = self.training
        self.eval()
        _, _, codes = self.forward(byte_ids)
        self.train(was_training)
        return codes

    @torch.no_grad()
    def decode(self, codes: Sequence[Tensor], output_length: int) -> Tensor:
        if len(codes) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        if not 1 <= output_length <= self.config.max_seq_len:
            raise ValueError("output_length must be within [1, max_seq_len]")
        batch = codes[0].shape[0]
        cumulative = self.byte_embedding.weight.new_zeros(
            batch, output_length, self.config.d_model
        )
        for level, ids in enumerate(codes):
            expected = level_length(output_length, level, self.config.levels)
            if ids.shape != (batch, expected):
                raise ValueError(f"level {level} codes must have shape [batch, {expected}]")
            cumulative = cumulative + resize_sequence(self.codebooks[level](ids), output_length)
        return self.decoder(self.decoder_norm(cumulative)).argmax(dim=-1)


class NCMGenerator(nn.Module):
    """Predict every code level in one training pass; generate one level per step.

    Inputs are cumulative reconstructions of *earlier* levels, not the target
    level's ground-truth IDs. Target codes are labels only. Independent
    recurrent scans keep levels isolated while allowing all positions within
    each level to be predicted in parallel.
    """

    def __init__(self, config: NCMConfig, tokenizer: ResidualVQTokenizer):
        super().__init__()
        if tokenizer.config != config:
            raise ValueError("tokenizer and generator must use the same NCMConfig")
        self.config = config
        self._tokenizer = (tokenizer,)
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        tokenizer.eval()
        linear_args = {
            "forward_quant": config.forward_quant,
            "backward_mode": config.backward_mode,
        }
        self.input_projection = NCMLinear(
            config.d_model, config.d_model, **linear_args
        )
        self.prefix_projection = NCMLinear(
            config.d_model, config.d_model, **linear_args
        )
        self.level_embeddings = nn.Embedding(config.levels, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        if config.model_kind == "moe":
            self.channel_mixer = SmaulSharedMoE(
                config.d_model, config.n_experts, config.top_k,
                config.expert_hidden, config.forward_quant, config.backward_mode,
            )
            self.dense_mixers = None
        else:
            self.channel_mixer = None
            self.dense_mixers = nn.ModuleList(
                SmaulDenseFFN(
                    config.d_model, config.ffn_hidden,
                    config.forward_quant, config.backward_mode,
                )
                for _ in range(config.n_layers)
            )
        self.blocks = nn.ModuleList(
            SmaulRecurrentBlock(
                config.d_model, config.forward_quant, config.backward_mode
            )
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.output_heads = nn.ModuleList(
            NCMLinear(config.d_model, config.codebook_size, **linear_args)
            for _ in range(config.levels)
        )

    @property
    def tokenizer(self) -> ResidualVQTokenizer:
        return self._tokenizer[0]

    def _build_level_inputs(
        self, codes: Sequence[Tensor], corrupt: bool
    ) -> tuple[Tensor, list[int]]:
        if len(codes) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        batch = codes[0].shape[0]
        length = codes[-1].shape[1]
        if length > self.config.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")
        device = codes[0].device
        cumulative = self.tokenizer.byte_embedding.weight.new_zeros(
            batch, length, self.config.d_model, device=device
        )
        parts: list[Tensor] = []
        lengths: list[int] = []
        for level, ids in enumerate(codes):
            expected = level_length(length, level, self.config.levels)
            if ids.shape != (batch, expected):
                raise ValueError(f"level {level} codes must have shape [batch, {expected}]")
            lengths.append(expected)
            # The current level's input contains only earlier levels.
            current = resize_sequence(cumulative, expected, mode="area")
            positions = torch.arange(expected, device=device)
            part = self.input_projection(current)
            part = part + self.position_embeddings(positions).unsqueeze(0)
            part = part + self.level_embeddings.weight[level].view(1, 1, -1)
            parts.append(part)
            # Corrupt conditioning IDs only; the original IDs remain the labels.
            conditioning_ids = ids
            if corrupt and self.training and self.config.conditioning_corruption > 0:
                replace = torch.rand(ids.shape, device=ids.device) < self.config.conditioning_corruption
                random_ids = torch.randint(
                    self.config.codebook_size, ids.shape, device=ids.device
                )
                conditioning_ids = torch.where(replace, random_ids, ids)
            quantized = self.tokenizer.codebooks[level](conditioning_ids)
            cumulative = cumulative + resize_sequence(quantized, length)
        return torch.cat(parts, dim=1), lengths

    def forward(
        self, codes: Sequence[Tensor], *, prefix_tokens: Tensor | None = None,
        corrupt: bool = True,
    ) -> list[Tensor]:
        inputs, lengths = self._build_level_inputs(codes, corrupt)
        level_inputs = list(torch.split(inputs, lengths, dim=1))
        prefix = None
        prefix_length = 0
        if prefix_tokens is not None:
            if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != inputs.shape[0]:
                raise ValueError("prefix_tokens must have shape [batch, prefix_length]")
            if not 1 <= prefix_tokens.shape[1] <= self.config.max_seq_len:
                raise ValueError("prefix length must be within [1, max_seq_len]")
            prefix_features = self.tokenizer.feature_norm(
                self.tokenizer.encoder(self.tokenizer.byte_embedding(prefix_tokens))
            )
            prefix_positions = torch.arange(prefix_tokens.shape[1], device=inputs.device)
            prefix = self.prefix_projection(prefix_features)
            prefix = prefix + self.position_embeddings(prefix_positions).unsqueeze(0)
            prefix_length = prefix.shape[1]

        # Each level gets its own recurrent state. When conditioning on a prompt,
        # repeat the known prefix for each level so no level can leak into another.
        segments = [
            torch.cat((prefix, part), dim=1) if prefix is not None else part
            for part in level_inputs
        ]
        segment_lengths = [part.shape[1] for part in segments]
        hidden = torch.cat(segments, dim=1)
        for layer_index, block in enumerate(self.blocks):
            mixer = (
                self.channel_mixer
                if self.config.model_kind == "moe"
                else self.dense_mixers[layer_index]
            )
            hidden = block(hidden, segment_lengths, mixer)
        hidden = self.final_norm(hidden)

        encoded_segments = torch.split(hidden, segment_lengths, dim=1)
        logits: list[Tensor] = []
        for level, encoded in enumerate(encoded_segments):
            if prefix is not None:
                encoded = encoded[:, prefix_length:]
            logits.append(self.output_heads[level](encoded))
        return logits

    def predict_level(
        self, codes: Sequence[Tensor], level: int,
        *, prefix_tokens: Tensor | None = None,
    ) -> Tensor:
        """Predict only one level, avoiding work on unrelated levels at inference."""
        if not 0 <= level < self.config.levels:
            raise ValueError("level is outside the configured hierarchy")
        if len(codes) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        batch = codes[0].shape[0]
        length = codes[-1].shape[1]
        if length > self.config.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")
        device = codes[0].device
        cumulative = self.tokenizer.byte_embedding.weight.new_zeros(
            batch, length, self.config.d_model, device=device
        )
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
        current = resize_sequence(
            cumulative, codes[level].shape[1], mode="area"
        )
        current = self.input_projection(current)
        positions = torch.arange(current.shape[1], device=device)
        current = current + self.position_embeddings(positions).unsqueeze(0)
        current = current + self.level_embeddings.weight[level].view(1, 1, -1)
        prefix_length = 0
        if prefix_tokens is None:
            hidden = current
        else:
            if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != batch:
                raise ValueError("prefix_tokens must have shape [batch, prefix_length]")
            if not 1 <= prefix_tokens.shape[1] <= self.config.max_seq_len:
                raise ValueError("prefix length must be within [1, max_seq_len]")
            prefix_features = self.tokenizer.feature_norm(
                self.tokenizer.encoder(self.tokenizer.byte_embedding(prefix_tokens))
            )
            prefix_positions = torch.arange(prefix_tokens.shape[1], device=device)
            prefix = self.prefix_projection(prefix_features)
            prefix = prefix + self.position_embeddings(prefix_positions).unsqueeze(0)
            prefix_length = prefix.shape[1]
            hidden = torch.cat((prefix, current), dim=1)
        segment_lengths = [hidden.shape[1]]
        for layer_index, block in enumerate(self.blocks):
            mixer = (
                self.channel_mixer
                if self.config.model_kind == "moe"
                else self.dense_mixers[layer_index]
            )
            hidden = block(hidden, segment_lengths, mixer)
        hidden = self.final_norm(hidden)
        if prefix_length:
            hidden = hidden[:, prefix_length:]
        return self.output_heads[level](hidden)

    @torch.no_grad()
    def generate_codes(
        self, lengths: Sequence[int], device: torch.device | str = "cpu",
        temperature: float = 0.0, top_k: int = 0,
        prefix_tokens: Tensor | None = None, top_p: float = 1.0,
    ) -> list[Tensor]:
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
        if prefix_tokens is not None:
            device = prefix_tokens.device
        codes = [
            torch.zeros(batch_size, n, dtype=torch.long, device=device) for n in lengths
        ]
        for level in range(self.config.levels):
            logits = self.predict_level(codes, level, prefix_tokens=prefix_tokens)
            if temperature == 0:
                codes[level] = logits.argmax(dim=-1)
            else:
                if top_k:
                    k = min(top_k, logits.shape[-1])
                    threshold = logits.topk(k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                    remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(-1, sorted_indices, sorted_probs)
                    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                codes[level] = torch.multinomial(
                    probs.reshape(-1, probs.shape[-1]), 1
                ).reshape(probs.shape[:-1])
        return codes

    @torch.no_grad()
    def generate(
        self, output_length: int, device: torch.device | str = "cpu",
        temperature: float = 0.0, top_k: int = 0,
        prefix_tokens: Tensor | None = None, top_p: float = 1.0,
    ) -> Tensor:
        lengths = [
            level_length(output_length, level, self.config.levels)
            for level in range(self.config.levels)
        ]
        codes = self.generate_codes(
            lengths, device, temperature, top_k, prefix_tokens, top_p=top_p
        )
        return self.tokenizer.decode(codes, output_length)


# Compatibility alias for existing scripts and user code that imported the old name.
# The implementation below is recurrent and contains no TransformerEncoder layers.
NCMTransformer = NCMGenerator


def ncm_loss(
    model: NCMGenerator, codes: Sequence[Tensor],
    *, prefix_tokens: Tensor | None = None,
) -> Tensor:
    """Length-normalized multilevel CE; alpha=1 weights each level equally."""
    logits = model(codes, prefix_tokens=prefix_tokens, corrupt=True)
    losses = [
        F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
        for pred, target in zip(logits, codes)
    ]
    alpha = model.config.level_loss_alpha
    weights = torch.tensor(
        [max(1, target.shape[1]) ** (1.0 - alpha) for target in codes],
        device=logits[0].device, dtype=logits[0].dtype,
    )
    weights = weights / weights.sum()
    return sum(weight * loss for weight, loss in zip(weights, losses))
