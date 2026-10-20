"""Experimental coarse-to-fine Next-Concept Modeling for SmaulBRAIN.

The default SmaulBRAIN model is not changed. Train ResidualVQTokenizer first,
freeze it, then train NCMGenerator against its code hierarchies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.codebook_size < 2:
            raise ValueError("vocab_size and codebook_size must be >= 2")
        if min(self.d_model, self.levels, self.n_layers, self.n_heads, self.n_experts) < 1:
            raise ValueError("model dimensions, levels, and experts must be positive")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        # n_heads remains accepted for older CLI/checkpoint configs. The
        # recurrent NCM generator does not split channels into attention heads.
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.conditioning_corruption <= 1:
            raise ValueError("conditioning_corruption must be in [0, 1]")
        if self.level_loss_alpha < 0:
            raise ValueError("level_loss_alpha must be >= 0")


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

    def __init__(self, dimension: int, hidden_dimension: int):
        super().__init__()
        self.W_in = nn.Linear(dimension, hidden_dimension, bias=False)
        self.W_out = nn.Linear(hidden_dimension, dimension, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_out(F.gelu(self.W_in(x)))


class SmaulSharedMoE(nn.Module):
    """One sparse router and expert pool shared by every recurrent block."""

    def __init__(self, dimension: int, n_experts: int, top_k: int):
        super().__init__()
        if n_experts < 1 or not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dimension, n_experts, bias=False)
        self.experts = nn.ModuleList(
            SmaulExpert(dimension, 4 * dimension) for _ in range(n_experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        route_logits = self.router(flat)
        chosen_logits, chosen_experts = route_logits.topk(self.top_k, dim=-1)
        route_weights = torch.softmax(chosen_logits, dim=-1)
        output = torch.zeros_like(flat)

        # Only run an expert for tokens routed to it; the pool itself is shared
        # by all NCM levels and all recurrent layers.
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

    _SCAN_CHUNK = 32

    def __init__(self, dimension: int):
        super().__init__()
        self.norm_attention = RMSNorm(dimension)
        self.norm_channel = RMSNorm(dimension)
        self.W_r = nn.Linear(dimension, dimension, bias=False)
        self.W_k = nn.Linear(dimension, dimension, bias=False)
        self.W_v = nn.Linear(dimension, dimension, bias=False)
        self.W_o = nn.Linear(dimension, dimension, bias=False)
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
        channel_mixer: SmaulSharedMoE,
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
            forward_state = self._scan(segment, decay_rate)
            reverse_state = self._scan(segment.flip(dims=(1,)), decay_rate).flip(dims=(1,))
            contexts.append(0.5 * (forward_state + reverse_state))
            offset += length
        context = torch.cat(contexts, dim=1)
        attended = self.W_o(read_gate * (context + self.bonus * key_value))
        x = x + attended

        channel_input = self.norm_channel(x)
        return x + channel_mixer(channel_input)


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
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.codebooks = nn.ModuleList(
            nn.Embedding(config.codebook_size, config.d_model)
            for _ in range(config.levels)
        )
        self.register_buffer(
            "_codebook_initialized", torch.zeros(config.levels, dtype=torch.bool)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size),
        )

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
        features = self.encoder(self.byte_embedding(byte_ids))
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
        logits = self.decoder(cumulative)
        reconstruction = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), byte_ids.reshape(-1)
        )
        return logits, reconstruction + vq_loss / self.config.levels, codes

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
        return self.decoder(cumulative).argmax(dim=-1)


class NCMGenerator(nn.Module):
    """Predict every code level in one training pass; generate one level per step.

    Inputs are cumulative reconstructions of *earlier* levels, not the target
    level's ground-truth IDs. Target codes are labels only. Block-diagonal
    attention lets positions interact within their level; each level's input
    already summarizes all coarser reconstructions.
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
        self.input_projection = nn.Linear(config.d_model, config.d_model)
        self.prefix_projection = nn.Linear(config.d_model, config.d_model)
        self.level_embeddings = nn.Embedding(config.levels, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        # A single router/expert pool is shared across every recurrent layer,
        # matching the shared-pool option in the normal SmaulBRAIN MoE design.
        self.channel_mixer = SmaulSharedMoE(
            config.d_model, config.n_experts, config.top_k
        )
        self.blocks = nn.ModuleList(
            SmaulRecurrentBlock(config.d_model) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.output_heads = nn.ModuleList(
            nn.Linear(config.d_model, config.codebook_size)
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
            prefix_features = self.tokenizer.encoder(
                self.tokenizer.byte_embedding(prefix_tokens)
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
        for block in self.blocks:
            hidden = block(hidden, segment_lengths, self.channel_mixer)
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
            prefix_features = self.tokenizer.encoder(
                self.tokenizer.byte_embedding(prefix_tokens)
            )
            prefix_positions = torch.arange(prefix_tokens.shape[1], device=device)
            prefix = self.prefix_projection(prefix_features)
            prefix = prefix + self.position_embeddings(prefix_positions).unsqueeze(0)
            prefix_length = prefix.shape[1]
            hidden = torch.cat((prefix, current), dim=1)
        segment_lengths = [hidden.shape[1]]
        for block in self.blocks:
            hidden = block(hidden, segment_lengths, self.channel_mixer)
        hidden = self.final_norm(hidden)
        if prefix_length:
            hidden = hidden[:, prefix_length:]
        return self.output_heads[level](hidden)

    @torch.no_grad()
    def generate_codes(
        self, lengths: Sequence[int], device: torch.device | str = "cpu",
        temperature: float = 0.0, top_k: int = 0,
        prefix_tokens: Tensor | None = None,
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
                codes[level] = torch.multinomial(
                    probs.reshape(-1, probs.shape[-1]), 1
                ).reshape(probs.shape[:-1])
        return codes

    @torch.no_grad()
    def generate(
        self, output_length: int, device: torch.device | str = "cpu",
        temperature: float = 0.0, top_k: int = 0,
        prefix_tokens: Tensor | None = None,
    ) -> Tensor:
        lengths = [
            level_length(output_length, level, self.config.levels)
            for level in range(self.config.levels)
        ]
        codes = self.generate_codes(lengths, device, temperature, top_k, prefix_tokens)
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
