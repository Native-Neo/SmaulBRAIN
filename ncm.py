"""Experimental coarse-to-fine Next-Concept Modeling for SmaulBRAIN.

The default SmaulBRAIN model is not changed. Train ResidualVQTokenizer first,
freeze it, then train NCMTransformer against its code hierarchies.
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
    max_seq_len: int = 1024
    dropout: float = 0.0
    commitment_weight: float = 0.25
    conditioning_corruption: float = 0.15
    level_loss_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.codebook_size < 2:
            raise ValueError("vocab_size and codebook_size must be >= 2")
        if min(self.d_model, self.levels, self.n_layers, self.n_heads) < 1:
            raise ValueError("model dimensions and levels must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
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
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def _quantize(self, residual: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        codebook = self.codebooks[level].weight
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


class NCMTransformer(nn.Module):
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
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=4 * config.d_model, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.d_model)
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
        level_ids = torch.cat([
            torch.full((n,), i, device=inputs.device, dtype=torch.long)
            for i, n in enumerate(lengths)
        ])
        prefix = None
        if prefix_tokens is not None:
            if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != inputs.shape[0]:
                raise ValueError("prefix_tokens must have shape [batch, prefix_length]")
            if not 1 <= prefix_tokens.shape[1] <= self.config.max_seq_len:
                raise ValueError("prefix length must be within [1, max_seq_len]")
            prefix_features = self.tokenizer.encoder(
                self.tokenizer.byte_embedding(prefix_tokens)
            )
            prefix = self.prefix_projection(prefix_features)
            inputs = torch.cat((prefix, inputs), dim=1)
        total = level_ids.numel()
        target_mask = level_ids[:, None] != level_ids[None, :]
        if prefix is None:
            mask = target_mask
        else:
            prefix_length = prefix.shape[1]
            mask = torch.zeros(
                prefix_length + total, prefix_length + total,
                dtype=torch.bool, device=inputs.device,
            )
            # Prefix queries cannot see target positions.
            mask[:prefix_length, prefix_length:] = True
            # Targets can see the prefix and their own level, but not other levels.
            mask[prefix_length:, prefix_length:] = target_mask
        hidden = self.final_norm(self.blocks(inputs, mask=mask))
        if prefix is not None:
            hidden = hidden[:, prefix.shape[1]:]
        logits: list[Tensor] = []
        offset = 0
        for level, length in enumerate(lengths):
            logits.append(self.output_heads[level](hidden[:, offset:offset + length]))
            offset += length
        return logits

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
        codes = [
            torch.zeros(1, n, dtype=torch.long, device=device) for n in lengths
        ]
        for level in range(self.config.levels):
            logits = self.forward(codes, prefix_tokens=prefix_tokens, corrupt=False)[level]
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


def ncm_loss(model: NCMTransformer, codes: Sequence[Tensor]) -> Tensor:
    """Length-normalized multilevel CE; alpha=1 weights each level equally."""
    logits = model(codes, corrupt=True)
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
