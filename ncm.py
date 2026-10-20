"""Experimental Next-Concept Modeling (NCM) components for SmaulBRAIN.

This is an isolated research path: the existing byte-level model and checkpoints
are unchanged. Train the multiscale tokenizer first, freeze it, then train the
coarse-to-fine code predictor. Requires PyTorch.
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

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.codebook_size < 2:
            raise ValueError("vocab_size and codebook_size must be >= 2")
        if self.d_model < 1 or self.levels < 1 or self.n_layers < 1:
            raise ValueError("d_model, levels and n_layers must be >= 1")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _level_length(length: int, level: int, levels: int) -> int:
    """Length at this scale, ordered from coarsest level to finest."""
    divisor = 1 << (levels - level - 1)
    return max(1, (length + divisor - 1) // divisor)


def _resize_sequence(x: Tensor, length: int) -> Tensor:
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear",
                         align_corners=False).transpose(1, 2)


class ResidualVQTokenizer(nn.Module):
    """Byte-to-multiscale-code tokenizer with residual vector quantization.

    Each successive level has a longer sequence. The decoder combines the
    upsampled quantized residuals to reconstruct the original byte sequence.
    This deliberately small implementation is a baseline, not a reproduction
    of every optimization in the published NCM system.
    """

    def __init__(self, config: NCMConfig):
        super().__init__()
        self.config = config
        self.byte_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.codebooks = nn.ModuleList(
            nn.Embedding(config.codebook_size, config.d_model)
            for _ in range(config.levels)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def _quantize(self, residual: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        # Compute distances in float32 for stable nearest-code selection.
        codebook = self.codebooks[level].weight
        flat = residual.reshape(-1, residual.shape[-1]).float()
        weights = codebook.float()
        distances = (flat.square().sum(1, keepdim=True)
                     - 2.0 * flat @ weights.t()
                     + weights.square().sum(1).unsqueeze(0))
        ids = distances.argmin(dim=-1).view(residual.shape[:2])
        quantized = self.codebooks[level](ids)
        codebook_loss = F.mse_loss(quantized, residual.detach())
        commitment_loss = F.mse_loss(residual, quantized.detach())
        straight_through = residual + (quantized - residual).detach()
        return ids, straight_through, codebook_loss + self.config.commitment_weight * commitment_loss

    def encode(self, byte_ids: Tensor) -> list[Tensor]:
        """Return code IDs per level, coarsest first; input shape is [batch, time]."""
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time]")
        if byte_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")
        with torch.no_grad():
            features = self.encoder(self.byte_embedding(byte_ids))
            cumulative = features.new_zeros(features.shape[0], 1, features.shape[-1])
            codes: list[Tensor] = []
            for level in range(self.config.levels):
                size = _level_length(byte_ids.shape[1], level, self.config.levels)
                target = F.adaptive_avg_pool1d(
                    features.transpose(1, 2), size
                ).transpose(1, 2)
                prior = _resize_sequence(cumulative, size)
                ids, quantized, _ = self._quantize(target - prior, level)
                cumulative = prior + self.codebooks[level](ids)
                codes.append(ids)
            return codes

    def forward(self, byte_ids: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time]")
        if byte_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")
        features = self.encoder(self.byte_embedding(byte_ids))
        cumulative = features.new_zeros(features.shape[0], 1, features.shape[-1])
        decoded = features.new_zeros(features.shape)
        vq_loss = features.new_zeros(())
        codes: list[Tensor] = []
        for level in range(self.config.levels):
            size = _level_length(byte_ids.shape[1], level, self.config.levels)
            target = F.adaptive_avg_pool1d(
                features.transpose(1, 2), size
            ).transpose(1, 2)
            prior = _resize_sequence(cumulative, size)
            ids, quantized, level_loss = self._quantize(target - prior, level)
            cumulative = prior + quantized
            codes.append(ids)
            decoded = decoded + _resize_sequence(quantized, byte_ids.shape[1])
            vq_loss = vq_loss + level_loss
        logits = self.decoder(decoded)
        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), byte_ids.reshape(-1)
        )
        return logits, reconstruction_loss + vq_loss, codes


class NCMTransformer(nn.Module):
    """Predict all code positions in parallel, conditioned on coarser levels.

    Training computes every level in one transformer invocation. The attention
    mask prevents a level's targets from leaking into its own predictions:
    query positions can see code embeddings from strictly coarser levels only.
    During generation, fill levels from coarse to fine and use predict().
    """

    def __init__(self, config: NCMConfig):
        super().__init__()
        self.config = config
        self.code_embeddings = nn.ModuleList(
            nn.Embedding(config.codebook_size, config.d_model)
            for _ in range(config.levels)
        )
        self.level_embeddings = nn.Embedding(config.levels, config.d_model)
        self.role_embeddings = nn.Embedding(2, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers,
                                                  enable_nested_tensor=False)
        self.output_heads = nn.ModuleList(
            nn.Linear(config.d_model, config.codebook_size)
            for _ in range(config.levels)
        )
        self.mask_embedding = nn.Parameter(torch.zeros(config.d_model))

    def _packed(self, code_ids: Sequence[Tensor]) -> tuple[Tensor, Tensor, list[int]]:
        if len(code_ids) != self.config.levels:
            raise ValueError("one code tensor is required per level")
        batch = code_ids[0].shape[0]
        device = code_ids[0].device
        context_parts: list[Tensor] = []
        query_parts: list[Tensor] = []
        level_parts: list[Tensor] = []
        lengths: list[int] = []
        for level, ids in enumerate(code_ids):
            if ids.ndim != 2 or ids.shape[0] != batch:
                raise ValueError("each code tensor must have shape [batch, level_length]")
            length = ids.shape[1]
            if length > self.config.max_seq_len:
                raise ValueError("code sequence exceeds config.max_seq_len")
            lengths.append(length)
            positions = torch.arange(length, device=device)
            pos = self.position_embeddings(positions).unsqueeze(0)
            lev = self.level_embeddings.weight[level].view(1, 1, -1)
            context_parts.append(self.code_embeddings[level](ids) + pos + lev
                                 + self.role_embeddings.weight[0].view(1, 1, -1))
            query_parts.append(self.mask_embedding.view(1, 1, -1) + pos + lev
                               + self.role_embeddings.weight[1].view(1, 1, -1))
            level_parts.append(torch.full((length,), level, device=device, dtype=torch.long))
        context = torch.cat(context_parts, dim=1)
        query = torch.cat(query_parts, dim=1)
        level_ids = torch.cat(level_parts)
        total = level_ids.numel()
        # Context entries are kept as isolated code embeddings. Query positions
        # attend only to context positions from strictly coarser levels.
        mask = torch.ones((2 * total, 2 * total), device=device, dtype=torch.bool)
        diag = torch.arange(total, device=device)
        mask[diag, diag] = False
        q_rows = total + diag
        allowed = level_ids.unsqueeze(0) < level_ids.unsqueeze(1)
        mask[q_rows.unsqueeze(1), diag.unsqueeze(0)] = ~allowed
        # Query tokens are masked placeholders, so positions at the same level
        # may exchange information without seeing their target code IDs.
        query_levels = level_ids.unsqueeze(1)
        key_levels = level_ids.unsqueeze(0)
        mask[total:, total:] = query_levels != key_levels
        return torch.cat((context, query), dim=1), mask, lengths

    def forward(self, code_ids: Sequence[Tensor]) -> list[Tensor]:
        packed, mask, lengths = self._packed(code_ids)
        hidden = self.transformer(packed, mask=mask)
        offset = sum(lengths)
        logits: list[Tensor] = []
        for level, length in enumerate(lengths):
            logits.append(self.output_heads[level](hidden[:, offset:offset + length]))
            offset += length
        return logits

    @torch.no_grad()
    def predict(self, code_ids: Sequence[Tensor], level: int) -> Tensor:
        """Predict all positions of one level simultaneously."""
        if not 0 <= level < self.config.levels:
            raise ValueError("level is outside the configured hierarchy")
        return self.forward(code_ids)[level].argmax(dim=-1)

    @torch.no_grad()
    def generate_codes(self, lengths: Sequence[int], device: torch.device | str = "cpu",
                       temperature: float = 0.0) -> list[Tensor]:
        """Generate a complete hierarchy, one level per sequential step."""
        if len(lengths) != self.config.levels:
            raise ValueError("one length is required per level")
        if any(n < 1 or n > self.config.max_seq_len for n in lengths):
            raise ValueError("each level length must be within [1, max_seq_len]")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        codes = [torch.zeros(1, n, dtype=torch.long, device=device) for n in lengths]
        for level in range(self.config.levels):
            logits = self.forward(codes)[level]
            if temperature == 0:
                codes[level] = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                codes[level] = torch.multinomial(
                    probs.reshape(-1, probs.shape[-1]), 1
                ).reshape(probs.shape[:-1])
        return codes


def ncm_loss(model: NCMTransformer, code_ids: Sequence[Tensor]) -> Tensor:
    """Mean cross-entropy across all hierarchy levels and positions."""
    logits = model(code_ids)
    if len(logits) != len(code_ids):
        raise RuntimeError("model returned an incomplete hierarchy")
    losses = [
        F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
        for pred, target in zip(logits, code_ids)
    ]
    return torch.stack(losses).mean()
