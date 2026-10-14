"""Opt-in full-model ternary quantization-aware training (QAT).

Every trainable affine weight is presented to the forward pass as row-scaled
{-1, 0, +1} values. Full-precision/BF16 master parameters, gradients, optimizer
state, normalization statistics, activations, and loss remain continuous.
This is QAT, not a claim that every operation in training is integer/additive.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experts import _TernaryQATLinear, ternary_qat_weight
from precision import HybridFP8Linear
from rmsnorm import RMSNorm


class TernaryQATLinear(nn.Module):
    """Linear layer with ternary weights, STE master weights, and optional bias."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        self.weight = source.weight
        self.bias = source.bias
        self.out_dtype = getattr(source, "out_dtype", source.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HybridFP8Linear used FP32 arithmetic around low-precision operands.
        # Match that contract here so weight/activation dtypes always agree.
        y = _TernaryQATLinear.apply(x.float(), self.weight.float(), True)
        if self.bias is not None:
            bias_q = ternary_qat_weight(self.bias.float().reshape(1, -1)).reshape(-1)
            y = y + bias_q
        return y.to(self.out_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, weight_mode=scaled-ternary-QAT")


class TernaryQATEmbedding(nn.Module):
    """Embedding lookup which ternarizes only selected rows (not the whole table)."""

    def __init__(self, source: nn.Embedding) -> None:
        super().__init__()
        if source.max_norm is not None or source.scale_grad_by_freq or source.sparse:
            raise ValueError(
                "full ternary QAT currently requires Embedding max_norm=None, "
                "scale_grad_by_freq=False, sparse=False"
            )
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.weight = source.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        selected = self.weight[ids.long()]
        quantized = ternary_qat_weight(selected)
        if self.padding_idx is not None:
            quantized = quantized.masked_fill((ids == self.padding_idx).unsqueeze(-1), 0)
        return quantized

    def extra_repr(self) -> str:
        return (f"num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim}, weight_mode=scaled-ternary-QAT")


class TernaryQATConv1d(nn.Module):
    """Conv1d with each output filter flattened and quantized to scaled trits."""

    def __init__(self, source: nn.Conv1d) -> None:
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self.weight = source.weight
        self.bias = source.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_q = ternary_qat_weight(
            self.weight.reshape(self.out_channels, -1)
        ).reshape_as(self.weight)
        bias_q = (ternary_qat_weight(self.bias.float().reshape(1, -1)).reshape(-1).to(x.dtype)
                  if self.bias is not None else None)
        padding = self.padding
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0
        return F.conv1d(x, weight_q, bias_q, self.stride, padding,
                        self.dilation, self.groups)

    @property
    def _reversed_padding_repeated_twice(self):
        # Matches torch.nn.Conv1d's padding-mode convention.
        values = self.padding if isinstance(self.padding, tuple) else (self.padding,)
        return tuple(v for p in reversed(values) for v in (p, p))


class TernaryQATRMSNorm(nn.Module):
    """RMSNorm whose learned gain is also constrained during the forward pass."""

    def __init__(self, source: RMSNorm) -> None:
        super().__init__()
        self.eps = source.eps
        self.weight = source.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        gain_q = ternary_qat_weight(self.weight.float().reshape(1, -1)).reshape(-1)
        var = xf.square().mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.eps) * gain_q).to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}, weight_mode=scaled-ternary-QAT"


def enable_full_ternary_qat(module: nn.Module) -> nn.Module:
    """Replace all standard trainable model layers with ternary-QAT equivalents.

    ExpertPool is an ordinary Python object rather than an nn.Module; its expert
    weights already use the ternary-aware expert path when the config flag is on.
    Parameter names and tensor shapes are preserved for checkpoint/optimizer use.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (TernaryQATLinear, TernaryQATEmbedding,
                              TernaryQATConv1d, TernaryQATRMSNorm)):
            continue
        if isinstance(child, (HybridFP8Linear, nn.Linear)):
            replacement = TernaryQATLinear(child)
        elif isinstance(child, nn.Embedding):
            replacement = TernaryQATEmbedding(child)
        elif isinstance(child, nn.Conv1d):
            replacement = TernaryQATConv1d(child)
        elif isinstance(child, RMSNorm):
            replacement = TernaryQATRMSNorm(child)
        else:
            enable_full_ternary_qat(child)
            continue
        setattr(module, name, replacement)
    return module


def clear_ternary_qat_caches(module: nn.Module) -> None:
    """Invalidate per-parameter packed/quantized caches at each model forward.

    The optimizer updates master Parameters in-place. Clearing at forward entry
    guarantees the subsequent recurrent applications reuse fresh weights even
    for optimizers whose .data updates do not advance Tensor._version.
    """
    for parameter in module.parameters():
        for attribute in ("_smaul_ternary_qat_cache",
                          "_smaul_packed_ternary_qat_cache"):
            if hasattr(parameter, attribute):
                delattr(parameter, attribute)
