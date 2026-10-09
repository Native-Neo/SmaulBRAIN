"""
SmaulBRAIN — Recurrent Core Modules
===================================
Contains recurrent components including:
- CausalByteConv: Local 1D depthwise convolution for byte n-gram patterns to lower BPB
- SmaulRecurrentLayer: Recurrent update cell with local gating
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalByteConv(nn.Module):
    """
    Causal 1D Depthwise Convolution over raw byte embeddings.
    Extracts local n-gram byte contexts (such as multi-byte UTF-8 prefixes)
    before feeding into the recurrent memory layers to lower BPB.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        x_padded = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        out = self.conv(x_padded).transpose(1, 2)
        return out


class SmaulRecurrentCell(nn.Module):
    """
    Recurrent cell with local byte convolution and dynamic gating.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.byte_conv = CausalByteConv(d_model=d_model, kernel_size=kernel_size)
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.update_linear = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        if state is None:
            state = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)

        conv_x = self.byte_conv(x)
        outputs = []

        for t in range(seq_len):
            x_t = conv_x[:, t, :]
            combined = torch.cat([x_t, state], dim=-1)
            gate = torch.sigmoid(self.gate_linear(combined))
            update = torch.tanh(self.update_linear(combined))
            state = (1 - gate) * state + gate * update
            outputs.append(state)

        out = torch.stack(outputs, dim=1)
        out = self.norm(out)
        return out, state
