"""
SmaulBRAIN — Model Architecture & Multi-Task Loss
=================================================
Model architecture with multi-step lookahead and boundary heads
for byte-level language modeling with optimized BPB.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recurrent import SmaulRecurrentCell

class SmaulBRAINModel(nn.Module):
    """
    Byte-level neural language model optimized for low Bits Per Byte (BPB).
    Integrates local byte n-gram convolutions, recurrent states, multi-step lookahead,
    and UTF-8 character boundary classification.
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 512, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SmaulRecurrentCell(d_model=d_model, kernel_size=4)
            for _ in range(num_layers)
        ])

        # Primary and auxiliary prediction heads
        self.head_next = nn.Linear(d_model, vocab_size, bias=False)
        self.head_lookahead = nn.Linear(d_model, vocab_size, bias=False)
        self.head_boundary = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor):
        x = self.byte_embedding(input_ids)
        for layer in self.layers:
            x, _ = layer(x)

        logits_next = self.head_next(x)
        logits_lookahead = self.head_lookahead(x)
        logits_boundary = self.head_boundary(x)

        return logits_next, logits_lookahead, logits_boundary


def compute_smaulbrain_loss(logits_next, logits_lookahead, logits_boundary, targets, alpha=0.15, beta=0.05):
    """
    Computes primary cross-entropy loss along with auxiliary multi-step lookahead
    and UTF-8 boundary losses, returning the exact Bits Per Byte (BPB).
    """
    # Primary next-byte cross-entropy loss (in nats)
    loss_next = F.cross_entropy(logits_next.view(-1, 256), targets.view(-1))

    # Multi-step lookahead loss (predicting t+2)
    targets_lookahead = targets[:, 1:]
    logits_lookahead_cut = logits_lookahead[:, :-1]
    loss_lookahead = F.cross_entropy(logits_lookahead_cut.reshape(-1, 256), targets_lookahead.reshape(-1))

    # UTF-8 character boundary classification loss
    # Bytes starting with (byte & 0xC0) != 0x80 represent ASCII or UTF-8 lead bytes
    utf8_boundaries = ((targets & 0xC0) != 0x80).float().unsqueeze(-1)
    loss_boundary = F.binary_cross_entropy_with_logits(logits_boundary, utf8_boundaries)

    total_loss = loss_next + (alpha * loss_lookahead) + (beta * loss_boundary)

    # Exact BPB calculation: Loss(nats) / ln(2)
    bpb = loss_next.item() / math.log(2)

    return total_loss, loss_next, bpb
