"""Continual learning: replay, retention measurement, slow-trunk discipline.

Mechanisms (all real, all measured — see tests/benchmarks):
  * slow shared trunk (``trunk_lr_mult``) + fast experts: new data lands in
    routed experts while shared parameters move slowly;
  * sparse routing confines updates to the active top-k experts;
  * replay/interleaving: a reservoir buffer mixes old batches into training;
  * retention evaluation: old-data loss/accuracy before vs. after learning new
    data, reported numerically (never "solved", only reduced and measured);
  * controlled growth/pruning add capacity for new domains and retire dead
    experts (see growth.py / pruning.py).
"""

from __future__ import annotations

import random
from collections import deque

import torch


class ReplayBuffer:
    """Reservoir of past batches (byte-id sequences) for interleaving."""

    def __init__(self, capacity: int = 512, seed: int = 0) -> None:
        self.capacity = capacity
        self.buf: deque[list[int]] = deque()
        self.rng = random.Random(seed)
        self.seen = 0

    def add(self, seq: list[int]) -> None:
        self.seen += 1
        if len(self.buf) < self.capacity:
            self.buf.append(seq)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.buf[j] = seq

    def sample(self, n: int) -> list[list[int]]:
        if not self.buf:
            return []
        return [list(self.rng.choice(self.buf)) for _ in range(min(n, len(self.buf)))]

    def __len__(self) -> int:
        return len(self.buf)


def batch_from_seqs(seqs: list[list[int]], context: int, pad_id: int = 0) -> torch.Tensor:
    """Pack variable-length byte seqs into a [B, T] batch (truncate/pad)."""
    rows = []
    for s in seqs:
        s = s[: context + 1]
        if len(s) < context + 1:
            s = s + [pad_id] * (context + 1 - len(s))
        rows.append(torch.tensor(s, dtype=torch.long))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def evaluate_loss(model, batch: torch.Tensor, context: int) -> dict:
    """Old/new-data evaluation: mean NLL + accuracy (no gradients)."""
    was_training = model.training
    model.eval()
    x = batch[:, :context]
    y = batch[:, 1 : context + 1]
    out = model.forward_infer(x)
    logits = out["logits"].float()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.cfg.vocab_size),
                                             y.reshape(-1)).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    if was_training:
        model.train()
    return {"loss": loss, "acc": acc, "mean_executed": float(out["n_executed"])}


def retention_report(before: dict, after: dict) -> dict:
    """Numerically report forgetting: old-data metrics before vs. after."""
    return {
        "old_loss_before": before["loss"],
        "old_loss_after": after["loss"],
        "old_loss_delta": after["loss"] - before["loss"],
        "old_acc_before": before["acc"],
        "old_acc_after": after["acc"],
        "old_acc_delta": after["acc"] - before["acc"],
        "retained": bool(after["loss"] <= before["loss"] * 1.5),
    }
