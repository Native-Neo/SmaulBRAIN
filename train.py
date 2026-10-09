"""
SmaulBRAIN — Training Pipeline
==============================
Training loop with context length expansion, cosine decay,
and real-time Bits Per Byte (BPB) tracking.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import SmaulBRAINModel, compute_smaulbrain_loss

class ByteDataset(Dataset):
    def __init__(self, data: bytes, seq_len: int = 2048):
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Configuration
    vocab_size = 256
    d_model = 512
    num_layers = 4
    seq_len = 2048  # Extended context window to lower BPB
    batch_size = 8
    learning_rate = 3e-4
    epochs = 5

    # Sample dummy bytes dataset for training initialization
    sample_text = ("SmaulBRAIN byte-level dynamic neural network training. " * 500).encode("utf-8")
    dataset = ByteDataset(sample_text, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmaulBRAINModel(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    model.train()
    step = 0

    for epoch in range(epochs):
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits_next, logits_lookahead, logits_boundary = model(input_ids)

            total_loss, loss_next, bpb = compute_smaulbrain_loss(
                logits_next, logits_lookahead, logits_boundary, targets
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Step {step:04d} | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"CE Loss (nats): {loss_next.item():.4f} | "
                    f"BPB: {bpb:.4f}"
                )

if __name__ == "__main__":
    train()
