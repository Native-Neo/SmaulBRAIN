"""Inference: same core model and checkpoint format, gradients off.

Generation reuses ``forward_infer`` (the exact training forward path minus
gradients and ponder weighting, plus early halting). Byte ids stream through
the incremental byte decoder; sampling supports temperature / top-k / top-p.
"""

from __future__ import annotations

import torch

from .bytes import IncrementalByteDecoder, decode_text, encode_text


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, generator: torch.Generator | None = None) -> int:
    """Sample one id from last-position logits [V]."""
    l = logits.float()
    if temperature <= 0:
        return int(l.argmax().item())
    l = l / max(temperature, 1e-6)
    if top_k > 0:
        k = min(top_k, l.numel())
        thresh = torch.topk(l, k).values[-1]
        l = torch.where(l >= thresh, l, torch.tensor(float("-inf")))
    if top_p < 1.0:
        order = torch.argsort(l, descending=True)
        probs = torch.softmax(l[order], dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = torch.cat([torch.tensor([True]), cum[:-1] < top_p])
        mask = torch.full_like(l, float("-inf"))
        mask[order[keep]] = l[order[keep]]
        l = mask
    probs = torch.softmax(l, dim=-1)
    if generator is None:
        return int(torch.multinomial(probs, 1).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int],
    max_new: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    context: int | None = None,
    seed: int = 0,
) -> dict:
    """Autoregressive byte generation. Returns ids, text, depths, paging stats."""
    model.eval()
    ctx = context or model.cfg.context_length
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids) or [10]
    depths: list[list[int]] = []
    decoder = IncrementalByteDecoder()
    text_parts: list[str] = []
    for _ in range(max_new):
        window = ids[-ctx:]
        x = torch.tensor([window], dtype=torch.long)
        out = model.forward_infer(x)
        nxt = sample_next(out["logits"][0, -1], temperature, top_k, top_p, gen)
        ids.append(nxt)
        depths.append([int(out["depths"][0, -1].item())])
        text_parts.append(decoder.feed([nxt]))
    text_parts.append(decoder.flush())
    return {"ids": ids, "text": "".join(text_parts), "depths": depths,
            "paging": model.pager.stats.to_dict()}


def encode_prompt(text: str) -> list[int]:
    return encode_text(text)


def decode_ids(ids: list[int]) -> str:
    return decode_text(ids)
