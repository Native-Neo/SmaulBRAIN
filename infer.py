"""Inference: same core model and checkpoint format, gradients off.

Generation reuses ``forward_infer`` (the exact training forward path minus
gradients and ponder weighting, plus early halting). Byte ids stream through
the incremental byte decoder; sampling supports temperature / top-k / top-p.
"""

from __future__ import annotations

import torch

from bytes import IncrementalByteDecoder, decode_text, encode_text

#: Prompt used when the caller passes no ids (embedding row 10 == newline).
DEFAULT_PROMPT_ID = 10


def _check_ids(ids: list[int], vocab_size: int, what: str) -> None:
    bad = [i for i in ids if not 0 <= i < vocab_size]
    if bad:
        raise ValueError(
            f"{what} has {len(bad)} ids outside [0, {vocab_size}) "
            f"(e.g. {bad[:5]}); vocabulary covers bytes + specials only"
        )


def _check_sampler(temperature: float, top_k: int = 0, top_p: float = 1.0) -> None:
    import math
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError(f"top_k must be a non-negative int, got {top_k!r}")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
            or not 0.0 < top_p <= 1.0
            or (isinstance(top_p, float) and not math.isfinite(top_p))):
        raise ValueError(f"top_p must be in (0, 1], got {top_p!r}")


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, generator: torch.Generator | None = None) -> int:
    """Sample one id from last-position logits [V]."""
    _check_sampler(temperature, top_k, top_p)
    l = logits.float()
    if temperature <= 0:
        return int(l.argmax().item())
    l = l / max(temperature, 1e-6)
    if top_k > 0:
        k = min(top_k, l.numel())
        thresh = torch.topk(l, k).values[-1]
        l = torch.where(l >= thresh, l, torch.full_like(l, float("-inf")))
    if top_p < 1.0:
        order = torch.argsort(l, descending=True)
        probs = torch.softmax(l[order], dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # First position always kept: top_p can never mask every candidate.
        keep = torch.cat([torch.ones(1, dtype=torch.bool, device=l.device),
                          cum[:-1] < top_p])
        if not bool(keep.any()):
            keep[0] = True
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
    """Autoregressive byte generation. Returns ids, text, depths, paging stats.

    ``context`` caps the prompt window (defaults to the model context
    length); the prompt keeps its LAST ``context`` ids. An empty prompt
    falls back to ``[DEFAULT_PROMPT_ID]``. Every id is range-checked
    against the model vocabulary before the first matmul.
    """
    was_training = model.training
    model.eval()
    dev = model.embed.weight.device
    ctx = model.cfg.context_length if context is None else int(context)
    if ctx <= 0:
        raise ValueError(f"context must be positive, got {context!r}")
    _check_sampler(temperature, top_k, top_p)
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids) or [DEFAULT_PROMPT_ID]
    _check_ids(ids, model.cfg.vocab_size, "prompt_ids")
    ids = ids[-ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out, attn_states = model.forward_infer_stateful(x)
    depths: list[list[int]] = []
    decoder = IncrementalByteDecoder()
    text_parts: list[str] = []
    if max_new <= 0:
        if was_training:
            model.train()
        return {"ids": ids, "text": "", "depths": depths,
                "paging": model.pager.stats.to_dict()}
    for _ in range(max_new):
        nxt = sample_next(out["logits"][0, -1], temperature, top_k, top_p, gen)
        ids.append(nxt)
        depths.append([int(out["depths"][0, -1].item())])
        text_parts.append(decoder.feed([nxt]))
        x = torch.tensor([[nxt]], dtype=torch.long, device=dev)
        out, attn_states = model.forward_infer_step(x, attn_states)

    text_parts.append(decoder.flush())
    if was_training:
        model.train()
    return {"ids": ids, "text": "".join(text_parts), "depths": depths,
            "paging": model.pager.stats.to_dict()}


def encode_prompt(text: str) -> list[int]:
    return encode_text(text)


def decode_ids(ids: list[int]) -> str:
    return decode_text(ids)
