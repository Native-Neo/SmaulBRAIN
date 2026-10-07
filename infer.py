"""Inference: same core model and checkpoint format, gradients off.

Generation reuses ``forward_infer`` (the exact training forward path minus
gradients and ponder weighting, plus early halting). Byte ids stream through
the incremental byte decoder; sampling supports temperature / top-k / top-p.

Special-token generation contract (issue #56):
- Vocabulary is bytes (0..255) + specials (256..); byte 0 (NUL) is ordinary
  data, never padding and never a terminator.
- Prompts may contain any id in [0, vocab_size), including BOS/EOS/PAD/SEP;
  ``generate`` does NOT auto-prepend BOS (continuation-only; callers use
  ``bytes.with_bos`` when a BOS prefix is wanted).
- Sampling never emits PAD or BOS as continuation text: both are masked to
  -inf before argmax/sampling. PAD is padding (masked from the loss, skipped
  by the decoder); BOS marks sequence start and must not re-appear
  mid-continuation.
- EOS is samplable and, when ``stop_on_eos=True`` (default), terminates
  generation early. The EOS id is included in ``ids`` (and gets a depth
  entry) but contributes no text (the decoder skips specials). With
  ``stop_on_eos=False`` generation runs the full ``max_new`` steps even
  through EOS (useful for probing/scoring).
- SEP is samplable ordinary structure (decoded as no text, no stopping).
- ``text`` is always the continuation-only decode (prompt excluded),
  matching ``decode_text(continuation_ids)``; streaming feed/flush output
  equals the bulk decode.
- Logits are validated against the configured vocabulary
  (``model.cfg.vocab_size``), never a hard-coded 256.
"""

from __future__ import annotations

import torch

from bytes import (
    BYTE_VOCAB,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_IDS,
    IncrementalByteDecoder,
    decode_text,
    encode_text,
)

#: Structural separator id (no dedicated constant in bytes.py).
SEP_ID = SPECIAL_IDS["<sep>"]

#: Continuation sampler never emits these (see module contract).
NEVER_SAMPLE_IDS = (BOS_ID, PAD_ID)

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


def _check_logits(logits: torch.Tensor, vocab_size: int) -> None:
    if not isinstance(logits, torch.Tensor):
        raise ValueError(f"logits must be a Tensor, got {type(logits).__name__}")
    if logits.ndim < 1:
        raise ValueError(f"logits must have >= 1 dim, got shape {tuple(logits.shape)}")
    if int(logits.shape[-1]) != int(vocab_size):
        raise ValueError(
            f"logits last dim {int(logits.shape[-1])} != vocab_size {int(vocab_size)}; "
            "logits must match the configured vocabulary (bytes + specials)"
        )


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, generator: torch.Generator | None = None,
                forbidden_ids: object = None,
                vocab_size: int | None = None) -> int:
    """Sample one id from last-position logits [V].

    ``forbidden_ids`` masks ids to -inf before argmax/sampling (used by
    ``generate`` to prevent PAD/BOS emission). Out-of-range entries
    (``< 0`` or ``>= V``) are ignored so tiny test logits still work;
    in-range ids can never be returned, including via greedy argmax.
    When ``vocab_size`` is given, the logits last dim must equal it
    (configured-vocabulary check, never a hard-coded 256).
    """
    _check_sampler(temperature, top_k, top_p)
    if vocab_size is not None:
        _check_logits(logits, int(vocab_size))
    # Clone: masking must never mutate the caller's logits tensor.
    l = logits.detach().float().clone()
    if forbidden_ids:
        try:
            banned = list(forbidden_ids)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError(f"forbidden_ids must be an iterable of ints, got {forbidden_ids!r}")
        v = int(l.numel()) if l.ndim == 1 else int(l.shape[-1])
        # 1-D fast path (the documented [V] shape); N-D falls back to
        # last-dim indexing so broadcast logits still mask correctly.
        flat = l.reshape(-1, v) if l.ndim > 1 else None
        for b in banned:
            if isinstance(b, bool) or not isinstance(b, int):
                raise ValueError(f"forbidden id must be an int, got {b!r}")
            if 0 <= b < v:
                if flat is None:
                    l[b] = float("-inf")
                else:
                    flat[:, b] = float("-inf")
        # All-masked guard: sampling/argmax over all -inf is undefined.
        check = flat if flat is not None else l.unsqueeze(0)
        if bool((check[0] == float("-inf")).all().item()) if check.numel() else False:
            raise ValueError("all logits masked by forbidden_ids; nothing left to sample")
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
    stop_on_eos: bool = True,
) -> dict:
    """Autoregressive byte generation. Returns ids, text, depths, paging stats.

    ``context`` caps the prompt window (defaults to the model context
    length); the prompt keeps its LAST ``context`` ids. An empty prompt
    falls back to ``[DEFAULT_PROMPT_ID]``. Every id is range-checked
    against the model vocabulary before the first matmul. No BOS is
    auto-prepended (continuation-only; pass ``with_bos(...)`` ids explicitly
    when a BOS prefix is wanted). Byte 0 stays ordinary data throughout.

    Special-token policy (see module contract): PAD/BOS are masked from the
    continuation sampler on every step; EOS ends the loop early when
    ``stop_on_eos`` is True (default: standard LM termination; ``False``
    runs the full ``max_new`` steps for probing). The EOS terminator is kept
    in ``ids`` with its depth entry but decodes to no text. Logits are
    validated against ``model.cfg.vocab_size`` each step.
    """
    was_training = model.training
    model.eval()
    dev = model.embed.weight.device
    ctx = model.cfg.context_length if context is None else int(context)
    if ctx <= 0:
        raise ValueError(f"context must be positive, got {context!r}")
    if isinstance(stop_on_eos, bool) is False:
        raise ValueError(f"stop_on_eos must be a bool, got {stop_on_eos!r}")
    _check_sampler(temperature, top_k, top_p)
    vocab = int(model.cfg.vocab_size)
    if vocab < BYTE_VOCAB:
        raise ValueError(f"vocab_size {vocab} < byte base {BYTE_VOCAB}")
    forbidden = [i for i in NEVER_SAMPLE_IDS if 0 <= i < vocab]
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids) or [DEFAULT_PROMPT_ID]
    _check_ids(ids, vocab, "prompt_ids")
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
        _check_logits(out["logits"], vocab)
        nxt = sample_next(out["logits"][0, -1], temperature, top_k, top_p, gen,
                          forbidden_ids=forbidden, vocab_size=vocab)
        ids.append(nxt)
        depths.append([int(out["depths"][0, -1].item())])
        text_parts.append(decoder.feed([nxt]))
        if nxt == EOS_ID and stop_on_eos and EOS_ID < vocab:
            break
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
