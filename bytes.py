"""Byte-level input/output representation.

The core representation is raw bytes (ids 0..255). A small set of structural
special tokens (ids 256..) may bracket sequences, but no BPE/WordPiece/
SentencePiece vocabulary is ever learned or required: arbitrary byte data
round-trips exactly, including invalid UTF-8 (decoded with replacement).
"""

from __future__ import annotations

import codecs

BYTE_VOCAB = 256

# Structural markers only; the model works fine with zero specials.
SPECIALS = ("<bos>", "<eos>", "<pad>", "<sep>")
SPECIAL_IDS = {tok: BYTE_VOCAB + i for i, tok in enumerate(SPECIALS)}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_IDS.items()}

PAD_ID = SPECIAL_IDS["<pad>"]
BOS_ID = SPECIAL_IDS["<bos>"]
EOS_ID = SPECIAL_IDS["<eos>"]


def total_vocab(num_specials: int = len(SPECIALS)) -> int:
    return BYTE_VOCAB + num_specials


def encode_bytes(data: bytes) -> list[int]:
    """Arbitrary bytes -> token ids (identity mapping)."""
    return list(data)


def encode_text(text: str) -> list[int]:
    """UTF-8 text -> byte token ids."""
    return list(text.encode("utf-8"))


def decode_bytes(ids: list[int]) -> bytes:
    """Token ids -> bytes; special ids (>=256) are skipped (structural only)."""
    return bytes(i for i in ids if 0 <= i < BYTE_VOCAB)


def decode_text(ids: list[int]) -> str:
    """Token ids -> text with replacement for invalid UTF-8 sequences."""
    return decode_bytes(ids).decode("utf-8", errors="replace")


def with_bos(ids: list[int]) -> list[int]:
    return [BOS_ID, *ids]


def with_eos(ids: list[int]) -> list[int]:
    return [*ids, EOS_ID]


class IncrementalByteDecoder:
    """Streaming decoder that buffers split UTF-8 tails across chunks.

    Backed by a codecs incremental decoder: O(n) amortized instead of
    re-scanning the whole buffer for the longest decodable prefix on every
    feed. Special ids (>=256) are structural and never enter the byte stream.
    """

    def __init__(self) -> None:
        # errors=replace: invalid bytes surface as U+FFFD at feed time instead
        # of stalling the decoder (the old prefix-scan buffered them forever).
        # Incomplete split tails are still buffered across feeds; only a
        # trailing tail at flush falls back to replacement, matching
        # decode_text semantics.
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if 0 <= i < BYTE_VOCAB)
        return self._dec.decode(raw, final=False)

    def flush(self) -> str:
        # A trailing split tail is incomplete input: decode with replacement,
        # matching decode_text semantics.
        buf, _ = self._dec.getstate()
        self._dec.reset()
        return bytes(buf).decode("utf-8", errors="replace")
