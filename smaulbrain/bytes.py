"""Byte-level input/output representation.

The core representation is raw bytes (ids 0..255). A small set of structural
special tokens (ids 256..) may bracket sequences, but no BPE/WordPiece/
SentencePiece vocabulary is ever learned or required: arbitrary byte data
round-trips exactly, including invalid UTF-8 (decoded with replacement).
"""

from __future__ import annotations

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
    """Streaming decoder that buffers split UTF-8 tails across chunks."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, ids: list[int]) -> str:
        self._buf.extend(i for i in ids if 0 <= i < BYTE_VOCAB)
        raw = bytes(self._buf)
        # Find longest decodable prefix; keep a possible split tail buffered.
        for end in range(len(raw), -1, -1):
            try:
                text = raw[:end].decode("utf-8")
                self._buf = bytearray(raw[end:])
                return text
            except UnicodeDecodeError:
                continue
        return ""

    def flush(self) -> str:
        text = bytes(self._buf).decode("utf-8", errors="replace")
        self._buf.clear()
        return text
