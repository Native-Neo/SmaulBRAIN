"""Byte-level vocabulary: fixed mapping of all 256 byte values + specials.

Byte b (0..255) maps to token id b. Specials follow contiguously.
No tokenizer dependency. Deterministic, lossless, streaming-safe.
"""
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
N_BYTE_TOKENS = 256

SPECIAL_NAMES = {"BOS": BOS_ID, "EOS": EOS_ID, "PAD": PAD_ID}


def encode_bytes(data: bytes) -> list:
    return list(data)


def decode_ids(ids) -> bytes:
    return bytes(int(i) for i in ids if 0 <= int(i) <= 255)


def is_special(tok: int) -> bool:
    return int(tok) >= 256


def vocab_meta() -> dict:
    return {
        "vocab_size": VOCAB_SIZE,
        "n_byte_tokens": N_BYTE_TOKENS,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "mapping": "identity-bytes-0..255",
    }
