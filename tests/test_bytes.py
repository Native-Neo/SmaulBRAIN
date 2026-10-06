"""Byte representation: identity round-trips, specials, streaming decode."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bytes import (
    IncrementalByteDecoder, decode_bytes, decode_text, encode_bytes,
    encode_text, with_bos, with_eos, BOS_ID, EOS_ID, PAD_ID, BYTE_VOCAB,
)


def test_identity_all_256_bytes():
    data = bytes(range(256))
    assert decode_bytes(encode_bytes(data)) == data


def test_arbitrary_binary_roundtrip():
    data = bytes([0, 255, 13, 0, 200, 1, 254, 128, 7] * 17)
    assert decode_bytes(encode_bytes(data)) == data


def test_text_utf8_multibyte():
    s = "hello wörld — bytes ✓"
    assert decode_text(encode_text(s)) == s


def test_invalid_utf8_replacement():
    assert isinstance(decode_text([0xFF, 0xFE, 65]), str)
    assert decode_text([65]) == "A"


def test_specials_are_structural_only():
    ids = with_eos(with_bos([65, 66]))
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert decode_bytes(ids) == b"AB"  # specials skipped in byte decode
    assert PAD_ID >= BYTE_VOCAB


def test_streaming_split_tail():
    enc = "é".encode("utf-8")  # 2-byte sequence
    dec = IncrementalByteDecoder()
    part1 = dec.feed([enc[0]])
    part2 = dec.feed([enc[1]])
    assert part1 + part2 + dec.flush() == "é"


def test_invalid_byte_does_not_stall_decoder():
    dec = IncrementalByteDecoder()
    assert dec.feed([0xFF]) == "�"  # surfaced immediately, not buffered forever
    assert dec.feed([65]) == "A"  # decoding continues after bad bytes
    assert dec.flush() == ""


def test_split_multibyte_across_single_byte_feeds():
    raw = "héllo ✓ wörld".encode("utf-8")
    dec = IncrementalByteDecoder()
    assert "".join(dec.feed([b]) for b in raw) + dec.flush() == "héllo ✓ wörld"


def test_trailing_split_tail_flushes_as_replacement():
    dec = IncrementalByteDecoder()
    assert dec.feed([0xC3]) == ""  # first byte of 2-byte seq stays buffered
    assert dec.flush() == "�"
    assert dec.feed([66]) == "B"  # reusable after flush
