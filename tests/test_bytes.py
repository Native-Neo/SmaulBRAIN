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


def test_long_invalid_sequence_streams_linearly():
    import time
    N = 20000
    dec = IncrementalByteDecoder()
    t0 = time.time()
    parts = []
    for i in range(0, N, 500):
        parts.append(dec.feed([0xFF] * 500))
    parts.append(dec.flush())
    dt = time.time() - t0
    text = "".join(parts)
    assert text == "�" * N  # surfaced at feed time, never buffered
    assert dt < 5.0, f"long invalid stream too slow ({dt:.2f}s): O(n^2) regression?"


def test_long_invalid_single_byte_feeds_do_not_stall():
    # Adversarial O(n^2) shape: many tiny feeds of always-invalid bytes.
    # Old prefix-scan buffered them forever (quadratic); incremental
    # decoder must surface each immediately in linear time.
    import time
    N = 5000
    dec = IncrementalByteDecoder()
    t0 = time.time()
    out = "".join(dec.feed([0xFE]) for _ in range(N))
    out += dec.flush()
    dt = time.time() - t0
    assert out == "�" * N
    assert dt < 5.0, f"single-byte invalid feeds too slow ({dt:.2f}s)"


def test_split_three_and_four_byte_chars_at_every_position():
    for s in ["€", "😀", "a€b😀c"]:
        raw = s.encode("utf-8")
        for split in range(len(raw) + 1):
            dec = IncrementalByteDecoder()
            a = dec.feed(list(raw[:split]))
            b = dec.feed(list(raw[split:]))
            assert a + b + dec.flush() == s, (s, split)


def test_arbitrary_binary_streaming_matches_bulk_decode():
    import random
    rng = random.Random(0)
    data = bytes(rng.randrange(256) for _ in range(5000))
    dec = IncrementalByteDecoder()
    parts = []
    for i in range(0, len(data), 7):
        parts.append(dec.feed(list(data[i:i + 7])))
    parts.append(dec.flush())
    assert "".join(parts) == data.decode("utf-8", errors="replace")
    assert "".join(parts) == decode_text(list(data))


def test_repeated_feed_flush_cycles():
    dec = IncrementalByteDecoder()
    for _ in range(5):
        assert dec.feed([65]) == "A"
        assert dec.flush() == ""
    # Split tail flushed repeatedly stays reusable.
    for _ in range(3):
        assert dec.feed([0xC3]) == ""
        assert dec.flush() == "�"
    assert dec.feed([66]) + dec.flush() == "B"
    assert dec.flush() == ""  # double flush is idempotent
    # Invalid + split-tail + valid across cycles.
    assert dec.feed([0xFF]) == "�"
    assert dec.feed([0xE2]) == ""  # first byte of € stays buffered
    assert dec.feed([0x82, 0xAC]) == "€"
    assert dec.flush() == ""


def test_streaming_ignores_special_ids():
    dec = IncrementalByteDecoder()
    assert dec.feed([65, BOS_ID, 66, PAD_ID, 67]) + dec.flush() == "ABC"
    # Split multibyte with specials interleaved still decodes.
    enc = "é".encode("utf-8")
    dec = IncrementalByteDecoder()
    assert dec.feed([enc[0], BOS_ID]) == ""
    assert dec.feed([enc[1], EOS_ID]) + dec.flush() == "é"
