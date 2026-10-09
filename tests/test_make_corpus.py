"""Bulk corpus builder: pure-function + fake-shard pipeline tests (no network)."""

import gzip
import json
import os
import sqlite3
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.make_corpus import (
    ShardWriter, doc_hash, english_ok, iter_shard_docs, normalize_doc,
    open_store, run,
)


def test_normalize_collapses_whitespace():
    assert normalize_doc("  hello\n\n  world \t ! ") == "hello world !"


def test_english_ok_gate():
    assert english_ok("The quick brown fox jumps over 13 lazy dogs.")
    assert not english_ok("")
    assert not english_ok("".join(chr(c) for c in range(0x4E00, 0x4E00 + 200)))
    assert not english_ok("x" * 10 + "\x00" * 100)


def test_doc_hash_stable_and_sensitive():
    assert doc_hash("abc") == doc_hash("abc")
    assert doc_hash("abc") != doc_hash("abd")


def test_shard_writer_rotates(tmp_path):
    w = ShardWriter(str(tmp_path), shard_bytes=10)
    w.write("1234567890")
    w.write("abcdefghij")
    w.close()
    assert open(os.path.join(str(tmp_path), "shard-00000.txt")).read() == "1234567890\n"
    assert open(os.path.join(str(tmp_path), "shard-00001.txt")).read() == "abcdefghij\n"


def test_iter_shard_docs_skips_bad_lines(tmp_path):
    p = str(tmp_path / "s.json.gz")
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({"text": "hello world this is long enough xx"}) + "\n")
        f.write("not json\n")
        f.write(json.dumps({"nope": 1}) + "\n")
    assert list(iter_shard_docs(p)) == ["hello world this is long enough xx"]


def _fake_shard(tmp_path, docs):
    p = str(tmp_path / "fake.json.gz")
    with gzip.open(p, "wt") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    return p


def test_run_dedupes_filters_and_resumes(tmp_path, monkeypatch):
    import scripts.make_corpus as mc
    docs = [
        "The server restarted at midnight after the update finished cleanly today",
        "The server restarted at midnight after the update finished cleanly today",  # dup
        "short",
        "Quantum zebras zebras zebras " * 3 + "end of line here",
    ]
    fake = _fake_shard(tmp_path, docs)
    calls = {"n": 0}

    def fake_dl(i, tmp):
        calls["n"] += 1
        assert i >= 100
        return f"f{i}", fake

    monkeypatch.setattr(mc, "download_shard", fake_dl)
    monkeypatch.setattr(mc, "N_SHARDS", 101)
    out = str(tmp_path / "c")
    stats = run(out, 10**9, 10**6, start=100, min_chars=20)
    assert stats["kept"] == 2 and stats["dup"] == 1 and stats["short"] == 1
    assert calls["n"] == 1
    # Resume: same file skipped, nothing re-downloaded, counts stable.
    stats2 = run(out, 10**9, 10**6, start=100, min_chars=20)
    assert calls["n"] == 1 and stats2["kept"] == 0
    con = sqlite3.connect(os.path.join(out, "dedupe.db"))
    assert con.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM done_files").fetchone()[0] == 1


def test_run_stops_at_target_bytes(tmp_path, monkeypatch):
    import scripts.make_corpus as mc
    fake = _fake_shard(tmp_path, ["A completely ordinary english sentence with many words in it"])
    monkeypatch.setattr(mc, "download_shard", lambda i, t: ("f", fake))
    monkeypatch.setattr(mc, "N_SHARDS", 5)
    out = str(tmp_path / "c")
    stats = run(out, 10, 10**6, start=0, min_chars=5)
    assert stats["bytes"] >= 10
