from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_input import load_training_sequences


def _joined(seqs: list[list[int]]) -> bytes:
    return bytes(token for seq in seqs for token in seq)


def test_directory_reads_nested_text_files_in_sorted_order(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "b" / "part.txt").write_text("SECOND", encoding="utf-8")
    (root / "a" / "part.txt").write_text("FIRST", encoding="utf-8")

    seqs, kind = load_training_sequences(root, context_length=5)

    assert kind == "directory"
    assert _joined(seqs) == b"FIRST\nSECOND"


def test_directory_samples_sequences_across_files(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("A" * 12, encoding="utf-8")
    (root / "b.txt").write_text("B" * 12, encoding="utf-8")

    seqs, _ = load_training_sequences(
        root, context_length=5, max_bytes=128, max_sequences=2
    )

    sample = _joined(seqs)
    assert len(seqs) == 2
    assert sample.startswith(b"A")
    assert sample.endswith(b"B")


def test_directory_serializes_jsonl_records_as_text(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "rows.jsonl").write_text(
        json.dumps({"question": "2+2", "answer": "4"}) + "\n",
        encoding="utf-8",
    )

    seqs, kind = load_training_sequences(root, context_length=64)

    assert kind == "directory"
    text = _joined(seqs).decode("utf-8")
    assert '"question":"2+2"' in text
    assert '"answer":"4"' in text


def test_directory_serializes_parquet_rows_as_text(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    root.mkdir()
    pq.write_table(
        pa.table({"question": ["2+2"], "answer": ["4"]}),
        root / "part-000.parquet",
    )

    seqs, kind = load_training_sequences(root, context_length=128)

    assert kind == "directory"
    text = _joined(seqs).decode("utf-8")
    assert '"question":"2+2"' in text
    assert '"answer":"4"' in text


def test_directory_ignores_hidden_and_unsupported_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / ".secret.txt").write_text("DO NOT LOAD", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01")
    (root / "visible.txt").write_text("VISIBLE", encoding="utf-8")

    seqs, _ = load_training_sequences(root, context_length=64)

    assert _joined(seqs) == b"VISIBLE"


def test_empty_directory_errors_instead_of_falling_back_to_demo(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="no supported dataset files"):
        load_training_sequences(root, context_length=8)


def test_single_file_preserves_raw_text_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello world")

    seqs, kind = load_training_sequences(path, context_length=5)

    assert kind == "file"
    assert _joined(seqs) == b"hello world"
