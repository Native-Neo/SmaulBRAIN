from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_input import _iter_file_text, load_training_sequences


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



@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_length": 0}, "context_length must be a positive int"),
        ({"context_length": -1}, "context_length must be a positive int"),
        ({"max_bytes": 0}, "max_bytes must be a positive int"),
        ({"max_sequences": 0}, "max_sequences must be a positive int"),
        ({"max_sequences": True}, "max_sequences must be a positive int"),
    ],
)
def test_loader_rejects_invalid_limits(tmp_path: Path, kwargs, message) -> None:
    """Dataset limits must be positive integers and reject boolean values."""
    path = tmp_path / "sample.txt"
    path.write_text("some data", encoding="utf-8")
    context_length = kwargs.pop("context_length", 4)
    with pytest.raises(ValueError, match=message):
        load_training_sequences(path, context_length=context_length, **kwargs)


def test_jsonl_budget_never_emits_a_partial_oversized_record(tmp_path: Path) -> None:
    """A JSONL record exceeding the byte budget yields no partial training text."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "rows.jsonl").write_text(
        json.dumps({"text": "this record is much larger than the budget"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no usable text"):
        load_training_sequences(root, context_length=4, max_bytes=12)


def test_jsonl_large_line_is_read_with_a_bounded_budget(tmp_path: Path) -> None:
    """A large JSONL line is rejected within the configured dataset byte budget."""
    root = tmp_path / "dataset"
    root.mkdir()
    # Directory mode uses the configured byte budget rather than file size;
    # this line must be rejected without loading/emitting a partial JSON row.
    (root / "rows.jsonl").write_bytes(b'{"text":"' + b"x" * 100_000 + b'"}\n')
    with pytest.raises(ValueError, match="no usable text"):
        load_training_sequences(root, context_length=4, max_bytes=64)



def test_csv_budget_does_not_emit_a_partial_structured_row(tmp_path: Path) -> None:
    """An oversized CSV record produces no partial training text."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "rows.csv").write_text(
        "text\nthis row is much larger than the budget\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no usable text"):
        load_training_sequences(root, context_length=4, max_bytes=12)



def test_jsonl_valid_newline_terminated_records_load(tmp_path: Path) -> None:
    """Complete JSONL records within the byte budget produce training sequences."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "rows.jsonl").write_text(
        json.dumps({"text": "hello"}) + "\n", encoding="utf-8"
    )
    seqs, kind = load_training_sequences(root, context_length=4, max_bytes=256)
    assert kind == "directory"
    assert seqs



def test_parquet_oversized_row_produces_no_partial_training_text(tmp_path: Path) -> None:
    """An oversized Parquet record produces no partial training text."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "dataset"
    root.mkdir()
    pq.write_table(pa.table({"text": ["x" * 10000]}), root / "large.parquet", compression="gzip")
    with pytest.raises(ValueError, match="no usable text"):
        load_training_sequences(root, context_length=4, max_bytes=64)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_csv_large_file_retains_complete_rows_within_budget(tmp_path: Path, newline) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes((f"text{newline}ok{newline}" + "x" * 10000).encode())

    assert list(_iter_file_text(path, 32)) == [b'{"text":"ok"}\n']


def test_csv_output_budget_preserves_complete_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(b"text\na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\n")

    assert list(_iter_file_text(path, 26)) == [b'{"text":"a"}\n', b'{"text":"b"}\n']


@pytest.mark.parametrize("tail", [
    '"quoted\n' + "x" * 10000,
    "x" * 15 + "\u20ac" * 10000,
])
def test_csv_budget_drops_cut_multiline_and_utf8_rows(tmp_path: Path, tail) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(("text\nok\n" + tail).encode("utf-8"))

    assert list(_iter_file_text(path, 24)) == [b'{"text":"ok"}\n']


@pytest.mark.parametrize("ending", ["", "\n"])
def test_csv_complete_multiline_record_at_eof_loads(tmp_path: Path, ending) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(('text\n"a\nb"' + ending).encode())

    assert list(_iter_file_text(path, 64)) == [b'{"text":"a\\nb"}\n']


def test_csv_still_rejects_malformed_complete_record(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(b'text\n"unfinished\n')

    with pytest.raises(ValueError, match="invalid CSV row"):
        list(_iter_file_text(path, 64))


@pytest.mark.parametrize("row_group_size", [1, 4])
def test_parquet_reads_usable_prefix_despite_large_row_groups(tmp_path: Path, row_group_size) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table({"text": ["a", "b", "x" * 10000, "c"]}), path,
        row_group_size=row_group_size, compression="gzip",
    )

    assert list(_iter_file_text(path, 26)) == [b'{"text":"a"}\n', b'{"text":"b"}\n']


@pytest.mark.parametrize("value", ["x" * 1024, [0] * 1000, {"nested": ["x"] * 100}])
def test_parquet_allows_python_overhead_for_rows_that_fit_text_budget(tmp_path: Path, value) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "rows.parquet"
    row = {"value": value}
    pq.write_table(pa.Table.from_pylist([row]), path)
    expected = (json.dumps(row, separators=(",", ":")) + "\n").encode()

    assert list(_iter_file_text(path, len(expected))) == [expected]


@pytest.mark.parametrize("large_value", [
    "x" * 10000,
    ["x"] * 10000,
    {"nested": ["x"] * 10000},
])
def test_parquet_checks_expansion_before_python_materialization(tmp_path: Path, monkeypatch, large_value) -> None:
    """Oversized strings and nested values must never reach to_pylist()."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "rows.parquet"
    path.touch()
    batch = pa.record_batch({"value": [large_value]})

    class GuardedBatch:
        schema = batch.schema

        def column(self, index):
            return batch.column(index)

        def to_pylist(self):
            pytest.fail("oversized values must be checked before Python conversion")

    class FakeParquet:
        def iter_batches(self, *, batch_size):
            assert batch_size == 1
            yield GuardedBatch()
            pytest.fail("reading must stop after the first oversized row")

    monkeypatch.setattr(pq, "ParquetFile", lambda path: FakeParquet())

    assert list(_iter_file_text(path, 64)) == []
