"""Bounded, deterministic loading of text and structured dataset folders.

The training driver operates on byte sequences. Structured rows are serialized
as UTF-8 JSON so their fields remain readable instead of training on file
container bytes (for example, Parquet's binary encoding).
"""

from __future__ import annotations

import csv
import io
import json
import os
import stat
from pathlib import Path
from typing import Iterator

MAX_DATA_BYTES = 64 * 1024 * 1024
MAX_SEQUENCES = 512
_SUPPORTED = {".txt", ".text", ".md", ".jsonl", ".json", ".csv", ".parquet"}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _iter_file_text(path: Path, remaining: int) -> Iterator[bytes]:
    """Yield readable text from one supported file, bounded by remaining bytes."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text", ".md"}:
        with path.open("rb") as stream:
            while remaining > 0:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        return

    if suffix == ".jsonl":
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"invalid JSONL row in {path}: {exc}") from exc
                encoded = _json_bytes(row)
                if len(encoded) > remaining:
                    yield encoded[:remaining]
                    return
                remaining -= len(encoded)
                yield encoded
                if remaining <= 0:
                    return
        return

    if suffix == ".json":
        if path.stat().st_size > remaining:
            raise ValueError(f"JSON file exceeds remaining dataset byte budget: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read JSON dataset {path}: {exc}") from exc
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            encoded = _json_bytes(row)
            if len(encoded) > remaining:
                yield encoded[:remaining]
                return
            remaining -= len(encoded)
            yield encoded
        return

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                encoded = _json_bytes(row)
                if len(encoded) > remaining:
                    yield encoded[:remaining]
                    return
                remaining -= len(encoded)
                yield encoded
                if remaining <= 0:
                    return
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError(
                f"reading Parquet requires pyarrow; cannot load {path}"
            ) from exc
        try:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    encoded = _json_bytes(row)
                    if len(encoded) > remaining:
                        yield encoded[:remaining]
                        return
                    remaining -= len(encoded)
                    yield encoded
                    if remaining <= 0:
                        return
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read Parquet dataset {path}: {exc}") from exc
        return

    raise ValueError(f"unsupported dataset file type: {path}")


def _regular_non_symlink(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"cannot inspect dataset path {path}: {exc}") from exc
    return stat.S_ISREG(mode) and not stat.S_ISLNK(mode)


def load_training_sequences(
    source: str | os.PathLike[str],
    context_length: int,
    *,
    max_bytes: int = MAX_DATA_BYTES,
    max_sequences: int = MAX_SEQUENCES,
) -> tuple[list[list[int]], str]:
    """Load a text file or a directory tree into bounded byte-id sequences.

    Directory traversal is recursive and deterministic. Hidden paths and
    symlinks are ignored. Only supported file extensions are read. Directory
    inputs are sampled in sorted path order up to max_bytes; files are never
    concatenated as raw binary containers.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"dataset path not found: {path}")
    if path.is_symlink():
        raise ValueError(f"dataset path must not be a symlink: {path}")

    is_directory = path.is_dir()
    if is_directory:
        files = []
        for root, dirs, names in os.walk(path, followlinks=False):
            dirs[:] = sorted(
                name for name in dirs
                if not name.startswith(".")
                and not (Path(root) / name).is_symlink()
            )
            for name in sorted(names):
                candidate = Path(root) / name
                if name.startswith(".") or candidate.suffix.lower() not in _SUPPORTED:
                    continue
                if candidate.is_symlink():
                    continue
                if _regular_non_symlink(candidate):
                    files.append(candidate)
        files.sort(key=lambda item: item.relative_to(path).as_posix())
        if not files:
            raise ValueError(
                f"no supported dataset files under {path}; supported: "
                + ", ".join(sorted(_SUPPORTED))
            )
    else:
        if not _regular_non_symlink(path):
            raise ValueError(f"dataset path is not a regular file: {path}")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"dataset file exceeds {max_bytes // (1024 * 1024)}MB cap: {path}")
        files = [path]

    raw = bytearray()
    for file_path in files:
        remaining = max_bytes - len(raw)
        if remaining <= 0:
            break
        for chunk in _iter_file_text(file_path, remaining):
            raw.extend(chunk)
            if len(raw) >= max_bytes:
                break
        if len(raw) >= max_bytes:
            break

    stride = context_length + 1
    seqs = [
        list(raw[offset:offset + stride])
        for offset in range(0, len(raw) - 1, stride)
    ][:max_sequences]
    if not seqs:
        raise ValueError(f"dataset contains no usable text: {path}")
    return seqs, ("directory" if is_directory else "file")
