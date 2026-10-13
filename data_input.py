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
import sys
from pathlib import Path
from typing import Iterator

MAX_DATA_BYTES = 64 * 1024 * 1024
MAX_SEQUENCES = 512
_SUPPORTED = {".txt", ".text", ".md", ".jsonl", ".json", ".csv", ".parquet"}


def _json_bytes(value) -> bytes:
    """Serialize a structured data value as UTF-8 JSON bytes."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _parquet_python_size(value, limit: int) -> int:
    """Conservatively bound scalar expansion without converting it to Python."""
    import pyarrow as pa

    if not value.is_valid:
        return sys.getsizeof(None)
    if pa.types.is_dictionary(value.type):
        return _parquet_python_size(value.value, limit)
    if (pa.types.is_string(value.type) or pa.types.is_large_string(value.type)
            or pa.types.is_binary(value.type) or pa.types.is_large_binary(value.type)
            or pa.types.is_fixed_size_binary(value.type)):
        # Python Unicode can use four bytes per character, even for ASCII
        # prefixes. as_buffer() exposes Arrow storage without making a str.
        return sys.getsizeof("\U0010ffff") + 4 * value.as_buffer().size
    if (pa.types.is_list(value.type) or pa.types.is_large_list(value.type)
            or pa.types.is_fixed_size_list(value.type) or pa.types.is_map(value.type)):
        children = value.values
        size = sys.getsizeof([]) + 16 * len(children)
    elif pa.types.is_struct(value.type):
        size = sys.getsizeof({}) + sum(
            128 + sys.getsizeof(field.name) for field in value.type
        )
        children = (value[index] for index in range(len(value.type)))
    elif pa.types.is_integer(value.type) or pa.types.is_boolean(value.type):
        return sys.getsizeof(2 ** 64)
    elif pa.types.is_floating(value.type):
        return sys.getsizeof(0.0)
    elif pa.types.is_temporal(value.type) or pa.types.is_decimal(value.type):
        # Fixed-width Python numbers/dates/decimals have bounded expansion.
        return 128
    else:
        raise ValueError(f"cannot bound Python expansion for Parquet type {value.type}")
    if size > limit:
        return size
    for child in children:
        size += _parquet_python_size(child, limit - size)
        if size > limit:
            break
    return size


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
            while remaining > 0:
                # readline(size) bounds memory even when a malicious or
                # malformed dataset contains a multi-gigabyte single line.
                line = stream.readline(remaining + 1)
                if not line:
                    break
                if len(line) > remaining and not line.endswith(b"\n"):
                    # Do not emit a partial serialized record; structured
                    # rows must remain intact when the byte budget is reached.
                    return
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"invalid JSONL row in {path}: {exc}") from exc
                encoded = _json_bytes(row)
                if len(encoded) > remaining:
                    return
                remaining -= len(encoded)
                yield encoded
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
                return
            remaining -= len(encoded)
            yield encoded
        return

    if suffix == ".csv":
        with path.open("rb") as stream:
            data = stream.read(remaining)
            truncated = stream.tell() < path.stat().st_size
        if truncated:
            # Drop an incomplete physical line, including a cut UTF-8 code
            # point. strict parsing below also drops cut multiline records.
            end = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1
            data = data[:end]
        with io.StringIO(data.decode("utf-8"), newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            try:
                for row in reader:
                    encoded = _json_bytes(row)
                    if len(encoded) > remaining:
                        return
                    remaining -= len(encoded)
                    yield encoded
                    if remaining <= 0:
                        return
            except csv.Error as exc:
                # A budget boundary inside a quoted multiline record is an
                # incomplete row, not a reason to discard preceding rows.
                if truncated and str(exc) == "unexpected end of data":
                    return
                raise ValueError(f"invalid CSV row in {path}: {exc}") from exc
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
            # Keep Arrow decoding incremental. Row-group storage sizes do not
            # bound Python objects, particularly nested or dictionary values.
            for batch in parquet.iter_batches(batch_size=1):
                # Python containers can be much larger than JSON text. Allow
                # bounded overhead, capped at the default dataset allocation.
                memory_limit = min(MAX_DATA_BYTES, max(4096, 64 * remaining))
                expanded_size = sys.getsizeof({})
                for index, name in enumerate(batch.schema.names):
                    expanded_size += 128 + sys.getsizeof(name)
                    expanded_size += _parquet_python_size(
                        batch.column(index)[0], memory_limit - expanded_size
                    )
                    if expanded_size > memory_limit:
                        return
                row = batch.to_pylist()[0]
                encoded = _json_bytes(row)
                if len(encoded) > remaining:
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
    """Return whether a path is a regular file or directory without following a symlink."""
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
    for name, value in (("context_length", context_length),
                         ("max_bytes", max_bytes),
                         ("max_sequences", max_sequences)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}")

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
    per_file_budget = max(1, max_bytes // len(files)) if is_directory else max_bytes
    for file_index, file_path in enumerate(files):
        if file_index and raw and raw[-1] not in b"\r\n\t " and len(raw) < max_bytes:
            raw.extend(b"\n")
        remaining = min(max_bytes - len(raw), per_file_budget)
        if remaining <= 0:
            break
        for chunk in _iter_file_text(file_path, remaining):
            raw.extend(chunk)
            if len(raw) >= max_bytes:
                break
        if len(raw) >= max_bytes:
            break

    stride = context_length + 1
    total_sequences = len(range(0, len(raw) - 1, stride))
    if is_directory and total_sequences > max_sequences:
        if max_sequences == 1:
            offsets = [0]
        else:
            offsets = [
                (index * (total_sequences - 1)) // (max_sequences - 1) * stride
                for index in range(max_sequences)
            ]
    else:
        offsets = list(range(0, len(raw) - 1, stride))[:max_sequences]
    seqs = [list(raw[offset:offset + stride]) for offset in offsets]
    if not seqs:
        raise ValueError(f"dataset contains no usable text: {path}")
    return seqs, ("directory" if is_directory else "file")
