"""Streaming byte-level dataset: never requires a file-backed corpus in RAM."""
import hashlib
import os

import torch


class ByteStream:
    """Reproducible cursor over raw bytes (file or in-memory buffer)."""

    def __init__(self, path=None, data: bytes = b""):
        if path is None and not data:
            raise ValueError("ByteStream needs path or non-empty data")
        self.path = path
        self._fh = None
        if data:
            self._data = bytes(data)
            self.size = len(self._data)
            self.identity = {
                "kind": "memory",
                "size": self.size,
                "sha256": hashlib.sha256(self._data).hexdigest(),
            }
        else:
            self._data = None
            assert path is not None
            self._fh = open(path, "rb")
            stat = os.fstat(self._fh.fileno())
            self.size = stat.st_size
            if self.size == 0:
                self.close()
                raise ValueError(f"empty corpus: {path}")
            self.identity = {
                "kind": "file",
                "path": os.path.realpath(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        self.cursor = 0

    def _read_at(self, off: int, n: int) -> bytes:
        if self._data is not None:
            return self._data[off:off + n]
        if self._fh is None or self._fh.closed:
            raise RuntimeError("ByteStream is closed")
        self._fh.seek(off)
        return self._fh.read(n)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def next_tokens(self, seq_len: int) -> list:
        if not isinstance(seq_len, int) or seq_len < 1:
            raise ValueError("seq_len must be an integer >= 1")
        n = seq_len + 1  # +1 so loss has a target for every input position
        if self.size < n:
            raise ValueError(f"corpus ({self.size}B) smaller than seq_len {seq_len}")
        if self.cursor + n > self.size:
            self.cursor = 0  # wrap (epoch boundary)
        chunk = self._read_at(self.cursor, n)
        if len(chunk) != n:
            raise RuntimeError(
                f"corpus read returned {len(chunk)} bytes, expected {n}; "
                "the file may have changed during training"
            )
        self.cursor += n
        return list(chunk)

    def state_dict(self) -> dict:
        return {
            "cursor": self.cursor,
            "path": self.path,
            "size": self.size,
            "identity": dict(self.identity),
        }

    def load_state_dict(self, sd: dict) -> None:
        saved_identity = sd.get("identity")
        if saved_identity is not None:
            # Identity uses real paths for files and a content hash for in-memory
            # corpora, so equivalent relative/absolute path spellings can resume.
            if saved_identity != self.identity:
                raise ValueError("dataset contents or file metadata changed since checkpoint")
        elif sd.get("size") != self.size or sd.get("path") != self.path:
            # Compatibility with checkpoints written before data identity existed.
            raise ValueError("dataset mismatch on resume")
        try:
            cursor = int(sd["cursor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid dataset cursor in checkpoint") from exc
        if cursor < 0 or cursor > self.size:
            raise ValueError(f"dataset cursor out of range: {cursor} for size {self.size}")
        self.cursor = cursor


def synth_bytes(n: int, seed: int, period: int = 64) -> bytes:
    if n < 1:
        raise ValueError("synthetic corpus size must be >= 1")
    if period < 1:
        raise ValueError("synthetic corpus period must be >= 1")
    rng = torch.Generator().manual_seed(int(seed))
    base = bytes((i * 37 + 11) % 251 + 2 for i in range(period))
    out = bytearray()
    while len(out) < n:
        out.extend(base)
        noise = torch.randint(0, 256, (max(1, period // 4),), generator=rng, dtype=torch.uint8)
        out.extend(noise.tolist())
    return bytes(out[:n])
