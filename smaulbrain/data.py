"""Streaming byte-level dataset: never requires the corpus in RAM."""
import os


class ByteStream:
    """Reproducible cursor over raw bytes (file or in-memory buffer)."""

    def __init__(self, path=None, data: bytes = b"", seed: int = 0):
        if path is None and not data:
            raise ValueError("ByteStream needs path or data")
        self.path = path
        self._fh = None
        if data:
            self._data = bytes(data)
            self.size = len(self._data)
        else:
            self._data = None
            assert path is not None
            self.size = os.path.getsize(path)
            if self.size == 0:
                raise ValueError(f"empty corpus: {path}")
            # Reuse one buffered file handle instead of open/seek/read/close
            # on every training step.
            self._fh = open(path, "rb")
        self.cursor = 0
        self.rng = __import__("numpy").random.default_rng(seed)

    def _read_at(self, off: int, n: int) -> bytes:
        if self._data is not None:
            return self._data[off:off + n]
        assert self.path is not None and self._fh is not None
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
        n = seq_len + 1  # +1 so loss has a target for every input position
        if self.size < n:
            raise ValueError(f"corpus ({self.size}B) smaller than seq_len {seq_len}")
        if self.cursor + n > self.size:
            self.cursor = 0  # wrap (epoch boundary)
        chunk = self._read_at(self.cursor, n)
        self.cursor += n
        return list(chunk)

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "path": self.path, "size": self.size}

    def load_state_dict(self, sd: dict) -> None:
        if sd.get("size") != self.size or sd.get("path") != self.path:
            raise ValueError("dataset mismatch on resume")
        self.cursor = int(sd["cursor"])


def synth_bytes(n: int, seed: int, period: int = 64) -> bytes:
    rng = __import__("numpy").random.default_rng(seed)
    base = bytes((i * 37 + 11) % 251 + 2 for i in range(period))
    out = bytearray()
    while len(out) < n:
        out += base
        out += rng.integers(0, 256, size=period // 4).astype("uint8").tobytes()
    return bytes(out[:n])
