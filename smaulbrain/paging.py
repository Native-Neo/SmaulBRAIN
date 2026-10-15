"""Expert paging subsystem with one common interface and five modes.

R   : experts resident in system RAM.
V   : experts resident in compatible VRAM (requires accelerator backend).
D2R : experts loaded from checkpoint-backed disk storage into bounded RAM cache (LRU).
D2V : disk -> bounded host staging buffers -> VRAM (requires backend).
R2V : RAM-resident set, moved into bounded device cache as needed (requires backend).

CPU-only hosts support R and D2R. V/D2V/R2V return actionable errors naming
the missing backend. VRAM is never claimed from system RAM / integrated
graphics / shared memory.
"""
import collections
import os
import time

import numpy as np


class PagingError(RuntimeError):
    pass


def accelerator_available() -> tuple:
    """(available: bool, reason: str). No compatible accelerator on CPU-only hosts."""
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            return True, "cuda"
        return False, ("no compatible accelerator backend: torch reports "
                       "cuda.is_available()==False; system RAM / integrated "
                       "graphics / shared memory is not supported VRAM")
    except ImportError:
        return False, ("no compatible accelerator backend: torch is not installed; "
                       "install a backend with device execution to use V/D2V/R2V")


class ExpertStore:
    """Common paging interface. Backend owns weight arrays for all experts.

    ram_weights: dict expert_id -> {"Win","Wout"} fp32 arrays (modes R, R2V source).
    disk_dir: directory holding per-expert .npz files (modes D2R, D2V source).
    ram_cache_max_experts: bound on cached experts for D2R.
    """

    def __init__(self, mode: str, n_experts: int, ram_weights=None, disk_dir=None,
                 ram_cache_max_experts: int = 8, staging_max_bytes: int = 1 << 24):
        if mode not in ("R", "V", "D2R", "D2V", "R2V"):
            raise ValueError(f"unknown paging mode {mode!r}")
        self.mode = mode
        self.n_experts = n_experts
        self.ram_weights = ram_weights or {}
        self.disk_dir = disk_dir
        self.ram_cache_max_experts = ram_cache_max_experts
        self.staging_max_bytes = staging_max_bytes
        self.cache = collections.OrderedDict()  # expert_id -> weights (D2R)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bytes_transferred = 0
        self.load_time_s = 0.0
        self.peak_cached = 0
        self.peak_staging_bytes = 0
        self._pinned = set()
        self._dirty = set()  # modified D2R/D2V entries awaiting disk write-back
        if mode in ("V", "D2V", "R2V"):
            ok, reason = accelerator_available()
            if not ok:
                raise PagingError(
                    f"paging mode {mode} requires a compatible VRAM device backend: {reason}")
        if mode in ("D2R", "D2V") and not disk_dir:
            raise ValueError(f"paging mode {mode} requires checkpoint-backed disk storage (disk_dir)")
        if mode in ("R", "R2V") and not self.ram_weights:
            raise ValueError(f"paging mode {mode} requires RAM-resident experts")

    # -- lifecycle --
    def pin(self, eid: int) -> None:
        self._pinned.add(int(eid))

    def unpin(self, eid: int) -> None:
        self._pinned.discard(int(eid))

    def get(self, eid: int):
        """Return weight dict for expert eid, loading through the cache as needed."""
        eid = int(eid)
        if self.mode in ("R", "R2V", "V"):
            if eid not in self.ram_weights:
                raise KeyError(f"expert {eid} not resident (mode {self.mode})")
            self.hits += 1
            return self.ram_weights[eid]
        # D2R / D2V
        if eid in self.cache:
            self.hits += 1
            self.cache.move_to_end(eid)
            return self.cache[eid]
        self.misses += 1
        t0 = time.perf_counter()
        w = self._load_from_disk(eid)
        nbytes = sum(a.nbytes for a in w.values())
        if nbytes > self.staging_max_bytes:
            raise PagingError(
                f"expert {eid} staging size {nbytes} exceeds bound {self.staging_max_bytes}")
        self.peak_staging_bytes = max(self.peak_staging_bytes, nbytes)
        self.bytes_transferred += nbytes
        self.load_time_s += time.perf_counter() - t0
        self.cache[eid] = w
        self.peak_cached = max(self.peak_cached, len(self.cache))
        while len(self.cache) > self.ram_cache_max_experts:
            victim, _ = next(iter(self.cache.items()))
            if victim in self._pinned:
                # move pinned to end and evict next unpinned; never evict pinned
                self.cache.move_to_end(victim)
                victim2 = None
                for k in self.cache:
                    if k not in self._pinned:
                        victim2 = k
                        break
                if victim2 is None:
                    raise PagingError("cache full of pinned experts; cannot evict")
                self._flush_dirty(victim2)
                del self.cache[victim2]
                self.evictions += 1
            else:
                self._flush_dirty(victim)
                del self.cache[victim]
                self.evictions += 1
        return w

    def _load_from_disk(self, eid: int):
        path = os.path.join(self.disk_dir, f"expert_{eid}.npz")
        if not os.path.exists(path):
            raise KeyError(f"expert {eid} backing file missing: {path}")
        with np.load(path) as z:
            return {"Win": z["Win"].copy(), "Wout": z["Wout"].copy()}

    def _write_expert_file(self, eid: int, weights) -> None:
        path = os.path.join(self.disk_dir, f"expert_{eid}.npz")
        tmp = path + ".tmp"
        # NOTE: pass an open handle: np.savez appends '.npz' to string paths
        # that lack the extension, which would break the atomic replace.
        with open(tmp, "wb") as f:
            np.savez(f, **{k: v for k, v in weights.items()})
        os.replace(tmp, path)

    def _flush_dirty(self, eid: int) -> None:
        if eid in self._dirty and eid in self.cache:
            self._write_expert_file(eid, self.cache[eid])
            self._dirty.discard(eid)

    def write_back(self, eid: int, weights) -> None:
        """Synchronize an optimizer-updated expert without flushing the whole cache."""
        eid = int(eid)
        if self.mode in ("R", "R2V", "V"):
            # Model parameters are updated in place; keep direct references so
            # the paging layer doesn't make another full copy of each expert.
            self.ram_weights[eid] = weights
            return

        if eid in self.cache:
            entry = self.cache[eid]
            for key, value in weights.items():
                if key in entry and entry[key].shape == value.shape:
                    np.copyto(entry[key], value)
                else:
                    entry[key] = value.copy()
            self._dirty.add(eid)
        else:
            # It may have been evicted earlier in the same forward pass. Write
            # its authoritative optimizer-updated weights directly to backing.
            self._write_expert_file(eid, weights)

    def invalidate_all(self) -> None:
        if self.mode in ("D2R", "D2V"):
            for eid in list(self._dirty):
                self._flush_dirty(eid)
            self.cache.clear()
            self._dirty.clear()

    def register_expert(self, eid: int, weights) -> None:
        eid = int(eid)
        if self.mode in ("R", "R2V", "V"):
            self.ram_weights[eid] = weights
        else:
            os.makedirs(self.disk_dir, exist_ok=True)
            np.savez(os.path.join(self.disk_dir, f"expert_{eid}.npz"), **weights)
        self.n_experts = max(self.n_experts, eid + 1)

    def remove_expert(self, eid: int) -> None:
        eid = int(eid)
        self.ram_weights.pop(eid, None)
        self.cache.pop(eid, None)
        self._dirty.discard(eid)
        if self.disk_dir:
            try:
                os.remove(os.path.join(self.disk_dir, f"expert_{eid}.npz"))
            except OSError:
                pass

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "mode": self.mode,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "evictions": self.evictions,
            "bytes_transferred": self.bytes_transferred,
            "load_time_s": self.load_time_s,
            "peak_cached_experts": self.peak_cached,
            "peak_staging_bytes": self.peak_staging_bytes,
            "cache_bound": self.ram_cache_max_experts,
        }
