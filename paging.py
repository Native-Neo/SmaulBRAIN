"""Expert paging: D2R / R2VR / D2VR as three genuinely different paths.

  D   = disk (per-expert files, via a loader callable)
  R   = RAM cache (LRU, ``ram_cache`` slots of dequantized compute weights)
  VR  = VRAM cache (LRU, ``vram_cache`` slots; CUDA tensors when available,
        otherwise explicitly tagged CPU-side VRAM simulation)

  D2R:  disk -> RAM; compute reads the RAM cache. VRAM is never touched.
  R2VR: disk -> RAM (bulk ``warm_ram`` at startup, on-demand staging after),
        then RAM -> VRAM per use; VRAM is only ever fed from staged RAM.
  D2VR: disk -> VRAM directly, bypassing RAM; the RAM cache stays empty.

Mode-specific counters prove which path executed (tests assert this).
Expert identity (stable id) is independent of cache slots, and optimizer
state lives on the ``ExpertRecord`` so it follows the expert, never a slot.

Prefetch: ``prefetch([ids])`` loads predicted experts asynchronously on a
background thread while the current expert computes.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import threading

import torch


@dataclass
class PagingStats:
    disk_reads: int = 0
    ram_hits: int = 0
    ram_evictions: int = 0
    ram_loads: int = 0
    vram_hits: int = 0
    vram_evictions: int = 0
    vram_loads: int = 0
    prefetch_submitted: int = 0
    prefetch_hits: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "disk_reads", "ram_hits", "ram_evictions", "ram_loads",
            "vram_hits", "vram_evictions", "vram_loads",
            "prefetch_submitted", "prefetch_hits")}


class ExpertPager:
    """Three-mode pager over one expert pool. Mode fixed at construction."""

    def __init__(
        self,
        pool,  # ExpertPool
        mode: str = "D2R",
        ram_cache: int = 8,
        vram_cache: int = 4,
        compute_dtype: torch.dtype = torch.bfloat16,
        load_from_disk=None,  # (expert_id) -> ExpertRecord (real file IO)
        max_staged: int = 128,  # R2VR RAM-staging ceiling (LRU; misses re-read)
    ) -> None:
        assert mode in ("D2R", "R2VR", "D2VR"), f"unknown paging mode {mode}"
        self.pool = pool
        self.mode = mode
        self.ram_cache = ram_cache
        self.vram_cache = vram_cache
        self.compute_dtype = compute_dtype
        self.load_from_disk = load_from_disk
        self.max_staged = max_staged
        self.ram: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self.vram: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self.ram_records: OrderedDict = OrderedDict()  # R2VR staging area
        self.stats = PagingStats()
        self._lock = threading.Lock()  # prefetch thread vs mutating main thread
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._pending: dict[str, Future] = {}
        self._prefetched: set[str] = set()
        self._tls = threading.local()  # marks the prefetch worker thread

    # -- device handling --
    @property
    def vram_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")  # tagged simulation; residency still separate

    def _to_vram(self, w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dev = self.vram_device
        return {k: (v.to(dev) if v.device != dev else v) for k, v in w.items()}

    # -- disk --
    def _read_disk(self, expert_id: str):
        """Authoritative disk read. Counts once per physical read."""
        self.stats.disk_reads += 1
        if self.load_from_disk is not None:
            return self.load_from_disk(expert_id)
        return self.pool.experts[expert_id]

    # -- public API --
    def provider(self, expert_id: str) -> dict[str, torch.Tensor]:
        """WeightProvider for ExpertPool.forward: mode-specific fetch.

        Joins an in-flight prefetch for the same expert instead of loading
        it twice: the background fetch warms the caches as a side effect,
        so after the join the normal path below is a cache hit. The
        prefetch worker itself bypasses the join (thread-local flag) —
        joining your own future would deadlock.
        """
        if not getattr(self._tls, "prefetching", False):
            with self._lock:
                fut = self._pending.pop(expert_id, None)
            if fut is not None:
                try:
                    fut.result()
                finally:
                    with self._lock:
                        self._prefetched.discard(expert_id)
        with self._lock:
            if self.mode == "D2R":
                return self._get_d2r(expert_id)
            if self.mode == "R2VR":
                return self._get_r2vr(expert_id)
            return self._get_d2vr(expert_id)

    def invalidate(self, expert_id: str) -> None:
        """Drop cached compute weights after an optimizer rewrite.

        The RAM stage keeps pointing at the pool's live record (updated in
        place by the optimizer) instead of re-reading a stale disk file.
        """
        with self._lock:
            self.ram.pop(expert_id, None)
            self.vram.pop(expert_id, None)
            if expert_id in self.ram_records:
                self.ram_records[expert_id] = self.pool.experts[expert_id]

    def forget(self, expert_id: str) -> None:
        """Drop every cached/staged trace of a pruned expert.

        Weights, optimizer state, and router rows are already gone; without
        this the caches would keep serving a ghost expert that no longer
        exists in the pool.
        """
        with self._lock:
            self.ram.pop(expert_id, None)
            self.vram.pop(expert_id, None)
            self.ram_records.pop(expert_id, None)
            # In-flight background loads must not resurrect the pruned
            # expert: drop (and cancel where still queued) its prefetch.
            fut = self._pending.pop(expert_id, None)
            if fut is not None:
                try:
                    fut.cancel()
                except Exception:
                    pass
            self._prefetched.discard(expert_id)

    # -- D2R: disk -> RAM --
    def _get_d2r(self, expert_id: str) -> dict[str, torch.Tensor]:
        if expert_id in self.ram:
            self.stats.ram_hits += 1
            self.ram.move_to_end(expert_id)
            return self.ram[expert_id]
        rec = self._read_disk(expert_id)
        w = {k: v.to(self.compute_dtype) for k, v in rec.dequantize(torch.float32).items()}
        self.ram[expert_id] = w
        while len(self.ram) > self.ram_cache:
            self.ram.popitem(last=False)
            self.stats.ram_evictions += 1
        assert len(self.vram) == 0, "D2R path must never populate VRAM"
        return w

    def _stage(self, expert_id: str):
        """Disk -> RAM staging with LRU bound; evicted records re-read later."""
        rec = self._read_disk(expert_id)
        self.stats.ram_loads += 1
        self.ram_records[expert_id] = rec
        self.ram_records.move_to_end(expert_id)
        while len(self.ram_records) > self.max_staged:
            self.ram_records.popitem(last=False)

    # -- R2VR: RAM -> VRAM (disk only via warm_ram) --
    def warm_ram(self) -> None:
        """Bulk-stage expert records from disk into RAM at startup.

        Later misses stage on demand through the same disk -> RAM path;
        VRAM is only ever fed from staged RAM records, never from disk.
        Staging never exceeds max_staged entries (LRU); evicted records
        are re-read from disk on their next miss.
        """
        for eid in self.pool.order:
            self._stage(eid)

    def _get_r2vr(self, expert_id: str) -> dict[str, torch.Tensor]:
        if expert_id in self.vram:
            self.stats.vram_hits += 1
            self.vram.move_to_end(expert_id)
            return self.vram[expert_id]
        if expert_id not in self.ram_records:
            # On-demand staging disk -> RAM (supports post-growth experts).
            # The RAM stage is never skipped: VRAM is only ever fed from RAM.
            self._stage(expert_id)
        else:
            self.ram_records.move_to_end(expert_id)
        rec = self.ram_records[expert_id]  # RAM-staged record, no disk IO below
        w = self._to_vram({k: v.to(self.compute_dtype)
                           for k, v in rec.dequantize(torch.float32).items()})
        self.vram[expert_id] = w
        self.stats.vram_loads += 1
        while len(self.vram) > self.vram_cache:
            self.vram.popitem(last=False)
            self.stats.vram_evictions += 1
        return w

    # -- D2VR: disk -> VRAM, bypassing RAM --
    def _get_d2vr(self, expert_id: str) -> dict[str, torch.Tensor]:
        if expert_id in self.vram:
            self.stats.vram_hits += 1
            self.vram.move_to_end(expert_id)
            return self.vram[expert_id]
        rec = self._read_disk(expert_id)
        w = self._to_vram({k: v.to(self.compute_dtype)
                           for k, v in rec.dequantize(torch.float32).items()})
        self.vram[expert_id] = w
        self.stats.vram_loads += 1
        while len(self.vram) > self.vram_cache:
            self.vram.popitem(last=False)
            self.stats.vram_evictions += 1
        return w

    # -- async prefetch --
    def prefetch(self, expert_ids: list[str]) -> None:
        """Begin loading predicted experts in the background."""
        for eid in expert_ids:
            if eid in self.ram or eid in self.vram or eid in self._pending:
                continue
            self.stats.prefetch_submitted += 1
            self._pending[eid] = self._exec.submit(self._prefetch_one, eid)

    def _prefetch_one(self, expert_id: str) -> None:
        self._tls.prefetching = True
        try:
            try:
                w = self.provider(expert_id)
            except KeyError:
                # Pruned mid-flight: the expert is gone from the pool, so there
                # is nothing to stage. A KeyError for a live expert is a real
                # bug and must still surface via the future.
                if expert_id in self.pool.experts:
                    raise
                return
        finally:
            self._tls.prefetching = False
        self._prefetched.add(expert_id)
        _ = w

    def await_prefetch(self, timeout: float | None = None) -> None:
        for eid, fut in list(self._pending.items()):
            fut.result(timeout=timeout)
            if eid in self._prefetched:
                self.stats.prefetch_hits += 1
            del self._pending[eid]
        self._prefetched.clear()

    def resident_counts(self) -> dict:
        return {"ram": len(self.ram), "vram": len(self.vram),
                "ram_staged": len(self.ram_records)}

    def close(self) -> None:
        self._exec.shutdown(wait=True)
