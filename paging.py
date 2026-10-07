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
        self._lock = threading.Lock()  # protects cache dicts, stats, and metadata
        self._exec = ThreadPoolExecutor(max_workers=4)
        self._pending: dict[str, Future] = {}
        self._prefetched: set[str] = set()
        self._forgotten: set[str] = set()
        self._versions: dict[str, int] = {}
        self._in_flight: dict[str, threading.Event] = {}
        self._closed: bool = False
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

    # -- version tracking --
    def _get_version(self, expert_id: str) -> int:
        try:
            rec_ver = getattr(self.pool.experts.get(expert_id), "version", 0) or 0
        except Exception:
            rec_ver = 0
        return rec_ver + self._versions.get(expert_id, 0)

    # -- disk --
    def _read_disk(self, expert_id: str):
        """Authoritative disk read. Counts once per physical read. Runs without global lock."""
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten:
                raise KeyError(f"expert {expert_id!r} was pruned")
            self.stats.disk_reads += 1
        if self.load_from_disk is not None:
            rec = self.load_from_disk(expert_id)
        else:
            rec = self.pool.experts[expert_id]
        return rec

    # -- public API --
    def provider(self, expert_id: str) -> dict[str, torch.Tensor]:
        """WeightProvider for ExpertPool.forward: mode-specific fetch.

        Joins an in-flight prefetch for the same expert instead of loading
        it twice: the background fetch warms the caches as a side effect,
        so after the join the normal path below is a cache hit. The
        prefetch worker itself bypasses the join (thread-local flag) —
        joining your own future would deadlock.
        """
        while True:
            # 1. Join in-flight prefetch if present (outside global lock)
            if not getattr(self._tls, "prefetching", False):
                with self._lock:
                    fut = self._pending.pop(expert_id, None)
                if fut is not None:
                    try:
                        fut.result()
                    except Exception:
                        with self._lock:
                            self._prefetched.discard(expert_id)
                        raise
                    finally:
                        with self._lock:
                            self._prefetched.discard(expert_id)

            # 2. Check cache hit under lock
            with self._lock:
                if self._closed:
                    raise RuntimeError("pager is closed")
                if expert_id in self._forgotten:
                    raise KeyError(f"expert {expert_id!r} was pruned")

                if self.mode == "D2R" and expert_id in self.ram:
                    self.stats.ram_hits += 1
                    self.ram.move_to_end(expert_id)
                    return self.ram[expert_id]
                elif self.mode in ("R2VR", "D2VR") and expert_id in self.vram:
                    self.stats.vram_hits += 1
                    self.vram.move_to_end(expert_id)
                    return self.vram[expert_id]

                # If another consumer thread is already loading this expert, wait for it
                if not getattr(self._tls, "prefetching", False) and expert_id in self._in_flight:
                    event = self._in_flight[expert_id]
                else:
                    event = None
                    if not getattr(self._tls, "prefetching", False):
                        self._in_flight[expert_id] = threading.Event()
                    break

            if event is not None:
                event.wait()

        # 3. Perform load/dequantization without holding global lock
        evt_to_set = self._in_flight.get(expert_id) if not getattr(self._tls, "prefetching", False) else None
        try:
            ver_before = self._get_version(expert_id)
            if self.mode == "D2R":
                w = self._load_d2r(expert_id, ver_before)
            elif self.mode == "R2VR":
                w = self._load_r2vr(expert_id, ver_before)
            else:
                w = self._load_d2vr(expert_id, ver_before)
            return w
        finally:
            if evt_to_set is not None:
                with self._lock:
                    self._in_flight.pop(expert_id, None)
                evt_to_set.set()

    def _load_d2r(self, expert_id: str, ver_before: int) -> dict[str, torch.Tensor]:
        rec = self._read_disk(expert_id)
        w = {k: v.to(self.compute_dtype) for k, v in rec.dequantize(torch.float32).items()}
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten or expert_id not in self.pool.experts:
                raise KeyError(f"expert {expert_id!r} was pruned")
            if self._get_version(expert_id) != ver_before:
                rec = self.pool.experts[expert_id]
                w = {k: v.to(self.compute_dtype) for k, v in rec.dequantize(torch.float32).items()}
            if expert_id in self.ram:
                return self.ram[expert_id]
            self.ram[expert_id] = w
            while len(self.ram) > self.ram_cache:
                self.ram.popitem(last=False)
                self.stats.ram_evictions += 1
            assert len(self.vram) == 0, "D2R path must never populate VRAM"
            return w

    def _load_r2vr(self, expert_id: str, ver_before: int) -> dict[str, torch.Tensor]:
        with self._lock:
            in_staged = expert_id in self.ram_records
        if not in_staged:
            rec = self._read_disk(expert_id)
            with self._lock:
                if expert_id not in self._forgotten:
                    self.stats.ram_loads += 1
                    self.ram_records[expert_id] = rec
                    self.ram_records.move_to_end(expert_id)
                    while len(self.ram_records) > self.max_staged:
                        self.ram_records.popitem(last=False)
        with self._lock:
            if expert_id in self._forgotten or expert_id not in self.ram_records:
                raise KeyError(f"expert {expert_id!r} was pruned")
            self.ram_records.move_to_end(expert_id)
            rec = self.ram_records[expert_id]

        w = self._to_vram({k: v.to(self.compute_dtype)
                           for k, v in rec.dequantize(torch.float32).items()})
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten or expert_id not in self.pool.experts:
                raise KeyError(f"expert {expert_id!r} was pruned")
            if self._get_version(expert_id) != ver_before:
                rec = self.pool.experts[expert_id]
                w = self._to_vram({k: v.to(self.compute_dtype)
                                   for k, v in rec.dequantize(torch.float32).items()})
            if expert_id in self.vram:
                return self.vram[expert_id]
            self.vram[expert_id] = w
            self.stats.vram_loads += 1
            while len(self.vram) > self.vram_cache:
                self.vram.popitem(last=False)
                self.stats.vram_evictions += 1
            return w

    def _load_d2vr(self, expert_id: str, ver_before: int) -> dict[str, torch.Tensor]:
        rec = self._read_disk(expert_id)
        w = self._to_vram({k: v.to(self.compute_dtype)
                           for k, v in rec.dequantize(torch.float32).items()})
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten or expert_id not in self.pool.experts:
                raise KeyError(f"expert {expert_id!r} was pruned")
            if self._get_version(expert_id) != ver_before:
                rec = self.pool.experts[expert_id]
                w = self._to_vram({k: v.to(self.compute_dtype)
                                   for k, v in rec.dequantize(torch.float32).items()})
            if expert_id in self.vram:
                return self.vram[expert_id]
            self.vram[expert_id] = w
            self.stats.vram_loads += 1
            while len(self.vram) > self.vram_cache:
                self.vram.popitem(last=False)
                self.stats.vram_evictions += 1
            return w

    def invalidate(self, expert_id: str) -> None:
        """Drop cached compute weights after an optimizer rewrite.

        The RAM stage keeps pointing at the pool's live record (updated in
        place by the optimizer) instead of re-reading a stale disk file.
        """
        with self._lock:
            self._versions[expert_id] = self._versions.get(expert_id, 0) + 1
            if expert_id in self.pool.experts:
                rec = self.pool.experts[expert_id]
                rec.version = getattr(rec, "version", 0) + 1
            self.ram.pop(expert_id, None)
            self.vram.pop(expert_id, None)
            if expert_id in self.ram_records and expert_id in self.pool.experts:
                self.ram_records[expert_id] = self.pool.experts[expert_id]

    def forget(self, expert_id: str) -> None:
        """Drop every cached/staged trace of a pruned expert.

        Weights, optimizer state, and router rows are already gone; without
        this the caches would keep serving a ghost expert that no longer
        exists in the pool.
        """
        with self._lock:
            self._forgotten.add(expert_id)
            self._versions[expert_id] = self._versions.get(expert_id, 0) + 1
            self.ram.pop(expert_id, None)
            self.vram.pop(expert_id, None)
            self.ram_records.pop(expert_id, None)
            fut = self._pending.pop(expert_id, None)
            if fut is not None:
                try:
                    fut.cancel()
                except Exception:
                    pass
            self._prefetched.discard(expert_id)

    def _stage(self, expert_id: str):
        """Disk -> RAM staging with LRU bound; evicted records re-read later."""
        rec = self._read_disk(expert_id)
        with self._lock:
            if expert_id not in self._forgotten:
                self.stats.ram_loads += 1
                self.ram_records[expert_id] = rec
                self.ram_records.move_to_end(expert_id)
                while len(self.ram_records) > self.max_staged:
                    self.ram_records.popitem(last=False)

    # -- R2VR: RAM -> VRAM (disk only via warm_ram) --
    def warm_ram(self) -> None:
        """Bulk-stage expert records from disk into RAM at startup."""
        for eid in list(self.pool.order):
            self._stage(eid)

    # -- async prefetch --
    def prefetch(self, expert_ids: list[str]) -> None:
        """Begin loading predicted experts in the background. Collapses duplicates."""
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            unique_ids = []
            seen = set()
            for eid in expert_ids:
                if eid not in seen:
                    seen.add(eid)
                    unique_ids.append(eid)

            for eid in unique_ids:
                if eid in self._forgotten:
                    continue
                if eid in self.ram or eid in self.vram:
                    continue
                if eid in self._pending:
                    continue  # already in flight: collapsed!
                expected_ver = self._get_version(eid)
                self.stats.prefetch_submitted += 1
                self._pending[eid] = self._exec.submit(self._prefetch_one, eid, expected_ver)

    def _prefetch_one(self, expert_id: str, expected_version: int) -> None:
        self._tls.prefetching = True
        try:
            with self._lock:
                if self._closed or expert_id in self._forgotten or expert_id not in self.pool.experts:
                    return
            try:
                w = self.provider(expert_id)
            except KeyError:
                with self._lock:
                    if expert_id in self.pool.experts and expert_id not in self._forgotten:
                        raise
                return

            with self._lock:
                current_ver = self._get_version(expert_id)
                if self._closed or expert_id in self._forgotten or current_ver != expected_version:
                    self.ram.pop(expert_id, None)
                    self.vram.pop(expert_id, None)
                    self.ram_records.pop(expert_id, None)
                    return
                self._prefetched.add(expert_id)
            _ = w
        finally:
            self._tls.prefetching = False

    def await_prefetch(self, timeout: float | None = None) -> None:
        with self._lock:
            items = list(self._pending.items())
        errors = []
        for eid, fut in items:
            try:
                fut.result(timeout=timeout)
                with self._lock:
                    if eid in self._prefetched:
                        self.stats.prefetch_hits += 1
            except Exception as e:
                errors.append((eid, e))
            finally:
                with self._lock:
                    self._pending.pop(eid, None)
                    self._prefetched.discard(eid)
        if errors:
            raise errors[0][1]

    def resident_counts(self) -> dict:
        with self._lock:
            for d in (self.ram, self.vram, self.ram_records):
                stale = [eid for eid in d if eid in self._forgotten or eid not in self.pool.experts]
                for eid in stale:
                    d.pop(eid, None)
            return {
                "ram": len(self.ram),
                "vram": len(self.vram),
                "ram_staged": len(self.ram_records),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending.values())
            self._pending.clear()
            self._prefetched.clear()
        for fut in pending:
            try:
                fut.cancel()
            except Exception:
                pass
        self._exec.shutdown(wait=True)
