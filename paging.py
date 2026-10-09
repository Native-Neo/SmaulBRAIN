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
        if mode not in ("D2R", "R2VR", "D2VR"):
            raise ValueError(f"unknown paging mode {mode}")
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
        # Version+identity tags for cached compute weights and staged records.
        # Each tag is (version, id(record)) captured at insertion; hits validate
        # against the live pool so replaced/updated experts can never be served
        # stale, and stale prefetches only evict entries they actually wrote.
        self._ram_tag: dict[str, tuple[int, int]] = {}
        self._vram_tag: dict[str, tuple[int, int]] = {}
        self._staged_tag: dict[str, tuple[int, int]] = {}
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

    def _live_tag(self, expert_id: str) -> tuple[int, int] | None:
        """Live (version, identity) for ``expert_id``; None if absent.

        Reads the pool without the pager lock (pool mutation never holds the
        pager lock, so holding it would not exclude writers anyway). Callers
        combine this with the pager lock to make check+insert atomic against
        invalidate/forget/restore, which do hold the pager lock.
        """
        try:
            rec = self.pool.experts.get(expert_id)
        except Exception:
            return None
        if rec is None:
            return None
        try:
            rec_ver = getattr(rec, "version", 0) or 0
        except Exception:
            rec_ver = 0
        return (rec_ver + self._versions.get(expert_id, 0), id(rec))

    def snapshot_stats(self) -> dict:
        """Thread-safe statistics snapshot (locked copy for contention)."""
        with self._lock:
            return self.stats.to_dict()

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

            # 2. Check cache hit under lock (version+identity validated: a
            # replaced or concurrently updated expert never serves stale).
            with self._lock:
                if self._closed:
                    raise RuntimeError("pager is closed")
                if expert_id in self._forgotten:
                    raise KeyError(f"expert {expert_id!r} was pruned")

                if self.mode == "D2R" and expert_id in self.ram:
                    live = self._live_tag(expert_id)
                    if live is not None and self._ram_tag.get(expert_id) == live:
                        self.stats.ram_hits += 1
                        self.ram.move_to_end(expert_id)
                        return self.ram[expert_id]
                    # Stale (updated/replaced) entry: drop and reload fresh.
                    self.ram.pop(expert_id, None)
                    self._ram_tag.pop(expert_id, None)
                elif self.mode in ("R2VR", "D2VR") and expert_id in self.vram:
                    live = self._live_tag(expert_id)
                    if live is not None and self._vram_tag.get(expert_id) == live:
                        self.stats.vram_hits += 1
                        self.vram.move_to_end(expert_id)
                        return self.vram[expert_id]
                    self.vram.pop(expert_id, None)
                    self._vram_tag.pop(expert_id, None)

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
            tag_before = self._live_tag(expert_id)
            ver_before = tag_before[0] if tag_before is not None else self._get_version(expert_id)
            if self.mode == "D2R":
                w = self._load_d2r(expert_id, tag_before)
            elif self.mode == "R2VR":
                w = self._load_r2vr(expert_id, tag_before)
            else:
                w = self._load_d2vr(expert_id, tag_before)
            return w
        finally:
            if evt_to_set is not None:
                with self._lock:
                    self._in_flight.pop(expert_id, None)
                evt_to_set.set()

    def _insert_ram_locked(self, expert_id: str, w: dict, tag: tuple[int, int]) -> dict:
        """Insert into RAM LRU under lock; return resident entry (fresh wins)."""
        existing = self._ram_tag.get(expert_id)
        if expert_id in self.ram and existing == tag:
            return self.ram[expert_id]
        # Stale resident (if any) loses to the fresh write.
        self.ram[expert_id] = w
        self._ram_tag[expert_id] = tag
        self.ram.move_to_end(expert_id)
        while len(self.ram) > self.ram_cache:
            old, _ = self.ram.popitem(last=False)
            self._ram_tag.pop(old, None)
            self.stats.ram_evictions += 1
        # Structural invariant (ValueError, not assert: must hold under -O
        # too): the D2R path never populates VRAM.
        if len(self.vram) != 0:
            raise ValueError("D2R path must never populate VRAM")
        return w

    def _insert_vram_locked(self, expert_id: str, w: dict, tag: tuple[int, int]) -> dict:
        """Insert into VRAM LRU under lock; return resident entry (fresh wins)."""
        existing = self._vram_tag.get(expert_id)
        if expert_id in self.vram and existing == tag:
            return self.vram[expert_id]
        # If a fresher entry landed first, keep it instead of overwriting.
        if expert_id in self.vram and existing is not None and existing != tag:
            live = self._live_tag(expert_id)
            if live is not None and existing == live:
                return self.vram[expert_id]
        self.vram[expert_id] = w
        self._vram_tag[expert_id] = tag
        self.stats.vram_loads += 1
        while len(self.vram) > self.vram_cache:
            old, _ = self.vram.popitem(last=False)
            self._vram_tag.pop(old, None)
            self.stats.vram_evictions += 1
        return w

    def _load_d2r(self, expert_id: str, tag_before: tuple[int, int] | None) -> dict[str, torch.Tensor]:
        rec = self._read_disk(expert_id)
        w = {k: v.to(self.compute_dtype) for k, v in rec.dequantize(torch.float32).items()}
        # Fine-grained: dequantize never holds the global lock. Revalidate the
        # live tag under lock; a concurrent update/replace bumps version or
        # identity so stale reads recompute from live instead of surviving.
        # A stale disk file (lagging a prior update) is also detected: when the
        # disk snapshot is not the live object and the live tag is dirty, the
        # live pool is authoritative.
        for _ in range(8):
            with self._lock:
                if self._closed:
                    raise RuntimeError("pager is closed")
                if expert_id in self._forgotten or expert_id not in self.pool.experts:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                cur = self._live_tag(expert_id)
                if cur is None:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                live_rec = self.pool.experts[expert_id]
                stale_concurrent = (cur != tag_before)
                stale_file = (live_rec is not rec and cur[0] != 0)
                if not stale_concurrent and not stale_file:
                    return self._insert_ram_locked(expert_id, w, cur)
                refresh_rec = live_rec
                refresh_tag = cur
            w = {k: v.to(self.compute_dtype)
                 for k, v in refresh_rec.dequantize(torch.float32).items()}
            rec = refresh_rec
            tag_before = refresh_tag
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten or expert_id not in self.pool.experts:
                raise KeyError(f"expert {expert_id!r} was pruned")
            cur = self._live_tag(expert_id)
            if cur is None:
                raise KeyError(f"expert {expert_id!r} was pruned")
            return self._insert_ram_locked(expert_id, w, cur)

    def _load_r2vr(self, expert_id: str, tag_before: tuple[int, int] | None) -> dict[str, torch.Tensor]:
        # Staging uses the live tag: a staged entry whose tag mismatches live
        # is refreshed to the live record without a disk read; a stale disk
        # snapshot never overwrites a fresher staged entry.
        with self._lock:
            if expert_id in self.ram_records and expert_id in self.pool.experts \
                    and expert_id not in self._forgotten:
                live = self._live_tag(expert_id)
                if live is not None and self._staged_tag.get(expert_id) != live:
                    self.ram_records[expert_id] = self.pool.experts[expert_id]
                    self._staged_tag[expert_id] = live
                    self.ram_records.move_to_end(expert_id)
            in_staged = expert_id in self.ram_records
        if not in_staged:
            rec = self._read_disk(expert_id)
            with self._lock:
                if self._closed:
                    raise RuntimeError("pager is closed")
                if expert_id in self._forgotten or expert_id not in self.pool.experts:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                if expert_id in self.ram_records:
                    # Another thread staged fresh first: keep it, drop ours.
                    pass
                else:
                    cur = self._live_tag(expert_id)
                    live_rec = self.pool.experts[expert_id]
                    if cur is not None and (cur != tag_before or
                            (live_rec is not rec and cur[0] != 0)):
                        # Disk snapshot went stale during the read: stage live.
                        self.ram_records[expert_id] = live_rec
                        self._staged_tag[expert_id] = cur
                    else:
                        self.stats.ram_loads += 1
                        self.ram_records[expert_id] = rec
                        if cur is not None:
                            self._staged_tag[expert_id] = cur
                    self.ram_records.move_to_end(expert_id)
                    while len(self.ram_records) > self.max_staged:
                        old, _ = self.ram_records.popitem(last=False)
                        self._staged_tag.pop(old, None)
        with self._lock:
            if expert_id in self._forgotten or expert_id not in self.ram_records:
                raise KeyError(f"expert {expert_id!r} was pruned")
            # Refresh a staged entry that went stale while we were away.
            if expert_id in self.pool.experts and expert_id not in self._forgotten:
                live = self._live_tag(expert_id)
                if live is not None and self._staged_tag.get(expert_id) != live:
                    self.ram_records[expert_id] = self.pool.experts[expert_id]
                    self._staged_tag[expert_id] = live
            self.ram_records.move_to_end(expert_id)
            rec = self.ram_records[expert_id]
            staged_tag = self._staged_tag.get(expert_id)

        w = self._to_vram({k: v.to(self.compute_dtype)
                           for k, v in rec.dequantize(torch.float32).items()})
        for _ in range(8):
            with self._lock:
                if self._closed:
                    raise RuntimeError("pager is closed")
                if expert_id in self._forgotten or expert_id not in self.pool.experts:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                cur = self._live_tag(expert_id)
                if cur is None:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                if cur == tag_before and cur == staged_tag:
                    return self._insert_vram_locked(expert_id, w, cur)
                refresh_rec = self.pool.experts[expert_id]
                refresh_tag = cur
            w = self._to_vram({k: v.to(self.compute_dtype)
                               for k, v in refresh_rec.dequantize(torch.float32).items()})
            tag_before = refresh_tag
            staged_tag = refresh_tag
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten or expert_id not in self.pool.experts:
                raise KeyError(f"expert {expert_id!r} was pruned")
            cur = self._live_tag(expert_id)
            if cur is None:
                raise KeyError(f"expert {expert_id!r} was pruned")
            return self._insert_vram_locked(expert_id, w, cur)

    def _load_d2vr(self, expert_id: str, tag_before: tuple[int, int] | None) -> dict[str, torch.Tensor]:
        rec = self._read_disk(expert_id)
        w = self._to_vram({k: v.to(self.compute_dtype)
                           for k, v in rec.dequantize(torch.float32).items()})
        for _ in range(8):
            with self._lock:
                if self._closed:
                    raise RuntimeError("pager is closed")
                if expert_id in self._forgotten or expert_id not in self.pool.experts:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                cur = self._live_tag(expert_id)
                if cur is None:
                    raise KeyError(f"expert {expert_id!r} was pruned")
                live_rec = self.pool.experts[expert_id]
                stale_concurrent = (cur != tag_before)
                stale_file = (live_rec is not rec and cur[0] != 0)
                if not stale_concurrent and not stale_file:
                    return self._insert_vram_locked(expert_id, w, cur)
                refresh_rec = live_rec
                refresh_tag = cur
            w = self._to_vram({k: v.to(self.compute_dtype)
                               for k, v in refresh_rec.dequantize(torch.float32).items()})
            rec = refresh_rec
            tag_before = refresh_tag
        with self._lock:
            if self._closed:
                raise RuntimeError("pager is closed")
            if expert_id in self._forgotten or expert_id not in self.pool.experts:
                raise KeyError(f"expert {expert_id!r} was pruned")
            cur = self._live_tag(expert_id)
            if cur is None:
                raise KeyError(f"expert {expert_id!r} was pruned")
            return self._insert_vram_locked(expert_id, w, cur)

    def invalidate(self, expert_id: str) -> None:
        """Drop cached compute weights after an optimizer rewrite.

        The RAM stage keeps pointing at the pool's live record (updated in
        place by the optimizer) instead of re-reading a stale disk file.
        A previously prefetched flag for this expert is discarded: the cached
        bytes it counted are gone, so a later await must not count a ghost hit.
        """
        with self._lock:
            self._versions[expert_id] = self._versions.get(expert_id, 0) + 1
            if expert_id in self.pool.experts:
                rec = self.pool.experts[expert_id]
                rec.version = getattr(rec, "version", 0) + 1
            self.ram.pop(expert_id, None)
            self._ram_tag.pop(expert_id, None)
            self.vram.pop(expert_id, None)
            self._vram_tag.pop(expert_id, None)
            self._prefetched.discard(expert_id)
            if expert_id in self.ram_records and expert_id in self.pool.experts:
                self.ram_records[expert_id] = self.pool.experts[expert_id]
                live = self._live_tag(expert_id)
                if live is not None:
                    self._staged_tag[expert_id] = live

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
            self._ram_tag.pop(expert_id, None)
            self.vram.pop(expert_id, None)
            self._vram_tag.pop(expert_id, None)
            self.ram_records.pop(expert_id, None)
            self._staged_tag.pop(expert_id, None)
            fut = self._pending.pop(expert_id, None)
            if fut is not None:
                try:
                    fut.cancel()
                except Exception:
                    pass
            self._prefetched.discard(expert_id)

    def restore(self, expert_id: str) -> None:
        """Clear the pruned marker so a restored (re-added) id is readable.

        Pruning calls forget(); re-adding the same id without clearing the
        forgotten-set would leave it permanently unreadable. Restoring drops
        the marker (caches are already empty from forget) without resetting
        the monotonic version, so older async results still mismatch.
        """
        with self._lock:
            self._forgotten.discard(expert_id)

    # Alias for callers that think in prune/restore vs forget/remember terms.
    remember = restore

    def _stage(self, expert_id: str):
        """Disk -> RAM staging with LRU bound; evicted records re-read later."""
        tag_before = self._live_tag(expert_id)
        rec = self._read_disk(expert_id)
        with self._lock:
            if self._closed or expert_id in self._forgotten:
                return
            if expert_id not in self.pool.experts:
                return
            if expert_id in self.ram_records:
                return  # fresher staged entry already present: keep it
            cur = self._live_tag(expert_id)
            live_rec = self.pool.experts.get(expert_id)
            if cur is not None and (cur != tag_before or
                    (live_rec is not None and live_rec is not rec and cur[0] != 0)):
                # Snapshot went stale during the read: stage live instead.
                if live_rec is not None:
                    self.ram_records[expert_id] = live_rec
                    self._staged_tag[expert_id] = cur
                    self.ram_records.move_to_end(expert_id)
                    while len(self.ram_records) > self.max_staged:
                        old, _ = self.ram_records.popitem(last=False)
                        self._staged_tag.pop(old, None)
                    return
            self.stats.ram_loads += 1
            self.ram_records[expert_id] = rec
            if cur is not None:
                self._staged_tag[expert_id] = cur
            self.ram_records.move_to_end(expert_id)
            while len(self.ram_records) > self.max_staged:
                old, _ = self.ram_records.popitem(last=False)
                self._staged_tag.pop(old, None)

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
                # Tag-validated residency: a stale entry does not suppress a
                # fresh prefetch.
                live = self._live_tag(eid)
                if eid in self.ram:
                    if live is not None and self._ram_tag.get(eid) == live:
                        continue
                if eid in self.vram:
                    if live is not None and self._vram_tag.get(eid) == live:
                        continue
                if eid in self._pending:
                    continue  # already in flight: collapsed!
                expected = self._live_tag(eid)
                self.stats.prefetch_submitted += 1
                self._pending[eid] = self._exec.submit(self._prefetch_one, eid, expected)

    def _evict_if_stale(self, expert_id: str, expected: tuple[int, int] | None) -> None:
        """Remove only the stale entry this prefetch wrote; preserve fresh."""
        if self.mode == "D2R":
            if self._ram_tag.get(expert_id) == expected:
                self.ram.pop(expert_id, None)
                self._ram_tag.pop(expert_id, None)
        else:
            if self._vram_tag.get(expert_id) == expected:
                self.vram.pop(expert_id, None)
                self._vram_tag.pop(expert_id, None)
        if self._staged_tag.get(expert_id) == expected:
            # Only drop the staged snapshot when live has moved on; a fresh
            # stage inserted after us must survive.
            live = self._live_tag(expert_id)
            if live != expected:
                self.ram_records.pop(expert_id, None)
                self._staged_tag.pop(expert_id, None)

    def _prefetch_one(self, expert_id: str, expected: tuple[int, int] | None) -> None:
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
                if self._closed or expert_id in self._forgotten:
                    self._evict_if_stale(expert_id, expected)
                    return
                current = self._live_tag(expert_id)
                if current != expected:
                    # An update/replace/prune won the race. The inner provider
                    # already revalidated, so a fresh entry (if any) must be
                    # preserved; only the stale write (if still resident) goes.
                    self._evict_if_stale(expert_id, expected)
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
            for d, t in ((self.ram, self._ram_tag),
                         (self.vram, self._vram_tag),
                         (self.ram_records, self._staged_tag)):
                stale = [eid for eid in d if eid in self._forgotten or eid not in self.pool.experts]
                for eid in stale:
                    d.pop(eid, None)
                    if t is not None:
                        t.pop(eid, None)
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
