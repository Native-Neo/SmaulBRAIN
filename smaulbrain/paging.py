"""Bounded expert storage for resident RAM, disk-to-RAM, and CUDA modes."""
import collections, os, time
import torch

class PagingError(RuntimeError): pass


def accelerator_available():
    if torch.cuda.is_available():return True,"cuda"
    return False,"no compatible CUDA backend: torch.cuda.is_available()==False; system RAM/integrated graphics is not VRAM"


class ExpertStore:
    def __init__(self,mode,n_experts,ram_weights=None,disk_dir=None,ram_cache_max_experts=8,staging_max_bytes=1<<24):
        if mode not in ("R","V","D2R","D2V","R2V"):raise ValueError(f"unknown paging mode {mode!r}")
        self.mode=mode;self.n_experts=n_experts;self.ram_weights=ram_weights or {};self.disk_dir=disk_dir
        self.ram_cache_max_experts=int(ram_cache_max_experts);self.staging_max_bytes=int(staging_max_bytes)
        self.cache=collections.OrderedDict();self.hits=self.misses=self.evictions=self.bytes_transferred=0
        self.load_time_s=0.0;self.peak_cached=self.peak_staging_bytes=0;self._pinned=set();self._dirty=set()
        if mode in ("V","D2V","R2V"):
            ok,reason=accelerator_available()
            if not ok:raise PagingError(f"paging mode {mode} requires a compatible VRAM device backend: {reason}")
        if mode in ("D2R","D2V") and not disk_dir:raise ValueError(f"paging mode {mode} requires disk_dir")
        if mode in ("R","R2V") and not self.ram_weights:raise ValueError(f"paging mode {mode} requires RAM-resident experts")

    def pin(self,eid):self._pinned.add(int(eid))
    def unpin(self,eid):self._pinned.discard(int(eid))

    def get(self,eid):
        eid=int(eid)
        if self.mode in ("R","R2V","V"):
            if eid not in self.ram_weights:raise KeyError(f"expert {eid} not resident (mode {self.mode})")
            self.hits+=1;return self.ram_weights[eid]
        if eid in self.cache:self.hits+=1;self.cache.move_to_end(eid);return self.cache[eid]
        self.misses+=1;t=time.perf_counter();w=self._load_from_disk(eid)
        nbytes=sum(a.numel()*a.element_size() for a in w.values())
        if nbytes>self.staging_max_bytes:raise PagingError(f"expert {eid} staging size {nbytes} exceeds bound {self.staging_max_bytes}")
        self.peak_staging_bytes=max(self.peak_staging_bytes,nbytes);self.bytes_transferred+=nbytes;self.load_time_s+=time.perf_counter()-t
        self.cache[eid]=w;self.peak_cached=max(self.peak_cached,len(self.cache))
        while len(self.cache)>self.ram_cache_max_experts:
            victim=next(iter(self.cache))
            if victim in self._pinned:
                self.cache.move_to_end(victim);victim=next((k for k in self.cache if k not in self._pinned),None)
                if victim is None:raise PagingError("cache full of pinned experts; cannot evict")
            self._flush_dirty(victim);del self.cache[victim];self.evictions+=1
        return w

    def _path(self,eid):return os.path.join(self.disk_dir,f"expert_{eid}.pt")
    def _load_from_disk(self,eid):
        path=self._path(eid)
        if os.path.exists(path):return torch.load(path,map_location="cpu",weights_only=True)
        legacy=os.path.join(self.disk_dir,f"expert_{eid}.npz")
        if os.path.exists(legacy):
            import numpy as np
            with np.load(legacy) as z:return {k:torch.from_numpy(z[k].copy()) for k in ("Win","Wout")}
        raise KeyError(f"expert {eid} backing file missing: {path}")

    def _write_expert_file(self,eid,weights):
        os.makedirs(self.disk_dir,exist_ok=True);path=self._path(eid);tmp=path+f".{os.getpid()}.tmp"
        torch.save({k:v.detach().cpu().contiguous() for k,v in weights.items()},tmp);os.replace(tmp,path)

    def _flush_dirty(self,eid):
        if eid in self._dirty and eid in self.cache:self._write_expert_file(eid,self.cache[eid]);self._dirty.discard(eid)

    def write_back(self,eid,weights):
        eid=int(eid)
        if self.mode in ("R","R2V","V"):
            self.ram_weights[eid]=weights;return
        if eid in self.cache:
            entry=self.cache[eid]
            for k,v in weights.items():
                if k in entry and entry[k].shape==v.shape:entry[k].copy_(v.detach().cpu())
                else:entry[k]=v.detach().cpu().clone()
            self._dirty.add(eid)
        else:self._write_expert_file(eid,weights)

    def invalidate_all(self):
        if self.mode in ("D2R","D2V"):
            for eid in list(self._dirty):self._flush_dirty(eid)
            self.cache.clear();self._dirty.clear()

    def register_expert(self,eid,weights):
        eid=int(eid)
        if self.mode in ("R","R2V","V"):self.ram_weights[eid]=weights
        else:self._write_expert_file(eid,weights)
        self.n_experts=max(self.n_experts,eid+1)

    def remove_expert(self,eid):
        eid=int(eid);self.ram_weights.pop(eid,None);self.cache.pop(eid,None);self._dirty.discard(eid)
        for path in (self._path(eid),os.path.join(self.disk_dir or "",f"expert_{eid}.npz")):
            try:os.remove(path)
            except OSError:pass

    def stats(self):
        total=self.hits+self.misses
        return {"mode":self.mode,"hits":self.hits,"misses":self.misses,"hit_rate":self.hits/total if total else 0.0,
                "evictions":self.evictions,"bytes_transferred":self.bytes_transferred,"load_time_s":self.load_time_s,
                "peak_cached_experts":self.peak_cached,"peak_staging_bytes":self.peak_staging_bytes,"cache_bound":self.ram_cache_max_experts}
