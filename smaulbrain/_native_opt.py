"""Optional native fused SmaulOPT update (see _optimizer_kernels.cpp).

Self-contained: builds its own cached shared object and never raises. If no
compiler is available, SMAULBRAIN_NATIVE_OPT=0 is set, or a tensor does not
qualify (dtype/layout), callers fall back to the NumPy reference path.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import cpu as _cpu

_LIB = None
_TRIED = False


def _load():
    global _LIB, _TRIED
    if _LIB is not None or _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("SMAULBRAIN_NATIVE_OPT", "1") == "0":
        return None
    try:
        source = Path(__file__).with_name("_optimizer_kernels.cpp")
        compiler = shutil.which(os.environ.get("CXX", "g++")) or shutil.which("c++")
        if compiler is None or not source.is_file():
            return None
        has_avx = bool(_cpu.detect().get("has_avx", False))
        variant = "-mavx" if has_avx else "-msse2"
        flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-fno-fast-math",
                 "-ffp-contract=off", "-pthread", variant]
        digest = hashlib.sha256(source.read_bytes() + compiler.encode()
                                + " ".join(flags).encode()).hexdigest()[:20]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "smaulbrain"
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"opt-{variant[1:]}-{digest}.so"
        if not target.is_file():
            tmp = cache / f"opt-{variant[1:]}-{digest}-{os.getpid()}.tmp.so"
            res = subprocess.run([compiler, *flags, str(source), "-o", str(tmp)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, timeout=90)
            if res.returncode != 0:
                return None
            os.replace(tmp, target)
        lib = ctypes.CDLL(str(target))
        fp = ctypes.POINTER(ctypes.c_float)
        f = ctypes.c_float
        lib.smaulbrain_smaulopt_factored.argtypes = [
            fp, fp, fp, fp, fp, ctypes.c_size_t, ctypes.c_size_t,
            f, f, f, f, f, f, f, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        lib.smaulbrain_smaulopt_factored.restype = ctypes.c_int
        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB


def _ok(a):
    return (isinstance(a, np.ndarray) and a.dtype == np.float32
            and a.flags.c_contiguous and a.flags.writeable)


def factored_step(opt, w, g, st, collect_norms):
    """Run the fused update in place. Returns the update RMS (or 0.0) on
    success, or None if the caller must use the NumPy path."""
    lib = _load()
    if lib is None or w.ndim != 2 or w.size < 1024:
        return None
    m, R, C = st["m"], st["R"], st["C"]
    if not (_ok(w) and _ok(m) and _ok(R) and _ok(C)
            and m.shape == w.shape and R.shape == (w.shape[0],) and C.shape == (w.shape[1],)):
        return None
    gf = np.ascontiguousarray(g, dtype=np.float32)
    if gf.shape != w.shape:
        return None
    f32 = np.float32
    p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sq = ctypes.c_double(0.0)
    rc = lib.smaulbrain_smaulopt_factored(
        p(w), p(m), p(gf), p(R), p(C), w.shape[0], w.shape[1],
        f32(opt.beta1), f32(1.0 - opt.beta1), f32(opt.beta2), f32(1.0 - opt.beta2),
        f32(opt.eps), f32(opt.lr), f32(opt.lr * opt.weight_decay) if opt.weight_decay else f32(0.0),
        1 if collect_norms else 0, ctypes.byref(sq))
    if rc != 0:
        return None
    return float(np.sqrt(sq.value / w.size)) if collect_norms else 0.0
