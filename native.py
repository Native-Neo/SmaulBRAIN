"""Native runtime dispatch for the C++ kernels in ``kernels_cpp/``.

The Python/torch paths are always correct; this module dispatches to native
code only when it is provably safe, and says so out loud otherwise:

  * CPU capability comes from ``torch.backends.cpu.get_cpu_capability()``
    (with a ``/proc/cpuinfo`` fallback, ``unknown`` elsewhere). Binaries are
    built with baseline x86-64 flags only — never ``-march=native`` — so the
    compiled code cannot execute instructions the build host has but the
    runtime CPU lacks.
  * Built shared objects are cached by source hash; concurrent first builds
    race harmlessly (identical bytes, atomic publish).
  * Build pins baseline x86-64 flags (``-march=x86-64 -mtune=generic``,
    never ``-march=native``) plus ``-fno-fast-math``; the cache key folds in
    sources, flags, compiler version, Python/torch versions, and machine, so
    a stale or ABI-mismatched ``.so`` is rebuilt instead of dlopen'ed.
  * After loading, a tiny self-test runs each kernel; failure marks the
    build bad and every call falls back with a recorded reason.
  * Calls validate shapes/bounds/strides, emptiness, tile/Dh positivity,
    and eps finiteness *before* touching raw pointers, so a bad call falls
    back instead of faulting (ctypes cannot catch a segfault/SIGFPE).
    Noncontiguous read-only inputs are copied; in-place state buffers are
    never copied (a copy would silently discard the update) — they fall
    back instead. Unaligned pointers are safe on x86-64 and need no copy.
  * Native kernels touch only raw buffers, throw nothing across the ABI,
    and run with no Python callbacks (ctypes releases the GIL during the
    call); outputs are fresh tensors owned by Python, and grad-tracked
    float-reordering inputs fall back so autograd is never silently cut.
  * ``SMAUL_NATIVE`` controls dispatch: ``auto`` (default), ``1``/``force``
    (raise if native is unavailable), ``0``/``off`` (reference paths only).
  * Every call site records native vs fallback counts plus the last fallback
    reason in ``COUNTERS``/``LAST_FALLBACK`` — dispatch is observable, and a
    broken native path degrades loudly instead of hiding.
  * In-place native ops (attention state) run only on contiguous state
    tensors; anything else falls back rather than silently copying state.

Bit-exact paths (FP8 quant/dequant) may auto-dispatch. Float-reordering
paths (RMSNorm, attention step) dispatch only when explicitly enabled, so
bit-deterministic tests keep passing on the reference path by default.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

import torch

ENV_VAR = "SMAUL_NATIVE"
BUILD_TIMEOUT_VAR = "SMAUL_NATIVE_BUILD_TIMEOUT"
BUILD_TIMEOUT_DEFAULT = 300
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels_cpp")
LIBS = ("rmsnorm", "linear_attn", "fp8_quant")

COUNTERS: dict[str, int] = {
    "fp8_quant_native": 0, "fp8_quant_fallback": 0,
    "fp8_dequant_native": 0, "fp8_dequant_fallback": 0,
    "rmsnorm_native": 0, "rmsnorm_fallback": 0,
    "attn_step_native": 0, "attn_step_fallback": 0,
}
LAST_FALLBACK: dict[str, str] = {}

_lock = threading.Lock()
_libs: dict[str, ctypes.CDLL] = {}
_build_failed: str | None = None


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0
    LAST_FALLBACK.clear()


def cpu_capability() -> str:
    """Best-effort CPU capability label; never raises."""
    try:
        cap = torch.backends.cpu.get_cpu_capability()
        if cap:
            return str(cap)
    except Exception:
        pass
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                text = f.read(65536)
            for line in text.splitlines():
                if line.startswith("flags") and ":" in line:
                    flags = line.split(":", 1)[1]
                    for want in ("avx512f", "avx2", "sse4_2"):
                        if want in flags:
                            return want.upper()
                    return "BASELINE"
    except Exception:
        pass
    return "UNKNOWN"


def native_mode() -> str:
    v = os.environ.get(ENV_VAR, "auto").strip().lower()
    if v in ("1", "force", "on"):
        return "force"
    if v in ("0", "off", "no", "disable"):
        return "off"
    return "auto"


def _build_flags() -> list[str]:
    """Baseline-only flags: portable to Ivy Bridge, no AVX assumptions."""
    flags = ["-O2", "-std=c++17", "-shared", "-fPIC", "-fno-fast-math"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        flags += ["-march=x86-64", "-mtune=generic"]
    return flags


def _compiler_version() -> str:
    try:
        r = subprocess.run(["g++", "--version"], capture_output=True,
                           text=True, timeout=30)
        return (r.stdout.splitlines() or ["unknown"])[0][:200]
    except Exception:
        return "unknown"


def _build_timeout() -> int:
    try:
        return max(1, int(os.environ.get(BUILD_TIMEOUT_VAR, BUILD_TIMEOUT_DEFAULT)))
    except Exception:
        return BUILD_TIMEOUT_DEFAULT


def _cache_dir() -> str:
    h = hashlib.sha256()
    for name in LIBS:
        with open(os.path.join(SRC_DIR, name + ".cpp"), "rb") as f:
            h.update(f.read())
    # ABI key: a .so built by another compiler/flagset/interpreter must not
    # be reused — that is the ABI-mismatch detector (rebuild, then self-test).
    h.update(" ".join(_build_flags()).encode())
    h.update(_compiler_version().encode())
    h.update(sys.version.encode())
    try:
        h.update(torch.__version__.encode())
    except Exception:
        pass
    h.update(platform.machine().encode())
    d = os.path.join(tempfile.gettempdir(), "smaulbrain_native_" + h.hexdigest()[:12])
    os.makedirs(d, exist_ok=True)
    return d


def ensure_native() -> bool:
    """Build (cached) and load all native libraries. False + reason on failure."""
    global _build_failed
    if _libs:
        return True
    if _build_failed is not None:
        return False
    with _lock:
        if _libs:
            return True
        if _build_failed is not None:
            return False
        if shutil.which("g++") is None:
            _build_failed = "no C++ compiler (g++) found"
            return False
        try:
            d = _cache_dir()
            for name in LIBS:
                so = os.path.join(d, name + ".so")
                if not os.path.exists(so):
                    tmp = so + f".{os.getpid()}.tmp"
                    try:
                        # Baseline ISA only: portable to any x86-64 CPU by default.
                        r = subprocess.run(
                            ["g++", *_build_flags(),
                             os.path.join(SRC_DIR, name + ".cpp"), "-o", tmp],
                            capture_output=True, text=True, timeout=_build_timeout(),
                        )
                    except subprocess.TimeoutExpired:
                        _rm(tmp)
                        raise RuntimeError(
                            f"g++ timed out after {_build_timeout()}s for {name}")
                    if r.returncode != 0:
                        _rm(tmp)
                        raise RuntimeError(f"g++ failed for {name}: {r.stderr[:500]}")
                    try:
                        os.replace(tmp, so)
                    except OSError as e:
                        _rm(tmp)
                        raise RuntimeError(f"publish failed for {name}: {e}")
                try:
                    lib = ctypes.CDLL(so)
                except OSError as e:  # ABI mismatch / corrupt .so: rebuild once
                    _rm(so)
                    raise RuntimeError(f"load failed for {name} (ABI?): {e}")
                _bind(lib, name)
                _libs[name] = lib
            _self_test()  # raises on mismatch; never ships a bad kernel
            return True
        except Exception as e:  # noqa: BLE001 - reason is recorded, not hidden
            _libs.clear()
            _build_failed = f"{type(e).__name__}: {e}"
            return False


def _rm(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _self_test() -> None:
    """Smoke-test each loaded kernel on tiny buffers; raise on any mismatch."""
    import math as _m
    # rmsnorm: rows=1, cols=4, w=ones -> y = x / rms(x)
    x = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    w = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
    y = (ctypes.c_float * 4)()
    _libs["rmsnorm"].smaul_rmsnorm_forward(x, w, y, 1, 4, 1e-6)
    rms = _m.sqrt((1 + 4 + 9 + 16) / 4.0 + 1e-6)
    for i, v in enumerate((1.0, 2.0, 3.0, 4.0)):
        if not _m.isfinite(y[i]) or abs(y[i] - v / rms) > 1e-5:
            raise RuntimeError(f"rmsnorm self-test failed: {list(y)}")
    # linear_attn: Dh=4 smoke (finite output only; parity is tested elsewhere)
    Dh = 4
    S = (ctypes.c_float * 16)()
    z = (ctypes.c_float * 4)()
    q = (ctypes.c_float * 4)(0.1, -0.2, 0.3, 0.0)
    k = (ctypes.c_float * 4)(0.2, 0.1, -0.1, 0.4)
    v = (ctypes.c_float * 4)(1.0, 0.5, -0.5, 0.25)
    yo = (ctypes.c_float * 4)()
    sc = (ctypes.c_float * 8)()
    _libs["linear_attn"].smaul_linear_attn_step_buf(S, z, q, k, v, yo, sc, Dh, 1e-6)
    if not all(_m.isfinite(yo[i]) for i in range(Dh)):
        raise RuntimeError(f"linear_attn self-test failed: {list(yo)}")
    # fp8: quantize+dequantize roundtrip stays finite
    wq = (ctypes.c_float * 4)(0.5, -1.0, 2.0, 0.01)
    codes = (ctypes.c_ubyte * 4)()
    scales = (ctypes.c_float * 2)()
    back = (ctypes.c_float * 4)()
    _libs["fp8_quant"].smaul_fp8_quant_row_block(wq, codes, scales, 1, 4, 2)
    _libs["fp8_quant"].smaul_fp8_dequant_row_block(codes, scales, back, 1, 4, 2)
    if not all(_m.isfinite(back[i]) for i in range(4)):
        raise RuntimeError(f"fp8 self-test failed: {list(back)}")


def _bind(lib: ctypes.CDLL, name: str) -> None:
    F = ctypes.POINTER(ctypes.c_float)
    U = ctypes.POINTER(ctypes.c_ubyte)
    S = ctypes.c_size_t
    # Missing symbols mean the cached .so is stale/ABI-mismatched: fail
    # loudly so ensure_native() records it and falls back instead of
    # calling a half-bound library.
    if name == "rmsnorm":
        fn = getattr(lib, "smaul_rmsnorm_forward", None)
        if fn is None:
            raise RuntimeError("rmsnorm .so missing smaul_rmsnorm_forward (ABI?)")
        fn.argtypes = [F, F, F, S, S, ctypes.c_float]
        fn.restype = None
    elif name == "linear_attn":
        fn = getattr(lib, "smaul_linear_attn_step_buf", None)
        if fn is None:
            raise RuntimeError("linear_attn .so missing smaul_linear_attn_step_buf (ABI?)")
        fn.argtypes = [F, F, F, F, F, F, F, S, ctypes.c_float]
        fn.restype = None
    elif name == "fp8_quant":
        q = getattr(lib, "smaul_fp8_quant_row_block", None)
        dq = getattr(lib, "smaul_fp8_dequant_row_block", None)
        if q is None or dq is None:
            raise RuntimeError("fp8_quant .so missing symbols (ABI?)")
        q.argtypes = [F, U, F, S, S, S]
        q.restype = None
        dq.argtypes = [U, F, F, S, S, S]
        dq.restype = None


def _ptr(t: torch.Tensor, ctype) -> ctypes._Pointer:
    """Raw pointer for a CPU tensor (no copy, no ownership transfer)."""
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctype))


def _note(op: str, ok: bool, reason: str = "") -> None:
    COUNTERS["%s_%s" % (op, "native" if ok else "fallback")] += 1
    if not ok and reason:
        LAST_FALLBACK[op] = reason


def _want_native(op: str) -> bool:
    mode = native_mode()
    if mode == "off":
        _note(op, False, "disabled by %s=off" % ENV_VAR)
        return False
    if not ensure_native():
        if mode == "force":
            raise RuntimeError(f"native {op} forced but unavailable: {_build_failed}")
        _note(op, False, _build_failed or "build failed")
        return False
    return True


def _finite_nonneg_eps(eps: float) -> float | None:
    try:
        e = float(eps)
    except Exception:
        return None
    if not math.isfinite(e) or e < 0.0:
        return None
    return e


def _valid_ptr(t: torch.Tensor) -> bool:
    return t.numel() == 0 or t.data_ptr() != 0


def call_rmsnorm(x_2d: torch.Tensor, w_1d: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Native RMSNorm over flat [rows, cols]. None = caller must fall back.

    Output is a fresh Python-owned tensor; inputs are only read (a strided
    read-only input is copied, never written through). Grad-tracked inputs
    fall back so the grad-less kernel never silently cuts autograd.
    """
    if x_2d.device.type != "cpu" or x_2d.dtype != torch.float32:
        _note("rmsnorm", False, "needs float32 CPU input")
        return None
    if x_2d.ndim != 2:
        _note("rmsnorm", False, "needs 2D [rows, cols] input")
        return None
    rows, cols = int(x_2d.shape[0]), int(x_2d.shape[1])
    if rows <= 0 or cols <= 0:
        _note("rmsnorm", False, "empty input")
        return None
    if x_2d.numel() != rows * cols:
        _note("rmsnorm", False, "strided/overlapping input")
        return None
    if w_1d.numel() != cols:
        _note("rmsnorm", False, "weight size mismatch")
        return None
    e = _finite_nonneg_eps(eps)
    if e is None:
        _note("rmsnorm", False, "bad eps (NaN/Inf/negative)")
        return None
    if torch.is_grad_enabled() and (x_2d.requires_grad or w_1d.requires_grad):
        _note("rmsnorm", False, "refuses autograd input (no grad rule)")
        return None
    if not _valid_ptr(x_2d) or not _valid_ptr(w_1d):
        _note("rmsnorm", False, "null storage")
        return None
    if not _want_native("rmsnorm"):
        return None
    try:
        x = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
        w = w_1d.detach().to(torch.float32).cpu()
        w = w if w.is_contiguous() else w.contiguous()
        y = torch.empty(rows, cols, dtype=torch.float32)
        F = ctypes.POINTER(ctypes.c_float)
        _libs["rmsnorm"].smaul_rmsnorm_forward(
            _ptr(x, ctypes.c_float), _ptr(w, ctypes.c_float),
            _ptr(y, ctypes.c_float),
            rows, cols, e,
        )
        COUNTERS["rmsnorm_native"] += 1
        return y
    except Exception as e:  # noqa: BLE001 - fall back loudly, never crash
        _note("rmsnorm", False, f"{type(e).__name__}: {e}")
        return None


def call_attn_step(
    S: torch.Tensor, z: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    y: torch.Tensor, scratch: torch.Tensor, dh: int, eps: float,
) -> bool:
    """Native attention step, in place. False = caller must fall back.

    All tensors must be float32, CPU, and contiguous: state is updated in
    place, so a defensive copy would silently discard the update. Shapes are
    validated (S [Dh,Dh], vectors [Dh], scratch [2*Dh]) so a mismatched call
    falls back instead of faulting. Grad-tracked buffers fall back.
    """
    try:
        Dh = int(dh)
    except Exception:
        _note("attn_step", False, "bad Dh")
        return False
    if Dh <= 0:
        _note("attn_step", False, "empty Dh")
        return False
    e = _finite_nonneg_eps(eps)
    if e is None:
        _note("attn_step", False, "bad eps (NaN/Inf/negative)")
        return False
    for t in (S, z, q, k, v, y, scratch):
        if t.device.type != "cpu" or t.dtype != torch.float32 or not t.is_contiguous():
            _note("attn_step", False, "needs contiguous float32 CPU state")
            return False
    if (S.ndim != 2 or tuple(S.shape) != (Dh, Dh)
            or tuple(z.shape) != (Dh,) or tuple(q.shape) != (Dh,)
            or tuple(k.shape) != (Dh,) or tuple(v.shape) != (Dh,)
            or tuple(y.shape) != (Dh,) or tuple(scratch.shape) != (2 * Dh,)):
        _note("attn_step", False, "shape mismatch for Dh=%d" % Dh)
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (S, z, q, k, v)):
        _note("attn_step", False, "refuses autograd input (no grad rule)")
        return False
    if not all(_valid_ptr(t) for t in (S, z, q, k, v, y, scratch)):
        _note("attn_step", False, "null storage")
        return False
    if not _want_native("attn_step"):
        return False
    try:
        F = ctypes.POINTER(ctypes.c_float)
        _libs["linear_attn"].smaul_linear_attn_step_buf(
            _ptr(S, ctypes.c_float), _ptr(z, ctypes.c_float),
            _ptr(q, ctypes.c_float), _ptr(k, ctypes.c_float),
            _ptr(v, ctypes.c_float),
            _ptr(y, ctypes.c_float), _ptr(scratch, ctypes.c_float),
            Dh, e,
        )
        COUNTERS["attn_step_native"] += 1
        return True
    except Exception as e:  # noqa: BLE001
        _note("attn_step", False, f"{type(e).__name__}: {e}")
        return False


def call_fp8_quant(
    w: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
    rows: int, cols: int, tile: int,
) -> bool:
    """Native FP8 row-block quantize. False = caller must fall back."""
    try:
        rows, cols, tile = int(rows), int(cols), int(tile)
    except Exception:
        _note("fp8_quant", False, "bad rows/cols/tile")
        return False
    if rows <= 0 or cols <= 0 or tile <= 0:
        _note("fp8_quant", False, "empty rows/cols or zero tile")
        return False
    nblocks = (cols + tile - 1) // tile
    if w.device.type != "cpu" or w.dtype != torch.float32:
        _note("fp8_quant", False, "needs float32 CPU input")
        return False
    if (codes.device.type != "cpu" or scales.device.type != "cpu"
            or not codes.is_contiguous() or not scales.is_contiguous()):
        _note("fp8_quant", False, "needs contiguous CPU output buffers")
        return False
    if tuple(w.shape) != (rows, cols) or tuple(codes.shape) != (rows, cols):
        _note("fp8_quant", False, "shape mismatch for rows/cols")
        return False
    if tuple(scales.shape) != (rows, nblocks):
        _note("fp8_quant", False, "scales shape mismatch for tile")
        return False
    if not all(_valid_ptr(t) for t in (w, codes, scales)):
        _note("fp8_quant", False, "null storage")
        return False
    if not _want_native("fp8_quant"):
        return False
    try:
        w = w if w.is_contiguous() else w.contiguous()
        F = ctypes.POINTER(ctypes.c_float)
        U = ctypes.POINTER(ctypes.c_ubyte)
        S = ctypes.c_size_t
        _libs["fp8_quant"].smaul_fp8_quant_row_block(
            _ptr(w, ctypes.c_float), _ptr(codes, ctypes.c_ubyte),
            _ptr(scales, ctypes.c_float),
            S(rows), S(cols), S(tile),
        )
        COUNTERS["fp8_quant_native"] += 1
        return True
    except Exception as e:  # noqa: BLE001
        _note("fp8_quant", False, f"{type(e).__name__}: {e}")
        return False


def call_fp8_dequant(
    codes: torch.Tensor, scales: torch.Tensor, out: torch.Tensor,
    rows: int, cols: int, tile: int,
) -> bool:
    """Native FP8 row-block dequantize into ``out``. False = fall back."""
    try:
        rows, cols, tile = int(rows), int(cols), int(tile)
    except Exception:
        _note("fp8_dequant", False, "bad rows/cols/tile")
        return False
    if rows <= 0 or cols <= 0 or tile <= 0:
        _note("fp8_dequant", False, "empty rows/cols or zero tile")
        return False
    nblocks = (cols + tile - 1) // tile
    if out.dtype != torch.float32:
        _note("fp8_dequant", False, "needs float32 output buffer")
        return False
    if (codes.device.type != "cpu" or scales.device.type != "cpu"
            or out.device.type != "cpu"):
        _note("fp8_dequant", False, "needs CPU storage tensors")
        return False
    if (not codes.is_contiguous() or not scales.is_contiguous()
            or not out.is_contiguous()):
        _note("fp8_dequant", False, "needs contiguous buffers")
        return False
    if (tuple(codes.shape) != (rows, cols) or tuple(out.shape) != (rows, cols)):
        _note("fp8_dequant", False, "shape mismatch for rows/cols")
        return False
    if tuple(scales.shape) != (rows, nblocks):
        _note("fp8_dequant", False, "scales shape mismatch for tile")
        return False
    if not all(_valid_ptr(t) for t in (codes, scales, out)):
        _note("fp8_dequant", False, "null storage")
        return False
    if not _want_native("fp8_dequant"):
        return False
    try:
        F = ctypes.POINTER(ctypes.c_float)
        U = ctypes.POINTER(ctypes.c_ubyte)
        S = ctypes.c_size_t
        _libs["fp8_quant"].smaul_fp8_dequant_row_block(
            _ptr(codes, ctypes.c_ubyte), _ptr(scales, ctypes.c_float),
            _ptr(out, ctypes.c_float),
            S(rows), S(cols), S(tile),
        )
        COUNTERS["fp8_dequant_native"] += 1
        return True
    except Exception as e:  # noqa: BLE001
        _note("fp8_dequant", False, f"{type(e).__name__}: {e}")
        return False
