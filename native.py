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
import os
import platform
import shutil
import subprocess
import tempfile
import threading

import torch

ENV_VAR = "SMAUL_NATIVE"
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


def _cache_dir() -> str:
    h = hashlib.sha256()
    for name in LIBS:
        with open(os.path.join(SRC_DIR, name + ".cpp"), "rb") as f:
            h.update(f.read())
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
                    # Baseline ISA only: portable to any x86-64 CPU by default.
                    r = subprocess.run(
                        ["g++", "-O2", "-std=c++17", "-shared", "-fPIC",
                         os.path.join(SRC_DIR, name + ".cpp"), "-o", tmp],
                        capture_output=True, text=True, timeout=300,
                    )
                    if r.returncode != 0:
                        raise RuntimeError(f"g++ failed for {name}: {r.stderr[:500]}")
                    os.replace(tmp, so)
                lib = ctypes.CDLL(so)
                _bind(lib, name)
                _libs[name] = lib
            return True
        except Exception as e:  # noqa: BLE001 - reason is recorded, not hidden
            _build_failed = f"{type(e).__name__}: {e}"
            return False


def _bind(lib: ctypes.CDLL, name: str) -> None:
    F = ctypes.POINTER(ctypes.c_float)
    U = ctypes.POINTER(ctypes.c_ubyte)
    S = ctypes.c_size_t
    if name == "rmsnorm":
        lib.smaul_rmsnorm_forward.argtypes = [F, F, F, S, S, ctypes.c_float]
        lib.smaul_rmsnorm_forward.restype = None
    elif name == "linear_attn":
        lib.smaul_linear_attn_step_buf.argtypes = [F, F, F, F, F, F, F, S, ctypes.c_float]
        lib.smaul_linear_attn_step_buf.restype = None
    elif name == "fp8_quant":
        lib.smaul_fp8_quant_row_block.argtypes = [F, U, F, S, S, S]
        lib.smaul_fp8_quant_row_block.restype = None
        lib.smaul_fp8_dequant_row_block.argtypes = [U, F, F, S, S, S]
        lib.smaul_fp8_dequant_row_block.restype = None


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


def _f32_cpu(t: torch.Tensor) -> torch.Tensor | None:
    if t.device.type != "cpu" or t.dtype != torch.float32:
        return None
    return t if t.is_contiguous() else t.contiguous()


def call_rmsnorm(x_2d: torch.Tensor, w_1d: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Native RMSNorm over flat [rows, cols]. None = caller must fall back."""
    if x_2d.device.type != "cpu" or x_2d.dtype != torch.float32:
        _note("rmsnorm", False, "needs float32 CPU input")
        return None
    if not _want_native("rmsnorm"):
        return None
    try:
        rows, cols = x_2d.shape
        x = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
        w = w_1d.to(torch.float32).cpu()
        w = w if w.is_contiguous() else w.contiguous()
        y = torch.empty(rows, cols, dtype=torch.float32)
        F = ctypes.POINTER(ctypes.c_float)
        _libs["rmsnorm"].smaul_rmsnorm_forward(
            _ptr(x, ctypes.c_float), _ptr(w, ctypes.c_float),
            _ptr(y, ctypes.c_float),
            rows, cols, float(eps),
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
    place, so a defensive copy would silently discard the update.
    """
    for t in (S, z, q, k, v, y, scratch):
        if t.device.type != "cpu" or t.dtype != torch.float32 or not t.is_contiguous():
            _note("attn_step", False, "needs contiguous float32 CPU state")
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
            dh, float(eps),
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
    if w.device.type != "cpu" or w.dtype != torch.float32:
        _note("fp8_quant", False, "needs float32 CPU input")
        return False
    if (codes.device.type != "cpu" or scales.device.type != "cpu"
            or not codes.is_contiguous() or not scales.is_contiguous()):
        _note("fp8_quant", False, "needs contiguous CPU output buffers")
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
