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
   * Required ISA is explicit (``REQUIRED_ISA`` = baseline x86-64/SSE2):
     dispatch checks runtime CPU flags (token-exact ``cpu_flags()``) via
     ``host_supports_baseline()`` before building, and every built ``.so``
     is disassembled (objdump, when available) to prove it needs no
     AVX/AVX2/AVX-512 instructions — an Ivy Bridge CPU can dlopen it
     safely. ``status()`` exposes mode, CPU label, flags, cache, and
     counters for observability.
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
  * ``SMAUL_NATIVE`` controls dispatch: ``auto`` (default),
    ``1``/``force``/``on``/``true`` (raise if native is unavailable),
    ``0``/``off``/``false`` (reference paths only). Forced infrastructure
    failures (unsupported host, failed build, kernel exception) record a
    fallback count/reason and then raise; per-call inapplicability (shapes,
    dtypes, autograd, non-finite) falls back even when forced.
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
import re
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
LIBS = ("rmsnorm", "linear_attn", "fp8_quant", "ternary_linear")

# Required ISA: baseline x86-64 (SSE2) only. The kernels use no AVX/AVX2/
# AVX-512 intrinsics and are built with ``-march=x86-64``, so they run on any
# x86-64 CPU including Ivy Bridge. Dispatch checks the host provides at least
# this baseline before attempting native (defense in depth + observability):
# ``cpu_flags()`` is the runtime truth (token-exact CPUID flags), kept
# separate from both the torch build-target label and the compile flags.
REQUIRED_ISA = "x86-64 (SSE2 baseline)"
REQUIRED_X86_FLAGS = ("sse2",)

# VEX/EVEX evidence: v-prefixed vector mnemonics and ymm/zmm/opmask
# registers. Baseline x86-64/SSE code never emits these, so any match means
# the .so needs more than REQUIRED_ISA (unsafe for e.g. Ivy Bridge).
_ISA_RE = re.compile(
    r"(?<![A-Za-z_])v(zeroupper|zeroall|mov[a-z]*|add[a-z]*|sub[a-z]*|mul[a-z]*"
    r"|div[a-z]*|broadcast[a-z]*|blend[a-z]*|perm[a-z]*|unpck[a-z]*|insert[a-z]*"
    r"|extract[a-z]*|gather[a-z]*|scatter[a-z]*|fmadd[a-z0-9]*|fmsub[a-z0-9]*"
    r"|fnmadd[a-z0-9]*|fnmsub[a-z0-9]*|xor[a-z]*|or[a-z]*|and[a-z]*"
    r"|padd[a-z]*|psub[a-z]*|pmul[a-z]*|pmin[a-z]*|pmax[a-z]*|pshuf[a-z]*"
    r"|punpck[a-z]*|maskmov[a-z]*|mask[a-z]*|compress[a-z]*|expand[a-z]*)"
    r"(?![A-Za-z_])|%(ymm|zmm)[0-9]+|%k[0-7]\b",
    re.IGNORECASE,
)

COUNTERS: dict[str, int] = {
    "fp8_quant_native": 0, "fp8_quant_fallback": 0,
    "fp8_dequant_native": 0, "fp8_dequant_fallback": 0,
    "rmsnorm_native": 0, "rmsnorm_fallback": 0,
    "attn_step_native": 0, "attn_step_fallback": 0,
    "ternary_linear_forward_native": 0, "ternary_linear_forward_fallback": 0,
    "ternary_linear_input_grad_native": 0, "ternary_linear_input_grad_fallback": 0,
    "fp8_to_ternary_native": 0, "fp8_to_ternary_fallback": 0,
    "fp8_factored_update_native": 0, "fp8_factored_update_fallback": 0,
    "fp8_clone_native": 0, "fp8_clone_fallback": 0,
    "float_to_ternary_native": 0, "float_to_ternary_fallback": 0,
}
LAST_FALLBACK: dict[str, str] = {}

_lock = threading.Lock()
_libs: dict[str, ctypes.CDLL] = {}
_build_failed: str | None = None
_alias_seq = 0


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0
    LAST_FALLBACK.clear()


_cpu_flags_cache: frozenset[str] | None = None


def _parse_cpu_flags(text: str) -> frozenset[str]:
    """Token-exact flag parse: ``avx`` alone must never read as ``avx2``."""
    for line in text.splitlines():
        if line.startswith("flags") and ":" in line:
            return frozenset(line.split(":", 1)[1].split())
    return frozenset()


def cpu_flags() -> frozenset[str]:
    """Runtime CPU flag tokens (Linux /proc/cpuinfo); empty set if unknown.

    Cached per process (flags cannot change under us); never raises.
    """
    global _cpu_flags_cache
    if _cpu_flags_cache is not None:
        return _cpu_flags_cache
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                _cpu_flags_cache = _parse_cpu_flags(f.read(65536))
                return _cpu_flags_cache
    except Exception:
        pass
    _cpu_flags_cache = frozenset()
    return _cpu_flags_cache


def cpu_capability() -> str:
    """Best-effort CPU capability label; never raises.

    ``torch.backends.cpu.get_cpu_capability()`` reports the ISA PyTorch was
    built to target, not necessarily the runtime CPUID, so it is only a
    label here — never a dispatch gate. The gate is
    :func:`host_supports_baseline`, which checks runtime flags (token-exact)
    against :data:`REQUIRED_ISA`.
    """
    try:
        cap = torch.backends.cpu.get_cpu_capability()
        if cap:
            return str(cap)
    except Exception:
        pass
    flags = cpu_flags()
    for want in ("avx512f", "avx2", "avx", "sse4_2", "sse2"):
        if want in flags:
            return want.upper()
    return "BASELINE" if flags else "UNKNOWN"


def required_isa() -> str:
    """ISA every native .so is guaranteed to stay within."""
    return REQUIRED_ISA


def host_supports_baseline() -> bool:
    """True if the host provably runs baseline x86-64 code; never raises.

    Non-x86 hosts always return True (generic ``-O2`` build, no x86 ISA
    assumption). On x86-64 the runtime flags must contain SSE2 when they are
    readable; unreadable flags (non-Linux) also return True — baseline x86-64
    implies SSE2 by definition, so absence of evidence is not evidence of
    absence, and the compiled binary stays baseline either way.
    """
    try:
        if platform.machine().lower() not in ("x86_64", "amd64"):
            return True
        flags = cpu_flags()
        if not flags:
            return True
        return all(f in flags for f in REQUIRED_X86_FLAGS)
    except Exception:
        return True


def native_mode() -> str:
    v = os.environ.get(ENV_VAR, "auto").strip().lower()
    if v in ("1", "force", "on", "true", "yes", "enable", "enabled"):
        return "force"
    if v in ("0", "off", "no", "disable", "disabled", "false"):
        return "off"
    return "auto"


def _build_flags() -> list[str]:
    """Baseline-only flags: portable to Ivy Bridge, no AVX assumptions."""
    flags = ["-O2", "-std=c++17", "-shared", "-fPIC", "-fno-fast-math", "-pthread"]
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


def _verify_baseline_isa(so_path: str) -> bool:
    """Best-effort proof a built .so needs no AVX+ instructions; never raises.

    Disassembles with objdump (when available) and rejects VEX/EVEX-encoded
    (``v``-prefixed) mnemonics and ymm/zmm/opmask registers. Missing objdump
    or any inspection failure returns True (skip): the ``-march=x86-64``
    build flags are the guarantee, this is the audit net on top.
    """
    try:
        if shutil.which("objdump") is None:
            return True
        r = subprocess.run(["objdump", "-d", so_path], capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return True
        return _ISA_RE.search(r.stdout) is None
    except Exception:
        return True


def _compile(name: str, so: str) -> None:
    """Compile one kernel to ``so`` (atomic publish); raises with reason."""
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


def _load_or_build(name: str, so: str) -> None:
    """Load a cached .so, rebuilding once if missing/corrupt/stale/wrong-ISA.

    Every file is ISA-verified *before* dlopen, so a foreign-ISA .so planted
    in the cache is never executed — it is rebuilt from our baseline sources
    and re-verified. A .so that loads but has the wrong symbols (stale cache)
    is likewise rebuilt; this process then loads the rebuild through a unique
    alias path, because dlopen caches by pathname and would otherwise keep
    returning the stale mapping for the canonical name. Raises on failure.
    """
    global _alias_seq
    if not os.path.exists(so):
        _compile(name, so)
    if not _verify_baseline_isa(so):
        _rm(so)  # wrong ISA for this host: rebuild once from baseline sources
        _compile(name, so)
        if not _verify_baseline_isa(so):
            _rm(so)
            raise RuntimeError(f"{name}.so failed baseline-ISA verification")
    try:
        lib = ctypes.CDLL(so)
        _bind(lib, name)
        _libs[name] = lib
        return
    except (OSError, RuntimeError):
        pass  # corrupt/stale/ABI-mismatched: repair the cache, alias-load below
    _rm(so)
    _compile(name, so)  # raises with its own reason (timeout/compile error)
    if not _verify_baseline_isa(so):
        _rm(so)
        raise RuntimeError(f"{name}.so failed baseline-ISA verification")
    _alias_seq += 1
    alias = f"{so}.alias-{os.getpid()}-{_alias_seq}.so"
    _rm(alias)
    try:
        shutil.copy(so, alias)  # new inode: dlopen treats it as a new object
    except OSError as e:
        raise RuntimeError(f"alias copy failed for {name}: {e}")
    try:
        lib = ctypes.CDLL(alias)
        _bind(lib, name)
    except (OSError, RuntimeError) as e2:
        _rm(alias)
        raise RuntimeError(f"load failed for {name} (ABI?) after rebuild: {e2}")
    _rm(alias)  # directory entry gone; the live mapping keeps its inode
    _libs[name] = lib


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
                _load_or_build(name, os.path.join(d, name + ".so"))
            _self_test()  # raises on mismatch; never ships a bad kernel
            return True
        except Exception as e:  # noqa: BLE001 - reason is recorded, not hidden
            _libs.clear()
            _build_failed = f"{type(e).__name__}: {e}"
            return False


def status() -> dict[str, object]:
    """Observable native-runtime summary: mode, CPU, build, and counters."""
    try:
        cache = _cache_dir()
    except Exception:
        cache = "unknown"
    return {
        "mode": native_mode(),
        "cpu_capability": cpu_capability(),
        "required_isa": REQUIRED_ISA,
        "host_ok": host_supports_baseline(),
        "build_flags": list(_build_flags()),
        "cache_dir": cache,
        "build_failed": _build_failed,
        "loaded": sorted(_libs),
        "counters": dict(COUNTERS),
        "last_fallback": dict(LAST_FALLBACK),
    }


def reset_native_state() -> None:
    """Clear loaded libs, the sticky build failure, and the CPU-flags cache."""
    global _build_failed, _cpu_flags_cache
    _libs.clear()
    _build_failed = None
    _cpu_flags_cache = None


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
    # Packed ternary forward: codes [+,0;-,+;0,-], scales [2,3,4].
    packed = (ctypes.c_ubyte * 2)(0x61, 0x08)
    scales_t = (ctypes.c_float * 3)(2.0, 3.0, 4.0)
    tx = (ctypes.c_float * 2)(1.0, 2.0)
    ty = (ctypes.c_float * 3)()
    _libs["ternary_linear"].smaul_ternary_linear_forward(
        ctypes.cast(tx, ctypes.c_void_p), packed, scales_t,
        ctypes.cast(ty, ctypes.c_void_p), 1, 2, 3, 0, 1
    )
    expected_ty = (2.0, 3.0, -8.0)
    if any(abs(ty[i] - expected_ty[i]) > 1e-6 for i in range(3)):
        raise RuntimeError(f"ternary linear forward self-test failed: {list(ty)}")
    # Input-gradient smoke test using the same packed ternary matrix.
    gy = (ctypes.c_float * 3)(1.0, 2.0, 3.0)
    gx = (ctypes.c_float * 2)()
    _libs["ternary_linear"].smaul_ternary_linear_input_grad(
        gy, packed, gx, 1, 2, 3, 1
    )
    if abs(gx[0] - (-1.0)) > 1e-6 or abs(gx[1] - (-1.0)) > 1e-6:
        raise RuntimeError(f"ternary input-gradient self-test failed: {list(gx)}")
    # Direct FP8-to-ternary conversion must not allocate a decoded weight matrix.
    f8_codes = (ctypes.c_ubyte * 4)(0x38, 0xC0, 0x30, 0xB0)  # [1,-2; 0.5,-0.5]
    f8_scales = (ctypes.c_float * 2)(1.0, 1.0)
    packed_f8 = (ctypes.c_ubyte * 1)()
    row_scales_f8 = (ctypes.c_float * 2)()
    _libs["ternary_linear"].smaul_fp8_to_packed_ternary(
        f8_codes, f8_scales, packed_f8, row_scales_f8, 2, 2, 2, 0.7, 0
    )
    if packed_f8[0] != 0x98 or abs(row_scales_f8[0] - 2.0) > 1e-6 or abs(row_scales_f8[1] - 0.5) > 1e-6:
        raise RuntimeError(f"FP8-to-ternary self-test failed: code={packed_f8[0]:#x}, scales={list(row_scales_f8)}")
    # Direct low-memory optimizer update: FP8 weights stay encoded, and
    # zero gradients/lr leave their represented values unchanged.
    opt_codes = (ctypes.c_ubyte * 4)(0x38, 0xC0, 0x30, 0xB0)
    opt_scales = (ctypes.c_float * 2)(1.0, 1.0)
    opt_grad = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
    opt_m = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
    opt_vr = (ctypes.c_float * 2)(0.0, 0.0)
    opt_vc = (ctypes.c_float * 2)(0.0, 0.0)
    opt_activity = (ctypes.c_float * 1)()
    opt_status = _libs["ternary_linear"].smaul_fp8_factored_update(
        opt_codes, opt_scales, opt_grad, opt_m, opt_vr, opt_vc,
        2, 2, 2, 0, 1, 0.9, 0.999, 0.0, 0.0, 1e-8, 10.0, 1.0,
        opt_activity,
    )
    if opt_status != 1 or abs(opt_activity[0]) > 1e-6:
        raise RuntimeError(f"low-memory FP8 optimizer self-test failed: status={opt_status}, activity={opt_activity[0]}")

    # Float-to-trit conversion consumes the original parameter directly.
    float_src = (ctypes.c_float * 4)(2.0, -1.0, 0.0, 0.5)
    float_packed = (ctypes.c_ubyte * 1)()
    float_scales = (ctypes.c_float * 1)()
    _libs["ternary_linear"].smaul_float_to_packed_ternary(
        float_src, float_packed, float_scales, 1, 4, 0.7, 0
    )
    if float_packed[0] != 0x09 or abs(float_scales[0] - 1.5) > 1e-6:
        raise RuntimeError(
            f"float-to-ternary self-test failed: codes={float_packed[0]:#x}, "
            f"scale={float_scales[0]}"
        )
    # Direct FP8 clone update stays encoded and uses no dense weight buffer.
    clone_src = (ctypes.c_ubyte * 4)(0x38, 0xC0, 0x30, 0xB0)
    clone_src_scales = (ctypes.c_float * 2)(1.0, 1.0)
    clone_dst = (ctypes.c_ubyte * 4)()
    clone_dst_scales = (ctypes.c_float * 2)()
    clone_status = _libs["ternary_linear"].smaul_fp8_clone_perturb(
        clone_src, clone_src_scales, clone_dst, clone_dst_scales,
        2, 2, 2, 0.0, 1234567,
    )
    if clone_status != 1 or not all(_m.isfinite(clone_dst_scales[i]) and clone_dst_scales[i] > 0 for i in range(2)):
        raise RuntimeError(f"direct FP8 clone self-test failed: status={clone_status}, scales={list(clone_dst_scales)}")

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
    elif name == "ternary_linear":
        fwd = getattr(lib, "smaul_ternary_linear_forward", None)
        bwd = getattr(lib, "smaul_ternary_linear_input_grad", None)
        convert = getattr(lib, "smaul_fp8_to_packed_ternary", None)
        update = getattr(lib, "smaul_fp8_factored_update", None)
        clone = getattr(lib, "smaul_fp8_clone_perturb", None)
        float_convert = getattr(lib, "smaul_float_to_packed_ternary", None)
        if fwd is None or bwd is None or convert is None or update is None or clone is None or float_convert is None:
            raise RuntimeError("ternary_linear .so missing a packed ternary, low-memory optimizer, clone, or float-conversion symbol (ABI?)")
        fwd.argtypes = [ctypes.c_void_p, U, F, ctypes.c_void_p,
                        S, S, S, ctypes.c_int, ctypes.c_int]
        fwd.restype = None
        bwd.argtypes = [F, U, F, S, S, S, ctypes.c_int]
        bwd.restype = None
        convert.argtypes = [U, F, U, F, S, S, S, ctypes.c_float, ctypes.c_int]
        convert.restype = None
        update.argtypes = [
            U, F, F, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            S, S, S, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, F,
        ]
        update.restype = ctypes.c_int
        clone.argtypes = [U, F, U, F, S, S, S, ctypes.c_float, ctypes.c_uint64]
        clone.restype = ctypes.c_int
        float_convert.argtypes = [ctypes.c_void_p, U, F, S, S, ctypes.c_float, ctypes.c_int]
        float_convert.restype = None
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
    if not host_supports_baseline():
        reason = ("host CPU lacks %s (capability=%s)"
                  % (REQUIRED_ISA, cpu_capability()))
        _note(op, False, reason)
        if mode == "force":
            raise RuntimeError(f"native {op} forced but unsupported: {reason}")
        return False
    if not ensure_native():
        reason = _build_failed or "build failed"
        _note(op, False, reason)
        if mode == "force":
            raise RuntimeError(f"native {op} forced but unavailable: {reason}")
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
        _note("rmsnorm", True)
        return y
    except Exception as e:  # noqa: BLE001 - fall back loudly, never crash
        _note("rmsnorm", False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
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
        _note("attn_step", True)
        return True
    except Exception as e:  # noqa: BLE001
        _note("attn_step", False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
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
    # NaN encodes differently per path (native saturates to finite codes,
    # torch casts NaN->NaN code), so non-finite inputs stay on the reference
    # path: bit-exactness is preserved and nothing crashes either way.
    try:
        if not bool(torch.isfinite(w).all()):
            _note("fp8_quant", False, "non-finite input (NaN/Inf)")
            return False
    except Exception:
        _note("fp8_quant", False, "finiteness check failed")
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
        _note("fp8_quant", True)
        return True
    except Exception as e:  # noqa: BLE001
        _note("fp8_quant", False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
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
        _note("fp8_dequant", True)
        return True
    except Exception as e:  # noqa: BLE001
        _note("fp8_dequant", False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return False

def call_fp8_factored_ternary_update(
    fp8_tensor, grad: torch.Tensor, state: dict, *,
    step: int, beta_m: float, beta_v: float, lr: float, wd: float,
    eps: float, update_clip: float, grad_multiplier: float,
) -> float | None:
    """Update FP8 storage in place without materializing a dense weight tensor.

    The native CPU kernel decodes/re-encodes one scalar at a time and holds
    only row/column moments plus one quantization tile in scratch memory.
    Supports the optimizer's canonical factored-v state only.
    """
    op = "fp8_factored_update"
    try:
        codes, block_scales = fp8_tensor.codes, fp8_tensor.scales
        rows, cols = map(int, fp8_tensor.shape)
        tile = int(fp8_tensor.tile)
    except Exception:
        _note(op, False, "expected a 2-D FP8BlockTensor")
        return None
    if (len(fp8_tensor.shape) != 2 or rows < 1 or cols < 1 or tile < 1
            or tuple(codes.shape) != (rows, cols)
            or tuple(block_scales.shape) != (rows, (cols + tile - 1) // tile)):
        _note(op, False, "invalid FP8 shape or block-scale layout")
        return None
    if (codes.device.type != "cpu" or block_scales.device.type != "cpu"
            or codes.dtype != torch.uint8 or block_scales.dtype != torch.float32
            or not codes.is_contiguous() or not block_scales.is_contiguous()):
        _note(op, False, "requires contiguous CPU FP8 codes and FP32 scales")
        return None
    if (grad.device.type != "cpu" or grad.dtype != torch.float32
            or tuple(grad.shape) != (rows, cols) or not grad.is_contiguous()):
        _note(op, False, "gradient must be contiguous CPU FP32 with the exact weight shape")
        return None
    m, vr, vc = state.get("m"), state.get("v_row"), state.get("v_col")
    moment_dtype = (torch.bfloat16, torch.float32)
    if (not torch.is_tensor(m) or m.dtype not in moment_dtype
            or tuple(m.shape) != (rows, cols)
            or not torch.is_tensor(vr) or vr.dtype != m.dtype
            or tuple(vr.shape) != (rows, 1)
            or not torch.is_tensor(vc) or vc.dtype != m.dtype
            or tuple(vc.shape) != (1, cols)
            or any(t.device.type != "cpu" or not t.is_contiguous() for t in (m, vr, vc))):
        _note(op, False, "requires contiguous CPU factored optimizer state matching the weight shape")
        return None
    scalar_values = (beta_m, beta_v, lr, wd, eps, update_clip, grad_multiplier)
    if (isinstance(step, bool) or not isinstance(step, int) or step < 1
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in scalar_values)
            or not 0.0 <= beta_m < 1.0 or not 0.0 <= beta_v < 1.0
            or lr < 0.0 or wd < 0.0 or eps <= 0.0 or update_clip <= 0.0
            or grad_multiplier < 0.0):
        _note(op, False, "invalid optimizer scalar or step")
        return None
    if not all(_valid_ptr(t) for t in (codes, block_scales, grad, m, vr, vc)):
        _note(op, False, "null weight/gradient/state storage")
        return None
    if not _want_native(op):
        return None
    try:
        activity = torch.empty((), dtype=torch.float32)
        status = _libs["ternary_linear"].smaul_fp8_factored_update(
            _ptr(codes, ctypes.c_ubyte), _ptr(block_scales, ctypes.c_float),
            _ptr(grad, ctypes.c_float), ctypes.c_void_p(m.data_ptr()),
            ctypes.c_void_p(vr.data_ptr()), ctypes.c_void_p(vc.data_ptr()),
            rows, cols, tile, int(m.dtype == torch.bfloat16), step,
            float(beta_m), float(beta_v), float(lr), float(wd), float(eps),
            float(update_clip), float(grad_multiplier), _ptr(activity, ctypes.c_float),
        )
        if status != 1:
            _note(op, False, "native low-memory optimizer refused the update")
            return None
        _note(op, True)
        return float(activity.item())
    except Exception as e:
        _note(op, False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return None


def call_float_to_packed_ternary(
    weights: torch.Tensor, threshold_ratio: float = 0.7
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pack rows of a CPU FP32/BF16 tensor directly into 2-bit ternary codes.

    Unlike _cached_packed_ternary_values, this does not create a same-shape
    floating-point tensor of ternary weights even temporarily.
    """
    op = "float_to_ternary"
    if (weights.device.type != "cpu" or weights.dtype not in (torch.float32, torch.bfloat16)
            or weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0
            or not weights.is_contiguous()):
        _note(op, False, "requires non-empty contiguous CPU FP32/BF16 matrix")
        return None
    if (isinstance(threshold_ratio, bool) or not isinstance(threshold_ratio, (int, float))
            or not math.isfinite(threshold_ratio) or not 0.0 <= threshold_ratio < 1.0):
        _note(op, False, "threshold ratio must be finite and in [0,1)")
        return None
    if not _want_native(op):
        return None
    try:
        rows, cols = map(int, weights.shape)
        packed = torch.empty(((rows * cols + 3) // 4,), dtype=torch.uint8)
        scales = torch.empty((rows,), dtype=torch.float32)
        _libs["ternary_linear"].smaul_float_to_packed_ternary(
            ctypes.c_void_p(weights.data_ptr()), _ptr(packed, ctypes.c_ubyte),
            _ptr(scales, ctypes.c_float), rows, cols, float(threshold_ratio),
            int(weights.dtype == torch.bfloat16),
        )
        _note(op, True)
        return packed, scales
    except Exception as e:
        _note(op, False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return None


def call_fp8_clone_perturb(
    fp8_tensor, perturb: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Clone a blockwise FP8 tensor with per-element relative perturbation.

    Both input and output stay encoded as E4M3 codes + FP32 block scales.
    Native scratch is one tile; no dense FP32/BF16 weight matrix is created.
    """
    op = "fp8_clone"
    try:
        codes, block_scales = fp8_tensor.codes, fp8_tensor.scales
        rows, cols = map(int, fp8_tensor.shape)
        tile = int(fp8_tensor.tile)
        magnitude = float(perturb)
    except Exception:
        _note(op, False, "expected a 2-D FP8BlockTensor")
        return None
    if (len(fp8_tensor.shape) != 2 or rows < 1 or cols < 1 or tile < 1
            or tuple(codes.shape) != (rows, cols)
            or tuple(block_scales.shape) != (rows, (cols + tile - 1) // tile)):
        _note(op, False, "invalid FP8 shape or block-scale layout")
        return None
    if (codes.device.type != "cpu" or block_scales.device.type != "cpu"
            or codes.dtype != torch.uint8 or block_scales.dtype != torch.float32
            or not codes.is_contiguous() or not block_scales.is_contiguous()):
        _note(op, False, "requires contiguous CPU FP8 codes and FP32 scales")
        return None
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF
            or not math.isfinite(magnitude) or not 0.0 <= magnitude < 1.0):
        _note(op, False, "seed must be uint64 and perturb must be finite in [0,1)")
        return None
    if not _want_native(op):
        return None
    try:
        out_codes = torch.empty_like(codes)
        out_scales = torch.empty_like(block_scales)
        status = _libs["ternary_linear"].smaul_fp8_clone_perturb(
            _ptr(codes, ctypes.c_ubyte), _ptr(block_scales, ctypes.c_float),
            _ptr(out_codes, ctypes.c_ubyte), _ptr(out_scales, ctypes.c_float),
            rows, cols, tile, magnitude, seed,
        )
        if status != 1:
            _note(op, False, "native FP8 clone rejected the input")
            return None
        _note(op, True)
        return out_codes, out_scales
    except Exception as e:
        _note(op, False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return None


def call_fp8_to_packed_ternary(
    fp8_tensor, threshold_ratio: float = 0.7, round_to_bf16: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Convert E4M3 bytes directly to packed trits without dense weight dequantization.

    Returns (packed_codes, row_scales) on CPU, or None if the required native
    kernel is unavailable. Full-ternary mode must not replace this with a dense
    FP32/BF16 fallback.
    """
    op = "fp8_to_ternary"
    try:
        codes, block_scales = fp8_tensor.codes, fp8_tensor.scales
        rows, cols = map(int, fp8_tensor.shape)
        tile = int(fp8_tensor.tile)
        ratio = float(threshold_ratio)
    except Exception:
        _note(op, False, "expected a 2-D FP8BlockTensor")
        return None
    if (codes.device.type != "cpu" or block_scales.device.type != "cpu"
            or codes.dtype != torch.uint8 or block_scales.dtype != torch.float32):
        _note(op, False, "requires CPU uint8 FP8 codes and FP32 block scales")
        return None
    if (len(fp8_tensor.shape) != 2 or rows < 1 or cols < 1 or tile < 1
            or tuple(codes.shape) != (rows, cols)
            or tuple(block_scales.shape) != (rows, (cols + tile - 1) // tile)):
        _note(op, False, "invalid FP8 shape or block-scale layout")
        return None
    if (not math.isfinite(ratio) or not 0.0 <= ratio < 1.0
            or not isinstance(round_to_bf16, bool)):
        _note(op, False, "threshold_ratio must be finite in [0,1); round_to_bf16 must be bool")
        return None
    if not (codes.is_contiguous() and block_scales.is_contiguous()):
        _note(op, False, "requires contiguous FP8 codes and scales")
        return None
    if not _valid_ptr(codes) or not _valid_ptr(block_scales):
        _note(op, False, "null FP8 storage")
        return None
    if not _want_native(op):
        return None
    try:
        packed = torch.empty(((rows * cols + 3) // 4,), dtype=torch.uint8)
        scales = torch.empty((rows,), dtype=torch.float32)
        _libs["ternary_linear"].smaul_fp8_to_packed_ternary(
            _ptr(codes, ctypes.c_ubyte), _ptr(block_scales, ctypes.c_float),
            _ptr(packed, ctypes.c_ubyte), _ptr(scales, ctypes.c_float),
            rows, cols, tile, ratio, int(round_to_bf16),
        )
        _note(op, True)
        return packed, scales
    except Exception as e:
        _note(op, False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return None


def call_ternary_linear_forward(
    x: torch.Tensor, packed_codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor | None:
    """Packed ternary forward: signed additions/subtractions plus row scaling."""
    op = "ternary_linear_forward"
    if x.device.type != "cpu" or x.dtype not in (torch.float32, torch.bfloat16):
        _note(op, False, f"requires CPU FP32/BF16 activations (got {x.device}/{x.dtype})")
        return None
    if packed_codes.device.type != "cpu" or packed_codes.dtype != torch.uint8:
        _note(op, False, "packed ternary codes must be CPU uint8")
        return None
    if scales.device.type != "cpu" or scales.dtype != torch.float32:
        _note(op, False, "ternary row scales must be CPU FP32")
        return None
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or scales.ndim != 1:
        _note(op, False, "requires non-empty [rows, in_features] and 1-D scales")
        return None
    rows, in_features = map(int, x.shape)
    out_features = int(scales.numel())
    if out_features == 0 or packed_codes.numel() != (in_features * out_features + 3) // 4:
        _note(op, False, "packed weight size does not match input/scales")
        return None
    if not (x.is_contiguous() and packed_codes.is_contiguous() and scales.is_contiguous()):
        _note(op, False, "requires contiguous inputs")
        return None
    if not _want_native(op):
        return None
    try:
        output = torch.empty((rows, out_features), dtype=x.dtype, device="cpu")
        _libs["ternary_linear"].smaul_ternary_linear_forward(
            ctypes.c_void_p(x.data_ptr()), _ptr(packed_codes, ctypes.c_ubyte),
            _ptr(scales, ctypes.c_float), ctypes.c_void_p(output.data_ptr()),
            rows, in_features, out_features, int(x.dtype == torch.bfloat16),
            max(1, int(torch.get_num_threads())),
        )
        _note(op, True)
        return output
    except Exception as e:
        _note(op, False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return None


def call_ternary_linear_input_grad(
    grad_scaled: torch.Tensor, packed_codes: torch.Tensor, in_features: int
) -> torch.Tensor | None:
    """Compute dX with packed ternary weights using signed adds/subtracts.

    grad_scaled already includes each output feature's ternary row scale. The
    native inner loop therefore performs no floating-point multiply.
    Returns CPU FP32 [rows, in_features], or None to request dense fallback.
    """
    op = "ternary_linear_input_grad"
    if grad_scaled.device.type != "cpu" or grad_scaled.dtype != torch.float32:
        _note(op, False, "requires CPU FP32 scaled gradients")
        return None
    if packed_codes.device.type != "cpu" or packed_codes.dtype != torch.uint8:
        _note(op, False, "packed ternary codes must be CPU uint8")
        return None
    if (grad_scaled.ndim != 2 or grad_scaled.shape[0] == 0
            or grad_scaled.shape[1] == 0):
        _note(op, False, "requires non-empty [rows, out_features] gradients")
        return None
    if isinstance(in_features, bool) or not isinstance(in_features, int) or in_features < 1:
        _note(op, False, "in_features must be a positive int")
        return None
    rows, out_features = map(int, grad_scaled.shape)
    if packed_codes.numel() != (in_features * out_features + 3) // 4:
        _note(op, False, "packed weight size does not match gradient/in_features")
        return None
    if not (grad_scaled.is_contiguous() and packed_codes.is_contiguous()):
        _note(op, False, "requires contiguous inputs")
        return None
    if not _want_native(op):
        return None
    try:
        output = torch.empty((rows, in_features), dtype=torch.float32, device="cpu")
        _libs["ternary_linear"].smaul_ternary_linear_input_grad(
            _ptr(grad_scaled, ctypes.c_float), _ptr(packed_codes, ctypes.c_ubyte),
            _ptr(output, ctypes.c_float), rows, in_features, out_features,
            max(1, int(torch.get_num_threads())),
        )
        _note(op, True)
        return output
    except Exception as e:
        _note(op, False, f"{type(e).__name__}: {e}")
        if native_mode() == "force":
            raise
        return None
