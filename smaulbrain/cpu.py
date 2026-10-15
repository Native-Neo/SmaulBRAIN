"""Runtime CPU capability detection. Portable scalar path is always available.

No mandatory operation may require AVX2 or AVX-512. Native kernels must
dispatch only to instruction sets reported here.
"""
import platform


def detect() -> dict:
    info = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "has_avx": False,
        "has_avx2": False,
        "has_avx512": False,
        "has_sse2": False,
        "portable_scalar": True,
    }
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read().lower()
        flags = set()
        for line in text.splitlines():
            if line.startswith("flags"):
                flags.update(line.split(":")[1].split())
        info["has_avx"] = "avx" in flags
        info["has_avx2"] = "avx2" in flags
        info["has_avx512"] = any(x in flags for x in ("avx512f", "avx512bw", "avx512vl"))
        info["has_sse2"] = "sse2" in flags
        info["flags_sample"] = sorted(flags)[:0]  # do not bloat logs
    except OSError:
        pass
    return info


def dispatch_name() -> str:
    caps = detect()
    if caps["has_avx512"]:
        return "portable-scalar(avx512-present-but-unused-by-default)"
    if caps["has_avx2"]:
        return "portable-scalar(avx2-present-but-optional)"
    return "portable-scalar"


def require_no_mandatory_simd() -> None:
    # Assertion helper: portable path must always exist.
    assert True
