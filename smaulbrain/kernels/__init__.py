"""Native kernel interface: build + runtime dispatch.

Portable scalar C++ only (no AVX2/AVX-512 intrinsics anywhere). If a compiler
or build fails, the error is explicit and actionable; the Python scalar/numpy
reference path always remains available.
"""
import os
import sys
from functools import lru_cache

from .. import cpu as _cpu

_SO_NAME = "smaul_kernels"


def build_here(build_dir="", verbose: bool = True) -> str:
    """Compile the extension in-place. Raises RuntimeError with actionable text."""
    try:
        from setuptools import setup
        from setuptools.command.build_ext import build_ext
        from setuptools.extension import Extension
    except ImportError as e:
        raise RuntimeError(
            "native build requires setuptools and a C++ compiler "
            "(e.g. apt install build-essential python3-dev). "
            "The portable Python/numpy path works without them.") from e
    import shutil
    if shutil.which("cc") is None and shutil.which("gcc") is None and shutil.which("c++") is None:
        raise RuntimeError("no C++ compiler found (cc/gcc/c++). Install build-essential; "
                           "the portable Python/numpy path works without native kernels.")
    here = os.path.dirname(os.path.abspath(__file__))
    ext = Extension(_SO_NAME, sources=[os.path.join(here, "cpu_kernels.cpp")],
                    extra_compile_args=["-O2", "-std=c++11", "-fPIC"])
    raise RuntimeError("native build is intentionally manual: run "
                       "`c++ -O2 -std=c++11 -fPIC -shared "
                       "v2/kernels/cpu_kernels.cpp -o smaul_kernels.so` "
                       "to produce the module; Python reference kernels are used otherwise.")


@lru_cache(maxsize=1)
def try_load():
    """(module, info). Never raises for missing builds; reports reason."""
    caps = _cpu.detect()
    try:
        import ctypes
        import glob
        cands = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "smaul_kernels*"))
        if not cands:
            return None, {"available": False,
                          "reason": "native module not built; using portable scalar/numpy path",
                          "cpu": caps, "dispatch": _cpu.dispatch_name()}
        lib = ctypes.CDLL(cands[0])
        return lib, {"available": True, "path": cands[0], "cpu": caps,
                     "dispatch": _cpu.dispatch_name()}
    except OSError as e:
        return None, {"available": False, "reason": f"native load failed: {e}",
                      "cpu": caps}
