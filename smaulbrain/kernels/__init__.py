"""Native kernel interface: build + runtime dispatch.

Portable scalar C++ only (no AVX2/AVX-512 intrinsics anywhere). If a compiler
or build fails, the error is explicit and actionable; the Python scalar/numpy
reference path always remains available.
"""
import errno
import os
import sys
from functools import lru_cache

from .. import cpu as _cpu

_SO_NAME = "smaul_kernels"


def build_here(build_dir="", verbose: bool = True) -> str:
    """Compile the optional native C++ kernels to a loadable shared library."""
    import hashlib
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    here = Path(__file__).resolve().parent
    source = here / "cpu_kernels.cpp"
    if not source.is_file():
        raise RuntimeError(f"native kernel source missing: {source}")
    configured = os.environ.get("CXX", "c++")
    compiler = shutil.which(configured) or shutil.which("c++")
    if compiler is None:
        raise RuntimeError(
            "native build requires a C++ compiler (c++/g++); "
            "the portable NumPy/Python path remains available.")

    caps = _cpu.detect()
    variant_flag = "-mavx" if caps.get("has_avx", False) else "-msse2"
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC",
             "-fno-fast-math", variant_flag]
    digest = hashlib.sha256(
        source.read_bytes() + compiler.encode("utf-8") +
        " ".join(flags).encode("ascii")
    ).hexdigest()[:16]
    target = here / f"{_SO_NAME}-{variant_flag[1:]}-{digest}.so"

    # Compile to a temporary file, then atomically publish a complete library.
    # build_dir optionally controls scratch space, not the final load location.
    scratch = Path(build_dir).resolve() if build_dir else here
    scratch.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{target.name}.{os.getpid()}.", suffix=".tmp.so", dir=scratch)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        if not target.is_file():
            command = [compiler, *flags, str(source), "-o", str(temporary)]
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, timeout=90)
            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.strip() or "native C++ kernel compilation failed")
            try:
                os.replace(temporary, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # build_dir may live on another filesystem. Copy to a unique
                # temporary sibling of the destination, then publish atomically.
                fd, publish_name = tempfile.mkstemp(
                    prefix=f"{target.name}.{os.getpid()}.", suffix=".publish.so",
                    dir=here)
                os.close(fd)
                publish_tmp = Path(publish_name)
                try:
                    shutil.copyfile(temporary, publish_tmp)
                    os.chmod(publish_tmp, 0o755)
                    os.replace(publish_tmp, target)
                    temporary.unlink(missing_ok=True)
                finally:
                    publish_tmp.unlink(missing_ok=True)
        else:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    # try_load is memoized, so clear a cached "missing" result after a build.
    try_load.cache_clear()
    if verbose:
        print(f"Built native kernels: {target} ({variant_flag})")
    return str(target)


@lru_cache(maxsize=1)
def try_load():
    """(module, info). Never raises for missing builds; reports reason."""
    caps = _cpu.detect()
    try:
        import ctypes
        import glob
        cands = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "smaul_kernels*"))
        compatible = []
        for path in cands:
            name = os.path.basename(path)
            if not name.endswith(".so") or ".tmp." in name or ".publish." in name:
                continue
            if "-mavx-" in name and not caps.get("has_avx", False):
                continue
            if "-mavx-" not in name and "-msse2-" not in name:
                continue
            compatible.append(path)
        compatible.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if not compatible:
            return None, {"available": False,
                          "reason": "no compatible native module built; using portable scalar/numpy path",
                          "cpu": caps, "dispatch": _cpu.dispatch_name()}
        lib = ctypes.CDLL(compatible[0])
        return lib, {"available": True, "path": compatible[0], "cpu": caps,
                     "dispatch": _cpu.dispatch_name()}
    except OSError as e:
        return None, {"available": False, "reason": f"native load failed: {e}",
                      "cpu": caps}
