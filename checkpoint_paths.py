"""Filesystem containment for checkpoint payload paths.

This module intentionally has no tensor-framework imports so CLI preflight
can validate checkpoint metadata on low-memory machines.
"""

import os


def _require(cond: bool, msg: str) -> None:
    """Raise ValueError with checkpoint context when a validation condition fails."""
    if not cond:
        raise ValueError(f"checkpoint validation failed: {msg}")


def _jailed_path(ckpt_dir: str, rel: str) -> str:
    """Join rel under ckpt_dir and refuse jail escapes (.., absolute, symlink).

    Returns the absolute jailed path. Raises ValueError on traversal.
    Both the lexical path and its realpath (symlink-resolved) must stay
    inside the checkpoint dir; existing symlinks anywhere along the path
    are refused.
    """
    _require(isinstance(rel, str) and rel != "" and "\x00" not in rel,
             f"bad checkpoint rel {rel!r}")
    _require(not os.path.isabs(rel), f"bad checkpoint rel {rel!r}")
    parts = rel.split("/")
    _require(all(part not in ("", ".", "..") for part in parts),
             f"bad checkpoint rel traversal {rel!r}")
    base = os.path.realpath(os.path.abspath(ckpt_dir))
    p = os.path.abspath(os.path.join(ckpt_dir, rel))
    _require(p == base or p.startswith(base + os.sep),
             f"checkpoint path escapes dir: {rel!r}")
    try:
        current = os.path.abspath(ckpt_dir)
        if os.path.lexists(current) and os.path.islink(current):
            raise ValueError(
                f"checkpoint validation failed: refusing symlink checkpoint root {ckpt_dir!r}"
            )
        for part in parts:
            current = os.path.join(current, part)
            if os.path.lexists(current) and os.path.islink(current):
                raise ValueError(
                    f"checkpoint validation failed: refusing symlink component {part!r} in {rel!r}"
                )
        r = os.path.realpath(p)
    except OSError as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable path {rel!r}: {e}"
        ) from e
    _require(r == base or r.startswith(base + os.sep),
             f"checkpoint path escapes dir via symlink: {rel!r}")
    try:
        if os.path.lexists(p) and os.path.islink(p):
            raise ValueError(
                f"checkpoint validation failed: refusing symlink {rel!r}"
            )
    except OSError as e:
        raise ValueError(
            f"checkpoint validation failed: unreadable path {rel!r}: {e}"
        ) from e
    return p
