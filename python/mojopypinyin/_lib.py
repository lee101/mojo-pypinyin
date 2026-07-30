"""Build and load the Mojo conversion kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIB = Path(
    os.environ.get("MOJOPYPINYIN_LIB", ROOT / "dist" / "libmojo-pypinyin.so")
)
I = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    sources = list(SRC.glob("*.mojo"))
    if (
        not force
        and LIB.exists()
        and sources
        and LIB.stat().st_mtime >= max(source.stat().st_mtime for source in sources)
    ):
        return str(LIB)
    if os.environ.get("MOJOPYPINYIN_LIB"):
        if LIB.exists():
            return str(LIB)
        raise BuildError(f"MOJOPYPINYIN_LIB does not exist: {LIB}")
    proc = subprocess.run(
        ["bash", str(ROOT / "build" / "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not LIB.exists():
        detail = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
        raise BuildError(detail[:4000] or f"build exited with status {proc.returncode}")
    return str(LIB)


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        fn = _library.mpy_convert
        fn.argtypes = [I] * 17
        fn.restype = None
    return _library
