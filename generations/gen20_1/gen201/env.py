"""Environment and git provenance for gen20.1 manifests (re-implemented; gen20k.manifest.environment
imports torch and is forbidden, IMPLEMENTATION_PLAN.md §8)."""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys

from . import bootstrap


def _total_ram_bytes() -> int | None:
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
            return None
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # pragma: no cover
        return None


def environment() -> dict:
    """Platform, interpreter, library versions, CPU/RAM, thread caps, and whether torch is loaded
    (it never should be outside S7; gen20k.runner imports it)."""
    import numpy
    import pandas
    import scipy
    import sklearn
    try:
        import pyarrow
        pa = pyarrow.__version__
    except Exception:  # pragma: no cover
        pa = None
    torch_loaded = "torch" in sys.modules
    torch_threads = None
    if torch_loaded:  # pragma: no cover
        try:
            torch_threads = int(sys.modules["torch"].get_num_threads())
        except Exception:
            torch_threads = None
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "executable": sys.executable,
            "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "pyarrow": pa,
            "cpu_count": os.cpu_count(), "ram_bytes": _total_ram_bytes(),
            "LOKY_MAX_CPU_COUNT": os.environ.get("LOKY_MAX_CPU_COUNT"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
            "torch_loaded": torch_loaded, "torch_threads": torch_threads}


def git_state() -> dict:
    """HEAD, branch, dirty flag and describe, read without writing anything: ``git status`` would otherwise
    refresh the stat cache and rewrite .git/index (``--no-optional-locks`` and GIT_OPTIONAL_LOCKS=0 on every
    call)."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def run(*cmd):
        try:
            return subprocess.run(["git", "--no-optional-locks", *cmd], cwd=bootstrap.ROOT, capture_output=True,
                                  text=True, check=True, env=env).stdout.strip()
        except Exception as exc:  # pragma: no cover
            return f"unavailable: {exc}"
    status = run("status", "--porcelain")
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) and not status.startswith("unavailable"),
            "describe": run("describe", "--always", "--dirty")}
