"""Path, thread and write-protection bootstrap for gen20.1.  Import it before any frozen module.

IMPLEMENTATION_PLAN.md §8 "Bootstrap hazards":

* ``sys.dont_write_bytecode = True`` so no ``__pycache__`` appears under a frozen generation;
* ``LOKY_MAX_CPU_COUNT`` and ``OMP_NUM_THREADS`` default to "2" (``setdefault``: an explicit
  environment value wins, and gen20k's own default of 4 OMP threads never applies);
* fail fast if the local, gitignored gen14 bench cache is missing, because
  ``gen14.dirbench.load`` would otherwise rebuild it and write into ``generations/gen14_direction``;
* fail fast if any gen13 output directory is missing, because ``gen13sep.paths`` creates them at
  import (``mkdir``), which would write into ``generations/gen13_separation`` (LT-27);
* then ``generations/gen20_kshot`` (and the frozen trees gen20k itself needs) go on ``sys.path`` and
  ``gen20k.bootstrap`` is imported.

Nothing here writes anything.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[3]
GENERATIONS = ROOT / "generations"
G201 = GENERATIONS / "gen20_1"
MANIFESTS = G201 / "manifests"
PREDICTIONS = G201 / "predictions"
RESULTS = G201 / "results"
PREREG = G201 / "PRE_REGISTRATION.md"
PLAN = G201 / "IMPLEMENTATION_PLAN.md"

G20 = GENERATIONS / "gen20_kshot"
G13 = GENERATIONS / "gen13_separation"
G14 = GENERATIONS / "gen14_direction"
G15 = GENERATIONS / "gen15_curve"
SRC = ROOT / "src"
BENCH_PKL = G14 / "cache" / "bench.pkl"

#: the directories ``gen13sep/paths.py`` creates at import time (paths.py:41-43)
GEN13_OUTPUT_DIRS: tuple[Path, ...] = tuple(G13 / d for d in (
    "config", "manifests", "predictions", "metrics", "bootstrap", "headline_tables", "analysis", "figures"))

#: directories under generations/ gen20.1 must never write into (LT-26): every directory there except
#: gen20_1 itself (gen12_*, gen13 .. gen20_kshot, the gen20.1 instructions directory and generations/tests).
#: The LT-26 file snapshot (gen201.guard.snapshot) covers more than this: the whole repository except .git,
#: .venv, .claude and generations/gen20_1, plus the three generations/gen20_1/*.md documents.
FROZEN_TREES: tuple[Path, ...] = tuple(sorted(p for p in GENERATIONS.iterdir() if p.is_dir() and p.name != G201.name))


def require_bench_cache(path: Path = BENCH_PKL) -> Path:
    """Raise unless the gen14 bench cache exists (gen14's loader would rebuild and write it)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing: gen14.dirbench.load would rebuild it and write into a frozen tree; "
            "gen20.1 refuses to run without the cached bench (IMPLEMENTATION_PLAN.md §8)")
    return Path(path)


def require_gen13_output_dirs(dirs=GEN13_OUTPUT_DIRS) -> None:
    """Raise if importing gen13sep.paths would create a directory inside gen13_separation."""
    missing = [str(d) for d in dirs if not Path(d).is_dir()]
    if missing:
        raise FileNotFoundError(f"gen13 output directories missing (gen13sep.paths would mkdir them): {missing}")


require_bench_cache()
require_gen13_output_dirs()

for _p in (SRC, G15, G14, G13, G20):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen20k.bootstrap  # noqa: E402,F401  (inserts gen15/gen14/gen13 the way gen20 does)

if str(G201) not in sys.path:
    sys.path.insert(0, str(G201))


#: a Git-Bash drive path ("/d/..." or, once pathlib has seen it, "\d\...") that Windows Python would resolve
#: to "<current drive>:\d\..."
_GITBASH_DRIVE = re.compile(r"^[\\/][A-Za-z](?:[\\/]|$)")


def scratch_roots() -> tuple[Path, ...]:
    """The only places outside the repository a gen20.1 script may write to: the system temporary directory
    (pytest's tmp_path and the session scratchpad live there)."""
    return (Path(tempfile.gettempdir()).resolve(),)


def _within(p: Path, base: Path) -> bool:
    a, b = os.path.normcase(str(p)), os.path.normcase(str(base)).rstrip("\\/")
    return a == b or a.startswith(b + os.sep)


def assert_under_g201(path) -> Path:
    r"""Every gen20.1 write target must resolve inside generations/gen20_1 or inside the system temporary
    directory (scratch); anything else, in the repository or not, is refused.  On Windows a Git-Bash drive
    path such as ``/d/ml_separator_gh/...`` is refused before resolving: Python would turn it into
    ``D:\d\ml_separator_gh\...`` (IMPLEMENTATION_PLAN.md §8, the runner.py:324 hazard)."""
    raw = os.fspath(path)
    if os.name == "nt" and _GITBASH_DRIVE.match(raw):
        raise PermissionError(f"refusing {raw!r}: a Git-Bash drive path ('/d/...') resolves to "
                              r"'<current drive>:\d\...' under Windows Python; pass a Windows path (D:/...) or a "
                              "path relative to the repository root")
    p = Path(raw).resolve()
    if _within(p, G201.resolve()):
        return p
    if _within(p, ROOT.resolve()):
        raise PermissionError(f"refusing to write {p}: outside generations/gen20_1")
    if any(_within(p, s) for s in scratch_roots()):
        return p          # scratch under the system temporary directory
    raise PermissionError(f"refusing to write {p}: outside generations/gen20_1 and outside the temporary directory "
                          f"{[str(s) for s in scratch_roots()]}")
