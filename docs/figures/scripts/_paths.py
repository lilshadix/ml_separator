"""Run-artefact locations, resolved from the repository root.

Every figure script imports this module and never hard-codes a path.

Resolution order for anything under ``runs/``:

1. the current checkout — after gen7-gen11 were committed this is normally
   enough, and it is what a clean clone gets;
2. ``$MLSEP_RUNS_EXTRA`` if set — point it at any tree that holds a ``runs/``
   directory (another checkout, an unpacked archive, a mounted share);
3. any git worktree of this repository, discovered via ``git worktree list``;
4. the historical agent worktree, kept only so an existing machine keeps working.

Steps 3 and 4 exist because gen10 and gen11 were produced in a worktree.  None
of them is required on a clean clone.  All are read-only inputs: no figure
script writes into ``runs/``.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: Kept so machines that still have the original agent worktree keep resolving.
LEGACY_WORKTREE = ROOT / ".claude" / "worktrees" / "lanthanide-separation-finalize-81154b"


@lru_cache(maxsize=1)
def search_roots() -> tuple[Path, ...]:
    """Trees that may hold a ``runs/`` directory, in resolution order."""
    roots: list[Path] = [ROOT]
    extra = os.environ.get("MLSEP_RUNS_EXTRA")
    if extra:
        roots.append(Path(extra).expanduser().resolve())
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False).stdout
        for line in out.splitlines():
            if line.startswith("worktree "):
                roots.append(Path(line[len("worktree "):]).resolve())
    except (OSError, subprocess.SubprocessError):
        pass
    roots.append(LEGACY_WORKTREE)
    seen: dict[Path, None] = {}
    for r in roots:
        seen.setdefault(r, None)
    return tuple(seen)


def _first(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "none of these exist:\n  " + "\n  ".join(str(c) for c in candidates)
        + "\n\nIf the artefact lives in another checkout, set MLSEP_RUNS_EXTRA to "
          "the directory that contains its runs/ folder.")


def run(relative: str) -> Path:
    """A path under ``runs/``, searched across every root in ``search_roots()``."""
    return _first(*(root / "runs" / relative for root in search_roots()))


def docs(relative: str) -> Path:
    return _first(*(root / "docs" / relative for root in search_roots()))


#: Rebuild output. Deleted 2026-09-04 and gitignored; recreated on demand.
FIG = ROOT / "figures"
MAIN = FIG / "main"
SUPP = FIG / "supplementary"
DERIVED = FIG / "derived"


def ensure_output_dirs() -> None:
    """Create the figure output directories."""
    for d in (MAIN, SUPP, DERIVED):
        d.mkdir(parents=True, exist_ok=True)


# The twenty plot/prepare scripts write into these directories and none of them
# creates one itself, so this stays an import-time side effect as it always was.
ensure_output_dirs()
