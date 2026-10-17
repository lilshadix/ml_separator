"""Write protection for the frozen trees (IMPLEMENTATION_PLAN.md §8 "forbidden" row; LT-26).

* ``install()`` replaces every forbidden writer of a frozen package with a function that raises
  ``ForbiddenWriterError``: gen20k.manifest (Run, preregistration_state, require_sealed_preregistration,
  seal_preregistration, environment), gen20k.runner.run, gen20k.report (load_tables, build_design,
  figures, setting_b_tables), gen13sep.runner.run_ladder, gen13sep.cohort.write_cohort and
  gen20k.nnc.NNC; ``gen14.dirbench.load`` is wrapped so that a rebuild (or a missing cache, which would
  trigger one) raises.  Modules whose import pulls in torch (gen20k.runner, gen20k.nnc) are never
  imported here: they are patched if already loaded, and otherwise the moment they are imported (a
  ``sys.meta_path`` post-import hook).
* ``snapshot()`` / ``diff(before, after)`` record (size, mtime_ns) of every file of the repository outside
  generations/gen20_1 (skipping only .git, .venv and .claude) plus the three generations/gen20_1/*.md
  documents, so a test or script can prove it wrote nothing there (including ``__pycache__``).  The seal
  step alone exempts PRE_REGISTRATION.md through ``exclude``.

Nothing here writes anything.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

from . import bootstrap


class ForbiddenWriterError(RuntimeError):
    """A forbidden writer of a frozen generation was called (IMPLEMENTATION_PLAN.md §8)."""


#: module -> attributes that write into a frozen tree, check gen20's seal, or import torch
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "gen20k.manifest": ("Run", "preregistration_state", "require_sealed_preregistration", "seal_preregistration",
                        "environment"),
    "gen20k.runner": ("run",),
    "gen20k.report": ("load_tables", "build_design", "figures", "setting_b_tables"),
    "gen13sep.runner": ("run_ladder",),
    "gen13sep.cohort": ("write_cohort",),
    "gen20k.nnc": ("NNC",),
}
#: modules that import torch: never imported by the guard itself
_LAZY = ("gen20k.runner", "gen20k.nnc")

#: read-only inputs outside generations/ that gen20.1 reads (named for the record; the snapshot covers them
#: and everything else of the repository anyway)
READ_ONLY_INPUTS: tuple[Path, ...] = (
    bootstrap.ROOT / "src",
    bootstrap.ROOT / "dataset with 3D structures",
    bootstrap.ROOT / "runs" / "gen6_provenance",
    bootstrap.ROOT / "runs" / "gen7_architecture" / "cache",
    bootstrap.GENERATIONS / "gen12_2_eu_pred" / "features",
)
#: the gen20.1 documents this build must not modify (only g201_seal.py's seal step may append the footer to
#: PRE_REGISTRATION.md, through ``snapshot(exclude=(bootstrap.PREREG,))``)
G201_DOCUMENTS: tuple[Path, ...] = (bootstrap.PREREG, bootstrap.PLAN, bootstrap.G201 / "README.md")
PROTECTED: tuple[Path, ...] = tuple(bootstrap.FROZEN_TREES) + READ_ONLY_INPUTS + G201_DOCUMENTS
#: directories the default snapshot does not walk (repo-relative posix): git's own store, the interpreter,
#: Claude Code's harness configuration (written by the harness, never by Python), and gen20.1's own tree
SNAPSHOT_SKIP: tuple[str, ...] = (".git", ".venv", ".claude", "generations/" + bootstrap.G201.name)

_INSTALLED: dict[tuple[str, str], object] = {}


def _raiser(qualname: str):
    def _forbidden(*args, **kwargs):
        raise ForbiddenWriterError(f"{qualname} is forbidden in gen20.1 (IMPLEMENTATION_PLAN.md §8): it writes "
                                   "into a frozen generation, checks gen20's seal, or instantiates gen20k.nnc")
    _forbidden.__name__ = qualname.rsplit(".", 1)[-1]
    _forbidden.__qualname__ = qualname
    _forbidden._gen201_forbidden = True
    return _forbidden


def _patch_module(mod) -> None:
    for attr in FORBIDDEN.get(mod.__name__, ()):
        if hasattr(mod, attr) and not getattr(getattr(mod, attr), "_gen201_forbidden", False):
            _INSTALLED.setdefault((mod.__name__, attr), getattr(mod, attr))
            setattr(mod, attr, _raiser(f"{mod.__name__}.{attr}"))


def _guard_dirbench(mod) -> None:
    load = getattr(mod, "load", None)
    if load is None or getattr(load, "_gen201_guarded", False):
        return
    _INSTALLED.setdefault((mod.__name__, "load"), load)
    cache = Path(getattr(mod, "CACHE", bootstrap.BENCH_PKL))

    def guarded_load(rebuild: bool = False):
        if rebuild or not cache.exists():
            raise ForbiddenWriterError(f"gen14.dirbench.load would rebuild and write {cache} (IMPLEMENTATION_PLAN.md §8)")
        return load(rebuild=False)
    guarded_load._gen201_guarded = True
    mod.load = guarded_load


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    """Patches the lazily imported modules as soon as something imports them."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _LAZY:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
            if spec is not None and spec.loader is not None:
                exec_module = spec.loader.exec_module

                def patched_exec(module, _exec=exec_module):
                    _exec(module)
                    _patch_module(module)
                spec.loader.exec_module = patched_exec
                return spec
        return None


_HOOK = _PostImportPatcher()


def install() -> dict:
    """Install every guard; idempotent.  Returns {module: [patched attributes]}."""
    for name in FORBIDDEN:
        if name in _LAZY:
            if name in sys.modules:
                _patch_module(sys.modules[name])
            continue
        _patch_module(importlib.import_module(name))
    _guard_dirbench(importlib.import_module("gen14.dirbench"))
    if _HOOK not in sys.meta_path:
        sys.meta_path.insert(0, _HOOK)
    return installed()


def installed() -> dict:
    out: dict[str, list[str]] = {}
    for name, attrs in FORBIDDEN.items():
        mod = sys.modules.get(name)
        out[name] = [a for a in attrs if mod is not None and getattr(getattr(mod, a, None), "_gen201_forbidden", False)]
    d = sys.modules.get("gen14.dirbench")
    out["gen14.dirbench"] = ["load"] if d is not None and getattr(getattr(d, "load", None), "_gen201_guarded", False) else []
    return out


def uninstall() -> None:
    """Restore the original callables (tests only)."""
    for (name, attr), orig in list(_INSTALLED.items()):
        mod = sys.modules.get(name)
        if mod is not None:
            setattr(mod, attr, orig)
    _INSTALLED.clear()
    if _HOOK in sys.meta_path:
        sys.meta_path.remove(_HOOK)


# --------------------------------------------------------------------------------------
# file snapshots
# --------------------------------------------------------------------------------------
def _rel(path) -> str | None:
    """Repository-relative posix path, or None for a path outside the repository."""
    try:
        return Path(path).resolve().relative_to(bootstrap.ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _walk(top: str, rel: str, skip: set, excl: set, out: dict) -> None:
    try:
        it = os.scandir(top)
    except OSError:  # pragma: no cover
        return
    with it:
        for e in it:
            r = f"{rel}/{e.name}" if rel else e.name
            if r in excl:
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    if r not in skip:
                        _walk(e.path, r, skip, excl, out)
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError:  # pragma: no cover
                continue
            out[r] = (int(st.st_size), int(st.st_mtime_ns))


def snapshot(trees=None, *, exclude=()) -> dict[str, tuple[int, int]]:
    """{posix path relative to the repo root: (size, mtime_ns)}.

    ``trees`` None (default): every file of the repository except under ``SNAPSHOT_SKIP`` (.git, .venv,
    .claude, generations/gen20_1), plus the three ``G201_DOCUMENTS``.  Otherwise every file under the given
    directories (or the given files).  ``exclude``: files or directories left out (the seal step exempts
    PRE_REGISTRATION.md); paths outside the repository are ignored there."""
    excl = {r for r in (_rel(p) for p in exclude) if r is not None}
    out: dict[str, tuple[int, int]] = {}
    if trees is None:
        _walk(str(bootstrap.ROOT), "", set(SNAPSHOT_SKIP), excl, out)
        trees = G201_DOCUMENTS
    for t in trees:
        t = Path(t)
        r = _rel(t)
        if r is None or r in excl or not t.exists():
            continue
        if t.is_file():
            st = t.stat()
            out[r] = (int(st.st_size), int(st.st_mtime_ns))
        else:
            _walk(str(t), r, set(), excl, out)
    return out


def diff(before: dict, after: dict) -> dict[str, list[str]]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def assert_unchanged(before: dict, after: dict | None = None, *, exclude=()) -> None:
    """Raise if any file of the default snapshot (or ``after``) differs from ``before``."""
    d = diff(before, snapshot(exclude=exclude) if after is None else after)
    if any(d.values()):
        raise AssertionError(f"frozen trees changed: { {k: v[:10] for k, v in d.items() if v} }")
