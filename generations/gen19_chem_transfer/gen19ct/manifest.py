"""``manifest.py`` -- deterministic writers and experiment tracking (brief sections 23-24).

Two files per script run:

* ``manifests/<name>.json`` -- deterministic: git commit, archive digest, arguments, seed, and the
  binary + LF-normalised digests of every input and output (repo-relative POSIX paths).  Re-running
  on the same commit and data reproduces it byte for byte.
* ``manifests/run_info/<name>.json`` -- volatile: date, hardware, python/package versions, runtime.
  Kept apart so wall-clock values never break a byte-identity check (the gen18 rule).

All text is written with ``newline="\\n"``.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gen19ct import paths

SCHEMA = "gen19.manifest.v1"


def write_json(path: Path, obj: Any) -> Path:
    paths.ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_finite(obj), fh, indent=2, sort_keys=True, ensure_ascii=False, default=_default,
                  allow_nan=False)
        fh.write("\n")
    return Path(path)


def _finite(o: Any) -> Any:
    """Strict JSON: NaN / +-inf become ``null`` (bare ``NaN`` tokens are not valid JSON)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            v = o.item()
        except (ValueError, AttributeError):
            return o
        return _finite(v) if isinstance(v, float) else v
    if isinstance(o, Mapping):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def write_csv(df: pd.DataFrame, path: Path, float_format: str = "%.6g") -> Path:
    paths.ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        df.to_csv(fh, index=False, float_format=float_format, lineterminator="\n")
    return Path(path)


def write_text(path: Path, text: str) -> Path:
    paths.ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return Path(path)


def _default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- a missing git is recorded, not fatal
        return None


def _file_record(p: Path) -> dict[str, Any]:
    p = Path(p)
    rec: dict[str, Any] = {"path": paths.rel(p)}
    rec.update(paths.digests(p) if p.exists() else {"missing": True})
    return rec


class Run:
    """Context manager: ``with Run("g19_audit_corpus", args=vars(ns), seed=0) as run: ...``;
    call ``run.inputs(...)`` / ``run.outputs(...)``; both manifests are written on exit."""

    def __init__(self, name: str, *, args: Mapping[str, Any] | None = None, seed: int | None = None,
                 extra: Mapping[str, Any] | None = None):
        self.name, self.args, self.seed = name, dict(args or {}), seed
        self.extra = dict(extra or {})
        self._in: list[Path] = []
        self._out: list[Path] = []

    def inputs(self, *ps: Path | Iterable[Path]) -> None:
        self._in.extend(_flatten(ps))

    def outputs(self, *ps: Path | Iterable[Path]) -> None:
        self._out.extend(_flatten(ps))

    def __enter__(self) -> "Run":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return
        body = {
            "schema": SCHEMA, "script": self.name, "git_head": git_head(), "seed": self.seed,
            "arguments": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.args.items()},
            "archive_master_sha256": paths.ARCHIVE_MASTER_SHA256,
            "inputs": [_file_record(p) for p in sorted(set(self._in))],
            "outputs": [_file_record(p) for p in sorted(set(self._out))],
            **self.extra,
        }
        write_json(paths.MANIFESTS_DIR / f"{self.name}.json", body)
        info = {
            "script": self.name,
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(time.perf_counter() - self._t0, 3),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "packages": _versions(["numpy", "pandas", "scipy", "sklearn", "rdkit", "catboost",
                                   "xgboost", "torch", "networkx"]),
        }
        write_json(paths.MANIFESTS_DIR / "run_info" / f"{self.name}.json", info)


def _flatten(ps: Iterable[Any]) -> list[Path]:
    out: list[Path] = []
    for p in ps:
        if isinstance(p, (str, Path)):
            out.append(Path(p))
        else:
            out.extend(Path(q) for q in p)
    return out


def _versions(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        mod = sys.modules.get(n)
        out[n] = getattr(mod, "__version__", None) if mod is not None else None
    return out
