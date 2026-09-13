"""``report.py`` -- tables that carry their regime, and manifests (DESIGN.md section 10.6).

Every number in gen18 is quoted with its regime: the **cohort** it was computed on, the
**hold-out** design, the **averaging unit**, and -- for regime (process) tables -- the **status of
the parameters** that produced it.  ``write_table`` refuses (``ValueError``) a table without those
keys, writes the regime as a ``# regime:`` header line in CSV and as the caption of a markdown
table; ``read_table`` reads the CSV back with its regime.  ``markdown_table`` renders a DataFrame
without any third-party dependency.  ``manifest`` hashes inputs and outputs (SHA-256) and records
the seed, the arguments and the git HEAD -- never a wall-clock value (section 1.5).
"""
from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from gen18proc.paths import REPO_ROOT, sha256_of

__all__ = [
    "REGIME_KEYS", "REGIME_TABLE_KEYS", "write_table", "read_table", "markdown_table",
    "manifest", "write_manifest", "regime_line", "check_regime",
]

REGIME_KEYS: tuple[str, ...] = ("cohort", "holdout", "averaging_unit")
"""Keys every table must carry (DESIGN.md section 10.6)."""

REGIME_TABLE_KEYS: tuple[str, ...] = REGIME_KEYS + ("status_of_parameters",)
"""Keys a regime (process / recipe / case) table must carry in addition."""

_REGIME_PREFIX = "# regime: "


def check_regime(regime: Mapping[str, Any] | None, *, regime_table: bool = False) -> dict[str, str]:
    """Validate the regime mapping; return it with string values.  ``ValueError`` when a key
    is missing or empty (the refusal of section 10.6)."""
    if regime is None or not isinstance(regime, Mapping):
        raise ValueError("write_table needs a regime mapping with keys "
                         f"{REGIME_TABLE_KEYS if regime_table else REGIME_KEYS}")
    required = REGIME_TABLE_KEYS if regime_table else REGIME_KEYS
    missing = [k for k in required if k not in regime or regime[k] in (None, "")]
    if missing:
        raise ValueError(f"table refused: regime is missing {missing}; required {required}")
    return {str(k): str(v) for k, v in regime.items()}


def regime_line(regime: Mapping[str, Any]) -> str:
    """``key=value; key=value`` in a fixed key order (required keys first, then the rest
    sorted) so the header is byte-stable across runs."""
    reg = {str(k): str(v) for k, v in regime.items()}
    ordered = [k for k in REGIME_TABLE_KEYS if k in reg] + sorted(
        k for k in reg if k not in REGIME_TABLE_KEYS)
    return "; ".join(f"{k}={reg[k].replace(';', ',').replace(chr(10), ' ')}" for k in ordered)


def _fmt(v: Any, floatfmt: str) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return format(v, floatfmt)
    if isinstance(v, (bool,)):
        return str(v)
    if isinstance(v, (int,)):
        return str(v)
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return _fmt(v.item(), floatfmt)
        except (AttributeError, ValueError):
            pass
    return str(v).replace("|", "\\|").replace("\n", " ")


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4g", *, index: bool = False,
                   caption: str | None = None) -> str:
    """A GitHub-flavoured markdown table; floats formatted with ``floatfmt``; an optional
    caption line above it.  Empty frames render the header row only."""
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = []
    if caption:
        lines.append(caption)
        lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(v, floatfmt) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, path: str | Path, *, regime: Mapping[str, Any],
                regime_table: bool = False, floatfmt: str = ".4g",
                float_format: str | None = None, index: bool = False) -> Path:
    """Write ``df`` with its regime (DESIGN.md section 10.6).

    Refuses (``ValueError``) unless ``regime`` has ``cohort``, ``holdout``, ``averaging_unit``
    (and ``status_of_parameters`` when ``regime_table`` is true).  ``.csv``: first line
    ``# regime: k=v; ...`` then the CSV (``pandas.read_csv(comment="#")`` skips it);
    ``.md`` / ``.markdown``: the regime as the caption line ``*regime: ...*`` above a markdown
    table.  Returns the path written.
    """
    reg = check_regime(regime, regime_table=regime_table)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    line = regime_line(reg)
    if out.suffix.lower() in {".md", ".markdown"}:
        text = markdown_table(df, floatfmt, index=index, caption=f"*regime: {line}*")
        out.write_text(text, encoding="utf-8", newline="\n")
    else:
        csv = df.to_csv(index=index, float_format=float_format, lineterminator="\n")
        out.write_text(_REGIME_PREFIX + line + "\n" + csv, encoding="utf-8", newline="\n")
    return out


def read_table(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a CSV written by ``write_table``: ``(frame, regime)``; a file without the header
    line returns an empty regime."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    regime: dict[str, str] = {}
    if first.startswith(_REGIME_PREFIX):
        body = first[len(_REGIME_PREFIX):].strip()
        for item in body.split("; "):
            k, _, v = item.partition("=")
            if k:
                regime[k] = v
        df = pd.read_csv(p, skiprows=1)
    else:
        df = pd.read_csv(p)
    return df, regime


def _git_head() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _hash_paths(paths: Iterable[str | Path]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in paths:
        pp = Path(p)
        key = pp.relative_to(REPO_ROOT).as_posix() if pp.is_absolute() and \
            pp.resolve().is_relative_to(REPO_ROOT.resolve()) else pp.as_posix()
        out[key] = sha256_of(pp) if pp.is_file() else None
    return dict(sorted(out.items()))


def manifest(paths: Sequence[str | Path], inputs: Sequence[str | Path], seed: int, *,
             arguments: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None,
             ) -> dict[str, Any]:
    """Manifest dict (DESIGN.md section 10.6 / 11): SHA-256 of every input and output path
    (``None`` for a path that does not exist), the ``seed``, the CLI ``arguments``, the git
    HEAD, and ``extra`` free entries.  No wall-clock values."""
    return {
        "schema": "gen18.manifest.1",
        "seed": int(seed),
        "git_head": _git_head(),
        "arguments": dict(arguments or {}),
        "inputs": _hash_paths(inputs),
        "outputs": _hash_paths(paths),
        **({"extra": dict(extra)} if extra else {}),
    }


def write_manifest(path: str | Path, paths: Sequence[str | Path], inputs: Sequence[str | Path],
                   seed: int, **kw: Any) -> Path:
    """``manifest(...)`` written as sorted, indented JSON; returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest(paths, inputs, seed, **kw), indent=2, sort_keys=True,
                              ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    return out
