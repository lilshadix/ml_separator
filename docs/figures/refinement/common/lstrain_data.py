"""Data access for the refined lanthanidestrain figures.

The upstream repository is ~111 MB and its model artefacts are not committed, so the
refined scripts read the vendored copy of its result tables instead:
``figure_refinement/lanthanidestrain/data/`` — every ``automl/reports/*.csv`` and
``*.json`` at the commit recorded in ``data/PROVENANCE.json``.

Nothing here recomputes a metric.  The refinement pass is layout, typography and
information design; the numbers are the ones the upstream tables already carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "lanthanidestrain" / "data"


def provenance() -> dict:
    return json.loads((DATA / "PROVENANCE.json").read_text())


def path(name: str) -> Path:
    """Resolve a table by bare name, with or without its extension."""
    for candidate in (DATA / name, DATA / f"{name}.csv", DATA / f"{name}.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{name!r} is not among the vendored tables in {DATA}. "
        "If it lives under automl/artifacts/ upstream it is not committed there either — "
        "see figure_refinement/lanthanidestrain/REPRODUCIBILITY.md.")


def read(name: str) -> pd.DataFrame:
    """A vendored CSV as a DataFrame."""
    return pd.read_csv(path(name))


def read_json(name: str) -> dict | list:
    return json.loads(path(name).read_text())


def available(pattern: str = "*") -> list[str]:
    return sorted(p.name for p in DATA.glob(pattern) if p.suffix in {".csv", ".json"})
