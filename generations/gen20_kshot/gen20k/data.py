"""The frozen gen13 cohort as gen20 sees it: cells, metals, pairs and the orientation contract.

* ``load_bench()`` returns gen14's cached ``BenchData`` and **asserts column-wise equality** with
  the frozen manifest ``gen13_separation/manifests/cohort_exact.parquet`` (identity columns, every
  ``logD__`` value NaN-aware, every other column), because ``Gen13Cohort.fingerprint()`` is not
  stable across pandas versions on this machine (AUDIT.md A.6) while the content is.
* ``y(A/B) = log10 D(A) - log10 D(B)`` with **A the lighter metal** (``ia < ib`` in the
  ``LANTHANIDES`` order), gen13's convention.  ``to_orientation`` converts to heavier-first for a
  caller that wants it; the two orientations are never mixed inside gen20.
* Strict adjacency is ``|Z_A - Z_B| == 1`` (promethium is absent, so Nd-Sm is not adjacent);
  gen13's adjacency (``dZ == 1`` or Nd-Sm) is kept as a named secondary.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bootstrap  # noqa: F401
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES  # noqa: E402

N_LN = len(LANTHANIDES)
Z_VEC = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES])
IDX = {m: i for i, m in enumerate(LANTHANIDES)}
MANIFEST_PARQUET = bootstrap.ROOT / "generations" / "gen13_separation" / "manifests" / "cohort_exact.parquet"
MANIFEST_JSON = bootstrap.ROOT / "generations" / "gen13_separation" / "manifests" / "cohort_exact.json"
IDENTITY = ("cell_id", "extractant", "publication_id", "condition_key", "chemotype", "ecfp_cluster",
            "n_metals", "metals")


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_matches_manifest(frame: pd.DataFrame) -> dict:
    """Column-wise equality of the live cohort frame with the frozen manifest parquet."""
    ref = pd.read_parquet(MANIFEST_PARQUET)
    if frame.shape != ref.shape or list(frame.columns) != list(ref.columns):
        raise AssertionError("cohort frame differs from the frozen manifest in shape or columns")
    for col in frame.columns:
        a, b = frame[col], ref[col]
        if a.dtype.kind in "fi":
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            if not (np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(np.nan_to_num(x), np.nan_to_num(y), atol=1e-9)):
                raise AssertionError(f"cohort column {col!r} differs from the frozen manifest")
        elif not (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all():
            raise AssertionError(f"cohort column {col!r} differs from the frozen manifest")
    return {"manifest_parquet": str(MANIFEST_PARQUET.relative_to(bootstrap.ROOT)),
            "manifest_sha256": sha256_of(MANIFEST_PARQUET), "n_cells": int(len(frame))}


def load_bench(check: bool = True):
    from gen14.dirbench import load
    bench = load()
    if check:
        assert_matches_manifest(bench.frame)
    return bench


def cohort_record(bench) -> dict:
    from gen13sep.cohort import Gen13Cohort
    cols = [c for c in bench.frame.columns]
    rec = assert_matches_manifest(bench.frame)
    try:
        fp = Gen13Cohort(frame=bench.frame, condition_columns=tuple(c for c in cols if c.startswith("cond__")),
                         key_mode="exact").fingerprint()
    except Exception:  # pragma: no cover
        fp = "n/a"
    rec.update({"fingerprint_obtained": fp, "n_extractants": int(bench.frame.extractant.nunique()),
                "n_chemotypes": int(bench.frame.chemotype.nunique()),
                "n_publications": int(bench.frame.publication_id.nunique())})
    return rec


# --------------------------------------------------------------------------------------
# pairs and orientation
# --------------------------------------------------------------------------------------
def observed_metals(y_row: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(~np.isnan(y_row))]


def pair_value(y_row: np.ndarray, ia: int, ib: int) -> float:
    """``log SF(A/B)`` for ``ia < ib`` (A lighter), from the cell's replicate-mean log D."""
    if not ia < ib:
        raise ValueError("pairs are stored lighter-first: ia < ib")
    return float(y_row[ia] - y_row[ib])


def is_strict_adjacent(ia: int, ib: int) -> bool:
    return abs(int(Z_VEC[ia]) - int(Z_VEC[ib])) == 1


def is_gen13_adjacent(ia: int, ib: int) -> bool:
    lo, hi = min(ia, ib), max(ia, ib)
    return is_strict_adjacent(lo, hi) or (LANTHANIDES[lo], LANTHANIDES[hi]) == ("Nd", "Sm")


def to_orientation(y_light_first: float, orientation: str) -> float:
    """Convert gen13's lighter-first value to the requested orientation."""
    if orientation == "lighter_first":
        return float(y_light_first)
    if orientation == "heavier_first":
        return -float(y_light_first)
    raise ValueError(orientation)


def pair_label(ia: int, ib: int, orientation: str = "lighter_first") -> str:
    a, b = LANTHANIDES[ia], LANTHANIDES[ib]
    return f"{a}/{b}" if orientation == "lighter_first" else f"{b}/{a}"


@dataclass(frozen=True)
class CellView:
    """What a held-out cell exposes before any label is revealed."""
    index: int
    cell_id: str
    extractant: str
    chemotype: str
    publication: str
    metals: tuple[int, ...]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        m = self.metals
        return [(m[i], m[j]) for i in range(len(m)) for j in range(i + 1, len(m))]


def cell_view(bench, ci: int) -> CellView:
    f = bench.frame
    return CellView(index=int(ci), cell_id=str(f.cell_id.iat[ci]), extractant=str(f.extractant.iat[ci]),
                    chemotype=str(f.chemotype.iat[ci]), publication=str(f.publication_id.iat[ci]),
                    metals=tuple(observed_metals(bench.Y[ci])))


def stable_seed(*parts) -> int:
    """SHA-256-derived 32-bit seed from stable strings (never Python's salted ``hash``)."""
    s = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(s).hexdigest()[:8], 16)
