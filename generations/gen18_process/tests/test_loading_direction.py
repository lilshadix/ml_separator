"""Validation test of DESIGN.md section 12.3 (marker ``validation``): on every loading-active
series (log D range >= 0.3; the publication-aware definition of ``systems/series.csv``) the
descriptive Spearman correlation of log D with log metal concentration is negative.  The
pre-registration (section 7 (c)) expects the rising TODGA/Ce series to fail this: that failure
is a reported result, not a suite blocker, and the series is not excluded anywhere.  One test
per series so the report can name which series fell.  Skipped when the database is absent.

Run:  .venv/Scripts/python.exe -m pytest \\
          generations/gen18_process/tests/test_loading_direction.py -q -m validation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import paths  # noqa: E402

RECORDS = paths.SYSTEMS_DIR / "corpus_records.csv"
SERIES = paths.SYSTEMS_DIR / "series.csv"

pytestmark = pytest.mark.validation


def _active_series() -> list[str]:
    if not SERIES.exists():
        return []
    s = pd.read_csv(SERIES)
    s = s.loc[(s["definition"] == "publication_aware") & s["loading_active"].astype(bool)]
    return sorted(s["loading_series_id"].astype(str))


@pytest.mark.skipif(not RECORDS.exists() or not SERIES.exists(),
                    reason="database not built (g18_build_db.py)")
@pytest.mark.parametrize("series_id", _active_series() or ["<no series>"])
def test_loading_active_series_spearman_negative(series_id: str):
    if series_id == "<no series>":
        pytest.skip("no loading-active series in series.csv")
    records = pd.read_csv(RECORDS, low_memory=False)
    sub = records.loc[records["loading_series_id"].astype("string") == series_id]
    sub = sub.loc[sub["fit_eligible"].astype(str).str.lower().isin({"true", "1"})]
    mm = pd.to_numeric(sub["metal_initial_mM"], errors="coerce").to_numpy(dtype=float)
    logd = pd.to_numeric(sub["log_d"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(mm) & (mm > 0) & np.isfinite(logd)
    assert ok.sum() >= 3, f"{series_id}: fewer than 3 usable points"
    rho, p = spearmanr(np.log10(mm[ok]), logd[ok])
    print(f"{series_id}: Spearman(log mM, log D) = {rho:+.3f} (p = {p:.3g}, n = {int(ok.sum())})")
    assert rho < 0, (f"{series_id}: Spearman = {rho:+.3f} is not negative (D does not fall with "
                     "loading in this series): reported result, series not excluded")
