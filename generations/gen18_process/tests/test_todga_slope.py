"""Validation test of DESIGN.md section 12.3 (marker ``validation``): the primary in-sample pooled
fit of ``sys_5cb78e5000d40860`` (TODGA / nitrate / aliphatic), band 20-30C, gives
``n_solvation`` inside the corpus-validated interval [2.36, 2.88] (gen5 MASSACTION block) and a
finite, reported jackknife SE.  A failure is a reported defect (PRE_REGISTRATION.md section 6),
never a suite blocker and never fixed by changing the fit.  Skipped when the database has not
been built.

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_todga_slope.py \\
          -q -m validation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import paths  # noqa: E402
from gen18proc.evalproto import fit_mass_action  # noqa: E402

TODGA_SYSTEM = "sys_5cb78e5000d40860"
N_RANGE = (2.36, 2.88)
RECORDS = paths.SYSTEMS_DIR / "corpus_records.csv"

pytestmark = pytest.mark.validation


@pytest.mark.skipif(not RECORDS.exists(), reason="database not built (g18_build_db.py)")
def test_todga_pooled_slope_in_corpus_validated_interval():
    records = pd.read_csv(RECORDS, low_memory=False)
    sub = records.loc[records["system_id"] == TODGA_SYSTEM]
    assert len(sub) > 0, "TODGA system has no records"
    fit = fit_mass_action(sub, ligand="TODGA", band="20-30C", seed=18, reliability=True)
    assert fit.status == "fitted"
    assert fit.slope_status["n"] == "fitted", "the ligand axis has fewer than 3 levels"
    jk = fit.jackknife_se
    assert jk is not None and np.isfinite(jk["n"][1]), "jackknife SE of n must be reported"
    print(f"TODGA in-sample n = {fit.n:.4f} (jackknife SE {jk['n'][1]:.4f}, "
          f"{fit.n_publications} publications, {fit.n_points} points); "
          f"interpretable = {fit.interpretable}")
    assert N_RANGE[0] <= fit.n <= N_RANGE[1], (
        f"TODGA n = {fit.n:.4f} outside [{N_RANGE[0]}, {N_RANGE[1]}] "
        f"(jackknife SE {jk['n'][1]:.4f}): reported defect, not fixed by changing the fit")
