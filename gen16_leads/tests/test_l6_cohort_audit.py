"""L6 regression tests: the switched rebuild must reproduce gen13's frozen cohorts exactly.

    /d/ml_separator_gh/.venv/Scripts/python.exe -m pytest gen16_leads/tests/test_l6_cohort_audit.py

Run from the repo root.  These are the guardrails for `gen16/cohort_audit.py`: if any of them
fails, no number in results/L6_cohort_audit/ may be trusted.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

GEN16_ROOT = Path(__file__).resolve().parents[1]
if str(GEN16_ROOT) not in sys.path:
    sys.path.insert(0, str(GEN16_ROOT))

from gen16 import cohort_audit as ca            # noqa: E402
from gen13sep import paths                      # noqa: E402

FROZEN = {"exact": "4c3c6628ea0be949", "relaxed": "97fb945371e30ac5",
          "series": "f0270097d692aaf3"}


def _fp(frame: pd.DataFrame) -> str:
    fr = frame.drop(columns=[c for c in ("n_publications_in_cell",) if c in frame.columns])
    cols = sorted(c for c in fr.columns if c != "safe_exp_ids")
    payload = pd.util.hash_pandas_object(fr[cols], index=False).to_numpy().tobytes()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


@pytest.fixture(scope="module")
def bundle():
    return ca.load_bundle_raw()


@pytest.mark.parametrize("mode", sorted(FROZEN))
def test_reproduces_frozen_key_mode(bundle, mode):
    c = ca.build_cohort_switched(ca.Switches(key_mode=mode), bundle=bundle)
    frozen = pd.read_parquet(paths.MANIFEST_DIR / f"cohort_{mode}.parquet")
    assert c.audit["n_cells"] == len(frozen)
    assert c.audit["n_extractants"] == frozen["extractant"].nunique()
    assert c.audit["n_chemotypes"] == frozen["chemotype"].nunique()
    assert _fp(c.frame) == FROZEN[mode]


def test_frozen_headline_numbers(bundle):
    c = ca.build_cohort_switched(ca.Switches(), bundle=bundle)
    assert (c.audit["n_cells"], c.audit["n_extractants"], c.audit["n_chemotypes"]) == (521, 90, 45)


def test_kish_matches_gen13_published_11_7():
    """gen13's 11.7 is effective_sample_size over DISTINCT EXTRACTANTS per chemotype."""
    frozen = pd.read_parquet(paths.MANIFEST_DIR / "cohort_exact.parquet")
    assert ca.kish_by_extractant(frozen) == pytest.approx(11.6715, abs=1e-3)
    assert ca.kish_by_cell(frozen) == pytest.approx(1.9134, abs=1e-3)


def test_series_mode_uses_the_relaxed_column_set(bundle):
    """cohort.py:140 makes `series` relaxed+series_id, NOT exact+series_id."""
    b = ca.cell_keys(bundle.copy(), ca.Switches(key_mode="series"))
    assert set(ca.RELAXABLE_CONDITION_COLUMNS).isdisjoint(b.attrs["key_cols"])


def test_no_extractant_with_two_lanthanides_is_excluded(bundle):
    """The L6 headline: gen13's filters throw away no measurable extractant."""
    per = bundle.groupby("canonical_smiles")["metal"].nunique()
    frozen = pd.read_parquet(paths.MANIFEST_DIR / "cohort_exact.parquet")
    assert set(per[per >= 2].index) == set(frozen["extractant"])
    assert int((per < 2).sum()) == 100


@pytest.mark.parametrize("sw", [
    ca.Switches(todga_quarantine=False), ca.Switches(sentinel_drop=False),
    ca.Switches(publication_in_key=False), ca.Switches(key_mode="relaxed"),
    ca.Switches(key_mode="series"), ca.Switches(key_mode="none"),
    ca.Switches(require_chemotype=False),
])
def test_no_relaxation_adds_a_chemotype(bundle, sw):
    c = ca.build_cohort_switched(sw, bundle=bundle)
    assert c.audit["n_chemotypes"] <= 45
    assert c.audit["n_extractants"] <= 90


def test_frozen_chemotype_map_is_single_linkage_tanimoto_07():
    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")
    clus = ca.single_linkage_clusters(list(chem.index), 0.7)
    cl = pd.DataFrame({"mine": list(clus.values())},
                      index=list(clus)).join(chem["chem__supercluster"])
    assert cl.groupby("mine")["chem__supercluster"].nunique().eq(1).all()
    assert cl.groupby("chem__supercluster")["mine"].nunique().eq(1).all()
