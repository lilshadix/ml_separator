"""Ingest invariants (DESIGN.md section 12.2, row 3) against the built database.

These tests read ``systems/*.csv``, ``results/audit/*.json`` and one entry; they skip with a
message when ``scripts/g18_build_db.py`` and ``scripts/g18_audit.py`` have not been run.  The
``slow``-marked test re-runs the ingest into a temporary directory.

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_ingest.py -q \
          -m "not slow"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))
from gen18proc import ingest as I  # noqa: E402
from gen18proc import paths  # noqa: E402
from gen18proc import systems as S  # noqa: E402

RECORDS = paths.SYSTEMS_DIR / "corpus_records.csv"
INGEST_AUDIT = paths.RESULTS_AUDIT_DIR / "ingest_audit.json"
AUDIT = paths.RESULTS_AUDIT_DIR / "audit.json"
SASAKI = "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1"
EXPECTED_SERIES = {
    "ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1",
    "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1",
    "ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.1",
    "ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.2",
    "ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.3",
    "ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1",
    "ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2",
    "ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2",
    "ls_sys_a7195d8a9d8696e0_Nd_pub_15b174237f_1_0.2",
    "ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1",
}


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    if not RECORDS.exists():
        pytest.skip("database not built: run scripts/g18_build_db.py")
    return pd.read_csv(RECORDS, dtype={"fit_ineligible_reason": object, "duplicate_flag": object,
                                       "loading_series_id": object})


@pytest.fixture(scope="module")
def audits() -> tuple[dict, dict]:
    if not (INGEST_AUDIT.exists() and AUDIT.exists()):
        pytest.skip("audit not run: run scripts/g18_build_db.py then scripts/g18_audit.py")
    return (json.loads(INGEST_AUDIT.read_text(encoding="utf-8")),
            json.loads(AUDIT.read_text(encoding="utf-8")))


def test_only_design_columns_are_read():
    """Section 4.1: the base columns plus every ``cond__*`` column; nothing else."""
    cols = I.bundle_columns()
    assert cols[:len(I.BUNDLE_BASE_COLUMNS)] == list(I.BUNDLE_BASE_COLUMNS)
    extra = [c for c in cols if c not in I.BUNDLE_BASE_COLUMNS and not c.startswith("cond__")]
    assert extra == []
    assert 64 <= sum(c.startswith("cond__") for c in cols) <= 70
    assert len(cols) < 100
    assert set(I.PROVENANCE_COLUMNS) == {"safe_exp_id", "publication_id", "experiment_series_id",
                                          "replicate_id", "condition_id", "cell_id"}


def test_ingest_audit_counts_equal_data_audit(audits):
    ingest, audit = audits
    for key in ("rows_in", "todga_name_mismatch_rows", "sentinel_rows", "rows_after_quarantine",
                "n_records", "n_publications", "n_loading_series_publication_aware",
                "n_loading_series_publication_blind", "n_unit_slip_rows", "n_tied_d_groups",
                "n_tied_d_rows", "replicate_groups", "n_nan_metal_rows", "n_nan_temperature_rows",
                "rows_fit_ineligible", "n_unit_slip_groups", "n_tied_d_tier_groups"):
        assert ingest[key] == audit[key], key
    assert ingest["n_systems"] == audit["n_systems_corpus"]
    assert ingest["replicate_sd_median"] == pytest.approx(audit["replicate_sd_median"], abs=1e-12)
    text = paths.DATA_AUDIT_MD.read_text(encoding="utf-8")
    assert f"{audit['rows_in']} -> {audit['rows_after_quarantine']}" in text
    assert f"**{audit['e1_groups']} groups in {audit['e1_systems']} systems**" in text
    assert f"**{audit['n_unit_slip_rows']}**" in text


def test_exactly_seven_unit_slip_rows_all_on_the_3M_side(records):
    slip = records[records["duplicate_flag"] == "UNIT_SLIP_DUPLICATE"]
    assert len(slip) == 7
    assert set(slip["publication_id"]) == {"pub_0e7f3e0563"}
    assert (slip["acid_nominal_M"] == 3.0).all()
    assert set(slip["record_id"]) == {f"Ca_SAFE:{i}" for i in range(2693, 2700)}
    assert (~slip["fit_eligible"]).all()
    assert (slip["fit_ineligible_reason"] == "UNIT_SLIP_DUPLICATE").all()
    pub = records[(records["publication_id"] == "pub_0e7f3e0563") & (records["metal"] == "Nd")]
    one_m = pub[pub["acid_nominal_M"] == 1.0]
    assert (one_m["fit_eligible"]).all() and len(one_m) == 18


def test_ten_publication_aware_loading_series(records):
    series = pd.read_csv(paths.SYSTEMS_DIR / "series.csv")
    aware = series[series["definition"] == "publication_aware"]
    assert len(aware) == 10 and set(aware["loading_series_id"]) == EXPECTED_SERIES
    assert (series["definition"] == "publication_blind").sum() == 11
    on_records = set(records["loading_series_id"].dropna())
    assert on_records == EXPECTED_SERIES
    d3 = aware.set_index("loading_series_id")
    assert d3.loc["ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2", "n_points"] == 8
    assert d3.loc["ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2", "n_points"] == 18
    tracers = records[records["is_tracer"]]
    assert len(tracers) == 10 and tracers.groupby("loading_series_id").size().eq(1).all()
    t = tracers.set_index("loading_series_id").loc[SASAKI]
    assert t["record_id"] == "Ca_SAFE:2222" and t["metal_initial_mM"] == 4.9


def test_nan_acid_rows_are_fit_ineligible(records):
    nan_acid = records[records["acid_nominal_M"].isna()]
    assert len(nan_acid) == 11
    assert (~nan_acid["fit_eligible"]).all()
    assert (nan_acid["fit_ineligible_reason"] == "acid_nan_or_nonpositive").all()
    nan_ext = records[records["ligand_M"].isna() | (records["ligand_M"] <= 0)]
    assert len(nan_ext) == 1 and (~nan_ext["fit_eligible"]).all()
    assert records["fit_eligible"].sum() == len(records) - 11 - 1 - 7


def test_quarantine_and_identity_counts(records):
    excl = pd.read_csv(paths.SYSTEMS_DIR / "exclusions.csv")
    assert (excl["reason"] == "todga_name_mismatch").sum() == 129
    assert (excl["reason"] == "sentinel_logD_le_-6").sum() == 3
    assert len(records) == 5860 and records["publication_id"].nunique() == 105
    assert records["system_id"].nunique() == 287
    assert (records["system_id"] == "sys_5cb78e5000d40860").sum() == 514
    assert records["record_id"].is_unique


def test_built_todga_entry_and_phase_literature(records):
    path = paths.SYSTEMS_DIR / "sys_5cb78e5000d40860.json"
    if not path.exists():
        pytest.skip("database not built")
    entry = S.load_system(path)
    assert len(entry.records) == 514 and entry.origin == "corpus"
    assert entry.phase.loc_metal_M.value == 0.008
    assert entry.phase.loc_metal_M.status is S.ProvStatus.LITERATURE
    assert entry.phase.loc_metal_M.source.doi == "10.1081/SEI-120016073"
    rec = {r.record_id: r for r in entry.records}["Ca_SAFE:2222"]
    assert rec.is_tracer and rec.loading_series_id == SASAKI
    assert rec.acid_eq_M is None and rec.oa_ratio is None and rec.anion_M == 3.0
    assert "ASSUMED initial aqueous" in rec.provenance.note
    dom = entry.applicability["TODGA|20-30C"]
    assert dom.hull_vertices is not None and len(dom.hull_vertices) >= 3
    assert dom.n_publications >= 2 and dom.loading_fraction is not None
    reg = S.load_registry(paths.SYSTEMS_DIR)
    assert len(reg) == 289 and set(reg["origin"]) == {"corpus", "literature"}


@pytest.mark.slow
def test_full_ingest_reproduces_the_audit(tmp_path, audits):
    ingest, _ = audits
    audit = I.ingest_corpus(tmp_path)
    for key in ("rows_in", "n_systems", "n_records", "n_publications",
                "n_loading_series_publication_aware", "n_loading_series_publication_blind",
                "n_unit_slip_rows", "n_tied_d_groups", "replicate_groups"):
        assert getattr(audit, key) == ingest[key], key
    assert (tmp_path / "corpus_records.csv").read_bytes() == RECORDS.read_bytes()
