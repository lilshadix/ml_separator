"""Tests for ``gen18proc.screen`` (DESIGN.md section 12.2, row ``test_screen``).

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_screen.py -q
       -m "not slow"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import paths  # noqa: E402
from gen18proc import screen as S  # noqa: E402

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"


def test_direction_prior_is_none_when_joblib_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "GEN15_DEPLOY", tmp_path / "absent" / "deploy_g15.joblib")
    assert S.gen15_available() is False
    assert S.direction_prior(TODGA, ("Nd", "Pr")) is None


def test_direction_prior_is_none_when_script_fails(monkeypatch, tmp_path):
    fake = tmp_path / "deploy_g15.joblib"
    fake.write_bytes(b"not a model")
    monkeypatch.setattr(paths, "GEN15_DEPLOY", fake)
    monkeypatch.setattr(paths, "GEN15_PREDICT_SCRIPT", tmp_path / "no_such_script.py")
    assert S.direction_prior(TODGA, ("Nd", "Pr")) is None


def test_parse_direction_line():
    out = "SMILES\n  direction   heavy-selective  (p_heavy 0.8123, confidence 0.6246)\n"
    assert S._parse_direction(out) == ("heavy-selective", 0.8123)
    assert S._parse_direction("nothing here") == (None, None)


def test_v6_rejects_gen15_sourced_record():
    """Validator V6 belongs to WB1 (``systems.validate_entry``); this test runs once it lands
    and is skipped (never passed silently) until then."""
    systems = pytest.importorskip("gen18proc.systems")
    validate_entry = getattr(systems, "validate_entry", None)
    literature = pytest.importorskip("gen18proc.literature")
    make_entry = getattr(literature, "pc88a_prnd_entry", None)
    if validate_entry is None or make_entry is None:
        pytest.skip("WB1 validate_entry / pc88a_prnd_entry not available yet")
    import dataclasses
    import math

    from gen18proc.types import DistributionRecord, Provenance, ProvStatus, Source
    entry = make_entry()
    rec = DistributionRecord(
        record_id="doi:none#gen15#1", metal="Nd", d=2.0, log_d=math.log10(2.0),
        acid_nominal_M=0.1, acid_eq_M=None, anion_M=0.31, ligand_M={"PC88A": 0.8},
        complexant_M=None, metals_initial_mM={"Nd": 10.0}, oa_ratio=1.0, temperature_C=25.0,
        contact_time_min=None, diluent_name="kerosene", publication_id=None,
        experiment_series_id=None, replicate_id=None, loading_series_id=None, is_tracer=True,
        fit_eligible=False, fit_ineligible_reason="gen15", duplicate_flag=None,
        provenance=Provenance(ProvStatus.LITERATURE,
                              Source("model", None, "deploy_g15 prediction"),
                              model_id="gen15 deploy_g15.joblib", note="must be refused"))
    bad = dataclasses.replace(entry, records=entry.records + (rec,))
    violations = validate_entry(bad)
    assert any(v.level == "error" and "V6" in v.message for v in violations)


@pytest.mark.slow
def test_direction_prior_real_model_when_present():
    if not S.gen15_available():
        pytest.skip("gen15 deploy model absent")
    prior = S.direction_prior(TODGA, ("Nd", "Pr"))
    assert prior is not None
    assert set(prior) == {"pair", "sign", "source", "note"}
    assert prior["pair"] == ("Nd", "Pr") and prior["sign"] in (1, -1)
    assert prior["source"] == "gen15 deploy_g15.joblib"
    assert prior["note"].startswith("pre-screen only; not a D source")
    flipped = S.direction_prior(TODGA, ("Pr", "Nd"))
    assert flipped["sign"] == -prior["sign"]
