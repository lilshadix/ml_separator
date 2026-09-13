"""Validator, loader and identity tests of ``gen18proc.systems`` (DESIGN.md section 12.2).

V1-V17 are each rejected by a minimal failing fixture derived from the PC88A entry of section
3.7; both example entries (3.7 and the 3.8-shaped corpus entry) load; ``write_system`` then
``load_system`` round-trips byte-identically; ``system_id`` recomputes the verified ids.

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_systems.py -q
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))
from gen18proc import literature as L  # noqa: E402
from gen18proc import paths  # noqa: E402
from gen18proc import systems as S  # noqa: E402

PC88A_ID = "sys_29976921e156a0a0"
CYANEX_ID = "sys_f02db527a94a5e86"
TODGA_ID = "sys_5cb78e5000d40860"
TODGA_SMILES = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
SASAKI = f"ls_{TODGA_ID}_Nd_pub_5a68dc5665_3_0.1"
MECH_DOI = "10.1016/j.jiec.2014.03.002"


def _src(kind: str = "none", doi: str | None = None, locator: str | None = None,
         publication_id: str | None = None, safe_exp_ids: list | None = None) -> dict:
    return {"kind": kind, "doi": doi, "locator": locator, "publication_id": publication_id,
            "safe_exp_ids": safe_exp_ids or []}


def _unknown(unit: str, kind: str = "none", locator: str | None = None, note: str = "") -> dict:
    return {"value": None, "unit": unit, "status": "unknown", "assumed_label": None,
            "range": None, "source": _src(kind, None, locator), "model_id": None,
            "fit_manifest_sha256": None, "note": note}


def _lit(value: float, doi: str, locator: str, note: str) -> dict:
    return {"value": value, "unit": "1", "status": "literature", "assumed_label": None,
            "range": None, "source": _src("doi", doi, locator), "model_id": None,
            "fit_manifest_sha256": None, "note": note}


def _corpus_record(rid: str, mm: float, d: float, tracer: bool) -> dict:
    return {"record_id": rid, "metal": "Nd", "d": d, "log_d": math.log10(d),
            "acid_nominal_M": 3.0, "acid_eq_M": None, "anion_M": 3.0, "ligand_M": {"TODGA": 0.1},
            "complexant_M": None, "metals_initial_mM": {"Nd": mm}, "oa_ratio": None,
            "temperature_C": 25.0, "contact_time_min": None, "diluent_name": "n_dodecane",
            "publication_id": "pub_5a68dc5665", "experiment_series_id": "caf22524baa8080e",
            "replicate_id": "c6fca1a8d31ed78d", "loading_series_id": SASAKI,
            "is_tracer": tracer, "fit_eligible": True, "fit_ineligible_reason": None,
            "duplicate_flag": None,
            "provenance": {
                "status": "measured_corpus",
                "source": _src("corpus", None, "dataset.parquet row", "pub_5a68dc5665", [rid]),
                "range": None, "assumed_label": None, "model_id": None,
                "fit_manifest_sha256": None,
                "note": "metal concentration semantics ASSUMED initial aqueous; O/A not reported"}}


def todga_example_json() -> dict:
    """DESIGN.md section 3.8 (three records shown).  One deviation, recorded in addenda/WB1.md
    A4: ``medium.salting_anion_M`` is ``null / unknown`` because a ``measured_corpus`` value
    without ``safe_exp_ids`` and ``publication_id`` violates V3 as written in section 1.2."""
    return {
        "schema_version": "gen18.1", "system_id": TODGA_ID,
        "name": "TODGA in aliphatic hydrocarbon, nitrate medium, no additive",
        "family": "diglycolamide", "origin": "corpus",
        "organic_ligands": [{
            "name": "TODGA", "smiles": TODGA_SMILES, "canonical_smiles": TODGA_SMILES,
            "role": "extractant", "mechanism": "solvating", "aggregation": "monomer",
            "scaffold_id": "DGA_core", "variant_tag": "N,N,N',N'-tetraoctyl",
            "concentration": _unknown("mol/L", "corpus",
                                      "per record: cond__extractant_concentration_M",
                                      "varies per record; see records[].ligand_M"),
            "stoichiometry": {
                "ligands_per_metal": _unknown(
                    "1", "none", "set from the fitted n_solvation by g18_fit_dmodels.py"),
                "protons_released_per_metal": _lit(
                    0.0, MECH_DOI, "solvating mechanism Ln3+ + 3 NO3- + n L",
                    "no proton release for neutral solvating extractants"),
                "anions_per_metal": _lit(3.0, MECH_DOI, "as above", "Ln(NO3)3.L_n")}}],
        "aqueous_complexants": [],
        "diluent": {"name": None, "family": "aliphatic_hydrocarbon", "components": []},
        "medium": {
            "acid": "HNO3", "acid_class": "nitrate", "anion": "nitrate", "salting_agent": None,
            "salting_anion_M": _unknown("mol/L", "corpus",
                                        "no salting additive column set in any record"),
            "ionic_strength_M": _unknown("mol/L"),
            "temperature_C": _unknown("Cel", "corpus", "per record: cond__temperature_C",
                                      "5-45 C across records; parameter sets are per band")},
        "params": {},
        "phase": {
            "loc_metal_M": _unknown("mol/L", note="third-phase limits stay null until a DOI"),
            "loc_acid_M": _unknown("mol/L"), "third_phase_observed": _unknown("1"),
            "disengagement_s": _unknown("s"),
            "ligand_loss_mol_per_L_aq": _unknown("mol/L_aq"),
            "max_loading_fraction_studied": _unknown(
                "1", "corpus", "computed by the fitter from loading-series records"),
            "regenerability_note": ""},
        "stream_records": [], "oxidation_state_routes": [], "direction_prior": None,
        "applicability": {},
        "records": [_corpus_record("Ca_SAFE:2222", 4.9, 23.5, True),
                    _corpus_record("Ca_SAFE:2221", 6.0, 14.0, False),
                    _corpus_record("Ca_SAFE:2217", 12.0, 0.6, False)],
        "notes": "514 records, 28 publications, 14 metals (three records shown).",
    }


def errors(obj: dict) -> list[str]:
    entry = S.entry_from_json(obj)
    return [v.message for v in S.validate_entry(entry) if v.level == "error"]


def rules(obj: dict) -> set[str]:
    return {m.split(":")[0] for m in errors(obj)}


def warning_rules(obj: dict) -> set[str]:
    entry = S.entry_from_json(obj)
    return {v.message.split(":")[0] for v in S.validate_entry(entry) if v.level == "warning"}


@pytest.fixture
def pc() -> dict:
    return copy.deepcopy(L.PC88A_ENTRY_JSON)


def test_example_entries_load_without_errors():
    for e in (L.pc88a_prnd_entry(), L.cyanex272_prnd_entry()):
        assert not [v for v in S.validate_entry(e) if v.level == "error"]
    assert not errors(todga_example_json())
    assert L.pc88a_prnd_entry().system_id == PC88A_ID
    assert L.cyanex272_prnd_entry().system_id == CYANEX_ID


def test_pc88a_matches_design_3_7_numbers():
    e = L.pc88a_prnd_entry()
    blk = e.params["PC88A"]["20-30C"]
    assert isinstance(blk, S.CationExchangeParams)
    assert blk.log_k["Nd"].value == -1.95 and blk.log_k["Nd"].range == (-2.9, -1.0)
    assert blk.log_k["Pr"].value == -2.10 and blk.log_k["Pr"].range == (-3.08, -1.11)
    assert blk.a_dimer.range == (2.0, 3.0) and blk.b_proton.range == (2.0, 3.0)
    assert e.organic_ligands[0].concentration.value == 0.8
    assert e.applicability.meta["PC88A|20-30C"]["_status"] == "assumed"
    for path, s in S.iter_sourced(e):
        if s.value is not None:
            assert s.status is S.ProvStatus.ASSUMED, path


def test_cyanex_entry_per_design_paragraph():
    e = L.cyanex272_prnd_entry()
    blk = e.params["Cyanex 272"]["20-30C"]
    assert blk.log_k["Nd"].range == (-4.5, -2.4)
    assert blk.b_proton.range == (2.0, 3.0)
    assert blk.log_k["Pr"].source.kind == "none"
    assert "10.1016/j.hydromet.2014.09.015" in blk.log_k["Pr"].provenance.note
    assert e.organic_ligands[0].smiles == "CC(C)(C)CC(C)CP(=O)(O)CC(C)CC(C)(C)C"


def test_system_id_recomputes_verified_ids():
    assert S.system_id(L.pc88a_prnd_entry()) == PC88A_ID
    assert S.system_id(L.cyanex272_prnd_entry()) == CYANEX_ID
    assert S.system_id(S.entry_from_json(todga_example_json())) == TODGA_ID
    key = ((TODGA_SMILES,), "nitrate", "aliphatic_hydrocarbon", (), ())
    assert S.system_id(key) == TODGA_ID


@pytest.mark.parametrize("entry_fn", [L.pc88a_prnd_entry, L.cyanex272_prnd_entry,
                                      lambda: S.entry_from_json(todga_example_json())])
def test_write_then_load_round_trips_byte_identically(tmp_path, entry_fn):
    entry = entry_fn()
    a = S.write_system(entry, tmp_path / "a.json")
    loaded = S.load_system(a)
    b = S.write_system(loaded, tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes()
    assert loaded.system_id == entry.system_id
    assert b"\r\n" not in a.read_bytes()


def test_built_todga_entry_round_trips_if_present(tmp_path):
    path = paths.SYSTEMS_DIR / f"{TODGA_ID}.json"
    if not path.exists():
        pytest.skip("database not built (run scripts/g18_build_db.py)")
    entry = S.load_system(path)
    assert len(entry.records) == 514
    out = S.write_system(entry, tmp_path / "t.json")
    assert out.read_bytes() == path.read_bytes()


# ---- V1-V17 minimal failing fixtures ---------------------------------------------------------

def test_v1_schema_version(pc):
    pc["schema_version"] = "gen17.9"
    assert "V1" in rules(pc)


def test_v2_value_with_unknown_status(pc):
    pc["medium"]["ionic_strength_M"]["value"] = 0.5
    assert "V2" in rules(pc)


def test_v3_literature_without_doi(pc):
    s = pc["organic_ligands"][0]["stoichiometry"]["protons_released_per_metal"]
    s["status"], s["assumed_label"], s["range"] = "literature", None, None
    s["source"] = _src("doi")
    assert "V3" in rules(pc)


def test_v3_measured_corpus_without_safe_exp_ids():
    obj = todga_example_json()
    obj["records"][0]["provenance"]["source"]["safe_exp_ids"] = []
    assert "V3" in rules(obj)


def test_v3_fitted_without_manifest(pc):
    s = pc["params"]["PC88A"]["20-30C"]["log_k"]["Nd"]
    s["status"], s["assumed_label"], s["model_id"] = "fitted_from_corpus", None, "M1_x"
    s["source"]["kind"] = "model"
    assert "V3" in rules(pc)


def test_v4_assumed_with_wrong_label(pc):
    pc["params"]["PC88A"]["20-30C"]["a_dimer"]["assumed_label"] = "GUESS"
    assert "V4" in rules(pc)


def test_v4_assumed_kind_none_without_reason(pc):
    s = pc["organic_ligands"][0]["stoichiometry"]["anions_per_metal"]
    s["source"]["locator"], s["note"] = None, ""
    assert "V4" in rules(pc)


def test_v5_phase_value_assumed(pc):
    pc["phase"]["loc_metal_M"] = {**pc["params"]["PC88A"]["20-30C"]["a_dimer"], "value": 0.05,
                                  "unit": "mol/L", "range": [0.01, 0.1]}
    assert "V5" in rules(pc)


def test_v5_hand_entered_domain_without_range_note(pc):
    pc["applicability"]["PC88A|20-30C"]["_range_note"] = ""
    assert "V5" in rules(pc)


def test_v6_gen15_sourced_value(pc):
    loc = "gen15 deploy_g15.joblib"
    pc["params"]["PC88A"]["20-30C"]["log_k"]["Nd"]["source"]["locator"] = loc
    assert "V6" in rules(pc)


def test_v6_gen15_sourced_record():
    obj = todga_example_json()
    obj["records"][0]["provenance"]["model_id"] = "deploy_g15"
    assert "V6" in rules(obj)


def test_v7_cross_anion_parameters(pc):
    pc["params"]["PC88A"]["20-30C"]["medium_anion"] = "nitrate"
    assert "V7" in rules(pc)


def test_v8_missing_log_k_for_record_metal(pc):
    rec = _corpus_record("Ca_SAFE:1", 1.0, 2.0, True)
    rec["metal"] = "Sm"
    pc["records"] = [rec]
    assert "V8" in rules(pc)


def test_v9_assumed_without_range(pc):
    pc["params"]["PC88A"]["20-30C"]["b_proton"]["range"] = None
    assert "V9" in rules(pc)


def test_v10_smiles(pc):
    pc["organic_ligands"][0]["canonical_smiles"] = "CCCC"
    assert "V10" in rules(pc)
    pc["organic_ligands"][0]["smiles"] = "C(C"
    assert "V10" in rules(pc)


def test_v11_system_id_mismatch(pc):
    pc["system_id"] = "sys_0000000000000000"
    assert "V11" in rules(pc)


def test_v12_duplicate_record_id(pc):
    pc["records"] = [_corpus_record("Ca_SAFE:1", 1.0, 2.0, True),
                     _corpus_record("Ca_SAFE:1", 2.0, 1.0, False)]
    assert "V12" in rules(pc)


def test_v13_nonpositive_d(pc):
    rec = _corpus_record("Ca_SAFE:1", 1.0, 2.0, True)
    rec["d"], rec["log_d"] = 0.0, -99.0
    pc["records"] = [rec]
    assert "V13" in rules(pc)


def test_v14_cation_exchange_stoichiometry(pc):
    pc["organic_ligands"][0]["stoichiometry"]["protons_released_per_metal"]["value"] = 2.0
    assert "V14" in rules(pc)
    pc2 = copy.deepcopy(L.PC88A_ENTRY_JSON)
    pc2["organic_ligands"][0]["aggregation"] = "monomer"
    assert "V14" in rules(pc2)


def test_v15_complexant_without_log_beta_for_record_metal(pc):
    pc["records"] = [_corpus_record("Ca_SAFE:1", 1.0, 2.0, True)]       # metal Nd
    cj = S._complexant_to_json(L.todga_hydrophilic_complexant_placeholder())
    del cj["log_beta"]["Nd"]
    pc["aqueous_complexants"] = [cj]
    pc["system_id"] = S.system_id(S.entry_from_json(pc))
    assert "V15" in rules(pc)


def test_v16_vocabulary(pc, tmp_path):
    pc["params"]["PC88A"]["20-30C"]["a_dimer"]["source"]["kind"] = "web"
    assert "V16" in rules(pc)
    pc2 = copy.deepcopy(L.PC88A_ENTRY_JSON)
    pc2["medium"]["salting_anion_M"]["status"] = "guessed"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(pc2), encoding="utf-8")
    with pytest.raises(S.SystemValidationError) as ei:
        S.load_system(p)
    assert any("V16" in v.message for v in ei.value.violations)


def test_v17_diluent_family_vocabulary(pc):
    pc["diluent"]["family"] = "kerosene"
    assert "V17" in rules(pc)


def test_load_system_raises_with_violations(tmp_path, pc):
    pc["schema_version"] = "x"
    p = tmp_path / "e.json"
    p.write_text(json.dumps(pc), encoding="utf-8")
    with pytest.raises(S.SystemValidationError) as ei:
        S.load_system(p)
    assert ei.value.violations and ei.value.violations[0].level == "error"


# ---- warnings and helpers --------------------------------------------------------------------

def test_warnings_w1_w2_w3_w4(pc):
    del pc["params"]["PC88A"]["20-30C"]["log_k"]["Pr"]
    pc["medium"]["temperature_C"] = _unknown("Cel")
    assert {"W1", "W2"} <= warning_rules(pc)
    obj = todga_example_json()
    obj["records"][0]["publication_id"] = None
    obj["records"][0]["provenance"]["source"]["publication_id"] = "pub_x"
    obj["params"] = {"TODGA": {"20-30C": {
        "model_type": "solvating", "medium_anion": "nitrate", "temperature_band": "20-30C",
        "log_k": {"Nd": _unknown("1")}, "n_solvation": _unknown("1"), "p_anion": _unknown("1"),
        "p_h": _unknown("1"), "k_acid_uptake": _unknown("L2/mol2"), "delta_h_kj_mol": None}}}
    assert {"W3", "W4"} <= warning_rules(obj)


def test_temperature_band():
    assert S.temperature_band(None) == "20-30C"
    assert S.temperature_band(float("nan")) == "20-30C"
    assert S.temperature_band(19.99) == "<20C" and S.temperature_band(20.0) == "20-30C"
    assert S.temperature_band(30.03) == "30-40C" and S.temperature_band(55.0) == ">=50C"


def test_derive_logk_reproduces_design_windows():
    mm = {"Nd": 1500 / 144.242, "Tb": 1500 / 158.925, "Dy": 1500 / 162.5}
    win = L.derive_logk_from_extraction({"Nd": 0.64, "Tb": 0.91, "Dy": 0.98}, mm, 0.8, 1.0,
                                        (1.02, 1.42), 3.0, 3.0)
    assert win["Nd"] == pytest.approx((-2.554, -1.354), abs=2e-3)
    win = L.derive_logk_from_extraction({"Nd": 0.27, "Tb": 0.61, "Dy": 0.76}, mm, 0.8, 1.0,
                                        (1.22, 1.70), 3.0, 3.0)
    assert win["Nd"] == pytest.approx((-4.176, -2.736), abs=2e-3)
    assert L.ph50_to_logk(2.0, 0.8, 3.0, 3.0) == pytest.approx(-6.0 - 3.0 * math.log10(0.4))


def test_complexant_placeholder_shape():
    c = L.todga_hydrophilic_complexant_placeholder()
    assert c.log_beta["Nd"][0].value is None and c.log_beta["Nd"][0].range == (1.0, 4.0)
    assert c.log_beta["Pr"][0].range == (1.0, 5.5) and c.protonation_logk[0].range == (1.0, 3.0)
    assert c.regeneration_fraction.value == 0.9
    assert c.regeneration_fraction.status is S.ProvStatus.ASSUMED
