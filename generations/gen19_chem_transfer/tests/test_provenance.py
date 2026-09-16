"""Tests for ``gen19ct.data.provenance``: free-text handling, derived quantities, the field registry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen19ct.data import provenance as P

STRUCTURED = ("Data Location: Figure 3; Additional Comments: {ac}; Complexant_Name: nan; Complexant_SMILES: "
              "O=S(=O)([O-])c1ccc(-c2nnc(-c3ccccc3)c(-c3ccc(S(=O)(=O)[O-])cc3)n2)cc1; Complexant_Concentration_M: nan; "
              "Publication_Year: 2020; Title: Insights into the third phase formation of TODGA at pH 2; Authors: A. B; "
              "No. of Extractants: 1; Aqueous Phase Metals: nan; Metal_Concentration_mM: 5; No. of Metals: 1")


def _row(i: int, **kw) -> dict:
    base = {
        "canonical_measurement_id": f"SAE:{i}", "g19_publication_id": "pub_a", "g19_publication_status": "DOI",
        "g19_study_id": "pub_a", "source_record_id": str(i), "doi_primary_corrected": "10.1/a", "doi_primary": "10.1/a",
        "doi_all": np.array(["10.1/a"], dtype=object), "doi_correction_rule": None, "data_location": None,
        "entry_author": "A", "comments_raw": None, "ini_comp_raw": "HNO3, TODGA, Nd, dodecane", "sub_source_file": None,
        "metal_symbol": "Nd", "metal_raw": "Nd", "metal_oxidation_state": 3.0, "metal_oxidation_state_source": "archive",
        "g19_metal_state": "Nd(III)", "metal_category": "lanthanide", "flags": np.array([], dtype=object),
        "extractant_names": np.array(["TODGA"], dtype=object), "extractant_primary_name": "TODGA",
        "extractant_name_raw": "TODGA", "extractant_system_key": "CCO", "extractant_primary_smiles": "CCO",
        "extractant_smiles_canonical": np.array(["CCO"], dtype=object),
        "components": np.array([{"role": "organic_extractant", "smiles_canonical": "CCO", "structure_source": "archive"}],
                               dtype=object),
        "system_component_class": "SINGLE_EXTRACTANT", "solvent_key": "dodecane:1",
        "solvent_components": np.array(["dodecane"], dtype=object), "solvent_fractions": np.array([1.0]),
        "solvent_n_components": 1, "acid_primary": "HNO3", "acid_signature": "HNO3", "acid_anion": "NO3",
        "nitrate_concentration_M": np.nan, "complexant_signature": None, "complexant_name": None,
        "complexant_concentration_M": np.nan, "holdback_smiles_canonical": None, "holdback_concentration_M": np.nan,
        "aqueous_phase_metals_declared": None, "extractant_primary_concentration_M": 0.1,
        "extractant_concentrations_M": np.array([0.1]), "metal_concentration_M": 0.001, "metal_concentration_raw": "1 mM",
        "acid_concentration_M": 3.0, "acid_concentration_organic_M": np.nan, "phase_ratio_org_aq": 1.0,
        "phase_ratio_raw": "1", "temperature_C": 25.0, "temperature_raw": "25", "contact_time_min": 30.0,
        "shaking_time_min": np.nan, "modifier_name": None, "modifier_concentration_M": np.nan, "D_value": 10.0,
        "log_D": 1.0, "D_raw": "10", "model_readiness": "A_model_ready", "duplicate_class": "UNIQUE",
        "in_value_conflict": False, "has_suspect_flag": False, "g19_tier": "MODEL",
    }
    base.update(kw)
    return base


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


@pytest.fixture()
def synthetic() -> pd.DataFrame:
    return _frame(
        _row(0),
        _row(1, phase_ratio_org_aq=np.nan, comments_raw="Just says equal volumes of aq/org phases were shaken"),
        _row(2, metal_symbol="U", metal_raw="UO2+2", metal_oxidation_state=np.nan, g19_metal_state=None,
             metal_category="actinide", metal_concentration_M=np.nan, comments_raw="Tracer levels of U233 used"),
        _row(3, metal_symbol="Sr", metal_category="alkaline_earth", g19_tier="TARGET_ONLY",
             data_location="Figure 2", comments_raw=STRUCTURED.format(ac="nan"), D_raw="0.591052825200136"),
        _row(4, metal_symbol=None, metal_category=None, g19_tier="NO_TARGET", log_D=np.nan, D_value=np.nan, D_raw=None,
             comments_raw="fig2 using log scale y nan extraction:graphreader1"),
        _row(5, complexant_signature="DTPA@0.01", data_location="Table 1", comments_raw=STRUCTURED.format(ac="pH 2.1 final"),
             solvent_n_components=2, solvent_key="kerosene:0.7|1-octanol:0.3", D_raw="1.23"),
    )


# --------------------------------------------------------------------------------------------- #
# free text
# --------------------------------------------------------------------------------------------- #

def test_comment_free_text_formats():
    df = _frame(_row(0, comments_raw=STRUCTURED.format(ac="nan")),
                _row(1, comments_raw=STRUCTURED.format(ac="Simulated raffinate; see text")),
                _row(2, comments_raw="file: ./ST102.json; nitrate concentration(M): 0.5; Holdback_Agent_SMILES: nan;"),
                _row(3, comments_raw="fig3 nan nan extraction:nan"), _row(4, comments_raw=None))
    ft = P.comment_free_text(df)
    assert ft.tolist() == [None, "Simulated raffinate; see text", None, "fig3 nan nan extraction:nan", None]


def test_title_mentions_do_not_count_as_comments():
    df = _frame(_row(0, comments_raw=STRUCTURED.format(ac="nan")))
    third = P.PATTERNS_BY_NAME["third_phase"]
    assert not P.text_hits(df, third).any()
    assert P.text_hits(df, third, ("comments_raw",)).all()


def test_ph_pattern_is_case_sensitive_and_ignores_phenyl():
    df = _frame(_row(0, comments_raw="SO3-Ph-BTP masking agent"), _row(1, comments_raw="equilibrium pH 3.2"),
                _row(2, comments_raw="phosphate buffer"))
    assert P.text_hits(df, P.PATTERNS_BY_NAME["pH"]).tolist() == [False, True, False]


@pytest.mark.parametrize("text,pattern", [("value ± 0.02", "uncertainty"), ("saponified to 40 %", "saponification"),
                                          ("O/A = 2", "phase_ratio"), ("shaken until equilibrium", "equilibrium"),
                                          ("Exp0.1 triplice nan", "replicate_tag"), ("0.1 M KBrO3 to hold", "redox_or_holdback_agent")])
def test_patterns_fire(text, pattern):
    assert P.text_hits(_frame(_row(0, comments_raw=text)), P.PATTERNS_BY_NAME[pattern]).all()


def test_scan_text_metadata_counts_and_examples(synthetic):
    sc = P.scan_text_metadata(synthetic, synthetic["g19_tier"].eq("MODEL"))
    r = sc[(sc["evidence_pattern"] == "phase_ratio") & (sc["text_column"] == P.FREE_TEXT_COLUMN)].iloc[0]
    assert (r["n_filled_all"], r["n_filled_model"]) == (1, 1)
    assert "equal volumes" in r["example_1"]
    ph = sc[(sc["evidence_pattern"] == "pH") & (sc["text_column"] == P.FREE_TEXT_COLUMN)].iloc[0]
    ph_raw = sc[(sc["evidence_pattern"] == "pH") & (sc["text_column"] == "comments_raw")].iloc[0]
    assert ph["n_filled_all"] == 1 and ph_raw["n_filled_all"] == 2  # the title's 'pH 2' only in the raw scan


# --------------------------------------------------------------------------------------------- #
# derived quantities
# --------------------------------------------------------------------------------------------- #

def test_nominal_max_loading_ratio():
    df = pd.DataFrame({"metal_concentration_M": [0.01, 0.01, np.nan, 0.01, 0.01],
                       "extractant_primary_concentration_M": [0.1, 0.1, 0.1, 0.0, 0.2],
                       "phase_ratio_org_aq": [1.0, np.nan, 1.0, 1.0, 0.5]})
    out = P.nominal_max_loading_ratio(df)
    assert out.iloc[0] == pytest.approx(0.1)
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])
    assert out.iloc[4] == pytest.approx(0.1)


def test_d_raw_float_artefact():
    df = pd.DataFrame({"D_raw": ["0.591052825200136", "4.32", None, "1.23e-05", "0.001000000000000023", "abc"]})
    assert P.d_raw_float_artefact(df).tolist() == [True, False, False, False, True, False]
    assert P.d_raw_significant_digits(df).tolist()[:4] == [15, 3, 0, 3]


def test_metal_group_and_location_kind():
    df = pd.DataFrame({"metal_category": ["lanthanide", "actinide", "transition_metal", None, "none", np.nan],
                       "data_location": ["Figure 3", "Table SI1", "Figure 10 and Table 1", "MAIN TEXT", None, "Graph 5"]})
    assert P.metal_group(df).tolist() == ["lanthanide", "actinide", "other", None, None, None]
    assert P.location_kind(df).tolist() == ["figure", "table", "mixed", "main_text", None, "figure"]


# --------------------------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------------------------- #

def test_registry_covers_every_brief_field_once():
    prim = [s.brief_field for s in P.FIELD_SPECS if s.source == "primary"]
    assert sorted(prim) == sorted(P.BRIEF_FIELDS) and len(prim) == len(set(prim)) == 26
    assert all(s.status in P.STATUSES for s in P.FIELD_SPECS)
    assert all(s.source in ("primary", "supplement") for s in P.FIELD_SPECS)
    absent = {s.brief_field for s in P.FIELD_SPECS if s.source == "primary" and s.status == "ABSENT"}
    assert {"equilibrium/final pH", "saponification", "reported uncertainty", "direct vs reconstructed D"} <= absent


def test_metadata_availability_fill_fractions(synthetic):
    av = P.metadata_availability(synthetic)
    assert (av["section"] == "field").all()
    prim = av[av["source"] == "primary"].set_index("brief_field")
    assert len(prim) == 26
    oa = prim.loc["O/A"]
    # MODEL rows are 0, 1, 2, 5; row 1 lacks O/A
    assert (oa["n_filled_all"], oa["n_all"], oa["n_model"], oa["n_filled_model"]) == (5, 6, 4, 3)
    assert oa["fill_model_lanthanide"] == pytest.approx(2 / 3) and oa["fill_model_actinide"] == pytest.approx(1.0)
    assert np.isnan(oa["fill_model_other"])  # no MODEL row in the "other" group
    assert prim.loc["saponification", "n_filled_all"] == 0
    assert prim.loc["metal", "n_filled_all"] == 5
    assert prim.loc["loading", "n_filled_all"] == 4  # rows 1 (no O/A) and 2 (no metal M) are not imputed
    assert prim.loc["complexants", "n_filled_all"] == 1
    assert "{" not in "".join(av["note"])  # every note template was filled
    sup = av[(av["source"] == "supplement") & (av["brief_field"] == "direct vs reconstructed D")]
    assert sup["n_filled_all"].max() >= 2


def test_availability_table_sections(synthetic):
    t = P.availability_table(synthetic)
    assert set(t["section"]) == {"field", "text_scan", "d_raw_precision_by_location"}
    d = t[t["section"] == "d_raw_precision_by_location"].set_index("location_kind")
    assert d.loc["figure", "n_filled_all"] == 1 and d.loc["table", "n_filled_all"] == 0


def test_metadata_availability_rejects_bad_specs(synthetic):
    bad = (P.FieldSpec("pH", "primary", "PRESENT", (), P._never, "x"),)
    with pytest.raises(ValueError):
        P.metadata_availability(synthetic, specs=bad)
    dup = (P.FieldSpec("O/A", "primary", "PRESENT", (), P._never, "x"),
           P.FieldSpec("O/A", "primary", "ABSENT", (), P._never, "y"))
    with pytest.raises(ValueError):
        P.metadata_availability(synthetic, specs=dup)


def test_archive_availability_matches_known_fills():
    from gen19ct.data.load import load_archive

    df = load_archive(copy=False)
    av = P.metadata_availability(df)
    prim = av[av["source"] == "primary"].set_index("brief_field")
    assert prim.loc["O/A", "n_filled_all"] == int(df["phase_ratio_org_aq"].notna().sum())
    assert prim.loc["initial metal concentration", "n_filled_all"] == int(df["metal_concentration_M"].notna().sum())
    assert prim.loc["table/figure/page", "n_filled_all"] == int(df["data_location"].notna().sum())
    assert prim.loc["measured D", "n_filled_model"] == prim.loc["measured D", "n_model"]
    for f in ("equilibrium/final pH", "saponification", "reported uncertainty", "extractant family"):
        assert prim.loc[f, "status"] == "ABSENT" and prim.loc[f, "n_filled_all"] == 0
    assert "{" not in "".join(av["note"])
