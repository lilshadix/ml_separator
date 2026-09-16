"""Tests for ``gen19ct.data.normalize`` (brief sections 4.3 and 27).

Fast tests run on small synthetic frames shaped like ``master_clean.parquet`` rows; the one
full-archive test is marked ``slow``.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from gen19ct.data import normalize as N

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
HEDTA = "O=C([O-])CN(CCO)CCN(CC(=O)[O-])CC(=O)[O-]"


def _comp(role, name, smiles, conc):
    return {"role": role, "name": name, "smiles_raw": smiles, "smiles_canonical": smiles,
            "structure_source": "archive", "concentration_M": conc, "concentration_raw": str(conc)}


def _row(**over):
    base = {
        "canonical_measurement_id": "SAE:1",
        "components": np.array([_comp("organic_extractant", "TODGA", TODGA, 0.1)], dtype=object),
        "acid_primary": "HNO3", "acid_signature": "HNO3", "acid_anion": "nitrate",
        "acid_concentration_M": 1.0, "acid_concentration_organic_M": np.nan, "nitrate_concentration_M": np.nan,
        "n_organic_extractants": 1, "extractant_primary_concentration_M": 0.1, "metal_concentration_M": np.nan,
        "phase_ratio_org_aq": 1.0, "solvent_key": "dodecane:1", "solvent_primary": "dodecane",
        "solvent_components": np.array(["dodecane"], dtype=object), "modifier_name": None,
        "modifier_concentration_M": np.nan, "temperature_C": 25.0, "contact_time_min": 30.0,
        "shaking_time_min": np.nan, "g19_publication_id": "pub_a", "extractant_system_key": TODGA,
        "metal_symbol": "Nd", "log_D": 0.5,
    }
    base.update(over)
    return base


def _frame(*rows):
    return pd.DataFrame([_row(**r) for r in rows])


# ----------------------------------------------------------------------------- significant figures
def test_format_sig_rounding_and_na():
    assert N.format_sig(0.123456789, 6) == "0.123457"
    assert N.format_sig(0.30000000000000004, 6) == "0.3"
    assert N.format_sig(1234567.0, 6) == "1.23457e+06"
    assert N.format_sig(2.5e-7, 6) == "2.5e-07"
    assert N.format_sig(-0.0) == "0" and N.format_sig(0.0) == "0"
    for missing in (None, np.nan, float("nan"), pd.NA, ""):
        assert N.format_sig(missing) == "NA"
    assert N.format_sig((0.2, 0.5000001), 6) == "(0.2,0.5)"
    assert N.format_sig(12, 6) == "12"


def test_round_sig():
    assert N.round_sig(0.123456789, 3) == pytest.approx(0.123)
    assert N.round_sig(98765.4321, 2) == 99000.0
    assert math.isnan(N.round_sig(None))


def test_sig_levels_merge_only_below_the_resolution():
    df = _frame({"acid_concentration_M": 1.0000001}, {"acid_concentration_M": 1.0000004})
    assert N.condition_key(df, sig=6).nunique() == 1   # differ in the 8th figure
    assert N.condition_key(df, sig=12).nunique() == 2


# ----------------------------------------------------------------------------- condition key
def test_condition_key_is_deterministic_and_order_invariant():
    df = _frame({}, {"temperature_C": 50.0}, {"canonical_measurement_id": "SAE:3", "log_D": -2.0})
    k1 = N.condition_key(df)
    k2 = N.condition_key(df.iloc[::-1].copy()).loc[df.index]
    assert k1.tolist() == k2.tolist()
    assert k1.tolist() == N.condition_key(df.copy()).tolist()
    # the id and the target are not part of the key
    assert k1.iloc[0] == k1.iloc[2]
    assert k1.iloc[0] != k1.iloc[1]
    h = N.condition_key(df, hashed=True)
    assert h.str.match(r"^ck_[0-9a-f]{16}$").all() and h.iloc[0] == h.iloc[2]


def test_condition_key_na_is_a_value():
    df = _frame({"temperature_C": np.nan}, {"temperature_C": np.nan}, {"temperature_C": 25.0})
    k = N.condition_key(df)
    assert k.iloc[0] == k.iloc[1]          # two missing temperatures match
    assert k.iloc[0] != k.iloc[2]          # missing never matches a recorded value
    items = dict(json.loads(k.iloc[0]))
    assert items["temperature_C"] == "NA"
    assert set(items) == set(N.CONDITION_KEY_FIELDS)


def test_condition_key_ignores_metal_and_accepts_condition_vector():
    df = _frame({"metal_symbol": "Pr"}, {"metal_symbol": "Nd"})
    k = N.condition_key(df)
    assert k.iloc[0] == k.iloc[1]
    cv = N.condition_vector(df)
    assert N.is_condition_vector(cv) and not N.is_condition_vector(df)
    assert N.condition_key(cv).tolist() == k.tolist()


def test_condition_key_exclude():
    df = _frame({"metal_concentration_M": 1e-3}, {"metal_concentration_M": 2e-3})
    assert N.condition_key(df).nunique() == 2
    assert N.condition_key(df, exclude=["metal_concentration_M"]).nunique() == 1
    with pytest.raises(ValueError):
        N.condition_key(df, exclude=["not_a_field"])


def test_multicomponent_order_invariance():
    ab = np.array([_comp("organic_extractant", "TODGA", TODGA, 0.2),
                   _comp("organic_extractant", "TBP", TBP, 0.5)], dtype=object)
    ba = np.array([_comp("organic_extractant", "TBP", TBP, 0.5),
                   _comp("organic_extractant", "TODGA", TODGA, 0.2)], dtype=object)
    df = _frame({"components": ab, "n_organic_extractants": 2}, {"components": ba, "n_organic_extractants": 2})
    cv = N.condition_vector(df)
    assert cv["extractant_concentrations_sorted_M"].iloc[0] == cv["extractant_concentrations_sorted_M"].iloc[1]
    # sorted by canonical SMILES: TBP's SMILES sorts after TODGA's
    assert cv["extractant_concentrations_sorted_M"].iloc[0] == tuple(
        c for _, c in sorted([(TODGA, 0.2), (TBP, 0.5)]))
    assert cv["extractant_total_concentration_M"].iloc[0] == pytest.approx(0.7)
    assert N.condition_key(df).nunique() == 1


def test_complexant_structure_key_and_name_fallback():
    comps = np.array([_comp("organic_extractant", "TODGA", TODGA, 0.1),
                      _comp("aqueous_complexant", "NaNO3", None, 1.0),
                      _comp("aqueous_complexant", "HEDTA", HEDTA, 0.05)], dtype=object)
    cv = N.condition_vector(_frame({"components": comps}))
    assert cv["complexant_structure_key"].iloc[0] == f"{HEDTA}|name:NaNO3"
    assert cv["complexant_concentrations_sorted_M"].iloc[0] == (0.05, 1.0)
    assert pd.isna(cv["holdback_structure_key"].iloc[0])


def test_experiment_key_is_publication_aware():
    df = _frame({"g19_publication_id": "pub_a"}, {"g19_publication_id": "pub_b"}, {"g19_publication_id": "pub_a"})
    ek = N.experiment_key(df)
    assert ek.iloc[0] == ek.iloc[2] and ek.iloc[0] != ek.iloc[1]
    assert N.condition_key(df).nunique() == 1
    with pytest.raises(KeyError):
        N.experiment_key(df.drop(columns=["g19_publication_id"]))


# ----------------------------------------------------------------------------- acid is not pH
def test_acid_molarity_is_never_turned_into_ph():
    df = _frame({"acid_concentration_M": 0.01}, {"acid_concentration_M": 1e-3, "acid_primary": "HCl"},
                {"acid_concentration_M": 3.0})
    cv = N.condition_vector(df)
    assert cv["pH"].isna().all()
    assert cv["pH_status"].str.startswith("NOT_RECORDED").all()
    assert "never converted to pH" in cv["pH_status"].iloc[0]
    # the log of the molarity is kept under its own name and is not substituted anywhere as pH
    assert cv["log10_acid_M"].tolist() == pytest.approx([-2.0, -3.0, math.log10(3.0)])
    ph_like = [c for c in cv.columns if c.lower() == "ph" or c.lower().startswith("ph_")]
    assert ph_like == ["pH", "pH_status"]
    for c in cv.columns:
        if c in ("pH", "log10_acid_M") or cv[c].dtype != float:
            continue
        assert not np.allclose(cv[c].to_numpy(), [2.0, 3.0, -math.log10(3.0)], equal_nan=False)
    assert N.condition_key(df).map(lambda k: "pH" not in dict(json.loads(k))).all()


def test_significant_figures():
    assert N.significant_figures(0.000977237) == 6
    assert N.significant_figures(2.63027e-7) == 6
    assert N.significant_figures(0.1) == 1 and N.significant_figures(100.0) == 1
    assert N.significant_figures(0.30000000000000004) == 1
    assert N.significant_figures(1.9733) == 5
    for missing in (None, np.nan, 0.0, float("inf")):
        assert N.significant_figures(missing) is None


@pytest.mark.parametrize("value,flag", [
    (0.000977237, True),     # 10^-3.01, six significant figures: pH-like / log-axis-like
    (2.63027e-7, True),      # 10^-6.58
    (1.99526, True),         # 10^0.30
    (0.001, False),          # round molarity, one significant figure
    (0.1, False), (3.0, False), (0.0001, False),
    (1.9733, False),         # arbitrary molarity off the grid
    (0.0, False), (-0.5, False), (np.nan, False), (None, False),
])
def test_acid_log10_grid_flag(value, flag):
    assert N.acid_log10_grid_flag(value) is flag


def test_acid_semantics_flags_never_change_values():
    df = _frame({"acid_concentration_M": 0.000977237}, {"acid_concentration_M": 1e-4},
                {"acid_concentration_M": 2.0}, {"acid_concentration_M": np.nan})
    cv = N.condition_vector(df)
    assert cv["acid_M_log10_grid"].tolist() == [True, False, False, False]
    assert cv["acid_M_below_1e-3"].tolist() == [True, True, False, False]
    assert cv["acid_M_log10_grid"].dtype == bool and cv["acid_M_below_1e-3"].dtype == bool
    np.testing.assert_array_equal(cv["acid_concentration_M"].to_numpy(), df["acid_concentration_M"].to_numpy())
    assert set(N.ACID_SEMANTICS_FLAGS) <= set(cv.columns)
    assert not set(N.ACID_SEMANTICS_FLAGS) & set(N.CONDITION_KEY_FIELDS)
    # the flags are not part of the key: identical conditions still share one key
    assert N.condition_key(_frame({"acid_concentration_M": 0.000977237},
                                  {"acid_concentration_M": 0.000977237})).nunique() == 1


def test_complexant_signature_passthrough():
    df = _frame({"complexant_signature": "HEDTA@0.05"}, {"complexant_signature": np.nan})
    cv = N.condition_vector(df)
    assert cv["complexant_signature"].iloc[0] == "HEDTA@0.05"
    assert pd.isna(cv["complexant_signature"].iloc[1])
    cv2 = N.condition_vector(_frame({}))          # optional column absent -> NA, no error
    assert pd.isna(cv2["complexant_signature"].iloc[0])


def test_log10_non_positive_is_na():
    cv = N.condition_vector(_frame({"acid_concentration_M": 0.0}, {"acid_concentration_M": np.nan}))
    assert cv["log10_acid_M"].isna().all()


def test_not_available_variables_are_na_with_reason():
    cv = N.condition_vector(_frame({}))
    for name in N.NOT_AVAILABLE:
        assert cv[name].isna().all()
        assert cv[f"{name}_status"].iloc[0] == N.NOT_AVAILABLE[name]


# ----------------------------------------------------------------------------- diluent family
@pytest.mark.parametrize("name,expected", [
    ("dodecane", "aliphatic"), ("n-dodecane", "aliphatic"), ("kerosene", "aliphatic"),
    ("sulfonated kerosene", "aliphatic"), ("tph", "aliphatic"), ("isopar l", "aliphatic"),
    ("exxsol d80", "aliphatic"), ("n-octane", "aliphatic"), ("hydrogenated tetrapropene", "aliphatic"),
    ("toluene", "aromatic"), ("tert-butylbenzene", "aromatic"), ("1,4-diisopropylbenzene", "aromatic"),
    ("mesitylene", "aromatic"), ("chloroform", "chlorinated"), ("1,2-dichloroethane", "chlorinated"),
    ("tetrachloroethylene", "chlorinated"), ("ch3cl", "chlorinated"), ("nitrobenzene", "nitroaromatic"),
    ("meta-nitrobenzotrifluoride", "nitroaromatic"), ("2-nitrophenyl hexyl ether", "nitroaromatic"),
    ("[c4mim][tf2n]", "ionic_liquid"), ("phenyl trifluoromethyl sulfone", "fluorinated"),
    ("cyclohexanone", "ketone"), ("1-octanol", "alcohol"), ("iso-decanol", "alcohol"),
    ("exxal 13", "alcohol"), ("tbp", "modifier"), ("dmso", "other"), ("diethylether", "other"),
    ("something unheard of", "other"),
])
def test_component_classes(name, expected):
    assert N.classify_solvent_component(name)[0] == expected


@pytest.mark.parametrize("components,family", [
    (["dodecane"], "aliphatic"),
    (["kerosene", "1-octanol"], "alcohol_modifier_containing"),
    (["1-octanol"], "alcohol_modifier_containing"),
    (["tbp", "dodecane"], "alcohol_modifier_containing"),
    (["isopar l", "exxal 13"], "alcohol_modifier_containing"),
    (["nitrobenzene"], "nitroaromatic"),
    (["meta-nitrobenzotrifluoride", "1-octanol"], "alcohol_modifier_containing"),
    (["toluene", "dodecane"], "other"),
    ([], "other"),
])
def test_diluent_family_rules(components, family):
    fam, classes, rule = N.diluent_family(np.array(components, dtype=object))
    assert fam == family
    assert fam in N.DILUENT_FAMILIES
    if family == "other" and len(set(components)) == 2:
        assert rule.startswith("mixed_classes:")


def test_inferred_trade_names_are_labelled():
    rule = N.classify_solvent_component("hyfrane")[1]
    assert "INFERRED" in rule


# ----------------------------------------------------------------------------- unit sanity
def test_unit_sanity_flags_counts():
    df = _frame(
        {},                                                       # clean
        {"acid_concentration_M": -0.1},                           # negative acid
        {"acid_concentration_M": 24.0},                           # > 16 M
        {"extractant_primary_concentration_M": 5.8,
         "components": np.array([_comp("organic_extractant", "X", TODGA, 5.8)], dtype=object)},
        {"temperature_C": -10.0}, {"temperature_C": 200.0},
        {"phase_ratio_org_aq": 0.0}, {"phase_ratio_org_aq": -1.0},
        {"acid_concentration_M": np.nan, "temperature_C": np.nan, "phase_ratio_org_aq": np.nan},  # NA never flags
        {"acid_concentration_M": 16.0, "temperature_C": 150.0, "extractant_primary_concentration_M": 5.0,
         "components": np.array([_comp("organic_extractant", "X", TODGA, 5.0)], dtype=object)},  # bounds inclusive
    )
    counts = N.unit_sanity_counts(df)
    assert counts["acid_M_negative"] == 1
    assert counts["acid_M_above_16"] == 1
    assert counts["extractant_M_above_5"] == 1
    assert counts["temperature_below_minus5"] == 1
    assert counts["temperature_above_150"] == 1
    assert counts["phase_ratio_nonpositive"] == 2
    flags = N.unit_sanity_flags(df)
    assert not flags.iloc[0].any() and not flags.iloc[8].any() and not flags.iloc[9].any()
    assert set(flags.columns) == set(N.UNIT_SANITY_RULES)


def test_condition_vector_preserves_rows_and_index():
    df = _frame({}, {}, {}).set_index(pd.Index([10, 20, 30]))
    cv = N.condition_vector(df)
    assert cv.index.tolist() == [10, 20, 30]
    assert "log_D" not in cv.columns
    with pytest.raises(KeyError):
        N.condition_vector(df.drop(columns=["components"]))


# ----------------------------------------------------------------------------- full archive
@pytest.mark.slow
def test_archive_condition_vector_invariants():
    from gen19ct.data.load import load_archive
    df = load_archive(copy=False)
    cv = N.condition_vector(df)
    assert len(cv) == len(df) == 16770
    assert cv["pH"].isna().all()
    assert set(cv["diluent_family"]) <= set(N.DILUENT_FAMILIES)
    # the sorted extractant tuple has one entry per organic extractant
    assert (cv["extractant_concentrations_sorted_M"].map(len) == df["n_organic_extractants"]).all()
    k1, k2 = N.condition_key(cv), N.condition_key(cv)
    assert k1.equals(k2)
    assert N.condition_key(df).equals(k1)            # archive frame and condition vector give the same keys
    # values pass through unaltered (never rescaled, never converted to pH)
    for c in ("acid_concentration_M", "extractant_primary_concentration_M", "metal_concentration_M",
              "phase_ratio_org_aq", "temperature_C"):
        np.testing.assert_array_equal(cv[c].to_numpy(), df[c].astype(float).to_numpy())
    assert cv["acid_M_log10_grid"].sum() < cv["acid_concentration_M"].notna().sum() * 0.05
    assert len(cv.index) == len(set(cv.index))
