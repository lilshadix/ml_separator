"""Tests for ``gen19ct.chemistry.metals`` (brief section 27: descriptor consistency, metal alias
normalisation) and for the files ``scripts/g19_build_metals.py`` writes.

Run from the repository root:
    .venv/Scripts/python.exe -m pytest generations/gen19_chem_transfer/tests/test_metals.py -q -p no:cacheprovider
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import metals as M
from gen19ct.manifest import write_csv

TABLE_CSV = paths.DESCRIPTORS_DIR / "metals.csv"


# ------------------------------------------------------------------------------------------------ #
# fixtures
# ------------------------------------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def archive() -> pd.DataFrame:
    from gen19ct.data.load import load_archive
    df = load_archive(copy=False)
    return df[["metal_raw", "metal_oxidation_state_raw", "metal_symbol", "metal_oxidation_state",
               "g19_metal", "g19_ox", "g19_metal_state", "g19_tier"]]


@pytest.fixture(scope="module")
def archive_keys(archive: pd.DataFrame) -> list[tuple[str, int | None]]:
    sub = archive[archive["metal_symbol"].notna()]
    return sorted({(s, None if pd.isna(o) else int(o))
                   for s, o in zip(sub["metal_symbol"], sub["metal_oxidation_state"])},
                  key=lambda k: (M.ATOMIC_NUMBER[k[0]], -1 if k[1] is None else k[1]))


@pytest.fixture(scope="module")
def table(archive_keys) -> pd.DataFrame:
    return M.build_descriptor_table(archive_keys, archive_keys=archive_keys)


def _row(table: pd.DataFrame, symbol: str, ox: int | None) -> pd.Series:
    sel = (table["symbol"] == symbol) & (table["oxidation_state"].isna() if ox is None
                                         else table["oxidation_state"] == ox)
    assert int(sel.sum()) == 1, (symbol, ox)
    return table[sel].iloc[0]


# ------------------------------------------------------------------------------------------------ #
# alias normalisation
# ------------------------------------------------------------------------------------------------ #

@pytest.mark.parametrize("label", ["Nd(III)", "Nd3+", "Nd+3", "Nd (3+)", "Nd III", "Nd+++", "Nd(3)",
                                   "nd(iii)", "Nd-III", "neodymium(III)"])
def test_neodymium_label_variants(label):
    assert M.normalize_metal(label) == ("Nd", 3, "free_ion")


def test_bare_symbol_has_no_state():
    assert M.normalize_metal("Nd") == ("Nd", None, None)
    assert M.normalize_metal("Nd", "III") == ("Nd", 3, "free_ion")
    assert M.normalize_metal("Am", 3.0) == ("Am", 3, "free_ion")


@pytest.mark.parametrize("label", ["UO2+2", "UO2(2+)", "UO2^2+", "UO22+", "UO2++", "UO2 2+", "uranyl"])
def test_uranyl_is_uranium_vi(label):
    assert M.normalize_metal(label) == ("U", 6, "uranyl")


def test_other_actinyls():
    assert M.normalize_metal("PuO2+2") == ("Pu", 6, "plutonyl")
    assert M.normalize_metal("NpO2+") == ("Np", 5, "neptunyl")
    assert M.normalize_metal("NpO2(+)") == ("Np", 5, "neptunyl")
    assert M.normalize_metal("U(VI)") == ("U", 6, "uranyl")
    assert M.normalize_metal("UO2+2", None) == ("U", 6, "uranyl")
    assert M.normalize_metal("UO2+2", "VI") == ("U", 6, "uranyl")


@pytest.mark.parametrize("label", ["Xx", "Q(III)", "Zz3+", "NdIII", "", None, float("nan")])
def test_unknown_or_malformed_label_raises(label):
    with pytest.raises(ValueError):
        M.normalize_metal(label)


def test_ambiguous_or_conflicting_labels_raise():
    with pytest.raises(ValueError):
        M.normalize_metal("UO2")               # no charge: solid UO2 (U(IV)) or uranyl?
    with pytest.raises(ValueError):
        M.normalize_metal("UO2+2", "IV")       # uranyl is U(VI)
    with pytest.raises(ValueError):
        M.normalize_metal("Nd(III)", "II")
    with pytest.raises(ValueError):
        M.normalize_metal("Nd(aq)")


def test_parse_oxidation_state():
    for v in ("III", "iii", 3, 3.0, "3", "+3", "3+", "(III)", "+++"):
        assert M.parse_oxidation_state(v) == 3
    assert M.parse_oxidation_state(None) is None
    assert M.parse_oxidation_state(float("nan")) is None
    for bad in ("IIII", "3.5", 3.5, -1, "x"):
        with pytest.raises(ValueError):
            M.parse_oxidation_state(bad)


def test_every_archive_metal_state_resolvable(archive):
    labels = sorted({s for s in archive["g19_metal_state"] if isinstance(s, str)})
    assert labels, "archive has no metal states"
    for lab in labels:
        sub = archive[archive["g19_metal_state"] == lab]
        sym, ox, form = M.normalize_metal(lab)
        assert (sym, ox) == (sub["g19_metal"].iloc[0], int(sub["g19_ox"].iloc[0])), lab
        assert form is not None
    combos = archive[archive["metal_raw"].notna()].drop_duplicates(
        ["metal_raw", "metal_oxidation_state_raw", "metal_symbol", "metal_oxidation_state"])
    for r in combos.itertuples(index=False):
        sym, ox, _ = M.normalize_metal(r.metal_raw, r.metal_oxidation_state_raw)
        assert sym == r.metal_symbol
        assert ox == (None if pd.isna(r.metal_oxidation_state) else int(r.metal_oxidation_state))


def test_unresolved_archive_rows_have_no_label(archive):
    # the only rows normalize_metal cannot see are those whose archive label is empty
    assert archive.loc[archive["metal_symbol"].isna(), "metal_raw"].isna().all()


# ------------------------------------------------------------------------------------------------ #
# descriptor table structure
# ------------------------------------------------------------------------------------------------ #

def test_table_covers_archive_and_series(table, archive_keys, archive):
    keys = {(s, None if pd.isna(o) else int(o)) for s, o in zip(table["symbol"], table["oxidation_state"])}
    assert set(archive_keys) <= keys
    assert {(s, 3) for s in M.LANTHANIDES} <= keys
    assert {(s, 3) for s in M.AN3_SERIES} <= keys
    model = archive[archive["g19_tier"] == "MODEL"]
    model_keys = {M.descriptor_key(s, o) for s, o in zip(model["g19_metal"], model["g19_ox"])}
    table_keys = {M.descriptor_key(s, o) for s, o in zip(table["symbol"], table["oxidation_state"])}
    assert model_keys <= table_keys


def test_no_duplicate_symbol_state_key(table):
    assert not table.duplicated(["symbol", "oxidation_state"]).any()


def test_z_unique_per_symbol(table):
    assert (table.groupby("symbol")["Z"].nunique() == 1).all()
    assert (table.groupby("Z")["symbol"].nunique() == 1).all()
    for sym, z in zip(table["symbol"], table["Z"]):
        assert M.ATOMIC_NUMBER[sym] == z
    assert len(M.PERIODIC_SYMBOLS) == 118 and len(set(M.PERIODIC_SYMBOLS)) == 118


def test_element_level_columns_constant_per_symbol(table):
    state_specific = pd.Series([(s, None if pd.isna(o) else int(o)) in M.PAULING_EN_STATE
                                for s, o in zip(table["symbol"], table["oxidation_state"])], index=table.index)
    for col in M.ELEMENT_LEVEL_COLUMNS:
        t = table[~state_specific] if col == "electronegativity_pauling" else table
        n = t.groupby("symbol")[col].apply(
            lambda s: pd.Series(["NA" if pd.isna(v) else str(v) for v in s]).nunique())
        assert (n == 1).all(), col


def test_state_specific_electronegativity_lead():
    """Allred's Pb(II) value replaces the tabulated element value 2.33, which is Pb(IV) (CHEM-08)."""
    t = M.build_descriptor_table([("Pb", 2), ("Pb", None)], archive_keys=[("Pb", 2), ("Pb", None)])
    pb2 = t[(t["symbol"] == "Pb") & (t["oxidation_state"] == 2)].iloc[0]
    pb = t[(t["symbol"] == "Pb") & t["oxidation_state"].isna()].iloc[0]
    assert pb2["electronegativity_pauling"] == pytest.approx(1.87)
    assert "+2 state" in pb2["electronegativity_pauling_source"]
    assert pb["electronegativity_pauling"] == pytest.approx(2.33)
    assert "Pb(IV)" in pb["electronegativity_pauling_source"]


def test_na_state_rows_carry_element_level_only(table):
    na = table[table["oxidation_state"].isna()]
    assert len(na) > 0
    for col in M.ION_LEVEL_COLUMNS:
        assert na[col].isna().all(), col
        assert na[f"{col}_source"].str.startswith("NA_REASON").all(), col
    assert na["Z"].notna().all() and na["category"].notna().all()


def test_every_value_has_a_source(table):
    for col in M.VALUE_COLUMNS:
        src = table[f"{col}_source"]
        assert src.notna().all() and (src.str.strip() != "").all(), col
        has = table[col].notna()
        assert not src[has].str.startswith("NA_REASON").any(), col
        assert src[~has].str.startswith("NA_REASON").all(), col
    assert set(M.VALUE_COLUMNS) | {"symbol"} <= set(table.columns)


def test_categories_and_groups(table):
    assert set(table["category"]) <= set(M.CATEGORIES)
    assert set(table.loc[table["symbol"].isin(["Y", "Sc"]), "category"]) == {"rare_earth_non_lanthanide"}
    f_block = table["block"] == "f"
    assert table.loc[f_block, "group"].isna().all()
    assert table.loc[~f_block, "group"].notna().all()
    assert set(table.loc[table["category"] == "lanthanide", "series"]) == {"Ln"}
    assert set(table.loc[table["category"] == "actinide", "series"]) == {"An"}
    assert set(table.loc[~table["category"].isin(["lanthanide", "actinide"]), "series"]) == {"none"}


# ------------------------------------------------------------------------------------------------ #
# descriptor consistency (chemistry identities)
# ------------------------------------------------------------------------------------------------ #

def _ln3(table: pd.DataFrame) -> pd.DataFrame:
    sub = table[(table["series"] == "Ln") & (table["oxidation_state"] == 3)].sort_values("Z")
    assert list(sub["symbol"]) == list(M.LANTHANIDES)
    return sub


@pytest.mark.parametrize("col", ["radius_cn8_A", "radius_cn6_A", "radius_cn9_A"])
def test_ln3_radius_strictly_decreasing(table, col):
    r = _ln3(table)[col].to_numpy(dtype=float)
    assert np.isfinite(r).all()
    assert (np.diff(r) < 0).all(), col


def test_radii_increase_with_coordination_number(table):
    for r in table.itertuples(index=False):
        vals = [(cn, getattr(r, f"radius_cn{cn}_A")) for cn in (6, 8, 9)]
        vals = [(cn, v) for cn, v in vals if not pd.isna(v)]
        for (c1, v1), (c2, v2) in zip(vals, vals[1:]):
            assert v1 < v2, (r.symbol, r.oxidation_state, c1, c2)


def test_series_index_and_f_count_identities(table):
    for r in table.itertuples(index=False):
        if r.series == "Ln":
            assert r.series_index == r.Z - 57
        elif r.series == "An":
            assert r.series_index == r.Z - 89
        else:
            assert pd.isna(r.series_index)
        if pd.isna(r.f_electron_count):
            continue
        if r.series == "Ln":
            assert r.f_electron_count == r.Z - 54 - r.oxidation_state
        if r.series == "An":
            assert r.f_electron_count == r.Z - 86 - r.oxidation_state
        if r.series in ("Ln", "An") and r.oxidation_state == 3:
            assert r.f_electron_count == r.series_index
        assert 0 <= r.f_electron_count <= 14 and 0 <= r.d_electron_count <= 10
    assert list(_ln3(table)["f_electron_count"]) == list(range(15))


@pytest.mark.parametrize("symbol,ox,f", [("La", 3, 0), ("Lu", 3, 14), ("U", 6, 0), ("Pu", 4, 4),
                                         ("Am", 3, 6), ("Cm", 3, 7), ("Th", 4, 0), ("Gd", 3, 7),
                                         ("Np", 5, 2), ("Cf", 3, 9)])
def test_f_count_spot_values(table, symbol, ox, f):
    row = _row(table, symbol, ox)
    assert row["f_electron_count"] == f
    assert row["electron_configuration"] == (("[Xe]" if symbol in M.LANTHANIDES else "[Rn]")
                                             + (f" {'4f' if symbol in M.LANTHANIDES else '5f'}{f}" if f else ""))


def test_d_block_and_p_block_configurations(table):
    assert _row(table, "Pd", 2)[["electron_configuration", "d_electron_count"]].tolist() == ["[Kr] 4d8", 8]
    assert _row(table, "Pd", 4)["d_electron_count"] == 6
    assert _row(table, "Cd", 2)["d_electron_count"] == 10
    assert _row(table, "Hf", 4)[["electron_configuration", "f_electron_count"]].tolist() == ["[Xe] 4f14", 14]
    assert _row(table, "Pb", 2)["electron_configuration"] == "[Xe] 4f14 5d10 6s2"
    assert _row(table, "Tc", 7)["electron_configuration"] == "[Kr]"


def test_species_and_charges(table):
    assert _row(table, "U", 6)[["species_form", "species_charge", "formal_charge"]].tolist() == ["uranyl", 2, 6]
    assert _row(table, "Np", 5)[["species_form", "species_charge"]].tolist() == ["neptunyl", 1]
    assert _row(table, "Pu", 6)["species_charge"] == 2
    assert _row(table, "Am", 6)["species_charge"] == 2
    assert _row(table, "Tc", 7)[["species_form", "species_charge"]].tolist() == ["pertechnetate", -1]
    free = table[table["species_form"] == "free_ion"]
    assert (free["species_charge"] == free["oxidation_state"]).all()
    assert (free["formal_charge"] == free["oxidation_state"]).all()
    # effective charge of an actinyl lies between its net charge and its oxidation state
    act = table[table["species_form"].isin(list(M.ACTINYL_NAMES.values())) & table["effective_charge"].notna()]
    assert len(act) >= 2
    assert ((act["effective_charge"] > act["species_charge"]) & (act["effective_charge"] < act["oxidation_state"])).all()


def test_implausible_state_keeps_element_level_only(table):
    row = _row(table, "Sr", 3)
    assert row["species_form"] == "implausible_state"
    assert not pd.isna(row["state_plausible"]) and not bool(row["state_plausible"])
    for col in M.ION_LEVEL_COLUMNS:
        if col != "species_form":
            assert pd.isna(row[col]), col
    assert row["Z"] == 38


def test_accessible_states_and_cn8_match_archive_reference_module():
    spec = importlib.util.spec_from_file_location("_t_sae_chem", paths.ARCHIVE_DIR / "scripts" / "sae_chem.py")
    chem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chem)
    for sym, states in chem.PLAUSIBLE_OX.items():
        assert M.ACCESSIBLE_STATES[sym] == states, sym
    assert set(M.ACCESSIBLE_STATES) - set(chem.PLAUSIBLE_OX) == {"Ac", "Bk"}
    for key, val in chem.SHANNON_CN8.items():
        assert M.SHANNON_RADII[key][8] == pytest.approx(val, abs=1e-9), key
    for sym in chem.ATOMIC_NUMBER:
        assert M.ATOMIC_NUMBER[sym] == chem.ATOMIC_NUMBER[sym]
        assert M._ELEMENT_ROWS[sym][4] == chem.metal_category(sym)


# ------------------------------------------------------------------------------------------------ #
# written artefacts
# ------------------------------------------------------------------------------------------------ #

@pytest.mark.skipif(not TABLE_CSV.exists(), reason="run scripts/g19_build_metals.py first")
def test_written_csv_reproduces_from_builder(tmp_path):
    written = pd.read_csv(TABLE_CSV, keep_default_na=True)
    arch = [(s, None if pd.isna(o) else int(o)) for s, o, a in
            zip(written["symbol"], written["oxidation_state"], written["in_archive"]) if a]
    rebuilt = M.build_descriptor_table(arch, archive_keys=arch)
    out = write_csv(rebuilt, tmp_path / "metals.csv")
    assert out.read_bytes() == TABLE_CSV.read_bytes()


@pytest.mark.skipif(not (paths.DATA_AUDIT_DIR / "metal_descriptor_coverage.csv").exists(),
                    reason="run scripts/g19_build_metals.py first")
def test_no_radius_disagreement_in_crosschecks():
    cov = pd.read_csv(paths.DATA_AUDIT_DIR / "metal_descriptor_coverage.csv")
    rad = cov[cov["column"].astype(str).str.startswith("radius") & (cov["section"] != "coverage")]
    assert len(rad) > 0
    assert not (rad["status"] == "DISAGREE").any()
    assert (rad["abs_diff"].dropna() <= 0.005).all()
