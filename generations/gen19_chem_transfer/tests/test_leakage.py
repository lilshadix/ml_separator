"""Tests for ``gen19ct.data.leakage``: detectors on small synthetic frames, and the fold / pair guards.

Brief section 27 requires tests for fold isolation, no source leakage, no duplicate crossing folds,
pair-generation isolation, metal / ligand alias handling, and an explicit check that a hidden
metal x ligand pair is absent from the training set (``test_v5_hidden_cell_absent_from_train``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen19ct.data import leakage as L

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
DHOA = "CCCCCCN(CCCCCC)C(=O)CCCC"


def _row(i: int, **kw) -> dict:
    base = {
        "canonical_measurement_id": f"SAE:{i}", "source_record_id": str(i), "source_file": "Nd.csv",
        "source_line_number": i + 2, "entry_author": "A", "addition_date": "2024-01-01 10:00:00",
        "g19_publication_id": "pub_a", "g19_publication_status": "DOI", "g19_publication_refs": "10.1/a",
        "g19_study_id": "pub_a", "doi_all": np.array(["10.1/a"], dtype=object), "doi_primary": "10.1/a",
        "doi_primary_corrected": "10.1/a", "reference_other": np.array([], dtype=object), "sub_source_file": None,
        "archive_citation_doi": None, "reference_title": "Paper A", "reference_year": 2020.0, "export_fanout_size": 1,
        "series_id": "s1", "extractant_system_key": TODGA, "extractant_primary_name": "TODGA",
        "g19_metal": "Nd", "g19_ox": 3.0, "g19_metal_state": "Nd(III)", "metal_raw": "Nd",
        "acid_primary": "HNO3", "solvent_key": "dodecane:1", "acid_concentration_M": 1.0,
        "extractant_primary_concentration_M": 0.1, "temperature_C": 25.0, "modifier_name": None,
        "complexant_signature": None, "holdback_smiles_canonical": None, "metal_concentration_M": np.nan,
        "phase_ratio_org_aq": 1.0, "modifier_concentration_M": np.nan, "nitrate_concentration_M": np.nan,
        "holdback_concentration_M": np.nan, "log_D": 0.5, "D_value": 10 ** 0.5, "D_raw": "3.16",
        "g19_tier": "MODEL", "duplicate_group_id": f"DG{i:04d}", "duplicate_class": "UNIQUE",
        "duplicate_class_reason": "", "duplicate_group_size": 1, "is_canonical_row": True,
        "group_representative_id": f"SAE:{i}", "identity_hash": f"h{i}", "model_readiness": "A_model_ready",
        "comments_raw": None, "data_location": None, "ini_comp_raw": "HNO3, TODGA, Nd, dodecane",
    }
    base.update(kw)
    return base


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------------------------- #

def test_round_sig():
    out = L.round_sig([123456.789, 0.000123456, 0.0, np.nan, -2.5551], 3)
    assert out[0] == pytest.approx(123000.0) and out[1] == pytest.approx(0.000123)
    assert out[2] == 0.0 and np.isnan(out[3]) and out[4] == pytest.approx(-2.56)


def test_near_duplicate_key_tiers_na_and_state():
    df = _frame(_row(0), _row(1, acid_concentration_M=1.0000001), _row(2, acid_concentration_M=1.004),
                _row(3, temperature_C=np.nan), _row(4, temperature_C=np.nan), _row(5, g19_ox=3),
                _row(6, g19_ox=np.nan, g19_metal_state=None))
    k6, k3 = L.near_duplicate_key(df, 6), L.near_duplicate_key(df, 3)
    assert k6[0] == k6[1] != k6[2]      # 1e-7 relative is equal at 6 s.f.; 4e-3 is not
    assert k3[0] == k3[2]               # ... but is at 3 s.f.
    assert k6[3] == k6[4] != k6[0]      # NA matches NA, not a value
    assert k6[5] == k6[0]               # 3 and 3.0 agree
    assert k6[6] != k6[0]               # unknown state is a different key ...
    wild = L.near_duplicate_key(df, 6, include_metal_state=False)
    assert wild[6] == wild[0]           # ... unless the state is wildcarded


def test_near_duplicate_key_strict_and_errors():
    df = _frame(_row(0, metal_concentration_M=0.001), _row(1, metal_concentration_M=0.01),
                _row(2, complexant_signature="DTPA@0.01"))
    assert L.near_duplicate_key(df, 6)[0] == L.near_duplicate_key(df, 6)[1] == L.near_duplicate_key(df, 6)[2]
    s = L.near_duplicate_key(df, 6, strict=True)
    assert len({s[0], s[1], s[2]}) == 3
    with pytest.raises(ValueError):
        L.near_duplicate_key(df, 0)
    with pytest.raises(KeyError):
        L.near_duplicate_key(df.drop(columns=["solvent_key"]), 6)


def test_near_duplicate_pairs_and_tier_table():
    df = _frame(_row(0), _row(1, log_D=0.51), _row(2, g19_publication_id="pub_b", log_D=np.nan),
                _row(3, acid_concentration_M=1.004), _row(4, acid_concentration_M=2.0))
    p6 = L.near_duplicate_pairs(df, 6)
    assert sorted(zip(p6["idx_a"], p6["idx_b"])) == [(0, 1), (0, 2), (1, 2)]
    r01 = p6[(p6["idx_a"] == 0) & (p6["idx_b"] == 1)].iloc[0]
    assert bool(r01["same_publication"]) and r01["abs_delta_log_D"] == pytest.approx(0.01)
    assert p6.loc[(p6["idx_a"] == 0) & (p6["idx_b"] == 2), "abs_delta_log_D"].isna().all()
    t = L.near_duplicate_pair_table(df, (6, 3))
    assert len(t) == 6  # 3 pairs at 6 s.f. plus (0,3), (1,3), (2,3) at 3 s.f. only
    row03 = t[(t["idx_a"] == 0) & (t["idx_b"] == 3)].iloc[0]
    assert not row03["near_dup_sig6"] and row03["near_dup_sig3"]
    assert t["near_dup_sig3"].all()


# --------------------------------------------------------------------------------------------- #
# exact duplicates
# --------------------------------------------------------------------------------------------- #

def _ab_frame() -> pd.DataFrame:
    return _frame(
        _row(0, duplicate_group_id="DG1", duplicate_class="A_EXACT_DATABASE_DUPLICATE", duplicate_group_size=2,
             group_representative_id="SAE:0", identity_hash="hA"),
        _row(1, duplicate_group_id="DG1", duplicate_class="A_EXACT_DATABASE_DUPLICATE", duplicate_group_size=2,
             group_representative_id="SAE:0", identity_hash="hA", is_canonical_row=False, g19_tier="TARGET_ONLY",
             model_readiness="D_redundant_duplicate"),
        _row(2, duplicate_group_id="DG2", duplicate_class="E_VALUE_CONFLICT", duplicate_group_size=2,
             group_representative_id="SAE:2", identity_hash="hE", log_D=1.5),
        _row(3, duplicate_group_id="DG2", duplicate_class="E_VALUE_CONFLICT", duplicate_group_size=2,
             group_representative_id="SAE:2", identity_hash="hE", log_D=0.2),
        _row(4),
    )


def test_exact_duplicate_groups_semantics_pass():
    groups, checks = L.exact_duplicate_groups(_ab_frame())
    assert set(groups["duplicate_group_id"]) == {"DG1", "DG2"}
    g2 = groups.set_index("duplicate_group_id").loc["DG2"]
    assert g2["n_rows"] == 2 and g2["log_D_spread"] == pytest.approx(1.3) and g2["n_canonical_rows"] == 2
    failing = [k for k, v in checks.items() if not v["pass"]]
    assert failing == []


def test_exact_duplicate_groups_detect_broken_semantics():
    df = _ab_frame()
    df.loc[4, "is_canonical_row"] = False          # a non-canonical row outside any A/B group
    df.loc[1, "log_D"] = 0.7                        # an A group whose members disagree
    _, checks = L.exact_duplicate_groups(df)
    assert not checks["noncanonical_only_in_AB"]["pass"]
    assert not checks["AB_groups_identical_log_D"]["pass"]
    assert not checks["model_tier_has_no_noncanonical_rows"]["pass"]


# --------------------------------------------------------------------------------------------- #
# copies and double digitisation
# --------------------------------------------------------------------------------------------- #

def test_cross_publication_copies():
    df = _frame(_row(0), _row(1, g19_publication_id="pub_b", log_D=0.503), _row(2, g19_publication_id="pub_c", log_D=0.6),
                _row(3, log_D=0.5))
    c = L.cross_publication_copies(df, 6)
    assert sorted(zip(c["idx_a"], c["idx_b"])) == [(0, 1), (1, 3)]
    assert (c["pub_a"] != c["pub_b"]).all()


def test_double_digitisation_candidates_and_location():
    df = _frame(
        _row(0, data_location="Figure 3"), _row(1, log_D=0.51, comments_raw="fig3 nan nan extraction:nan"),
        _row(2, log_D=0.515, data_location="Table 1"),
        _row(3, g19_publication_id="pub_b"),                                   # other publication: not a candidate
        _row(4, log_D=0.8),                                                    # value too far
        _row(5, duplicate_group_id="DG0000", duplicate_class="A_EXACT_DATABASE_DUPLICATE"),
        _row(6, duplicate_group_id="DG0000", duplicate_class="A_EXACT_DATABASE_DUPLICATE", is_canonical_row=False),
    )
    df.loc[0, ["duplicate_group_id", "duplicate_class"]] = ["DG0000", "A_EXACT_DATABASE_DUPLICATE"]
    dd = L.double_digitisation(df, 3)
    pairs = set(zip(dd["idx_a"], dd["idx_b"]))
    assert (0, 5) not in pairs and (0, 6) not in pairs and (5, 6) not in pairs   # one A/B group
    assert (0, 1) in pairs and (1, 2) in pairs and not any(3 in p or 4 in p for p in pairs)
    rel = {(a, b): r for a, b, r in zip(dd["idx_a"], dd["idx_b"], dd["location_relation"])}
    assert rel[(0, 1)] == "SAME_LOCATION" and rel[(0, 2)] == "DIFFERENT_LOCATION"
    assert rel[(1, 5)] == "LOCATION_UNKNOWN"
    assert not dd.loc[(dd["idx_a"] == 1) & (dd["idx_b"] == 6), "both_canonical"].iloc[0]


def test_location_token_and_digitiser_tag():
    df = _frame(_row(0, data_location="Figure S4A"), _row(1, comments_raw="FIG2 or S1 information nan"),
                _row(2, comments_raw="tables1 nan"), _row(3, comments_raw="nan nan nan extraction:graphreaderM"))
    assert L.location_token(df).tolist() == ["figs4a", "fig2", "tables1", None]
    assert L.digitiser_tag(df).tolist() == [None, None, None, "graphreaderm"]


# --------------------------------------------------------------------------------------------- #
# DOI multiplicity and publication groups
# --------------------------------------------------------------------------------------------- #

SELF = L.SAFE_SELF_CITATION


def _doi_frame() -> pd.DataFrame:
    comp = "10.1009/review"
    return _frame(
        _row(0, g19_publication_id="pub_p1", doi_all=np.array(["10.1001/orig", SELF], dtype=object), doi_primary="10.1001/orig",
             doi_primary_corrected="10.1001/orig", reference_title="Original", entry_author="A"),
        _row(1, g19_publication_id="pub_p1c", doi_all=np.array([comp, "10.1001/orig"], dtype=object), doi_primary=comp,
             doi_primary_corrected=comp, reference_title="Review", entry_author="B"),
        _row(2, g19_publication_id="pub_p2c", doi_all=np.array([comp, "10.1002/orig"], dtype=object), doi_primary=comp,
             doi_primary_corrected=comp, reference_title="Review", entry_author="B"),
        _row(3, g19_publication_id="pub_p3c", doi_all=np.array([comp + "x", "10.1003/orig"], dtype=object),
             doi_primary=comp + "x", doi_primary_corrected=comp, reference_title="Review", entry_author="B"),
        _row(4, g19_publication_id="pub_t", doi_all=np.array(["10.1005/typox"], dtype=object), doi_primary="10.1005/typox",
             doi_primary_corrected="10.1005/typo", reference_title="Typo paper"),
        _row(5, g19_publication_id="pub_t2", doi_all=np.array(["10.1005/typo"], dtype=object), doi_primary="10.1005/typo",
             doi_primary_corrected="10.1005/typo", reference_title="Typo paper"),
    )


def test_row_source_dois_and_roles():
    df = _doi_frame()
    dois = L.row_source_dois(df)
    assert dois[0] == ("10.1001/orig",)                     # self-citation removed
    assert dois[3] == ("10.1003/orig", "10.1009/review")        # raw 'x' spelling replaced by the corrected DOI
    roles = L.doi_roles(df)
    assert roles["10.1009/review"] == "COMPILATION_OR_SECONDARY" and roles["10.1001/orig"] == "PRIMARY_SOURCE"


def test_doi_source_multiplicity_risks():
    out = L.doi_source_multiplicity(_doi_frame())
    risk = {(k, s): r for k, s, r in zip(out["kind"], out["source"], out["risk"])}
    assert risk[("doi", "10.1001/orig")] == "V1_SPLIT_SAME_SOURCE"
    assert risk[("doi", "10.1009/review")] == "COMPILATION_FANOUT"
    assert risk[("doi", "10.1005/typo")] == "V1_SPLIT_SAME_SOURCE"
    assert risk[("reference_title", "typo paper")] == "SAME_TITLE_MULTIPLE_RAW_DOIS"


def test_corrected_publication_id_merges_spelling_split():
    df = _doi_frame()
    from gen19ct.data.load import publication_key

    for i in range(len(df)):  # the synthetic ids above are placeholders; use the loader's real ids
        df.loc[i, "g19_publication_id"] = publication_key(df.loc[i, "doi_all"], df.loc[i, "reference_other"], None)[0]
    corr = L.corrected_publication_id(df)
    assert df.loc[4, "g19_publication_id"] != df.loc[5, "g19_publication_id"]
    assert corr[4] == corr[5] == df.loc[5, "g19_publication_id"]
    assert corr[0] == df.loc[0, "g19_publication_id"]


def test_publication_link_components_cumulative():
    df = _doi_frame()
    copies = pd.DataFrame({"pub_a": ["pub_p2c"], "pub_b": ["pub_t2"]})
    comps = L.publication_link_components(df, copy_pairs=copies).set_index("g19_publication_id")
    g = comps["group_primary_source_doi"]
    assert g["pub_p1"] == g["pub_p1c"]                   # shared primary-source DOI 10.1001/orig
    assert g["pub_p2c"] != g["pub_p3c"]                  # the review alone does not link
    assert comps.loc["pub_t", "group_corrected_doi"] == comps.loc["pub_t2", "group_corrected_doi"]
    assert comps.loc["pub_p2c", "group_cross_publication_copy"] == comps.loc["pub_t2", "group_cross_publication_copy"]
    c = comps["group_compilation_doi"]
    assert c["pub_p1"] == c["pub_p2c"] == c["pub_p3c"]  # the most conservative rule merges the review's papers
    assert (comps["n_publications_in_group_compilation_doi"] >= comps["n_publications_in_group_corrected_doi"]).all()


def test_union_components():
    comp = L.union_components(["b", "c", "x"], ["a", "b", "y"], nodes=["z"])
    assert comp["a"] == comp["b"] == comp["c"] == "a"
    assert comp["x"] == comp["y"] == "x" and comp["z"] == "z"


# --------------------------------------------------------------------------------------------- #
# metal aliases
# --------------------------------------------------------------------------------------------- #

def test_metal_alias_split_risk():
    df = _frame(
        _row(0, g19_metal="U", g19_ox=6.0, g19_metal_state="U(VI)", source_file="U.csv"),
        _row(1, g19_metal="U", g19_ox=np.nan, g19_metal_state=None, source_file="Eu.csv"),   # same conditions, no state
        _row(2, g19_metal="U", g19_ox=4.0, g19_metal_state="U(IV)", g19_publication_id="pub_b", source_file="U.csv"),
        _row(3, g19_metal="U", g19_ox=6.0, g19_metal_state="U(VI)", g19_publication_id="pub_b", acid_concentration_M=3.0),
        _row(4, g19_metal="Nd", source_file="Nd.csv"),
    )
    out = L.metal_alias_split_risk(df)
    el = out[out["scope"] == "element"].set_index("g19_metal")
    assert el.loc["U", "risk"] == "UNKNOWN_AND_KNOWN_STATE" and el.loc["Nd", "risk"] == "NONE"
    assert el.loc["U", "n_model_rows_unknown_state"] == 1
    pe = out[out["scope"] == "publication_element"].set_index("g19_publication_id")
    assert pe.loc["pub_a", "risk"] == "UNKNOWN_AND_KNOWN_STATE+SAME_CONDITIONS"
    assert pe.loc["pub_b", "risk"] == "MULTIPLE_KNOWN_STATES"      # different acid M: not the same experiment
    ef = out[out["scope"] == "export_file"].set_index("g19_metal")
    assert ef.loc["U", "n_rows_file_is_other_metal"] == 2 and ef.loc["Nd", "n_rows_file_is_other_metal"] == 0


# --------------------------------------------------------------------------------------------- #
# metadata -> target
# --------------------------------------------------------------------------------------------- #

def test_target_in_text_detects_planted_value():
    rows = [_row(i, log_D=float(i) / 10, D_value=10 ** (i / 10), D_raw=f"{10 ** (i / 10):.4f}",
                 comments_raw="fig1 nan nan") for i in range(20)]
    rows[7]["comments_raw"] = f"value read as {rows[7]['D_raw']} from the plot"
    out = L.target_in_text(_frame(*rows), columns=("comments_raw",), seed=1).set_index("match_type")
    assert out.loc["D_raw_substring", "n_rows_matched"] == 1
    assert out.loc["D_value_3sf", "n_rows_matched"] == 1
    assert out.loc["D_value_exact", "n_rows_matched"] == 0     # '5.0119' is 6e-6 from 10**0.7: not exact
    assert out.loc["D_raw_substring", "n_rows_matched_permuted"] <= 1


def test_loo_group_mean_is_leave_one_out():
    y = np.array([1.0, 1.0, 5.0, 5.0, 7.0])
    labels = pd.Series(["a", "a", "b", "b", "c"])
    s = L.loo_group_mean_r2(y, labels)
    assert s["n_groups"] == 3 and s["n_singleton_rows"] == 1
    assert s["eta2_in_sample"] == pytest.approx(1.0)
    assert s["loo_r2"] < 1.0                                        # the singleton cannot predict itself
    s_same_value = L.loo_group_mean_r2(np.array([2.0, 2.0, 9.0, 9.0]), pd.Series(["a", "a", "b", "b"]))
    assert s_same_value["loo_r2"] == pytest.approx(1.0)


def test_provenance_group_predictability_and_serial():
    rng = np.random.default_rng(0)
    rows = [_row(i, g19_publication_id=f"pub_{i % 4}", log_D=float(rng.normal()), series_id=f"s{i % 5}")
            for i in range(40)]
    df = _frame(*rows)
    pg = L.provenance_group_predictability(df)
    assert {"g19_publication_id", "source_line_block20", "near_duplicate_key_sig6"} <= set(pg["grouping"])
    assert set(pg["role"]) == {"provenance", "chemistry_reference"}
    sc = L.serial_target_correlation(df).set_index("grouping")
    assert sc.loc["all_consecutive", "n_rows"] == 39
    assert sc.loc["same_publication", "n_rows"] + sc.loc["different_publication", "n_rows"] == 39
    meta = L.metadata_target_leak(df)
    assert set(meta["check"]) == {"target_in_text", "provenance_group_predictability", "serial_target_correlation"}


def test_cross_boundary_burden():
    df = _frame(_row(0), _row(1, g19_publication_id="pub_b"), _row(2, g19_publication_id="pub_c", log_D=2.0),
                _row(3, g19_publication_id="pub_d", acid_concentration_M=4.0))
    t = L.near_duplicate_pair_table(df, (6,))
    b = L.cross_boundary_burden(df, t, flag="near_dup_sig6", key_col="key_id_sig6")
    assert b["n_pairs_cross_publication"] == 3 and b["n_rows_in_cross_publication_pairs"] == 3
    assert b["n_merged_components"] == 1 and b["n_publication_groups_after_merge"] == 2
    tight = L.cross_boundary_burden(df, t, flag="near_dup_sig6", tol=0.005, key_col="key_id_sig6")
    assert tight["n_pairs_cross_publication"] == 1 and tight["largest_component_n_publications"] == 2


# --------------------------------------------------------------------------------------------- #
# bundle overlap
# --------------------------------------------------------------------------------------------- #

def test_bundle_overlap():
    arch = _frame(_row(0, log_D=1.0), _row(1, g19_ox=np.nan, g19_metal_state=None),
                  _row(2, duplicate_group_id="DG9"), _row(3, duplicate_group_id="DG9"))
    arch["g19_bundle_exp_id"] = pd.array([0, 1, 2, 3], dtype="Int64")
    arch["extractant_primary_smiles"] = TODGA
    arch["extractant_smiles_canonical"] = [np.array([TODGA], dtype=object)] * 4
    arch["metal_category"] = "lanthanide"
    bundle = pd.DataFrame({"safe_exp_id": ["Nd_SAFE:0", "Nd_SAFE:1", "Nd_SAFE:2", "Nd_SAFE:3", "Nd_SAFE:99"],
                           "metal_symbol": ["Nd"] * 5, "metal_ox": [3] * 5, "log_D": [1.0, 0.5, 0.5, 0.5, 0.1],
                           "canonical_smiles": [TODGA] * 5, "cond__acid_concentration_M": [1.0] * 5,
                           "cond__extractant_concentration_M": [0.1] * 5, "cond__temperature_C": [25.0] * 5})
    gen6 = pd.DataFrame({"safe_exp_id": bundle["safe_exp_id"], "publication_id": ["pub_a"] * 5})
    out = L.bundle_overlap(bundle, arch, gen6=gen6)
    assert out["joined"].tolist() == [True, True, True, True, False]
    assert out.loc[0, "abs_delta_log_D"] == 0.0 and out.loc[:3, "metal_agree"].all()
    assert out["ox_agree"].tolist()[:2] == [True, False]          # the bundle's +3 where the archive has no state
    assert out.loc[2, "n_bundle_rows_in_duplicate_group"] == 2
    assert out.loc[:3, "publication_id_identical"].all() and out.loc[:3, "smiles_equals_archive_primary"].all()


# --------------------------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------------------------- #

def test_pair_isolation_check():
    pairs = pd.DataFrame({"idx_a": [0, 1, 2], "idx_b": [1, 3, 4]})
    folds = {0: "train", 1: "train", 2: "test", 3: "test"}
    with pytest.raises(AssertionError):
        L.pair_isolation_check(pairs, folds)
    bad = L.pair_isolation_check(pairs, folds, raise_on_violation=False)
    assert bad["reason"].tolist() == ["CROSSES_FOLDS", "UNASSIGNED"]
    ok = L.pair_isolation_check(pairs.iloc[:1], folds)
    assert ok.empty
    with pytest.raises(ValueError):
        L.pair_isolation_check(pairs, pd.Series(["train", "test"], index=[0, 0]))


def test_pairs_generated_after_fold_assignment_are_isolated():
    df = _frame(*[_row(i, g19_publication_id=f"pub_{i % 3}", log_D=float(i)) for i in range(9)])
    folds = pd.Series(np.where(df["g19_publication_id"] == "pub_0", "test", "train"), index=df.index)
    within = []
    for label, part in df.groupby(folds):
        idx = part.index.to_numpy()
        a, b = np.triu_indices(len(idx), k=1)
        within.append(pd.DataFrame({"idx_a": idx[a], "idx_b": idx[b]}))
    assert L.pair_isolation_check(pd.concat(within), folds).empty
    a, b = np.triu_indices(len(df), k=1)                            # pairs built BEFORE the split
    with pytest.raises(AssertionError):
        L.pair_isolation_check(pd.DataFrame({"idx_a": a, "idx_b": b}), folds)


def _guard_frame() -> pd.DataFrame:
    return _frame(
        _row(0, g19_publication_id="pub_a"),
        _row(1, g19_publication_id="pub_b", acid_concentration_M=2.0),
        _row(2, g19_publication_id="pub_c", g19_metal="Pr", g19_metal_state="Pr(III)", acid_concentration_M=3.0),
        _row(3, g19_publication_id="pub_d", extractant_system_key=f"{DHOA}|{TODGA}", acid_concentration_M=4.0),
        _row(4, g19_publication_id="pub_e", g19_ox=np.nan, g19_metal_state=None, acid_concentration_M=5.0),
        _row(5, g19_publication_id="pub_f", extractant_system_key=DHOA, g19_metal="Pr", g19_metal_state="Pr(III)",
             acid_concentration_M=6.0),
    )


def test_fold_isolation_v1():
    df = _guard_frame()
    rep = L.fold_isolation_check([1, 2, 3, 4, 5], [0], df, "V1")
    assert rep["ok"] and rep["violations"]["V1_shared_publication"] == 0
    df.loc[1, "g19_publication_id"] = "pub_a"
    with pytest.raises(AssertionError, match="V1_shared_publication"):
        L.fold_isolation_check([1, 2, 3, 4, 5], [0], df, "V1")


def test_fold_isolation_index_and_level_errors():
    df = _guard_frame()
    with pytest.raises(AssertionError, match="index_overlap"):
        L.fold_isolation_check([0, 1], [1, 2], df, "V1", near_dup_sig=None)
    rep = L.fold_isolation_check([0, 99], [2], df, "V1", raise_on_violation=False, near_dup_sig=None)
    assert rep["violations"]["index_not_in_frame"] == 1
    with pytest.raises(ValueError):
        L.fold_isolation_check([0], [1], df, "V9")


def test_fold_isolation_v2_state_and_unknown_state_alias():
    df = _guard_frame()
    nd = df.index[df["g19_metal"] == "Nd"]
    pr = df.index[df["g19_metal"] == "Pr"]
    # hiding Nd(III) while Nd(?) (row 4) stays in train is an alias violation
    rep = L.fold_isolation_check(list(pr) + [4], [0, 1, 3], df, "V2", raise_on_violation=False)
    assert rep["violations"]["V2_shared_metal_state"] == 0
    assert rep["violations"]["V2_unknown_state_alias"] == 1
    assert L.fold_isolation_check(list(pr), list(nd), df, "V2")["ok"]
    rep = L.fold_isolation_check([0, 2], [1, 5], df, "V2", raise_on_violation=False)
    assert rep["violations"]["V2_shared_metal_state"] == 2


def test_fold_isolation_v2_element_level_and_other_state_warning():
    df = _frame(_row(0, g19_metal="U", g19_ox=6.0, g19_metal_state="U(VI)"),
                _row(1, g19_metal="U", g19_ox=4.0, g19_metal_state="U(IV)", acid_concentration_M=2.0))
    rep = L.fold_isolation_check([1], [0], df, "V2")
    assert rep["ok"] and rep["warnings"]["V2_other_known_state_same_publication_rows"] == 1
    with pytest.raises(AssertionError, match="V2_shared_element"):
        L.fold_isolation_check([1], [0], df, "V2", element_level=True)


def test_v5_hidden_cell_absent_from_train():
    """Brief section 27: explicitly confirm that a hidden metal x ligand pair is absent from training."""
    df = _guard_frame()
    hidden = ("Nd(III)", TODGA)
    in_cell = (df["g19_metal_state"] == hidden[0]) & (df["extractant_system_key"] == hidden[1])
    test_idx = df.index[in_cell]
    train_idx = df.index[~in_cell & df["g19_metal_state"].notna()]
    train_cells = set(zip(df.loc[train_idx, "g19_metal_state"], df.loc[train_idx, "extractant_system_key"]))
    assert hidden not in train_cells
    rep = L.fold_isolation_check(train_idx, test_idx, df, "V5")
    assert rep["ok"] and rep["violations"]["V5_shared_cell"] == 0
    # Nd(III) x TODGA|DHOA stays in train (allowed by V5: a different system) but is reported;
    # Pr(III) x DHOA shares a component too, but not the hidden metal state, so it is not counted
    assert rep["warnings"]["V5_train_rows_sharing_component_with_hidden_cell"] == 1
    leaky = list(train_idx) + [int(df.index[in_cell][0])]
    with pytest.raises(AssertionError, match="V5_shared_cell"):
        L.fold_isolation_check(leaky, list(test_idx[1:]) or [99], df, "V5", near_dup_sig=None)


def test_v5_unknown_state_alias():
    df = _guard_frame()
    rep = L.fold_isolation_check([2, 3, 4, 5], [0, 1], df, "V5", raise_on_violation=False)
    assert rep["violations"]["V5_shared_cell"] == 0 and rep["violations"]["V5_unknown_state_alias"] == 1


def test_no_duplicate_group_crossing_folds():
    df = _frame(_row(0, g19_publication_id="pub_a", duplicate_group_id="DG_E", duplicate_class="E_VALUE_CONFLICT"),
                _row(1, g19_publication_id="pub_b", duplicate_group_id="DG_E", duplicate_class="E_VALUE_CONFLICT",
                     acid_concentration_M=1.5, log_D=2.0))
    with pytest.raises(AssertionError, match="duplicate_group_shared"):
        L.fold_isolation_check([0], [1], df, "V1", near_dup_sig=None)
    df.loc[1, "duplicate_group_id"] = "DG_other"
    assert L.fold_isolation_check([0], [1], df, "V1")["ok"]


def test_no_near_duplicate_crossing_folds():
    df = _frame(_row(0, g19_publication_id="pub_a", log_D=0.5), _row(1, g19_publication_id="pub_b", log_D=0.5),
                _row(2, g19_publication_id="pub_c", log_D=0.9), _row(3, g19_publication_id="pub_d", acid_concentration_M=2.0))
    with pytest.raises(AssertionError, match="near_duplicate_key_sig6_shared"):
        L.fold_isolation_check([0, 3], [1], df, "V1")
    rep = L.fold_isolation_check([2, 3], [1], df, "V1", raise_on_violation=False)
    assert rep["violations"]["near_duplicate_key_sig6_shared"] == 1
    tol = L.fold_isolation_check([2, 3], [1], df, "V1", near_dup_value_tol=0.005)
    assert tol["ok"]                                                # same conditions, different value
    with pytest.raises(AssertionError, match="within_0.005"):
        L.fold_isolation_check([0, 3], [1], df, "V1", near_dup_value_tol=0.005)
    assert L.fold_isolation_check([0, 1], [3], df, "V1")["ok"]


# --------------------------------------------------------------------------------------------- #
# the archive
# --------------------------------------------------------------------------------------------- #

def test_archive_duplicate_semantics_and_model_rows():
    from gen19ct.data.load import load_archive

    df = load_archive(copy=False)
    _, checks = L.exact_duplicate_groups(df)
    assert all(v["pass"] for v in checks.values()), {k: v for k, v in checks.items() if not v["pass"]}
    model = df[df["g19_tier"] == "MODEL"]
    p = L.near_duplicate_pairs(model, 6)
    assert not p["both_ab_group"].any()                              # MODEL rows hold one member per A/B group
    k = L.near_duplicate_key(df, 6)
    assert k.notna().all() and len(k) == len(df)


# --------------------------------------------------------------------------------------------- #
# corrections of the Phase A/B verification (P3, P6)
# --------------------------------------------------------------------------------------------- #

def test_publication_link_components_merge_archive_duplicate_groups():
    """P3: an archive duplicate group (here an E_VALUE_CONFLICT pair) spanning two publications merges them
    before the copy rule, so a leave-group-out fold cannot split it; the mask limits the linking rows."""
    df = _frame(_row(0, g19_publication_id="pub_a", duplicate_group_id="DG_E", duplicate_class="E_VALUE_CONFLICT"),
                _row(1, g19_publication_id="pub_b", duplicate_group_id="DG_E", duplicate_class="E_VALUE_CONFLICT",
                     doi_all=np.array(["10.1/b"], dtype=object), doi_primary="10.1/b", doi_primary_corrected="10.1/b",
                     acid_concentration_M=1.5, log_D=2.0),
                _row(2, g19_publication_id="pub_c", doi_all=np.array(["10.1/c"], dtype=object), doi_primary="10.1/c",
                     doi_primary_corrected="10.1/c"))
    comps = L.publication_link_components(df).set_index("g19_publication_id")
    assert comps.loc["pub_a", "group_primary_source_doi"] != comps.loc["pub_b", "group_primary_source_doi"]
    for rule in ("archive_duplicate_group", "cross_publication_copy", "compilation_doi"):
        assert comps.loc["pub_a", f"group_{rule}"] == comps.loc["pub_b", f"group_{rule}"], rule
    assert comps.loc["pub_c", "group_cross_publication_copy"] != comps.loc["pub_a", "group_cross_publication_copy"]
    masked = L.publication_link_components(df, duplicate_group_mask=[True, False, True]).set_index("g19_publication_id")
    assert masked.loc["pub_a", "group_cross_publication_copy"] != masked.loc["pub_b", "group_cross_publication_copy"]
    # the merged groups pass the guard, the unmerged ones do not
    g = df["g19_publication_id"].map(comps["group_cross_publication_copy"])
    te = g == g[0]
    assert L.fold_isolation_check(df.index[~te], df.index[te], df, "V1", near_dup_sig=None,
                                  raise_on_violation=False)["violations"]["duplicate_group_shared"] == 0


def test_v5_component_aware_guard_and_v6_level():
    """P6: with component_aware (and always for V6) the hidden state or an X(?) row of its element left in a
    component-sharing system is a violation, not a warning."""
    df = _frame(
        _row(0, g19_publication_id="pub_a"),                                                   # Nd(III) x TODGA
        _row(1, g19_publication_id="pub_b", extractant_system_key=f"{DHOA}|{TODGA}", acid_concentration_M=2.0),
        _row(2, g19_publication_id="pub_c", extractant_system_key=f"{DHOA}|{TODGA}", g19_ox=np.nan,
             g19_metal_state=None, acid_concentration_M=3.0),                                  # Nd(?) in the mixture
        _row(3, g19_publication_id="pub_d", g19_metal="Pr", g19_metal_state="Pr(III)", acid_concentration_M=4.0),
        _row(4, g19_publication_id="pub_e", extractant_system_key=DHOA, acid_concentration_M=5.0),   # Nd(III) x DHOA
    )
    lax = L.fold_isolation_check([1, 2, 3, 4], [0], df, "V5", near_dup_sig=None)
    assert lax["ok"] and lax["warnings"]["V5_train_rows_sharing_component_with_hidden_cell"] == 1
    rep = L.fold_isolation_check([1, 2, 3, 4], [0], df, "V5", near_dup_sig=None, component_aware=True,
                                 raise_on_violation=False)
    assert rep["violations"]["V5_hidden_state_in_component_sharing_system"] == 2   # rows 1 (Nd(III)) and 2 (Nd(?))
    with pytest.raises(AssertionError, match="V5_hidden_state_in_component_sharing_system"):
        L.fold_isolation_check([1, 2, 3, 4], [0], df, "V6", near_dup_sig=None)
    assert L.fold_isolation_check([3, 4], [0], df, "V6", near_dup_sig=None)["ok"]     # DHOA alone shares nothing
    # a component map that equates TODGA and DHOA makes row 4 a violation too
    rep = L.fold_isolation_check([3, 4], [0], df, "V5", near_dup_sig=None, component_aware=True,
                                 component_map={TODGA: "X", DHOA: "X"}, raise_on_violation=False)
    assert rep["violations"]["V5_hidden_state_in_component_sharing_system"] == 1


def test_v5_component_aware_guard_is_state_level():
    """Pre-seal design decision: Pu(VI) x TODGA hidden.  In TODGA|DHOA a Pu(VI) or Pu(?) train row is a violation;
    a Pu(IV) train row is allowed there, exactly as Pu(IV) x TODGA is allowed in the hidden cell's own system."""
    mix = f"{DHOA}|{TODGA}"
    pu = dict(g19_metal="Pu", metal_raw="Pu")
    df = _frame(
        _row(0, g19_publication_id="pub_a", g19_ox=6.0, g19_metal_state="Pu(VI)", **pu),               # test cell
        _row(1, g19_publication_id="pub_b", g19_ox=4.0, g19_metal_state="Pu(IV)", acid_concentration_M=2.0, **pu),
        _row(2, g19_publication_id="pub_c", extractant_system_key=mix, g19_ox=4.0, g19_metal_state="Pu(IV)",
             acid_concentration_M=3.0, **pu),
        _row(3, g19_publication_id="pub_d", extractant_system_key=mix, g19_ox=6.0, g19_metal_state="Pu(VI)",
             acid_concentration_M=4.0, **pu),
        _row(4, g19_publication_id="pub_e", extractant_system_key=mix, g19_ox=np.nan, g19_metal_state=None,
             acid_concentration_M=5.0, **pu),
    )
    ok = L.fold_isolation_check([1, 2], [0], df, "V5", near_dup_sig=None, component_aware=True)
    assert ok["ok"] and ok["warnings"]["V5_other_known_state_of_hidden_element_in_component_sharing_system_rows"] == 1
    rep = L.fold_isolation_check([1, 2, 3, 4], [0], df, "V5", near_dup_sig=None, component_aware=True,
                                 raise_on_violation=False)
    assert rep["violations"]["V5_hidden_state_in_component_sharing_system"] == 2          # rows 3 and 4
    assert rep["violations"]["V5_shared_cell"] == 0 and rep["violations"]["V5_unknown_state_alias"] == 0


def test_publication_group_guard_sees_a_copy_group_row_with_another_publication_id():
    """Task X, leakage finding VR-02: V5-P masks the scored row's copy group (``group_cross_publication_copy``).  A
    training row of that group whose ``g19_publication_id`` differs from every scored row's passes the raw-id V1 level
    but is caught once the guard reads the group (``io.publication_group_frame``); ``cell_holdout.guard_v5p`` and
    ``source_holdout.guard_v1`` use that basis."""
    from gen19ct.folds import cell_holdout as CH
    from gen19ct.folds import io as FI
    from gen19ct.folds import source_holdout as SH

    df = _frame(
        _row(0, g19_publication_id="pub_a"),                                                      # scored cell row, gA
        _row(1, g19_publication_id="pub_b", g19_metal="Pr", g19_metal_state="Pr(III)", acid_concentration_M=2.0),  # gA
        _row(2, g19_publication_id="pub_c", g19_metal="Eu", g19_metal_state="Eu(III)", acid_concentration_M=3.0),  # gC
    )
    df[FI.GROUP_COL] = ["gA", "gA", "gC"]
    fold = FI.make_fold(design="V5P", variant="base", scheme="cell_x_group", fold_id="t", half="S", seed=None,
                        hidden=["SAE:0"], scored=["SAE:0"], unit_type="cell", units=["Nd(III) x TODGA"],
                        meta={"cells": [["Nd(III)", TODGA]], "publication_group": "gA", "component_aware": True})
    raw = FI.guard(fold, df, df.index, ["SAE:0"], "V1", near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL)
    assert raw["ok"] and raw["publication_basis"] == "g19_publication_id"                    # the blind spot
    with pytest.raises(AssertionError, match="V1_shared_publication"):
        FI.guard(fold, df, df.index, ["SAE:0"], "V1", publication_col=FI.GROUP_COL, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL)
    with pytest.raises(AssertionError, match="V1_shared_publication"):
        CH.guard_v5p(fold, df)
    with pytest.raises(AssertionError, match="V1_shared_publication"):
        SH.guard_v1(FI.make_fold(design="V1", variant="copy", scheme="exact", fold_id="gA", half="S", seed=None,
                                 hidden=["SAE:0"], scored=["SAE:0"], unit_type="publication_group", units=["gA"]), df)
    # hiding the whole group passes on both bases
    whole = FI.make_fold(design="V5P", variant="base", scheme="cell_x_group", fold_id="t2", half="S", seed=None,
                         hidden=["SAE:0", "SAE:1"], scored=["SAE:0"], unit_type="cell", units=["Nd(III) x TODGA"],
                         meta={"cells": [["Nd(III)", TODGA]], "publication_group": "gA", "component_aware": True})
    assert all(r["ok"] for r in CH.guard_v5p(whole, df))
    assert CH.guard_v5p(whole, df)[1]["publication_basis"] == FI.GROUP_COL
    with pytest.raises(KeyError, match="absent"):
        FI.publication_group_frame(df.drop(columns=[FI.GROUP_COL]))


def test_v3_and_v4_levels():
    df = _guard_frame()
    todga_rows = df.index[df["extractant_system_key"] == TODGA]
    rest = df.index.difference(todga_rows)
    rep = L.fold_isolation_check(rest, todga_rows, df, "V3", near_dup_sig=None, raise_on_violation=False)
    assert rep["violations"]["V3_shared_system"] == 0 and rep["violations"]["V3_component_sharing_system"] == 1
    clean = df.index[df["extractant_system_key"] == DHOA]
    assert L.fold_isolation_check(clean, todga_rows, df, "V3", near_dup_sig=None)["ok"]          # DHOA alone shares nothing
    fam = {TODGA: "diglycolamide", DHOA: "monoamide"}
    rep = L.fold_isolation_check(rest, todga_rows, df, "V4", near_dup_sig=None, hidden_families=["diglycolamide"],
                                 family_map=fam, raise_on_violation=False)
    assert rep["violations"]["V4_train_system_contains_hidden_family"] == 1                # DHOA|TODGA
    assert L.fold_isolation_check(clean, todga_rows, df, "V4", near_dup_sig=None, hidden_families=["diglycolamide"],
                                  family_map=fam)["ok"]
    with pytest.raises(ValueError):
        L.fold_isolation_check(clean, todga_rows, df, "V4")


# --------------------------------------------------------------------------------------------- #
# wildcard copies (verification finding VL-04)
# --------------------------------------------------------------------------------------------- #

def test_wildcard_copy_pairs_state_token_and_structure_key() -> None:
    df = _frame(
        # 0/1: the same Zr measurement recorded as Zr(IV) and Zr(?) in two publications (0.2688 vs 0.269)
        _row(0, g19_metal="Zr", g19_ox=4.0, g19_metal_state="Zr(IV)", log_D=float(np.log10(0.2688)), D_raw="0.2688",
             g19_publication_id="pub_a"),
        _row(1, g19_metal="Zr", g19_ox=np.nan, g19_metal_state=None, log_D=float(np.log10(0.269)), D_raw="0.269",
             g19_publication_id="pub_b"),
        # 2/3: a bit-identical, non-decade D under two different systems of one figure
        _row(2, g19_metal="Am", g19_ox=3.0, g19_metal_state="Am(III)", log_D=float(np.log10(8.36282369671159)),
             D_raw="8.36282369671159", data_location="Figure 3"),
        _row(3, g19_metal="Am", g19_ox=3.0, g19_metal_state="Am(III)", extractant_system_key=DHOA,
             log_D=float(np.log10(8.36282369671159)), D_raw="8.36282369671159", data_location="Figure 3"),
        # 4/5: a decade value shared by two systems (a coincidence or a detection floor: not strict)
        _row(4, g19_metal="Eu", g19_ox=3.0, g19_metal_state="Eu(III)", log_D=-2.0, D_raw="0.01"),
        _row(5, g19_metal="Eu", g19_ox=3.0, g19_metal_state="Eu(III)", extractant_system_key=DHOA, log_D=-2.0,
             D_raw="0.01"),
        # 6/7: the same state and system at the same conditions -- the registered key already sees it; no wildcard pair
        _row(6, g19_metal="Ce", g19_ox=3.0, g19_metal_state="Ce(III)", log_D=0.1),
        _row(7, g19_metal="Ce", g19_ox=3.0, g19_metal_state="Ce(III)", log_D=0.1),
        # 8: Zr(IV) far in value from row 1 -> no pair with it beyond the tolerance
        _row(8, g19_metal="Zr", g19_ox=4.0, g19_metal_state="Zr(IV)", log_D=0.9, g19_publication_id="pub_c"),
    )
    w = L.wildcard_copy_pairs(df)
    got = {(r.kind, tuple(sorted((r.id_a, r.id_b)))): r for r in w.itertuples(index=False)}
    assert set(got) == {("STATE_WILDCARD", ("SAE:0", "SAE:1")), ("STRUCTURE_WILDCARD", ("SAE:2", "SAE:3")),
                        ("STRUCTURE_WILDCARD", ("SAE:4", "SAE:5"))}
    a = got[("STATE_WILDCARD", ("SAE:0", "SAE:1"))]
    assert a.strict_copy and not a.same_publication and not a.D_raw_identical
    b = got[("STRUCTURE_WILDCARD", ("SAE:2", "SAE:3"))]
    assert b.strict_copy and b.D_raw_identical and not b.decade_value and b.same_location and b.abs_delta_log_D == 0
    c = got[("STRUCTURE_WILDCARD", ("SAE:4", "SAE:5"))]
    assert not c.strict_copy and c.decade_value
    # beyond the value tolerance nothing is paired
    assert len(L.wildcard_copy_pairs(df, tol=1e-6)) == 2          # the 0.2688 / 0.269 pair drops out
