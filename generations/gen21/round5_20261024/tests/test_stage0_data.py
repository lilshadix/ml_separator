"""Tests for the Stage-0 data rules: r5/dupes.py (bit identity, offset / near-identity, quantisation, lab-1
near-copy), r5/population.py (the E-CT sibling population is metadata-only) and r5/supply.py (metadata-only reads,
the frozen mechanism-class map, author parsing, the P4 / PC P4 ledger counts and the supply-audit kill).

Synthetic data, plus label-based construction checks on V1 (which pairs the frozen rule flags; population counts).
No prediction, residual or score is computed.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402,F401
from r5 import dupes as DU  # noqa: E402
from r5 import population as POP  # noqa: E402
from r5 import supply as SU  # noqa: E402

NAN = np.nan


# ------------------------------------------------------------------------------------------------------------
# bit identity
# ------------------------------------------------------------------------------------------------------------
def test_bit_identity_rule_synthetic():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 14)
    Y = np.full((6, 14), NAN)
    Y[0, :7] = base[:7]
    Y[1, :6] = base[:6]
    Y[1, 0:3] += 0.1                                  # 3 of 6 shared identical (50 %) -> flagged
    Y[2, :7] = base[:7]
    Y[2, 3:7] += 0.2                                  # 3 of 7 identical (43 %) -> not flagged
    Y[3, :2] = base[:2]                               # 2 of 2 identical (< 3) -> not flagged
    Y[4, :7] = base[:7] * (1 + 1e-15)                 # float64 inequality -> not identical
    Y[5, 8:14] = base[8:14]                           # no shared metal with cell 0
    bt = DU.bit_identity(Y, [f"c{i}" for i in range(6)], keep_min_ident=1)
    get = {(r.cell_a, r.cell_b): r for r in bt.itertuples(index=False)}
    assert get[("c0", "c1")].flag_bit and get[("c0", "c1")].n_ident == 3 and get[("c0", "c1")].n_shared == 6
    assert not get[("c0", "c2")].flag_bit and get[("c0", "c2")].n_ident == 3
    assert not get[("c0", "c3")].flag_bit and get[("c0", "c3")].n_ident == 2
    assert ("c0", "c4") not in get or get[("c0", "c4")].n_ident == 0 or not get[("c0", "c4")].flag_bit
    assert ("c0", "c5") not in get
    assert DU.flagged_links(bt) == [("c0", "c1")]


@pytest.fixture(scope="module")
def bench():
    from gen20k import data as D
    b = D.load_bench()
    return b.frame.reset_index(drop=True), np.asarray(b.Y, float)


def test_bit_identity_flags_the_known_items(bench):
    f, Y = bench
    ids = f.cell_id.astype(str).tolist()
    bt = DU.bit_identity(Y, ids)
    fl = {frozenset((r.cell_a, r.cell_b)): (r.n_ident, r.n_shared) for r in bt.itertuples(index=False) if r.flag_bit}
    for k, (a, b, ne, ns) in DU.KNOWN_BIT.items():
        assert fl.get(frozenset((a, b))) == (ne, ns), k
    assert len(fl) == len(DU.KNOWN_BIT)               # nothing else, within or across publications
    allp = {frozenset((r.cell_a, r.cell_b)): r for r in bt.itertuples(index=False)}
    for k, (a, b) in DU.KNOWN_NOT_FLAGGED.items():
        r = allp[frozenset((a, b))]
        assert not r.flag_bit and 2 <= r.n_ident <= 5, k
    L2 = POP.v2_lists()["V2-2"]
    for k in ("b9_pair1", "b9_pair2", "b9_pair3"):
        a, b = DU.KNOWN_BIT[k][:2]
        assert L2.get(a) == b or L2.get(b) == a, k    # the three further 8-of-8 pairs are the V2-2 pairs


# ------------------------------------------------------------------------------------------------------------
# offset / near-identity / quantisation / near-copy
# ------------------------------------------------------------------------------------------------------------
def test_offset_and_nearid_scan_synthetic():
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.normal(0, 0.3, 14))
    Y = np.vstack([a, a.copy(), a + 0.004 * rng.normal(size=14), a.copy(), rng.normal(0, 1, 14)])
    Y[1, 6:] = a[6:] - 5.37 + 0.01 * rng.normal(size=8)   # constant shift on Gd-Lu only
    Y[3] = a + 0.5 * np.sin(np.arange(14))                 # not a copy
    off = DU.offset_scan(Y, ["a", "shift", "near", "wave", "rand"])
    r = off[(off.cell_a == "a") & (off.cell_b == "shift")]
    assert len(r) == 1 and r.flag_offset.iloc[0] and abs(r.off_mean.iloc[0] - 5.37) < 0.02   # d = a - shift
    assert r.off_len.iloc[0] >= 8 and r.off_segment.iloc[0].endswith("Lu")
    n = off[(off.cell_a == "a") & (off.cell_b == "near")]
    assert len(n) == 1 and n.flag_nearid.iloc[0] and not n.flag_offset.iloc[0]
    assert off[(off.cell_a == "a") & (off.cell_b == "wave")].empty
    off2 = DU.offset_scan(Y, ["a", "shift", "near", "wave", "rand"], bit_flagged=[("a", "near")])
    assert not off2[(off2.cell_a == "a") & (off2.cell_b == "near")].flag_nearid.any()


def test_quant_scan_synthetic():
    Y = np.full((4, 14), NAN)
    Y[0] = np.log10(np.array([1, 2, 3, 2, 5, 6, 4, 3, 2, 6, 5, 4, 3, 2], float))       # small integer D
    Y[1] = np.log10(np.array([0.3, 0.4, 0.6, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6, 3.1, 3.5, 4.1]))  # one decimal
    Y[2, :6] = [1.134, 1.134, 1.134, 0.5, 0.6, 0.7]                                       # ties
    Y[3, :6] = [0.1234567, 0.2345671, 0.345, 0.456, 0.567, 0.678]
    Y[3, 1] = Y[3, 0]                                                                       # La == Ce
    q = DU.quant_scan(Y, ["int", "dec1", "tie", "adj"]).set_index("cell_id")
    assert q.loc["int", "Q_SMALLINT"] and q.loc["int", "dp_cell"] == 0
    assert q.loc["dec1", "Q_COARSE"] and not q.loc["dec1", "Q_SMALLINT"]
    assert q.loc["tie", "Q_TIES"] and q.loc["tie", "max_tie"] == 3
    assert q.loc["adj", "Q_ADJEQ"] and q.loc["adj", "n_adjeq"] == 1


def test_near_copy_mask_and_medium():
    y = np.array([1.0, 0.5, 0.0, -0.5])
    don = np.array([[1.1, 0.6, -0.1, -0.4], [1.1, 0.6, -0.1, -0.7], [1.1, 0.6, np.nan, -0.4], [1.0, 0.5, 0.0, -0.5]])
    same = np.array([True, True, True, False])
    assert DU.near_copy_mask(y, don, same).tolist() == [True, False, False, False]


def test_metal_conc_duplicates_need_shared_metals():
    f = pd.DataFrame({"cell_id": ["a", "b", "c", "d"], "publication_id": ["p"] * 4, "extractant": ["X", "X", "X", "Y"],
                      "chemotype": ["s"] * 4, "metals": ["La;Ce;Pr", "La;Ce;Pr;Nd", "Gd;Tb;Dy", "La;Ce;Pr"],
                      "cond__acid__hno3": [1, 1, 1, 1], "cond__acid_concentration_M": [3.0] * 4,
                      "cond__contact_time_min": [30.0] * 4, "cond__diluent__n_dodecane": [1, 1, 1, 1],
                      "cond__extractant_concentration_M": [0.1] * 4,
                      "cond__metal_concentration_mM": [0.1, 0.2, 0.3, 0.2], "cond__temperature_C": [25.0] * 4})
    fin = POP.finite_from_metals(f)
    lk = DU.metal_conc_links(f, fin)
    assert len(lk) == 3                               # a-b, a-c, b-c (d is another extractant)
    assert DU.metal_conc_duplicates(f, fin) == [["b", "a"]]   # c shares no metal: split series, not a duplicate


# ------------------------------------------------------------------------------------------------------------
# population: metadata only
# ------------------------------------------------------------------------------------------------------------
def _synthetic_frame():
    rows = []

    def cell(cid, pub, ext, metals, acid, ct=30.0, mconc=0.1, dil=1):
        rows.append({"cell_id": cid, "publication_id": pub, "extractant": ext, "chemotype": "sc1", "metals": metals,
                     "cond__acid__hno3": 1, "cond__acid_concentration_M": acid, "cond__contact_time_min": ct,
                     "cond__diluent__n_dodecane": dil, "cond__diluent__toluene": 1 - dil,
                     "cond__extractant_concentration_M": 0.1, "cond__metal_concentration_mM": mconc,
                     "cond__temperature_C": 25.0})
    full = ";".join(POP.LN)
    cell("t", "p1", "X", "La;Ce;Pr;Nd;Gd", 1.0)
    cell("s1", "p1", "X", full, 2.0)                  # acid differs: CLEAN sibling
    cell("s2", "p1", "X", full, 1.0, ct=60.0)         # contact time only: not CLEAN
    cell("s3", "p1", "X", "La;Ce;Pr", 3.0)            # does not cover Nd, Gd: not a sibling
    cell("s4", "p1", "X", full, 1.0, dil=0)           # diluent differs: CLEAN
    cell("s5", "p1", "Z", full, 1.0)                  # other extractant: placebo plc_pub, not a sibling
    cell("s6", "p2", "X", full, 1.0)                  # other publication, same extractant: plc_xext
    cell("v", "p1", "X", full, 4.0)                   # V2-dropped
    return pd.DataFrame(rows)


def test_population_rules_synthetic():
    f = _synthetic_frame()
    fin = POP.finite_from_metals(f)
    P = POP.build_pools(f, fin, ["t", "s1"], {"v"}, bit_links=[("t", "s4")])
    t = P["t"]
    assert sorted(t["v1"]) == ["s1", "s2", "s4", "v"]
    assert sorted(t["clean"]) == ["s1", "s4"]
    assert t["bit"] == ["s1"] and t["bit_removed"] == ["s4"]      # the link is removed, the cell is kept
    assert "s4" in P["s1"]["clean"]                                # s4 stays a sibling of s1
    assert t["plc_pub"] == ["s5"] and t["plc_xext"] == ["s6"]
    assert t["off_sibs"] == ["s1"] and t["off_all"] == ["s1"]
    assert t["sib_fams"]["s2"] == "contact_time"
    S = POP.strata(P)
    assert "t" in S["PRIMARY_NOBIT"] and "t" not in S["PRIMARY"] and "t" in S["EQ1"]


def test_population_builder_is_metadata_only(bench):
    f, Y = bench
    assert "Y" not in inspect.signature(POP.build_pools).parameters
    with pytest.raises(AssertionError):
        POP.build_pools(f, np.isfinite(Y), ["x"], set())         # a frame with logD__ columns is refused
    meta = POP.metadata_frame(f)
    assert not any(c.startswith(("logD__", "repsd__")) for c in meta.columns)
    fin = POP.finite_from_metals(meta)
    assert np.array_equal(fin, np.isfinite(Y))
    c3 = POP.c3_cells_from_ref()
    v2 = POP.v2drop_set()
    P0 = POP.build_pools(meta, fin, c3, v2)
    # scrambling every V1 value (same NaN pattern) cannot change a metadata-only population
    f2 = f.copy()
    rng = np.random.default_rng(3)
    for k, m in enumerate(POP.LN):
        col = f"logD__{m}"
        f2[col] = np.where(np.isfinite(Y[:, k]), rng.normal(size=len(f2)), np.nan)
    P1 = POP.build_pools(POP.metadata_frame(f2), POP.finite_from_metals(POP.metadata_frame(f2)), c3, v2)
    assert P0 == P1
    S = POP.strata(P0)
    rows = POP.rows_per_seed_from_ref()
    c = POP.counts(S["PRIMARY_NOBIT"], P0, rows)
    assert (c["cells"], c["publications"], c["pub_x_chemotype_units"]) == (99, 12, 14)
    assert c["rows_per_seed"]["BPG"]["distinct"] == [690]
    # the bit-identity rule (the only label-derived step) removes links, not cells
    bt = DU.bit_identity(Y, meta.cell_id.astype(str).tolist())
    Pb = POP.build_pools(meta, fin, c3, v2, DU.flagged_links(bt))
    assert set(Pb) == set(P0)
    nrem = sum(len(p["bit_removed"]) for p in Pb.values())
    assert nrem == sum(len(P0[t]["clean"]) - len(Pb[t]["bit"]) for t in Pb) and nrem > 0


# ------------------------------------------------------------------------------------------------------------
# supply
# ------------------------------------------------------------------------------------------------------------
def test_read_meta_refuses_value_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"doi,log_D\n10.1/x,0.1\n")
    with pytest.raises(AssertionError):
        SU.read_meta(p)
    p.write_bytes(b"doi,logD__La\n10.1/x,0.1\n")
    with pytest.raises(AssertionError):
        SU.read_meta(p)
    p.write_bytes(b"doi,authors,n_ln_metals,data_locations\n10.1/x,A B,3,Table 1\n")
    assert list(SU.read_meta(p).columns) == ["doi", "authors", "n_ln_metals", "data_locations"]


@pytest.mark.parametrize("label,cls,cgd", [
    ("acidic organophosphorus", "CX_ACIDIC_OP", "Y"), ("HDEHP in [C4mim][Tf2N] vs toluene", "CX_ACIDIC_OP", "Y"),
    ("beta-diketone synergist", "CX_CHELATING", "Y"), ("HTTA + TOPO", "CX_CHELATING", "Y"),
    ("8-HQ + crown synergist", "CX_CHELATING", "Y"), ("diglycolamic acid", "CX_CARBOXYLIC", "Y"),
    ("organophosphorus-anion bifunctional IL", "IL_BIFUNCTIONAL", "N"), ("[A336][P507]", "IL_BIFUNCTIONAL", "N"),
    ("quaternary-onium nitrate IL", "IL_QUATERNARY", "N"), ("TOMANO3", "IL_QUATERNARY", "N"),
    ("quaternary ammonium nitrate + TBP", "IL_QUATERNARY", "N"), ("CMPO", "NEUTRAL_CMPO", "N"),
    ("CMPO-like carbamoylphosphine oxide", "NEUTRAL_CMPO", "N"), ("phosphorylurea", "NEUTRAL_CMPO", "N"),
    ("neutral phosphine oxide", "NEUTRAL_OP", "N"), ("pyridine dicarboxamide", "SOFT_N", "N"),
    ("phen-diamide", "SOFT_N", "N"), ("BTBP", "SOFT_N", "N"), ("malonamide", "NEUTRAL_AMIDE", "N"),
    ("DGA", "NEUTRAL_AMIDE", "N"), ("eutectic solvent", "OTHER_EUTECTIC", "N"), ("calixarene", "NEUTRAL_MACROCYCLE", "N"),
    ("crown ether carboxylic acid", "CX_CARBOXYLIC", "Y"), ("aqueous chelator", "NOT_LLE", "N")])
def test_mechanism_class_map(label, cls, cgd):
    m = SU.classify_extractant(label)
    assert (m["mechanism_class"], m["cgd_stratum"]) == (cls, cgd)


def test_mechanism_flags_and_media():
    m = SU.classify_extractant("DGA in quaternary IL")
    assert m["mechanism_class"] == "NEUTRAL_AMIDE" and m["il_diluent_flag"]
    m = SU.classify_extractant("DGA + aqueous BLPhen")
    assert m["mechanism_class"] == "NEUTRAL_AMIDE" and m["aq_complexant_flag"]
    assert SU.classify_medium("acetate/acetic acid; nitrate salts")["gd_medium_eligible"] == "N"
    assert SU.classify_medium("DTPA-nitrate pH")["medium_class"] == "AQ_COMPLEXANT"
    assert SU.classify_medium("HNO3 3 M")["gd_medium_eligible"] == "Y"
    assert SU.classify_medium("0.1 M NaClO4")["medium_class"] == "INERT_SALT"
    assert SU.t_classes("pH and HDEHP concentration x 2 diluents", cx=True) == ["T0", "T2"]
    assert SU.t_classes("pH series", cx=True) == ["T0"]
    assert SU.t_classes("HNO3 x diluent composition", cx=False) == ["T1", "T2"]
    tab = SU.mech_rules_table()
    assert {"extractant", "medium", "sibling_factor", "flag"} <= set(tab.rule_type)
    assert tab[tab.rule_type == "extractant"].rule_id.iloc[-1] == "X99"


def test_author_parsing_repairs():
    P = SU.parse_reference("Sun Guoxin; Liu Min; Cui Yu; Yuan Meilong; Yin Shaohong")
    assert set(P["families"]) == {"sun", "liu", "cui", "yuan", "yin"}            # Family-Given order
    P = SU.parse_reference("Yuji SASAKI; Zhi-Xuan ZHU; Yumi SUGO; Takaumi KIMURA")
    assert set(P["families"]) == {"sasaki", "zhu", "sugo", "kimura"} and P["senior"][0] == "kimura"
    P = SU.parse_reference("E.A. Mowafy, D. Mohamed; E.A. Mowafy; D. Mohamed")
    assert set(P["families"]) == {"mowafy", "mohamed"} and P["senior"] == ("mohamed", "d")
    P = SU.parse_reference("Dr. Andreas Wilden, Dr. Piotr M. Kowalski, Prof. Dr. Giuseppe Modolo")
    assert set(P["families"]) == {"wilden", "kowalski", "modolo"}
    P = SU.parse_reference("SA Ansari, PK Mohapatra, W Verboom")
    assert set(P["families"]) == {"ansari", "mohapatra", "verboom"}
    links = SU.corpus_links({"a": SU.parse_reference("E.A. Mowafy; D. Mohamed"),
                             "b": SU.parse_reference("E. A. Mowafy; H. F. Aly"),
                             "c": SU.parse_reference("Yan Liu; Wei Zhang; Q. Smithers"),
                             "d": SU.parse_reference("Bo Liu; Lei Zhang; R. Jones")})
    pairs = {(r.key_a, r.key_b): r for r in links.itertuples(index=False)}
    assert pairs[("a", "b")].LPI and pairs[("a", "b")].single_pi_only    # single-PI link
    assert ("c", "d") not in pairs                                        # liu + zhang never link


@pytest.fixture(scope="module")
def frozen_candidates():
    cand = SU.read_meta(SU.CAND, dtype=str).fillna("")
    reg = SU.read_meta(SU.REG)
    _, _, parsed = SU.corpus_key(reg)
    return SU.candidates_frozen(cand, parsed)


def test_ledger_reproduces_p4_and_pc_p4(frozen_candidates):
    c = frozen_candidates
    n = lambda t, e, l: len(SU.new_labs(c, t, e, l))  # noqa: E731
    assert n({"E1"}, "eligibility_P4", "lab_in_corpus_P4") == 7
    assert n({"E1", "E2"}, "eligibility_P4", "lab_in_corpus_P4") == 16
    assert n({"E1", "E2", "U"}, "eligibility_P4", "lab_in_corpus_P4") == 24
    assert n({"E1"}, "eligibility_frozen", "lab_in_corpus_frozen") == 5
    assert n({"E1", "E2"}, "eligibility_frozen", "lab_in_corpus_frozen") == 11
    assert abs(SU.expected_yield(c, "eligibility_P4", "lab_in_corpus_P4", 0.5, 0.25) - 13.5) < 1e-9
    assert abs(SU.expected_yield(c, "eligibility_frozen", "lab_in_corpus_frozen", 2 / 6, 0.0) - 7.0) < 1e-9


def test_supply_audit_kill(frozen_candidates):
    """D4 (s0-statistics-3, s0-fidelity-5): decision supply = in hand (+ SAFE max); C-ECT in linked chemotype units;
    the kill depends only on the claims with role 'test'; the brief's bracket is reported apart."""
    cs = SU.claim_supply(frozen_candidates)
    a = SU.supply_audit(cs, {"pc": SU.expected_claims(cs, 2 / 6, 0.0)}, cand=frozen_candidates)
    assert a["kill_fires"] and all(r["below_min"] for r in a["claims"])
    by = {r["claim"]: r for r in a["claims"]}
    assert by["C-ECT"]["min_G"] == 7 and by["C-ECT"]["unit"].startswith("chemotype units")
    assert all(r["decision_supply"] == r["supply_in_hand"] for r in a["claims"])
    assert [by[c]["decision_supply"] for c in ("C-ECT", "C-GD")] == [2, 1]
    assert by["C-ECT"]["supply_in_hand_laboratories"] == 2 and by["C-ECT"]["supply_all_convert"] == 6
    assert all(r["bracket_high"] == max(r["supply_in_hand"], r["realistic"][1]) for r in a["claims"])
    assert {r["claim"] for r in a["claims"] if r["role"] == "test"} == {"C-ECT"}
    # the kill reads the test claims only: a descriptive claim reaching its minimum does not lift it
    fake = {**cs, "G_HS": {**cs["G_HS"], "verified": [f"L{i}" for i in range(9)]}}
    b = SU.supply_audit(fake, {"pc": SU.expected_claims(cs, 2 / 6, 0.0)}, cand=frozen_candidates)
    assert b["kill_fires"] and not {r["claim"]: r for r in b["claims"]}["C-GD"]["below_min"]
    with pytest.raises(ValueError):
        SU.supply_audit(cs, {"pc": SU.expected_claims(cs, 2 / 6, 0.0)})
