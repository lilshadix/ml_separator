"""Tests for r5/ekx.py (Stage-1 item 1.8, M07 E-KX): synthetic data or construction-only checks; no score is computed
on real labels (real data enter only as metadata counts, identity checks of the round-4 parquet, or with every label
replaced by seeded noise).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ekx.py -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest

Covered: the Nd chord (zero on a radius-linear series, recovers a dip); shrinkage and publication-first means (empty
pools, weighting, index-order independence of the result); the Nd point potential (reads exact, additive form equals
r4.refs.predict_levels, dphi signs); the frozen external object on the real metadata (n = 12 sc009 publications with
Pr/Nd/Sm, w = 12/17); donor isolation for every target and configuration and the guard's positive controls; the
round-4 identities on the REF rows (B1 from the fold betas, B0, y, row counts; construction only); arm construction on
a noise surrogate (arms equal B1 where they must); the unit-level chemotype permutation; perm_eval against
r5.common; the top-3 report rule on a synthetic frame; the external rider's decision logic; the intervention on a
noise surrogate (passes, with positive controls caught); the frozen value's provenance wording (the V1 means never
move it), the consistency lines (a failed line is reported; empty records tolerated), the BPG SENSITIVITY label and
the E-KX-LOPO note on arm / contrast rows, and the per-item table's extra sections (every rule row; the lines).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402,F401
from r5 import common as CM  # noqa: E402
from r5 import ekx as E  # noqa: E402
from r3 import interp  # noqa: E402

ALL = tuple(range(14))
PAIRS = tuple((a, b) for a, b in zip(ALL[:-1], ALL[1:]) if (a, b) != (3, 4))       # Nd/Sm is not adjacent
X = np.asarray(interp._coords("R"), float)


def radius_linear(a=0.3, b=2.0):
    return a + b * (X - X.mean())


# ------------------------------------------------------------------------------------------------------------
# pure helpers
# ------------------------------------------------------------------------------------------------------------
def test_nd_chord_zero_on_radius_linear_series_and_recovers_a_dip():
    y = radius_linear()
    assert abs(E.nd_chord(y)[0]) < 1e-12
    y2 = y.copy()
    y2[E.ND] -= 0.11
    assert abs(E.nd_chord(y2)[0] + 0.11) < 1e-12
    y3 = y.copy()
    y3[E.SM] = np.nan
    assert np.isnan(E.nd_chord(y3)[0])
    assert 0.0 < E.W_SM < 1.0


def test_shrink_and_publication_first_means():
    assert E.shrink(12) == pytest.approx(12 / 17)
    assert E.shrink(0) == 0.0
    pcode = np.array([0, 0, 0, 1, 2, 2])
    v = np.array([1.0, 1.0, 4.0, 3.0, np.nan, 0.0])
    raw, n, m = E.pub_means(v, np.arange(6), pcode, 3)
    assert (n, m) == (3, 5)
    assert raw == pytest.approx(np.mean([2.0, 3.0, 0.0]))
    rawc, nc, mc = E.pub_means(v, np.arange(6), pcode, 3, "cell")
    assert (nc, mc) == (3, 5) and rawc == pytest.approx(np.nanmean(v))
    assert np.isnan(E.pub_means(v, np.array([4]), pcode, 3)[0]) and E.pub_means(v, np.array([], int), pcode, 3)[1] == 0
    full = E.pub_means(v, np.arange(6), pcode, 3, full=True)
    assert full[3] == pytest.approx(np.std([2.0, 3.0, 0.0], ddof=1))
    a = E.pub_means(v, np.array([0, 1, 2, 3]), pcode, 3)
    b = E.pub_means(v, np.array([0, 1, 2, 3, 4]), pcode, 3)          # a NaN donor changes nothing, bit for bit
    assert a == b


def test_nd_potential_reads_exact_and_additive_form_matches_predict_levels():
    rng = np.random.default_rng(3)
    y = radius_linear() + rng.normal(0.0, 0.2, 14)
    for read in (frozenset({0, 6, 13}), frozenset({0, 3, 9, 13}), frozenset({2, 6, 10}), frozenset({1, 4, 8, 12})):
        ym = E.masked(y, read)
        b1 = E.levels(E.b1_fit(-0.09), ym, ALL)
        kx = E.levels(E.kx_fit(-0.09, -0.08), ym, ALL)
        for m in read:
            assert kx[m] == y[m] and b1[m] == y[m]
        pairs = [p for p in PAIRS if not (p[0] in read and p[1] in read)]
        d = E.dphi_pairs(read, ALL, pairs, E.ND)
        assert np.max(np.abs(E.pair_sf(kx, pairs) - (E.pair_sf(b1, pairs) - 0.08 * d))) < 1e-12
        dg = E.dphi_pairs(read, ALL, pairs, E.GD)
        b0 = E.pair_sf(interp.pwl_levels(ym, ALL, "R"), pairs)
        assert np.max(np.abs(b0 - 0.09 * dg - E.pair_sf(b1, pairs))) < 1e-12


def test_dphi_nd_signs():
    pr_nd = [(E.PR, E.ND)]
    # Nd unread, Pr read: the Pr/Nd SF moves by -u
    assert E.dphi_pairs(frozenset({0, E.PR, 9, 13}), ALL, pr_nd, E.ND)[0] == -1.0
    # Nd read, Pr unread: Pr interpolates from Nd, so the potential reaches the Ce/Pr and Pr/Nd rows through -w_Nd(Pr)
    d = E.dphi_pairs(frozenset({0, E.ND, 9, 13}), ALL, [(1, 2), (2, 3)], E.ND)
    assert d[0] != 0.0 and d[1] != 0.0
    # a cell without Nd: no effect anywhere
    mets = tuple(m for m in ALL if m != E.ND)
    pairs = [p for p in PAIRS if E.ND not in p]
    assert not np.any(E.dphi_pairs(frozenset({0, 6, 13}), mets, pairs, E.ND))


# ------------------------------------------------------------------------------------------------------------
# real metadata and the round-4 rows (construction only)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def donors():
    return E.Donors(log=lambda *a: None)


@pytest.fixture(scope="module")
def rows_bp(donors):
    return E.build_rows(donors, "BP", log=lambda *a: None)


@pytest.fixture(scope="module")
def noise(donors):
    rng = np.random.default_rng(11)
    return np.where(np.isfinite(donors.Y), rng.normal(0.0, 1.0, donors.Y.shape), np.nan)


def test_frozen_external_object_is_metadata_only(donors):
    ext = E.frozen_external(donors)
    assert ext["n_donor_publications"] == 12
    assert ext["w"] == pytest.approx(12 / 17) and ext["u_ext"] == pytest.approx(12 / 17 * -0.11)
    assert ext["u_raw"] == E.U_RAW == -0.11
    # counted from the metals metadata: identical when every log D is replaced
    d2 = E.Donors.__new__(E.Donors)
    d2.__dict__.update(donors.__dict__)
    d2.Y = np.where(np.isfinite(donors.Y), 0.0, np.nan)
    assert E.frozen_external(d2)["donor_publications"] == ext["donor_publications"]


def test_frozen_value_provenance_never_moves_the_frozen_value(donors):
    ext = E.frozen_external(donors)
    sc = {"u_raw": -0.12190, "u_raw_cell_weighted": -0.11048, "consistent_2dp": False}
    ext2 = E.frozen_external(donors, sc=sc)
    assert ext2["u_raw"] == ext["u_raw"] == E.U_RAW and ext2["u_ext"] == ext["u_ext"]
    assert "publication means first" not in ext["definition"] and E.U_RAW_SOURCE in ext["definition"]
    assert "V1 publication-first mean -0.122, cell-weighted -0.110 (computed" in ext2["definition"]
    assert "FAILED" in ext2["definition"]


def test_consistency_lines_report_a_failed_line_and_tolerate_empty_records():
    cons = {"BPG": {"sc009_prnd_rows_per_seed": {1: 46, 2: 46}, "seen": 46, "match": True},
            "BP": {"sc009_prnd_rows_per_seed": {1: 45, 2: 46}, "seen": 46, "match": False},
            "frozen_value_source": {"u_raw": -0.1219, "u_raw_cell_weighted": -0.1105, "consistent_2dp": False,
                                    "n_negative": 11, "n_publications": 12}}
    t = E.consistency_table(cons)
    assert list(t.result[:2]) == ["match", "MISMATCH"] and t.result.iloc[2].startswith("FAILED")
    assert t.observed.iloc[2] == "-0.1219" and t.result.iloc[3] == "match" and t.result.iloc[4] == "descriptive"
    assert len(E.consistency_table(cons, designs=("BP",))) == 4
    ok = {**cons, "frozen_value_source": {**cons["frozen_value_source"], "u_raw": -0.112, "consistent_2dp": True}}
    assert E.consistency_table(ok).result.iloc[2].startswith("consistent")
    assert E.consistency_table({"BPG": {}, "frozen_value_source": {}}).empty      # the gate test's stub records


def test_donor_isolation_every_target_and_configuration(donors):
    for t in donors.c3_ids:
        i = donors.pos[t]
        for c5, var in E.CONFIGS:
            for lab in (None, donors.chem):
                idx = donors.donor_idx(i, variant=var, c5=c5, lab=lab)
                assert i not in set(idx)
                assert not ((donors.pub[idx] == donors.pub[i]) | (donors.cg[idx] == donors.cg[i])).any()
                if var != "NB":
                    assert not (set(idx.tolist()) & donors.bit[i])
                if c5 == "C5_out":
                    assert not donors.c5pub[idx].any()
                if c5 == "C5_cell_out":
                    assert not donors.c5cell[idx].any()
                if lab is not None:
                    assert (donors.chem[idx] == donors.chem[i]).all()


def test_guard_positive_controls(donors):
    t = donors.c3_ids[0]
    i = donors.pos[t]
    idx = donors.donor_idx(i)
    with pytest.raises(E.LeakageError):
        donors.guard(i, np.append(idx, i))
    same = np.flatnonzero((donors.pub == donors.pub[i]) & (np.arange(donors.N) != i))
    if len(same):
        with pytest.raises(E.LeakageError):
            donors.guard(i, np.append(idx, same[0]))
    for a, b in donors.bit_links:                       # a bit-identical copy in another copy group is refused
        ia, ib = donors.pos[a], donors.pos[b]
        if donors.cg[ia] != donors.cg[ib]:
            with pytest.raises(E.LeakageError):
                donors.guard(ia, np.array([ib]))
            donors.guard(ia, np.array([ib]), "NB")
            break


def test_round4_identities_on_ref_rows(rows_bp, donors):
    R, ck = rows_bp
    assert ck["b1_predict_levels_maxabs"] <= E.B1_TOL and ck["b0_maxabs"] <= E.B1_TOL and ck["y_maxabs"] == 0.0
    assert ck["b1_additive_maxabs"] <= E.ADD_TOL and ck["kx_additive_identity_maxabs"] <= E.ADD_TOL
    assert set(ck["rows_per_seed"].values()) == {E.ROWS_PER_SEED} and ck["cells"] == len(donors.c3_ids) == 229
    assert ck["both_ends_read_rows"] == 0 and ck["adjacent_lighter_first"]
    assert ck["dphi_nd_on_cells_without_nd"] == 0
    assert set(ck["sc009_prnd_rows_per_seed"].values()) == {46}         # the seen count of IR M07 (a count)


def test_arms_on_noise_surrogate(rows_bp, donors, noise):
    R = rows_bp[0].copy()
    rng = np.random.default_rng(5)
    for c in ("y", "B1", "B0"):
        R[c] = rng.normal(0.0, 0.3, len(R))
    TV = E.all_target_values(donors, donors.c3_ids, Y=noise)
    betas = E.fold_betas("BP")
    OFF = {c5: E.offsets(donors, R, betas=betas, Y=noise, c5=c5) for c5 in E.C5_VARIANTS}
    E.add_arms(R, TV, OFF, -0.07)
    x = E.sfx("C5_out")
    b1 = R.B1.to_numpy()
    flat = R.dphi_Nd.to_numpy() == 0
    for a in ("KX_CHEM", "KX_SC009", "NDU", "KX_FROZEN"):
        assert np.array_equal(R[a + x].to_numpy()[flat], b1[flat])
    assert np.array_equal(R["KX_SC009" + x].to_numpy()[~R.is_sc009.to_numpy()], b1[~R.is_sc009.to_numpy()])
    sc = R.is_sc009.to_numpy()
    assert np.array_equal(R["KX_SC009" + x].to_numpy()[sc], R["KX_CHEM" + x].to_numpy()[sc])
    npn = ~R.is_prnd.to_numpy()
    for a in ("PRND_OFFSET", "PRND_COPY"):
        assert np.array_equal(R[a + x].to_numpy()[npn], b1[npn])
    tv = TV[("C5_out", "PRIMARY")]
    assert (tv.loc[tv.n_chem_pub == 0, "u_chem"] == 0).all()
    assert np.allclose(tv.u_chem, E.shrink(tv.n_chem_pub.to_numpy(float)) * np.nan_to_num(tv.u_chem_raw))
    V = E.view(R, endpoint="SC009_PRND")
    assert len(V) == 46 * 5 and set(E.ARMS) <= set(V.columns)


def test_permutation_keeps_units_per_chemotype_and_is_seeded(donors):
    labs, seed = E.perm_labels(donors, 5)
    labs2, seed2 = E.perm_labels(donors, 5)
    assert seed == seed2 and np.array_equal(labs, labs2)
    want = np.bincount(donors.unit_chem, minlength=len(donors.chem_names))
    for lab in labs:
        ul = pd.Series(lab).groupby(donors.unit_code).nunique()
        assert (ul == 1).all()                                             # one label per unit
        got = np.bincount(pd.Series(lab).groupby(donors.unit_code).first().to_numpy(), minlength=len(want))
        assert np.array_equal(got, want)
    assert not np.array_equal(labs[0], donors.chem_code)


def test_perm_eval_matches_common():
    rng = np.random.default_rng(7)
    n = 300
    V = pd.DataFrame({"split_seed": np.repeat([1, 2, 3], n // 3), "y": rng.normal(0, 1, n), "B1": rng.normal(0, 1, n),
                      "dphi_Nd": rng.choice([0.0, -1.0, 0.4], n), "tix": rng.integers(0, 20, n),
                      "is_sc009": rng.random(n) < 0.5})
    u = rng.normal(0, 0.2, (4, 20))
    got = E.perm_eval(V, u, sc_only=True, chunk=3)
    for b in range(4):
        W = V.copy()
        W["A1"] = W.B1 + u[b][W.tix] * W.dphi_Nd * W.is_sc009
        assert got[b] == pytest.approx(CM.r2_seedmean(W, "A1") - CM.r2_seedmean(W, "B1"), abs=1e-12)


def test_top3_rule_on_synthetic_frame():
    rng = np.random.default_rng(9)
    rows = []
    for s in (1, 2):
        for p in range(8):
            for _ in range(10):
                y = rng.normal(0, 1)
                b1 = y + rng.normal(0, 0.5)
                ndu = b1 + 0.1 * (y - b1)                                  # a small universal gain everywhere
                kx = y + (0.05 * rng.normal()) if p < 3 else b1           # the chemotype gain lives in 3 publications
                rows.append({"split_seed": s, "publication": f"p{p}", "y": y, "B1": b1, "NDU": ndu, "KX_CHEM": kx})
    V = pd.DataFrame(rows)
    top, share, _ = E._top3(V, "KX_CHEM")
    assert sorted(top) == ["p0", "p1", "p2"] and share == pytest.approx(1.0)
    r = E.rule_row(V, cand="KX_CHEM", design="BP", c5="C5_out", endpoint="POOLED", removal="rows", top=top,
                   share=share, primary=True)
    assert r["narrowed"] and r["rule_stat"] < 0 and r["label"].startswith(E.CC)
    s = E.rule_row(V, cand="KX_CHEM", design="BP", c5="C5_out", endpoint="POOLED", removal="rows", top=["p7"],
                   share=share, primary=False)
    assert not s["narrowed"]
    rule = pd.DataFrame([r, s])
    note = E.rule_note(rule, "BP")
    assert "BP decides" in note and "Narrowed in 1 of 2 rows; not narrowed: C5_out POOLED rows (+" in note
    assert "every row is a rule sensitivity" in E.rule_note(rule, "BPG")


def test_bpg_label_and_the_estimand_note_on_arm_and_contrast_rows():
    assert E.design_label("BP") == E.SEEN and E.design_label("BPG") == f"{E.SENS}; {E.SEEN}"
    rng = np.random.default_rng(19)
    n = 60
    V = pd.DataFrame({"split_seed": np.repeat([1, 2], n // 2), "cell_id": [f"c{k % 12}" for k in range(n)],
                      "publication": [f"p{k % 4}" for k in range(n)], "y": rng.normal(0, 1, n)})
    V["B1"] = V.y + rng.normal(0, 0.5, n)
    V["KX_CHEM"] = V.y + rng.normal(0, 0.4, n)
    meta = {"design": "BP", "c5": "C5_in", "variant": "PRIMARY", "endpoint": "POOLED"}
    rows = E.arm_rows(V, ("B1", "KX_CHEM"), meta, "BP", balanced=False)
    cons = E.contrast_rows(V, (("KX_CHEM", "B1"),), meta, "BP", balanced=False)
    for x in rows + cons:
        assert x["label"] == f"{E.SENS}; {E.SEEN}; {E.NOT_NEXT_TO}"
    assert list(E._bare(pd.DataFrame(rows)).label) == [f"{E.SENS}; {E.SEEN}"] * 2


def test_per_item_table_writes_the_extra_sections(tmp_path):
    item = {k: "n/a" for k, _ in CM.ITEM_FIELDS}
    item["label"] = E.design_label("BPG")
    item["rule"] = pd.DataFrame({"role": ["rule sensitivity"], "rule_stat": [0.0057], "narrowed": [False]})
    item["rule_note"] = "BP decides."
    item["consistency"] = pd.DataFrame({"line": ["x"], "result": ["FAILED"]})
    item["consistency_note"] = "lines."
    p = tmp_path / "t.md"
    text = E.per_item_table(item, p, title="t")
    assert p.read_bytes().decode("utf-8") == text and b"\r" not in p.read_bytes()
    assert text.startswith(CM.per_item_table(item, None, title="t"))
    assert "## Report rule: every rule row\n\nBP decides.\n\n| role | rule_stat | narrowed |" in text
    assert "| rule sensitivity | 0.0057 | False |" in text and "## Consistency lines (do not stop the item)" in text
    assert "**Label: SENSITIVITY; EXPLORATORY (seen)**" in text


def _cell(rng, dip, read=(0, E.PR, 9, 13)):
    y = radius_linear() + rng.normal(0.0, 0.01, 14)
    y[E.ND] += dip
    return {"y": y, "read_set": read}


def test_external_rider_logic():
    rng = np.random.default_rng(13)
    good = [{"publication": f"n{k}", "cells": [_cell(rng, -0.12), _cell(rng, -0.10)]} for k in range(4)]
    r = E.external_rider(good, beta=-0.09, u_nd=-0.08)
    assert r["verdict"] == "NOT KILLED" and r["n_falls"] == 4 and r["n_chord_nonneg"] == 0
    bad = good[:2] + [{"publication": f"m{k}", "cells": [_cell(rng, +0.05)]} for k in range(2)]
    assert E.external_rider(bad, beta=-0.09, u_nd=-0.08)["verdict"] == "KILLED"
    assert E.external_rider(good[:3], beta=-0.09, u_nd=-0.08)["verdict"] == "PENDING"
    noprsm = [{"publication": "x", "cells": [{"y": np.where(np.arange(14) == E.SM, np.nan, radius_linear()),
                                              "read_set": (0, 9, 13)}]}]
    r2 = E.external_rider(noprsm + good, beta=-0.09, u_nd=-0.08)
    assert r2["publications"][0]["eligible"] is False and r2["verdict"] == "NOT KILLED"


def test_intervention_on_noise_surrogate(rows_bp, donors, noise):
    R = rows_bp[0].copy()
    R["B0"] = np.random.default_rng(17).normal(0.0, 0.3, len(R))
    some = list(dict.fromkeys(donors.c3_ids[:4] + [t for t in donors.c3_ids if donors.sc[donors.pos[t]]][:4]))
    iv = E.intervention(donors, R, "BP", betas=E.fold_betas("BP"), Y=noise, targets=some)
    assert iv["passed"] and iv["n_mismatches"] == 0 and iv["pc_self_guard_caught"] == len(some)
    assert iv["pc_leaky_caught"] == iv["pc_leaky_applicable"]
