"""Tests for r5/elab.py (Stage-1 item 1.3, E-LAB lab-1): synthetic data or construction-only checks; no score is
computed on real labels (real data enter only as metadata counts, the frozen duplicate links, guard behaviour, or with
every label replaced by seeded noise).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_elab.py -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest

Covered: the quadratic detrend; the shape-vector interpolation (exact at the reads; equals r3.interp.pwl_levels and
r4.refs.predict_levels in B1's form); level-matched and unmatched shapes (no donor -> B1, w = n / (n + 6.1), the 1.0
tolerance, the mask override); the controls' draw functions (E_RM equals MX_LM10 when level matching keeps every donor;
E_PCM's count matching and weights; E_PBC's chemotype-matched replacement from the pool only); Skeptic A's offset sd;
the statistics on identical rows (expected-error columns, seed means, pooled convention, unit statistics against
r5.common, leave-two-out against brute force); the global-K choice (ties, LOPO independence of the left-out labels);
labels and variant frames; the A2 table wording (unit named in W/L/T, seen pools, BP seen values and SENSITIVITY,
fixed-K rows not [O], the exact-enumeration SF_POOL p, the headline-identity and C5-change columns) on synthetic tables;
the RNG scheme; and, on the real metadata, the POPX counts, the frozen bit link of I2 item
4 removed from the clean pool, the lab key coverage, the guard's positive controls and a single-target intervention on
a noise surrogate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402,F401
from r5 import common as CM  # noqa: E402
from r5 import elab as E  # noqa: E402
from r3 import interp  # noqa: E402
from r4 import refs as RF  # noqa: E402

X = E.X_R
ALL = list(range(14))


def quad(a=0.2, b=-1.0, c=0.5):
    return a + b * X + c * X ** 2


# ------------------------------------------------------------------------------------------------------------
# pure core
# ------------------------------------------------------------------------------------------------------------
def test_qdev_zero_on_a_quadratic_and_nan_below_six_metals():
    assert np.nanmax(np.abs(E.qdev(quad()))) < 1e-10
    y = quad()
    y[6] += 0.3
    d = E.qdev(y)
    assert d[6] > 0.2 and np.isfinite(d).all()
    y5 = np.full(14, np.nan)
    y5[:5] = 1.0
    assert np.isnan(E.qdev(y5)).all()


def test_interp_levels_exact_at_reads_and_matches_round4_b1():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 14)
    reads, metals = [0, 3, 6, 12], ALL
    L0 = E.interp_levels(y, reads, metals, np.zeros(14))
    ym = np.full(14, np.nan)
    ym[reads] = y[reads]
    assert np.allclose(L0, interp.pwl_levels(ym, metals, "R"), atol=1e-14)
    u = np.zeros(14)
    u[E.GD] = -0.07
    L1 = E.interp_levels(y, reads, metals, u)
    assert np.array_equal(L1[reads], y[reads])
    Lp = RF.predict_levels({"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([-0.07])}, SimpleNamespace(y_masked=ym), metals)
    assert np.allclose(L1, Lp, atol=1e-14)


def test_lm_shape_no_donor_weights_tolerance_and_mask():
    Dv = np.zeros((4, 14))
    Dv[:, 2] = [0.1, 0.3, 0.5, np.nan]
    Y = np.zeros((4, 14))
    Y[:, 2] = [0.0, 0.5, 2.0, 0.0]
    lv = np.zeros(14)
    c, w = E.lm_shape(Dv, Y, [], lv)
    assert not c.any() and not w.any()
    c, w = E.lm_shape(Dv, Y, [0, 1, 2, 3], lv)
    assert c[2] == pytest.approx(0.2) and w[2] == pytest.approx(2 / 8.1)          # donor 2 is > 1 log away, 3 is NaN
    c, w = E.lm_shape(Dv, Y, [0, 1, 2, 3], lv, tol=2.0)
    assert c[2] == pytest.approx(0.3) and w[2] == pytest.approx(3 / 9.1)
    msk = np.zeros((4, 14), bool)
    msk[2, 2] = True
    c, w = E.lm_shape(Dv, Y, [0, 1, 2, 3], lv, mask=msk)
    assert c[2] == pytest.approx(0.5) and w[2] == pytest.approx(1 / 7.1)
    cu, wu = E.shape_all(Dv, [0, 1, 2, 3], 12.0)
    ci, wi = E.lm_shape(Dv, Y, [0, 1, 2, 3], lv, tol=np.inf, K=12.0)
    assert np.allclose(cu, ci) and np.allclose(wu, wi) and wu[2] == pytest.approx(3 / 15)
    u = np.zeros(14)
    u[E.GD] = -0.1
    assert np.array_equal(E.blend(u, np.ones(14), np.zeros(14)), u)
    assert E.blend(u, np.ones(14), np.full(14, 0.5), sub=[2])[2] == pytest.approx(0.5)
    assert E.blend(u, np.ones(14), np.full(14, 0.5), sub=[2])[E.GD] == pytest.approx(-0.1)


def test_e_rm_equals_mx_lm10_when_every_donor_is_level_matched():
    rng = np.random.default_rng(2)
    Dv = rng.normal(0, 0.1, (6, 14))
    Y = rng.normal(0, 0.2, (6, 14))
    lv = np.zeros(14)
    ref = E.lm_shape(Dv, Y, np.arange(6), lv)
    for c, w in E.draws_rm(np.random.default_rng(0), Dv, Y, np.arange(6), lv, ndraw=4):
        assert np.allclose(c, ref[0]) and np.allclose(w, ref[1])
    assert len(E.draws_rm(np.random.default_rng(0), Dv, Y, [], lv, ndraw=3)) == 3


def test_e_pcm_count_matching_and_e_pbc_pool_only():
    rng = np.random.default_rng(3)
    Dv = rng.normal(0, 0.1, (40, 14))
    Y = np.zeros((40, 14))
    Y[30:, :] = 5.0                                          # cells 30-39 never level-match
    nm = np.zeros(14, int)
    nm[[1, 5]] = [2, 50]
    out = E.draws_pcm(np.random.default_rng(0), Dv, Y, nm, np.zeros(14), np.arange(40), ndraw=3)
    for c, w in out:
        assert w[1] == pytest.approx(2 / 8.1) and w[5] == pytest.approx(30 / 36.1) and w[0] == 0
    assert all(not w.any() for _, w in E.draws_pcm(np.random.default_rng(0), Dv, Y, nm, np.zeros(14), np.zeros(0, int), ndraw=2))
    chem = np.array(["a"] * 20 + ["b"] * 20)
    pool = np.arange(10, 40)
    seen = []
    rng2 = np.random.default_rng(5)
    for _ in range(20):
        rep = []
        for j in [0, 25]:
            pa = np.setdiff1d(pool, rep)
            pc = pa[chem[pa] == chem[j]]
            rep.append(int(rng2.choice(pc if len(pc) else pa)))
        seen.append(rep)
    assert all(10 <= r[0] < 20 and 20 <= r[1] < 40 for r in seen)
    assert len(E.draws_pbc(np.random.default_rng(0), Dv, Y, [0, 25], np.zeros(14), chem, pool, ndraw=5)) == 5


def test_offsd():
    a = quad()
    assert E.offsd(a, a + 0.7) == pytest.approx(0.0, abs=1e-12)
    b = np.full(14, np.nan)
    b[:5] = 0.0
    assert E.offsd(a, b) == np.inf


# ------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------
def synth_frame(n_pub=5, cells=3, pairs=4, seeds=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seeds:
        for p in range(n_pub):
            for c in range(cells):
                for q in range(pairs):
                    y = rng.normal(0, 1)
                    rows.append({"split_seed": s, "publication": f"p{p}", "cell_id": f"p{p}c{c}", "extractant": f"e{c}",
                                 "chemotype": f"h{c % 2}", "y": y, "B1": y + rng.normal(0, 0.5), "MX_LM10": y + rng.normal(0, 0.4),
                                 "E_PCM": rng.uniform(0.1, 0.5), "lab_cons": f"L{p // 2}", "in_PRIMARY": True})
    d = pd.DataFrame(rows)
    for k, a in zip(E.K_GRID, E.K_ARMS):
        d[a] = d.y + rng.normal(0, 0.3 + 0.01 * k, len(d))
    return d


def test_seed_mean_statistics_and_expected_error_columns():
    d = synth_frame()
    man = []
    for _, g in d.groupby("split_seed"):
        man.append(1 - ((g.MX_LM10 - g.y) ** 2).sum() / ((g.B1 - g.y) ** 2).sum())
    assert E.relsse(d, "MX_LM10") == pytest.approx(np.mean(man))
    assert np.array_equal(E.err2(d, "E_PCM"), d.E_PCM.to_numpy())
    r2m = np.mean([1 - ((g.B1 - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum() for _, g in d.groupby("split_seed")])
    assert E.r2(d, "B1") == pytest.approx(r2m)
    assert E.relsse_pooled(d, "MX_LM10") == pytest.approx(1 - ((d.MX_LM10 - d.y) ** 2).sum() / ((d.B1 - d.y) ** 2).sum())
    us = E.unit_stats(d, "MX_LM10", "B1", "publication")
    uc = CM.unit_stats(d, "MX_LM10", "B1", "publication")
    assert np.allclose(us[["n", "SSE_ref", "SSE_cand", "dSSE", "L", "rel"]].to_numpy(),
                       uc[["n", "SSE_ref", "SSE_cand", "dSSE", "L", "rel"]].to_numpy())
    r = E.infer(d, "MX_LM10", "B1", "publication", "TEST|ELAB")
    assert r["G"] == 5 and r["SF_POOL_exact"]


def test_leave_two_out_matches_brute_force():
    d = synth_frame()
    l2 = E.leave_out(d, "MX_LM10", "B1", 2)
    assert len(l2) == 10
    for row in l2.itertuples(index=False):
        q = row.left_out.split(";")
        assert row.rel == pytest.approx(E.relsse(d[~d.publication.isin(q)], "MX_LM10", "B1"))


def test_select_k_ties_and_lopo_independence():
    d = synth_frame()
    d[E.K_ARMS[1]] = d[E.K_ARMS[0]]
    ks = E.select_k(d)
    assert len(ks) in (1, 2) and (len(ks) == 1 or set(ks) == set(E.K_ARMS[:2]))
    choice = E.gk_lopo(d, E.PRIMARY_C5)
    assert "MX_GK_LOPO" in d and d.MX_GK_LOPO.notna().all() and len(choice) == 5
    gp = E.gk_poison_check(d)
    assert gp["passed"] and gp["pc_runs"] == 5 and gp["pc_caught"] > 0 and not gp["changed"]
    dd, ko = E.with_oracle(d)
    assert np.allclose(dd.MX_GK_O, d[ko].to_numpy().mean(1))


def test_labels_and_variant_frames():
    lb = E.label(design="BPG", c5="C5_in", stratum="PRIMARY", arms=("MX_LM10", "B1"))
    assert lb.startswith(E.SEEN) and E.BEST25 in lb and E.SENS not in lb
    assert E.label(design="BP", c5="C5_in", stratum="PRIMARY", arms=("MX_LM10",)).count(E.SENS) == 1
    assert E.label(design="BPG", c5="C5_in", stratum="PRIMARY", arms=("MX_GK_O",)).startswith(E.ORACLE)
    assert E.label(design="BPG", c5="C5_in", stratum="SM_DM", arms=("MX_LM10",)).startswith(E.DIAG)
    assert any(x in E.label(design="BPG", c5="C5_out", stratum="PRIMARY") for x in CM.LABELS)
    R = pd.DataFrame({"publication": ["a", paths.C5], "cell_id": ["x", paths.C5_CELL], "MX_LM10": [1.0, 2.0],
                      "MX_LM10@C5_out": [3.0, 4.0], "MX_LM10@C5_cell_out": [5.0, 6.0]})
    v = E.vframe(R, "C5_out")
    assert list(v.MX_LM10) == [3.0] and "MX_LM10@C5_out" not in v
    assert list(E.vframe(R, "C5_cell_out").MX_LM10) == [5.0]
    assert E.vframe(R, "C5_in") is R


def test_label_options_rows_label_and_seen_text():
    kw = dict(c5="C5_in", stratum="PRIMARY", arms=("MX_LM10", "B1"))
    assert E.label(design="BPG", **kw) == f"{E.SEEN}; {E.BEST25}"
    assert E.label(design="BPG", lead=(E.CC,), **kw) == f"{E.CC}; {E.SEEN}; {E.BEST25}"
    assert E.label(design="BPG", extra=("bridge",), **kw) == f"{E.SEEN}; bridge; {E.BEST25}"
    assert E.label(design="BP", extra=("bridge",), **kw) == f"{E.SEEN}; bridge; {E.SENS}; {E.BEST25}"
    assert E.label(design="BP", sens=True, **kw).count(E.SENS) == 1
    assert E.label(design="BPG", seen_note="x", **kw).startswith(f"{E.SEEN} (x); ")
    assert E.SENS in E.rows_label("BP") and E.rows_label("BP").startswith(f"{E.SEEN}; {E.SENS}")
    assert E.rows_label("BPG").startswith(f"{E.SEEN}; {E.BEST25}") and "@C5_out" in E.rows_label("BPG")
    s = E.seen_text("BPG", "POPX0_rel")
    assert s.startswith(E.SEEN_VALUES["POPX0_rel"]) and E.SEEN_POOL["src"] in s and "BPG seen value" not in s
    assert E.seen_text("BP", "POPX0_rel").startswith("+11.34 %") and "BP seen value" in E.seen_text("BP", "POPX0_rel")
    assert "+9.20 %" in E.seen_text("BP", "POPX_B1_to_MX_LM10")
    assert "BPG seen value" in E.seen_text("BP", "vs_E_PBC") and E.SEEN_POOL["nc"] in E.seen_text("BPG", "cleaned_rel", pool="nc")


def test_infer_exact_enumeration_beside_the_registered_random_flips():
    d = synth_frame(n_pub=16, cells=2, pairs=2, seed=7)
    r0 = E.infer(d, "MX_LM10", "B1", "publication", "TEST|ELAB|ENUM")
    r1 = E.infer(d, "MX_LM10", "B1", "publication", "TEST|ELAB|ENUM", enum=True)
    assert r0["G"] == 16 and not r0["SF_POOL_exact"] and np.isnan(r0["SF_POOL_p_enum"])
    assert r1["SF_POOL_p"] == r0["SF_POOL_p"]                                  # the registered p is kept
    x = E.unit_stats(d, "MX_LM10", "B1", "publication").dSSE.to_numpy()
    signs = 1 - 2 * ((np.arange(2 ** 16)[:, None] >> np.arange(16)) & 1)
    assert r1["SF_POOL_p_enum"] == pytest.approx(((signs @ x) >= x.sum() - 1e-10 * np.abs(x).sum()).mean(), abs=1e-15)


def test_headline_identity_and_c5_changes():
    R = synth_frame()
    R["MX_LM10_NC"] = R.MX_LM10.copy()
    R["MX_LM10@C5_out"] = R.MX_LM10.copy()
    R["E_PCM@C5_out"] = R.E_PCM + 0.01
    assert E.equals_headline(R, R, "MX_LM10_NC", "B1") and not E.equals_headline(R, R, "MX_LM10", "MX_LM10_NC")
    assert not E.equals_headline(R.iloc[1:], R, "MX_LM10_NC", "B1")             # other rows: not the headline
    assert E.changed_vs_c5_in(R, E.vframe(R, "C5_out"), "C5_out") == (0, 1)
    R["MX_LM10@C5_out"] = R.MX_LM10 + 1e-12
    assert E.changed_vs_c5_in(R, E.vframe(R, "C5_out"), "C5_out") == (1, 1)
    assert E.changed_vs_c5_in(R, R, "C5_in") == (0, 0)


def _crow(stratum, contrast, unit, c5="C5_in", **k):
    base = {"stratum": stratum, "contrast": contrast, "unit": unit, "c5": c5, "cand": contrast.split(" - ")[0],
            "rel": 0.05, "LOPO_lo": -0.01, "LOPO_hi": 0.11, "W/L/T": "3/1/0", "SF_POOL_p": 0.2, "SF_POOL_p_enum": np.nan,
            "G": 4, "top3_share": 0.9, "without_LEAD2": 0.01, "LTPO_min": 0.0, "LTPO_min_without": "a;b",
            "unit_median_rel": 0.04, "IM_REL_lb95_1s": 0.01, "oracle_K": "", "equals_headline": False,
            "point_arms_changed_vs_C5_in": 0, "draw_arms_changed_vs_C5_in": 0}
    return {**base, **k}


def synth_tables():
    hd = dict(equals_headline=True, G=16, SF_POOL_p=0.0966, SF_POOL_p_enum=0.0932, **{"W/L/T": "10/4/2"})
    C = pd.DataFrame([_crow("PRIMARY", "MX_LM10 - B1", "lab_cons", G=9, **{"W/L/T": "9/0/0"}),
                      _crow("PRIMARY", "MX_LM10 - B1", "lab_lib"), _crow("PRIMARY", "MX_LM10 - B1", "publication", **hd),
                      _crow("PRIMARY", "MX_LM10_NC - B1", "publication", **{**hd, "SF_POOL_p": 0.098}),
                      _crow("PRIMARY", "MX_LM10 - B1", "publication", c5="C5_out", draw_arms_changed_vs_C5_in=5, **hd),
                      _crow("SRC", "MX_LM10_NB - B1", "publication", rel=0.09, without_LEAD2=0.026, top3_share=0.935),
                      _crow("POPX0", "MX_LM10 - B1", "publication", rel=0.094), _crow("POPX0", "MX_LM10_NB - B1", "publication", rel=0.11),
                      _crow("PRIMARY", "MX_LM10 - E_PCM", "publication", rel=0.1),
                      _crow("SS_OS", "MX_LM10_SS - B1", "publication", rel=0.075, **{"W/L/T": "4/0/0"}),
                      _crow("SS_OS", "MX_LM10_OS - B1", "publication", rel=-0.079, **{"W/L/T": "2/1/1"}),
                      _crow("SS_OS", "MX_LM10_SS - MX_LM10_OS", "publication", rel=0.143, **{"W/L/T": "2/2/0"})])
    K = pd.DataFrame({"by": ["chem_family"], "class": ["dga"], "harm_flag_reading": [False]})
    P = pd.DataFrame({"stratum": ["SS_OS"], "c5": ["C5_in"], "cells": [35], "publications": [4], "clusters": [1]})
    return {"contrasts": C, "classes": K, "populations": P}


def test_rules_name_the_unit_the_seen_pool_and_bp_sensitivity():
    T = synth_tables()
    for design in ("BPG", "BP"):
        r = E.t_rules(T, design).set_index("rule")
        head = r.loc[[x for x in r.index if x.startswith("headline: ")][0]]
        assert "laboratories (conservative key) W/L/T 9/0/0" in head.in_corpus_value and "publications W/L/T" not in head.in_corpus_value
        pub = r.loc[[x for x in r.index if x.startswith("headline beside it") and "publication" in x][0]]
        assert "publications W/L/T 10/4/2" in pub.in_corpus_value and "SF_POOL p 0.0966" in pub.in_corpus_value
        assert "exact enumeration over 2^16 sign vectors 0.0932" in pub.in_corpus_value
        assert E.SEEN_POOL["nc"] in pub.seen and pub.in_corpus_on_seen_pool.startswith("MX_LM10_NC - B1")
        p0 = r.loc["POPX0 (new extractants): MX_LM10 - B1"]
        assert E.SEEN_POOL["src"] in p0.seen and p0.in_corpus_on_seen_pool.startswith("MX_LM10_NB - B1 on POPX0: +0.1100")
        lt = r.loc[[x for x in r.index if x.startswith("leave-two-publications-out")][0]]
        assert "MX_LM10_NB - B1 on SRC: +0.0260" in lt.in_corpus_on_seen_pool
        assert r.loc["top-3 publication share of the MX_LM10 - B1 gain"].in_corpus_on_seen_pool == "MX_LM10_NB - B1 on SRC: 0.935"
        ss = r.loc[[x for x in r.index if x.startswith("same-series")][0]]
        assert "35 cells, 4 publications, 1 source cluster" in ss.name
        assert all(f"{c} " in ss.in_corpus_value for c in ("MX_LM10_SS - B1", "MX_LM10_OS - B1", "MX_LM10_SS - MX_LM10_OS"))
        note = r.loc[[x for x in r.index if x.startswith("note: ")][0]]
        assert "bitwise on every PRIMARY row: yes" in note.in_corpus_value and "C5_out 0" in note.in_corpus_value
        assert "over all 2^16 sign vectors 0.0932" in note.in_corpus_value
        if design == "BP":
            assert r.label.str.contains(E.SENS).all() and p0.seen.startswith("+11.34 %")
            assert "BPG seen value" in r.loc[[x for x in r.index if x.startswith("control: MX_LM10 - E_PCM (unit publication")][0]].seen
        else:
            assert E.SENS not in head.label and "BPG seen value" not in " ".join(r.seen)


def test_globalk_and_classes_labels():
    d = synth_frame()
    d["in_POPX0"] = d["in_SCOPE_IN"] = True
    d["chem_family"], d["mech_class"], d["scope_in"] = "dga", "NEUTRAL_AMIDE", True
    for design in ("BPG", "BP"):
        g = E.t_globalk(d, design)
        k = g[g.stratum != "LOPO_CHOICE"]
        assert k[k.oracle_best.astype(bool)].label.str.startswith(E.ORACLE).all()
        assert not k[~k.oracle_best.astype(bool)].label.str.contains(E.ORACLE, regex=False).any()
        ch = g[g.stratum == "LOPO_CHOICE"]
        assert ch[ch.c5 != E.PRIMARY_C5].label.str.contains(E.SENS).all()
        assert ch[ch.c5 == E.PRIMARY_C5].label.str.contains(E.SENS).all() == (design == "BP")
        assert k.label.str.contains(E.SENS).all() == (design == "BP")
        cl = E.t_classes(d, design)
        assert cl.label.str.startswith(E.CC).all() and cl.label.str.contains(E.SENS).all() == (design == "BP")


def test_rng_scheme_is_deterministic_and_keyed():
    a = E.rng_for("E_PCM", "C5_in", "BPG", 104729, "t1").random(3)
    b = E.rng_for("E_PCM", "C5_in", "BPG", 104729, "t1").random(3)
    c = E.rng_for("E_PCM", "C5_in", "BP", 104729, "t1").random(3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert E.kname(6.1) == "MX_K6p1" and E.K_ARMS[0] == "MX_K6p1"


# ------------------------------------------------------------------------------------------------------------
# construction on the real metadata (no score)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def camp():
    return E.Campaign(log=lambda *a: None)


def test_popx_counts_and_frozen_duplicate_links(camp):
    st = camp.static
    src = st[st.cell_id.isin(camp.popx_src)]
    assert (len(src), src.publication.nunique()) == (E.REPRO["POPX"]["cells"], E.REPRO["POPX"]["publications"])
    assert (src.n_ms_src == 0).sum() == E.REPRO["POPX0"]["cells"]
    assert set(camp.primary) <= set(camp.popx_src)
    a, b = "259e68bb986bb484", "26cbed1ed70886fe"            # I2 item 4: TDDGA == TDdDGA, bit-identical at 8 metals
    assert frozenset((a, b)) in camp.bit
    P = camp.pools(a)
    assert camp.pos[b] in set(P["src"].tolist()) and camp.pos[b] not in set(P["clean"].tolist())
    assert camp.pos[b] not in set(P["xpub"].tolist())
    assert not src.lab_key_missing.any()
    for t in camp.popx_src:
        i = camp.pos[t]
        for k, allow in E.GUARD:
            camp.guard(i, camp.pools(t)[k], allow_bit=allow, name=k)          # every frozen pool passes the guard


def test_guard_positive_controls(camp):
    a = "259e68bb986bb484"
    i = camp.pos[a]
    with pytest.raises(E.LeakageError):
        camp.guard(i, np.array([i]), name="self")
    with pytest.raises(E.LeakageError):
        camp.guard(i, camp.pools(a)["src"], allow_bit=False, name="src as strict")


def test_single_target_intervention_on_a_noise_surrogate(camp):
    from g21 import cache as GC
    Ys = E.surrogate_Y(camp.Y, "TEST|ELAB|SURROGATE")
    Ds = E.detrend(Ys)
    rec = GC.load("BPG", paths.DEV_SEEDS[0], 0)
    cell = next(c for c in rec["outer"]["cells"] if c["cell_id"] in set(camp.primary))
    t, i = cell["cell_id"], camp.pos[cell["cell_id"]]
    reads = sorted({int(m) for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
    metals = [int(m) for m in cell["metals"]]
    ia = np.array([m for m in metals[:-1]])
    ib = np.array([metals[k + 1] for k in range(len(metals) - 1)])
    kw = dict(design="BPG", seed=paths.DEV_SEEDS[0])
    p0, d0, info = camp.target_arms(t, reads, metals, -0.07, ia, ib, Yv=Ys, Dv=Ds, **kw)
    assert set(p0) == set(E.POINT_ARMS) and set(d0) == set(E.DRAW_ARMS)
    assert all(np.isfinite(v).all() for v in p0.values()) and all(v.shape == (E.NDRAW, len(ia)) for v in d0.values())
    un = [m for m in metals if m not in reads]
    Yp = Ys.copy()
    Yp[i, un] = 9.0
    Dp = Ds.copy()
    Dp[i] = E.qdev(Yp[i])
    p1, d1, _ = camp.target_arms(t, reads, metals, -0.07, ia, ib, Yv=Yp, Dv=Dp, **kw)
    for a in p0:
        if a not in E.LABEL_BASED:
            assert np.array_equal(p0[a], p1[a], equal_nan=True), a
    for a in d0:
        assert np.array_equal(d0[a], d1[a], equal_nan=True), a
    with pytest.raises(E.LeakageError):
        camp.target_arms(t, reads, metals, -0.07, ia, ib, Yv=Ys, Dv=Ds, inject={"clean": [i]}, **kw)
    q0 = camp.target_arms(t, reads, metals, -0.07, ia, ib, Yv=Ys, Dv=Ds, inject={"prim": [i]}, guard=False, **kw)[0]
    q1 = camp.target_arms(t, reads, metals, -0.07, ia, ib, Yv=Yp, Dv=Dp, inject={"prim": [i]}, guard=False, **kw)[0]
    assert not np.array_equal(q0["MX_LM10"], q1["MX_LM10"])                  # a leaked target is caught
    pr = camp.target_arms(t, reads, metals, -0.07, ia, ib, Yv=Ys, Dv=Ds, mode="repro", **kw)[0]
    assert set(pr) == set(E.REPRO_ARMS) and np.array_equal(pr["B1"], p0["B1"])
    assert info["n_prim"] <= info["n_clean"] <= info["n_src"]


def test_gk_poison_check_catches_a_leaky_lopo(monkeypatch):
    """s0-mechanics-15: the check runs the production gk_lopo, so a leaky gk_lopo (K chosen on every publication, the
    held-out one included) is caught; the old check, which dropped p before selecting, could not see it."""
    d = synth_frame(n_pub=6, seed=4)

    def leaky(R, c5):
        dd = E.vframe(R, c5)
        P = dd[dd.in_PRIMARY]
        k = E.select_k(P)
        R["MX_GK_LOPO"] = E.tied_mean(R, k)
        return pd.DataFrame([{"c5": c5, "publication": p, "K_chosen": ";".join(k), "in_primary": True}
                             for p in sorted(dd.publication.unique())])
    assert E.gk_poison_check(d)["passed"]
    monkeypatch.setattr(E, "gk_lopo", leaky)
    assert not E.gk_poison_check(d)["passed"]

