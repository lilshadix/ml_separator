"""Tests for r5/edec.py and scripts/s1_edec.py (Stage-1 item 1.5, E-DEC): synthetic data and construction-only
checks; no score is computed on real labels.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_edec.py
        -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest

Covered: the false-sign rule (|y| >= 0.10; p = 0 never called); the hull / bracketing rule; grid counts, n_min, LTT
fixed-sequence, CRC and plug-in against brute force; leave-one-cluster-out thresholds never read the held-out labels
(and the leaky positive control is caught); the full fit-poisoning check on a synthetic frame; risk-coverage with ties
and the matched-coverage weights against the expectation over tie permutations; split-conformal ranks; Platt
recovery; net benefit; E-RANK endpoints and the RANDOM expectation against brute force; LINE shares; the D7 identity;
the leave-own-publication-out CHPT majority; savings; the smoke surrogate; configuration names; every table of one
design on synthetic rows (E-DEC and E-REC); and the script's refusal without the seal.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
from r5 import edec as ED  # noqa: E402

LN = ED.LN
READSETS = (("La", "Nd", "Gd", "Lu"), ("La", "Pr", "Gd", "Ho"), ("Ce", "Sm", "Dy", "Er"))


def synth(n_cells: int = 48, seeds=(11, 12), clusters: int = 6, rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in seeds:
        for c in range(n_cells):
            reads = READSETS[c % len(READSETS)]
            amp = rng.normal(0.0, 0.4)
            for q in ED.PAIR_ORDER:
                A, B = q.split("/")
                if A in reads and B in reads:
                    continue
                y = amp + rng.normal(0.0, 0.25)
                b1 = y + rng.normal(0.0, 0.2)
                rows.append({"split_seed": s, "fold": c % 5, "cell_id": f"c{c:03d}", "A": A, "B": B, "ia": LN.index(A),
                             "ib": LN.index(B), "read_set": ",".join(sorted(reads, key=LN.index)), "a_read": A in reads,
                             "b_read": B in reads, "y": y, "B1": b1, "B0": b1 + rng.normal(0, 0.05),
                             "HBK@ML": b1 + rng.normal(0, 0.05), "RIDGE2": b1 + rng.normal(0, 0.1),
                             "LINE": amp + rng.normal(0, 0.3), "G14@ZERO": rng.normal(0, 0.3), "HIST_D7": b1 + rng.normal(0, 0.01),
                             "cluster_cpg": f"g{c % clusters}", "publication": f"p{c % (2 * clusters):02d}",
                             "chemotype": "sc009" if c % 3 else "sc001", "n_metals": 14})
    d = pd.DataFrame(rows)
    d["pubchem"] = d.publication + "|" + d.chemotype
    return ED.prepare(d)


def erec_synth(d: pd.DataFrame, rng_seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    F = d.copy()
    n = len(F)
    cells = sorted(F.cell_id.unique())
    nsib = {c: i % 4 for i, c in enumerate(cells)}
    F["n_sib"] = F.cell_id.map(nsib)
    has = F.n_sib >= 1
    for c in ("COPY", "SH_r2", "SH_r4", "NB_COPY", "NB_SH_r2", "NB_SH_r4"):
        F[c] = np.where(has, F.y + rng.normal(0, 0.2, n), np.nan)
    for c in ("B1_r2", "B0_r2", "B1_r4", "B0_r4", "G14Z", "D7_r2", "D7_r4"):
        F[c] = F.y + rng.normal(0, 0.3, n)
    F["in_GE1"] = has.to_numpy()
    F["in_GE1_WITH_C5"] = has.to_numpy()
    F["in_GE1_NOBIT"] = has.to_numpy()
    F["t_quant"] = F.cell_id.isin(cells[:3])
    F["cls"] = np.where(F.n_sib % 2 == 0, "acid_conc_only", "mixed")
    F["dga"] = F.chemotype == ED.DGA
    maj = ED.chpt_majority(d)
    k3 = list(zip(F.publication, F.chemotype, F.pair))
    F["CHPT_MAJ"] = [maj[k][0] for k in k3]
    F["chpt_source"] = [maj[k][1] for k in k3]
    return F


# ------------------------------------------------------------------------------------------------------------
# rules
# ------------------------------------------------------------------------------------------------------------
def test_false_sign_rule():
    y = np.array([0.05, -0.05, 0.2, -0.2, 0.2, 0.3])
    p = np.array([-1.0, 1.0, -0.1, -0.5, 0.0, np.nan])
    assert ED.false_sign(p, y).tolist() == [False, False, True, False, False, False]
    assert ED.wrong_sign(p, y).tolist() == [True, True, True, False, False, False]
    assert ED.score(p).tolist()[-2:] == [-1.0, -1.0]


def test_hull_flags_bracketing():
    rs = ["La,Nd,Gd,Er"] * 4 + ["La,Nd,Gd,Lu"] * 2
    a = [LN.index(x) for x in ("La", "Gd", "Er", "Tm", "Yb", "Tm")]
    b = [LN.index(x) for x in ("Ce", "Tb", "Tm", "Yb", "Lu", "Yb")]
    assert ED.hull_flags(rs, a, b).tolist() == [True, True, False, False, True, True]


def test_prepare_refuses_both_ends_read():
    d = synth(n_cells=6, seeds=(1,))
    bad = d.copy()
    bad.loc[0, ["a_read", "b_read"]] = True
    with pytest.raises(AssertionError):
        ED.prepare(bad)


# ------------------------------------------------------------------------------------------------------------
# calibration
# ------------------------------------------------------------------------------------------------------------
def test_grid_counts_bruteforce():
    rng = np.random.default_rng(3)
    s = np.round(rng.uniform(-1, 2, 300), 2)
    e = rng.random(300) < 0.2
    n, k = ED.grid_counts(s, e)
    for j in (0, 100, 300, 450, 599):
        t = ED.GRID[j]
        assert n[j] == int((s >= t).sum()) and k[j] == int(((s >= t) & e).sum())


def test_n_min():
    for a in ED.ALPHAS:
        m = ED.n_min(a)
        assert (1 - a) ** m <= ED.DELTA < (1 - a) ** (m - 1)
    assert ED.n_min(0.10) == 22 and ED.n_min(0.05) == 45


def _ltt_brute(n, k, alpha):
    from scipy import stats as SS
    tau = math.inf
    started = False
    for j in range(len(ED.GRID)):
        if not started:
            if n[j] < ED.n_min(alpha):
                continue
            started = True
        if SS.binom.cdf(k[j], n[j], alpha) <= ED.DELTA:
            tau = ED.GRID[j]
        else:
            break
    return tau


def test_ltt_crc_plugin_against_bruteforce():
    rng = np.random.default_rng(5)
    for rep in range(20):
        s = np.round(np.abs(rng.normal(0, 0.6, 400)), 3)
        e = rng.random(400) < (0.4 * np.exp(-3 * s))
        n, k = ED.grid_counts(s, e)
        N = len(s)
        for a in ED.ALPHAS:
            assert ED.calibrate(n, k, N, a, "LTT")[0] == _ltt_brute(n, k, a)
            crc = [ED.GRID[j] for j in range(len(ED.GRID)) if (k[j] + 1) / (N + 1) <= a]
            assert ED.calibrate(n, k, N, a, "CRC")[0] == (min(crc) if crc else math.inf)
            pl = [ED.GRID[j] for j in range(len(ED.GRID)) if n[j] >= 1 and k[j] / n[j] <= a]
            assert ED.calibrate(n, k, N, a, "PLUGIN")[0] == (min(pl) if pl else math.inf)


def test_ltt_no_calls_when_risky():
    s = np.linspace(0.01, 2.0, 200)
    e = np.ones(200, bool)
    n, k = ED.grid_counts(s, e)
    assert ED.calibrate(n, k, 200, 0.10, "LTT") == (math.inf, -1)
    assert ED.calibrate(n, k, 200, 0.10, "PLUGIN") == (math.inf, -1)
    tau, j = ED.calibrate(n, k, 200, 0.10, "CRC")          # per-pair control: (K + 1) / 201 <= 0.10 -> 19 calls
    assert n[j] == 19 and k[j] == 19


def test_loco_thresholds_ignore_held_out_labels_and_leak_is_caught():
    d = synth()
    rng = np.random.default_rng(9)
    for proc in ED.PROCS:
        for mond in (False, True):
            base = ED.loco_taus(d, "B1", 0.10, proc, mondrian=mond)
            leak = ED.loco_taus(d, "B1", 0.10, proc, mondrian=mond, leak=True)
            dp = d.copy()
            m = dp.cluster_cpg == "g2"
            dp.loc[m, "y"] = rng.normal(0, 3, int(m.sum()))
            b2 = ED.loco_taus(dp, "B1", 0.10, proc, mondrian=mond)
            sel = (base.cluster_cpg == "g2").to_numpy()
            assert base[sel].reset_index(drop=True).equals(b2[(b2.cluster_cpg == "g2").to_numpy()].reset_index(drop=True))
            l2 = ED.loco_taus(dp, "B1", 0.10, proc, mondrian=mond, leak=True)
            assert not leak[["n_cal_calls", "k_cal_fs"]].equals(l2[["n_cal_calls", "k_cal_fs"]]) or proc == "CRC"


def test_poison_check_passes_on_synthetic():
    d = synth(n_cells=30, seeds=(11,))
    rep = ED.poison_check(d, "TEST", log=lambda *a: None)
    assert rep["checks_ok"] and rep["clusters"] == 6 and rep["quantities_compared"] > 0
    # s0-mechanics-5: one seed and 6 clusters is not the registered coverage, so the check does not pass
    assert not rep["coverage_full"] and not rep["passed"] and rep["scope"].startswith("calibration labels only")


def test_pooled_counts_beside_the_seed_mean():
    d = pd.DataFrame({"split_seed": [1, 1, 1, 2, 2, 2, 3], "cell_id": ["a", "b", "c", "a", "b", "c", "a"],
                      "A": ["La"] * 7, "B": ["Ce"] * 7, "y": [0.5, -0.5, 0.05, 0.5, -0.5, 0.5, 0.5],
                      "B1": [0.4, 0.4, -0.3, 0.4, 0.4, 0.2, 0.4]})
    call = np.array([True, True, True, True, True, False, False])
    pc = ED.pooled_counts(d, call, "B1")
    # seed 1: 3 calls, 1 false sign (b; c has |y| < 0.10); seed 2: 2 calls, 1 false sign (b); seed 3: none
    assert (pc["calls_pooled"], pc["false_signs_pooled"], pc["seeds_with_calls"], pc["seeds"]) == (5, 2, 2, 3)
    assert math.isclose(pc["fs_pooled"], 0.4) and pc["calls_distinct"] == 3 and pc["false_signs_distinct"] == 1
    assert pc["calls_per_seed"] == "3;2;0" and pc["n_min"] == 22 and "few calls" in pc["few_calls"]
    assert "without calls dropped" in pc["few_calls"] and ED.counts_str(pc).startswith("2/5 = 0.4000 pooled, 3 distinct")
    big = pd.concat([d.assign(cell_id=d.cell_id + str(i)) for i in range(30)], ignore_index=True)
    many = ED.pooled_counts(big[big.split_seed < 3], np.ones(int((big.split_seed < 3).sum()), bool), "B1")
    assert many["few_calls"] == "" and many["calls_per_seed"] == "90;90"
    assert ED.counts_str(ED.pooled_counts(d, np.zeros(7, bool), "B1")) == "no calls in 3 seeds"


def test_support_record_is_counted():
    d = synth(n_cells=12, seeds=(1, 2))
    va = np.where(d.a_read, -4.5, np.nan)
    vb = np.where(d.b_read, -1.0, np.nan)
    s = ED.support_record(d, va, vb)
    assert s["ok"] and s["both_ends_read_rows"] == 0 and s["non_strict_adjacent_rows"] == 0
    assert s["hull_recount_mismatch_rows"] == 0 and s["strict_adjacent"] and s["hull_from_read_sets_only"]
    assert s["bound_rows"] == int(d.a_read.sum()) and s["bound_rows_unread_end"] == 0 and s["v1_values_at_unread_ends"] == 0
    bad = d.copy()
    bad.loc[0, "hull"] = not bad.loc[0, "hull"]
    va2 = va.copy()
    i = int(np.flatnonzero(~d.a_read.to_numpy())[0])
    va2[i] = -4.5                                          # a V1 value at an unread A end, below the limit
    s2 = ED.support_record(bad, va2, vb)
    assert not s2["ok"] and s2["hull_recount_mismatch_rows"] == 1 and s2["bound_rows_unread_end"] == 1
    assert s2["v1_values_at_unread_ends"] == 1 and "bound_rows" not in ED.support_record(d)


def test_no_hull_note():
    d = synth(n_cells=12, seeds=(1,))
    z = d.assign(B1=np.where(d.hull, d.B1, 0.0))
    assert "never calls outside the read hull" in ED.no_hull_note(z, "B1", "LTT_a10_all_glob")
    assert "denominator" in ED.no_hull_note(z, "B1", "CRC_a10_all_glob")
    assert "callable for B0" in ED.no_hull_note(z, "B0", "LTT_a10_all_glob")


def test_apply_calls_respects_hull_and_threshold():
    d = synth(n_cells=12, seeds=(1,))
    taus = ED.loco_taus(d, "B1", 0.10, "PLUGIN")
    c = ED.apply_calls(d, "B1", taus, hull=True, mondrian=False)
    assert not (c & ~d.hull.to_numpy()).any()
    tm = dict(zip(taus.cluster_cpg, taus.tau))
    assert (np.abs(d.B1.to_numpy()[c]) >= d.cluster_cpg.map(tm).to_numpy()[c]).all()


# ------------------------------------------------------------------------------------------------------------
# risk-coverage, matched coverage
# ------------------------------------------------------------------------------------------------------------
def _perm_expect(sc, e):
    """Brute force: mean risk curve over every ordering consistent with descending score (ties permuted)."""
    o = np.argsort(-sc, kind="mergesort")
    groups = [list(g) for _, g in itertools.groupby(o, key=lambda i: sc[i])]
    curves = []
    for perm in itertools.product(*[list(itertools.permutations(g)) for g in groups]):
        order = [i for g in perm for i in g]
        ee = e[order].astype(float)
        curves.append(np.cumsum(ee) / np.arange(1, len(ee) + 1))
    return np.mean(curves, axis=0)


def test_rc_expected_matches_tie_permutations():
    sc = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.1, 0.9])
    e = np.array([1, 0, 0, 1, 1, 0, 0], bool)
    assert np.allclose(ED.rc_expected(sc, e), _perm_expect(sc, e))
    assert ED.aurc_oracle(sc, e) <= ED.aurc(sc, e)


def test_topm_weights_expectation():
    sc = np.array([0.9, 0.5, 0.5, 0.5, 0.2, -1.0])
    for m in range(0, 7):
        w = ED.topm_weights(sc, m)
        assert math.isclose(w.sum(), min(m, 5)) and w[-1] == 0
    w = ED.topm_weights(sc, 2)
    assert np.allclose(w, [1, 1 / 3, 1 / 3, 1 / 3, 0, 0])


def test_cls3_and_threeway():
    assert ED.cls3([0.1, 0.099, -0.1, -0.5, 0]).tolist() == [1, 0, -1, -1, 0]


# ------------------------------------------------------------------------------------------------------------
# Platt, net benefit, conformal
# ------------------------------------------------------------------------------------------------------------
def test_platt_recovers_parameters():
    rng = np.random.default_rng(2)
    z = np.abs(rng.normal(0, 1, 40000))
    pr = 1 / (1 + np.exp(-(-1.0 + 2.0 * z)))
    w = ED.platt_fit(z, rng.random(len(z)) < pr)
    assert abs(w[0] + 1.0) < 0.08 and abs(w[1] - 2.0) < 0.1


def test_net_benefit_formula():
    pi = np.array([0.9, 0.6, 0.4, 0.1])
    ev = np.array([True, False, True, False])
    assert math.isclose(ED.net_benefit(pi, ev, 0.5), (1 - 1 * 1.0) / 4)
    assert math.isclose(ED.net_benefit(pi, ev, 0.3), (2 - 1 * 0.3 / 0.7) / 4)
    assert math.isclose(ED.net_benefit(np.ones(4), ev, 0.2), (2 - 2 * 0.25) / 4)


def test_conformal_rank():
    assert ED.conf_rank(9, 0.9) == 9 and ED.conf_rank(8, 0.9) == 9 and ED.conf_rank(99, 0.95) == 95
    q = ED.conf_quantiles(np.arange(1, 9, dtype=float), [0.5, 0.9])
    assert q[0] == 5.0 and math.isinf(q[1])


def test_interval_and_bounds_shapes():
    d = synth(n_cells=30, seeds=(11,))
    hw, cov, ch = ED.interval_loco(d)
    assert len(hw) == len(d) and set(ch.nominal_chosen) <= set(ED.NOMINAL) and len(ch) == 6
    va = np.where(d.a_read, -4.5, np.nan)
    vb = np.where(d.b_read, 0.0, np.nan)
    R, q = ED.bounds_loco(d, va, vb)
    assert len(R) == int(d.a_read.sum()) and set(R.side) == {"upper (A read at or below L)"} and len(q) == 6


# ------------------------------------------------------------------------------------------------------------
# E-RANK
# ------------------------------------------------------------------------------------------------------------
def test_rank_endpoints_constructed():
    r = ED.rank_endpoints([3, 2, 1, 0, 0], [1, 5, 2, 0, 0])
    assert r["top1"] == 0.0 and r["top3"] == 1.0 and r["regret"] == 4.0
    r = ED.rank_endpoints([1, 1, 0], [2, 1, 0])
    assert r["top1"] == 0.5 and math.isclose(r["regret"], 0.5)
    r = ED.rank_endpoints([0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5])
    assert math.isclose(r["spearman"], 1.0) and r["top1"] == 1.0


def test_rank_random_is_the_permutation_expectation():
    v = np.array([0.3, 1.2, 0.5, 0.1, 0.9])
    t1, t3, rg = [], [], []
    for perm in itertools.permutations(range(5)):
        a = np.array(perm, float)
        r = ED.rank_endpoints(a, v)
        t1.append(r["top1"]), t3.append(r["top3"]), rg.append(r["regret"])
    R = ED.rank_random(v)
    assert math.isclose(np.mean(t1), R["top1"]) and math.isclose(np.mean(t3), R["top3"]) and math.isclose(np.mean(rg), R["regret"])


def test_line_shares():
    ps = {"B1": [{"top1": 0.3, "top3": 0.6, "regret": 0.2, "spearman": 0.5}],
          "LINE": [{"top1": 0.2, "top3": 0.5, "regret": 0.3, "spearman": 0.3}],
          "RANDOM": [{"top1": 0.1, "top3": 0.3, "regret": 0.4, "spearman": 0.0}]}
    s = ED.line_shares(ps)
    assert math.isclose(s["top1"], 0.5) and math.isclose(s["top3"], 2 / 3) and math.isclose(s["regret"], 0.5)
    assert math.isclose(s["spearman"], 0.6)


def test_amp_oracle_excludes_own_pair():
    d = pd.DataFrame({"split_seed": [1, 1, 1, 1], "cell_id": ["a", "a", "a", "b"], "y": [1.0, -2.0, 3.0, 5.0]})
    sc = ED.amp_oracle_score(d)
    assert np.allclose(sc[:3], [2.5, 2.0, 1.5]) and np.isneginf(sc[3])


# ------------------------------------------------------------------------------------------------------------
# E-REC helpers
# ------------------------------------------------------------------------------------------------------------
def test_d7_identity():
    rng = np.random.default_rng(4)
    b0, h = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    beta = rng.uniform(-0.14, -0.03, 50)
    d7 = ED.d7_from_b1(b0, b0 + beta * h, beta)
    assert np.max(np.abs(d7 - (b0 + ED.HIST_D7_BETA * h))) < 1e-12
    with pytest.raises(ValueError):
        ED.d7_from_b1(b0, b0, np.zeros(50))


def test_chpt_majority_leave_own_publication_out():
    rows = []
    for pub, chem, ys in (("P1", "X", [1.0, 1.0, 1.0]), ("P2", "X", [-1.0, -1.0, -1.0]), ("P3", "Y", [0.5]),
                          (paths.C5, "X", [9.0, 9.0, 9.0, 9.0])):
        for i, y in enumerate(ys):
            rows.append({"split_seed": 1, "cell_id": f"{pub}{i}", "publication": pub, "chemotype": chem, "pair": "La/Ce", "y": y})
    d = pd.DataFrame(rows)
    m = ED.chpt_majority(d)
    assert m[("P1", "X", "La/Ce")] == (-1.0, "chemotype x pair type", 3)       # own publication and C5-BTBP excluded
    assert m[("P2", "X", "La/Ce")] == (1.0, "chemotype x pair type", 3)
    v, src, n = m[("P3", "Y", "La/Ce")]
    assert src == "pair type" and n == 6 and v == 0.0                           # fallback, C5-BTBP never a donor


def test_erec_tables_on_synthetic():
    d = synth()
    F = erec_synth(d)
    T = ED.t_erec(F, "BPG")
    pop = T["erec_population"].set_index("stratum")
    assert pop.loc["PRIMARY_OWN", "cells"] == F[F.in_GE1].cell_id.nunique()
    arms = T["erec_arms"]
    assert set(arms[arms.stratum == "PRIMARY_OWN"].arm) == set(ED.EREC_ARMS)
    assert len(T["erec_unit_tests"]) == 4 and len(T["erec_lopo"]) == 4
    assert all("EXPLORATORY (seen)" in s for s in arms.label)


# ------------------------------------------------------------------------------------------------------------
# savings, surrogate, configurations, all tables, script gate
# ------------------------------------------------------------------------------------------------------------
def test_savings_seed():
    g = pd.DataFrame({"cell_id": ["a"] * 3 + ["b"] * 2, "read_set": ["La,Pr,Gd,Lu"] * 3 + ["La,Nd,Gd,Lu"] * 2,
                      "A": ["La", "Ce", "Gd", "Ce", "Pr"], "B": ["Ce", "Pr", "Tb", "Pr", "Nd"]})
    s = ED.savings_seed(g, np.array([True, True, True, False, True]))
    assert s["cells_fully_called"] == 1 and s["unread_pair_end_metals"] == 2 + 2 and s["metals_still_needed"] == 2
    assert s["pairs_signed"] == 4


def test_surrogate_keeps_nan_pattern():
    d = synth(n_cells=6, seeds=(1,))
    d.loc[3, "LINE"] = np.nan
    S = ED.surrogate(d, "BPG", ED.ARMS)
    assert np.isnan(S.loc[3, "LINE"]) and np.isfinite(S.B1).all() and not np.allclose(S.y, d.y)
    assert S.y.equals(ED.surrogate(d, "BPG", ED.ARMS).y)


def test_configurations_and_labels():
    cf = list(ED.configs())
    assert len(cf) == len(ED.ARMS) * 3 * 2 * 3
    assert all(h for _, _, _, h, m in cf if m)
    assert ED.cfg_name(*ED.PRIMARY) == "LTT_a10_hull_glob"
    assert "PRIMARY" in ED.cfg_label("LTT_a10_hull_glob") and "SENSITIVITY" in ED.cfg_label("CRC_a05_all_glob")
    assert ED.label("BP").startswith("EXPLORATORY (seen)") and "SENSITIVITY (BP" in ED.label("BP")


def test_design_tables_on_synthetic():
    d = synth(n_cells=36, seeds=(11, 12))
    va = np.where(d.a_read, -4.2, np.nan)
    vb = np.where(d.b_read, -1.0, np.nan)
    T, rows, extra = ED.design_tables(d, "BPG", va=va, vb=vb, F=erec_synth(d), log=lambda *a: None)
    for k in ("selective", "pairtype", "contrasts", "risk_coverage", "rc_curve", "netbenefit", "citl", "platt", "clusters",
              "unit_tests", "interval", "interval_clusters", "bounds", "savings", "rank", "rank_line_share", "primary",
              "thresholds", "erec_arms", "erec_units"):
        assert k in T and len(T[k]), k
    assert set(T["selective"].c5) == {"C5_in", "C5_out"} and len(rows["C5_in"]) == len(d)
    assert T["selective"].label.str.contains("EXPLORATORY").all()
    assert (T["rank"][T["rank"].arm == "AMP_ORACLE"].label.str.contains("ORACLE")).all()
    assert extra["erec_status"] == "DONE"
    # labels on every table (platt, bounds_rows), the oracle tag of E_AURC, the S03 note, the pooled counts
    for k in ("platt", "bounds_rows"):
        assert T[k].label.str.startswith("EXPLORATORY (seen)").all(), k
    assert T["erec_arms"].label.str.contains("E_AURC uses the oracle order: ORACLE").all()
    sel = T["selective"]
    assert (sel.no_hull_note.str.len() > 0).eq(sel.config.str.contains("_all_")).all()
    for t in (sel, T["pairtype"]):
        assert set(ED.POOLED_KEYS) | {"calls_per_seed", "few_calls"} <= set(t.columns)
        assert (t.calls_pooled == (t.calls * 2).round().astype(int)).all()       # 2 seeds, every seed with rows
    assert {"dfs_rate_seeds", "dfs_rate_pooled", "fs_pooled_counts"} <= set(T["contrasts"].columns)
    assert "fs_counts_B1" in T["clusters"]
    P = T["primary"].set_index("statistic")
    assert list(P.index) == list(ED.PRIMARY_KEYS) + list(ED.COUNT_KEYS)
    assert P.loc[list(ED.COUNT_KEYS), "P"].isna().all() and (P.loc[list(ED.PRIMARY_KEYS), "P"] > 0).all()
    assert {"LOPO_se", "LOPO_n_undefined", "few_calls"} <= set(P.columns)
    item = ED.per_item(T, extra, "BPG", d, seal_tag="TEST", leakage="synthetic", seconds=0.0, ect_checks=None)
    lt = item["lopo_top3"]
    assert {"LOPO_min (spread)", "LOPO_max (spread)", "LOPO_min_by", "LOPO_max_by", f"{ED.JK_LABEL} lo"} <= set(lt.columns)
    assert "LOPO_lo" not in lt and list(lt.index) == list(ED.PRIMARY_KEYS)
    assert {"calls_pooled", "calls_distinct", "seeds_with_calls"} <= set(item["per_seed"].columns)
    ch = item["changed"]
    assert ch.startswith("EXPLORATORY (seen): ") and f"{P.loc['cov_B1', 'point']:.4f}" in ch and "CRC" in ch
    from r5 import common as CM
    assert "per-item" in CM.per_item_table(item, None, title="1.5 per-item table")


def test_design_tables_mark_erec_pending_without_ect():
    d = synth(n_cells=18, seeds=(11,))
    T, _, extra = ED.design_tables(d, "BPG", F=None, erec_status="PENDING-ECT (test)", log=lambda *a: None)
    assert extra["erec_status"] == "PENDING-ECT (test)" and "erec_arms" not in T


def test_script_support_target_is_counted():
    spec = importlib.util.spec_from_file_location("s1_edec_sup", paths.R5 / "scripts" / "s1_edec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = synth(n_cells=12, seeds=(1,))
    key = "support contract (no pair with both ends read; strict adjacency; hull flags from read sets)"
    ok = mod.repro_targets(d, None)[key]
    assert ok["ok"] and ok["hull_recount_mismatch_rows"] == 0 and ok["rows"] == len(d)
    bad = d.copy()
    bad.loc[0, "hull"] = not bad.loc[0, "hull"]
    assert not mod.repro_targets(bad, None)[key]["ok"]


def test_script_refuses_without_seal(monkeypatch):
    spec = importlib.util.spec_from_file_location("s1_edec", paths.R5 / "scripts" / "s1_edec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from r5 import sealcheck

    def refuse(*a, **k):
        raise SystemExit("round-5 seal does not verify (test)")
    monkeypatch.setattr(sealcheck, "require_seal", refuse)
    monkeypatch.setattr(sys, "argv", ["s1_edec.py", "--design", "BPG"])
    with pytest.raises(SystemExit):
        mod.main()
