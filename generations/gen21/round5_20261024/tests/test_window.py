"""Tests for r5/window.py (item 1.4, M09 window axis): synthetic data and construction-only checks -- no score of the
real rows is computed or asserted.

* levels: read metals keep their reads, the B1 path equals r4.refs.predict_levels, the intervention passes an honest
  predictor and catches a block-Y peek;
* rules: the frozen grid, the operating-point quantile, the whole-cell and reads-only flags, the nested choice;
* controls: exact matched counts, the Monte Carlo mean agrees with the closed-form expectation, whole-cell masks make the
  CELL control deterministic (flagged, its shares n/a in the tables), common random numbers; the literal 2.5 / 97.5 %
  percentile ranks;
* tail ratio and its bootstrap, the conformal quantile, leave-one-cluster-out Mondrian calibration, the unit
  contributions summing to the excess, top-k cells, the report rules, every table of analyse() on a synthetic frame;
* (construction-only, real metadata and training labels) the operating-point rows of one fold come only from training
  cells, are bit-identical under fit poisoning, and the positive control is caught;
* labels, RNG seed scheme and the frozen item specification.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_window.py
        -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
from r5 import window as WIN  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN  # noqa: E402

IDX = WIN.IDX
PTS = [f"{a}/{b}" for a, b in zip(LANTHANIDES[:-1], LANTHANIDES[1:]) if (a, b) != ("Nd", "Sm")]   # 12 strict pairs


# ------------------------------------------------------------------------------------------------------------
# synthetic frames
# ------------------------------------------------------------------------------------------------------------
def synth_rows(n_seeds=3, n_cells=24, per_cell=8, seed=0, with_c5=True) -> pd.DataFrame:
    """A synthetic E-A-like frame: cells nested in publications and clusters, levels, reads and residuals whose tails grow
    with |level| (so the window is informative), every rule's membership and per-fold thresholds / choices."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        for c in range(n_cells):
            pub = paths.C5 if (with_c5 and c < 3) else f"pub_{c % 8}"
            fold = c % 5
            lv = rng.normal(0, 1.6, per_cell + 1)
            reads = rng.normal(0, 1.5, 4)
            for j in range(per_cell):
                la, lb = lv[j], lv[j + 1]
                sc = max(abs(la), abs(lb))
                yv = rng.normal(0, 0.25)
                err = rng.normal(0, 0.06 + 0.12 * (sc > 2) + 0.4 * (sc > 3))
                rows.append({"split_seed": 1000 + s, "fold": fold, "cell_id": f"c{c:03d}", "publication": pub,
                             "chemotype": f"ch{c % 4}", "cluster_cpg": f"g{c % 6}", "A": PTS[j % 12].split("/")[0],
                             "B": PTS[j % 12].split("/")[1], "pt": PTS[j % 12], "y": yv, "B1": yv - err,
                             "B0": yv - err - 0.01, "HIST_D7": yv - err + 0.005, "levA": la, "levB": lb,
                             "readmin": reads.min(), "readmax": reads.max()})
    R = pd.DataFrame(rows)
    R["s"] = np.maximum(R.levA.abs(), R.levB.abs())
    R["lmin"], R["lmax"] = np.minimum(R.levA, R.levB), np.maximum(R.levA, R.levB)
    R["inst"] = R.split_seed.astype(str) + "|" + R.cell_id
    O = []
    for s in sorted(R.split_seed.unique()):
        for f in range(5):
            O.append({"split_seed": s, "fold": f, "thr_OP75": 1.9 + 0.05 * f, "thr_OP90": 2.6,
                      "choice_NEST_HW": ["H2", "H2.5"][f % 2], "choice_NEST_FAM": ["H2", "OP75", "READMIN-2", "H1.5", "WHOLECELL2"][f]})
    O = pd.DataFrame(O)
    R = R.merge(O, on=["split_seed", "fold"], how="left")
    for r in WIN.RULES:
        R[f"in_{r}"] = WIN.rule_inside(R, r)
    return R, O


# ------------------------------------------------------------------------------------------------------------
# levels and the intervention check
# ------------------------------------------------------------------------------------------------------------
def test_levels_reads_exact_and_equal_r4_path():
    rng = np.random.default_rng(1)
    metals = tuple(range(N_LN))
    y = rng.normal(size=N_LN)
    read = frozenset(IDX[m] for m in ("La", "Gd", "Er", "Lu"))
    for beta in (0.0, -0.072, -0.13):
        L = WIN.levels(y, metals, read, beta)
        assert np.array_equal(L[sorted(read)], y[sorted(read)])
        ym = np.full(N_LN, np.nan)
        ym[sorted(read)] = y[sorted(read)]
        ref = RF.predict_levels({"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([beta])},
                                SimpleNamespace(y_masked=ym), metals)
        assert np.array_equal(L, ref)
    assert np.array_equal(WIN.levels(y, metals, read, 0.0), interp.pwl_levels(np.where(np.isin(np.arange(N_LN), sorted(read)), y, np.nan), metals, "R"))


def test_intervention_honest_passes_peek_caught():
    from g21 import evaluate as E
    rng = np.random.default_rng(2)
    metals = tuple(range(N_LN))
    y = rng.normal(size=N_LN)
    read = frozenset(IDX[m] for m in ("La", "Nd", "Gd", "Yb"))
    ys = E.scramble_hidden(y, tuple(sorted(read)), rng)
    assert not np.allclose(ys, y)
    assert np.array_equal(WIN.levels(ys, metals, read, -0.09), WIN.levels(y, metals, read, -0.09))
    q = [IDX["Ce"], IDX["Pr"]]
    assert not np.array_equal(WIN.peek_levels(ys, metals, read, -0.09)[q], WIN.peek_levels(y, metals, read, -0.09)[q])


def test_read_set_of_is_union_of_steps():
    seq = [(0, 13), (6,), (3, 9), (11,)]
    assert WIN.read_set_of(seq, 3) == frozenset({0, 13, 6, 3, 9})
    assert WIN.read_set_of(seq, 1) == frozenset({0, 13})


# ------------------------------------------------------------------------------------------------------------
# rules
# ------------------------------------------------------------------------------------------------------------
def test_grid_names_and_rule_definitions():
    assert WIN.GRID == ("H1", "H1.5", "H2", "H2.5", "H3", "Hinf", "LO-2", "LO-2.5", "LO-3", "LO-4", "UP2", "UP2.5", "UP3")
    F = pd.DataFrame({"levA": [0.5, -2.2, 1.9, 3.5, -4.5], "levB": [1.0, 0.0, -2.0, 0.0, 0.0],
                      "readmin": [-1.0, -2.5, -2.0, 0.0, -3.0], "readmax": [1.0, 2.5, 2.0, 2.1, 1.0],
                      "inst": ["a", "a", "b", "b", "c"], "publication": ["p", paths.C5, "p", "p", "p"],
                      "thr_OP75": [1.0, 3.0, 2.0, 2.0, 5.0], "thr_OP90": 9.0})
    F["s"] = np.maximum(F.levA.abs(), F.levB.abs())
    F["lmin"], F["lmax"] = np.minimum(F.levA, F.levB), np.maximum(F.levA, F.levB)
    m = {r: WIN.rule_inside(F, r).tolist() for r in WIN.GRID + WIN.OP_RULES + WIN.READ_RULES + ("NOC5",)}
    assert m["H2"] == [True, False, True, False, False]                  # closed interval [-2, 2]
    assert m["Hinf"] == [True] * 5
    assert m["LO-2"] == [True, False, True, True, False]
    assert m["LO-4"] == [True, True, True, True, False]
    assert m["UP3"] == [True, True, True, False, True]
    assert m["OP75"] == [True, True, True, False, True]                 # per-row fold thresholds
    assert m["READMIN-2"] == [True, False, True, True, False]
    assert m["READSIN2"] == [True, False, True, False, False]
    assert m["WHOLECELL2"] == [True, True, True, True, False]            # H2 keeps 1/2 of a, 1/2 of b, 0/1 of c
    assert m["NOC5"] == [True, False, True, True, True]
    F["choice_NEST_FAM"] = ["H2", "H2", "UP3", "UP3", "Hinf"]
    assert WIN.rule_inside(F, "NEST_FAM").tolist() == [True, False, True, False, True]


def test_quantile_threshold_is_the_inverted_cdf():
    rng = np.random.default_rng(3)
    s = np.round(rng.exponential(1.5, 997), 2)
    for c in (0.75, 0.90):
        thr = WIN.quantile_threshold(s, c)
        assert (s <= thr).mean() >= c > (s < thr).mean()
        assert thr in set(s.tolist())


def test_select_rule_criterion_and_ties():
    rng = np.random.default_rng(4)
    n = 4000
    s = np.abs(rng.normal(0, 1.4, n))
    y = rng.normal(0, 0.3, n)
    pred = y - rng.normal(0, 0.05 + 0.5 * (s > 2.0), n)
    F = pd.DataFrame({"s": s, "lmin": -s, "lmax": s, "y": y, "pred": pred, "readmin": 0.0, "readmax": 0.0,
                      "inst": np.arange(n) // 8, "thr_OP75": np.quantile(s, 0.75), "thr_OP90": np.quantile(s, 0.9)})
    name, tab = WIN.select_rule(F, WIN.HW_RULES)
    ok = tab[tab.train_coverage >= WIN.NEST_MIN_COVERAGE]
    assert name in set(ok.rule)
    assert tab.set_index("rule").train_R2_in[name] == ok.train_R2_in.max()
    assert tab.set_index("rule").train_coverage["H1"] < WIN.NEST_MIN_COVERAGE   # the narrowest window is not eligible
    F2 = F.assign(s=np.minimum(F.s, 2.9))                                        # H3 and Hinf keep identical rows
    name2, tab2 = WIN.select_rule(F2.assign(pred=F2.y), ("Hinf", "H3"))          # all R2 = 1: tie -> wider, then order
    assert name2 == "Hinf"
    name3, _ = WIN.select_rule(F2.assign(pred=F2.y), ("H3", "Hinf"))
    assert name3 == "H3"
    name4, _ = WIN.select_rule(F, ("H1",))
    assert name4 == "Hinf"                                                       # nothing qualifies: keep everything


# ------------------------------------------------------------------------------------------------------------
# controls
# ------------------------------------------------------------------------------------------------------------
def test_control_matched_counts_and_closed_form_expectation():
    R, _ = synth_rows(seed=5)
    v = WIN.Rows(R)
    m = v.m("H2")
    D = WIN.control_draws(v, "CELL", {"H2": m}, 3000, 7)
    ex = WIN.expected_control(v, m, "CELL")
    se = D["H2"]["relMSE"].std() / math.sqrt(3000)
    assert abs(D["H2"]["relMSE"].mean() - ex["relMSE"]) <= 4 * se + 1e-12
    # the closed form is exact: E[SSE] per seed = sum over cells of (kept share) x cell SSE
    for s, ix in enumerate(v.idx):
        g = pd.DataFrame({"c": R.cell_id.to_numpy()[ix], "e2": v.e2[ix], "m": m[ix]}).groupby("c")
        assert np.isclose(ex["E_SSE_per_seed"][s], float((g.m.mean() * g.e2.sum()).sum()), rtol=1e-12)


def test_control_whole_cell_mask_is_deterministic_and_full_mask_is_all_rows():
    R, _ = synth_rows(seed=6)
    v = WIN.Rows(R)
    cellmask = v.m("WHOLECELL2")
    full = np.ones(len(R), bool)
    D = WIN.control_draws(v, "CELL", {"W": cellmask, "F": full}, 40, 11)
    st = WIN.mask_stats(v, cellmask)
    assert np.allclose(D["W"]["R2"], st["R2_in"], atol=1e-12)
    assert np.allclose(D["W"]["relMSE"], st["relMSE_in"], atol=1e-12)
    DU = WIN.control_draws(v, "UNSTRAT", {"F": full}, 10, 12)
    assert np.allclose(DU["F"]["R2"], WIN.mask_stats(v, full)["R2_all"], atol=1e-12)
    assert np.allclose(DU["F"]["relMSE"], 0.0, atol=1e-12)


def test_control_common_random_numbers_and_seed():
    R, _ = synth_rows(seed=7)
    v = WIN.Rows(R)
    a = WIN.control_draws(v, "PUBPT", {"x": v.m("H2"), "y": v.m("OP75")}, 30, 99)
    b = WIN.control_draws(v, "PUBPT", {"x": v.m("H2")}, 30, 99)
    assert np.array_equal(a["x"]["R2"], b["x"]["R2"])                         # same order whatever the other rules
    c = WIN.control_draws(v, "PUBPT", {"x": v.m("H2")}, 30, 100)
    assert not np.array_equal(a["x"]["R2"], c["x"]["R2"])


def test_codes_nest_seed_outermost():
    R, _ = synth_rows(seed=8)
    v = WIN.Rows(R)
    for c in WIN.CONTROLS:
        codes = v.codes(c)
        per_seed = [np.unique(codes[ix]) for ix in v.idx]
        for a, b in zip(per_seed[:-1], per_seed[1:]):
            assert a.max() < b.min()


# ------------------------------------------------------------------------------------------------------------
# tail, bulk, units
# ------------------------------------------------------------------------------------------------------------
def test_tail_ratio_point_and_unit_bootstrap():
    R, _ = synth_rows(seed=9)
    v = WIN.Rows(R)
    m = v.m("H2")
    pubs = np.sort(R.publication.unique())
    pc = np.searchsorted(pubs, R.publication.to_numpy())
    W = np.ones((1, len(pubs)))
    ts = WIN.tail_stats(v, m, 0.3, pc, len(pubs), W)
    big = np.abs(v.e) > 0.3
    pin = np.mean([big[ix][m[ix]].mean() for ix in v.idx])
    pout = np.mean([big[ix][~m[ix]].mean() for ix in v.idx])
    assert np.isclose(ts["P_in"], pin) and np.isclose(ts["P_out"], pout) and np.isclose(ts["ratio"], pout / pin)
    assert np.isclose(ts["boot_lo"], ts["ratio"]) and np.isclose(ts["boot_hi"], ts["ratio"])
    from gen20k import inference as INF
    W2 = INF._draw_counts(len(pubs), 200, 5)
    ts2 = WIN.tail_stats(v, m, 0.3, pc, len(pubs), W2)
    assert ts2["boot_B"] == 200 and ts2["boot_lo"] <= ts2["boot_hi"]


def test_conformal_quantile():
    x = np.arange(1.0, 10.0)                                                   # n = 9 -> k = ceil(10 x 0.8) = 8
    assert WIN.conformal_q(x, 0.8) == 8.0
    assert math.isinf(WIN.conformal_q(np.arange(3.0), 0.8))                    # k = 4 > n = 3
    assert math.isinf(WIN.conformal_q(np.array([]), 0.8))


def test_mondrian_calibration_never_uses_own_cluster():
    rng = np.random.default_rng(10)
    n = 60
    F = pd.DataFrame({"split_seed": 1, "pt": "La/Ce", "cluster_cpg": np.repeat(["gA", "gB"], n // 2),
                      "y": rng.normal(size=n)})
    F["B1"] = F.y - np.where(F.cluster_cpg == "gA", 10.0 + rng.random(n), 0.01 * rng.random(n))
    v = WIN.Rows(F)
    m = np.tile([True, False], n // 2)
    hp, hw = WIN.mondrian_halfwidths(v, m)
    ae = np.abs(v.e)
    a, b = (F.cluster_cpg == "gA").to_numpy(), (F.cluster_cpg == "gB").to_numpy()
    assert np.all(hp[a] == WIN.conformal_q(ae[b]))                             # gA rows calibrated on gB only
    assert np.all(hp[b] == WIN.conformal_q(ae[a]))
    assert np.all(hw[a & m] == WIN.conformal_q(ae[b & m]))                     # window-aware: same status, other cluster
    assert np.all(hp[a] < 1.0) and np.all(hp[b] > 9.0)


def test_unit_contributions_sum_to_the_excess():
    R, _ = synth_rows(seed=11)
    v = WIN.Rows(R)
    for r in ("H2", "OP75", "NEST_FAM"):
        m = v.m(r)
        us = WIN.unit_frame(v, m)
        exc = WIN.mask_stats(v, m)["relMSE_in"] - WIN.expected_control(v, m, "CELL")["relMSE"]
        assert np.isclose(us.c_u.sum(), exc, atol=1e-12)
        assert np.allclose(us.dSSE, us.SSE_cell_expected - us.SSE_window)
        t = WIN.unit_tests_row(us, f"test|{r}")
        assert t["G"] == R.cluster_cpg.nunique() and t["SF_POOL_exact"]


def test_top_cells_rank_by_b1_sse():
    F = pd.DataFrame({"cell_id": ["a", "a", "b", "c", "c"], "y": [0, 0, 0, 0, 0.0], "B1": [1, 1, 3, 0.5, 0.5]})
    assert WIN.top_cells(F, 2) == ["b", "a"]


def test_frame_stat_matches_mask_stats():
    R, _ = synth_rows(seed=12)
    v = WIN.Rows(R)
    assert np.isclose(WIN.frame_stat(R, "H2", "R2_in"), WIN.mask_stats(v, v.m("H2"))["R2_in"])
    assert np.isclose(WIN.frame_stat(R, "OP75", "coverage"), WIN.mask_stats(v, v.m("OP75"))["coverage"])
    m = v.m("H2")
    assert np.isclose(WIN.frame_stat(R, "H2", "excess_relMSE_cell"),
                      WIN.mask_stats(v, m)["relMSE_in"] - WIN.expected_control(v, m, "CELL")["relMSE"])
    assert len(PTS) == 12


# ------------------------------------------------------------------------------------------------------------
# analyse() end to end on synthetic rows, report rules
# ------------------------------------------------------------------------------------------------------------
def test_analyse_builds_every_table_with_labels():
    R, O = synth_rows(seed=13)
    T = WIN.analyse(R, O, "BPG", reps_control=30, reps_boot=30, lopo=True, log=lambda s: None)
    need = {"curve", "winsor", "controls", "tail", "bulk", "alternatives", "optimism", "units", "unit_tests", "pairtypes",
            "lopo", "topk", "c5", "duplicates", "report_rules"}
    assert need <= set(T)
    for k, df in T.items():
        assert "label" in df.columns and df.label.astype(str).str.len().min() > 0, k
    cur = T["curve"]
    assert set(cur.rule) == set(WIN.RULES) and {"all", "noC5", "noC5cell"} <= set(cur.variant)
    assert set(T["controls"].control) == set(WIN.CONTROLS)
    rr = T["report_rules"]
    assert set(rr.rule_name) == {"row_flag", "tail_flag", "bulk_precision"}
    assert rr[rr.rule_name == "row_flag"].outcome.isin(["ROW FLAG", "CELL OR CONDITION FLAG"]).all()
    assert rr.label.str.contains("CONSISTENCY CHECK").all()
    alt = T["alternatives"]
    assert set(alt.rule) == set(WIN.ALT_RULES)
    S = WIN.summary(T, "BPG", {"folds": 15}, {"ok": True}, 1.0, "SMOKE")
    json.dumps(S, default=str)
    h = S["headline"][0]
    assert "excess_relMSE_vs_CELL_MC" in h and "excess_relMSE_vs_CELL" not in h and "CELL_ctrl_reproduces_rule" in h


def test_controls_that_reproduce_the_rule_are_flagged_na():
    R, O = synth_rows(seed=14)
    T = WIN.analyse(R, O, "BPG", reps_control=30, reps_boot=30, lopo=False, log=lambda s: None)
    c = T["controls"][T["controls"].variant == "all"].set_index(["control", "rule"])
    for r in ("READMIN-2", "READSIN2", "WHOLECELL2", "NOC5"):                 # constant within (seed, cell)
        assert c.loc[("CELL", r), "ctrl_reproduces_rule"]
        assert c.loc[("CELL", r), "P_ctrl_R2_ge_rule"] == WIN.NA_REPRO == c.loc[("CELL", r), "P_ctrl_relMSE_ge_rule"]
    assert c.loc[("PUBPT", "NOC5"), "ctrl_reproduces_rule"]                   # publication is a PUBPT key
    assert all(c.loc[(k, "Hinf"), "ctrl_reproduces_rule"] for k in WIN.CONTROLS)   # coverage 1
    for k in WIN.CONTROLS:                                                     # a row-level window is never flagged
        assert not c.loc[(k, "H2"), "ctrl_reproduces_rule"]
        assert isinstance(c.loc[(k, "H2"), "P_ctrl_relMSE_ge_rule"], float)
    alt = T["alternatives"][T["alternatives"].variant == "all"].set_index("rule")
    assert alt.alt_CELL_ctrl_reproduces_rule["WHOLECELL2"] and not alt.alt_CELL_ctrl_reproduces_rule["LO-4"]
    opt = T["optimism"][T["optimism"].variant == "all"].set_index("rule")
    assert opt.fold_fits_choosing_cell_level_rule["NEST_FAM"] == 6               # READMIN-2, WHOLECELL2 in 2 of 5 folds
    assert np.isclose(opt.share_choosing_cell_level_rule["NEST_FAM"], 6 / 15)
    rr = T["report_rules"]
    d = rr[(rr.rule_name == "row_flag") & rr.primary].detail.iloc[0]
    assert "cell-level rule in 6 of 15 fold-fits" in d


def test_percentile_bounds_are_the_literal_ranks():
    x = np.random.default_rng(15).permutation(np.arange(1.0, 2001.0))           # B = 2000 draws, ranks 1 .. 2000
    assert WIN._q(x, WIN.CI_LO) == 50.0 and WIN._q(x, WIN.CI_HI) == 1950.0     # inverted_cdf at 0.025 / 0.975
    assert WIN._q(x, (1.0 - WIN.CI_LEVEL) / 2) == 51.0                         # the float-rounded bound is one rank up
    assert np.isclose(WIN.CI_HI - WIN.CI_LO, WIN.CI_LEVEL)


def test_report_rules_logic():
    ctrl = pd.DataFrame([{"rule": "NEST_FAM", "variant": v, "control": "CELL", "relMSE_in": 0.30, "ctrl_relMSE_q975": q,
                          "R2_in": 0.5, "ctrl_R2_mean": 0.51, "ctrl_R2_q975": 0.53} for v, q in (("all", 0.25), ("noC5", 0.35))])
    tail = pd.DataFrame([{"rule": r, "variant": v, "cut": 0.5, "ratio": 4.0 if v == "all" else 2.5, "boot_lo": 1.5,
                          "boot_hi": 9.0} for r in ("OP75", "OP90", "H2", "NEST_FAM") for v in WIN.DRAW_VARIANTS])
    bulk = pd.DataFrame([{"rule": r, "variant": v, "narrowing_mean": 0.12, "hw_ptw_mean": 0.2, "hw_pt_mean": 0.23,
                          "med_in": 0.05, "med_all": 0.06} for r in WIN.MAIN_RULES for v in WIN.DRAW_VARIANTS])
    rr = WIN.report_rules({"controls": ctrl, "tail": tail, "bulk": bulk}, "BPG")
    rf = rr[rr.rule_name == "row_flag"].set_index("variant").outcome
    assert rf["all"] == "ROW FLAG" and rf["noC5"] == "CELL OR CONDITION FLAG"
    dt = rr[rr.rule_name == "row_flag"].set_index("variant").detail["all"]    # the rule's scale, then R2 beside it
    assert "NEST_FAM on inside relative SSE 0.3000 vs CELL q97.5 0.2500" in dt
    assert "inside R2 0.5000 is below the CELL R2 mean 0.5100 (q97.5 0.5300)" in dt and "cell-level" not in dt
    tf = rr[(rr.rule_name == "tail_flag") & rr.primary]
    assert len(tf) == 1 and tf.outcome.iloc[0] == "FAIL"                       # noC5 ratio 2.5 < 3
    bp = rr[(rr.rule_name == "bulk_precision") & rr.primary]
    assert len(bp) == 1 and bp.outcome.iloc[0] == "FAIL (expected)"


def test_checks_failed_flags():
    ok = {"folds": 2, "isolation_ok": 2, "peek_control_runs": 3, "peek_control_caught": 3, "poison_control_runs": 2,
          "poison_control_caught": 2, "B1_identity_max_abs": 0.0}
    assert WIN.checks_failed(ok) == []
    assert "intervention_failed" in WIN.checks_failed({**ok, "intervention_failed": 1})
    assert "peek_control_not_caught" in WIN.checks_failed({**ok, "peek_control_caught": 2})
    assert "B1_identity" in WIN.checks_failed({**ok, "B1_identity_max_abs": 1e-12})
    assert "isolation" in WIN.checks_failed({**ok, "isolation_ok": 1})


# ------------------------------------------------------------------------------------------------------------
# labels, seeds, spec
# ------------------------------------------------------------------------------------------------------------
def test_labels_name_section_9_1():
    from r5 import common as CM
    for r in WIN.RULES:
        for v in WIN.VARIANTS:
            for d in ("BPG", "BP"):
                lb = WIN.label_for(r, v, d)
                assert any(x in lb for x in CM.LABELS)
    assert "fixed after seeing V1" in WIN.label_for("H2")
    assert "SENSITIVITY (C5-BTBP out)" in WIN.label_for("OP75", "noC5")
    assert "SENSITIVITY (design BP)" in WIN.label_for("OP75", "all", "BP")
    assert "outcome-selected" in WIN.label_for("OP75", "top5")


def test_rng_seed_scheme():
    import zlib
    assert WIN.control_seed("BPG", "all", "CELL") == 20261023 + zlib.crc32(b"WINDOW|control|BPG|all|CELL")
    assert WIN.boot_seed("BP", "noC5") == 20261023 + zlib.crc32(b"WINDOW|pubboot|BP|noC5")
    assert WIN.poison_seed(104729, 3) == 1047293


def test_item_spec_lists_the_frozen_rules():
    txt = WIN.item_spec_text()
    for s in ("fixed after seeing V1", "H1, H1.5, H2, H2.5, H3, Hinf", "LO-2, LO-2.5, LO-3, LO-4", "UP2, UP2.5, UP3",
              "OP75, OP90", "READMIN-2", "WHOLECELL2", "NEST_FAM", "CELL = (seed, cell) [primary]", "B = 2000",
              "0.540", "8.9e-16", "s1_window.py --smoke", "s1_window.py --repro", "Row flag", "Tail flag", "Bulk precision",
              str(WIN.control_seed("BPG", "all", "CELL")), str(WIN.boot_seed("BPG", "all"))):
        assert s in txt, s
    assert "\r" not in txt


# ------------------------------------------------------------------------------------------------------------
# construction-only on one real fold record (training labels are read; no score is computed or asserted)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench():
    from gen20k.data import load_bench
    return load_bench()


def test_operating_point_rows_are_training_only_and_poison_invariant(bench):
    from g21 import cache as GC
    from r4 import leakage as RL
    rec = GC.load("BPG", 104729, 1)
    fitted = json.loads((paths.R4RES / "REF" / "fitted_BPG.json").read_bytes())
    beta = float(fitted["104729_1"]["ref:B1"]["coef"][0])
    o, F = WIN.fold_ops(bench, rec, beta)
    tr, te = set(map(int, rec["outer"]["train_index"])), set(map(int, rec["outer"]["test_index"]))
    assert set(F.ci) <= tr and not set(F.ci) & te
    c3_train = {int(c["ci"]) for c in RF.training_cells(bench, rec) if c["k_max"] >= 3 and c["query"]}
    assert set(F.ci) == c3_train
    assert o["choice_NEST_HW"] in WIN.HW_RULES and o["choice_NEST_FAM"] in WIN.FAMILY
    pb, _ = RL.poisoned_bench(bench, rec["outer"]["train_index"], seed=WIN.poison_seed(104729, 1))
    assert WIN.ops_equal((o, F), WIN.fold_ops(pb, rec, beta))
    o2, F2 = WIN.fold_ops(bench, rec, beta, include_test=True)
    assert set(F2.ci) & te
    assert not WIN.ops_equal((o2, F2), WIN.fold_ops(pb, rec, beta, include_test=True))   # positive control caught
    with pytest.raises(AssertionError):
        bad = dict(rec)
        bad["outer"] = dict(rec["outer"])
        bad["outer"]["train_index"] = np.asarray(sorted(tr - {next(iter(set(F.ci)))}))
        WIN.training_frame(bench, bad, beta)
