"""Tests for r5/ew.py (item 1.7, E-W): synthetic data and construction-only checks -- no score is computed on real data.

* the frozen cut-pair set (33 pair types; dZ by atomic number with the Pm gap; the feed pairs);
* wider_pairs never scores a pair with both ends read or an unmeasured metal;
* level curves: exact supports, B1 at beta 0 = B0, LINE = gen20k naive_line, and the prediction intervention passes
  an honest predictor while the block-Y peek is caught;
* cell rows: families, orientation and the leakage counters;
* the per-seed sufficient-sum statistics agree with r5.common (R2, relative SSE, leave-one-publication-out);
* the leave-one-cluster-out jackknife-t and the Nd/Sm report rule (retire logic, giant cluster, row threshold);
* sign agreement counts zero predictions; level_block fields; labels; the frozen item specification;
* amendment A2: Nd/Sm contrast labels and interval units, the WIDE / STRICT population, the per-item table (per-seed
  values of the 'all' variant, BP marked, absolute MSE) and the per-item leakage record;
* (construction-only, real data) one fold: counters clean, row keys unique, STRICT rows and read sets equal round 4's.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ew.py
        -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as SS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
from r5 import common as CM  # noqa: E402
from r5 import ew as EW  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import adapters as AD  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN  # noqa: E402

IDX = EW.IDX
BASIS = np.vstack([np.linspace(-1.0, 1.0, N_LN), np.cos(np.linspace(0.0, 3.0, N_LN))])


def synth_cell(metals=tuple(range(N_LN)), seq=((0, 12), (0, 2), (0, 6), (0, 10), (0, 4), (0, 8))):
    ms = tuple(sorted(metals))
    q = tuple((a, b) for a, b in zip(ms[:-1], ms[1:]) if EW.ZV[b] - EW.ZV[a] == 1)
    return {"ci": 0, "cell_id": "t", "extractant": "e", "chemotype": "c", "publication": "p", "metals": ms, "query": q,
            "k_max": 6, "seqs": {"DOPT_Q": list(seq)}}


def synth_y(rng, metals=tuple(range(N_LN))):
    y = np.full(N_LN, np.nan)
    y[list(metals)] = rng.normal(0.0, 1.0, len(metals)) + np.linspace(0, 2, N_LN)[list(metals)]
    return y


def synth_frame(rng, n_cells=24, pairs=("Nd/Sm", "La/Pr", "Ho/Tm", "Eu/Tb", "Nd/Dy")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic WIDE and STRICT rows (prepared) over the 5 dev seeds; arms differ by their noise level."""
    rows = []
    for s in EW.SEEDS:
        for c in range(n_cells):
            base = {"design": "BPG", "split_seed": s, "fold": c % 5, "cell_id": f"c{c:02d}", "publication": f"p{c % 7}",
                    "cluster_cpg": f"g{c % 4}", "chemotype": f"h{c % 3}", "extractant": f"e{c % 5}", "read_set": "La,Gd"}
            for fam, names in (("WIDE", pairs), ("STRICT", ("La/Ce", "Gd/Tb"))):
                for nm in names:
                    a, b = (IDX[x] for x in nm.split("/"))
                    y = float(rng.normal(0.0, 1.0))
                    rows.append({**base, "family": fam, "A": LANTHANIDES[a], "B": LANTHANIDES[b], "pair": nm,
                                 "dZ": int(EW.ZV[b] - EW.ZV[a]), "cut_class": EW.cut_class(int(EW.ZV[b] - EW.ZV[a])),
                                 "region": EW.region(a, b), "feed": nm in EW.FEED_NAMES, "ndfeb": nm in EW.NDFEB_NAMES,
                                 "heavy_nonclaim": fam == "WIDE" and EW.region(a, b) == "heavy",
                                 "tm_read": bool(c % 2), "lu_read": bool(c % 3 == 0), "y": y,
                                 "p_B1": y + rng.normal(0, 0.3), "p_B0": y + rng.normal(0, 0.4),
                                 "p_LINE": y + rng.normal(0, 0.6), "p_HIST_D7": y + rng.normal(0, 0.3)})
    P = EW.prepare(pd.DataFrame(rows).sort_values(EW.KEY).reset_index(drop=True))
    return P[P.family == "WIDE"], P[P.family == "STRICT"]


# ------------------------------------------------------------------------------------------------------------
# the frozen pair set
# ------------------------------------------------------------------------------------------------------------
def test_pair_universe():
    assert len(EW.PAIRS) == 33 and len(set(EW.PAIRS)) == 33
    dz = {n: int(EW.ZV[b] - EW.ZV[a]) for (a, b), n in zip(EW.PAIRS, EW.PAIR_NAMES)}
    assert all(EW.ZV[b] > EW.ZV[a] for a, b in EW.PAIRS)
    assert sum(v == 2 for v in dz.values()) == 11 and sum(v == 3 for v in dz.values()) == 10
    assert sum(v == 4 for v in dz.values()) == 9
    assert dz["Nd/Sm"] == 2 and dz["Pr/Sm"] == 3 and dz["Eu/Tb"] == 2           # Pm (Z 61) is absent
    assert dz["Nd/Tb"] == 5 and dz["Nd/Dy"] == 6 and dz["Pr/Dy"] == 7
    assert set(EW.FEED_NAMES) <= set(EW.PAIR_NAMES) and min(dz.values()) == 2
    assert "Pr/Nd" not in EW.PAIR_NAMES and "Sm/Eu" not in EW.PAIR_NAMES


def test_region_and_cut_class():
    assert EW.region(IDX["Nd"], IDX["Sm"]) == "light|middle"
    assert EW.region(IDX["Ho"], IDX["Tm"]) == "heavy" and EW.region(IDX["Dy"], IDX["Er"]) == "middle|heavy"
    assert EW.cut_class(2) == "dZ2" and EW.cut_class(6) == "feed dZ5+"


def test_wider_pairs_excludes_known_and_unmeasured():
    metals = tuple(IDX[m] for m in ("La", "Pr", "Nd", "Sm", "Gd", "Dy", "Ho", "Tm", "Lu"))
    read = {IDX["La"], IDX["Pr"], IDX["Gd"], IDX["Lu"]}
    wp = EW.wider_pairs(metals, read)
    assert (IDX["La"], IDX["Pr"]) not in wp                                     # both ends read: known, never scored
    assert (IDX["Nd"], IDX["Sm"]) in wp and (IDX["Ho"], IDX["Tm"]) in wp
    assert all(a in metals and b in metals and not (a in read and b in read) for a, b in wp)
    assert all((a, b) in EW.PAIRS for a, b in wp)
    assert (IDX["Ce"], IDX["Nd"]) not in wp                                      # Ce not measured


# ------------------------------------------------------------------------------------------------------------
# level curves and cell rows
# ------------------------------------------------------------------------------------------------------------
def test_arm_levels_exact_supports_and_line():
    rng = np.random.default_rng(3)
    cell, y = synth_cell(), synth_y(rng)
    obs, L = EW.arm_levels(BASIS, cell, y, -0.09)
    read = list(obs.read)
    assert sorted(read) == sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
    for a in ("B1", "B0", "HIST_D7"):
        assert np.array_equal(L[a][read], y[read])
    _, L0 = EW.arm_levels(BASIS, cell, y, 0.0)
    assert np.array_equal(L0["B1"], L["B0"])
    assert np.array_equal(L["LINE"], AD.naive_line(BASIS, obs.support))
    assert not np.array_equal(L["B1"], L["B0"])                                  # the Gd potential moves unread metals


def test_intervention_passes_honest_and_catches_peek():
    rng = np.random.default_rng(4)
    cell, y = synth_cell(), synth_y(rng)
    obs, L = EW.arm_levels(BASIS, cell, y, -0.09)
    ys = E.scramble_hidden(y, tuple(obs.read), rng)
    _, Ls = EW.arm_levels(BASIS, cell, ys, -0.09)
    for a in EW.ARMS:
        assert np.array_equal(L[a], Ls[a], equal_nan=True)
    assert not np.array_equal(y, ys, equal_nan=True)                             # a block-Y peek would see the change
    y2 = y.copy()
    y2[sorted(obs.read)[0]] += 1.0
    _, L2 = EW.arm_levels(BASIS, cell, y2, -0.09)
    assert not np.array_equal(L["B1"], L2["B1"])                                 # reads do enter


def test_cell_rows_families_and_counters():
    rng = np.random.default_rng(5)
    metals = tuple(IDX[m] for m in ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"))
    cell, y = synth_cell(metals), synth_y(rng, metals)
    chk = EW.Checks()
    rows = EW.cell_rows(BASIS, cell, y, -0.09, {"design": "BPG", "split_seed": 1, "fold": 0, "cell_id": "t"}, chk, rng)
    R = pd.DataFrame(rows)
    assert EW.bad_checks(chk.c, intervention=False) == [] and chk.c["intervention_checked"] == 1
    read = set(int(m) for m in cell["seqs"]["DOPT_Q"][0] + cell["seqs"]["DOPT_Q"][1] + cell["seqs"]["DOPT_Q"][2])
    assert (R[R.family == "STRICT"].dZ == 1).all() and (R[R.family == "WIDE"].dZ >= 2).all()
    assert len(R[R.family == "STRICT"]) == len(cell["query"]) == chk.c["strict_rows"]
    known = sum(1 for a, b in EW.PAIRS if a in read and b in read)
    assert chk.c["wide_both_read_excluded"] == known and len(R[R.family == "WIDE"]) == len(EW.PAIRS) - known
    assert not (R.a_read & R.b_read).any()
    assert np.isfinite(R[[f"p_{a}" for a in EW.ARMS]].to_numpy(float)).all()
    assert R.read_set.nunique() == 1 and R.read_set.iloc[0] == EW.rs_str(read)
    assert np.allclose(R.y, [y[IDX[a]] - y[IDX[b]] for a, b in zip(R.A, R.B)])


def test_bad_checks():
    c = EW.Checks().c
    assert EW.bad_checks(c, intervention=False) == []
    assert EW.bad_checks({**c, "both_ends_read_bad": 1}, intervention=False) == ["both_ends_read_bad"]
    assert EW.bad_checks({**c, "peek_control_runs": 1}, intervention=False) == ["peek_control_not_caught"]
    assert EW.bad_checks({**c, "intervention_failed": 2}, intervention=False) == ["intervention_failed"]
    # s0-mechanics-5: a sealed run needs full coverage of the intervention and one caught peek per fold
    full = {**c, "folds": 2, "cell_instances": 10, "intervention_checked": 10, "peek_control_runs": 2,
            "peek_control_caught": 2}
    assert EW.bad_checks(full) == []
    assert EW.bad_checks(c) == ["intervention_coverage"]          # nothing intervened on
    assert "intervention_coverage" in EW.bad_checks({**full, "intervention_checked": 9})
    assert "peek_control_coverage" in EW.bad_checks({**full, "peek_control_runs": 1, "peek_control_caught": 1})


def test_peek_control_goes_through_the_predictor():
    """s0-mechanics-6: a predictor that reads the whole DOPT_Q sequence (leaky) changes when the metals outside the
    K_EA read set are scrambled; the honest K_EA predictor does not."""
    rng = np.random.default_rng(11)
    metals = tuple(IDX[m] for m in ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"))
    cell, y = synth_cell(metals), synth_y(rng, metals)
    seq = cell["seqs"]["DOPT_Q"]
    read = set(m for p in seq[:EW.K_EA] for m in p)
    assert set(m for p in seq for m in p) - read
    ys = E.scramble_hidden(y, tuple(read), rng)
    _, H1 = EW.arm_levels(BASIS, cell, y, -0.09)
    _, H2 = EW.arm_levels(BASIS, cell, ys, -0.09)
    assert all(np.array_equal(H1[a][list(metals)], H2[a][list(metals)], equal_nan=True) for a in EW.ARMS)
    _, L1 = EW.arm_levels(BASIS, cell, y, -0.09, k=len(seq))
    _, L2 = EW.arm_levels(BASIS, cell, ys, -0.09, k=len(seq))
    assert not all(np.array_equal(L1[a][list(metals)], L2[a][list(metals)], equal_nan=True) for a in EW.ARMS)


# ------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------
def test_sum_statistics_match_common():
    W, S = synth_frame(np.random.default_rng(7))
    F = W[W.pair == "Nd/Sm"]
    a = EW.sums(F)[0][0]
    assert abs(EW.smean(EW.r2_ps(a, "B1")) - CM.r2_seedmean(F, "p_B1")) < 1e-12
    assert abs(EW.smean(EW.skill_ps(a, "B1", "LINE")) - CM.rel_sse(F, "p_B1", "p_LINE")) < 1e-12
    d = EW.smean(EW.delta_ps(a, "B1", "LINE"))
    assert abs(d - (CM.r2_seedmean(F, "p_B1") - CM.r2_seedmean(F, "p_LINE"))) < 1e-12
    assert abs(EW.smean(EW.mse_ps(a, "B1")) - CM.mse_seedmean(F, "p_B1")) < 1e-12
    mine = EW.lopo(F, lambda x: EW.smean(EW.r2_ps(x, "B1")))
    ref = CM.lopo(F, lambda t: CM.r2_seedmean(t, "p_B1"))
    for k in ("point", "se", "lo", "hi", "min", "max"):
        assert abs(mine[k] - ref[k]) < 1e-10, k
    assert mine["min_by"] == ref["min_by"] and mine["max_by"] == ref["max_by"] and mine["P"] == ref["P"]
    A, groups = EW.sums(F, "cluster_cpg")
    assert np.allclose(A.sum(0), a) and groups == sorted(F.cluster_cpg.unique())


def test_jackknife_mean_is_the_t_interval():
    v = np.array([0.1, -0.05, 0.2, 0.03, 0.07])
    jk = EW.jackknife_mean(v)
    se = v.std(ddof=1) / np.sqrt(len(v))
    q = SS.t.ppf(0.975, len(v) - 1)
    assert abs(jk["se"] - se) < 1e-12 and abs(jk["lo"] - (v.mean() - q * se)) < 1e-12 and jk["df"] == 4
    assert np.isnan(EW.jackknife_mean([0.3])["lo"]) and EW.jackknife_mean([0.3])["G"] == 1


def test_ndsm_rule_and_clusters():
    def cl(vals):
        return pd.DataFrame({"cluster_cpg": [f"g{i}" for i in range(len(vals))], "B1_minus_LINE": vals,
                             "qualifies": [True] * len(vals)})
    assert EW.ndsm_rule(cl([0.20, 0.25, 0.22, 0.18]))["retired"] is False
    assert EW.ndsm_rule(cl([0.04, 0.05, 0.03]))["retired"] is True                 # contrast <= +0.05
    assert EW.ndsm_rule(cl([0.40, -0.30, 0.10]))["retired"] is True                # lower bound <= 0
    assert EW.ndsm_rule(cl([0.40]))["retired"] is True                             # G < 2
    assert "RETIRED" in EW.ndsm_rule(cl([0.0, 0.01]))["outcome"]
    W, _ = synth_frame(np.random.default_rng(8))
    W = W.assign(cluster_cpg=np.where(W.cluster_cpg == "g0", EW.GIANT, W.cluster_cpg))
    c = EW.ndsm_clusters(W)
    assert set(c.cluster_cpg) == {EW.GIANT, "g1", "g2", "g3"}
    assert not c[c.cluster_cpg == EW.GIANT].qualifies.any()
    assert (c.rows_per_seed == 6.0).all() and not c.qualifies.any()               # 6 rows per seed < 10
    assert c[c.cluster_cpg == "g1"].qualifies.tolist() == [False]
    c2 = EW.ndsm_clusters(W, min_rows=5)
    assert set(c2[c2.qualifies].cluster_cpg) == {"g1", "g2", "g3"}
    c3 = EW.ndsm_clusters(W, min_rows=5, drop_giant=False)
    assert c3.qualifies.all()


def test_sign_agreement_counts_zero_predictions():
    F = pd.DataFrame({"split_seed": [1] * 4, "y": [0.3, -0.2, 0.5, 0.0], "p_B1": [0.1, -0.1, 0.0, 0.0],
                      "p_B0": [0.0] * 4, "p_LINE": [-0.1, -0.3, 0.2, 0.1], "p_HIST_D7": [0.1, 0.1, 0.1, 0.1]})
    sr = EW.seed_rows(F).iloc[0]
    assert sr.agree_B1 == 0.75 and sr.zero_B1 == 0.5                            # y = 0 & p = 0 agree; p = 0 & y > 0 not
    assert sr.agree_B0 == 0.25 and sr.zero_B0 == 1.0
    assert sr.agree_LINE == 0.5


def test_level_block_and_tables_construct():
    W, S = synth_frame(np.random.default_rng(9))
    b = EW.level_block(W[W.pair == "Nd/Sm"], S)
    for k in ("rows_per_seed", "R2_B1", "R2_LINE", "R2_B0", "R2_HIST_D7", "skill_vs_LINE", "skill_vs_B0", "MSE_mult_B1",
              "var_mult", "MAE_rel_median_B1", "sign_agree_B1", "zero_pred_share_B1", "clusters", "clusters_ge10",
              "largest_share", "kish_clusters", "units_WLT_vs_LINE", "top3_unit_share_vs_LINE", "lopo_R2_B1_lo",
              "lopo_skill_vs_LINE_min_by", "R2_B1_per_seed", "skill_vs_LINE_per_seed"):
        assert k in b, k
    assert b["rows_per_seed"] == 24.0 and b["clusters"] == 4 and len(b["R2_B1_per_seed"].split(";")) == 5
    assert EW.level_block(W.iloc[:0], S) == {"rows_per_seed": 0.0}
    cuts = EW.cut_table(W, S, "BPG", variants=("all",), lopo_on=False)
    assert len(cuts) == len(EW.PAIR_NAMES) and cuts.label.str.contains("EXPLORATORY").all()
    assert cuts[cuts.pair == "Ho/Tm"].label.str.contains("NON-CLAIM").all()
    cl = EW.class_table(W, S, cuts, "BPG", variants=("all",), lopo_on=False)
    assert "dZ2" in set(cl.frame) and "Nd/Sm" in set(cl.frame)
    assert int(cl[cl.frame == "dZ2"].pair_types.iloc[0]) == 4                  # Nd/Sm, La/Pr, Ho/Tm, Eu/Tb
    tr = EW.substitution_trap(W, S, "BPG")
    assert "SUBSTITUTION TRAP" in tr.label.iloc[0] and tr.rows_per_seed.iloc[0] == 24 * 3
    sr = EW.strict_reference(S, "BPG", variants=("all",))
    assert "R2_B1" not in sr.columns and "E-W" not in sr.label.iloc[0]


def test_analyse_constructs_every_table():
    W, S = synth_frame(np.random.default_rng(10), pairs=("Nd/Sm", "La/Pr", "Ho/Tm", "Eu/Tb", "Nd/Dy", "Tm/Lu", "Er/Yb",
                                                         "La/Nd"))
    R = pd.concat([W, S], ignore_index=True)
    R["cluster_cpg"] = np.where(R.cluster_cpg.isin(["g0", "g1"]), EW.GIANT, R.cluster_cpg)
    T = EW.analyse(R, "BPG", log=lambda s: None)
    assert set(T) == {"cuts", "classes", "strict_reference", "substitution_trap", "signs", "heavy", "ndsm_contrast",
                      "ndsm_clusters", "units", "unit_tests", "duplicates", "seen"}
    for k, v in T.items():
        assert "label" in v.columns and v.label.astype(str).str.len().gt(0).all(), k
    assert len(T["seen"]) == len(EW.SEEN_IR) and T["seen"].label.str.contains("CONSISTENCY CHECK").all()
    nd = T["ndsm_contrast"]
    assert nd.spec.str.startswith("PRIMARY").sum() == 1 and bool(nd[nd.spec.str.startswith("PRIMARY")].retired.iloc[0])
    s = EW.summary(T, "BPG", EW.Checks().c, {"ok": True}, 0.0)
    assert set(s["frames"]) == set(EW.MAIN_FRAMES) and "outcome" in s["ndsm_primary"]
    R2 = R.assign(cluster_cpg=np.where(R.cluster_cpg == EW.GIANT, "g9", R.cluster_cpg))
    with pytest.raises(AssertionError):
        EW.analyse(R2, "BPG", log=lambda s: None)                                 # the frozen giant cluster is asserted


def _script():
    """scripts/s1_ew.py as a module (its item table and leakage record; nothing is run or written)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s1_ew_under_test", paths.R5 / "scripts" / "s1_ew.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_rows(seed=12) -> pd.DataFrame:
    """Synthetic STRICT + WIDE rows with the frozen giant cluster and the C5-BTBP cell / publication present, so that
    the variants all, noC5 and noC5cell differ."""
    W, S = synth_frame(np.random.default_rng(seed), pairs=("Nd/Sm", "La/Pr", "Ho/Tm", "Eu/Tb", "Nd/Dy", "Pr/Dy", "Nd/Tb",
                                                           "La/Nd"))
    R = pd.concat([W, S], ignore_index=True)
    R["cluster_cpg"] = np.where(R.cluster_cpg.isin(["g0", "g1"]), EW.GIANT, R.cluster_cpg)
    c5 = R.cell_id == "c03"
    R.loc[c5, "cell_id"], R.loc[c5, "publication"] = paths.C5_CELL, paths.C5
    return R


def _full_checks() -> dict:
    return {**EW.Checks().c, "folds": 1, "isolation_ok": 1, "cell_instances": 1, "intervention_checked": 1,
            "peek_control_runs": 1, "peek_control_caught": 1}


def test_ndsm_contrast_labels_and_interval_units():
    """Amendment A2 (EW-4, EW-6): the PRIMARY row carries the E-W marker and the registered CONSISTENCY CHECK label,
    BP's design SENSITIVITY once; the pooled rows give the interval's publication count P in its own column."""
    P = EW.prepare(_synthetic_rows())
    W, S = P[P.family == "WIDE"], P[P.family == "STRICT"]
    F = W[W.pair == "Nd/Sm"]
    for d in paths.DESIGNS:
        nd, ncl = EW.ndsm_contrast(W, S, d)
        prim = nd[nd.spec.str.startswith("PRIMARY")].iloc[0]
        assert prim.label.startswith(EW.SEEN) and "E-W (wider pairs" in prim.label and "CONSISTENCY CHECK" in prim.label
        assert all(lab.count("SENSITIVITY (design BP)") == (d == "BP") for lab in nd.label)
        assert (ncl[ncl.spec == prim.spec].label == prim.label).all()
        cl_rows, pooled = nd[~nd.spec.str.startswith("pooled")], nd[nd.spec.str.startswith("pooled")]
        assert cl_rows.P.isna().all() and cl_rows.interval.str.contains("leave-one-cluster-out").all()
        assert pooled.interval.str.contains("leave-one-publication-out").all() and (pooled.df == pooled.P - 1).all()
        allc = pooled[pooled.spec.str.contains("all clusters")].iloc[0]
        assert allc.P == F.publication.nunique() and allc.G == F.cluster_cpg.nunique()


def test_population_table_wide_and_strict_lines():
    """Amendment A2 (EW-2): the WIDE population per design and the STRICT E-A rows on their own labelled line."""
    R = _synthetic_rows()
    R = R[~((R.family == "WIDE") & (R.cell_id == "c05"))]                       # a C3 cell without a WIDE row
    for d in paths.DESIGNS:
        P = EW.population_table(R, d)
        assert P.family.tolist() == ["WIDE", "STRICT"]
        for f in ("WIDE", "STRICT"):
            F, r = R[R.family == f], P[P.family == f].iloc[0]
            assert r.cells == F.cell_id.nunique() and r.publications == F.publication.nunique()
            assert r.units == F.cluster_cpg.nunique() and r.unit == "cluster_cpg"
            assert r.rows_per_seed == len(F) / len(EW.SEEDS) and r.rows_all_seeds == len(F)
        w, s = P.iloc[0], P.iloc[1]
        assert w.cells == s.cells - 1 and "without a WIDE row: 1" in w.role
        assert "E-W (wider pairs" in w.label and "E-W" not in s.label and "STRICT = E-A rows" in s.label
        assert ("SENSITIVITY (design BP)" in w.label) == (d == "BP") == ("SENSITIVITY (design BP)" in s.label)


def test_item_table_per_seed_population_and_labels():
    """Amendment A2 (EW-1, EW-2, EW-5, EW-7): the per-seed row is the 'all' variant with BP marked SENSITIVITY
    (design BP); the population sub-table has WIDE lines then STRICT lines; the C5 rows of BP are marked; the metrics
    carry the absolute MSE and var(y)."""
    mod = _script()
    R = _synthetic_rows()
    S, T, POP = {}, {}, {}
    for d in paths.DESIGNS:
        T[d] = EW.analyse(R.assign(design=d), d, log=lambda s: None)
        S[d] = EW.summary(T[d], d, _full_checks(), {"ok": True}, 0.0)
        POP[d] = EW.population_table(R, d)
    txt = mod.item_table(S, T, POP, 0.0)
    ps = next(x for x in txt.splitlines() if x.startswith("| Per-seed values |"))
    assert "variant all" in ps and "BP (SENSITIVITY (design BP))" in ps
    for d in S:
        c = T[d]["classes"]
        for fr in EW.MAIN_FRAMES:
            a = c[(c.variant == "all") & (c.frame == fr)].R2_B1_per_seed.iloc[0]
            b = c[(c.variant == "noC5cell") & (c.frame == fr)].R2_B1_per_seed.iloc[0]
            assert a != b and a in ps and b not in ps, fr
    sec = txt.split("## Population (cells, publications, units, rows per seed)")[1].split("\n## ")[0]
    rows = [x.split(" | ") for x in sec.splitlines() if x.startswith("| BP")]
    assert [(r[0], r[1]) for r in rows] == [("| BPG", "WIDE"), ("| BP", "WIDE"), ("| BPG", "STRICT"), ("| BP", "STRICT")]
    met = txt.split("## R², MSE, relative SSE vs B1")[1].split("\n## ")[0]
    assert "| MSE_B1 | MSE_LINE | MSE_B0 | var_y | MSE_mult_B1 |" in met
    c5 = txt.split("## C5-BTBP in / out")[1].split("\n## ")[0]
    for x in (x for x in c5.splitlines() if x.startswith("| BP")):
        assert ("SENSITIVITY (design BP)" in x) == x.startswith("| BP |")


def test_leakage_record_carries_the_counters():
    """Amendment A2 (EW-3): results/leakage/EW_<design>.json holds the checks counters, the seal and the code digests."""
    mod = _script()
    full = _full_checks()
    rec = mod.leakage_record({**full, "seal": "x", "code_lf_sha256": {}}, "BPG", "TAG")
    assert rec["passed"] and rec["failed"] == [] and rec["counters"] == full and rec["seal"] == "TAG"
    assert set(rec["code_lf_sha256"]) == set(mod.CODE) and mod.LEAK == paths.RES / "leakage"
    bad = mod.leakage_record({**full, "both_ends_read_bad": 1}, "BP", "TAG")
    assert not bad["passed"] and bad["failed"] == ["both_ends_read_bad"] and bad["support"]["both_ends_read_bad"] == 1


def test_labels_name_section_9_1():
    for v in EW.VARIANTS + EW.DUP_VARIANTS:
        for d in paths.DESIGNS:
            lab = EW.label_for(v, d)
            assert any(x in lab for x in CM.LABELS) and lab.startswith(EW.SEEN)
    assert "SENSITIVITY (design BP)" in EW.label_for("all", "BP")
    assert "SENSITIVITY (C5-BTBP out)" in EW.label_for("noC5", "BPG")
    assert "never next to E-A's 0.452" in EW.label_for() and "E-W" not in EW.label_for(ew=False)


def test_item_spec_lists_the_frozen_arms():
    txt = EW.item_spec_text()
    for s in ("B1", "B0", "LINE", "HIST_D7", EW.GIANT, "jackknife-t", "RETIRED", "--smoke", "--repro", "Nd/Sm",
              "NON-CLAIM", "SUBSTITUTION TRAP", "seed_for('EW|intervention|<design>')", "8.9e-16", "Pr/Dy = 7"):
        assert s in txt, s
    assert "\r" not in txt


def test_duplicate_sets_flag_known_items():
    dup = EW.duplicate_sets()
    assert "noV2" in dup and len(dup["noV2"]) > 0
    if "noBITDUP" in dup:
        assert "5169c377777d282b" in dup["noBITDUP"]                                # item 1 (13 of 14 identical)


# ------------------------------------------------------------------------------------------------------------
# construction-only on real data (no score)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench():
    from gen20k import data as D
    return D.load_bench()


def test_one_fold_construction(bench):
    res = EW.compute_design(bench, "BPG", folds=1, cluster_of=EW.cluster_map(), log=lambda s: None)
    R, c = res["rows"], res["checks"]
    assert EW.bad_checks(c) == [] and c["folds"] == 1 and c["peek_control_runs"] == 1
    assert c["intervention_checked"] == c["cell_instances"] > 0
    assert not R.duplicated(EW.KEY).any() and not (R.a_read & R.b_read).any()
    assert set(R.pair[R.family == "WIDE"]) <= set(EW.PAIR_NAMES) and (R.cluster_cpg != "").all()
    rep = EW.repro_checks(R, "BPG")
    assert rep["rows_matched"] == rep["rows_REF"] == rep["rows_STRICT"] and rep["read_sets_equal_share"] == 1.0
    assert rep["max_abs_diff"]["y"] == 0.0 and rep["max_abs_diff"]["B1"] <= EW.TARGETS["B1_vs_REF_max_abs"]
    assert rep["identity_ok"] and not rep["full_run"] and not rep["ok"]
