"""Tests for ``gen19ct.evaluation.metrics``, ``pairs`` and ``calibration`` on hand-computed and synthetic data.

No archive row is read and no model is fitted.  Every expected value below is worked out by hand in the
comment next to it.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from gen19ct.evaluation import calibration as CAL
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as PR

SYS = EM.SYSTEM_COL
STATE = EM.METAL_STATE_COL
PUB = EM.PUB_GROUP_COL
CK = EM.CONDITION_KEY_COL


def _no_v6(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index)


def _frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {SYS: ["S1"] * n, STATE: ["Nd(III)"] * n, PUB: ["g1"] * n, CK: [f"c{i}" for i in range(n)]}
    base.update(cols)
    return pd.DataFrame(base, index=[f"r{i}" for i in range(n)])


# --------------------------------------------------------------------------------------------- #
# metrics: MAE / RMSE / R2 / Spearman
# --------------------------------------------------------------------------------------------- #

def test_macro_mae_rmse_hand_example():
    # unit A: errors 1 and 3 -> MAE 2, RMSE sqrt(5); unit B: error 0.5 -> MAE 0.5, RMSE 0.5
    df = _frame(unit=["A", "A", "B"], log_D=[0.0, 0.0, 0.0], pred=[1.0, 3.0, 0.5], **{SYS: ["S1", "S1", "S2"]})
    pu = EM.per_unit_table(df, unit_cols=["unit"])
    assert pu["mae"].tolist() == [2.0, 0.5]
    assert pu["rmse"].tolist() == pytest.approx([math.sqrt(5), 0.5])
    s = EM.logd_summary(df, EM.Regime(design="V5", arm="B0"), unit_cols=["unit"], v6_mask=_no_v6(df))
    assert EM.metric_value(s, "mae", "unit_macro") == pytest.approx(1.25)          # (2 + 0.5) / 2
    assert EM.metric_value(s, "mae", "row_pooled") == pytest.approx(1.5)           # (1 + 3 + 0.5) / 3
    assert EM.metric_value(s, "mae", "system_macro") == pytest.approx(1.25)        # systems = units here
    assert EM.metric_value(s, "rmse", "unit_macro") == pytest.approx((math.sqrt(5) + 0.5) / 2)
    assert EM.metric_value(s, "rmse", "row_pooled") == pytest.approx(math.sqrt((1 + 9 + 0.25) / 3))
    for col in EM.REGIME_COLUMNS:
        assert col in s.columns
    row = s[(s["metric"] == "mae") & (s["aggregation"] == "unit_macro")].iloc[0]
    assert (row["design"], row["arm"], row["half"], row["averaging_unit"], row["n_units"], row["n_rows"]) == \
        ("V5", "B0", "selection", "unit", 2, 3)
    assert row["role"] == "primary"
    assert set(s.loc[s["aggregation"].isin(["row_pooled", "system_macro"]), "role"]) == {"side"}


def test_equal_weight_within_unit_v6_metal_half_weights():
    # system S: Pr rows errors 1, 1, 1 and one Nd row error 3 -> each metal weight 1/2 -> (1 + 3) / 2 = 2
    df = _frame(log_D=[0.0] * 4, pred=[1.0, 1.0, -1.0, 3.0], **{STATE: ["Pr(III)"] * 3 + ["Nd(III)"]})
    pu = EM.per_unit_table(df, unit_cols=[SYS], equal_weight_within=STATE)
    assert pu["mae"].iloc[0] == pytest.approx(2.0)
    assert EM.per_unit_table(df, unit_cols=[SYS])["mae"].iloc[0] == pytest.approx(1.5)


def test_r2_pooled_and_unit_weighted():
    # y = 1, 2, 3; pred = 1, 2, 4: SS_res = 1, SS_tot = 2 -> R2 = 0.5
    assert EM.r2_score(np.array([1.0, 2, 3]), np.array([1.0, 2, 4])) == pytest.approx(0.5)
    # weights 1/2, 1/2, 1: weighted mean = (0.5 + 1 + 3) / 2 = 2.25;
    # SS_tot = 0.5*1.5625 + 0.5*0.0625 + 1*0.5625 = 1.375; SS_res = 1 -> R2 = 1 - 1/1.375
    assert EM.r2_score(np.array([1.0, 2, 3]), np.array([1.0, 2, 4]), np.array([0.5, 0.5, 1.0])) == \
        pytest.approx(1 - 1 / 1.375)
    assert math.isnan(EM.r2_score(np.array([1.0, 1.0]), np.array([0.0, 2.0])))


def test_spearman_across_rows_hand_value():
    # ranks of pred 1,3,2,4 vs y 1,2,3,4: sum d^2 = 2 -> rho = 1 - 6*2 / (4*15) = 0.8
    df = _frame(log_D=[1.0, 2.0, 3.0, 4.0], pred=[1.0, 3.0, 2.0, 4.0])
    s = EM.logd_summary(df, EM.Regime(design="V1", arm="B3"), unit_cols=[PUB], v6_mask=_no_v6(df))
    assert EM.metric_value(s, "spearman_rows", "rows") == pytest.approx(0.8)
    assert EM.spearman_rho([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254037844386)  # average ranks for ties


def test_within_cell_spearman_eligibility_and_macro():
    rows = []
    # cell A (Nd x S1): 5 rows over 3 condition keys, pred perfectly ordered -> rho 1
    for i, c in enumerate(["k1", "k1", "k2", "k2", "k3"]):
        rows.append({STATE: "Nd(III)", SYS: "S1", CK: c, "log_D": float(i), "pred": float(i)})
    # cell B (Pr x S1): 5 rows over 3 keys, pred reversed -> rho -1
    for i, c in enumerate(["k1", "k2", "k3", "k3", "k3"]):
        rows.append({STATE: "Pr(III)", SYS: "S1", CK: c, "log_D": float(i), "pred": float(-i)})
    # cell C: 6 rows but only 2 condition keys -> ineligible
    for i, c in enumerate(["k1", "k1", "k1", "k2", "k2", "k2"]):
        rows.append({STATE: "Sm(III)", SYS: "S1", CK: c, "log_D": float(i), "pred": float(i)})
    # cell D: 4 rows -> ineligible
    for i in range(4):
        rows.append({STATE: "Eu(III)", SYS: "S1", CK: f"k{i}", "log_D": float(i), "pred": float(i)})
    # cell E: 5 rows, 3 keys, constant prediction -> eligible, rho scored 0
    for i, c in enumerate(["k1", "k2", "k3", "k3", "k3"]):
        rows.append({STATE: "Gd(III)", SYS: "S1", CK: c, "log_D": float(i), "pred": 0.7})
    df = pd.DataFrame(rows)
    df[PUB] = "g1"
    tab = EM.within_cell_spearman_table(df)
    by = tab.set_index(STATE)
    assert by.loc["Nd(III)", "eligible"] and by.loc["Nd(III)", "rho"] == pytest.approx(1.0)
    assert by.loc["Pr(III)", "rho"] == pytest.approx(-1.0)
    assert not by.loc["Sm(III)", "eligible"] and by.loc["Sm(III)", "ineligible_reason"] == "condition_keys<3"
    assert not by.loc["Eu(III)", "eligible"] and by.loc["Eu(III)", "ineligible_reason"] == "rows<5"
    assert by.loc["Gd(III)", "eligible"] and by.loc["Gd(III)", "prediction_constant"] and by.loc["Gd(III)", "rho"] == 0
    s = EM.logd_summary(df, EM.Regime(design="V5", arm="B3x"), unit_cols=list(EM.CELL_COLS), v6_mask=_no_v6(df))
    row = s[s["metric"] == "within_cell_spearman"].iloc[0]
    assert row["value"] == pytest.approx((1 - 1 + 0) / 3) and row["n_units"] == 3 and row["n_rows"] == 15


def test_rank_accuracy_ties_threshold_and_publication():
    # one unit, one publication: y = 0, 0.05, 0.5, 1.0; pred = 0, 0, 0.2, 0.2
    # eligible pairs (|dy| >= 0.1): (0,2) 1, (0,3) 1, (1,2) 1, (1,3) 1, (2,3) tie 1/2; (0,1) excluded -> 4.5 / 5
    df = _frame(log_D=[0.0, 0.05, 0.5, 1.0], pred=[0.0, 0.0, 0.2, 0.2])
    tab = EM.rank_accuracy_table(df, unit_cols=[SYS])
    assert tab["n_pairs"].iloc[0] == 5 and tab["rank_accuracy"].iloc[0] == pytest.approx(0.9)
    # |dy| exactly 0.1 in float (0.3 - 0.2 = 0.09999999999999998) still qualifies; a wrong order scores 0
    df2 = _frame(log_D=[0.3, 0.2], pred=[0.0, 1.0])
    t2 = EM.rank_accuracy_table(df2, unit_cols=[SYS])
    assert t2["n_pairs"].iloc[0] == 1 and t2["rank_accuracy"].iloc[0] == 0.0
    # rows of different publication groups never pair
    df3 = _frame(log_D=[0.0, 1.0], pred=[0.0, 1.0], **{PUB: ["g1", "g2"]})
    t3 = EM.rank_accuracy_table(df3, unit_cols=[SYS])
    assert t3["n_pairs"].iloc[0] == 0 and math.isnan(t3["rank_accuracy"].iloc[0])


def test_rank_accuracy_macro_over_units_and_pooled():
    # unit A: 1 pair correct (1.0); unit B: 3 pairs, 1 correct -> 1/3; macro 2/3, pooled 2/4
    df = _frame(unit=["A", "A", "B", "B", "B"], log_D=[0, 1, 0, 1, 2], pred=[0, 1, 2, 1, 0],
                **{SYS: ["S1", "S1", "S2", "S2", "S2"]})
    s = EM.logd_summary(df, EM.Regime(design="V5", arm="x"), unit_cols=["unit"], v6_mask=_no_v6(df))
    # unit B pairs: (0,1) dy -1 dp 1 wrong; (0,2) dy -2 dp 2 wrong; (1,2) dy -1 dp 1 wrong -> 0 of 3
    assert EM.metric_value(s, "rank_accuracy", "unit_macro") == pytest.approx(0.5)   # (1 + 0) / 2
    assert EM.metric_value(s, "rank_accuracy", "pair_pooled") == pytest.approx(0.25)  # 1 / 4


def test_interval_coverage_width_per_unit_macro_and_pooled():
    # unit A: rows covered at 80 %: yes, no -> 0.5; unit B: yes -> 1.0; macro 0.75, pooled 2/3
    df = _frame(unit=["A", "A", "B"], log_D=[0.0, 2.0, 0.0], pred=[0.0, 0.0, 0.0],
                lower_80=[-1.0, -1.0, -0.5], upper_80=[1.0, 1.0, 0.5], **{SYS: ["S1", "S1", "S2"]})
    s = EM.interval_summary(df, EM.Regime(design="V5", arm="B3x"), unit_cols=["unit"], v6_mask=_no_v6(df),
                            levels=(0.8,))
    assert EM.metric_value(s, "coverage_80", "unit_macro") == pytest.approx(0.75)
    assert EM.metric_value(s, "coverage_80", "row_pooled") == pytest.approx(2 / 3)
    assert EM.metric_value(s, "width_80", "unit_macro") == pytest.approx(1.5)      # (2 + 1) / 2
    assert EM.metric_value(s, "width_80", "row_pooled") == pytest.approx(5 / 3)    # (2 + 2 + 1) / 3
    # an unbounded conformal interval covers and has infinite width
    df.loc["r1", ["lower_80", "upper_80"]] = [-np.inf, np.inf]
    s2 = EM.interval_summary(df, EM.Regime(design="V5", arm="B3x"), unit_cols=["unit"], v6_mask=_no_v6(df),
                             levels=(0.8,))
    assert EM.metric_value(s2, "coverage_80", "unit_macro") == pytest.approx(1.0)
    assert math.isinf(EM.metric_value(s2, "width_80", "unit_macro"))


def test_crps_gaussian_closed_form():
    # CRPS(N(0,1), 0) = 2 phi(0) - 1/sqrt(pi) = (sqrt(2) - 1) / sqrt(pi)
    assert EM.crps_gaussian(0.0, 0.0, 1.0) == pytest.approx((math.sqrt(2) - 1) / math.sqrt(math.pi))
    # sd = 0 reduces to absolute error; scale equivariance CRPS(N(0, s), s z) = s CRPS(N(0,1), z)
    assert EM.crps_gaussian(1.5, 0.5, 0.0) == pytest.approx(1.0)
    assert EM.crps_gaussian(3.0, 0.0, 2.0) == pytest.approx(2 * EM.crps_gaussian(1.5, 0.0, 1.0))
    # numeric check: CRPS = int_{-inf}^{y} F(x)^2 dx + int_{y}^{inf} (1 - F(x))^2 dx
    from scipy.integrate import quad
    from scipy.special import ndtr
    mu, sd, y = 0.3, 0.7, -0.4
    left = quad(lambda x: ndtr((x - mu) / sd) ** 2, -np.inf, y)[0]
    right = quad(lambda x: (1 - ndtr((x - mu) / sd)) ** 2, y, np.inf)[0]
    assert float(EM.crps_gaussian(y, mu, sd)) == pytest.approx(left + right, abs=1e-8)
    with pytest.raises(ValueError):
        EM.crps_gaussian(0.0, 0.0, -1.0)


def test_spearman_abs_error_vs_sd_and_crps_in_summary():
    # |errors| 0.1, 0.2, 0.3, 0.4 ranked like sd 1, 2, 3, 4 -> rho 1
    df = _frame(log_D=[0.0] * 4, pred=[0.1, -0.2, 0.3, -0.4], pred_sd=[1.0, 2.0, 3.0, 4.0],
                lower_50=[-9] * 4, upper_50=[9] * 4)
    s = EM.interval_summary(df, EM.Regime(design="V1", arm="B5", seed=104729, seed_set="discovery"),
                            unit_cols=[PUB], v6_mask=_no_v6(df), levels=(0.5,), sd_col="pred_sd")
    assert EM.metric_value(s, "spearman_abs_error_sd", "rows") == pytest.approx(1.0)
    exp = np.mean(EM.crps_gaussian(np.zeros(4), [0.1, -0.2, 0.3, -0.4], [1, 2, 3, 4]))
    assert EM.metric_value(s, "crps", "unit_macro") == pytest.approx(exp)
    assert s["seed"].iloc[0] == 104729 and s["seed_set"].iloc[0] == "discovery"


def test_majority_label_ties_go_to_smallest_label():
    df = pd.DataFrame({STATE: ["Nd(III)"] * 4 + ["Pr(III)"] * 2, SYS: ["S"] * 6,
                       PUB: ["g2", "g2", "g1", "g1", "g9", "g3"]})
    m = EM.majority_label(df, list(EM.CELL_COLS), PUB).set_index(STATE)[PUB]
    assert m["Nd(III)"] == "g1" and m["Pr(III)"] == "g3"


def test_scoring_guard_and_regime_validation():
    df = _frame(log_D=[0.0, 1.0], pred=[0.0, 1.0])
    mask = pd.Series([False, True], index=df.index)
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        EM.logd_summary(df, EM.Regime(design="V5", arm="B0"), unit_cols=[SYS], v6_mask=mask)
    with pytest.raises(ValueError, match="v6_mask"):
        EM.logd_summary(df, EM.Regime(design="V5", arm="B0"), unit_cols=[SYS], v6_mask=None)
    # V6 is the only design scored without the mask, and never in discovery
    EM.logd_summary(df, EM.Regime(design="V6", arm="M2", half="none", seed=123457, seed_set="confirmation"),
                    unit_cols=[SYS], v6_mask=None)
    with pytest.raises(ValueError, match="V6 is run once"):
        EM.Regime(design="V6", arm="M2", half="none", seed=104729, seed_set="discovery")
    with pytest.raises(ValueError):
        EM.Regime(design="V5", arm="B0", seed=19)  # a seed with seed_set deterministic
    assert EM.Regime(design="V5", arm="B0", half="C").half == "confirmation"
    with pytest.raises(ValueError, match="non-finite"):
        EM.logd_summary(_frame(log_D=[0.0, 1.0], pred=[0.0, np.nan]), EM.Regime(design="V5", arm="B0"),
                        unit_cols=[SYS], v6_mask=pd.Series(False, index=["r0", "r1"]))


def test_strata_rows_split_before_units():
    df = _frame(log_D=[0.0, 0.0, 0.0], pred=[1.0, 2.0, 4.0], acid=["HNO3", "HCl", "HNO3"])
    s = EM.logd_summary(df, EM.Regime(design="V5", arm="B0"), unit_cols=[SYS], v6_mask=_no_v6(df), strata_col="acid")
    assert EM.metric_value(s, "mae", "unit_macro", "acid=HNO3") == pytest.approx(2.5)
    assert EM.metric_value(s, "mae", "unit_macro", "acid=HCl") == pytest.approx(2.0)
    assert EM.metric_value(s, "mae", "unit_macro", "all") == pytest.approx(7 / 3)


# --------------------------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------------------------- #

def _pair_rows() -> pd.DataFrame:
    rows = [
        # fold test, pub g1, system S1, condition c1: Pr, Nd, Nd, Sm, X(?) -> Pr-Nd x2, Pr-Sm, Nd-Sm x2 = 5 pairs
        ("t0", "test", "g1", "S1", "c1", "Pr(III)", 1.0), ("t1", "test", "g1", "S1", "c1", "Nd(III)", 1.5),
        ("t2", "test", "g1", "S1", "c1", "Nd(III)", 1.2), ("t3", "test", "g1", "S1", "c1", "Sm(III)", 2.0),
        ("t4", "test", "g1", "S1", "c1", None, 1.4),
        # same key but another fold -> pairs only among themselves (Pr-Nd = 1)
        ("t5", "train", "g1", "S1", "c1", "Pr(III)", 0.0), ("t6", "train", "g1", "S1", "c1", "Nd(III)", 0.5),
        # other publication / system / condition: no partner
        ("t7", "test", "g2", "S1", "c1", "Eu(III)", 3.0), ("t8", "test", "g1", "S2", "c1", "Eu(III)", 3.0),
        ("t9", "test", "g1", "S1", "c2", "Eu(III)", 3.0),
        # An-Ln pair and same-element different states
        ("u0", "test", "g3", "S3", "c1", "Am(III)", 1.0), ("u1", "test", "g3", "S3", "c1", "Eu(III)", 0.2),
        ("u2", "test", "g4", "S4", "c1", "Pu(IV)", 1.0), ("u3", "test", "g4", "S4", "c1", "Pu(VI)", 0.4),
    ]
    df = pd.DataFrame(rows, columns=["id", "fold", PUB, SYS, CK, STATE, "log_D"]).set_index("id")
    return df


def test_comparable_pairs_rules_and_orientation():
    df = _pair_rows()
    p = PR.comparable_pairs(df)
    assert len(p) == 5 + 1 + 1 + 1
    for r in p.itertuples():
        a, b = df.loc[r.idx_a], df.loc[r.idx_b]
        assert a["fold"] == b["fold"] and a[PUB] == b[PUB] and a[SYS] == b[SYS] and a[CK] == b[CK]
        assert isinstance(a[STATE], str) and isinstance(b[STATE], str) and a[STATE] != b[STATE]
        assert PR.state_order_key(r.state_a) > PR.state_order_key(r.state_b)
        assert r.logsf_obs == pytest.approx(a["log_D"] - b["log_D"])
    assert "t4" not in set(p["idx_a"]) | set(p["idx_b"])
    prnd = p[(p["state_a"] == "Nd(III)") & (p["state_b"] == "Pr(III)") & (p["fold"] == "test")]
    assert sorted(prnd["idx_a"]) == ["t1", "t2"] and set(prnd["idx_b"]) == {"t0"}   # logSF_Nd/Pr orientation
    cls = dict(zip(zip(p["state_a"], p["state_b"]), p["category_class"]))
    assert cls[("Am(III)", "Eu(III)")] == "An-Ln" and cls[("Pu(VI)", "Pu(IV)")] == "An-An"
    assert cls[("Sm(III)", "Nd(III)")] == "Ln-Ln"


def test_comparable_pairs_never_cross_boundaries_random_frame():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({"fold": rng.choice(["train", "test", "val"], n), PUB: rng.choice(["g1", "g2"], n),
                       SYS: rng.choice(["S1", "S2", "S3"], n), CK: rng.choice(["c1", "c2"], n),
                       STATE: rng.choice(["La(III)", "Nd(III)", "Pr(III)", "Am(III)", None], n),
                       "log_D": rng.normal(size=n)}, index=[f"x{i}" for i in range(n)])
    p = PR.comparable_pairs(df)
    brute = set()
    for i, j in itertools.combinations(df.index, 2):
        a, b = df.loc[i], df.loc[j]
        if (a["fold"] == b["fold"] and a[PUB] == b[PUB] and a[SYS] == b[SYS] and a[CK] == b[CK]
                and isinstance(a[STATE], str) and isinstance(b[STATE], str) and a[STATE] != b[STATE]):
            brute.add(frozenset((i, j)))
    got = {frozenset((a, b)) for a, b in zip(p["idx_a"], p["idx_b"])}
    assert got == brute and len(p) == len(got)
    with pytest.raises(KeyError, match="after fold assignment"):
        PR.comparable_pairs(df.drop(columns="fold"))


def test_logsf_mae_direction_flat_heavier_hand_example():
    # three test pairs in one cell pair (Nd/Pr under S1): observed logSF 0.5, 0.2, -0.4
    pairs = pd.DataFrame({"idx_a": ["a0", "a1", "a2"], "idx_b": ["b0", "b1", "b2"], "fold": "test",
                          SYS: "S1", "state_a": "Nd(III)", "state_b": "Pr(III)", "category_class": "Ln-Ln",
                          "logsf_obs": [0.5, 0.2, -0.4]})
    folds = pd.Series("test", index=["a0", "a1", "a2", "b0", "b1", "b2"])
    reg = EM.Regime(design="V5-PAIR", arm="B3x")
    pred = pd.Series([0.0, 0.1, 0.2], index=pairs.index)
    s = PR.pair_summary(pairs, pred, reg, v6_mask=pd.Series(False, index=folds.index), folds=folds)
    m = EM.as_mapping(s)
    assert m["logsf_mae"] == pytest.approx((0.5 + 0.1 + 0.6) / 3)
    # |obs| >= 0.3: pairs 0 (pred 0 -> 1/2) and 2 (pred +, obs - -> 0) -> 0.25, n = 2
    assert m["direction_accuracy_abs_ge_0.3"] == pytest.approx(0.25)
    assert "direction_accuracy_abs_ge_0.1" not in m          # 0.1 is registered for V6 only
    assert "n_pairs=2" in s.loc[s["metric"] == "direction_accuracy_abs_ge_0.3", "note"].iloc[0]
    flat = PR.pair_summary(pairs, PR.flat_logsf(pairs), reg, v6_mask=pd.Series(False, index=folds.index), folds=folds)
    fm = EM.as_mapping(flat)
    assert fm["logsf_mae"] == pytest.approx((0.5 + 0.2 + 0.4) / 3) and fm["direction_accuracy_abs_ge_0.3"] == 0.5
    heavy = PR.pair_summary(pairs, PR.heavier_direction(pairs), reg, v6_mask=pd.Series(False, index=folds.index),
                            folds=folds, direction_only=True)
    hm = EM.as_mapping(heavy)
    assert "logsf_mae" not in hm and hm["direction_accuracy_abs_ge_0.3"] == pytest.approx(0.5)  # right on 0, wrong on 2
    # V6 adds the 0.1 threshold: pairs 0, 1, 2 -> 1/2 + 1 + 0 = 1.5 / 3
    v6 = PR.pair_summary(pairs, pred, EM.Regime(design="V6", arm="M2", half="none", seed=123457,
                                                seed_set="confirmation"), v6_mask=None, folds=folds)
    assert EM.as_mapping(v6)["direction_accuracy_abs_ge_0.1"] == pytest.approx(0.5)


def test_heavier_only_ln_iii_pairs():
    pairs = pd.DataFrame({"state_a": ["Nd(III)", "Am(III)", "Pu(VI)", "Ce(IV)"],
                          "state_b": ["La(III)", "Eu(III)", "Pu(IV)", "La(III)"]})
    h = PR.heavier_direction(pairs)
    assert h.iloc[0] == 1.0 and h.iloc[1:].isna().all()


def test_pair_macro_over_cell_pairs_and_derived_logsf():
    # cell pair 1 (S1 Nd/Pr): errors 0.1, 0.3 -> 0.2; cell pair 2 (S2 Nd/Pr): error 1.0 -> macro 0.6, pooled 1.4/3
    preds = pd.Series({"a": 1.0, "b": 0.6, "c": 2.0, "d": 1.4, "e": 0.0, "f": 1.0})
    pairs = pd.DataFrame({"idx_a": ["a", "c", "e"], "idx_b": ["b", "d", "f"], "fold": "test",
                          SYS: ["S1", "S1", "S2"], "state_a": "Nd(III)", "state_b": "Pr(III)",
                          "category_class": "Ln-Ln", "logsf_obs": [0.5, 0.9, 0.0]})
    d = PR.derived_logsf(pairs, preds)
    assert d.tolist() == pytest.approx([0.4, 0.6, -1.0])
    folds = pd.Series("test", index=list(preds.index))
    s = PR.pair_summary(pairs, d, EM.Regime(design="V5-PAIR", arm="B3x"), v6_mask=pd.Series(False, index=folds.index),
                        folds=folds)
    assert EM.metric_value(s, "logsf_mae", "unit_macro") == pytest.approx(0.6)
    assert EM.metric_value(s, "logsf_mae", "pair_pooled") == pytest.approx(1.4 / 3)
    with pytest.raises(ValueError, match="lack a prediction"):
        PR.derived_logsf(pairs, preds.drop("f"))


def test_pair_scoring_requires_isolated_test_pairs_and_v6_guard():
    pairs = pd.DataFrame({"idx_a": ["a"], "idx_b": ["b"], "fold": "test", SYS: "S1", "state_a": "Nd(III)",
                          "state_b": "Pr(III)", "category_class": "Ln-Ln", "logsf_obs": [0.5]})
    pred = pd.Series([0.1], index=pairs.index)
    reg = EM.Regime(design="V5-PAIR", arm="M2", seed=104729, seed_set="discovery")
    no = pd.Series(False, index=["a", "b"])
    with pytest.raises(AssertionError, match="pair isolation"):
        PR.pair_summary(pairs, pred, reg, v6_mask=no, folds=pd.Series({"a": "test", "b": "train"}))
    with pytest.raises(AssertionError, match="another fold"):
        PR.pair_summary(pairs, pred, reg, v6_mask=no, folds=pd.Series({"a": "train", "b": "train"}))
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        PR.pair_summary(pairs, pred, reg, v6_mask=pd.Series({"a": False, "b": True}),
                        folds=pd.Series({"a": "test", "b": "test"}))


# --------------------------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------------------------- #

def test_conformal_rank_and_quantile_finite_sample():
    assert CAL.conformal_rank(99, 0.2) == 80          # (100 * 0.8) is 80.00000000000001 in float
    assert CAL.conformal_rank(9, 0.05) == 10           # ceil(9.5)
    assert math.isinf(CAL.conformal_quantile(np.arange(9.0), 0.05))   # rank 10 > n = 9: unbounded
    assert CAL.conformal_quantile(np.arange(1.0, 100.0), 0.2) == 80.0  # 80th smallest of 1..99
    assert CAL.conformal_quantile(np.array([3.0, 1.0, 2.0, 4.0]), 0.5) == 3.0  # ceil(2.5) = 3rd smallest


def _gaussian_split(n: int, seed: int, hetero: bool):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=2 * n)
    sd = 0.2 + 1.5 * x if hetero else np.full(2 * n, 0.5)
    y = 2 * x + rng.normal(0, sd)
    pred = 2 * x
    idx = pd.Index([f"r{i}" for i in range(2 * n)])
    return idx[:n], idx[n:], pd.Series(y, index=idx), pd.Series(pred, index=idx), pd.Series(sd, index=idx)


@pytest.mark.parametrize("method,hetero", [("absolute", False), ("normalized", True)])
def test_split_conformal_coverage_gaussian_n5000(method, hetero):
    """Calibration and test sets of n = 5000 each.  At a single split the 50 % coverage has SD about 0.01
    (calibration and test noise together), so the +-0.02 check is made on the mean over five independent
    splits (SD about 0.0045) and every single split must lie within +-0.035."""
    covs = {lvl: [] for lvl in CAL.LEVELS}
    for seed in (11, 12, 13, 14, 15):
        cal, test, y, pred, sd = _gaussian_split(5000, seed, hetero)
        no = pd.Series(False, index=y.index)
        c = CAL.fit_split_conformal(y[cal], pred[cal], calibration_index=cal, outer_test_index=test, v6_mask=no,
                                    design="V5", sd=sd[cal] if method == "normalized" else None, method=method)
        iv = CAL.apply_conformal(c, pred[test], index=test, sd=sd[test] if method == "normalized" else None)
        for lvl in CAL.LEVELS:
            lo, hi = EM.interval_columns(lvl)
            covs[lvl].append(float(((y[test] >= iv[lo]) & (y[test] <= iv[hi])).mean()))
    for lvl, vals in covs.items():
        assert abs(np.mean(vals) - lvl) <= 0.02, (method, lvl, vals)
        assert max(abs(v - lvl) for v in vals) <= 0.035, (method, lvl, vals)


def test_conformal_refuses_outer_or_v6_rows_and_prediction_overlap():
    cal, test, y, pred, _ = _gaussian_split(50, 3, False)
    no = pd.Series(False, index=y.index)
    with pytest.raises(AssertionError, match="inner rows only"):
        CAL.fit_split_conformal(y[cal], pred[cal], calibration_index=cal, outer_test_index=list(cal[:2]),
                                v6_mask=no, design="V5")
    v6 = no.copy()
    v6.iloc[0] = True
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        CAL.fit_split_conformal(y[cal], pred[cal], calibration_index=cal, outer_test_index=test, v6_mask=v6,
                                design="V5")
    c = CAL.fit_split_conformal(y[cal], pred[cal], calibration_index=cal, outer_test_index=test, v6_mask=no,
                                design="V5")
    with pytest.raises(AssertionError, match="calibration rows"):
        CAL.apply_conformal(c, pred[cal], index=cal)


def test_mondrian_small_or_unseen_category_is_unbounded():
    idx = pd.Index([f"c{i}" for i in range(12)])
    y = pd.Series(np.arange(12.0), index=idx)
    pred = pd.Series(np.zeros(12), index=idx)
    cats = ["IN_DOMAIN"] * 10 + ["UNSUPPORTED"] * 2
    c = CAL.fit_mondrian_conformal(y, pred, cats, calibration_index=idx, outer_test_index=[], design="V5",
                                   v6_mask=pd.Series(False, index=idx), levels=(0.5, 0.95))
    # IN_DOMAIN scores 0..9: 50 % rank ceil(11 * 0.5) = 6 -> 5.0; 95 % rank ceil(10.45) = 11 > 10 -> inf
    assert c.quantiles["IN_DOMAIN"][0.5] == 5.0 and math.isinf(c.quantiles["IN_DOMAIN"][0.95])
    # UNSUPPORTED scores 10, 11 (n = 2): 50 % rank ceil(1.5) = 2 -> 11.0; 95 % rank ceil(2.85) = 3 > 2 -> inf
    assert c.quantiles["UNSUPPORTED"][0.5] == 11.0 and math.isinf(c.quantiles["UNSUPPORTED"][0.95])
    iv = CAL.apply_conformal(c, pd.Series([0.0, 0.0, 0.0], index=["t0", "t1", "t2"]), index=["t0", "t1", "t2"],
                             categories=["IN_DOMAIN", "UNSUPPORTED", "FAMILY_EXTRAPOLATION"])
    assert iv["upper_50"].tolist()[:2] == [5.0, 11.0]
    assert math.isinf(iv["upper_95"].iloc[1]) and math.isinf(iv["upper_50"].iloc[2])   # n too small; unseen


def test_cv_plus_hand_example_and_group_check():
    # four calibration residuals 1, 2, 3, 4 (two folds), every fold model predicts 0 at the test point
    cal = pd.Index(["c0", "c1", "c2", "c3"])
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=cal)
    oof = pd.Series(0.0, index=cal)
    fold = pd.Series(["f1", "f1", "f2", "f2"], index=cal)
    tp = pd.DataFrame({"f1": [0.0], "f2": [0.0]}, index=["t0"])
    no = pd.Series(False, index=list(cal) + ["t0"])
    iv = CAL.cv_plus_intervals(y, oof, fold, tp, calibration_index=cal, v6_mask=no, design="V1", levels=(0.5, 0.8))
    # alpha 0.5: upper rank ceil(2.5) = 3 -> 3; lower rank floor(2.5) = 2 -> 2nd smallest of -1..-4 = -3
    assert (iv["lower_50"].iloc[0], iv["upper_50"].iloc[0]) == (-3.0, 3.0)
    # alpha 0.2: upper rank ceil(4.0) = 4 -> 4; lower rank floor(1.0) = 1 -> -4
    assert (iv["lower_80"].iloc[0], iv["upper_80"].iloc[0]) == (-4.0, 4.0)
    iv95 = CAL.cv_plus_intervals(y, oof, fold, tp, calibration_index=cal, v6_mask=no, design="V1", levels=(0.95,))
    assert math.isinf(iv95["upper_95"].iloc[0]) and math.isinf(iv95["lower_95"].iloc[0])  # ranks outside 1..4
    with pytest.raises(AssertionError, match="split"):
        CAL.cv_plus_intervals(y, oof, fold, tp, calibration_index=cal, v6_mask=no, design="V1",
                              groups=["g1", "g2", "g2", "g3"])


def test_cv_plus_coverage_synthetic():
    rng = np.random.default_rng(5)
    n, m, k = 2000, 2000, 5
    x = rng.uniform(0, 1, n + m)
    y = 3 * x + rng.normal(0, 0.4, n + m)
    cal = pd.Index([f"c{i}" for i in range(n)])
    fold = np.arange(n) % k
    slopes = {}
    for f in range(k):
        tr = fold != f
        slopes[f] = float(np.polyfit(x[:n][tr], y[:n][tr], 1)[0])
    oof = np.array([slopes[f] * xi for f, xi in zip(fold, x[:n])])
    test = pd.Index([f"t{i}" for i in range(m)])
    tp = pd.DataFrame({f: slopes[f] * x[n:] for f in range(k)}, index=test)
    no = pd.Series(False, index=list(cal) + list(test))
    iv = CAL.cv_plus_intervals(pd.Series(y[:n], index=cal), pd.Series(oof, index=cal), pd.Series(fold, index=cal),
                               tp, calibration_index=cal, v6_mask=no, design="V1")
    for lvl in CAL.LEVELS:
        lo, hi = EM.interval_columns(lvl)
        cov = float(((y[n:] >= iv[lo].to_numpy()) & (y[n:] <= iv[hi].to_numpy())).mean())
        assert cov >= lvl - 0.02, (lvl, cov)


def test_coverage_by_category_counts_and_bands():
    df = _frame(unit=["A", "A", "B", "C"], log_D=[0.0, 2.0, 0.0, 0.0], pred=[0.0] * 4, lower_80=[-1.0] * 4,
                upper_80=[1.0] * 4, status=["IN_DOMAIN", "IN_DOMAIN", "IN_DOMAIN", "UNSUPPORTED"])
    tab = CAL.coverage_by_category(df, EM.Regime(design="V5", arm="B3x"), category_col="status", unit_cols=["unit"],
                                   v6_mask=_no_v6(df), levels=(0.8,))
    ind = tab[(tab["category"] == "IN_DOMAIN") & (tab["metric"] == "coverage_80") & (tab["aggregation"] == "unit_macro")]
    assert ind["value"].iloc[0] == pytest.approx(0.75) and ind["category_units"].iloc[0] == 2   # (0.5 + 1) / 2
    fam = tab[(tab["category"] == "FAMILY_EXTRAPOLATION") & (tab["metric"] == "coverage_80")]
    assert (fam["category_units"] == 0).all() and fam["value"].isna().all()
    assert CAL.s1d_check({0.5: 0.5, 0.8: 0.8, 0.95: 0.95}, {"IN_DOMAIN": (0.93, 25), "UNSUPPORTED": (0.2, 19)})["pass"] \
        is False                                                     # 0.93 outside [0.65, 0.92] at 25 cells
    assert CAL.s1d_check({0.5: 0.5, 0.8: 0.8, 0.95: 0.95}, {"UNSUPPORTED": (0.2, 19)})["pass"] is True   # < 20 cells
    assert CAL.s1d_check({0.5: 0.5, 0.8: 0.8}, {})["pass"] is False                                    # 95 % missing
    assert CAL.f3_check(0.59, 0.9)["failure"] and CAL.f3_check(0.8, 0.84)["failure"]
    assert not CAL.f3_check(0.8, 0.9)["failure"]
    k = CAL.knows_when_it_does_not_know(0.2, 0.05, {"IN_DOMAIN": {0.5: 0.5, 0.8: 0.8, 0.95: 0.95},
                                                    "UNSUPPORTED": {0.5: 0.45, 0.8: 0.75, 0.95: 0.9},
                                                    "FAMILY_EXTRAPOLATION": {0.5: 0.5, 0.8: 0.69, 0.95: 0.95}})
    assert k["spearman_pass"] and not k["coverage_gap_pass"] and not k["established"]   # 0.80 - 0.69 = 0.11


def test_reliability_and_width_tables():
    rng = np.random.default_rng(2)
    n = 20000
    sd = rng.uniform(0.1, 1.0, n)
    df = pd.DataFrame({"log_D": rng.normal(0, sd), "pred": 0.0, "pred_sd": sd, "dist": sd * 3})
    reg = EM.Regime(design="V5", arm="M7", seed=104729, seed_set="discovery")
    no = pd.Series(False, index=df.index)
    rel = CAL.sd_reliability_table(df, reg, v6_mask=no)
    assert rel["arm"].eq("M7").all() and rel["averaging_unit"].eq("row").all()
    assert len(rel) == 10 and (rel["n_rows"] == 2000).all()
    assert np.allclose(rel["mean_abs_error"], rel["expected_mean_abs_error"], atol=0.05)
    iv = CAL.gaussian_intervals(df["pred"], df["pred_sd"], index=df.index)
    df = pd.concat([df, iv], axis=1)
    sp, bn = CAL.width_vs_distance(df, reg, distance_cols=["dist"], v6_mask=no, levels=(0.8,))
    assert sp["spearman_width_distance"].iloc[0] == pytest.approx(1.0) and len(bn) == 10
    curve = CAL.gaussian_coverage_curve(df, reg, v6_mask=no)
    assert np.allclose(curve["empirical"], curve["nominal"], atol=0.04)
