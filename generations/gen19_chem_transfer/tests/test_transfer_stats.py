"""Tests for ``gen19ct.evaluation.transfer``: cluster bootstrap, BCa, R19, TOST, Benjamini-Hochberg, the
section 9 margin and the signal-injection helper.  Synthetic data only; no model is fitted."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import transfer as T


def _units(deltas_by_cluster: dict[str, list[float]]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """comparator = delta, candidate = 0, so the macro Delta is the mean of the deltas."""
    idx, d, c = [], [], []
    for cl, vals in deltas_by_cluster.items():
        for k, v in enumerate(vals):
            idx.append(f"{cl}_u{k}")
            d.append(v)
            c.append(cl)
    index = pd.Index(idx)
    return pd.Series(d, index=index), pd.Series(0.0, index=index), pd.Series(c, index=index)


# --------------------------------------------------------------------------------------------- #
# cluster bootstrap
# --------------------------------------------------------------------------------------------- #

def test_cluster_bootstrap_reproducible_with_seed_19_and_manual_draws():
    a, b, c = _units({"A": [1.0, 1.0], "B": [0.4], "C": [-0.1, 0.1, 0.3]})
    r1 = T.paired_cluster_bootstrap(a, b, c)
    r2 = T.paired_cluster_bootstrap(a, b, c)
    assert r1.seed == 19 and r1.n_resamples == 10_000
    assert np.array_equal(r1.draws, r2.draws)
    assert not np.array_equal(r1.draws, T.paired_cluster_bootstrap(a, b, c, seed=20).draws)
    picks = np.random.default_rng(19).integers(0, 3, size=(10_000, 3))
    assert np.array_equal(T.draw_cluster_picks(3), picks)
    # clusters sorted A, B, C; units of a drawn cluster enter as often as it is drawn
    members = {0: [1.0, 1.0], 1: [0.4], 2: [-0.1, 0.1, 0.3]}
    for r in (0, 1, 2, 777, 9999):
        vals = [v for j in picks[r] for v in members[int(j)]]
        assert r1.draws[r] == pytest.approx(np.mean(vals))
    assert r1.point == pytest.approx(np.mean([1, 1, 0.4, -0.1, 0.1, 0.3]))


def test_higher_is_better_flips_the_sign():
    a, b, c = _units({"A": [0.6, 0.7], "B": [0.8]})
    lower = T.paired_cluster_bootstrap(a, b, c)
    higher = T.paired_cluster_bootstrap(a, b, c, higher_is_better=True)
    assert higher.point == pytest.approx(-lower.point)
    assert np.allclose(higher.draws, -lower.draws)


def test_leave_one_cluster_out_deltas_hand_values():
    # sums A 2.0 (2 units), B 0.4 (1), C 0.3 (3): total 2.7 over 6
    a, b, c = _units({"A": [1.0, 1.0], "B": [0.4], "C": [-0.1, 0.1, 0.3]})
    loco = T.paired_cluster_bootstrap(a, b, c).loco_deltas
    assert loco["A"] == pytest.approx(0.7 / 4)
    assert loco["B"] == pytest.approx(2.3 / 5)
    assert loco["C"] == pytest.approx(2.4 / 3)


def test_p_value_mde_and_record():
    rng = np.random.default_rng(1)
    a, b, c = _units({f"s{i}": list(rng.normal(0.5, 0.05, 3)) for i in range(15)})
    r = T.paired_cluster_bootstrap(a, b, c, contrast="demo", cluster_unit="system")
    assert r.p_two_sided == 0.0
    assert r.mde_80 == pytest.approx(2.80 * np.std(r.draws, ddof=1))
    rec = r.record()
    assert rec["decides"] and rec["n_clusters"] == 15 and rec["n_units"] == 45 and rec["percentile_low"] > 0
    # a null contrast: symmetric deltas around 0 -> p well above 0.05
    a0, b0, c0 = _units({f"s{i}": [(-1) ** i * 0.3] for i in range(20)})
    assert T.paired_cluster_bootstrap(a0, b0, c0).p_two_sided > 0.5


def test_paired_bootstrap_rejects_unpaired_units():
    a, b, c = _units({"A": [1.0], "B": [0.5]})
    with pytest.raises(ValueError, match="same units"):
        T.paired_cluster_bootstrap(a, b.iloc[:1], c)
    with pytest.raises(ValueError, match="finite value in both arms"):
        T.paired_cluster_bootstrap(a, b.where(b.index != b.index[0]), c)


def test_bca_vs_percentile_on_skewed_statistic():
    rng = np.random.default_rng(3)
    x = rng.exponential(1.0, size=40)
    idx = pd.Index([f"u{i:02d}" for i in range(40)])
    r = T.unclustered_bootstrap(pd.Series(x, index=idx), pd.Series(0.0, index=idx))
    plo, phi = r.percentile_interval()
    blo, bhi = r.bca_interval()
    # right-skewed mean: positive acceleration and bias correction move both BCa bounds up
    assert r.acceleration > 0
    assert blo > plo and bhi > phi
    # the acceleration is the jackknife formula: sum(c^3) / (6 (sum c^2)^1.5), c = mean(jk) - jk
    jk = np.array([(x.sum() - v) / 39 for v in x])
    cen = jk.mean() - jk
    assert r.acceleration == pytest.approx((cen ** 3).sum() / (6 * (cen ** 2).sum() ** 1.5))
    # symmetric data: BCa is close to the percentile interval
    y = rng.normal(0, 1, size=60)
    iy = pd.Index(range(60))
    s = T.unclustered_bootstrap(pd.Series(y, index=iy), pd.Series(0.0, index=iy))
    assert np.allclose(s.bca_interval(), s.percentile_interval(), atol=0.01)


def test_bca_matches_scipy_formula_on_identical_draws():
    resampling = pytest.importorskip("scipy.stats._resampling")
    if not hasattr(resampling, "_bca_interval"):
        pytest.skip("scipy private BCa helper not available")
    rng = np.random.default_rng(3)
    x = rng.exponential(1.0, size=40)
    idx = pd.Index([f"u{i:02d}" for i in range(40)])
    r = T.unclustered_bootstrap(pd.Series(x, index=idx), pd.Series(0.0, index=idx))
    try:
        a1, a2, _ = resampling._bca_interval((x,), lambda v, axis=-1: np.mean(v, axis=axis), -1, 0.025,
                                             r.draws[None, :], None, np)
    except TypeError:
        pytest.skip("scipy private BCa helper has another signature")
    assert np.quantile(r.draws, [float(np.squeeze(a1)), float(np.squeeze(a2))]) == pytest.approx(r.bca_interval())


def test_unclustered_reference_and_generic_statistic_share_draws():
    a, b, c = _units({"A": [1.0, 0.5], "B": [0.4], "C": [-0.1, 0.2]})
    u = T.unclustered_bootstrap(a, b)
    assert not u.clustered and u.n_clusters == u.n_units == 5 and not u.record()["decides"]
    frame = pd.DataFrame({"delta": a - b, "cluster": c})
    g = T.cluster_bootstrap_statistic(frame, "cluster", lambda f: float(f["delta"].mean()), n_resamples=500)
    p = T.paired_cluster_bootstrap(a, b, c, n_resamples=500)
    assert np.allclose(g.draws, p.draws) and np.allclose(g.jackknife, p.jackknife) and g.point == pytest.approx(p.point)


# --------------------------------------------------------------------------------------------- #
# R19
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def passing_v5():
    rng = np.random.default_rng(42)
    d = {f"sys{i:02d}": list(rng.normal(0.30, 0.05, 3)) for i in range(12)}
    a, b, sysc = _units(d)
    pubc = pd.Series([f"pub{k % 8}" for k in range(len(a))], index=a.index)
    boots = {"system": T.paired_cluster_bootstrap(a, b, sysc, cluster_unit="system"),
             "publication_group": T.paired_cluster_bootstrap(a, b, pubc, cluster_unit="publication_group")}
    sens = {name: 0.2 for name in T.REGISTERED_SENSITIVITIES["V5"]}
    kw = dict(design="V5", stage="discovery", point=boots["system"].point, margin=0.1, bootstraps=boots,
              seed_deltas=[0.25, 0.31, 0.28, 0.30, 0.27], deterministic=False, sensitivity_deltas=sens)
    return kw


def test_r19_passes_when_every_item_holds(passing_v5):
    res = T.r19(**passing_v5)
    assert res.verdict == "PASS" and res.passes
    assert [i["status"] for i in res.items] == ["PASS"] * 6
    assert set(res.to_frame()["item"]) == {1, 2, 3, 4, 5, 6}


def test_r19_item1_margin_fails(passing_v5):
    res = T.r19(**{**passing_v5, "margin": 1.0})
    assert res.verdict == "FAIL" and res.item(1)["status"] == "FAIL" and not res.passes


def test_r19_item2_interval_under_every_cluster_unit(passing_v5):
    boots = dict(passing_v5["bootstraps"])
    pub = boots["publication_group"]
    boots["publication_group"] = dataclasses.replace(pub, draws=pub.draws - pub.point)  # centred on 0
    res = T.r19(**{**passing_v5, "bootstraps": boots})
    assert res.verdict == "FAIL" and res.item(2)["status"] == "FAIL"
    # a registered cluster unit without a bootstrap, and an unregistered bootstrap, also fail item 2
    res2 = T.r19(**{**passing_v5, "bootstraps": {"system": boots["system"]}})
    assert res2.verdict == "FAIL" and res2.item(2)["status"] == "FAIL"
    short = dataclasses.replace(passing_v5["bootstraps"]["system"], n_resamples=999)
    res3 = T.r19(**{**passing_v5, "bootstraps": {**passing_v5["bootstraps"], "system": short}})
    assert res3.verdict == "FAIL" and res3.item(2)["status"] == "FAIL"


def test_r19_item3_p_value_fails_alone():
    # V6 (percentile only): 250 of 10,000 draws just below 0 -> 2.5 % quantile interpolates above 0, p = 0.05
    draws = np.r_[np.full(250, -1e-6), np.full(9750, 0.5)]
    br = T.BootstrapResult(contrast="c", cluster_unit="system", point=0.4, draws=draws,
                           jackknife=pd.Series([0.35 + 0.01 * i for i in range(13)], index=[f"s{i}" for i in range(13)]),
                           n_units=13, n_clusters=13, n_resamples=10_000, seed=19, clustered=True,
                           higher_is_better=False)
    assert br.percentile_interval()[0] > 0 and br.p_two_sided == pytest.approx(0.05)
    sens = {n: 0.1 for n in T.REGISTERED_SENSITIVITIES["V6"]}
    res = T.r19(design="V6", stage="confirmation", point=0.4, margin=0.02, bootstraps={"system": br},
                seed_deltas=[0.3] * 5, deterministic=False, sensitivity_deltas=sens)
    assert res.verdict == "FAIL"
    assert res.item(3)["status"] == "FAIL"
    assert [res.item(i)["status"] for i in (1, 2, 4, 5, 6)] == ["PASS"] * 5


def _fixed_bootstrap(draws: np.ndarray, point: float, unit: str, n_clusters: int) -> T.BootstrapResult:
    # a symmetric positive jackknife -> acceleration 0, every leave-one-cluster-out Delta > 0
    jk = pd.Series(point + 0.01 * (np.arange(n_clusters) - (n_clusters - 1) / 2), index=[f"c{i}" for i in range(n_clusters)])
    return T.BootstrapResult(contrast="c", cluster_unit=unit, point=point, draws=draws, jackknife=jk, n_units=n_clusters,
                             n_clusters=n_clusters, n_resamples=10_000, seed=19, clustered=True, higher_is_better=False)


def test_r19_item3_p_under_every_registered_cluster_unit_secondary_alone_fails():
    """Resolution 2026-09-15: item 3 needs p < 0.05 under EVERY registered cluster unit.  V5: the system cluster passes
    everything; the publication-group cluster has 250 of 10,000 draws just below 0 -> p = 0.05 exactly (not < 0.05)
    while its percentile and BCa lower bounds stay > 0 (item 2 passes) -> only item 3 fails."""
    rng = np.random.default_rng(11)
    point = 0.5
    sys_b = _fixed_bootstrap(rng.normal(point, 0.05, 10_000), point, "system", 18)
    pub_draws = np.r_[np.full(250, -1e-6), np.full(9750, point)]
    pub_b = _fixed_bootstrap(pub_draws, point, "publication_group", 27)
    assert sys_b.p_two_sided < 0.05 and pub_b.p_two_sided == pytest.approx(0.05)
    assert pub_b.percentile_interval()[0] > 0 and pub_b.bca_interval()[0] > 0
    sens = {n: 0.1 for n in T.REGISTERED_SENSITIVITIES["V5"]}
    kw = dict(design="V5", stage="discovery", point=point, margin=0.1, seed_deltas=None, deterministic=True,
              sensitivity_deltas=sens)
    res = T.r19(bootstraps={"system": sys_b, "publication_group": pub_b}, **kw)
    assert res.verdict == "FAIL" and res.item(3)["status"] == "FAIL"
    assert [res.item(i)["status"] for i in (1, 2, 5, 6)] == ["PASS"] * 4 and res.item(4)["status"] == "VACUOUS"
    assert "publication_group: p=0.05" in res.item(3)["detail"] and "system" not in res.item(3)["detail"]
    assert res.item(3)["name"] == "two_sided_p_below_0.05_every_cluster_unit"
    # the same p failing under the primary cluster alone also fails item 3; both passing -> PASS with p per unit
    res2 = T.r19(bootstraps={"system": dataclasses.replace(pub_b, cluster_unit="system", n_clusters=18,
                                                           jackknife=sys_b.jackknife),
                             "publication_group": dataclasses.replace(sys_b, cluster_unit="publication_group")}, **kw)
    assert res2.item(3)["status"] == "FAIL" and "system: p=0.05" in res2.item(3)["detail"]
    ok = T.r19(bootstraps={"system": sys_b, "publication_group": dataclasses.replace(sys_b, cluster_unit="publication_group")},
               **kw)
    assert ok.passes and "system: p=" in ok.item(3)["detail"] and "publication_group: p=" in ok.item(3)["detail"]
    # a registered cluster unit whose bootstrap is missing fails items 2 and 3 (p cannot be read)
    miss = T.r19(bootstraps={"system": sys_b}, **kw)
    assert miss.item(3)["status"] == "FAIL" and "publication_group: no bootstrap" in miss.item(3)["detail"]
    # V1 (one registered cluster unit) and V2: item 3 reads exactly that unit
    v1 = T.r19(design="V1", stage="discovery", point=point, margin=0.05, deterministic=True, seed_deltas=None,
               bootstraps={"publication_group": pub_b},
               sensitivity_deltas={n: 0.1 for n in T.REGISTERED_SENSITIVITIES["V1"]})
    assert v1.item(3)["status"] == "FAIL" and v1.item(2)["status"] == "PASS"


def test_r19_item4_seed_agreement(passing_v5):
    res = T.r19(**{**passing_v5, "seed_deltas": [0.2, -0.1, -0.2, 0.3, 0.1]})
    assert res.verdict == "FAIL" and res.item(4)["status"] == "FAIL"                  # 3 of 5
    ok = T.r19(**{**passing_v5, "seed_deltas": [0.2, -0.1, 0.2, 0.3, 0.1]})
    assert ok.item(4)["status"] == "PASS" and ok.passes                                 # 4 of 5 in discovery
    conf = T.r19(**{**passing_v5, "stage": "confirmation", "seed_deltas": [0.2, -0.1, 0.2, 0.3, 0.1]})
    assert conf.verdict == "FAIL" and conf.item(4)["status"] == "FAIL"                 # confirmation needs 5 of 5
    short = T.r19(**{**passing_v5, "seed_deltas": [0.2, 0.2, 0.2]})
    assert short.verdict == "FAIL" and short.item(4)["status"] == "FAIL"
    det = T.r19(**{**passing_v5, "seed_deltas": None, "deterministic": True})
    assert det.item(4)["status"] == "VACUOUS" and det.passes


def test_r19_item5_leave_one_cluster_out(passing_v5):
    loco = {"system": [0.3] * 11 + [0.0], "publication_group": [0.3] * 8}
    res = T.r19(**{**passing_v5, "loco_deltas": loco})
    assert res.verdict == "FAIL" and res.item(5)["status"] == "FAIL"
    assert (passing_v5["bootstraps"]["system"].loco_deltas > 0).all()


def test_r19_item6_sensitivities(passing_v5):
    sens = dict(passing_v5["sensitivity_deltas"])
    res = T.r19(**{**passing_v5, "sensitivity_deltas": {**sens, "HNO3_only_cells": -0.01}})
    assert res.verdict == "FAIL" and res.item(6)["status"] == "FAIL"
    missing = {k: v for k, v in sens.items() if k != "parent_structure_hiding"}
    res2 = T.r19(**{**passing_v5, "sensitivity_deltas": missing})
    assert res2.verdict == "FAIL" and "parent_structure_hiding: missing" in res2.item(6)["detail"]
    res3 = T.r19(**{**passing_v5, "sensitivity_deltas": {**sens, "V5-P": T.UNTESTABLE}})
    assert res3.verdict == "UNDECIDED" and not res3.passes and res3.item(6)["status"] == "UNTESTABLE"
    res4 = T.r19(**{**passing_v5, "sensitivity_deltas": {**sens, "exploratory_extra": -5.0}})
    assert res4.passes and "unregistered" in res4.item(6)["detail"]
    # section 2 resolution: the any-partner wildcard-copy filter decides (V5 and V1); the strict one never does
    res5 = T.r19(**{**passing_v5, "sensitivity_deltas": {**sens, "wildcard_copies_excluded_scoring": -0.001}})
    assert res5.verdict == "FAIL" and "wildcard_copies_excluded_scoring: Delta" in res5.item(6)["detail"]
    no_wc = {k: v for k, v in sens.items() if k != "wildcard_copies_excluded_scoring"}
    assert T.r19(**{**passing_v5, "sensitivity_deltas": no_wc}).item(6)["status"] == "FAIL"
    res6 = T.r19(**{**passing_v5, "sensitivity_deltas": {**sens, "wildcard_copies_strict_excluded_scoring": -1.0}})
    assert res6.passes and "exploratory, not deciding: ['wildcard_copies_strict_excluded_scoring']" in res6.item(6)["detail"]


def test_registered_sensitivity_and_cluster_lists():
    assert set(T.REGISTERED_SENSITIVITIES["V5"]) == {
        "loose_setting", "strict_setting", "V5-P", "V5-cell-only", "non_DGA_stratum", "HNO3_only_cells",
        "parent_structure_hiding", "acid_grid_rows_excluded", "censoring_candidates_excluded_scoring",
        "sr_iii_dropped_training", "wildcard_copies_excluded_scoring"}
    assert set(T.REGISTERED_SENSITIVITIES["V1"]) == {
        "censoring_candidates_excluded_scoring", "acid_grid_rows_excluded", "sr_iii_dropped_training",
        "near_duplicate_key_groups_value_blind", "compilation_doi_groups", "wildcard_copies_excluded_scoring"}
    for design, names in T.REGISTERED_SENSITIVITIES.items():
        assert "censoring_candidates_excluded_scoring" in names, design
        assert "wildcard_copies_strict_excluded_scoring" not in names, design          # strict stays exploratory
        # section 2 resolution, extended to V5-P and V5-PAIR by the section 8 R19 item 6 resolution (2026-09-15)
        assert ("wildcard_copies_excluded_scoring" in names) == (design in ("V5", "V1", "V5-P", "V5-PAIR")), design
    assert T.EXPLORATORY_SENSITIVITIES == {d: ("wildcard_copies_strict_excluded_scoring",)
                                           for d in ("V5", "V1", "V5-P", "V5-PAIR")}
    assert T.sensitivity_status("V1", "wildcard_copies_excluded_scoring") == "registered"
    assert T.sensitivity_status("V5", "wildcard_copies_strict_excluded_scoring") == "exploratory"
    assert T.sensitivity_status("V2", "wildcard_copies_excluded_scoring") == "unregistered"
    assert T.scoring_filters("V5") == ("acid_grid_rows_excluded_scoring", "censoring_candidates_excluded_scoring",
                                       "wildcard_copies_excluded_scoring")
    assert T.scoring_filters("V2") == ("censoring_candidates_excluded_scoring", "acid_grid_rows_excluded_scoring")
    assert T.scoring_filters("V1", "exploratory") == ("wildcard_copies_strict_excluded_scoring",)
    assert T.REGISTERED_CLUSTER_UNITS["V5"] == ("system", "publication_group")
    assert T.REGISTERED_CLUSTER_UNITS["V2"] == ("metal_state",)


# --------------------------------------------------------------------------------------------- #
# TOST, BH, margins
# --------------------------------------------------------------------------------------------- #

def test_tost_verdicts():
    rng = np.random.default_rng(8)
    tight = _units({f"s{i}": list(rng.normal(0.0, 0.02, 2)) for i in range(30)})
    better = _units({f"s{i}": list(rng.normal(0.2, 0.1, 2)) for i in range(30)})
    wide = _units({f"s{i}": list(rng.normal(0.0, 0.6, 1)) for i in range(10)})
    assert T.tost(T.paired_cluster_bootstrap(*tight))["verdict"] == "NO_DIFFERENCE"
    t_better = T.tost(T.paired_cluster_bootstrap(*better))
    assert t_better["verdict"] == "PASS" and t_better["non_inferior"] and not t_better["equivalent"]
    t_wide = T.tost(T.paired_cluster_bootstrap(*wide))
    assert t_wide["verdict"] == "UNDECIDED" and t_wide["low_90"] <= -0.05
    worse = _units({f"s{i}": list(rng.normal(-0.3, 0.05, 2)) for i in range(30)})
    assert T.tost(T.paired_cluster_bootstrap(*worse))["verdict"] == "UNDECIDED"   # never "no effect"
    r = T.paired_cluster_bootstrap(*better)
    both = T.tost(r)
    assert both["low_90"] == pytest.approx(min(r.percentile_interval(0.9)[0], r.bca_interval(0.9)[0]))
    with pytest.raises(ValueError, match="primary-cluster"):
        T.tost(T.unclustered_bootstrap(better[0], better[1]))


def test_benjamini_hochberg_known_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005, 0.01, 0.03, 0.04 -> x 4/i: 0.02, 0.02, 0.04, 0.04 -> running min from the top unchanged
    assert T.benjamini_hochberg(p).tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    from scipy.stats import false_discovery_control
    q = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251, 0.269, 0.275,
                  0.34, 0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94, 0.942, 0.975, 0.986])
    assert np.allclose(T.benjamini_hochberg(q), false_discovery_control(q, method="bh"))
    with_nan = T.benjamini_hochberg([0.01, np.nan, 0.03])
    # a missing p counts toward m = 3: 0.01 * 3/1 = 0.03, 0.03 * 3/2 = 0.045
    assert math.isnan(with_nan[1]) and with_nan[0] == pytest.approx(0.03) and with_nan[2] == pytest.approx(0.045)
    assert T.benjamini_hochberg([0.01, np.nan, 0.03], nan_counts_toward_m=False).tolist()[0] == pytest.approx(0.02)


def test_delta5_formula_and_constants():
    assert T.N0 == pytest.approx(0.299 * math.sqrt(2 / math.pi)) and round(T.N0, 3) == 0.239
    assert T.delta5(0.8) == pytest.approx(0.2 * (0.8 - T.N0))           # 0.1123
    assert T.delta5(0.4) == 0.05                                         # 0.2 * 0.161 = 0.032 -> floor
    assert (T.RHO5, T.GAMMA5, T.ETA5, T.EPSILON, T.N_RESAMPLES, T.BOOTSTRAP_SEED) == (0.20, 0.05, 0.02, 0.05, 10_000, 19)
    assert T.stronger_lookup({"B3x": 0.61, "B3i": 0.58}) == ("B3i", 0.58)
    assert T.stronger_lookup({"B3x": 0.6, "B3i": 0.6}) == ("B3x", 0.6)


def test_superseded_s1c_check_raises_s1e_and_reliability_helpers():
    rng = np.random.default_rng(4)
    # the fixed-number S1(c) reading was redefined (section 9, 2026-09-15): the old helper must not be usable
    cand = _units({f"s{i}": list(rng.normal(0.1, 0.03, 3)) for i in range(15)})
    dir_boot = T.paired_cluster_bootstrap(cand[1], cand[0], cand[2], higher_is_better=True)
    with pytest.raises(NotImplementedError, match="redefined"):
        T.s1c_check(dir_boot, 0.03, 0.025)
    frame = pd.DataFrame({"mae": np.linspace(1, 0.2, 60) + rng.normal(0, 0.05, 60), "support": np.linspace(0, 1, 60),
                          "system": [f"s{i % 12}" for i in range(60)]})
    sp = T.cluster_bootstrap_statistic(frame, "system", lambda f: EM.spearman_rho(f["mae"], f["support"]),
                                       n_resamples=2000)
    assert T.s1e_check(sp, components_reliable=True)["verdict"] == "PASS"
    assert T.s1e_check(sp, components_reliable=False)["verdict"] == "FAIL"
    assert T.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert T.jackknife_reliability(1.0, 3.0) == pytest.approx(0.25)
    assert T.reliability_gate(0.25) == "UNDECIDED_UNRELIABLE" and T.reliability_gate(0.3) == "RELIABLE"
    assert T.reliability_gate(float("nan")) == "UNDECIDED_UNRELIABLE"


# --------------------------------------------------------------------------------------------- #
# signal injection
# --------------------------------------------------------------------------------------------- #

def _grid() -> pd.DataFrame:
    states = ["Nd(III)", "Pr(III)", "Am(III)", "Eu(III)"]
    systems = ["S1", "S2", "S3"]
    rows = [{EM.METAL_STATE_COL: m, EM.SYSTEM_COL: s} for m in states for s in systems for _ in range(2)]
    rows.append({EM.METAL_STATE_COL: None, EM.SYSTEM_COL: "S1"})
    return pd.DataFrame(rows, index=[f"r{i}" for i in range(len(rows))])


def test_injected_signal_structure_seed_and_standardisation():
    df = _grid()
    raw = T.injected_signal(df, seed=104729, standardise=False)
    rng = np.random.default_rng(104729)
    u = dict(zip(sorted(["Nd(III)", "Pr(III)", "Am(III)", "Eu(III)"]), rng.standard_normal(4)))
    v = dict(zip(["S1", "S2", "S3"], rng.standard_normal(3)))
    for i, r in df.iterrows():
        if isinstance(r[EM.METAL_STATE_COL], str):
            assert raw[i] == pytest.approx(u[r[EM.METAL_STATE_COL]] * v[r[EM.SYSTEM_COL]])
    assert math.isnan(raw.iloc[-1])                                        # unknown state: no signal
    s = T.injected_signal(df, seed=104729)
    assert s.equals(T.injected_signal(df, seed=104729))
    assert not s.equals(T.injected_signal(df, seed=130363))
    assert s.dropna().mean() == pytest.approx(0.0, abs=1e-12) and s.dropna().std(ddof=0) == pytest.approx(1.0)
    # rank-1 before the row standardisation: s(m1,s1) s(m2,s2) = s(m1,s2) s(m2,s1)
    cell = raw.groupby([df[EM.METAL_STATE_COL], df[EM.SYSTEM_COL]]).first()
    assert cell[("Nd(III)", "S1")] * cell[("Pr(III)", "S2")] == pytest.approx(cell[("Nd(III)", "S2")] * cell[("Pr(III)", "S1")])


def test_injected_signal_u_share_and_injection_guards():
    df = _grid()
    shared = T.injected_signal(df, seed=7, standardise=False, u_share={"Am(III)": "Eu(III)"})
    am = shared[df[EM.METAL_STATE_COL] == "Am(III)"].to_numpy()
    eu = shared[df[EM.METAL_STATE_COL] == "Eu(III)"].to_numpy()
    assert np.allclose(am, eu)
    y = pd.Series(1.0, index=df.index)
    with pytest.raises(ValueError, match="no injected signal"):
        T.inject_targets(y, shared, 0.25)
    known = df[EM.METAL_STATE_COL].notna()
    out = T.inject_targets(y[known], shared[known], 0.5)
    assert np.allclose(out, 1.0 + 0.5 * shared[known])
    with pytest.raises(ValueError, match="kappa"):
        T.inject_targets(y[known], shared[known], 0.3)
    # task X findings V-L3 / V-P04: the two NULL labels apply only to a contrast that IS a null (its un-injected
    # reported-scope verdict FAILs); the record also carries `informative` and what it knew about the un-injected run
    assert T.power_verdict({0.1: False, 0.25: True, 0.5: True, 1.0: True}) == \
        {"kappa_min": 0.25, "verdict": "INFORMATIVE_NULL", "informative": True, "uninjected_is_a_null": None,
         "uninjected_verdict": None}
    assert T.power_verdict({0.1: False, 0.25: False, 0.5: True, 1.0: True})["verdict"] == "UNDECIDED_UNDERPOWERED"
    assert T.power_verdict({0.1: False, 0.25: False, 0.5: False, 1.0: False})["kappa_min"] is None
    assert T.power_verdict({0.1: False, 0.25: True, 0.5: True, 1.0: True}, uninjected_verdict="FAIL")["verdict"] \
        == "INFORMATIVE_NULL"
    assert T.power_verdict({0.1: False, 0.25: True, 0.5: True, 1.0: True}, uninjected_verdict="PASS")["verdict"] \
        == "POWERED_NOT_A_NULL"
