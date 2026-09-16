"""Code consequences of the orchestrator's pre-registration resolutions of 2026-09-15 (task S and the last pre-seal
resolutions): the V1 outer-fold scoring unit and its remainder fold as ONE publication-group cluster (sections 3.2, 8),
the V0 F1 interval (section 3.6), the paired S1(c) evaluator with the registered "same fitted folds" reading (B3x / B3i
re-fitted on the candidate's batched V5-PAIR folds) and the seed combination at confirmation (section 9), the
wildcard-copy sensitivity registered for V5-P and V5-PAIR (section 8 R19 item 6), the B3x / B3i re-fit helper
(``models.s1c_yardsticks``), multi-seed conformal intervals of the deterministic arms (section 15) and the power-check
X(?) rule (section 8).  R19 item 3 and the s4 reading are tested in ``test_transfer_stats.py`` and ``test_support.py``.

Synthetic data with hand-worked values unless marked ``slow`` (which reads pre-seal outputs and scores nothing new).
No learned model is fitted, nothing runs on an outer fold of a registered design (the re-fit helper runs on a synthetic
mini-corpus whose fold files are written to pytest's ``tmp_path`` only), and no file of the tree is written.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import calibration as EC
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as EP
from gen19ct.evaluation import transfer as T
from gen19ct.models import baselines as B
from gen19ct.models import interface as I

SYS, STATE, PUB, CK = EM.SYSTEM_COL, EM.METAL_STATE_COL, EM.PUB_GROUP_COL, EM.CONDITION_KEY_COL


def _no_v6(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index)


# --------------------------------------------------------------------------------------------- #
# 1. V1 scoring unit = outer fold (section 3.2)
# --------------------------------------------------------------------------------------------- #

def _v1_frame(fold_ids: list[str]) -> pd.DataFrame:
    # fold gA: group gA, errors 1, 3 (MAE 2); fold gB: error 0.5; remainder fold: group r1 error 1, group r2 errors 0,0,0
    groups = ["gA", "gA", "gB", "r1", "r2", "r2", "r2"]
    err = [1.0, 3.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    n = len(groups)
    return pd.DataFrame({EM.FOLD_COL: fold_ids, PUB: groups, SYS: [f"S{i % 2}" for i in range(n)],
                         STATE: ["Nd(III)"] * n, CK: [f"c{i}" for i in range(n)], "log_D": [0.0] * n, "pred": err,
                         "lower_80": [e - 1.5 for e in err], "upper_80": [e + 1.5 for e in err]},
                        index=[f"r{i}" for i in range(n)])


EXACT_FOLDS = ["gA", "gA", "gB", "REMAINDER", "REMAINDER", "REMAINDER", "REMAINDER"]


def test_v1_scoring_units_map_fold_ids_and_check_groups():
    from gen19ct.folds import source_holdout as SH

    assert EM.V1_REMAINDER_UNIT == SH.REMAINDER
    fr = _v1_frame(EXACT_FOLDS)
    u = EM.v1_scoring_units(fr[EM.FOLD_COL], groups=fr[PUB], remainder_groups={"r1", "r2"})
    assert u.tolist() == EXACT_FOLDS and u.name == EM.V1_UNIT_COL
    # a single-group fold whose rows carry another group is refused (wrong grouping passed)
    bad = fr[PUB].copy()
    bad.iloc[1] = "gZ"
    with pytest.raises(ValueError, match="another group than the fold id"):
        EM.v1_scoring_units(fr[EM.FOLD_COL], groups=bad)
    with pytest.raises(ValueError, match="not a remainder group"):
        EM.v1_scoring_units(fr[EM.FOLD_COL], groups=fr[PUB], remainder_groups={"r1"})
    # grouped outer folds of the heavy arms: every row takes its exact-design unit
    grouped = pd.Series(["s1_f0", "s1_f1", "s1_f0", "s1_f1", "s1_f0", "s1_f1", "s1_f1"], index=fr.index)
    g = EM.v1_scoring_units(grouped, scheme="grouped", groups=fr[PUB], remainder_groups={"r1", "r2"})
    assert g.tolist() == EXACT_FOLDS
    with pytest.raises(ValueError, match="grouped scheme needs"):
        EM.v1_scoring_units(grouped, scheme="grouped", groups=fr[PUB])
    assert EM.registered_unit_cols("V1") == (EM.V1_UNIT_COL,) and EM.registered_unit_cols("V0") == (PUB,)
    with pytest.raises(ValueError):
        EM.registered_unit_cols("V5-PAIR")


def test_v1_design_summaries_use_the_outer_fold_and_print_the_group_reading():
    fr = _v1_frame(EXACT_FOLDS)
    reg = EM.Regime(design="V1", arm="B3")
    s = EM.design_logd_summary(fr, reg, v6_mask=_no_v6(fr))
    mae = s[(s["metric"] == "mae") & (s["aggregation"] == "unit_macro") & (s["stratum"] == "all")]
    registered = mae[mae["status"] == "registered"].iloc[0]
    exploratory = mae[mae["status"] == "exploratory"].iloc[0]
    # outer folds: gA 2, gB 0.5, REMAINDER (1 + 0 + 0 + 0) / 4 = 0.25 -> 2.75 / 3
    assert registered["value"] == pytest.approx(2.75 / 3) and registered["n_units"] == 3
    assert registered["averaging_unit"] == EM.V1_UNIT_COL and registered["unit_reading"].startswith("registered: outer fold")
    # exploratory per publication group: 2, 0.5, 1, 0 -> 0.875 over 4 units
    assert exploratory["value"] == pytest.approx(0.875) and exploratory["n_units"] == 4
    assert exploratory["unit_reading"].startswith("exploratory: publication group")
    # the registered reading alone, and the grouped heavy-arm folds scored on the same units give the same macro
    only = EM.design_logd_summary(fr, reg, v6_mask=_no_v6(fr), exploratory_unit_readings=False)
    assert set(only["status"]) == {"registered"}
    gfr = fr.assign(**{EM.FOLD_COL: ["s1_f0", "s1_f1", "s1_f0", "s1_f1", "s1_f0", "s1_f1", "s1_f1"]})
    gs = EM.design_logd_summary(gfr, EM.Regime(design="V1", arm="M2", seed=104729, seed_set="discovery"),
                                v6_mask=_no_v6(gfr), v1_scheme="grouped", remainder_groups={"r1", "r2"},
                                exploratory_unit_readings=False)
    assert EM.metric_value(gs, "mae") == pytest.approx(2.75 / 3)
    # intervals pred +- 1.5 around y = 0: covered unless |error| > 1.5 -> gA (1 yes, 3 no) 0.5, gB 1, REMAINDER 1 -> 2.5 / 3
    iv = EM.design_interval_summary(fr, reg, v6_mask=_no_v6(fr), levels=(0.8,), exploratory_unit_readings=False)
    assert EM.metric_value(iv, "coverage_80") == pytest.approx(2.5 / 3)
    # per-unit table and clusters: the remainder fold is ONE unit and ONE publication-group cluster
    pu = EM.design_per_unit_table(fr, "V1", v6_mask=_no_v6(fr))
    assert list(pu.index) == ["REMAINDER", "gA", "gB"] and pu.loc["REMAINDER", "mae"] == pytest.approx(0.25)
    cl = EM.design_unit_clusters(fr, "V1")
    assert list(cl) == ["publication_group"] and cl["publication_group"].to_dict() == {
        "REMAINDER": "REMAINDER", "gA": "gA", "gB": "gB"}
    assert set(EM.design_unit_clusters(fr, "V1", reading="publication_group")["publication_group"]) == {"gA", "gB", "r1", "r2"}
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        EM.design_per_unit_table(fr, "V1", v6_mask=pd.Series([True] + [False] * 6, index=fr.index))
    # a paired contrast over the registered units: comparator (B0) - candidate (B3) with the design's clusters
    b0 = EM.design_per_unit_table(fr.assign(pred=[2.0] * 7), "V1", v6_mask=_no_v6(fr))
    br = T.paired_cluster_bootstrap(b0["mae"], pu["mae"], cl["publication_group"], n_resamples=200)
    assert br.n_units == 3 and br.n_clusters == 3 and br.point == pytest.approx(2.0 - 2.75 / 3)


def test_v5_design_clusters_system_and_majority_group():
    fr = pd.DataFrame({STATE: ["Nd(III)"] * 3 + ["Pr(III)"] * 2, SYS: ["S1"] * 3 + ["S2"] * 2,
                       PUB: ["g2", "g1", "g1", "g3", "g4"]}, index=[f"r{i}" for i in range(5)])
    cl = EM.design_unit_clusters(fr, "V5")
    assert cl["system"].to_dict() == {"Nd(III) x S1": "S1", "Pr(III) x S2": "S2"}
    assert cl["publication_group"].to_dict() == {"Nd(III) x S1": "g1", "Pr(III) x S2": "g3"}   # tie -> smallest label


@pytest.mark.slow
def test_v1_outer_fold_unit_reproduces_preseal_remainder_as_one_unit_values():
    """On the pre-seal V1 predictions (selection half), B3 and B0 macro MAE under the registered outer-fold unit
    equal ``difficulty.json`` -> ``V1.unit_sensitivities_exploratory.remainder_fold_as_one_unit`` (B3 1.033, B0 1.142)
    and the exploratory per-group reading equals the as-run ``V1.macro_mae_selection``.  Re-scores existing
    predictions only."""
    import json

    from gen19ct import paths
    from gen19ct.data import load

    ev = paths.G19_ROOT / "evaluation" / "preseal"
    if not (ev / "predictions" / "V1__copy.parquet").exists():
        pytest.skip("pre-seal outputs absent")
    diff = json.loads((ev / "difficulty.json").read_text(encoding="utf-8"))["V1"]
    one_unit = diff["unit_sensitivities_exploratory"]["remainder_fold_as_one_unit"]["macro_mae_selection"]
    model = load.load_model_rows(copy=False)
    y = pd.Series(model["log_D"].to_numpy(dtype=float), index=model["canonical_measurement_id"].astype(str))
    p = pd.read_parquet(ev / "predictions" / "V1__copy.parquet", columns=["row_id", "fold_id", "arm", "half", "unit",
                                                                          "mean_logD"])
    for arm, rounded in (("B3", 1.033), ("B0", 1.142)):
        fr = p[(p["arm"] == arm) & (p["half"] == "S")].set_index("row_id", drop=False)
        fr = fr.assign(log_D=y.reindex(fr.index).to_numpy(), pred=fr["mean_logD"], **{PUB: fr["unit"]})
        v6 = pd.Series(False, index=fr.index)
        reg = EM.design_per_unit_table(fr, "V1", v6_mask=v6)
        grp = EM.design_per_unit_table(fr, "V1", v6_mask=v6, reading="publication_group")
        assert len(reg) == 39 and len(grp) == 65
        assert reg["mae"].mean() == pytest.approx(one_unit[arm], abs=1e-12) and round(reg["mae"].mean(), 3) == rounded
        assert grp["mae"].mean() == pytest.approx(diff["macro_mae_selection"][arm], abs=1e-12)
        # the pooled remainder fold is one unit AND one publication-group cluster; its 27 groups are not clusters
        cl = EM.design_unit_clusters(fr, "V1")["publication_group"]
        rem_groups = set(fr.loc[fr["fold_id"] == "REMAINDER", PUB])
        assert list(cl.index) == list(reg.index) and cl["REMAINDER"] == "REMAINDER"
        assert len(rem_groups) == 27 and not set(cl) & rem_groups and len(set(cl)) == 39


# --------------------------------------------------------------------------------------------- #
# 2. V0 F1 interval (section 3.6)
# --------------------------------------------------------------------------------------------- #

def _v0_tables():
    idx = pd.Index(["g1", "g2", "g3"])
    b3 = pd.DataFrame({104729: [1.0, 0.8, 0.5], 130363: [1.2, 0.6, np.nan]}, index=idx)
    m2 = pd.DataFrame({104729: [0.7, 0.9, 0.1], 130363: [1.0, 0.5, np.nan]}, index=idx)
    return b3, m2


def test_v0_seed_mean_bootstrap_hand_values_and_shared_draws():
    b3, m2 = _v0_tables()
    r = T.seed_mean_cluster_bootstrap(b3, m2, n_resamples=500)
    # Delta per group: g1 (0.3, 0.2), g2 (-0.1, 0.1), g3 (0.4, not scored); seed macros 0.2 and 0.15 -> 0.175
    assert r.point == pytest.approx(0.175) and r.cluster_unit == "publication_group" and r.n_clusters == 3
    # leave g1 out: (0.15 + 0.1) / 2; g2: (0.35 + 0.2) / 2; g3: (0.1 + 0.15) / 2
    assert r.jackknife.to_dict() == pytest.approx({"g1": 0.125, "g2": 0.275, "g3": 0.125})
    D = {"g1": (0.3, 0.2), "g2": (-0.1, 0.1), "g3": (0.4, None)}
    picks = T.draw_cluster_picks(3, 500, 19)
    names = ["g1", "g2", "g3"]
    for row in (0, 1, 7, 250, 499):
        drawn = [names[j] for j in picks[row]]
        per_seed = []
        for s in (0, 1):
            vals = [D[g][s] for g in drawn if D[g][s] is not None]
            per_seed.append(np.mean(vals) if vals else np.nan)
        want = np.mean(per_seed)
        assert (math.isnan(want) and math.isnan(r.draws[row])) or r.draws[row] == pytest.approx(want)
    # the same resampled groups in every seed: identical seed columns give exactly the single-seed cluster bootstrap
    one = pd.DataFrame({1: [1.0, 0.8, 0.5, 0.9], 2: [1.0, 0.8, 0.5, 0.9]}, index=["a", "b", "c", "d"])
    zero = one * 0.0 + 0.2
    both = T.seed_mean_cluster_bootstrap(one, zero)
    ref = T.paired_cluster_bootstrap(one[1], zero[1], pd.Series(one.index, index=one.index))
    assert np.allclose(both.draws, ref.draws) and both.point == pytest.approx(ref.point)
    assert both.n_resamples == 10_000 and both.seed == 19
    with pytest.raises(ValueError, match="same groups in every seed"):
        T.seed_mean_cluster_bootstrap(b3, m2.fillna(0.0))


def test_f1_check_reads_the_v0_percentile_interval():
    rng = np.random.default_rng(3)
    groups = [f"g{i:02d}" for i in range(40)]
    b3 = pd.DataFrame({s: rng.normal(1.0, 0.1, 40) for s in (1, 2, 3, 4, 5)}, index=groups)
    better = b3 - 0.3 + pd.DataFrame(rng.normal(0, 0.05, (40, 5)), index=groups, columns=b3.columns)
    v0 = T.seed_mean_cluster_bootstrap(b3, better)
    assert v0.percentile_interval()[0] > 0
    idx = pd.Index(groups)
    v1_null = T.paired_cluster_bootstrap(pd.Series(rng.normal(0, 0.3, 40), index=idx), pd.Series(0.0, index=idx),
                                         pd.Series(groups, index=idx), cluster_unit="publication_group")
    v1_gain = T.paired_cluster_bootstrap(pd.Series(rng.normal(0.4, 0.05, 40), index=idx), pd.Series(0.0, index=idx),
                                         pd.Series(groups, index=idx), cluster_unit="publication_group")
    assert T.f1_check(v0, s1a_passed=False, v1=v1_null)["failure"] is True
    assert T.f1_check(v0, s1a_passed=True, v1=v1_null)["failure"] is False
    assert T.f1_check(v0, s1a_passed=False, v1=v1_gain)["failure"] is False
    worse = T.seed_mean_cluster_bootstrap(b3, b3 + 0.2)
    assert T.f1_check(worse, s1a_passed=False, v1=v1_null)["failure"] is False
    with pytest.raises(ValueError, match="registered publication-group"):
        T.f1_check(T.seed_mean_cluster_bootstrap(b3, better, n_resamples=100), s1a_passed=False, v1=v1_null)

# --------------------------------------------------------------------------------------------- #
# 3. S1(c) paired evaluator (section 9 redefinition; "same fitted folds" and the seed combination resolved 2026-09-15)
# --------------------------------------------------------------------------------------------- #

def _hand_pairs():
    rows = [("S1", "Nd(III)", "Pr(III)", "Ln-Ln", 0.5), ("S1", "Nd(III)", "Pr(III)", "Ln-Ln", -0.4),
            ("S1", "Nd(III)", "Pr(III)", "Ln-Ln", 0.1), ("S1", "Am(III)", "Eu(III)", "An-Ln", 1.0),
            ("S1", "Am(III)", "Eu(III)", "An-Ln", 0.6), ("S2", "Sm(III)", "Nd(III)", "Ln-Ln", 0.8),
            ("S2", "Sm(III)", "Nd(III)", "Ln-Ln", 0.35)]
    pairs = pd.DataFrame(rows, columns=[SYS, "state_a", "state_b", "category_class", "logsf_obs"])
    pairs["fold"] = ["F1" if sy == "S1" else "F2" for sy in pairs[SYS]]
    pairs["idx_a"] = [f"{f}|a{i}" for i, f in enumerate(pairs["fold"])]            # fold-qualified member labels
    pairs["idx_b"] = [f"{f}|b{i}" for i, f in enumerate(pairs["fold"])]
    pairs["half"] = "confirmation"                                                  # the half every pair belongs to
    s = lambda v: pd.Series(v, index=pairs.index)  # noqa: E731
    cand = s([0.2, -0.1, 0.3, 0.5, -0.2, 0.1, 0.0])
    b3x = s([0.1, 0.1, 0.0, 0.4, 0.3, -0.2, 0.2])
    b3i = s([0.3, -0.3, 0.1, 0.0, 0.1, 0.4, 0.1])
    return pairs, cand, b3x, b3i


BATCHED_ID = T.s1c_fold_design_id("V5PAIR__primary__batched", "a" * 64)
UNBATCHED_ID = T.s1c_fold_design_id("V5PAIR__primary__cell_pair", "b" * 64)
DESIGNS = {k: BATCHED_ID for k in T.S1C_FOLD_DESIGN_KEYS}
#: five withheld-seed stand-ins (never discovery seeds)
WITHHELD = (123457, 234567, 345679, 456781, 567891)


def _guards(pairs: pd.DataFrame, **over) -> dict:
    """The S1(c) guard arguments of a test pair frame: every member in the test fold, none in V6_TARGET_ROWS, every arm
    fitted on one batched V5-PAIR fold design."""
    labels = list(pairs["idx_a"]) + list(pairs["idx_b"])
    kw = {"folds": pd.Series("test", index=labels), "v6_mask": pd.Series(False, index=labels), "fold_designs": DESIGNS}
    kw.update(over)
    return kw


def test_s1c_registered_fold_reading_constants():
    from gen19ct.folds import io as FI

    assert T.S1C_REGISTERED_FOLD_READING == "yardsticks_refit_on_candidate_batched_folds"
    assert set(T.S1C_FOLD_READINGS) == {T.S1C_REGISTERED_FOLD_READING}
    assert "re-fitted" in T.S1C_FOLD_READINGS[T.S1C_REGISTERED_FOLD_READING]
    assert T.S1C_REFIT_YARDSTICKS == ("B3x", "B3i") and "HEAVIER" in T.S1C_UNFITTED_RULES
    assert T.DISCOVERY_SEEDS == FI.DISCOVERY_SEEDS and T.S1C_SELECTION_SEED == FI.DISCOVERY_SEEDS[0]
    assert T.is_batched_v5pair_design(BATCHED_ID) and T.is_batched_v5pair_design("V5PAIR__primary__batched_max4")
    assert not T.is_batched_v5pair_design(UNBATCHED_ID) and not T.is_batched_v5pair_design("V5P__base__batched")
    assert T.split_fold_label(T.fold_qualified_label("s1_S_b000", "ID:1")) == ("s1_S_b000", "ID:1")
    for bad in ("ID:1", "|ID:1", "F1|", 7, None):
        with pytest.raises(ValueError, match="fold-qualified"):
            T.split_fold_label(bad)
    # a fold design id carries the folds.io.design_hash of the fitted file (task X verification, finding 5)
    assert T.split_s1c_fold_design_id(BATCHED_ID) == ("V5PAIR__primary__batched", "a" * 64)
    for bad_hash in ("", "a" * 63, "A" * 64, "g" * 64, "a" * 65):
        with pytest.raises(ValueError):
            T.s1c_fold_design_id("V5PAIR__primary__batched", bad_hash)
        assert not T.is_batched_v5pair_design(f"V5PAIR__primary__batched@{bad_hash}")      # malformed: not a design
    for bad in ("V5PAIR__primary__batched", "V5PAIR__primary__batched@", "@" + "a" * 64, BATCHED_ID + "@" + "a" * 64,
                None, 7):
        with pytest.raises(ValueError, match="not a fold design id"):
            T.split_s1c_fold_design_id(bad)


def test_s1c_half_hand_values_on_identical_pairs():
    pairs, cand, b3x, b3i = _hand_pairs()
    h = T.s1c_half(pairs, cand, half="confirmation", seed=None, lookup_logsf={"B3x": b3x, "B3i": b3i}, n_resamples=200,
                   **_guards(pairs))
    assert h["fold_reading"] == T.S1C_REGISTERED_FOLD_READING and h["n_pairs_excluded_by_filter"] == 0
    assert {k: h["fold_designs"][k] for k in DESIGNS} == DESIGNS and h["fold_design"] == BATCHED_ID
    assert h["fold_designs"]["HEAVIER"].startswith("no fit")
    d = h["direction"]
    # HEAVIER is defined on the Ln-Ln cell pairs only: candidate (1 + 0.75) / 2, HEAVIER (0.5 + 1) / 2
    assert d["HEAVIER"]["candidate_accuracy"] == pytest.approx(0.875)
    assert d["HEAVIER"]["yardstick_accuracy"] == pytest.approx(0.75) and d["HEAVIER"]["delta"] == pytest.approx(0.125)
    assert d["HEAVIER"]["n_pairs"] == 4 and d["HEAVIER"]["n_pairs_yardstick_undefined"] == 2
    assert d["HEAVIER"]["n_cell_pairs"] == 2
    # B3x / B3i on every qualifying pair: candidate (1 + 0.5 + 0.75) / 3; B3x (0.5 + 1 + 0.5) / 3; B3i (1 + 0.75 + 1) / 3
    assert d["B3x"]["candidate_accuracy"] == pytest.approx(0.75) and d["B3x"]["delta"] == pytest.approx(0.75 - 2 / 3)
    assert d["B3i"]["delta"] == pytest.approx(0.75 - 2.75 / 3) and d["B3i"]["n_pairs"] == 6
    assert h["min_delta_yardstick"] == "B3i" and h["min_delta"] == pytest.approx(0.75 - 2.75 / 3)
    # the per-cell-pair table the seed combination reads
    per = d["B3x"]["per_cell_pair"]
    assert per["candidate"].to_dict() == pytest.approx({"S1 x Am(III) x Eu(III)": 0.5, "S1 x Nd(III) x Pr(III)": 1.0,
                                                       "S2 x Sm(III) x Nd(III)": 0.75})
    assert per["yardstick"].to_dict() == pytest.approx({"S1 x Am(III) x Eu(III)": 1.0, "S1 x Nd(III) x Pr(III)": 0.5,
                                                       "S2 x Sm(III) x Nd(III)": 0.5})
    assert per["system"].to_dict() == {"S1 x Am(III) x Eu(III)": "S1", "S1 x Nd(III) x Pr(III)": "S1",
                                       "S2 x Sm(III) x Nd(III)": "S2"} and per["n_pairs"].sum() == 6
    # the yardstick side equals the tidy pair summary's unit-macro direction accuracy of B3x
    folds = pd.Series("test", index=list(pairs["idx_a"]) + list(pairs["idx_b"]))
    ps = EP.pair_summary(pairs, b3x, EM.Regime(design="V5-PAIR", arm="B3x", half="confirmation"),
                         v6_mask=pd.Series(False, index=folds.index), folds=folds)
    assert EM.metric_value(ps, "direction_accuracy_abs_ge_0.3") == pytest.approx(d["B3x"]["yardstick_accuracy"])
    # logSF MAE on all 7 pairs, cell-pair macro: candidate (0.8/3 + 0.65 + 0.525) / 3; FLAT (1/3 + 0.8 + 0.575) / 3;
    # B3i (0.1 + 0.75 + 0.325) / 3
    cand_mae = (0.8 / 3 + 0.65 + 0.525) / 3
    assert h["logsf_mae"]["FLAT"]["gain"] == pytest.approx((1 / 3 + 0.8 + 0.575) / 3 - cand_mae)
    assert h["logsf_mae"]["B3i"]["gain"] == pytest.approx((0.1 + 0.75 + 0.325) / 3 - cand_mae)
    assert h["logsf_mae"]["B3i"]["per_cell_pair"]["candidate_logsf_mae"].mean() == pytest.approx(cand_mae)
    # the pair set identity strips the fold part of the labels
    moved = pairs.assign(idx_a=pairs["idx_a"].str.replace("F", "G"), idx_b=pairs["idx_b"].str.replace("F", "G"),
                         fold=pairs["fold"].str.replace("F", "G"))
    hm = T.s1c_half(moved, cand, half="confirmation", seed=None, lookup_logsf={"B3x": b3x, "B3i": b3i}, n_resamples=50,
                    **_guards(moved))
    assert hm["pair_set_sha256"] == h["pair_set_sha256"]
    with pytest.raises(ValueError, match="predict every pair"):
        T.s1c_half(pairs, cand.where(cand.index != 0), half="confirmation", seed=None,
                   lookup_logsf={"B3x": b3x, "B3i": b3i}, n_resamples=50, **_guards(pairs))


def _base_pairs(rng, *, n_systems: int = 12, classes=("Ln-Ln",)) -> pd.DataFrame:
    """Comparable row pairs of one half (row ids without a fold): 10 pairs per cell pair, observed logSF +-1."""
    rows = []
    for s in range(n_systems):
        for k, cls in enumerate(classes * 2):
            a, b = (("Nd(III)", "Pr(III)") if k % 2 == 0 else ("Sm(III)", "Nd(III)")) if cls == "Ln-Ln" \
                else (("Am(III)", "Eu(III)") if k % 2 == 0 else ("Cm(III)", "Eu(III)"))
            for j in range(10):
                rows.append((f"S{s:02d}", a, b, cls, float(rng.choice([-1.0, 1.0])), f"a{s}_{k}_{j}", f"b{s}_{k}_{j}"))
    return pd.DataFrame(rows, columns=[SYS, "state_a", "state_b", "category_class", "logsf_obs", "row_a", "row_b"])


def _half(base: pd.DataFrame, rng, *, half: str, seed: int, candidate_right: float, magnitude: float = 0.9,
          n_resamples: int = 500, **kw) -> dict:
    """One fitted run of a half: the base pairs under the seed's own batched fold ids and fold design, a candidate right
    in sign on a fraction of pairs, and re-fitted lookups (random signs)."""
    fold = f"s{seed}_b" + base[SYS]
    pairs = base.drop(columns=["row_a", "row_b"]).assign(fold=fold, idx_a=fold + "|" + base["row_a"],
                                                         idx_b=fold + "|" + base["row_b"],
                                                         half=half)
    n, obs = len(pairs), pairs["logsf_obs"].to_numpy()
    ser = lambda v: pd.Series(v, index=pairs.index)  # noqa: E731
    cand = ser(np.where(rng.random(n) < candidate_right, obs * magnitude, -obs * magnitude))
    look = {"B3x": ser(rng.choice([-0.5, 0.5], n)), "B3i": ser(rng.choice([-0.5, 0.5], n))}
    design = T.s1c_fold_design_id("V5PAIR__primary__batched", f"{seed:064x}")
    return T.s1c_half(pairs, cand, half=half, seed=seed, lookup_logsf=look, n_resamples=n_resamples,
                      **_guards(pairs, fold_designs={k: design for k in T.S1C_FOLD_DESIGN_KEYS}), **kw)


def _reference_seed_mean_bootstrap(delta: pd.DataFrame, members: dict, n_resamples: int,
                                   seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    """The seed-mean cluster bootstrap written out from its definition (sections 3.6 and 9), independent of
    ``transfer``: ``delta`` = per-unit comparator - candidate (units x seeds, NaN = not scored in that seed), ``members``
    = cluster -> units.  The clusters, sorted by label, are drawn once per resample with
    ``default_rng(seed).integers(0, G, size=(n_resamples, G))`` and applied to every seed; per seed the macro is the
    plain mean of the scored units of the drawn clusters (a cluster drawn twice counts twice, no unit -> NaN); the
    statistic is the mean over seeds.  The jackknife leaves one cluster out of every seed."""
    names = sorted(members)
    cols = list(delta.columns)
    val = {(u, c): float(delta.at[u, c]) for u in delta.index for c in cols}

    def stat(clusters: list) -> float:
        per_seed = []
        for c in cols:
            xs = [val[(u, c)] for k in clusters for u in members[k] if not math.isnan(val[(u, c)])]
            per_seed.append(sum(xs) / len(xs) if xs else float("nan"))
        return sum(per_seed) / len(per_seed)
    picks = np.random.default_rng(seed).integers(0, len(names), size=(n_resamples, len(names)))
    draws = np.array([stat([names[j] for j in row]) for row in picks])
    jk = np.array([stat([k for k in names if k != left]) for left in names])
    return stat(names), draws, jk


def test_seed_mean_nested_cluster_bootstrap_hand_values():
    # units u1, u2 in system A, u3 in B; two seeds; comparator - candidate per unit
    idx = pd.Index(["u1", "u2", "u3"])
    comp = pd.DataFrame({1: [0.5, 0.3, 0.2], 2: [0.4, np.nan, 0.1]}, index=idx)
    cand = pd.DataFrame({1: [0.1, 0.2, 0.4], 2: [0.1, np.nan, 0.3]}, index=idx)
    clusters = pd.Series(["A", "A", "B"], index=idx)
    r = T.seed_mean_nested_cluster_bootstrap(comp, cand, clusters, n_resamples=400)
    D = {"u1": (0.4, 0.3), "u2": (0.1, None), "u3": (-0.2, -0.2)}
    # seed 1: (0.4 + 0.1 - 0.2) / 3 = 0.1; seed 2: (0.3 - 0.2) / 2 = 0.05 -> 0.075
    assert r.point == pytest.approx(0.075) and r.n_units == 3 and r.n_clusters == 2 and r.cluster_unit == "system"
    # leave A out: (-0.2 + -0.2) / 2; leave B out: ((0.5 / 2) + 0.3) / 2
    assert r.jackknife.to_dict() == pytest.approx({"A": -0.2, "B": 0.275})
    members = {0: ["u1", "u2"], 1: ["u3"]}
    picks = T.draw_cluster_picks(2, 400, 19)
    for row in (0, 3, 99, 399):
        per_seed = []
        for s in (0, 1):
            vals = [D[u][s] for j in picks[row] for u in members[int(j)] if D[u][s] is not None]
            per_seed.append(np.mean(vals))
        assert r.draws[row] == pytest.approx(np.mean(per_seed))
    # every one of the draws, the point and the jackknife equal an independent per-draw recomputation of the definition
    # (its own default_rng(19) draw matrix, plain Python means); a check against the definition, since both bootstrap
    # functions now call one core and the pre-sharing V0 code is not kept (task X verification, finding 9)
    ref_point, ref_draws, ref_jk = _reference_seed_mean_bootstrap(comp - cand, {"A": ["u1", "u2"], "B": ["u3"]}, 400, 19)
    assert r.point == pytest.approx(ref_point, abs=1e-12)
    np.testing.assert_allclose(r.draws, ref_draws, rtol=0, atol=1e-12)
    np.testing.assert_allclose(r.jackknife.to_numpy(), ref_jk, rtol=0, atol=1e-12)
    # one unit per cluster: the V0 seed-mean bootstrap and the nested one both equal the reference on all 10,000 draws
    b3, m2 = _v0_tables()
    v0 = T.seed_mean_cluster_bootstrap(b3, m2)
    nested = T.seed_mean_nested_cluster_bootstrap(b3, m2, pd.Series(b3.index, index=b3.index), cluster_unit="publication_group")
    ref_point, ref_draws, ref_jk = _reference_seed_mean_bootstrap(b3 - m2, {g: [g] for g in b3.index}, 10_000, 19)
    assert np.isnan(ref_draws).any()                                   # draws holding only g3 leave seed 130363 empty
    for res in (v0, nested):
        assert res.point == pytest.approx(ref_point, abs=1e-12) and res.n_resamples == 10_000
        np.testing.assert_allclose(res.draws, ref_draws, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res.jackknife.to_numpy(), ref_jk, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="same units in every seed"):
        T.seed_mean_nested_cluster_bootstrap(comp, cand.fillna(0.0), clusters)
    with pytest.raises(ValueError, match="every scored unit needs a cluster"):
        T.seed_mean_nested_cluster_bootstrap(comp, cand, clusters.iloc[:2])


def test_s1c_seed_combination_seed_mean_and_shared_system_draws():
    rng = np.random.default_rng(130363)
    base = _base_pairs(rng)
    conf = [_half(base, rng, half="confirmation", seed=s, candidate_right=0.9, n_resamples=200) for s in WITHHELD[::-1]]
    comb = T.s1c_seed_combination(conf)
    assert comb["seeds"] == sorted(WITHHELD) and comb["n_pairs"] == len(base)
    systems = sorted(base[SYS].unique())
    picks = T.draw_cluster_picks(len(systems))
    for y in T.S1C_YARDSTICKS:
        c = comb["direction"][y]
        by_seed = {h["seed"]: h["direction"][y] for h in conf}
        assert c["per_seed_delta"] == pytest.approx({s: by_seed[s]["delta"] for s in WITHHELD})
        assert c["delta"] == pytest.approx(np.mean([by_seed[s]["delta"] for s in WITHHELD]))
        assert c["bootstrap"].n_resamples == 10_000 and c["bootstrap"].seed == 19 and c["n_systems"] == len(systems)
        assert c["n_seeds_positive"] == 5 and c["all_seeds_positive"] is True
        # a draw by hand: the SAME drawn systems in every seed, per-seed cell-pair macro, then the mean over seeds
        for row in (0, 4321):
            drawn = [systems[int(j)] for j in picks[row]]
            per_seed = []
            for s in sorted(WITHHELD):
                t = by_seed[s]["per_cell_pair"]
                vals = [v for sy in drawn for v in (t.loc[t["system"] == sy, "candidate"]
                                                   - t.loc[t["system"] == sy, "yardstick"])]
                per_seed.append(np.mean(vals))
            assert c["bootstrap"].draws[row] == pytest.approx(np.mean(per_seed))
    g = comb["logsf_mae"]["FLAT"]
    assert g["gain"] == pytest.approx(np.mean([h["logsf_mae"]["FLAT"]["gain"] for h in conf]))
    # five copies of one run: the seed-mean draws equal the single-run cluster bootstrap of that run
    one = conf[0]
    ids = {s: T.s1c_fold_design_id("V5PAIR__primary__batched", f"{s:064x}") for s in WITHHELD}
    copies = [{**one, "seed": s, "fold_design": ids[s],
               "fold_designs": {**one["fold_designs"], **dict.fromkeys(T.S1C_FOLD_DESIGN_KEYS, ids[s])}} for s in WITHHELD]
    same = T.s1c_seed_combination(copies)["direction"]["B3i"]
    t = one["direction"]["B3i"]["per_cell_pair"]
    ref = T.paired_cluster_bootstrap(t["yardstick"], t["candidate"], t["system"], higher_is_better=True)
    assert np.allclose(same["bootstrap"].draws, ref.draws) and same["delta"] == pytest.approx(ref.point)
    # refusals: not 5 seeds, a discovery seed, duplicate seeds or fold designs, another half, another pair set / filter
    with pytest.raises(ValueError, match="one s1c_half result per withheld seed"):
        T.s1c_seed_combination(conf[:4])
    with pytest.raises(TypeError):
        T.s1c_seed_combination(conf[0])
    with pytest.raises(ValueError, match="not on discovery seeds"):
        T.s1c_seed_combination(conf[:4] + [{**conf[4], "seed": 104729}])
    with pytest.raises(ValueError, match="distinct"):
        T.s1c_seed_combination(conf[:4] + [{**conf[4], "seed": conf[0]["seed"]}])
    with pytest.raises(ValueError, match="its own batched V5-PAIR folds"):
        T.s1c_seed_combination(conf[:4] + [{**conf[4], "fold_designs": conf[0]["fold_designs"]}])
    # every half's fold designs are re-checked: bare stems (the same for every seed) are refused even when all five
    # halves are bare, and so is a half whose yardstick was fitted on other folds than its candidate
    bare = [{**h, "fold_designs": {**h["fold_designs"], **dict.fromkeys(T.S1C_FOLD_DESIGN_KEYS, "V5PAIR__primary__batched")}}
            for h in conf]
    with pytest.raises(ValueError, match="design hash"):
        T.s1c_seed_combination(bare)
    with pytest.raises(AssertionError, match="same fitted folds"):
        T.s1c_seed_combination(conf[:4] + [{**conf[4], "fold_designs": {**conf[4]["fold_designs"], "B3x": BATCHED_ID}}])
    with pytest.raises(ValueError, match="confirmation half"):
        T.s1c_seed_combination(conf[:4] + [{**conf[4], "half": "selection"}])
    other = _half(_base_pairs(np.random.default_rng(1)), rng, half="confirmation", seed=conf[4]["seed"], candidate_right=0.9,
                  n_resamples=50)                                  # same row ids, other observed logSF
    assert other["pair_set_sha256"] != conf[4]["pair_set_sha256"]
    with pytest.raises(ValueError, match="identical confirmation pair set"):
        T.s1c_seed_combination(conf[:4] + [other])
    with pytest.raises(ValueError, match="scoring filters"):
        T.s1c_seed_combination(conf[:4] + [{**conf[4], "scoring_filter": T.WILDCARD_COPY_SENSITIVITY}])


def test_s1c_paired_verdict_pass_fail_undecided():
    rng = np.random.default_rng(104729)
    base, sel_base = _base_pairs(rng), _base_pairs(rng)
    conf = [_half(base, rng, half="confirmation", seed=s, candidate_right=1.0) for s in WITHHELD]
    sel = _half(sel_base, rng, half="selection", seed=104729, candidate_right=1.0)
    v = T.s1c_paired_verdict(confirmation=conf, selection=sel)
    assert v["verdict"] == "PASS" and all(v["checks"].values()) and v["fold_reading_registered"] == T.S1C_REGISTERED_FOLD_READING
    comb = v["confirmation_combined"]
    assert comb["min_delta"] >= 0.05 and all(d["interval_excludes_zero"] for d in comb["direction"].values())
    assert set(v["checks"]) >= {"confirmation_delta_positive_in_5_of_5_seeds", "confirmation_logsf_mae_below_FLAT_by_eta5",
                                "confirmation_logsf_mae_below_B3i_by_eta5", "selection_min_delta_at_least_-0.02"}
    # selection counterweight: a candidate far below the yardsticks on the selection half fails S1(c)
    bad_sel = _half(sel_base, rng, half="selection", seed=104729, candidate_right=0.2)
    assert bad_sel["min_delta"] < -0.02
    vf = T.s1c_paired_verdict(confirmation=conf, selection=bad_sel)
    assert vf["verdict"] == "FAIL" and vf["checks"]["selection_min_delta_at_least_-0.02"] is False
    # an evaluation that did not happen as registered is refused, never recorded as a FAIL (task X verification,
    # findings 6 and 8): the selection half on another seed than 104729, a result of another half passed as the
    # selection half, a selection run on the fold design of a withheld seed, or fold designs named without their hash
    for bad_seed in (130363, 262147, None, True, "104729"):
        with pytest.raises(ValueError, match="discovery seed 104729"):
            T.s1c_paired_verdict(confirmation=conf, selection={**sel, "seed": bad_seed})
    with pytest.raises(ValueError, match="counterweight is the selection half"):
        T.s1c_paired_verdict(confirmation=conf, selection={**sel, "half": "confirmation"})
    with pytest.raises(ValueError, match="counterweight is the selection half"):
        T.s1c_paired_verdict(confirmation=None, selection=conf[0])
    with pytest.raises(ValueError, match="fold design of a withheld seed"):
        T.s1c_paired_verdict(confirmation=conf, selection={**sel, "fold_designs": conf[2]["fold_designs"]})
    with pytest.raises(ValueError, match="design hash"):
        T.s1c_paired_verdict(confirmation=conf, selection={
            **sel, "fold_designs": dict.fromkeys(T.S1C_FOLD_DESIGN_KEYS, "V5PAIR__primary__batched")})
    with pytest.raises(TypeError, match="one s1c_half result"):
        T.s1c_paired_verdict(confirmation=conf, selection=[sel])
    # 5 of 5 withheld seeds: one seed with the candidate wrong everywhere fails although the seed mean stays >= gamma5
    wrong = conf[:4] + [_half(base, rng, half="confirmation", seed=WITHHELD[4], candidate_right=0.0)]
    vs = T.s1c_paired_verdict(confirmation=wrong, selection=sel)
    assert vs["verdict"] == "FAIL" and vs["checks"]["confirmation_delta_positive_in_5_of_5_seeds"] is False
    assert vs["checks"]["confirmation_min_delta_at_least_gamma5"] is True
    assert vs["confirmation_combined"]["direction"]["HEAVIER"]["n_seeds_positive"] == 4
    # the logSF MAE part: a candidate right in sign but no better than FLAT in magnitude fails eta5 (seed mean)
    flat_like = [_half(base, rng, half="confirmation", seed=s, candidate_right=1.0, magnitude=2.1) for s in WITHHELD]
    vm = T.s1c_paired_verdict(confirmation=flat_like, selection=sel)
    assert vm["verdict"] == "FAIL" and vm["checks"]["confirmation_logsf_mae_below_FLAT_by_eta5"] is False
    # untestable: V5-PAIR dropped, a missing half, or no pair where HEAVIER is defined (all An-Ln) -> UNDECIDED
    assert T.s1c_paired_verdict(confirmation=conf, selection=sel, v5_pair_dropped=True)["verdict"] == "UNDECIDED"
    assert T.s1c_paired_verdict(confirmation=None, selection=sel)["verdict"] == "UNDECIDED"
    assert T.s1c_paired_verdict(confirmation=conf, selection=None)["verdict"] == "UNDECIDED"
    an_base = _base_pairs(rng, classes=("An-Ln",))
    an = [_half(an_base, rng, half="confirmation", seed=s, candidate_right=1.0, n_resamples=100) for s in WITHHELD]
    assert not an[0]["direction"]["HEAVIER"]["testable"] and an[0]["direction"]["HEAVIER"]["n_pairs"] == 0
    va = T.s1c_paired_verdict(confirmation=an, selection=sel)
    assert va["verdict"] == "UNDECIDED" and va["checks"]["confirmation_min_delta_at_least_gamma5"] is None
    # refused: one confirmation half instead of the five seeds, a half under an unregistered reading, mixed filters
    with pytest.raises(TypeError, match="5 per-seed"):
        T.s1c_paired_verdict(confirmation=conf[0], selection=sel)
    with pytest.raises(ValueError, match="registered reading"):
        T.s1c_paired_verdict(confirmation=conf, selection={**sel, "fold_reading": "same_pairs"})
    with pytest.raises(ValueError, match="scoring filter"):
        T.s1c_paired_verdict(confirmation=conf, selection={**sel, "scoring_filter": T.WILDCARD_COPY_SENSITIVITY})
    with pytest.raises(ValueError, match="half"):
        _half(base, rng, half="S", seed=104729, candidate_right=1.0)


def test_s1c_guards_refuse_cross_fold_training_v6_members_unqualified_exclusions_and_other_fold_designs():
    """Task X: leakage finding VR-01 (the S1(c) evaluator ran no pair guard) and consistency finding VR-02 (nothing tied
    the arms to one fold design), resolved 2026-09-15: B3x / B3i re-fitted on exactly the candidate's batched V5-PAIR
    folds.  Each corruption that the old evaluator accepted silently is refused."""
    pairs, cand, b3x, b3i = _hand_pairs()
    look = {"B3x": b3x, "B3i": b3i}
    base = T.s1c_half(pairs, cand, half="confirmation", seed=None, lookup_logsf=look, n_resamples=50, **_guards(pairs))
    # (1) member b predicted in another fold than the pair's (a row scored in several V5-PAIR folds)
    cross = pairs.copy()
    cross.loc[0, "idx_b"] = "F2|b0"
    with pytest.raises(AssertionError, match="another fold than the pair"):
        T.s1c_half(cross, cand, half="confirmation", seed=None, lookup_logsf=look, n_resamples=50, **_guards(cross))
    # (2) member a is a training row of its fold
    folds = _guards(pairs)["folds"].copy()
    folds[pairs.loc[1, "idx_a"]] = "train"
    with pytest.raises(AssertionError, match="pair isolation violated|another fold role"):
        T.s1c_half(pairs, cand, half="confirmation", seed=None, lookup_logsf=look, n_resamples=50,
                   **_guards(pairs, folds=folds))
    # (3) a member in V6_TARGET_ROWS; (4) a mask keyed by row id instead of the fold-qualified labels
    v6 = _guards(pairs)["v6_mask"].copy()
    v6[pairs.loc[2, "idx_b"]] = True
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        T.s1c_half(pairs, cand, half="confirmation", seed=None, lookup_logsf=look, n_resamples=50,
                   **_guards(pairs, v6_mask=v6))
    with pytest.raises(ValueError, match="does not cover"):
        T.s1c_half(pairs, cand, half="confirmation", seed=None, lookup_logsf=look, n_resamples=50,
                   **_guards(pairs, v6_mask=pd.Series(False, index=["a0", "b0"])))
    # (5) unqualified member labels; (6) the helpers guard on their own
    plain = pairs.assign(idx_a=[f"a{i}" for i in range(len(pairs))], idx_b=[f"b{i}" for i in range(len(pairs))])
    with pytest.raises(ValueError, match="fold-qualified"):
        T.s1c_half(plain, cand, half="confirmation", seed=None, lookup_logsf=look, n_resamples=50, **_guards(plain))
    with pytest.raises(AssertionError, match="another fold than the pair"):
        T.paired_logsf_mae_gain(cross, cand, b3i, comparator="B3i", n_resamples=50,
                                folds=_guards(cross)["folds"], v6_mask=_guards(cross)["v6_mask"])
    with pytest.raises(TypeError):
        T.paired_direction_contrast(pairs, cand, b3x, yardstick="B3x", yardstick_direction_only=False, n_resamples=50,
                                    folds=_guards(pairs)["folds"], v6_mask=None)
    # (7) "same fitted folds": B3x from the unbatched V5-PAIR folds while the candidate used the batched ones
    kw = dict(half="confirmation", seed=None, lookup_logsf=look, n_resamples=50)
    with pytest.raises(AssertionError, match="different outer fold designs"):
        T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs={**DESIGNS, "B3x": UNBATCHED_ID}))
    # ... or re-fitted on the batched folds of another seed; or a pair set of other folds
    other_seed = T.s1c_fold_design_id("V5PAIR__primary__batched", "c" * 64)
    for key in ("B3i", "pairs"):
        with pytest.raises(AssertionError, match="same fitted folds"):
            T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs={**DESIGNS, key: other_seed}))
    # a candidate that is not fitted on batched V5-PAIR folds is refused whatever the yardsticks
    with pytest.raises(ValueError, match="not a batched V5-PAIR design"):
        T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs={k: UNBATCHED_ID for k in DESIGNS}))
    # a fold design without its hash cannot show that the arms share folds (the batched stem is the same for every
    # seed): bare stems are refused even when every arm names the same one, and so is an empty or malformed hash
    # (task X verification, finding 5)
    stem = "V5PAIR__primary__batched"
    for designs in (dict.fromkeys(DESIGNS, stem), dict.fromkeys(DESIGNS, stem + "@"), {**DESIGNS, "pairs": stem},
                    {**DESIGNS, "B3i": stem + "@" + "A" * 64}, dict.fromkeys(DESIGNS, stem + "@" + "a" * 12)):
        with pytest.raises(ValueError, match="design hash"):
            T.check_s1c_fold_designs(designs)
        with pytest.raises(ValueError, match="design hash"):
            T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs=designs))
    # the pairs must all belong to the half being judged (task X verification, finding 7): confirmation-half pairs are
    # never scored as the selection half, and a pair frame without the half column is refused
    with pytest.raises(ValueError, match="belong to another half"):
        T.s1c_half(pairs, cand, **{**kw, "half": "selection"}, **_guards(pairs))
    mixed = pairs.assign(half=["confirmation"] * (len(pairs) - 1) + ["selection"])
    with pytest.raises(ValueError, match="1 pair\\(s\\) belong to another half"):
        T.s1c_half(mixed, cand, **kw, **_guards(mixed))
    with pytest.raises(KeyError, match="lack the column 'half'"):
        T.s1c_half(pairs.drop(columns=["half"]), cand, **kw, **_guards(pairs))
    # HEAVIER (and FLAT) need no fit: naming any fold design for them is accepted and never compared
    ok = T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs={**DESIGNS, "HEAVIER": UNBATCHED_ID,
                                                                       "FLAT": "anything"}))
    assert ok["direction"]["HEAVIER"]["delta"] == pytest.approx(base["direction"]["HEAVIER"]["delta"])
    with pytest.raises(ValueError, match="lacks"):
        T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs={"candidate": BATCHED_ID}))
    with pytest.raises(ValueError, match="not part of S1"):
        T.s1c_half(pairs, cand, **kw, **_guards(pairs, fold_designs={**DESIGNS, "B3": BATCHED_ID}))
    # an unregistered reading is refused (the old NotImplementedError / UNDECIDED path is gone)
    for reading in ("same_pairs", "same_fold_design", "anything"):
        with pytest.raises(ValueError, match="not the registered reading"):
            T.s1c_half(pairs, cand, **kw, fold_reading=reading, **_guards(pairs))
    # the scoring-filter hook drops every pair touching an excluded member for every arm alike ...
    excl = T.s1c_half(pairs, cand, **kw, exclude_rows=[pairs.loc[5, "idx_a"]],
                      scoring_filter="censoring_candidates_excluded_scoring", **_guards(pairs))
    assert excl["n_pairs"] == base["n_pairs"] - 1 and excl["n_pairs_excluded_by_filter"] == 1
    assert excl["scoring_filter"] == "censoring_candidates_excluded_scoring" and excl["n_exclude_labels"] == 1
    assert excl["direction"]["B3i"]["n_pairs"] == base["direction"]["B3i"]["n_pairs"] - 1
    assert excl["pair_set_sha256"] != base["pair_set_sha256"]
    # ... and refuses labels that are not fold-qualified (a bare row id would silently exclude nothing)
    for bad in (["a5"], [pairs.loc[5, "idx_a"], "b5"], ["|a5"], [5]):
        with pytest.raises(ValueError, match="fold-qualified"):
            T.s1c_half(pairs, cand, **kw, exclude_rows=bad, scoring_filter=T.WILDCARD_COPY_SENSITIVITY,
                       **_guards(pairs))
    with pytest.raises(TypeError, match="iterable of fold-qualified"):
        T.s1c_half(pairs, cand, **kw, exclude_rows=pairs.loc[5, "idx_a"], **_guards(pairs))


# --------------------------------------------------------------------------------------------- #
# 4. wildcard-copy sensitivity registered for V5-P and V5-PAIR (section 8 R19 item 6 resolution)
# --------------------------------------------------------------------------------------------- #

def test_wildcard_copy_sensitivity_registered_for_v5p_and_v5pair_strict_stays_exploratory():
    for d in ("V1", "V5", "V5-P", "V5-PAIR"):
        assert T.sensitivity_status(d, T.WILDCARD_COPY_SENSITIVITY) == "registered", d
        assert T.sensitivity_status(d, T.WILDCARD_COPY_STRICT_SENSITIVITY) == "exploratory", d
        assert T.WILDCARD_COPY_SENSITIVITY in T.scoring_filters(d), d
        assert T.scoring_filters(d, "exploratory") == (T.WILDCARD_COPY_STRICT_SENSITIVITY,), d
    assert T.REGISTERED_SENSITIVITIES["V5-PAIR"] == T.COMMON_SENSITIVITIES + (T.WILDCARD_COPY_SENSITIVITY,)
    assert T.REGISTERED_SENSITIVITIES["V5-P"] == T.COMMON_SENSITIVITIES + (T.WILDCARD_COPY_SENSITIVITY,)
    for d in ("V2", "V3", "V4", "V6", "V7"):
        assert T.sensitivity_status(d, T.WILDCARD_COPY_SENSITIVITY) == "unregistered", d
    # R19 item 6 on V5-P / V5-PAIR: the any-partner filter decides, the strict one never does
    rng = np.random.default_rng(7)
    idx = pd.Index([f"cp{i:02d}" for i in range(36)])
    comp = pd.Series(rng.normal(0.3, 0.05, len(idx)), index=idx)
    zero = pd.Series(0.0, index=idx)
    boots = {"system": T.paired_cluster_bootstrap(comp, zero, pd.Series([f"s{i % 12}" for i in range(36)], index=idx),
                                                  cluster_unit="system"),
             "publication_group": T.paired_cluster_bootstrap(comp, zero, pd.Series([f"g{i % 9}" for i in range(36)],
                                                                                   index=idx),
                                                             cluster_unit="publication_group")}
    for d in ("V5-P", "V5-PAIR"):
        sens = {n: 0.1 for n in T.REGISTERED_SENSITIVITIES[d]}
        kw = dict(design=d, stage="discovery", point=boots["system"].point, margin=0.05, bootstraps=boots,
                  seed_deltas=[0.3] * 5, deterministic=False)
        ok = T.r19(**kw, sensitivity_deltas={**sens, T.WILDCARD_COPY_STRICT_SENSITIVITY: -1.0})
        assert ok.passes and "exploratory, not deciding" in ok.item(6)["detail"], d
        neg = T.r19(**kw, sensitivity_deltas={**sens, T.WILDCARD_COPY_SENSITIVITY: -0.01})
        assert neg.verdict == "FAIL" and "wildcard_copies_excluded_scoring: Delta" in neg.item(6)["detail"], d
        miss = T.r19(**kw, sensitivity_deltas={k: v for k, v in sens.items() if k != T.WILDCARD_COPY_SENSITIVITY})
        assert miss.item(6)["status"] == "FAIL" and "wildcard_copies_excluded_scoring: missing" in miss.item(6)["detail"]


# --------------------------------------------------------------------------------------------- #
# 5. V1: the pooled remainder fold is ONE publication-group cluster under the outer-fold unit (section 8)
# --------------------------------------------------------------------------------------------- #

GROUPED_FOLDS = ["s1_f0", "s1_f1", "s1_f0", "s1_f1", "s1_f0", "s1_f1", "s1_f1"]


def test_v1_remainder_fold_is_one_publication_group_cluster_under_the_outer_fold_unit():
    fr = _v1_frame(EXACT_FOLDS)
    rem = {"r1", "r2"}
    for scheme, fold_ids in (("exact", EXACT_FOLDS), ("grouped", GROUPED_FOLDS)):
        f = fr.assign(**{EM.FOLD_COL: fold_ids})
        kw = dict(v1_scheme=scheme, remainder_groups=rem)
        units = EM.with_registered_units(f, "V1", **kw)[EM.V1_UNIT_COL]
        assert set(units[f[PUB].isin(rem)]) == {"REMAINDER"}                       # every remainder row, one unit
        cl = EM.design_unit_clusters(f, "V1", **kw)
        assert list(cl) == list(T.REGISTERED_CLUSTER_UNITS["V1"]) == ["publication_group"]
        cl = cl["publication_group"]
        assert cl.to_dict() == {"REMAINDER": "REMAINDER", "gA": "gA", "gB": "gB"}, scheme
        assert not set(cl) & rem                                                      # its groups are not clusters
        pu = EM.design_per_unit_table(f, "V1", v6_mask=_no_v6(f), **kw)
        assert list(pu.index) == list(cl.index)
        # the paired cluster bootstrap draws the remainder fold whole: 3 clusters for 3 units
        b0 = EM.design_per_unit_table(f.assign(pred=[2.0] * 7), "V1", v6_mask=_no_v6(f), **kw)
        br = T.paired_cluster_bootstrap(b0["mae"], pu["mae"], cl, n_resamples=300, cluster_unit="publication_group")
        assert br.n_clusters == br.n_units == 3 and list(br.jackknife.index) == ["REMAINDER", "gA", "gB"]
        d = (b0["mae"] - pu["mae"]).to_dict()                     # REMAINDER 1.75, gA 0.0, gB 1.5
        picks = T.draw_cluster_picks(3, 300, 19)
        for row in (0, 17, 299):
            assert br.draws[row] == pytest.approx(np.mean([d[["REMAINDER", "gA", "gB"][int(j)]] for j in picks[row]]))
        assert br.jackknife["REMAINDER"] == pytest.approx((d["gA"] + d["gB"]) / 2)
    # the exploratory per-group reading splits the remainder fold into its groups (never deciding)
    assert set(EM.design_unit_clusters(fr, "V1", reading="publication_group")["publication_group"]) == {"gA", "gB", "r1", "r2"}
    # the remainder fold holding a group that is not declared a remainder group is refused (never split into clusters)
    with pytest.raises(ValueError, match="not a remainder group"):
        EM.design_unit_clusters(fr, "V1", remainder_groups={"r1"})


# --------------------------------------------------------------------------------------------- #
# 6. B3x / B3i re-fitted on batched V5-PAIR folds (models.s1c_yardsticks; synthetic mini-corpus only)
# --------------------------------------------------------------------------------------------- #

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TDDGA = "CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCC)CCCCCCCCCC"
LN_Y = {"La(III)": 0.0, "Pr(III)": 0.7, "Nd(III)": 1.1, "Sm(III)": 1.6, "Eu(III)": 2.2}
ACIDS = (0.0, 0.5, 1.0)


def _mini_corpus() -> pd.DataFrame:
    """Two DGA systems x five Ln(III) x three acid conditions (one publication group per system), plus an Nd X(?) row
    under TODGA whose target is a 1e6 sentinel: hidden with every Nd(III) x TODGA cell, never a candidate."""
    recs = []

    def add(state, system, smiles, y, acid, pub, ck, elem=None):
        k = len(recs)
        recs.append({"_label": f"r{k:03d}", STATE: state, SG.ELEMENT_COL: elem or SG.metal_properties(state)["symbol"],
                     SYS: system, SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: -1.0,
                     SG.TEMP_COL: 25.0, SG.DILUENT_COL: "aliphatic", I.ID_COL: f"ID:{k:05d}", I.TARGET_COL: y,
                     I.PUB_GROUP_COL: pub, SG.PUB_COL: f"pub_{pub}", SG.FAMILY_COL: "diglycolamide",
                     SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: smiles, CK: ck})
    for system, smiles, offset, pub in (("S_TODGA", TODGA, 0.0, "g1"), ("S_TDDGA", TDDGA, 0.3, "g2")):
        for c, acid in enumerate(ACIDS):
            for st, y in LN_Y.items():
                add(st, system, smiles, y + offset - 0.8 * acid, acid, pub, f"c{c}")
    add(None, "S_TODGA", TODGA, 1e6, 0.5, "g1", "c1", elem="Nd")
    return pd.DataFrame(recs).set_index("_label")


def _mini_folds(fr: pd.DataFrame, *, scheme: str = "batched", seed: int | None = 104729) -> list:
    """V5-PAIR folds by the registered hiding (``support_graph.hide_cell``, component-aware) of each unit's two cells:
    two selection-half batches under TODGA (one row is scored in both) and one confirmation-half batch under TDDGA."""
    from gen19ct.folds import io as FI

    ids = fr[I.ID_COL]

    def fold(fid, half, cells):
        hidden = pd.Index([])
        for st, sy in cells:
            hidden = hidden.union(fr.index.difference(SG.hide_cell(fr, st, sy, component_aware=True).index))
        scored = fr.index[np.logical_or.reduce([((fr[STATE] == st) & (fr[SYS] == sy)).to_numpy() for st, sy in cells])]
        return FI.make_fold(design="V5PAIR", variant="primary", scheme=scheme, fold_id=fid, half=half, seed=seed,
                            hidden=ids[hidden], scored=ids[scored], unit_type="cell",
                            units=[f"{st} x {sy}" for st, sy in cells], batch_id=fid,
                            meta={"cells": [list(c) for c in cells], "component_aware": True, "parent_structure": False})
    tag = f"s{seed}" if seed is not None else "u"
    return [fold(f"{tag}_S_b000", "S", [("Nd(III)", "S_TODGA"), ("Pr(III)", "S_TODGA")]),
            fold(f"{tag}_S_b001", "S", [("Nd(III)", "S_TODGA"), ("Sm(III)", "S_TODGA")]),
            fold(f"{tag}_C_b000", "C", [("Nd(III)", "S_TDDGA"), ("Pr(III)", "S_TDDGA")])]


def _builder_pairs(fr: pd.DataFrame, folds: list) -> pd.DataFrame:
    recs = []
    for f in folds:
        sc = fr[fr[I.ID_COL].isin(f.scored_row_ids)]
        for _, g in sc.groupby([I.PUB_GROUP_COL, SYS, CK]):
            for a, b in itertools.combinations(g.index, 2):
                if g.loc[a, STATE] != g.loc[b, STATE]:
                    recs.append({"fold_id": f.fold_id, "row_id_a": g.loc[a, I.ID_COL], "row_id_b": g.loc[b, I.ID_COL]})
    return pd.DataFrame(recs)


def test_refit_lookup_yardsticks_on_batched_v5pair_folds_of_a_synthetic_mini_corpus(tmp_path):
    import json

    from gen19ct.folds import io as FI
    from gen19ct.models import s1c_yardsticks as SY

    fr = _mini_corpus()
    folds = _mini_folds(fr)
    FI.write_design(folds, tmp_path)
    stem = "V5PAIR__primary__batched"
    calls = []

    def guard(fold, universe):
        calls.append(fold.fold_id)
        assert universe.equals(fr.index)
        return {"ok": True}

    v6 = pd.Series(False, index=fr.index)
    refit = SY.refit_lookup_yardsticks(stem, fr, folds_dir=tmp_path, v6_mask=v6, guard=guard)
    assert calls == [f.fold_id for f in folds]                                        # the outer guard runs before every fit
    index_hash = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))["summary"]["design_hash"]
    assert refit.design_hash == FI.design_hash(folds) == index_hash and refit.seed == 104729
    assert refit.fold_design == T.s1c_fold_design_id(stem, index_hash) and T.is_batched_v5pair_design(refit.fold_design)
    p = refit.predictions
    assert list(p.columns) == list(SY.PREDICTION_COLUMNS) and set(p["arm"]) == {"B3x", "B3i"} and len(p) == 2 * 18
    assert (p["label"] == p["fold_id"] + "|" + p["row_id"]).all() and not p.duplicated(["label", "arm"]).any()
    assert set(refit.fold_log["status"]) == {"fitted"} and (p["mean_logD"].abs() < 1e5).all()   # sentinel never used
    # each fold's predictions equal a direct fit of the arm on the fold's training rows (MODEL rows minus hidden rows)
    for f in folds:
        hidden = fr.index[fr[I.ID_COL].isin(f.hidden_row_ids)]
        assert "r030" in hidden or f.half == "C"                                    # the Nd X(?) row hides with Nd x TODGA
        scored = fr.index[fr[I.ID_COL].isin(f.scored_row_ids)]
        for arm in ("B3x", "B3i"):
            want = B.BaselineArm(arm).fit(fr.drop(index=hidden), I.FitContext(hidden_index=hidden)).predict(fr.loc[scored])
            want = pd.Series(want["mean_logD"].to_numpy(), index=fr.loc[want["row_id"], I.ID_COL].to_numpy())
            got = p[(p["fold_id"] == f.fold_id) & (p["arm"] == arm)].set_index("row_id")["mean_logD"]
            assert got.reindex(want.index).to_numpy() == pytest.approx(want.to_numpy(), abs=1e-12), (f.fold_id, arm)
    # the same Nd(III) x TODGA row is predicted differently in its two folds, so predictions are fold-qualified:
    # b000 hides Pr (nearest-radius training metal Sm, log D 1.6), b001 hides Sm (nearest Pr, 0.7)
    nd_c0 = fr.index[(fr[STATE] == "Nd(III)") & (fr[SYS] == "S_TODGA") & (fr[CK] == "c0")][0]
    rid = fr.loc[nd_c0, I.ID_COL]
    b3x = refit.logd("B3x")
    assert b3x[f"s104729_S_b000|{rid}"] == pytest.approx(1.6) and b3x[f"s104729_S_b001|{rid}"] == pytest.approx(0.7)
    # halves, scoring exclusions, the V6 guard, a failing guard, other arms and other fold files
    only_s = SY.refit_lookup_yardsticks(stem, fr, folds_dir=tmp_path, v6_mask=v6, guard=guard, halves=("S",))
    assert set(only_s.predictions["half"]) == {"selection"} and len(only_s.predictions) == 2 * 12
    pr_c = fr.index[(fr[STATE] == "Pr(III)") & (fr[SYS] == "S_TDDGA") & (fr[CK] == "c2")][0]
    ex = SY.refit_lookup_yardsticks(stem, fr, folds_dir=tmp_path, v6_mask=v6, guard=guard,
                                    exclude_from_scoring=pd.Series(fr.index == pr_c, index=fr.index))
    assert int(ex.fold_log.set_index("fold_id").loc["s104729_C_b000", "n_excluded_from_scoring"]) == 1
    assert len(ex.predictions) == 2 * 17 and fr.loc[pr_c, I.ID_COL] not in set(ex.predictions["row_id"])
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        SY.refit_lookup_yardsticks(stem, fr, folds_dir=tmp_path, guard=guard,
                                   v6_mask=pd.Series(fr.index == nd_c0, index=fr.index))
    with pytest.raises(AssertionError, match="outer guard failed"):
        SY.refit_lookup_yardsticks(stem, fr, folds_dir=tmp_path, v6_mask=v6, guard=lambda f, u: {"ok": False})
    with pytest.raises(ValueError, match="lookup yardsticks"):
        SY.refit_lookup_yardsticks(stem, fr, folds_dir=tmp_path, v6_mask=v6, guard=guard, arms=("B3",))
    unbatched = tmp_path / "unbatched"
    FI.write_design(_mini_folds(fr, scheme="cell_pair", seed=None), unbatched)
    with pytest.raises(ValueError, match="batched V5-PAIR folds only"):
        SY.refit_lookup_yardsticks("V5PAIR__primary__cell_pair", fr, folds_dir=unbatched, v6_mask=v6, guard=guard)
    with pytest.raises(KeyError, match="columns missing"):
        SY.registered_guard(fr)                                         # the registered guard needs the archive key columns

    # pairs regenerated inside each fold after fold assignment, checked against the fold builder's list
    attrs = fr.rename(columns={I.PUB_GROUP_COL: PUB}).set_index(I.ID_COL)[[PUB, SYS, CK, STATE, "log_D"]]
    v6_ids = pd.Series(False, index=attrs.index)
    bp = _builder_pairs(fr, folds)
    assert len(bp) == 9
    inp = SY.yardstick_pair_inputs(refit, attrs, v6_mask=v6_ids, builder_pairs=bp)
    pairs = inp["pairs"]
    assert len(pairs) == 9 and pairs.groupby("half").size().to_dict() == {"confirmation": 3, "selection": 6}
    assert (pairs["idx_a"].str.split("|", n=1).str[0] == pairs["fold"]).all()
    assert inp["fold_designs"] == {"pairs": refit.fold_design, "B3x": refit.fold_design, "B3i": refit.fold_design}
    for arm in ("B3x", "B3i"):
        lk = refit.logd(arm)
        assert inp["lookup_logsf"][arm].to_numpy() == pytest.approx(
            (pairs["idx_a"].map(lk) - pairs["idx_b"].map(lk)).to_numpy())
    with pytest.raises(AssertionError, match="regenerated"):
        SY.yardstick_pair_inputs(refit, attrs, v6_mask=v6_ids, builder_pairs=bp.iloc[1:])
    # S1(c) on the selection half: a candidate fitted on the same batched folds is accepted ...
    sel = SY.select_half(inp, "selection")
    cand = sel["lookup_logsf"]["B3i"] + 0.05
    kw = dict(half="selection", seed=104729, lookup_logsf=sel["lookup_logsf"], folds=sel["folds"],
              v6_mask=sel["v6_mask"], n_resamples=100)
    h = T.s1c_half(sel["pairs"], cand, fold_designs={**sel["fold_designs"], "candidate": refit.fold_design}, **kw)
    assert h["fold_design"] == refit.fold_design and h["n_pairs"] == 6
    obs, lk = sel["pairs"]["logsf_obs"], sel["lookup_logsf"]["B3i"]
    cp = sel["pairs"][list(EP.CELL_PAIR_COLS)].astype(str).agg(" x ".join, axis=1)
    want_gain = ((lk - obs).abs().groupby(cp).mean() - (cand - obs).abs().groupby(cp).mean()).mean()
    assert h["logsf_mae"]["B3i"]["gain"] == pytest.approx(want_gain)
    # ... yardsticks from the unbatched folds, or a candidate on other folds, are refused
    with pytest.raises(AssertionError, match="same fitted folds"):
        T.s1c_half(sel["pairs"], cand, fold_designs={**sel["fold_designs"], "candidate": refit.fold_design,
                                                     "B3x": UNBATCHED_ID}, **kw)
    with pytest.raises(AssertionError, match="same fitted folds"):
        T.s1c_half(sel["pairs"], cand, fold_designs={**sel["fold_designs"], "candidate": BATCHED_ID}, **kw)
    # ... and the confirmation-half pairs of the same run are refused as the selection half (the half column the helper
    # supplies is checked against the half argument)
    conf_in = SY.select_half(inp, "confirmation")
    with pytest.raises(ValueError, match="belong to another half"):
        T.s1c_half(conf_in["pairs"], conf_in["lookup_logsf"]["B3i"] + 0.05,
                   fold_designs={**conf_in["fold_designs"], "candidate": refit.fold_design},
                   **{**kw, "lookup_logsf": conf_in["lookup_logsf"]})


# --------------------------------------------------------------------------------------------- #
# 7. multi-seed conformal intervals of the deterministic arms (section 15)
# --------------------------------------------------------------------------------------------- #

def _grouped_frame(seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    effects = rng.normal(0, 2, size=6)
    for g in range(30):
        for _ in range(8):
            s = int(rng.integers(6))
            k = len(recs)
            recs.append({"_label": f"r{k:04d}", SG.METAL_COL: "Nd(III)", SG.ELEMENT_COL: "Nd", SG.SYSTEM_COL: f"S{s}",
                         SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: 0.0, SG.LOG_EXT_COL: -1.0, SG.TEMP_COL: 25.0,
                         I.ID_COL: f"ID:{k:05d}", I.TARGET_COL: float(effects[s] + rng.normal()),
                         I.PUB_GROUP_COL: f"g{g:02d}", SG.PUB_COL: f"pub_g{g:02d}", SG.FAMILY_COL: "fam",
                         SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: "CCCCOP(=O)(OCCCC)OCCCC"})
    return pd.DataFrame(recs).set_index("_label")


def test_conformal_wrapper_multi_seed_intervals_and_seed_mean_metrics():
    fr = _grouped_frame()
    table = I.RowTable(fr)
    test_groups = {f"g{g:02d}" for g in range(24, 30)}
    mask = ~fr[I.PUB_GROUP_COL].isin(test_groups).to_numpy()
    pos = np.flatnonzero(~mask)

    def ctx(seed):
        return I.FitContext(seed=seed, table=table, v6_mask=pd.Series(False, index=fr.index),
                            isolation_check=lambda tr, te: {"ok": True})
    seeds = (104729, 130363, 155921)
    # default: one draw with context.seed, unchanged layout (the disclosed single-seed pre-seal run)
    single = {s: I.ConformalWrapper(B.BaselineArm("B2"), splitter=I.GroupKFoldCalibration(3)).fit_table(table, mask, ctx(s))
              for s in seeds}
    p0 = single[seeds[0]].predict_positions(pos)
    assert list(p0.columns) == list(I.PREDICTION_COLUMNS) and len(p0) == len(pos)
    assert single[seeds[0]].seeds is None and single[seeds[0]].fit_seed == seeds[0]
    # multi-seed: per-seed calibration equals the single-seed fit with that seed; one centre for every seed
    multi = I.ConformalWrapper(B.BaselineArm("B2"), splitter=I.GroupKFoldCalibration(3), seeds=seeds)
    multi.fit_table(table, mask, ctx(None))
    assert multi.multi_seed and multi.residuals is None and multi.quantiles == {}
    for s in seeds:
        assert multi.quantiles_by_seed[s] == single[s].quantiles
        assert np.array_equal(multi.residuals_by_seed[s], single[s].residuals)
        assert multi.calibration_units_by_seed[s] == single[s].calibration_units
    assert len({tuple(q.values()) for q in multi.quantiles_by_seed.values()}) > 1        # the seed moves the intervals
    by_seed = multi.predict_positions_by_seed(pos)
    for s in seeds:
        pd.testing.assert_frame_equal(by_seed[s], single[s].predict_positions(pos))
        assert np.array_equal(by_seed[s]["mean_logD"], p0["mean_logD"])                  # point predictions unaffected
    long = multi.predict_positions(pos)
    assert len(long) == len(seeds) * len(pos) and list(long["conformal_seed"].unique()) == list(seeds)
    assert multi.clone().seeds == seeds
    # seed-mean coverage / width over the section 4 unit (publication group here)
    y = table.y[pos]
    units = fr[I.PUB_GROUP_COL].to_numpy()[pos]
    frames, met = multi.seed_interval_metrics(pos, y, units=units)
    assert set(frames) == set(seeds)
    for metric in ("coverage_80", "width_95"):
        for agg in ("unit_macro", "row_pooled"):
            rows = met[(met["metric"] == metric) & (met["aggregation"] == agg)]
            per = rows[rows["seed"] != EC.SEED_MEAN_LABEL]
            mean = rows[rows["seed"] == EC.SEED_MEAN_LABEL]
            assert len(per) == 3 and len(mean) == 1 and mean["n_seeds"].iloc[0] == 3
            assert mean["value"].iloc[0] == pytest.approx(per["value"].mean())
    # hand check of one per-seed value: unit-macro 80 % coverage of the first seed
    f = by_seed[seeds[0]]
    cov = pd.Series(((y >= f["lower_80"].to_numpy()) & (y <= f["upper_80"].to_numpy())).astype(float)).groupby(units).mean()
    got = met[(met["seed"] == seeds[0]) & (met["metric"] == "coverage_80") & (met["aggregation"] == "unit_macro")]
    assert got["value"].iloc[0] == pytest.approx(cov.mean()) and got["n_units"].iloc[0] == 6
    with pytest.raises(ValueError):
        I.ConformalWrapper(B.BaselineArm("B2"), splitter=I.GroupKFoldCalibration(3), seeds=[1, 1])
    with pytest.raises(TypeError):
        I.ConformalWrapper(B.BaselineArm("B2"), splitter=I.GroupKFoldCalibration(3), seeds=104729)


# --------------------------------------------------------------------------------------------- #
# 8. power check: X(?) rows dropped from injected refits, scored rows unchanged (section 8)
# --------------------------------------------------------------------------------------------- #

def _power_frame() -> pd.DataFrame:
    states = ["Nd(III)", "Pr(III)", "Eu(III)"]
    rows = [{STATE: m, SYS: s, "log_D": float(i)} for i, (m, s) in enumerate((m, s) for m in states for s in ("S1", "S2"))
            for _ in range(2)]
    rows += [{STATE: None, SYS: "S1", "log_D": 5.0}, {STATE: None, SYS: "S3", "log_D": 6.0}]   # X(?), one in its own system
    return pd.DataFrame(rows, index=[f"r{i:02d}" for i in range(len(rows))])


def test_power_check_drops_unknown_state_rows_and_keeps_scored_rows():
    fr = _power_frame()
    xq = fr.index[fr[STATE].isna()]
    scored = pd.Index(["r00", "r03", "r08"])
    run = T.prepare_injected_run(fr, kappa=0.5, seed=104729, scored_index=scored)
    assert list(run.dropped_unknown_state) == list(xq) and not run.frame[STATE].isna().any()
    assert run.frame.index.equals(fr.index.difference(xq, sort=False))
    # the signal is drawn on every input row (the draw does not depend on the drop), then injected on the kept rows
    s = T.injected_signal(fr, seed=104729)
    assert s.equals(run.signal) and s.loc[xq].isna().all()
    assert np.allclose(run.frame["log_D"], fr.loc[run.frame.index, "log_D"] + 0.5 * s.loc[run.frame.index])
    assert run.scored_index.equals(scored)
    # every arm of the contrast gets the same injected training rows (no X(?)) and the un-injected scored rows; the
    # training rows follow from the fold's HIDDEN rows (task X, leakage finding VR-03)
    hidden = scored.append(pd.Index(["r01"]))                       # r01: hidden with the fold, not scored
    outer_train = fr.index.difference(hidden, sort=False)
    arm_inputs = [run.fold_inputs(hidden, scored, outer_training_index=outer_train, uninjected_scored_index=scored)
                  for _arm in ("M2", "B3i")]
    (tr_a, sc_a), (tr_b, sc_b) = arm_inputs
    assert tr_a.equals(tr_b) and sc_a.equals(sc_b) and sc_a.equals(scored)
    assert not tr_a.index.isin(xq).any() and len(tr_a) == len(outer_train) - len(xq)
    assert not tr_a.index.isin(hidden).any() and np.allclose(tr_a["log_D"], run.frame.loc[tr_a.index, "log_D"])
    assert run.training_index(hidden).equals(tr_a.index)
    # a hidden-but-unscored row passed back as training (the old helper accepted it) is refused
    with pytest.raises(AssertionError, match="not the input rows minus the fold's hidden rows"):
        run.fold_inputs(hidden, scored, outer_training_index=outer_train.append(pd.Index(["r01"])))
    with pytest.raises(AssertionError, match="differ from the un-injected"):
        run.fold_inputs(hidden, scored[:2], uninjected_scored_index=scored)
    with pytest.raises(AssertionError, match="does not score"):
        run.fold_inputs(pd.Index(["r01"]), pd.Index(["r01"]))
    with pytest.raises(AssertionError, match="not hidden in the fold"):
        run.fold_inputs(scored[:2], scored)
    with pytest.raises(KeyError):
        run.training_index(["nope"])
    with pytest.raises(AssertionError, match="X\\(\\?\\) row is among the scored rows"):
        T.prepare_injected_run(fr, kappa=0.5, seed=1, scored_index=scored.append(xq[:1]))
    with pytest.raises(ValueError, match="kappa"):
        T.prepare_injected_run(fr, kappa=0.3, seed=1, scored_index=scored)
    T.assert_same_scored_rows(pd.Index(["a", "b"]), pd.Index(["b", "a"]))
