"""Tests for the gen6 level/shape/offset decomposition, hard-chemistry endpoints and bootstrap.

The three fixtures that matter are the identity ones: a pure-offset predictor, a
pure-shape-free predictor, and the exact SSE identity on random data. If those
hold, the decomposition means what the report says it means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.metrics import (
    decompose_level_shape, effective_sample_size, gen6_metric_table, hard_chemistry_endpoints,
    ood_calibration_table, ood_statistics, paired_unit_bootstrap, per_unit_statistics,
)


def _panel(n_ligands: int = 6, n_rows: int = 8, seed: int = 11) -> pd.DataFrame:
    """A small level panel: several ligands, several rows each, two clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ligands):
        for j in range(n_rows):
            rows.append({
                "extractant": f"L{i}",
                "ecfp_cluster": f"c{i // 2}",
                "tanimoto_cluster": f"t{i // 3}",
                "log_D": float(rng.normal(loc=i - 2.0, scale=1.2)),
                "nn_reference_tanimoto": 0.2 + 0.15 * i,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #

def test_pure_offset_predictor_has_zero_shape_error():
    """prediction = truth + per-ligand constant -> shape is perfect, offset is the constant."""
    frame = _panel()
    offsets = {f"L{i}": (i - 2) * 0.5 for i in range(6)}
    pred = frame["log_D"] + frame["extractant"].map(offsets)
    out = decompose_level_shape(frame["log_D"], pred, frame["extractant"])
    assert out.summary["shape_mae"] == pytest.approx(0.0, abs=1e-12)
    assert out.summary["shape_r2"] == pytest.approx(1.0, abs=1e-12)
    assert out.summary["offset_mae"] == pytest.approx(np.mean([abs(v) for v in offsets.values()]))
    assert out.summary["offset_share_of_sse"] == pytest.approx(1.0, abs=1e-12)


def test_per_ligand_mean_predictor_has_zero_offset_and_zero_shape_r2():
    """prediction = that ligand's own mean truth -> no level error, no shape skill."""
    frame = _panel()
    pred = frame.groupby("extractant")["log_D"].transform("mean")
    out = decompose_level_shape(frame["log_D"], pred, frame["extractant"])
    assert out.summary["offset_mae"] == pytest.approx(0.0, abs=1e-12)
    assert out.summary["shape_r2"] == pytest.approx(0.0, abs=1e-12)
    assert out.summary["offset_share_of_sse"] == pytest.approx(0.0, abs=1e-12)
    # the shape error it leaves behind is the mean absolute deviation of the truth
    expected = frame.assign(c=lambda d: (d["log_D"] - d.groupby("extractant")["log_D"].transform("mean")).abs()) \
        .groupby("extractant")["c"].mean().mean()
    assert out.summary["shape_mae"] == pytest.approx(expected)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sse_identity_holds_exactly_on_random_data(seed):
    """SSE_total == SSE_centred + sum_l n_l * b_l^2 — the identity the report leans on."""
    rng = np.random.default_rng(seed)
    frame = _panel(n_ligands=7, n_rows=5, seed=seed)
    pred = frame["log_D"] + rng.normal(scale=1.5, size=len(frame))
    out = decompose_level_shape(frame["log_D"], pred, frame["extractant"])
    assert out.summary["sse_total"] == pytest.approx(
        out.summary["sse_centred"] + out.summary["sse_offset"], rel=1e-12)


def test_identity_holds_with_singleton_ligands():
    """A one-row ligand contributes all of its error to the offset term."""
    frame = pd.DataFrame({
        "extractant": ["A", "A", "A", "B"],
        "log_D": [1.0, 2.0, 3.0, -1.0],
    })
    pred = pd.Series([1.5, 2.5, 2.5, 0.5])
    out = decompose_level_shape(frame["log_D"], pred, frame["extractant"])
    assert out.summary["sse_total"] == pytest.approx(out.summary["sse_centred"] + out.summary["sse_offset"])
    singleton = out.per_ligand.set_index("extractant").loc["B"]
    assert singleton["sse_centred"] == pytest.approx(0.0)
    assert singleton["sse_offset"] == pytest.approx(1.5 ** 2)
    assert np.isnan(singleton["shape_mae"])          # excluded from shape statistics
    assert out.summary["n_ligands"] == 2 and out.summary["n_ligands_with_shape"] == 1
    # A's residuals are +0.5, +0.5, -0.5 -> bias 1/6;  B's single residual is +1.5
    assert out.per_ligand.set_index("extractant").loc["A", "bias"] == pytest.approx(0.5 / 3)
    assert out.summary["offset_mae"] == pytest.approx(np.mean([0.5 / 3, 1.5]))


def test_perfect_prediction_scores_zero_everywhere():
    frame = _panel()
    out = decompose_level_shape(frame["log_D"], frame["log_D"], frame["extractant"])
    assert out.summary["offset_mae"] == pytest.approx(0.0)
    assert out.summary["shape_mae"] == pytest.approx(0.0)
    assert out.summary["pooled_mae"] == pytest.approx(0.0)
    assert out.summary["sign_accuracy"] == pytest.approx(1.0)
    assert out.summary["rank_spearman"] == pytest.approx(1.0)


def test_constant_predictor_scores_half_on_sign_accuracy():
    frame = _panel()
    out = decompose_level_shape(frame["log_D"], np.zeros(len(frame)), frame["extractant"])
    assert out.summary["sign_accuracy"] == pytest.approx(0.5)


def test_reversed_predictor_scores_zero_sign_accuracy():
    frame = _panel(n_ligands=2, n_rows=6)
    out = decompose_level_shape(frame["log_D"], -frame["log_D"], frame["extractant"])
    assert out.summary["sign_accuracy"] == pytest.approx(0.0)
    assert out.summary["rank_spearman"] == pytest.approx(-1.0)


def test_decomposition_rejects_bad_input():
    with pytest.raises(ValueError, match="equal length"):
        decompose_level_shape([1.0, 2.0], [1.0], ["A", "A"])
    with pytest.raises(ValueError, match="no rows"):
        decompose_level_shape([], [], [])
    with pytest.raises(ValueError, match="non-finite"):
        decompose_level_shape([1.0, 2.0], [1.0, np.nan], ["A", "A"])


def test_worst_quartile_is_at_least_the_median():
    frame = _panel(n_ligands=8, n_rows=6)
    rng = np.random.default_rng(5)
    pred = frame["log_D"] + rng.normal(scale=frame["extractant"].str[1:].astype(int) * 0.4 + 0.1)
    out = decompose_level_shape(frame["log_D"], pred, frame["extractant"])
    assert out.summary["worst_quartile_ligand_mae"] >= out.summary["median_ligand_mae"]
    assert out.summary["max_ligand_mae"] >= out.summary["worst_quartile_ligand_mae"]


def test_effective_sample_size_matches_kish():
    assert effective_sample_size([10, 10, 10, 10]) == pytest.approx(4.0)
    assert effective_sample_size([100, 1, 1]) == pytest.approx(102 ** 2 / (100 ** 2 + 2))
    assert np.isnan(effective_sample_size([]))


# --------------------------------------------------------------------------- #
# Hard chemistry endpoints
# --------------------------------------------------------------------------- #

def test_hard_chemistry_endpoints_subset_and_counts():
    frame = _panel()
    frame["prediction_arm"] = frame["log_D"] + 0.4
    table = hard_chemistry_endpoints(frame, prediction_column="prediction_arm")
    assert list(table["endpoint"]) == ["nn<0.4", "nn<0.6", "all"]
    hard = table.set_index("endpoint")
    assert hard.loc["nn<0.4", "n_rows"] == int((frame["nn_reference_tanimoto"] < 0.4).sum())
    assert hard.loc["all", "n_rows"] == len(frame)
    # a pure offset of 0.4 shows up as offset error, not shape error, in every bin
    assert hard.loc["all", "offset_mae"] == pytest.approx(0.4)
    assert hard.loc["nn<0.6", "shape_mae"] == pytest.approx(0.0, abs=1e-12)


def test_hard_chemistry_endpoint_with_no_rows_is_reported_not_dropped():
    frame = _panel()
    frame["nn_reference_tanimoto"] = 0.9
    frame["prediction_arm"] = frame["log_D"]
    table = hard_chemistry_endpoints(frame, prediction_column="prediction_arm").set_index("endpoint")
    assert table.loc["nn<0.4", "n_rows"] == 0
    assert np.isnan(table.loc["nn<0.4", "macro_mae"])


def test_hard_chemistry_requires_the_similarity_column():
    frame = _panel().drop(columns=["nn_reference_tanimoto"])
    frame["prediction_arm"] = 0.0
    with pytest.raises(KeyError, match="similarity column"):
        hard_chemistry_endpoints(frame, prediction_column="prediction_arm")


# --------------------------------------------------------------------------- #
# OOD layer
# --------------------------------------------------------------------------- #

def test_ood_statistics_attach_by_ligand():
    frame = _panel(n_ligands=3, n_rows=4)
    neighbours = pd.DataFrame({
        "extractant": ["L0", "L1", "L2"],
        "nn_tanimoto": [0.30, 0.65, 0.92],
        "n_above_0_5": [0, 3, 12],
        "n_above_0_7": [0, 0, 7],
        "n_above_0_8": [0, 0, 4],
        "supercluster_support": [0, 2, 40],
    })
    out = ood_statistics(frame, neighbour_table=neighbours,
                         prediction_spread=np.full(len(frame), 0.5))
    assert out.loc[out["extractant"] == "L2", "ood__nn_tanimoto"].unique().tolist() == [0.92]
    assert (out["ood__prediction_sd"] == 0.5).all()
    assert out.loc[out["extractant"] == "L0", "ood__n_above_0_7"].unique().tolist() == [0]


def test_ood_calibration_table_bins_by_similarity():
    frame = _panel(n_ligands=4, n_rows=5)
    frame["ood__nn_tanimoto"] = frame["extractant"].map({"L0": 0.2, "L1": 0.5, "L2": 0.7, "L3": 0.95})
    frame["prediction_arm"] = frame["log_D"] + frame["ood__nn_tanimoto"].rsub(1.0)  # worse when far
    table = ood_calibration_table(frame, prediction_column="prediction_arm")
    assert len(table) == 4
    assert table["pooled_mae"].is_monotonic_decreasing  # error falls as similarity rises
    # never a bare "mae": a row mean must not sit unlabelled beside macro tables
    assert "mae" not in table.columns


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def _two_arm_predictions() -> pd.DataFrame:
    frame = _panel(n_ligands=6, n_rows=8, seed=4)
    rng = np.random.default_rng(9)
    frame["prediction_good"] = frame["log_D"] + rng.normal(scale=0.2, size=len(frame))
    frame["prediction_bad"] = frame["log_D"] + rng.normal(scale=1.4, size=len(frame)) + 0.8
    return frame


def test_per_unit_statistics_shape_and_membership():
    frame = _two_arm_predictions()
    table = per_unit_statistics(frame, ["good", "bad"])
    assert set(table["arm"]) == {"good", "bad"}
    assert set(table["unit"]) == set(frame["ecfp_cluster"])
    assert (table["n_rows"] > 0).all()
    # the mean over units of the per-unit MAE is exactly the macro MAE
    macro = frame.assign(e=(frame["prediction_good"] - frame["log_D"]).abs()) \
        .groupby("ecfp_cluster")["e"].mean().mean()
    assert table[table["arm"] == "good"]["mae"].mean() == pytest.approx(macro)


def test_paired_bootstrap_point_estimate_equals_the_macro_delta():
    frame = _two_arm_predictions()
    per_unit = per_unit_statistics(frame, ["good", "bad"])
    boot = paired_unit_bootstrap(per_unit, {"bad_vs_good": ("bad", "good")},
                                 statistics=("mae",), replicates=200)
    row = boot.iloc[0]
    expected = (per_unit[per_unit.arm == "bad"].set_index("unit")["mae"]
                - per_unit[per_unit.arm == "good"].set_index("unit")["mae"]).mean()
    assert row["point_delta"] == pytest.approx(expected)
    assert row["ci95_low"] <= row["point_delta"] <= row["ci95_high"]


def test_bootstrap_intervals_do_not_depend_on_comparison_order():
    frame = _two_arm_predictions()
    per_unit = per_unit_statistics(frame, ["good", "bad"])
    a = paired_unit_bootstrap(per_unit, {"x": ("bad", "good"), "y": ("good", "bad")},
                              statistics=("mae",), replicates=300)
    b = paired_unit_bootstrap(per_unit, {"y": ("good", "bad"), "x": ("bad", "good")},
                              statistics=("mae",), replicates=300)
    a_x = a[a.comparison == "x"].iloc[0]
    b_x = b[b.comparison == "x"].iloc[0]
    assert a_x["ci95_low"] == pytest.approx(b_x["ci95_low"])
    assert a_x["ci95_high"] == pytest.approx(b_x["ci95_high"])


def test_bootstrap_resamples_blocks_not_units():
    """With every unit inside one block, every replicate is identical -> zero width."""
    frame = _two_arm_predictions()
    per_unit = per_unit_statistics(frame, ["good", "bad"])
    one_block = {u: "single" for u in per_unit["unit"].unique()}
    boot = paired_unit_bootstrap(per_unit, {"d": ("bad", "good")}, statistics=("mae",),
                                 block_of_unit=one_block, replicates=200)
    row = boot.iloc[0]
    assert row["bootstrap_blocks"] == 1
    assert row["ci95_low"] == pytest.approx(row["ci95_high"])


def test_bootstrap_is_deterministic_for_a_seed():
    frame = _two_arm_predictions()
    per_unit = per_unit_statistics(frame, ["good", "bad"])
    kwargs = dict(statistics=("mae", "offset_mae"), replicates=200, seed=1234)
    a = paired_unit_bootstrap(per_unit, {"d": ("bad", "good")}, **kwargs)
    b = paired_unit_bootstrap(per_unit, {"d": ("bad", "good")}, **kwargs)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Arm table
# --------------------------------------------------------------------------- #

def test_gen6_metric_table_reports_counts_and_both_endpoints():
    frame = _two_arm_predictions()
    overall, per_ligand, hard = gen6_metric_table(frame, ["good", "bad"])
    assert list(overall["arm"]) == ["good", "bad"]
    for column in ("macro_mae", "offset_mae", "shape_mae", "n_ligands", "n_ecfp_clusters",
                   "n_superclusters", "n_eff_pooled_rows", "n_macro_units"):
        assert column in overall.columns
    # the two n's answer different questions and must not be one column: macro
    # weights clusters equally (its unit count is n_macro_units), pooled weights
    # rows (its effective n is the Kish number over rows per cluster)
    assert overall["n_macro_units"].iloc[0] == frame["ecfp_cluster"].nunique()
    assert overall["n_eff_pooled_rows"].iloc[0] <= overall["n_macro_units"].iloc[0]
    assert overall.set_index("arm").loc["good", "macro_mae"] < overall.set_index("arm").loc["bad", "macro_mae"]
    assert set(per_ligand["arm"]) == {"good", "bad"}
    assert set(hard["endpoint"]) == {"nn<0.4", "nn<0.6", "all"}


def test_gen6_metric_table_rejects_missing_or_nan_predictions():
    frame = _two_arm_predictions()
    with pytest.raises(KeyError, match="missing prediction column"):
        gen6_metric_table(frame, ["absent"])
    frame.loc[0, "prediction_good"] = np.nan
    with pytest.raises(ValueError, match="missing predictions"):
        gen6_metric_table(frame, ["good"])


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #

def test_all_endpoint_keeps_rows_with_unknown_similarity():
    """`NaN < inf` is False: a naive filter would drop unknown-distance rows from
    the endpoint that is supposed to be the entire test set."""
    frame = _panel(n_ligands=4, n_rows=4)
    frame.loc[frame["extractant"] == "L3", "nn_reference_tanimoto"] = np.nan
    frame["prediction_arm"] = frame["log_D"] + 0.3
    table = hard_chemistry_endpoints(frame, prediction_column="prediction_arm").set_index("endpoint")
    assert table.loc["all", "n_rows"] == len(frame)
    assert table.loc["all", "n_ligands"] == 4
    # unknown-distance rows are excluded from the *threshold* subsets, and counted
    assert table.loc["nn<0.4", "n_rows"] == int((frame["nn_reference_tanimoto"] < 0.4).sum())
    assert (table["n_rows_unknown_similarity"] == 4).all()


def test_ood_statistics_joins_by_name_not_position():
    frame = _panel(n_ligands=3, n_rows=3)
    neighbours = pd.DataFrame({
        "nn_tanimoto": [0.30, 0.65, 0.92],
        "extractant": ["L0", "L1", "L2"],      # key deliberately not first
        "n_above_0_5": [0, 3, 12],
    })
    out = ood_statistics(frame, neighbour_table=neighbours)
    assert out.loc[out["extractant"] == "L2", "ood__nn_tanimoto"].unique().tolist() == [0.92]
    with pytest.raises(KeyError, match="key column"):
        ood_statistics(frame, neighbour_table=neighbours.drop(columns=["extractant"]))
