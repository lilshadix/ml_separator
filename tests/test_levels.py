"""Tests for the gen5 level (``log D``) cohort, estimator and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.levels import (
    LEVEL_ARMS,
    LEVEL_TARGET_COLUMN,
    LevelForestParameters,
    LevelRegressor,
    build_level_dataset,
    condition_labels,
    ecfp_cluster_labels,
    equal_group_macro_mae,
    group_balanced_weights,
    level_metric_table,
    paired_group_bootstrap,
    replicate_noise_floor,
    shuffle_block,
)

METALS = {"La": (57, 1.160), "Nd": (60, 1.109), "Eu": (63, 1.066), "Dy": (66, 1.027), "Lu": (71, 0.977)}


def _synthetic_source(
    seed: int = 0,
    n_extractants: int = 6,
    n_conditions: int = 4,
    n_ecfp: int = 12,
    twin_pairs: int = 1,
    replicate_rows: int = 3,
    missing_condition_rows: int = 5,
) -> pd.DataFrame:
    """Raw-dataset-shaped frame: one row per (extractant, condition, metal).

    ``twin_pairs`` extractants are given a bit-identical fingerprint to another
    extractant, reproducing the ECFP-homolog structure the real data has.
    """
    rng = np.random.default_rng(seed)
    fps = rng.integers(0, 2, size=(n_extractants, n_ecfp))
    for t in range(twin_pairs):  # make extractant (t+1) a fingerprint twin of t
        fps[t + 1] = fps[t]
    ligand_level = rng.normal(0, 1.5, size=n_extractants)
    rows = []
    for e in range(n_extractants):
        for c in range(n_conditions):
            acid = float(rng.uniform(0.1, 5.0))
            for m, (z, radius) in METALS.items():
                y = ligand_level[e] + 0.35 * acid + 0.08 * (z - 60) + rng.normal(0, 0.1)
                row = {
                    "canonical_smiles": f"SMILES{e}", "metal_symbol": m,
                    "Atomic Number_metal": float(z), "Ionic Radius_metal": radius,
                    "lanthanide_index": float(z - 57), LEVEL_TARGET_COLUMN: y,
                    "geometry_ok": True, "cond__acid_concentration_M": acid,
                    "cond__diluent__kerosene": float(c % 2), "cond__temperature_C": 25.0 + c,
                    "MolWt": 300.0 + 10 * e, "MolLogP": 3.0 + 0.1 * e,
                }
                row.update({f"ecfp_{i}": float(fps[e, i]) for i in range(n_ecfp)})
                row.update({f"feat3d__d{i}": float(rng.normal()) for i in range(3)})
                rows.append(row)
    frame = pd.DataFrame(rows)
    if replicate_rows:
        frame = pd.concat([frame, frame.head(replicate_rows)], ignore_index=True)
    if missing_condition_rows:
        frame.loc[frame.index[:missing_condition_rows], "cond__temperature_C"] = np.nan
    return frame


# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #

def test_ecfp_cluster_collapses_fingerprint_twins():
    src = _synthetic_source(n_extractants=6, twin_pairs=1)
    ec = [c for c in src.columns if c.startswith("ecfp_")]
    lig = src.drop_duplicates("canonical_smiles")
    clusters = ecfp_cluster_labels(lig, ec)
    assert lig["canonical_smiles"].nunique() == 6
    assert clusters.nunique() == 5  # two extractants share a fingerprint
    data = build_level_dataset(src, min_rows_per_extractant=1)
    assert data.audit["extractants"] == 6 and data.audit["ecfp_clusters"] == 5


def test_condition_labels_are_stable_and_treat_missing_as_a_value():
    frame = pd.DataFrame({"cond__a": [1.0, 1.0, np.nan, np.nan], "cond__b": ["x", "x", "x", "y"]})
    labels = condition_labels(frame, ["cond__a", "cond__b"])
    assert labels.iloc[0] == labels.iloc[1]      # identical rows
    assert labels.iloc[2] != labels.iloc[0]      # missing is its own value, not a wildcard
    assert labels.iloc[2] != labels.iloc[3]
    again = condition_labels(frame, ["cond__a", "cond__b"])
    pd.testing.assert_series_equal(labels, again)  # deterministic across calls


def test_replicate_policies_collapse_cells_and_record_counts():
    src = _synthetic_source(replicate_rows=4, missing_condition_rows=0)
    cell = ["extractant", "condition_id", "metal_symbol"]
    mean = build_level_dataset(src, min_rows_per_extractant=1, replicate_policy="mean")
    uniq = build_level_dataset(src, min_rows_per_extractant=1, replicate_policy="unique")
    every = build_level_dataset(src, min_rows_per_extractant=1, replicate_policy="all")
    assert not mean.frame.duplicated(cell).any()
    assert not uniq.frame.duplicated(cell).any()
    assert len(every.frame) == len(mean.frame) + 4
    assert (mean.frame["n_replicates"] > 1).sum() == 4
    assert len(mean.frame) == len(uniq.frame)


def test_min_rows_and_log_d_floor_filter_the_cohort():
    src = _synthetic_source(n_extractants=4, n_conditions=4)
    src.loc[src.index[:3], LEVEL_TARGET_COLUMN] = -12.5
    small = src[src.canonical_smiles == "SMILES0"].head(3)
    src = pd.concat([src[src.canonical_smiles != "SMILES0"], small], ignore_index=True)
    data = build_level_dataset(src, min_rows_per_extractant=10, drop_below_log_d=-6.0)
    assert "SMILES0" not in set(data.frame["extractant"])
    assert data.frame[LEVEL_TARGET_COLUMN].min() > -6.0
    assert data.audit["dropped_below_floor"] >= 1


def test_identity_columns_are_never_features():
    data = build_level_dataset(_synthetic_source(), min_rows_per_extractant=1)
    features = set(data.block_columns(data.blocks))
    for identity in ("row_id", "extractant", "ecfp_cluster", "condition_id", "metal_symbol", LEVEL_TARGET_COLUMN):
        assert identity not in features
    assert "n_replicates" not in features


def test_arm_ladder_is_nested_and_uses_expected_blocks():
    data = build_level_dataset(_synthetic_source(), min_rows_per_extractant=1)
    m = set(data.arm_columns("A_metal"))
    mc = set(data.arm_columns("MC"))
    mce = set(data.arm_columns("MC_ecfp"))
    assert m < mc < mce
    assert all(c.startswith("cond__") for c in mc - m)
    assert all(c.startswith("ecfp_") for c in mce - mc)
    assert "ecfp_cluster" not in mce
    # every family arm is family-only; every MC_ family arm is MC plus that family
    for fam, alone, on_mc in (("PHYSCHEM", "A_physchem", "MC_physchem"), ("ECFP", "A_ecfp", "MC_ecfp")):
        if fam in data.blocks:
            assert set(data.arm_columns(alone)) == set(data.blocks[fam])
            assert set(data.arm_columns(on_mc)) == mc | set(data.blocks[fam])


def test_donor_census_and_descriptor_join():
    src = _synthetic_source(n_extractants=3)
    src["DONOR_TYPES"] = '["O(amide_carbonyl)", "O(ether)", "O(amide_carbonyl)"]'
    src["DENTATE"] = 3; src["coreCN"] = 9; src["n_ligs"] = 3; src["n_fill"] = 0
    desc = pd.DataFrame({"canonical_smiles": [f"SMILES{e}" for e in range(3)],
                         "lig2d__a": [1.0, 2.0, 3.0], "lig2d__b": [0.1, 0.2, 0.3]})
    data = build_level_dataset(src, min_rows_per_extractant=1, ligand_descriptors=desc)
    assert "DONORS" in data.blocks and "LIG2D_EXT" in data.blocks
    f = data.frame
    assert (f["donor__O(amide_carbonyl)"] == 2.0).all() and (f["donor__O(ether)"] == 1.0).all()
    assert (f["donor__n_total"] == 3.0).all()
    assert set(data.blocks["LIG2D_EXT"]) == {"lig2d__a", "lig2d__b"}
    assert (f.loc[f.extractant == "SMILES2", "lig2d__a"] == 3.0).all()
    assert data.audit["ligand_descriptor_coverage"] == 1.0


def test_alternative_learners_fit_and_predict():
    data = build_level_dataset(_synthetic_source(n_extractants=5, n_conditions=5), min_rows_per_extractant=1)
    frame = data.frame
    cols = data.arm_columns("MC")
    y = frame[LEVEL_TARGET_COLUMN]
    for learner in ("extratrees", "hgb", "ridge"):
        params = LevelForestParameters(n_estimators=30, n_jobs=1, learner=learner)
        pred = LevelRegressor(cols, params).fit(frame, y, groups=frame["ecfp_cluster"]).predict(frame)
        assert pred.shape == (len(frame),) and np.isfinite(pred).all()
    with pytest.raises(ValueError):
        LevelRegressor(cols, LevelForestParameters(learner="nope")).fit(frame, y)


def test_replicate_noise_floor_reports_within_cell_spread():
    src = _synthetic_source(replicate_rows=6)
    noise = replicate_noise_floor(src)
    assert noise["replicated_cells"] >= 1
    assert noise["mae_floor_median_sd"] >= 0.0 and noise["pooled_within_cell_sd"] >= 0.0


# --------------------------------------------------------------------------- #
# Estimator and controls
# --------------------------------------------------------------------------- #

def test_regressor_learns_the_synthetic_signal_and_beats_the_mean():
    data = build_level_dataset(_synthetic_source(n_extractants=6, n_conditions=6), min_rows_per_extractant=1)
    frame = data.frame
    train = frame[frame["condition_id"] != frame["condition_id"].iloc[0]]
    test = frame[frame["condition_id"] == frame["condition_id"].iloc[0]]
    params = LevelForestParameters(n_estimators=60, n_jobs=1)
    model = LevelRegressor(data.arm_columns("MC_ecfp"), params).fit(
        train, train[LEVEL_TARGET_COLUMN], groups=train["ecfp_cluster"])
    pred = model.predict(test)
    mae = float(np.abs(pred - test[LEVEL_TARGET_COLUMN]).mean())
    trivial = float(np.abs(train[LEVEL_TARGET_COLUMN].mean() - test[LEVEL_TARGET_COLUMN]).mean())
    assert mae < 0.6 * trivial


def test_regressor_is_reproducible_and_rejects_unknown_columns():
    data = build_level_dataset(_synthetic_source(), min_rows_per_extractant=1)
    frame = data.frame
    params = LevelForestParameters(n_estimators=25, n_jobs=1)
    cols = data.arm_columns("MC")
    a = LevelRegressor(cols, params).fit(frame, frame[LEVEL_TARGET_COLUMN]).predict(frame)
    b = LevelRegressor(cols, params).fit(frame, frame[LEVEL_TARGET_COLUMN]).predict(frame)
    assert np.allclose(a, b)
    with pytest.raises(KeyError):
        LevelRegressor(("missing_column",), params).fit(frame, frame[LEVEL_TARGET_COLUMN])
    with pytest.raises(RuntimeError):
        LevelRegressor(cols, params).predict(frame)


def test_group_balanced_weights_equalise_group_totals():
    groups = ["a", "a", "a", "b"]
    w = group_balanced_weights(groups)
    assert np.isclose(w[:3].sum(), w[3])


def test_shuffle_block_permutes_only_the_named_block():
    data = build_level_dataset(_synthetic_source(), min_rows_per_extractant=1)
    frame = data.frame
    block = data.blocks["ECFP"]
    other = data.blocks["COND"]
    out = shuffle_block(frame, block, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(out[list(other)], frame[list(other)])
    assert not out[list(block)].equals(frame[list(block)])
    # the multiset of rows is preserved — only their association with y is destroyed
    assert np.allclose(np.sort(out[list(block)].to_numpy(), axis=0),
                       np.sort(frame[list(block)].to_numpy(), axis=0))


def test_shuffle_block_within_group_keeps_group_membership():
    data = build_level_dataset(_synthetic_source(), min_rows_per_extractant=1)
    frame = data.frame.reset_index(drop=True)
    block = list(data.blocks["COND"])
    out = shuffle_block(frame, block, rng=np.random.default_rng(1), within="extractant")
    for ext, g in frame.groupby("extractant"):
        got = np.sort(out.loc[g.index, block].to_numpy(), axis=0)
        want = np.sort(g[block].to_numpy(), axis=0)
        assert np.allclose(got, want, equal_nan=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_macro_mae_is_equal_weighted_over_groups():
    truth = [0.0] * 10 + [0.0] * 2
    pred = [1.0] * 10 + [3.0] * 2
    groups = ["big"] * 10 + ["small"] * 2
    assert np.isclose(equal_group_macro_mae(truth, pred, groups), 2.0)
    pooled = float(np.mean(np.abs(np.array(pred) - np.array(truth))))
    assert not np.isclose(pooled, 2.0)


def test_level_metric_table_reports_macro_pooled_and_deltas():
    n = 40
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        LEVEL_TARGET_COLUMN: rng.normal(size=n),
        "ecfp_cluster": ["a"] * 20 + ["b"] * 20,
    })
    frame["prediction_good"] = frame[LEVEL_TARGET_COLUMN] + rng.normal(0, 0.1, size=n)
    frame["prediction_bad"] = rng.normal(size=n)
    over, per_group = level_metric_table(frame, ["good", "bad"], baseline_arm="good")
    good = over[over.arm == "good"].iloc[0]
    bad = over[over.arm == "bad"].iloc[0]
    assert good.macro_mae < bad.macro_mae and good.pooled_r2 > bad.pooled_r2
    assert set(per_group["group"]) == {"a", "b"}
    assert np.isclose(over.loc[over.arm == "good", "delta_macro_vs_baseline"].iloc[0], 0.0)
    assert over.loc[over.arm == "bad", "delta_macro_vs_baseline"].iloc[0] < 0


def test_bootstrap_unit_is_the_cluster_not_the_row():
    rng = np.random.default_rng(3)
    # one huge cluster where the candidate wins, many small ones where it loses
    rows = []
    for i in range(200):
        rows.append({"ecfp_cluster": "huge", LEVEL_TARGET_COLUMN: 0.0,
                     "prediction_ref": 1.0, "prediction_cand": 0.0})
    for c in range(12):
        for i in range(3):
            rows.append({"ecfp_cluster": f"small{c}", LEVEL_TARGET_COLUMN: 0.0,
                         "prediction_ref": 0.0, "prediction_cand": 1.0})
    frame = pd.DataFrame(rows)
    out = paired_group_bootstrap(frame, {"ref_vs_cand": ("ref", "cand")}, replicates=400, seed=1)
    row = out.iloc[0]
    assert row.groups_improved == 1 and row.groups_total == 13
    assert row.point_delta_mae < 0            # row-weighted would say the candidate wins
    assert row.p_worse_one_sided > 0.5
    assert row.p_worse_one_sided >= 1 / 401   # add-one floor: never exactly zero
    _ = rng


# --------------------------------------------------------------------------- #
# Fold logic (imported from the script)
# --------------------------------------------------------------------------- #

def _load_script():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_gen5_levels.py"
    spec = importlib.util.spec_from_file_location("run_gen5_levels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_group_kfold_never_splits_a_group_and_seeds_change_the_partition():
    mod = _load_script()
    groups = np.array([f"g{i % 12}" for i in range(120)])
    seen = {}
    for seed in (1, 2, 3):
        labels = np.full(len(groups), -1)
        for k, (tr, te) in enumerate(mod._seeded_group_kfold(groups, 4, seed)):
            assert set(groups[tr]).isdisjoint(set(groups[te]))   # a group is never on both sides
            assert (labels[te] == -1).all()                        # every row tested exactly once
            labels[te] = k
        assert (labels >= 0).all()
        seen[seed] = tuple(labels)
    assert len(set(seen.values())) == 3                            # different seeds -> different partitions


def test_regimes_hold_out_the_right_unit():
    mod = _load_script()
    src = _synthetic_source(n_extractants=6, n_conditions=4)
    data = build_level_dataset(src, min_rows_per_extractant=1)
    frame = data.frame
    for regime, unit in (("unseen_ligand", "ecfp_cluster"), ("unseen_chemotype", "tanimoto_cluster"),
                         ("unseen_series", "series_id"), ("unseen_conditions", "condition_id")):
        groups = mod._fold_groups(frame, regime)
        for tr, te in mod._seeded_group_kfold(groups, 3, 0):
            assert set(frame[unit].iloc[tr]).isdisjoint(set(frame[unit].iloc[te]))
    # unseen_ligand must keep fingerprint twins together
    twins = frame[frame.extractant.isin(["SMILES0", "SMILES1"])]
    assert twins.ecfp_cluster.nunique() == 1


def test_nearest_condition_null_copies_from_the_same_ligand_and_metal():
    mod = _load_script()
    src = _synthetic_source(n_extractants=3, n_conditions=5)
    data = build_level_dataset(src, min_rows_per_extractant=1)
    frame = data.frame
    test = frame[frame.condition_id == frame.condition_id.iloc[0]]
    train = frame.drop(test.index)
    pred = mod._nearest_condition_copy(train, test, data.blocks["COND"],
                                       fallback=train.groupby("extractant")[LEVEL_TARGET_COLUMN].mean(),
                                       global_mean=float(train[LEVEL_TARGET_COLUMN].mean()))
    assert pred.shape == (len(test),)
    # every prediction is an actual training value of the same ligand+metal
    for (e, m), val in zip(zip(test.extractant, test.metal_symbol), pred):
        pool = train[(train.extractant == e) & (train.metal_symbol == m)][LEVEL_TARGET_COLUMN]
        assert np.isclose(pool, val).any()


def test_hgb_survives_an_all_nan_column_in_the_training_fold():
    """unseen_chemotype can leave a descriptor with no observed value at all.

    HGB has no imputer (it keeps NaN on purpose), and its binner used to die with
    "window shape cannot be larger than input array shape".  Regression test for
    the crash that killed run gen5_levels_20260818T180930Z.
    """
    import numpy as np
    from lanthanide_separation.levels import LevelRegressor, LevelForestParameters

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.normal(size=120),
        "empty": np.full(120, np.nan),      # no observed value at all
        "partial": np.where(rng.random(120) < 0.25, np.nan, rng.normal(size=120)),
    })
    y = frame["a"] * 2.0 + rng.normal(scale=0.1, size=120)

    model = LevelRegressor(("a", "empty", "partial"),
                           LevelForestParameters(learner="hgb", n_estimators=10, n_jobs=1))
    model.fit(frame, y)
    assert model.predict(frame).shape == (120,)
    assert int((~model.pipeline.named_steps["drop_empty"].keep_).sum()) == 1


def test_drop_all_nan_columns_is_a_noop_without_empty_columns():
    import numpy as np
    from lanthanide_separation.levels import DropAllNaNColumns

    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    x[:10, 1] = np.nan                      # partial missingness must survive
    out = DropAllNaNColumns().fit_transform(x)
    assert out.shape == (50, 3)
    assert np.isnan(out).sum() == 10


def test_bootstrap_ci_does_not_depend_on_comparison_order():
    """One shared index matrix, so a pair's CI is not a function of its position."""
    import numpy as np
    from lanthanide_separation.levels import paired_group_bootstrap

    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({
        "ecfp_cluster": np.repeat([f"c{i}" for i in range(12)], n // 12),
        "log_D": rng.normal(size=n),
    })
    for arm in ("A", "B", "C"):
        frame[f"prediction_{arm}"] = frame["log_D"] + rng.normal(scale=0.5, size=n)
    comps = {"ab": ("A", "B"), "ac": ("A", "C"), "bc": ("B", "C")}
    first = paired_group_bootstrap(frame, comps, replicates=500).set_index("comparison")
    reordered = {k: comps[k] for k in ("bc", "ac", "ab")}
    second = paired_group_bootstrap(frame, reordered, replicates=500).set_index("comparison")
    for key in comps:
        assert first.loc[key, "ci95_low"] == pytest.approx(second.loc[key, "ci95_low"])
        assert first.loc[key, "ci95_high"] == pytest.approx(second.loc[key, "ci95_high"])


def test_bootstrap_resamples_the_held_out_block_and_keeps_the_point_estimate():
    """Scoring stays per ECFP cluster; resampling follows the coarser held-out unit."""
    import numpy as np
    from lanthanide_separation.levels import paired_group_bootstrap

    rng = np.random.default_rng(4)
    n = 480
    clusters = np.repeat([f"c{i}" for i in range(24)], n // 24)
    frame = pd.DataFrame({
        "ecfp_cluster": clusters,
        # every 3 ECFP clusters nest inside one super-cluster
        "tanimoto_cluster": [f"t{int(c[1:]) // 3}" for c in clusters],
        "log_D": rng.normal(size=n),
    })
    frame["prediction_A"] = frame["log_D"] + rng.normal(scale=0.6, size=n)
    frame["prediction_B"] = frame["log_D"] + rng.normal(scale=0.5, size=n)
    comps = {"ab": ("A", "B")}
    fine = paired_group_bootstrap(frame, comps, replicates=800).iloc[0]
    coarse = paired_group_bootstrap(frame, comps, resample_column="tanimoto_cluster",
                                    replicates=800).iloc[0]
    assert fine["point_delta_mae"] == pytest.approx(coarse["point_delta_mae"])
    assert coarse["bootstrap_blocks"] == 8 and fine["bootstrap_blocks"] == 24
    # fewer independent units must not give a tighter interval
    assert (coarse["ci95_high"] - coarse["ci95_low"]) > (fine["ci95_high"] - fine["ci95_low"])


def test_bootstrap_rejects_a_scoring_group_that_straddles_two_blocks():
    import numpy as np
    from lanthanide_separation.levels import paired_group_bootstrap

    frame = pd.DataFrame({
        "ecfp_cluster": ["a", "a", "b", "b"],
        "series_id": ["s1", "s2", "s1", "s2"],   # cuts across the cluster
        "log_D": [0.0, 1.0, 2.0, 3.0],
        "prediction_A": [0.1, 1.1, 2.1, 3.1],
        "prediction_B": [0.2, 1.2, 2.2, 3.2],
    })
    with pytest.raises(ValueError, match="must nest inside"):
        paired_group_bootstrap(frame, {"ab": ("A", "B")}, resample_column="series_id", replicates=10)


def test_within_ligand_r2_separates_deployable_from_shape():
    """Both are 0 for the ORACLE per-ligand mean; the deployable one goes negative
    as soon as the offset is anything the model did not get for free — which is the
    real deployment case, where only a training mean is available."""
    import numpy as np
    from lanthanide_separation.levels import level_metric_table

    rng = np.random.default_rng(5)
    ext = np.repeat(["e1", "e2", "e3"], 40)
    offsets = {"e1": -2.0, "e2": 0.0, "e3": 3.0}
    y = np.array([offsets[e] for e in ext]) + rng.normal(size=len(ext))
    oracle = pd.Series(y).groupby(ext).transform("mean").to_numpy()

    base = {"extractant": ext, "ecfp_cluster": ext, "log_D": y}
    over, _ = level_metric_table(
        pd.DataFrame({**base, "prediction_MC_ecfp": oracle}), ["MC_ecfp"], baseline_arm="MC_ecfp")
    row = over.iloc[0]
    # the oracle offset is exactly what the shape metric hands out for free
    assert row["within_ligand_r2_shape"] == pytest.approx(0.0, abs=1e-12)
    assert row["within_ligand_r2"] == pytest.approx(0.0, abs=1e-12)

    # a per-ligand constant that is NOT the held-out mean (i.e. a training mean)
    drifted = oracle + np.array([0.4 if e == "e1" else -0.3 for e in ext])
    over2, _ = level_metric_table(
        pd.DataFrame({**base, "prediction_MC_ecfp": drifted}), ["MC_ecfp"], baseline_arm="MC_ecfp")
    row2 = over2.iloc[0]
    assert row2["within_ligand_r2_shape"] == pytest.approx(0.0, abs=1e-12)
    assert row2["within_ligand_r2"] < 0.0
