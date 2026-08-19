"""Tests for the Experiment C models: cross-fit provenance, component attribution, pair identities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.levels import LevelData, LevelForestParameters, group_balanced_weights
from lanthanide_separation.gen6.hierarchical import (
    HierarchicalRidge, cell_frame, derived_pairs, fit_design, fit_two_stage,
    pair_consistency, pair_difference_rows, pair_label_mean_null, pair_metrics,
    predict_two_stage, ridge_feature_frame, tune_ridge,
)


def _synthetic(n_ligands: int = 8, n_conditions: int = 4, metals=(57, 60, 63, 66, 71), seed: int = 0):
    """A level frame whose target really is α_l + F_cond + F_metal + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ligands):
        alpha = rng.normal(scale=1.5)
        d1, d2 = rng.normal(), rng.normal()
        for c in range(n_conditions):
            log_l = rng.uniform(-2, 0)
            log_h = rng.uniform(-1, 1)
            for z in metals:
                radius = 1.2 - 0.01 * (z - 57)
                y = alpha + 2.0 * log_l + 0.5 * log_h + 0.8 * radius * d1 + rng.normal(scale=0.1)
                rows.append({
                    "row_id": f"L{i}-c{c}-{z}", "extractant": f"L{i}", "ecfp_cluster": f"e{i}",
                    "tanimoto_cluster": f"t{i % 3}", "series_id": f"s{i}", "condition_id": f"c{c}",
                    "metal_symbol": f"M{z}", "metal_Z": float(z), "n_replicates": 1, "log_D": y,
                    "Atomic Number_metal": float(z), "lanimoto": 0.0,
                    "lanthanide_index": float(z - 57), "Ionic Radius_metal": radius,
                    "cond__acid_concentration_M": 10 ** log_h, "cond__extractant_concentration_M": 10 ** log_l,
                    "cond__diluent__kerosene": 1.0,
                    "massact__log10_cond__acid_concentration_M": log_h,
                    "massact__log10_cond__extractant_concentration_M": log_l,
                    "massact__logL_x_DENTATE": 3 * log_l,
                    "MolWt": 300 + 20 * d1, "MolLogP": 4 + d2,
                    "donor__O(amide_carbonyl)": 2.0 + (i % 2), "donor__O(ether)": 1.0,
                    "DENTATE": 3.0, "coreCN": 9.0,
                    "lig2d__a": d1, "lig2d__b": d2,
                })
    frame = pd.DataFrame(rows)
    blocks = {
        "METAL": ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal"),
        "COND": ("cond__acid_concentration_M", "cond__extractant_concentration_M", "cond__diluent__kerosene"),
        "PHYSCHEM": ("MolWt", "MolLogP"),
        "DONORS": ("donor__O(amide_carbonyl)", "donor__O(ether)", "DENTATE", "coreCN"),
        "MASSACTION": ("massact__log10_cond__acid_concentration_M",
                       "massact__log10_cond__extractant_concentration_M", "massact__logL_x_DENTATE"),
        "LIG2D_EXT": ("lig2d__a", "lig2d__b"),
    }
    return LevelData(frame=frame, blocks=blocks, audit={})


def _split(frame, held_out=("L0", "L1")):
    test = frame["extractant"].isin(held_out).to_numpy()
    return frame[~test].reset_index(drop=True), frame[test].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Design and ridge
# --------------------------------------------------------------------------- #

def test_design_builds_the_declared_interaction_block_only():
    data = _synthetic()
    features, groups = ridge_feature_frame(data.frame, data)
    interactions = [c for c, g in groups.items() if g == "INTERACTION"]
    assert interactions
    assert all(c.startswith(("ix__donor_x_radius__", "ix__logL_x_donor__", "ix__cond_x_radius__"))
               for c in interactions)
    # no LIG2D_EXT column in the ridge design — deliberately compact
    assert not any(c.startswith("lig2d__") for c in features.columns)


def test_design_transform_is_fold_local_and_reproducible():
    data = _synthetic()
    train, test = _split(data.frame)
    design = fit_design(train, data)
    x_train = design.transform(train)
    assert np.allclose(x_train.mean(axis=0), 0.0, atol=1e-9)
    # the test transform uses TRAINING statistics: its mean need not be zero
    x_test = design.transform(test)
    assert x_test.shape[1] == x_train.shape[1]
    np.testing.assert_allclose(design.transform(train), x_train)


def test_hierarchical_ridge_recovers_a_ligand_intercept_in_training():
    data = _synthetic()
    train, _ = _split(data.frame, held_out=())
    design = fit_design(train, data)
    x = design.transform(train)
    y = train["log_D"].to_numpy()
    w = group_balanced_weights(train["ecfp_cluster"])
    hier = HierarchicalRidge(design, lambda_fixed=10.0, lambda_ligand=1.0).fit(
        x, y, weights=w, ligands=train["extractant"])
    plain = HierarchicalRidge(design, lambda_fixed=10.0, lambda_ligand=None).fit(x, y, weights=w)
    err_hier = np.abs(hier.predict(x, train["extractant"]) - y).mean()
    err_plain = np.abs(plain.predict(x) - y).mean()
    assert err_hier < err_plain                    # intercepts absorb the ligand level
    assert set(hier.gamma_) == set(train["extractant"].unique())


def test_unseen_ligand_gets_zero_intercept():
    data = _synthetic()
    train, test = _split(data.frame)
    design = fit_design(train, data)
    hier = HierarchicalRidge(design, 10.0, 1.0).fit(
        design.transform(train), train["log_D"].to_numpy(), weights=None, ligands=train["extractant"])
    components = hier.predict_components(design.transform(test), test["extractant"])
    assert (components["ligand_intercept"] == 0.0).all()
    # components sum to the prediction (before clamping, which does not bind here)
    parts = components.drop(columns=["prediction"]).sum(axis=1)
    np.testing.assert_allclose(parts, components["prediction"], atol=1e-9)


def test_tune_ridge_searches_the_full_grid_and_returns_its_argmin():
    """The choice must be the inner-CV argmin over the whole declared grid.

    (No direction is asserted: on synthetic data where the ligand level is pure
    noise, a SMALL ligand penalty can win under a ligand hold-out, because the
    intercepts absorb that noise and the transferable slopes are estimated more
    cleanly — a legitimate mechanism. On the real cohort the inner CV chose the
    plain-ridge limit under a chemotype hold-out; that is a finding, not a test.)
    """
    from lanthanide_separation.gen6.hierarchical import FIXED_PENALTY_GRID, LIGAND_PENALTY_GRID
    data = _synthetic(n_ligands=12)
    train, _ = _split(data.frame, held_out=("L0",))
    tuned = tune_ridge(train, data, group_column="extractant", hierarchical=True, seed=3)
    table = tuned["inner_table"]
    assert len(table) == len(FIXED_PENALTY_GRID) * len(LIGAND_PENALTY_GRID)
    best = table.sort_values("inner_macro_mae", kind="stable").iloc[0]
    assert tuned["lambda_fixed"] == best["lambda_fixed"]
    assert tuned["lambda_ligand"] == best["lambda_ligand"]
    assert tuned["lambda_ligand"] in LIGAND_PENALTY_GRID


def test_tune_ridge_plain_has_no_ligand_penalty():
    data = _synthetic()
    train, _ = _split(data.frame)
    tuned = tune_ridge(train, data, group_column="extractant", hierarchical=False, seed=3)
    assert tuned["lambda_ligand"] is None


# --------------------------------------------------------------------------- #
# Two-stage: cross-fit provenance and oracles
# --------------------------------------------------------------------------- #

def test_cell_frame_has_one_row_per_cell_with_series():
    data = _synthetic()
    cells = cell_frame(data.frame, data)
    assert len(cells) == data.frame.groupby(["extractant", "condition_id"]).ngroups
    assert (cells["n_metals"] == 5).all()
    assert "series_id" in cells.columns
    assert not any(c.startswith("Atomic") for c in cells.columns)   # no metal feature in Stage A


def test_two_stage_residuals_are_cross_fitted():
    """No training row's residual may come from a Stage A model that saw its cell."""
    data = _synthetic(n_ligands=9)
    train, test = _split(data.frame, held_out=("L0",))
    params = LevelForestParameters(n_estimators=20, random_state=1, n_jobs=1)
    fit = fit_two_stage(train, data, params=params, group_column="tanimoto_cluster", seed=5, crossfit_folds=3)
    audit = fit.crossfit_audit
    # every row got an inner fold, and the inner fold of a row equals the inner
    # fold of its cell (the residual was formed from that cell's OOF prediction)
    assert (audit["row_inner_fold"] >= 0).all()
    cell_fold = dict(zip(audit["cell_keys"], audit["cell_inner_fold"]))
    assert all(cell_fold[k] == f for k, f in zip(audit["row_cell_keys"], audit["row_inner_fold"]))
    # and the inner folds were grouped on chemotype: one chemotype never sits in two inner folds
    cells = cell_frame(train, data)
    by_chemotype = pd.DataFrame({"t": cells["tanimoto_cluster"], "f": audit["cell_inner_fold"]})
    assert (by_chemotype.groupby("t")["f"].nunique() == 1).all()


def test_two_stage_oracles_bracket_the_prediction():
    data = _synthetic(n_ligands=9)
    train, test = _split(data.frame, held_out=("L0", "L1"))
    params = LevelForestParameters(n_estimators=30, random_state=1, n_jobs=1)
    fit = fit_two_stage(train, data, params=params, group_column="tanimoto_cluster", seed=5, crossfit_folds=3)
    out = predict_two_stage(fit, test, data)
    y = test["log_D"].to_numpy()
    assert np.isfinite(out["prediction_C2_TWO_STAGE"]).all()
    # ORACLE_METAL error is exactly Stage A's level error: |Â − ȳ_cell|
    np.testing.assert_allclose(np.abs(out["prediction_ORACLE_METAL"] - y),
                               np.abs(out["stage_a"] - out["cell_true_mean"]))
    # ORACLE_LEVEL error is exactly the metal-response error: |B̂_true − (y − ȳ_cell)|
    np.testing.assert_allclose(np.abs(out["prediction_ORACLE_LEVEL"] - y),
                               np.abs(out["stage_b_truecentre"] - (y - out["cell_true_mean"])))
    # on this synthetic data the level is the whole story: the level oracle is far better
    assert np.abs(out["prediction_ORACLE_LEVEL"] - y).mean() < np.abs(out["prediction_ORACLE_METAL"] - y).mean()


# --------------------------------------------------------------------------- #
# Pairs: antisymmetry, transitivity, null
# --------------------------------------------------------------------------- #

def test_derived_pairs_are_antisymmetric_and_transitive_exactly():
    data = _synthetic(n_ligands=3)
    frame = data.frame.copy()
    rng = np.random.default_rng(0)
    frame["prediction_X"] = frame["log_D"] + rng.normal(scale=0.3, size=len(frame))
    pairs = derived_pairs(frame, ["prediction_X"])
    assert len(pairs) == 3 * 4 * (5 * 4 // 2)            # ligands x conditions x C(5,2)
    assert (pairs["delta_z"] > 0).all()                    # A is always the lighter metal
    consistency = pair_consistency(pairs, "pair_prediction_X")
    assert consistency["n_triples"] > 0
    assert consistency["transitivity_max"] < 1e-9
    assert consistency["antisymmetry_max"] < 1e-12


def test_pair_metrics_and_null():
    data = _synthetic(n_ligands=4)
    train, test = _split(data.frame, held_out=("L0",))
    test = test.assign(prediction_X=test["log_D"])        # perfect level predictions
    pairs = derived_pairs(test, ["prediction_X"])
    perfect = pair_metrics(pairs, "pair_prediction_X")
    assert perfect["pair_macro_mae"] == pytest.approx(0.0, abs=1e-12)
    assert perfect["pair_sign_accuracy"] == pytest.approx(1.0)
    null = pair_label_mean_null(train, pairs)
    assert null.shape == (len(pairs),)
    pairs["pair_NULL"] = null
    assert pair_metrics(pairs, "pair_NULL")["pair_macro_mae"] > 0


def test_pair_difference_rows_cancel_ligand_level():
    """x_A − x_B removes every ligand-constant column, which is the point of C3."""
    data = _synthetic(n_ligands=3)
    design = fit_design(data.frame, data)
    x = design.transform(data.frame)
    w = np.ones(len(x))
    dx, dy, dw, table = pair_difference_rows(data.frame, x, w)
    assert len(dx) == len(table) > 0
    ligand_cols = [i for i, c in enumerate(design.columns)
                   if design.groups[c] in ("PHYSCHEM", "DONORS")]
    assert np.allclose(dx[:, ligand_cols], 0.0)
    assert np.all(dw > 0)


def test_antisymmetry_residual_is_measured_not_hard_coded():
    """A corrupted derived pair must show up as a non-zero antisymmetry residual."""
    data = _synthetic(n_ligands=2)
    frame = data.frame.copy()
    frame["prediction_X"] = frame["log_D"]
    pairs = derived_pairs(frame, ["prediction_X"])
    assert pair_consistency(pairs, "pair_prediction_X")["antisymmetry_max"] == pytest.approx(0.0)
    corrupted = pairs.copy()
    corrupted.loc[0, "pair_prediction_X"] += 0.25
    assert pair_consistency(corrupted, "pair_prediction_X")["antisymmetry_max"] == pytest.approx(0.25)
