"""Tests for gen6 Experiment B — the diversity-versus-depth learning curve.

The curve is only interpretable if four things are literally true, and each has a
test here:

* **the budget is a budget** — every policy spends exactly the row count it was
  given, or the whole pool when the pool is smaller.  A policy that quietly
  under-spends would look better or worse for a reason that has nothing to do
  with chemistry;
* **the policies are label-free** — permuting ``log_D`` must not move a single
  selected row index.  An acquisition policy that peeked at the target would be
  selecting easy rows, and the whole experiment would be circular;
* **the curve table has exactly one row per (split seed, policy, budget, draw)**,
  with the folds aggregated — a duplicated or missing point silently reweights
  every average taken over the table;
* **``all`` is the whole training pool** — the right-hand end of the curve must be
  the un-subsampled model, which is what ties Experiment B back to Experiment A.

Two further properties are checked because the script depends on them for speed
and for the nesting of the curve: the vectorised nearest-neighbour helper agrees
with ``ChemistryMap.nearest_neighbour``, and the fit cache returns the identical
prediction it computed the first time instead of re-fitting.

The script is loaded through ``importlib`` (it lives in ``scripts/``, which is not
an importable package) and registered in ``sys.modules`` first, because a module
holding dataclasses cannot be executed while absent from ``sys.modules``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.chemistry import ChemistryMap, build_chemistry_map
from lanthanide_separation.gen6.cohorts import (
    ACQUISITION_POLICIES, BASE_ARM, EXPANDED_ARM, diversity_splits,
)
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN, build_level_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_diversity_learning_curve.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_diversity_learning_curve", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registering before execution is required: dataclasses resolve their own
    # module out of sys.modules while the class body is being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lc = _load_script()

METALS = {"La": (57, 1.160), "Nd": (60, 1.109), "Eu": (63, 1.066), "Dy": (66, 1.027)}


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _synthetic_source(
    n_dense: int = 6, n_sparse: int = 4, dense_conditions: int = 6, sparse_conditions: int = 1,
    bits_per_extractant: int = 4, seed: int = 0,
) -> pd.DataFrame:
    """A raw-dataset-shaped frame with dense and sparse extractants.

    Fingerprints are disjoint bit blocks, so every extractant is its own ECFP
    cluster *and* its own Tanimoto super-cluster: the fold structure is then
    completely determined by the number of extractants, which keeps the tests
    from depending on an accidental clustering of random bits.

    Dense extractants get ``dense_conditions x 4`` cells (>= 10, so they are in
    BASE); sparse ones get ``sparse_conditions x 4`` (>= 3, so EXPANDED only).
    """
    rng = np.random.default_rng(seed)
    n_extractants = n_dense + n_sparse
    n_bits = bits_per_extractant * n_extractants
    rows: list[dict] = []
    for e in range(n_extractants):
        conditions = dense_conditions if e < n_dense else sparse_conditions
        level = float(rng.normal(0.0, 1.5))
        for c in range(conditions):
            acid = float(rng.uniform(0.1, 5.0))
            for metal, (z, radius) in METALS.items():
                row = {
                    "canonical_smiles": f"SMILES{e:02d}",
                    "metal_symbol": metal,
                    "Atomic Number_metal": float(z),
                    "Ionic Radius_metal": radius,
                    "lanthanide_index": float(z - 57),
                    LEVEL_TARGET_COLUMN: level + 0.35 * acid + 0.08 * (z - 60)
                    + float(rng.normal(0.0, 0.1)),
                    "cond__acid_concentration_M": acid,
                    "cond__temperature_C": 25.0 + c,
                    "cond__diluent": "kerosene" if c % 2 else "dodecane",
                    "MolWt": 300.0 + 10 * e,
                    "MolLogP": 3.0 + 0.1 * e,
                }
                row.update({f"ecfp_{b}": int(bits_per_extractant * e <= b
                                             < bits_per_extractant * (e + 1))
                            for b in range(n_bits)})
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cohort() -> tuple[pd.DataFrame, ChemistryMap]:
    source = _synthetic_source()
    data = build_level_dataset(source, min_rows_per_extractant=3, drop_below_log_d=None)
    chemistry = build_chemistry_map(source)
    return data.frame.reset_index(drop=True), chemistry


@pytest.fixture(scope="module")
def pool(cohort) -> np.ndarray:
    """The full training pool of the first fold — what a budget is drawn from."""
    frame, _ = cohort
    splits = diversity_splits(frame, n_splits=3, seed=104729, arms=(BASE_ARM, EXPANDED_ARM))
    return splits[0].train_index_by_arm[EXPANDED_ARM]


# --------------------------------------------------------------------------- #
# The budget is a budget
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy", list(ACQUISITION_POLICIES))
@pytest.mark.parametrize("budget", [7, 25, 60])
def test_every_policy_spends_the_budget_exactly(cohort, pool, policy, budget):
    frame, chemistry = cohort
    assert budget < len(pool), "the fixture must be able to under-spend the pool"
    for draw in range(3):
        point = lc.CurvePoint(policy=policy, budget=budget, draw=draw)
        selection = lc.select_training_rows(
            frame, pool, point, chemistry=chemistry, split_seed=104729, fold=0)
        assert len(selection) == budget
        assert len(set(selection.tolist())) == budget, "rows must not be selected twice"
        assert set(selection.tolist()) <= set(pool.tolist()), "selection must come from the pool"


@pytest.mark.parametrize("policy", list(ACQUISITION_POLICIES))
def test_all_budget_is_the_full_training_pool(cohort, pool, policy):
    frame, chemistry = cohort
    point = lc.CurvePoint(policy=policy, budget=None, draw=0)
    selection = lc.select_training_rows(
        frame, pool, point, chemistry=chemistry, split_seed=104729, fold=0)
    np.testing.assert_array_equal(selection, np.sort(pool))


@pytest.mark.parametrize("policy", list(ACQUISITION_POLICIES))
def test_budget_larger_than_the_pool_yields_the_pool(cohort, pool, policy):
    """The honest shortfall: a budget the fold cannot afford collapses to the pool."""
    frame, chemistry = cohort
    point = lc.CurvePoint(policy=policy, budget=len(pool) + 500, draw=0)
    selection = lc.select_training_rows(
        frame, pool, point, chemistry=chemistry, split_seed=104729, fold=0)
    np.testing.assert_array_equal(selection, np.sort(pool))


@pytest.mark.parametrize("policy", list(ACQUISITION_POLICIES))
def test_draws_within_a_policy_are_nested_across_budgets(cohort, pool, policy):
    """The acquisition RNG must not see the budget, or the curve's slope is noise.

    Ligands enter whole and in the policy's order, so the extractants bought at a
    smaller budget must still be present at a larger one within the same draw.
    """
    frame, chemistry = cohort
    extractants = frame["extractant"].to_numpy()
    seen: set[str] = set()
    for budget in (10, 30, 70):
        point = lc.CurvePoint(policy=policy, budget=budget, draw=1)
        selection = lc.select_training_rows(
            frame, pool, point, chemistry=chemistry, split_seed=104729, fold=0)
        chosen = set(extractants[selection])
        assert seen <= chosen, f"{policy}: budget {budget} dropped previously acquired chemistry"
        seen = chosen


# --------------------------------------------------------------------------- #
# The policies are label-free
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy", list(ACQUISITION_POLICIES))
@pytest.mark.parametrize("budget", [25, None])
def test_policies_are_label_free(cohort, pool, policy, budget):
    """Permuting ``log_D`` must not move one selected row index."""
    frame, chemistry = cohort
    rng = np.random.default_rng(11)
    permuted = frame.copy()
    permuted[LEVEL_TARGET_COLUMN] = frame[LEVEL_TARGET_COLUMN].to_numpy()[rng.permutation(len(frame))]
    assert not np.allclose(permuted[LEVEL_TARGET_COLUMN], frame[LEVEL_TARGET_COLUMN])

    point = lc.CurvePoint(policy=policy, budget=budget, draw=0)
    original = lc.select_training_rows(frame, pool, point, chemistry=chemistry,
                                       split_seed=104729, fold=0)
    shuffled = lc.select_training_rows(permuted, pool, point, chemistry=chemistry,
                                       split_seed=104729, fold=0)
    np.testing.assert_array_equal(original, shuffled)


def test_label_free_helper_reports_pass_on_the_real_policies(cohort, pool):
    frame, chemistry = cohort
    points = lc.curve_points(list(ACQUISITION_POLICIES), [15, None], draws=2)
    report = lc.selection_is_label_free(frame, pool, points, chemistry=chemistry,
                                        split_seed=104729, fold=0)
    assert report["ok"] is True
    assert report["points_that_moved"] == []
    assert report["n_points_checked"] == len(points)


def test_label_free_helper_would_catch_a_target_peeking_policy(cohort, pool, monkeypatch):
    """The guard has to be able to fail, or it is decoration.

    A fake policy that sorts rows by ``log_D`` is injected in place of the real
    selection; the helper must report it as moved.
    """
    frame, chemistry = cohort

    def peeking_selection(frame_, pool_, point, **kwargs):
        order = np.argsort(frame_[LEVEL_TARGET_COLUMN].to_numpy()[pool_])
        return np.sort(pool_[order[: point.budget or len(pool_)]])

    monkeypatch.setattr(lc, "select_training_rows", peeking_selection)
    report = lc.selection_is_label_free(
        frame, pool, [lc.CurvePoint(policy="random", budget=20, draw=0)],
        chemistry=chemistry, split_seed=104729, fold=0)
    assert report["ok"] is False
    assert report["points_that_moved"] == ["random@20#d0"]


# --------------------------------------------------------------------------- #
# Chemistry helper and fit cache
# --------------------------------------------------------------------------- #

def test_vectorised_nearest_neighbour_matches_the_chemistry_map(cohort):
    """The fast path must equal the public, slow, per-query implementation."""
    frame, chemistry = cohort
    names = list(chemistry.extractants)
    query, reference = names[:4], names[2:]
    fast = lc.max_similarity_to_reference(chemistry, query, reference)
    slow = chemistry.nearest_neighbour(query, reference)["nn_tanimoto"].to_numpy(dtype=float)
    np.testing.assert_allclose(fast, slow, atol=1e-6)


def test_nearest_neighbour_helper_handles_an_empty_reference(cohort):
    _, chemistry = cohort
    out = lc.max_similarity_to_reference(chemistry, list(chemistry.extractants)[:3], [])
    np.testing.assert_array_equal(out, np.zeros(3))


def test_nearest_neighbour_helper_is_loud_about_unknown_extractants(cohort):
    _, chemistry = cohort
    with pytest.raises(KeyError, match="frozen chemistry map"):
        lc.max_similarity_to_reference(chemistry, ["not-a-ligand"], list(chemistry.extractants))


def test_fit_cache_reuses_an_identical_training_set(cohort):
    frame, _ = cohort
    from lanthanide_separation.levels import LevelForestParameters

    columns = ("Atomic Number_metal", "cond__acid_concentration_M", "cond__temperature_C")
    cache = lc.FoldModelCache(columns, LevelForestParameters(n_estimators=8, n_jobs=1))
    train = np.arange(0, 80)
    test = np.arange(80, 100)
    first = cache.predict(frame, train, test, split_seed=1, fold=0)
    second = cache.predict(frame, train, test, split_seed=1, fold=0)
    np.testing.assert_array_equal(first, second)
    assert (cache.n_fits, cache.n_hits) == (1, 1)
    # a different training set is a different model, and must actually be fitted
    cache.predict(frame, np.arange(0, 60), test, split_seed=1, fold=0)
    assert cache.n_fits == 2


# --------------------------------------------------------------------------- #
# End to end: the shape of the curve table
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pilot_run(tmp_path_factory) -> tuple[int, Path, pd.DataFrame]:
    """One tiny end-to-end run of the script on synthetic data."""
    workspace = tmp_path_factory.mktemp("gen6_curve")
    dataset = workspace / "dataset.parquet"
    _synthetic_source().to_parquet(dataset, index=False)
    output = workspace / "run"
    status = lc.main([
        "--dataset", str(dataset), "--descriptors", "", "--feature-set", "MC",
        "--budgets", "30", "all", "--draws", "2", "--split-seeds", "104729",
        "--folds", "3", "--n-estimators", "20", "--replicates", "50", "--n-jobs", "1",
        "--eval-min-cells", "3", "--output-dir", str(output),
    ])
    curve = pd.read_csv(output / "diversity_learning_curve.csv")
    return status, output, curve


def test_curve_has_one_row_per_seed_policy_budget_draw(pilot_run):
    status, _, curve = pilot_run
    assert status == 0
    n_policies, n_budgets, n_draws, n_seeds = len(ACQUISITION_POLICIES), 2, 2, 1
    assert len(curve) == n_seeds * n_policies * n_budgets * n_draws
    assert not curve.duplicated(["split_seed", "policy", "budget", "draw"]).any()
    assert set(curve["policy"]) == set(ACQUISITION_POLICIES)
    assert set(curve["budget"]) == {"30", "all"}
    assert set(curve["draw"]) == {0, 1}
    # folds are aggregated into the point: every point scores the whole cohort once
    assert curve["n_rows"].nunique() == 1


def test_all_budget_points_agree_across_policies_and_draws(pilot_run):
    """At budget ``all`` every policy trains on the same pool, so the metrics must coincide."""
    _, _, curve = pilot_run
    at_all = curve[curve["budget"] == "all"]
    assert len(at_all) == len(ACQUISITION_POLICIES) * 2
    assert at_all["macro_mae"].nunique() == 1
    assert at_all["offset_mae"].nunique() == 1


def test_run_writes_the_standard_artifact_set_and_validates(pilot_run):
    _, output, _ = pilot_run
    import json

    for name in ("diversity_learning_curve.csv", "acquisition_audit.csv",
                 "learning_curve_bootstrap.csv", "oof_predictions.parquet",
                 "acquisition_selections.json", "decision_report.md", "summary.json",
                 "manifest.json", "artifact_hashes.json", "validation.json", "_SUCCESS.json",
                 "log.txt"):
        assert (output / name).is_file(), f"missing artifact {name}"
    validation = json.loads((output / "validation.json").read_text())
    assert validation["ok"] is True, validation
    assert validation["checks"]["curve_shape"]["ok"] is True
    assert validation["checks"]["budgets_met_exactly"]["ok"] is True
    assert validation["checks"]["label_free_selection"]["ok"] is True
    assert validation["checks"]["split_integrity"]["ok"] is True
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["split_definition"]["experiment"] == "B_diversity_vs_depth"
    assert len(manifest["folds"]) == 3


def test_acquisition_audit_records_the_chemistry_every_point_bought(pilot_run):
    _, output, curve = pilot_run
    audit = pd.read_csv(output / "acquisition_audit.csv")
    # one row per (seed, fold, policy, budget, draw)
    assert len(audit) == 3 * len(ACQUISITION_POLICIES) * 2 * 2
    assert {"n_rows", "n_extractants", "n_ecfp_clusters", "n_superclusters",
            "budget_met", "pool_rows", "selection_sha256"} <= set(audit.columns)
    spent = audit[(audit["budget"] == "30") & audit["budget_met"]]
    assert (spent["n_rows"] == 30).all()
    # the point of the experiment: at an equal budget the policies buy different chemistry
    by_policy = spent.groupby("policy")["n_superclusters"].mean()
    assert by_policy["diversity"] > by_policy["depth"]


def test_curve_carries_both_hard_chemistry_readings(pilot_run):
    """Own-training and BASE-referenced bins are both present and clearly named."""
    _, _, curve = pilot_run
    own = [c for c in curve.columns if c.startswith("hard_own_train_nn")]
    fixed = [c for c in curve.columns if c.startswith("hard_vs_BASEtrain_nn")]
    assert any(c.endswith("__macro_mae") for c in own)
    assert any(c.endswith("__macro_mae") for c in fixed)
    for threshold in ("0.4", "0.6"):
        assert f"hard_own_train_nn{threshold}__macro_mae" in curve.columns
        assert f"hard_vs_BASEtrain_nn{threshold}__macro_mae" in curve.columns


def test_bootstrap_compares_diversity_against_depth_at_every_budget(pilot_run):
    _, output, _ = pilot_run
    boot = pd.read_csv(output / "learning_curve_bootstrap.csv")
    for budget in ("30", "all"):
        row = boot[(boot["comparison"] == f"diversity_vs_depth@{budget}")
                   & (boot["statistic"] == "mae") & (boot["endpoint"] == "overall")]
        assert len(row) == 1
        assert row.iloc[0]["reference"] == f"depth@{budget}"
        assert row.iloc[0]["candidate"] == f"diversity@{budget}"
        assert row.iloc[0]["ci95_low"] <= row.iloc[0]["ci95_high"]
    # at budget "all" both policies are the identical model, so the delta is exactly zero
    at_all = boot[(boot["comparison"] == "diversity_vs_depth@all")
                  & (boot["statistic"] == "mae") & (boot["endpoint"] == "overall")]
    assert abs(float(at_all.iloc[0]["point_delta"])) < 1e-12


def test_hypotheses_are_scored_and_reported(pilot_run):
    import json

    _, output, _ = pilot_run
    summary = json.loads((output / "summary.json").read_text())
    for key in ("B1", "B2", "B3"):
        assert key in summary["hypotheses"]
        assert summary["hypotheses"][key]["verdict"]
    report = (output / "decision_report.md").read_text()
    assert "B1" in report and "B2" in report and "B3" in report
    assert "hard_vs_BASEtrain" in report
