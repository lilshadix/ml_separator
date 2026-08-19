"""Tests for the gen6 BASE/EXPANDED split machinery and the acquisition policies.

The experiment is only interpretable if three things are literally true: the test
rows do not move between arms, no held-out chemistry appears in any arm's
training rows, and the acquisition policies never look at the target. Each has a
test here, and the fold algorithm is checked against an independently written
reference so it cannot drift away from gen5's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.cohorts import (
    ACQUISITION_POLICIES, BASE_ARM, DEFAULT_ARMS, EXPANDED_ARM, ROWMATCHED_ARM, SHUFFLED_ARM,
    acquisition_audit, arm_membership, assert_split_integrity, cells_per_extractant,
    cohort_comparison, diversity_splits, eligible_extractants, ligand_acquisition_order,
    row_budget_subsample, seeded_group_kfold,
)


def _cohort(n_dense: int = 8, n_sparse: int = 6, dense_cells: int = 12, sparse_cells: int = 4,
            seed: int = 7) -> pd.DataFrame:
    """A synthetic shared cohort: dense and sparse extractants across super-clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dense + n_sparse):
        dense = i < n_dense
        name = f"{'D' if dense else 'S'}{i:02d}"
        n_cells = dense_cells if dense else sparse_cells
        for j in range(n_cells):
            rows.append({
                "row_id": f"{name}-{j:03d}",
                "extractant": name,
                "condition_id": f"cond{j:03d}",
                "metal_symbol": ["Nd", "Eu", "Yb"][j % 3],
                "ecfp_cluster": f"e{i:02d}",
                "tanimoto_cluster": f"t{i % 5:02d}",
                "log_D": float(rng.normal(loc=i * 0.3, scale=1.0)),
            })
    return pd.DataFrame(rows)


def _reference_fold_assignment(groups, n_splits, seed):
    """Independent re-derivation of gen5's fold algorithm (run_gen5_levels._seeded_group_kfold)."""
    labels = np.asarray([str(g) for g in groups])
    names = np.unique(labels)
    dealt = np.random.default_rng(seed).permutation(names)
    fold_of = {g: i % n_splits for i, g in enumerate(dealt)}
    return np.array([fold_of[g] for g in labels])


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #

def test_cells_per_extractant_counts_cells_not_replicate_rows():
    frame = _cohort(n_dense=1, n_sparse=0, dense_cells=5)
    duplicated = pd.concat([frame, frame], ignore_index=True)   # every cell measured twice
    assert cells_per_extractant(duplicated).iloc[0] == 5


def test_cells_per_extractant_requires_the_cell_columns():
    with pytest.raises(KeyError, match="cell columns"):
        cells_per_extractant(pd.DataFrame({"extractant": ["A"]}))


def test_arm_membership_splits_dense_from_sparse():
    frame = _cohort()
    dense = arm_membership(frame, min_cells=10)
    assert set(frame.loc[dense, "extractant"]) == {f"D{i:02d}" for i in range(8)}
    assert set(frame.loc[~dense, "extractant"]) == {f"S{i:02d}" for i in range(8, 14)}
    assert eligible_extractants(frame, min_cells=3) == tuple(sorted(frame["extractant"].unique()))


# --------------------------------------------------------------------------- #
# Fold algorithm
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [104729, 130363, 155921])
def test_fold_algorithm_matches_the_gen5_reference(seed):
    frame = _cohort()
    groups = frame["tanimoto_cluster"].to_numpy()
    reference = _reference_fold_assignment(groups, 5, seed)
    for fold, (train, test) in enumerate(seeded_group_kfold(groups, 5, seed)):
        np.testing.assert_array_equal(test, np.flatnonzero(reference == fold))
        np.testing.assert_array_equal(train, np.flatnonzero(reference != fold))


def test_fold_algorithm_keeps_a_group_whole():
    frame = _cohort()
    groups = frame["tanimoto_cluster"].to_numpy()
    for train, test in seeded_group_kfold(groups, 5, 104729):
        assert not set(groups[train]) & set(groups[test])


def test_different_seeds_give_different_partitions():
    frame = _cohort(n_dense=10, n_sparse=10)
    groups = frame["ecfp_cluster"].to_numpy()
    a = [tuple(t) for _, t in seeded_group_kfold(groups, 5, 1)]
    b = [tuple(t) for _, t in seeded_group_kfold(groups, 5, 2)]
    assert a != b


# --------------------------------------------------------------------------- #
# The two guarantees the experiment stands on
# --------------------------------------------------------------------------- #

def test_identical_test_rows_across_arms():
    frame = _cohort()
    for split in diversity_splits(frame, n_splits=5, seed=104729):
        # one test index per fold, shared by every arm by construction
        assert split.test_index.size > 0
        for arm in split.arms:
            assert not set(split.train_index_by_arm[arm].tolist()) & set(split.test_index.tolist())


def test_no_chemistry_leakage_and_full_coverage():
    frame = _cohort()
    splits = diversity_splits(frame, n_splits=5, seed=104729)
    report = assert_split_integrity(frame, splits)
    assert report["ok"] is True
    assert report["all_rows_tested_once"] is True
    assert report["n_rows_never_tested"] == 0
    assert all(entry["leaks"] == {} for entry in report["folds"])


def test_integrity_check_catches_an_injected_leak():
    frame = _cohort()
    splits = diversity_splits(frame, n_splits=5, seed=104729)
    broken = splits[0]
    leaked = np.concatenate([broken.train_index_by_arm[BASE_ARM], broken.test_index[:1]])
    broken.train_index_by_arm[BASE_ARM] = leaked
    report = assert_split_integrity(frame, splits)
    assert report["ok"] is False
    assert report["folds"][0]["leaks"][BASE_ARM]["row_overlap"] is True


def test_base_training_rows_are_a_strict_subset_of_expanded():
    frame = _cohort()
    for split in diversity_splits(frame, n_splits=5, seed=104729):
        base = set(split.train_index_by_arm[BASE_ARM].tolist())
        expanded = set(split.train_index_by_arm[EXPANDED_ARM].tolist())
        assert base < expanded
        assert set(frame.iloc[sorted(base)]["extractant"]).isdisjoint(
            {f"S{i:02d}" for i in range(8, 14)})


def test_rowmatched_arm_matches_the_base_row_budget_and_keeps_all_sparse():
    frame = _cohort()
    for split in diversity_splits(frame, n_splits=5, seed=104729):
        base = split.train_index_by_arm[BASE_ARM]
        matched = split.train_index_by_arm[ROWMATCHED_ARM]
        assert len(matched) == len(base)
        sparse_rows = split.train_index_by_arm[EXPANDED_ARM][
            ~np.isin(split.train_index_by_arm[EXPANDED_ARM], base)]
        assert set(sparse_rows.tolist()) <= set(matched.tolist())
        assert split.audit["rowmatch_overshoot_rows"] == 0


def test_shuffled_arm_permutes_only_the_added_sparse_targets():
    frame = _cohort()
    target = frame["log_D"].to_numpy()
    for split in diversity_splits(frame, n_splits=5, seed=104729):
        index = split.train_index_by_arm[SHUFFLED_ARM]
        shuffled = split.training_target(SHUFFLED_ARM, target)
        honest = target[index]
        dense_positions = np.flatnonzero(np.isin(index, split.train_index_by_arm[BASE_ARM]))
        sparse_positions = np.flatnonzero(~np.isin(index, split.train_index_by_arm[BASE_ARM]))
        np.testing.assert_allclose(shuffled[dense_positions], honest[dense_positions])
        # same multiset of sparse targets, different order (with enough sparse rows)
        np.testing.assert_allclose(np.sort(shuffled[sparse_positions]),
                                   np.sort(honest[sparse_positions]))
        if sparse_positions.size > 3:
            assert not np.allclose(shuffled[sparse_positions], honest[sparse_positions])


def test_honest_arms_do_not_touch_the_target():
    frame = _cohort()
    target = frame["log_D"].to_numpy()
    for split in diversity_splits(frame, n_splits=5, seed=104729):
        for arm in (BASE_ARM, EXPANDED_ARM, ROWMATCHED_ARM):
            np.testing.assert_allclose(split.training_target(arm, target),
                                       target[split.train_index_by_arm[arm]])


def test_splits_are_deterministic_for_a_seed():
    frame = _cohort()
    a = diversity_splits(frame, n_splits=5, seed=104729)
    b = diversity_splits(frame, n_splits=5, seed=104729)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.test_index, y.test_index)
        for arm in DEFAULT_ARMS:
            np.testing.assert_array_equal(x.train_index_by_arm[arm], y.train_index_by_arm[arm])


def test_split_construction_rejects_unknown_arms_and_missing_columns():
    frame = _cohort()
    with pytest.raises(ValueError, match="unknown arms"):
        diversity_splits(frame, arms=("NOPE",))
    with pytest.raises(KeyError, match="fold-grouping column"):
        diversity_splits(frame.drop(columns=["tanimoto_cluster"]))


def test_splits_are_label_free():
    """Permuting log_D must not change a single training or test index."""
    frame = _cohort()
    permuted = frame.assign(log_D=np.random.default_rng(0).permutation(frame["log_D"].to_numpy()))
    for a, b in zip(diversity_splits(frame, seed=104729), diversity_splits(permuted, seed=104729)):
        np.testing.assert_array_equal(a.test_index, b.test_index)
        for arm in DEFAULT_ARMS:
            np.testing.assert_array_equal(a.train_index_by_arm[arm], b.train_index_by_arm[arm])


# --------------------------------------------------------------------------- #
# Acquisition policies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy", ACQUISITION_POLICIES)
def test_acquisition_policies_are_label_free(policy):
    frame = _cohort()
    index = np.arange(len(frame))
    names = sorted(frame["extractant"].unique())
    similarity = np.eye(len(names)) + 0.1
    kwargs = dict(policy=policy, similarity=similarity, similarity_names=names)
    permuted = frame.assign(log_D=np.random.default_rng(3).permutation(frame["log_D"].to_numpy()))
    a = ligand_acquisition_order(frame, index, rng=np.random.default_rng(5), **kwargs)
    b = ligand_acquisition_order(permuted, index, rng=np.random.default_rng(5), **kwargs)
    assert a == b
    assert sorted(a) == names


def test_depth_policy_prefers_the_best_measured_ligands():
    frame = _cohort()
    order = ligand_acquisition_order(frame, np.arange(len(frame)), policy="depth",
                                     rng=np.random.default_rng(0))
    assert all(name.startswith("D") for name in order[:8])


def test_diversity_policy_covers_superclusters_first():
    frame = _cohort()
    order = ligand_acquisition_order(frame, np.arange(len(frame)), policy="diversity",
                                     rng=np.random.default_rng(0))
    first_five = frame.set_index("extractant").loc[order[:5], "tanimoto_cluster"]
    assert first_five.nunique() == 5      # one ligand from each of the five super-clusters


def test_maxmin_policy_needs_a_similarity_matrix():
    frame = _cohort()
    with pytest.raises(ValueError, match="similarity matrix"):
        ligand_acquisition_order(frame, np.arange(len(frame)), policy="maxmin",
                                 rng=np.random.default_rng(0))


def test_maxmin_walks_away_from_what_it_already_has():
    names = ["A", "B", "C", "D"]
    # A and B are near-duplicates; C is middling; D is far from everything
    similarity = np.array([
        [1.0, 0.95, 0.40, 0.05],
        [0.95, 1.0, 0.42, 0.06],
        [0.40, 0.42, 1.0, 0.20],
        [0.05, 0.06, 0.20, 1.0],
    ])
    frame = pd.DataFrame({
        "row_id": [f"r{i}" for i in range(8)],
        "extractant": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "condition_id": [f"c{i}" for i in range(8)],
        "metal_symbol": ["Nd"] * 8,
        "ecfp_cluster": ["e"] * 8,
        "tanimoto_cluster": ["t"] * 8,
        "log_D": np.arange(8.0),
    })
    order = ligand_acquisition_order(frame, np.arange(8), policy="maxmin",
                                     rng=np.random.default_rng(0), similarity=similarity,
                                     similarity_names=names)
    assert order[1] == "D" if order[0] in {"A", "B"} else True
    assert order.index("B") > order.index("D") or order[0] == "B"


@pytest.mark.parametrize("budget", [10, 37, 100])
@pytest.mark.parametrize("policy", ACQUISITION_POLICIES)
def test_row_budget_is_met_exactly(budget, policy):
    frame = _cohort()
    names = sorted(frame["extractant"].unique())
    similarity = np.eye(len(names)) + 0.1
    index = np.arange(len(frame))
    taken = row_budget_subsample(frame, index, budget=budget, policy=policy,
                                 rng=np.random.default_rng(1), similarity=similarity,
                                 similarity_names=names)
    assert len(taken) == budget
    assert len(set(taken.tolist())) == budget
    assert set(taken.tolist()) <= set(index.tolist())


def test_row_budget_none_returns_everything():
    frame = _cohort()
    index = np.arange(len(frame))
    taken = row_budget_subsample(frame, index, budget=None, policy="random",
                                 rng=np.random.default_rng(1))
    np.testing.assert_array_equal(taken, index)
    over = row_budget_subsample(frame, index, budget=10 ** 6, policy="random",
                                rng=np.random.default_rng(1))
    np.testing.assert_array_equal(over, index)


def test_diversity_covers_more_chemistry_than_depth_at_equal_budget():
    frame = _cohort()
    index = np.arange(len(frame))
    budget = 60
    depth = row_budget_subsample(frame, index, budget=budget, policy="depth",
                                 rng=np.random.default_rng(2))
    diverse = row_budget_subsample(frame, index, budget=budget, policy="diversity",
                                   rng=np.random.default_rng(2))
    assert acquisition_audit(frame, diverse)["n_superclusters"] >= \
        acquisition_audit(frame, depth)["n_superclusters"]
    assert acquisition_audit(frame, diverse)["n_extractants"] >= \
        acquisition_audit(frame, depth)["n_extractants"]
    assert acquisition_audit(frame, depth)["n_rows"] == budget


# --------------------------------------------------------------------------- #
# Cohort comparison
# --------------------------------------------------------------------------- #

def test_cohort_comparison_counts_what_expansion_adds():
    frame = _cohort()
    comparison = cohort_comparison(frame)
    assert comparison["base"]["n_extractants"] == 8
    assert comparison["added_by_expansion"]["n_extractants"] == 6
    assert comparison["expanded"]["n_extractants"] == 14
    assert comparison["new_ecfp_clusters"] == 6      # every sparse ligand has its own cluster
    assert 0.0 < comparison["row_cost_fraction"] < 1.0
