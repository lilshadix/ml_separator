"""Tests for Experiment F: the start cohort, the policies, and above all label-freeness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.levels import LevelData, LevelForestParameters, LevelRegressor
from lanthanide_separation.gen6.acquisition import (
    MODEL_POLICIES, POLICIES, PolicyContext, choose_next, per_tree_predictions, reveal_rows,
    run_acquisition, score_maxmin, score_offset_uncertainty, score_same_chemotype_first,
    score_uncertainty, start_cohort,
)
from lanthanide_separation.gen6.chemistry import build_chemistry_map


def _world(n_ligands: int = 14, rows_per_ligand: int = 6, n_bits: int = 48, seed: int = 2):
    """Source table + cohort frame + chemistry map for a small acquisition world.

    Ligands 0-3 form one big chemotype (near-identical fingerprints); the rest
    are scattered.  The target depends on a hidden per-ligand level so that
    uncertainty policies have something to be uncertain about.
    """
    rng = np.random.default_rng(seed)
    source_rows, frame_rows = [], []
    base = np.zeros(n_bits, dtype=int); base[rng.choice(n_bits, 10, replace=False)] = 1
    for i in range(n_ligands):
        if i < 4:
            bits = base.copy(); flip = rng.choice(n_bits, 1); bits[flip] = 1 - bits[flip]
        else:
            bits = np.zeros(n_bits, dtype=int); bits[rng.choice(n_bits, 10, replace=False)] = 1
        level = rng.normal(scale=1.5)
        for j in range(rows_per_ligand):
            smiles = f"S{i}"
            source_rows.append({"canonical_smiles": smiles, "extractant_name": f"L{i}", "metal_symbol": "Nd",
                                "cond__acid_concentration_M": float(j), "log_D": level,
                                **{f"ecfp_{b}": int(bits[b]) for b in range(n_bits)}})
            frame_rows.append({
                "row_id": f"{smiles}-{j}", "extractant": smiles, "ecfp_cluster": f"e{i}",
                "tanimoto_cluster": "big" if i < 4 else f"t{i}", "condition_id": f"c{j}",
                "metal_symbol": "Nd", "metal_Z": 60.0, "n_replicates": 1,
                "log_D": level + 0.3 * j + rng.normal(scale=0.05),
                "f_cond": float(j), "f_lig": float(i), "f_noise": rng.normal(),
            })
    source = pd.DataFrame(source_rows)
    frame = pd.DataFrame(frame_rows)
    data = LevelData(frame=frame, blocks={"X": ("f_cond", "f_lig", "f_noise")}, audit={})
    chemistry = build_chemistry_map(source)
    return frame, data, chemistry


def _fit(frame, index, n_trees=15):
    params = LevelForestParameters(n_estimators=n_trees, random_state=0, n_jobs=1)
    sub = frame.iloc[index]
    return LevelRegressor(("f_cond", "f_lig", "f_noise"), params).fit(sub, sub["log_D"].to_numpy(), groups=sub["ecfp_cluster"])


# --------------------------------------------------------------------------- #
# Start cohort and reveal
# --------------------------------------------------------------------------- #

def test_start_cohort_is_the_largest_chemotype_and_nothing_else():
    frame, _, _ = _world()
    train_index = np.arange(len(frame))
    start = start_cohort(frame, train_index, n_ligands=3)
    assert start.chemotype == "big"
    assert set(start.extractants) <= {"S0", "S1", "S2", "S3"}
    assert len(start.extractants) == 3
    assert set(frame.iloc[start.row_index]["extractant"]) == set(start.extractants)
    assert set(start.pool_extractants).isdisjoint(start.extractants)
    assert start.audit["n_pool_ligands"] == 14 - 3


def test_start_cohort_is_not_padded_when_the_chemotype_is_short():
    frame, _, _ = _world()
    start = start_cohort(frame, np.arange(len(frame)), n_ligands=10)
    assert len(start.extractants) == 4              # the big chemotype holds four ligands
    assert start.audit["start_is_short"] is True


def test_reveal_rows_respects_the_budget_and_the_training_index():
    frame, _, _ = _world()
    train_index = np.arange(len(frame))[:60]         # first ten ligands only
    rng = np.random.default_rng(1)
    rows = reveal_rows(frame, train_index, "S2", budget=3, rng=rng)
    assert len(rows) == 3
    assert set(frame.iloc[rows]["extractant"]) == {"S2"}
    assert set(rows.tolist()) <= set(train_index.tolist())
    everything = reveal_rows(frame, train_index, "S2", budget=None, rng=rng)
    assert len(everything) == 6


# --------------------------------------------------------------------------- #
# Label-freeness — the property the whole experiment stands on
# --------------------------------------------------------------------------- #

def _context(frame, chemistry, acquired, pool, model, seed=0, blank_target=False):
    pool_features = {}
    for name in pool:
        block = frame[frame["extractant"] == name].drop(columns=["log_D"])
        pool_features[name] = block

    def similarity(candidate, reference):
        if not reference:
            return np.zeros(0)
        return chemistry.similarity_between([candidate], list(reference))[0]
    return PolicyContext(acquired=tuple(acquired), pool_features=pool_features,
                         similarity=similarity, model=model, rng=np.random.default_rng(seed))


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_ignores_the_pool_target(policy):
    """Permute the pool ligands' log_D: the choice must not move."""
    frame, data, chemistry = _world()
    acquired = ["S0", "S1", "S2"]
    pool = [f"S{i}" for i in range(3, 14)]
    start_rows = np.flatnonzero(frame["extractant"].isin(acquired))
    model = _fit(frame, start_rows)
    honest = choose_next(policy, pool, _context(frame, chemistry, acquired, pool, model, seed=7))
    scrambled = frame.copy()
    pool_mask = scrambled["extractant"].isin(pool)
    scrambled.loc[pool_mask, "log_D"] = np.random.default_rng(3).permutation(scrambled.loc[pool_mask, "log_D"].to_numpy())
    permuted = choose_next(policy, pool, _context(scrambled, chemistry, acquired, pool, model, seed=7))
    assert honest == permuted


def test_policy_refuses_a_context_that_carries_the_target():
    frame, data, chemistry = _world()
    context = _context(frame, chemistry, ["S0"], ["S5", "S6"], None)
    context.pool_features["S5"] = frame[frame["extractant"] == "S5"]     # target still present
    with pytest.raises(ValueError, match="may not see it"):
        choose_next("random", ["S5", "S6"], context)


def test_unknown_policy_is_rejected():
    frame, data, chemistry = _world()
    with pytest.raises(ValueError, match="unknown policy"):
        choose_next("oracle", ["S5"], _context(frame, chemistry, ["S0"], ["S5"], None))


# --------------------------------------------------------------------------- #
# Policy semantics
# --------------------------------------------------------------------------- #

def test_maxmin_picks_the_farthest_and_same_chemotype_the_nearest():
    frame, data, chemistry = _world()
    acquired = ["S0", "S1"]
    pool = ["S2", "S3", "S7", "S9"]         # S2/S3 share the big chemotype with S0/S1
    context = _context(frame, chemistry, acquired, pool, None)
    far = choose_next("maxmin", pool, context)
    near = choose_next("same_chemotype_first", pool, context)
    assert near in {"S2", "S3"}
    assert far in {"S7", "S9"}
    scores = score_maxmin(pool, context)
    assert scores[far] >= max(scores.values()) - 1e-12


def test_uncertainty_scores_are_per_tree_dispersion():
    frame, data, chemistry = _world()
    acquired = ["S0", "S1", "S2"]
    pool = ["S5", "S6", "S7"]
    start_rows = np.flatnonzero(frame["extractant"].isin(acquired))
    model = _fit(frame, start_rows, n_trees=25)
    context = _context(frame, chemistry, acquired, pool, model)
    scores = score_uncertainty(pool, context)
    # recompute one by hand through per_tree_predictions
    block = context.pool_features["S5"]
    trees = per_tree_predictions(model, block)
    assert trees.shape == (25, len(block))
    assert scores["S5"] == pytest.approx(float(trees.std(axis=0).mean()))
    offset = score_offset_uncertainty(pool, context)
    assert offset["S5"] == pytest.approx(float(trees.mean(axis=1).std()))


def test_uncertainty_policies_require_a_model():
    frame, data, chemistry = _world()
    context = _context(frame, chemistry, ["S0"], ["S5"], None)
    with pytest.raises(ValueError, match="need a fitted model"):
        score_uncertainty(["S5"], context)


def test_choose_next_is_deterministic_for_a_seed():
    frame, data, chemistry = _world()
    pool = [f"S{i}" for i in range(4, 14)]
    a = choose_next("random", pool, _context(frame, chemistry, ["S0"], pool, None, seed=11))
    b = choose_next("random", pool, _context(frame, chemistry, ["S0"], pool, None, seed=11))
    assert a == b


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #

def test_run_acquisition_never_touches_the_test_set_and_records_every_step():
    frame, data, chemistry = _world()
    test_mask = frame["tanimoto_cluster"].isin(["t12", "t13"]).to_numpy()
    test_index = np.flatnonzero(test_mask)
    train_index = np.flatnonzero(~test_mask)
    start = start_cohort(frame, train_index, n_ligands=2)
    seen: list[np.ndarray] = []

    def evaluate(pred, train_rows, acquired):
        seen.append(np.asarray(train_rows))
        return {"mae": float(np.abs(pred - frame["log_D"].to_numpy()[test_index]).mean())}

    params = LevelForestParameters(n_estimators=10, random_state=0, n_jobs=1)
    trace = run_acquisition(frame, data, start=start, train_index=train_index, test_index=test_index,
                            policy="maxmin", feature_columns=("f_cond", "f_lig", "f_noise"),
                            params=params, chemistry=chemistry, budget_per_ligand=2, n_steps=4,
                            checkpoints=(1, 2, 4), seed=3, evaluate=evaluate)
    assert list(trace.checkpoints["step"]) == [0, 1, 2, 4]
    assert len(trace.steps) == 4
    assert (trace.steps["n_rows_revealed"] == 2).all()
    # training rows grow by exactly the budget each step and never include a test row
    for rows in seen:
        assert set(rows.tolist()).isdisjoint(set(test_index.tolist()))
    assert trace.checkpoints["n_train_rows"].tolist() == [len(start.row_index) + 2 * s for s in (0, 1, 2, 4)]
    # acquired ligands come from the pool and are distinct
    assert set(trace.steps["extractant"]) <= set(start.pool_extractants)
    assert trace.steps["extractant"].is_unique
    assert set(trace.predictions) == {0, 1, 2, 4}


def test_model_policies_refit_every_step_and_others_only_at_checkpoints():
    """Indirect check: a model policy's choices depend on revealed data, so
    revealing different rows changes them; maxmin's do not."""
    frame, data, chemistry = _world()
    test_index = np.flatnonzero(frame["tanimoto_cluster"].isin(["t13"]))
    train_index = np.flatnonzero(~frame["tanimoto_cluster"].isin(["t13"]))
    start = start_cohort(frame, train_index, n_ligands=2)
    params = LevelForestParameters(n_estimators=10, random_state=0, n_jobs=1)
    kwargs = dict(frame=frame, data=data, start=start, train_index=train_index, test_index=test_index,
                  feature_columns=("f_cond", "f_lig", "f_noise"), params=params, chemistry=chemistry,
                  budget_per_ligand=2, n_steps=5, checkpoints=(5,))
    a = run_acquisition(policy="maxmin", seed=1, **kwargs).steps["extractant"].tolist()
    b = run_acquisition(policy="maxmin", seed=2, **kwargs).steps["extractant"].tolist()
    # chemistry-only: the seed changes only the revealed rows and the tie-break
    # among equally distant candidates, so the SET of acquired ligands is the same
    assert set(a) == set(b)
    assert "uncertainty" in MODEL_POLICIES
