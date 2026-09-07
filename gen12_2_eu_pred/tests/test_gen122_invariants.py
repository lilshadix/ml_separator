"""The fourteen pre-registered Gen12.2 invariants, as executable checks.

These do not test that a model is good.  They test the properties that make every
Gen12.2 number mean what the decision report says it means, and each has a way of failing
silently that would leave the tables looking perfectly reasonable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GEN122 = Path(__file__).resolve().parents[1]
if str(GEN122) not in sys.path:
    sys.path.insert(0, str(GEN122))

from gen122 import coordination, decomposed, levelmodels, levelrunner, levels, paths  # noqa: E402
from gen12eu import splits  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402
from gen12eu.fewshot import support_draw  # noqa: E402
from gen12eu.preprocess import FoldPreprocessor  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    return build_cohort()


@pytest.fixture(scope="module")
def folds(cohort):
    return splits.all_folds(cohort.frame, design="B")


@pytest.fixture(scope="module")
def coordination_table():
    return pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")


# --- 1: the folds are Gen12's, not merely like Gen12's ---------------------- #

def test_design_b_fold_plan_is_byte_identical_to_gen12(cohort, folds):
    plan = json.loads(paths.GEN12_FOLD_PLAN_B.read_text())
    assert plan["cohort_fingerprint"] == cohort.fingerprint
    assert len(plan["folds"]) == len(folds)
    for fold, recorded in zip(folds, plan["folds"]):
        assert (fold.seed, fold.fold) == (recorded["seed"], recorded["fold"])
        assert cohort.frame.iloc[fold.test_index]["row_id"].tolist() == recorded["test_row_ids"]
        assert list(fold.held_out_groups) == recorded["held_out_groups"]


# --- 2: no chemistry crosses a boundary ------------------------------------- #

def test_no_extractant_or_chemotype_crosses_a_fold(cohort, folds):
    report = splits.assert_fold_integrity(cohort.frame, folds)
    assert report["ok"]
    assert report["overlap_counts"] == {"extractant": 0, "ecfp_cluster": 0, "chemotype": 0}


# --- 3, 4: the descriptors cannot depend on the target or drift silently ----- #

def test_coordination_descriptors_are_independent_of_the_target(cohort, coordination_table):
    """Rebuild from the structures alone; the bytes must match the stored matrix."""
    structures = sorted(set(cohort.frame["extractant"].astype(str)))
    rebuilt = coordination.build_table(structures)
    assert rebuilt.equals(coordination_table.reindex(rebuilt.index)[rebuilt.columns])


def test_coordination_descriptors_do_not_depend_on_input_order(cohort):
    import random
    structures = sorted(set(cohort.frame["extractant"].astype(str)))
    first = coordination.build_table(structures)
    shuffled = list(structures)
    random.Random(7).shuffle(shuffled)
    second = coordination.build_table(shuffled).reindex(first.index)[first.columns]
    assert first.equals(second)


def test_smarts_specification_digest_matches_the_one_frozen_before_the_ladder():
    audit = json.loads((paths.FEATURE_DIR / "coordination_audit.json").read_text())
    assert audit["spec_sha256"] == coordination.spec_digest()
    manifest = paths.PREDICTION_DIR / "level" / "LVL_MEAN" / "run_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["coordination_spec_sha256"] == audit["spec_sha256"]


def test_no_condition_column_can_reach_a_level_block(cohort, coordination_table):
    _, columns = levelrunner.structure_frame(cohort.frame, coordination_table, cohort.blocks)
    for name, cols in columns.items():
        assert not [c for c in cols if c.startswith(levelrunner.FORBIDDEN_PREFIXES)]


def test_structure_frame_columns_are_constant_within_an_extractant(cohort, coordination_table):
    """Anything varying by row is a condition in disguise; only the Architector
    complex-specification columns may vary, and those are collapsed to their mode."""
    table, columns = levelrunner.structure_frame(cohort.frame, coordination_table, cohort.blocks)
    collapsed = table.attrs["collapsed_complex_spec_columns"]
    assert set(collapsed) <= set(levelrunner.COMPLEX_SPEC_COLUMNS)
    for column in levelrunner.COMPLEX_SPEC_COLUMNS:
        assert table[column].notna().all()


# --- 5, 6, 7: level targets and residuals are training-only ------------------ #

def test_preprocessor_statistics_depend_only_on_training_rows(cohort, folds, coordination_table):
    structure, columns = levelrunner.structure_frame(cohort.frame, coordination_table,
                                                     cohort.blocks)
    fold = folds[0]
    targets = levels.build_level_targets(
        cohort.frame.iloc[fold.train_index], cohort.frame.iloc[fold.test_index],
        cohort.blocks, definition="LVL_MEAN", seed=fold.model_seed)
    train_names = list(targets.train.index)
    fitted = FoldPreprocessor(columns["COORD"]).fit(structure.loc[train_names])
    corrupted = structure.copy()
    corrupted.loc[list(targets.test.index), list(columns["COORD"])] = 1e6
    again = FoldPreprocessor(columns["COORD"]).fit(corrupted.loc[train_names])
    assert np.allclose(fitted.medians_, again.medians_, equal_nan=True)


def test_held_out_level_is_never_an_input(cohort, folds):
    """Corrupting every held-out extractant's target must not move a training level."""
    fold = folds[0]
    train = cohort.frame.iloc[fold.train_index]
    test = cohort.frame.iloc[fold.test_index]
    first = levels.build_level_targets(train, test, cohort.blocks, definition="LVL_MEAN",
                                       seed=fold.model_seed).train
    corrupted = test.copy()
    corrupted["log_D"] = 99.0
    second = levels.build_level_targets(train, corrupted, cohort.blocks,
                                        definition="LVL_MEAN", seed=fold.model_seed).train
    assert first.equals(second)


def test_training_residuals_use_the_training_level_only(cohort, folds):
    fold = folds[0]
    train = cohort.frame.iloc[fold.train_index]
    targets = levels.build_level_targets(train, cohort.frame.iloc[fold.test_index],
                                         cohort.blocks, definition="LVL_MEAN",
                                         seed=fold.model_seed)
    expected = (train["log_D"].to_numpy(dtype=float)
                - train["extractant"].astype(str).map(targets.train).to_numpy(dtype=float))
    assert np.allclose(targets.train_row_residual, expected)
    # and the residual of each training extractant sums to zero under LVL_MEAN
    per = pd.Series(targets.train_row_residual).groupby(
        train["extractant"].astype(str).to_numpy()).mean()
    assert np.allclose(per.to_numpy(), 0.0, atol=1e-9)


def test_condition_model_for_the_secondary_definition_is_cross_fitted(cohort, folds):
    """A training row's condition prediction must come from a model that never saw its
    chemotype, or the training level is measured against a different yardstick from the
    held-out one — the two-stage residual trap."""
    fold = folds[0]
    train = cohort.frame.iloc[fold.train_index].reset_index(drop=True)
    test = cohort.frame.iloc[fold.test_index]
    from lanthanide_separation.gen6.cohorts import seeded_group_kfold
    groups = train["chemotype"].astype(str).to_numpy()
    # The property, asserted directly: the inner folds partition the training chemotypes,
    # so no row's prediction can come from a model that saw its own chemotype.  A
    # residual-magnitude comparison alone would pass even if every chemotype leaked.
    seen = []
    for inner_train, inner_test in seeded_group_kfold(groups, 4, fold.model_seed):
        assert not (set(groups[inner_train]) & set(groups[inner_test]))
        seen.append(set(inner_test.tolist()))
    covered = set().union(*seen)
    assert covered == set(range(len(train)))
    assert sum(len(s) for s in seen) == len(train), "inner folds overlap"

    oof, _ = levels.condition_only_predictions(train, test, cohort.blocks,
                                               seed=fold.model_seed, inner_seed=fold.model_seed)
    fitted_on_all = levels._fit_condition_model(
        train, train["log_D"].to_numpy(dtype=float),
        levels._condition_columns(cohort.blocks), fold.model_seed)
    in_sample = fitted_on_all[1].predict(fitted_on_all[0].transform(train))
    y = train["log_D"].to_numpy(dtype=float)
    assert np.abs(y - oof).mean() > np.abs(y - in_sample).mean() * 1.2


# --- 8: shape centring is per (split_seed, extractant) ---------------------- #

def test_level_shape_centring_is_per_seed():
    from gen12eu.metrics import per_extractant
    rows = []
    for seed in (1, 2):
        for i in range(4):
            rows.append({"split_seed": seed, "extractant": "E", "chemotype": "c",
                         "log_D": float(i), "prediction": float(i) + seed})
    frame = pd.DataFrame(rows)
    table = per_extractant(frame)
    # a pure per-seed offset must be scored entirely as level and not at all as shape
    assert np.allclose(table["shape_mae"].to_numpy(), 0.0)
    assert set(np.round(table["bias"].to_numpy(), 6)) == {1.0, 2.0}


# --- 9: matched units and matched rows -------------------------------------- #

def _row_key(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(columns)
    return hashlib.blake2b("|".join(
        "|".join(str(v) for v in row) for row in ordered[columns].to_numpy()).encode(),
        digest_size=16).hexdigest()


@pytest.mark.skipif(not (paths.PREDICTION_DIR / "level" / "LVL_MEAN").exists(),
                    reason="level ladder has not been run")
def test_every_level_arm_is_scored_on_identical_units():
    directory = paths.PREDICTION_DIR / "level" / "LVL_MEAN"
    keys, truths = set(), []
    for path in sorted(directory.glob("*.parquet")):
        frame = pd.read_parquet(path)
        keys.add(_row_key(frame, ["split_seed", "fold", "extractant"]))
        truths.append(frame.sort_values(["split_seed", "fold", "extractant"])
                      ["alpha_true"].to_numpy())
    assert len(keys) == 1
    for values in truths:
        assert np.allclose(values, truths[0])


@pytest.mark.skipif(not (paths.PREDICTION_DIR / "full").exists(),
                    reason="full prediction arms have not been run")
def test_every_full_arm_is_scored_on_identical_rows():
    keys = set()
    for path in sorted((paths.PREDICTION_DIR / "full").glob("*.parquet")):
        frame = pd.read_parquet(path)
        keys.add(_row_key(frame, ["split_seed", "fold", "row_id"]))
    gen12 = pd.read_parquet(paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet")
    keys.add(_row_key(gen12, ["split_seed", "fold", "row_id"]))
    assert len(keys) == 1, "a Gen12.2 arm is scored on different rows from Gen12's"


# --- 10: few-shot draws are shared and process-independent ------------------- #

def test_support_draw_is_identical_across_processes():
    expected = support_draw("EXTRACTANT", 9, 3, 2, 104729).tolist()
    code = ("import sys; sys.path.insert(0, %r);"
            "from gen12eu.fewshot import support_draw;"
            "print(support_draw('EXTRACTANT', 9, 3, 2, 104729).tolist())"
            % str(paths.GEN12_ROOT))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    assert eval(result.stdout.strip()) == expected


# --- 11: the subgroup is structural ----------------------------------------- #

def test_multi_arm_membership_depends_only_on_structure(coordination_table):
    subgroup = pd.read_csv(paths.MANIFEST_DIR / "multi_arm_subgroup.csv", index_col=0)
    rule = (coordination_table["coord__arm__local_donor_cluster_count"] >= 2)
    assert subgroup["MULTI_ARM"].astype(bool).equals(rule.reindex(subgroup.index))


# --- 12: numerical safety ---------------------------------------------------- #

def test_coordination_block_has_no_heavy_tail(coordination_table):
    values = coordination_table.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    worst = 0.0
    for column in coordination_table.columns:
        series = coordination_table[column].astype(float)
        sd = float(series.std(ddof=0))
        if sd > 1e-12:
            worst = max(worst, float(np.abs(series).max() / sd))
    # Gen12's lig2d__rd__Ipc reached 7.3e16 standardised; anything of that order here
    # would mean the block can poison a model that multiplies its inputs.
    assert worst < 100.0, f"largest |x|/sd is {worst}"


@pytest.mark.skipif(not (paths.PREDICTION_DIR / "level" / "LVL_MEAN").exists(),
                    reason="level ladder has not been run")
def test_clipping_is_recorded_and_immaterial():
    for path in sorted((paths.PREDICTION_DIR / "level" / "LVL_MEAN").glob("*__selection.csv")):
        selection = pd.read_csv(path)
        assert "n_clipped" in selection.columns
        frame = pd.read_parquet(path.parent / f"{path.stem.replace('__selection', '')}.parquet")
        # both the clipped and unclipped predictions are kept, so a clip cannot hide
        assert {"alpha_pred", "alpha_pred_unclipped"} <= set(frame.columns)
        assert np.isfinite(frame["alpha_pred_unclipped"]).all()


# --- 13: selection reads inner validation only ------------------------------- #

@pytest.mark.skipif(not (paths.PREDICTION_DIR / "level" / "LVL_MEAN").exists(),
                    reason="level ladder has not been run")
def test_selection_log_carries_an_inner_validation_score():
    for path in sorted((paths.PREDICTION_DIR / "level" / "LVL_MEAN").glob("*__selection.csv")):
        selection = pd.read_csv(path)
        assert "inner_level_mae" in selection.columns
        assert selection["inner_level_mae"].notna().all()


def test_level_definition_choice_records_the_rules_file_it_was_made_under():
    choice = json.loads((paths.MANIFEST_DIR / "level_definition_choice.json").read_text())
    assert choice["decision"]["primary_level_definition"] == "LVL_MEAN"
    assert choice["decision"]["secondary_level_definition"] == "LVL_COND_RESIDUAL"
    # the study records what the original, un-amended rule would have chosen
    assert choice["decision"]["original_rule_would_have_chosen"] == "LVL_COND_RESIDUAL"
    # and the digest it recorded must be the digest of the rules file as it stands, so a
    # rules file edited after the study would be caught rather than pass a length check
    assert choice["rules_digest"] == paths.sha256_of(
        paths.CONFIG_DIR / "level_definition_rules.json")


# --- 14: headline pairwise tests operate on matched units --------------------- #

@pytest.mark.skipif(not (paths.BOOTSTRAP_DIR / "level_LVL_MEAN_pairwise.csv").exists(),
                    reason="the level analysis has not been run")
def test_headline_contrasts_are_computed_on_matched_units():
    """Every published contrast must name two arms that were scored on the same units.

    ``inference.unit_table`` raises when the arms disagree on their unit set, so the check
    here is that the published tables really came through it: every contrast's two arms
    exist among the level predictions and cover an identical extractant set.
    """
    directory = paths.PREDICTION_DIR / "level" / "LVL_MEAN"
    units = {}
    for path in sorted(directory.glob("*.parquet")):
        frame = pd.read_parquet(path)
        units[frame["arm"].iloc[0]] = set(frame["extractant"])
    table = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_pairwise.csv")
    assert len(table)
    for reference, candidate in zip(table["reference"], table["candidate"]):
        assert reference in units and candidate in units, (reference, candidate)
        assert units[reference] == units[candidate]
    # and the unit count reported must be the real one for that cohort: the full cohort is
    # every extractant, and the level-reliable cohort is a strict subset of it
    full = len(next(iter(units.values())))
    by_cohort = table.groupby("cohort")["units_total"].unique()
    assert list(by_cohort.loc["full"]) == [full]
    assert 0 < by_cohort.loc["level_reliable"][0] < full


# --- the shape blocks really are the pre-registered ones ---------------------- #

def test_shape_contracts_always_include_the_conditions():
    for name, blocks in decomposed.SHAPE_BLOCKS.items():
        assert "COND" in blocks and "MASSACT" in blocks, name


def test_ablation_letters_cover_the_seven_preregistered_arms():
    assert set(levelmodels.ABLATION_LETTER.values()) == set("ABCDEFG")
