"""Run one level contender over Gen12's frozen design-B fold plan.

The runner owns the contract:

* the fold plan is Gen12's, rebuilt by Gen12's own splitter from Gen12's own cohort;
* the structural frame is one row per extractant, and a condition column reaching it is
  a fatal error rather than a silent extra feature;
* the training level target is computed from the fold's training rows only, and the
  inner-validation level target from the inner-validation rows only, so no selection
  step can see an outer test extractant;
* the held-out extractant's level is the evaluation truth and is attached *after* the
  prediction is produced;
* predictions are clipped to the training level range plus one decade, and the number of
  clipped values is recorded per fold, so a clip can never make a divergence invisible.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from gen12eu import splits
from gen12eu.chemistry import band_of

from . import coordination, levels
from .levelmodels import LevelContext, _clip

TARGET = "log_D"
FORBIDDEN_PREFIXES = ("cond__", "massact__")
#: Built per experiment by the Architector complex recipe rather than per ligand, so they
#: can vary between two rows of the same extractant.  See ``structure_frame``.
COMPLEX_SPEC_COLUMNS: tuple[str, ...] = ("DENTATE", "coreCN", "n_ligs", "n_fill")


def structure_frame(frame: pd.DataFrame, coordination_table: pd.DataFrame,
                    blocks: dict) -> tuple[pd.DataFrame, dict]:
    """One row per extractant: every structural column, and no condition column.

    Asserts that each structural column really is constant within an extractant; a
    column that varies by row is a condition in disguise and would smuggle the
    experiment into a structure-only model.
    """
    structural: list[str] = []
    for name in ("ECFP", "PHYSCHEM", "DONORS", "LIG2D"):
        structural.extend(blocks[name])
    structural = list(dict.fromkeys(structural))
    varying = [c for c in structural
               if frame.groupby("extractant")[c].nunique(dropna=False).max() > 1]
    unexpected = [c for c in varying if c not in COMPLEX_SPEC_COLUMNS]
    if unexpected:
        raise AssertionError(
            f"{len(unexpected)} structural columns vary within an extractant and are not "
            f"complex-specification columns: {unexpected[:5]}")
    table = (frame.drop_duplicates("extractant")
             .set_index("extractant")[structural + ["chemotype", "chem_family"]])
    collapsed: dict[str, int] = {}
    for column in varying:
        # The Architector complex-specification columns are built per experiment, not per
        # ligand, and for ten extractants two different complexes were built.  A column
        # that varies by row is not a structural property, and letting it into a
        # structure-only level model would smuggle the experiment in.  The per-extractant
        # modal value is taken instead, with ties broken by the smaller value so the
        # collapse is deterministic, and the count is recorded rather than absorbed.
        mode = (frame.groupby("extractant")[column]
                .agg(lambda s: sorted(s.mode().tolist())[0]))
        collapsed[column] = int((frame.groupby("extractant")[column].nunique() > 1).sum())
        table[column] = mode.reindex(table.index).to_numpy()
    coordination_columns = list(coordination_table.columns)
    table = table.join(coordination_table, how="left")
    if table[coordination_columns].isna().any().any():
        raise AssertionError("a cohort extractant has no coordination descriptor row")
    columns = {"ECFP": tuple(blocks["ECFP"]), "PHYSCHEM": tuple(blocks["PHYSCHEM"]),
               "DONORS": tuple(blocks["DONORS"]), "LIG2D": tuple(blocks["LIG2D"]),
               "COORD": tuple(coordination_columns)}
    for name, cols in columns.items():
        bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise AssertionError(f"condition column {bad[:3]} reached the level block {name!r}")
    for column in structural + list(coordination_columns):
        if table[column].isna().any() and column not in blocks["LIG2D"]:
            raise AssertionError(f"structural column {column!r} is missing for some extractant")
    table.attrs["collapsed_complex_spec_columns"] = collapsed
    return table, columns


def level_targets_by_fold(frame: pd.DataFrame, folds: Sequence[splits.Fold], blocks: dict,
                          *, definition: str = "LVL_MEAN") -> dict:
    """Level targets for every fold, computed once and shared by every arm.

    Sharing them is not only cheaper — it is what makes the arms comparable, because a
    difference between two arms is then the representation and never a difference in the
    target they were fitted against.
    """
    out: dict = {}
    for fold in folds:
        outer = levels.build_level_targets(
            frame.iloc[fold.train_index], frame.iloc[fold.test_index], blocks,
            definition=definition, seed=fold.model_seed, inner_seed=fold.model_seed)
        inner = levels.build_level_targets(
            frame.iloc[fold.inner_train_index], frame.iloc[fold.inner_validation_index], blocks,
            definition=definition, seed=fold.model_seed, inner_seed=fold.model_seed)
        out[(fold.seed, fold.fold)] = (outer, inner)
    return out


def run_level_contender(contender, frame: pd.DataFrame, structure: pd.DataFrame,
                        columns: dict, folds: Sequence[splits.Fold],
                        similarity: pd.DataFrame, *, definition: str = "LVL_MEAN",
                        blocks: dict | None = None, verbose: bool = False,
                        targets_by_fold: dict | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns ``(predictions, per_fold_selection)`` — one prediction per held-out extractant."""
    records: list[pd.DataFrame] = []
    selections: list[dict] = []
    if targets_by_fold is None:
        targets_by_fold = level_targets_by_fold(frame, folds, blocks or {}, definition=definition)
    for fold in folds:
        started = time.time()
        targets, inner = targets_by_fold[(fold.seed, fold.fold)]
        if targets.definition != definition:
            raise AssertionError("target cache holds a different level definition")

        train_names = list(targets.train.index)
        test_names = list(targets.test.index)
        if set(train_names) & set(test_names):
            raise AssertionError("an extractant is both a training and a held-out level unit")
        context = LevelContext(
            columns=columns, model_seed=fold.model_seed,
            inner_train=structure.loc[list(inner.train.index)],
            inner_validation=structure.loc[list(inner.test.index)],
            alpha_inner_train=inner.train.to_numpy(dtype=float),
            alpha_inner_validation=inner.test.to_numpy(dtype=float),
            split_seed=fold.seed, fold=fold.fold)
        alpha_train = targets.train.to_numpy(dtype=float)
        raw = np.asarray(contender.fit_predict(
            structure.loc[train_names], alpha_train, structure.loc[test_names], context), float)
        if raw.shape != (len(test_names),):
            raise ValueError(f"{contender.name} returned {raw.shape}, expected {(len(test_names),)}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{contender.name} returned non-finite level predictions")
        prediction, n_clipped = _clip(raw, alpha_train)

        block = pd.DataFrame({
            "arm": contender.name, "definition": definition, "split_seed": fold.seed,
            "fold": fold.fold, "extractant": test_names,
            "chemotype": structure.loc[test_names, "chemotype"].to_numpy(),
            "alpha_true": targets.test.reindex(test_names).to_numpy(dtype=float),
            "alpha_pred_unclipped": raw, "alpha_pred": prediction,
        })
        records.append(block)
        selections.append({
            "arm": contender.name, "split_seed": fold.seed, "fold": fold.fold,
            "n_train_extractants": len(train_names), "n_test_extractants": len(test_names),
            "n_clipped": n_clipped, "seconds": round(time.time() - started, 2),
            **getattr(contender, "selected_", {})})
        if verbose:
            print(f"  {contender.name} seed={fold.seed} fold={fold.fold} "
                  f"n_test={len(test_names)} {time.time() - started:.1f}s", flush=True)

    predictions = pd.concat(records, ignore_index=True)
    key = ["split_seed", "fold", "extractant"]
    predictions = predictions.merge(
        similarity[key + ["max_train_tanimoto", "nearest_train_extractant"]],
        on=key, how="left", validate="one_to_one")
    if predictions["max_train_tanimoto"].isna().any():
        raise AssertionError("a held-out extractant has no similarity entry")
    predictions["band"] = band_of(predictions["max_train_tanimoto"]).to_numpy()
    predictions["level_abs_error"] = (predictions["alpha_pred"] - predictions["alpha_true"]).abs()
    predictions["level_signed_error"] = predictions["alpha_pred"] - predictions["alpha_true"]
    return predictions, pd.DataFrame(selections)
