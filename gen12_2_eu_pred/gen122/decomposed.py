"""The decomposed predictor: ``y_hat_ij = alpha_hat_i + delta_hat_ij``.

The level comes from **molecular structure only**; the shape is a model of the deviation
from that level under experimental conditions.  Every residualisation is fold-local, and
the order of operations is the pre-registered one:

1. level targets from the fold's training rows only;
2. level model fitted on the training extractants;
3. training residuals against the **observed training level** — never against a
   cross-fitted prediction.  The repository measured that centring a second stage on a
   cross-fitted residual makes it *worse* under a chemotype hold-out (1.25 against 1.08)
   and that centring on the true cell mean is what works (1.04);
4. shape model fitted on those residuals;
5. the held-out extractant's level predicted from structure alone;
6. its query residual predicted from conditions and structure;
7. the two summed and clipped to the training target range.

**A held-out extractant's mean is never computed before its prediction.**  It appears only
afterwards, as evaluation truth.

Hyperparameters for the shape model are chosen on the fold's inner validation block
against the *deployed* quantity — the level model fitted on inner-training extractants
plus the shape model, scored on inner-validation rows — so the selection criterion is the
thing the arm is finally judged on, and no outer test row is involved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from gen12eu import splits
from gen12eu.chemistry import band_of
from gen12eu.preprocess import (
    FoldPreprocessor, clip_to_training_range, extractant_balanced_weights,
)

from . import levels
from .levelmodels import TreeLevel, _clip

TARGET = "log_D"

#: The pre-registered shape contracts.  ``COND`` and ``MASSACT`` are always present: a
#: shape model that cannot see the conditions is not a shape model.
SHAPE_BLOCKS: dict[str, tuple[str, ...]] = {
    "S0": ("COND", "MASSACT"),
    "S1": ("COND", "MASSACT", "ECFP"),
    "S2": ("COND", "MASSACT", "COORD"),
    "S3": ("COND", "MASSACT", "ECFP", "COORD"),
}


def _grid() -> list[dict]:
    return [{"max_features": f, "min_samples_leaf": leaf}
            for f in (0.30, "sqrt") for leaf in (1, 2)]


def _fit_trees(x, y, weights, params, seed, n_estimators: int = 500):
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(n_estimators=n_estimators, max_features=params["max_features"],
                                min_samples_leaf=int(params["min_samples_leaf"]),
                                random_state=seed, n_jobs=-1)
    model.fit(x, y, sample_weight=weights)
    return model


def _macro_mae(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    table = pd.DataFrame({"e": frame["extractant"].astype(str).to_numpy(),
                          "ae": np.abs(np.asarray(prediction, float) - np.asarray(truth, float))})
    return float(table.groupby("e")["ae"].mean().mean())


@dataclass
class DecomposedContender:
    """One level representation plus one shape representation."""

    name: str
    level_blocks: tuple[str, ...]
    shape: str = "S1"
    selected_: dict = field(default_factory=dict)

    def _level_predictions(self, structure, columns, train_names, alpha_train, test_names,
                           seed) -> np.ndarray:
        arm = TreeLevel(name=f"{self.name}__level", blocks=self.level_blocks)
        level_columns = []
        for block in self.level_blocks:
            level_columns.extend(columns[block])
        level_columns = list(dict.fromkeys(level_columns))
        best, best_score = None, np.inf
        # The level model's hyperparameter is chosen on a grouped split of *this fold's
        # training extractants* by chemotype.  That is leakage-safe and it is NOT the same
        # block the frozen ladder used, which splits the fold's training *rows*: the level
        # ladder's inner block is carved from rows and this one from extractants, so the two
        # models agree in accuracy without being identical.  Measured on the shipped arms:
        # per-extractant mean absolute difference 0.167 with correlation 0.967, and level
        # MAE 1.0268 here against 1.0281 in the ladder for L7, 1.0620 against 1.0624 for L3.
        groups = structure.loc[train_names, "chemotype"].astype(str).to_numpy()
        from lanthanide_separation.gen6.cohorts import seeded_group_kfold
        inner_train, inner_validation = next(iter(seeded_group_kfold(groups, 4, seed)))
        pre = FoldPreprocessor(tuple(level_columns)).fit(
            structure.loc[[train_names[i] for i in inner_train]])
        x_inner = pre.transform(structure.loc[[train_names[i] for i in inner_train]])
        x_valid = pre.transform(structure.loc[[train_names[i] for i in inner_validation]])
        for params in _grid():
            model = _fit_trees(x_inner, alpha_train[inner_train], None, params, seed)
            score = float(np.abs(model.predict(x_valid) - alpha_train[inner_validation]).mean())
            if score < best_score:
                best, best_score = params, score
        outer = FoldPreprocessor(tuple(level_columns)).fit(structure.loc[train_names])
        model = _fit_trees(outer.transform(structure.loc[train_names]), alpha_train, None,
                           best, seed)
        self.selected_["level_params"] = str(best)
        self.selected_["level_inner_mae"] = float(best_score)
        return model.predict(outer.transform(structure.loc[test_names]))

    def fit_predict(self, context) -> np.ndarray:
        train, test = context["train"], context["test"]
        structure, columns = context["structure"], context["columns"]
        seed = context["model_seed"]
        alpha = context["alpha_train"]                       # training extractant -> level
        train_names = list(alpha.index)
        test_names = sorted(set(test["extractant"].astype(str)))

        # --- level, from structure only ------------------------------------- #
        raw_level = self._level_predictions(structure, columns, train_names,
                                            alpha.to_numpy(dtype=float), test_names, seed)
        level, n_clipped = _clip(raw_level, alpha.to_numpy(dtype=float))
        level_of = dict(zip(test_names, level))

        # --- shape, on residuals against the OBSERVED training level ---------- #
        shape_columns: list[str] = []
        for block in SHAPE_BLOCKS[self.shape]:
            shape_columns.extend(columns[block] if block in columns else context["blocks"][block])
        shape_columns = list(dict.fromkeys(shape_columns))
        residual = (train[TARGET].to_numpy(dtype=float)
                    - train["extractant"].astype(str).map(alpha).to_numpy(dtype=float))

        inner_train_rows = context["inner_train"]
        inner_validation_rows = context["inner_validation"]
        inner_alpha = context["alpha_inner_train"]
        inner_residual = (inner_train_rows[TARGET].to_numpy(dtype=float)
                          - inner_train_rows["extractant"].astype(str)
                          .map(inner_alpha).to_numpy(dtype=float))
        inner_level = self._level_predictions(
            structure, columns, list(inner_alpha.index), inner_alpha.to_numpy(dtype=float),
            sorted(set(inner_validation_rows["extractant"].astype(str))), seed)
        inner_level_of = dict(zip(sorted(set(inner_validation_rows["extractant"].astype(str))),
                                  inner_level))
        inner_base = inner_validation_rows["extractant"].astype(str).map(
            inner_level_of).to_numpy(dtype=float)

        pre_inner = FoldPreprocessor(tuple(shape_columns)).fit(inner_train_rows)
        x_inner = pre_inner.transform(inner_train_rows)
        x_valid = pre_inner.transform(inner_validation_rows)
        w_inner = extractant_balanced_weights(inner_train_rows["extractant"])
        best, best_score = None, np.inf
        for params in _grid():
            model = _fit_trees(x_inner, inner_residual, w_inner, params, seed)
            combined = inner_base + model.predict(x_valid)
            score = _macro_mae(inner_validation_rows,
                               inner_validation_rows[TARGET].to_numpy(dtype=float), combined)
            if score < best_score:
                best, best_score = params, score

        pre = FoldPreprocessor(tuple(shape_columns)).fit(train)
        model = _fit_trees(pre.transform(train), residual,
                           extractant_balanced_weights(train["extractant"]), best, seed)
        delta = model.predict(pre.transform(test))
        base = test["extractant"].astype(str).map(level_of).to_numpy(dtype=float)
        self.selected_.update({"shape_params": str(best), "inner_macro_mae": float(best_score),
                               "n_level_clipped": int(n_clipped),
                               "shape_n_features": int(pre.transform(train).shape[1])})
        prediction = clip_to_training_range(base + delta, train[TARGET].to_numpy(dtype=float))
        return prediction, base, delta


def run_decomposed(contender: DecomposedContender, frame: pd.DataFrame,
                   structure: pd.DataFrame, columns: dict, blocks: dict,
                   folds: Sequence[splits.Fold], similarity: pd.DataFrame,
                   targets_by_fold: dict, *, verbose: bool = False
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, selections = [], []
    for fold in folds:
        started = time.time()
        outer, inner = targets_by_fold[(fold.seed, fold.fold)]
        train = frame.iloc[fold.train_index]
        test = frame.iloc[fold.test_index]
        context = {
            "train": train, "test": test, "structure": structure, "columns": columns,
            "blocks": blocks, "model_seed": fold.model_seed, "alpha_train": outer.train,
            "inner_train": frame.iloc[fold.inner_train_index],
            "inner_validation": frame.iloc[fold.inner_validation_index],
            "alpha_inner_train": inner.train,
        }
        prediction, level, delta = contender.fit_predict(context)
        if not np.isfinite(prediction).all():
            raise ValueError(f"{contender.name} produced non-finite predictions")
        records.append(pd.DataFrame({
            "arm": contender.name, "design": "B", "split_seed": fold.seed, "fold": fold.fold,
            "row_id": test["row_id"].to_numpy(), "extractant": test["extractant"].to_numpy(),
            "chemotype": test["chemotype"].to_numpy(),
            "ecfp_cluster": test["ecfp_cluster"].to_numpy(),
            "series_id": test["series_id"].to_numpy(), "doi": test["doi"].to_numpy(),
            "review_flags": test["review_flags"].to_numpy(),
            TARGET: test[TARGET].to_numpy(dtype=float), "prediction": prediction,
            "level_component": level, "shape_component": delta}))
        selections.append({"arm": contender.name, "split_seed": fold.seed, "fold": fold.fold,
                           "seconds": round(time.time() - started, 2),
                           **getattr(contender, "selected_", {})})
        if verbose:
            print(f"  {contender.name} seed={fold.seed} fold={fold.fold} "
                  f"{time.time() - started:.1f}s", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    key = ["design", "split_seed", "fold", "extractant"]
    predictions = predictions.merge(
        similarity[key + ["max_train_tanimoto", "nearest_train_extractant"]],
        on=key, how="left", validate="many_to_one")
    predictions["band"] = band_of(predictions["max_train_tanimoto"]).to_numpy()
    return predictions, pd.DataFrame(selections)
