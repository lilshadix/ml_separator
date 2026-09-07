"""The level ladder: molecular structure -> the extractant's intrinsic Eu level.

One learner and one contract, so the leaderboard compares **representations** and not
algorithms.  Experimental conditions cannot enter: a level contender is handed a frame
with one row per extractant and only structural columns, and the runner asserts that no
condition column is among them.

Three properties the runner enforces rather than documents:

* the training level target is computed from that fold's **training rows only**;
* the held-out extractant contributes structural features and nothing else — its own
  targets are the evaluation truth and are never an input;
* hyperparameters are chosen on the fold's **inner validation block**, whose level
  targets are themselves computed from inner-training and inner-validation rows
  separately, so the selection never sees an outer test extractant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from gen12eu.chemistry import ECFP_COLUMNS, tanimoto_matrix
from gen12eu.preprocess import FoldPreprocessor

#: Gen12's generic molecular content — exactly the molecular blocks of its best arm.
GENERIC_BLOCKS: tuple[str, ...] = ("PHYSCHEM", "DONORS", "LIG2D")
COORD_BLOCK = "COORD"
CLIP_MARGIN = 1.0


@dataclass
class LevelContext:
    """Everything a level contender may legitimately use."""

    columns: dict                      # block name -> tuple of columns
    model_seed: int
    inner_train: pd.DataFrame          # one row per inner-training extractant
    inner_validation: pd.DataFrame     # one row per inner-validation extractant
    alpha_inner_train: np.ndarray
    alpha_inner_validation: np.ndarray
    split_seed: int = 0
    fold: int = 0

    def block(self, *names: str) -> tuple[str, ...]:
        out: list[str] = []
        for name in names:
            out.extend(self.columns[name])
        return tuple(dict.fromkeys(out))


def _mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.abs(np.asarray(prediction, float) - np.asarray(truth, float)).mean())


def _clip(prediction: np.ndarray, alpha_train: np.ndarray, margin: float = CLIP_MARGIN):
    low, high = float(np.min(alpha_train)) - margin, float(np.max(alpha_train)) + margin
    clipped = np.clip(np.asarray(prediction, float), low, high)
    return clipped, int((np.asarray(prediction, float) != clipped).sum())


@dataclass
class GlobalMeanLevel:
    name: str = "L0_GLOBAL_MEAN"
    blocks: tuple[str, ...] = ()
    selected_: dict = field(default_factory=dict)

    def fit_predict(self, train, alpha_train, test, context):
        self.selected_ = {"inner_level_mae": _mae(
            context.alpha_inner_validation,
            np.full(len(context.inner_validation), float(np.mean(context.alpha_inner_train))))}
        return np.full(len(test), float(np.mean(alpha_train)))


@dataclass
class NearestTanimotoLevel:
    """The level of the most similar training extractant.

    gen7 measured that on a condition-adjusted level target no model beat a
    1-nearest-neighbour Tanimoto lookup.  A level claim that does not beat this is not a
    claim, so the arm is mandatory rather than decorative.
    """

    name: str = "L0_NN_TANIMOTO"
    blocks: tuple[str, ...] = ()
    selected_: dict = field(default_factory=dict)

    @staticmethod
    def _predict(train, alpha_train, test):
        train_bits = train[list(ECFP_COLUMNS)].to_numpy(dtype=np.uint8)
        test_bits = test[list(ECFP_COLUMNS)].to_numpy(dtype=np.uint8)
        similarity = tanimoto_matrix(test_bits, train_bits)
        return np.asarray(alpha_train, float)[similarity.argmax(axis=1)]

    def fit_predict(self, train, alpha_train, test, context):
        self.selected_ = {"inner_level_mae": _mae(
            context.alpha_inner_validation,
            self._predict(context.inner_train, context.alpha_inner_train,
                          context.inner_validation))}
        return self._predict(train, alpha_train, test)


@dataclass
class TreeLevel:
    """Extremely randomised trees on one structural feature contract.

    The grid is Gen12's, chosen on the inner validation block of the outer fold and
    refitted on the whole outer training set.
    """

    name: str
    blocks: tuple[str, ...]
    family: str = "extratrees"
    n_estimators: int = 500
    selected_: dict = field(default_factory=dict)

    def _grid(self) -> list[dict]:
        if self.family in ("extratrees", "randomforest"):
            return [{"max_features": f, "min_samples_leaf": leaf}
                    for f in (0.30, "sqrt") for leaf in (1, 2)]
        if self.family == "xgboost":
            return [{"max_depth": d, "min_child_weight": w} for d in (3, 6) for w in (1.0, 5.0)]
        if self.family == "catboost":
            return [{"depth": d, "l2_leaf_reg": l} for d in (4, 6) for l in (3.0, 10.0)]
        raise KeyError(self.family)

    def _fit(self, x, y, params, seed):
        if self.family == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=400, learning_rate=0.05,
                                 max_depth=int(params["max_depth"]),
                                 min_child_weight=float(params["min_child_weight"]),
                                 subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0,
                                 objective="reg:absoluteerror", random_state=seed,
                                 n_jobs=-1, tree_method="hist")
            model.fit(x, y)
            return model
        if self.family == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=600, learning_rate=0.05,
                                      depth=int(params["depth"]),
                                      l2_leaf_reg=float(params["l2_leaf_reg"]),
                                      loss_function="MAE", random_seed=seed, verbose=False,
                                      allow_writing_files=False, thread_count=-1)
            model.fit(x, y, verbose=False)
            return model
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        cls = RandomForestRegressor if self.family == "randomforest" else ExtraTreesRegressor
        model = cls(n_estimators=self.n_estimators, max_features=params["max_features"],
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    random_state=seed, n_jobs=-1)
        model.fit(x, y)
        return model

    def fit_predict(self, train, alpha_train, test, context):
        columns = context.block(*self.blocks)
        inner = FoldPreprocessor(columns).fit(context.inner_train)
        x_inner = inner.transform(context.inner_train)
        x_valid = inner.transform(context.inner_validation)
        best, best_score = None, np.inf
        for params in self._grid():
            model = self._fit(x_inner, context.alpha_inner_train, params, context.model_seed)
            score = _mae(context.alpha_inner_validation, model.predict(x_valid))
            if score < best_score:
                best, best_score = params, score
        outer = FoldPreprocessor(columns).fit(train)
        x_train = outer.transform(train)
        model = self._fit(x_train, alpha_train, best, context.model_seed)
        prediction = model.predict(outer.transform(test))
        self.selected_ = {**{k: str(v) for k, v in best.items()},
                          "inner_level_mae": float(best_score),
                          "n_features_after_preprocess": int(x_train.shape[1]),
                          "max_abs_input": float(np.abs(x_train).max())}
        self.model_, self.preprocessor_ = model, outer
        return prediction


#: The pre-registered ladder.  The letter is the feature-family ablation letter.
LADDER_BLOCKS: dict[str, tuple[str, ...]] = {
    "L1_ECFP": ("ECFP",),
    "L2_GENERIC": GENERIC_BLOCKS,
    "L3_ECFP_GENERIC": ("ECFP",) + GENERIC_BLOCKS,
    "L4_COORD": (COORD_BLOCK,),
    "L5_ECFP_COORD": ("ECFP", COORD_BLOCK),
    "L6_GENERIC_COORD": GENERIC_BLOCKS + (COORD_BLOCK,),
    "L7_ALL": ("ECFP",) + GENERIC_BLOCKS + (COORD_BLOCK,),
}
ABLATION_LETTER: dict[str, str] = {
    "L1_ECFP": "A", "L2_GENERIC": "B", "L4_COORD": "C", "L3_ECFP_GENERIC": "D",
    "L5_ECFP_COORD": "E", "L6_GENERIC_COORD": "F", "L7_ALL": "G",
}


def default_ladder(family: str = "extratrees") -> list:
    arms: list = [GlobalMeanLevel(), NearestTanimotoLevel()]
    for name, blocks in LADDER_BLOCKS.items():
        arms.append(TreeLevel(name=name, blocks=blocks, family=family))
    return arms
