"""The Gen12 model ladder.  One interface, so the leaderboard compares science.

Every contender implements ``fit_predict(train, y_train, test, context)`` and
returns one prediction per test row in test-row order.  A contender that needs a
hyperparameter chooses it inside ``fit_predict`` on the *inner validation block
of that outer fold*, which the context carries.  No contender can reach an outer
test row: it is handed a feature frame and never the fold indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .chemistry import ECFP_COLUMNS, tanimoto_matrix
from .preprocess import (
    FoldPreprocessor, as_float, clip_to_training_range, extractant_balanced_weights,
)

TARGET = "log_D"
#: The default molecular + condition contract.  LIG2D is deliberately absent: gen7
#: measured that deleting exactly that block was a significant improvement.
DEFAULT_BLOCKS: tuple[str, ...] = ("COND", "MASSACT", "ECFP", "PHYSCHEM", "DONORS")


@dataclass
class FoldContext:
    """Everything a contender may legitimately use."""

    blocks: dict
    model_seed: int
    inner_train: pd.DataFrame
    inner_validation: pd.DataFrame
    y_inner_train: np.ndarray
    y_inner_validation: np.ndarray
    split_seed: int = 0
    fold: int = 0

    def columns(self, *names: str) -> tuple[str, ...]:
        out: list[str] = []
        for name in names:
            out.extend(self.blocks[name])
        return tuple(dict.fromkeys(out))


def _macro_mae(frame: pd.DataFrame, y: np.ndarray, prediction: np.ndarray) -> float:
    """Extractant-macro MAE — the selection criterion, matching the headline metric."""
    table = pd.DataFrame({"e": frame["extractant"].astype(str).to_numpy(),
                          "ae": np.abs(np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float))})
    return float(table.groupby("e")["ae"].mean().mean())


# --------------------------------------------------------------------------- #
# Tier 0 — sanity baselines
# --------------------------------------------------------------------------- #

@dataclass
class GlobalMean:
    name: str = "B0_GLOBAL_MEAN"
    tier: str = "T0"

    def fit_predict(self, train, y_train, test, context):
        return np.full(len(test), float(np.mean(y_train)))


@dataclass
class ConditionOnly:
    """Gradient boosting on conditions alone — no molecular information at all.

    The honest floor: whatever this scores is available without knowing which
    molecule was used, so it is the part of the task that is not chemistry.
    """

    name: str = "B1_COND_ONLY"
    tier: str = "T0"
    blocks: tuple[str, ...] = ("COND", "MASSACT")

    def fit_predict(self, train, y_train, test, context):
        from sklearn.ensemble import HistGradientBoostingRegressor
        columns = context.columns(*self.blocks)
        pre = FoldPreprocessor(columns).fit(train)
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=False, random_state=context.model_seed)
        model.fit(pre.transform(train), y_train,
                  sample_weight=extractant_balanced_weights(train["extractant"]))
        return clip_to_training_range(model.predict(pre.transform(test)), y_train)


@dataclass
class ChemicalNearestNeighbour:
    """Predict a new extractant from the behaviour of its nearest training chemistry.

    For a test row the candidate pool is the rows of the ``m`` training
    extractants most similar by Tanimoto.  Each candidate is weighted by the
    product of a fingerprint-similarity weight and a Gaussian kernel on the
    standardised continuous-condition axes, so a neighbour measured at the wrong
    acidity counts for little.  ``m`` and the bandwidth are chosen on the inner
    validation block; nothing reads a test target.

    This is the baseline a model has to beat to have learned anything beyond
    "copy the closest molecule you have seen".
    """

    name: str = "B2_NN_CHEMICAL"
    tier: str = "T0"
    neighbours: tuple[int, ...] = (1, 3, 5)
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)
    axes: tuple[str, ...] = ("massact__log10_cond__acid_concentration_M",
                             "massact__log10_cond__extractant_concentration_M")

    def _predict(self, train, y_train, test, m: int, bandwidth: float) -> np.ndarray:
        names = sorted(set(train["extractant"].astype(str)))
        index = {n: i for i, n in enumerate(names)}
        train_bits = (train.drop_duplicates("extractant").set_index("extractant")
                      .loc[names, list(ECFP_COLUMNS)].to_numpy(dtype=np.uint8))
        test_names = sorted(set(test["extractant"].astype(str)))
        test_bits = (test.drop_duplicates("extractant").set_index("extractant")
                     .loc[test_names, list(ECFP_COLUMNS)].to_numpy(dtype=np.uint8))
        similarity = tanimoto_matrix(test_bits, train_bits)

        pre = FoldPreprocessor(self.axes, standardise=True, add_indicator=False,
                               drop_constant=False, clip_to_train_range=True).fit(train)
        train_axes, test_axes = pre.transform(train), pre.transform(test)
        train_extractant = train["extractant"].astype(str).to_numpy()
        # Categorical setting: a neighbour in a different acid/diluent is penalised.
        categorical = [c for c in train.columns
                       if c.startswith(("cond__acid__", "cond__diluent__"))]
        train_cat = train[categorical].to_numpy(dtype=float)
        test_cat = test[categorical].to_numpy(dtype=float)

        out = np.empty(len(test), dtype=float)
        test_row_extractant = test["extractant"].astype(str).to_numpy()
        for position, name in enumerate(test_names):
            order = np.argsort(-similarity[position])[:m]
            chosen = {names[j] for j in order}
            weight_of = {names[j]: max(similarity[position][j], 1e-6) for j in order}
            mask = np.isin(train_extractant, list(chosen))
            pool_axes, pool_y = train_axes[mask], y_train[mask]
            pool_cat = train_cat[mask]
            pool_weight = np.array([weight_of[e] for e in train_extractant[mask]])
            rows = np.flatnonzero(test_row_extractant == name)
            for row in rows:
                distance = np.linalg.norm(pool_axes - test_axes[row], axis=1)
                mismatch = np.abs(pool_cat - test_cat[row]).sum(axis=1) / 2.0
                kernel = np.exp(-0.5 * ((distance / bandwidth) ** 2) - mismatch)
                w = pool_weight * np.maximum(kernel, 1e-12)
                out[row] = float(np.average(pool_y, weights=w))
        return out

    def fit_predict(self, train, y_train, test, context):
        best, best_score = None, np.inf
        for m in self.neighbours:
            for bandwidth in self.bandwidths:
                prediction = self._predict(context.inner_train, context.y_inner_train,
                                           context.inner_validation, m, bandwidth)
                score = _macro_mae(context.inner_validation, context.y_inner_validation, prediction)
                if score < best_score:
                    best, best_score = (m, bandwidth), score
        m, bandwidth = best
        self.selected_ = {"neighbours": m, "bandwidth": bandwidth, "inner_macro_mae": best_score}
        return clip_to_training_range(self._predict(train, y_train, test, m, bandwidth), y_train)


@dataclass
class ExtractantMeanOracleFree:
    """PAIRMEAN's Gen12 analogue: the training-chemotype mean nearest the test one.

    Not an oracle — it uses only training targets — but deliberately crude: it
    ignores conditions entirely and reports the level of the nearest training
    chemistry.  Its role is to show how much of any model's score is a level
    lookup.
    """

    name: str = "B3_NN_LEVEL_ONLY"
    tier: str = "T0"

    def fit_predict(self, train, y_train, test, context):
        names = sorted(set(train["extractant"].astype(str)))
        train_bits = (train.drop_duplicates("extractant").set_index("extractant")
                      .loc[names, list(ECFP_COLUMNS)].to_numpy(dtype=np.uint8))
        level = (pd.DataFrame({"e": train["extractant"].astype(str), "y": y_train})
                 .groupby("e")["y"].mean().loc[names].to_numpy())
        test_names = sorted(set(test["extractant"].astype(str)))
        test_bits = (test.drop_duplicates("extractant").set_index("extractant")
                     .loc[test_names, list(ECFP_COLUMNS)].to_numpy(dtype=np.uint8))
        similarity = tanimoto_matrix(test_bits, train_bits)
        nearest = {name: level[similarity[i].argmax()] for i, name in enumerate(test_names)}
        return clip_to_training_range(
            test["extractant"].astype(str).map(nearest).to_numpy(dtype=float), y_train)


# --------------------------------------------------------------------------- #
# Tier 1 — fingerprint / tabular
# --------------------------------------------------------------------------- #

@dataclass
class TreeContender:
    """CatBoost / XGBoost / RandomForest / ExtraTrees on one shared feature contract.

    The grid is pre-declared and small; the choice is made on the inner
    validation block of the outer fold and the chosen configuration is then
    refitted on the whole outer training set.  For the boosted families the
    number of rounds is chosen the same way and reused, so early stopping never
    touches an outer test row.
    """

    family: str = "catboost"
    name: str = ""
    tier: str = "T1"
    blocks: tuple[str, ...] = DEFAULT_BLOCKS
    selected_: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = f"T1_{self.family.upper()}"

    def _grid(self) -> list[dict]:
        if self.family == "catboost":
            return [{"depth": d, "l2_leaf_reg": l} for d in (4, 6) for l in (3.0, 10.0)]
        if self.family == "xgboost":
            return [{"max_depth": d, "min_child_weight": w} for d in (3, 6) for w in (1.0, 5.0)]
        if self.family in ("randomforest", "extratrees"):
            return [{"max_features": f, "min_samples_leaf": leaf}
                    for f in (0.30, "sqrt") for leaf in (1, 2)]
        raise KeyError(self.family)

    def _fit(self, x, y, weights, params: dict, seed: int, x_val=None, y_val=None):
        if self.family == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                iterations=int(params.get("iterations", 2000)), learning_rate=0.05,
                depth=int(params["depth"]), l2_leaf_reg=float(params["l2_leaf_reg"]),
                loss_function="MAE", random_seed=seed, verbose=False, allow_writing_files=False,
                thread_count=-1)
            if x_val is not None:
                model.fit(x, y, sample_weight=weights, eval_set=(x_val, y_val),
                          early_stopping_rounds=100, verbose=False)
            else:
                model.fit(x, y, sample_weight=weights, verbose=False)
            return model
        if self.family == "xgboost":
            from xgboost import XGBRegressor
            kwargs = dict(n_estimators=int(params.get("iterations", 2000)), learning_rate=0.05,
                          max_depth=int(params["max_depth"]),
                          min_child_weight=float(params["min_child_weight"]),
                          subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0,
                          objective="reg:absoluteerror", random_state=seed, n_jobs=-1,
                          tree_method="hist")
            if x_val is not None:
                kwargs["early_stopping_rounds"] = 100
            model = XGBRegressor(**kwargs)
            if x_val is not None:
                model.fit(x, y, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
            else:
                model.fit(x, y, sample_weight=weights)
            return model
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        cls = RandomForestRegressor if self.family == "randomforest" else ExtraTreesRegressor
        model = cls(n_estimators=500, max_features=params["max_features"],
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    random_state=seed, n_jobs=-1)
        model.fit(x, y, sample_weight=weights)
        return model

    def fit_predict(self, train, y_train, test, context):
        columns = context.columns(*self.blocks)
        inner = FoldPreprocessor(columns).fit(context.inner_train)
        x_inner = inner.transform(context.inner_train)
        x_valid = inner.transform(context.inner_validation)
        w_inner = extractant_balanced_weights(context.inner_train["extractant"])
        best, best_score, best_rounds = None, np.inf, None
        for params in self._grid():
            boosted = self.family in ("catboost", "xgboost")
            model = self._fit(x_inner, context.y_inner_train, w_inner, params,
                              context.model_seed,
                              x_valid if boosted else None,
                              context.y_inner_validation if boosted else None)
            prediction = model.predict(x_valid)
            score = _macro_mae(context.inner_validation, context.y_inner_validation, prediction)
            rounds = None
            if self.family == "catboost":
                rounds = int(model.get_best_iteration() or 0) + 1
            elif self.family == "xgboost":
                rounds = int(getattr(model, "best_iteration", 0) or 0) + 1
            if score < best_score:
                best, best_score, best_rounds = params, score, rounds
        params = dict(best)
        if best_rounds:
            params["iterations"] = max(50, best_rounds)
        self.selected_ = {**params, "inner_macro_mae": float(best_score)}
        outer = FoldPreprocessor(columns).fit(train)
        model = self._fit(outer.transform(train), y_train,
                          extractant_balanced_weights(train["extractant"]),
                          params, context.model_seed)
        return clip_to_training_range(model.predict(outer.transform(test)), y_train)


# --------------------------------------------------------------------------- #
# Tier 2 — descriptor MLP
# --------------------------------------------------------------------------- #

@dataclass
class DescriptorMLP:
    """A deliberately modest network: 183 independent extractants, not 1,329 rows.

    Width, depth, dropout and weight decay are fixed; only the early-stopping
    epoch is learned, on the inner validation block.  Three model seeds are
    averaged because a single neural seed on this cohort was measured elsewhere
    in the programme to be misleading.
    """

    name: str = "T2_MLP"
    tier: str = "T2"
    blocks: tuple[str, ...] = DEFAULT_BLOCKS
    hidden: tuple[int, ...] = (256, 64)
    dropout: float = 0.3
    weight_decay: float = 1e-4
    max_epochs: int = 400
    patience: int = 40
    n_seeds: int = 3
    selected_: dict = field(default_factory=dict)

    def _run(self, x_train, y_train, w_train, x_valid, y_valid, x_test, seed: int,
             epochs: int | None):
        import torch
        from torch import nn
        torch.manual_seed(seed)
        device = "cpu"
        net = []
        width = x_train.shape[1]
        for size in self.hidden:
            net += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        net += [nn.Linear(width, 1)]
        model = nn.Sequential(*net).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      weight_decay=self.weight_decay)
        xt = torch.tensor(x_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        wt = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)
        xv = torch.tensor(x_valid, dtype=torch.float32) if x_valid is not None else None
        yv = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1) if y_valid is not None else None
        best_state, best_score, best_epoch, bad = None, np.inf, 0, 0
        limit = epochs or self.max_epochs
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, limit + 1):
            model.train()
            order = torch.randperm(len(xt), generator=generator)
            for start in range(0, len(order), 64):
                batch = order[start:start + 64]
                optimiser.zero_grad()
                loss = (wt[batch] * (model(xt[batch]) - yt[batch]).abs()).mean()
                loss.backward()
                optimiser.step()
            if xv is not None:
                model.eval()
                with torch.no_grad():
                    score = float((model(xv) - yv).abs().mean())
                if score < best_score - 1e-5:
                    best_score, best_epoch, bad = score, epoch, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if xv is not None and best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor(x_test, dtype=torch.float32)).squeeze(1).numpy()
        return prediction, best_epoch, best_score

    def fit_predict(self, train, y_train, test, context):
        columns = context.columns(*self.blocks)
        inner = FoldPreprocessor(columns, standardise=True,
                                 clip_to_train_range=True).fit(context.inner_train)
        x_inner, x_valid = inner.transform(context.inner_train), inner.transform(context.inner_validation)
        w_inner = extractant_balanced_weights(context.inner_train["extractant"])
        epochs, scores = [], []
        for offset in range(self.n_seeds):
            _, best_epoch, best_score = self._run(
                x_inner, context.y_inner_train, w_inner, x_valid, context.y_inner_validation,
                x_valid, context.model_seed + offset, None)
            epochs.append(max(best_epoch, 1)); scores.append(best_score)
        chosen = int(np.median(epochs))
        self.selected_ = {"epochs": chosen, "inner_val_mae": float(np.mean(scores))}
        outer = FoldPreprocessor(columns, standardise=True,
                                 clip_to_train_range=True).fit(train)
        x_train, x_test = outer.transform(train), outer.transform(test)
        w_train = extractant_balanced_weights(train["extractant"])
        predictions = [self._run(x_train, y_train, w_train, None, None, x_test,
                                 context.model_seed + offset, chosen)[0]
                       for offset in range(self.n_seeds)]
        return clip_to_training_range(np.mean(predictions, axis=0), y_train)


def default_ladder() -> list:
    return [
        GlobalMean(), ConditionOnly(), ChemicalNearestNeighbour(), ExtractantMeanOracleFree(),
        TreeContender(family="catboost"), TreeContender(family="xgboost"),
        TreeContender(family="randomforest"), TreeContender(family="extratrees"),
        DescriptorMLP(),
    ]
