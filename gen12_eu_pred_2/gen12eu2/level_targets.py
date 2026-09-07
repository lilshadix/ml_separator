"""Candidate definitions of the extractant-specific level ``alpha_i``.  Training-only.

Conceptual model: ``y_ij = alpha_i + g(c_ij) + eps_ij``.  Five candidates for
``alpha_i`` are computed *inside a fold from that fold's training rows* and,
for held-out extractants, from their own observed cells using only quantities
(``g``, the grand mean, variance ratios) fitted on training rows.  No candidate
lets a held-out extractant's target influence anything it is later predicted from.

1. ``raw_mean``      — mean ``log_D`` over the extractant's observed cells.
2. ``median``        — median over the cells.
3. ``fe_intercept``  — fixed-effects intercept: ``mean_j(y_ij - g(c_ij))`` where
                        ``g`` is a within-extractant ridge on the condition blocks,
                        fitted on training rows demeaned per extractant (so ``g``
                        carries no level information by construction) and centred
                        so that its extractant-balanced training mean is zero.
4. ``shrunk_mean``   — James-Stein shrinkage of the raw mean towards the training
                        grand mean: ``mu + n/(n + sigma2/tau2) * (ybar - mu)``, with
                        ``sigma2`` (within) and ``tau2`` (between) estimated on training.
5. ``resid_condonly`` — mean residual after a *global* condition-only ExtraTrees model
                        fitted on the training rows (in-sample).  Included because it
                        is the "natural" definition a reader might propose; expected
                        to be distorted because a global condition model absorbs
                        level through each extractant's condition signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gen12eu.preprocess import FoldPreprocessor, extractant_balanced_weights  # noqa: E402

TARGET = "log_D"
CONDITION_BLOCKS: tuple[str, ...] = ("COND", "MASSACT")
RIDGE_LAMBDA = 1.0          # pre-declared; on standardised demeaned features


@dataclass
class ConditionResponse:
    """``g(c)``: a within-extractant ridge on the condition blocks.  Level-free by construction."""

    columns: tuple[str, ...]
    ridge_lambda: float = RIDGE_LAMBDA
    pre_: FoldPreprocessor | None = field(default=None, init=False)
    beta_: np.ndarray | None = field(default=None, init=False)
    centre_: float = field(default=0.0, init=False)
    n_extractants_informing_: int = field(default=0, init=False)

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "ConditionResponse":
        self.pre_ = FoldPreprocessor(self.columns, standardise=True, add_indicator=True,
                                     drop_constant=True, clip_to_train_range=True).fit(train)
        x = self.pre_.transform(train)
        e = train["extractant"].astype(str).to_numpy()
        table = pd.DataFrame(x)
        table["_e"] = e
        # demean within extractant: one-row extractants become exact zeros and carry no slope
        xd = (table.groupby("_e").transform(lambda s: s - s.mean())).to_numpy(dtype=float)
        yd = pd.Series(y).groupby(e).transform(lambda s: s - s.mean()).to_numpy(dtype=float)
        counts = pd.Series(e).map(pd.Series(e).value_counts()).to_numpy()
        keep = counts >= 2
        self.n_extractants_informing_ = int(len(set(e[keep])))
        # extractant-balanced weights so one 306-row extractant does not define g
        w = extractant_balanced_weights(pd.Series(e[keep]))
        xk, yk = xd[keep], yd[keep]
        sw = np.sqrt(w)[:, None]
        a = (xk * sw).T @ (xk * sw) + self.ridge_lambda * np.eye(xk.shape[1])
        b = (xk * sw).T @ (yk * sw[:, 0])
        self.beta_ = np.linalg.solve(a, b)
        g_train = x @ self.beta_
        # centre g so its extractant-balanced training mean is zero: alpha then reads
        # as "log_D at the average training condition"
        self.centre_ = float(np.average(g_train, weights=extractant_balanced_weights(pd.Series(e))))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pre_.transform(frame) @ self.beta_ - self.centre_


@dataclass
class ConditionOnlyTrees:
    """Definition 5's global condition-only model (the same learner as Gen12's ABL_A)."""

    columns: tuple[str, ...]
    seed: int = 42
    pre_: FoldPreprocessor | None = field(default=None, init=False)
    model_: object = field(default=None, init=False)

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "ConditionOnlyTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        self.pre_ = FoldPreprocessor(self.columns).fit(train)
        self.model_ = ExtraTreesRegressor(n_estimators=300, max_features=0.30, min_samples_leaf=2,
                                          random_state=self.seed, n_jobs=-1)
        self.model_.fit(self.pre_.transform(train), y,
                        sample_weight=extractant_balanced_weights(train["extractant"]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.pre_.transform(frame))


@dataclass
class LevelDefinitions:
    """All five candidates, fitted on one fold's training rows."""

    columns: tuple[str, ...]
    seed: int = 42
    g_: ConditionResponse | None = field(default=None, init=False)
    trees_: ConditionOnlyTrees | None = field(default=None, init=False)
    mu_: float = field(default=0.0, init=False)
    sigma2_: float = field(default=1.0, init=False)
    tau2_: float = field(default=1.0, init=False)

    def fit(self, train: pd.DataFrame, y: np.ndarray, *, with_trees: bool = True) -> "LevelDefinitions":
        self.g_ = ConditionResponse(self.columns).fit(train, y)
        if with_trees:
            self.trees_ = ConditionOnlyTrees(self.columns, seed=self.seed).fit(train, y)
        e = train["extractant"].astype(str).to_numpy()
        per = pd.Series(y).groupby(e)
        means = per.mean()
        self.mu_ = float(means.mean())                     # extractant-balanced grand mean
        within = per.var(ddof=1).dropna()
        self.sigma2_ = float(within.mean()) if len(within) else 1.0
        self.tau2_ = max(float(means.var(ddof=1)) - self.sigma2_ * float((1.0 / per.size()).mean()), 1e-3)
        return self

    def per_extractant(self, frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """One row per extractant present in ``frame``, all five definitions."""
        e = frame["extractant"].astype(str).to_numpy()
        g = self.g_.predict(frame)
        table = pd.DataFrame({"extractant": e, "y": y, "y_minus_g": y - g})
        if self.trees_ is not None:
            table["y_minus_tree"] = y - self.trees_.predict(frame)
        grouped = table.groupby("extractant")
        out = pd.DataFrame({
            "n_rows": grouped.size(),
            "raw_mean": grouped["y"].mean(),
            "median": grouped["y"].median(),
            "fe_intercept": grouped["y_minus_g"].mean(),
        })
        n = out["n_rows"].to_numpy(dtype=float)
        weight = n / (n + self.sigma2_ / self.tau2_)
        out["shrunk_mean"] = self.mu_ + weight * (out["raw_mean"].to_numpy() - self.mu_)
        if self.trees_ is not None:
            out["resid_condonly"] = grouped["y_minus_tree"].mean()
        out["within_sd_raw"] = grouped["y"].std(ddof=1)
        out["within_sd_fe"] = grouped["y_minus_g"].std(ddof=1)
        return out.reset_index()


DEFINITIONS: tuple[str, ...] = ("raw_mean", "median", "fe_intercept", "shrunk_mean", "resid_condonly")
