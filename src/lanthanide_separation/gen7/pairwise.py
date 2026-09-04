"""Pairwise difference regression on the level (brief §3, Loss 3, in tree form).

The literature's strongest recommendation for our regime — Tynes et al., *Pairwise
Difference Regression*, JCIM 61(8) 2021 — makes the same argument the brief makes
about relational supervision, but without a neural network.  Instead of learning
``y = f(x)`` from *n* examples, learn ``y_i − y_j = g(x_i, x_j)`` from *n²* pairs,
then predict a new molecule by averaging its predicted difference against every
training anchor:

.. code-block:: text

    ŷ_t = mean_j ( y_j + g(x_t, x_j) )

Three reasons it is the right thing to try on *this* problem specifically:

1. **The level task has ~120 training examples, not 5,248 rows.**  Squaring 120 into
   ~14,000 pairs is the largest honest increase in effective sample size available
   without a single new measurement.
2. **Differences are better determined than absolutes here.**  Five generations of
   this project have found the *contrast* between two systems far easier to predict
   than either level — that is what a separation factor is — and PADRE is the
   formulation that makes the model predict contrasts natively.
3. **The averaging is a shrinkage estimator.**  Each anchor contributes an
   independent estimate of the query's level; their mean has variance ~1/n of a
   single prediction, which is exactly the medicine for a 120-point regression.

The prediction is also free uncertainty: the *spread* across anchors says how much
the anchors disagree about where this molecule sits, and that is reported.

Two failure modes this implementation is written around:

* **Anchor leakage.**  Anchors must come from the training fold only.  A test ligand
  is never its own anchor and never anchors another test ligand.
* **Antisymmetry.**  ``g(x_i, x_j)`` should equal ``−g(x_j, x_i)``.  Rather than hope
  a tree learns it, every pair is included in both orders and the prediction is
  explicitly antisymmetrised, which halves the variance for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..levels import _as_float_frame, group_balanced_weights
from .contenders import extratrees
from .harness import FoldContext


def _pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``[x_a, x_b, x_a − x_b]`` — the difference is handed over, not implied."""
    return np.hstack([a, b, a - b])


@dataclass
class PairwiseDifferenceLevel:
    """PADRE on the per-ligand level, added to a conventional response model.

    The level head is pairwise; the response head is the ordinary within-ligand
    centred regression, because the response is already the part that works.
    """

    level_blocks: tuple[str, ...] = ("DONORS", "PHYSCHEM", "LIGPHYS")
    response_blocks: tuple[str, ...] = ("METAL", "METALPHYS", "COND", "MASSACTION", "DONORS")
    estimator: Callable[[int], object] = extratrees
    max_pairs: int = 40_000
    #: Hard ceiling on the pair design matrix in elements.  ``_pair_features`` triples
    #: the width, so 40,000 pairs over a 2,048-bit fingerprint is 246 M floats — 2 GB
    #: before the forest allocates anything.  The pair count is reduced to fit and the
    #: reduction is recorded in ``context.extras`` rather than being silent.
    max_matrix_elements: int = 40_000_000
    n_anchors: int = 120
    name: str = "PADRE_level"
    weighting: str = "cluster"

    def _ligand_view(self, frame: pd.DataFrame, columns: Sequence[str]):
        first = frame.drop_duplicates("extractant")
        x = _as_float_frame(first, columns).to_numpy()
        return first["extractant"].astype(str).to_numpy(), x

    @staticmethod
    def _clean(train_x, test_x):
        keep = ~np.all(np.isnan(train_x), axis=0)
        train_x, test_x = train_x[:, keep], test_x[:, keep]
        median = np.nan_to_num(np.nanmedian(train_x, axis=0))
        train_x = np.where(np.isfinite(train_x), train_x, median)
        test_x = np.where(np.isfinite(test_x), test_x, median)
        return train_x, test_x

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        cohort = context.cohort
        y = np.asarray(y_train, dtype=float)
        rng = np.random.default_rng(context.model_seed)

        level_columns = cohort.block_columns(
            tuple(b for b in self.level_blocks if b in cohort.blocks))
        train_names, train_x = self._ligand_view(train, level_columns)
        test_names, test_x = self._ligand_view(test, level_columns)
        train_x, test_x = self._clean(train_x, test_x)
        train_level = pd.Series(y).groupby(train["extractant"].astype(str).to_numpy()).mean()
        train_level = train_level.reindex(train_names).to_numpy(float)

        n = len(train_names)
        n_pairs_used = 0.0
        if n < 8:
            level_prediction = np.full(len(test_names), float(np.mean(y)))
            spread = np.zeros(len(test_names))
        else:
            left, right = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            left, right = left.ravel(), right.ravel()
            keep = left != right
            left, right = left[keep], right[keep]
            width = 3 * train_x.shape[1]
            budget = min(self.max_pairs, max(2_000, self.max_matrix_elements // max(1, width)))
            if len(left) > budget:
                take = rng.choice(len(left), size=budget, replace=False)
                left, right = left[take], right[take]
            # per ROW, not per ligand: the harness writes extras into ``arr[test_index]``
            n_pairs_used = float(len(left))
            features = _pair_features(train_x[left], train_x[right])
            target = train_level[left] - train_level[right]
            model = self.estimator(context.model_seed)
            model.fit(features, target)

            anchors = np.arange(n) if n <= self.n_anchors else rng.choice(
                n, size=self.n_anchors, replace=False)
            estimates = np.empty((len(test_names), len(anchors)))
            for k, anchor in enumerate(anchors):
                repeated = np.repeat(train_x[anchor][None, :], len(test_names), axis=0)
                forward = model.predict(_pair_features(test_x, repeated))
                backward = model.predict(_pair_features(repeated, test_x))
                # explicit antisymmetrisation: g(a,b) = -g(b,a) by construction
                estimates[:, k] = train_level[anchor] + 0.5 * (forward - backward)
            level_prediction = estimates.mean(axis=1)
            spread = estimates.std(axis=1)

        level_of = dict(zip(test_names, level_prediction))
        spread_of = dict(zip(test_names, spread))
        test_ligand = test["extractant"].astype(str).to_numpy()

        # --- response head, unchanged ---------------------------------------- #
        response_columns = cohort.block_columns(
            tuple(b for b in self.response_blocks if b in cohort.blocks))
        rx_train, rx_test = self._clean(
            _as_float_frame(train, response_columns).to_numpy(),
            _as_float_frame(test, response_columns).to_numpy())
        centred = y - pd.Series(y).groupby(
            train["extractant"].astype(str).to_numpy()).transform("mean").to_numpy()
        response_model = self.estimator(context.model_seed + 1)
        response_model.fit(rx_train, centred,
                           sample_weight=group_balanced_weights(train["ecfp_cluster"])
                           if self.weighting == "cluster" else None)
        response = np.asarray(response_model.predict(rx_test), dtype=float)

        offset = np.array([level_of[n] for n in test_ligand])
        context.extras["offset_hat"] = offset
        context.extras["anchor_spread"] = np.array([spread_of[n] for n in test_ligand])
        context.extras["padre_pairs"] = np.full(len(test), n_pairs_used)
        prediction = offset + response
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)
