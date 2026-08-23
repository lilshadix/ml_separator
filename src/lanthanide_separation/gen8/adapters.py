"""The calibration-adapter interface — one signature for every few-shot method.

Every gen8 method that turns *k measurements of a new ligand* into *predictions
for the rest of its surface* implements the same three-line protocol, and the
protocol is designed so that **target leakage is structurally impossible**:

.. code-block:: python

    class Adapter(Protocol):
        name: str
        def predict(self, block, prediction, selected, observed, context) -> np.ndarray

``block``
    every row of the held-out ligand, features only — conditions, metal, ligand
    chemistry.  The target column is *not* in it.
``prediction``
    the frozen global model's zero-shot prediction for those rows.  An adapter may
    ignore it entirely (a Neural Process does), use it as an anchor (offset
    correction does) or use it as one feature among several.
``selected``
    positional indices of the rows the experimentalist is allowed to have measured.
``observed``
    **the only targets in the signature**: ``truth[selected]``, and nothing else.
    An adapter physically cannot read the target of a row it did not select,
    because that number is never passed to it.
``context``
    which split seed and fold this ligand was held out in, so an adapter carrying a
    fold-specific trained model can look up the right one.

Two consequences worth stating.

*Everything becomes paired.*  Offset correction, a ridge-shrunk response-coefficient
update, a Conditional Neural Process and a physics-latent hypernetwork are all
scored on the same ligand, the same repeat, the same candidate pool, the same
selected rows and the same evaluation rows.  A difference between them is a
difference in method, not in what they happened to be given.

*Acquisition and adaptation stay separable.*  A policy chooses ``selected`` without
seeing any target; an adapter turns ``selected`` into predictions.  The 2x2 of
(policy, adapter) is therefore fully crossed, which is what lets the final report
attribute a gain to the measurement, the architecture, or the choice of point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .kshot import DEFAULT_RIDGE, design_matrix, ridge_fit


@dataclass(frozen=True)
class AdaptContext:
    """Which held-out fold this ligand belongs to, and who it is."""

    split_seed: int
    fold: int
    extractant: str


class Adapter(Protocol):
    name: str

    def predict(self, block: pd.DataFrame, prediction: np.ndarray, selected: np.ndarray,
                observed: np.ndarray, context: AdaptContext) -> np.ndarray:
        """Predictions for **every** row of ``block``; only ``selected`` rows are known."""


# --------------------------------------------------------------------------- #
# The reference adapters
# --------------------------------------------------------------------------- #

@dataclass
class ZeroShot:
    """The frozen global model, untouched.  k is ignored."""

    name: str = "ZERO_SHOT"

    def predict(self, block, prediction, selected, observed, context):
        return np.asarray(prediction, dtype=float)


@dataclass
class RidgeOffset:
    """gen7's offset correction, generalised to K1 / K2 / K3 (brief §7).

    ``K1`` moves the level only, ``K2`` adds a lanthanide trend, ``K3`` adds acid
    and extractant-concentration slope adjustments.  The intercept is never
    penalised and everything else always is, so a mode with more freedom than the
    data supports degenerates gracefully to the mode below it rather than
    exploding — which is what happened to gen7's unrestricted affine fit at k = 2.
    """

    mode: str = "K1"
    penalty: float = DEFAULT_RIDGE
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", f"OFFSET_{self.mode}") if False else None
            self.name = f"OFFSET_{self.mode}"

    def predict(self, block, prediction, selected, observed, context):
        prediction = np.asarray(prediction, dtype=float)
        if len(selected) == 0:
            return prediction
        design = design_matrix(block, self.mode)
        residual = observed - prediction[selected]
        beta = ridge_fit(design[selected], residual, penalty=self.penalty)
        return prediction + design @ beta


@dataclass
class NoModel:
    """The null: predict the mean of what you measured.  No model at all.

    gen5's k-shot study died on this null and gen8 carries it in every table.
    """

    name: str = "NO_MODEL"

    def predict(self, block, prediction, selected, observed, context):
        if len(selected) == 0:
            return np.full(len(block), float("nan"))
        return np.full(len(block), float(np.mean(observed)))


@dataclass
class NearestObserved:
    """Predict each row from the measured row nearest it in condition space.

    A local lookup with no global model in it at all.  It is here because gen7's
    level benchmark found that *no model beat a 1-nearest-neighbour lookup* on the
    condition-adjusted level, and the same question has to be asked of the
    calibrated surface: if this wins, the response surface is a lookup too.
    """

    name: str = "NEAREST_OBSERVED"
    axes: tuple = ()

    def predict(self, block, prediction, selected, observed, context):
        from .protocols import _standardised_axes
        if len(selected) == 0:
            return np.full(len(block), float("nan"))
        axes = _standardised_axes(block)
        distance = np.linalg.norm(axes[:, None, :] - axes[selected][None, :, :], axis=2)
        return observed[np.argmin(distance, axis=1)]


def default_adapters(penalty: float = DEFAULT_RIDGE) -> list:
    return [ZeroShot(), RidgeOffset(mode="K1", penalty=penalty),
            RidgeOffset(mode="K2", penalty=penalty), RidgeOffset(mode="K3", penalty=penalty),
            NoModel(), NearestObserved()]


# --------------------------------------------------------------------------- #
# Trainable adapters
# --------------------------------------------------------------------------- #

class TrainableAdapter(Adapter, Protocol):
    """An adapter carrying its own model, refitted once per held-out fold.

    ``fit_fold`` is called with **only the training rows of that fold** — the same
    rows the frozen global model was fitted on, under the same chemotype-blocked
    plan — so a trainable adapter is under exactly the evaluation contract every
    other gen8 method is.  It is called once per (split seed, fold) and the fitted
    state is reused for every ligand, every repeat, every k and every policy in
    that fold, which is also what makes the comparison paired.
    """

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *,
                 split_seed: int, fold: int, model_seed: int) -> None: ...


def is_trainable(adapter) -> bool:
    return hasattr(adapter, "fit_fold")
