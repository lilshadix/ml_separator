"""``CurveBoost`` — boosted forests whose loss knows what a titration is.

The gen9 hypothesis in one sentence: *the model draws unseen titrations flat not
because it lacks a molecular representation but because a row-wise objective is
minimised by conditional averaging, and averaging destroys derivatives.*  Testing
that needs a learner whose objective can be changed and otherwise held fixed —
which rules out the frozen ``ExtraTreesRegressor``, whose splitting rule is not a
loss at all.

So gen9 boosts *forests* on the functional gradient of

.. code-block:: text

    L_total = L_row  +  lambda_delta * L_delta  +  lambda_span * L_span

``L_row``
    Huber on ``f(x_i) - y_i``, cluster-balanced exactly as every gen5-gen8 arm is.
``L_delta``
    Huber on ``(f(x_j) - f(x_i)) - (y_j - y_i)`` over pairs of rows on the same
    reconstructed curve.  Invariant to any per-curve constant — it asks for the
    *shape* of the response and never for the level, which is the quantity six
    generations of evidence say is not predictable from structure.
``L_span``
    Huber on ``(max f - min f) - (max y - min y)`` over a curve.  Also
    offset-invariant.  It exists because the delta term can be made small by a
    curve that is uniformly a little too flat, whereas the span term states the
    measured pathology directly: predicted dynamic range at 5 % of truth.  Its
    subgradient lives on two rows per curve, so it is deliberately a *small* term.

**Why a forest per stage, and why that matters.**  Each stage fits a small
``ExtraTreesRegressor`` to the pseudo-residual ``z = -g / w`` with
``sample_weight = w``, so a leaf's value is the weight-normalised mean negative
gradient in that leaf.  Two consequences, and the second is the reason the design
is worth the sentence:

* the weak learner averages, so a stage's step has the variance-reduction that
  makes ExtraTrees strong on 5,000 noisy rows in 2,100 mostly-sparse dimensions —
  single-tree boosting measured 0.09 macro MAE worse here, which would have handed
  every shape arm a handicap it could not have earned its way out of;
* **one stage at ``learning_rate = 1`` with a squared row loss and no curve terms
  is algebraically the frozen model**: ``F = ybar + ExtraTrees(y - ybar)``.  The
  gen9 arm family therefore *contains* ``REC_ecfp_plus_recovered`` as an exact
  special case, which is asserted in
  ``tests/test_gen9_curve_objective.py``.  The control is not a
  near-equivalent of the thing gen9 wants to beat; it is that thing.

**Why plain gradient steps and not Newton leaf values.**  The delta term couples
rows, so its Hessian is not diagonal and per-leaf Newton refinement is unavailable
without approximating the coupling away.  Rather than approximate silently, each
stage takes a shrunk functional-gradient step.

Per-stage values of every term are recorded in :attr:`CurveBoost.history`, because
a shape arm that "works" by having silently switched the row loss off is a result
about loss scaling and not about chemistry (brief §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .curves import CurvePairs

#: Huber transition points, in log-D units, fixed before any arm was scored.
#: 1.0 for rows — the loss is then quadratic on ordinary rows and linear on the
#: corrupted-decade outliers the data audit keeps finding; 0.5 for deltas and
#: spans, which are differences and already an order of magnitude tighter.
DEFAULT_ROW_DELTA = 1.0
DEFAULT_PAIR_DELTA = 0.5
DEFAULT_SPAN_DELTA = 0.5
#: ``row_delta`` at or above this is squared error for every residual in the corpus.
SQUARED_LOSS_DELTA = 1e6

#: Decimal places the running prediction is rounded to after every stage.
#:
#: This is not cosmetic and it is not a tolerance — it is a correctness fix.
#: ``ExtraTreesRegressor(n_jobs=-1)`` sums its trees in thread-completion order, so
#: its predictions move in the last bit between runs.  In a *single* forest that is
#: 1e-15 and irrelevant.  Inside a boosting recursion it is not: the next stage's
#: pseudo-residual inherits the perturbation, a near-tied split comparison flips,
#: the stage grows a **different tree**, and the divergence compounds.  Measured on
#: one fold of the real cohort, two runs of the identical configuration in the same
#: process differed by up to **0.139 log units** on every one of 382 held-out rows —
#: enough to move a pooled MAE by 0.007, which is the size of effects this study
#: reports.
#:
#: Rounding the running prediction to 1e-9 log units — nine orders of magnitude
#: below any measurement in the corpus — removes the perturbation before it can
#: reach a split comparison, and restores bit-identical results while keeping the
#: forests parallel.  ``n_jobs=1`` also fixes it and costs roughly six times the
#: runtime.  Both are tested.
DETERMINISM_DECIMALS = 9


@dataclass(frozen=True)
class BoostConfig:
    """Everything about the learner except the loss weights.

    Held identical across every arm of the shape sweep, so a difference between
    arms is a difference in objective.  Selected once by inner group-CV on
    training partitions only (``scripts/gen9_tune.py``) and then frozen.
    """

    n_stages: int = 10
    trees_per_stage: int = 60
    learning_rate: float = 0.3
    max_features: float = 0.30
    min_samples_leaf: int = 2
    max_depth: int | None = None
    subsample: float = 1.0
    row_delta: float = DEFAULT_ROW_DELTA
    pair_delta: float = DEFAULT_PAIR_DELTA
    span_delta: float = DEFAULT_SPAN_DELTA
    clip_to_training_range: bool = True
    n_jobs: int = -1
    #: See :data:`DETERMINISM_DECIMALS`.  ``None`` disables the rounding, which makes
    #: the arm irreproducible under ``n_jobs != 1`` and is only useful for measuring
    #: that fact.
    round_decimals: int | None = DETERMINISM_DECIMALS

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def total_trees(self) -> int:
        return int(self.n_stages * self.trees_per_stage)


#: The configuration that reproduces ``REC_ecfp_plus_recovered`` exactly.
FROZEN_EQUIVALENT = BoostConfig(n_stages=1, trees_per_stage=400, learning_rate=1.0,
                                row_delta=SQUARED_LOSS_DELTA, min_samples_leaf=2,
                                max_features=0.30, subsample=1.0,
                                clip_to_training_range=True)


def _huber_gradient(error: np.ndarray, delta: float) -> np.ndarray:
    """d/de of ``0.5 e^2`` for ``|e| <= delta`` and ``delta(|e| - delta/2)`` beyond."""
    return np.clip(error, -delta, delta)


def _huber_value(error: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(absolute <= delta, 0.5 * error ** 2, delta * (absolute - 0.5 * delta))


@dataclass
class CurveBoost:
    """Boosted forests under ``L_row + lambda_delta L_delta + lambda_span L_span``."""

    config: BoostConfig = field(default_factory=BoostConfig)
    lambda_delta: float = 0.0
    lambda_span: float = 0.0
    random_state: int = 0

    _stages: list = field(default_factory=list, init=False, repr=False)
    _base: float = field(default=0.0, init=False, repr=False)
    _lo: float = field(default=-np.inf, init=False, repr=False)
    _hi: float = field(default=np.inf, init=False, repr=False)
    history: pd.DataFrame | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # loss terms
    # ------------------------------------------------------------------ #

    def _row_term(self, F: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
        error = F - y
        gradient = w * _huber_gradient(error, self.config.row_delta)
        value = float(np.sum(w * _huber_value(error, self.config.row_delta)) / max(len(w), 1))
        return gradient, value

    def _delta_term(self, F: np.ndarray, y: np.ndarray, pairs: CurvePairs,
                    n: int) -> tuple[np.ndarray, float]:
        gradient = np.zeros(n)
        if self.lambda_delta <= 0 or pairs is None or len(pairs) == 0:
            return gradient, 0.0
        i, j = pairs.pairs[:, 0], pairs.pairs[:, 1]
        error = (F[j] - F[i]) - (y[j] - y[i])
        weight = pairs.pair_weight * float(n)
        contribution = self.lambda_delta * weight * _huber_gradient(error, self.config.pair_delta)
        np.add.at(gradient, j, contribution)
        np.add.at(gradient, i, -contribution)
        value = float(np.sum(pairs.pair_weight * _huber_value(error, self.config.pair_delta)))
        return gradient, value

    def _span_term(self, F: np.ndarray, y: np.ndarray, pairs: CurvePairs,
                   n: int) -> tuple[np.ndarray, float]:
        """Subgradient of the predicted range, carried by the two extreme rows.

        ``max`` and ``min`` are subdifferentiable and the subgradient is a one-hot
        on the arg-extremum — which is exactly the row a step should move.  Ties go
        to ``argmax``/``argmin``'s first index, deterministic in row order.
        """
        gradient = np.zeros(n)
        if self.lambda_span <= 0 or pairs is None or pairs.n_curves == 0:
            return gradient, 0.0
        total = 0.0
        for c, members in enumerate(pairs.curve_members):
            if len(members) < 2:
                continue
            fv, yv = F[members], y[members]
            hi, lo = int(np.argmax(fv)), int(np.argmin(fv))
            if hi == lo:
                continue
            error = (fv[hi] - fv[lo]) - (float(yv.max()) - float(yv.min()))
            weight = float(pairs.curve_weight[c])
            step = (self.lambda_span * weight * float(n)
                    * float(_huber_gradient(np.array([error]), self.config.span_delta)[0]))
            gradient[members[hi]] += step
            gradient[members[lo]] -= step
            total += weight * float(_huber_value(np.array([error]), self.config.span_delta)[0])
        return gradient, total

    # ------------------------------------------------------------------ #
    # fit / predict
    # ------------------------------------------------------------------ #

    def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None,
            pairs: CurvePairs | None = None, record_history: bool = True) -> "CurveBoost":
        from sklearn.ensemble import ExtraTreesRegressor

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if x.shape[0] != n:
            raise ValueError(f"x has {x.shape[0]} rows, y has {n}")
        weight = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weight.shape != (n,):
            raise ValueError(f"sample_weight has shape {weight.shape}, expected {(n,)}")
        if not np.all(weight > 0):
            raise ValueError("sample weights must be strictly positive")
        # Normalised to mean 1, so a gradient entry is O(1) and the learning rate
        # means the same thing whatever the cohort size.
        weight = weight / weight.sum() * n

        if pairs is not None and len(pairs):
            flat = pairs.pairs.reshape(-1)
            if flat.min() < 0 or flat.max() >= n:
                raise AssertionError(
                    "a curve pair addresses a row outside the training partition — "
                    "this is the train/test straddle gen9 forbids")

        rng = np.random.default_rng(self.random_state)
        self._base = float(np.average(y, weights=weight))
        # The initial constant is exact by construction and carries no parallel
        # noise, so it is *not* rounded: rounding it would perturb the first stage's
        # pseudo-residual by ~5e-10 and — by the same amplification this rounding
        # exists to prevent — grow a different first tree, breaking the exact nesting
        # of the frozen model inside this arm family.
        F = np.full(n, self._base)
        self._stages = []
        span = float(y.max() - y.min())
        self._lo, self._hi = float(y.min()) - 0.5 * span, float(y.max()) + 0.5 * span

        rows = np.arange(n)
        records: list[dict] = []
        for stage in range(self.config.n_stages):
            g_row, v_row = self._row_term(F, y, weight)
            g_delta, v_delta = self._delta_term(F, y, pairs, n)
            g_span, v_span = self._span_term(F, y, pairs, n)
            gradient = g_row + g_delta + g_span
            if record_history:
                records.append({
                    "stage": stage, "L_row": v_row,
                    "L_delta_weighted": self.lambda_delta * v_delta,
                    "L_span_weighted": self.lambda_span * v_span,
                    "L_delta_raw": v_delta, "L_span_raw": v_span,
                    "grad_row_absmean": float(np.abs(g_row).mean()),
                    "grad_delta_absmean": float(np.abs(g_delta).mean()),
                    "grad_span_absmean": float(np.abs(g_span).mean()),
                    "train_mae": float(np.abs(F - y).mean()),
                })
            # Pseudo-residual: dividing by the weight and then fitting *with* the
            # weight makes a leaf value the weight-normalised mean negative
            # gradient, which is what turns "one stage of this" into "the frozen
            # ExtraTrees" under a squared row loss.
            pseudo = -gradient / weight
            index = rows
            if self.config.subsample < 1.0:
                size = max(2, int(round(self.config.subsample * n)))
                index = rng.choice(n, size=size, replace=False)
            forest = ExtraTreesRegressor(
                n_estimators=self.config.trees_per_stage,
                max_features=self.config.max_features,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                random_state=int(rng.integers(1, 2 ** 31 - 1)),
                n_jobs=self.config.n_jobs)
            forest.fit(x[index], pseudo[index], sample_weight=weight[index])
            # Rounded before it can reach the next stage's split comparisons; see
            # DETERMINISM_DECIMALS.
            F = self._stabilise(F + self.config.learning_rate * forest.predict(x))
            self._stages.append(forest)
        self.history = pd.DataFrame.from_records(records) if records else None
        return self

    def _stabilise(self, values: np.ndarray) -> np.ndarray:
        if self.config.round_decimals is None:
            return values
        return np.round(values, self.config.round_decimals)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape[0], self._base)
        for forest in self._stages:
            out = self._stabilise(out + self.config.learning_rate * forest.predict(x))
        if self.config.clip_to_training_range:
            # The same guard the gen7 ``Tabular`` contender applies, kept so no gen9
            # arm can win by being allowed to extrapolate further than the arm it is
            # compared with.
            out = np.clip(out, self._lo, self._hi)
        return out

    def prediction_spread(self, x: np.ndarray, *, n_members: int = 40) -> np.ndarray:
        """An ensemble spread, so gen9 arms can feed the uncertainty baselines too.

        Each "member" takes one tree from every stage instead of that stage's forest
        mean, giving ``n_members`` alternative paths through the same boosted model;
        the spread across them is the proxy.  It is not a posterior and is not
        claimed to be one — gen8 established that no uncertainty-driven rule picks a
        good first experiment.  It exists so that ``MAX_ENSEMBLE_SD`` and
        ``MIN_ENSEMBLE_SD`` appear in a gen9 arm's table rather than silently falling
        back to random, which would make the table ragged for a reason unrelated to
        the result.
        """
        x = np.asarray(x, dtype=float)
        if not self._stages:
            return np.zeros(x.shape[0])
        members = min(n_members, min(len(f.estimators_) for f in self._stages))
        paths = np.zeros((members, x.shape[0]))
        for m in range(members):
            total = np.zeros(x.shape[0])
            for forest in self._stages:
                total += self.learning_rate_of(forest) * forest.estimators_[m].predict(x)
            paths[m] = total
        return paths.std(axis=0)

    def learning_rate_of(self, _forest) -> float:
        return self.config.learning_rate

    def staged_predict(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape[0], self._base)
        for i, forest in enumerate(self._stages, start=1):
            out = out + self.config.learning_rate * forest.predict(x)
            yield i, (np.clip(out, self._lo, self._hi)
                      if self.config.clip_to_training_range else out.copy())


# --------------------------------------------------------------------------- #
# Arm registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ShapeArm:
    """One point of the gen9-A design: a sampler and a pair of loss weights."""

    name: str
    strategy: str
    lambda_delta: float
    lambda_span: float
    axis_set: str = "LHM"
    balance: str = "curve"

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def phase1_arms(lambda_delta: float = 0.5) -> list[ShapeArm]:
    """The brief's §4 screen: which *sampling* strategy moves the flattening.

    One loss weight for all five, so the comparison is about the sampler and
    nothing else.  ``ROW_ONLY`` carries ``lambda_delta = 0`` by construction — it
    is the control, and it is the same learner.
    """
    return [
        ShapeArm("A0_ROW_ONLY", "ROW_ONLY", 0.0, 0.0),
        ShapeArm("A1_ROW_ADJACENT", "ROW_ADJACENT", lambda_delta, 0.0),
        ShapeArm("A2_ROW_ENDPOINT", "ROW_ENDPOINT", lambda_delta, 0.0),
        ShapeArm("A3_ROW_RANDOM_PAIR", "ROW_RANDOM_PAIR", lambda_delta, 0.0),
        ShapeArm("A4_ROW_MULTISCALE", "ROW_MULTISCALE", lambda_delta, 0.0),
    ]


#: The brief's §6 grid, run only for the sampler(s) phase 1 keeps.
LAMBDA_DELTA_GRID: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
LAMBDA_SPAN_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25)

#: An upward extension, added after the phase-1 screen showed the pre-registered
#: grid's top end barely moving the extractant slope.  The reason is mechanical and
#: worth stating: the Huber gradient of the delta term saturates at ``pair_delta``,
#: while the row term's gradient is scaled by a cluster-balanced weight that reaches
#: ~7 for rows in rare ECFP clusters — so at ``lambda_delta <= 1`` the row term
#: outguns the shape term precisely on the chemistry the shape term exists to fix.
#: Every arm drawn from this extension is labelled EXPLORATORY.
LAMBDA_DELTA_EXTENSION: tuple[float, ...] = (2.0, 4.0, 8.0)
