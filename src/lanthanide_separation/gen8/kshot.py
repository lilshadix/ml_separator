"""Few-shot calibration of a frozen global model on a genuinely new ligand.

This is the gen8 engine.  Everything it does happens *after* a global model has
been fitted on a chemotype-blocked training set and has emitted an out-of-fold
prediction for every row of a held-out ligand — so nothing here can leak: the
model never saw the ligand, and the calibration only ever reads the targets of
the rows it is explicitly allowed to "measure".

Three things are separated that gen7 ran together.

**The adaptation mode** — how many ligand-specific degrees of freedom the k
measurements are allowed to move (brief §7).  ``K1`` moves the level only;
``K2`` adds a lanthanide trend; ``K3`` adds response-coefficient adjustments.
Every mode above ``K1`` is ridge-shrunk with an unpenalised intercept, because
gen7 showed an unrestricted affine map at k = 2 is unstable (it scored 3.32
against 0.64 for a plain offset).

**The acquisition policy** — *which* k rows get measured (brief §8).  Policies
see the candidate rows' conditions, the model's prediction and its uncertainty,
and never a candidate's target.  ``ORACLE`` is the exception and is marked
non-deployable everywhere it appears.

**The protocol** — which rows may be selected and which rows are scored.

``P1`` (gen7-compatible)
    every row of the ligand is a candidate; the score is the MAE on all the rows
    *not* selected.  ``RANDOM`` is the mean over candidates, ``ORACLE`` the
    minimum, so the two are exactly paired and the gap between them is the value
    of choosing well.  Comparable to ``runs/gen7_architecture/kshot``.

``P2`` (acquisition-fair)
    the ligand's rows are split once per repeat into a **candidate pool** and a
    disjoint **evaluation set**.  Every policy selects from the identical pool and
    is scored on the identical evaluation rows, which is the only way two policies
    that pick different points can be compared at all (brief §28).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import hashlib

import numpy as np
import pandas as pd

def stable_hash(text: str) -> int:
    """A process-independent 31-bit hash of a string.

    Python's builtin ``hash`` is salted per interpreter process (PYTHONHASHSEED),
    so seeding an RNG with ``hash(ligand)`` makes the draw reproducible *within* a
    run and different *between* runs.  That would quietly break the property this
    whole study rests on: the pool/evaluation split has to be a pure function of
    (seed, repeat, ligand, n_rows) so that an architecture evaluated in a separate
    later process is paired row-for-row with the primary run.  It is not a
    performance detail — with the salted hash, two runs of two different adapters
    are scored on different rows and the "paired" bootstrap between them is not
    paired at all.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(),
                          "big") % (2 ** 31)


#: Condition axes an acquisition policy may look at.  All target-free.
POLICY_AXES: tuple[str, ...] = (
    "massact__log10_cond__acid_concentration_M",
    "massact__log10_cond__extractant_concentration_M",
    "lanthanide_index",
    "cond__temperature_C",
)
#: Design columns the ``K3`` adaptation mode is allowed to move, beyond the intercept.
K3_DESIGN: tuple[str, ...] = (
    "lanthanide_index",
    "massact__log10_cond__acid_concentration_M",
    "massact__log10_cond__extractant_concentration_M",
)
#: ...and ``K2`` moves only the lanthanide trend.
K2_DESIGN: tuple[str, ...] = ("lanthanide_index",)

#: Ridge penalty on the *non-intercept* coefficients of K2/K3, on standardised
#: columns.  Deliberately strong: with k = 2 there is one residual degree of
#: freedom left after the intercept, so an unpenalised slope is pure noise.
DEFAULT_RIDGE = 4.0


# --------------------------------------------------------------------------- #
# Design matrices
# --------------------------------------------------------------------------- #

def _standardise(block_values: np.ndarray) -> np.ndarray:
    """Centre and scale a per-ligand design column using that ligand's own rows.

    Uses features only — never the target — so it is available at inference for a
    ligand whose targets are entirely unknown.  A constant or all-missing column
    collapses to zeros, which makes its coefficient unidentifiable and therefore
    shrunk to zero rather than arbitrary.
    """
    values = np.asarray(block_values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values)
    centre = float(np.nanmean(values[finite]))
    scale = float(np.nanstd(values[finite]))
    out = np.where(finite, values - centre, 0.0)
    return out / scale if scale > 1e-9 else np.zeros_like(values)


def design_matrix(block: pd.DataFrame, mode: str) -> np.ndarray:
    """The per-row design a given adaptation mode may move, intercept first."""
    n = len(block)
    intercept = np.ones((n, 1))
    if mode == "K1":
        return intercept
    columns = K2_DESIGN if mode == "K2" else K3_DESIGN
    parts = [intercept]
    for name in columns:
        values = block[name].to_numpy(dtype=float) if name in block.columns else np.zeros(n)
        parts.append(_standardise(values)[:, None])
    return np.hstack(parts)


def ridge_fit(design: np.ndarray, residual: np.ndarray, *, penalty: float) -> np.ndarray:
    """Ridge with an unpenalised intercept.

    With any ``penalty > 0`` the augmented Gram matrix is non-singular and the
    solve cannot fail: a design column the data cannot identify has its coefficient
    shrunk to zero rather than set to something arbitrary.  That is the whole point
    — gen7 found an *unrestricted* affine recalibration at k = 2 catastrophic
    (3.32 macro MAE against 0.64 for a plain offset), and the diagnosis was that
    with two observations and three free coefficients the fit is not determined.

    ``penalty = 0`` is therefore kept available and is **not** silently repaired: it
    falls back to the minimum-norm least-squares solution, which is exactly what an
    unrestricted fit does on a rank-deficient design.  The degrees-of-freedom
    ablation needs that arm to reproduce gen7's failure rather than crash on it.
    """
    penalties = np.full(design.shape[1], float(penalty))
    penalties[0] = 0.0
    gram = design.T @ design + np.diag(penalties)
    moment = design.T @ residual
    try:
        return np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(design, residual, rcond=None)[0]


# --------------------------------------------------------------------------- #
# Calibration estimators
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Calibration:
    """A fitted adaptation: maps the frozen model's prediction to a calibrated one."""

    mode: str
    coefficients: np.ndarray

    def apply(self, block: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
        design = design_matrix(block, self.mode)
        return prediction + design @ self.coefficients


def fit_calibration(block: pd.DataFrame, prediction: np.ndarray, truth: np.ndarray,
                    selected: np.ndarray, mode: str, *, penalty: float = DEFAULT_RIDGE) -> Calibration:
    """Fit the ligand-specific correction from the selected (measured) rows only."""
    if selected.size == 0:
        width = design_matrix(block.iloc[:1], mode).shape[1]
        return Calibration(mode=mode, coefficients=np.zeros(width))
    design = design_matrix(block, mode)[selected]
    residual = (truth - prediction)[selected]
    return Calibration(mode=mode, coefficients=ridge_fit(design, residual, penalty=penalty))


def apply_no_model(truth: np.ndarray, selected: np.ndarray, n: int) -> np.ndarray:
    """The null that killed gen5's k-shot story: predict the mean of what you measured."""
    return np.full(n, float(np.mean(truth[selected])))


# --------------------------------------------------------------------------- #
# Acquisition policies
# --------------------------------------------------------------------------- #

@dataclass
class PolicyContext:
    """Everything a policy may look at.  Contains no candidate target."""

    block: pd.DataFrame
    prediction: np.ndarray
    pool: np.ndarray                      # positional indices selectable this repeat
    evaluation: np.ndarray                # positional indices scored this repeat
    axes: np.ndarray                      # standardised condition coordinates, all rows
    uncertainty: np.ndarray               # per-row model uncertainty (may be all-NaN)
    disagreement: np.ndarray              # per-row across-model prediction sd
    rng: np.random.Generator
    truth: np.ndarray | None = None       # ORACLE only; None for every deployable policy
    observed: dict = field(default_factory=dict)   # position -> measured value, filled sequentially


Policy = Callable[[PolicyContext, list[int]], int]


def _remaining(context: PolicyContext, selected: Sequence[int]) -> np.ndarray:
    taken = set(selected)
    return np.array([i for i in context.pool if i not in taken], dtype=int)


def policy_random(context: PolicyContext, selected: list[int]) -> int:
    remaining = _remaining(context, selected)
    return int(context.rng.choice(remaining))


def policy_central(context: PolicyContext, selected: list[int]) -> int:
    remaining = _remaining(context, selected)
    centre = context.axes[context.pool].mean(axis=0)
    distance = np.linalg.norm(context.axes[remaining] - centre, axis=1)
    return int(remaining[int(np.argmin(distance))])


def policy_farthest(context: PolicyContext, selected: list[int]) -> int:
    remaining = _remaining(context, selected)
    if not selected:
        centre = context.axes[context.pool].mean(axis=0)
        distance = np.linalg.norm(context.axes[remaining] - centre, axis=1)
        return int(remaining[int(np.argmax(distance))])
    taken = context.axes[np.asarray(selected, dtype=int)]
    distance = np.linalg.norm(context.axes[remaining][:, None, :] - taken[None, :, :], axis=2).min(axis=1)
    return int(remaining[int(np.argmax(distance))])


def policy_coverage(context: PolicyContext, selected: list[int]) -> int:
    """Greedy k-centre: pick the point that most reduces the pool's covering radius."""
    remaining = _remaining(context, selected)
    pool_axes = context.axes[context.pool]
    best, best_radius = int(remaining[0]), np.inf
    for candidate in remaining:
        centres = context.axes[np.asarray(selected + [int(candidate)], dtype=int)]
        radius = np.linalg.norm(pool_axes[:, None, :] - centres[None, :, :], axis=2).min(axis=1).max()
        if radius < best_radius:
            best, best_radius = int(candidate), float(radius)
    return best


def _argmax_finite(remaining: np.ndarray, values: np.ndarray, rng: np.random.Generator) -> int:
    scores = values[remaining]
    if not np.isfinite(scores).any():
        return int(rng.choice(remaining))
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return int(remaining[int(np.argmax(scores))])


def policy_max_sd(context: PolicyContext, selected: list[int]) -> int:
    return _argmax_finite(_remaining(context, selected), context.uncertainty, context.rng)


def policy_min_sd(context: PolicyContext, selected: list[int]) -> int:
    remaining = _remaining(context, selected)
    scores = context.uncertainty[remaining]
    if not np.isfinite(scores).any():
        return int(context.rng.choice(remaining))
    return int(remaining[int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))])


def policy_max_disagreement(context: PolicyContext, selected: list[int]) -> int:
    return _argmax_finite(_remaining(context, selected), context.disagreement, context.rng)


def _gp_posterior_variance(axes: np.ndarray, observed_idx: np.ndarray, query_idx: np.ndarray,
                           *, length_scale: float = 1.0, noise: float = 0.25,
                           amplitude: float = 1.0) -> np.ndarray:
    """Posterior variance of an RBF GP over the *residual field* of one ligand.

    The GP is deliberately low-dimensional and local: it models how the frozen
    model's residual varies over this ligand's condition axes, which is the
    problem gen7's whole-dataset GP was the wrong tool for.
    """
    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return amplitude * np.exp(-0.5 * d2 / (length_scale ** 2))

    prior = np.full(len(query_idx), amplitude)
    if observed_idx.size == 0:
        return prior
    k_oo = kernel(axes[observed_idx], axes[observed_idx]) + noise * np.eye(len(observed_idx))
    k_qo = kernel(axes[query_idx], axes[observed_idx])
    solved = np.linalg.solve(k_oo, k_qo.T)
    return np.clip(prior - np.einsum("ij,ji->i", k_qo, solved), 0.0, None)


def policy_max_predictive_variance(context: PolicyContext, selected: list[int]) -> int:
    remaining = _remaining(context, selected)
    variance = _gp_posterior_variance(context.axes, np.asarray(selected, dtype=int), remaining)
    return int(remaining[int(np.argmax(variance))])


def policy_max_variance_reduction(context: PolicyContext, selected: list[int]) -> int:
    """Pick the point that most reduces the GP's *average* variance over the pool."""
    remaining = _remaining(context, selected)
    pool = np.asarray(context.pool, dtype=int)
    best, best_score = int(remaining[0]), np.inf
    for candidate in remaining:
        observed = np.asarray(selected + [int(candidate)], dtype=int)
        mean_variance = float(_gp_posterior_variance(context.axes, observed, pool).mean())
        if mean_variance < best_score:
            best, best_score = int(candidate), mean_variance
    return best


def policy_d_optimal(context: PolicyContext, selected: list[int]) -> int:
    """Greedy D-optimal on the K3 design — maximise ``det(X'X + eI)``.

    Under ``K1`` this criterion is *degenerate* (every row contributes the same
    single column of ones), which is itself a finding: if offset-only calibration
    is what wins, there is no experimental-design question left to optimise, only
    a shape-error question.  The design is therefore always taken at K3 so the
    policy is non-trivial and can be compared across adaptation modes.
    """
    remaining = _remaining(context, selected)
    design = design_matrix(context.block, "K3")
    best, best_score = int(remaining[0]), -np.inf
    for candidate in remaining:
        rows = design[np.asarray(selected + [int(candidate)], dtype=int)]
        gram = rows.T @ rows + 1e-6 * np.eye(design.shape[1])
        sign, logdet = np.linalg.slogdet(gram)
        score = logdet if sign > 0 else -np.inf
        if score > best_score:
            best, best_score = int(candidate), float(score)
    return best


def make_oracle_policy(mode: str, penalty: float) -> Policy:
    """Exhaustive best next point.  **Not deployable** — reads candidate targets."""
    def policy(context: PolicyContext, selected: list[int]) -> int:
        remaining = _remaining(context, selected)
        truth = context.truth
        assert truth is not None, "oracle policy requires targets"
        best, best_score = int(remaining[0]), np.inf
        for candidate in remaining:
            chosen = np.asarray(selected + [int(candidate)], dtype=int)
            calibration = fit_calibration(context.block, context.prediction, truth, chosen,
                                          mode, penalty=penalty)
            adjusted = calibration.apply(context.block, context.prediction)
            score = float(np.abs(adjusted[context.evaluation] - truth[context.evaluation]).mean())
            if score < best_score:
                best, best_score = int(candidate), score
        return best
    return policy


#: Deployable policies, plus the non-deployable oracle which is added by name.
POLICIES: dict[str, Policy] = {
    "RANDOM": policy_random,
    "CENTRAL": policy_central,
    "FARTHEST_FROM_EXISTING": policy_farthest,
    "MAX_CONDITION_COVERAGE": policy_coverage,
    "MAX_ENSEMBLE_SD": policy_max_sd,
    "MIN_ENSEMBLE_SD": policy_min_sd,
    "MAX_MODEL_DISAGREEMENT": policy_max_disagreement,
    "MAX_PREDICTIVE_VARIANCE": policy_max_predictive_variance,
    "MAX_EXPECTED_VARIANCE_REDUCTION": policy_max_variance_reduction,
    "D_OPTIMAL": policy_d_optimal,
}
NON_DEPLOYABLE: frozenset[str] = frozenset({"ORACLE"})


# --------------------------------------------------------------------------- #
# Policies derived from the calibration-geography measurement
# --------------------------------------------------------------------------- #
# The exhaustive one-shot scan (``scripts/gen8_calibration_geography.py``) found
# that the *stratum* a measurement sits in changes its value by more than any
# uncertainty signal does: on paired within-ligand comparisons a low-acid point is
# 0.244 worse than a mid-acid one (CI [0.175, 0.322], 5/5 seeds), an extreme
# lanthanide 0.095 worse than a central one, and a point in the bottom third of the
# model's own predicted range 0.143 worse than one in the middle.  These three
# policies are the deployable form of that finding; they are declared here rather
# than tuned, and each is a single rule with no free parameter.

def _median_pick(context: PolicyContext, selected: list[int], values: np.ndarray) -> int:
    remaining = _remaining(context, selected)
    pool_values = values[context.pool]
    finite = np.isfinite(pool_values)
    if finite.sum() < 2:
        return int(context.rng.choice(remaining))
    target = float(np.median(pool_values[finite]))
    distance = np.abs(values[remaining] - target)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    if not np.isfinite(distance).any():
        return int(context.rng.choice(remaining))
    return int(remaining[int(np.argmin(distance))])


def policy_median_prediction(context: PolicyContext, selected: list[int]) -> int:
    """Measure where the model predicts a *typical* value for this ligand.

    Needs nothing but the model's own output, which makes it the cheapest policy in
    the study and the only one that can be applied before any conditions are chosen.
    """
    return _median_pick(context, selected, context.prediction)


def policy_mid_acid(context: PolicyContext, selected: list[int]) -> int:
    """Measure at the ligand's median acidity.  The single-variable recommendation."""
    column = context.block["massact__log10_cond__acid_concentration_M"] \
        if "massact__log10_cond__acid_concentration_M" in context.block.columns else None
    if column is None:
        return policy_central(context, selected)
    return _median_pick(context, selected, column.to_numpy(dtype=float))


def policy_medoid(context: PolicyContext, selected: list[int]) -> int:
    """L1 medoid of the candidate pool — the robust form of CENTRAL.

    CENTRAL minimises the Euclidean distance to the pool *mean*, which a single
    outlying condition can drag; the medoid minimises the summed distance to every
    other candidate and cannot be moved by one extreme point.
    """
    remaining = _remaining(context, selected)
    pool_axes = context.axes[context.pool]
    cost = np.abs(context.axes[remaining][:, None, :] - pool_axes[None, :, :]).sum(axis=(1, 2))
    return int(remaining[int(np.argmin(cost))])


def policy_central_then_spread(context: PolicyContext, selected: list[int]) -> int:
    """Anchor the level first, then buy leverage for the slopes.

    The geography result and the adaptation-mode result point in opposite
    directions — a *typical* point is the best single measurement, but fitting a
    slope needs *spread* — so the obvious policy is to do both in order.  This is
    the only composite policy in the study and it is stated before it is measured.
    """
    if not selected:
        return policy_medoid(context, selected)
    return policy_farthest(context, selected)


POLICIES.update({
    "MEDIAN_PREDICTION": policy_median_prediction,
    "MID_ACID": policy_mid_acid,
    "MEDOID": policy_medoid,
    "CENTRAL_THEN_SPREAD": policy_central_then_spread,
})
