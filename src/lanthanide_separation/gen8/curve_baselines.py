"""gen8 §3 — series-aware curve baselines: the strong simple methods, before any net.

Motivation, measured rather than assumed.  On a *held-out chemotype* the frozen
global model does not merely mis-level a new ligand, it **flattens** its response
surface.  Centred within a titration curve, the truth moves 2.58 log units per
decade of extractant concentration while the model's own prediction moves 0.14;
on the acid axis the truth moves 0.94 and the model 0.33.  The within-curve
residual of the frozen model therefore correlates 0.90 with the swept coordinate
on the extractant axis and 0.41 on the acid axis — it is *shape*, not noise, and
it is shape of a kind the mass-action law says should transfer between ligands.

Three baselines are built on that observation, all as :class:`~.adapters.Adapter`
implementations so they are scored on byte-identical rows against offset
correction (:class:`~.adapters.RidgeOffset`, which is baseline A and is *not*
duplicated here).

``B`` — :class:`CurveShape` with ``mode="spline"``
    A natural-cubic-spline shape ``f_a(x)`` per swept axis, fitted **across
    training ligands** on the within-curve-centred target, optionally with a
    per-donor-family delta shrunk toward the pooled fit.  At inference the frozen
    model's own within-curve shape is *replaced* by ``f_a`` while that curve's level
    is left untouched, so the correction is level-preserving by construction and
    the k calibration points are still free to set the level (and, from k = 2, an
    amplitude on the correction).

``C`` — :class:`CurveShape` with ``mode="linear"``
    The same machinery restricted to a straight line in ``log10[L]`` and
    ``log10[H+]`` — i.e. ``log D = a + n log10[L] + m log10[H+] (+ metal term)``,
    the extraction equilibrium itself.  ``n`` and ``m`` are transferable priors
    fitted per donor family, because the families genuinely disagree: a neutral
    solvating extractant gains with acid and an acidic cation exchanger loses.

``D`` — :class:`LocalCurveGP`
    A Gaussian process over the *continuous condition axes only* (log acid, log
    extractant, lanthanide coordinate), fitted to the frozen model's residual for
    the held-out ligand from its k observed points.  Kernel hyperparameters are
    estimated by marginal likelihood on **training ligands' fields**, never on the
    held-out ligand.  A very large constant-kernel term keeps the ligand level
    free, which makes the GP reduce *exactly* to offset correction at k = 1 and
    become local only as k grows — so any difference from baseline A is
    attributable to locality and nothing else.

Leakage.  Everything a baseline needs from training data comes through
``fit_fold``, which sees only that fold's training rows.  Nothing here reads the
frozen model's *training-row* predictions: those exist only as out-of-fold values
produced by models that did see the held-out fold, and using them would let the
held-out ligand's targets reach its own correction through the shape prior.  The
transferable shape is therefore fitted on the **within-curve-centred target**,
a pure training-data quantity, and the frozen model enters only through its
prediction on the held-out ligand, which carries no target at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .kshot import DEFAULT_RIDGE, ridge_fit

# --------------------------------------------------------------------------- #
# Axis bookkeeping
# --------------------------------------------------------------------------- #

#: Short name -> the ``axis`` label used by :mod:`.series` curve membership.
AXIS_SOURCE: dict[str, str] = {
    "ext": "cond__extractant_concentration_M",
    "acid": "cond__acid_concentration_M",
    "metal": "metal",
    "temp": "cond__temperature_C",
}
#: Order in which a row is claimed by an axis.  A row sitting on both an extractant
#: titration and an acid titration is corrected along **one** of them, because the
#: two corrections are not additive (each replaces the model's whole within-curve
#: shape) and summing them would double-count the level.  Extractant first: it is
#: the axis where the frozen model is most badly flattened.
DEFAULT_PRIORITY: tuple[str, ...] = ("ext", "acid")

CURVE_COLUMN = "g8__curve__{axis}"
X_COLUMN = "g8__x__{axis}"
FAMILY_COLUMN = "g8__family"

#: Donor families.  Derived from the DONORS block plus the canonical SMILES, both
#: available for a ligand whose targets are entirely unknown.
FAMILIES: tuple[str, ...] = ("ACIDIC", "NEUTRAL_P", "S_DONOR", "DGA", "OTHER_N_O")
_ACIDIC_PATTERN = r"P\(=O\)\(O\)|OP\(=O\)\(O|C\(=O\)O(?![A-Za-z])"


def donor_family(frame: pd.DataFrame) -> pd.Series:
    """One label per row from ligand structure only — never from conditions or target.

    The families are the ones the brief names: S donors, P donors, diglycolamides,
    acidic versus neutral.  They are ordered so the most mechanistically specific
    label wins: an acidic organophosphorus extractant is ``ACIDIC`` (cation
    exchange, ``log D`` *falls* with acid) rather than ``NEUTRAL_P`` (solvation,
    ``log D`` *rises* with acid), which is the distinction a pooled prior destroys.
    """
    smiles = frame["extractant"].astype(str)
    has_p = smiles.str.contains("P", regex=False)
    acidic = smiles.str.contains(_ACIDIC_PATTERN, regex=True)
    has_s = frame.get("donor__S(donor)", pd.Series(0.0, index=frame.index)).fillna(0) > 0
    ether = frame.get("donor__O(ether)", pd.Series(0.0, index=frame.index)).fillna(0) > 0
    amide = frame.get("donor__O(amide_carbonyl)", pd.Series(0.0, index=frame.index)).fillna(0) >= 2
    label = pd.Series("OTHER_N_O", index=frame.index, dtype=object)
    label[ether & amide] = "DGA"
    label[has_s] = "S_DONOR"
    label[has_p & ~acidic] = "NEUTRAL_P"
    label[acidic] = "ACIDIC"
    return label


def prepare_curve_columns(cohort: pd.DataFrame, membership: pd.DataFrame,
                          axes: Sequence[str] = tuple(AXIS_SOURCE)) -> pd.DataFrame:
    """Attach ``g8__curve__*`` / ``g8__x__*`` / ``g8__family`` to a cohort frame.

    The curve membership is metadata: which rows of one series sweep one variable
    with the others held fixed.  Nothing in it is derived from ``log_D`` (see
    :func:`.series.build_curve_table`), so carrying it into inference is legal for
    a ligand whose targets are unknown — unlike ``curve_table``'s slope and
    ``linear_r2``, which are target-derived and are used **only** for reporting.
    """
    out = cohort.copy()
    out[FAMILY_COLUMN] = donor_family(out).to_numpy()
    for axis in axes:
        source = AXIS_SOURCE[axis]
        sub = membership[membership["axis"] == source]
        # A row can appear once per curve of a given axis at most, but be defensive.
        sub = sub.drop_duplicates("row_id")[["row_id", "curve_id", "axis_value"]]
        merged = out[["row_id"]].merge(sub, on="row_id", how="left")
        out[CURVE_COLUMN.format(axis=axis)] = merged["curve_id"].fillna("").to_numpy()
        out[X_COLUMN.format(axis=axis)] = merged["axis_value"].to_numpy(dtype=float)
    return out


def curve_feature_columns(axes: Sequence[str] = tuple(AXIS_SOURCE)) -> list[str]:
    columns = [FAMILY_COLUMN]
    for axis in axes:
        columns += [CURVE_COLUMN.format(axis=axis), X_COLUMN.format(axis=axis)]
    return columns


# --------------------------------------------------------------------------- #
# Bases
# --------------------------------------------------------------------------- #

def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (ESL 5.2.1), no intercept column.

    Chosen over a smoothing spline because it is a *fixed, transferable* basis:
    the same ``K - 1`` columns are evaluated for a training ligand and for a
    held-out one, so the fitted coefficients mean the same thing on both.  A
    natural spline is linear beyond the boundary knots, which is the behaviour the
    mass-action law wants at the ends of a titration rather than a cubic blow-up.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    n_knots = len(knots)
    if n_knots < 3:
        return x[:, None]

    def d(k: int) -> np.ndarray:
        span = knots[-1] - knots[k]
        if span <= 1e-9:
            return np.zeros_like(x)
        return (np.clip(x - knots[k], 0.0, None) ** 3
                - np.clip(x - knots[-1], 0.0, None) ** 3) / span

    last = d(n_knots - 2)
    return np.column_stack([x] + [d(k) - last for k in range(n_knots - 2)])


def _basis(x: np.ndarray, knots: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "linear" or knots is None or len(knots) < 3:
        return np.asarray(x, dtype=float)[:, None]
    return natural_spline_basis(x, knots)


def _group_centre(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Subtract each group's mean.  ``values`` may be 1-D or 2-D (column-wise)."""
    frame = pd.DataFrame(np.atleast_2d(values.T).T)
    means = frame.groupby(keys, sort=False).transform("mean").to_numpy()
    return (values - means.reshape(values.shape))


def _weighted_ridge(design: np.ndarray, target: np.ndarray, weight: np.ndarray,
                    penalty: float) -> np.ndarray:
    root = np.sqrt(weight)[:, None]
    a = design * root
    b = target * root[:, 0]
    gram = a.T @ a + penalty * np.eye(design.shape[1])
    try:
        return np.linalg.solve(gram, a.T @ b)
    except np.linalg.LinAlgError:            # pragma: no cover - ridge makes this unreachable
        return np.zeros(design.shape[1])


# --------------------------------------------------------------------------- #
# The transferable per-axis shape
# --------------------------------------------------------------------------- #

#: Blend weights the per-axis shrinkage is chosen from.  A coarse grid on purpose:
#: it is selected by leave-one-ligand-out on a few dozen training ligands, and a
#: finer grid would only be fitting the selection noise.
SHRINK_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: Features of the in-fold surrogate whose cross-fitted prediction stands in for the
#: frozen model when the shrinkage is chosen.  Deliberately the *cheap physical*
#: blocks — metal, mass-action log-concentrations, donor counts, bulk physchem —
#: because what has to be reproduced is not the frozen model's accuracy but the way
#: a tree flattens a titration on a chemotype it has never seen.
SURROGATE_FEATURES: tuple[str, ...] = (
    "Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal",
    "massact__log10_cond__acid_concentration_M", "massact__log10_cond__contact_time_min",
    "massact__log10_cond__extractant_concentration_M",
    "massact__log10_cond__metal_concentration_mM", "massact__log10_cond__temperature_C",
    "massact__logL_x_DENTATE", "massact__logL_x_coreCN", "massact__logL_x_logH",
    "donor__O(amide_carbonyl)", "donor__O(ether)", "donor__N(aromatic)", "donor__N(amine)",
    "donor__S(donor)", "donor__O(hydroxyl)", "donor__O(ester_carbonyl)", "donor__O(carbonyl)",
    "donor__n_total", "DENTATE", "coreCN",
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
    "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP",
)
#: ``(split_seed, fold, n_train)`` -> cross-fitted surrogate prediction.  Shared so
#: that every baseline in one run pays for it once and, more importantly, so that
#: every baseline chooses its shrinkage against the *same* stand-in surface.
_SURROGATE_CACHE: dict[tuple, np.ndarray] = {}


def surrogate_oof(train: pd.DataFrame, y: np.ndarray, *, model_seed: int, split_seed: int,
                  fold: int, n_inner: int = 3) -> np.ndarray | None:
    """Chemotype-blocked cross-fitted prediction for the fold's own training rows.

    The shrinkage question is "is the transferable prior's within-curve shape better
    than what a global tree model already predicts for a chemotype it has not seen".
    Answering it needs a *held-out* prediction on training ligands, and the frozen
    model's out-of-fold values cannot supply one: those come from models fitted with
    the held-out fold in their training set, so using them would let the evaluation
    ligand's targets reach its own correction.  An inner chemotype-blocked
    cross-fit inside the fold's training rows is the clean substitute — every
    prediction here comes from a model trained on a strict subset of ``train``.
    """
    key = (int(split_seed), int(fold), len(train))
    cached = _SURROGATE_CACHE.get(key)
    if cached is not None and len(cached) == len(train):
        return cached
    columns = [c for c in SURROGATE_FEATURES if c in train.columns]
    if len(columns) < 8 or "tanimoto_cluster" not in train.columns:
        return None
    from sklearn.ensemble import HistGradientBoostingRegressor

    from ..gen6.cohorts import seeded_group_kfold
    matrix = train[columns].to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    groups = train["tanimoto_cluster"].astype(str).to_numpy()
    out = np.full(len(train), np.nan)
    for index, (inner_train, inner_test) in enumerate(
            seeded_group_kfold(groups, n_inner, int(model_seed))):
        model = HistGradientBoostingRegressor(random_state=int(model_seed) + index,
                                              max_iter=200, early_stopping=False)
        model.fit(matrix[inner_train], y[inner_train])
        out[inner_test] = model.predict(matrix[inner_test])
    _SURROGATE_CACHE[key] = out
    return out


@dataclass
class AxisShape:
    """One axis' transferable shape: knots, a pooled fit, per-family deltas, a weight."""

    axis: str
    mode: str
    knots: np.ndarray | None
    scale: np.ndarray
    beta_pooled: np.ndarray
    beta_family: dict[str, np.ndarray] = field(default_factory=dict)
    shrink: float = 1.0
    n_curves: int = 0
    n_ligands: int = 0

    def evaluate(self, x: np.ndarray, family: np.ndarray | str) -> np.ndarray:
        design = _basis(x, self.knots, self.mode) / self.scale[None, :]
        if isinstance(family, str):
            return design @ self.beta_family.get(family, self.beta_pooled)
        out = np.zeros(len(x))
        for label in np.unique(family):
            mask = family == label
            out[mask] = design[mask] @ self.beta_family.get(str(label), self.beta_pooled)
        return out


def fit_axis_shape(train: pd.DataFrame, y: np.ndarray, axis: str, *, mode: str = "spline",
                   penalty: float = 1.0, family_penalty_factor: float = 6.0,
                   n_knots: int = 5, min_family_ligands: int = 4,
                   min_family_rows: int = 60, use_families: bool = True,
                   shrink: float | None = None,
                   baseline: np.ndarray | None = None) -> AxisShape | None:
    """Fit ``f_axis`` on the within-curve-centred training target.

    Centring within the curve is what makes the fit a *shape* fit and not a level
    fit: every curve's own intercept is projected out, so nothing about how high a
    ligand sits can enter the coefficients, and the estimand is purely "how does
    ``log D`` move when this one variable is swept".  Rows are weighted so that
    each **training ligand** contributes equally (the macro convention this repo
    uses everywhere) — otherwise the two ligands carrying 30-point titrations
    would write the prior for all 152.
    """
    curve_column = CURVE_COLUMN.format(axis=axis)
    x_column = X_COLUMN.format(axis=axis)
    if curve_column not in train.columns:
        return None
    curve = train[curve_column].to_numpy(dtype=object)
    x = train[x_column].to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (curve != "") & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return None
    curve, x, y = curve[mask], x[mask], y[mask]
    ligand = train["extractant"].to_numpy(dtype=object)[mask]
    family = train[FAMILY_COLUMN].to_numpy(dtype=object)[mask]
    stand_in = None
    if baseline is not None:
        stand_in = np.asarray(baseline, dtype=float)[mask]
        if not np.isfinite(stand_in).all():
            stand_in = None

    knots = None
    if mode != "linear":
        quantiles = np.linspace(5.0, 95.0, n_knots)
        knots = np.unique(np.percentile(x, quantiles))
        if len(knots) < 3:
            knots = None
    design = _basis(x, knots, mode)
    design = _group_centre(design, curve)
    target = _group_centre(y, curve)

    counts = pd.Series(ligand).value_counts()
    weight = 1.0 / pd.Series(ligand).map(counts).to_numpy(dtype=float)
    weight = weight / weight.mean()

    scale = np.sqrt(np.maximum((design ** 2 * weight[:, None]).sum(axis=0) / weight.sum(), 1e-12))
    design = design / scale[None, :]

    def solve(mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pooled = _weighted_ridge(design[mask], target[mask], weight[mask], penalty)
        families: dict[str, np.ndarray] = {}
        if use_families:
            residual = target - design @ pooled
            for label in np.unique(family[mask]):
                sub = mask & (family == label)
                if sub.sum() < min_family_rows or len(np.unique(ligand[sub])) < min_family_ligands:
                    continue
                families[str(label)] = pooled + _weighted_ridge(
                    design[sub], residual[sub], weight[sub], penalty * family_penalty_factor)
        return pooled, families

    def apply(pooled: np.ndarray, families: dict, index: np.ndarray) -> np.ndarray:
        out = design[index] @ pooled
        for label, beta in families.items():
            sub = family[index] == label
            if sub.any():
                out[sub] = design[index][sub] @ beta
        return out

    everything = np.ones(len(target), dtype=bool)
    beta_pooled, beta_family = solve(everything)

    # How far to trust the prior, chosen leave-one-training-ligand-out and scored
    # exactly as the correction is deployed: the blend that is graded here is
    # ``(1 - w) * model shape + w * prior shape``, with the fold's cross-fitted
    # surrogate standing in for the model.  Grading against a *flat* curve instead
    # would ask the wrong question — the prior beats flat on every axis, including
    # the two where it is worse than what the model already predicts — and would
    # pick w = 1 everywhere.  One ligand, one vote, because the axis is dominated by
    # a handful of ligands carrying long titrations.
    scores = {value: [] for value in SHRINK_GRID}
    if shrink is None:
        model_shape = (_group_centre(stand_in, curve) if stand_in is not None
                       else np.zeros(len(target)))
        for label in np.unique(ligand):
            held = ligand == label
            if held.sum() < 3 or held.all():
                continue
            pooled, families = solve(~held)
            fitted = apply(pooled, families, held)
            for value in SHRINK_GRID:
                blend = (1.0 - value) * model_shape[held] + value * fitted
                scores[value].append(float(np.abs(target[held] - blend).mean()))
        chosen = (min(SHRINK_GRID, key=lambda value: float(np.mean(scores[value])))
                  if scores[SHRINK_GRID[0]] else 1.0)
    else:
        chosen = float(shrink)
    return AxisShape(axis=axis, mode=mode, knots=knots, scale=scale, beta_pooled=beta_pooled,
                     beta_family=beta_family, shrink=float(chosen),
                     n_curves=int(len(np.unique(curve))), n_ligands=int(len(np.unique(ligand))))


# --------------------------------------------------------------------------- #
# Baselines B and C — one adapter, two bases
# --------------------------------------------------------------------------- #

@dataclass
class CurveShape:
    """Frozen prediction with its within-curve shape replaced by a transferable one.

    For every row claimed by an axis ``a`` (priority order, one axis per row) the
    correction is

    .. code-block:: text

        corr(i) = [ f_a(x_i) - mean_curve f_a ]  -  [ p_i - mean_curve p ]

    — the training-derived shape *minus* the frozen model's own shape.  Both
    brackets are curve-centred, so ``mean corr = 0`` over the rows of the curve that
    *claimed* the row, exactly, and the correction cannot move that curve's level.
    That is the whole point: the level is the k measurements' job, the shape is the
    prior's job, and the two are orthogonal by construction rather than by hope.  (A
    row claimed by its extractant titration is not re-levelled inside its acid
    titration too; the resulting shift between acid curves is the extractant
    stoichiometry showing up on the other axis, which is what it should do.)

    ``adapt``
        ``"none"``  ignore the measurements entirely — this is the **k = 0**
        number, and the driver reports the same value at every k because the
        adapter never reads ``observed``;
        ``"level"``  offset correction on top (identical freedom to ``OFFSET_K1``);
        ``"amplitude"``  offset plus a ridge-shrunk scalar on the correction, which
        needs k >= 2 and degrades to ``"level"`` below that.
    """

    mode: str = "spline"
    adapt: str = "level"
    priority: tuple[str, ...] = DEFAULT_PRIORITY
    use_families: bool = True
    penalty: float = 1.0
    offset_penalty: float = DEFAULT_RIDGE
    n_knots: int = 5
    shrink: float | None = None          # None -> chosen per axis, leave-one-ligand-out
    name: str = ""
    shapes: dict = field(default_factory=dict, repr=False)
    fit_log: list = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.name:
            stem = "CURVE_SPLINE" if self.mode == "spline" else "MASSACTION"
            if not self.use_families:
                stem += "_POOLED"
            if self.shrink is not None:
                stem += f"_W{self.shrink:g}".replace(".", "")
            if self.priority != DEFAULT_PRIORITY:
                stem += "_" + "".join(a[0].upper() for a in self.priority)
            suffix = {"none": "_SHAPEONLY", "level": "_K1", "amplitude": "_AMP"}[self.adapt]
            self.name = stem + suffix

    # -- training ---------------------------------------------------------- #
    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *, split_seed: int,
                 fold: int, model_seed: int) -> None:
        self.shapes = {}
        baseline = (None if self.shrink is not None else
                    surrogate_oof(train, y_train, model_seed=model_seed, split_seed=split_seed,
                                  fold=fold))
        for axis in self.priority:
            shape = fit_axis_shape(train, y_train, axis, mode=self.mode, penalty=self.penalty,
                                   n_knots=self.n_knots, use_families=self.use_families,
                                   shrink=self.shrink, baseline=baseline)
            if shape is not None:
                self.shapes[axis] = shape
                self.fit_log.append({"adapter": self.name, "split_seed": int(split_seed),
                                     "fold": int(fold), "axis": axis, "shrink": shape.shrink,
                                     "n_curves": shape.n_curves, "n_ligands": shape.n_ligands,
                                     "families": ",".join(sorted(shape.beta_family))})
        self._cache = {}

    # -- inference --------------------------------------------------------- #
    def _correction(self, block: pd.DataFrame, prediction: np.ndarray,
                    context) -> np.ndarray:
        key = (context.split_seed, context.fold, context.extractant, len(block))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        n = len(block)
        correction = np.zeros(n)
        claimed = np.zeros(n, dtype=bool)
        family = block[FAMILY_COLUMN].to_numpy(dtype=object) if FAMILY_COLUMN in block.columns \
            else np.array(["OTHER_N_O"] * n, dtype=object)
        for axis in self.priority:
            shape = self.shapes.get(axis)
            if shape is None:
                continue
            curve_column, x_column = CURVE_COLUMN.format(axis=axis), X_COLUMN.format(axis=axis)
            if curve_column not in block.columns:
                continue
            curve = block[curve_column].to_numpy(dtype=object)
            x = block[x_column].to_numpy(dtype=float)
            on_axis = (curve != "") & np.isfinite(x)
            if not on_axis.any():
                continue
            target = np.zeros(n)
            target[on_axis] = shape.evaluate(x[on_axis], family[on_axis])
            # Centre both the prior shape and the model's shape over *all* rows of
            # the curve present in the block, including rows already claimed by a
            # higher-priority axis: the curve mean is a property of the curve, not
            # of which rows this adapter happens to be allowed to move.
            keys = curve[on_axis]
            centred_target = _group_centre(target[on_axis], keys)
            centred_model = _group_centre(prediction[on_axis], keys)
            delta = shape.shrink * (centred_target - centred_model)
            take = on_axis & ~claimed
            correction[take] = delta[(~claimed)[on_axis]]
            claimed |= on_axis
        self._cache[key] = correction
        return correction

    def predict(self, block, prediction, selected, observed, context):
        prediction = np.asarray(prediction, dtype=float)
        try:
            correction = self._correction(block, prediction, context)
        except Exception:                     # pragma: no cover - contract: never fail loudly
            return prediction
        base = prediction + correction
        if self.adapt == "none" or len(selected) == 0:
            return base
        residual = np.asarray(observed, dtype=float) - base[selected]
        design = np.ones((len(block), 1))
        if self.adapt == "amplitude" and len(selected) >= 2:
            spread = float(np.std(correction))
            if spread > 1e-6 and float(np.std(correction[selected])) > 1e-6:
                column = (correction - float(np.mean(correction))) / spread
                design = np.hstack([design, column[:, None]])
        beta = ridge_fit(design[selected], residual, penalty=self.offset_penalty)
        out = base + design @ beta
        return np.where(np.isfinite(out), out, prediction)


# --------------------------------------------------------------------------- #
# Baseline D — a GP over the condition axes only
# --------------------------------------------------------------------------- #

#: Continuous condition axes the GP lives on.  Three dimensions, all physical.
GP_AXES: tuple[str, ...] = (
    "massact__log10_cond__acid_concentration_M",
    "massact__log10_cond__extractant_concentration_M",
    "lanthanide_index",
)
#: Variance of the constant kernel term that carries the *ligand level*.  Large on
#: purpose: the level is exactly what the k measurements exist to determine, so it
#: is given an effectively flat prior.  The consequence is that at k = 1 the GP
#: posterior mean is the measured residual everywhere — offset correction, to four
#: decimal places — and every difference from baseline A at k >= 2 is locality.
LEVEL_VARIANCE = 1.0e4


def _rbf(a: np.ndarray, b: np.ndarray, length_scale) -> np.ndarray:
    """RBF with one length scale per axis (ARD).

    Anisotropy is not a refinement here, it is the point: a lanthanide series steps
    through fifteen integers while an acid titration covers three decades, and one
    shared length scale would either smooth the metal trend into a constant or make
    the acid axis behave like a lookup.  A scalar is broadcast, so an isotropic call
    still works.
    """
    scales = np.atleast_1d(np.asarray(length_scale, dtype=float))
    if scales.size == 1:
        scales = np.repeat(scales, a.shape[1])
    diff = (a[:, None, :] - b[None, :, :]) / scales[None, None, :]
    return np.exp(-0.5 * (diff ** 2).sum(axis=2))


@dataclass
class LocalCurveGP:
    """GP on the frozen model's residual over the continuous condition axes.

    Hyperparameters come from **training ligands** by marginal likelihood.  The
    field they are estimated on is each training ligand's own ``log_D`` centred by
    its ligand mean — a pure training-data quantity.  That is the right proxy for
    the residual field the GP will actually see, because the frozen model's
    within-ligand prediction moves very little (its within-curve standard deviation
    is 0.07 on the extractant axis and 0.29 on the acid axis against 0.89 and 0.96
    for the truth), so ``y - p`` and ``y - const`` have nearly the same covariance
    structure.  Estimating them from the frozen model's *training-row* residuals
    instead would require predictions from models that saw the held-out fold.
    """

    name: str = "CURVE_GP"
    axes: tuple[str, ...] = GP_AXES
    length_scales: tuple[float, ...] = (0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0)
    amplitudes: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    noises: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    max_rows_per_ligand: int = 60
    base: object | None = None            # optional CurveShape applied first
    centre: np.ndarray = field(default_factory=lambda: np.zeros(len(GP_AXES)), repr=False)
    scale: np.ndarray = field(default_factory=lambda: np.ones(len(GP_AXES)), repr=False)
    length_scale: object = 1.0
    amplitude: float = 1.0
    noise: float = 0.25
    fit_log: list = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    # -- training ---------------------------------------------------------- #
    def _standardise(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.axes:
            values = (frame[name].to_numpy(dtype=float) if name in frame.columns
                      else np.zeros(len(frame)))
            columns.append(values)
        matrix = np.vstack(columns).T
        matrix = np.where(np.isfinite(matrix), matrix, self.centre[None, :])
        return (matrix - self.centre[None, :]) / self.scale[None, :]

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *, split_seed: int,
                 fold: int, model_seed: int) -> None:
        raw = np.vstack([(train[name].to_numpy(dtype=float) if name in train.columns
                          else np.zeros(len(train))) for name in self.axes]).T
        with np.errstate(invalid="ignore"):
            self.centre = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        self.centre = np.where(np.isfinite(self.centre), self.centre, 0.0)
        spread = np.nanstd(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        self.scale = np.where(np.isfinite(spread) & (spread > 1e-6), spread, 1.0)

        rng = np.random.default_rng(model_seed)
        axes = self._standardise(train)
        y = np.asarray(y_train, dtype=float)
        ligand = train["extractant"].to_numpy(dtype=object)
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for _, index in sorted(pd.Series(ligand).groupby(ligand, sort=True).indices.items()):
            index = np.asarray(index, dtype=int)
            if len(index) < 4:
                continue
            if len(index) > self.max_rows_per_ligand:
                index = np.sort(rng.choice(index, size=self.max_rows_per_ligand, replace=False))
            target = y[index]
            if not np.isfinite(target).all():
                continue
            blocks.append((axes[index], target - target.mean()))
        if not blocks:
            return

        def negative_log_likelihood(scales, amplitude: float, noise: float) -> float:
            total = 0.0
            for coordinates, target in blocks:
                gram = (amplitude ** 2 * _rbf(coordinates, coordinates, scales)
                        + (noise ** 2) * np.eye(len(target)))
                try:
                    factor = np.linalg.cholesky(gram)
                except np.linalg.LinAlgError:
                    return np.inf
                alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, target))
                total += 0.5 * float(target @ alpha) + float(np.log(np.diag(factor)).sum())
            return total

        # Stage 1: isotropic grid, so the anisotropic search starts somewhere sane.
        best, best_nll = (self.length_scale, self.amplitude, self.noise), np.inf
        for length_scale in self.length_scales:
            for amplitude in self.amplitudes:
                for noise in self.noises:
                    value = negative_log_likelihood(length_scale, amplitude, noise)
                    if value < best_nll:
                        best_nll, best = value, (length_scale, amplitude, noise)
        scales = np.repeat(float(best[0]), len(self.axes))
        amplitude, noise = float(best[1]), float(best[2])
        # Stage 2: coordinate descent on the per-axis scales, then on amplitude and
        # noise.  Two sweeps: a third never moved the optimum in practice and every
        # extra sweep is another chance to chase marginal-likelihood noise.
        for _ in range(2):
            for axis in range(len(scales)):
                for candidate in self.length_scales:
                    trial = scales.copy()
                    trial[axis] = candidate
                    value = negative_log_likelihood(trial, amplitude, noise)
                    if value < best_nll:
                        best_nll, scales = value, trial
            for candidate_amplitude in self.amplitudes:
                for candidate_noise in self.noises:
                    value = negative_log_likelihood(scales, candidate_amplitude, candidate_noise)
                    if value < best_nll:
                        best_nll, amplitude, noise = value, candidate_amplitude, candidate_noise
        self.length_scale, self.amplitude, self.noise = scales, amplitude, noise
        self.fit_log.append({"adapter": self.name, "split_seed": int(split_seed),
                             "fold": int(fold), "n_ligands": len(blocks),
                             "length_scale": ",".join(f"{v:g}" for v in np.atleast_1d(scales)),
                             "amplitude": float(amplitude), "noise": float(noise),
                             "nll": float(best_nll)})
        self._cache = {}

    # -- inference --------------------------------------------------------- #
    def _axes_for(self, block: pd.DataFrame, context) -> np.ndarray:
        key = (context.split_seed, context.fold, context.extractant, len(block))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._standardise(block)
            self._cache[key] = cached
        return cached

    def predict(self, block, prediction, selected, observed, context):
        prediction = np.asarray(prediction, dtype=float)
        base = prediction
        if self.base is not None:
            try:
                base = prediction + self.base._correction(block, prediction, context)
            except Exception:                 # pragma: no cover
                base = prediction
        if len(selected) == 0:
            return base
        try:
            axes = self._axes_for(block, context)
            observed_axes = axes[selected]
            residual = np.asarray(observed, dtype=float) - base[selected]
            k_oo = (LEVEL_VARIANCE + self.amplitude ** 2 * _rbf(observed_axes, observed_axes,
                                                                self.length_scale)
                    + (self.noise ** 2) * np.eye(len(selected)))
            k_qo = LEVEL_VARIANCE + self.amplitude ** 2 * _rbf(axes, observed_axes,
                                                               self.length_scale)
            out = base + k_qo @ np.linalg.solve(k_oo, residual)
        except Exception:                     # pragma: no cover
            return base
        return np.where(np.isfinite(out), out, base)


# --------------------------------------------------------------------------- #
# Mass-action prior reporting (baseline C's chemistry claim)
# --------------------------------------------------------------------------- #

def massaction_priors(frame: pd.DataFrame, y: np.ndarray, axis: str, *,
                      n_boot: int = 2000, seed: int = 20260820) -> pd.DataFrame:
    """Per-family stoichiometric slope, bootstrapped over **ligands**.

    The unit of resampling is the ligand, not the curve and not the row: two acid
    titrations of the same extractant are not independent evidence about that
    family's ``m``, and a per-row bootstrap would report a confidence interval an
    order of magnitude too tight.  ``pooled`` is the same estimator with the family
    label ignored, so the two rows of the table are directly comparable and the
    question "do the families differ" has an answer rather than an impression.
    """
    curve_column, x_column = CURVE_COLUMN.format(axis=axis), X_COLUMN.format(axis=axis)
    curve = frame[curve_column].to_numpy(dtype=object)
    x = frame[x_column].to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (curve != "") & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return pd.DataFrame()
    work = pd.DataFrame({"curve": curve[mask], "x": x[mask], "y": y[mask],
                         "ligand": frame["extractant"].to_numpy(dtype=object)[mask],
                         "family": frame[FAMILY_COLUMN].to_numpy(dtype=object)[mask]})
    work["dx"] = work["x"] - work.groupby("curve")["x"].transform("mean")
    work["dy"] = work["y"] - work.groupby("curve")["y"].transform("mean")

    def slope(sub: pd.DataFrame) -> float:
        denominator = float((sub["dx"] ** 2).sum())
        return float((sub["dx"] * sub["dy"]).sum() / denominator) if denominator > 1e-9 else np.nan

    per_ligand = work.groupby(["ligand", "family"], sort=True).apply(
        slope, include_groups=False).rename("slope").reset_index().dropna(subset=["slope"])
    rng = np.random.default_rng(seed)
    curves_per_family = work.groupby("family")["curve"].nunique()
    records = []
    for label, sub in list(per_ligand.groupby("family")) + [("pooled", per_ligand)]:
        values = sub["slope"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        draws = np.median(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
        records.append({"axis": axis, "family": label, "n_ligands": int(len(values)),
                        "n_curves": int(curves_per_family.get(label, work["curve"].nunique())),
                        "median_slope": float(np.median(values)),
                        "ci_lo": float(np.percentile(draws, 2.5)),
                        "ci_hi": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(records)


def family_difference_test(frame: pd.DataFrame, y: np.ndarray, axis: str, *,
                           n_permutations: int = 5000, seed: int = 20260820) -> dict:
    """Permutation test on the spread of per-family median slopes, ligands as units."""
    curve_column, x_column = CURVE_COLUMN.format(axis=axis), X_COLUMN.format(axis=axis)
    curve = frame[curve_column].to_numpy(dtype=object)
    x = frame[x_column].to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (curve != "") & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return {}
    work = pd.DataFrame({"curve": curve[mask], "x": x[mask], "y": y[mask],
                         "ligand": frame["extractant"].to_numpy(dtype=object)[mask],
                         "family": frame[FAMILY_COLUMN].to_numpy(dtype=object)[mask]})
    work["dx"] = work["x"] - work.groupby("curve")["x"].transform("mean")
    work["dy"] = work["y"] - work.groupby("curve")["y"].transform("mean")
    per_ligand = work.groupby(["ligand", "family"], sort=True).apply(
        lambda s: float((s["dx"] * s["dy"]).sum() / max(float((s["dx"] ** 2).sum()), 1e-9)),
        include_groups=False).rename("slope").reset_index().dropna(subset=["slope"])
    keep = per_ligand.groupby("family")["slope"].transform("size") >= 3
    per_ligand = per_ligand[keep]
    if per_ligand["family"].nunique() < 2:
        return {}
    labels = per_ligand["family"].to_numpy(dtype=object)
    values = per_ligand["slope"].to_numpy(dtype=float)

    def statistic(assignment: np.ndarray) -> float:
        medians = [np.median(values[assignment == label]) for label in np.unique(assignment)]
        return float(np.ptp(medians))

    observed = statistic(labels)
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(labels)) for _ in range(n_permutations)])
    return {"axis": axis, "statistic": observed, "n_ligands": int(len(values)),
            "n_families": int(len(np.unique(labels))),
            "p_value": float((1 + (null >= observed).sum()) / (1 + n_permutations))}


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def build_curve_adapters(*, penalty: float = 1.0, offset_penalty: float = DEFAULT_RIDGE,
                         n_knots: int = 5, with_metal: bool = True) -> list:
    """Every §3 baseline, in the order the report reads them.

    ``*_SHAPEONLY`` variants never read ``observed`` at all, so the number the
    driver prints for them at k = 1 *is* their k = 0 number — which is the only way
    to see a zero-measurement gain inside a harness whose loop starts at k = 1.
    """
    spline_shape_only = CurveShape(mode="spline", adapt="none", penalty=penalty, n_knots=n_knots)
    spline_level = CurveShape(mode="spline", adapt="level", penalty=penalty,
                              offset_penalty=offset_penalty, n_knots=n_knots)
    spline_amplitude = CurveShape(mode="spline", adapt="amplitude", penalty=penalty,
                                  offset_penalty=offset_penalty, n_knots=n_knots)
    spline_unshrunk = CurveShape(mode="spline", adapt="level", penalty=penalty,
                                 offset_penalty=offset_penalty, n_knots=n_knots, shrink=1.0)
    linear_shape_only = CurveShape(mode="linear", adapt="none", penalty=penalty)
    linear_level = CurveShape(mode="linear", adapt="level", penalty=penalty,
                              offset_penalty=offset_penalty)
    linear_pooled = CurveShape(mode="linear", adapt="level", penalty=penalty,
                               offset_penalty=offset_penalty, use_families=False)
    adapters = [spline_shape_only, spline_level, spline_amplitude, spline_unshrunk,
                linear_shape_only, linear_level, linear_pooled]
    if with_metal:
        adapters.append(CurveShape(mode="linear", adapt="level", penalty=penalty,
                                   offset_penalty=offset_penalty,
                                   priority=("ext", "acid", "metal"),
                                   name="MASSACTION_WITH_METAL_K1"))
    gp = LocalCurveGP()
    combined = LocalCurveGP(name="CURVE_SPLINE_GP", base=spline_level)
    adapters += [gp, combined]
    return adapters
