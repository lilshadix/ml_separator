"""Learning which experiment to run first.

gen8 closed the acquisition question analytically and then left it open
practically.  Under one-shot offset calibration the best candidate is exactly

.. code-block:: text

    argmin_i | r_i - median(r) |        with  r_i = y_i - yhat_i

verified to attain the realised oracle to 6.7e-16 over 715 blocks.  The catch is
that ``r_i`` is a *measurement*: it is the thing you learn by running the
experiment, so the rule is not a policy, it is a description of hindsight.

gen9 asks whether the deviation ``q_i = |r_i - median(r)|`` can be *predicted*
from what is knowable before anyone picks up a pipette.  Note what this is not:
it is **not** uncertainty estimation.  gen8 showed why — a perfectly calibrated
uncertainty approximates ``argmin |r_i|``, which scores 0.780 against random's
0.709, i.e. worse than choosing blindly.  The quantity that matters is distance
from the *median* residual, not distance from zero, and those two rank candidates
almost orthogonally (Spearman 0.23).

Everything here obeys one rule, enforced by :data:`FEATURE_PROVENANCE` and tested:
**a deployable feature may read the candidate pool's conditions and the frozen
model's predictions, and nothing else.**  No truth, no residual, no measured span,
no statistic computed from a held-out target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen8.kshot import POLICY_AXES, stable_hash
from ..gen8.protocols import _standardised_axes, make_p2_split

#: One entry per emitted feature.  ``uses_target`` must be False for every column
#: that reaches a deployable policy; the test suite reads this table rather than
#: trusting the docstring.
FEATURE_PROVENANCE: dict[str, dict] = {}


def _register(name: str, *, scope: str, uses_target: bool = False,
              available_pre_measurement: bool = True, note: str = "") -> str:
    FEATURE_PROVENANCE[name] = {
        "feature_name": name, "scope": scope, "uses_target": uses_target,
        "available_pre_measurement": available_pre_measurement, "note": note}
    return name


AXIS_NAMES = ("acid", "extractant", "lanthanide", "temperature")

for _axis in AXIS_NAMES:
    _register(f"pos_{_axis}", scope="pool", note="rank percentile of the candidate on this axis within the pool")
    _register(f"centrality_{_axis}", scope="pool", note="|percentile - 0.5|")
    _register(f"z_{_axis}", scope="pool", note="standardised axis coordinate")
    _register(f"boundary_{_axis}", scope="pool", note="distance to the nearer end of the pool's range, in pool units")
_register("dist_medoid_l1", scope="pool", note="L1 cost to every pool point — gen8's MEDOID objective, in pool-standardised coordinates")
_register("dist_medoid_rank", scope="pool")
_register("dist_centre_l2", scope="pool", note="Euclidean distance to the pool mean — gen8's CENTRAL objective, in pool-standardised coordinates")
_register("dist_centre_rank", scope="pool")
_register("nn1_pool", scope="pool", note="distance to the nearest other pool candidate")
_register("nn3_pool", scope="pool", note="mean distance to the three nearest pool candidates")
_register("gp_design_var", scope="pool", note="RBF leave-one-out design variance over the pool axes; gen8's best single signal")
_register("pred", scope="candidate", note="the frozen model's prediction for this candidate")
_register("pred_pos", scope="pool", note="rank percentile of the prediction within the pool")
_register("pred_centrality", scope="pool")
_register("pred_dev_from_median", scope="pool", note="|pred - median(pred over pool)| — the MEDIAN_PREDICTION objective")
_register("pred_spread_pool", scope="pool", note="IQR of the pool's predictions")
_register("pred_curve_slope", scope="pool", note="slope of the model's own predictions along the candidate's curve")
_register("pred_curve_span", scope="pool", note="range of the model's own predictions along the candidate's curve")
_register("pred_curve_pos", scope="pool", note="where on its own curve the candidate sits, 0..1")
_register("pred_local_gradient", scope="pool", note="finite-difference gradient of the prediction at the candidate along its curve")
_register("n_curves_through", scope="pool", note="how many reconstructed curves the candidate lies on")
_register("n_pool_same_curve", scope="pool")
_register("n_pool", scope="pool")
_register("n_rows", scope="pool")
_register("log_n_pool", scope="pool")
for _t in ("acid", "extractant", "metal_series", "temperature", "contact_time", "metal_concentration", "none"):
    _register(f"curve_is_{_t}", scope="pool", note="curve type of the candidate's primary curve")

#: **Why there is no molecular-descriptor group.**  The brief asks for ligand
#: descriptors as an explicit ablation, expecting them to add little.  They cannot add
#: anything here, and the reason is structural rather than empirical: a molecular
#: descriptor is *constant across a ligand's candidates*.  The deployed rule ranks
#: candidates **within one pool**, so a per-ligand constant has zero variance in every
#: block it appears in and cannot change an ordering at all — a linear ranker's
#: coefficient on it is unidentifiable, and a tree can only use it to select which
#: *other* feature's threshold applies.  Running the ablation would measure the value
#: of those interactions, not of the chemistry.  It is recorded as not run, with this
#: argument, rather than as a null result.
#:
#: Feature groups the §15 ablation switches on and off.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "geometry": tuple(n for n in FEATURE_PROVENANCE
                      if n.startswith(("pos_", "centrality_", "z_", "boundary_", "dist_", "nn"))
                      or n in ("gp_design_var", "n_pool", "n_rows", "log_n_pool")),
    "prediction": ("pred", "pred_pos", "pred_centrality", "pred_dev_from_median",
                   "pred_spread_pool", "pred_local_gradient"),
    "curve": ("pred_curve_slope", "pred_curve_span", "pred_curve_pos",
              "n_curves_through", "n_pool_same_curve") + tuple(
        n for n in FEATURE_PROVENANCE if n.startswith("curve_is_")),
}
FEATURE_GROUPS["geometry+prediction"] = FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["prediction"]
FEATURE_GROUPS["full"] = (FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["prediction"]
                          + FEATURE_GROUPS["curve"])
#: The §15 control.  Same columns as ``full``; the *runner* shuffles the curve block
#: across candidates before fitting, breaking the correspondence between a candidate
#: and the response-surface context it sits in while leaving every marginal
#: distribution untouched.  If this scores like ``full``, the context was decoration.
FEATURE_GROUPS["full_shuffled"] = FEATURE_GROUPS["full"]

CURVE_TYPES = ("acid", "extractant", "metal_series", "temperature",
               "contact_time", "metal_concentration", "none")


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #

def _rank_percentile(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(len(values), 0.5)
    if finite.sum() < 2:
        return out
    out[finite] = pd.Series(values[finite]).rank(pct=True, method="average").to_numpy()
    return out


def _gp_design_variance(axes: np.ndarray, jitter: float = 1e-3) -> np.ndarray:
    """Leave-one-out RBF posterior variance over the pool's own design.

    gen8 found this the one 'uncertainty' signal whose minimum beats random, and
    diagnosed it as geometric centrality in an uncertainty costume.  gen9 keeps it
    as a *feature* for exactly that reason: it is a compact summary of how well a
    candidate is surrounded by its own pool, and it reads no target at all.
    """
    n = len(axes)
    if n < 3:
        return np.full(n, np.nan)
    distance = np.linalg.norm(axes[:, None, :] - axes[None, :, :], axis=2)
    upper = distance[np.triu_indices(n, 1)]
    scale = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    kernel = np.exp(-0.5 * (distance / max(scale, 1e-6)) ** 2) + jitter * np.eye(n)
    try:
        inverse = np.linalg.inv(kernel)
    except np.linalg.LinAlgError:                        # pragma: no cover
        return np.full(n, np.nan)
    diagonal = np.clip(np.diag(inverse), 1e-12, None)
    return 1.0 / diagonal - jitter


def _curve_context(block: pd.DataFrame, pool: np.ndarray, prediction: np.ndarray,
                   membership: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """Model-predicted shape around each candidate, from geometry plus predictions.

    This is where gen9-A and gen9-B meet: once the global model draws curves with
    the right shape, *its own* predicted slope and span become informative about
    where a curve's typical point is.  Every quantity is a function of the model's
    predictions and the condition columns — never of ``log_D``.
    """
    n = len(block)
    out = {
        "pred_curve_slope": np.zeros(n), "pred_curve_span": np.zeros(n),
        "pred_curve_pos": np.full(n, 0.5), "pred_local_gradient": np.zeros(n),
        "n_curves_through": np.zeros(n), "n_pool_same_curve": np.zeros(n),
        "curve_type": np.array(["none"] * n, dtype=object),
    }
    if membership is None or membership.empty:
        return out
    row_ids = block["row_id"].astype(str).to_numpy()
    position = {r: i for i, r in enumerate(row_ids)}
    present = membership[membership["row_id"].astype(str).isin(position)]
    if present.empty:
        return out
    in_pool = np.zeros(n, dtype=bool)
    in_pool[pool] = True
    # A row can sit on several curves.  It is described by its *longest* one,
    # because that is the curve whose slope is most identifiable; ties are broken
    # by ``curve_id`` order, which the sorted groupby makes deterministic.
    best_points = np.zeros(n)
    for _, sub in present.groupby("curve_id", sort=True):
        idx = np.array([position[str(r)] for r in sub["row_id"]], dtype=int)
        x = sub["axis_value"].to_numpy(dtype=float)
        ok = np.isfinite(x)
        idx, x = idx[ok], x[ok]
        if len(idx) < 3 or np.ptp(x) <= 0:
            continue
        order = np.argsort(x, kind="stable")
        idx, x = idx[order], x[order]
        y = prediction[idx]
        slope = float(np.polyfit(x, y, 1)[0])
        span = float(np.ptp(y))
        pool_here = float(in_pool[idx].sum())
        label = str(sub["axis_label"].iloc[0])
        rank = (x - x.min()) / max(np.ptp(x), 1e-12)
        gradient = np.gradient(y, x) if len(x) > 2 else np.full(len(x), slope)

        out["n_curves_through"][idx] += 1
        wins = len(idx) > best_points[idx]
        best_points[idx[wins]] = len(idx)
        for j, row in enumerate(idx):
            if not wins[j]:
                continue
            out["pred_curve_slope"][row] = slope
            out["pred_curve_span"][row] = span
            out["n_pool_same_curve"][row] = pool_here
            out["curve_type"][row] = label
            out["pred_curve_pos"][row] = float(rank[j])
            out["pred_local_gradient"][row] = float(gradient[j])
    return out


def _pool_standardised_axes(block: pd.DataFrame, pool: np.ndarray) -> np.ndarray:
    """Condition coordinates z-scored **within the candidate pool**.

    gen8's policies standardise over the ligand's whole row block, which is
    target-free and legitimate.  gen9's learned features do not, for a different
    reason: a feature whose value moves when a row *outside* the candidate list
    changes is not reproducible at deployment, where the candidate list is
    literally what the user supplies and the rest of the surface may not exist yet.
    Making the scope the pool is what lets :data:`FEATURE_PROVENANCE` say
    ``scope="pool"`` and mean it.
    """
    columns = []
    for name in POLICY_AXES:
        values = (block[name].to_numpy(dtype=float) if name in block.columns
                  else np.zeros(len(block)))[pool]
        finite = np.isfinite(values)
        if finite.sum() >= 2:
            centre, scale = float(values[finite].mean()), float(values[finite].std())
            column = np.where(finite, values - centre, 0.0)
            column = column / scale if scale > 1e-9 else np.zeros_like(column)
        else:
            column = np.zeros_like(values)
        columns.append(column)
    return np.vstack(columns).T


def candidate_features(block: pd.DataFrame, prediction: np.ndarray, pool: np.ndarray,
                       *, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per *pool* candidate, features only, no target anywhere.

    Every normalisation is against the pool, because that is what an
    experimentalist has in hand: the list of conditions they are choosing between.
    A feature that moved when a non-candidate row moved would be unreproducible at
    deployment, and ``test_changing_rows_outside_the_pool_cannot_move_a_pool_feature``
    is what stops one being added by accident.
    """
    pool = np.asarray(pool, dtype=int)
    axes = _pool_standardised_axes(block, pool)
    predictions = np.asarray(prediction, dtype=float)[pool]
    n = len(pool)
    features: dict[str, np.ndarray] = {}

    for j, name in enumerate(AXIS_NAMES):
        raw = axes[:, j]
        percentile = _rank_percentile(raw)
        features[f"pos_{name}"] = percentile
        features[f"centrality_{name}"] = np.abs(percentile - 0.5)
        features[f"z_{name}"] = raw
        lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
        width = max(hi - lo, 1e-9)
        features[f"boundary_{name}"] = np.minimum(raw - lo, hi - raw) / width

    l1 = np.abs(axes[:, None, :] - axes[None, :, :]).sum(axis=(1, 2))
    features["dist_medoid_l1"] = l1
    features["dist_medoid_rank"] = _rank_percentile(l1)
    centre = axes.mean(axis=0)
    l2 = np.linalg.norm(axes - centre[None, :], axis=1)
    features["dist_centre_l2"] = l2
    features["dist_centre_rank"] = _rank_percentile(l2)

    distance = np.linalg.norm(axes[:, None, :] - axes[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    ordered = np.sort(distance, axis=1)
    features["nn1_pool"] = ordered[:, 0] if n > 1 else np.zeros(n)
    features["nn3_pool"] = ordered[:, :3].mean(axis=1) if n > 3 else features["nn1_pool"]
    features["gp_design_var"] = _gp_design_variance(axes)

    features["pred"] = predictions
    percentile = _rank_percentile(predictions)
    features["pred_pos"] = percentile
    features["pred_centrality"] = np.abs(percentile - 0.5)
    features["pred_dev_from_median"] = np.abs(predictions - float(np.nanmedian(predictions)))
    spread = float(np.nanquantile(predictions, 0.75) - np.nanquantile(predictions, 0.25)) if n > 3 else 0.0
    features["pred_spread_pool"] = np.full(n, spread)

    curve = _curve_context(block, pool, np.asarray(prediction, dtype=float), membership)
    for name in ("pred_curve_slope", "pred_curve_span", "pred_curve_pos",
                 "pred_local_gradient", "n_curves_through", "n_pool_same_curve"):
        features[name] = curve[name][pool]
    types = curve["curve_type"][pool]
    for label in CURVE_TYPES:
        features[f"curve_is_{label}"] = (types == label).astype(float)

    features["n_pool"] = np.full(n, float(n))
    features["log_n_pool"] = np.full(n, float(np.log1p(n)))
    features["n_rows"] = np.full(n, float(len(block)))

    frame = pd.DataFrame(features)
    frame.insert(0, "candidate", pool.astype(int))
    return frame


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def oracle_labels(truth: np.ndarray, prediction: np.ndarray, pool: np.ndarray,
                  evaluation: np.ndarray) -> pd.DataFrame:
    """The supervised target, and the realised score it is meant to stand in for.

    ``q`` is the brief's ``|r_i - median(r)|`` with the median taken over the
    candidate *and* evaluation rows — the population the calibration will be judged
    on.  ``realised_mae`` is what actually happens if candidate ``i`` is measured
    and a constant offset applied: ``mean_e |r_e - r_i|``.  Both are computed here
    so :func:`verify_oracle_identity` can check that minimising the first really
    does minimise the second *under the gen9 protocol* — gen8 verified the identity
    in its exhaustive protocol, where the candidate set and the scored set coincide,
    and that proof does not transfer for free to disjoint pool/evaluation sets.
    """
    residual = np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float)
    population = np.concatenate([residual[pool], residual[evaluation]])
    median = float(np.median(population))
    q = np.abs(residual[pool] - median)
    realised = np.array([float(np.abs(residual[evaluation] - residual[i]).mean()) for i in pool])
    best = float(realised.min()) if len(realised) else float("nan")
    return pd.DataFrame({
        "candidate": pool.astype(int),
        "residual": residual[pool],
        "median_residual": median,
        "oracle_deviation": q,
        "realised_mae": realised,
        "regret": realised - best,
        "rank_percentile": _rank_percentile(realised),
    })


def verify_oracle_identity(labels: pd.DataFrame) -> dict:
    """Does ``argmin oracle_deviation`` attain ``min realised_mae``?"""
    if labels.empty:
        return {"n": 0}
    pick = int(labels["oracle_deviation"].to_numpy().argmin())
    gap = float(labels["realised_mae"].iloc[pick] - labels["realised_mae"].min())
    from scipy.stats import spearmanr

    rho = float("nan")
    a = labels["oracle_deviation"].to_numpy(dtype=float)
    b = labels["realised_mae"].to_numpy(dtype=float)
    # A block where every candidate scores the same has no ordering to correlate;
    # scipy warns and returns NaN, so the degenerate case is caught here instead of
    # filling the log with ConstantInputWarning on every such block.
    if len(labels) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0:
        rho = float(spearmanr(a, b).statistic)
    return {"n": int(len(labels)), "gap": gap, "exact": bool(gap <= 1e-12), "spearman": rho}


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AcquisitionDataset:
    features: pd.DataFrame
    labels: pd.DataFrame
    audit: dict


def build_dataset(oof: pd.DataFrame, *, membership: pd.DataFrame | None,
                  repeats: int, seed: int, min_rows: int = 4, pool_cap: int = 48,
                  ligands: Sequence[str] | None = None) -> AcquisitionDataset:
    """Features and labels for every (ligand, repeat) block in ``oof``.

    ``oof`` must already be restricted to the *training* ligands of a fold.  The
    pool/evaluation draw uses gen8's function with gen8's seeding, so a training
    block looks exactly like the deployment blocks the policy will face.
    """
    feature_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []
    identity_checks: list[dict] = []
    skipped = 0
    for ligand, block in oof.groupby("extractant", sort=True):
        if ligands is not None and ligand not in ligands:
            continue
        block = block.reset_index(drop=True)
        n = len(block)
        if n < min_rows:
            skipped += 1
            continue
        truth = block["log_D"].to_numpy(dtype=float)
        prediction = block["prediction"].to_numpy(dtype=float)
        for repeat in range(repeats):
            rng = np.random.default_rng((seed, repeat, stable_hash(str(ligand))))
            split = make_p2_split(n, rng, pool_cap=pool_cap)
            if len(split.pool) < 2 or len(split.evaluation) < 2:
                continue
            features = candidate_features(block, prediction, split.pool, membership=membership)
            labels = oracle_labels(truth, prediction, split.pool, split.evaluation)
            key = {"extractant": str(ligand), "repeat": int(repeat),
                   "split_seed": int(block["split_seed"].iloc[0]),
                   "fold": int(block["fold"].iloc[0]),
                   "tanimoto_cluster": str(block["tanimoto_cluster"].iloc[0])}
            for name, value in key.items():
                features[name] = value
                labels[name] = value
            feature_rows.append(features)
            label_rows.append(labels)
            identity_checks.append({**key, **verify_oracle_identity(labels)})
    if not feature_rows:
        return AcquisitionDataset(pd.DataFrame(), pd.DataFrame(), {"n_blocks": 0, "skipped": skipped})
    features = pd.concat(feature_rows, ignore_index=True)
    labels = pd.concat(label_rows, ignore_index=True)
    checks = pd.DataFrame(identity_checks)
    audit = {
        "n_blocks": int(len(checks)), "n_candidates": int(len(features)),
        "n_ligands": int(features["extractant"].nunique()), "skipped_ligands": int(skipped),
        "identity_exact_share": float(checks["exact"].mean()) if len(checks) else float("nan"),
        "identity_max_gap": float(checks["gap"].max()) if len(checks) else float("nan"),
        "identity_mean_gap": float(checks["gap"].mean()) if len(checks) else float("nan"),
        "identity_spearman_median": float(checks["spearman"].median()) if len(checks) else float("nan"),
    }
    return AcquisitionDataset(features=features, labels=labels, audit=audit)


# --------------------------------------------------------------------------- #
# Learners
# --------------------------------------------------------------------------- #

def _matrix(features: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise KeyError(f"acquisition features missing {missing}")
    values = features[list(columns)].to_numpy(dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class ScalarAcquisition:
    """B1 — regress ``oracle_deviation`` and pick its minimum."""

    columns: tuple[str, ...]
    name: str = "LEARNED_SCALAR"
    target: str = "oracle_deviation"
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 0
    _model: object = field(default=None, init=False, repr=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _scale: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> "ScalarAcquisition":
        from sklearn.ensemble import HistGradientBoostingRegressor

        x = _matrix(features, self.columns)
        # Standardisation is fitted on the training fold only; nothing about a
        # held-out chemotype touches these moments.
        self._mean = x.mean(axis=0)
        self._scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
        y = labels[self.target].to_numpy(dtype=float)
        self._model = HistGradientBoostingRegressor(
            max_iter=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, min_samples_leaf=20,
            random_state=self.random_state).fit((x - self._mean) / self._scale, y)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Lower is better, matching ``oracle_deviation``."""
        x = (_matrix(features, self.columns) - self._mean) / self._scale
        return np.asarray(self._model.predict(x), dtype=float)


@dataclass
class PairwiseAcquisition:
    """B2 — learn the *ordering*, and specifically the top of it.

    The downstream task never needs the value of ``q``; it needs to know which of
    two candidates is closer to the residual median.  A ranking objective says only
    that, so it is not penalised for mis-estimating the scale of a quantity whose
    per-block level varies by an order of magnitude between ligands and would
    otherwise dominate a squared-error fit.

    Two refinements, both from the structure of the task rather than from any
    observed score, and both declared before the finalist evaluation.

    ``top_fraction`` — the deployed rule takes an **argmin**, so being right about
    the ordering of two mediocre candidates is worth nothing.  Every training pair
    therefore has at least one member in the block's best ``top_fraction``, which
    concentrates the fit on the decision the policy actually makes.

    ``pair weights`` — a pair whose two candidates differ by 0.001 in ``q`` is a coin
    flip whichever way it is labelled, and fitting it hard is fitting noise.  Pairs
    are weighted by ``|q_a - q_b|``, so decisive comparisons carry the fit.
    """

    columns: tuple[str, ...]
    name: str = "LEARNED_RANK"
    target: str = "oracle_deviation"
    max_pairs_per_block: int = 120
    top_fraction: float = 0.34
    C: float = 1.0
    random_state: int = 0
    _weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _scale: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> "PairwiseAcquisition":
        from sklearn.linear_model import LogisticRegression

        x = _matrix(features, self.columns)
        self._mean = x.mean(axis=0)
        self._scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
        z = (x - self._mean) / self._scale
        y = labels[self.target].to_numpy(dtype=float)
        block_key = (features["extractant"].astype(str) + "|"
                     + features["repeat"].astype(str)).to_numpy()
        rng = np.random.default_rng(self.random_state)
        left, right, target, weight = [], [], [], []
        for key in pd.unique(block_key):
            index = np.flatnonzero(block_key == key)
            if len(index) < 2:
                continue
            order = index[np.argsort(y[index], kind="stable")]
            n_top = max(1, int(np.ceil(self.top_fraction * len(order))))
            top = order[:n_top]
            budget = min(self.max_pairs_per_block, len(index) * (len(index) - 1) // 2)
            for _ in range(budget):
                a = int(rng.choice(top))
                b = int(rng.choice(index))
                if a == b or y[a] == y[b]:
                    continue
                left.append(a)
                right.append(b)
                target.append(1.0 if y[a] < y[b] else 0.0)
                weight.append(abs(float(y[a] - y[b])))
        if not left:
            raise ValueError("no usable ranking pairs")
        weights = np.asarray(weight)
        weights = weights / weights.mean() if weights.mean() > 0 else np.ones_like(weights)
        difference = z[np.asarray(left)] - z[np.asarray(right)]
        model = LogisticRegression(fit_intercept=False, C=self.C, max_iter=5000)
        model.fit(difference, np.asarray(target), sample_weight=weights)
        # P(a beats b) rises with w.(z_a - z_b); "beats" means lower q, so the
        # deployable score is the negative of that projection.
        self._weights = -np.asarray(model.coef_).ravel()
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        z = (_matrix(features, self.columns) - self._mean) / self._scale
        return z @ self._weights


# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #

@dataclass
class LearnedPolicy:
    """A fold-local acquisition model wearing gen8's ``Policy`` signature.

    ``evaluate_fewshot`` resolves policies by name out of a registry and hands them
    a :class:`PolicyContext` with ``truth=None``.  This object reads the fold from
    the block's own columns, looks up the model fitted on *that fold's training
    ligands*, and scores the remaining pool candidates.  It never touches
    ``context.truth`` (which is ``None``) and never touches ``context.evaluation``
    — asserted in :mod:`tests.test_gen9_acquisition`, because using the evaluation
    indices would be a subtle way of learning which rows will be scored.

    For ``k >= 2`` it delegates: gen8's finding that the first point wants
    *centrality* and later points want *leverage* is a statement about two different
    estimation problems, and one rule for both is known to be wrong.
    """

    models: Mapping[tuple[int, int], object]
    columns: tuple[str, ...]
    name: str = "LEARNED"
    membership: pd.DataFrame | None = None
    later_policy: str = "FARTHEST_FROM_EXISTING"

    def __call__(self, context, selected: list[int]) -> int:
        from ..gen8.kshot import POLICIES, _remaining

        remaining = _remaining(context, selected)
        if len(remaining) == 0:                          # pragma: no cover - guarded upstream
            return int(context.pool[0])
        if selected:
            return int(POLICIES[self.later_policy](context, selected))
        block = context.block
        key = (int(block["split_seed"].iloc[0]), int(block["fold"].iloc[0]))
        model = self.models.get(key)
        if model is None:
            # A fold with no fitted model must fail loudly in the runner, not
            # silently degrade to a different policy and be reported as "learned".
            raise KeyError(f"{self.name}: no acquisition model fitted for fold {key}")
        features = candidate_features(block, context.prediction, np.asarray(remaining, dtype=int),
                                      membership=self.membership)
        scores = np.asarray(model.score(features), dtype=float)
        scores = np.where(np.isfinite(scores), scores, np.inf)
        return int(remaining[int(np.argmin(scores))])


# --------------------------------------------------------------------------- #
# Baselines the brief names that gen8's registry does not carry
# --------------------------------------------------------------------------- #

def _remaining_of(context, selected):
    from ..gen8.kshot import _remaining
    return _remaining(context, selected)


def policy_min_prediction(context, selected: list[int]) -> int:
    """Measure where the model says extraction is weakest."""
    remaining = _remaining_of(context, selected)
    values = np.where(np.isfinite(context.prediction[remaining]), context.prediction[remaining], np.inf)
    return int(remaining[int(np.argmin(values))])


def policy_max_prediction(context, selected: list[int]) -> int:
    """...and where it says extraction is strongest.  Both are in the brief because
    a plausible-sounding rule that is *wrong* is worth measuring once."""
    remaining = _remaining_of(context, selected)
    values = np.where(np.isfinite(context.prediction[remaining]), context.prediction[remaining], -np.inf)
    return int(remaining[int(np.argmax(values))])


def policy_min_gp_design_variance(context, selected: list[int]) -> int:
    """gen8's one uncertainty signal that beat random — and its diagnosis.

    The RBF leave-one-out variance over the ligand's own condition design is low
    exactly where a candidate is well surrounded by its siblings, so minimising it
    is geometric centrality in an uncertainty costume.  Carried as a *named
    baseline* so gen9 can say whether a learned policy beats it, rather than
    beating random and calling that acquisition.
    """
    remaining = _remaining_of(context, selected)
    pool = np.asarray(context.pool, dtype=int)
    variance = _gp_design_variance(context.axes[pool])
    lookup = {int(p): float(v) for p, v in zip(pool, variance)}
    values = np.array([lookup.get(int(r), np.inf) for r in remaining])
    values = np.where(np.isfinite(values), values, np.inf)
    if not np.isfinite(values).any():                    # pragma: no cover
        return int(context.rng.choice(remaining))
    return int(remaining[int(np.argmin(values))])


#: Registered into gen8's ``POLICIES`` by the gen9 scripts, never at import time —
#: mutating a shared registry on import would change what another study's
#: ``sorted(POLICIES)`` default means.
EXTRA_POLICIES: dict = {
    "MIN_PREDICTION": policy_min_prediction,
    "MAX_PREDICTION": policy_max_prediction,
    "MIN_GP_DESIGN_VAR": policy_min_gp_design_variance,
}


def register_extra_policies() -> list[str]:
    from ..gen8.kshot import POLICIES

    POLICIES.update(EXTRA_POLICIES)
    return sorted(EXTRA_POLICIES)


# --------------------------------------------------------------------------- #
# B3 — a blend that nests the baseline it has to beat
# --------------------------------------------------------------------------- #

def _within_pool_rank(values: np.ndarray) -> np.ndarray:
    """Rank a score inside its own candidate pool, so two scores can be averaged.

    A learned score and a medoid distance live on incomparable scales; averaging
    them raw would let whichever has the larger variance decide everything.  Both
    are converted to a rank in [0, 1] within the pool first.
    """
    return _rank_percentile(np.asarray(values, dtype=float))


@dataclass
class BlendedAcquisition:
    """B3 — ``alpha * MEDOID + (1 - alpha) * LEARNED``, with alpha fitted in-fold.

    B1 and B2 both have to beat ``MEDOID``, and either could fail by learning a
    combination that generalises worse than the one rule gen8 already validated.
    This arm removes that failure mode by *containing* the baseline: at
    ``alpha = 1`` it is exactly ``MEDOID``, so the question becomes "does any
    departure from MEDOID help", which is answerable rather than all-or-nothing.

    ``alpha`` is chosen by scoring each candidate value on the fold's **training**
    blocks — whose labels are known because those ligands were held out in other
    folds — and never on the fold's own test chemotypes.  That is the same nested
    selection gen7's ``SELECT_inner`` used, one level down.
    """

    inner: object
    columns: tuple[str, ...]
    name: str = "LEARNED_BLEND"
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    #: The centrality rules gen8 validated, as candidate anchors.  Which one is the
    #: stronger baseline is itself a fold-local question — gen8 measured MEDOID and
    #: CENTRAL within 0.005 of each other — so it is chosen rather than assumed.
    anchors: tuple[str, ...] = ("dist_medoid_l1", "dist_centre_l2")
    alpha: float = 1.0
    anchor: str = "dist_medoid_l1"
    selection: dict = field(default_factory=dict)

    @staticmethod
    def ligand_halves(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Split the training blocks by ligand, deterministically and process-stably.

        BLAKE2b, not Python's salted ``hash`` — the same rule the pool/evaluation
        draw uses, for the same reason.
        """
        ligands = features["extractant"].astype(str).to_numpy()
        first = np.array([stable_hash(str(x)) % 2 == 0 for x in ligands])
        return first, ~first

    def fit_alpha(self, features: pd.DataFrame, labels: pd.DataFrame) -> "BlendedAcquisition":
        """Pick the anchor and alpha by realised one-shot MAE on the given blocks.

        **The blocks passed here must come from ligands ``self.inner`` never saw.**
        The first version of this selected alpha on the very blocks the ranker had
        been fitted on, and the consequence was not subtle: the ranker reproduces
        those blocks better than any fixed geometric rule can, so ``alpha = 0`` — pure
        learned score, no fallback — won in **25 folds out of 25**, and lost to
        ``MEDOID`` on every held-out chemotype. An arm built to nest its own baseline
        was thereby guaranteed never to fall back to it.
        """
        block_key = (features["extractant"].astype(str) + "|"
                     + features["repeat"].astype(str)).to_numpy()
        learned = np.asarray(self.inner.score(features), dtype=float)
        anchors = {name: _matrix(features, (name,)).ravel() for name in self.anchors}
        realised = labels["realised_mae"].to_numpy(dtype=float)
        scores: dict[tuple[str, float], list[float]] = {
            (name, a): [] for name in self.anchors for a in self.alphas}
        for key in pd.unique(block_key):
            index = np.flatnonzero(block_key == key)
            if len(index) < 2:
                continue
            rank_learned = _within_pool_rank(learned[index])
            for name, values in anchors.items():
                rank_anchor = _within_pool_rank(values[index])
                for a in self.alphas:
                    combined = a * rank_anchor + (1.0 - a) * rank_learned
                    scores[(name, a)].append(float(realised[index[int(np.argmin(combined))]]))
        means = {key: float(np.mean(v)) if v else np.inf for key, v in scores.items()}
        self.anchor, self.alpha = min(means, key=means.get)
        self.selection = {
            "means": {f"{name}@{a}": value for (name, a), value in means.items()},
            "chosen_anchor": self.anchor, "chosen_alpha": self.alpha,
            "n_blocks": int(len(pd.unique(block_key)))}
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        learned = _within_pool_rank(np.asarray(self.inner.score(features), dtype=float))
        anchor = _within_pool_rank(_matrix(features, (self.anchor,)).ravel())
        return self.alpha * anchor + (1.0 - self.alpha) * learned
