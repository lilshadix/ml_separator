"""A parametric physics-latent model of the extraction surface (brief section 6).

The idea
--------
Every other gen8 method treats the held-out ligand as *a set of rows to be
corrected*.  This module treats it as *a set of physical parameters to be
inferred*.  The response law is the solvent-extraction mass-action law

.. math::

    \\log D \\;=\\; b_L
                 \\;+\\; n_L \\log_{10}[\\mathrm{extractant}]
                 \\;+\\; m_L \\log_{10}[\\mathrm{acid}]
                 \\;+\\; \\sum_j c_{L,j}\\,\\phi_j(r_{\\mathrm{ionic}})
                 \\;+\\; t_L \\, \\tilde{T}

so a ligand *is* the vector ``theta_L = [b_L, n_L, m_L, t_L, c_L]`` -- eight
numbers, not a row of a table.  Three consequences follow, and they are the
whole reason the module exists:

* **calibration updates parameters, not weights.**  Given ``k`` measurements the
  posterior over ``theta_L`` has a closed form.  Nothing is retrained, nothing is
  fine-tuned, and the same fitted state serves ``k = 0, 1, 2, 3, 5``;
* **the ``k = 1`` behaviour is derived, not hand-written.**  A single observation
  enters through a rank-one update ``X_S^T X_S / sigma^2``.  Measured on the
  held-out ligands, 85% of the resulting change *in function space* is a constant
  level shift and only 15% is shape (at ``k = 3`` it is 65 / 35).  Offset
  correction is therefore a special case this model **derives**, not a competitor
  it has to be told to imitate.  Note the decomposition only looks like that in
  function space: the RBF metal columns are collinear with the intercept, so in
  parameter space the level shift is shared out among them;
* **the failure mode is visible in the algebra.**  ``n_L`` is identifiable only
  where the ligand actually has an extractant-concentration titration and ``m_L``
  only where it has an acid titration.  Of the 143 held-out ligands the study
  scores, 26 have three or more extractant concentrations, 75 have three or more
  acid concentrations, 26 have both and 68 have neither.  For the rest
  ``X_L^T X_L`` is rank-deficient in those columns and the posterior returns the
  prior -- which is the honest answer, not a silent zero.

Fitting, in four stages
-----------------------
1. **Hierarchical fit on the fold's training rows.**  A Gaussian random-effects
   model ``y_L = X_L theta_L + eps``, ``theta_L ~ N(mu, Sigma)``, fitted by EM.
   The shrinkage strength is *not* a hyperparameter: ``Sigma`` is estimated from
   the spread of the training ligands, on training rows only.
2. **Chemistry -> prior map.**  A regression from a compact ligand
   representation (``DONORS`` + ``PHYSCHEM`` + ``mech__*`` from
   :mod:`.mechanism`) onto the posterior means ``theta_L``.  Ridge, gradient
   boosting and a small neural hypernetwork are all available; each output is
   weighted by that ligand's posterior precision for that coefficient, so a
   ligand whose ``n_L`` is pure prior does not teach the map about ``n_L``.
3. **Recalibration, and only then a prior.**  The map's predictions are rescaled
   by a per-coefficient slope fitted on *out-of-sample* inner-fold predictions,
   so a coefficient the chemistry cannot predict gets slope zero and the prior
   falls back to the population value instead of injecting noise.  This is not a
   refinement, it is load-bearing: without it the zero-shot residual arm scored
   1.24 against the frozen model's 1.01, i.e. the uncalibrated chemistry prior
   actively damaged a model it was meant to improve.
4. **Prior strength by inner replay.**  ``(trust, spread)`` -- how much of the
   chemistry prior to keep and how freely ``k`` measurements may move the
   parameters -- are chosen by replaying the k-shot protocol on inner
   chemotype-blocked *training* ligands.  ``trust = 0`` with a large ``spread``
   *is* ridge-shrunk offset correction on the physical design, so that baseline
   is a point of the search grid and the model cannot be beaten by it through a
   bad variance estimate.

Contract
--------
``predict`` reads only ``observed``; the target column of ``block`` is never
touched, and ``scripts/gen8_physics_latent.py`` proves it by re-running every
adapter with ``log_D`` destroyed and asserting byte-identical output.
``fit_fold`` sees only the rows it is handed.  Everything is deterministic given
``model_seed``.  Any numerical failure falls back to the frozen model's
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .mechanism import MECHANISM_COLUMNS, mechanism_features

# --------------------------------------------------------------------------- #
# Column vocabulary
# --------------------------------------------------------------------------- #

#: The two mass-action axes, in raw log10 units so the fitted coefficients *are*
#: the stoichiometric exponents and can be compared with ``curve_table.parquet``.
X_EXT = "massact__log10_cond__extractant_concentration_M"
X_ACID = "massact__log10_cond__acid_concentration_M"
#: A "small term": standardised, because log10(T/degC) has a spread of 0.06 and a
#: raw-unit coefficient would be numerically incomparable with the others.
X_TEMP = "massact__log10_cond__temperature_C"
RADIUS = "Ionic Radius_metal"

#: Compact ligand representation the chemistry -> theta map is allowed to see.
DONOR_COLUMNS: tuple[str, ...] = (
    "donor__O(amide_carbonyl)", "donor__O(ether)", "donor__N(aromatic)",
    "donor__N(amine)", "donor__S(donor)", "donor__O(hydroxyl)",
    "donor__O(ester_carbonyl)", "donor__O(carbonyl)", "donor__n_total",
    "DENTATE", "coreCN", "n_ligs", "n_fill",
)
PHYSCHEM_COLUMNS: tuple[str, ...] = (
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumAromaticRings", "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP",
)

_MECH_CACHE: dict[str, np.ndarray] = {}


def _mechanism_matrix(smiles: Sequence[str]) -> np.ndarray:
    """Cached ``mech__*`` rows for a list of SMILES, in input order."""
    missing = [s for s in dict.fromkeys(smiles) if s not in _MECH_CACHE]
    if missing:
        frame = mechanism_features(missing)
        for name, row in zip(missing, frame.to_numpy(dtype=float)):
            _MECH_CACHE[name] = row
    return np.vstack([_MECH_CACHE[s] for s in smiles])


def ligand_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct extractant, in a deterministic (sorted) order.

    ``groupby(...).head(1)`` would return the *frame's* order, not the group order,
    so every downstream array would silently disagree about which row is which
    ligand.  Sorting explicitly is the only way the chemistry matrix, the theta
    matrix and the inner-CV cluster labels stay aligned.
    """
    first = frame.drop_duplicates(subset="extractant").set_index("extractant")
    return first.loc[sorted(first.index.astype(str))]


def ligand_chemistry(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """One compact chemistry row per distinct extractant in ``frame``.

    Returns ``(smiles, matrix)`` with ``matrix`` of shape ``(n_ligands, d)``.
    Target-free by construction: only structural columns are read.
    """
    first = ligand_table(frame)
    smiles = [str(s) for s in first.index]
    tabular = [c for c in DONOR_COLUMNS + PHYSCHEM_COLUMNS if c in first.columns]
    left = first[tabular].to_numpy(dtype=float) if tabular else np.zeros((len(first), 0))
    return smiles, np.hstack([left, _mechanism_matrix(smiles)])


# --------------------------------------------------------------------------- #
# The physical design matrix
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DesignSpec:
    """Everything needed to turn rows into the design ``X`` of the response law.

    All centring/scaling constants come from the fold's *training* rows, so the
    same map applies unchanged to a ligand that has never been seen.
    """

    metal_basis: str                 # "rbf" | "categorical"
    ext_centre: float
    acid_centre: float
    temp_centre: float
    temp_scale: float
    radius_centre: float
    radius_scale: float
    rbf_centres: np.ndarray
    rbf_width: float
    metal_levels: tuple[str, ...]
    metal_column_means: np.ndarray
    terms: tuple[str, ...]
    names: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.names)


def _finite(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.where(np.isfinite(values), values, fill)


#: The non-metal terms of the law, and the column each reads.
TERMS: dict[str, str] = {"n_ext": X_EXT, "m_acid": X_ACID, "t_temp": X_TEMP}
DEFAULT_TERMS: tuple[str, ...] = ("n_ext", "m_acid", "t_temp")


def build_design_spec(train: pd.DataFrame, *, metal_basis: str = "rbf",
                      n_basis: int = 4, terms: tuple = DEFAULT_TERMS) -> DesignSpec:
    """Fit the design's constants on training rows only.

    ``terms = ()`` and ``n_basis = 0`` leave the intercept alone, which is the
    physics-latent spelling of offset-only calibration and is what the attribution
    ablation uses to separate "the law" from "the Bayesian machinery".
    """
    def column(name: str) -> np.ndarray:
        return (train[name].to_numpy(dtype=float) if name in train.columns
                else np.full(len(train), np.nan))

    ext, acid, temp = column(X_EXT), column(X_ACID), column(X_TEMP)
    radius = column(RADIUS)
    ext_centre = float(np.nanmean(ext)) if np.isfinite(ext).any() else 0.0
    acid_centre = float(np.nanmean(acid)) if np.isfinite(acid).any() else 0.0
    temp_centre = float(np.nanmean(temp)) if np.isfinite(temp).any() else 0.0
    temp_scale = float(np.nanstd(temp)) if np.isfinite(temp).any() else 1.0
    temp_scale = temp_scale if temp_scale > 1e-9 else 1.0
    radius_centre = float(np.nanmean(radius)) if np.isfinite(radius).any() else 0.0
    radius_scale = float(np.nanstd(radius)) if np.isfinite(radius).any() else 1.0
    radius_scale = radius_scale if radius_scale > 1e-9 else 1.0

    terms = tuple(x for x in terms if x in TERMS)
    head = ("intercept",) + terms
    metal_levels: tuple[str, ...] = ()
    rbf_centres = np.zeros(0)
    rbf_width = 1.0
    if metal_basis == "categorical":
        metal_levels = tuple(sorted(train["metal_symbol"].astype(str).unique()))
        names = head + tuple(f"metal[{s}]" for s in metal_levels)
    elif n_basis > 0:
        z = (_finite(radius, radius_centre) - radius_centre) / radius_scale
        rbf_centres = np.quantile(z, np.linspace(0.08, 0.92, max(2, n_basis)))
        spread = float(rbf_centres.max() - rbf_centres.min())
        rbf_width = max(spread / max(1, len(rbf_centres) - 1), 1e-3)
        names = head + tuple(f"rbf{j}" for j in range(len(rbf_centres)))
    else:
        names = head

    spec = DesignSpec(metal_basis=metal_basis, ext_centre=ext_centre, acid_centre=acid_centre,
                      temp_centre=temp_centre, temp_scale=temp_scale,
                      radius_centre=radius_centre, radius_scale=radius_scale,
                      rbf_centres=rbf_centres, rbf_width=rbf_width,
                      metal_levels=metal_levels,
                      metal_column_means=np.zeros(len(names) - len(head)),
                      terms=terms, names=names)
    # Centre the metal block on the training mean so the intercept keeps its meaning
    # as "the ligand's level at average conditions" rather than "at zero radius".
    raw = _metal_block(train, spec)
    return DesignSpec(**{**spec.__dict__, "metal_column_means": raw.mean(axis=0)})


def _metal_block(frame: pd.DataFrame, spec: DesignSpec) -> np.ndarray:
    n = len(frame)
    if spec.metal_basis != "categorical" and len(spec.rbf_centres) == 0:
        return np.zeros((n, 0))
    if spec.metal_basis == "categorical":
        symbols = frame["metal_symbol"].astype(str).to_numpy()
        out = np.zeros((n, len(spec.metal_levels)))
        index = {s: j for j, s in enumerate(spec.metal_levels)}
        for i, s in enumerate(symbols):
            j = index.get(s)
            if j is not None:
                out[i, j] = 1.0
        return out
    radius = (frame[RADIUS].to_numpy(dtype=float) if RADIUS in frame.columns
              else np.full(n, np.nan))
    z = (_finite(radius, spec.radius_centre) - spec.radius_centre) / spec.radius_scale
    return np.exp(-0.5 * ((z[:, None] - spec.rbf_centres[None, :]) / spec.rbf_width) ** 2)


def design(frame: pd.DataFrame, spec: DesignSpec) -> np.ndarray:
    """``X`` for the rows of ``frame``.  Missing conditions impute to the centre."""
    n = len(frame)

    def column(name: str, centre: float, scale: float = 1.0) -> np.ndarray:
        values = (frame[name].to_numpy(dtype=float) if name in frame.columns
                  else np.full(n, np.nan))
        return _finite(values - centre, 0.0) / scale

    centres = {"n_ext": (X_EXT, spec.ext_centre, 1.0),
               "m_acid": (X_ACID, spec.acid_centre, 1.0),
               "t_temp": (X_TEMP, spec.temp_centre, spec.temp_scale)}
    parts = [np.ones((n, 1))]
    parts += [column(*centres[name])[:, None] for name in spec.terms]
    parts.append(_metal_block(frame, spec) - spec.metal_column_means[None, :])
    return np.hstack(parts)


# --------------------------------------------------------------------------- #
# Stage 1 -- hierarchical (random-effects) fit of theta_L
# --------------------------------------------------------------------------- #

@dataclass
class HierarchicalFit:
    mu: np.ndarray                       # population mean theta
    sigma: np.ndarray                    # population covariance of theta
    noise: float                         # residual variance
    theta: dict[str, np.ndarray]         # ligand -> posterior mean
    posterior_cov: dict[str, np.ndarray]  # ligand -> posterior covariance
    identifiable: dict[str, np.ndarray]  # ligand -> bool per coefficient


def _psd(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.clip(values, floor, None)) @ vectors.T


def fit_hierarchical(train: pd.DataFrame, y: np.ndarray, spec: DesignSpec, *,
                     iterations: int = 40, diagonal_shrink: float = 0.25,
                     tolerance: float = 1e-6) -> HierarchicalFit:
    """EM for ``y_L = X_L theta_L + eps``, ``theta_L ~ N(mu, Sigma)``.

    The shrinkage strength is the fitted ``Sigma``: a coefficient the training
    ligands genuinely disagree about is left free, and one they do not is pulled
    to the population value.  Nothing here is tuned on anything but train rows.
    """
    ligands = train["extractant"].astype(str).to_numpy()
    order = sorted(set(ligands))
    X_all = design(train, spec)
    p = X_all.shape[1]

    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in order:
        mask = ligands == name
        Xl, yl = X_all[mask], np.asarray(y, dtype=float)[mask]
        blocks[name] = (Xl, yl)
        grams[name] = (Xl.T @ Xl, Xl.T @ yl)

    # Pooled ridge start: mu0 and a first noise estimate.
    gram = X_all.T @ X_all + 1e-3 * np.eye(p)
    mu = np.linalg.solve(gram, X_all.T @ np.asarray(y, dtype=float))
    residual = np.asarray(y, dtype=float) - X_all @ mu
    noise = max(float(np.mean(residual ** 2)), 1e-3)
    sigma = np.eye(p) * max(float(np.var(np.asarray(y, dtype=float))), 1e-2)

    theta: dict[str, np.ndarray] = {}
    cov: dict[str, np.ndarray] = {}
    for _ in range(iterations):
        precision = np.linalg.inv(_psd(sigma))
        theta, cov = {}, {}
        for name, (XtX, Xty) in grams.items():
            post_cov = np.linalg.inv(_psd(XtX / noise + precision, floor=1e-9))
            theta[name] = post_cov @ (Xty / noise + precision @ mu)
            cov[name] = post_cov
        stacked = np.vstack([theta[n] for n in order])
        new_mu = stacked.mean(axis=0)
        centred = stacked - new_mu[None, :]
        new_sigma = (centred.T @ centred) / len(order)
        new_sigma += sum(cov[n] for n in order) / len(order)
        new_sigma = (1.0 - diagonal_shrink) * new_sigma + diagonal_shrink * np.diag(
            np.diag(new_sigma))
        # macro noise: every ligand counts once, so the 1,488-row extractant does
        # not set sigma for the other 151.
        per_ligand = []
        for name, (Xl, yl) in blocks.items():
            r = yl - Xl @ theta[name]
            trace = float(np.trace(Xl @ cov[name] @ Xl.T)) / len(yl)
            per_ligand.append(float(np.mean(r ** 2)) + trace)
        new_noise = max(float(np.mean(per_ligand)), 1e-3)
        delta = float(np.max(np.abs(new_mu - mu))) + abs(new_noise - noise)
        mu, sigma, noise = new_mu, _psd(new_sigma), new_noise
        if delta < tolerance:
            break

    identifiable: dict[str, np.ndarray] = {}
    for name, (Xl, _) in blocks.items():
        spread = Xl.std(axis=0)
        flags = spread > 1e-8
        flags[0] = len(Xl) >= 1
        identifiable[name] = flags

    return HierarchicalFit(mu=mu, sigma=sigma, noise=noise, theta=theta,
                           posterior_cov=cov, identifiable=identifiable)


def unshrunk_slopes(train: pd.DataFrame, y: np.ndarray, spec: DesignSpec,
                    *, min_levels: int = 3) -> pd.DataFrame:
    """Per-ligand OLS exponents, for the sanity check against ``curve_table``.

    Only ligands with at least ``min_levels`` distinct values on an axis get a
    number for that axis; the rest are reported as unidentifiable rather than
    given the prior's value under a different name.
    """
    ligands = train["extractant"].astype(str).to_numpy()
    y = np.asarray(y, dtype=float)
    rows = []
    for name in sorted(set(ligands)):
        mask = ligands == name
        sub = train[mask]
        Xl, yl = design(sub, spec), y[mask]
        record = {"extractant": name, "n_rows": int(mask.sum())}
        for label, column, index in (("n_ext", X_EXT, 1), ("m_acid", X_ACID, 2)):
            values = (sub[column].to_numpy(dtype=float) if column in sub.columns
                      else np.full(int(mask.sum()), np.nan))
            distinct = len(np.unique(np.round(values[np.isfinite(values)], 6)))
            record[f"{label}_levels"] = distinct
            record[label] = np.nan
            if distinct >= min_levels and len(yl) > 3:
                keep = [0, index]
                sub_X = Xl[:, keep]
                gram = sub_X.T @ sub_X + 1e-8 * np.eye(2)
                try:
                    beta = np.linalg.solve(gram, sub_X.T @ yl)
                    record[label] = float(beta[1])
                except np.linalg.LinAlgError:
                    pass
        rows.append(record)
    return pd.DataFrame(rows)


def within_series_slopes(frame: pd.DataFrame, y: np.ndarray, *,
                         min_levels: int = 3) -> pd.DataFrame:
    """Per-ligand exponents with the *series* baseline swept out.

    ``curve_table.parquet`` measures a slope inside one titration curve, where the
    diluent, the paper and the metal are all held fixed.  A single ``(b_L, n_L,
    m_L)`` per ligand cannot hold them fixed, and pooling across a ligand's series
    attenuates the slope toward zero.  Demeaning within ``(series_id,
    metal_symbol)`` reproduces the curve-table quantity and so isolates how much of
    the gap between the two is model mis-specification rather than a fitting bug.
    """
    y = pd.Series(np.asarray(y, dtype=float), index=frame.index)
    rows = []
    for ligand, group in frame.groupby("extractant", sort=True):
        record = {"extractant": str(ligand), "n_rows": len(group)}
        for label, column in (("n_ext", X_EXT), ("m_acid", X_ACID)):
            record[label] = np.nan
            if column not in group.columns:
                continue
            sub = group[np.isfinite(group[column].to_numpy(dtype=float))]
            if len(sub) < 4:
                continue
            keys = ["series_id", "metal_symbol"]
            keys = [k for k in keys if k in sub.columns]
            grouper = sub.groupby(keys) if keys else None
            x = sub[column].astype(float)
            target = y.loc[sub.index]
            if grouper is not None:
                x = x - grouper[column].transform("mean")
                target = target - target.groupby([sub[k] for k in keys]).transform("mean")
            if x.round(6).nunique() < min_levels or float(x.std()) < 1e-9:
                continue
            denominator = float((x * x).sum())
            if denominator > 1e-9:
                record[label] = float((x * target).sum() / denominator)
        rows.append(record)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage 2 -- chemistry -> theta prior
# --------------------------------------------------------------------------- #

@dataclass
class ChemistryMap:
    kind: str
    centre: np.ndarray
    scale: np.ndarray
    fill: np.ndarray
    models: list
    fallback: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = self._standardise(features)
        out = np.repeat(self.fallback[None, :], len(z), axis=0)
        for j, model in enumerate(self.models):
            if model is None:
                continue
            out[:, j] = model.predict(z)
        return out

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        z = np.asarray(features, dtype=float).copy()
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = np.take(self.fill, np.nonzero(bad)[1])
        return (z - self.centre[None, :]) / self.scale[None, :]


def _fit_one_output(z: np.ndarray, target: np.ndarray, weight: np.ndarray,
                    kind: str, model_seed: int, groups: np.ndarray | None = None):
    from sklearn.linear_model import Ridge
    if kind == "ridge":
        from ..gen6.cohorts import seeded_group_kfold
        alphas = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
        labels = groups if groups is not None else np.arange(len(z)).astype(str)
        # Chemotype-blocked, exactly like the outer plan.  A plain shuffled KFold
        # here would put near-duplicate ligands on both sides of the inner split and
        # would choose an alpha that is far too small -- the map would then look
        # skilful in-sample and inject noise into the prior at deployment.
        splits = list(seeded_group_kfold(np.asarray(labels).astype(str),
                                         min(5, max(2, len(np.unique(labels)))), 20260820))
        best, best_score = alphas[-1], np.inf
        for alpha in alphas:
            errors = []
            for tr, te in splits:
                if len(tr) < 4:
                    continue
                model = Ridge(alpha=alpha).fit(z[tr], target[tr], sample_weight=weight[tr])
                errors.append(float(np.average((model.predict(z[te]) - target[te]) ** 2,
                                               weights=weight[te] + 1e-9)))
            score = float(np.mean(errors)) if errors else np.inf
            if score < best_score:
                best, best_score = alpha, score
        return Ridge(alpha=best).fit(z, target, sample_weight=weight)
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=5,
            l2_regularization=1.0, random_state=model_seed % (2 ** 31),
        ).fit(z, target, sample_weight=weight)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=3000,
                            learning_rate_init=3e-3, random_state=model_seed % (2 ** 31),
                            early_stopping=False).fit(z, target)
    raise ValueError(f"unknown map kind {kind!r}")


def fit_chemistry_map(features: np.ndarray, theta: np.ndarray, precision: np.ndarray,
                      *, kind: str, model_seed: int,
                      groups: np.ndarray | None = None) -> ChemistryMap:
    """Learn ``chemistry -> theta``, one output at a time, precision-weighted.

    A ligand with no extractant titration has a posterior ``n_L`` that is simply
    the population mean; letting it vote on the ``n_L`` map with full weight would
    teach the map to predict the mean everywhere.  Weighting each output by that
    ligand's posterior precision for *that* coefficient removes the votes that
    carry no information, without discarding the ligand from the other outputs.
    """
    features = np.asarray(features, dtype=float)
    fill = np.nanmedian(np.where(np.isfinite(features), features, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    filled = np.where(np.isfinite(features), features, fill[None, :])
    centre = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (filled - centre[None, :]) / scale[None, :]

    models: list = []
    fallback = theta.mean(axis=0)
    for j in range(theta.shape[1]):
        weight = precision[:, j]
        weight = weight / max(float(weight.mean()), 1e-12)
        weight = np.clip(weight, 1e-3, 50.0)
        if float(np.std(theta[:, j])) < 1e-9 or len(z) < 8:
            models.append(None)
            continue
        try:
            models.append(_fit_one_output(z, theta[:, j], weight, kind, model_seed + j,
                                          groups=groups))
        except Exception:
            models.append(None)
    return ChemistryMap(kind=kind, centre=centre, scale=scale, fill=fill,
                        models=models, fallback=fallback)


# --------------------------------------------------------------------------- #
# The fitted per-fold state
# --------------------------------------------------------------------------- #

#: Grid for the two scalars the inner loop is allowed to choose.  ``trust`` is how
#: much of the chemistry prior's *deviation from the anchor* to keep; ``spread``
#: multiplies the prior covariance, i.e. how freely ``k`` measurements may move the
#: parameters.  A large ``spread`` with ``trust = 0`` is exactly ridge-shrunk offset
#: correction on the physical design, so the search contains that baseline as a
#: point and cannot lose to it by construction of the prior.
TRUST_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
SPREAD_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 60.0)


@dataclass
class FoldState:
    spec: DesignSpec
    hierarchy: HierarchicalFit
    chemistry: ChemistryMap
    recalibration: np.ndarray        # (p, 2): per-coefficient intercept and slope
    anchor: np.ndarray               # what the prior shrinks to when the map is useless
    prior_cov: np.ndarray
    prior_precision: np.ndarray
    noise: float
    trust: float
    spread: float
    diagnostics: dict


def _recalibrate(theta: np.ndarray, oos: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Per-coefficient Platt-style rescaling of the map's out-of-sample prediction.

    ``theta_j ~ a_j + g_j * oos_j`` fitted on the training ligands.  A map with no
    out-of-sample skill gets ``g_j = 0`` and ``a_j = mean(theta_j)``, so the prior
    silently degrades to the population value instead of injecting noise -- which
    is what an uncalibrated map did in the first version of this module (it made
    the zero-shot residual arm *worse* than the frozen model it was correcting).
    """
    p = theta.shape[1]
    out = np.zeros((p, 2))
    for j in range(p):
        # Bounded weights.  Unbounded posterior precisions differ by six orders of
        # magnitude between a ligand with a real titration and one without, and the
        # weighted mean then *is* that one ligand -- which silently turns the
        # recalibration intercept into an outlier.
        w = np.clip(weight[:, j] / max(float(weight[:, j].mean()), 1e-12), 0.1, 10.0)
        x, y = oos[:, j], theta[:, j]
        xbar = float(np.average(x, weights=w))
        ybar = float(np.average(y, weights=w))
        variance = float(np.average((x - xbar) ** 2, weights=w))
        covariance = float(np.average((x - xbar) * (y - ybar), weights=w))
        slope = covariance / variance if variance > 1e-12 else 0.0
        out[j] = (ybar - slope * xbar, float(np.clip(slope, 0.0, 2.0)))
    return out


class PhysicsLatentFitter:
    """Fits and caches one fold's physics-latent state; shared by several adapters."""

    def __init__(self, *, target: str = "logD", metal_basis: str = "rbf",
                 map_kind: str = "ridge", n_basis: int = 4,
                 diagonal_shrink: float = 0.25, inner_folds: int = 5,
                 tune: bool = True, tune_repeats: int = 3, tune_row_cap: int = 60,
                 trust: float = 1.0, spread: float = 1.0,
                 trust_grid: tuple = TRUST_GRID, spread_grid: tuple = SPREAD_GRID,
                 terms: tuple = DEFAULT_TERMS):
        self.target = target
        self.metal_basis = metal_basis
        self.map_kind = map_kind
        self.n_basis = n_basis
        self.terms = tuple(terms)
        self.diagonal_shrink = diagonal_shrink
        self.inner_folds = inner_folds
        self.tune = tune
        self.tune_repeats = tune_repeats
        self.tune_row_cap = tune_row_cap
        self.trust = trust
        self.spread = spread
        # Restricting the grid is how an ablation is expressed: ``trust_grid=(0.0,)``
        # switches the chemistry prior off without touching any other code path, so
        # the ablated arm differs from the tuned one by exactly one knob.
        self.trust_grid = tuple(trust_grid)
        self.spread_grid = tuple(spread_grid)
        self.state: FoldState | None = None
        self._key: tuple | None = None
        self.history: list[dict] = []

    # ---------------------------------------------------------------- fitting
    def fit(self, train: pd.DataFrame, y_train: np.ndarray, *, split_seed: int,
            fold: int, model_seed: int) -> None:
        key = (int(split_seed), int(fold), int(model_seed), self.target,
               self.metal_basis, self.map_kind, self.tune, self.terms, self.n_basis)
        if key == self._key:
            return
        y = np.asarray(y_train, dtype=float)
        if self.target == "residual":
            if "prediction" not in train.columns:
                raise ValueError("the residual target needs a 'prediction' column on train")
            y = y - train["prediction"].to_numpy(dtype=float)

        spec = build_design_spec(train, metal_basis=self.metal_basis, n_basis=self.n_basis,
                                 terms=self.terms)
        hierarchy = fit_hierarchical(train, y, spec, diagonal_shrink=self.diagonal_shrink)
        smiles, chem = ligand_chemistry(train)
        table = ligand_table(train)
        clusters = (table["tanimoto_cluster"].astype(str).to_numpy()
                    if "tanimoto_cluster" in table.columns else np.asarray(smiles))
        theta = np.vstack([hierarchy.theta[s] for s in smiles])
        precision = np.vstack([1.0 / np.clip(np.diag(hierarchy.posterior_cov[s]), 1e-9, None)
                               for s in smiles])

        oos, covered = self._out_of_sample_prior(chem, theta, precision, clusters,
                                                 int(model_seed))
        recalibration = _recalibrate(theta[covered], oos[covered], precision[covered])
        calibrated = recalibration[:, 0][None, :] + oos * recalibration[:, 1][None, :]
        errors = theta[covered] - calibrated[covered]
        # Prior covariance = what the chemistry map does NOT explain.  Taking the
        # population covariance from the EM and subtracting the covariance the
        # calibrated map reproduces is stable in a way the raw error covariance is
        # not: ``hierarchy.sigma`` already accounts for the shrinkage in the
        # per-ligand estimates, while a raw error covariance over ligands whose
        # ``n_L`` is pure prior is dominated by whichever ligand had a titration.
        centred = calibrated[covered] - calibrated[covered].mean(axis=0)[None, :]
        explained = (centred.T @ centred) / max(len(centred), 1)
        prior_cov_base = hierarchy.sigma - explained
        floor = 0.05 * np.diag(hierarchy.sigma)
        prior_cov_base[np.diag_indices_from(prior_cov_base)] = np.maximum(
            np.diag(prior_cov_base), floor)
        prior_cov_base = _psd(prior_cov_base, floor=1e-8)

        chemistry = fit_chemistry_map(chem, theta, precision, kind=self.map_kind,
                                      model_seed=int(model_seed), groups=clusters)
        # ``anchor`` is where a useless prior lands: the population law for the raw
        # target, and "leave the frozen model alone" for the residual target.
        anchor = (np.zeros(theta.shape[1]) if self.target == "residual"
                  else hierarchy.mu.copy())

        trust, spread, tuning = self.trust, self.spread, {}
        if self.tune:
            trust, spread, tuning = self._tune(train, y, spec, smiles, calibrated, covered,
                                               hierarchy.noise, prior_cov_base, anchor,
                                               int(model_seed))

        prior_cov = _psd(prior_cov_base * float(spread), floor=1e-8)
        diagnostics = {
            "split_seed": int(split_seed), "fold": int(fold), "target": self.target,
            "metal_basis": self.metal_basis, "map_kind": self.map_kind,
            "n_train_ligands": len(smiles), "noise": float(hierarchy.noise),
            "names": list(spec.names),
            "mu": hierarchy.mu.tolist(),
            "sigma_diag": np.diag(hierarchy.sigma).tolist(),
            "prior_cov_diag": np.diag(prior_cov).tolist(),
            "map_slope": recalibration[:, 1].tolist(),
            "map_oos_rmse": np.sqrt(np.diag((errors.T @ errors) / max(len(errors), 1))).tolist(),
            "trust": float(trust), "spread": float(spread), **tuning,
        }
        self.state = FoldState(spec=spec, hierarchy=hierarchy, chemistry=chemistry,
                               recalibration=recalibration, anchor=anchor,
                               prior_cov=prior_cov,
                               prior_precision=np.linalg.inv(_psd(prior_cov, floor=1e-8)),
                               noise=float(hierarchy.noise), trust=float(trust),
                               spread=float(spread), diagnostics=diagnostics)
        self._key = key
        self.history.append(diagnostics)

    def _out_of_sample_prior(self, chem, theta, precision, clusters, model_seed):
        """The map's prediction for each training ligand, from a map that never saw it."""
        from ..gen6.cohorts import seeded_group_kfold
        oos = np.repeat(theta.mean(axis=0)[None, :], len(theta), axis=0)
        covered = np.zeros(len(theta), dtype=bool)
        n_splits = max(2, min(self.inner_folds, len(np.unique(clusters))))
        for tr, te in seeded_group_kfold(clusters, n_splits, 20260820 + model_seed % 997):
            if len(tr) < 8:
                continue
            inner = fit_chemistry_map(chem[tr], theta[tr], precision[tr], kind=self.map_kind,
                                      model_seed=model_seed, groups=clusters[tr])
            oos[te] = inner.predict(chem[te])
            covered[te] = True
        if covered.sum() < 8:
            covered[:] = True
        return oos, covered

    # ------------------------------------------------------------------ tuning
    def _tune(self, train, y, spec, smiles, calibrated, covered, noise, prior_cov_base,
              anchor, model_seed):
        """Choose ``(trust, spread)`` by replaying the k-shot protocol on train ligands.

        The prior handed to each training ligand here is the *out-of-sample* one
        already computed above -- produced by a map fitted on an inner
        chemotype-blocked split that never saw that ligand's chemotype -- so the
        replay is under the same contract as deployment.  Nothing outside the fold's
        training rows is read: the objective is the macro MAE over inner-held-out
        training ligands at k = 0, 1, 2, 3.

        ``trust = 0`` with a large ``spread`` is precisely ridge-shrunk offset
        correction on the physical design, so that baseline is a *point of the grid*.
        The model can therefore only lose to it by the width of the tuning noise, and
        when the search lands there it is telling us the chemistry prior is worthless.
        """
        ligands = train["extractant"].astype(str).to_numpy()
        X_all = design(train, spec)
        rng = np.random.default_rng(20260820 + model_seed % 9973)

        tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for position, ligand in enumerate(smiles):
            if not covered[position]:
                continue
            rows = np.flatnonzero(ligands == ligand)
            if len(rows) < 4:
                continue
            if len(rows) > self.tune_row_cap:
                rows = np.sort(rng.choice(rows, size=self.tune_row_cap, replace=False))
            tasks.append((X_all[rows], y[rows], calibrated[position]))
        if not tasks:
            return self.trust, self.spread, {"tuned": False}

        precisions = [np.linalg.inv(_psd(prior_cov_base * s, floor=1e-8))
                      for s in self.spread_grid]
        scores = np.zeros((len(self.trust_grid), len(self.spread_grid)))
        counts = 0
        for X, target, prior_raw in tasks:
            n = len(X)
            for _ in range(self.tune_repeats):
                order = rng.permutation(n)
                cut = max(2, n // 2)
                evaluation, pool = np.sort(order[:cut]), np.sort(order[cut:])
                if len(pool) < 1:
                    continue
                chosen = pool[:min(3, len(pool))]
                Xe, ye = X[evaluation], target[evaluation]
                grams = [(X[chosen[:k]].T @ X[chosen[:k]],
                          X[chosen[:k]].T @ target[chosen[:k]])
                         for k in range(1, len(chosen) + 1)]
                counts += 1
                for a, trust in enumerate(self.trust_grid):
                    prior = anchor + trust * (prior_raw - anchor)
                    for b, precision0 in enumerate(precisions):
                        anchored = precision0 @ prior
                        total = float(np.abs(Xe @ prior - ye).mean())
                        for XtX, Xty in grams:
                            fitted = np.linalg.solve(_psd(precision0 + XtX / noise, floor=1e-10),
                                                     anchored + Xty / noise)
                            total += float(np.abs(Xe @ fitted - ye).mean())
                        scores[a, b] += total
        if counts == 0:
            return self.trust, self.spread, {"tuned": False}
        a, b = np.unravel_index(int(np.argmin(scores)), scores.shape)
        return float(self.trust_grid[a]), float(self.spread_grid[b]), {
            "tuned": True, "n_tune_ligands": len(tasks),
            "tune_score": float(scores[a, b] / max(counts, 1))}

    # -------------------------------------------------------------- inference
    def theta_prior(self, block: pd.DataFrame) -> np.ndarray:
        """The chemistry prior for one ligand, recalibrated and trust-weighted."""
        state = self.state
        assert state is not None
        _, chem = ligand_chemistry(block)
        raw = state.chemistry.predict(chem)[0]
        calibrated = state.recalibration[:, 0] + state.recalibration[:, 1] * raw
        return state.anchor + state.trust * (calibrated - state.anchor)

    def map_update(self, X: np.ndarray, prior: np.ndarray, selected: np.ndarray,
                   residual: np.ndarray) -> np.ndarray:
        """Closed-form Gaussian posterior mean of ``theta_L`` given ``k`` rows.

        With ``k = 1`` this is a rank-one update in the direction
        ``Sigma x / (x' Sigma x + sigma^2)``.  Because ``Sigma[b, b]`` dominates,
        essentially the whole step lands on the level -- offset correction falls out
        of the algebra rather than being coded as a special case.
        """
        assert self.state is not None
        if len(selected) == 0:
            return prior
        Xs = X[selected]
        precision = self.state.prior_precision + Xs.T @ Xs / self.state.noise
        rhs = self.state.prior_precision @ prior + Xs.T @ residual / self.state.noise
        return np.linalg.solve(_psd(precision, floor=1e-10), rhs)


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

@dataclass
class PhysicsAdapter:
    """One adapter view onto a shared :class:`PhysicsLatentFitter`."""

    fitter: PhysicsLatentFitter
    name: str
    use_observations: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *, split_seed: int,
                 fold: int, model_seed: int) -> None:
        self._cache.clear()
        self.fitter.fit(train, y_train, split_seed=split_seed, fold=fold,
                        model_seed=model_seed)

    def predict(self, block, prediction, selected, observed, context):
        prediction = np.asarray(prediction, dtype=float)
        state = self.fitter.state
        if state is None:
            return prediction
        try:
            key = (context.split_seed, context.fold, context.extractant, len(block))
            entry = self._cache.get(key)
            if entry is None:
                X = design(block, state.spec)
                prior = self.fitter.theta_prior(block)
                entry = (X, prior)
                self._cache[key] = entry
            X, prior = entry

            base = prediction if self.fitter.target == "residual" else np.zeros(len(block))
            if not self.use_observations or len(selected) == 0:
                theta = prior
            else:
                target = np.asarray(observed, dtype=float) - base[np.asarray(selected, int)]
                theta = self.fitter.map_update(X, prior, np.asarray(selected, int), target)
            out = base + X @ theta
        except Exception:
            return prediction
        if not np.isfinite(out).all():
            return prediction
        return out


def build_physics_adapters(*, include_gbm: bool = True, include_mlp: bool = True,
                           include_categorical: bool = True,
                           n_basis: int = 4) -> list:
    """The gen8 physics-latent adapter suite.

    ``PHYS_prior`` and ``PHYS_map`` share one fitter, so the zero-shot and the
    calibrated numbers come from *the same* fitted parameters -- the difference
    between them is exactly the value of the measurements, with nothing else moving.
    The same holds for the ``PHYS_residual`` pair.
    """
    adapters: list = []
    main = PhysicsLatentFitter(target="logD", metal_basis="rbf", map_kind="ridge",
                               n_basis=n_basis)
    adapters += [PhysicsAdapter(main, "PHYS_prior", use_observations=False),
                 PhysicsAdapter(main, "PHYS_map", use_observations=True)]
    residual = PhysicsLatentFitter(target="residual", metal_basis="rbf", map_kind="ridge",
                                   n_basis=n_basis)
    adapters += [PhysicsAdapter(residual, "PHYS_residual_prior", use_observations=False),
                 PhysicsAdapter(residual, "PHYS_residual", use_observations=True)]
    if include_categorical:
        cat = PhysicsLatentFitter(target="logD", metal_basis="categorical", map_kind="ridge")
        adapters.append(PhysicsAdapter(cat, "PHYS_map_metalcat", use_observations=True))
        cat_res = PhysicsLatentFitter(target="residual", metal_basis="categorical",
                                      map_kind="ridge")
        adapters.append(PhysicsAdapter(cat_res, "PHYS_residual_metalcat",
                                       use_observations=True))
    if include_mlp:
        mlp = PhysicsLatentFitter(target="residual", metal_basis="rbf", map_kind="mlp",
                                  n_basis=n_basis)
        adapters.append(PhysicsAdapter(mlp, "PHYS_residual_hypernet", use_observations=True))
    if include_gbm:
        gbm = PhysicsLatentFitter(target="residual", metal_basis="rbf", map_kind="gbm",
                                  n_basis=n_basis)
        adapters.append(PhysicsAdapter(gbm, "PHYS_residual_gbm", use_observations=True))
    return adapters
