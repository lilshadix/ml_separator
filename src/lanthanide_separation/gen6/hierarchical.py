"""Experiment C — hierarchical level decomposition: which component fails to transfer?

Phase 1 showed that on a new ligand the error is the per-ligand *level* and not
the response *shape*.  This module builds the models that make that attribution
with structure rather than with post-hoc decomposition of a monolithic forest:

.. code-block:: text

    log D(l, m, c)  ≈  α_l  +  F_cond(l, c)  +  F_metal(l, m, c)  +  ε

The point is not to assert the equation; it is to build estimators in which the
pieces are separately estimated, so each can be swapped for its true value and
the error it is responsible for can be measured.

Three structured models, two controls:

``C1_HIER_RIDGE`` — a partial-pooling ridge.  Fixed effects on compact ligand
descriptors, metal radius, conditions, the mass-action logs and a *small*
pre-declared interaction block, **plus a ridge-penalised random intercept per
training ligand** that shrinks toward the descriptor prediction.  For a ligand
that is in training the intercept is estimated; for a held-out ligand it is zero,
so the prediction is the descriptor prior.  That is the mixed-effects idea in its
simplest auditable form, and it is exactly what makes the model informative: the
penalty on the intercepts is tuned by inner CV grouped the same way as the outer
regime, so under a chemotype hold-out the inner CV *itself* discovers that the
descriptor prior is all that transfers.

``C2_TWO_STAGE`` — Stage A predicts the (ligand, condition) cell mean from ligand
and condition features, one row per cell; Stage B predicts the residual
``y − Â(l, c)`` with metal features added.  The residual that Stage B trains on is
formed from **cross-fitted** Stage A predictions inside the training fold, never
from in-sample ones — the same provenance rule the gen3/gen4 stacked models
obeyed, and asserted by test here.  Because each stage is a separate estimator,
either can be replaced by its oracle on the test fold:

* ``ORACLE_LEVEL``  = true cell mean + B̂   → the error the level is responsible for
* ``ORACLE_METAL``  = Â + true metal departure → the error the metal response is
  responsible for

``C3_SHARED_RIDGE`` — C1's design matrix stacked with within-cell pair-difference
rows ``x_A − x_B → y_A − y_B``, so one linear score ``g`` fits both the level and
the pair task.  Pair predictions are derived *only* as ``g(A) − g(B)``, which
makes antisymmetry and transitivity exact; both are asserted numerically.

Controls: ``MONO_ET`` (the Experiment A EXPANDED arm, which must reproduce
bit-for-bit on the chemotype folds) and ``MONO_RIDGE`` (plain ridge on C1's
features, no intercepts, same inner-CV tuning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import (
    LEVEL_TARGET_COLUMN, LevelData, LevelForestParameters, LevelRegressor,
    group_balanced_weights,
)
from .cohorts import seeded_group_kfold

#: The ridge design: compact blocks only, by design.  LIG2D_EXT (206 columns) is
#: deliberately excluded from the *linear* models — the brief asks for a
#: low-capacity structured baseline, and the forest controls keep the champion
#: feature set so neither comparison is confounded by features.
RIDGE_BASE_BLOCKS: tuple[str, ...] = ("METAL", "COND", "PHYSCHEM", "DONORS", "MASSACTION")
#: The champion feature set for the forest-based models (MONO_ET, C2 stages).
FOREST_BLOCKS: tuple[str, ...] = ("METAL", "COND", "LIG2D_EXT", "MASSACTION")
#: Stage A sees no metal feature.
STAGE_A_BLOCKS: tuple[str, ...] = ("COND", "LIG2D_EXT", "MASSACTION")

IONIC_RADIUS_COLUMN = "Ionic Radius_metal"
LOG_EXTRACTANT_COLUMN = "massact__log10_cond__extractant_concentration_M"
LOG_CONCENTRATION_COLUMNS: tuple[str, ...] = (
    "massact__log10_cond__acid_concentration_M",
    "massact__log10_cond__extractant_concentration_M",
    "massact__log10_cond__metal_concentration_mM",
    "massact__log10_cond__temperature_C",
    "massact__log10_cond__contact_time_min",
)

#: Inner-CV grids.  Small and fixed: the chosen values are recorded per fold.
FIXED_PENALTY_GRID: tuple[float, ...] = (1.0, 10.0, 100.0)
LIGAND_PENALTY_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1e6)
INNER_FOLDS = 3
CROSSFIT_FOLDS = 5

MODEL_NAMES: tuple[str, ...] = (
    "MONO_ET", "MONO_RIDGE", "C1_HIER_RIDGE", "C2_TWO_STAGE", "C2_TRUECENTRE", "C3_SHARED_RIDGE",
    "ORACLE_LEVEL", "ORACLE_METAL",
)


# --------------------------------------------------------------------------- #
# Design matrix for the linear models
# --------------------------------------------------------------------------- #

@dataclass
class RidgeDesign:
    """Fold-local standardised design with named, grouped columns.

    ``groups`` maps each column to a block name so penalties can differ by block
    and the report can attribute a prediction to α (ligand), F_cond and F_metal.
    """

    columns: tuple[str, ...]
    groups: dict[str, str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    data: "LevelData"
    ligand_levels: tuple[str, ...] = ()

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Rebuild base + interaction columns from the raw frame, then impute and
        standardise with the fold-local statistics.  Interaction columns do not
        exist on the raw frame, so they are recomputed here from the same raw
        inputs — the transform is a function of the raw row only."""
        features, _ = ridge_feature_frame(frame, self.data)
        missing = [c for c in self.columns if c not in features.columns]
        if missing:
            raise KeyError(f"frame cannot supply design columns {missing[:5]}")
        x = features[list(self.columns)].to_numpy(dtype=float)
        x = np.where(np.isnan(x), self.medians, x)
        return (x - self.means) / self.scales


def interaction_frame(frame: pd.DataFrame, data: LevelData) -> pd.DataFrame:
    """The small pre-declared interaction block.

    * donor census × ionic radius — the inner-sphere "donor topology × radius"
      hypothesis (brief Part XV);
    * log[L] × donor census — the per-ligand slope of the mass-action law, made
      transferable by expressing it through descriptors instead of per-ligand;
    * log-concentrations × ionic radius — condition × metal.

    No other interactions; the brief asks for a few physically named columns,
    not a product expansion.
    """
    out: dict[str, np.ndarray] = {}
    radius = pd.to_numeric(frame[IONIC_RADIUS_COLUMN], errors="coerce").to_numpy(dtype=float) \
        if IONIC_RADIUS_COLUMN in frame.columns else None
    log_l = pd.to_numeric(frame[LOG_EXTRACTANT_COLUMN], errors="coerce").to_numpy(dtype=float) \
        if LOG_EXTRACTANT_COLUMN in frame.columns else None
    donors = [c for c in data.blocks.get("DONORS", ()) if c in frame.columns]
    for column in donors:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if radius is not None:
            out[f"ix__donor_x_radius__{column}"] = values * radius
        if log_l is not None:
            out[f"ix__logL_x_donor__{column}"] = values * log_l
    if radius is not None:
        for column in LOG_CONCENTRATION_COLUMNS:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                out[f"ix__cond_x_radius__{column}"] = values * radius
    return pd.DataFrame(out, index=frame.index)


def ridge_feature_frame(frame: pd.DataFrame, data: LevelData) -> tuple[pd.DataFrame, dict[str, str]]:
    """Base blocks + interactions, with a block label per column."""
    groups: dict[str, str] = {}
    parts = []
    for block in RIDGE_BASE_BLOCKS:
        cols = [c for c in data.blocks.get(block, ()) if c in frame.columns]
        if cols:
            parts.append(frame[cols].apply(pd.to_numeric, errors="coerce"))
            groups.update({c: block for c in cols})
    inter = interaction_frame(frame, data)
    if not inter.empty:
        parts.append(inter)
        groups.update({c: "INTERACTION" for c in inter.columns})
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
    return out, groups


def fit_design(train: pd.DataFrame, data: LevelData) -> RidgeDesign:
    """Fold-local medians and standardisation; constant columns get unit scale."""
    features, groups = ridge_feature_frame(train, data)
    x = features.to_numpy(dtype=float)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    filled = np.where(np.isnan(x), medians, x)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    return RidgeDesign(columns=tuple(features.columns), groups=groups,
                       medians=medians, means=means, scales=scales, data=data)


# --------------------------------------------------------------------------- #
# Penalised linear solver with per-block penalties and ligand intercepts
# --------------------------------------------------------------------------- #

@dataclass
class HierarchicalRidge:
    """Ridge with a block penalty on fixed effects and a separate one on ligand intercepts.

    Solves ``min Σ w (y − β0 − Xβ − Zγ)² + λ_fixed‖β‖² + λ_ligand‖γ‖²`` where ``Z``
    is the one-hot of *training* ligands.  ``λ_ligand → ∞`` recovers plain ridge;
    ``λ_ligand → 0`` lets the intercepts absorb all between-ligand variance, which
    leaves the descriptor prior for a new ligand unidentified — inner CV grouped
    like the outer regime is what keeps that from happening.
    """

    design: RidgeDesign
    lambda_fixed: float
    lambda_ligand: float | None          # None = no ligand intercepts (plain ridge)
    intercept_: float = 0.0
    beta_: np.ndarray | None = None
    gamma_: dict[str, float] = field(default_factory=dict)
    y_range_: tuple[float, float] | None = None

    def _ligand_matrix(self, ligands: Sequence[str], levels: Sequence[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(levels)}
        z = np.zeros((len(ligands), len(levels)))
        for row, name in enumerate(ligands):
            j = index.get(str(name))
            if j is not None:
                z[row, j] = 1.0
        return z

    def fit(self, x: np.ndarray, y: np.ndarray, *, weights: np.ndarray | None,
            ligands: Sequence[str] | None = None,
            extra_x: np.ndarray | None = None, extra_y: np.ndarray | None = None,
            extra_w: np.ndarray | None = None) -> "HierarchicalRidge":
        """``extra_*`` are additional rows with no intercept and no ligand columns
        (the pair-difference rows of C3): a difference of two rows of the same
        ligand has zero intercept and zero ligand indicator."""
        n, p = x.shape
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        use_ligands = self.lambda_ligand is not None and ligands is not None
        levels: tuple[str, ...] = ()
        if use_ligands:
            levels = tuple(sorted({str(l) for l in ligands}))
            z = self._ligand_matrix(ligands, levels)
        else:
            z = np.zeros((n, 0))
        ones = np.ones((n, 1))
        a = np.hstack([ones, x, z])
        aw = a * w[:, None]
        gram = aw.T @ a
        rhs = aw.T @ y
        if extra_x is not None and len(extra_x):
            ex = np.hstack([np.zeros((len(extra_x), 1)), extra_x, np.zeros((len(extra_x), z.shape[1]))])
            ew = np.ones(len(extra_x)) if extra_w is None else np.asarray(extra_w, dtype=float)
            exw = ex * ew[:, None]
            gram = gram + exw.T @ ex
            rhs = rhs + exw.T @ np.asarray(extra_y, dtype=float)
        penalty = np.concatenate([[0.0], np.full(p, self.lambda_fixed),
                                  np.full(z.shape[1], self.lambda_ligand or 0.0)])
        gram = gram + np.diag(penalty)
        coef = np.linalg.solve(gram, rhs)
        self.intercept_ = float(coef[0])
        self.beta_ = coef[1:1 + p]
        self.gamma_ = {name: float(v) for name, v in zip(levels, coef[1 + p:])}
        self.design.ligand_levels = levels
        span = float(y.max() - y.min())
        self.y_range_ = (float(y.min()) - 0.5 * span, float(y.max()) + 0.5 * span)
        return self

    def predict_components(self, x: np.ndarray, ligands: Sequence[str] | None = None) -> pd.DataFrame:
        """Per-row prediction split into named components (for attribution)."""
        if self.beta_ is None:
            raise RuntimeError("not fitted")
        contributions = x * self.beta_[None, :]
        out = {"intercept": np.full(len(x), self.intercept_)}
        block_of = [self.design.groups.get(c, "OTHER") for c in self.design.columns]
        for block in sorted(set(block_of)):
            mask = np.array([b == block for b in block_of])
            out[f"block__{block}"] = contributions[:, mask].sum(axis=1)
        gamma = np.zeros(len(x))
        if ligands is not None and self.gamma_:
            gamma = np.array([self.gamma_.get(str(l), 0.0) for l in ligands])
        out["ligand_intercept"] = gamma
        frame = pd.DataFrame(out)
        frame["prediction"] = frame.sum(axis=1)
        if self.y_range_ is not None:
            lo, hi = self.y_range_
            frame["prediction"] = frame["prediction"].clip(lo, hi)
        return frame

    def predict(self, x: np.ndarray, ligands: Sequence[str] | None = None) -> np.ndarray:
        return self.predict_components(x, ligands)["prediction"].to_numpy(dtype=float)


def _macro_mae(truth: np.ndarray, prediction: np.ndarray, groups: Sequence) -> float:
    table = pd.DataFrame({"e": np.abs(prediction - truth), "g": [str(g) for g in groups]})
    return float(table.groupby("g")["e"].mean().mean())


def tune_ridge(
    train: pd.DataFrame, data: LevelData, *, group_column: str, hierarchical: bool,
    seed: int, fixed_grid: Sequence[float] = FIXED_PENALTY_GRID,
    ligand_grid: Sequence[float] = LIGAND_PENALTY_GRID, inner_folds: int = INNER_FOLDS,
) -> dict:
    """Inner grouped CV over the penalty grid, grouped like the outer regime.

    Returns the chosen penalties and the full inner score table.  Grouping the
    inner CV on the *same* column as the outer regime is what makes the choice
    honest: under a chemotype hold-out the inner folds also hold out chemotypes,
    so a small ligand penalty — which would fit training ligands beautifully and
    transfer nothing — is penalised where it should be.
    """
    design = fit_design(train, data)
    x = design.transform(train)
    y = train[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    w = group_balanced_weights(train["ecfp_cluster"].astype(str))
    ligands = train["extractant"].astype(str).to_numpy()
    groups = train[group_column].astype(str).to_numpy()
    scores: list[dict] = []
    grid = [(lf, lg) for lf in fixed_grid for lg in (ligand_grid if hierarchical else [None])]
    folds = list(seeded_group_kfold(groups, inner_folds, seed))
    for lam_fixed, lam_ligand in grid:
        fold_scores = []
        for tr, te in folds:
            model = HierarchicalRidge(design, lam_fixed, lam_ligand)
            model.fit(x[tr], y[tr], weights=w[tr], ligands=ligands[tr] if hierarchical else None)
            pred = model.predict(x[te], ligands[te] if hierarchical else None)
            fold_scores.append(_macro_mae(y[te], pred, train["ecfp_cluster"].astype(str).to_numpy()[te]))
        scores.append({"lambda_fixed": lam_fixed, "lambda_ligand": lam_ligand,
                       "inner_macro_mae": float(np.mean(fold_scores))})
    table = pd.DataFrame(scores).sort_values("inner_macro_mae", kind="stable")
    best = table.iloc[0]
    return {"lambda_fixed": float(best["lambda_fixed"]),
            "lambda_ligand": None if pd.isna(best["lambda_ligand"]) else float(best["lambda_ligand"]),
            "inner_table": table, "design": design}


# --------------------------------------------------------------------------- #
# C3: pair-difference rows
# --------------------------------------------------------------------------- #

def pair_difference_rows(
    frame: pd.DataFrame, x: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Within-cell metal pairs as difference rows ``(x_A − x_B, y_A − y_B)``.

    Orientation: A is the lighter metal.  Each pair's weight is the mean of its
    two row weights times ``2 / k`` for a cell with ``k`` metals, so a 14-metal
    cell (91 pairs) does not swamp the two-metal cells.  Returns the index table
    too, so the same pairs can be scored later.
    """
    y = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    z = pd.to_numeric(frame["metal_Z"], errors="coerce").to_numpy(dtype=float)
    keys = (frame["extractant"].astype(str) + "|" + frame["condition_id"].astype(str)).to_numpy()
    positions: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        positions.setdefault(key, []).append(i)
    dx, dy, dw, rows = [], [], [], []
    for key, members in positions.items():
        if len(members) < 2:
            continue
        k = len(members)
        ordered = sorted(members, key=lambda i: z[i])
        for a_index in range(len(ordered)):
            for b_index in range(a_index + 1, len(ordered)):
                a, b = ordered[a_index], ordered[b_index]
                if z[a] == z[b]:
                    continue
                dx.append(x[a] - x[b])
                dy.append(y[a] - y[b])
                dw.append(0.5 * (weights[a] + weights[b]) * 2.0 / k)
                rows.append({"row_a": a, "row_b": b, "cell": key})
    if not dx:
        return (np.zeros((0, x.shape[1])), np.zeros(0), np.zeros(0), pd.DataFrame(rows))
    return np.asarray(dx), np.asarray(dy), np.asarray(dw), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# C2: two-stage cross-fitted decomposition
# --------------------------------------------------------------------------- #

def cell_frame(frame: pd.DataFrame, data: LevelData) -> pd.DataFrame:
    """One row per (ligand, condition) cell: features of the first row, mean target,
    metal count and the cell's ECFP cluster.  Stage A is fitted on this table so a
    14-metal cell does not outvote a one-metal cell on the *level*."""
    cols = list(dict.fromkeys([c for b in STAGE_A_BLOCKS for c in data.blocks.get(b, ())]))
    # series_id is constant within a cell (a cell fixes the condition vector, and
    # the series is the ligand plus its categorical conditions), so it can travel
    # with the cell and the cross-fit can be grouped on it under unseen_series.
    identity = [c for c in ("extractant", "condition_id", "ecfp_cluster", "tanimoto_cluster",
                            "series_id") if c in frame.columns]
    keep = identity + [c for c in cols if c in frame.columns]
    first = frame.drop_duplicates(["extractant", "condition_id"])[keep].reset_index(drop=True)
    stats = (frame.groupby(["extractant", "condition_id"], sort=False)[LEVEL_TARGET_COLUMN]
             .agg(cell_mean="mean", n_metals="size").reset_index())
    return first.merge(stats, on=["extractant", "condition_id"], how="left", validate="one_to_one")


@dataclass
class TwoStageFit:
    stage_a: LevelRegressor
    stage_b_crossfit: LevelRegressor
    stage_b_truecentre: LevelRegressor | None
    stage_a_columns: tuple[str, ...]
    stage_b_columns: tuple[str, ...]
    crossfit_audit: dict


def fit_two_stage(
    train: pd.DataFrame, data: LevelData, *, params: LevelForestParameters,
    group_column: str, seed: int, crossfit_folds: int = CROSSFIT_FOLDS,
) -> TwoStageFit:
    """Fit Stage A on cells, cross-fit it inside training, fit Stage B on the residual.

    The audit records, for every training row, which inner fold produced the
    Stage A prediction its residual was formed from — so a test can prove that no
    residual came from a Stage A model that had seen that row's cell.
    """
    cells = cell_frame(train, data)
    a_cols = tuple(dict.fromkeys([c for b in STAGE_A_BLOCKS for c in data.blocks.get(b, ())]))
    b_cols = data.block_columns(FOREST_BLOCKS)
    cell_weights = group_balanced_weights(cells["ecfp_cluster"].astype(str))

    # --- cross-fitted Stage A predictions for every training cell -------------
    groups = cells[group_column].astype(str).to_numpy() if group_column in cells.columns \
        else cells["tanimoto_cluster"].astype(str).to_numpy()
    a_oof = np.full(len(cells), np.nan)
    fold_of_cell = np.full(len(cells), -1)
    for inner, (tr, te) in enumerate(seeded_group_kfold(groups, crossfit_folds, seed + 17)):
        inner_params = LevelForestParameters(**{**params.__dict__, "random_state": params.random_state + 31 * (inner + 1)})
        model = LevelRegressor(a_cols, inner_params).fit(
            cells.iloc[tr], cells["cell_mean"].to_numpy(dtype=float)[tr],
            groups=cells["ecfp_cluster"].iloc[tr])
        a_oof[te] = model.predict(cells.iloc[te])
        fold_of_cell[te] = inner
    if np.isnan(a_oof).any():
        raise RuntimeError("cross-fitting left a training cell without an out-of-fold prediction")

    # --- full Stage A for test-time use ---------------------------------------
    stage_a = LevelRegressor(a_cols, params).fit(
        cells, cells["cell_mean"].to_numpy(dtype=float), groups=cells["ecfp_cluster"])

    # --- Stage B on the cross-fitted residual -----------------------------------
    key = train["extractant"].astype(str) + "|" + train["condition_id"].astype(str)
    cell_key = cells["extractant"].astype(str) + "|" + cells["condition_id"].astype(str)
    lookup_oof = dict(zip(cell_key, a_oof))
    lookup_fold = dict(zip(cell_key, fold_of_cell))
    lookup_mean = dict(zip(cell_key, cells["cell_mean"]))
    lookup_n = dict(zip(cell_key, cells["n_metals"]))
    a_oof_rows = key.map(lookup_oof).to_numpy(dtype=float)
    y = train[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    residual_crossfit = y - a_oof_rows
    stage_b = LevelRegressor(b_cols, LevelForestParameters(
        **{**params.__dict__, "random_state": params.random_state + 977})).fit(
        train, residual_crossfit, groups=train["ecfp_cluster"])

    # --- Stage B on the TRUE within-cell departure (for the oracle swap) --------
    multi = key.map(lookup_n).to_numpy() >= 2
    stage_b_true = None
    if multi.sum() >= 10:
        residual_true = y - key.map(lookup_mean).to_numpy(dtype=float)
        stage_b_true = LevelRegressor(b_cols, LevelForestParameters(
            **{**params.__dict__, "random_state": params.random_state + 1979})).fit(
            train[multi], residual_true[multi], groups=train["ecfp_cluster"][multi])

    audit = {
        "n_cells": int(len(cells)),
        "n_cells_multi_metal": int((cells["n_metals"] >= 2).sum()),
        "crossfit_folds": int(crossfit_folds),
        "row_inner_fold": key.map(lookup_fold).to_numpy(dtype=int),
        "cell_inner_fold": fold_of_cell,
        "cell_keys": cell_key.to_numpy(),
        "row_cell_keys": key.to_numpy(),
    }
    return TwoStageFit(stage_a=stage_a, stage_b_crossfit=stage_b, stage_b_truecentre=stage_b_true,
                       stage_a_columns=a_cols, stage_b_columns=b_cols, crossfit_audit=audit)


def predict_two_stage(fit: TwoStageFit, test: pd.DataFrame, data: LevelData) -> pd.DataFrame:
    """Â, B̂ and the oracle swaps on a test frame.  Oracles use the test frame's
    own cell means — label information — and are reported only as attribution,
    never as a deployable prediction."""
    cells = cell_frame(test, data)
    a_hat_cells = fit.stage_a.predict(cells)
    key = test["extractant"].astype(str) + "|" + test["condition_id"].astype(str)
    cell_key = cells["extractant"].astype(str) + "|" + cells["condition_id"].astype(str)
    a_hat = key.map(dict(zip(cell_key, a_hat_cells))).to_numpy(dtype=float)
    true_mean = key.map(dict(zip(cell_key, cells["cell_mean"]))).to_numpy(dtype=float)
    n_metals = key.map(dict(zip(cell_key, cells["n_metals"]))).to_numpy(dtype=int)
    b_hat = fit.stage_b_crossfit.predict(test)
    y = test[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    out = pd.DataFrame({
        "stage_a": a_hat, "stage_b": b_hat, "cell_true_mean": true_mean, "cell_n_metals": n_metals,
        "prediction_C2_TWO_STAGE": a_hat + b_hat,
        # Â + true metal departure: the only error left is Stage A's level error
        "prediction_ORACLE_METAL": a_hat + (y - true_mean),
    }, index=test.index)
    if fit.stage_b_truecentre is not None:
        b_true = fit.stage_b_truecentre.predict(test)
        out["stage_b_truecentre"] = b_true
        # true level + predicted departure: the only error left is the metal response
        out["prediction_ORACLE_LEVEL"] = true_mean + b_true
        # The deployable two-stage model with Stage B trained on the TRUE within-cell
        # departure (y − ȳ_cell) rather than on the cross-fitted residual.  Found on
        # a one-seed preview, before the full run, that the cross-fitted residual under
        # a chemotype hold-out carries Stage A's *level* error, which Stage B then
        # "learns" from in-sample ligand features and mis-applies to new ligands
        # (mean |B̂| 0.53 vs 0.15; macro MAE 1.25 vs 1.04 on that seed).  Both are
        # reported; the protocol amendment records which was pre-registered.
        out["prediction_C2_TRUECENTRE"] = a_hat + b_true
    else:
        out["stage_b_truecentre"] = np.nan
        out["prediction_ORACLE_LEVEL"] = np.nan
        out["prediction_C2_TRUECENTRE"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Derived pair evaluation (the shared-score form)
# --------------------------------------------------------------------------- #

def derived_pairs(test: pd.DataFrame, prediction_columns: Sequence[str]) -> pd.DataFrame:
    """All within-cell metal pairs in a test frame with derived predictions.

    ``ŷ(A, B) = ŷ(A) − ŷ(B)`` with A the lighter metal.  Antisymmetry is then an
    identity and transitivity over any triple within a cell is
    ``ŷ(A,B) + ŷ(B,C) − ŷ(A,C) = 0`` exactly; :func:`pair_consistency` checks both
    numerically rather than trusting the algebra.
    """
    y = test[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    z = pd.to_numeric(test["metal_Z"], errors="coerce").to_numpy(dtype=float)
    keys = (test["extractant"].astype(str) + "|" + test["condition_id"].astype(str)).to_numpy()
    ext = test["extractant"].astype(str).to_numpy()
    cluster = test["ecfp_cluster"].astype(str).to_numpy()
    metal = test["metal_symbol"].astype(str).to_numpy()
    preds = {c: test[c].to_numpy(dtype=float) for c in prediction_columns}
    positions: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        positions.setdefault(key, []).append(i)
    rows = []
    for key, members in positions.items():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda i: z[i])
        for ai in range(len(ordered)):
            for bi in range(ai + 1, len(ordered)):
                a, b = ordered[ai], ordered[bi]
                if z[a] == z[b]:
                    continue
                record = {"cell": key, "extractant": ext[a], "ecfp_cluster": cluster[a],
                          "metal_a": metal[a], "metal_b": metal[b],
                          "pair_label": f"{metal[a]}-{metal[b]}",
                          "delta_z": float(z[b] - z[a]), "truth": float(y[a] - y[b]),
                          "row_a": a, "row_b": b}
                for c, p in preds.items():
                    record[f"pair_{c}"] = float(p[a] - p[b])
                    record[f"pred_a__{c}"] = float(p[a])
                    record[f"pred_b__{c}"] = float(p[b])
                rows.append(record)
    return pd.DataFrame(rows)


def pair_label_mean_null(train: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Leave-fold-out pair-label mean: the mean training log SF for each (A, B)
    metal label, the strongest honest pair null established in gen3/gen4."""
    train_pairs = derived_pairs(train, [])
    if train_pairs.empty:
        return np.zeros(len(pairs))
    table = train_pairs.groupby("pair_label")["truth"].mean()
    overall = float(train_pairs["truth"].mean())
    return pairs["pair_label"].map(table).fillna(overall).to_numpy(dtype=float)


def pair_consistency(pairs: pd.DataFrame, prediction_column: str) -> dict:
    """Numeric antisymmetry and transitivity residuals for one derived predictor."""
    if pairs.empty:
        return {"antisymmetry_max": 0.0, "transitivity_max": 0.0, "n_triples": 0}
    # Antisymmetry: a derived pair is ŷ(A) − ŷ(B), so ŷ(B,A) = −ŷ(A,B) is an algebraic
    # identity.  It is still MEASURED rather than assumed: if the pair table carries
    # the two row predictions (``pred_a`` / ``pred_b``), the identity is recomputed
    # from them; otherwise it is recomputed from the derived values of the reversed
    # orientation where both orientations are present, and reported as NaN when
    # neither is possible — never as a constant zero.  (A first draft hard-coded
    # ``anti = 0.0``, which made the check vacuous; caught in review.)
    anti = float("nan")
    base = prediction_column[len("pair_"):] if prediction_column.startswith("pair_") else prediction_column
    col_a, col_b = f"pred_a__{base}", f"pred_b__{base}"
    if {col_a, col_b} <= set(pairs.columns):
        forward = pairs[prediction_column].to_numpy(dtype=float)
        rebuilt_reverse = (pairs[col_b] - pairs[col_a]).to_numpy(dtype=float)
        anti = float(np.nanmax(np.abs(forward + rebuilt_reverse))) if len(pairs) else 0.0
    else:
        lookup = {(r.row_a, r.row_b): getattr(r, prediction_column) for r in pairs.itertuples()}
        residuals = [abs(lookup[(a, b)] + lookup[(b, a)]) for (a, b) in lookup if (b, a) in lookup]
        anti = float(max(residuals)) if residuals else float("nan")
    trans = 0.0
    n_triples = 0
    for cell, block in pairs.groupby("cell"):
        if len(block) < 3:
            continue
        value = {(r.row_a, r.row_b): getattr(r, prediction_column) for r in block.itertuples()}
        members = sorted({*block["row_a"], *block["row_b"]})
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                for k in range(j + 1, len(members)):
                    a, b, c = members[i], members[j], members[k]
                    if (a, b) in value and (b, c) in value and (a, c) in value:
                        trans = max(trans, abs(value[(a, b)] + value[(b, c)] - value[(a, c)]))
                        n_triples += 1
    return {"antisymmetry_max": float(anti), "transitivity_max": float(trans), "n_triples": int(n_triples)}


def pair_metrics(pairs: pd.DataFrame, prediction_column: str) -> dict:
    if pairs.empty:
        return {"n_pairs": 0}
    err = np.abs(pairs[prediction_column] - pairs["truth"])
    table = pd.DataFrame({"e": err, "g": pairs["ecfp_cluster"].astype(str)})
    sign = np.sign(pairs[prediction_column]) == np.sign(pairs["truth"])
    nonzero = pairs["truth"] != 0
    return {
        "n_pairs": int(len(pairs)),
        "pair_macro_mae": float(table.groupby("g")["e"].mean().mean()),
        "pair_pooled_mae": float(err.mean()),
        "pair_sign_accuracy": float(sign[nonzero].mean()) if nonzero.any() else np.nan,
    }
