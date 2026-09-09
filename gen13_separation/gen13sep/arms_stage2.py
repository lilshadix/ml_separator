"""Stage-2 exploratory arms (``S2_*``), written after the locked Gen13 ladder was scored.

Every arm here is exploratory in the pre-registration's sense: it is reported with an ``S2_``
prefix, it is never promoted to the primary endpoint, and it is fitted on the same frozen fold
plan, the same cohort and the same feature blocks as the locked ladder so that a contrast against
``C_DIRECT_ROW`` / ``M_SELECTED`` is paired pair-for-pair.

The gaps in the locked ladder these arms address, each stated as the defect it targets:

* **The row control learns the level it then throws away.**  ``C_DIRECT_ROW`` fits trees on raw
  ``log D``, of which 89 % of the variance is the per-cell level (DATA_AUDIT §3) and only 11 % is
  the lanthanide-axis curve that the target actually is.  Every split the trees spend on the level
  is capacity not spent on the curve.  ``CentredRowArm`` fits the same learner on the *centred*
  row target ``log D(Ln) - mean_observed(log D)``, which is the quantity the metric differences.
  It keeps the row model's freedom (no rank-1/2/3 basis constraint) without the level.
* **The learner minimises squared error while the metric is MAE.**  ``aggregate="median"`` takes
  the median over the forest's trees instead of the mean, which is the L1-consistent readout of
  the same fitted ensemble and costs nothing extra to fit.
* **A two-metal cell and a fourteen-metal cell carry the same weight.**  A cell's curve target is
  determined by its observed metals; a two-metal cell fixes one contrast and its basis coefficient
  is shrunk 30-40 % by the per-cell ridge, yet it enters the coefficient regression with the same
  chemotype-balanced weight as a full cell.  ``reliability_weights`` multiplies the chemotype
  weight by ``n_obs / (n_obs + k)`` and renormalises *within* each chemotype, so chemotype shares
  are exactly preserved and only the within-chemotype allocation changes.
* **The metal side of the row model is two numbers.**  ``C_DIRECT_ROW`` gives the trees ``Z`` and
  the Shannon CN8 radius.  ``metal_features="rich"`` adds CN9 radius, hydration free energy, the
  4f count and the two Jorgensen tetrad shapes, so a row model can express non-smooth structure
  along the series if there is any.
* **Bagging is equal-weight.**  ``StackArm`` chooses convex weights over its members on the
  inner-validation chemotypes with the same one-vote-per-chemotype macro MAE that ``SelectedArm``
  uses, so it can interpolate between members instead of choosing one or averaging blindly.

Nothing here reads a held-out row.  Selection, weighting and gains are all decided on the inner
split of the training fold.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .basis import centre_rows, curves, fit_all_coefficients
from .metals import (ATOMIC_NUMBER, F_COUNT, HYDRATION_FREE_ENERGY_KJ, LANTHANIDES,
                     SHANNON_RADIUS_CN8, SHANNON_RADIUS_CN9, jorgensen_e1, jorgensen_e3)
from .models import Arm, FitContext, chemotype_balanced_weights, tree_pipeline

N_LN = len(LANTHANIDES)


def _standardise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def metal_feature_matrix(kind: str = "rich") -> np.ndarray:
    """``14 x p`` target-free descriptors of the lanthanide itself.

    ``basic`` reproduces ``C_DIRECT_ROW``'s two columns (Z and the raw CN8 radius) exactly, so a
    ``basic`` centred-row arm isolates the effect of centring the target and nothing else.
    """
    z = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES], dtype=float)
    r8 = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)
    if kind == "basic":
        return np.c_[z, r8]
    if kind != "rich":
        raise ValueError(f"metal_features must be 'basic' or 'rich', got {kind!r}")
    r9 = np.array([SHANNON_RADIUS_CN9[m] for m in LANTHANIDES], dtype=float)
    ghyd = np.array([HYDRATION_FREE_ENERGY_KJ[m] for m in LANTHANIDES], dtype=float)
    q = np.array([F_COUNT[m] for m in LANTHANIDES], dtype=float)
    return np.c_[z, r8, _standardise(r9), _standardise(ghyd), q,
                 _standardise(jorgensen_e1(q)), _standardise(jorgensen_e3(q))]


def reliability_weights(groups: Sequence, n_observed: np.ndarray, *, k: float = 2.0) -> np.ndarray:
    """Chemotype-balanced weights re-allocated inside each chemotype by ``n_obs / (n_obs + k)``.

    The total weight of every chemotype is exactly what ``chemotype_balanced_weights`` gives it,
    so the chemotype-macro character of the fit is unchanged; only the share a poorly determined
    cell takes inside its own chemotype falls.
    """
    base = chemotype_balanced_weights(groups)
    n_observed = np.asarray(n_observed, dtype=float)
    raw = base * (n_observed / (n_observed + k))
    frame = pd.DataFrame({"g": list(groups), "base": base, "raw": raw})
    per = frame.groupby("g")[["base", "raw"]].transform("sum")
    scale = per["base"].to_numpy() / np.maximum(per["raw"].to_numpy(), 1e-12)
    return raw * scale


def _forest_median(pipeline, X: np.ndarray) -> np.ndarray:
    """Median over the forest's trees instead of the mean (the L1 readout of the same fit)."""
    imputer, forest = pipeline[0], pipeline[-1]
    Xi = imputer.transform(X)
    stack = np.stack([tree.predict(Xi) for tree in forest.estimators_], axis=0)
    return np.median(stack, axis=0)


class CentredRowArm(Arm):
    """Trees on ``(cell features, metal descriptors)`` predicting the **centred** row value.

    Target for a training row is ``log D(cell, Ln) - mean over the metals observed in that cell``,
    i.e. the cell's centred curve evaluated at that metal -- exactly the quantity whose differences
    the metric scores.  Predictions are centred over the 14 metals before being returned; a
    constant per-cell offset cancels in every pairwise difference, so the centring convention at
    prediction time cannot change a score.
    """

    def __init__(self, *, metal_features: str = "rich", aggregate: str = "mean",
                 weights: str = "balanced", n_estimators: int = 400, max_features: float = 0.5,
                 min_samples_leaf: int = 2, name: str | None = None):
        if aggregate not in ("mean", "median"):
            raise ValueError("aggregate must be 'mean' or 'median'")
        if weights not in ("balanced", "reliability"):
            raise ValueError("weights must be 'balanced' or 'reliability'")
        self.metal_features = metal_features
        self.aggregate = aggregate
        self.weights = weights
        self.params = dict(n_estimators=n_estimators, max_features=max_features,
                           min_samples_leaf=min_samples_leaf)
        suffix = "" if metal_features == "rich" else f"_{metal_features}"
        suffix += "" if aggregate == "mean" else "_median"
        suffix += "" if weights == "balanced" else "_rel"
        self.name = name or f"S2_CENTRED_ROW{suffix}"

    def _rows(self, X: np.ndarray, C: np.ndarray | None):
        n = len(X)
        ci = np.repeat(np.arange(n), N_LN)
        mi = np.tile(np.arange(N_LN), n)
        feats = np.hstack([X[ci], self.metals_[mi]])
        if C is None:
            return feats, ci, mi, None
        y = C[ci, mi]
        keep = ~np.isnan(y)
        return feats[keep], ci[keep], mi[keep], y[keep]

    def fit(self, ctx: FitContext):
        self.metals_ = metal_feature_matrix(self.metal_features)
        C = ctx.centred_train                      # centred over each cell's observed metals
        feats, ci, mi, y = self._rows(ctx.X_train, C)
        n_obs = (~np.isnan(ctx.Y_train)).sum(axis=1).astype(float)
        cell_w = (chemotype_balanced_weights(ctx.groups_train) if self.weights == "balanced"
                  else reliability_weights(ctx.groups_train, n_obs))
        w = cell_w[ci] / n_obs[ci]                 # a cell keeps its weight, spread over its metals
        w = w * len(w) / w.sum()
        self.model_ = tree_pipeline(ctx.seed, **self.params)
        self.model_.fit(feats, y, extratreesregressor__sample_weight=w)
        return self

    def predict_curves(self, X: np.ndarray, fingerprints=None) -> np.ndarray:
        feats, _, _, _ = self._rows(X, None)
        raw = (self.model_.predict(feats) if self.aggregate == "mean"
               else _forest_median(self.model_, feats))
        p = np.asarray(raw, dtype=float).reshape(len(X), N_LN)
        return p - p.mean(axis=1, keepdims=True)


class MedianCoefficientArm(Arm):
    """Any coefficient-predicting arm, read out with the forest median instead of the mean.

    Wraps an arm that owns a ``model_`` pipeline and a ``basis_`` after ``fit`` (the low-rank and
    physics arms do).  Fitting is untouched; only the aggregation over trees changes, which is the
    cheapest available move toward the L1 metric.
    """

    def __init__(self, inner: Arm, *, name: str | None = None):
        self.inner = inner
        self.name = name or f"S2_MED[{inner.name}]"

    def fit(self, ctx: FitContext):
        self.inner.fit(ctx)
        if not hasattr(self.inner, "model_") or not hasattr(self.inner, "basis_"):
            raise TypeError(f"{self.inner.name} has no model_/basis_ to read out")
        return self

    def predict_curves(self, X: np.ndarray, fingerprints=None) -> np.ndarray:
        drop = getattr(self.inner, "drop_", None)
        Xp = X
        if drop is not None and len(drop):
            keep = np.ones(X.shape[1], dtype=bool)
            keep[drop] = False
            Xp = X[:, keep]
        coef = _forest_median(self.inner.model_, Xp)
        return curves(np.asarray(coef).reshape(len(X), self.inner.basis_.shape[0]), self.inner.basis_)


class ReliabilityWeightedArm(Arm):
    """Refit a curve arm with ``reliability_weights`` in place of the plain chemotype weights.

    Implemented by monkey-free delegation: the inner arm is fitted on a context whose chemotype
    labels are unchanged but whose per-cell weight is applied by duplicating the arm's own
    weighting call.  Only arms that build their weights from ``chemotype_balanced_weights`` on
    ``ctx.groups_train`` are supported, which is every tree-based curve arm in the ladder.
    """

    def __init__(self, inner: Arm, *, k: float = 2.0, name: str | None = None):
        self.inner = inner
        self.k = k
        self.name = name or f"S2_REL[{inner.name}]"

    def fit(self, ctx: FitContext):
        n_obs = (~np.isnan(ctx.Y_train)).sum(axis=1).astype(float)
        w = reliability_weights(ctx.groups_train, n_obs, k=self.k)
        # Re-express the reliability weight as a change of the *group* multiset is impossible, so
        # the inner arm is fitted directly here for the two shapes we support.
        basis_arm = hasattr(self.inner, "basis_") or hasattr(self.inner, "rank") or hasattr(self.inner, "names")
        if not basis_arm:
            raise TypeError(f"{self.inner.name} is not a basis arm")
        self.inner.fit(ctx)                      # establishes basis_ and drop_ the same way
        coef = fit_all_coefficients(ctx.centred_train, self.inner.basis_)
        Xp = ctx.X_train
        drop = getattr(self.inner, "drop_", None)
        if drop is not None and len(drop):
            keep = np.ones(Xp.shape[1], dtype=bool)
            keep[drop] = False
            Xp = Xp[:, keep]
        self.inner.model_ = tree_pipeline(ctx.seed, **getattr(self.inner, "params", {}))
        self.inner.model_.fit(Xp, coef, extratreesregressor__sample_weight=w * len(w) / w.sum())
        return self

    def predict_curves(self, X: np.ndarray, fingerprints=None) -> np.ndarray:
        return self.inner.predict_curves(X, fingerprints)


class StackArm(Arm):
    """Convex combination of member curves with weights chosen on the inner-validation chemotypes.

    The score minimised is the same one ``SelectedArm`` uses: mean over inner-validation chemotypes
    of the mean absolute pairwise error, one vote per chemotype.  Weights are searched on a simplex
    grid of step ``grid_step``; the members are then refitted on the full training fold.  With two
    members and step 0.1 this is 11 evaluations, none of which touches a held-out row.
    """

    def __init__(self, members: Sequence[Arm], *, grid_step: float = 0.1, name: str = "S2_STACK"):
        self.members = list(members)
        self.grid_step = grid_step
        self.name = name
        self.selection_: list[dict] = []

    @staticmethod
    def _simplex(n: int, step: float) -> list[tuple[float, ...]]:
        levels = int(round(1.0 / step))

        def rec(k: int, left: int) -> list[tuple[int, ...]]:
            if k == 1:
                return [(left,)]
            out: list[tuple[int, ...]] = []
            for i in range(left + 1):
                out.extend((i,) + rest for rest in rec(k - 1, left - i))
            return out

        return [tuple(v / levels for v in combo) for combo in rec(n, levels)]

    @staticmethod
    def _macro_pair_mae(curve: np.ndarray, Y: np.ndarray, groups: np.ndarray) -> float:
        errs: dict = {}
        for i in range(len(Y)):
            obs = np.flatnonzero(~np.isnan(Y[i]))
            for a in obs:
                for b in obs:
                    if a < b:
                        errs.setdefault(groups[i], []).append(
                            abs((Y[i, a] - Y[i, b]) - (curve[i, a] - curve[i, b])))
        return float(np.mean([np.mean(v) for v in errs.values()]))

    def fit(self, ctx: FitContext):
        if ctx.inner_train is None or ctx.inner_validation is None:
            raise ValueError("StackArm needs the inner split positions")
        it, iv = ctx.inner_train, ctx.inner_validation
        inner_ctx = ctx.subset(it)
        member_curves = []
        for m in self.members:
            m.fit(inner_ctx)
            member_curves.append(m.predict_curves(ctx.X_train[iv], ctx.fingerprints_train[iv]))
        Y_val, g_val = ctx.Y_train[iv], ctx.groups_train[iv]
        best, best_w = np.inf, None
        for wts in self._simplex(len(self.members), self.grid_step):
            blend = sum(w * c for w, c in zip(wts, member_curves))
            score = self._macro_pair_mae(blend, Y_val, g_val)
            if score < best:
                best, best_w = score, wts
        self.weights_ = best_w
        self.selection_.append({"selected": "+".join(f"{m.name}:{w:.2f}" for m, w in zip(self.members, best_w)),
                                "inner_score": best})
        for m in self.members:
            m.fit(ctx)
        return self

    def predict_curves(self, X: np.ndarray, fingerprints=None) -> np.ndarray:
        return sum(w * m.predict_curves(X, fingerprints) for w, m in zip(self.weights_, self.members))


def stage2_arms() -> list[Arm]:
    """The stage-2 ladder: centring, L1 readout, reliability weights and inner-validated stacking.

    ``C_DIRECT_ROW`` and ``M_PHYSICS_radius+radius_sq`` are added by the runner script as the
    paired references, so every contrast is computed on byte-identical pairs.
    """
    from .models import DirectRowArm, LowRankArm, PhysicsBasisArm

    def phys() -> Arm:
        return PhysicsBasisArm(("radius", "radius_sq"), name="M_PHYSICS_radius+radius_sq")

    return [
        CentredRowArm(metal_features="basic", name="S2_CENTRED_ROW_basic"),
        CentredRowArm(metal_features="rich", name="S2_CENTRED_ROW"),
        CentredRowArm(metal_features="rich", aggregate="median", name="S2_CENTRED_ROW_median"),
        CentredRowArm(metal_features="rich", weights="reliability", name="S2_CENTRED_ROW_rel"),
        MedianCoefficientArm(phys(), name="S2_MED_PHYSICS"),
        MedianCoefficientArm(LowRankArm(rank=2), name="S2_MED_LOWRANK_K2"),
        ReliabilityWeightedArm(phys(), name="S2_REL_PHYSICS"),
        StackArm([DirectRowArm(), phys()], name="S2_STACK_DIRECT+PHYSICS"),
        StackArm([CentredRowArm(metal_features="rich", name="cr"), phys()], name="S2_STACK_CENTRED+PHYSICS"),
    ]


# ---------------------------------------------------------------------------
# Stage-3 arms: the hierarchical split the variance decomposition asks for
# ---------------------------------------------------------------------------

def extractant_key(X: np.ndarray, blocks: dict, *, block: str = "ECFP") -> np.ndarray:
    """Identify the extractant of every cell without an identity column.

    The ligand blocks are constant within an extractant by construction, so the fingerprint rows
    partition the cells exactly by extractant.  Grouping on a feature the model already sees is a
    modelling choice, not a leak: no target, provenance or held-out value is consulted.
    """
    cols = np.asarray(blocks[block], dtype=int)
    fp = np.nan_to_num(X[:, cols], nan=-1.0).astype(np.float32)
    return np.array([hash(row.tobytes()) for row in fp])


class HierarchicalCurveArm(Arm):
    """Ligand level plus condition deviation, fitted as two separate models.

    On the cells whose curve is well determined, 86 % of the sum of squares of the radius-ramp
    amplitude lies **between** extractants and the rest lies between the conditions of one
    extractant (``analysis/stage2``).  A single flat regression has to learn both at once from a
    corpus where one chemotype holds 375 of 521 cells, so the between part is estimated from cells
    that disagree with each other for reasons the ligand features cannot express.

    This arm splits the target instead:

    * the **ligand model** is fitted on one row per training extractant -- the shrunk mean basis
      coefficient of that extractant's cells -- from the ligand blocks only, so an extractant
      counts once however many cells it contributed;
    * the **condition model** is fitted on the within-extractant deviations of extractants with at
      least two cells, from the condition blocks only, so it can only learn what conditions do at
      fixed chemistry.

    The prediction is their sum.  ``shrink`` is the pseudo-count pulling a sparsely measured
    extractant's mean toward its chemotype mean; ``use_conditions=False`` gives the ligand-only
    ablation that isolates the effect of aggregating to the extractant level.
    """

    def __init__(self, *, basis_names: tuple[str, ...] = ("radius", "radius_sq"),
                 condition_blocks: tuple[str, ...] = ("COND", "MASSACT"),
                 shrink: float = 2.0, use_conditions: bool = True, n_estimators: int = 400,
                 max_features: float = 0.5, min_samples_leaf: int = 2, name: str | None = None):
        self.basis_names = tuple(basis_names)
        self.condition_blocks = tuple(condition_blocks)
        self.shrink = shrink
        self.use_conditions = use_conditions
        self.params = dict(n_estimators=n_estimators, max_features=max_features,
                           min_samples_leaf=min_samples_leaf)
        self.name = name or ("S3_HIER" if use_conditions else "S3_EXT_LEVEL")

    def fit(self, ctx: FitContext):
        from .basis import physics_basis_matrix
        blocks = ctx.extra.get("blocks") or {}
        if not blocks:
            raise KeyError("HierarchicalCurveArm needs ctx.extra['blocks']")
        self.basis_ = physics_basis_matrix(self.basis_names)
        coef = fit_all_coefficients(ctx.centred_train, self.basis_)          # cells x k
        key = extractant_key(ctx.X_train, blocks)
        self.cond_cols_ = np.concatenate([np.asarray(blocks[b], dtype=int) for b in self.condition_blocks])
        self.lig_cols_ = np.setdiff1d(np.arange(ctx.X_train.shape[1]), self.cond_cols_)

        frame = pd.DataFrame({"key": key, "chemotype": ctx.groups_train})
        for j in range(coef.shape[1]):
            frame[f"c{j}"] = coef[:, j]
        cols = [f"c{j}" for j in range(coef.shape[1])]
        per_ext = frame.groupby("key")[cols].mean()
        n_ext = frame.groupby("key").size()
        chemo_of = frame.groupby("key")["chemotype"].first()
        per_chemo = frame.groupby("chemotype")[cols].mean()
        prior = per_chemo.reindex(chemo_of.to_numpy()).to_numpy()
        shrunk = ((n_ext.to_numpy()[:, None] * per_ext.to_numpy() + self.shrink * prior)
                  / (n_ext.to_numpy()[:, None] + self.shrink))

        first = frame.reset_index().groupby("key")["index"].first().reindex(per_ext.index).to_numpy()
        X_ext = ctx.X_train[first][:, self.lig_cols_]
        w_ext = chemotype_balanced_weights(chemo_of.reindex(per_ext.index).to_numpy())
        self.ligand_model_ = tree_pipeline(ctx.seed, **self.params)
        self.ligand_model_.fit(X_ext, shrunk, extratreesregressor__sample_weight=w_ext)

        self.condition_model_ = None
        if self.use_conditions:
            mean_of = per_ext.reindex(frame["key"].to_numpy()).to_numpy()      # unshrunk: dev = 0 if n_e = 1
            dev = coef - mean_of
            multi = n_ext.reindex(frame["key"].to_numpy()).to_numpy() >= 2
            if multi.sum() >= 20:
                w_dev = chemotype_balanced_weights(ctx.groups_train[multi])
                self.condition_model_ = tree_pipeline(ctx.seed + 1, **self.params)
                self.condition_model_.fit(ctx.X_train[multi][:, self.cond_cols_], dev[multi],
                                          extratreesregressor__sample_weight=w_dev)
        return self

    def predict_curves(self, X: np.ndarray, fingerprints=None) -> np.ndarray:
        k = self.basis_.shape[0]
        coef = np.asarray(self.ligand_model_.predict(X[:, self.lig_cols_]), dtype=float).reshape(len(X), k)
        if self.condition_model_ is not None:
            coef = coef + np.asarray(self.condition_model_.predict(X[:, self.cond_cols_]),
                                     dtype=float).reshape(len(X), k)
        return curves(coef, self.basis_)


def stage3_arms() -> list[Arm]:
    """Hierarchical arms plus their two ablations, all on the physics basis."""
    return [
        HierarchicalCurveArm(use_conditions=False, name="S3_EXT_LEVEL"),
        HierarchicalCurveArm(use_conditions=True, name="S3_HIER"),
        HierarchicalCurveArm(use_conditions=True, shrink=0.0, name="S3_HIER_noshrink"),
        HierarchicalCurveArm(use_conditions=True, shrink=8.0, name="S3_HIER_shrink8"),
    ]


def extractant_balanced_weights(groups: Sequence, keys: np.ndarray) -> np.ndarray:
    """Equal weight per chemotype, split equally between that chemotype's extractants.

    The headline metric is a macro average over *extractants* inside a macro average over seeds,
    while the training weights are macro over *chemotypes* only.  Inside the diglycolamide
    chemotype that lets one extractant with 171 cells take 46 % of the chemotype's weight, so the
    fit is tuned to one molecule that the metric counts once.  This weight matches the metric one
    level deeper without changing the chemotype shares.
    """
    frame = pd.DataFrame({"g": list(groups), "k": list(keys)})
    n_chemo = frame["g"].nunique()
    ext_per_chemo = frame.groupby("g")["k"].transform("nunique").to_numpy()
    cells_per_ext = frame.groupby(["g", "k"])["k"].transform("size").to_numpy()
    w = 1.0 / (n_chemo * ext_per_chemo * cells_per_ext)
    return w * len(w) / w.sum()


class ExtractantBalancedArm(Arm):
    """Refit a basis arm with :func:`extractant_balanced_weights` instead of chemotype weights."""

    def __init__(self, inner: Arm, *, name: str | None = None):
        self.inner = inner
        self.name = name or f"S3_EXTW[{inner.name}]"

    def fit(self, ctx: FitContext):
        blocks = ctx.extra.get("blocks") or {}
        keys = extractant_key(ctx.X_train, blocks)
        self.inner.fit(ctx)                                  # establishes basis_ / drop_ / params
        coef = fit_all_coefficients(ctx.centred_train, self.inner.basis_)
        Xp = ctx.X_train
        drop = getattr(self.inner, "drop_", None)
        if drop is not None and len(drop):
            keep = np.ones(Xp.shape[1], dtype=bool)
            keep[drop] = False
            Xp = Xp[:, keep]
        w = extractant_balanced_weights(ctx.groups_train, keys)
        self.inner.model_ = tree_pipeline(ctx.seed, **getattr(self.inner, "params", {}))
        self.inner.model_.fit(Xp, coef, extratreesregressor__sample_weight=w)
        return self

    def predict_curves(self, X: np.ndarray, fingerprints=None) -> np.ndarray:
        return self.inner.predict_curves(X, fingerprints)


def stage3_weight_arms() -> list[Arm]:
    from .models import LowRankArm, PhysicsBasisArm
    return [
        ExtractantBalancedArm(PhysicsBasisArm(("radius", "radius_sq"), name="M_PHYSICS_radius+radius_sq"),
                              name="S3_EXTW_PHYSICS"),
        ExtractantBalancedArm(LowRankArm(rank=2), name="S3_EXTW_LOWRANK_K2"),
    ]


# ---------------------------------------------------------------------------
# Stage-4: the centred-row learner inside the block-subset bag that won B_v2
# ---------------------------------------------------------------------------

def stage4_arms() -> list[Arm]:
    """`V2_BAG_MIX3` / `V2_BAG_MIX5` with the row member swapped for the centred-row learner.

    The two changes are independent: `B_v2` fixed *which blocks* the members see (dropping the
    fingerprint blocks), this module fixes *what the row learner spends its capacity on* (the
    centred curve rather than the per-cell level).  Both the original bags and the swapped bags are
    returned so the contrast is paired inside one run, on byte-identical pairs.
    """
    from .models import BagArm, BlockSubsetArm, DirectRowArm, LowRankArm, PhysicsBasisArm

    cond = ("COND", "MASSACT")
    lean = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "COORD")

    def phys(tag: str) -> Arm:
        return PhysicsBasisArm(("radius", "radius_sq"), name=f"p@{tag}")

    def centred(tag: str) -> Arm:
        return CentredRowArm(metal_features="rich", name=f"cr@{tag}")

    return [
        BagArm([BlockSubsetArm(DirectRowArm(), cond, name="d@cond"),
                BlockSubsetArm(phys("lean"), lean, name="p@lean"),
                BlockSubsetArm(LowRankArm(rank=2), lean, name="k2@lean")], name="V2_BAG_MIX3"),
        BagArm([BlockSubsetArm(centred("cond"), cond, name="cr@cond"),
                BlockSubsetArm(phys("lean"), lean, name="p@lean"),
                BlockSubsetArm(LowRankArm(rank=2), lean, name="k2@lean")], name="S4_BAG_MIX3C"),
        BagArm([BlockSubsetArm(DirectRowArm(), cond, name="d@cond"),
                BlockSubsetArm(DirectRowArm(), lean, name="d@lean"),
                BlockSubsetArm(phys("lean"), lean, name="p@lean"),
                BlockSubsetArm(LowRankArm(rank=2), lean, name="k2@lean"),
                BlockSubsetArm(LowRankArm(rank=1), lean, name="k1@lean")], name="V2_BAG_MIX5"),
        BagArm([BlockSubsetArm(centred("cond"), cond, name="cr@cond"),
                BlockSubsetArm(centred("lean"), lean, name="cr@lean"),
                BlockSubsetArm(phys("lean"), lean, name="p@lean"),
                BlockSubsetArm(LowRankArm(rank=2), lean, name="k2@lean"),
                BlockSubsetArm(LowRankArm(rank=1), lean, name="k1@lean")], name="S4_BAG_MIX5C"),
        BlockSubsetArm(centred("lean"), lean, name="S4_CENTRED_ROW@lean"),
    ]
