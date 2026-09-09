"""Arms.  Every arm sees the same training cells and predicts, for each held-out
cell, either a centred 14-vector curve (``predict_curves``) or, for pairwise-only
arms, a value per requested (cell, A, B) triple (``predict_pairs``).  The runner
turns curves into pairwise ``log SF`` predictions so that every arm is scored on
byte-identical pairs.

Learners are deliberately the repository's proven ones: extremely randomised
trees with a median imputer and chemotype-balanced sample weights.  Nothing here
selects on a held-out row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from .basis import centre_rows, curves, fit_all_coefficients, fit_data_basis, physics_basis_matrix
from .metals import ATOMIC_NUMBER, LANTHANIDES, SHANNON_RADIUS_CN8

N_LN = len(LANTHANIDES)
Z_VEC = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES], dtype=float)
R_VEC = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)


def chemotype_balanced_weights(groups: Sequence) -> np.ndarray:
    counts = pd.Series(list(groups)).value_counts()
    w = np.array([1.0 / counts[g] for g in groups], dtype=float)
    return w * len(w) / w.sum()


def tree_pipeline(seed: int, *, n_estimators: int = 400, max_features: float = 0.5,
                  min_samples_leaf: int = 2) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesRegressor(n_estimators=n_estimators, max_features=max_features,
                            min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=-1),
    )


@dataclass
class FitContext:
    """Everything an arm may use: training cells only."""
    X_train: np.ndarray
    Y_train: np.ndarray            # training cells x 14 raw log D (NaN = unmeasured)
    groups_train: np.ndarray       # chemotype per training cell
    fingerprints_train: np.ndarray  # ECFP bits for the NN baseline
    seed: int
    inner_train: np.ndarray | None = None        # positions into the training arrays
    inner_validation: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def subset(self, positions: np.ndarray) -> "FitContext":
        return FitContext(X_train=self.X_train[positions], Y_train=self.Y_train[positions],
                          groups_train=self.groups_train[positions],
                          fingerprints_train=self.fingerprints_train[positions], seed=self.seed)

    @property
    def centred_train(self) -> np.ndarray:
        return centre_rows(self.Y_train)


class Arm:
    name: str = "ARM"
    pairwise_only: bool = False

    def fit(self, ctx: FitContext) -> "Arm":
        raise NotImplementedError

    def predict_curves(self, X: np.ndarray, fingerprints: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def predict_pairs(self, X: np.ndarray, ia: np.ndarray, ib: np.ndarray, cell_rows: np.ndarray) -> np.ndarray:
        curve = self.predict_curves(X)
        return curve[cell_rows, ia] - curve[cell_rows, ib]


class ZeroArm(Arm):
    name = "B0_ZERO"

    def fit(self, ctx):
        return self

    def predict_curves(self, X, fingerprints=None):
        return np.zeros((len(X), N_LN))


class MeanCurveArm(Arm):
    """Corpus mean centred curve of the training fold (chemotype-balanced)."""
    name = "B1_MEAN_CURVE"

    def fit(self, ctx):
        C = ctx.centred_train
        w = chemotype_balanced_weights(ctx.groups_train)
        num = np.nansum(C * w[:, None], axis=0)
        den = np.nansum((~np.isnan(C)) * w[:, None], axis=0)
        self.curve_ = num / np.maximum(den, 1e-12)
        return self

    def predict_curves(self, X, fingerprints=None):
        return np.tile(self.curve_, (len(X), 1))


class PairMeanArm(Arm):
    """Per pair-type mean log SF over training pairs (no ligand information)."""
    name = "B2_PAIRMEAN"
    pairwise_only = True

    def fit(self, ctx):
        Y = ctx.Y_train
        w = chemotype_balanced_weights(ctx.groups_train)
        num = np.zeros((N_LN, N_LN)); den = np.zeros((N_LN, N_LN))
        for i in range(len(Y)):
            obs = np.flatnonzero(~np.isnan(Y[i]))
            for a in obs:
                for b in obs:
                    if a < b:
                        num[a, b] += w[i] * (Y[i, a] - Y[i, b]); den[a, b] += w[i]
        self.table_ = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        return self

    def predict_pairs(self, X, ia, ib, cell_rows):
        return self.table_[ia, ib]


class HeavierAlwaysArm(Arm):
    """'Heavier lanthanide always preferred': prediction = -(training mean |log SF| of the pair
    type), so the sign is always heavy-preferred and the magnitude is typical.  It exists so the
    sign-accuracy yardstick is scored with the same unit and weighting as every other arm."""
    name = "B4_HEAVIER_ALWAYS"
    pairwise_only = True

    def fit(self, ctx):
        Y = ctx.Y_train
        w = chemotype_balanced_weights(ctx.groups_train)
        num = np.zeros((N_LN, N_LN)); den = np.zeros((N_LN, N_LN))
        for i in range(len(Y)):
            obs = np.flatnonzero(~np.isnan(Y[i]))
            for a in obs:
                for b in obs:
                    if a < b:
                        num[a, b] += w[i] * abs(Y[i, a] - Y[i, b]); den[a, b] += w[i]
        self.table_ = -np.where(den > 0, num / np.maximum(den, 1e-12), 0.3)
        return self

    def predict_pairs(self, X, ia, ib, cell_rows):
        return self.table_[ia, ib]


class NearestNeighbourArm(Arm):
    """Curve of the training cell whose extractant is nearest in Tanimoto (rank-K basis)."""
    name = "B3_NN_TANIMOTO"

    def __init__(self, rank: int = 2):
        self.rank = rank

    def fit(self, ctx):
        C = ctx.centred_train
        self.basis_ = fit_data_basis(C, self.rank)
        self.coef_ = fit_all_coefficients(C, self.basis_)
        self.fp_ = ctx.fingerprints_train.astype(bool)
        return self

    def predict_curves(self, X, fingerprints=None):
        if fingerprints is None:
            raise ValueError("NN arm needs fingerprints")
        q = fingerprints.astype(bool)
        inter = q.astype(np.int32) @ self.fp_.astype(np.int32).T
        union = q.sum(1)[:, None] + self.fp_.sum(1)[None, :] - inter
        sim = inter / np.maximum(union, 1)
        # average the coefficients of all training cells tied at the maximum similarity
        best = sim.max(axis=1, keepdims=True)
        tie = (sim >= best - 1e-12).astype(float)
        coef = (tie @ self.coef_) / tie.sum(1, keepdims=True)
        return curves(coef, self.basis_)


class LowRankArm(Arm):
    """Rank-K data basis fitted in-fold; trees predict the K coefficients from cell features."""

    def __init__(self, rank: int = 2, *, name: str | None = None, n_estimators: int = 400,
                 max_features: float = 0.5, min_samples_leaf: int = 2):
        self.rank = rank
        self.name = name or f"M_LOWRANK_K{rank}"
        self.params = dict(n_estimators=n_estimators, max_features=max_features,
                           min_samples_leaf=min_samples_leaf)

    def fit(self, ctx):
        C = ctx.centred_train
        self.basis_ = fit_data_basis(C, self.rank)
        coef = fit_all_coefficients(C, self.basis_)
        self.model_ = tree_pipeline(ctx.seed, **self.params)
        w = chemotype_balanced_weights(ctx.groups_train)
        self.model_.fit(ctx.X_train, coef, extratreesregressor__sample_weight=w)
        return self

    def predict_curves(self, X, fingerprints=None):
        return curves(np.asarray(self.model_.predict(X)).reshape(len(X), self.basis_.shape[0]), self.basis_)


class PhysicsBasisArm(Arm):
    """Fixed physics basis; trees predict its coefficients."""

    def __init__(self, names: tuple[str, ...] = ("radius", "radius_sq"), *, name: str | None = None,
                 n_estimators: int = 400, max_features: float = 0.5, min_samples_leaf: int = 2):
        self.names = tuple(names)
        self.name = name or "M_PHYSICS_" + "+".join(self.names)
        self.params = dict(n_estimators=n_estimators, max_features=max_features,
                           min_samples_leaf=min_samples_leaf)

    def fit(self, ctx):
        self.basis_ = physics_basis_matrix(self.names)
        coef = fit_all_coefficients(ctx.centred_train, self.basis_)
        self.model_ = tree_pipeline(ctx.seed, **self.params)
        w = chemotype_balanced_weights(ctx.groups_train)
        # a physics arm never sees the opt-in RESP3D block: it is the paired reference for the helpers
        blocks = ctx.extra.get("blocks") or {}
        self.drop_ = np.asarray(sorted(set(np.concatenate([np.asarray(blocks.get(b, []), dtype=int)
                                                            for b in ("RESP3D", "LOGK")]).tolist())), dtype=int)
        self.model_.fit(_drop_columns(ctx.X_train, self.drop_), coef, extratreesregressor__sample_weight=w)
        return self

    def predict_curves(self, X, fingerprints=None):
        Xp = _drop_columns(X, self.drop_) if len(getattr(self, "drop_", [])) else X
        return curves(np.asarray(self.model_.predict(Xp)).reshape(len(X), self.basis_.shape[0]), self.basis_)


class DirectRowArm(Arm):
    """gen5-gen12 style row model: trees on log D with metal descriptors, differenced in-cell.

    The level is learned and then cancelled by the difference, so this is the
    control that asks whether modelling the curve directly beats modelling rows.
    """
    name = "C_DIRECT_ROW"

    def __init__(self, *, n_estimators: int = 400, max_features: float = 0.5, min_samples_leaf: int = 2):
        self.params = dict(n_estimators=n_estimators, max_features=max_features,
                           min_samples_leaf=min_samples_leaf)

    @staticmethod
    def _rows(X: np.ndarray, Y: np.ndarray | None):
        n = len(X)
        ci = np.repeat(np.arange(n), N_LN); mi = np.tile(np.arange(N_LN), n)
        feats = np.hstack([X[ci], Z_VEC[mi][:, None], R_VEC[mi][:, None]])
        if Y is None:
            return feats, ci, mi, None
        y = Y[ci, mi]
        keep = ~np.isnan(y)
        return feats[keep], ci[keep], mi[keep], y[keep]

    def fit(self, ctx):
        feats, ci, mi, y = self._rows(ctx.X_train, ctx.Y_train)
        n_obs = (~np.isnan(ctx.Y_train)).sum(axis=1).astype(float)
        # a cell keeps exactly its chemotype-balanced weight, spread over its observed metals,
        # so chemotype shares are identical to the curve arms (1/n_chemotypes each)
        w = chemotype_balanced_weights(ctx.groups_train)[ci] / n_obs[ci]
        w = w * len(w) / w.sum()
        self.model_ = tree_pipeline(ctx.seed, **self.params)
        self.model_.fit(feats, y, extratreesregressor__sample_weight=w)
        return self

    def predict_curves(self, X, fingerprints=None):
        feats, ci, mi, _ = self._rows(X, None)
        p = self.model_.predict(feats).reshape(len(X), N_LN)
        return p - p.mean(axis=1, keepdims=True)


class AntisymmetricPairArm(Arm):
    """gen2-gen4 style pair model: trees on (cell features, metal pair features), trained on all
    training pairs plus their reversals, predicted as ``(f(A,B) - f(B,A)) / 2``."""
    name = "C_PAIR_ANTISYM"
    pairwise_only = True

    def __init__(self, *, n_estimators: int = 200, max_features: float = 0.3, min_samples_leaf: int = 2,
                 max_pairs_per_cell: int | None = None):
        self.params = dict(n_estimators=n_estimators, max_features=max_features,
                           min_samples_leaf=min_samples_leaf)
        self.max_pairs_per_cell = max_pairs_per_cell

    @staticmethod
    def _feats(X: np.ndarray, ci: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
        za, zb, ra, rb = Z_VEC[ia], Z_VEC[ib], R_VEC[ia], R_VEC[ib]
        metal = np.c_[za, zb, zb - za, ra, rb, rb - ra, (za + zb) / 2.0, (ra + rb) / 2.0]
        return np.hstack([X[ci], metal]).astype(np.float32)

    def fit(self, ctx):
        Y = ctx.Y_train
        cis, ias, ibs, ys = [], [], [], []
        rng = np.random.default_rng(ctx.seed)
        for i in range(len(Y)):
            obs = np.flatnonzero(~np.isnan(Y[i]))
            pairs = [(a, b) for a in obs for b in obs if a < b]
            if self.max_pairs_per_cell and len(pairs) > self.max_pairs_per_cell:
                pairs = [pairs[j] for j in rng.choice(len(pairs), self.max_pairs_per_cell, replace=False)]
            for a, b in pairs:
                cis.append(i); ias.append(a); ibs.append(b); ys.append(Y[i, a] - Y[i, b])
        ci, ia, ib, y = map(np.asarray, (cis, ias, ibs, ys))
        Xp = np.vstack([self._feats(ctx.X_train, ci, ia, ib), self._feats(ctx.X_train, ci, ib, ia)])
        yp = np.concatenate([y, -y])
        n_pairs = np.bincount(ci, minlength=len(Y)).astype(float)
        w = chemotype_balanced_weights(ctx.groups_train)[ci] / n_pairs[ci]
        w = w * len(w) / w.sum()
        w = np.concatenate([w, w])
        self.model_ = tree_pipeline(ctx.seed, **self.params)
        self.model_.fit(Xp, yp, extratreesregressor__sample_weight=w)
        return self

    def predict_pairs(self, X, ia, ib, cell_rows):
        f = self.model_.predict(self._feats(X, cell_rows, ia, ib))
        r = self.model_.predict(self._feats(X, cell_rows, ib, ia))
        return (f - r) / 2.0


class SelectedArm(Arm):
    """Pre-registered selection: fit every candidate on the inner-training chemotypes, score on
    the inner-validation chemotypes (macro MAE over *chemotypes* of all pairwise log SF, one vote
    per validation chemotype), keep the best, refit it on the whole training fold.  No held-out
    row is ever consulted.  The inner training/validation sizes are recorded because the
    round-robin inner split can deal the 375-cell diglycolamide chemotype to validation."""

    def __init__(self, candidates: Sequence[Arm], *, name: str = "M_SELECTED"):
        self.candidates = list(candidates)
        self.name = name
        self.selection_: list[dict] = []

    @staticmethod
    def _inner_score(arm: Arm, ctx_in: FitContext, X_val, Y_val, fp_val, groups_val) -> float:
        arm.fit(ctx_in)
        errs: dict = {}
        if arm.pairwise_only:
            cells, ias, ibs, ys = [], [], [], []
            for i in range(len(Y_val)):
                obs = np.flatnonzero(~np.isnan(Y_val[i]))
                for a in obs:
                    for b in obs:
                        if a < b:
                            cells.append(i); ias.append(a); ibs.append(b); ys.append(Y_val[i, a] - Y_val[i, b])
            pred = arm.predict_pairs(X_val, np.array(ias), np.array(ibs), np.array(cells))
            for c, y, p in zip(cells, ys, pred):
                errs.setdefault(groups_val[c], []).append(abs(y - p))
        else:
            curve = arm.predict_curves(X_val, fp_val)
            for i in range(len(Y_val)):
                obs = np.flatnonzero(~np.isnan(Y_val[i]))
                for a in obs:
                    for b in obs:
                        if a < b:
                            errs.setdefault(groups_val[i], []).append(
                                abs((Y_val[i, a] - Y_val[i, b]) - (curve[i, a] - curve[i, b])))
        # one vote per inner-validation chemotype (extractant identity is not in the context)
        return float(np.mean([np.mean(v) for v in errs.values()]))

    def fit(self, ctx: FitContext):
        if ctx.inner_train is None or ctx.inner_validation is None:
            raise ValueError("SelectedArm needs inner_train / inner_validation positions")
        ctx_in = ctx.subset(ctx.inner_train)
        X_val = ctx.X_train[ctx.inner_validation]; Y_val = ctx.Y_train[ctx.inner_validation]
        fp_val = ctx.fingerprints_train[ctx.inner_validation]; g_val = ctx.groups_train[ctx.inner_validation]
        scores = {}
        for arm in self.candidates:
            scores[arm.name] = self._inner_score(arm, ctx_in, X_val, Y_val, fp_val, g_val)
        best = min(scores, key=scores.get)
        self.selection_.append({"selected": best, "n_inner_train": int(len(ctx.inner_train)),
                                "n_inner_validation": int(len(ctx.inner_validation)),
                                **{f"inner_{k}": v for k, v in scores.items()}})
        self.chosen_ = next(a for a in self.candidates if a.name == best)
        self.chosen_.fit(ctx)
        self.pairwise_only = self.chosen_.pairwise_only
        return self

    def predict_curves(self, X, fingerprints=None):
        return self.chosen_.predict_curves(X, fingerprints)

    def predict_pairs(self, X, ia, ib, cell_rows):
        return self.chosen_.predict_pairs(X, ia, ib, cell_rows)


class ProjectedDirectArm(DirectRowArm):
    """Exploratory: the direct row model's predicted curve projected onto the in-fold rank-K basis
    (a denoiser: keeps the part of the row model's curve that the corpus says is shape)."""

    def __init__(self, rank: int = 2, **kw):
        super().__init__(**kw)
        self.rank = rank
        self.name = f"X_DIRECT_PROJ_K{rank}"

    def fit(self, ctx):
        super().fit(ctx)
        self.basis_ = fit_data_basis(ctx.centred_train, self.rank)
        return self

    def predict_curves(self, X, fingerprints=None):
        c = super().predict_curves(X, fingerprints)
        coef = fit_all_coefficients(c, self.basis_, ridge=1e-6)
        return curves(coef, self.basis_)


class EnsembleArm(Arm):
    """Exploratory: equal-weight mean of member curves (members must be curve arms)."""

    def __init__(self, members: Sequence[Arm], *, name: str = "X_ENSEMBLE"):
        self.members = list(members)
        self.name = name

    def fit(self, ctx):
        for m in self.members:
            m.fit(ctx)
        return self

    def predict_curves(self, X, fingerprints=None):
        return np.mean([m.predict_curves(X, fingerprints) for m in self.members], axis=0)


class KernelCoefficientArm(Arm):
    """Exploratory: kernel ridge on a product kernel Tanimoto(ECFP) x RBF(standardised numeric
    conditions) predicting physics-basis coefficients.  A smooth-in-chemical-similarity inductive
    bias, the opposite of trees; no feature block other than the fingerprint and conditions."""

    def __init__(self, names: tuple[str, ...] = ("radius", "radius_sq"), *, alpha: float = 0.3,
                 gamma: float = 0.5, name: str | None = None):
        self.names = tuple(names); self.alpha = float(alpha); self.gamma = float(gamma)
        self.name = name or "X_KRR_TANIMOTO_" + "+".join(self.names)

    @staticmethod
    def _tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32); b = b.astype(np.float32)
        inter = a @ b.T
        return inter / np.maximum(a.sum(1)[:, None] + b.sum(1)[None, :] - inter, 1.0)

    def _cond(self, X: np.ndarray) -> np.ndarray:
        cols = self.cond_cols_
        Z = X[:, cols]
        Z = np.where(np.isnan(Z), self.cond_median_, Z)
        return (Z - self.cond_mean_) / self.cond_sd_

    def fit(self, ctx: FitContext):
        # numeric conditions = the columns of the context's 'cond_index' if provided, else the
        # first 72 columns (COND + MASSACT in the default block order) restricted to non-binary ones
        X = ctx.X_train
        idx = ctx.extra.get("cond_index")
        if idx is None:
            blocks = ctx.extra.get("blocks") or {}
            head_cols = np.concatenate([np.asarray(blocks[b], dtype=int) for b in ("COND", "MASSACT") if b in blocks])                 if any(b in blocks for b in ("COND", "MASSACT")) else np.arange(min(72, X.shape[1]))
            nonbin = [int(j) for j in head_cols if len(np.unique(X[~np.isnan(X[:, j]), j])) > 2]
            idx = np.array(nonbin, dtype=int)
        self.cond_cols_ = idx
        Z = X[:, idx]
        self.cond_median_ = np.nanmedian(Z, axis=0); self.cond_median_ = np.where(np.isnan(self.cond_median_), 0.0, self.cond_median_)
        Zf = np.where(np.isnan(Z), self.cond_median_, Z)
        self.cond_mean_ = Zf.mean(0); self.cond_sd_ = Zf.std(0) + 1e-9
        self.fp_ = ctx.fingerprints_train
        self.Zc_ = self._cond(X)
        self.basis_ = physics_basis_matrix(self.names)
        coef = fit_all_coefficients(ctx.centred_train, self.basis_)
        K = self._tanimoto(self.fp_, self.fp_) * np.exp(-self.gamma * ((self.Zc_[:, None, :] - self.Zc_[None, :, :]) ** 2).mean(-1))
        w = chemotype_balanced_weights(ctx.groups_train)
        # weighted kernel ridge: (K + alpha * W^-1) a = coef
        A = K + self.alpha * np.diag(1.0 / w)
        self.mean_ = np.average(coef, axis=0, weights=w)
        self.dual_ = np.linalg.solve(A, coef - self.mean_)
        return self

    def predict_curves(self, X, fingerprints=None):
        if fingerprints is None:
            raise ValueError("kernel arm needs fingerprints")
        Zq = self._cond(X)
        K = self._tanimoto(fingerprints, self.fp_) * np.exp(-self.gamma * ((Zq[:, None, :] - self.Zc_[None, :, :]) ** 2).mean(-1))
        coef = self.mean_ + K @ self.dual_
        return curves(coef, self.basis_)


class HybridPhysicsResidualArm(Arm):
    """Exploratory: physics basis (radius, radius^2) plus a rank-1 data basis fitted to the
    residual of the physics fit on training cells; trees predict all coefficients."""

    def __init__(self, *, n_estimators: int = 400, max_features: float = 0.5, min_samples_leaf: int = 2):
        self.name = "X_PHYSICS+RESID_K1"
        self.params = dict(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples_leaf)

    def fit(self, ctx: FitContext):
        C = ctx.centred_train
        pb = physics_basis_matrix(("radius", "radius_sq"))
        pc = fit_all_coefficients(C, pb)
        resid = C - curves(pc, pb)
        rb = fit_data_basis(resid, 1)
        self.basis_ = np.vstack([pb, rb])
        coef = fit_all_coefficients(C, self.basis_)
        self.model_ = tree_pipeline(ctx.seed, **self.params)
        self.model_.fit(ctx.X_train, coef, extratreesregressor__sample_weight=chemotype_balanced_weights(ctx.groups_train))
        return self

    def predict_curves(self, X, fingerprints=None):
        return curves(np.asarray(self.model_.predict(X)).reshape(len(X), self.basis_.shape[0]), self.basis_)


def _drop_columns(X: np.ndarray, cols: np.ndarray) -> np.ndarray:
    keep = np.ones(X.shape[1], dtype=bool); keep[cols] = False
    return X[:, keep]


class HelperBlockArm(Arm):
    """3D (or any block) as a *helper*, never the basis of the prediction.

    ``mode``:
      * ``add``      - base learner with the block appended (the naive test);
      * ``gated``    - inner validation compares base-without-block vs base-with-block and
                       keeps the block only if it wins by more than ``gate_margin``;
      * ``residual`` - base learner without the block; a ridge on the block columns alone
                       predicts the cross-fitted coefficient residuals of the base, added with
                       shrinkage ``lam``;
      * ``shuffled`` - as ``add`` but the block rows are permuted across training cells
                       (the width-matched null the repository demands).
    The base is a physics-basis tree model; the block is located through
    ``ctx.extra["blocks"][block]`` (column positions set by the runner).
    """

    def __init__(self, block: str = "RESP3D", mode: str = "gated", *, names: tuple[str, ...] = ("radius", "radius_sq"),
                 gate_margin: float = 0.0, lam: float = 0.5, ridge: float = 10.0, n_estimators: int = 400,
                 max_features: float = 0.5, min_samples_leaf: int = 2, name: str | None = None):
        if mode not in ("add", "gated", "residual", "shuffled"):
            raise ValueError(mode)
        self.block = block; self.mode = mode; self.names = tuple(names)
        self.gate_margin = float(gate_margin); self.lam = float(lam); self.ridge = float(ridge)
        self.params = dict(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples_leaf)
        self.name = name or f"X_{block}_{mode.upper()}"
        self.selection_: list[dict] = []

    def _cols(self, ctx: FitContext) -> np.ndarray:
        blocks = ctx.extra.get("blocks") or {}
        if self.block not in blocks:
            raise KeyError(f"block {self.block!r} not in ctx.extra[blocks]")
        return np.asarray(blocks[self.block], dtype=int)

    def _other_helper_cols(self, ctx: FitContext) -> np.ndarray:
        blocks = ctx.extra.get("blocks") or {}
        others = [np.asarray(blocks[b], dtype=int) for b in ("RESP3D", "LOGK") if b in blocks and b != self.block]
        return np.concatenate(others) if others else np.zeros(0, dtype=int)

    def _fit_base(self, X, coef, groups, seed):
        model = tree_pipeline(seed, **self.params)
        model.fit(X, coef, extratreesregressor__sample_weight=chemotype_balanced_weights(groups))
        return model

    @staticmethod
    def _score(pred_curves, Y, groups):
        errs: dict = {}
        for i in range(len(Y)):
            obs = np.flatnonzero(~np.isnan(Y[i]))
            for a in obs:
                for b in obs:
                    if a < b:
                        errs.setdefault(groups[i], []).append(abs((Y[i, a] - Y[i, b]) - (pred_curves[i, a] - pred_curves[i, b])))
        return float(np.mean([np.mean(v) for v in errs.values()]))

    def fit(self, ctx: FitContext):
        cols = self._cols(ctx)
        self.cols_ = cols
        self.other_ = self._other_helper_cols(ctx)
        self.basis_ = physics_basis_matrix(self.names)
        coef = fit_all_coefficients(ctx.centred_train, self.basis_)
        X = ctx.X_train.copy()
        if len(self.other_):
            X[:, self.other_] = np.nan          # the other opt-in block is neutralised (constant NaN -> imputed constant)
        if self.mode == "shuffled":
            rng = np.random.default_rng(ctx.seed + 17)
            X[:, cols] = X[rng.permutation(len(X))][:, cols]
        self.use_block_ = self.mode in ("add", "shuffled")
        if self.mode == "gated":
            if ctx.inner_train is None:
                raise ValueError("gated mode needs inner split positions")
            it, iv = ctx.inner_train, ctx.inner_validation
            Yv = ctx.Y_train[iv]; gv = ctx.groups_train[iv]
            without = self._fit_base(_drop_columns(X[it], cols), coef[it], ctx.groups_train[it], ctx.seed)
            with_ = self._fit_base(X[it], coef[it], ctx.groups_train[it], ctx.seed)
            s0 = self._score(curves(np.asarray(without.predict(_drop_columns(X[iv], cols))).reshape(len(iv), -1), self.basis_), Yv, gv)
            s1 = self._score(curves(np.asarray(with_.predict(X[iv])).reshape(len(iv), -1), self.basis_), Yv, gv)
            self.use_block_ = (s0 - s1) > self.gate_margin
            self.selection_.append({"selected": "with_block" if self.use_block_ else "without_block",
                                    "inner_without": s0, "inner_with": s1})
        if self.mode == "residual":
            it, iv = ctx.inner_train, ctx.inner_validation
            resid = np.zeros_like(coef)
            for tr_pos, te_pos in ((it, iv), (iv, it)):
                base = self._fit_base(_drop_columns(X[tr_pos], cols), coef[tr_pos], ctx.groups_train[tr_pos], ctx.seed)
                resid[te_pos] = coef[te_pos] - np.asarray(base.predict(_drop_columns(X[te_pos], cols))).reshape(len(te_pos), -1)
            Zb = X[:, cols]
            self.z_median_ = np.nanmedian(Zb, axis=0); self.z_median_ = np.where(np.isnan(self.z_median_), 0.0, self.z_median_)
            Zf = np.where(np.isnan(Zb), self.z_median_, Zb)
            self.z_mean_ = Zf.mean(0); self.z_sd_ = Zf.std(0) + 1e-9
            Zs = (Zf - self.z_mean_) / self.z_sd_
            w = chemotype_balanced_weights(ctx.groups_train)
            A = Zs.T @ (Zs * w[:, None]) + self.ridge * np.eye(Zs.shape[1])
            self.ridge_coef_ = np.linalg.solve(A, Zs.T @ (resid * w[:, None]))
            self.model_ = self._fit_base(_drop_columns(X, cols), coef, ctx.groups_train, ctx.seed)
            return self
        Xfit = X if self.use_block_ else _drop_columns(X, cols)
        self.model_ = self._fit_base(Xfit, coef, ctx.groups_train, ctx.seed)
        return self

    def predict_curves(self, X, fingerprints=None):
        cols = self.cols_
        if len(getattr(self, "other_", [])):
            X = X.copy(); X[:, self.other_] = np.nan
        if self.mode == "residual":
            base = np.asarray(self.model_.predict(_drop_columns(X, cols))).reshape(len(X), -1)
            Zb = X[:, cols]; Zf = np.where(np.isnan(Zb), self.z_median_, Zb); Zs = (Zf - self.z_mean_) / self.z_sd_
            coef = base + self.lam * (Zs @ self.ridge_coef_)
            return curves(coef, self.basis_)
        Xp = X if self.use_block_ else _drop_columns(X, cols)
        return curves(np.asarray(self.model_.predict(Xp)).reshape(len(X), -1), self.basis_)


def helper_arms(block: str = "RESP3D") -> list[Arm]:
    """Paired reference (physics arm without the block) plus the four helper modes for ``block``."""
    return [
        PhysicsBasisArm(("radius", "radius_sq"), name="M_PHYSICS_radius+radius_sq"),
        HelperBlockArm(block, "add"), HelperBlockArm(block, "gated"),
        HelperBlockArm(block, "residual"), HelperBlockArm(block, "shuffled"),
    ]


def helper3d_arms() -> list[Arm]:
    return helper_arms("RESP3D")


def default_ladder() -> list[Arm]:
    return [
        ZeroArm(), MeanCurveArm(), PairMeanArm(), HeavierAlwaysArm(), NearestNeighbourArm(rank=2),
        LowRankArm(rank=1), LowRankArm(rank=2), LowRankArm(rank=3),
        PhysicsBasisArm(("radius", "radius_sq")),
        PhysicsBasisArm(("radius", "radius_sq", "gd_break")),
        PhysicsBasisArm(("radius", "radius_sq", "tetrad_e1", "tetrad_e3")),
        DirectRowArm(),
    ]


def selected_arm() -> "SelectedArm":
    return SelectedArm([
        LowRankArm(rank=1), LowRankArm(rank=2), LowRankArm(rank=3),
        PhysicsBasisArm(("radius", "radius_sq")),
        PhysicsBasisArm(("radius", "radius_sq", "gd_break")),
        PhysicsBasisArm(("radius", "radius_sq", "tetrad_e1", "tetrad_e3")),
    ])


def exploratory_arms() -> list[Arm]:
    return [
        KernelCoefficientArm(), HybridPhysicsResidualArm(),
        ProjectedDirectArm(rank=2),
        EnsembleArm([DirectRowArm(), LowRankArm(rank=2)], name="X_ENS_DIRECT+LOWRANK_K2"),
        EnsembleArm([DirectRowArm(), PhysicsBasisArm(("radius", "radius_sq"))], name="X_ENS_DIRECT+PHYSICS"),
    ]


# ---------------------------------------------------------------------------
# gen13.1 wrappers (exploratory, labelled X_ / V2_): block subsets, bagging, dispersion calibration
# ---------------------------------------------------------------------------

def _keep_blocks(ctx: FitContext, keep: tuple[str, ...]) -> np.ndarray:
    blocks = ctx.extra.get("blocks") or {}
    missing = [b for b in keep if b not in blocks]
    if missing:
        raise KeyError(f"blocks {missing} not in ctx.extra[blocks]")
    return np.concatenate([np.asarray(blocks[b], dtype=int) for b in keep])


class BlockSubsetArm(Arm):
    """Run ``inner`` on the columns of ``keep`` blocks only (a feature-subset candidate)."""

    def __init__(self, inner: Arm, keep: tuple[str, ...], *, name: str | None = None):
        self.inner = inner; self.keep = tuple(keep)
        self.name = name or f"{inner.name}@{'+'.join(keep)}"
        self.pairwise_only = inner.pairwise_only

    def fit(self, ctx: FitContext):
        self.cols_ = _keep_blocks(ctx, self.keep)
        sub = FitContext(X_train=ctx.X_train[:, self.cols_], Y_train=ctx.Y_train, groups_train=ctx.groups_train,
                         fingerprints_train=ctx.fingerprints_train, seed=ctx.seed,
                         inner_train=ctx.inner_train, inner_validation=ctx.inner_validation,
                         extra={"blocks": _reindex_blocks(ctx, self.keep)})
        self.inner.fit(sub)
        self.selection_ = getattr(self.inner, "selection_", [])
        return self

    def predict_curves(self, X, fingerprints=None):
        return self.inner.predict_curves(X[:, self.cols_], fingerprints)

    def predict_pairs(self, X, ia, ib, cell_rows):
        return self.inner.predict_pairs(X[:, self.cols_], ia, ib, cell_rows)


def _reindex_blocks(ctx: FitContext, keep: tuple[str, ...]) -> dict:
    blocks = ctx.extra.get("blocks") or {}
    out = {}; start = 0
    for b in keep:
        n = len(blocks[b]); out[b] = np.arange(start, start + n); start += n
    return out


class BagArm(Arm):
    """Equal-weight average of member curves (members refit each fold); the bagging alternative to
    selecting one candidate by inner validation."""

    def __init__(self, members: Sequence[Arm], *, name: str = "V2_BAG"):
        self.members = list(members); self.name = name

    def fit(self, ctx: FitContext):
        for m in self.members:
            m.fit(ctx)
        return self

    def predict_curves(self, X, fingerprints=None):
        return np.mean([m.predict_curves(X, fingerprints) for m in self.members], axis=0)


class CalibratedGainArm(Arm):
    """Dispersion calibration: the inner arm's centred curve is multiplied by a gain ``g`` chosen on
    inner-validation pairs (grid), because tree predictions of contrasts are systematically shrunk.
    The gain is a single scalar per fold, chosen without any held-out row."""

    def __init__(self, inner: Arm, *, grid: Sequence[float] = (1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0),
                 name: str | None = None):
        self.inner = inner; self.grid = tuple(grid)
        self.name = name or f"V2_CAL[{inner.name}]"
        self.selection_: list[dict] = []

    @staticmethod
    def _score(curve, Y, groups, g):
        errs: dict = {}
        for i in range(len(Y)):
            obs = np.flatnonzero(~np.isnan(Y[i]))
            for a in obs:
                for b in obs:
                    if a < b:
                        errs.setdefault(groups[i], []).append(abs((Y[i, a] - Y[i, b]) - g * (curve[i, a] - curve[i, b])))
        return float(np.mean([np.mean(v) for v in errs.values()]))

    def fit(self, ctx: FitContext):
        if ctx.inner_train is None:
            raise ValueError("CalibratedGainArm needs inner split positions")
        it, iv = ctx.inner_train, ctx.inner_validation
        self.inner.fit(ctx.subset(it))
        curve_v = self.inner.predict_curves(ctx.X_train[iv], ctx.fingerprints_train[iv])
        scores = {g: self._score(curve_v, ctx.Y_train[iv], ctx.groups_train[iv], g) for g in self.grid}
        self.gain_ = min(scores, key=scores.get)
        self.selection_.append({"selected": f"gain={self.gain_}", **{f"inner_g{g}": s for g, s in scores.items()}})
        self.inner.fit(ctx)
        return self

    def predict_curves(self, X, fingerprints=None):
        return self.gain_ * self.inner.predict_curves(X, fingerprints)


def v2_arms() -> list[Arm]:
    """gen13.1 exploratory ladder: block subsets (cond = conditions only; lean = conditions +
    physchem + donors + coordination), bagging over bases/direct, gain calibration, mixed bags."""
    cond = ("COND", "MASSACT")
    lean = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "COORD")
    def phys(): return PhysicsBasisArm(("radius", "radius_sq"), name="M_PHYSICS_radius+radius_sq")
    def bag(): return BagArm([DirectRowArm(), phys(), LowRankArm(rank=1), LowRankArm(rank=2)], name="V2_BAG4")
    return [
        BlockSubsetArm(DirectRowArm(), cond, name="V2_DIRECT@cond"),
        BlockSubsetArm(phys(), cond, name="V2_PHYSICS@cond"),
        BlockSubsetArm(LowRankArm(rank=2), cond, name="V2_LOWRANK_K2@cond"),
        BlockSubsetArm(bag(), cond, name="V2_BAG4@cond"),
        BlockSubsetArm(DirectRowArm(), lean, name="V2_DIRECT@lean"),
        BlockSubsetArm(phys(), lean, name="V2_PHYSICS@lean"),
        BlockSubsetArm(LowRankArm(rank=2), lean, name="V2_LOWRANK_K2@lean"),
        BlockSubsetArm(bag(), lean, name="V2_BAG4@lean"),
        BlockSubsetArm(CalibratedGainArm(bag(), name="V2_CAL_BAG4"), lean, name="V2_CAL_BAG4@lean"),
        BagArm([BlockSubsetArm(DirectRowArm(), cond, name="d@cond"), BlockSubsetArm(phys(), lean, name="p@lean"),
                BlockSubsetArm(LowRankArm(rank=2), lean, name="k2@lean")], name="V2_BAG_MIX3"),
        BagArm([BlockSubsetArm(DirectRowArm(), cond, name="d@cond"), BlockSubsetArm(DirectRowArm(), lean, name="d@lean"),
                BlockSubsetArm(phys(), lean, name="p@lean"), BlockSubsetArm(LowRankArm(rank=2), lean, name="k2@lean"),
                BlockSubsetArm(LowRankArm(rank=1), lean, name="k1@lean")], name="V2_BAG_MIX5"),
        bag(),
    ]
