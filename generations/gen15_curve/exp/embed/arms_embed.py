"""Gen15 arms driven by a per-ligand representation (pretrained LM embedding or Morgan counts).

Every arm below is the gen14 skeleton with exactly one part replaced, so a gain can only come from
the representation:

    coef = ( sign(direction) * magnitude , curvature )

* ``direction_arm``  swaps gen14's 39-column topology logistic for a logistic on the representation;
  everything else (training-fold mean magnitude, training-fold mean curvature) is gen14's.
* ``magnitude_arm``  keeps gen14's direction and replaces the constant magnitude with a
  ridge / kernel-ridge prediction of ``log|a|`` from the representation.
* ``curvature_arm``  keeps gen14's direction and constant magnitude and predicts ``b``.
* ``composite_arm``  the representation on both halves.

The representation is a deterministic function of the SMILES, so it is precomputed for all 521
cells; the *fitting* -- imputation, standardisation, PCA, the penalty -- happens strictly inside the
training fold of every fold of every design, which is where the leakage discipline has to live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                    # noqa: E402
from gen15.arms import _curve, _logistic_sign               # noqa: E402
from gen15.valuebench import Ctx                            # noqa: E402

EPS = 0.05


# --------------------------------------------------------------------------------------
# estimator fronts
# --------------------------------------------------------------------------------------
def _front(n_pca: int | None, seed: int = 0):
    f = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if n_pca:
        f += [PCA(n_components=n_pca, random_state=seed), StandardScaler()]
    return f


def _logit_p(Xtr, y, w, Xte, C: float, n_pca: int | None, seed: int) -> np.ndarray:
    if len(set(y.tolist())) < 2:
        return np.full(len(Xte), float(y.mean()))
    n_pca = None if n_pca is None else int(min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1]))
    m = make_pipeline(*_front(n_pca, seed),
                      LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
    m.fit(Xtr, y, logisticregression__sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def _emb_sign(ctx: Ctx, name: str, C: float, n_pca: int | None) -> np.ndarray:
    X = FEAT.cells(ctx, name)
    rtr = ctx.rich_train()
    p = _logit_p(X[rtr], (ctx.amp[rtr] < 0).astype(int), ctx.rich_weights(), X[ctx.test],
                 C, n_pca, ctx.model_seed)
    return np.where(p >= 0.5, -1.0, 1.0)


def _ridge_predict(Xtr, y, w, Xte, alpha: float, n_pca: int | None, seed: int) -> np.ndarray:
    n_pca = None if n_pca is None else int(min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1]))
    m = make_pipeline(*_front(n_pca, seed), Ridge(alpha=alpha))
    m.fit(Xtr, y, ridge__sample_weight=w)
    return np.asarray(m.predict(Xte), dtype=float)


def _kernel_ridge_predict(Xtr, y, w, Xte, alpha: float, kernel: str, gamma_scale: float,
                          center: bool = True) -> np.ndarray:
    """Weighted kernel ridge with an RBF or cosine kernel on standardised columns.

    ``sklearn``'s KernelRidge takes ``sample_weight``, so the chemotype balancing is kept.  The RBF
    length scale is set from the median pairwise distance of the *training* fold, which is the
    standard scale-free choice and uses no held-out information.
    """
    sc = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    A = sc.fit_transform(Xtr)
    Bm = sc.transform(Xte)
    mu = float(np.average(y, weights=w)) if center else 0.0
    if kernel == "cosine":
        A, Bm = normalize(A), normalize(Bm)
        m = KernelRidge(alpha=alpha, kernel="linear")
    else:
        from scipy.spatial.distance import pdist
        d = pdist(A[:min(len(A), 400)])
        med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma_scale / max(med ** 2, 1e-9))
    m.fit(A, y - mu, sample_weight=w)
    return np.asarray(m.predict(Bm), dtype=float) + mu


# --------------------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------------------
def direction_arm(name: str, C: float = 1.0, n_pca: int | None = None):
    """Gen14 with the direction called from the representation instead of TOPO39."""
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(_emb_sign(ctx, name, C, n_pca) * ctx.train_mean_magnitude(),
                      ctx.train_mean_curvature(), len(ctx.test))
    return f


def magnitude_arm(name: str, how: str = "ridge", alpha: float = 10.0, n_pca: int | None = None,
                  gamma_scale: float = 1.0, direction: str = "g14"):
    """Gen14's direction, magnitude regressed from the representation on ``log(|a| + eps)``."""
    def f(ctx: Ctx) -> np.ndarray:
        s = _logistic_sign(ctx) if direction == "g14" else _emb_sign(ctx, name, 1.0, n_pca)
        X = FEAT.cells(ctx, name)
        rtr, w = ctx.rich_train(), ctx.rich_weights()
        y = np.log(np.abs(ctx.amp[rtr]) + EPS)
        if how == "ridge":
            pred = _ridge_predict(X[rtr], y, w, X[ctx.test], alpha, n_pca, ctx.model_seed)
        else:
            pred = _kernel_ridge_predict(X[rtr], y, w, X[ctx.test], alpha, how, gamma_scale)
        lo, hi = np.quantile(y, [0.02, 0.98])
        mag = np.clip(np.exp(np.clip(pred, lo, hi)) - EPS, 0.0, None)
        return _curve(s * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def curvature_arm(name: str, how: str = "ridge", alpha: float = 10.0, n_pca: int | None = None,
                  gamma_scale: float = 1.0):
    """Gen14's amplitude, curvature regressed (``how='ridge'/'rbf'/'cosine'``) or classified."""
    def f(ctx: Ctx) -> np.ndarray:
        s = _logistic_sign(ctx)
        X = FEAT.cells(ctx, name)
        rtr, w = ctx.rich_train(), ctx.rich_weights()
        b_tr = ctx.cur[rtr]
        if how == "sign":
            p = _logit_p(X[rtr], (b_tr < 0).astype(int), w, X[ctx.test], alpha, n_pca,
                         ctx.model_seed)
            sb = np.where(p >= 0.5, -1.0, 1.0)
            b = sb * float(np.average(np.abs(b_tr), weights=w))
        elif how == "ridge":
            b = _ridge_predict(X[rtr], b_tr, w, X[ctx.test], alpha, n_pca, ctx.model_seed)
        else:
            b = _kernel_ridge_predict(X[rtr], b_tr, w, X[ctx.test], alpha, how, gamma_scale)
        lo, hi = np.quantile(b_tr, [0.02, 0.98])
        return _curve(s * ctx.train_mean_magnitude(), np.clip(b, lo, hi), len(ctx.test))
    return f


def composite_arm(name: str, C: float = 1.0, dir_pca: int | None = None, how: str = "ridge",
                  alpha: float = 10.0, mag_pca: int | None = None, gamma_scale: float = 1.0):
    """The representation on the direction *and* the magnitude; curvature stays the training mean."""
    def f(ctx: Ctx) -> np.ndarray:
        s = _emb_sign(ctx, name, C, dir_pca)
        X = FEAT.cells(ctx, name)
        rtr, w = ctx.rich_train(), ctx.rich_weights()
        y = np.log(np.abs(ctx.amp[rtr]) + EPS)
        if how == "ridge":
            pred = _ridge_predict(X[rtr], y, w, X[ctx.test], alpha, mag_pca, ctx.model_seed)
        else:
            pred = _kernel_ridge_predict(X[rtr], y, w, X[ctx.test], alpha, how, gamma_scale)
        lo, hi = np.quantile(y, [0.02, 0.98])
        mag = np.clip(np.exp(np.clip(pred, lo, hi)) - EPS, 0.0, None)
        return _curve(s * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def concat_arm(name: str, C: float = 1.0, n_pca: int | None = 16):
    """Direction from the representation *concatenated with* gen14's 39 topology columns.

    The fair question is not only "is the embedding better than TOPO39" but "does it add anything
    TOPO39 does not already have".
    """
    def f(ctx: Ctx) -> np.ndarray:
        E = FEAT.cells(ctx, name)
        T = ctx.feat("TOPO39")
        rtr, w = ctx.rich_train(), ctx.rich_weights()
        y = (ctx.amp[rtr] < 0).astype(int)
        if len(set(y.tolist())) < 2:
            s = np.ones(len(ctx.test))
        else:
            k = None if n_pca is None else int(min(n_pca, len(rtr) - 1, E.shape[1]))
            pre = make_pipeline(*_front(k, ctx.model_seed))
            Etr = pre.fit_transform(E[rtr])
            Ete = pre.transform(E[ctx.test])
            Xtr = np.c_[Etr, T[rtr]]
            Xte = np.c_[Ete, T[ctx.test]]
            p = _logit_p(Xtr, y, w, Xte, C, None, ctx.model_seed)
            s = np.where(p >= 0.5, -1.0, 1.0)
        return _curve(s * ctx.train_mean_magnitude(), ctx.train_mean_curvature(), len(ctx.test))
    return f
