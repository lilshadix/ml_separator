"""Similarity-kernel arms on molecular fingerprints and RDKit descriptors.

The cohort has carried an ECFP block (2048 Morgan bits) and a LIG2D block (206 RDKit descriptors)
since gen6, but they have only ever been *columns* inside a ~2500-column extra-trees matrix, where
gen13's audit found they diluted the compact chemistry blocks.  A similarity kernel is a different
estimator, not a different feature set: it never selects columns, it has one effective smoothing
parameter, and it is the standard thing to reach for at n ~ 80-300 distinct ligands.

Everything here plugs into the gen15 value bench as an ``arm(ctx) -> (n_test, 2)`` so the numbers
are byte-comparable with G14 (0.500 extractant-macro MAE under BP) and FLAT (0.589).

Composition rules, so that a gain can be attributed:

* a *direction* arm keeps gen14's magnitude (the training fold's mean |a|) and gen14's curvature
  (the training fold's mean b) and replaces only the sign -- so ``D_* - G14`` is the value of the
  new direction call and nothing else;
* a *magnitude* arm keeps gen14's direction and curvature and replaces only |a|;
* a *curvature* arm keeps gen14's a entirely and replaces only b;
* a *joint* arm predicts (a, b) outright.

Grams.  The Tanimoto (Jaccard) similarity on binary vectors is positive definite, so it is a
legitimate kernel for ``kernel='precomputed'``; it is computed once over all 521 cells because it
uses no fitted statistic and so cannot leak.  The RDKit descriptor kernels DO use fitted statistics
(median imputation + standardisation + a median-heuristic length scale), so they are refitted on
the training fold of every fold and cached per fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "gen15_curve") not in sys.path:
    sys.path.insert(0, str(ROOT / "gen15_curve"))

from gen15.arms import _curve, _logistic_sign  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402

EPS = 0.05
#: corpus replicate sd propagated onto the radius coefficient (gen13 DATA_AUDIT); this is the noise
#: level the GP arms use instead of a tuned ridge.
REPLICATE_SD_AMP = 0.237


# ======================================================================================
# caches -- every one is keyed by the fold, never by anything from the test cells
# ======================================================================================
_SIGN: dict[tuple, np.ndarray] = {}
_TAN: dict[int, np.ndarray] = {}
_DESC: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _fold_key(ctx: Ctx) -> tuple:
    return (ctx.design, ctx.seed, ctx.fold, len(ctx.train), len(ctx.test))


def g14_sign(ctx: Ctx) -> np.ndarray:
    """Gen14's deployed direction call, memoised per fold (it is deterministic)."""
    k = _fold_key(ctx)
    if k not in _SIGN:
        _SIGN[k] = _logistic_sign(ctx)
    return _SIGN[k]


# ======================================================================================
# Gram matrices
# ======================================================================================
def tanimoto_gram(ctx: Ctx) -> np.ndarray:
    """Full 521 x 521 Tanimoto similarity of the ECFP block.  Uses no fitted statistic."""
    key = id(ctx.bench)
    if key not in _TAN:
        E = ctx.bench.frames["ECFP"].to_numpy(dtype=float)
        inter = E @ E.T
        n = E.sum(axis=1)
        den = n[:, None] + n[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            T = np.where(den > 0, inter / np.maximum(den, 1e-12), 0.0)
        np.fill_diagonal(T, 1.0)
        _TAN[key] = 0.5 * (T + T.T)
    return _TAN[key]


def _descriptor_space(ctx: Ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-imputed, standardised LIG2D for every cell, fitted on THIS fold's training cells.

    Returns (Z, pairwise-distance matrix over all cells, median heuristic length scale).

    The cache holds ONE fold: ``run_arms`` calls every arm of a fold consecutively, and a 521x521
    distance matrix per fold across 125 folds would be ~400 MB on a machine shared by four agents.
    """
    k = _fold_key(ctx)
    if k in _DESC:
        return _DESC[k]
    _DESC.clear()
    L = ctx.bench.frames["LIG2D"].to_numpy(dtype=float)
    tr = ctx.train
    med = np.nanmedian(L[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Z = np.where(np.isfinite(L), L, med[None, :])
    mu = Z[tr].mean(axis=0)
    sd = Z[tr].std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (Z - mu) / sd
    Z = np.clip(Z, -8.0, 8.0)                     # a few RDKit columns have extreme outliers
    sq = (Z * Z).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    D = np.sqrt(D2)
    tri = D[np.ix_(tr, tr)]
    ell = float(np.median(tri[np.triu_indices(len(tr), 1)]))
    ell = ell if ell > 1e-6 else 1.0
    _DESC[k] = (Z, D, ell)
    return _DESC[k]


def rbf_gram(ctx: Ctx) -> np.ndarray:
    _, D, ell = _descriptor_space(ctx)
    return np.exp(-0.5 * (D / ell) ** 2)


def matern_gram(ctx: Ctx, nu: float = 1.5) -> np.ndarray:
    _, D, ell = _descriptor_space(ctx)
    if nu == 1.5:
        s = np.sqrt(3.0) * D / ell
        return (1.0 + s) * np.exp(-s)
    s = np.sqrt(5.0) * D / ell
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


GRAMS = {"TAN": tanimoto_gram,
         "RBF": rbf_gram,
         "MAT": lambda ctx: matern_gram(ctx, 1.5)}


# ======================================================================================
# weighted kernel ridge / kernel logistic on a precomputed Gram
# ======================================================================================
def _norm_w(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w * (len(w) / max(w.sum(), 1e-12))


def wkrr(K_tt: np.ndarray, y: np.ndarray, w: np.ndarray, K_st: np.ndarray, alpha: float,
         *, centre: bool = True) -> np.ndarray:
    """Weighted kernel ridge:  min  sum_i w_i (y_i - f_i)^2 + alpha * f' K^-1 f.

    Solved in the symmetrised form  a = S (S K S + alpha I)^-1 S y  with S = diag(sqrt(w)),
    which is the same estimator sklearn's KernelRidge(kernel='precomputed', sample_weight=w)
    computes but lets the same factorisation serve the GP posterior variance.
    """
    w = _norm_w(w)
    y0 = float(np.average(y, weights=w)) if centre else 0.0
    s = np.sqrt(w)
    A = (K_tt * s[:, None]) * s[None, :]
    A.flat[:: len(A) + 1] += alpha
    u = np.linalg.solve(A, s * (y - y0))
    return K_st @ (s * u) + y0


def gp_mean_var(K_tt: np.ndarray, y: np.ndarray, w: np.ndarray, K_st: np.ndarray,
                sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """GP predictive mean and variance with the signal variance read off the training fold.

    The kernel is scaled to the weighted variance of the target, and the noise is the corpus
    replicate sd -- so the amount of shrinkage is set by measured quantities and nothing is tuned.
    """
    w = _norm_w(w)
    y0 = float(np.average(y, weights=w))
    s2 = float(np.average((y - y0) ** 2, weights=w))
    s2 = max(s2, 1e-8)
    alpha = (sigma ** 2) / s2
    mean = wkrr(K_tt, y, w, K_st, alpha, centre=True)
    A = K_tt.copy()
    A.flat[:: len(A) + 1] += alpha
    v = s2 * np.maximum(1.0 - np.einsum("ij,ij->i", K_st, np.linalg.solve(A, K_st.T).T), 0.0)
    return mean, v


def wklr(K_tt: np.ndarray, y01: np.ndarray, w: np.ndarray, K_st: np.ndarray,
         lam: float) -> np.ndarray:
    """Weighted kernel logistic regression with an unpenalised intercept; returns P(y=1) on test.

    min_{a,c}  sum_i w_i [log(1+e^{f_i}) - y_i f_i] + (lam/2) a' K a,   f = K a + c.
    L-BFGS on (a, c); n_train is a few hundred so each fit is milliseconds.
    """
    n = len(y01)
    w = _norm_w(w)
    if len(set(np.asarray(y01).tolist())) < 2:
        return np.full(len(K_st), float(np.mean(y01)))
    y = np.asarray(y01, dtype=float)

    def obj(theta):
        a, c = theta[:n], theta[n]
        f = K_tt @ a + c
        p = 1.0 / (1.0 + np.exp(-np.clip(f, -60, 60)))
        ll = np.sum(w * (np.logaddexp(0.0, f) - y * f))
        Ka = K_tt @ a
        val = ll + 0.5 * lam * float(a @ Ka)
        r = w * (p - y)
        return val, np.concatenate([K_tt @ r + lam * Ka, [float(r.sum())]])

    res = minimize(obj, np.zeros(n + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-8})
    a, c = res.x[:n], res.x[n]
    f = K_st @ a + c
    return 1.0 / (1.0 + np.exp(-np.clip(f, -60, 60)))


# ======================================================================================
# training-fold assembly
# ======================================================================================
def _rich(ctx: Ctx) -> tuple[np.ndarray, np.ndarray]:
    return ctx.rich_train(), ctx.rich_weights()


def _ligand_units(ctx: Ctx, tr: np.ndarray, w: np.ndarray):
    """Collapse the training fold to one row per extractant: the ligand is the kNN unit.

    61 of 82 scored extractants contribute one cell and one contributes 63, so a cell-level
    nearest-neighbour pool would let a single much-measured ligand fill every neighbourhood.
    """
    ext = ctx.bench.frame.extractant.to_numpy()[tr]
    a, b = ctx.amp[tr], ctx.cur[tr]
    rows, A, B, W = [], [], [], []
    for e in np.unique(ext):
        m = ext == e
        rows.append(int(tr[np.flatnonzero(m)[0]]))
        A.append(float(np.average(a[m], weights=w[m])))
        B.append(float(np.average(b[m], weights=w[m])))
        W.append(float(w[m].sum()))
    return np.array(rows), np.array(A), np.array(B), np.array(W)


# ======================================================================================
# arms: DIRECTION  (gen14 magnitude and curvature; only the sign changes)
# ======================================================================================
def _dir_arm(sign_fn):
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(sign_fn(ctx) * ctx.train_mean_magnitude(),
                      ctx.train_mean_curvature(), len(ctx.test))
    return f


def dir_kernel_logistic(gram: str = "TAN", lam: float = 1.0):
    """Kernel logistic on a precomputed similarity Gram, target = heavy-selective."""
    def sign_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        p = wklr(K[np.ix_(tr, tr)], (ctx.amp[tr] < 0).astype(float), w,
                 K[np.ix_(ctx.test, tr)], lam)
        return np.where(p >= 0.5, -1.0, 1.0)
    return _dir_arm(sign_fn)


def dir_kernel_svc(gram: str = "TAN", C: float = 1.0):
    """SVC(kernel='precomputed'), sign of the decision function -- a hinge loss on the same Gram."""
    from sklearn.svm import SVC

    def sign_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        y = (ctx.amp[tr] < 0).astype(int)
        K = GRAMS[gram](ctx)
        if len(set(y.tolist())) < 2:
            return np.full(len(ctx.test), -1.0 if y[0] == 1 else 1.0)
        m = SVC(C=C, kernel="precomputed")
        m.fit(K[np.ix_(tr, tr)], y, sample_weight=_norm_w(w))
        d = m.decision_function(K[np.ix_(ctx.test, tr)])
        return np.where(d >= 0.0, -1.0, 1.0)
    return _dir_arm(sign_fn)


def dir_kernel_ridge(gram: str = "TAN", alpha: float = 1.0):
    """Kernel ridge on the signed amplitude; the direction is the sign of the fitted value."""
    def sign_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        yh = wkrr(K[np.ix_(tr, tr)], ctx.amp[tr], w, K[np.ix_(ctx.test, tr)], alpha)
        return np.where(yh < 0, -1.0, 1.0)
    return _dir_arm(sign_fn)


_ECFP: dict[str, object] = {}


def ecfp_matrix(ctx) -> "np.ndarray":
    """The 2048 bits as float32, materialised once (8 GB shared between four agents)."""
    if "E" not in _ECFP:
        _ECFP["E"] = ctx.bench.frames["ECFP"].to_numpy(dtype=np.float32)
    return _ECFP["E"]


def dir_ecfp_logistic(C: float = 1.0, scale: bool = True):
    """Plain L2 logistic on the 2048 raw bits -- the linear counterpart of the Tanimoto kernel.

    ``scale=True`` standardises every bit on the training fold, which is what a default sklearn
    pipeline does and which multiplies a bit present in 2 % of ligands by about 7; ``scale=False``
    keeps the raw 0/1 encoding, so the L2 penalty is uniform over bits.  Both are reported, because
    the choice is a real modelling decision on a sparse binary block and not a detail.
    """
    from sklearn.linear_model import LogisticRegression

    def sign_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        E = ecfp_matrix(ctx)
        y = (ctx.amp[tr] < 0).astype(int)
        if len(set(y.tolist())) < 2:
            return np.full(len(ctx.test), -1.0 if y[0] == 1 else 1.0)
        Xtr, Xte = E[tr], E[ctx.test]
        if scale:
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            sd = np.where(sd > 1e-9, sd, np.float32(1.0)).astype(np.float32)
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
        m = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
        m.fit(Xtr, y, sample_weight=_norm_w(w))
        return np.where(m.predict_proba(Xte)[:, 1] >= 0.5, -1.0, 1.0)
    return _dir_arm(sign_fn)


def dir_ridge_lig2d(alpha: float = 10.0):
    """Plain L2 ridge on the standardised 206 RDKit descriptors; sign of the fitted amplitude."""
    def sign_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        Z, _, _ = _descriptor_space(ctx)
        ww = _norm_w(w)
        y = ctx.amp[tr]
        y0 = float(np.average(y, weights=ww))
        s = np.sqrt(ww)
        A = Z[tr] * s[:, None]
        G = A.T @ A
        G.flat[:: len(G) + 1] += alpha
        beta = np.linalg.solve(G, A.T @ (s * (y - y0)))
        return np.where(Z[ctx.test] @ beta + y0 < 0, -1.0, 1.0)
    return _dir_arm(sign_fn)


def dir_knn(k: int = 5, gram: str = "TAN"):
    """Sign of the similarity-weighted mean amplitude of the k nearest training ligands."""
    def sign_fn(ctx: Ctx) -> np.ndarray:
        a, _, _ = _knn_predict(ctx, k, gram)
        return np.where(a < 0, -1.0, 1.0)
    return _dir_arm(sign_fn)


# ======================================================================================
# arms: MAGNITUDE  (gen14 direction and curvature; only |a| changes)
# ======================================================================================
def _mag_arm(mag_fn):
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(g14_sign(ctx) * np.clip(mag_fn(ctx), 0.0, None),
                      ctx.train_mean_curvature(), len(ctx.test))
    return f


def mag_kernel_ridge(gram: str = "TAN", alpha: float = 1.0, log: bool = True):
    """Kernel ridge on log|a| (or |a|) with a precomputed similarity Gram."""
    def mag_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        y = np.abs(ctx.amp[tr])
        yy = np.log(y + EPS) if log else y
        out = wkrr(K[np.ix_(tr, tr)], yy, w, K[np.ix_(ctx.test, tr)], alpha)
        return np.exp(out) - EPS if log else out
    return _mag_arm(mag_fn)


def mag_gp(gram: str = "TAN", sigma: float = REPLICATE_SD_AMP):
    """GP predictive mean for |a| with the noise level fixed at the corpus replicate sd."""
    def mag_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        m, _ = gp_mean_var(K[np.ix_(tr, tr)], np.abs(ctx.amp[tr]), w, K[np.ix_(ctx.test, tr)], sigma)
        return m
    return _mag_arm(mag_fn)


def mag_knn(k: int = 5, gram: str = "TAN"):
    """Similarity-weighted mean |a| of the k nearest training ligands."""
    def mag_fn(ctx: Ctx) -> np.ndarray:
        return _knn_predict(ctx, k, gram, target="absamp")[0]
    return _mag_arm(mag_fn)


def mag_ridge_lig2d(alpha: float = 10.0):
    def mag_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        Z, _, _ = _descriptor_space(ctx)
        ww = _norm_w(w)
        y = np.log(np.abs(ctx.amp[tr]) + EPS)
        y0 = float(np.average(y, weights=ww))
        s = np.sqrt(ww)
        A = Z[tr] * s[:, None]
        G = A.T @ A
        G.flat[:: len(G) + 1] += alpha
        beta = np.linalg.solve(G, A.T @ (s * (y - y0)))
        return np.exp(Z[ctx.test] @ beta + y0) - EPS
    return _mag_arm(mag_fn)


# ======================================================================================
# arms: CURVATURE  (gen14's a untouched; only b changes)
# ======================================================================================
def _cur_arm(cur_fn):
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(g14_sign(ctx) * ctx.train_mean_magnitude(), cur_fn(ctx), len(ctx.test))
    return f


def cur_kernel_ridge(gram: str = "TAN", alpha: float = 1.0):
    def cur_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        return wkrr(K[np.ix_(tr, tr)], ctx.cur[tr], w, K[np.ix_(ctx.test, tr)], alpha)
    return _cur_arm(cur_fn)


def cur_gp(gram: str = "TAN", sigma: float = REPLICATE_SD_AMP):
    def cur_fn(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        m, _ = gp_mean_var(K[np.ix_(tr, tr)], ctx.cur[tr], w, K[np.ix_(ctx.test, tr)], sigma)
        return m
    return _cur_arm(cur_fn)


def cur_knn(k: int = 5, gram: str = "TAN"):
    def cur_fn(ctx: Ctx) -> np.ndarray:
        return _knn_predict(ctx, k, gram)[1]
    return _cur_arm(cur_fn)


# ======================================================================================
# arms: JOINT  (both coefficients straight out of the kernel)
# ======================================================================================
def joint_kernel_ridge(gram: str = "TAN", alpha: float = 1.0):
    def f(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        Ktt, Kst = K[np.ix_(tr, tr)], K[np.ix_(ctx.test, tr)]
        a = wkrr(Ktt, ctx.amp[tr], w, Kst, alpha)
        b = wkrr(Ktt, ctx.cur[tr], w, Kst, alpha)
        return np.c_[a, b]
    return f


def joint_gp(gram: str = "TAN", sigma: float = REPLICATE_SD_AMP):
    def f(ctx: Ctx) -> np.ndarray:
        tr, w = _rich(ctx)
        K = GRAMS[gram](ctx)
        Ktt, Kst = K[np.ix_(tr, tr)], K[np.ix_(ctx.test, tr)]
        a, _ = gp_mean_var(Ktt, ctx.amp[tr], w, Kst, sigma)
        b, _ = gp_mean_var(Ktt, ctx.cur[tr], w, Kst, sigma)
        return np.c_[a, b]
    return f


def joint_knn(k: int = 5, gram: str = "TAN"):
    def f(ctx: Ctx) -> np.ndarray:
        a, b, _ = _knn_predict(ctx, k, gram)
        return np.c_[a, b]
    return f


def knn_sign_g14mag(k: int = 5, gram: str = "TAN"):
    """kNN calls the direction, gen14's constant supplies the magnitude (the G14 composition)."""
    return dir_knn(k, gram)


# ======================================================================================
# k nearest neighbours in similarity, and the applicability-domain quantity
# ======================================================================================
def _knn_predict(ctx: Ctx, k: int, gram: str = "TAN", target: str = "coef"):
    """Similarity-weighted mean of the k most similar *training ligands*.

    Returns (a_hat, b_hat, s_max) where ``s_max`` is the similarity to the single nearest training
    ligand -- the applicability-domain coordinate.  ``target='absamp'`` returns |a| in slot 0.
    """
    tr, w = _rich(ctx)
    rows, A, B, W = _ligand_units(ctx, tr, w)
    K = GRAMS[gram](ctx)
    S = K[np.ix_(ctx.test, rows)]
    kk = min(k, S.shape[1])
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    a_hat = np.empty(len(ctx.test))
    b_hat = np.empty(len(ctx.test))
    s_max = S.max(axis=1) if S.shape[1] else np.zeros(len(ctx.test))
    fb_a = float(np.average(np.abs(A) if target == "absamp" else A, weights=W))
    fb_b = float(np.average(B, weights=W))
    src = np.abs(A) if target == "absamp" else A
    for j in range(len(ctx.test)):
        nb = idx[j]
        sw = S[j, nb]
        if sw.sum() <= 1e-9:
            a_hat[j], b_hat[j] = fb_a, fb_b
            continue
        a_hat[j] = float(np.average(src[nb], weights=sw))
        b_hat[j] = float(np.average(B[nb], weights=sw))
    return a_hat, b_hat, s_max


# ======================================================================================
# a combined Gram: does ECFP similarity add anything on top of gen14's 39 topology columns?
# ======================================================================================
_TOPO: dict[tuple, np.ndarray] = {}


def _topo_gram(ctx: Ctx) -> np.ndarray:
    """Cosine-normalised linear kernel on the standardised TOPO39 block (training-fold stats)."""
    k = _fold_key(ctx)
    if k in _TOPO:
        return _TOPO[k]
    X = ctx.feat("TOPO39").astype(float)
    tr = ctx.train
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    Z = (X - mu) / np.where(sd > 1e-9, sd, 1.0)
    nrm = np.sqrt((Z * Z).sum(axis=1))
    nrm = np.where(nrm > 1e-9, nrm, 1.0)
    Zn = Z / nrm[:, None]
    G = Zn @ Zn.T
    _TOPO.clear()
    _TOPO[k] = G
    return G


def combo_gram(ctx: Ctx, wt: float = 0.5) -> np.ndarray:
    """(1-wt) * Tanimoto(ECFP) + wt * cosine(TOPO39).  A sum of PSD kernels is PSD."""
    return (1.0 - wt) * tanimoto_gram(ctx) + wt * _topo_gram(ctx)


GRAMS["TOPO"] = _topo_gram
GRAMS["COMBO"] = lambda ctx: combo_gram(ctx, 0.5)


# ======================================================================================
# applicability domain and GP confidence -- diagnostics, not arms
# ======================================================================================
def ad_probe(ctx: Ctx) -> dict:
    """Per-test-cell applicability-domain and GP-confidence quantities for one fold.

    Returns arrays aligned with ``ctx.test``:
      ``s_max``   Tanimoto to the single nearest *training ligand*
      ``gp_sd``   posterior sd of the Tanimoto GP on |a|
      ``err_mag`` |predicted |a| - true |a||  for the GP magnitude arm
      ``err_amp`` |predicted a - true a|      for the G14 composition with that magnitude
    """
    tr, w = _rich(ctx)
    K = tanimoto_gram(ctx)
    Ktt, Kst = K[np.ix_(tr, tr)], K[np.ix_(ctx.test, tr)]
    m, v = gp_mean_var(Ktt, np.abs(ctx.amp[tr]), w, Kst, REPLICATE_SD_AMP)
    rows, A, B, W = _ligand_units(ctx, tr, w)
    S = K[np.ix_(ctx.test, rows)]
    s_max = S.max(axis=1) if S.shape[1] else np.zeros(len(ctx.test))
    truth = ctx.amp[ctx.test]
    sign = g14_sign(ctx)
    return {"cell": ctx.test.copy(), "s_max": s_max, "gp_sd": np.sqrt(v),
            "gp_mag": m, "true_mag": np.abs(truth),
            "err_mag": np.abs(np.clip(m, 0, None) - np.abs(truth)),
            "err_amp": np.abs(sign * np.clip(m, 0, None) - truth),
            "err_g14": np.abs(sign * ctx.train_mean_magnitude() - truth),
            "rich": ctx.rich[ctx.test].copy()}
