"""The learners this experiment adds to the programme, all with the same interface.

Every function takes the fold context and returns a prediction for the held-out cells:

* ``*_dir``   -> probability the cell is heavy-selective (a < 0), gen14's direction target
* ``*_mag``   -> the magnitude |a|
* ``*_curv``  -> the curvature b

Fitting always uses the rich (>= 5 metals) training cells with the frozen chemotype-balanced
weights, exactly as ``gen14.models`` does, so the only thing that changes between these and the
deployed logistic is the estimator.  No hyperparameter is chosen on a held-out row; where one is
chosen at all it is chosen by the repository's own chemotype-grouped splitter re-dealt inside the
training fold, and the cost of doing so is measured by running the fixed-hyperparameter twin.
"""
from __future__ import annotations

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import common as K

BITE = "coord__dist__frac_donor_pairs_within_3"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _xy(ctx, features: str):
    fd = K.fold_data(ctx)
    Xa = ctx.feat(features)
    Xtr, Xte = K.median_impute(Xa[fd.rtr], Xa[ctx.test])
    return fd, Xtr, Xte


def _col_index(ctx, name: str) -> int:
    lean = ctx.bench.columns(K.LEAN_BLOCKS)
    return lean.index(name)


def _wcorr(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    mx = (X * w[:, None]).sum(0)
    my = float((y * w).sum())
    cx = X - mx
    cy = y - my
    num = (w[:, None] * cx * cy[:, None]).sum(0)
    den = np.sqrt((w[:, None] * cx ** 2).sum(0) * float((w * cy ** 2).sum())) + 1e-12
    r = num / den
    r[~np.isfinite(r)] = 0.0
    r[X.std(0) < 1e-12] = 0.0          # a constant column carries nothing and breaks the splines
    return r


# --------------------------------------------------------------------------------------
# CatBoost
# --------------------------------------------------------------------------------------
_CB_FIXED = dict(depth=3, l2_leaf_reg=20.0, learning_rate=0.03, rsm=0.5, subsample=0.8,
                 bootstrap_type="Bernoulli", random_strength=2.0, thread_count=2,
                 verbose=0, allow_writing_files=False)


def catboost_dir(features: str = "TOPO39", early: bool = False, iterations: int = 400):
    """Heavily regularised gradient boosting; ``early`` adds chemotype-grouped early stopping.

    The two are run side by side on purpose: the difference is the price of letting an inner
    chemotype split choose the number of trees, which is the same kind of inner-CV selection that
    cost gen14 -0.086 under BP when it chose the logistic penalty.
    """
    from catboost import CatBoostClassifier

    def f(ctx) -> np.ndarray:
        fd, Xtr, Xte = _xy(ctx, features)
        if len(set(fd.y_dir.tolist())) < 2:
            return np.full(len(ctx.test), float(fd.y_dir.mean()))
        if not early:
            m = CatBoostClassifier(iterations=iterations, random_seed=ctx.model_seed, **_CB_FIXED)
            m.fit(Xtr, fd.y_dir, sample_weight=fd.w)
            return m.predict_proba(Xte)[:, 1]
        it, iv = K.inner_folds(fd.groups, ctx.model_seed)[0]
        if len(set(fd.y_dir[it].tolist())) < 2 or len(set(fd.y_dir[iv].tolist())) < 2:
            m = CatBoostClassifier(iterations=iterations, random_seed=ctx.model_seed, **_CB_FIXED)
            m.fit(Xtr, fd.y_dir, sample_weight=fd.w)
            return m.predict_proba(Xte)[:, 1]
        m = CatBoostClassifier(iterations=3000, random_seed=ctx.model_seed,
                               od_type="Iter", od_wait=100, **_CB_FIXED)
        m.fit(Xtr[it], fd.y_dir[it], sample_weight=fd.w[it],
              eval_set=(Xtr[iv], fd.y_dir[iv]), use_best_model=True)
        best = max(int(getattr(m, "best_iteration_", iterations) or iterations) + 1, 20)
        m2 = CatBoostClassifier(iterations=best, random_seed=ctx.model_seed, **_CB_FIXED)
        m2.fit(Xtr, fd.y_dir, sample_weight=fd.w)
        return m2.predict_proba(Xte)[:, 1]
    return f


def catboost_reg(target: str = "logmag", features: str = "TOPO39", iterations: int = 400):
    from catboost import CatBoostRegressor

    def f(ctx) -> np.ndarray:
        fd, Xtr, Xte = _xy(ctx, features)
        y = fd.y_logmag if target == "logmag" else fd.y_curv
        m = CatBoostRegressor(iterations=iterations, loss_function="MAE",
                              random_seed=ctx.model_seed, **_CB_FIXED)
        m.fit(Xtr, y, sample_weight=fd.w)
        p = np.asarray(m.predict(Xte), dtype=float)
        return np.clip(np.exp(p) - K.EPS, 0.0, None) if target == "logmag" else p
    return f


# --------------------------------------------------------------------------------------
# monotone / shape-constrained
# --------------------------------------------------------------------------------------
def isotonic_dir(column: str = BITE, increasing: bool | None = True):
    """P(heavy) as a monotone function of one bite descriptor.

    ``increasing=True`` is the chemistry's own prior -- a tighter chelate bite favours the smaller,
    heavier lanthanide -- fixed before any fold is run.  ``increasing=None`` lets the training fold
    pick the direction, which is the cheapest possible piece of inner selection and is reported
    beside it so the price of that one choice is visible.
    """
    j = None

    def f(ctx) -> np.ndarray:
        nonlocal j
        if j is None:
            j = _col_index(ctx, column)
        fd = K.fold_data(ctx)
        v_tr = ctx.X[fd.rtr, j]
        v_te = ctx.X[ctx.test, j]
        ok = np.isfinite(v_tr)
        if ok.sum() < 20:
            return np.full(len(ctx.test), float(np.average(fd.y_dir, weights=fd.w)))
        fill = float(np.median(v_tr[ok]))
        v_te = np.where(np.isfinite(v_te), v_te, fill)
        if increasing is None:
            best, best_loss = None, np.inf
            for inc in (True, False):
                m = IsotonicRegression(increasing=inc, out_of_bounds="clip")
                m.fit(v_tr[ok], fd.y_dir[ok], sample_weight=fd.w[ok])
                loss = float(np.average(np.abs(m.predict(v_tr[ok]) - fd.y_dir[ok]),
                                        weights=fd.w[ok]))
                if loss < best_loss:
                    best, best_loss = m, loss
            return np.asarray(best.predict(v_te), dtype=float)
        m = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        m.fit(v_tr[ok], fd.y_dir[ok], sample_weight=fd.w[ok])
        return np.asarray(m.predict(v_te), dtype=float)
    return f


def xgb_dir(features: str = "TOPO39", monotone_column: str | None = BITE):
    """Gradient boosting with a hard monotone constraint on the bite descriptor (or without).

    ``min_child_weight`` is XGBoost's own default of 1.  The first setting tried here was 5, which
    is degenerate on this corpus and not a tuning question: a BP fold has as few as 66 rich
    training cells, so no split ever reaches the required hessian mass, every tree is a stump with
    no split, and the model returns one constant probability for every cell (training accuracy
    0.44, i.e. worse than the majority class).  The change was made on that training-fold
    diagnostic, before any five-design score was read.
    """
    from xgboost import XGBClassifier

    def f(ctx) -> np.ndarray:
        fd, Xtr, Xte = _xy(ctx, features)
        if len(set(fd.y_dir.tolist())) < 2:
            return np.full(len(ctx.test), float(fd.y_dir.mean()))
        cons = [0] * Xtr.shape[1]
        if monotone_column is not None:
            names = [ctx.bench.columns(K.LEAN_BLOCKS)[i] for i in ctx.fs[features]]
            if monotone_column in names:
                cons[names.index(monotone_column)] = 1
        m = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.6, reg_lambda=10.0, min_child_weight=1.0,
                          monotone_constraints="(" + ",".join(map(str, cons)) + ")",
                          tree_method="hist", n_jobs=2, random_state=ctx.model_seed,
                          eval_metric="logloss")
        m.fit(Xtr, fd.y_dir, sample_weight=fd.w)
        return m.predict_proba(Xte)[:, 1]
    return f


# --------------------------------------------------------------------------------------
# generalised additive model
# --------------------------------------------------------------------------------------
def gam_dir(features: str = "TOPO39", k: int = 8, n_knots: int = 5, C: float = 1.0,
            inner_pick_k: bool = False):
    """Splines per column on the ``k`` strongest topology columns, then an L2 logistic.

    The columns are ranked by their chemotype-weighted point-biserial correlation with the
    direction label *on the training fold*.  ``inner_pick_k`` prices letting a chemotype-grouped
    inner split choose ``k`` instead of fixing it a priori.
    """
    def _fit(Xtr, ytr, wtr, Xte, cols, kk, seed):
        sel = cols[:kk]
        pipe = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant"),
            StandardScaler(),
            LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
        pipe.fit(Xtr[:, sel], ytr, logisticregression__sample_weight=wtr)
        return pipe.predict_proba(Xte[:, sel])[:, 1]

    def f(ctx) -> np.ndarray:
        fd, Xtr, Xte = _xy(ctx, features)
        if len(set(fd.y_dir.tolist())) < 2:
            return np.full(len(ctx.test), float(fd.y_dir.mean()))
        live = np.flatnonzero(Xtr.std(0) > 1e-12)
        if len(live) == 0:
            return np.full(len(ctx.test), float(np.average(fd.y_dir, weights=fd.w)))
        order = live[np.argsort(-np.abs(_wcorr(Xtr, fd.y_dir.astype(float), fd.w))[live])]
        kk = k
        if inner_pick_k:
            best, best_acc = k, -np.inf
            splits = K.inner_folds(fd.groups, ctx.model_seed)
            for cand in (3, 5, 8, 12, 20):
                if cand > Xtr.shape[1]:
                    continue
                accs = []
                for it, iv in splits:
                    if len(set(fd.y_dir[it].tolist())) < 2:
                        continue
                    lv = np.flatnonzero(Xtr[it].std(0) > 1e-12)
                    if len(lv) == 0:
                        continue
                    o = lv[np.argsort(-np.abs(
                        _wcorr(Xtr[it], fd.y_dir[it].astype(float), fd.w[it]))[lv])]
                    p = _fit(Xtr[it], fd.y_dir[it], fd.w[it], Xtr[iv], o, cand, ctx.model_seed)
                    accs.append(float(np.average((p >= 0.5).astype(int) == fd.y_dir[iv],
                                                 weights=fd.w[iv])))
                if accs and np.mean(accs) > best_acc:
                    best, best_acc = cand, float(np.mean(accs))
            kk = best
        return _fit(Xtr, fd.y_dir, fd.w, Xte, order, kk, ctx.model_seed)
    return f


def gam_reg(target: str = "logmag", features: str = "TOPO39", k: int = 8, n_knots: int = 5,
            alpha: float = 10.0):
    def f(ctx) -> np.ndarray:
        fd, Xtr, Xte = _xy(ctx, features)
        y = fd.y_logmag if target == "logmag" else fd.y_curv
        w = fd.w
        live = np.flatnonzero(Xtr.std(0) > 1e-12)
        if len(live) == 0:
            v = float(np.average(y, weights=w))
            p = np.full(len(ctx.test), v)
            return np.clip(np.exp(p) - K.EPS, 0.0, None) if target == "logmag" else p
        sel = live[np.argsort(-np.abs(_wcorr(Xtr, y, w))[live])][:k]
        pipe = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant"),
            StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(Xtr[:, sel], y, ridge__sample_weight=w)
        p = np.asarray(pipe.predict(Xte[:, sel]), dtype=float)
        return np.clip(np.exp(p) - K.EPS, 0.0, None) if target == "logmag" else p
    return f


# --------------------------------------------------------------------------------------
# symbolic / sparse equation search
# --------------------------------------------------------------------------------------
def _scale01(Xtr: np.ndarray, Xte: np.ndarray):
    """Map every column into [0.1, 1.1] with the training fold's own range."""
    lo = Xtr.min(0)
    hi = Xtr.max(0)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (Xtr - lo) / rng + 0.1, np.clip((Xte - lo) / rng, 0.0, 1.0) + 0.1


def _pair_terms(A: np.ndarray, names: list[str], B: np.ndarray | None = None,
                bnames: list[str] | None = None):
    """Products and ratios of two blocks of columns (or all unordered pairs when B is None)."""
    if B is None:
        i, j = np.triu_indices(A.shape[1], k=1)
        prod = A[:, i] * A[:, j]
        pn = [f"{names[a]}*{names[b]}" for a, b in zip(i, j)]
        i2, j2 = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[1]), indexing="ij")
        m = i2 != j2
        rat = A[:, i2[m]] / A[:, j2[m]]
        rn = [f"{names[a]}/{names[b]}" for a, b in zip(i2[m], j2[m])]
        return np.c_[prod, rat], pn + rn
    prod = np.einsum("ni,nj->nij", A, B).reshape(len(A), -1)
    pn = [f"({a})*{b}" for a in names for b in bnames]
    rat = (A[:, :, None] / B[:, None, :]).reshape(len(A), -1)
    rn = [f"({a})/{b}" for a in names for b in bnames]
    return np.c_[prod, rat], pn + rn


def _best_threshold(F: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Per column: the weighted-accuracy-optimal threshold rule.  Returns (thr, sign, acc)."""
    n, m = F.shape
    order = np.argsort(F, axis=0, kind="stable")
    Y = y[order] * w[order]
    W = w[order]
    cum_pos = np.cumsum(Y, axis=0)
    cum_w = np.cumsum(W, axis=0)
    tot_pos = cum_pos[-1]
    tot_w = cum_w[-1]
    # rule "predict 1 above the cut": correct = (positives above) + (negatives below)
    acc_hi = (tot_pos - cum_pos) + (cum_w - cum_pos)
    acc_lo = cum_pos + (tot_w - cum_w - (tot_pos - cum_pos))
    A = np.maximum(acc_hi, acc_lo) / tot_w
    k = np.argmax(A, axis=0)
    cols = np.arange(m)
    acc = A[k, cols]
    sign = np.where(acc_hi[k, cols] >= acc_lo[k, cols], 1.0, -1.0)
    fs = np.take_along_axis(F, order, axis=0)
    lo = fs[k, cols]
    hi = fs[np.minimum(k + 1, n - 1), cols]
    return 0.5 * (lo + hi), sign, acc


def _apply_threshold(F: np.ndarray, thr: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return ((F > thr[None, :]) == (sign[None, :] > 0)).astype(float)


def symbolic_search(features: str = "TOPO39", top1: int = 30, audit: list | None = None):
    """Exhaustive 1- and 2-column terms, then a hill-climb into the 3-column terms.

    Stage 1 scores every column, every product and every ratio of two columns (2 262 terms) by
    chemotype-grouped 4-fold inner CV *inside the training fold*, with a weighted-accuracy-optimal
    threshold rule as the classifier.  Stage 2 takes the best ``top1`` terms and multiplies and
    divides each by every single column (2 340 more), so the search covers products and ratios of
    at most three columns without enumerating all 27 000 of them.  The winner is refitted on the
    whole training fold and applied out of fold.

    When ``audit`` is a list, one record per fold is appended: the winner's inner-CV score, its
    honest out-of-fold accuracy, the out-of-fold accuracy of the a-priori bite column, and the
    best out-of-fold accuracy any term in the search space achieves (an oracle, reported only to
    size how much the search *could* overfit).
    """
    def f(ctx) -> np.ndarray:
        fd, Xtr, Xte = _xy(ctx, features)
        names = [ctx.bench.columns(K.LEAN_BLOCKS)[i] for i in ctx.fs[features]]
        if len(set(fd.y_dir.tolist())) < 2:
            return np.full(len(ctx.test), float(fd.y_dir.mean()))
        Atr, Ate = _scale01(Xtr, Xte)
        y, w = fd.y_dir.astype(float), fd.w
        P1tr, n1 = _pair_terms(Atr, names)
        P1te, _ = _pair_terms(Ate, names)
        F1tr = np.c_[Atr, P1tr].astype(np.float64)
        F1te = np.c_[Ate, P1te].astype(np.float64)
        nm1 = list(names) + n1

        splits = K.inner_folds(fd.groups, ctx.model_seed)

        def inner_score(Ftr):
            sc = np.zeros(Ftr.shape[1])
            tot = 0.0
            for it, iv in splits:
                if len(set(fd.y_dir[it].tolist())) < 2 or len(iv) == 0:
                    continue
                thr, sign, _ = _best_threshold(Ftr[it], y[it], w[it])
                pred = _apply_threshold(Ftr[iv], thr, sign)
                sc += (w[iv][:, None] * (pred == y[iv][:, None])).sum(0) / w[iv].sum()
                tot += 1.0
            return sc / max(tot, 1.0)

        s1 = inner_score(F1tr)
        keep = np.argsort(-s1)[:top1]
        B2tr, n2 = _pair_terms(F1tr[:, keep], [nm1[i] for i in keep], Atr, names)
        B2te, _ = _pair_terms(F1te[:, keep], [nm1[i] for i in keep], Ate, names)
        Ftr = np.c_[F1tr, B2tr]
        Fte = np.c_[F1te, B2te]
        nm = nm1 + n2
        s = np.r_[s1, inner_score(B2tr)]
        j = int(np.argmax(s))
        thr, sign, _ = _best_threshold(Ftr[:, [j]], y, w)
        p = _apply_threshold(Fte[:, [j]], thr, sign).ravel()

        if audit is not None:
            te_ok = ctx.rich[ctx.test]
            yte = (ctx.amp[ctx.test] < 0).astype(float)
            rec = {"design": ctx.design, "split_seed": ctx.seed, "fold": ctx.fold,
                   "n_terms": Ftr.shape[1], "winner": nm[j], "inner_cv_acc": float(s[j]),
                   "outer_acc": float((p[te_ok] == yte[te_ok]).mean()) if te_ok.any() else np.nan}
            if te_ok.any():
                thr_a, sign_a, _ = _best_threshold(Ftr, y, w)
                Pte = _apply_threshold(Fte, thr_a, sign_a)
                acc_all = (Pte[te_ok] == yte[te_ok][:, None]).mean(0)
                rec["oracle_best_outer_acc"] = float(acc_all.max())
                bj = names.index(BITE) if BITE in names else 0
                rec["bite_outer_acc"] = float(acc_all[bj])
                rec["oracle_winner"] = nm[int(np.argmax(acc_all))]
            audit.append(rec)
        return p
    return f
