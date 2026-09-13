"""Arms that feed the phys3d block (GFN2-xTB complex geometries) into the gen15 curve bench.

Every arm keeps gen14's architecture -- ``coef = (sign x magnitude, curvature)`` -- and swaps in
exactly one piece at a time, so a gain can be attributed:

    direction   L2 logistic (gen14's own estimator) on a chosen column set
    magnitude   the training fold's mean |a| (gen14) or a ridge on log |a|
    curvature   the training fold's mean b (gen14) or a ridge on b

The phys3d columns are attached per *cell* by the cell's extractant; the block is constant within
an extractant, so nothing about a held-out cell's conditions or publication enters through it.
One cohort extractant has no accepted geometry and gets NaN, which the median imputer inside each
pipeline fills from the training fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
from gen15.arms import _curve, _logistic_sign  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402

EPS = 0.05

STATIC = ["d_mean", "d_spread", "d_range", "q_metal", "q_donor", "q_donor_spread",
          "cn", "frac_N", "dipole", "ang_std", "ang_min", "asym"]
RESP = ["dd_dr_A", "dspread_dr_A", "dqm_dr_A", "dqd_dr_A", "dd_dr_B", "dcn_dr_raw",
        "dE_dr_A", "d2E_dr2_A", "E_rms_A", "dE_dr_B", "d2E_dr2_B", "E_rms_B"]
SUPPORT = ["n_series", "n_block", "comp_varies", "anion_nitrate"]
GROUPS = {"P3_STATIC": STATIC, "P3_RESP": RESP, "P3_SUPPORT": SUPPORT,
          "P3_PHYS": STATIC + RESP, "P3_ALL": STATIC + RESP + SUPPORT,
          "P3_ENERGY": ["dE_dr_A", "d2E_dr2_A", "E_rms_A", "dE_dr_B", "d2E_dr2_B", "E_rms_B"]}

_CACHE: dict[int, dict[str, np.ndarray]] = {}


def cell_matrix(ctx: Ctx, group: str = "P3_ALL") -> np.ndarray:
    """The phys3d columns of ``group``, one row per cell of the frozen cohort."""
    key = id(ctx.bench)
    if key not in _CACHE:
        blk = pd.read_parquet(HERE / "phys3d_block.parquet").set_index("extractant")
        ext = ctx.bench.frame["extractant"].to_numpy()
        aligned = blk.reindex(ext)
        _CACHE[key] = {c.replace("phys3d__", ""): aligned[c].to_numpy(dtype=float)
                       for c in blk.columns if c.startswith("phys3d__")}
    cols = _CACHE[key]
    return np.column_stack([cols[c] for c in GROUPS[group]])


def _with_topo(ctx: Ctx, group: str) -> np.ndarray:
    return np.hstack([ctx.feat("TOPO39"), cell_matrix(ctx, group)])


def _X(ctx: Ctx, spec: str) -> np.ndarray:
    """``spec`` is a phys3d group name, ``TOPO39``, or ``TOPO39+<group>``."""
    if spec == "TOPO39":
        return ctx.feat("TOPO39")
    if spec.startswith("TOPO39+"):
        return _with_topo(ctx, spec.split("+", 1)[1])
    return cell_matrix(ctx, spec)


def _pipe(model):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), model)


# --------------------------------------------------------------------------------------
# the three pieces
# --------------------------------------------------------------------------------------
def sign_from(ctx: Ctx, spec: str, C: float = 1.0) -> np.ndarray:
    """gen14's direction call on an arbitrary column set: -1 heavy-selective, +1 light."""
    tr, w = ctx.rich_train(), ctx.rich_weights()
    X = _X(ctx, spec)
    y = (ctx.amp[tr] < 0).astype(int)
    if len(set(y.tolist())) < 2:
        return np.full(len(ctx.test), -1.0 if y[0] else 1.0)
    m = _pipe(LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
    m.fit(X[tr], y, logisticregression__sample_weight=w)
    p = m.predict_proba(X[ctx.test])[:, 1]
    return np.where(p >= 0.5, -1.0, 1.0)


def magnitude_from(ctx: Ctx, spec: str, alpha: float = 10.0) -> np.ndarray:
    """Ridge on log(|a| + eps); falls back to the training constant if it degenerates."""
    tr, w = ctx.rich_train(), ctx.rich_weights()
    X = _X(ctx, spec)
    m = _pipe(Ridge(alpha=alpha))
    m.fit(X[tr], np.log(np.abs(ctx.amp[tr]) + EPS), ridge__sample_weight=w)
    out = np.exp(m.predict(X[ctx.test])) - EPS
    return np.clip(out, 0.0, float(np.abs(ctx.amp[tr]).max()))


def curvature_from(ctx: Ctx, spec: str, alpha: float = 10.0) -> np.ndarray:
    tr, w = ctx.rich_train(), ctx.rich_weights()
    X = _X(ctx, spec)
    m = _pipe(Ridge(alpha=alpha))
    m.fit(X[tr], ctx.cur[tr], ridge__sample_weight=w)
    lo, hi = np.percentile(ctx.cur[tr], [2, 98])
    return np.clip(m.predict(X[ctx.test]), lo, hi)


# --------------------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------------------
def dir_arm(spec: str, C: float = 1.0):
    """gen14 with the direction taken from ``spec``; magnitude and curvature unchanged."""
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(sign_from(ctx, spec, C) * ctx.train_mean_magnitude(),
                      ctx.train_mean_curvature(), len(ctx.test))
    return f


def mag_arm(spec: str, alpha: float = 10.0):
    """gen14's direction (TOPO39) with the magnitude regressed on ``spec``."""
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(_logistic_sign(ctx) * magnitude_from(ctx, spec, alpha),
                      ctx.train_mean_curvature(), len(ctx.test))
    return f


def curv_arm(spec: str, alpha: float = 10.0):
    """gen14 exactly, except the curvature is regressed on ``spec``."""
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(),
                      curvature_from(ctx, spec, alpha), len(ctx.test))
    return f


def composite(spec: str, C: float = 1.0, alpha: float = 10.0):
    """All three pieces from ``spec``: direction, magnitude and curvature."""
    def f(ctx: Ctx) -> np.ndarray:
        return _curve(sign_from(ctx, spec, C) * magnitude_from(ctx, spec, alpha),
                      curvature_from(ctx, spec, alpha), len(ctx.test))
    return f


def _pick_column(ctx: Ctx, y_tr: np.ndarray, group: str) -> tuple[int, float]:
    """The phys3d column most rank-associated with ``y_tr`` **inside the training fold**.

    Choosing the column on the whole corpus would be the researcher degree of freedom that this
    programme has twice been fooled by, so the choice is refitted on every fold and pays for
    itself in the held-out score.
    """
    from scipy import stats
    tr = ctx.rich_train()
    M = cell_matrix(ctx, group)[tr]
    best, best_rho = -1, 0.0
    for j in range(M.shape[1]):
        x = M[:, j]
        ok = np.isfinite(x) & np.isfinite(y_tr)
        if ok.sum() < 40 or len(np.unique(x[ok])) < 5:
            continue
        rho = stats.spearmanr(x[ok], y_tr[ok]).statistic
        if np.isfinite(rho) and abs(rho) > abs(best_rho):
            best, best_rho = j, float(rho)
    return best, best_rho


def curv_1d(group: str = "P3_ALL"):
    """Curvature from ONE phys3d column, the column chosen inside the training fold.

    With about a dozen effective chemotype-level training units, a 28-column ridge is already
    over-parameterised; this is the smallest model that can use the block at all.
    """
    def f(ctx: Ctx) -> np.ndarray:
        tr, w = ctx.rich_train(), ctx.rich_weights()
        b_tr = ctx.cur[tr]
        j, _ = _pick_column(ctx, b_tr, group)
        if j < 0:
            return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(),
                          ctx.train_mean_curvature(), len(ctx.test))
        M = cell_matrix(ctx, group)[:, j]
        fill = float(np.nanmedian(M[tr]))
        x_tr = np.where(np.isfinite(M[tr]), M[tr], fill)
        x_te = np.where(np.isfinite(M[ctx.test]), M[ctx.test], fill)
        A = np.c_[np.ones(len(x_tr)), x_tr]
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], b_tr * sw, rcond=None)
        lo, hi = np.percentile(b_tr, [5, 95])
        pred = np.clip(beta[0] + beta[1] * x_te, lo, hi)
        return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(), pred, len(ctx.test))
    return f


def mag_1d(group: str = "P3_ALL", how: str = "isotonic"):
    """Magnitude from ONE phys3d column chosen inside the training fold, mapped with the
    programme's own 1-D fitter (monotone, both directions tried, absolute loss)."""
    from gen15.arms import _fit_magnitude_1d

    def f(ctx: Ctx) -> np.ndarray:
        tr, w = ctx.rich_train(), ctx.rich_weights()
        mag_tr = np.abs(ctx.amp[tr])
        j, _ = _pick_column(ctx, mag_tr, group)
        if j < 0:
            mag = np.full(len(ctx.test), ctx.train_mean_magnitude())
        else:
            M = cell_matrix(ctx, group)[:, j]
            mag = _fit_magnitude_1d(M[tr], mag_tr, w, M[ctx.test], how)
        return _curve(_logistic_sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def tree_arm(spec: str, n_estimators: int = 300):
    """A small extra-trees regression straight onto both coefficients -- gen13's shape of model,
    on this block, as the non-linear counterpart to the ridge/logistic arms."""
    def f(ctx: Ctx) -> np.ndarray:
        tr, w = ctx.rich_train(), ctx.rich_weights()
        X = _X(ctx, spec)
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          ExtraTreesRegressor(n_estimators=n_estimators, max_features=0.5,
                                              min_samples_leaf=2, random_state=ctx.model_seed,
                                              n_jobs=2))
        m.fit(X[tr], ctx.bench.coef[tr], extratreesregressor__sample_weight=w)
        return np.asarray(m.predict(X[ctx.test]), dtype=float).reshape(len(ctx.test), 2)
    return f
