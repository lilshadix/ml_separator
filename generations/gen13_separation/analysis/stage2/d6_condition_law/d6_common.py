"""D6 shared helpers: amplitude extraction, condition column taxonomy, design matrices.

Amplitude of a cell = coefficient of the standardised Shannon radius (CN8) in an OLS
quadratic fit  y_centred ~ 1 + r_z + r_z^2  over the metals observed in that cell,
where y_centred = logD_m - mean_m(logD).  This is the rank-1 "how steep is the curve
across the lanthanide axis" number.  Negative amplitude = heavier (smaller radius)
extracted more strongly.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

REPO = "D:/ml_separator_gh"
sys.path.insert(0, REPO + "/gen13_separation")
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8, ATOMIC_NUMBER  # noqa: E402

COHORT = REPO + "/gen13_separation/manifests/cohort_exact.parquet"
PRED_DIR = REPO + "/gen13_separation/predictions/B_primary"
OUT = REPO + "/gen13_separation/analysis/stage2/d6_condition_law"
FIG = REPO + "/gen13_separation/figures/stage2"

MIN_METALS = 4  # need >=4 metals so the quadratic fit has >=1 residual dof
MIN_RZ_SPAN = 1.5  # standardised-radius span of the observed metals (full La-Lu = 3.147).
# Below this the quadratic design matrix is near-collinear and the amplitude blows up:
# cells with span <= 1.0 have amplitude sd 4.34 vs 0.47 for full-span cells.

# --- standardised radius over the 14 lanthanides -------------------------------------
_r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)
RZ = {m: (SHANNON_RADIUS_CN8[m] - _r.mean()) / _r.std(ddof=0) for m in LANTHANIDES}


def load_cohort() -> pd.DataFrame:
    return pd.read_parquet(COHORT)


def cond_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("cond__")]


def cond_taxonomy(cols: list[str]) -> pd.DataFrame:
    """Label every cond__ column with the physical family it belongs to."""
    rows = []
    for c in cols:
        if c == "cond__acid_concentration_M":
            fam, kind = "acid_concentration", "numeric"
        elif c.startswith("cond__acid__"):
            fam, kind = "acid_identity", "onehot"
        elif c == "cond__extractant_concentration_M":
            fam, kind = "extractant_concentration", "numeric"
        elif c == "cond__temperature_C":
            fam, kind = "temperature", "numeric"
        elif c == "cond__contact_time_min":
            fam, kind = "contact_time", "numeric"
        elif c == "cond__metal_concentration_mM":
            fam, kind = "metal_loading", "numeric"
        elif c.startswith("cond__diluent__"):
            fam, kind = "diluent_identity", "onehot"
        elif c.startswith("cond__additive__"):
            fam, kind = "additive_identity", "onehot"
        else:
            fam, kind = "other", "unknown"
        rows.append(dict(column=c, family=fam, kind=kind))
    return pd.DataFrame(rows)


def collapse_onehot(df: pd.DataFrame, prefix: str, name: str) -> pd.Series:
    """Turn a block of 0/1 one-hot columns into a single categorical label."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    sub = df[cols].astype(float).to_numpy()
    lab = np.full(len(df), "none", dtype=object)
    for i in range(len(df)):
        j = np.flatnonzero(sub[i] > 0.5)
        if len(j) == 1:
            lab[i] = cols[j[0]][len(prefix):]
        elif len(j) > 1:
            lab[i] = "+".join(cols[k][len(prefix):] for k in j)
    return pd.Series(lab, index=df.index, name=name)


def cell_amplitudes(df: pd.DataFrame, min_metals: int = MIN_METALS) -> pd.DataFrame:
    """Observed rank-1 amplitude + quadratic term for every cell with >= min_metals."""
    out = []
    for _, row in df.iterrows():
        ms = [m for m in LANTHANIDES if pd.notna(row.get("logD__" + m))]
        if len(ms) < min_metals:
            continue
        y = np.array([row["logD__" + m] for m in ms], dtype=float)
        y = y - y.mean()
        x = np.array([RZ[m] for m in ms], dtype=float)
        X = np.column_stack([np.ones_like(x), x, x ** 2])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(ms) - 3
        if dof > 0:
            s2 = float(resid @ resid) / dof
            cov = s2 * np.linalg.pinv(X.T @ X)
            se = float(np.sqrt(max(cov[1, 1], 0.0)))
        else:
            se = np.nan
        Xl = np.column_stack([np.ones_like(x), x])
        bl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
        out.append(dict(cell_id=row["cell_id"], amp=float(beta[1]), quad=float(beta[2]),
                        amp_lin=float(bl[1]), amp_se=se, n_metals_fit=len(ms),
                        rz_span=float(x.max() - x.min()),
                        rmse_fit=float(np.sqrt((resid @ resid) / len(ms))),
                        curve_rms=float(np.sqrt((y @ y) / len(ms))),
                        metal_set=",".join(ms)))
    return pd.DataFrame(out)


def predicted_curves(pred: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a predicted centred curve per (split_seed, cell) from the pairwise
    predictions, then take its amplitude the same way.  Every cell has the complete
    pair graph, so the sum-zero least-squares curve is c_m = mean_k d(m,k)."""
    rows = []
    for (seed, cid), g in pred.groupby(["split_seed", "cell_id"], sort=False):
        ms = sorted(set(g["A"]) | set(g["B"]), key=lambda m: ATOMIC_NUMBER[m])
        if len(ms) < MIN_METALS:
            continue
        idx = {m: i for i, m in enumerate(ms)}
        n = len(ms)
        D = np.zeros((n, n))
        Y = np.zeros((n, n))
        a = g["A"].map(idx).to_numpy()
        b = g["B"].map(idx).to_numpy()
        D[a, b] = g["prediction"].to_numpy()
        D[b, a] = -g["prediction"].to_numpy()
        Y[a, b] = g["y"].to_numpy()
        Y[b, a] = -g["y"].to_numpy()
        cp = D.sum(axis=1) / n
        co = Y.sum(axis=1) / n
        x = np.array([RZ[m] for m in ms])
        X = np.column_stack([np.ones_like(x), x, x ** 2])
        bp, *_ = np.linalg.lstsq(X, cp - cp.mean(), rcond=None)
        bo, *_ = np.linalg.lstsq(X, co - co.mean(), rcond=None)
        rows.append(dict(split_seed=seed, cell_id=cid, amp_pred=float(bp[1]),
                         amp_obs_pairs=float(bo[1]), n_metals_fit=n,
                         rz_span=float(x.max() - x.min())))
    return pd.DataFrame(rows)


def seq_ss(y: np.ndarray, blocks: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    """Sequential (Type I) sums of squares for a list of (name, design-block) pairs."""
    n = len(y)
    X = np.ones((n, 1))
    prev_rss = float(y @ y) - n * y.mean() ** 2
    prev_rank = 1
    fit = np.linalg.lstsq(X, y, rcond=None)
    prev_rss = float(((y - X @ fit[0]) ** 2).sum())
    total = prev_rss
    rows = []
    for name, B in blocks:
        Xn = np.column_stack([X, B]) if B.size else X
        beta, *_ = np.linalg.lstsq(Xn, y, rcond=None)
        rss = float(((y - Xn @ beta) ** 2).sum())
        rank = int(np.linalg.matrix_rank(Xn))
        rows.append(dict(term=name, df=rank - prev_rank, ss=prev_rss - rss,
                         frac_of_total=(prev_rss - rss) / total if total > 0 else np.nan,
                         rss_after=rss))
        X, prev_rss, prev_rank = Xn, rss, rank
    rows.append(dict(term="residual", df=n - prev_rank, ss=prev_rss,
                     frac_of_total=prev_rss / total if total > 0 else np.nan,
                     rss_after=prev_rss))
    return pd.DataFrame(rows)


def dummies(labels: pd.Series, drop_first: bool = True) -> np.ndarray:
    d = pd.get_dummies(labels.astype(str), drop_first=drop_first)
    return d.to_numpy(dtype=float)
