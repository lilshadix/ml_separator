"""D6 step 3: nested ANOVA-style decomposition of the within-extractant amplitude."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from d6_common import (load_cohort, cell_amplitudes, collapse_onehot, seq_ss, OUT,
                       MIN_RZ_SPAN)

df = load_cohort()
cell = df.merge(cell_amplitudes(df), on="cell_id")
cell = cell[cell.rz_span >= MIN_RZ_SPAN].copy()
cell["diluent_id"] = collapse_onehot(cell, "cond__diluent__", "diluent_id")
cell["acid_id"] = collapse_onehot(cell, "cond__acid__", "acid_id")
cell["log_acid"] = np.log10(cell["cond__acid_concentration_M"].astype(float))
cell["log_ext"] = np.log10(cell["cond__extractant_concentration_M"].astype(float))
cell["temp"] = cell["cond__temperature_C"].astype(float)

# restrict to extractants with >=3 amplitude cells (a within-extractant question)
cell = cell[cell.groupby("extractant").cell_id.transform("size") >= 3].copy()
# fill the few missing numerics with the extractant median so the design stays full rank
for c in ["log_acid", "log_ext", "temp"]:
    cell[c] = cell[c].fillna(cell.groupby("extractant")[c].transform("median"))
    cell[c] = cell[c].fillna(cell[c].median())
print(f"analysis set: {len(cell)} cells, {cell.extractant.nunique()} extractants, "
      f"{cell.chemotype.nunique()} chemotypes, {cell.publication_id.nunique()} publications")

y = cell.amp.to_numpy(float)
ext = cell.extractant.astype(str)


def nested_num(col: str) -> np.ndarray:
    """one slope per extractant for a numeric column, centred within extractant"""
    x = cell[col].astype(float)
    xr = (x - cell.groupby("extractant")[col].transform("mean")).to_numpy()
    E = pd.get_dummies(ext, drop_first=False).to_numpy(float)
    B = E * xr[:, None]
    keep = [j for j in range(B.shape[1]) if np.ptp(B[:, j]) > 1e-12]
    return B[:, keep]


def nested_cat(col: str) -> np.ndarray:
    """extractant-specific effect of a categorical column"""
    lab = (ext + "||" + cell[col].astype(str))
    A = pd.get_dummies(lab, drop_first=False).to_numpy(float)
    E = pd.get_dummies(ext, drop_first=False).to_numpy(float)
    M = np.column_stack([E, A])
    # orthogonalise the interaction against the extractant main effect
    q, _ = np.linalg.qr(E)
    R = A - q @ (q.T @ A)
    keep = [j for j in range(R.shape[1]) if np.linalg.norm(R[:, j]) > 1e-8]
    R = R[:, keep]
    if R.size == 0:
        return R
    u, s, _ = np.linalg.svd(R, full_matrices=False)
    return u[:, s > 1e-8 * max(s.max(), 1e-12)]


E = pd.get_dummies(ext, drop_first=True).to_numpy(float)

ORDERS = {
    "primary (conditions before provenance)": [
        ("extractant identity", E),
        ("measured metal set (design nuisance)", nested_cat("metal_set")),
        ("acid concentration (log10 M, slope per extractant)", nested_num("log_acid")),
        ("acid identity", nested_cat("acid_id")),
        ("extractant concentration (log10 M, slope per extractant)", nested_num("log_ext")),
        ("temperature (C, slope per extractant)", nested_num("temp")),
        ("diluent identity", nested_cat("diluent_id")),
        ("publication_id", nested_cat("publication_id")),
    ],
    "provenance first (publication before conditions)": [
        ("extractant identity", E),
        ("measured metal set (design nuisance)", nested_cat("metal_set")),
        ("publication_id", nested_cat("publication_id")),
        ("acid concentration (log10 M, slope per extractant)", nested_num("log_acid")),
        ("acid identity", nested_cat("acid_id")),
        ("extractant concentration (log10 M, slope per extractant)", nested_num("log_ext")),
        ("temperature (C, slope per extractant)", nested_num("temp")),
        ("diluent identity", nested_cat("diluent_id")),
    ],
}

tables = []
for name, blocks in ORDERS.items():
    t = seq_ss(y, blocks)
    tot_within = float(t.ss.sum()) - float(t.loc[t.term == "extractant identity", "ss"].iloc[0])
    t["frac_of_within_extractant_SS"] = t.ss / tot_within
    t.loc[t.term == "extractant identity", "frac_of_within_extractant_SS"] = np.nan
    t["order"] = name
    t["ms"] = t.ss / t.df.replace(0, np.nan)
    res = t[t.term == "residual"].iloc[0]
    ms_res = res.ss / res.df
    t["F"] = t.ms / ms_res
    t.loc[t.term == "residual", "F"] = np.nan
    t["p"] = [float(stats.f.sf(f, d, res.df)) if np.isfinite(f) and d > 0 else np.nan
              for f, d in zip(t.F, t.df)]
    tables.append(t)
    print(f"\n=== sequential (Type I) SS of cell amplitude - {name} ===")
    print(f"total SS about the grand mean = {t.ss.sum():.3f} over {len(cell)} cells; "
          f"within-extractant SS = {tot_within:.3f}")
    print(t[["term", "df", "ss", "ms", "F", "p", "frac_of_total",
             "frac_of_within_extractant_SS"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

anova = pd.concat(tables, ignore_index=True)
anova.to_csv(OUT + "/d6_step3_nested_anova.csv", index=False)

# --- each condition block added LAST (Type-II-like unique contribution) --------------
print("\n=== unique contribution: each block added last, after everything else ===")
full = ORDERS["primary (conditions before provenance)"]
names = [n for n, _ in full]
uni = []
for i, (nm, B) in enumerate(full):
    if nm == "extractant identity":
        continue
    others = [b for j, (n2, b) in enumerate(full) if j != i and b.size]
    Xo = np.column_stack([np.ones(len(y))] + others)
    bo, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    rss_o = float(((y - Xo @ bo) ** 2).sum())
    Xf = np.column_stack([Xo, B]) if B.size else Xo
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    rss_f = float(((y - Xf @ bf) ** 2).sum())
    dfu = int(np.linalg.matrix_rank(Xf) - np.linalg.matrix_rank(Xo))
    uni.append(dict(term=nm, df_unique=dfu, ss_unique=rss_o - rss_f,
                    rss_full=rss_f))
uni = pd.DataFrame(uni)
tot_within = float(tables[0].ss.sum()) - float(tables[0].ss.iloc[0])
uni["frac_of_within_extractant_SS"] = uni.ss_unique / tot_within
uni["SS_within_reference"] = tot_within
uni.to_csv(OUT + "/d6_step3_unique_contributions.csv", index=False)
print(uni.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --- the honest question: does any of this predict a HELD-OUT cell? -----------------
# leave-one-cell-out within the same extractant.  Min-norm least squares, so a
# categorical level seen only in the held-out cell simply contributes nothing.
# Ridge shrinks the nested condition block only; the extractant means stay free.
def loo(X: np.ndarray, yv: np.ndarray, n_free: int, lam: float) -> np.ndarray:
    p = X.shape[1]
    P = np.eye(p) * lam
    P[:n_free, :n_free] = 0.0
    preds = np.empty(len(yv))
    for i in range(len(yv)):
        m = np.ones(len(yv), bool)
        m[i] = False
        Xt, yt = X[m], yv[m]
        beta = np.linalg.pinv(Xt.T @ Xt + P) @ (Xt.T @ yt)
        preds[i] = float(X[i] @ beta)
    return preds


cv_rows = []
for min_cells in (3, 6):
    sel = cell.groupby("extractant").cell_id.transform("size") >= min_cells
    c2 = cell[sel].copy()
    idx = sel.to_numpy()
    y2 = y[idx]
    e2 = c2.extractant.astype(str)
    Eall = pd.get_dummies(e2, drop_first=False).to_numpy(float)
    nf = Eall.shape[1]

    def nn(col):  # nested numeric slope on this subset
        x = c2[col].astype(float)
        xr = (x - c2.groupby("extractant")[col].transform("mean")).to_numpy()
        B = Eall * xr[:, None]
        return B[:, [j for j in range(B.shape[1]) if np.ptp(B[:, j]) > 1e-12]]

    def nc(col):  # nested categorical effect on this subset
        A = pd.get_dummies(e2 + "||" + c2[col].astype(str), drop_first=False).to_numpy(float)
        q, _ = np.linalg.qr(Eall)
        R = A - q @ (q.T @ A)
        R = R[:, [j for j in range(R.shape[1]) if np.linalg.norm(R[:, j]) > 1e-8]]
        if R.size == 0:
            return R
        u, s, _ = np.linalg.svd(R, full_matrices=False)
        return u[:, s > 1e-8 * max(s.max(), 1e-12)]

    MODELS = {
        "extractant mean only (baseline)": [],
        "+ nested acid-concentration slope": [nn("log_acid")],
        "+ nested extractant-concentration slope": [nn("log_ext")],
        "+ nested temperature slope": [nn("temp")],
        "+ nested diluent identity": [nc("diluent_id")],
        "+ acid + ext + temperature slopes": [nn("log_acid"), nn("log_ext"), nn("temp")],
        "all conditions (acid, ext, temp, acid id, diluent)": [
            nn("log_acid"), nn("log_ext"), nn("temp"), nc("acid_id"), nc("diluent_id")],
    }
    print(f"\n=== leave-one-cell-out within-extractant CV of the amplitude, "
          f"extractants with >= {min_cells} analysis cells "
          f"({len(c2)} cells, {c2.extractant.nunique()} extractants) ===")
    print("    (ridge lambda shrinks the condition block only; lam=0 is plain OLS)")
    base = {}
    for label, blocks in MODELS.items():
        X = np.column_stack([Eall] + [b for b in blocks if b.size]) if blocks else Eall
        for lam in (0.0, 1.0):
            pr = loo(X, y2, nf, lam)
            err = y2 - pr
            mae, rmse = float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))
            if label.startswith("extractant mean"):
                base[lam] = (mae, rmse, err.copy())
            b_mae, b_rmse, b_err = base[lam]
            cv_rows.append(dict(min_cells_per_extractant=min_cells, model=label,
                                ridge_lambda=lam, n_params=X.shape[1], n_cells=len(y2),
                                loo_mae=mae, loo_rmse=rmse,
                                delta_mae_vs_baseline=mae - b_mae,
                                r2_vs_baseline=1 - (err ** 2).sum() / (b_err ** 2).sum()))
            print(f"  lam={lam:<4g} {label:52s} p={X.shape[1]:3d}  MAE {mae:.3f}  "
                  f"RMSE {rmse:6.3f}  dMAE {mae-b_mae:+.3f}  "
                  f"R2 {1 - (err**2).sum()/(b_err**2).sum():+.3f}")
pd.DataFrame(cv_rows).to_csv(OUT + "/d6_step3_loo_cv_amplitude.csv", index=False)

# --- residual vs measurement noise ---------------------------------------------------
res_row = tables[0][tables[0].term == "residual"].iloc[0]
resid_ms = res_row.ss / res_row.df
print(f"\nresidual mean square {resid_ms:.4f} -> residual sd {np.sqrt(resid_ms):.3f} "
      f"amplitude units, on {int(res_row.df)} df")
print(f"median per-cell amplitude standard error from the quadratic fit: "
      f"{cell.amp_se.median():.3f} -> replicate-noise variance ~ "
      f"{cell.amp_se.median()**2:.4f}")
print(f"so the unexplained within-extractant amplitude variation is "
      f"{np.sqrt(resid_ms)/cell.amp_se.median():.1f}x the measurement noise sd")
print("\nwrote d6_step3_nested_anova.csv, d6_step3_unique_contributions.csv")
