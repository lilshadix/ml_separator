"""Second verification pass: the things that could invalidate the headline.

(a) is the 82.5-87.0% amplitude share an artefact of a few huge cells?
    -> recompute it weighting each cell equally, each extractant equally,
       and excluding the dominant chemotype.
(b) is the centring really over each cell's OWN observed metals?
    -> cross-check the metal set implied by the pairs against the non-NaN
       logD columns of cohort_exact.parquet, cell by cell.
(c) how concentrated is the pooled SSE (top cell / top chemotype)?
(d) amplitude regression: slope through origin, slope with intercept, R^2,
    and the sample size behind each.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
PRED = os.path.join(ROOT, "generations/gen13_separation/predictions/B_primary")
OUT = os.path.join(ROOT, "generations/gen13_separation/analysis/stage2/d4_amplitude_vs_shape/verify")

sys.path.insert(0, os.path.join(ROOT, "generations", "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

LN_IDX = {m: i for i, m in enumerate(LANTHANIDES)}
RADIUS = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], float)
ARMS = ["C_DIRECT_ROW", "M_SELECTED", "M_PHYSICS_radius+radius_sq",
        "M_LOWRANK_K2", "X_ENS_DIRECT+LOWRANK_K2"]

coh = pd.read_parquet(os.path.join(ROOT, "generations/gen13_separation/manifests/cohort_exact.parquet"))
logd_cols = [f"logD__{m}" for m in LANTHANIDES]
coh_metals = {
    r.cell_id: frozenset(m for m, c in zip(LANTHANIDES, logd_cols)
                         if pd.notna(getattr(r, c)))
    for r in coh.itertuples()
}
coh_chemo = dict(zip(coh["cell_id"], coh["chemotype"]))


def per_unit_table(arm):
    df = pd.read_parquet(os.path.join(PRED, arm + ".parquet"))
    recs = []
    mismatch = 0
    for (seed, cid), sub in df.groupby(["split_seed", "cell_id"], sort=False):
        metals = sorted(set(sub["A"]).union(sub["B"]), key=lambda x: LN_IDX[x])
        m = len(metals)
        pos = {mm: i for i, mm in enumerate(metals)}
        n = len(sub)
        P = np.zeros((n, m))
        P[np.arange(n), [pos[a] for a in sub["A"]]] = 1.0
        P[np.arange(n), [pos[b] for b in sub["B"]]] = -1.0
        r = RADIUS[[LN_IDX[mm] for mm in metals]]
        u = r - r.mean()
        u = u / np.linalg.norm(u)
        y = sub["y"].values.astype(float)
        p = sub["prediction"].values.astype(float)
        c_obs = P.T @ y / m
        c_pred = P.T @ p / m
        err = c_pred - c_obs
        a_err = float(err @ u)
        shape = err - a_err * u
        if frozenset(metals) != coh_metals.get(cid, frozenset()):
            mismatch += 1
        recs.append(dict(
            split_seed=seed, cell_id=cid, extractant=sub["extractant"].iloc[0],
            chemotype=sub["chemotype"].iloc[0], m=m, n_pairs=n,
            a_obs=float(c_obs @ u), a_pred=float(c_pred @ u),
            sse_amp=m * a_err ** 2, sse_shape=m * float(shape @ shape),
            sse_tot=m * float(err @ err)))
    return pd.DataFrame(recs), mismatch


rows = []
regr = []
for arm in ARMS:
    U, mismatch = per_unit_table(arm)
    U["share_amp_unit"] = U["sse_amp"] / U["sse_tot"].replace(0, np.nan)

    pooled = U["sse_amp"].sum() / U["sse_tot"].sum()

    # equal weight per (seed, cell): mean of the per-unit amplitude shares
    eq_cell = float(U["share_amp_unit"].dropna().mean())

    # equal weight per extractant: pool inside an extractant, then average
    per_ext = U.groupby(["split_seed", "extractant"])[["sse_amp", "sse_tot"]].sum()
    eq_ext = float((per_ext["sse_amp"] / per_ext["sse_tot"]).groupby("split_seed").mean().mean())

    # drop the dominant chemotype
    dom = U["chemotype"].value_counts().idxmax()
    nod = U[U["chemotype"] != dom]
    no_dom = nod["sse_amp"].sum() / nod["sse_tot"].sum()

    # concentration of the pooled SSE
    s = U["sse_tot"].sort_values(ascending=False)
    top1 = float(s.iloc[0] / s.sum())
    top10 = float(s.iloc[:10].sum() / s.sum())
    top1pct = float(s.iloc[:max(1, len(s) // 100)].sum() / s.sum())
    dom_sse = float(U.loc[U["chemotype"] == dom, "sse_tot"].sum() / U["sse_tot"].sum())

    # amplitude regression
    x, yv = U["a_pred"].values, U["a_obs"].values
    slope0 = float((x @ yv) / (x @ x))
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
    # bootstrap the through-origin slope over extractants
    exts = U["extractant"].unique()
    grp = {e: U.index[U["extractant"] == e].values for e in exts}
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        pick = rng.choice(len(exts), len(exts), replace=True)
        idx = np.concatenate([grp[exts[k]] for k in pick])
        xx, yy = x[idx], yv[idx]
        bs.append((xx @ yy) / (xx @ xx))
    lo, hi = np.percentile(bs, [2.5, 97.5])

    rows.append(dict(arm=arm, n_units=len(U), cellset_mismatches=mismatch,
                     amp_share_pooled=pooled, amp_share_equal_cell=eq_cell,
                     amp_share_equal_extractant=eq_ext,
                     amp_share_excl_dominant_chemotype=no_dom,
                     dominant_chemotype=dom, n_units_excl_dom=len(nod),
                     sse_share_dominant_chemotype=dom_sse,
                     sse_share_top1_unit=top1, sse_share_top10_units=top10,
                     sse_share_top1pct_units=top1pct))
    regr.append(dict(arm=arm, n_cells=len(U), slope_through_origin=slope0,
                     ci_lo=float(lo), ci_hi=float(hi), n_extractants=len(exts),
                     slope_with_intercept=float(coef[0]), intercept=float(coef[1]),
                     r_squared=float(r2)))
    print(f"{arm}: pooled={pooled:.4f} eq-cell={eq_cell:.4f} eq-extr={eq_ext:.4f} "
          f"no-{dom}={no_dom:.4f} | top1 unit={top1:.4f} top1%={top1pct:.4f} "
          f"{dom} SSE share={dom_sse:.4f} | mismatches={mismatch}")
    print(f"    slope0={slope0:.3f} [{lo:.3f},{hi:.3f}]  slope_int={coef[0]:.3f} "
          f"int={coef[1]:.3f} R2={r2:.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "v_d4_share_robustness.csv"), index=False)
pd.DataFrame(regr).to_csv(os.path.join(OUT, "v_d4_amplitude_regression.csv"), index=False)
print("\nwrote v_d4_share_robustness.csv, v_d4_amplitude_regression.csv")
