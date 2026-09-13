"""D2 part 5 - is there conditional structure the current arms are NOT using?

Part 5 of d2_analysis.py was degenerate: y and prediction are both differences of a
per-cell metal curve, so the pair residuals inside a cell are exactly additive
(r_AB = u_A - u_B) and "predict one pair's residual from the other pairs" is an
identity (R2 = 1.000).  What that identity really says is that a cell's m(m-1)/2
held-out pairs carry only m-1 independent numbers.

The operational question instead: if ONE measured log SF were revealed for a
held-out cell, how much of the remaining held-out pairs' error would that fix?
Answer with the Gaussian conditional built from the residual variogram, estimated
leave-one-chemotype-out so nothing from the evaluated cell's chemotype leaks in.

Also: per-metal "independence" - how well each metal is predicted from its
neighbours / from all the others, and its unique variance after 1-3 components.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER, SHANNON_RADIUS_CN8, physics_basis

OUT = ROOT / "gen13_separation/analysis/stage2/d2_metal_dependency"
FIG = ROOT / "gen13_separation/figures/stage2"
M = list(LANTHANIDES)
NM = len(M)
IDX = {m: i for i, m in enumerate(M)}
Z = np.array([ATOMIC_NUMBER[m] for m in M], float)
J = np.eye(NM) - np.ones((NM, NM)) / NM

log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


df = pd.read_parquet(ROOT / "gen13_separation/manifests/cohort_exact.parquet")
Yraw = df[["logD__" + m for m in M]].to_numpy(float)
OBS = ~np.isnan(Yraw)
nmet = OBS.sum(axis=1)
C = Yraw - np.nanmean(Yraw, axis=1, keepdims=True)
full = nmet == NM
Cfull = Yraw[full] - Yraw[full].mean(axis=1, keepdims=True)


def loo_linear(Xd, y):
    n = len(y)
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    H = Xd @ XtXi @ Xd.T
    h = np.clip(np.diag(H), 0, 1 - 1e-9)
    beta = XtXi @ Xd.T @ y
    eloo = (y - Xd @ beta) / (1 - h)
    mloo = (y.sum() - y) / (n - 1)
    e0 = y - mloo
    return dict(n=n, r2_loo=1 - float((eloo**2).sum()) / float((e0**2).sum()),
                mae_loo=float(np.abs(eloo).mean()),
                mae_baseline=float(np.abs(e0).mean()))


# ---------------------------------------------------------------- per-metal independence
rows = []
for j in range(NM):
    y = Cfull[:, j]
    others = [k for k in range(NM) if k != j]
    nb = [1, 2] if j == 0 else ([NM - 3, NM - 2] if j == NM - 1 else [j - 1, j + 1])
    rec = dict(metal=M[j], Z=int(Z[j]), var_centred_complete14=float(y.var(ddof=1)),
               sd_centred_complete14=float(y.std(ddof=1)))
    r = loo_linear(np.column_stack([np.ones(len(y)), Cfull[:, nb]]), y)
    rec.update(r2_flank=r["r2_loo"], mae_flank=r["mae_loo"], mae_baseline=r["mae_baseline"])
    r = loo_linear(np.column_stack([np.ones(len(y)), Cfull[:, others]]), y)
    rec.update(r2_all13=r["r2_loo"], mae_all13=r["mae_loo"])
    best = max(((loo_linear(np.column_stack([np.ones(len(y)), Cfull[:, k]]), y)["r2_loo"], M[k])
                for k in others))
    rec.update(r2_best_single=best[0], best_single_partner=best[1])
    rows.append(rec)
per_metal = pd.DataFrame(rows)

# unique variance after k principal components of the complete-14 covariance
S = np.cov(Cfull, rowvar=False, ddof=1)
w, Vv = np.linalg.eigh(S)
o = np.argsort(w)[::-1]
w, Vv = w[o], Vv[:, o]
for k in (1, 2, 3):
    rec = Vv[:, :k] @ np.diag(w[:k]) @ Vv[:, :k].T
    per_metal[f"communality_pc{k}"] = np.diag(rec) / np.diag(S)
    per_metal[f"unique_sd_pc{k}"] = np.sqrt(np.clip(np.diag(S) - np.diag(rec), 0, None))
per_metal.to_csv(OUT / "d2_per_metal_independence.csv", index=False, float_format="%.5g")
say("[5a] per-metal independence on the 78 complete-14 cells:\n" +
    per_metal[["metal", "sd_centred_complete14", "r2_best_single", "best_single_partner",
               "r2_flank", "mae_flank", "r2_all13", "mae_all13",
               "communality_pc1", "communality_pc3", "unique_sd_pc3"]]
    .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
say(f"[5a] least predictable from the rest (LOO R2 predict-from-all-13): " +
    ", ".join(f"{r.metal}={r.r2_all13:.3f}" for r in
              per_metal.sort_values('r2_all13').head(4).itertuples()))
say(f"[5a] residual sd after 3 components (log10 units), largest: " +
    ", ".join(f"{r.metal}={r.unique_sd_pc3:.3f}" for r in
              per_metal.sort_values('unique_sd_pc3', ascending=False).head(5).itertuples()))
say(f"[5a] median within-replicate sd of logD on this cohort = 0.302 -> a metal whose "
    f"unique sd after 3 components is below that is fully explained by the 3-component family")

# ------------------------------------------------------- residual variogram of each arm
chemo = df["chemotype"].to_numpy()
cellrow = {c: i for i, c in enumerate(df["cell_id"])}

arm_rows, curve_store = [], {}
ARMS = ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2", "M_SELECTED", "B1_MEAN_CURVE"]
for arm in ARMS:
    p = pd.read_parquet(ROOT / f"gen13_separation/predictions/B_primary/{arm}.parquet")
    seed = p.split_seed.min()
    p = p[p.split_seed == seed].copy()
    p["resid"] = p.y - p.prediction
    # exact residual curve per cell (residuals are additive: r_AB = u_A - u_B)
    U = np.full((len(df), NM), np.nan)
    for cid, g in p.groupby("cell_id", sort=False):
        mets = sorted(set(g.A) | set(g.B), key=lambda m: IDX[m])
        loc = {m: t for t, m in enumerate(mets)}
        D = np.zeros((len(g), len(mets)))
        D[np.arange(len(g)), [loc[a] for a in g.A]] = 1
        D[np.arange(len(g)), [loc[b] for b in g.B]] = -1
        u, *_ = np.linalg.lstsq(D, g.resid.to_numpy(), rcond=None)
        u -= u.mean()
        U[cellrow[cid], [IDX[m] for m in mets]] = u
    curve_store[arm] = U
    ok = ~np.isnan(U)
    # smoothness of the residual curve: lag-1 correlation along Z on complete cells
    fullU = ok.all(axis=1)
    lag1 = np.nan
    if fullU.sum() > 10:
        a = np.concatenate([U[fullU, i] for i in range(NM - 1)])
        b = np.concatenate([U[fullU, i + 1] for i in range(NM - 1)])
        lag1 = float(np.corrcoef(a, b)[0, 1])
    arm_rows.append(dict(arm=arm, seed=int(seed), n_cells=int(ok.any(axis=1).sum()),
                         pair_mae=float(p.resid.abs().mean()),
                         resid_curve_sd=float(np.nanstd(U)),
                         resid_lag1_corr_along_Z=lag1))
    say(f"[5b] {arm}: pooled pair MAE {p.resid.abs().mean():.3f}, residual-curve sd "
        f"{np.nanstd(U):.3f}, lag-1 correlation of the residual curve along Z "
        f"{lag1:+.3f} (on {int(fullU.sum())} fully covered cells)")
    del p
pd.DataFrame(arm_rows).to_csv(OUT / "d2_arm_residual_curves.csv", index=False,
                              float_format="%.5g")


# ---------------------------------------------- reveal one measured SF, correct the rest
def variogram_sums(U, mask):
    """per-pair sum, sumsq, n of the residual difference over the masked cells."""
    s = np.zeros((NM, NM)); ss = np.zeros((NM, NM)); n = np.zeros((NM, NM))
    Um = U[mask]
    ok = ~np.isnan(Um)
    for i in range(NM):
        for j in range(NM):
            if i == j:
                continue
            k = ok[:, i] & ok[:, j]
            d = Um[k, i] - Um[k, j]
            s[i, j] = d.sum(); ss[i, j] = (d ** 2).sum(); n[i, j] = k.sum()
    return s, ss, n


gain_rows = []
for arm in ARMS:
    U = curve_store[arm]
    S_all, SS_all, N_all = variogram_sums(U, np.ones(len(df), bool))
    per_ct = {}
    for ct in np.unique(chemo):
        per_ct[ct] = variogram_sums(U, chemo == ct)
    recs = []
    for r in range(len(df)):
        u = U[r]
        obs = np.where(~np.isnan(u))[0]
        if len(obs) < 4:
            continue
        ct = chemo[r]
        s, ss, n = (S_all - per_ct[ct][0], SS_all - per_ct[ct][1], N_all - per_ct[ct][2])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_d = np.where(n > 0, s / np.maximum(n, 1), 0.0)
            var_d = np.where(n > 3, (ss - n * mean_d ** 2) / np.maximum(n - 1, 1), np.nan)
        Vr = np.nan_to_num(var_d, nan=float(np.nanmedian(var_d)))
        np.fill_diagonal(Vr, 0.0)
        Vr = 0.5 * (Vr + Vr.T)
        Gr = -0.5 * J @ Vr @ J
        # reveal the widest-dZ observed pair
        a, b = obs[0], obs[-1]
        d_obs = u[a] - u[b]
        for ii in range(len(obs)):
            for jj in range(ii + 1, len(obs)):
                i, j = obs[ii], obs[jj]
                if {i, j} == {a, b}:
                    continue
                y = u[i] - u[j]
                m_ij, m_ab = mean_d[i, j], mean_d[a, b]
                cov = Gr[i, a] - Gr[i, b] - Gr[j, a] + Gr[j, b]
                var_ab = Vr[a, b]
                beta = cov / var_ab if var_ab > 1e-9 else 0.0
                recs.append((df.extractant.iat[r], abs(y),
                             abs(y - m_ij),
                             abs(y - (m_ij + beta * (d_obs - m_ab)))))
    rec = pd.DataFrame(recs, columns=["extractant", "mae_arm", "mae_bias", "mae_cond"])
    macro = rec.groupby("extractant")[["mae_arm", "mae_bias", "mae_cond"]].mean().mean()
    gain_rows.append(dict(arm=arm, n_eval_pairs=len(rec),
                          n_extractants=rec.extractant.nunique(),
                          pooled_mae_arm=rec.mae_arm.mean(),
                          pooled_mae_plus_bias=rec.mae_bias.mean(),
                          pooled_mae_plus_conditional=rec.mae_cond.mean(),
                          macro_mae_arm=macro.mae_arm, macro_mae_plus_bias=macro.mae_bias,
                          macro_mae_plus_conditional=macro.mae_cond,
                          macro_gain=macro.mae_arm - macro.mae_cond))
    say(f"[5c] {arm}: reveal the widest observed log SF in each held-out cell, correct the "
        f"other pairs with the leave-chemotype-out residual covariance -> extractant-macro "
        f"MAE {macro.mae_arm:.3f} -> {macro.mae_cond:.3f} "
        f"(bias-only step {macro.mae_bias:.3f}); gain {macro.mae_arm - macro.mae_cond:+.3f} "
        f"over {len(rec)} held-out pairs in {rec.extractant.nunique()} extractants")
    del rec
pd.DataFrame(gain_rows).to_csv(OUT / "d2_conditional_gain_one_revealed_SF.csv", index=False,
                               float_format="%.5g")

# ------------------------------------------------------------- radius-residual pattern
sf = pd.read_csv(OUT / "d2_pair_separation_factor_stats.csv")
byheavy = sf.groupby("B").mean_abs_logSF_resid_dr.mean().reindex(M).dropna()
bylight = sf.groupby("A").mean_abs_logSF_resid_dr.mean().reindex(M).dropna()
pd.DataFrame({"metal": M,
              "mean_resid_as_heavy_partner": [byheavy.get(m, np.nan) for m in M],
              "mean_resid_as_light_partner": [bylight.get(m, np.nan) for m in M]}
             ).to_csv(OUT / "d2_radius_residual_by_partner.csv", index=False,
                      float_format="%.4f")
say("[4b] mean radius-fit residual of mean|logSF| when the metal is the HEAVY partner: " +
    ", ".join(f"{k}={v:+.3f}" for k, v in byheavy.items()))
say("[4b] ... when it is the LIGHT partner: " +
    ", ".join(f"{k}={v:+.3f}" for k, v in bylight.items()))

# ------------------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
ax.bar(np.arange(NM) - 0.2, per_metal.r2_flank, 0.4, label="from 2 flanking metals")
ax.bar(np.arange(NM) + 0.2, per_metal.r2_all13, 0.4, label="from all 13 others")
ax.set_xticks(range(NM)); ax.set_xticklabels(M, rotation=90, fontsize=8)
ax.set_ylabel("LOO out-of-sample R2"); ax.set_ylim(0, 1)
ax.set_title("D2.5  how well each metal is pinned by the others\n(78 complete-14 cells)",
             fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
ax = axes[1]
U = curve_store["C_DIRECT_ROW"]
ax.plot(Z, np.nanstd(U, axis=0), "-o", ms=4, label="C_DIRECT_ROW residual curve sd")
ax.plot(Z, per_metal.sd_centred_complete14, "-s", ms=4, label="observed centred sd")
ax.plot(Z, per_metal.unique_sd_pc3, "-^", ms=4, label="unique sd after 3 components")
ax.axhline(0.302, color="r", ls="--", lw=1, label="within-replicate sd (0.302)")
ax.set_xticks(Z); ax.set_xticklabels(M, rotation=90, fontsize=8)
ax.set_ylabel("sd of centred logD (log10 units)")
ax.set_title("where the variance and the leftover error live", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "d2_metal_independence.png", dpi=150, bbox_inches="tight")
plt.close(fig)

(OUT / "d2_stdout_part5.txt").write_text("\n".join(log), encoding="utf-8")
print("\nDONE")
