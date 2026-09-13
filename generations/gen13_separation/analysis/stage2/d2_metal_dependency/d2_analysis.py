"""D2 - dependency between the lanthanides themselves.

Runs on gen13_separation/manifests/cohort_exact.parquet (521 cells x 14 metals).
Writes CSVs into gen13_separation/analysis/stage2/d2_metal_dependency/ and
figures into gen13_separation/figures/stage2/ (prefix d2_).

Design notes that matter for reading the numbers
------------------------------------------------
* The target is the *centred* curve: c_i = logD_i - mean(logD over the metals
  observed in that cell).  Row-centring is a projection, so it INDUCES negative
  correlation between metals inside a cell.  For a cell with m observed metals the
  induced correlation between two of them is exactly -1/(m-1) when the true curve
  is isotropic; for m = 2 it is -1 deterministically (c_j = -c_i).  138 of 521
  cells have m = 2.  Therefore we report three estimators:
    (N)  naive pairwise-complete correlation of the row-centred matrix, all cells
         -> what the literal question asks for, but artefact-dominated;
    (V)  variogram / double-centred estimator, all cells, LEVEL-INVARIANT:
         V[i,j] = Var over cells of (logD_i - logD_j) needs no centring choice at
         all, and G = -0.5 * J V J (J = I - 11'/14) is the covariance of the curve
         centred over the *full* 14-metal axis.  This is the primary estimator.
    (C)  plain sample covariance of the 78 cells that measure all 14 metals,
         centred over all 14 -> unbiased but small-n check on (V).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import (LANTHANIDES, ATOMIC_NUMBER, SHANNON_RADIUS_CN8,
                             physics_basis)

OUT = ROOT / "gen13_separation/analysis/stage2/d2_metal_dependency"
FIG = ROOT / "gen13_separation/figures/stage2"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

M = list(LANTHANIDES)
NM = len(M)
Z = np.array([ATOMIC_NUMBER[m] for m in M], float)
R8 = np.array([SHANNON_RADIUS_CN8[m] for m in M], float)
PHYS = physics_basis()

log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


# ----------------------------------------------------------------------------- load
df = pd.read_parquet(ROOT / "gen13_separation/manifests/cohort_exact.parquet")
Yraw = df[["logD__" + m for m in M]].to_numpy(float)          # 521 x 14, NaN = unmeasured
OBS = ~np.isnan(Yraw)
nmet = OBS.sum(axis=1)
assert (nmet == df["n_metals"].to_numpy()).all()

# row-centred over each cell's own observed metals
C = Yraw - np.nanmean(Yraw, axis=1, keepdims=True)
say(f"[load] {len(df)} cells, {NM} metals, {OBS.sum()} observed logD values, "
    f"median n_metals {int(np.median(nmet))}")

# ============================================================ 1. cov / corr matrices
def pairwise_cov(X: np.ndarray, obs: np.ndarray):
    """Pairwise-complete covariance / correlation / count."""
    cov = np.full((NM, NM), np.nan)
    cnt = np.zeros((NM, NM), int)
    for i in range(NM):
        for j in range(NM):
            k = obs[:, i] & obs[:, j]
            n = int(k.sum())
            cnt[i, j] = n
            if n >= 3:
                a, b = X[k, i], X[k, j]
                cov[i, j] = np.dot(a - a.mean(), b - b.mean()) / (n - 1)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return cov, corr, cnt


cov_N, corr_N, cnt_N = pairwise_cov(C, OBS)

# --- (V) variogram, level-invariant --------------------------------------------
V = np.zeros((NM, NM))
Vn = np.zeros((NM, NM), int)
for i in range(NM):
    for j in range(NM):
        if i == j:
            continue
        k = OBS[:, i] & OBS[:, j]
        d = Yraw[k, i] - Yraw[k, j]
        Vn[i, j] = len(d)
        V[i, j] = d.var(ddof=1) if len(d) >= 3 else np.nan
J = np.eye(NM) - np.ones((NM, NM)) / NM
G = -0.5 * J @ V @ J
G = 0.5 * (G + G.T)
sdG = np.sqrt(np.clip(np.diag(G), 1e-12, None))
corr_V = G / np.outer(sdG, sdG)

# --- (C) complete-14 subset -----------------------------------------------------
full = nmet == NM
Cfull = Yraw[full] - Yraw[full].mean(axis=1, keepdims=True)
cov_C = np.cov(Cfull, rowvar=False, ddof=1)
sdC = np.sqrt(np.clip(np.diag(cov_C), 1e-12, None))
corr_C = cov_C / np.outer(sdC, sdC)
say(f"[1] complete-14 subset: {int(full.sum())} cells")

for nm, mat in [("cov_pairwise_naive", cov_N), ("corr_pairwise_naive", corr_N),
                ("n_pairwise", cnt_N), ("variogram_var_of_difference", V),
                ("cov_variogram_doublecentred", G), ("corr_variogram_doublecentred", corr_V),
                ("cov_complete14", cov_C), ("corr_complete14", corr_C)]:
    pd.DataFrame(mat, index=M, columns=M).to_csv(OUT / f"d2_{nm}.csv",
                                                 float_format="%.6g")

say(f"[1] pairwise co-observation count: min {cnt_N[~np.eye(NM, dtype=bool)].min()} "
    f"(pair {M[np.unravel_index(np.argmin(np.where(np.eye(NM, dtype=bool), 10**6, cnt_N)), (NM,NM))[0]]}-"
    f"{M[np.unravel_index(np.argmin(np.where(np.eye(NM, dtype=bool), 10**6, cnt_N)), (NM,NM))[1]]}), "
    f"median {int(np.median(cnt_N[~np.eye(NM, dtype=bool)]))}, max {cnt_N[~np.eye(NM, dtype=bool)].max()}")
say(f"[1] diag of naive centred cov (var of centred logD, log10 units^2): "
    f"{np.round(np.diag(cov_N),3).tolist()}")
say(f"[1] diag of variogram cov G: {np.round(np.diag(G),3).tolist()}")
say(f"[1] mean off-diag corr  naive={np.nanmean(corr_N[~np.eye(NM, dtype=bool)]):.3f}  "
    f"variogram={np.nanmean(corr_V[~np.eye(NM, dtype=bool)]):.3f}  "
    f"complete14={np.nanmean(corr_C[~np.eye(NM, dtype=bool)]):.3f}")
# adjacent-pair correlations
adj_N = [corr_N[i, i + 1] for i in range(NM - 1)]
adj_V = [corr_V[i, i + 1] for i in range(NM - 1)]
adj_C = [corr_C[i, i + 1] for i in range(NM - 1)]
say("[1] adjacent-pair corr (V): " +
    ", ".join(f"{M[i]}-{M[i+1]}={adj_V[i]:.3f}" for i in range(NM - 1)))
say(f"[1] La-Lu corr: naive={corr_N[0,-1]:.3f} variogram={corr_V[0,-1]:.3f} "
    f"complete14={corr_C[0,-1]:.3f}")

# how much of the naive matrix is centring artefact: expected induced corr
contrib2 = np.zeros((NM, NM), int)
for i in range(NM):
    for j in range(NM):
        contrib2[i, j] = int((OBS[:, i] & OBS[:, j] & (nmet == 2)).sum())
frac2 = contrib2 / np.maximum(cnt_N, 1)
pd.DataFrame(frac2, index=M, columns=M).to_csv(OUT / "d2_fraction_from_2metal_cells.csv",
                                               float_format="%.4f")
say(f"[1] fraction of each pair's cells that are 2-metal cells (corr forced to -1): "
    f"mean {frac2[~np.eye(NM, dtype=bool)].mean():.3f}, max {frac2[~np.eye(NM, dtype=bool)].max():.3f}")

# heatmap
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
for ax, mat, ttl in zip(axes, [corr_N, corr_V, corr_C],
                        ["(N) naive row-centred, pairwise, 521 cells",
                         "(V) variogram double-centred, 521 cells",
                         "(C) complete-14 cells only (n=78)"]):
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(NM)); ax.set_xticklabels(M, rotation=90, fontsize=8)
    ax.set_yticks(range(NM)); ax.set_yticklabels(M, fontsize=8)
    ax.set_title(ttl, fontsize=9)
    for i in range(NM):
        for j in range(NM):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "-."),
                        ha="center", va="center", fontsize=5.0,
                        color="white" if abs(v) > 0.55 else "black")
fig.colorbar(im, ax=axes, shrink=0.8, label="correlation of centred logD")
fig.suptitle("D2.1  Correlation of the centred lanthanide curve across cells "
             "(metals ordered by atomic number)", fontsize=11)
fig.savefig(FIG / "d2_corr_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================ 2. eigen-decomposition
def shrink_to_isotropic(S, alpha):
    """(1-a)S + a * (tr S / rank) * P, P = J = projector onto the mean-zero subspace."""
    return (1 - alpha) * S + alpha * (np.trace(S) / (NM - 1)) * J


def eig_in_subspace(S):
    """Eigen-decompose inside the 13-dim mean-zero subspace (the centred curve lives there)."""
    w, Vv = np.linalg.eigh(0.5 * (S + S.T))
    order = np.argsort(w)[::-1]
    return w[order], Vv[:, order]


alpha_grid = np.arange(0.0, 0.51, 0.01)
chosen = None
for a in alpha_grid:
    w, _ = eig_in_subspace(shrink_to_isotropic(G, a))
    if w[NM - 2] > 1e-6 * w[0]:          # 13 leading eigenvalues strictly positive
        chosen = float(a)
        break
say(f"[2] shrinkage: linear shrinkage of the variogram covariance G toward an "
    f"isotropic covariance on the mean-zero subspace, alpha = {chosen:.2f} "
    f"(smallest alpha on a 0.01 grid that makes all 13 in-subspace eigenvalues > 0)")

rows = []
for label, S, a in [("V_variogram_shrunk", shrink_to_isotropic(G, chosen), chosen),
                    ("V_variogram_raw", G, 0.0),
                    ("C_complete14_shrunk", shrink_to_isotropic(cov_C, 0.05), 0.05),
                    ("N_naive_shrunk", shrink_to_isotropic(np.nan_to_num(cov_N), 0.05), 0.05)]:
    w, Vv = eig_in_subspace(S)
    wpos = np.clip(w[:NM - 1], 0, None)
    tot = wpos.sum()
    rows.append(dict(matrix=label, alpha=a, total_var=tot,
                     **{f"frac_pc{k+1}": wpos[k] / tot for k in range(5)},
                     cum_pc3=wpos[:3].sum() / tot))
    if label == "V_variogram_shrunk":
        W_main, EV_main = w, Vv
eigsum = pd.DataFrame(rows)
eigsum.to_csv(OUT / "d2_eigen_variance_fractions.csv", index=False, float_format="%.5g")
say("[2] variance fractions:\n" + eigsum.to_string(index=False,
                                                   float_format=lambda x: f"{x:.4f}"))

# sign convention: PC1 positive slope toward heavy
for k in range(3):
    if np.dot(EV_main[:, k], Z - Z.mean()) < 0 and k == 0:
        EV_main[:, k] *= -1
    elif k > 0 and EV_main[np.argmax(np.abs(EV_main[:, k])), k] < 0:
        EV_main[:, k] *= -1

ev_df = pd.DataFrame({"metal": M, "Z": Z.astype(int), "radius_CN8": R8})
for k in range(4):
    ev_df[f"pc{k+1}"] = EV_main[:, k]
ev_df["mean_centred_curve"] = np.array([np.nanmean(C[OBS[:, i], i]) for i in range(NM)])
ev_df.to_csv(OUT / "d2_eigenvectors.csv", index=False, float_format="%.5g")

# correlate eigenvectors with physics basis
pb = pd.DataFrame(PHYS, index=M)
rows = []
for k in range(4):
    v = EV_main[:, k]
    for name in pb.columns:
        b = pb[name].to_numpy()
        r = np.corrcoef(v, b)[0, 1]
        rows.append(dict(component=f"pc{k+1}", basis=name, pearson_r=r, r2=r * r))
phys_tab = pd.DataFrame(rows)
phys_tab.to_csv(OUT / "d2_eigenvector_physics_correlation.csv", index=False,
                float_format="%.4f")
for k in range(3):
    sub = phys_tab[phys_tab.component == f"pc{k+1}"].reindex(
        phys_tab[phys_tab.component == f"pc{k+1}"].r2.sort_values(ascending=False).index)
    say(f"[2] pc{k+1} vs physics basis (top 4 by r2): " +
        ", ".join(f"{r.basis} r={r.pearson_r:+.3f}" for r in sub.head(4).itertuples()))

# best 2-term physics fit to each PC
rows = []
for k in range(3):
    v = EV_main[:, k]
    best = None
    for a1, a2 in itertools.combinations(pb.columns, 2):
        Xd = np.column_stack([pb[a1], pb[a2]])
        beta, *_ = np.linalg.lstsq(Xd, v, rcond=None)
        r2 = 1 - ((v - Xd @ beta) ** 2).sum() / (v ** 2).sum()
        if best is None or r2 > best[2]:
            best = (a1, a2, r2)
    rows.append(dict(component=f"pc{k+1}", term1=best[0], term2=best[1], r2=best[2]))
    say(f"[2] pc{k+1} best 2-term physics fit: {best[0]} + {best[1]}, R2 = {best[2]:.3f}")
pd.DataFrame(rows).to_csv(OUT / "d2_eigenvector_best2term.csv", index=False,
                          float_format="%.4f")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
for k, cstyle in zip(range(3), ["-o", "-s", "-^"]):
    wpos = np.clip(W_main[:NM - 1], 0, None)
    ax.plot(Z, EV_main[:, k], cstyle, ms=4,
            label=f"PC{k+1} ({100*wpos[k]/wpos.sum():.0f}% of centred variance)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(Z); ax.set_xticklabels(M, rotation=90, fontsize=8)
ax.set_ylabel("eigenvector loading"); ax.set_title("D2.2  first three eigenvectors")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax = axes[1]
for name in ["radius", "radius_sq", "gd_break", "tetrad_e3"]:
    ax.plot(Z, pb[name] / np.linalg.norm(pb[name]), "--", lw=1.2, alpha=0.75, label=name)
for k, cstyle in zip(range(2), ["-o", "-s"]):
    ax.plot(Z, EV_main[:, k], cstyle, ms=4, color="k" if k == 0 else "0.45",
            label=f"PC{k+1}")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(Z); ax.set_xticklabels(M, rotation=90, fontsize=8)
ax.set_title("vs unit-normalised physics basis"); ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "d2_eigenvectors.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================ 3. conditional prediction
def loo_linear(Xd, y):
    """Leave-one-out OLS via the hat matrix. Xd includes the intercept column."""
    n = len(y)
    XtX = Xd.T @ Xd
    try:
        XtXi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    H = Xd @ XtXi @ Xd.T
    h = np.clip(np.diag(H), 0, 1 - 1e-9)
    beta = XtXi @ Xd.T @ y
    e = y - Xd @ beta
    eloo = e / (1 - h)
    # intercept-only LOO baseline
    mloo = (y.sum() - y) / (n - 1)
    e0 = y - mloo
    sse, sse0 = float((eloo ** 2).sum()), float((e0 ** 2).sum())
    return dict(n=n, r2_loo=1 - sse / sse0, mae_loo=float(np.abs(eloo).mean()),
                rmse_loo=float(np.sqrt(sse / n)),
                mae_loo_baseline=float(np.abs(e0).mean()), slope=float(beta[1]) if Xd.shape[1] > 1 else np.nan)


def build_pairtable(mask_cells, tag, min_n=25):
    rows = []
    for i in range(NM):
        for j in range(NM):
            if i == j:
                continue
            k = mask_cells & OBS[:, i] & OBS[:, j]
            if k.sum() < min_n:
                continue
            x, y = C[k, i], C[k, j]
            Xd = np.column_stack([np.ones(k.sum()), x])
            res = loo_linear(Xd, y)
            if res is None:
                continue
            res.update(source=M[i], target=M[j], dZ=int(abs(Z[i] - Z[j])),
                       didx=abs(i - j), subset=tag,
                       n_2metal=int((k & (nmet == 2)).sum()))
            rows.append(res)
    return pd.DataFrame(rows)


pt_all = build_pairtable(np.ones(len(df), bool), "all521")
pt_r6 = build_pairtable(nmet >= 6, "nmet>=6")
pt_f14 = build_pairtable(full, "complete14", min_n=25)
pairtab = pd.concat([pt_all, pt_r6, pt_f14], ignore_index=True)
pairtab.to_csv(OUT / "d2_pairwise_loo_prediction.csv", index=False, float_format="%.5g")

for tag, pt in [("all521", pt_all), ("nmet>=6", pt_r6), ("complete14", pt_f14)]:
    say(f"[3] {tag}: {len(pt)} ordered pairs with n>=25; "
        f"mean LOO R2 {pt.r2_loo.mean():.3f}, median {pt.r2_loo.median():.3f}, "
        f"mean LOO MAE {pt.mae_loo.mean():.3f} vs baseline MAE {pt.mae_loo_baseline.mean():.3f}")

for tag, pt in [("all521", pt_all), ("nmet>=6", pt_r6), ("complete14", pt_f14)]:
    piv = pt.pivot(index="source", columns="target", values="r2_loo").reindex(index=M, columns=M)
    piv.to_csv(OUT / f"d2_loo_r2_{tag.replace('>=','ge').replace('=','')}.csv", float_format="%.4f")
    piv2 = pt.pivot(index="source", columns="target", values="mae_loo").reindex(index=M, columns=M)
    piv2.to_csv(OUT / f"d2_loo_mae_{tag.replace('>=','ge').replace('=','')}.csv", float_format="%.4f")
    pn = pt.pivot(index="source", columns="target", values="n").reindex(index=M, columns=M)
    pn.to_csv(OUT / f"d2_loo_n_{tag.replace('>=','ge').replace('=','')}.csv")

# decay of R2 with dZ
dec = (pairtab.groupby(["subset", "dZ"])
       .agg(n_pairs=("r2_loo", "size"), mean_r2=("r2_loo", "mean"),
            median_r2=("r2_loo", "median"), mean_mae=("mae_loo", "mean"),
            mean_n=("n", "mean")).reset_index())
dec.to_csv(OUT / "d2_r2_decay_with_dZ.csv", index=False, float_format="%.4f")
say("[3] R2 decay with |dZ| (subset complete14):\n" +
    dec[dec.subset == "complete14"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
say("[3] R2 decay with |dZ| (subset nmet>=6):\n" +
    dec[dec.subset == "nmet>=6"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# --- two flanking neighbours ---------------------------------------------------
rows = []
for tag, mask in [("all521", np.ones(len(df), bool)), ("nmet>=6", nmet >= 6),
                  ("complete14", full)]:
    for j in range(NM):
        if j == 0:
            nb = [1, 2]
        elif j == NM - 1:
            nb = [NM - 3, NM - 2]
        else:
            nb = [j - 1, j + 1]
        k = mask & OBS[:, j] & OBS[:, nb[0]] & OBS[:, nb[1]]
        if k.sum() < 25:
            continue
        Xd = np.column_stack([np.ones(k.sum()), C[k, nb[0]], C[k, nb[1]]])
        res = loo_linear(Xd, C[k, j])
        if res is None:
            continue
        res.pop("slope", None)
        res.update(target=M[j], neighbours=f"{M[nb[0]]}+{M[nb[1]]}", subset=tag)
        rows.append(res)
nbtab = pd.DataFrame(rows)
nbtab.to_csv(OUT / "d2_neighbour_pair_loo_prediction.csv", index=False, float_format="%.5g")
for tag in ["all521", "nmet>=6", "complete14"]:
    s = nbtab[nbtab.subset == tag]
    if len(s):
        say(f"[3] two flanking neighbours, {tag}: {len(s)} metals, mean LOO R2 "
            f"{s.r2_loo.mean():.3f}, mean LOO MAE {s.mae_loo.mean():.3f} "
            f"(baseline {s.mae_loo_baseline.mean():.3f})")
say("[3] flanking-neighbour detail (complete14):\n" +
    nbtab[nbtab.subset == "complete14"][["target", "neighbours", "n", "r2_loo",
                                         "mae_loo", "mae_loo_baseline"]]
    .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
piv = pt_f14.pivot(index="source", columns="target", values="r2_loo").reindex(index=M, columns=M)
im = ax.imshow(piv.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(NM)); ax.set_xticklabels(M, rotation=90, fontsize=8)
ax.set_yticks(range(NM)); ax.set_yticklabels(M, fontsize=8)
ax.set_xlabel("target j"); ax.set_ylabel("source i")
ax.set_title("LOO out-of-sample R2, predict centred j from centred i\n(complete-14 cells, n=78)",
             fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.85)
ax = axes[1]
for tag, mk in [("complete14", "o"), ("nmet>=6", "s"), ("all521", "^")]:
    s = pairtab[pairtab.subset == tag]
    g = s.groupby("dZ").r2_loo.mean()
    ax.plot(g.index, g.values, mk + "-", ms=4, label=tag)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("|Z_i - Z_j|"); ax.set_ylabel("mean LOO R2")
ax.set_title("D2.3  decay of pairwise predictability with atomic-number distance", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "d2_conditional_prediction.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================ 4. separation factors
extr = df["extractant"].to_numpy()
rows = []
for i in range(NM):
    for j in range(i + 1, NM):
        k = OBS[:, i] & OBS[:, j]
        n = int(k.sum())
        sf = Yraw[k, i] - Yraw[k, j]          # light minus heavy
        e = extr[k]
        macro_abs = pd.Series(np.abs(sf)).groupby(pd.Series(e)).mean().mean()
        rows.append(dict(A=M[i], B=M[j], dZ=int(Z[j] - Z[i]), didx=j - i,
                         d_radius_CN8=R8[i] - R8[j], n_cells=n,
                         n_extractants=int(pd.unique(e).size),
                         mean_logSF=float(sf.mean()), sd_logSF=float(sf.std(ddof=1)),
                         median_logSF=float(np.median(sf)),
                         frac_heavy_preferred=float((sf < 0).mean()),
                         mean_abs_logSF=float(np.abs(sf).mean()),
                         mean_abs_logSF_extractant_macro=float(macro_abs),
                         p95_abs_logSF=float(np.percentile(np.abs(sf), 95))))
sf_tab = pd.DataFrame(rows)

x = sf_tab.d_radius_CN8.to_numpy()
for ycol in ["mean_abs_logSF", "mean_abs_logSF_extractant_macro"]:
    yv = sf_tab[ycol].to_numpy()
    Xd = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    pred = Xd @ beta
    r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    sf_tab[ycol + "_pred_dr"] = pred
    sf_tab[ycol + "_resid_dr"] = yv - pred
    say(f"[4] OLS {ycol} ~ |d radius CN8|: slope {beta[1]:.3f} per Angstrom, "
        f"intercept {beta[0]:.3f}, R2 = {r2:.3f} over 91 pairs")
    # no-intercept version (physically, zero radius difference -> zero SF)
    b0 = float(np.dot(x, yv) / np.dot(x, x))
    r2_0 = 1 - ((yv - b0 * x) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    say(f"[4]   through-origin version: slope {b0:.3f}, R2 vs mean = {r2_0:.3f}")

sf_tab.to_csv(OUT / "d2_pair_separation_factor_stats.csv", index=False, float_format="%.5g")
say(f"[4] 91 pairs; n_cells min {sf_tab.n_cells.min()} median "
    f"{int(sf_tab.n_cells.median())} max {sf_tab.n_cells.max()}")
say(f"[4] mean |logSF| pooled: min {sf_tab.mean_abs_logSF.min():.3f} "
    f"({sf_tab.loc[sf_tab.mean_abs_logSF.idxmin(),'A']}-{sf_tab.loc[sf_tab.mean_abs_logSF.idxmin(),'B']}), "
    f"max {sf_tab.mean_abs_logSF.max():.3f} "
    f"({sf_tab.loc[sf_tab.mean_abs_logSF.idxmax(),'A']}-{sf_tab.loc[sf_tab.mean_abs_logSF.idxmax(),'B']})")
say(f"[4] fraction heavy-preferred over the 91 pairs: mean "
    f"{sf_tab.frac_heavy_preferred.mean():.3f}, range "
    f"{sf_tab.frac_heavy_preferred.min():.3f}-{sf_tab.frac_heavy_preferred.max():.3f}")
adj = sf_tab[sf_tab.didx == 1]
say("[4] adjacent pairs: " + ", ".join(
    f"{r.A}-{r.B} n={r.n_cells} mean={r.mean_logSF:+.3f} sd={r.sd_logSF:.3f} "
    f"|SF|={r.mean_abs_logSF:.3f} heavy={r.frac_heavy_preferred:.2f}"
    for r in adj.itertuples()))
res = sf_tab.sort_values("mean_abs_logSF_resid_dr")
say("[4] most NEGATIVE residuals (less separation than radius predicts): " + ", ".join(
    f"{r.A}-{r.B} {r.mean_abs_logSF_resid_dr:+.3f}" for r in res.head(6).itertuples()))
say("[4] most POSITIVE residuals (more separation than radius predicts): " + ", ".join(
    f"{r.A}-{r.B} {r.mean_abs_logSF_resid_dr:+.3f}" for r in res.tail(6).itertuples()))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
sc = ax.scatter(sf_tab.d_radius_CN8, sf_tab.mean_abs_logSF, c=sf_tab.dZ,
                cmap="plasma", s=26)
xs = np.linspace(0, sf_tab.d_radius_CN8.max(), 10)
Xd = np.column_stack([np.ones(len(x)), x]); beta, *_ = np.linalg.lstsq(Xd, sf_tab.mean_abs_logSF, rcond=None)
ax.plot(xs, beta[0] + beta[1] * xs, "k--", lw=1)
ax.set_xlabel("|Shannon radius difference| CN8 (A)"); ax.set_ylabel("mean |log SF|")
ax.set_title("D2.4  mean |log SF| vs radius difference (91 pairs)", fontsize=9)
fig.colorbar(sc, ax=ax, label="dZ", shrink=0.85)
ax = axes[1]
Rr = np.full((NM, NM), np.nan)
for r in sf_tab.itertuples():
    i, j = M.index(r.A), M.index(r.B)
    Rr[i, j] = r.mean_abs_logSF_resid_dr
    Rr[j, i] = r.mean_abs_logSF_resid_dr
mx = np.nanmax(np.abs(Rr))
im = ax.imshow(Rr, cmap="PuOr_r", vmin=-mx, vmax=mx)
ax.set_xticks(range(NM)); ax.set_xticklabels(M, rotation=90, fontsize=8)
ax.set_yticks(range(NM)); ax.set_yticklabels(M, fontsize=8)
ax.set_title("residual of mean |log SF| after radius fit", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.85)
ax = axes[2]
ax.bar(range(len(adj)), adj.mean_logSF, yerr=adj.sd_logSF, color="steelblue", capsize=2)
ax.set_xticks(range(len(adj)))
ax.set_xticklabels([f"{r.A}-{r.B}" for r in adj.itertuples()], rotation=90, fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("mean log SF (light - heavy)")
ax.set_title("adjacent-pair log SF, mean +- sd", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "d2_separation_factors.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================ 5. what the arms leave
# per-cell residual curves of the best single-model arm, and whether the residual
# of one pair in a cell predicts the residual of another pair in the same cell.
arm_rows = []
for arm in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2", "M_LOWRANK_K2", "M_SELECTED"]:
    p = pd.read_parquet(ROOT / f"gen13_separation/predictions/B_primary/{arm}.parquet")
    p = p[p.split_seed == p.split_seed.min()].copy()      # one seed keeps memory small
    p["resid"] = p.y - p.prediction
    # within each cell, fit the residual to an additive metal curve  r_AB = u_A - u_B
    keep, rank1 = [], []
    for cid, g in p.groupby("cell_id", sort=False):
        mets = sorted(set(g.A) | set(g.B), key=lambda m: M.index(m))
        if len(mets) < 4:
            continue
        idx = {m: t for t, m in enumerate(mets)}
        D = np.zeros((len(g), len(mets)))
        D[np.arange(len(g)), [idx[a] for a in g.A]] = 1
        D[np.arange(len(g)), [idx[b] for b in g.B]] = -1
        r = g.resid.to_numpy()
        u, *_ = np.linalg.lstsq(D, r, rcond=None)
        u -= u.mean()
        fit = D @ u
        keep.append((float((fit ** 2).sum()), float((r ** 2).sum())))
        rank1.append(len(mets))
    ex = np.array(keep)
    frac_additive = ex[:, 0].sum() / ex[:, 1].sum()
    # leave-one-pair-out: predict a pair's residual from the additive fit of the OTHER pairs
    sse_loo, sse_0, n_used, abs_loo = 0.0, 0.0, 0, []
    for cid, g in p.groupby("cell_id", sort=False):
        mets = sorted(set(g.A) | set(g.B), key=lambda m: M.index(m))
        if len(mets) < 5:
            continue
        idx = {m: t for t, m in enumerate(mets)}
        D = np.zeros((len(g), len(mets)))
        D[np.arange(len(g)), [idx[a] for a in g.A]] = 1
        D[np.arange(len(g)), [idx[b] for b in g.B]] = -1
        r = g.resid.to_numpy()
        for t in range(len(g)):
            m = np.ones(len(g), bool); m[t] = False
            Dm = D[m]
            if np.linalg.matrix_rank(Dm) < len(mets) - 1:
                continue
            u, *_ = np.linalg.lstsq(Dm, r[m], rcond=None)
            pr = float(D[t] @ u)
            sse_loo += (r[t] - pr) ** 2
            sse_0 += r[t] ** 2
            abs_loo.append(abs(r[t] - pr))
            n_used += 1
    r2_struct = 1 - sse_loo / sse_0 if sse_0 > 0 else np.nan
    arm_rows.append(dict(arm=arm, n_cells=len(ex), seed=int(p.split_seed.iloc[0]),
                         resid_mae=float(p.resid.abs().mean()),
                         frac_resid_variance_additive_in_cell=frac_additive,
                         n_loo_pairs=n_used, loo_r2_resid_from_other_pairs=r2_struct,
                         loo_mae_resid=float(np.mean(abs_loo)) if abs_loo else np.nan))
    say(f"[5] {arm}: pooled residual MAE {p.resid.abs().mean():.3f}; "
        f"{100*frac_additive:.1f}% of within-cell residual variance is an additive metal curve; "
        f"leave-one-pair-out R2 of a pair's residual from the other pairs in the same cell "
        f"= {r2_struct:+.3f} over {n_used} held-out pairs")
    del p
pd.DataFrame(arm_rows).to_csv(OUT / "d2_arm_residual_structure.csv", index=False,
                              float_format="%.5g")

(OUT / "d2_stdout.txt").write_text("\n".join(log), encoding="utf-8")
print("\nDONE")
