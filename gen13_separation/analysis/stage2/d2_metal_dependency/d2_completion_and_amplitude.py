"""D2 fixes + the actionable decomposition.

(a) "predict metal j from all the others" was degenerate in d2_conditional_gain.py:
    on complete-14 cells the centred curve sums to zero, so c_j = -sum(c_k) exactly
    and the LOO R2 was 1.000 by construction.  Re-do it in the level-free form that
    a chemist actually faces: metal j was NOT measured, the other 13 were, predict
    t_j = logD_j - mean(logD over the other 13) from the curve of those 13
    (which carries only 12 free numbers, so 12 predictors + intercept).

(b) How much of each arm's leftover error is ONE scalar per cell (an amplitude
    error along a fixed shape) rather than metal-specific error?  Oracle bound.
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
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER

OUT = ROOT / "gen13_separation/analysis/stage2/d2_metal_dependency"
FIG = ROOT / "gen13_separation/figures/stage2"
M = list(LANTHANIDES); NM = len(M); IDX = {m: i for i, m in enumerate(M)}
Z = np.array([ATOMIC_NUMBER[m] for m in M], float)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


df = pd.read_parquet(ROOT / "gen13_separation/manifests/cohort_exact.parquet")
Yraw = df[["logD__" + m for m in M]].to_numpy(float)
OBS = ~np.isnan(Yraw)
nmet = OBS.sum(axis=1)
full = nmet == NM
Yf = Yraw[full]
extr = df["extractant"].to_numpy()
say(f"[6] complete-14 cells: {int(full.sum())}")


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
                mae_loo=float(np.abs(eloo).mean()), mae_baseline=float(np.abs(e0).mean()),
                sd_target=float(y.std(ddof=1)))


# --------------------------------------------- (a) hold one metal out of a complete cell
rows = []
for j in range(NM):
    others = [k for k in range(NM) if k != j]
    mo = Yf[:, others].mean(axis=1)
    t = Yf[:, j] - mo                      # level-free target
    Cx = Yf[:, others] - mo[:, None]       # 13 columns, rank 12
    Xd = np.column_stack([np.ones(len(t)), Cx[:, :-1]])   # drop one to make it full rank
    r = loo_linear(Xd, t)
    nb = [1, 2] if j == 0 else ([NM - 3, NM - 2] if j == NM - 1 else [j - 1, j + 1])
    rnb = loo_linear(np.column_stack([np.ones(len(t)), Cx[:, [others.index(k) for k in nb]]]), t)
    rows.append(dict(metal=M[j], Z=int(Z[j]), sd_target=r["sd_target"],
                     r2_from_other13=r["r2_loo"], mae_from_other13=r["mae_loo"],
                     r2_from_2_flanking=rnb["r2_loo"], mae_from_2_flanking=rnb["mae_loo"],
                     mae_baseline=r["mae_baseline"]))
comp = pd.DataFrame(rows)
comp.to_csv(OUT / "d2_holdout_one_metal_completion.csv", index=False, float_format="%.5g")
say("[6] hold ONE metal out of a complete-14 cell, predict its offset from the other 13 "
    "(level-free, LOO over the 78 cells):\n" +
    comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
say(f"[6] mean over the 14 metals: R2 from other 13 = {comp.r2_from_other13.mean():.3f}, "
    f"MAE = {comp.mae_from_other13.mean():.3f} log10 units "
    f"(baseline MAE {comp.mae_baseline.mean():.3f}); "
    f"from 2 flanking metals R2 = {comp.r2_from_2_flanking.mean():.3f}, "
    f"MAE = {comp.mae_from_2_flanking.mean():.3f}")
worst = comp.sort_values("r2_from_other13").head(4)
say("[6] least completable metals: " +
    ", ".join(f"{r.metal} R2={r.r2_from_other13:.3f} MAE={r.mae_from_other13:.3f}"
              for r in worst.itertuples()))

# --------------------------------------------- (b) amplitude decomposition of arm error
# fixed shape = PC1 of the observed centred curve on complete-14 cells
Cf = Yf - Yf.mean(axis=1, keepdims=True)
S = np.cov(Cf, rowvar=False, ddof=1)
w, Vv = np.linalg.eigh(S)
o = np.argsort(w)[::-1]
V1, V2 = Vv[:, o[0]], Vv[:, o[1]]
if np.dot(V1, Z - Z.mean()) < 0:
    V1 = -V1
cellrow = {c: i for i, c in enumerate(df["cell_id"])}

rows = []
for arm in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2", "M_SELECTED", "M_LOWRANK_K2",
            "M_PHYSICS_radius+radius_sq", "B1_MEAN_CURVE"]:
    p = pd.read_parquet(ROOT / f"gen13_separation/predictions/B_primary/{arm}.parquet")
    seed = int(p.split_seed.min())
    p = p[p.split_seed == seed].copy()
    p["resid"] = p.y - p.prediction
    recs = []
    for cid, g in p.groupby("cell_id", sort=False):
        mets = sorted(set(g.A) | set(g.B), key=lambda m: IDX[m])
        if len(mets) < 3:
            # a 2-metal cell has a single pair; one scalar removes it exactly - excluded
            continue
        loc = {m: t for t, m in enumerate(mets)}
        D = np.zeros((len(g), len(mets)))
        D[np.arange(len(g)), [loc[a] for a in g.A]] = 1
        D[np.arange(len(g)), [loc[b] for b in g.B]] = -1
        u, *_ = np.linalg.lstsq(D, g.resid.to_numpy(), rcond=None)
        u -= u.mean()
        ii = [IDX[m] for m in mets]
        for nk, basis in [(1, [V1]), (2, [V1, V2])]:
            B = np.column_stack([b[ii] - b[ii].mean() for b in basis])
            a, *_ = np.linalg.lstsq(B, u, rcond=None)
            res = u - B @ a
            if nk == 1:
                u1 = res
            else:
                u2 = res
        e = df.extractant.iat[cellrow[cid]]
        recs.append((e, np.abs(D @ u).mean(), np.abs(D @ u1).mean(), np.abs(D @ u2).mean(),
                     float(g.y.abs().mean()), len(g)))
    r = pd.DataFrame(recs, columns=["extractant", "mae", "mae_after1", "mae_after2",
                                    "mean_abs_y", "n_pairs"])
    macro = r.groupby("extractant")[["mae", "mae_after1", "mae_after2", "mean_abs_y"]].mean().mean()
    rows.append(dict(arm=arm, seed=seed, n_cells=len(r), n_extractants=r.extractant.nunique(),
                     macro_mae=macro.mae, macro_mae_oracle_1scalar=macro.mae_after1,
                     macro_mae_oracle_2scalar=macro.mae_after2,
                     frac_error_removed_1scalar=1 - macro.mae_after1 / macro.mae,
                     frac_error_removed_2scalar=1 - macro.mae_after2 / macro.mae,
                     macro_mean_abs_y=macro.mean_abs_y))
    say(f"[7] {arm}: extractant-macro MAE {macro.mae:.3f}; if ONE oracle scalar per cell "
        f"(amplitude along the fixed PC1 shape) were known -> {macro.mae_after1:.3f} "
        f"({100*(1-macro.mae_after1/macro.mae):.0f}% of the error); two scalars -> "
        f"{macro.mae_after2:.3f} ({100*(1-macro.mae_after2/macro.mae):.0f}%); "
        f"mean |log SF| in this set {macro.mean_abs_y:.3f}, {len(r)} cells >=3 metals")
    del p, r
amp = pd.DataFrame(rows)
amp.to_csv(OUT / "d2_amplitude_oracle_decomposition.csv", index=False, float_format="%.5g")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
xs = np.arange(len(amp))
ax.bar(xs - 0.25, amp.macro_mae, 0.25, label="arm as it stands")
ax.bar(xs, amp.macro_mae_oracle_1scalar, 0.25, label="oracle 1 scalar / cell (PC1 amplitude)")
ax.bar(xs + 0.25, amp.macro_mae_oracle_2scalar, 0.25, label="oracle 2 scalars / cell")
ax.axhline(0.302, color="r", ls="--", lw=1, label="within-replicate sd 0.302")
ax.set_xticks(xs); ax.set_xticklabels(amp.arm, rotation=25, ha="right", fontsize=7)
ax.set_ylabel("extractant-macro MAE of log SF")
ax.set_title("D2.7  how much of each arm's error is a single per-cell amplitude", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(FIG / "d2_amplitude_oracle.png", dpi=150, bbox_inches="tight")
plt.close(fig)

(OUT / "d2_stdout_part67.txt").write_text("\n".join(log), encoding="utf-8")
print("\nDONE")
