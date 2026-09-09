"""Third verification pass for D7: weighting-robustness stress tests.
(a) does one big cell dominate the extractant-macro pair average?
(b) does the per-element residual curve survive extractant-macro weighting?
(c) does the reproducible amplitude survive dropping the dominant chemotype sc009?
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

ROOT = "gen13_separation"
OUT = os.path.join(ROOT, "analysis", "stage2", "d7_tetrad_residual", "verify")
LN = list(LANTHANIDES)
NEL = len(LN)
IDX = {m: i for i, m in enumerate(LN)}
r_raw = np.array([SHANNON_RADIUS_CN8[m] for m in LN], float)
Z = (r_raw - r_raw.mean()) / r_raw.std(ddof=0)

coh = pd.read_parquet(os.path.join(ROOT, "manifests", "cohort_exact.parquet"))
logD = coh[[f"logD__{m}" for m in LN]].to_numpy(float)
obs = np.isfinite(logD)
n_obs = obs.sum(1)
Y = np.where(obs, logD - (np.nansum(np.where(obs, logD, 0.0), 1) /
                          np.maximum(n_obs, 1))[:, None], np.nan)
sel = n_obs >= 6
FIT = np.full((len(coh), NEL), np.nan)
R = np.full((len(coh), NEL), np.nan)
for i in np.where(sel)[0]:
    m = obs[i]
    X = np.vander(Z[m], 3, increasing=True)
    beta, *_ = np.linalg.lstsq(X, Y[i, m], rcond=None)
    FIT[i, m] = X @ beta
    R[i, m] = Y[i, m] - FIT[i, m]

extr = coh["extractant"].to_numpy()
chem = coh["chemotype"].to_numpy()

# ---------------- (a) pair-mass concentration on the oracle baseline
recs = []
for i in np.where(sel)[0]:
    ms = np.where(obs[i])[0]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            recs.append((i, extr[i], ms[a], ms[b]))
pairs = pd.DataFrame(recs, columns=["row", "extractant", "ia", "ib"])
cellsz = pairs.groupby(["extractant", "row"]).size().rename("npair").reset_index()
share = cellsz.groupby("extractant")["npair"].apply(lambda s: s.max() / s.sum())
print("(a) within-extractant pair mass held by that extractant's LARGEST cell:")
print(f"    median {share.median():.2f}, mean {share.mean():.2f}, "
      f"{(share > 0.9).mean()*100:.0f}% of the 77 extractants are >90% one cell "
      f"({(cellsz.groupby('extractant').size() == 1).sum()} extractants have only one cell)")

row = pairs["row"].to_numpy()
ia = pairs["ia"].to_numpy()
ib = pairs["ib"].to_numpy()
err = (Y[row, ia] - Y[row, ib]) - (FIT[row, ia] - FIT[row, ib])
uniq = np.unique(extr[sel])
loeo = {}
for e in uniq:
    keep = (extr[sel] != e)
    with np.errstate(invalid="ignore"):
        loeo[e] = np.nan_to_num(np.nanmean(R[sel][keep], axis=0))
C = np.array([loeo[e] for e in pairs["extractant"].to_numpy()])
corr = C[np.arange(len(pairs)), ia] - C[np.arange(len(pairs)), ib]

df = pd.DataFrame({"extractant": pairs["extractant"].to_numpy(), "row": row})


def macro_pairpool(ev):
    df["ae"] = np.abs(ev)
    return float(df.groupby("extractant")["ae"].mean().mean())


def macro_cellfirst(ev):
    df["ae"] = np.abs(ev)
    return float(df.groupby(["extractant", "row"])["ae"].mean()
                 .groupby("extractant").mean().mean())


for tag, fn in [("pair-pooled-in-extractant (D7 convention)", macro_pairpool),
                ("cell-macro then extractant-macro", macro_cellfirst)]:
    b = fn(err)
    best = min((fn(err - lam * corr), lam) for lam in np.round(np.arange(0, 2.01, 0.05), 2))
    print(f"    {tag:42s} base={b:.4f} lam=1 -> {fn(err-corr):.4f} "
          f"({fn(err-corr)-b:+.4f})  best={best[0]:.4f} at lam={best[1]:.2f} ({best[0]-b:+.4f})")

# ---------------- (b) per-element residual under extractant-macro weighting
print("\n(b) per-element mean residual: cell-equal vs extractant-macro weighting")
sub = pd.DataFrame(R[sel], columns=LN)
sub["extractant"] = extr[sel]
sub["chem"] = chem[sel]
cell_eq = sub[LN].mean()
extr_macro = sub.groupby("extractant")[LN].mean().mean()
no_sc009 = sub[sub["chem"] != "sc009"][LN].mean()
tab = pd.DataFrame({"cell_equal": cell_eq, "extractant_macro": extr_macro,
                    "excl_sc009": no_sc009,
                    "n_cells_sc009": sub[sub.chem == "sc009"][LN].notna().sum(),
                    "n_cells_other": sub[sub.chem != "sc009"][LN].notna().sum()})
print(tab.round(4).to_string())
tab.to_csv(os.path.join(OUT, "V_c8_weighting_robustness.csv"))

# ---------------- (c) reproducible amplitude excluding sc009
print("\n(c) reproducible rms amplitude, extractant half-splits")
for tag, mask in [("all 268 cells", np.ones(sub.shape[0], bool)),
                  ("excluding sc009", (sub["chem"] != "sc009").to_numpy())]:
    Rm = R[sel][mask]
    em = extr[sel][mask]
    u = np.unique(em)
    gi = {g: np.where(em == g)[0] for g in u}
    rr = np.random.default_rng(5)
    vals, cors = [], []
    for _ in range(1000):
        pm = rr.permutation(len(u))
        h1 = np.concatenate([gi[g] for g in u[pm[: len(u) // 2]]])
        h2 = np.concatenate([gi[g] for g in u[pm[len(u) // 2:]]])
        with np.errstate(invalid="ignore"):
            c1, c2 = np.nanmean(Rm[h1], 0), np.nanmean(Rm[h2], 0)
        ok = np.isfinite(c1) & np.isfinite(c2)
        cv = float((c1[ok] * c2[ok]).mean())
        vals.append(np.sign(cv) * np.sqrt(abs(cv)))
        cors.append(float(np.corrcoef(c1[ok], c2[ok])[0, 1]))
    print(f"    {tag:18s} n_cells={mask.sum():3d} n_extr={len(u):2d}  "
          f"amplitude={np.median(vals):.4f}  split-half r={np.median(cors):+.3f}")
