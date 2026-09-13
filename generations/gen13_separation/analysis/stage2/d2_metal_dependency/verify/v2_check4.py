"""Part 4: is the amplitude scalar really PER CELL, or would one global recalibration do?
Plus the naive pairwise-complete correlation artefact (claimed La-Lu = -1.035)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen13sep.metals import LANTHANIDES  # noqa: E402

OUT = ROOT / "generations/gen13_separation/analysis/stage2/d2_metal_dependency/verify"
M = list(LANTHANIDES)
NM = len(M)
IDX = {m: i for i, m in enumerate(M)}
J = np.eye(NM) - np.ones((NM, NM)) / NM

coh = pd.read_parquet(ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet")
X = coh[[f"logD__{m}" for m in M]].to_numpy(dtype=float)
obs = ~np.isnan(X)
C = X - np.nanmean(X, axis=1)[:, None]

print("=" * 78)
print("F1  naive pairwise-complete centred correlation (summary: La-Lu = -1.035)")
print("=" * 78)
cov = np.zeros((NM, NM))
for i in range(NM):
    for j in range(NM):
        k = obs[:, i] & obs[:, j]
        if k.sum() < 2:
            cov[i, j] = np.nan
            continue
        a, b = C[k, i], C[k, j]
        cov[i, j] = ((a - a.mean()) * (b - b.mean())).sum() / (k.sum() - 1)
d = np.sqrt(np.diag(cov))
corr = cov / np.outer(d, d)
print(f"  naive corr(La,Lu) = {corr[IDX['La'], IDX['Lu']]:.4f}   (summary -1.035)")
off = corr[~np.eye(NM, dtype=bool)]
print(f"  entries outside [-1,1]: {int((np.abs(off) > 1).sum())} of {off.size}; "
      f"min={np.nanmin(off):.3f} max={np.nanmax(off):.3f}")
print(f"  m==2 cells: {(obs.sum(1) == 2).sum()} of {len(coh)} "
      f"(their centred pair is exactly (+d/2, -d/2), corr forced to -1)")

print()
print("=" * 78)
print("F2  per-cell amplitude vs ONE global scalar (m>=8, seed 0, C_DIRECT_ROW)")
print("=" * 78)
# PC1 shape from the shrunk variogram
V = np.zeros((NM, NM))
for i, j in itertools.combinations(range(NM), 2):
    k = obs[:, i] & obs[:, j]
    V[i, j] = V[j, i] = np.var(X[k, i] - X[k, j], ddof=1)
G = -0.5 * J @ V @ J
Gs = 0.78 * G + 0.22 * (np.trace(G) / 13) * J
w, v = np.linalg.eigh(Gs)
pc1 = v[:, np.argsort(w)[::-1][0]]
pc1 = (pc1 - pc1.mean()) / np.linalg.norm(pc1 - pc1.mean())

PRED = ROOT / "generations/gen13_separation/predictions/B_primary"
rows = []
for arm_name in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2", "M_SELECTED"]:
    a = pd.read_parquet(PRED / f"{arm_name}.parquet")
    SEED = sorted(a.split_seed.unique())[0]
    a = a[(a.split_seed == SEED) & (a.n_metals >= 8)].copy()
    a["resid"] = a["y"] - a["prediction"]
    a["abserr"] = a["resid"].abs()
    a["x"] = pc1[a.A.map(IDX).to_numpy()] - pc1[a.B.map(IDX).to_numpy()]

    def macro(col):
        return float(a.groupby("extractant")[col].mean().mean())

    base = macro("abserr")
    # one GLOBAL scalar for every cell
    cg = float((a.x * a.resid).sum() / (a.x ** 2).sum())
    a["e_global"] = (a.resid - cg * a.x).abs()
    # one scalar per EXTRACTANT (still not per cell)
    ce = a.groupby("extractant").apply(
        lambda g: float((g.x * g.resid).sum() / (g.x ** 2).sum()), include_groups=False)
    a["e_ext"] = (a.resid - a.extractant.map(ce) * a.x).abs()
    # one scalar per CELL
    cc = a.groupby("cell_id").apply(
        lambda g: float((g.x * g.resid).sum() / (g.x ** 2).sum()), include_groups=False)
    a["e_cell"] = (a.resid - a.cell_id.map(cc) * a.x).abs()
    print(f"  {arm_name:24s} base={base:.4f}  global scalar={macro('e_global'):.4f}  "
          f"per-extractant={macro('e_ext'):.4f}  PER-CELL={macro('e_cell'):.4f}  "
          f"(global c={cg:+.3f}, sd of per-cell c={cc.std():.3f})")
    rows.append({"arm": arm_name, "base": base, "one_global_scalar": macro("e_global"),
                 "per_extractant_scalar": macro("e_ext"), "per_cell_scalar": macro("e_cell"),
                 "global_c": cg, "sd_per_cell_c": float(cc.std()),
                 "frac_removed_per_cell": 1 - macro("e_cell") / base,
                 "frac_removed_global": 1 - macro("e_global") / base})
pd.DataFrame(rows).to_csv(OUT / "v2_global_vs_percell.csv", index=False)
print("\n  -> if the global-scalar column were close to the per-cell column, the finding "
      "would be a\n     recalibration artefact rather than a per-cell amplitude.")
