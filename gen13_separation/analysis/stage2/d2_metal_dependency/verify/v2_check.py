"""Independent re-derivation of the three load-bearing numbers of diagnostic d2.

Written from scratch without reading the d2 scripts.

Checks
  C1  PC1 variance fraction of the centred lanthanide curve (variogram covariance,
      shrunk / unshrunk / complete-14 sample covariance) and its correlation with the
      Shannon CN8 radius basis.
  C2  Fraction of C_DIRECT_ROW extractant-macro MAE removed by one oracle per-cell
      amplitude scalar along the fixed PC1 shape, on held-out cells with >= 8 metals.
  C3  Convention sanity: C_DIRECT_ROW extractant-macro MAE over all cells / 5 seeds
      (programme reference 0.495), plus centring and dominance controls.
  C4  mean |log SF| vs |Shannon CN8 radius difference| over the 91 pairs.

Run:  /d/ml_separator_gh/.venv/Scripts/python.exe <this file>
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import (  # noqa: E402
    ATOMIC_NUMBER,
    LANTHANIDES,
    SHANNON_RADIUS_CN8,
    physics_basis,
)

OUT = ROOT / "gen13_separation/analysis/stage2/d2_metal_dependency/verify"
OUT.mkdir(parents=True, exist_ok=True)
M = list(LANTHANIDES)
NM = len(M)
IDX = {m: i for i, m in enumerate(M)}
rng = np.random.default_rng(20260908)


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# --------------------------------------------------------------------------- data
coh = pd.read_parquet(ROOT / "gen13_separation/manifests/cohort_exact.parquet")
X = coh[[f"logD__{m}" for m in M]].to_numpy(dtype=float)  # 521 x 14, NaN = unmeasured
obs = ~np.isnan(X)
n_obs = obs.sum(1)
assert (n_obs == coh["n_metals"].to_numpy()).all(), "n_metals column disagrees with NaN pattern"

# centre each cell over the metals OBSERVED IN THAT CELL (not over all 14)
row_mean = np.nanmean(X, axis=1)
C = X - row_mean[:, None]  # centred curve, NaN preserved
# proof that centring used the observed subset only:
_chk = np.nanmean(C, axis=1)
assert np.nanmax(np.abs(_chk)) < 1e-12, "row centring is not zero-mean over observed metals"
print(f"cells={len(coh)}  metals={NM}  complete-14 cells={(n_obs == 14).sum()}  "
      f"m>=8 cells={(n_obs >= 8).sum()}  m==2 cells={(n_obs == 2).sum()}")
print(f"row-centring check: max |mean of centred curve over observed metals| = "
      f"{np.nanmax(np.abs(_chk)):.2e}")

# --------------------------------------------------------- C1: covariance and PCA
hdr("C1  PC1 variance fraction and radius correlation")

# (a) variogram estimator: V[i,j] = Var over co-measuring cells of (logD_i - logD_j).
#     This is level-free, so raw logD and centred logD give the same V.
V = np.zeros((NM, NM))
Npair = np.zeros((NM, NM), dtype=int)
for i, j in itertools.combinations(range(NM), 2):
    both = obs[:, i] & obs[:, j]
    d = X[both, i] - X[both, j]
    Npair[i, j] = Npair[j, i] = both.sum()
    V[i, j] = V[j, i] = np.var(d, ddof=1)
J = np.eye(NM) - np.ones((NM, NM)) / NM
G = -0.5 * J @ V @ J
print(f"variogram pair counts: min={Npair[Npair > 0].min()} median="
      f"{int(np.median(Npair[Npair > 0]))} max={Npair.max()}  (91 pairs)")

# also confirm V is identical when built from the centred curves
V2 = np.zeros((NM, NM))
for i, j in itertools.combinations(range(NM), 2):
    both = obs[:, i] & obs[:, j]
    V2[i, j] = V2[j, i] = np.var(C[both, i] - C[both, j], ddof=1)
print(f"max |V(raw) - V(centred)| = {np.abs(V - V2).max():.2e}  (level-free, as claimed)")


def var_fracs(S: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    w, vecs = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    w, vecs = w[order], vecs[:, order]
    return w, vecs


def report_pc(S: np.ndarray, label: str) -> np.ndarray:
    w, vecs = var_fracs(S)
    tr = np.trace(S)
    fr_tr = w / tr
    pos = w[w > 0]
    fr_pos = w / pos.sum()
    print(f"{label:38s} PC1={fr_tr[0]:.3f} PC2={fr_tr[1]:.3f} PC3={fr_tr[2]:.3f} "
          f"cum3={fr_tr[:3].sum():.3f}   [/sum(pos): {fr_pos[0]:.3f}]  "
          f"min eig={w.min():+.4f}")
    return vecs[:, 0], vecs[:, 1], w, fr_tr


pc1_u, pc2_u, w_u, fr_u = report_pc(G, "variogram, unshrunk")

# (b) shrinkage: linear toward isotropic on the 13-dim mean-zero subspace
def shrink(S: np.ndarray, a: float) -> np.ndarray:
    iso = (np.trace(S) / (NM - 1)) * J
    return (1 - a) * S + a * iso


alphas = np.round(np.arange(0.0, 1.0001, 0.01), 2)
alpha_star = None
for a in alphas:
    ev = np.linalg.eigvalsh(shrink(G, a))
    ev_sub = np.sort(ev)[::-1][: NM - 1]  # 13 in-subspace eigenvalues
    if ev_sub.min() > 0:
        alpha_star = a
        break
print(f"smallest alpha on a 0.01 grid making all 13 subspace eigenvalues > 0: {alpha_star}")
G_s = shrink(G, 0.22)
pc1_s, pc2_s, w_s, fr_s = report_pc(G_s, "variogram, shrunk alpha=0.22")
if alpha_star is not None and alpha_star != 0.22:
    _ = report_pc(shrink(G, alpha_star), f"variogram, shrunk alpha={alpha_star}")

# (c) complete-14 sample covariance
full = n_obs == 14
Cf = C[full][:, :]
S14 = np.cov(Cf, rowvar=False, ddof=1)
pc1_f, pc2_f, w_f, fr_f = report_pc(S14, f"complete-14 sample cov (n={full.sum()})")

# (d) correlation of PC1 with the physics basis
basis = physics_basis()
rows = []
for name, pc1, tag in [("unshrunk", pc1_u, "variogram_unshrunk"),
                       ("shrunk", pc1_s, "variogram_shrunk_0.22"),
                       ("complete14", pc1_f, "complete14")]:
    # sign convention: make the La end positive-radius aligned check via |r|
    line = {"estimator": tag}
    for bname, bvec in basis.items():
        line[bname] = float(np.corrcoef(pc1, bvec)[0, 1])
    rows.append(line)
pc1_phys = pd.DataFrame(rows)
print("\nPC1 vs physics basis (Pearson r over the 14 metals):")
print(pc1_phys.round(4).to_string(index=False))

# 2-term radius + inv_radius fit to PC1
A2 = np.column_stack([basis["radius"], basis["inv_radius"], np.ones(NM)])
for tag, pc1 in [("unshrunk", pc1_u), ("shrunk", pc1_s), ("complete14", pc1_f)]:
    beta, *_ = np.linalg.lstsq(A2, pc1, rcond=None)
    resid = pc1 - A2 @ beta
    r2 = 1 - resid.var() / pc1.var()
    print(f"PC1 ({tag}) ~ radius + inv_radius : R2 = {r2:.4f}")

pc1_phys.to_csv(OUT / "v2_pc1_physics_correlation.csv", index=False)
pd.DataFrame({
    "metal": M,
    "pc1_unshrunk": pc1_u, "pc2_unshrunk": pc2_u,
    "pc1_shrunk": pc1_s, "pc2_shrunk": pc2_s,
    "pc1_complete14": pc1_f, "pc2_complete14": pc2_f,
    "radius_cn8": [SHANNON_RADIUS_CN8[m] for m in M],
}).to_csv(OUT / "v2_eigenvectors.csv", index=False)
pd.DataFrame({
    "estimator": ["variogram_unshrunk", "variogram_shrunk_0.22", "complete14"],
    "pc1": [fr_u[0], fr_s[0], fr_f[0]],
    "pc2": [fr_u[1], fr_s[1], fr_f[1]],
    "pc3": [fr_u[2], fr_s[2], fr_f[2]],
    "cum3": [fr_u[:3].sum(), fr_s[:3].sum(), fr_f[:3].sum()],
}).to_csv(OUT / "v2_variance_fractions.csv", index=False)

# per-metal sd of the centred curve on complete-14 cells (claimed U shape)
sd14 = Cf.std(axis=0, ddof=1)
print("\nsd of centred curve, complete-14 cells: " +
      " ".join(f"{m}={s:.3f}" for m, s in zip(M, sd14)))

# ------------------------------------------------- C2/C3: arm scoring and oracle
hdr("C2/C3  extractant-macro MAE and the one-scalar amplitude oracle")

PRED = ROOT / "gen13_separation/predictions/B_primary"


def load_arm(name: str) -> pd.DataFrame:
    d = pd.read_parquet(PRED / f"{name}.parquet")
    d["abserr"] = (d["y"] - d["prediction"]).abs()
    return d


def ext_macro(d: pd.DataFrame, col: str = "abserr", per_cell_first: bool = False) -> float:
    """extractant-macro: mean within extractant, then over extractants, then over seeds."""
    if per_cell_first:
        g = d.groupby(["split_seed", "extractant", "cell_id"])[col].mean().reset_index()
        g = g.groupby(["split_seed", "extractant"])[col].mean().reset_index()
    else:
        g = d.groupby(["split_seed", "extractant"])[col].mean().reset_index()
    return float(g.groupby("split_seed")[col].mean().mean())


arm = load_arm("C_DIRECT_ROW")
print(f"C_DIRECT_ROW: {len(arm)} rows, {arm.cell_id.nunique()} cells, "
      f"{arm.extractant.nunique()} extractants, {arm.split_seed.nunique()} seeds")
print(f"[C3] extractant-macro MAE, all cells, 5 seeds = {ext_macro(arm):.4f}   "
      f"(programme reference 0.495)")
print(f"[C3] pooled MAE, all cells, 5 seeds          = {arm.abserr.mean():.4f}")
for nm in ["X_ENS_DIRECT+LOWRANK_K2", "M_SELECTED", "B1_MEAN_CURVE", "B4_HEAVIER_ALWAYS"]:
    a = load_arm(nm)
    print(f"[C3] extractant-macro MAE {nm:26s} = {ext_macro(a):.4f}")

SEED = sorted(arm.split_seed.unique())[0]
print(f"\nusing single split seed {SEED} for the oracle (as the summary did)")


def oracle_table(d1: pd.DataFrame, shapes: dict[str, np.ndarray], min_m: int) -> pd.DataFrame:
    """Per-pair residual after fitting per-cell amplitude scalars along fixed shapes."""
    sub = d1[d1["n_metals"] >= min_m].copy()
    sub["resid"] = sub["y"] - sub["prediction"]
    ai = sub["A"].map(IDX).to_numpy()
    bi = sub["B"].map(IDX).to_numpy()
    r = sub["resid"].to_numpy()
    cells = sub["cell_id"].to_numpy()
    out = {}
    # group row indices by cell
    order = np.argsort(cells, kind="stable")
    cs = cells[order]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    groups = [order[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]
    for label, W in shapes.items():
        Wm = np.atleast_2d(W)  # k x 14
        k = Wm.shape[0]
        e_all = np.empty(len(sub))
        e_lofo = np.empty(len(sub))  # leave-the-scored-pair-out
        for g in groups:
            Dm = (Wm[:, ai[g]] - Wm[:, bi[g]]).T  # n_pairs x k
            rg = r[g]
            beta, *_ = np.linalg.lstsq(Dm, rg, rcond=None)
            e_all[g] = rg - Dm @ beta
            if len(g) > k + 1:
                for t in range(len(g)):
                    keep = np.ones(len(g), dtype=bool)
                    keep[t] = False
                    b2, *_ = np.linalg.lstsq(Dm[keep], rg[keep], rcond=None)
                    e_lofo[t_i := g[t]] = rg[t] - Dm[t] @ b2
            else:
                e_lofo[g] = rg
        out[f"e_{label}"] = np.abs(e_all)
        out[f"elofo_{label}"] = np.abs(e_lofo)
    res = sub[["split_seed", "extractant", "cell_id", "n_metals", "A", "B", "dZ"]].copy()
    res["abserr"] = np.abs(r)
    for kk, vv in out.items():
        res[kk] = vv
    return res


# fixed shapes, all mean-zero over the 14 metals and unit norm
def unitise(v):
    v = np.asarray(v, float)
    v = v - v.mean()
    return v / np.linalg.norm(v)


shapes = {
    "pc1_shrunk": unitise(pc1_s),
    "pc1_complete14": unitise(pc1_f),
    "radius": unitise(basis["radius"]),
    "pc2_shrunk": unitise(pc2_s),
    "pc1pc2": np.vstack([unitise(pc1_s), unitise(pc2_s)]),
}
# null controls: random mean-zero directions
for t in range(5):
    shapes[f"rand{t}"] = unitise(rng.normal(size=NM))

d1 = arm[arm.split_seed == SEED]
tab = oracle_table(d1, shapes, min_m=8)
ncells = tab.cell_id.nunique()
nexts = tab.extractant.nunique()
print(f"\n[C2] held-out cells with m>=8, seed {SEED}: {ncells} cells, {nexts} extractants, "
      f"{len(tab)} pairs")
base = ext_macro(tab, "abserr")
print(f"[C2] baseline extractant-macro MAE (m>=8)        = {base:.4f}   (summary 0.441)")
print(f"[C2] baseline pooled MAE (m>=8)                  = {tab.abserr.mean():.4f}")
rows = [{"variant": "baseline", "ext_macro_mae": base, "pooled_mae": tab.abserr.mean(),
         "frac_removed": 0.0}]
for label in shapes:
    for pre, tag in [("e_", "oracle_all_pairs"), ("elofo_", "leave_scored_pair_out")]:
        v = ext_macro(tab, pre + label)
        rows.append({"variant": f"{label}__{tag}", "ext_macro_mae": v,
                     "pooled_mae": float(tab[pre + label].mean()),
                     "frac_removed": 1 - v / base})
oc = pd.DataFrame(rows)
print(oc.round(4).to_string(index=False))
oc.to_csv(OUT / "v2_amplitude_oracle.csv", index=False)

# dominance control: per-cell-first averaging inside the extractant
print(f"\n[C2] per-cell-first extractant-macro: baseline "
      f"{ext_macro(tab, 'abserr', per_cell_first=True):.4f} -> pc1 oracle "
      f"{ext_macro(tab, 'e_pc1_shrunk', per_cell_first=True):.4f}")
big = tab.groupby("extractant").size().sort_values(ascending=False)
print(f"[C2] pairs per extractant: max={big.iloc[0]} ({big.index[0][:28]}...), "
      f"median={int(big.median())}, min={big.min()}")
cellsz = tab.groupby("cell_id").size()
print(f"[C2] pairs per cell: max={cellsz.max()}, median={int(cellsz.median())}, "
      f"share of all pairs held by the largest cell = {cellsz.max()/len(tab):.3%}")

# is the arm's pair residual exactly additive in a per-cell metal curve?
addit = []
for cid, g in tab.assign(resid=(d1.set_index(d1.index).loc[tab.index, 'y'] -
                                d1.loc[tab.index, 'prediction'])).groupby("cell_id"):
    if len(g) < 3:
        continue
    ms = sorted(set(g.A) | set(g.B))
    if len(ms) < 3:
        continue
    mi = {m: i for i, m in enumerate(ms)}
    D = np.zeros((len(g), len(ms)))
    for t, (_, row) in enumerate(g.iterrows()):
        D[t, mi[row.A]] = 1.0
        D[t, mi[row.B]] = -1.0
    rr = g["resid"].to_numpy()
    beta, *_ = np.linalg.lstsq(D, rr, rcond=None)
    addit.append(1 - ((rr - D @ beta) ** 2).sum() / max((rr ** 2).sum(), 1e-30))
print(f"[C2] additivity of the arm pair-residual in a per-cell metal curve: "
      f"mean R2 = {np.mean(addit):.6f} over {len(addit)} cells "
      f"(min {np.min(addit):.4f}) -> a cell's residual has only m-1 free numbers")

# seed-to-seed spread of the headline oracle number
seed_rows = []
for sd in sorted(arm.split_seed.unique()):
    t = oracle_table(arm[arm.split_seed == sd], {"pc1_shrunk": shapes["pc1_shrunk"]}, min_m=8)
    b = ext_macro(t, "abserr")
    o = ext_macro(t, "e_pc1_shrunk")
    seed_rows.append({"split_seed": sd, "baseline": b, "oracle_pc1": o,
                      "frac_removed": 1 - o / b})
sr = pd.DataFrame(seed_rows)
print("\n[C2] all five seeds (summary used one):")
print(sr.round(4).to_string(index=False))
sr.to_csv(OUT / "v2_amplitude_oracle_by_seed.csv", index=False)

# ------------------------------------------------------- C4: radius law on pairs
hdr("C4  mean |log SF| vs |Shannon CN8 radius difference| over the 91 pairs")
recs = []
for i, j in itertools.combinations(range(NM), 2):
    both = obs[:, i] & obs[:, j]
    if both.sum() < 2:
        continue
    sf = X[both, i] - X[both, j]  # logD(light) - logD(heavy), Z_i < Z_j
    recs.append({"A": M[i], "B": M[j], "n_cells": int(both.sum()),
                 "mean_abs_logSF": float(np.mean(np.abs(sf))),
                 "sd_logSF": float(np.std(sf, ddof=1)),
                 "frac_heavy_pref": float(np.mean(sf < 0)),
                 "dr": abs(SHANNON_RADIUS_CN8[M[i]] - SHANNON_RADIUS_CN8[M[j]]),
                 "dZ": abs(ATOMIC_NUMBER[M[j]] - ATOMIC_NUMBER[M[i]])})
pw = pd.DataFrame(recs)
A = np.column_stack([pw.dr, np.ones(len(pw))])
beta, *_ = np.linalg.lstsq(A, pw.mean_abs_logSF, rcond=None)
r2 = 1 - ((pw.mean_abs_logSF - A @ beta) ** 2).sum() / \
    ((pw.mean_abs_logSF - pw.mean_abs_logSF.mean()) ** 2).sum()
print(f"pairs={len(pw)}  n_cells min={pw.n_cells.min()} median={int(pw.n_cells.median())} "
      f"max={pw.n_cells.max()}")
print(f"OLS mean|logSF| = {beta[0]:.3f} * |dr| + {beta[1]:.3f}   R2 = {r2:.4f}  "
      f"(summary 9.60 / 0.024 / 0.961)")
print(f"fraction heavy-preferred: mean {pw.frac_heavy_pref.mean():.3f}, "
      f"range {pw.frac_heavy_pref.min():.3f}-{pw.frac_heavy_pref.max():.3f}")
pw.to_csv(OUT / "v2_pair_stats.csv", index=False)
print("\nwrote:", ", ".join(sorted(p.name for p in OUT.glob("v2_*.csv"))))
