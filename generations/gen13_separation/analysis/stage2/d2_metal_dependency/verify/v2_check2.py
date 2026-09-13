"""Part 2 of the independent d2 check.

  D1  variants of the complete-14 covariance, to locate the summary's PC1 = 0.857
  D2  variants of the "two scalars" oracle, to locate the summary's 0.129
  D3  PC2 correlations with the physics basis
  D4  the non-oracle claim: reveal ONE measured log SF of a held-out cell and correct
      the remaining pairs with a leave-chemotype-out rule; is it genuinely out of sample,
      and is the revealed pair excluded from scoring?
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, physics_basis  # noqa: E402

OUT = ROOT / "generations/gen13_separation/analysis/stage2/d2_metal_dependency/verify"
M = list(LANTHANIDES)
NM = len(M)
IDX = {m: i for i, m in enumerate(M)}
J = np.eye(NM) - np.ones((NM, NM)) / NM
rng = np.random.default_rng(7)

coh = pd.read_parquet(ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet")
X = coh[[f"logD__{m}" for m in M]].to_numpy(dtype=float)
obs = ~np.isnan(X)
n_obs = obs.sum(1)
C = X - np.nanmean(X, axis=1)[:, None]
full = n_obs == 14
Cf = C[full]


def fracs(S):
    w = np.linalg.eigvalsh(S)[::-1]
    return w / w[w > 0].sum(), w


def top(S):
    w, v = np.linalg.eigh(S)
    o = np.argsort(w)[::-1]
    return v[:, o[0]], v[:, o[1]], (w[o] / w[o][w[o] > 0].sum())


print("=" * 78)
print("D1  complete-14 covariance variants (summary claims PC1=0.857 PC2=0.065 PC3=0.018)")
print("=" * 78)
variants = {
    "cov of row-centred, ddof=1": np.cov(Cf, rowvar=False, ddof=1),
    "cov of row-centred, ddof=0": np.cov(Cf, rowvar=False, ddof=0),
    "second moment (no column mean)": Cf.T @ Cf / (len(Cf) - 1),
    "double-centred second moment": J @ (Cf.T @ Cf / (len(Cf) - 1)) @ J,
    "variogram of the 78 complete cells": None,
    "corr matrix of row-centred": np.corrcoef(Cf, rowvar=False),
}
V78 = np.zeros((NM, NM))
for i, j in itertools.combinations(range(NM), 2):
    V78[i, j] = V78[j, i] = np.var(X[full][:, i] - X[full][:, j], ddof=1)
variants["variogram of the 78 complete cells"] = -0.5 * J @ V78 @ J
for name, S in variants.items():
    f, w = fracs(S)
    print(f"  {name:36s} PC1={f[0]:.3f} PC2={f[1]:.3f} PC3={f[2]:.3f} cum3={f[:3].sum():.3f}")
# a shrunk complete-14, to see whether 0.857 is a lightly-shrunk version
S14 = np.cov(Cf, rowvar=False, ddof=1)
for a in (0.02, 0.05, 0.08, 0.10):
    Sa = (1 - a) * S14 + a * (np.trace(S14) / 13) * J
    f, _ = fracs(Sa)
    print(f"  complete-14 shrunk alpha={a:<5}          PC1={f[0]:.3f} PC2={f[1]:.3f} "
          f"PC3={f[2]:.3f} cum3={f[:3].sum():.3f}")

print()
print("=" * 78)
print("D3  PC2 vs physics basis (summary: radius_sq +0.656, gd_break +0.037, tetrad_e3 -0.106)")
print("=" * 78)
Vall = np.zeros((NM, NM))
for i, j in itertools.combinations(range(NM), 2):
    both = obs[:, i] & obs[:, j]
    Vall[i, j] = Vall[j, i] = np.var(X[both, i] - X[both, j], ddof=1)
G = -0.5 * J @ Vall @ J
Gs = 0.78 * G + 0.22 * (np.trace(G) / 13) * J
pc1_s, pc2_s, f_s = top(Gs)
basis = physics_basis()
row = {b: float(np.corrcoef(pc2_s, v)[0, 1]) for b, v in basis.items()}
print("  PC2 (shrunk variogram): " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))
print("  PC2 loadings: " + " ".join(f"{m}{v:+.3f}" for m, v in zip(M, pc2_s)))
# best 2-term physics fit to PC2
best = None
for a, b in itertools.combinations(basis, 2):
    A = np.column_stack([basis[a], basis[b], np.ones(NM)])
    beta, *_ = np.linalg.lstsq(A, pc2_s, rcond=None)
    r2 = 1 - ((pc2_s - A @ beta) ** 2).sum() / ((pc2_s - pc2_s.mean()) ** 2).sum()
    if best is None or r2 > best[0]:
        best = (r2, a, b)
print(f"  best 2-term physics fit to PC2: R2={best[0]:.3f} ({best[1]} + {best[2]})  "
      f"(summary 0.580)")

# ------------------------------------------------------------------ arms / oracle
PRED = ROOT / "generations/gen13_separation/predictions/B_primary"


def unitise(v):
    v = np.asarray(v, float) - np.mean(v)
    return v / np.linalg.norm(v)


def ext_macro(d, col):
    g = d.groupby(["extractant"])[col].mean()
    return float(g.mean())


arm = pd.read_parquet(PRED / "C_DIRECT_ROW.parquet")
SEED = sorted(arm.split_seed.unique())[0]
d1 = arm[arm.split_seed == SEED].copy()
d1["resid"] = d1["y"] - d1["prediction"]
d1["abserr"] = d1["resid"].abs()

print()
print("=" * 78)
print("D2  'two scalars' variants on m>=8 cells (summary claims 0.129)")
print("=" * 78)
sub = d1[d1.n_metals >= 8].copy()
ai = sub.A.map(IDX).to_numpy()
bi = sub.B.map(IDX).to_numpy()
r = sub.resid.to_numpy()
groups = [g.to_numpy() for _, g in sub.reset_index(drop=True).groupby("cell_id").groups.items()]


def fit_shapes(W):
    W = np.atleast_2d(W)
    e = np.empty(len(sub))
    for g in groups:
        D = (W[:, ai[g]] - W[:, bi[g]]).T
        beta, *_ = np.linalg.lstsq(D, r[g], rcond=None)
        e[g] = r[g] - D @ beta
    return e


pc1_f, pc2_f, _ = top(np.cov(Cf, rowvar=False, ddof=1))
cands = {
    "PC1+PC2 (shrunk variogram)": np.vstack([unitise(pc1_s), unitise(pc2_s)]),
    "PC1+PC2 (complete-14)": np.vstack([unitise(pc1_f), unitise(pc2_f)]),
    "radius+radius_sq": np.vstack([unitise(basis["radius"]), unitise(basis["radius_sq"])]),
    "PC1+PC2+PC3 (shrunk)": None,
}
w_, v_ = np.linalg.eigh(Gs)
o_ = np.argsort(w_)[::-1]
cands["PC1+PC2+PC3 (shrunk)"] = np.vstack([unitise(v_[:, o_[k]]) for k in range(3)])
base8 = ext_macro(sub.assign(e=sub.abserr), "e")
print(f"  baseline (m>=8, seed {SEED}) extractant-macro MAE = {base8:.4f}")
for name, W in cands.items():
    e = np.abs(fit_shapes(W))
    v = ext_macro(sub.assign(e=e), "e")
    print(f"  {name:30s} -> {v:.4f}   ({1 - v / base8:.1%} removed)")

# ------------------------------------------------- D4 the non-oracle revealed-SF claim
print()
print("=" * 78)
print("D4  reveal ONE measured log SF per held-out cell (summary: 0.433 -> 0.224)")
print("=" * 78)
shape = unitise(pc1_s)
res_rows = []
for arm_name in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2"]:
    a = pd.read_parquet(PRED / f"{arm_name}.parquet")
    a = a[a.split_seed == SEED].copy()
    a["resid"] = a["y"] - a["prediction"]
    a["abserr"] = a["resid"].abs()
    a = a[a.n_metals >= 3].copy()
    a["x"] = shape[a.A.map(IDX).to_numpy()] - shape[a.B.map(IDX).to_numpy()]
    # choose the revealed pair per cell: the widest dZ (ties -> first)
    a = a.sort_values(["cell_id", "dZ"], kind="stable")
    rev_idx = a.groupby("cell_id").tail(1).index
    a["revealed"] = a.index.isin(rev_idx)
    rev = a.loc[rev_idx, ["cell_id", "chemotype", "resid", "x"]].rename(
        columns={"resid": "r_rev", "x": "x_rev"})
    a = a.merge(rev[["cell_id", "r_rev", "x_rev"]], on="cell_id", how="left")
    # amplitude estimate from the revealed pair: c = kappa * r_rev / x_rev, with the
    # shrinkage kappa fitted on cells of OTHER chemotypes (leave-chemotype-out).
    #   per-cell LS amplitude from all pairs = c_full ; regress c_full on c_naive
    cell = a.groupby(["cell_id", "chemotype"]).apply(
        lambda g: pd.Series({
            "c_full": float((g.x * g.resid).sum() / (g.x ** 2).sum()),
            "c_naive": float(g.r_rev.iloc[0] / g.x_rev.iloc[0]) if abs(g.x_rev.iloc[0]) > 1e-9
            else 0.0}), include_groups=False).reset_index()
    kap = {}
    for ct in cell.chemotype.unique():
        o = cell[cell.chemotype != ct]
        kap[ct] = float((o.c_naive * o.c_full).sum() / (o.c_naive ** 2).sum())
    cell["kappa"] = cell.chemotype.map(kap)
    cell["c_hat"] = cell.kappa * cell.c_naive
    a = a.merge(cell[["cell_id", "c_hat", "kappa"]], on="cell_id", how="left")
    a["e"] = (a.resid - a.c_hat * a.x).abs()
    incl = ext_macro(a, "e")
    excl = ext_macro(a[~a.revealed], "e")
    base_incl = ext_macro(a, "abserr")
    base_excl = ext_macro(a[~a.revealed], "abserr")
    print(f"  {arm_name}: {len(a)} pairs, {a.extractant.nunique()} extractants, "
          f"{a.cell_id.nunique()} cells (m>=3), mean kappa={cell.kappa.mean():.3f}")
    print(f"    scoring INCLUDING the revealed pair : {base_incl:.4f} -> {incl:.4f}")
    print(f"    scoring EXCLUDING the revealed pair : {base_excl:.4f} -> {excl:.4f}")
    res_rows.append({"arm": arm_name, "n_pairs": len(a), "n_ext": a.extractant.nunique(),
                     "base_incl": base_incl, "corrected_incl": incl,
                     "base_excl": base_excl, "corrected_excl": excl})
pd.DataFrame(res_rows).to_csv(OUT / "v2_revealed_sf.csv", index=False)
print("\nwrote v2_revealed_sf.csv")
