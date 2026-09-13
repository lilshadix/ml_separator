"""Part 3: match the conditional-gain subset (m>=4, revealed pair excluded from scoring),
and test whether the PC1 result survives dropping the dominant chemotype."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, physics_basis  # noqa: E402

OUT = ROOT / "gen13_separation/analysis/stage2/d2_metal_dependency/verify"
M = list(LANTHANIDES)
NM = len(M)
IDX = {m: i for i, m in enumerate(M)}
J = np.eye(NM) - np.ones((NM, NM)) / NM

coh = pd.read_parquet(ROOT / "gen13_separation/manifests/cohort_exact.parquet")
X = coh[[f"logD__{m}" for m in M]].to_numpy(dtype=float)
obs = ~np.isnan(X)
n_obs = obs.sum(1)


def vario_pc(mask):
    V = np.zeros((NM, NM))
    nmin = 10 ** 9
    for i, j in itertools.combinations(range(NM), 2):
        both = obs[:, i] & obs[:, j] & mask
        if both.sum() < 4:
            return None, None, None, 0
        nmin = min(nmin, both.sum())
        V[i, j] = V[j, i] = np.var(X[both, i] - X[both, j], ddof=1)
    G = -0.5 * J @ V @ J
    Gs = 0.78 * G + 0.22 * (np.trace(G) / 13) * J
    w, v = np.linalg.eigh(Gs)
    o = np.argsort(w)[::-1]
    f = w[o] / w[o][w[o] > 0].sum()
    wr, vr = np.linalg.eigh(G)
    orr = np.argsort(wr)[::-1]
    fr = wr[orr] / wr[orr][wr[orr] > 0].sum()
    return f, v[:, o[0]], fr, nmin


print("=" * 78)
print("E1  does PC1 survive dropping the dominant chemotype?")
print("=" * 78)
ct = coh["chemotype"].to_numpy()
top_ct = pd.Series(ct).value_counts()
print(f"  top chemotypes: {top_ct.head(3).to_dict()}")
radius = physics_basis()["radius"]
rows = []
for label, mask in [
    ("all 521 cells", np.ones(len(coh), bool)),
    (f"only {top_ct.index[0][:20]} ({top_ct.iloc[0]})", ct == top_ct.index[0]),
    (f"excluding {top_ct.index[0][:20]}", ct != top_ct.index[0]),
]:
    f, pc1, fr, nmin = vario_pc(mask)
    if f is None:
        print(f"  {label:44s} -> not estimable (a metal pair has <4 cells)")
        continue
    r = float(np.corrcoef(pc1, radius)[0, 1])
    print(f"  {label:44s} n={mask.sum():3d} min pair n={nmin:3d}  "
          f"PC1_shrunk={f[0]:.3f} PC1_raw={fr[0]:.3f}  |r with radius|={abs(r):.4f}")
    rows.append({"subset": label, "n_cells": int(mask.sum()), "min_pair_n": int(nmin),
                 "pc1_shrunk": f[0], "pc1_unshrunk": fr[0], "abs_r_radius": abs(r)})
pd.DataFrame(rows).to_csv(OUT / "v2_pc1_by_chemotype.csv", index=False)

print()
print("=" * 78)
print("E2  reveal-one-SF on the SAME subset the summary used (m>=4, revealed pair excluded)")
print("=" * 78)
PRED = ROOT / "gen13_separation/predictions/B_primary"


def unitise(v):
    v = np.asarray(v, float) - np.mean(v)
    return v / np.linalg.norm(v)


# PC1 shape from the shrunk variogram over all cells
f_all, pc1_all, _, _ = vario_pc(np.ones(len(coh), bool))
shape = unitise(pc1_all)
res = []
for arm_name in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2"]:
    a = pd.read_parquet(PRED / f"{arm_name}.parquet")
    SEED = sorted(a.split_seed.unique())[0]
    a = a[(a.split_seed == SEED) & (a.n_metals >= 4)].copy()
    a["resid"] = a["y"] - a["prediction"]
    a["x"] = shape[a.A.map(IDX).to_numpy()] - shape[a.B.map(IDX).to_numpy()]
    a = a.sort_values(["cell_id", "dZ"], kind="stable")
    rev_idx = a.groupby("cell_id").tail(1).index
    a["revealed"] = a.index.isin(rev_idx)
    rev = a.loc[rev_idx, ["cell_id", "resid", "x"]].rename(
        columns={"resid": "r_rev", "x": "x_rev"})
    a = a.merge(rev, on="cell_id", how="left")
    cell = (a.groupby(["cell_id", "chemotype"], as_index=False)
            .apply(lambda g: pd.Series({
                "c_full": float((g.x * g.resid).sum() / (g.x ** 2).sum()),
                "c_naive": float(g.r_rev.iloc[0] / g.x_rev.iloc[0])}), include_groups=False))
    kap = {c: float((o.c_naive * o.c_full).sum() / (o.c_naive ** 2).sum())
           for c in cell.chemotype.unique() for o in [cell[cell.chemotype != c]]}
    cell["c_hat"] = cell.chemotype.map(kap) * cell.c_naive
    a = a.merge(cell[["cell_id", "c_hat"]], on="cell_id", how="left")
    a["e"] = (a.resid - a.c_hat * a.x).abs()
    a["abserr"] = a.resid.abs()
    ev = a[~a.revealed]
    b = float(ev.groupby("extractant").abserr.mean().mean())
    c = float(ev.groupby("extractant").e.mean().mean())
    print(f"  {arm_name}: {len(ev)} scored pairs (revealed pair excluded), "
          f"{ev.extractant.nunique()} extractants, {ev.cell_id.nunique()} cells")
    print(f"    extractant-macro MAE {b:.4f} -> {c:.4f}   gain {b - c:+.4f}   "
          f"(summary: 0.433 -> 0.224, gain 0.210)")
    res.append({"arm": arm_name, "n_pairs": len(ev), "n_ext": ev.extractant.nunique(),
                "base": b, "corrected": c, "gain": b - c})
pd.DataFrame(res).to_csv(OUT / "v2_revealed_sf_m4.csv", index=False)
