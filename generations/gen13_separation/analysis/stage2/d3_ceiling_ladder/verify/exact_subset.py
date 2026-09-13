"""Decisive test: rebuild the like-for-like subset with the same-acid-identity rule and
check the two headline numbers (L3 = 0.207, best arm = 0.457 on 299 cells / 20 ext / 7228 pairs).
Also test how outcome-correlated that subset restriction is."""
import os
import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
OUT = os.path.join(ROOT, "generations/gen13_separation/analysis/stage2/d3_ceiling_ladder/verify")
PRED = os.path.join(ROOT, "generations/gen13_separation/predictions/B_primary")
LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Zn = dict(zip(LN, [57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]))
IDX = {m: i for i, m in enumerate(LN)}

coh = pd.read_parquet(os.path.join(ROOT, "generations/gen13_separation/manifests/cohort_exact.parquet"))
ext = coh.extractant.to_numpy(); chemo = coh.chemotype.to_numpy()
pub = coh.publication_id.to_numpy(); cells = coh.cell_id.to_numpy()
acid_cols = [c for c in coh.columns if c.startswith("cond__acid__")]
acid_id = coh[acid_cols].astype(int).astype(str).agg("".join, axis=1).to_numpy()
conc = coh["cond__acid_concentration_M"].to_numpy(dtype=float)
X = coh[[f"logD__{m}" for m in LN]].to_numpy(dtype=float)
OBS = ~np.isnan(X); NC = len(cells)
C = X - np.nanmean(np.where(OBS, X, np.nan), axis=1)[:, None]

rows = []
for i in range(NC):
    ms = [m for m in LN if OBS[i, IDX[m]]]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            rows.append((cells[i], i, ms[a], ms[b], X[i, IDX[ms[a]]] - X[i, IDX[ms[b]]]))
P = pd.DataFrame(rows, columns=["cell_id", "ri", "A", "B", "y"])
P["extractant"] = ext[P.ri.to_numpy()]
IA = P.A.map(IDX).to_numpy(); IB = P.B.map(IDX).to_numpy(); RI = P.ri.to_numpy()

def two_way(idx):
    Xs, Os = X[idx], OBS[idx]; mu = np.zeros(14)
    for _ in range(300):
        b = np.where(Os, Xs - mu[None, :], 0.0).sum(1) / np.maximum(Os.sum(1), 1)
        r = np.where(Os, Xs - b[:, None], 0.0)
        mn = np.where(Os.sum(0) > 0, r.sum(0) / np.maximum(Os.sum(0), 1), np.nan)
        mn = mn - np.nanmean(mn)
        if np.nanmax(np.abs(np.nan_to_num(mn) - mu)) < 1e-11:
            mu = mn; break
        mu = np.nan_to_num(mn)
    return np.where(Os.sum(0) > 0, mu, np.nan)

D3 = np.full((NC, 14), np.nan); D4 = np.full((NC, 14), np.nan); D5 = np.full((NC, 14), np.nan)
D5b = np.full((NC, 14), np.nan)
s = pd.Series(ext)
groups = {k: np.asarray(list(v)) for k, v in s.groupby(s).groups.items()}
for i in range(NC):
    er = groups[ext[i]]; er = er[er != i]
    if not len(er): continue
    D3[i] = two_way(er)
    pr = er[pub[er] == pub[i]]
    if len(pr): D4[i] = two_way(pr)
    cand = er[(acid_id[er] == acid_id[i]) & np.isfinite(conc[er])]
    if len(cand) and np.isfinite(conc[i]):
        d = np.abs(conc[cand] - conc[i]); don = cand[d == d.min()]
        D5[i] = two_way(don)
        nd = don[(pub[don] == pub[i]) & (conc[don] == conc[i])]
        if len(nd): D5b[i] = two_way(nd)

def pv(D): return D[RI, IA] - D[RI, IB]
mask = np.isfinite(pv(D3)) & np.isfinite(pv(D4)) & np.isfinite(pv(D5))
S = P[mask].copy()
print(f"=== like-for-like subset (same-acid-identity L5 rule): "
      f"{S.cell_id.nunique()} cells / {S.extractant.nunique()} ext / {len(S)} pairs "
      f"[D3 claim 299 / 20 / 7228] ===")

def em(d): return float(d.groupby("extractant", observed=True)["ae"].mean().mean())
key = set(zip(S.cell_id, S.A, S.B))
out = []
for lab, D in [("L3 extractant", D3), ("L4 +publication", D4), ("L5 +nearest acid", D5)]:
    d = S.assign(p=pv(D)[mask]).dropna(subset=["p"]).copy(); d["ae"] = (d.y - d.p).abs()
    out.append(dict(predictor=lab, ext_macro=round(em(d), 4), pooled=round(float(d.ae.mean()), 4),
                    n_ext=d.extractant.nunique(), n_cells=d.cell_id.nunique(), n_pairs=len(d)))
    print(f"  {lab:20s} ext-macro {em(d):.4f}  pooled {d.ae.mean():.4f}")
arm_per_ext = {}
for a in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "B1_MEAN_CURVE"]:
    pr_ = pd.read_parquet(os.path.join(PRED, a + ".parquet"))
    k = pd.Series(list(zip(pr_.cell_id, pr_.A, pr_.B)))
    sel = pr_[k.isin(key).to_numpy()].copy(); sel["ae"] = (sel.y - sel.prediction).abs()
    ps = sel.groupby("split_seed").apply(
        lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
    arm_per_ext[a] = sel.groupby(["split_seed", "extractant"], observed=True)["ae"].mean() \
        .groupby("extractant").mean()
    out.append(dict(predictor="ARM " + a, ext_macro=round(float(ps.mean()), 4),
                    pooled=round(float(sel.ae.mean()), 4), n_ext=sel.extractant.nunique(),
                    n_cells=sel.cell_id.nunique(), n_pairs=len(sel) // 5))
    print(f"  ARM {a:24s} ext-macro {ps.mean():.4f}  pooled {sel.ae.mean():.4f}  "
          f"seed sd {ps.std(ddof=1):.4f}")
pd.DataFrame(out).to_csv(os.path.join(OUT, "v_exact_subset.csv"), index=False)

# ---- how outcome-correlated is the subset restriction? ----
print("\n=== effect of the subset restriction (L3-defined set -> like-for-like subset) ===")
m3 = np.isfinite(pv(D3))
F = P[m3].assign(p=pv(D3)[m3]); F["ae"] = (F.y - F.p).abs()
Sx = S.assign(p=pv(D3)[mask]); Sx["ae"] = (Sx.y - Sx.p).abs()
print(f"  L3: full L3-defined set {em(F):.4f} ({F.extractant.nunique()} ext / {len(F)} pairs)"
      f"  ->  subset {em(Sx):.4f} ({Sx.extractant.nunique()} ext / {len(Sx)} pairs)")
keyF = set(zip(F.cell_id, F.A, F.B))
pr_ = pd.read_parquet(os.path.join(PRED, "X_ENS_DIRECT+LOWRANK_K2.parquet"))
k = pd.Series(list(zip(pr_.cell_id, pr_.A, pr_.B)))
selF = pr_[k.isin(keyF).to_numpy()].copy(); selF["ae"] = (selF.y - selF.prediction).abs()
psF = selF.groupby("split_seed").apply(
    lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
print(f"  arm: full L3-defined set {psF.mean():.4f}  ->  subset {arm_per_ext['X_ENS_DIRECT+LOWRANK_K2'].mean():.4f}")
print(f"  gap: {psF.mean()-em(F):.4f}  ->  {arm_per_ext['X_ENS_DIRECT+LOWRANK_K2'].mean()-em(Sx):.4f}")

# which extractants are dropped, and were they good or bad for L3?
inn = set(Sx.extractant.unique()); dropped = sorted(set(F.extractant.unique()) - inn)
peF = F.groupby("extractant", observed=True)["ae"].mean()
print(f"  {len(dropped)} extractants dropped; their L3 MAE on the full set: "
      f"{[round(float(peF[e]),3) for e in dropped]}  (kept-extractant mean {peF[list(inn)].mean():.3f})")

# bootstrap the subset gap over its extractants
gp = (arm_per_ext["X_ENS_DIRECT+LOWRANK_K2"] - Sx.groupby("extractant", observed=True)["ae"].mean()).dropna()
bs = np.array([gp.sample(len(gp), replace=True, random_state=int(i)).mean() for i in range(2000)])
print(f"  subset gap {gp.mean():.4f} over {len(gp)} extractants, "
      f"bootstrap 95% CI [{np.quantile(bs,.025):.3f}, {np.quantile(bs,.975):.3f}]")
pd.DataFrame({"arm": arm_per_ext["X_ENS_DIRECT+LOWRANK_K2"],
              "L3": Sx.groupby("extractant", observed=True)["ae"].mean()}).to_csv(
    os.path.join(OUT, "v_exact_subset_per_extractant.csv"))

# L5b near-duplicate
m5b = np.isfinite(pv(D5b))
d = P[m5b].assign(p=pv(D5b)[m5b]); d["ae"] = (d.y - d.p).abs()
print(f"\n  L5b near-duplicate donor: ext-macro {em(d):.4f} over {d.extractant.nunique()} ext / "
      f"{d.cell_id.nunique()} cells / {len(d)} pairs  [D3 claim 0.177 / 16 / 200 / 4455]")
