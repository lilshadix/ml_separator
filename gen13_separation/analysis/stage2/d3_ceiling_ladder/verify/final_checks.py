"""Final verification: the headline gap on a subset-free set, its fragility to which
extractants enter the macro average, and the 'arms buy nothing over the corpus mean' claim."""
import os
import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
OUT = os.path.join(ROOT, "gen13_separation/analysis/stage2/d3_ceiling_ladder/verify")
PRED = os.path.join(ROOT, "gen13_separation/predictions/B_primary")
LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Zn = dict(zip(LN, [57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]))
IDX = {m: i for i, m in enumerate(LN)}
rng = np.random.default_rng(0)

coh = pd.read_parquet(os.path.join(ROOT, "gen13_separation/manifests/cohort_exact.parquet"))
ext = coh.extractant.to_numpy(); chemo = coh.chemotype.to_numpy()
pub = coh.publication_id.to_numpy(); cells = coh.cell_id.to_numpy()
X = coh[[f"logD__{m}" for m in LN]].to_numpy(dtype=float)
OBS = ~np.isnan(X); NC = len(cells)
C = X - np.nanmean(np.where(OBS, X, np.nan), axis=1)[:, None]

rows = []
for i in range(NC):
    ms = [m for m in LN if OBS[i, IDX[m]]]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            rows.append((cells[i], i, ms[a], ms[b], Zn[ms[b]] - Zn[ms[a]],
                         X[i, IDX[ms[a]]] - X[i, IDX[ms[b]]]))
P = pd.DataFrame(rows, columns=["cell_id", "ri", "A", "B", "dZ", "y"])
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

def build(groups):
    D = np.full((NC, 14), np.nan); g = pd.Series(groups)
    for k, ix in g.groupby(g).groups.items():
        ix = np.asarray(list(ix))
        if len(ix) < 2: continue
        for i in ix: D[i] = two_way(ix[ix != i])
    return D

D3 = build(ext); D4 = build([f"{a}|{b}" for a, b in zip(ext, pub)]); D2 = build(chemo)
Cz = np.where(OBS, C, 0.0)
cts = pd.Series(chemo).value_counts()
w = np.array([1.0 / cts[g] for g in chemo]); w = w * NC / w.sum()
tot = (Cz * w[:, None]).sum(0); cnt = (OBS * w[:, None]).sum(0)
D1 = np.full((NC, 14), np.nan)
for i in range(NC):
    n = cnt - OBS[i] * w[i]
    D1[i] = np.where(n > 1e-12, (tot - Cz[i] * w[i]) / np.maximum(n, 1e-12), np.nan)

# ------- the definition-free like-for-like set: every pair where L3 exists -------
p3 = D3[RI, IA] - D3[RI, IB]
m = ~np.isnan(p3)
S = P[m].copy()
key = set(zip(S.cell_id, S.A, S.B))
print(f"=== like-for-like set = all pairs where L3 is defined: "
      f"{S.cell_id.nunique()} cells / {S.extractant.nunique()} ext / {len(S)} pairs ===")

per_ext = {}
res = []
for lab, D in [("L1 corpus mean (LOO, chemo-bal)", D1), ("L2 chemotype (LOO)", D2),
               ("L3 extractant (LOO)", D3), ("L4 extractant+publication (LOO)", D4)]:
    d = S.assign(p=(D[RI, IA] - D[RI, IB])[m]).dropna(subset=["p"]).copy()
    d["ae"] = (d.y - d.p).abs()
    pe = d.groupby("extractant", observed=True)["ae"].mean()
    per_ext[lab] = pe
    res.append(dict(predictor=lab, ext_macro=round(float(pe.mean()), 4),
                    pooled=round(float(d.ae.mean()), 4), n_ext=len(pe), n_pairs=len(d)))
    print(f"  {lab:34s} ext-macro {pe.mean():.4f}  pooled {d.ae.mean():.4f}  n_ext={len(pe)}  n={len(d)}")

for a in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE", "B4_HEAVIER_ALWAYS"]:
    pr = pd.read_parquet(os.path.join(PRED, a + ".parquet"))
    k = pd.Series(list(zip(pr.cell_id, pr.A, pr.B)))
    sel = pr[k.isin(key).to_numpy()].copy()
    sel["ae"] = (sel.y - sel.prediction).abs()
    pe = sel.groupby(["split_seed", "extractant"], observed=True)["ae"].mean().groupby("extractant").mean()
    per_ext["ARM " + a] = pe
    ps = sel.groupby("split_seed").apply(
        lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
    res.append(dict(predictor="ARM " + a, ext_macro=round(float(ps.mean()), 4),
                    pooled=round(float(sel.ae.mean()), 4), n_ext=len(pe), n_pairs=len(sel) // 5))
    print(f"  ARM {a:30s} ext-macro {ps.mean():.4f}  pooled {sel.ae.mean():.4f}  seed sd {ps.std(ddof=1):.4f}")
pd.DataFrame(res).to_csv(os.path.join(OUT, "v_final_likeforlike.csv"), index=False)

best = per_ext["ARM X_ENS_DIRECT+LOWRANK_K2"]; l3 = per_ext["L3 extractant (LOO)"]
gap = (best - l3).dropna()
print(f"\n  headline gap (arm - L3) on {len(gap)} extractants: mean {gap.mean():.4f}, "
      f"sd {gap.std(ddof=1):.4f}, se {gap.std(ddof=1)/np.sqrt(len(gap)):.4f}")
bs = np.array([gap.sample(len(gap), replace=True, random_state=int(s)).mean() for s in range(2000)])
print(f"  bootstrap over extractants: 95% CI [{np.quantile(bs,.025):.3f}, {np.quantile(bs,.975):.3f}]")

# fragility: how far can the gap move by dropping 4 of 24 extractants?
combined = pd.DataFrame({"arm": best, "l3": l3}).dropna()
drops = []
for _ in range(4000):
    keep = combined.sample(len(combined) - 4, random_state=int(rng.integers(1e9)))
    drops.append((keep.arm.mean(), keep.l3.mean(), keep.arm.mean() - keep.l3.mean()))
dr = pd.DataFrame(drops, columns=["arm", "l3", "gap"])
print(f"  dropping any 4 of {len(combined)} extractants: L3 ranges "
      f"[{dr.l3.quantile(.01):.3f}, {dr.l3.quantile(.99):.3f}], arm "
      f"[{dr.arm.quantile(.01):.3f}, {dr.arm.quantile(.99):.3f}], gap "
      f"[{dr.gap.quantile(.01):.3f}, {dr.gap.quantile(.99):.3f}]")
combined.assign(gap=combined.arm - combined.l3).sort_values("gap").to_csv(
    os.path.join(OUT, "v_per_extractant_gap.csv"))

# 'arms buy nothing over the corpus mean' on this set
print(f"\n  arms vs LOO corpus mean on the same set: arm {best.mean():.4f} vs "
      f"L1 {per_ext['L1 corpus mean (LOO, chemo-bal)'].mean():.4f} "
      f"(D3 claims 0.457 vs 0.456, i.e. no gain)")

# ---------------- variance decomposition cross-check ----------------
print("\n=== variance of pairwise log SF, one-way ANOVA within each pair type ===")
sb, sw, dfb, dfw, n0n, n0d = 0.0, 0.0, 0, 0, 0.0, 0.0
for (A, B), g in P.groupby(["A", "B"]):
    grp = g.groupby("extractant", observed=True)["y"]
    ns = grp.size(); k = len(ns); n = len(g)
    if k < 2: continue
    mus = grp.mean()
    sb += float(((mus - g.y.mean()) ** 2 * ns).sum()); dfb += k - 1
    sw += float(((g.y - g.extractant.map(mus)) ** 2).sum()); dfw += n - k
    n0n += n - (ns ** 2).sum() / n; n0d += k - 1
msb, msw = sb / dfb, sw / dfw
n0 = n0n / n0d
vb = max((msb - msw) / n0, 0.0)
print(f"  between-extractant {vb:.3f} (sd {np.sqrt(vb):.3f}, {dfb} df), "
      f"within {msw:.3f} (sd {np.sqrt(msw):.3f}, {dfw} df), n0 {n0:.3f}, "
      f"between share {100*vb/(vb+msw):.1f}%  (D3: 0.393 / 0.186, 5382 / 8677 df, n0 2.385, 67.9%)")
print(f"  total var of y {P.y.var(ddof=1):.3f} (D3: 0.544)")
