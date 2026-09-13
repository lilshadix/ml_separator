"""Pin down (a) the L1 corpus-mean estimator and (b) the like-for-like subset,
then re-score the headline comparison under the estimator that matches D3's rungs."""
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
ext = coh["extractant"].to_numpy(); chemo = coh["chemotype"].to_numpy()
pub = coh["publication_id"].to_numpy(); cells = coh["cell_id"].to_numpy()
acid = coh["cond__acid_concentration_M"].to_numpy(dtype=float)
X = coh[[f"logD__{m}" for m in LN]].to_numpy(dtype=float)
OBS = ~np.isnan(X); NC = len(cells)
C = X - np.nanmean(np.where(OBS, X, np.nan), axis=1)[:, None]
nmet = OBS.sum(1)

rows = []
for i in range(NC):
    ms = [m for m in LN if OBS[i, IDX[m]]]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            rows.append((cells[i], i, ms[a], ms[b], X[i, IDX[ms[a]]] - X[i, IDX[ms[b]]]))
P = pd.DataFrame(rows, columns=["cell_id", "ri", "A", "B", "y"])
P["extractant"] = ext[P.ri.to_numpy()]
IA = P.A.map(IDX).to_numpy(); IB = P.B.map(IDX).to_numpy(); RI = P.ri.to_numpy()

def em(d):
    return float(d.groupby("extractant", observed=True)["ae"].mean().mean())

def sc(pred, sub=None, label=""):
    d = P.assign(p=pred[RI, IA] - pred[RI, IB])
    if sub is not None:
        d = d[sub]
    d = d.dropna(subset=["p"]).copy()
    d["ae"] = (d.y - d.p).abs()
    return dict(predictor=label, ext_macro=round(em(d), 4), pooled=round(float(d.ae.mean()), 4),
                n_ext=d.extractant.nunique(), n_cells=d.cell_id.nunique(), n_pairs=len(d))

# ---------- L1 variants ----------
print("=== L1 corpus mean curve: estimator variants, leave-one-cell-out ===")
Cz = np.where(OBS, C, 0.0)
def l1_weighted(w):
    tot = (Cz * w[:, None]).sum(0); cnt = (OBS * w[:, None]).sum(0)
    D = np.full((NC, 14), np.nan)
    for i in range(NC):
        n = cnt - OBS[i] * w[i]
        D[i] = np.where(n > 1e-12, (tot - Cz[i] * w[i]) / np.maximum(n, 1e-12), np.nan)
    return D
cts = pd.Series(chemo).value_counts()
w_chem = np.array([1.0 / cts[g] for g in chemo]); w_chem = w_chem * NC / w_chem.sum()
res = [sc(l1_weighted(np.ones(NC)), label="L1 unweighted mean-of-centred"),
       sc(l1_weighted(w_chem), label="L1 chemotype-balanced (programme B1 rule)")]
cte = pd.Series(ext).value_counts()
w_ext = np.array([1.0 / cte[g] for g in ext]); w_ext = w_ext * NC / w_ext.sum()
res.append(sc(l1_weighted(w_ext), label="L1 extractant-balanced"))
for r in res:
    print(f"  {r['predictor']:44s} ext-macro {r['ext_macro']:.4f}  pooled {r['pooled']:.4f}")
pd.DataFrame(res).to_csv(os.path.join(OUT, "v_L1_variants.csv"), index=False)

# ---------- two-way (masked LS) donor curves, which reproduced L2/L3/L4 ----------
def two_way(idx):
    Xs, Os = X[idx], OBS[idx]
    mu = np.zeros(14)
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
    D = np.full((NC, 14), np.nan)
    g = pd.Series(groups)
    for k, ix in g.groupby(g).groups.items():
        ix = np.asarray(list(ix))
        if len(ix) < 2: continue
        for i in ix:
            D[i] = two_way(ix[ix != i])
    return D

D3 = build(ext)
D4 = build([f"{a}|{b}" for a, b in zip(ext, pub)])
D2 = build(chemo)

# ---------- L5: nearest acid concentration, single-donor vs tie-averaged ----------
def l5(single):
    D = np.full((NC, 14), np.nan)
    s = pd.Series(ext)
    for k, ix in s.groupby(s).groups.items():
        ix = np.asarray(list(ix))
        if len(ix) < 2: continue
        for i in ix:
            o = ix[ix != i]
            dd = np.abs(acid[o] - acid[i]) if not np.isnan(acid[i]) else np.zeros(len(o))
            dd = np.where(np.isnan(dd), np.inf, dd)
            if not np.isfinite(dd).any():
                sel = o
            else:
                sel = o[dd == dd.min()]
                if single:
                    sel = sel[[int(np.argmax(nmet[sel]))]]
            D[i] = two_way(sel) if len(sel) > 1 else np.where(OBS[sel[0]], C[sel[0]], np.nan)
    return D

for single in (False, True):
    D5 = l5(single)
    m = (~np.isnan(D3[RI, IA] - D3[RI, IB])) & (~np.isnan(D4[RI, IA] - D4[RI, IB])) \
        & (~np.isnan(D5[RI, IA] - D5[RI, IB]))
    S = P[m]
    tag = "single nearest donor" if single else "tie-averaged donors"
    print(f"\n=== like-for-like subset, L5 = {tag}: "
          f"{S.cell_id.nunique()} cells / {S.extractant.nunique()} ext / {len(S)} pairs "
          f"(D3 claim: 299 / 20 / 7228) ===")
    if single:
        keep = m
        D5keep = D5

m = keep
S = P[m].copy()
key = set(zip(S.cell_id, S.A, S.B))
out = []
for lab, D in [("L1 corpus (chemotype-bal)", l1_weighted(w_chem)), ("L2 chemotype", D2),
               ("L3 extractant", D3), ("L4 +publication", D4), ("L5 +nearest acid", D5keep)]:
    r = sc(D, sub=m, label=lab); out.append(r)
    print(f"  {lab:26s} ext-macro {r['ext_macro']:.4f}  pooled {r['pooled']:.4f}  n={r['n_pairs']}")
for a in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "B1_MEAN_CURVE"]:
    pr = pd.read_parquet(os.path.join(PRED, a + ".parquet"))
    k = pd.Series(list(zip(pr.cell_id, pr.A, pr.B)))
    sel = pr[k.isin(key).to_numpy()].copy()
    sel["ae"] = (sel.y - sel.prediction).abs()
    ps = sel.groupby("split_seed").apply(
        lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
    out.append(dict(predictor="ARM_" + a, ext_macro=round(float(ps.mean()), 4),
                    pooled=round(float(sel.ae.mean()), 4), n_ext=sel.extractant.nunique(),
                    n_cells=sel.cell_id.nunique(), n_pairs=len(sel) // 5))
    print(f"  ARM {a:22s} ext-macro {ps.mean():.4f}  pooled {sel.ae.mean():.4f}  "
          f"seed sd {ps.std(ddof=1):.4f}  n/seed={len(sel)//5}")
pd.DataFrame(out).to_csv(os.path.join(OUT, "v_headline_subset.csv"), index=False)

# ---------- headline gap on the definition-free set (all pairs where L3 exists) ----------
print("\n=== headline gap, all 9840 pairs where L3 is defined (twoway estimator) ===")
r = sc(D3, sub=(~np.isnan(D3[RI, IA] - D3[RI, IB])), label="L3 extractant")
print(f"  L3 {r['ext_macro']:.4f} over {r['n_ext']} ext / {r['n_cells']} cells / {r['n_pairs']} pairs")
