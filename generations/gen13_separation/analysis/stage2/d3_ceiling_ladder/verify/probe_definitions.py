"""Probe which donor-curve estimator reproduces the D3 rung values, and score the
arms on a subset definition that does not depend on the L5 tie-breaking rule."""
import os
import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
OUT = os.path.join(ROOT, "generations/gen13_separation/analysis/stage2/d3_ceiling_ladder/verify")
LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Zn = dict(zip(LN, [57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]))
IDX = {m: i for i, m in enumerate(LN)}

coh = pd.read_parquet(os.path.join(ROOT, "generations/gen13_separation/manifests/cohort_exact.parquet"))
ext = coh["extractant"].to_numpy(); chemo = coh["chemotype"].to_numpy()
pub = coh["publication_id"].to_numpy(); cells = coh["cell_id"].to_numpy()
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

def em(d, c="ae"):
    return float(d.groupby("extractant", observed=True)[c].mean().mean())

def two_way_curve(idx):
    """min_{b_i, mu_m} sum_obs (X_im - b_i - mu_m)^2  -- the unbiased 'mean curve'
    when metal coverage is unbalanced across donor cells."""
    Xs, Os = X[idx], OBS[idx]
    mu = np.zeros(14); b = np.zeros(len(idx))
    for _ in range(200):
        R = np.where(Os, Xs - mu[None, :], 0.0)
        b = R.sum(1) / np.maximum(Os.sum(1), 1)
        R2 = np.where(Os, Xs - b[:, None], 0.0)
        mu_new = np.where(Os.sum(0) > 0, R2.sum(0) / np.maximum(Os.sum(0), 1), np.nan)
        mu_new = mu_new - np.nanmean(mu_new)
        if np.nanmax(np.abs(mu_new - mu)) < 1e-10:
            mu = mu_new; break
        mu = np.nan_to_num(mu_new, nan=0.0)
    mu = np.where(Os.sum(0) > 0, mu, np.nan)
    return mu

def naive_curve(idx):
    n = OBS[idx].sum(0)
    s = np.where(OBS[idx], C[idx], 0.0).sum(0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)

def build(groups, fn):
    D = np.full((NC, 14), np.nan)
    g = pd.Series(groups)
    for k, ix in g.groupby(g).groups.items():
        ix = np.asarray(list(ix))
        if len(ix) < 2: continue
        for i in ix:
            D[i] = fn(ix[ix != i])
    return D

def build_corpus(fn):
    D = np.full((NC, 14), np.nan)
    allix = np.arange(NC)
    for i in range(NC):
        D[i] = fn(allix[allix != i])
    return D

def score(D, tag, sub=None):
    ri = P.ri.to_numpy()
    p = D[ri, P.A.map(IDX).to_numpy()] - D[ri, P.B.map(IDX).to_numpy()]
    d = P.assign(p=p)
    if sub is not None:
        d = d[sub]
    d = d.dropna(subset=["p"]).copy()
    d["ae"] = (d.y - d.p).abs()
    return em(d), float(d.ae.mean()), d.extractant.nunique(), d.cell_id.nunique(), len(d)

print("=== donor-curve estimator: naive mean-of-centred vs two-way least squares ===")
res = []
for tag, groups in [("L2_chemotype", chemo), ("L3_extractant", ext),
                    ("L4_ext_pub", [f"{a}|{b}" for a, b in zip(ext, pub)])]:
    for name, fn in [("naive", naive_curve), ("twoway", two_way_curve)]:
        v = score(build(groups, fn), tag)
        res.append(dict(rung=tag, estimator=name, ext_macro=round(v[0], 4), pooled=round(v[1], 4),
                        n_ext=v[2], n_cells=v[3], n_pairs=v[4]))
        print(f"  {tag:14s} {name:7s} ext-macro {v[0]:.4f} pooled {v[1]:.4f} "
              f"({v[2]} ext / {v[3]} cells / {v[4]} pairs)")
for name, fn in [("naive", naive_curve), ("twoway", two_way_curve)]:
    v = score(build_corpus(fn), "L1")
    res.append(dict(rung="L1_corpus", estimator=name, ext_macro=round(v[0], 4), pooled=round(v[1], 4),
                    n_ext=v[2], n_cells=v[3], n_pairs=v[4]))
    print(f"  {'L1_corpus':14s} {name:7s} ext-macro {v[0]:.4f} pooled {v[1]:.4f} "
          f"({v[2]} ext / {v[3]} cells / {v[4]} pairs)")
pd.DataFrame(res).to_csv(os.path.join(OUT, "v_estimator_probe.csv"), index=False)

# ---- like-for-like on a definition-free set: every pair where L3 is defined ----
print("\n=== like-for-like on ALL pairs where L3 is defined (no L5 tie-break needed) ===")
D3n, D3t = build(ext, naive_curve), build(ext, two_way_curve)
ri = P.ri.to_numpy()
p3n = D3n[ri, P.A.map(IDX).to_numpy()] - D3n[ri, P.B.map(IDX).to_numpy()]
p3t = D3t[ri, P.A.map(IDX).to_numpy()] - D3t[ri, P.B.map(IDX).to_numpy()]
S = P.assign(p3n=p3n, p3t=p3t)
S = S[S.p3n.notna()].copy()
print(f"  set: {S.cell_id.nunique()} cells / {S.extractant.nunique()} extractants / {len(S)} pairs")
key = set(zip(S.cell_id, S.A, S.B))
out = []
for nm, col in [("L3_naive", "p3n"), ("L3_twoway", "p3t")]:
    d = S.assign(ae=(S.y - S[col]).abs())
    out.append(dict(predictor=nm, ext_macro=round(em(d), 4), pooled=round(float(d.ae.mean()), 4),
                    n_pairs=len(d)))
    print(f"  {nm:26s} ext-macro {em(d):.4f}  pooled {d.ae.mean():.4f}")
PRED = os.path.join(ROOT, "generations/gen13_separation/predictions/B_primary")
for a in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE"]:
    pr = pd.read_parquet(os.path.join(PRED, a + ".parquet"))
    k = pd.Series(list(zip(pr.cell_id, pr.A, pr.B)))
    sel = pr[k.isin(key).to_numpy()].copy()
    sel["ae"] = (sel.y - sel.prediction).abs()
    ps = sel.groupby("split_seed").apply(
        lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
    out.append(dict(predictor="ARM_" + a, ext_macro=round(float(ps.mean()), 4),
                    pooled=round(float(sel.ae.mean()), 4), n_pairs=len(sel) // 5))
    print(f"  ARM {a:22s} ext-macro {ps.mean():.4f}  pooled {sel.ae.mean():.4f}  "
          f"seed sd {ps.std(ddof=1):.4f}")
pd.DataFrame(out).to_csv(os.path.join(OUT, "v_likeforlike_L3defined.csv"), index=False)

# ---- fragility of a 20-24 extractant macro average ----
print("\n=== per-extractant L3 error and macro-average fragility ===")
d = S.assign(ae=(S.y - S.p3n).abs())
pe = d.groupby("extractant", observed=True).agg(mae=("ae", "mean"), n_pairs=("ae", "size"),
                                                n_cells=("cell_id", "nunique")).sort_values("mae")
pe.to_csv(os.path.join(OUT, "v_L3_per_extractant.csv"))
print(pe.round(3).to_string())
print(f"  ext-macro {pe.mae.mean():.4f}; sd across {len(pe)} extractants {pe.mae.std(ddof=1):.4f}; "
      f"se {pe.mae.std(ddof=1)/np.sqrt(len(pe)):.4f}")
print(f"  drop the 3 worst extractants -> {pe.mae.iloc[:-3].mean():.4f}; "
      f"drop the 4 worst -> {pe.mae.iloc[:-4].mean():.4f}")
