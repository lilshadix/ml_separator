"""Second verification pass for diagnostic d1.

  G. the three-level similarity ladder (curve MAE 0.230/0.343/0.380 and
     log-SF MAE 0.342/0.519/0.566 over 3208/7628/20000 pairs, >=4 shared metals),
     plus the permissive >=2 threshold (0.303/0.479/0.526 over 9237/27024/20000).
  H. the median/mean Pearson r within extractant (0.941 / 0.703).
  I. same-publication vs different-publication pair MAE (0.279 vs 0.374,
     1066 / 2142 pairs).
  J. how the "pooled (cell-weighted) 0.295" companion to the 0.286 headline is
     formed, and whether any single extractant/cell dominates the pooled tables.
  K. the acid-gap Spearman 0.165 over 1417 pairs, and its effective sample size.
  L. coverage counts (metals per cell, corpus concentration).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER  # noqa: E402

OUT = "gen13_separation/analysis/stage2/d1_extractant_atlas/verify"
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet("gen13_separation/manifests/cohort_exact.parquet")
LN = sorted([m for m in LANTHANIDES if f"logD__{m}" in df.columns],
            key=lambda m: ATOMIC_NUMBER[m])
logD = df[[f"logD__{m}" for m in LN]].to_numpy(float)
obs = ~np.isnan(logD)
extr = df.extractant.to_numpy()
chemo = df.chemotype.to_numpy()
pub = df.publication_id.to_numpy()
cid = df.cell_id.to_numpy()
n = len(df)


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def pair_stats(i, j, min_shared):
    m = obs[i] & obs[j]
    k = int(m.sum())
    if k < min_shared:
        return None
    a = logD[i][m]
    b = logD[j][m]
    a = a - a.mean()          # re-centre on the SHARED metals
    b = b - b.mean()
    curve_mae = float(np.abs(a - b).mean())
    v = a - b
    d = v[:, None] - v[None, :]
    iu = np.triu_indices(k, 1)
    sf = np.abs(d[iu])
    r = np.nan
    if k >= 3 and a.std() > 0 and b.std() > 0:
        r = float(pearsonr(a, b)[0])
    elif k >= 2 and a.std() > 0 and b.std() > 0:
        r = float(pearsonr(a, b)[0])
    return dict(i=i, j=j, k=k, curve_mae=curve_mae, sf_mae=float(sf.mean()),
                sf_sum=float(sf.sum()), n_sf=len(sf), r=r)


def collect(level, min_shared, rng=None, n_random=20000):
    recs = []
    if level == "random":
        assert rng is not None
        seen = set()
        tries = 0
        while len(recs) < n_random and tries < n_random * 60:
            tries += 1
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            s = pair_stats(int(key[0]), int(key[1]), min_shared)
            if s:
                recs.append(s)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if level == "extractant" and extr[i] != extr[j]:
                    continue
                if level == "chemotype" and not (chemo[i] == chemo[j]
                                                 and extr[i] != extr[j]):
                    continue
                s = pair_stats(i, j, min_shared)
                if s:
                    recs.append(s)
    r = pd.DataFrame(recs)
    r["extr_i"] = extr[r.i.to_numpy()]
    r["same_pub"] = pub[r.i.to_numpy()] == pub[r.j.to_numpy()]
    return r


hr("G. three-level similarity ladder")
rows = []
for min_shared, tag in [(4, ">=4 shared"), (2, ">=2 shared")]:
    rng = np.random.default_rng(12345)
    for level in ["extractant", "chemotype", "random"]:
        r = collect(level, min_shared, rng=rng)
        rows.append(dict(threshold=tag, level=level, n_pairs=len(r),
                         mean_shared=r.k.mean(),
                         curve_mae=r.curve_mae.mean(),
                         sf_mae_pairmean=r.sf_mae.mean(),
                         sf_mae_pooled=r.sf_sum.sum() / r.n_sf.sum(),
                         r_median=r.r.median(), r_mean=r.r.mean(),
                         pct_anticorr=100 * (r.r < 0).mean()))
        print(f"{tag:12s} {level:11s} n={len(r):6d} shared={r.k.mean():.2f} "
              f"curveMAE={r.curve_mae.mean():.4f} sfMAE(pairmean)={r.sf_mae.mean():.4f} "
              f"sfMAE(pooled)={r.sf_sum.sum()/r.n_sf.sum():.4f} "
              f"r_med={r.r.median():.4f} r_mean={r.r.mean():.4f} "
              f"anti={100*(r.r<0).mean():.1f}%")
        if level == "extractant" and min_shared == 4:
            R4 = r
L = pd.DataFrame(rows)
L.to_csv(f"{OUT}/v2_similarity_levels.csv", index=False)

hr("I. same-pub vs different-pub within extractant (>=4 shared metals)")
for flag, lab in [(True, "same publication"), (False, "different publication")]:
    s = R4[R4.same_pub == flag]
    print(f"{lab:24s} n={len(s):5d} sfMAE(pairmean)={s.sf_mae.mean():.4f} "
          f"pooled={s.sf_sum.sum()/s.n_sf.sum():.4f} curveMAE={s.curve_mae.mean():.4f}")

hr("J. independence / dominance of the pooled pair tables")
c = R4.extr_i.value_counts()
nm = df.drop_duplicates("extractant").set_index("extractant").extractant_name
print("within-extractant pairs (>=4 shared) by extractant, top 6:")
for k, v in c.head(6).items():
    print(f"   {nm.get(k, k)[:24]:24s} {v:5d} pairs  {100*v/len(R4):5.1f}% of the 3208")
print(f"   distinct extractants contributing = {c.size}; "
      f"distinct cells involved = {len(set(R4.i) | set(R4.j))}")
print("=> the 'median r = 0.941 over 3208 pairs' is NOT 3208 independent "
      "observations: the pairs come from %d cells of %d extractants."
      % (len(set(R4.i) | set(R4.j)), c.size))
# extractant-macro version of the pairwise table, for comparison with 0.342
print("extractant-macro (mean within extractant, then over extractants) of the "
      "within-extractant sfMAE = %.4f (vs pooled-over-pairs %.4f)"
      % (R4.groupby("extr_i").sf_mae.mean().mean(), R4.sf_mae.mean()))
R4.groupby("extr_i").agg(n_pairs=("sf_mae", "size"), sf_mae=("sf_mae", "mean"),
                         r_med=("r", "median")).assign(
    name=lambda d: d.index.map(nm)).to_csv(f"{OUT}/v2_within_extr_by_extractant.csv")

hr("J2. reconciling the claimed pooled companion 0.295 of the 0.286 headline")
# rebuild the directed transfer table (target,donor) as in v1
recs = []
for i in range(n):
    for j in range(n):
        if i == j or extr[i] != extr[j]:
            continue
        m = obs[i] & obs[j]
        k = int(m.sum())
        if k < 2:
            continue
        v = logD[i][m] - logD[j][m]
        d = v[:, None] - v[None, :]
        iu = np.triu_indices(k, 1)
        e = np.abs(d[iu])
        recs.append(dict(target=cid[i], extractant=extr[i], donor=cid[j],
                         mean_abs=float(e.mean()), sum_abs=float(e.sum()),
                         n_sf=len(e)))
T = pd.DataFrame(recs)
cell = T.groupby(["extractant", "target"]).agg(
    cell_mean=("mean_abs", "mean"),
    cell_pooled=("sum_abs", lambda s: np.nan)).reset_index()
cell["cell_mean"] = T.groupby(["extractant", "target"]).mean_abs.mean().values
g = T.groupby(["extractant", "target"])[["sum_abs", "n_sf"]].sum()
cell["cell_pairpooled"] = (g.sum_abs / g.n_sf).values
variants = {
    "mean over the 450 target cells of (mean over donors)": cell.cell_mean.mean(),
    "mean over the 450 target cells of (pair-pooled)": cell.cell_pairpooled.mean(),
    "mean over all 18474 (target,donor) records": T.mean_abs.mean(),
    "pooled over all individual SF observations": T.sum_abs.sum() / T.n_sf.sum(),
    "extractant-macro of cell means (the 0.286 route)":
        cell.groupby("extractant").cell_mean.mean().mean(),
}
for k, v in variants.items():
    print(f"   {v:.4f}   {k}")
print("claimed pooled/cell-weighted companion = 0.295 -> matches "
      "'mean over the 450 target cells' (%.4f), i.e. it is CELL-macro, not "
      "SF-observation-pooled (%.4f)." % (cell.cell_mean.mean(),
                                         T.sum_abs.sum() / T.n_sf.sum()))
cell.to_csv(f"{OUT}/v2_transfer_cell_level.csv", index=False)

hr("K. acid gap")
acid_col = "cond__acid_concentration_M"
hno3 = None
for cand in ["cond__acid__hno3", "cond__acid_hno3", "cond__acid__HNO3"]:
    if cand in df.columns:
        hno3 = cand
        break
print("acid columns found:", acid_col in df.columns, "| hno3 flag:", hno3)
if hno3 and acid_col in df.columns:
    isn = df[hno3].to_numpy() == 1
    mol = df[acid_col].to_numpy(float)
    ok = isn & np.isfinite(mol) & (mol > 0)
    print(f"cells with HNO3 and a positive known molarity: {ok.sum()}")
    sub = R4 if False else None
    # use the permissive within-extractant pair set at >=4 shared metals
    q = R4[[ok[i] and ok[j] for i, j in zip(R4.i, R4.j)]].copy()
    q["dlog"] = [abs(np.log10(mol[i]) - np.log10(mol[j])) for i, j in zip(q.i, q.j)]
    rho, p = spearmanr(q.dlog, q.sf_mae)
    print(f"pairs = {len(q)}  Spearman(|dlog10 M|, sfMAE) = {rho:.4f} (p={p:.2e})")
    bins = [-1e-9, 1e-9, 0.2, 0.5, 1.0, 99]
    labs = ["identical", "<0.2 dex", "0.2-0.5", "0.5-1", ">1"]
    q["bin"] = pd.cut(q.dlog, bins, labels=labs)
    bt = q.groupby("bin", observed=True).agg(n=("sf_mae", "size"),
                                             sf_mae=("sf_mae", "mean"))
    print(bt.to_string())
    bt.to_csv(f"{OUT}/v2_acid_gap_bins.csv")
    ncells = len(set(q.i) | set(q.j))
    print(f"=> those {len(q)} pairs are built from only {ncells} distinct cells "
          f"({len(q)/ncells:.1f} pairs per cell); the pairs are strongly "
          f"dependent, so n=1417 is not an effective sample size.")

hr("L. coverage counts")
k = obs.sum(1)
print(f"metals/cell: median={np.median(k):.0f} mean={k.mean():.3f}; "
      f"exactly 2: {(k==2).sum()} ({100*(k==2).mean():.1f}%); "
      f">=4: {(k>=4).sum()} ({100*(k>=4).mean():.1f}%); >=5: {(k>=5).sum()}")
vc = df.extractant.value_counts()
print(f"extractants={vc.size}; with exactly 1 cell={int((vc==1).sum())}; "
      f"top12 cells={int(vc.head(12).sum())} ({100*vc.head(12).sum()/n:.1f}%); "
      f"top1={vc.iloc[0]} ({nm.get(vc.index[0])}, {100*vc.iloc[0]/n:.1f}%)")
print(f"TODGA rows(entries)={int(obs[extr==vc.index[0]].sum())}, "
      f"publications={df[df.extractant==vc.index[0]].publication_id.nunique()}")
print(f"replicate_sd_median over cells: median={df.replicate_sd_median.median():.4f}")
print("\nwrote:", OUT)
