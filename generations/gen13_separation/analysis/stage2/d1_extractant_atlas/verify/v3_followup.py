"""Third pass: pin down the pair-count discrepancies and finish the like-for-like
model comparison that the headline rests on."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, "generations/gen13_separation")
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER  # noqa: E402

OUT = "generations/gen13_separation/analysis/stage2/d1_extractant_atlas/verify"
df = pd.read_parquet("generations/gen13_separation/manifests/cohort_exact.parquet")
LN = sorted([m for m in LANTHANIDES if f"logD__{m}" in df.columns],
            key=lambda m: ATOMIC_NUMBER[m])
logD = df[[f"logD__{m}" for m in LN]].to_numpy(float)
obs = ~np.isnan(logD)
extr = df.extractant.to_numpy()
chemo = df.chemotype.to_numpy()
n = len(df)

print("=== count reconciliation at >=4 shared metals ===")
for level in ["extractant", "chemotype"]:
    tot = degen = 0
    for i in range(n):
        for j in range(i + 1, n):
            if level == "extractant" and extr[i] != extr[j]:
                continue
            if level == "chemotype" and not (chemo[i] == chemo[j] and extr[i] != extr[j]):
                continue
            m = obs[i] & obs[j]
            if m.sum() < 4:
                continue
            tot += 1
            a = logD[i][m] - logD[i][m].mean()
            b = logD[j][m] - logD[j][m].mean()
            if a.std() == 0 or b.std() == 0:
                degen += 1
    print(f"{level:11s} total={tot}  degenerate(flat curve, r undefined)={degen}  "
          f"with r defined={tot-degen}")

print("\n=== alternative chemotype-pair definitions (looking for 7628) ===")
cnt = {}
for name, cond in [
    ("same chemotype, different extractant", lambda i, j: chemo[i] == chemo[j] and extr[i] != extr[j]),
    ("same chemotype, any extractant", lambda i, j: chemo[i] == chemo[j]),
    ("same ecfp_cluster, different extractant",
     lambda i, j: df.ecfp_cluster.iloc[i] == df.ecfp_cluster.iloc[j] and extr[i] != extr[j]),
    ("same chem_family, different extractant",
     lambda i, j: df.chem_family.iloc[i] == df.chem_family.iloc[j] and extr[i] != extr[j]),
]:
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not cond(i, j):
                continue
            if (obs[i] & obs[j]).sum() >= 4:
                t += 1
    cnt[name] = t
    print(f"  {name:44s} {t}")

print("\n=== D (finished): like-for-like scope for the 0.286-vs-0.481 headline ===")
PRED = "generations/gen13_separation/predictions/B_primary"
# oracle-covered cells: cells with >=1 same-extractant sibling sharing >=2 metals
cov = []
for i in range(n):
    ok = any(j != i and extr[j] == extr[i] and (obs[i] & obs[j]).sum() >= 2
             for j in range(n))
    if ok:
        cov.append(df.cell_id.iloc[i])
cov = set(cov)
print(f"oracle-covered cells = {len(cov)}")
rows = []
for arm in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE"]:
    p = pd.read_parquet(f"{PRED}/{arm}.parquet")
    p["e"] = (p.y - p.prediction).abs()

    def macro(q):
        return q.groupby(["split_seed", "extractant"]).e.mean().groupby("split_seed").mean().mean()
    full = macro(p)
    restr = macro(p[p.cell_id.isin(cov)])
    rows.append(dict(arm=arm, macro_all_90_extractants=round(full, 4),
                     macro_on_24_oracle_extractants=round(restr, 4),
                     oracle=0.2864, gap_claimed=round(full - 0.2864, 4),
                     gap_like_for_like=round(restr - 0.2864, 4)))
    print(f"  {arm:26s} all={full:.4f}  on the 24 oracle extractants={restr:.4f}  "
          f"like-for-like gap vs oracle 0.2864 = {restr-0.2864:+.4f} "
          f"(claimed gap {full-0.2864:+.4f})")
pd.DataFrame(rows).to_csv(f"{OUT}/v3_like_for_like.csv", index=False)

print("\n=== replicate sd reference ===")
rep = df[[f"repsd__{m}" for m in LN]].to_numpy(float)
v = rep[np.isfinite(rep)]
print(f"median over {len(v)} (cell,metal) within-replicate sds = {np.nanmedian(v):.4f}")
print(f"median over cells of replicate_sd_median = {df.replicate_sd_median.median():.4f} "
      f"({df.replicate_sd_median.notna().sum()} non-null cells)")
