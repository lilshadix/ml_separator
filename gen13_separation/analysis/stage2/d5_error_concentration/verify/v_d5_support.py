"""Second pass: supporting numbers behind the D5 modelling implications.

Covers (a) how much DATA the 'worst 10' concentration actually covers, (b) the
replicate-noise-floor statistics, (c) the label-free abstention controls.
    .venv/Scripts/python.exe gen13_separation/analysis/stage2/d5_error_concentration/verify/v_d5_support.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "gen13_separation"
OUT = os.path.join(ROOT, "analysis", "stage2", "d5_error_concentration", "verify")
LN = "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
res = {}

d = pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary",
                                 "X_ENS_DIRECT+LOWRANK_K2.parquet"))
co = pd.read_parquet(os.path.join(ROOT, "manifests", "cohort_exact.parquet"))
d = d.assign(ae=(d.y - d.prediction).abs(), absy=d.y.abs())
name = co.drop_duplicates("extractant").set_index("extractant")["extractant_name"]

# ---- (a) coverage of the worst 10 -------------------------------------------
tab = d.groupby(["split_seed", "extractant"])["ae"].mean().unstack(0).mean(axis=1)
tab = tab.sort_values(ascending=False)
one = d[d.split_seed == d.split_seed.iloc[0]]
pps = one.groupby("extractant").size()
ncell = d.groupby("extractant")["cell_id"].nunique()
rows = []
for k in (5, 10, 20):
    sel = tab.index[:k]
    rows.append({"k": k,
                 "share_of_summed_per_extractant_MAE": tab.head(k).sum() / tab.sum(),
                 "share_of_held_out_pairs": pps[sel].sum() / pps.sum(),
                 "n_cells_covered": int(ncell[sel].sum()),
                 "macro_after_drop": float(
                     d[~d.extractant.isin(sel)].groupby(["split_seed", "extractant"])["ae"]
                     .mean().groupby("split_seed").mean().mean()),
                 "pooled_after_drop": float(d[~d.extractant.isin(sel)].ae.mean())})
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(OUT, "v_worst_k_coverage.csv"), index=False)
print(cov.to_string(index=False))
res["worst10_share_of_pairs"] = float(cov.loc[cov.k == 10, "share_of_held_out_pairs"].iloc[0])
res["pooled_all"] = float(d.ae.mean())

print("\npairs/seed per extractant: max = %d (%s), 2nd = %d (%s), min = %d (%s)" % (
    pps.max(), name[pps.idxmax()],
    pps.sort_values().iloc[-2], name[pps.sort_values().index[-2]],
    pps.min(), name[pps.idxmin()]))
res["max_pairs_per_seed"] = int(pps.max())
res["max_pairs_extractant"] = str(name[pps.idxmax()])
res["max_cells_one_extractant"] = int(ncell.max())

# ---- (b) replicate noise floor ----------------------------------------------
sub = co[co.replicate_sd_median.notna()]
rs = sub[[f"repsd__{m}" for m in LN]]
res["n_cells_with_replicates"] = int(len(sub))
res["median_of_per_cell_medians"] = float(sub.replicate_sd_median.median())
res["median_of_individual_repsd"] = float(rs.stack(future_stack=True).dropna().median())
res["n_repsd_values"] = int(rs.notna().sum().sum())
res["n_logD_values"] = int(co[[f"logD__{m}" for m in LN]].notna().sum().sum())
print("\nreplicates: %d/521 cells, %d/%d logD values; median of per-cell medians %.4f, "
      "median of the %d individual repsd %.4f -- SAME 41 cells, not two populations"
      % (len(sub), res["n_repsd_values"], res["n_logD_values"],
         res["median_of_per_cell_medians"], res["n_repsd_values"],
         res["median_of_individual_repsd"]))

# ---- (c) abstention controls (label-free, deterministic) --------------------
def norm(s):
    return float(s.ae.sum() / s.absy.sum())


rng = np.random.default_rng(0)
ab = []
for frac in (0.10, 0.20):
    thr = d.prediction.abs().quantile(1 - frac)
    k1 = d[d.prediction.abs() <= thr]
    dz = d.sort_values("dZ", ascending=False)
    k2 = dz.iloc[int(frac * len(dz)):]
    k3 = d.iloc[rng.permutation(len(d))[int(frac * len(d)):]]
    k4 = d[d.ae <= d.ae.quantile(1 - frac)]
    for lbl, k in [("abs_prediction", k1), ("dZ_only", k2), ("random", k3), ("oracle", k4)]:
        ab.append({"abstain_frac": frac, "rule": lbl,
                   "pooled_MAE": float(k.ae.mean()), "MAE_over_mean_abs_y": norm(k)})
ab = pd.DataFrame(ab)
ab.to_csv(os.path.join(OUT, "v_abstention_controls.csv"), index=False)
print("\nbaseline pooled %.4f  normalised %.4f" % (d.ae.mean(), norm(d)))
print(ab.to_string(index=False))

res["spearman_absprediction_abserror"] = float(spearmanr(d.prediction.abs(), d.ae).statistic)
res["spearman_dZ_abserror"] = float(spearmanr(d.dZ, d.ae).statistic)
w = [(len(g), spearmanr(g.prediction.abs(), g.ae).statistic)
     for _, g in d.groupby("dZ") if len(g) > 30]
res["within_dZ_spearman_absprediction"] = float(sum(a * b for a, b in w) / sum(a for a, _ in w))
print("\nSpearman(|pred|,|err|)=%.4f  Spearman(dZ,|err|)=%.4f  within-dZ(|pred|)=%.4f"
      % (res["spearman_absprediction_abserror"], res["spearman_dZ_abserror"],
         res["within_dZ_spearman_absprediction"]))

with open(os.path.join(OUT, "v_support_results.json"), "w") as f:
    json.dump(res, f, indent=2)
print("\nwrote", os.path.join(OUT, "v_support_results.json"))
