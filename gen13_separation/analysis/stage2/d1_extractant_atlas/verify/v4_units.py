"""Fourth pass: put every transfer oracle and every model arm into ONE unit
(extractant-macro, mean over the same extractant set) so the headline
comparisons can be judged, and quantify the pooled-vs-macro label."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER  # noqa: E402

OUT = "gen13_separation/analysis/stage2/d1_extractant_atlas/verify"
df = pd.read_parquet("gen13_separation/manifests/cohort_exact.parquet")
LN = sorted([m for m in LANTHANIDES if f"logD__{m}" in df.columns],
            key=lambda m: ATOMIC_NUMBER[m])
logD = df[[f"logD__{m}" for m in LN]].to_numpy(float)
obs = ~np.isnan(logD)
extr = df.extractant.to_numpy()
chemo = df.chemotype.to_numpy()
cid = df.cell_id.to_numpy()
n = len(df)


def sf_err(i, j):
    m = obs[i] & obs[j]
    k = int(m.sum())
    if k < 2:
        return None
    v = logD[i][m] - logD[j][m]
    d = v[:, None] - v[None, :]
    return np.abs(d[np.triu_indices(k, 1)])


def oracle(mask, label):
    """cell score = mean over allowed donors of that donor's mean SF error;
    extractant-macro = mean of cell scores in an extractant, then over extractants."""
    recs = []
    tot_sum = tot_n = 0.0
    for i in range(n):
        es = []
        for j in range(n):
            if i == j or not mask(i, j):
                continue
            e = sf_err(i, j)
            if e is None:
                continue
            es.append(e.mean())
            tot_sum += e.sum()
            tot_n += len(e)
        if es:
            recs.append(dict(cell_id=cid[i], extractant=extr[i], score=np.mean(es),
                             n_donors=len(es)))
    R = pd.DataFrame(recs)
    macro = R.groupby("extractant").score.mean().mean()
    print(f"{label:44s} macro={macro:.4f} over {R.extractant.nunique():2d} extractants "
          f"/ {len(R)} cells; cell-weighted={R.score.mean():.4f}; "
          f"SF-observation-pooled={tot_sum/tot_n:.4f} ({int(tot_n)} SF obs)")
    return R, macro


print("=== every oracle in the SAME extractant-macro unit ===")
own_R, own_m = oracle(lambda i, j: extr[i] == extr[j], "copy same-extractant sibling")
ctl_R, ctl_m = oracle(lambda i, j: chemo[i] == chemo[j] and extr[i] != extr[j],
                      "copy same-chemotype other-extractant cell")
rnd_R, rnd_m = oracle(lambda i, j: True, "copy any other cell (random level)")

print("\n=== model arms on matching extractant sets ===")
sets = {"all 90 extractants": None,
        "the 24 own-oracle extractants": set(own_R.extractant),
        "the 56 control-oracle extractants": set(ctl_R.extractant)}
rows = []
for arm in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "B1_MEAN_CURVE",
            "B4_HEAVIER_ALWAYS"]:
    p = pd.read_parquet(f"gen13_separation/predictions/B_primary/{arm}.parquet")
    p["e"] = (p.y - p.prediction).abs()
    r = {"arm": arm}
    for lab, s in sets.items():
        q = p if s is None else p[p.extractant.isin(s)]
        r[lab] = round(q.groupby(["split_seed", "extractant"]).e.mean()
                       .groupby("split_seed").mean().mean(), 4)
    rows.append(r)
    print(f"{arm:26s} " + "  ".join(f"{k}={r[k]:.4f}" for k in sets))
M = pd.DataFrame(rows)
M.to_csv(f"{OUT}/v4_arms_on_matched_extractant_sets.csv", index=False)

print("\n=== headline comparisons, restated on matched scopes ===")
best_all = float(M.loc[M.arm == "X_ENS_DIRECT+LOWRANK_K2", "all 90 extractants"].iloc[0])
best_24 = float(M.loc[M.arm == "X_ENS_DIRECT+LOWRANK_K2",
                      "the 24 own-oracle extractants"].iloc[0])
best_56 = float(M.loc[M.arm == "X_ENS_DIRECT+LOWRANK_K2",
                      "the 56 control-oracle extractants"].iloc[0])
print(f"claimed  : own-oracle {own_m:.3f} vs best arm {best_all:.3f} -> gap "
      f"{best_all-own_m:+.3f} (summary says +0.195)")
print(f"matched  : own-oracle {own_m:.3f} vs best arm on the SAME 24 extractants "
      f"{best_24:.3f} -> gap {best_24-own_m:+.3f}")
print(f"control  : {ctl_m:.3f} vs best arm on the same 56 extractants {best_56:.3f} "
      f"-> gap {best_56-ctl_m:+.3f}")
print(f"random-copy oracle in extractant-macro units = {rnd_m:.4f}; the summary "
      f"quotes 0.526 for this level, which is a mean over sampled cell PAIRS, "
      f"not extractant-macro; best arm all-extractants = {best_all:.4f}")
pd.DataFrame([dict(quantity="own oracle macro", value=own_m),
              dict(quantity="control oracle macro", value=ctl_m),
              dict(quantity="random oracle macro", value=rnd_m),
              dict(quantity="best arm macro all 90", value=best_all),
              dict(quantity="best arm macro on 24", value=best_24),
              dict(quantity="best arm macro on 56", value=best_56),
              dict(quantity="gap claimed (own vs arm all)", value=best_all - own_m),
              dict(quantity="gap matched (own vs arm on 24)", value=best_24 - own_m),
              ]).to_csv(f"{OUT}/v4_headline_restated.csv", index=False)
