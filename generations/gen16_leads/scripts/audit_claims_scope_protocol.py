"""Audit lens: claims, scope and protocol.  Read-only over gen16 artefacts.

Computes (a) the chemotype-blocked family-wise permutation bar for the post-hoc
composition-position family the refuters searched, using the SAME construction the
pre-registration fixed for L1 (chemotype-level permutation of the target, max |rho|
over the family), and (b) the diglycolamide control on the descriptors and on the
"attenuation-free" cycle estimators the DECISION_REPORT quotes.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "D:/ml_separator_gh/gen16_leads"
OUT = f"{ROOT}/results/audit/claims_scope_protocol"

d = pd.read_csv(f"{ROOT}/results/refutation/L1NULL/B/checkD_composition_descriptors.csv")
DESC = [c for c in d.columns if c not in
        ("extractant", "series", "a", "b", "abs_a", "n_metals", "chemotype",
         "S8", "S14", "S3", "naive_slope")]
print("descriptors in lens-B family:", len(DESC))

rows = []
for s in ("S8", "S14"):
    sub = d[d[s].astype(bool)]
    for c in DESC:
        m = sub[c].notna() & sub["a"].notna()
        if m.sum() >= 10:
            r, p = spearmanr(sub.loc[m, c], sub.loc[m, "a"])
            rows.append(dict(set=s, descriptor=c, n=int(m.sum()),
                             n_chemotypes=int(sub.loc[m, "chemotype"].nunique()),
                             rho=r, p=p))
obs = pd.DataFrame(rows)
obs.to_csv(f"{OUT}/composition_family_observed.csv", index=False)
print(obs.reindex(obs["rho"].abs().sort_values(ascending=False).index).head(6).to_string())

# family-wise chemotype-permutation null: permute the target between chemotype blocks
rng = np.random.default_rng(8675309)
REPS = 2000
maxabs = np.empty(REPS)
blocks = {s: d[d[s].astype(bool)].copy() for s in ("S8", "S14")}
for i in range(REPS):
    perm_maps = {}
    for s, sub in blocks.items():
        ch = sub["chemotype"].to_numpy()
        uch = np.unique(ch)
        # permute the per-chemotype label sets: shuffle chemotype identities of the target
        order = rng.permutation(len(uch))
        mapping = dict(zip(uch, uch[order]))
        pooled = {c: sub.loc[sub["chemotype"] == c, "a"].to_numpy() for c in uch}
        newa = np.empty(len(sub))
        cursor = {c: 0 for c in uch}
        for j, c in enumerate(ch):
            src = mapping[c]
            vals = pooled[src]
            newa[j] = vals[cursor[src] % len(vals)]
            cursor[src] += 1
        perm_maps[s] = newa
    best = 0.0
    for s, sub in blocks.items():
        av = perm_maps[s]
        for c in DESC:
            m = sub[c].notna().to_numpy() & ~np.isnan(av)
            if m.sum() >= 10:
                r, _ = spearmanr(sub.loc[m, c], av[m])
                if np.isfinite(r):
                    best = max(best, abs(r))
    maxabs[i] = best

bar = dict(family_size=int(len(obs)), reps=REPS,
           null_p95_max_abs_rho=float(np.percentile(maxabs, 95)),
           null_p50_max_abs_rho=float(np.percentile(maxabs, 50)),
           null_p99_max_abs_rho=float(np.percentile(maxabs, 99)),
           observed_max_abs_rho=float(obs["rho"].abs().max()),
           observed_argmax=str(obs.loc[obs["rho"].abs().idxmax(), "descriptor"]),
           familywise_p=float((maxabs >= obs["rho"].abs().max()).mean()))
print(json.dumps(bar, indent=2))

# diglycolamide control on the top composition descriptors
dga = []
for s in ("S8", "S14"):
    sub = d[d[s].astype(bool)]
    for c in ("steppos_n_H2O", "slope_n_H2O_vs_r", "steppos_n_fill", "slope_n_fill_vs_r"):
        m = sub[c].notna()
        r_all, _ = spearmanr(sub.loc[m, c], sub.loc[m, "a"])
        k = m & (sub["chemotype"] != "sc009")
        r_no, p_no = (spearmanr(sub.loc[k, c], sub.loc[k, "a"]) if k.sum() >= 6 else (np.nan, np.nan))
        o = m & (sub["chemotype"] == "sc009")
        r_only, _ = (spearmanr(sub.loc[o, c], sub.loc[o, "a"]) if o.sum() >= 6 else (np.nan, np.nan))
        dga.append(dict(set=s, descriptor=c, n=int(m.sum()), rho_all=r_all,
                        n_no_sc009=int(k.sum()), rho_no_sc009=r_no, p_no_sc009=p_no,
                        n_sc009=int(o.sum()), rho_sc009_only=r_only,
                        rho_vs_naive_slope=float(spearmanr(sub.loc[m, c], sub.loc[m, "naive_slope"])[0])))
dga = pd.DataFrame(dga)
dga.to_csv(f"{OUT}/composition_diglycolamide_control.csv", index=False)
print(dga.to_string())
with open(f"{OUT}/composition_familywise_bar.json", "w") as f:
    json.dump(bar, f, indent=2)
