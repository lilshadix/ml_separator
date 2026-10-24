"""Skeptic A, lab-4: are the same-lab rank correlations of candidate-B1 contrasts driven by exact ties or one lab? EXPLORATORY."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np, pandas as pd
from r4 import paths
from gen20k import data as D
from lab4_recompute_skA import build_rules
RNG = np.random.default_rng(11)
f = D.load_bench().frame
rules, labof, links, sn = build_rules(set(f.publication_id.astype(str)))
base_cols = ["split_seed", "cell_id", "read_set", "A", "B", "draw", "k", "policy"]
ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet"); ref = ref[ref.k == 3].copy()
con = {}
for s in ("S02", "S05", "S06"):
    sl = pd.read_parquet(paths.PRED / f"{s}_BPG.parquet"); sl = sl[sl.k == 3]
    new = [c for c in sl.columns if c not in ref.columns]
    ref = ref.merge(sl[base_cols + new], on=base_cols, how="left", validate="one_to_one"); con[s] = new[0]
ref["publication"] = ref.publication.astype(str)
up = sorted(ref.publication.unique()); G = len(up)
PA = [(a, b) for i, a in enumerate(up) for b in up[i + 1:]]
ia = np.array([up.index(a) for a, _ in PA]); ib = np.array([up.index(b) for _, b in PA])
perms = np.argsort(RNG.random((20000, G)), axis=1)
for s, c in con.items():
    d = ref.assign(sr=(ref.B1 - ref.y) ** 2, sc=(ref[c] - ref.y) ** 2)
    g = d.groupby(["publication", "split_seed"])[["sr", "sc"]].sum().groupby("publication").mean().loc[up]
    L = np.log(g.sr / g.sc)
    ties = (L.abs() < 1e-12)
    print(f"{s} ({c}): pooled rel {1 - g.sc.sum() / g.sr.sum():+.4f}; exact-tie publications {int(ties.sum())} of {G}")
    for rn in ("R0", "R2"):
        m = np.array([rules[rn](a, b) for a, b in PA])
        labs = sorted({labof['R0c' if rn == 'R0' else rn][a] for (a, b), mm in zip(PA, m) if mm})
        rk = L.rank().to_numpy(); z = (rk - rk.mean()) / rk.std()
        obs = (z[ia][m] * z[ib][m]).mean()
        # without ties
        nt = ~ties.to_numpy()
        m2 = m & nt[ia] & nt[ib]
        rk2 = L[nt].rank().reindex(L.index).to_numpy(); z2 = np.where(nt, (rk2 - np.nanmean(rk2)) / np.nanstd(rk2), 0.0)
        obs2 = (z2[ia][m2] * z2[ib][m2]).mean() if m2.any() else np.nan
        zp = z2[perms]; nullv = (zp[:, ia[m2]] * zp[:, ib[m2]]).mean(1)
        # leave-one-lab-out on the full statistic
        loo = []
        for lab in labs:
            keep = np.array([not (labof['R0c' if rn == 'R0' else rn][a] == lab) for a, _ in PA])
            loo.append((lab, round(float((z[ia][m & keep] * z[ib][m & keep]).mean()), 2)))
        print(f"   {rn}: rank r_SL {obs:+.2f} ({int(m.sum())} pairs); without tie pubs {obs2:+.2f} ({int(m2.sum())} pairs, perm p {(nullv >= obs2 - 1e-12).mean():.3f}); LOO-lab {loo}")
    lab2 = pd.Series({p: labof["R2"][p] for p in up})
    t = pd.DataFrame({"L": L, "lab": lab2, "tie": ties})
    print(t[t.lab.map(t.lab.value_counts()) > 1].sort_values("lab").round(3).to_string())
