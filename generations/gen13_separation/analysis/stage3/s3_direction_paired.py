"""Paired chemotype-blocked comparison of the direction models against 'always heavy-selective',
using one shared resample per design, over every prediction file produced so far."""
import sys, glob
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
RNG = np.random.default_rng(8675309); REPS = 10000
frames = [pd.read_parquet(p) for p in
          glob.glob("generations/gen13_separation/analysis/stage3/s3_direction_predictions*.parquet")]
t = pd.concat(frames, ignore_index=True)
t = t[t.model != "always_heavy"].drop_duplicates(["design", "split_seed", "fold", "model", "cell_id"])
rows, gains = [], []
for design, blk in t.groupby("design"):
    base = blk[blk.model == "lean_all"][["cell_id", "extractant", "chemotype", "y"]].drop_duplicates("cell_id")
    units = base.groupby(["extractant", "chemotype"]).size().reset_index()[["extractant", "chemotype"]]
    acc = {}
    hb = base.assign(h=(base.y == 1).astype(float)).groupby(["extractant", "chemotype"]).h.mean().reset_index()
    acc["always_heavy"] = units.merge(hb, on=["extractant", "chemotype"]).h.to_numpy()
    for model, b in blk.groupby("model"):
        h = b.assign(h=((b.p >= 0.5).astype(int) == b.y).astype(float)) \
             .groupby(["extractant", "chemotype"]).h.mean().reset_index()
        acc[model] = units.merge(h, on=["extractant", "chemotype"]).h.to_numpy()
    ch = sorted(set(units.chemotype)); mem = [np.flatnonzero(units.chemotype.to_numpy() == c) for c in ch]
    take = [np.concatenate([mem[j] for j in row]) for row in RNG.integers(0, len(ch), size=(REPS, len(ch)))]
    draws = {m: np.array([v[i].mean() for i in take]) for m, v in acc.items()}
    for m, v in acc.items():
        rows.append(dict(design=design, model=m, macro_accuracy=float(v.mean()),
                         ci_low=float(np.quantile(draws[m], .025)), ci_high=float(np.quantile(draws[m], .975)),
                         n_units=len(v)))
    for m in acc:
        if m == "always_heavy":
            continue
        d = draws[m] - draws["always_heavy"]
        gains.append(dict(design=design, model=m, gain=float(acc[m].mean() - acc["always_heavy"].mean()),
                          ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                          p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                          units_better=int((acc[m] > acc["always_heavy"]).sum()),
                          units_worse=int((acc[m] < acc["always_heavy"]).sum())))
A = pd.DataFrame(rows).sort_values(["design", "macro_accuracy"], ascending=[True, False])
G = pd.DataFrame(gains).sort_values(["design", "gain"], ascending=[True, False])
A.to_csv("generations/gen13_separation/analysis/stage3/s3_direction_accuracy_all_designs.csv", index=False)
G.to_csv("generations/gen13_separation/analysis/stage3/s3_direction_gain_all_designs.csv", index=False)
print(A.round(4).to_string(index=False)); print()
print(G[G.model == "donor_geometry"].round(4).to_string(index=False))
