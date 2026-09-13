"""How much of the remaining error is a shortage of training chemotypes?

For every fold of the publication-masked design the training set is thinned to a fraction of its
chemotypes and the model refitted; the held-out cells are unchanged, so the only thing that varies
is how many chemically independent units the model saw.  The slope of the resulting curve at the
full corpus says whether collecting more chemotypes would buy anything, and how much.
"""
import sys, time
sys.path.insert(0, "gen13_separation")
import numpy as np, pandas as pd
from gen13sep.amplitude_bench import (load_bench, cell_weights, _pair_frame, LEAN_BLOCKS)
from gen13sep.metrics import per_extractant, summarise
from gen13sep.models import tree_pipeline
from gen13sep.splits import all_folds

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
FRACTIONS = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
DRAWS = 3

bench = load_bench()
X = bench.matrix(LEAN_BLOCKS)
folds = all_folds(bench.frame, design=DESIGN)
print(f"design {DESIGN}, {len(folds)} folds, {X.shape[1]} lean columns", flush=True)

rows = []
t0 = time.time()
for f in folds:
    tr, te = f.train_index, f.test_index
    pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
    if pairs.empty:
        continue
    ia = pairs["ia"].to_numpy(); ib = pairs["ib"].to_numpy(); loc = pairs["cell_local"].to_numpy()
    chemos = np.array(sorted(set(bench.groups[tr])))
    for frac in FRACTIONS:
        k = max(2, int(round(frac * len(chemos))))
        n_draws = 1 if k == len(chemos) else DRAWS
        for d in range(n_draws):
            rng = np.random.default_rng(f.model_seed * 131 + int(frac * 1000) * 17 + d)
            keep = set(rng.choice(chemos, size=k, replace=False).tolist())
            mask = np.array([g in keep for g in bench.groups[tr]])
            sub = tr[mask]
            if len(sub) < 20:
                continue
            w = cell_weights(bench.groups[sub], bench.n_obs[sub])
            m = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
            m.fit(X[sub], bench.coef[sub], extratreesregressor__sample_weight=w)
            curve = np.asarray(m.predict(X[te])).reshape(len(te), bench.basis.shape[0]) @ bench.basis
            t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
            t["prediction"] = curve[loc, ia] - curve[loc, ib]
            t["fraction"] = frac; t["draw"] = d
            t["n_chemotypes"] = k; t["n_cells"] = len(sub)
            rows.append(t)
    print(f"  seed {f.seed} fold {f.fold} done ({time.time()-t0:.0f}s)", flush=True)

table = pd.concat(rows, ignore_index=True)
table.to_parquet(f"gen13_separation/analysis/stage3/s3_learning_curve_{DESIGN}_pairs.parquet", index=False)

out = []
for (frac, draw), block in table.groupby(["fraction", "draw"]):
    pe = per_extractant(block, ["prediction"])
    s = summarise(pe, block, ["prediction"])
    out.append({"fraction": frac, "draw": draw,
                "n_chemotypes": float(block["n_chemotypes"].mean()),
                "n_cells": float(block["n_cells"].mean()),
                "macro_mae": float(s["macro_mae_extractant"].iat[0]),
                "macro_mae_chemotype": float(s["macro_mae_chemotype"].iat[0]),
                "macro_mae_far": float(s["macro_mae_far"].iat[0])})
curve = pd.DataFrame(out)
agg = curve.groupby("fraction").agg(n_chemotypes=("n_chemotypes", "mean"),
                                    n_cells=("n_cells", "mean"),
                                    macro_mae=("macro_mae", "mean"),
                                    sd=("macro_mae", "std"),
                                    macro_mae_far=("macro_mae_far", "mean")).reset_index()
agg.to_csv(f"gen13_separation/analysis/stage3/s3_learning_curve_{DESIGN}.csv", index=False)
print()
print(agg.round(4).to_string(index=False))

# power-law extrapolation MAE = c + a * n^-b on the mean curve
from scipy.optimize import curve_fit
x = agg["n_chemotypes"].to_numpy(); y = agg["macro_mae"].to_numpy()
try:
    p, _ = curve_fit(lambda n, c, a, b: c + a * n ** (-b), x, y,
                     p0=[y.min() * 0.8, 1.0, 0.5], maxfev=20000)
    c, a, b = p
    print(f"\npower law MAE = {c:.3f} + {a:.3f} * n^-{b:.3f}")
    for n in (45, 60, 90, 135, 200, 400):
        print(f"  {n:4d} chemotypes -> {c + a * n ** (-b):.3f}")
    print(f"  asymptote        -> {c:.3f}")
except Exception as e:
    print("fit failed:", e)
