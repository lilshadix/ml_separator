"""Which donor-topology descriptors carry the direction of selectivity, and does topology also carry
its magnitude?

Part 1: permutation importance of every donor-topology column, measured on the held-out folds under
the publication-masked design, with a chemotype-blocked interval. A column matters if permuting it
across held-out cells costs accuracy.

Part 2: the same features against a three-way magnitude class -- strongly heavy-selective, weak, and
light-selective -- to see whether topology carries how much, not only which way.
"""
import sys, time
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from gen13sep.amplitude_bench import load_bench, cell_weights, LEAN_BLOCKS
from gen13sep.splits import all_folds

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
RNG = np.random.default_rng(8675309); REPS = 4000
bench = load_bench()
lean = bench.columns(LEAN_BLOCKS)
X = bench.matrix(LEAN_BLOCKS)
GEOM = [c for c in lean if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
gi = np.array([lean.index(c) for c in GEOM], dtype=int)
rich = bench.frame.n_metals.to_numpy() >= 5
amp = bench.coef[:, 0]
y_dir = (amp < 0).astype(int)
thr = np.quantile(amp[rich], [1/3, 2/3])
y_mag = np.digitize(amp, thr)                     # 0 = most heavy-selective ... 2 = light
print(f"{DESIGN}: {rich.sum()} well-determined cells; magnitude class thresholds {thr.round(3)}")


def clf(seed, n=400):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         ExtraTreesClassifier(n_estimators=n, max_features=0.5, min_samples_leaf=2,
                                              random_state=seed, n_jobs=-1))


def run(y, cols, permute=None, seed_offset=0):
    """Out-of-fold hits; ``permute`` names a column index to shuffle in the held-out block."""
    recs = []
    for f in all_folds(bench.frame, design=DESIGN):
        tr = f.train_index[rich[f.train_index]]; te = f.test_index[rich[f.test_index]]
        if len(tr) < 40 or len(te) < 2 or len(set(y[tr])) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        m = clf(f.model_seed + seed_offset)
        m.fit(X[tr][:, cols], y[tr], extratreesclassifier__sample_weight=w)
        Xte = X[te][:, cols].copy()
        if permute is not None:
            j = list(cols).index(permute)
            rng = np.random.default_rng(f.model_seed * 7 + permute)
            Xte[:, j] = Xte[rng.permutation(len(Xte)), j]
        pred = m.predict(Xte)
        for i, ci in enumerate(te):
            recs.append({"extractant": bench.frame.extractant.iat[ci],
                         "chemotype": bench.frame.chemotype.iat[ci],
                         "hit": float(pred[i] == y[ci])})
    return pd.DataFrame(recs)


def macro_and_draws(df, picks_cache={}):
    u = df.groupby(["extractant", "chemotype"]).hit.mean().reset_index()
    ch = sorted(set(u.chemotype)); key = tuple(ch)
    if key not in picks_cache:
        mem = [np.flatnonzero(u.chemotype.to_numpy() == c) for c in ch]
        picks = RNG.integers(0, len(ch), size=(REPS, len(ch)))
        picks_cache[key] = [np.concatenate([mem[j] for j in row]) for row in picks]
    v = u.hit.to_numpy()
    return float(v.mean()), np.array([v[i].mean() for i in picks_cache[key]]), u


t0 = time.time()
base = run(y_dir, gi)
base_m, base_d, base_u = macro_and_draws(base)
print(f"baseline donor-topology accuracy {base_m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

rows = []
for c in GEOM:
    j = lean.index(c)
    df = run(y_dir, gi, permute=j)
    m, d, u = macro_and_draws(df)
    drop = base_d - d
    rows.append({"feature": c, "accuracy_permuted": m, "importance": base_m - m,
                 "ci_low": float(np.quantile(drop, .025)), "ci_high": float(np.quantile(drop, .975)),
                 "p_two_sided": float(2 * min((drop <= 0).mean(), (drop >= 0).mean()))})
    print(f"  {c[:52]:52s} drop {base_m - m:+.4f}", flush=True)
imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
imp.to_csv(f"generations/gen13_separation/analysis/stage3/s3_permutation_importance_{DESIGN}.csv", index=False)
print("\ntop 10 by permutation importance:")
print(imp.head(10).round(4).to_string(index=False))

print("\n=== three-way magnitude class ===")
for name, cols in (("donor_topology", gi), ("full_lean", np.arange(X.shape[1]))):
    df = run(y_mag, cols, seed_offset=11)
    m, d, u = macro_and_draws(df)
    print(f"  {name:16s} macro accuracy {m:.4f}  CI [{np.quantile(d,.025):.4f}, {np.quantile(d,.975):.4f}]")
maj = pd.DataFrame({"extractant": bench.frame.extractant[rich], "chemotype": bench.frame.chemotype[rich],
                    "hit": (y_mag[rich] == np.bincount(y_mag[rich]).argmax()).astype(float)})
m, d, _ = macro_and_draws(maj)
print(f"  {'majority class':16s} macro accuracy {m:.4f}  CI [{np.quantile(d,.025):.4f}, {np.quantile(d,.975):.4f}]")
