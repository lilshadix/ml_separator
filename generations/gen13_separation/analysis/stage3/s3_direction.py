"""Can the direction of lanthanide selectivity be called from 2D structure for an unseen chemotype?

Practical question behind it: given a candidate ligand and nothing else, can we say whether it will
prefer the heavy or the light end of the series, and how strongly? That is what decides whether a
molecule is worth making, and it is a much easier target than the separation factor itself.

Target: the sign of the cell's radius-coefficient (negative = heavy-selective), on cells whose curve
is well determined (>= 5 metals). Scored out of fold on the frozen plans, per extractant then macro,
under three designs so that a design-specific fluke cannot survive.

Predictors, from the cheapest up:
  * always-heavy       the majority rule, the yardstick any claim must clear
  * donor_pair_min     one number: the shortest donor-to-donor bond path, i.e. the smallest chelate
                       ring the ligand can close with the metal (ring size = path + 2). Threshold
                       chosen inside the training fold.
  * donor geometry     the coord__dist__ / coord__arm__ family
  * full lean set      conditions + physchem + donors + coordination
"""
import sys, time
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from gen13sep.amplitude_bench import load_bench, cell_weights, LEAN_BLOCKS
from gen13sep.splits import all_folds

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP", "B", "A"]
RNG = np.random.default_rng(8675309)
REPS = 10000

bench = load_bench()
lean_cols = bench.columns(LEAN_BLOCKS)
X = bench.matrix(LEAN_BLOCKS)
GEOM = [c for c in lean_cols if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
gi = np.array([lean_cols.index(c) for c in GEOM], dtype=int)
pmin = lean_cols.index("coord__dist__donor_pair_min")

rich = bench.frame.n_metals.to_numpy() >= 5
amp = bench.coef[:, 0]
y = (amp < 0).astype(int)                      # 1 = heavy-selective
print(f"{rich.sum()} well-determined cells; heavy-selective in {y[rich].mean()*100:.0f}% of them, "
      f"{(bench.frame.extractant[rich].nunique())} extractants")


def clf(seed):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         ExtraTreesClassifier(n_estimators=400, max_features=0.5,
                                              min_samples_leaf=2, random_state=seed, n_jobs=-1))


def threshold_rule(v_tr, y_tr, w_tr, v_te):
    """Best split on one descriptor, chosen on the training fold by weighted accuracy."""
    cand = np.unique(v_tr[np.isfinite(v_tr)])
    if len(cand) < 2:
        return np.full(len(v_te), y_tr.mean())
    best_t, best_s, best_dir = cand[0], -1.0, 1
    for t in cand:
        for d in (1, -1):
            pred = ((v_tr <= t) if d == 1 else (v_tr > t)).astype(int)
            s = float(np.average((pred == y_tr), weights=w_tr))
            if s > best_s:
                best_t, best_s, best_dir = t, s, d
    p = ((v_te <= best_t) if best_dir == 1 else (v_te > best_t)).astype(float)
    return p


rows = []
for design in DESIGNS:
    folds = all_folds(bench.frame, design=design)
    for f in folds:
        tr = f.train_index[rich[f.train_index]]
        te = f.test_index[rich[f.test_index]]
        if len(tr) < 40 or len(te) < 2 or len(set(y[tr])) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        preds = {}
        preds["always_heavy"] = np.full(len(te), float(np.average(y[tr], weights=w)))
        vtr = np.nan_to_num(X[tr, pmin], nan=np.nanmedian(X[tr, pmin]))
        vte = np.nan_to_num(X[te, pmin], nan=np.nanmedian(X[tr, pmin]))
        preds["donor_pair_min"] = threshold_rule(vtr, y[tr], w, vte)
        for nm, cols in (("donor_geometry", gi), ("lean_all", np.arange(X.shape[1]))):
            m = clf(f.model_seed)
            m.fit(X[tr][:, cols], y[tr], extratreesclassifier__sample_weight=w)
            preds[nm] = m.predict_proba(X[te][:, cols])[:, 1]
        for nm, p in preds.items():
            for j, ci in enumerate(te):
                rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                             "model": nm, "cell_id": bench.frame.cell_id.iat[ci],
                             "extractant": bench.frame.extractant.iat[ci],
                             "chemotype": bench.frame.chemotype.iat[ci],
                             "y": int(y[ci]), "p": float(p[j]), "amp": float(amp[ci])})
    print(f"  {design} done", flush=True)

t = pd.DataFrame(rows)
t.to_parquet("generations/gen13_separation/analysis/stage3/s3_direction_predictions.parquet", index=False)


def macro_acc(block):
    hit = (block.p >= 0.5).astype(int) == block.y
    return float(hit.groupby([block.split_seed, block.extractant]).mean().groupby(level=0).mean().mean())


out = []
for (design, model), block in t.groupby(["design", "model"]):
    per_ext = (block.assign(hit=((block.p >= 0.5).astype(int) == block.y).astype(float))
               .groupby(["split_seed", "extractant", "chemotype"])["hit"].mean().reset_index())
    unit = per_ext.groupby(["extractant", "chemotype"])["hit"].mean().reset_index()
    ch = sorted(set(unit.chemotype)); mem = [np.flatnonzero(unit.chemotype.to_numpy() == c) for c in ch]
    vals = unit["hit"].to_numpy()
    picks = RNG.integers(0, len(ch), size=(REPS, len(ch)))
    draws = np.array([vals[np.concatenate([mem[j] for j in row])].mean() for row in picks])
    try:
        auc = roc_auc_score(block.y, block.p) if block.y.nunique() > 1 else np.nan
    except Exception:
        auc = np.nan
    out.append({"design": design, "model": model, "n_cells": int(block.cell_id.nunique()),
                "n_extractants": int(unit.shape[0]), "n_chemotypes": len(ch),
                "macro_accuracy": float(vals.mean()),
                "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
                "pooled_accuracy": float(((block.p >= 0.5).astype(int) == block.y).mean()),
                "auc_pooled": float(auc)})
res = pd.DataFrame(out).sort_values(["design", "macro_accuracy"], ascending=[True, False])
res.to_csv("generations/gen13_separation/analysis/stage3/s3_direction_accuracy.csv", index=False)
print()
print(res.round(4).to_string(index=False))
