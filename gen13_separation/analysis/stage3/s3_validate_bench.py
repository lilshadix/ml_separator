"""Validate the amplitude bench against a locked arm: it must reproduce M_PHYSICS to 3 decimals."""
import sys, time
sys.path.insert(0, "gen13_separation")
import numpy as np
from gen13sep.amplitude_bench import load_bench, run_candidate, score, ALL_BLOCKS
from gen13sep.models import tree_pipeline

t0 = time.time()
bench = load_bench()
print(f"bench loaded in {time.time()-t0:.0f}s: {len(bench.frame)} cells, "
      f"coef {bench.coef.shape}, blocks {list(bench.frames)}", flush=True)

def physics_trees(Xtr, coef, w, groups, Xte, seed):
    m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
    m.fit(Xtr, coef, extratreesregressor__sample_weight=w)
    return m.predict(Xte)

t0 = time.time()
tab = run_candidate(bench, "BENCH_PHYSICS", physics_trees, blocks=ALL_BLOCKS, design="B")
s = score(tab, ["BENCH_PHYSICS"])
print(f"\n{time.time()-t0:.0f}s for 25 folds")
print(s[["arm","macro_mae_extractant","macro_mae_chemotype","macro_mae_far","macro_sign_acc_strong","pooled_mae"]].round(4).to_string(index=False))
print("\nlocked M_PHYSICS_radius+radius_sq: 0.4979 / 0.5425 / 0.6572 / 0.7440 / 0.4912")
