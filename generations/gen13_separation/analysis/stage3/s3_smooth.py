"""Should a model's prediction vary between two cells of the same extractant at all?

Under the publication-masked design a conditions-only model is worse than the corpus mean curve, and
no arm reproduces more than 4-21 % of the observed within-extractant amplitude spread, with a
correlation between -0.06 and +0.08. So whatever variation an arm does produce between two cells of
one extractant is not the condition effect: it is noise the condition columns inject.

This tests removing it. Within a fold, the predicted coefficients of every held-out cell that shares
an extractant are averaged, wholly or partly. That uses no label and no held-out value, only the
knowledge of which ligand each cell contains, which a deployment always has.
"""
import sys, time
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
from gen13sep.amplitude_bench import (load_bench, cell_weights, _pair_frame, LEAN_BLOCKS, CHEM_BLOCKS)
from gen13sep.inference import paired_contrasts
from gen13sep.metrics import per_extractant, summarise
from gen13sep.models import tree_pipeline
from gen13sep.splits import all_folds

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP", "B", "A"]
LAMBDAS = [0.0, 0.5, 1.0]
bench = load_bench()
X = bench.matrix(LEAN_BLOCKS)
Xc = bench.matrix(CHEM_BLOCKS)

for design in DESIGNS:
    folds = all_folds(bench.frame, design=design)
    parts = []
    t0 = time.time()
    for f in folds:
        tr, te = f.train_index, f.test_index
        pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs.empty:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        m = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(X[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
        coef_te = np.asarray(m.predict(X[te]), dtype=float).reshape(len(te), bench.basis.shape[0])
        ext_te = bench.frame.extractant.to_numpy()[te]
        grp = pd.DataFrame(coef_te).groupby(ext_te).transform("mean").to_numpy()
        mc = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        mc.fit(Xc[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
        coef_chem = np.asarray(mc.predict(Xc[te]), dtype=float).reshape(len(te), bench.basis.shape[0])
        ia = pairs["ia"].to_numpy(); ib = pairs["ib"].to_numpy(); loc = pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        for lam in LAMBDAS:
            curve = ((1 - lam) * coef_te + lam * grp) @ bench.basis
            t[f"SMOOTH{int(lam*100)}"] = curve[loc, ia] - curve[loc, ib]
        curve = coef_chem @ bench.basis
        t["CHEM_ONLY"] = curve[loc, ia] - curve[loc, ib]
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    names = [f"SMOOTH{int(l*100)}" for l in LAMBDAS] + ["CHEM_ONLY"]
    board = summarise(per_extractant(table, names), table, names)
    print("=== " + design + f"  ({time.time()-t0:.0f}s)", flush=True)
    print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype",
                 "macro_mae_far", "macro_sign_acc_strong"]].round(4).to_string(index=False), flush=True)
    board.to_csv("generations/gen13_separation/analysis/stage3/s3_smooth_" + design + ".csv", index=False)
    pe = per_extractant(table, names)
    comps = {design + "|" + n + "_vs_SMOOTH0": ("SMOOTH0", n) for n in names if n != "SMOOTH0"}
    r = paired_contrasts(pe, comps, value="mae_all", replicates=10000)
    print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "seeds_positive",
             "units_improved", "n_units", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False), flush=True)
    r.to_csv("generations/gen13_separation/analysis/stage3/s3_smooth_contrasts_" + design + ".csv", index=False)
