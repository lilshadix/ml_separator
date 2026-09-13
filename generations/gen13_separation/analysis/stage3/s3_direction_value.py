"""What is a correct direction call worth on the programme's own metric?

Three predictors of a held-out cell's curve, all built from the training fold's mean amplitude
magnitude and a direction:
  * DIR_TRAIN_MAJORITY  the direction every cell is given by the training majority
  * DIR_PREDICTED       the direction called out of fold from donor topology
  * DIR_ORACLE          the true direction (an upper bound, not achievable)
against the corpus mean curve and a full model, on byte-identical pairs.
"""
import sys, time
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from gen13sep.amplitude_bench import (load_bench, cell_weights, _pair_frame, LEAN_BLOCKS)
from gen13sep.inference import paired_contrasts
from gen13sep.metrics import per_extractant, summarise
from gen13sep.models import tree_pipeline
from gen13sep.splits import all_folds

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
bench = load_bench()
lean = bench.columns(LEAN_BLOCKS); X = bench.matrix(LEAN_BLOCKS)
gi = np.array([lean.index(c) for c in lean
               if c.startswith("coord__dist__") or c.startswith("coord__arm__")], dtype=int)
amp = bench.coef[:, 0]
y = (amp < 0).astype(int)
parts = []
t0 = time.time()
for f in all_folds(bench.frame, design=DESIGN):
    tr, te = f.train_index, f.test_index
    pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
    if pairs.empty:
        continue
    w = cell_weights(bench.groups[tr], bench.n_obs[tr])
    mag = float(np.average(np.abs(amp[tr]), weights=w))
    maj = 1 if float(np.average(y[tr], weights=w)) >= 0.5 else 0
    c = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                      ExtraTreesClassifier(n_estimators=400, max_features=0.5, min_samples_leaf=2,
                                           random_state=f.model_seed, n_jobs=-1))
    c.fit(X[tr][:, gi], y[tr], extratreesclassifier__sample_weight=w)
    p_dir = (c.predict_proba(X[te][:, gi])[:, 1] >= 0.5).astype(int)
    m = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
    m.fit(X[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
    full = np.asarray(m.predict(X[te]), dtype=float).reshape(len(te), 2)
    bmean = float(np.average(bench.coef[tr, 1], weights=w))
    amean = float(np.average(bench.coef[tr, 0], weights=w))
    def curve_from(sign_vec):
        a = np.where(sign_vec == 1, -mag, mag)
        return np.c_[a, np.full(len(a), bmean)] @ bench.basis
    curves = {"DIR_TRAIN_MAJORITY": curve_from(np.full(len(te), maj)),
              "DIR_PREDICTED": curve_from(p_dir),
              "DIR_ORACLE": curve_from(y[te]),
              "MEAN_CURVE": np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1)),
              "FULL_MODEL": full @ bench.basis}
    ia = pairs["ia"].to_numpy(); ib = pairs["ib"].to_numpy(); loc = pairs["cell_local"].to_numpy()
    t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
    for nm, cv in curves.items():
        t[nm] = cv[loc, ia] - cv[loc, ib]
    parts.append(t)
table = pd.concat(parts, ignore_index=True)
names = ["DIR_TRAIN_MAJORITY", "DIR_PREDICTED", "DIR_ORACLE", "MEAN_CURVE", "FULL_MODEL"]
board = summarise(per_extractant(table, names), table, names)
print(f"design {DESIGN}  ({time.time()-t0:.0f}s)\n")
print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype",
             "macro_mae_far", "macro_sign_acc_strong"]].round(4).to_string(index=False))
board.to_csv(f"generations/gen13_separation/analysis/stage3/s3_direction_value_{DESIGN}.csv", index=False)
pe = per_extractant(table, names)
comps = {"PREDICTED_vs_MAJORITY": ("DIR_TRAIN_MAJORITY", "DIR_PREDICTED"),
         "PREDICTED_vs_MEANCURVE": ("MEAN_CURVE", "DIR_PREDICTED"),
         "ORACLE_vs_PREDICTED": ("DIR_PREDICTED", "DIR_ORACLE"),
         "FULL_vs_PREDICTED": ("DIR_PREDICTED", "FULL_MODEL"),
         "ORACLE_vs_FULL": ("FULL_MODEL", "DIR_ORACLE")}
r = paired_contrasts(pe, comps, value="mae_all", replicates=10000)
print()
print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "seeds_positive",
         "units_improved", "n_units", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False))
r.to_csv(f"generations/gen13_separation/analysis/stage3/s3_direction_value_contrasts_{DESIGN}.csv", index=False)
