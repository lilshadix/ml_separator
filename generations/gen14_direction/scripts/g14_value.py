"""Gen14 step 3: what the direction call plus an amplitude prior is worth on the pairwise metric.

The programme's endpoint is the extractant-macro MAE of a predicted log separation factor, not the
accuracy of a bit.  This script assembles a curve for every held-out cell out of two numbers -- a
direction and a magnitude -- and scores it on byte-identical pairs against the corpus mean curve,
against gen13's full 209-column regression and against the oracle direction.

Three decisions are separated so their contributions can be told apart:

* which *direction* model (gen13's trees vs gen14's logistic);
* which *magnitude* (the training fold's mean |amplitude|, gen13's prior, vs a ridge on
  log |amplitude| that lets the magnitude depend on the ligand);
* which *rule* turns a probability and a magnitude into a coefficient -- the hard 0.5 threshold
  gen13 used, or the posterior mean ``(1 - 2p) * magnitude``, which shrinks toward the flat curve
  exactly as far as the classifier is unsure.

Usage:  python generations/gen14_direction/scripts/g14_value.py [design,design,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))
from gen14 import dirbench as db
from gen14 import models as M
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights
from gen13sep.inference import paired_contrasts
from gen13sep.metrics import per_extractant, summarise
from gen13sep.models import tree_pipeline
from gen13sep.splits import all_folds

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP", "B"]
bench = db.load()
FS = db.feature_sets(bench)
amp = bench.coef[:, 0]
rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS

#: (name, direction model, amplitude prior, feature set, decision rule)
ARMS = [
    ("G13_DIR_HARD",     M.dir_extratrees(),          M.amp_constant,       "TOPO39", "hard"),
    ("G14_DIR_HARD",     M.dir_logistic(),            M.amp_constant,       "TOPO39", "hard"),
    ("G14_DIR_EXPECTED", M.dir_logistic(),            M.amp_constant,       "TOPO39", "expected"),
    ("G14_DIR_SQRT",     M.dir_logistic(),            M.amp_constant,       "TOPO39", "sqrt"),
    ("G14_MAGREG_HARD",  M.dir_logistic(),            M.amp_regression(),   "TOPO39", "hard"),
    ("G14_MAGREG_EXP",   M.dir_logistic(),            M.amp_regression(),   "TOPO39", "expected"),
    ("G14_MAGTREE_HARD", M.dir_logistic(),            M.amp_trees(),        "TOPO39", "hard"),
    ("DIR_ORACLE",       None,                        M.amp_constant,       "TOPO39", "hard"),
]

for design in DESIGNS:
    t0 = time.time()
    # 1. out-of-fold direction + magnitude for every candidate, keyed by (seed, fold, cell)
    oof = {}
    for name, dirm, ampm, fs, _ in ARMS:
        if dirm is None:
            continue
        o = db.run(bench, name, M.candidate(dirm, ampm), features=FS[fs], design=design)
        oof[name] = o.cells.set_index(["split_seed", "fold", "cell_index"])[["p", "mag"]]
        print(f"  {design} {name:16s} {o.seconds:5.1f}s", flush=True)

    # 2. one pass over the folds to build the pair table and the two reference arms
    X_lean = bench.matrix(LEAN_BLOCKS)
    parts = []
    for f in all_folds(bench.frame, design=design):
        tr, te = f.train_index, f.test_index
        pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs.empty:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        rtr = tr[rich[tr]]
        wr = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
        bmean = float(np.average(bench.coef[tr, 1], weights=w))
        amean = float(np.average(bench.coef[tr, 0], weights=w))
        mag_const = float(np.average(np.abs(amp[rtr]), weights=wr))
        curves = {}
        for name, dirm, ampm, fs, rule in ARMS:
            if dirm is None:
                a = np.where(amp[te] < 0, -mag_const, mag_const)
            else:
                sub = oof[name].loc[(f.seed, f.fold)].reindex(te)
                a = db.decision(sub["p"].to_numpy(), sub["mag"].to_numpy(), rule)
            curves[name] = np.c_[a, np.full(len(te), bmean)] @ bench.basis
        curves["MEAN_CURVE"] = np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1))
        m = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(X_lean[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
        curves["G13_FULL_MODEL"] = np.asarray(m.predict(X_lean[te]), dtype=float).reshape(len(te), 2) @ bench.basis
        ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        for nm, cv in curves.items():
            t[nm] = cv[loc, ia] - cv[loc, ib]
        parts.append(t)

    table = pd.concat(parts, ignore_index=True)
    names = [a[0] for a in ARMS] + ["MEAN_CURVE", "G13_FULL_MODEL"]
    pe = per_extractant(table, names)
    board = summarise(pe, table, names)
    board.insert(0, "design", design)
    db.RESULTS.mkdir(parents=True, exist_ok=True)
    board.to_csv(db.RESULTS / f"g14_value_{design}.csv", index=False)
    # keep the per-extractant table: every headline p-value is a paired mean over these units,
    # and few-cluster re-inference (gen16_protocol/scripts/g16_dir_signflip.py) needs them.
    pe.to_csv(db.RESULTS / f"g14_value_per_extractant_{design}.csv", index=False)
    print(f"\n=== design {design} ({time.time() - t0:.0f}s) ===")
    print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
                 "macro_mae_chemotype", "macro_mae_far", "macro_sign_acc_strong",
                 "macro_pair_spearman"]].round(4).to_string(index=False))

    comps = {
        "G14hard_vs_MEANCURVE":  ("MEAN_CURVE", "G14_DIR_HARD"),
        "G14hard_vs_G13dir":     ("G13_DIR_HARD", "G14_DIR_HARD"),
        "G14hard_vs_FULL":       ("G13_FULL_MODEL", "G14_DIR_HARD"),
        "G14exp_vs_G14hard":     ("G14_DIR_HARD", "G14_DIR_EXPECTED"),
        "G14exp_vs_FULL":        ("G13_FULL_MODEL", "G14_DIR_EXPECTED"),
        "G14exp_vs_MEANCURVE":   ("MEAN_CURVE", "G14_DIR_EXPECTED"),
        "MAGREGexp_vs_G14exp":   ("G14_DIR_EXPECTED", "G14_MAGREG_EXP"),
        "ORACLE_vs_G14hard":     ("G14_DIR_HARD", "DIR_ORACLE"),
        "ORACLE_vs_FULL":        ("G13_FULL_MODEL", "DIR_ORACLE"),
    }
    r = paired_contrasts(pe, comps, value="mae_all", replicates=10000)
    r.insert(0, "design", design)
    r.to_csv(db.RESULTS / f"g14_value_contrasts_{design}.csv", index=False)
    print()
    print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "units_improved",
             "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False))
