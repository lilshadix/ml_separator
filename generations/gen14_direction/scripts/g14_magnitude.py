"""Gen14 step 4: the amplitude prior, fitted to the loss it is judged by.

Step 3 left the direction call 0.014 MAE short of the oracle direction, so the remaining error is
in the *magnitude*.  Gen13's prior was the training fold's mean |amplitude| -- a mean, under a
metric that is an absolute error, on a distribution whose median is 0.18 and whose mean is pulled
out by a long tail.  Two things are wrong with it and both are fixable without new data:

1. **Wrong statistic.**  Under MAE the loss-minimising constant is a median-like quantity, not a
   mean.  ``MAG_LOSS_*`` chooses the magnitude that minimises the *training fold's own* pairwise
   MAE by a one-dimensional scan -- the same loss the arm will be scored on, fitted where it is
   allowed to be fitted.
2. **Wrong conditioning.**  The magnitude is applied behind a direction call that is wrong for
   about one extractant in six.  A magnitude fitted against the *oracle* direction is therefore too
   large: every wrong call pays twice.  ``MAG_LOSS_INNER`` fits the scalar against directions
   predicted by an inner chemotype-grouped CV inside the training fold, so the prior knows the
   error rate of the classifier it sits behind.

Upper bounds are run alongside: the true magnitude behind the predicted direction, and the true
coefficient (stage 2's "perfect amplitude" bound).

Usage:  python generations/gen14_direction/scripts/g14_magnitude.py [design,design,...]
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

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(db.DESIGNS)
GRID = np.round(np.arange(0.0, 1.501, 0.005), 3)
N_INNER = 4

bench = db.load()
FS = db.feature_sets(bench)
AMP = bench.coef[:, 0]
RICH = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
BASIS = bench.basis
TOPO = FS["TOPO39"]
X = bench.matrix(LEAN_BLOCKS)


def cell_pairs(index: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per cell: (observed log SF, radius-basis contrast, curvature-basis contrast) over its pairs."""
    out = []
    for ci in index:
        obs = np.flatnonzero(~np.isnan(bench.Y[ci]))
        a, b = np.triu_indices(len(obs), k=1)
        ia, ib = obs[a], obs[b]
        out.append((bench.Y[ci][ia] - bench.Y[ci][ib],
                    BASIS[0][ia] - BASIS[0][ib], BASIS[1][ia] - BASIS[1][ib]))
    return out


def fit_magnitude(pairs, signs: np.ndarray, w: np.ndarray, bmean: float) -> float:
    """The magnitude minimising the weighted per-cell mean absolute pair error on this fold."""
    cost = np.zeros(len(GRID))
    tot = 0.0
    for (y, dr, dq), s, wi in zip(pairs, signs, w):
        if len(y) == 0:
            continue
        resid = y[None, :] - (s * GRID[:, None] * dr[None, :] + bmean * dq[None, :])
        cost += wi * np.abs(resid).mean(axis=1)
        tot += wi
    return float(GRID[int(np.argmin(cost))]) if tot > 0 else 0.0


def inner_directions(tr: np.ndarray, seed: int) -> np.ndarray:
    """Honest in-fold direction calls: chemotype-grouped inner CV with the gen14 classifier."""
    groups = bench.groups[tr]
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    block = np.array([{g: i % N_INNER for i, g in enumerate(rng.permutation(uniq))}[g] for g in groups])
    p = np.full(len(tr), np.nan)
    model = M.dir_logistic()
    for k in range(N_INNER):
        itr = np.flatnonzero((block != k) & RICH[tr])
        ite = np.flatnonzero(block == k)
        if len(itr) < 20 or len(ite) == 0 or len(set(AMP[tr][itr] < 0)) < 2:
            continue
        p[ite] = model(X[tr][itr][:, TOPO], AMP[tr][itr],
                       cell_weights(groups[itr], bench.n_obs[tr][itr]), groups[itr],
                       X[tr][ite][:, TOPO], seed)
    p = np.where(np.isnan(p), 0.5, p)
    return np.where(p >= 0.5, -1.0, 1.0)


for design in DESIGNS:
    t0 = time.time()
    parts = []
    diag = []
    for f in all_folds(bench.frame, design=design):
        tr, te = f.train_index, f.test_index
        pairs_te = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs_te.empty:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        rtr = tr[RICH[tr]]
        wr = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
        bmean = float(np.average(bench.coef[tr, 1], weights=w))
        amean = float(np.average(bench.coef[tr, 0], weights=w))

        # --- direction: gen14's logistic on the 39 topology columns, trained on rich cells
        p_te = M.dir_logistic()(X[rtr][:, TOPO], AMP[rtr], wr, bench.groups[rtr],
                                X[te][:, TOPO], f.model_seed)
        s_te = np.where(np.asarray(p_te) >= 0.5, -1.0, 1.0)
        p_g13 = M.dir_extratrees()(X[rtr][:, TOPO], AMP[rtr], wr, bench.groups[rtr],
                                   X[te][:, TOPO], f.model_seed)
        s_g13 = np.where(np.asarray(p_g13) >= 0.5, -1.0, 1.0)

        # --- magnitude priors
        tr_pairs = cell_pairs(tr)
        s_oracle_tr = np.where(AMP[tr] < 0, -1.0, 1.0)
        s_inner_tr = inner_directions(tr, f.model_seed)
        mag_mean = float(np.average(np.abs(AMP[rtr]), weights=wr))
        mag_median = float(np.median(np.abs(AMP[rtr])))
        mag_loss_oracle = fit_magnitude(tr_pairs, s_oracle_tr, w, bmean)
        mag_loss_inner = fit_magnitude(tr_pairs, s_inner_tr, w, bmean)
        heavy = s_inner_tr < 0
        mag_h = fit_magnitude([q for q, m in zip(tr_pairs, heavy) if m], s_inner_tr[heavy],
                              w[heavy], bmean) if heavy.any() else mag_loss_inner
        mag_l = fit_magnitude([q for q, m in zip(tr_pairs, ~heavy) if m], s_inner_tr[~heavy],
                              w[~heavy], bmean) if (~heavy).any() else mag_loss_inner
        diag.append({"design": design, "seed": f.seed, "fold": f.fold, "mag_mean": mag_mean,
                     "mag_median": mag_median, "mag_loss_oracle": mag_loss_oracle,
                     "mag_loss_inner": mag_loss_inner, "mag_heavy": mag_h, "mag_light": mag_l,
                     "bmean": bmean, "inner_dir_acc": float(np.mean(s_inner_tr == s_oracle_tr))})

        mag_tree = np.asarray(M.amp_trees()(X[rtr][:, TOPO], AMP[rtr], wr, bench.groups[rtr],
                                            X[te][:, TOPO], f.model_seed), dtype=float)
        coefs = {
            "G13_DIR_MEAN":     s_g13 * mag_mean,
            "G14_DIR_MEAN":     s_te * mag_mean,
            "G14_MAG_MEDIAN":   s_te * mag_median,
            "G14_MAG_LOSS_ORA": s_te * mag_loss_oracle,
            "G14_MAG_LOSS":     s_te * mag_loss_inner,
            "G14_MAG_BYDIR":    s_te * np.where(s_te < 0, mag_h, mag_l),
            "G14_MAG_TREE":     s_te * mag_tree,
            "ORACLE_DIR_LOSS":  np.where(AMP[te] < 0, -1.0, 1.0) * mag_loss_inner,
            "ORACLE_MAG":       s_te * np.abs(AMP[te]),
            "ORACLE_BOTH":      AMP[te],
        }
        curves = {k: np.c_[v, np.full(len(te), bmean)] @ BASIS for k, v in coefs.items()}
        curves["MEAN_CURVE"] = np.tile(np.array([amean, bmean]) @ BASIS, (len(te), 1))
        m = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(X[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
        curves["G13_FULL_MODEL"] = np.asarray(m.predict(X[te]), dtype=float).reshape(len(te), 2) @ BASIS
        ia, ib, loc = pairs_te["ia"].to_numpy(), pairs_te["ib"].to_numpy(), pairs_te["cell_local"].to_numpy()
        t = pairs_te.drop(columns=["ia", "ib", "cell_local"]).copy()
        for nm, cv in curves.items():
            t[nm] = cv[loc, ia] - cv[loc, ib]
        parts.append(t)
        print(f"  {design} seed {f.seed} fold {f.fold}  mean {mag_mean:.3f} median {mag_median:.3f} "
              f"loss(ora) {mag_loss_oracle:.3f} loss(inner) {mag_loss_inner:.3f} "
              f"h/l {mag_h:.3f}/{mag_l:.3f}", flush=True)

    table = pd.concat(parts, ignore_index=True)
    names = [c for c in table.columns if c.isupper() and c not in
             {"A", "B"}] if False else list(curves.keys())
    pe = per_extractant(table, names)
    board = summarise(pe, table, names)
    board.insert(0, "design", design)
    db.RESULTS.mkdir(parents=True, exist_ok=True)
    board.to_csv(db.RESULTS / f"g14_magnitude_{design}.csv", index=False)
    pd.DataFrame(diag).to_csv(db.RESULTS / f"g14_magnitude_diag_{design}.csv", index=False)
    print(f"\n=== design {design} ({time.time() - t0:.0f}s) ===")
    print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
                 "macro_mae_chemotype", "macro_mae_adjacent", "macro_mae_far",
                 "macro_sign_acc_strong", "macro_pair_spearman"]].round(4).to_string(index=False))

    comps = {
        "LOSS_vs_MEAN":        ("G14_DIR_MEAN", "G14_MAG_LOSS"),
        "LOSS_vs_MEDIAN":      ("G14_MAG_MEDIAN", "G14_MAG_LOSS"),
        "LOSS_vs_LOSSORACLE":  ("G14_MAG_LOSS_ORA", "G14_MAG_LOSS"),
        "BYDIR_vs_LOSS":       ("G14_MAG_LOSS", "G14_MAG_BYDIR"),
        "LOSS_vs_G13DIR":      ("G13_DIR_MEAN", "G14_MAG_LOSS"),
        "LOSS_vs_FULL":        ("G13_FULL_MODEL", "G14_MAG_LOSS"),
        "LOSS_vs_MEANCURVE":   ("MEAN_CURVE", "G14_MAG_LOSS"),
        "ORACLEDIR_vs_LOSS":   ("G14_MAG_LOSS", "ORACLE_DIR_LOSS"),
        "ORACLEMAG_vs_LOSS":   ("G14_MAG_LOSS", "ORACLE_MAG"),
    }
    r = paired_contrasts(pe, comps, value="mae_all", replicates=10000)
    r.insert(0, "design", design)
    r.to_csv(db.RESULTS / f"g14_magnitude_contrasts_{design}.csv", index=False)
    print()
    print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "units_improved",
             "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False))
