"""Is the model's amplitude shrinkage inherited from its own training targets?

A cell's coefficients come from a ridge fit on the metals that cell measured:
    coef_i = (B_obs B_obs' + lam I)^-1 B_obs y_i
so the estimate is attenuated by A_i = (B_obs B_obs' + lam I)^-1 (B_obs B_obs') relative to the
unpenalised one. A cell measured on 13 metals has A_i almost the identity; a two-metal cell spanning
a narrow slice of the radius axis has A_i far below it. A third of the corpus is in the second
category, so a third of the training targets are systematically pulled toward zero -- which is the
same direction as the 1.8-2.4x too-narrow amplitude the models predict.

Three fixes are tested against the untouched target, all on the frozen folds:
  * de-shrink   : divide each training target by its own attenuation (unbiased, noisier)
  * precision   : keep the target, weight cells by how well determined they are
  * well-only   : train only on cells whose attenuation exceeds a threshold
"""
import sys, time
sys.path.insert(0, "gen13_separation")
import numpy as np, pandas as pd
from gen13sep.amplitude_bench import load_bench, compare, LEAN_BLOCKS
from gen13sep.basis import DEFAULT_RIDGE
from gen13sep.inference import paired_contrasts
from gen13sep.metrics import per_extractant
from gen13sep.models import tree_pipeline

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP","B"]
bench = load_bench()
B = bench.basis
lam = DEFAULT_RIDGE
k = B.shape[0]

att = np.zeros((len(bench.Y), k, k))
for i in range(len(bench.Y)):
    m = ~np.isnan(bench.Y[i])
    Bo = B[:, m]
    G = Bo @ Bo.T
    att[i] = np.linalg.solve(G + lam * np.eye(k), G)
a11 = att[:, 0, 0]
print(f"attenuation of the radius coefficient: median {np.median(a11):.3f}, "
      f"1st pct {np.quantile(a11,0.01):.3f}, min {a11.min():.3f}")
for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
    print(f"  cells with attenuation >= {thr}: {(a11>=thr).sum():3d} of {len(a11)}")

deshrunk = np.zeros_like(bench.coef)
for i in range(len(bench.coef)):
    try:
        deshrunk[i] = np.linalg.solve(att[i], bench.coef[i])
    except np.linalg.LinAlgError:
        deshrunk[i] = bench.coef[i]
deshrunk = np.clip(deshrunk, -4.0, 4.0)          # a two-metal cell can invert to nonsense
print(f"\nde-shrunk amplitude: sd {deshrunk[:,0].std():.3f} against {bench.coef[:,0].std():.3f} "
      f"for the ridge target; clipped at |4|: {(np.abs(deshrunk[:,0])>=4).sum()} cells")

IDX = {"a11": a11, "deshrunk": deshrunk}


def base(Xtr, coef, w, g, Xte, seed):
    m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
    m.fit(Xtr, coef[:, 0], extratreesregressor__sample_weight=w)
    a = np.asarray(m.predict(Xte), dtype=float)
    return np.c_[a, np.full_like(a, float(np.average(coef[:, 1], weights=w)))]


def make(kind, thr=0.8):
    """The bench hands over training rows in fold order, so the per-cell auxiliaries are matched by
    a nearest-row lookup on the untouched coefficient matrix (exact: coefficients are unique)."""
    lookup = {tuple(np.round(c, 12)): i for i, c in enumerate(bench.coef)}

    def rows_of(coef):
        return np.array([lookup[tuple(np.round(c, 12))] for c in coef], dtype=int)

    def f(Xtr, coef, w, g, Xte, seed):
        idx = rows_of(coef)
        if kind == "deshrink":
            target = deshrunk[idx]
        else:
            target = coef
        ww = w.copy()
        if kind == "precision":
            ww = w * a11[idx]
            ww = ww * len(ww) / ww.sum()
        keep = np.ones(len(coef), dtype=bool)
        if kind == "wellonly":
            keep = a11[idx] >= thr
            if keep.sum() < 30:
                keep = np.ones(len(coef), dtype=bool)
        m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(Xtr[keep], target[keep, 0], extratreesregressor__sample_weight=ww[keep])
        a = np.asarray(m.predict(Xte), dtype=float)
        b = float(np.average(coef[:, 1], weights=w))
        return np.c_[a, np.full_like(a, b)]
    return f


CAND = {"AMP_BASE": base, "AMP_DESHRINK": make("deshrink"),
        "AMP_PRECISION": make("precision"), "AMP_WELLONLY80": make("wellonly", 0.8),
        "AMP_WELLONLY95": make("wellonly", 0.95)}


for DESIGN in DESIGNS:
    t0 = time.time()
    board, table = compare(bench, CAND, blocks=LEAN_BLOCKS, design=DESIGN)
    print("=== " + DESIGN + f"  ({time.time()-t0:.0f}s)", flush=True)
    print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype",
                 "macro_mae_far", "macro_sign_acc_strong"]].round(4).to_string(index=False), flush=True)
    board.to_csv("gen13_separation/analysis/stage3/s3_attenuation_" + DESIGN + ".csv", index=False)
    pe = per_extractant(table, list(CAND))
    comps = {DESIGN + "|" + c + "_vs_BASE": ("AMP_BASE", c) for c in CAND if c != "AMP_BASE"}
    r = paired_contrasts(pe, comps, value="mae_all", replicates=10000)
    print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "seeds_positive",
             "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False), flush=True)
    r.to_csv("gen13_separation/analysis/stage3/s3_attenuation_contrasts_" + DESIGN + ".csv", index=False)
