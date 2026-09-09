"""Test one hypothesis-driven change across every available design: predict the radius amplitude
only and take the curvature from the training fold.

The reason is not a search result. Stage 2 measured that 82-87 % of the pairwise squared error is
amplitude error and that curvature has an extractant ICC of 0.19, i.e. it is mostly noise; a model
that spends half its output on curvature is fitting noise and paying variance for it. The prediction
was made before the arm was run.

Designs: B (primary, comparable with gen2-gen12), BP (publication-masked, the honest one), BR and BQ
(the size-matched controls) and A (exact-extractant hold-out). A change that is real should move the
same way in all of them.
"""
import sys, time
sys.path.insert(0, "gen13_separation")
import numpy as np, pandas as pd
from gen13sep.amplitude_bench import load_bench, compare, LEAN_BLOCKS
from gen13sep.inference import paired_contrasts
from gen13sep.metrics import per_extractant
from gen13sep.models import tree_pipeline

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["B", "BP", "BR", "BQ", "A"]
bench = load_bench()


def both_coefficients(Xtr, coef, w, g, Xte, seed):
    m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
    m.fit(Xtr, coef, extratreesregressor__sample_weight=w)
    return m.predict(Xte)


def amplitude_only(curvature="mean"):
    def f(Xtr, coef, w, g, Xte, seed):
        m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(Xtr, coef[:, 0], extratreesregressor__sample_weight=w)
        a = np.asarray(m.predict(Xte), dtype=float)
        if curvature == "zero":
            b = np.zeros_like(a)
        elif curvature == "mean":
            b = np.full_like(a, float(np.average(coef[:, 1], weights=w)))
        else:
            denom = float(np.average(coef[:, 0] ** 2, weights=w))
            kappa = float(np.average(coef[:, 0] * coef[:, 1], weights=w)) / max(denom, 1e-9)
            b = kappa * a
        return np.c_[a, b]
    return f


CAND = {"BOTH_COEF": both_coefficients,
        "AMP_ONLY_meanb": amplitude_only("mean"),
        "AMP_ONLY_kappab": amplitude_only("kappa"),
        "AMP_ONLY_zerob": amplitude_only("zero")}

boards, contrasts = [], []
for design in DESIGNS:
    t0 = time.time()
    board, table = compare(bench, CAND, blocks=LEAN_BLOCKS, design=design)
    board.insert(0, "design", design)
    boards.append(board)
    pe = per_extractant(table, list(CAND))
    comps = {f"{design}|{c}_vs_BOTH": ("BOTH_COEF", c) for c in CAND if c != "BOTH_COEF"}
    for val in ("mae_all", "mae_far", "sign_acc_strong"):
        r = paired_contrasts(pe, comps, value=val, replicates=10000)
        if len(r):
            r.insert(0, "design", design)
            contrasts.append(r)
    print(f"--- {design} ({time.time()-t0:.0f}s)")
    print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
                 "macro_mae_chemotype", "macro_mae_far", "macro_sign_acc_strong"]].round(4).to_string(index=False),
          flush=True)

B = pd.concat(boards, ignore_index=True); B.to_csv("gen13_separation/analysis/stage3/s3_amponly_boards.csv", index=False)
C = pd.concat(contrasts, ignore_index=True); C.to_csv("gen13_separation/analysis/stage3/s3_amponly_contrasts.csv", index=False)
print("\n=== contrasts, mae_all (positive favours the amplitude-only arm) ===")
cols = ["design", "comparison", "point", "ci95_low", "ci95_high", "bca_low", "p_two_sided",
        "seeds_positive", "units_improved", "n_units", "loco_sign_stable", "passes_intervals", "passes_P1"]
print(C[C.value == "mae_all"][cols].round(4).to_string(index=False))
print("\n=== sign accuracy on strong pairs ===")
print(C[C.value == "sign_acc_strong"][["design", "comparison", "point", "ci95_low", "ci95_high",
                                       "p_two_sided", "seeds_positive"]].round(4).to_string(index=False))
