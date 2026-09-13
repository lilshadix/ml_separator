"""Gen15 step 3d: WHICH pair should the laboratory measure first?

The locked stage-2 protocol measures the widest available dZ, which is the obvious heuristic: the
biggest contrast pins the amplitude best.  The estimator-aware alternative is greedy D-optimal
selection under the residual covariance, which maximises the information a measurement carries about
the *whole* curve rather than about one contrast.  A first sweep suggested D-optimal is much better
at k >= 2, but that comparison was not admissible: each strategy removes its own support pairs from
scoring, and removing the widest pair takes the hardest pair out of the test set.  On k = 0, where no
measurement is used at all, the two strategies differed by 0.02 for that reason alone.

Here every strategy is run in one pass and the UNION of their support pairs is excluded from scoring
for all of them, so the comparison is on byte-identical pairs and a difference can only come from
which measurements were taken.

Usage:  python gen15_curve/scripts/g15_support.py [designs] [--cov empirical|blend|smooth]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen15 import arms as A  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
DESIGNS = argv[0].split(",") if argv else list(V.DESIGNS)
COV = "empirical"
for i, a in enumerate(sys.argv):
    if a == "--cov" and i + 1 < len(sys.argv):
        COV = sys.argv[i + 1]

ARMS = {"G14": A.g14, "FLAT": A.flat, "OBOTH": A.o_both}
HOWS = ("widest", "dopt", "random")
KS = (0, 1, 2, 3)

COMPS = {
    "dopt_vs_widest@k1": ("G14@widestk1", "G14@doptk1"),
    "dopt_vs_widest@k2": ("G14@widestk2", "G14@doptk2"),
    "dopt_vs_widest@k3": ("G14@widestk3", "G14@doptk3"),
    "widest_vs_random@k1": ("G14@randomk1", "G14@widestk1"),
    "dopt_vs_random@k1": ("G14@randomk1", "G14@doptk1"),
    "doptk1_vs_k0": ("G14@doptk0", "G14@doptk1"),
    "doptk3_vs_doptk1": ("G14@doptk1", "G14@doptk3"),
    "G14_vs_NAIVE@doptk1": ("NAIVE_LINE@doptk1", "G14@doptk1"),
    "G14_vs_NAIVE@doptk3": ("NAIVE_LINE@doptk3", "G14@doptk3"),
    "G14_vs_FLAT@doptk1": ("FLAT@doptk1", "G14@doptk1"),
    "OBOTH_vs_G14@doptk3": ("G14@doptk3", "OBOTH@doptk3"),
}

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    boards, cons = [], []
    for d in DESIGNS:
        r = FS.evaluate(bench, ARMS, d, ks=KS, hows=HOWS, cov_kind=COV,
                        mask_publication=True, verbose=True)
        if r.pairs.empty:
            print(f"  {d}: no scorable cells")
            continue
        pe = per_extractant(r.pairs, r.modes)
        bd = summarise(pe, r.pairs, r.modes)
        bd.insert(0, "design", d)
        boards.append(bd)
        c = paired_contrasts(pe, COMPS, value="mae_all", replicates=10_000)
        if len(c):
            c.insert(0, "design", d)
            cons.append(c)
        V.RESULTS.mkdir(parents=True, exist_ok=True)
        pe.to_parquet(V.RESULTS / f"g15_support_perext_{d}_{COV}.parquet")
    B = pd.concat(boards, ignore_index=True)
    B.to_csv(V.RESULTS / f"g15_support_board_{COV}.csv", index=False)
    C = pd.concat(cons, ignore_index=True) if cons else pd.DataFrame()
    if len(C):
        C.to_csv(V.RESULTS / f"g15_support_contrasts_{COV}.csv", index=False)
    pd.set_option("display.width", 240)
    order = [f"{a}@{h}k{k}" for a in ("G14", "FLAT", "OBOTH") for h in HOWS for k in KS] + \
            [f"NAIVE_LINE@{h}k{k}" for h in HOWS for k in KS if k]
    w = V.wide(B)
    print(f"\n=== extractant-macro MAE, common scoring set, cov={COV} ===")
    print(w.reindex([m for m in order if m in w.index]).round(4).to_string())
    for col in ("macro_sign_acc_strong", "macro_pair_spearman"):
        ww = V.wide(B, col)
        print(f"\n=== {col} ===")
        print(ww.reindex([m for m in order if m in ww.index]).round(4).to_string())
    if len(C):
        print("\n=== paired contrasts (positive favours the candidate) ===")
        print(C[["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                 "units_improved", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]]
              .round(4).to_string(index=False))
    print(f"\n[g15-support] total {time.time() - t0:.0f}s")
