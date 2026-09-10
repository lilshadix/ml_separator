"""Fourth stage of the lens-B refutation of CLAIM L3A: the cheapest competitor that is not
degenerate.

`PRE_REGISTRATION.md` L3a names "always heavier" as L3a's cheapest competitor; that rule saves
0.000000 by algebra, not by measurement, because a call that is constant across a candidate set
cannot partition it.  `START_HERE.md` section 1 item 5 -- the validity contract the
pre-registration inherits in its section 0 -- names, for **direction**, the 13-column gen6 donor
census.  This runs the deployed arm with its direction call taken from the cheapest blocks
(DONORS13, BITE6) instead of TOPO39: the same estimator, the same folds, the same seeds, one
argument changed.  No new classifier is written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import bootstrap                      # noqa: E402
from gen16 import l3_decision as L               # noqa: E402
from gen14.dirbench import load                  # noqa: E402
from gen15 import arms as A                      # noqa: E402
from gen15 import valuebench as V                # noqa: E402
from refute_l3a_B import OUT, block_boot         # noqa: E402


def dir_arm(features: str):
    def arm(ctx):
        return A._curve(A._logistic_sign(ctx, features) * ctx.train_mean_magnitude(),
                        ctx.train_mean_curvature(), len(ctx.test))
    return arm


def main() -> int:
    t0 = time.time()
    bench = load()
    arms = {"G14": A.g14, "DIR_DONORS13": dir_arm("DONORS13"), "DIR_BITE6": dir_arm("BITE6"),
            "DIR_MASSACT8": dir_arm("MASSACT8")}
    rows = []
    for d in L.DESIGNS:
        tab = V.run_arms(bench, arms, d, verbose=False)
        names = [a for a in arms]
        T = L.build_tasks(tab, d, names)
        one = np.ones((1, len(T.ext)))
        W = block_boot(np.asarray(T.chem_of))
        for a in names:
            calls = [L.calls_of(p) for p in T.pred[a]]
            mats = L.l3a_matrices(T, calls)
            pt = L.l3a_saved_weighted(one, mats)
            sv = float(np.atleast_1d(L.seed_macro(pt["saved"], T.seeds))[0])
            er = float(np.atleast_1d(L.seed_macro(pt["e_random"], T.seeds))[0])
            dr = L.seed_macro(L.l3a_saved_weighted(W, mats)["saved"], T.seeds)
            perm = L.seed_macro(L.l3a_permutation(T, calls, seed=20260911)["saved"], T.seeds)
            rows.append(dict(design=d, arm=a, saved=sv, e_random=er, saved_frac=sv / er,
                             ci_low=L.pct(dr, 0.025), ci_high=L.pct(dr, 0.975),
                             p_block=L.two_sided_p(dr), p_perm=float((perm >= sv).mean()),
                             n_tasks=int(T.n)))
        print(f"[{d}] competitor arms done t={time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    # increment of the model over each cheap competitor, per design (paired on the same tasks)
    piv = D.pivot(index="design", columns="arm", values="saved")
    inc = pd.DataFrame({f"G14_minus_{a}": piv["G14"] - piv[a]
                        for a in ("DIR_DONORS13", "DIR_BITE6", "DIR_MASSACT8")})
    D.to_csv(OUT / "refute_l3a_B_competitors.csv", index=False)
    inc.to_csv(OUT / "refute_l3a_B_competitor_increments.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(D.round(4).to_string(index=False))
        print()
        print(piv.round(4).to_string())
        print()
        print(inc.round(4).to_string())
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
