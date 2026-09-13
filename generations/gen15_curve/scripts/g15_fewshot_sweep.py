"""Gen15 step 3b: does anything in the measurement-assisted estimator recover the remaining headroom?

Under BP with a publication-masked covariance, one measured pair takes the extractant-macro MAE from
0.439 to 0.231 while the cell's own two coefficients would give 0.182, so 0.049 is left on the table
after one measurement.  This sweeps the parts of the estimator that could hold it:

  support   which pair the laboratory is told to measure -- widest dZ, greedy D-optimal, or random
  cov       the residual covariance -- empirical, a two-parameter smooth kernel in the ionic radius,
            or a half-and-half blend (the empirical one is 105 parameters from a few dozen curves)
  noise     the assumed variance of one measured log SF, fixed or taken from the cell's replicates

This is a sensitivity study, not a selection: a variant is adopted only if it wins in ALL FIVE
designs, which is the project's own rule.  Everything is reported.

Usage:  python gen15_curve/scripts/g15_fewshot_sweep.py [designs]
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
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP"]
ARMS = {"G14": A.g14, "FLAT": A.flat, "OBOTH": A.o_both}
KS = (0, 1, 2, 3)

VARIANTS = [
    ("widest_empirical", dict(how="widest", cov_kind="empirical")),
    ("widest_smooth", dict(how="widest", cov_kind="smooth")),
    ("widest_blend", dict(how="widest", cov_kind="blend")),
    ("widest_cellnoise", dict(how="widest", cov_kind="empirical", per_cell_noise=True)),
    ("widest_noise04", dict(how="widest", cov_kind="empirical", noise_var=0.04)),
    ("widest_noise20", dict(how="widest", cov_kind="empirical", noise_var=0.20)),
    ("dopt_empirical", dict(how="dopt", cov_kind="empirical")),
    ("dopt_blend", dict(how="dopt", cov_kind="blend")),
    ("random_empirical", dict(how="random", cov_kind="empirical")),
]

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    rows = []
    for name, kw in VARIANTS:
        for d in DESIGNS:
            r = FS.evaluate(bench, ARMS, d, ks=KS, mask_publication=True, verbose=False, **kw)
            if r.pairs.empty:
                continue
            pe = per_extractant(r.pairs, r.modes)
            bd = summarise(pe, r.pairs, r.modes)
            for _, s in bd.iterrows():
                rows.append({"variant": name, "design": d, "mode": s["arm"],
                             "mae": s["macro_mae_extractant"],
                             "mae_far": s["macro_mae_far"],
                             "sign_acc": s["macro_sign_acc_strong"],
                             "spearman": s["macro_pair_spearman"]})
            print(f"  {name:18s} {d}  k1={bd.set_index('arm').loc['G14@k1', 'macro_mae_extractant']:.4f}"
                  f"  k3={bd.set_index('arm').loc['G14@k3', 'macro_mae_extractant']:.4f}"
                  f"  ({r.seconds:.0f}s)", flush=True)
    T = pd.DataFrame(rows)
    V.RESULTS.mkdir(parents=True, exist_ok=True)
    T.to_csv(V.RESULTS / "g15_fewshot_sweep.csv", index=False)
    pd.set_option("display.width", 240)
    for d in DESIGNS:
        sub = T[T.design == d]
        print(f"\n=== design {d}: extractant-macro MAE by variant x mode ===")
        print(sub.pivot(index="variant", columns="mode", values="mae")
              .reindex(columns=[m for m in ["G14@k0", "G14@k1", "G14@k2", "G14@k3",
                                            "FLAT@k1", "NAIVE_LINE@k1", "OBOTH@k1"]
                                if m in set(sub["mode"])]).round(4).to_string())
    print(f"\n[sweep] total {time.time() - t0:.0f}s")
