"""Gen15 step 1: locate the remaining information.

Gen14 ended with "the magnitude is not in the 2D structure" and seven priors inside 0.007 of a
constant.  Before another prior is tried, this script asks where the magnitude *could* come from,
by scoring a ladder of oracles alongside the deployed arm under all five designs:

  FLAT              predict no separation at all -- the floor the programme has never printed
  MEAN_CURVE        the corpus mean curve
  G14               the deployed arm: logistic direction x constant magnitude
  G13_FULL          gen13's 209-column regression
  O_SIGN            true direction x constant magnitude
  O_AMP             the cell's own radius coefficient
  O_BOTH            both of the cell's own coefficients -- the two-parameter representation ceiling
  O_CURV            gen14's amplitude with the cell's own curvature
  O_EXTMAG          magnitude = mean |amplitude| of the same extractant's other cells
  O_PUBMAG          magnitude = mean |amplitude| of the same publication's other cells
  O_LEVEL_MEAN/MAX  magnitude as a monotone function of the cell's own level of log D
  QUERY_SPAN        magnitude as a monotone function of the radius span of the metals asked about
  N_METALS          magnitude as a monotone function of how many metals are asked about

Usage:  python generations/gen15_curve/scripts/g15_locate.py [design,design,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen15 import arms as A  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)

ARMS = {
    "FLAT": A.flat,
    "MEAN_CURVE": A.mean_curve,
    "G14": A.g14,
    "G13_FULL": A.g13_full,
    "O_SIGN": A.o_sign,
    "O_AMP": A.o_amp,
    "O_BOTH": A.o_both,
    "O_CURV": A.o_curvature,
    "O_EXTMAG": A.o_extractant_magnitude,
    "O_PUBMAG": A.o_publication_magnitude,
    "O_LEVEL_MEAN": A.o_level("mean"),
    "O_LEVEL_MAX": A.o_level("max"),
    "QUERY_SPAN": A.query_span_magnitude(),
    "N_METALS": A.n_metals_magnitude(),
}

COMPS = {
    "G14_vs_FLAT": ("FLAT", "G14"),
    "G14_vs_MEANCURVE": ("MEAN_CURVE", "G14"),
    "OAMP_vs_G14": ("G14", "O_AMP"),
    "OEXTMAG_vs_G14": ("G14", "O_EXTMAG"),
    "OPUBMAG_vs_G14": ("G14", "O_PUBMAG"),
    "OLEVELMAX_vs_G14": ("G14", "O_LEVEL_MAX"),
    "QUERYSPAN_vs_G14": ("G14", "QUERY_SPAN"),
    "NMETALS_vs_G14": ("G14", "N_METALS"),
    "OCURV_vs_G14": ("G14", "O_CURV"),
}

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    print(f"[g15] bench loaded {time.time() - t0:.0f}s; {len(ARMS)} arms x {len(DESIGNS)} designs",
          flush=True)
    B, C, tables = V.score(bench, ARMS, DESIGNS, comps=COMPS)
    V.RESULTS.mkdir(parents=True, exist_ok=True)
    B.to_csv(V.RESULTS / "g15_locate_board.csv", index=False)
    C.to_csv(V.RESULTS / "g15_locate_contrasts.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== extractant-macro MAE of log SF ===")
    print(V.wide(B).round(4).to_string())
    for col in ("macro_mae_far", "macro_sign_acc_strong", "macro_pair_spearman"):
        print(f"\n=== {col} ===")
        print(V.wide(B, col).round(4).to_string())
    if len(C):
        print("\n=== paired contrasts (positive favours the candidate) ===")
        print(C[["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                 "units_improved", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]]
              .round(4).to_string(index=False))
    print(f"\n[g15] total {time.time() - t0:.0f}s")
