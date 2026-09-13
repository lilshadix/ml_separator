"""Gen15 step 2: is the second coefficient predictable, and what is each rung of the ladder worth?

Usage:  python generations/gen15_curve/scripts/g15_shape.py [design,design,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen15 import arms as A  # noqa: E402
from gen15 import shape as S  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)

ARMS = {
    "G14": A.g14,
    "B_MEDIAN": S.curv_constant("median"),
    "B_MEAN_RICH": S.curv_constant("mean"),
    "B_DIRCOND_MED": S.curv_direction_conditional("median"),
    "B_DIRCOND_MEAN": S.curv_direction_conditional("mean"),
    "B_LOGIT": S.curv_sign_model(direction_conditional=False),
    "B_LOGIT_DC": S.curv_sign_model(direction_conditional=True),
    "B_LOGIT_ORACLE": S.curv_sign_model(direction_conditional=True, oracle=True),
    "SHAPE3": S.shape_alphabet(3),
    "SHAPE4": S.shape_alphabet(4),
    "SHAPE6": S.shape_alphabet(6),
    "SHAPE4_ORACLE": S.shape_alphabet(4, oracle=True),
    "SHAPE6_ORACLE": S.shape_alphabet(6, oracle=True),
    "O_CURV": A.o_curvature,
    "O_BOTH": A.o_both,
    "FLAT": A.flat,
}

COMPS = {
    "BMEDIAN_vs_G14": ("G14", "B_MEDIAN"),
    "BDIRCONDMED_vs_G14": ("G14", "B_DIRCOND_MED"),
    "BDIRCONDMEAN_vs_G14": ("G14", "B_DIRCOND_MEAN"),
    "BLOGIT_vs_G14": ("G14", "B_LOGIT"),
    "BLOGITDC_vs_G14": ("G14", "B_LOGIT_DC"),
    "BLOGITDC_vs_BDIRCONDMED": ("B_DIRCOND_MED", "B_LOGIT_DC"),
    "SHAPE3_vs_G14": ("G14", "SHAPE3"),
    "SHAPE4_vs_G14": ("G14", "SHAPE4"),
    "SHAPE6_vs_G14": ("G14", "SHAPE6"),
    "SHAPE4ORACLE_vs_G14": ("G14", "SHAPE4_ORACLE"),
    "G14_vs_FLAT": ("FLAT", "G14"),
}

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    print(f"[g15-shape] {len(ARMS)} arms x {len(DESIGNS)} designs", flush=True)
    B, C, _ = V.score(bench, ARMS, DESIGNS, comps=COMPS)
    V.RESULTS.mkdir(parents=True, exist_ok=True)
    B.to_csv(V.RESULTS / "g15_shape_board.csv", index=False)
    C.to_csv(V.RESULTS / "g15_shape_contrasts.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== extractant-macro MAE ===")
    print(V.wide(B).round(4).to_string())
    print("\n=== macro sign accuracy on strong pairs ===")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n=== macro pair Spearman ===")
    print(V.wide(B, "macro_pair_spearman").round(4).to_string())
    if len(C):
        print("\n=== paired contrasts (positive favours the candidate) ===")
        print(C[["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                 "units_improved", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]]
              .round(4).to_string(index=False))
    print(f"\n[g15-shape] total {time.time() - t0:.0f}s")
