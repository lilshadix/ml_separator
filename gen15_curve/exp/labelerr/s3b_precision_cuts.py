"""Step 3b: the sharpest version of the precision idea -- drop the noisy training labels.

If the magnitude and the direction look unpredictable because a third of the training labels are
inside their own measurement noise, then refusing to fit on those labels should help.  gen14 calls a
cell well determined at >= 5 measured metals; here that cut is moved, and replaced by a cut on the
*computed* standard error of the cell's own ``a``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from arms_labelerr import labelerr_arm, rich_threshold_arm  # noqa: E402
from gen15 import arms as A, valuebench as V  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

ARMS = {
    "FLAT": A.flat,
    "G14": A.g14,
    "PRECW_SHARP": labelerr_arm(weights="precision", tau2=0.001),
    "RICH3": rich_threshold_arm(min_metals=3),
    "RICH8": rich_threshold_arm(min_metals=8),
    "RICH10": rich_threshold_arm(min_metals=10),
    "SE_A_LT_050": rich_threshold_arm(min_metals=2, max_se_a=0.050),
    "SE_A_LT_040": rich_threshold_arm(min_metals=2, max_se_a=0.040),
}
COMPS = {f"{k}_vs_G14": ("G14", k) for k in ARMS if k not in ("FLAT", "G14")}
COMPS.update({f"{k}_vs_FLAT": ("FLAT", k) for k in ARMS if k != "FLAT"})


def main() -> None:
    t0 = time.time()
    bench = V.load()
    B, C, _ = V.score(bench, ARMS, ["B", "BR", "BQ", "A", "BP"], comps=COMPS)
    B.to_csv(OUT / "s3b_board.csv", index=False)
    C.to_csv(OUT / "s3b_contrasts.csv", index=False)
    print("\n=== extractant-macro MAE ===")
    print(V.wide(B).round(4).to_string())
    print("\n=== macro sign accuracy on strong pairs ===")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n=== contrasts vs G14, BP ===")
    print(C[(C.design == "BP")][["comparison", "point", "bca_low", "bca_high", "p_two_sided",
                                 "seeds_positive", "loco_sign_stable", "passes_P1"]]
          .round(4).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
