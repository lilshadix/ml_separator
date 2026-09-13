"""The completion run: every arm that came within sight of G14, under all five designs.

The first pass scored the direction arms and the ridge/cosine magnitude arms under all five designs
but ran the RBF kernel and the Morgan magnitude arms under BP only.  Those are exactly the arms that
landed closest to G14, so they are the ones that most need the other four designs before anything is
claimed.  FLAT and G14 ride along in every batch as the reproducibility check: they must come back
at 0.5885 and 0.5001 under BP.

Usage:  python run_final.py [designs]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT / "generations" / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arms_embed as E                                      # noqa: E402
from gen15 import arms as A, valuebench as V                 # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

PRIMARY = "chemberta_mtr__mean"


def batch() -> dict:
    d = {"FLAT": A.flat, "G14": A.g14}
    # --- magnitude: gen14's direction, |a| regressed from the representation -----------
    for tag, short in ((PRIMARY, "cbmtr_mean"), ("morgan2", "morgan2"), ("morgan3", "morgan3")):
        d[f"MAG_{short}_rbf"] = E.magnitude_arm(tag, how="rbf", alpha=1.0)
        d[f"MAG_{short}_cos"] = E.magnitude_arm(tag, how="cosine", alpha=1.0)
    # --- curvature: gen14's amplitude, b regressed from the representation -------------
    d["CUR_cbmtr_mean_cos"] = E.curvature_arm(PRIMARY, how="cosine", alpha=1.0)
    d["CUR_morgan2_ridge"] = E.curvature_arm("morgan2", how="ridge", alpha=100.0)
    # --- the strongest embedding magnitude with a stronger penalty --------------------
    d["MAG_cbmtr_mean_ridge1k"] = E.magnitude_arm(PRIMARY, how="ridge", alpha=1000.0)
    return d


def main() -> None:
    designs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)
    arms = batch()
    comps = {f"{k}_vs_G14": ("G14", k) for k in arms if k not in ("G14", "FLAT")}
    comps.update({f"{k}_vs_FLAT": ("FLAT", k) for k in arms if k not in ("G14", "FLAT")})
    bench = V.load()
    t0 = time.time()
    B, C, _ = V.score(bench, arms, designs, comps=comps)
    B.to_csv(OUT / "board_final.csv", index=False)
    if len(C):
        C.to_csv(OUT / "contrast_final.csv", index=False)
    print(f"\n=== final  extractant-macro MAE  ({time.time() - t0:.0f}s) ===")
    print(V.wide(B).round(4).to_string())
    print("\n--- macro_sign_acc_strong ---")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n--- macro_pair_spearman ---")
    print(V.wide(B, "macro_pair_spearman").round(4).to_string())
    if len(C):
        print("\n--- contrasts ---")
        print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
