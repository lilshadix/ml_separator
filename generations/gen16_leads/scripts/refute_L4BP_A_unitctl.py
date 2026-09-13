"""Refuter lens A, step 8: do the size-matched controls survive on the chemotype scoring unit?

Step 7 found that the registered extractant-unit sign inconsistency (B and BQ negative) becomes
positive in all five designs when the scoring unit is the chemotype rather than the extractant.
The registered unit is the extractant, so that is an observation and not a promotion -- but the
size-matched controls must be re-read on the same unit or the observation is not answered.

Reads the refuter's own ABC frames from `refute_L4BP_A_rerun.py` (BP and B) and re-runs
AOPT vs BIGGEST / RANDCM / RANDOM with the chemotype as the scoring unit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.inference import paired_contrasts  # noqa: E402

OUT = ROOT / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
COMPS = {"AOPTR_vs_RANDOM": ("RANDOM", "AOPT_R"), "BIGGEST_vs_RANDOM": ("RANDOM", "BIGGEST"),
         "AOPTR_vs_BIGGEST": ("BIGGEST", "AOPT_R"), "AOPTR_vs_RANDCM": ("RANDCM", "AOPT_R"),
         "RANDCM_vs_RANDOM": ("RANDOM", "RANDCM")}
pd.set_option("display.width", 260)


def main() -> None:
    rows = []
    for design in ("BP", "B"):
        A = pd.read_parquet(OUT / f"rerun_abc_{design}.parquet")
        for unit in ("extractant", "chemotype"):
            if unit == "extractant":
                pe = A.rename(columns={"order": "arm"})
            else:
                pe = (A.groupby(["order", "split_seed", "chemotype"])["mae_all"].mean()
                      .reset_index().rename(columns={"order": "arm"}))
                pe["extractant"] = pe["chemotype"]
            c = paired_contrasts(pe, COMPS, value="mae_all")
            c.insert(0, "unit", unit)
            c.insert(0, "design", design)
            rows.append(c)
    C = pd.concat(rows, ignore_index=True)
    C.to_csv(OUT / "unit_controls.csv", index=False)
    print(C[["design", "unit", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
             "seeds_positive", "loco_sign_stable", "passes_P1", "n_units"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
