#!/usr/bin/env python
"""Summary of the block-A synthetic discrimination study (reporting only; reads results/synthetic_a/*.json).

Counts k/n with exact Clopper-Pearson 95 % intervals for the rules AS THEY ARE APPLIED (PROTOCOL section 10): the
specificity rules only for a slot that passes criterion 1, so the composite claims are
  electronic claim     = crit1(S02) and S02 - S03 >= 0.005 and S02 > every S04 draw
  donor claim (route)  = crit1(route) and route > every S21 draw and mean S21 < 50 % of the route's gain
Also: the share of replicates with ΔR² >= 0.01 (without the balanced-MAE leg) and B1's synthetic RMSE.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/report/r4_synthetic_summary.py
writes results/synthetic_a/SUMMARY.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as SS

R4 = Path(__file__).resolve().parents[1]
WORLDS = ("NULL", "CHEM_D7", "GDCLASS", "ELEC", "PROJ", "UNIVERSAL")


def cp(v) -> str:
    v = np.asarray(v, bool)
    k, n = int(v.sum()), len(v)
    lo = 0.0 if k == 0 else SS.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else SS.beta.ppf(0.975, k + 1, n - k)
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    lines = ["# Block A synthetic discrimination study, run 2", "",
             "Two BPG folds, synthetic targets with class-agnostic chemotype nuisance and noise sd 0.11. Counts k/n with exact "
             "95 % intervals. Rules are an upper bound of the sealed ones (vs B1 only, no harm rule, no B2, one fold, no seed "
             "mean).", "",
             "| world | B1 RMSE | crit1 S01 | crit1 S02 | crit1 S03 | ΔR²≥0.01 S01 / S02 | electronic claim | donor claim S01 | "
             "donor claim S02 | mean ΔR² S01 / S02 / S03 | S04 / S21:S01 / S21:S02 draws |", "|" + "---|" * 11]
    for w in WORLDS:
        p = R4 / "results" / "synthetic_a" / f"{w}.json"
        if not p.exists():
            continue
        reps = json.loads(p.read_text())["replicates"]
        c1 = {s: [r["crit1"][s] for r in reps] for s in ("S01", "S02", "S03")}
        r2 = {s: [r["gain"][f"slot:{s}"] >= 0.01 for r in reps] for s in ("S01", "S02")}
        elec = [r["crit1"]["S02"] and r["elec_specific"] for r in reps]
        d1 = [r["crit1"]["S01"] and r["chem_specific_S01"] for r in reps]
        d2 = [r["crit1"]["S02"] and r["chem_specific_S02"] for r in reps]
        g = lambda key: np.mean([r["gain"][key] for r in reps])  # noqa: E731
        gd = lambda pre: np.mean([np.mean([v for k, v in r["gain"].items() if k.startswith(pre)]) for r in reps])  # noqa: E731
        rm = np.mean([r.get("b1_rmse", np.nan) for r in reps])
        lines.append(f"| {w} | {rm:.3f} | {cp(c1['S01'])} | {cp(c1['S02'])} | {cp(c1['S03'])} | "
                     f"{int(np.sum(r2['S01']))} / {int(np.sum(r2['S02']))} of {len(reps)} | {cp(elec)} | {cp(d1)} | {cp(d2)} | "
                     f"{g('slot:S01'):+.4f} / {g('slot:S02'):+.4f} / {g('slot:S03'):+.4f} | "
                     f"{gd('slot:S04'):+.4f} / {gd('slot:S21:S01'):+.4f} / {gd('slot:S21:S02'):+.4f} |")
    out = R4 / "results" / "synthetic_a" / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
