"""The constant-direction floors, on gen14's own macro-over-extractants metric.

The programme quotes "0.821 vs 0.559 for always-heavy".  ``run_majority`` in ``direction_acc`` fits
the *chemotype-weighted* majority inside each training fold, which on this corpus comes out light,
so it is a different (and much weaker) reference.  This computes the two fixed rules directly --
always heavy, always light -- reusing the same out-of-fold cell list, unit and weighting as every
other row, so the floor in the report is the one the programme means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT / "generations" / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from direction_acc import run_topo                                 # noqa: E402
from gen14 import dirbench as D                                    # noqa: E402
from gen15.valuebench import DESIGNS                               # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def constant_oof(template: D.OOF, p: float, name: str) -> D.OOF:
    """The same held-out cells, answered by a fixed probability."""
    c = template.cells.copy()
    c["p"] = p
    c["name"] = name
    return D.OOF(cells=c, design=template.design, name=name, seconds=0.0)


def main() -> None:
    designs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DESIGNS)
    bench = D.load()
    rows = []
    for design in designs:
        t = run_topo(bench, design)
        oofs = [t,
                constant_oof(t, 1.0, "ALWAYS_HEAVY"),
                constant_oof(t, 0.0, "ALWAYS_LIGHT")]
        b = D.direction_board(oofs, baseline="TOPO39")
        b.insert(0, "design_", design)
        rows.append(b)
    B = pd.concat(rows, ignore_index=True)
    B.to_csv(OUT / "constant_floor.csv", index=False)
    w = B.pivot(index="model", columns="design", values="macro_accuracy")
    w = w[[d for d in DESIGNS if d in w.columns]]
    print("=== constant-direction floors, macro accuracy over extractants ===")
    print(w.round(4).to_string())
    print()
    print(B[["design", "model", "macro_accuracy", "ci_low", "ci_high",
             "pooled_accuracy"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
