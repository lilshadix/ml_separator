"""The headline direction table: TOPO39 vs the majority floor vs the best of each representation.

One row per (design, model), all five designs, gen14's macro accuracy over extractants with its
chemotype-blocked bootstrap.  The three numbers this experiment turns on are all here: TOPO39's
0.821 under BP, the always-majority floor, and what a pretrained embedding actually reaches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                            # noqa: E402
from direction_acc import run_majority, run_matrix, run_topo       # noqa: E402
from gen14 import dirbench as D                                    # noqa: E402
from gen15.valuebench import DESIGNS                               # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

#: representation -> the single configuration reported, fixed at gen14's rule (C = 1, no PCA).
TAGS = ["chemberta_mtr__mean", "chemberta_mtr__cls", "chemberta_mlm__mean", "chemberta_mlm__cls",
        "morgan2", "morgan3"]


def main() -> None:
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else TAGS
    designs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DESIGNS)
    bench = D.load()
    ext = tuple(bench.frame.extractant.astype(str).tolist())
    out = []
    for design in designs:
        oofs = [run_topo(bench, design), run_majority(bench, design)]
        for t in tags:
            oofs.append(run_matrix(bench, t, FEAT.cell_block(t, ext), design, C=1.0))
        b = D.direction_board(oofs, baseline="TOPO39")
        out.append(b)
        print(f"\n=== direction macro accuracy, design {design} ===", flush=True)
        print(b[["model", "macro_accuracy", "ci_low", "ci_high", "pooled_accuracy",
                 "gain", "p_two_sided"]].round(4).to_string(index=False), flush=True)
    B = pd.concat(out, ignore_index=True)
    B.to_csv(OUT / "headline_direction.csv", index=False)
    w = B.pivot(index="model", columns="design", values="macro_accuracy")
    w = w[[d for d in DESIGNS if d in w.columns]]
    print("\n=== macro accuracy, model x design ===")
    print(w.round(4).sort_values(w.columns[-1], ascending=False).to_string())


if __name__ == "__main__":
    main()
