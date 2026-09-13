"""What a confidence filter actually selects.

A risk-coverage curve is easy to misread: an MAE that falls as coverage tightens can mean the model
knows where it is right, or it can mean the retained pairs simply have smaller separations.  This
records, for both confidence definitions and all five designs, what the retained set looks like.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from gen15 import valuebench as V              # noqa: E402
from run_metrics import DESIGNS, load_design   # noqa: E402

COV = (1.0, 0.5, 0.25, 0.10, 0.05)


def main() -> None:
    bench = V.load()
    rows = []
    for d in DESIGNS:
        tab = load_design(bench, d)
        for lab, conf in [("|prediction|", tab["G14"].abs().to_numpy()),
                          ("gen14 |p-0.5|", tab["dir_conf"].to_numpy())]:
            rng = np.random.default_rng(20260909)
            t = tab.assign(_c=conf, _j=rng.random(len(tab))).sort_values(
                ["_c", "_j"], ascending=[False, False], kind="stable")
            for q in COV:
                keep = t.iloc[:int(round(q * len(t)))]
                rows.append({"design": d, "confidence": lab, "coverage": q,
                             "n_pairs": len(keep), "n_cells": int(keep["cell_id"].nunique()),
                             "mean_abs_y": float(keep["y"].abs().mean()),
                             "frac_strong": float((keep["y"].abs() >= 0.3).mean()),
                             "mean_dZ": float(keep["dZ"].mean()),
                             "pooled_mae": float((keep["y"] - keep["G14"]).abs().mean())})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "q4_confidence_selection.csv", index=False)
    print(out[out.design == "BP"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
