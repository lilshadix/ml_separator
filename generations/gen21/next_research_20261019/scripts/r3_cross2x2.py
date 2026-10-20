#!/usr/bin/env python
"""2 x 2 (read policy x one-parameter Gd correction) on identical rows from the stored predictions: R23 (Gd,
POL_ERR) and R14 (Gd, DOPT_Q) with their PWL_R columns.  Paired chemotype-bootstrap contrasts (r3.metrics defaults).

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_cross2x2.py
writes results/policy_cross/contrasts_2x2_<design>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import metrics as MT, paths  # noqa: E402

KEY = ["split_seed", "fold", "cell_id", "A", "B", "k"]


def main() -> int:
    for d in ("BP", "BPG"):
        a = pd.read_parquet(paths.PRED / f"R23_{d}.parquet").rename(columns={"COR[Dnc]": "Dnc@POL_ERR", "PWL_R": "PWL_R@POL_ERR"})
        b = pd.read_parquet(paths.PRED / f"R14_{d}.parquet").rename(columns={"COR[Dnc]": "Dnc@DOPT_Q", "PWL_R": "PWL_R@DOPT_Q"})
        m = a.merge(b[KEY + ["y", "Dnc@DOPT_Q", "PWL_R@DOPT_Q"]], on=KEY, suffixes=("", "_b"))
        assert len(m) == len(a) == len(b) and np.allclose(m.y, m.y_b)
        pairs = [("Dnc@POL_ERR", "Dnc@DOPT_Q"), ("PWL_R@POL_ERR", "PWL_R@DOPT_Q"), ("Dnc@POL_ERR", "PWL_R@DOPT_Q")]
        c = MT.contrasts(m, pairs, K=3, k=3)
        c.insert(0, "design", d)
        c.to_csv(paths.RES / "policy_cross" / f"contrasts_2x2_{d}.csv", index=False)
        print(d, c[c.statistic == "R2"][["name", "point", "ci95_low", "ci95_high"]].round(4).values.tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
