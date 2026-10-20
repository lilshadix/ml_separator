#!/usr/bin/env python
"""Budget ladders on the C6 cohort from the stored row-level predictions (no refit): for R20 (DOPT_Q, R08's
corrector trained per budget) and R24 (POL_ERR(k), the one-parameter Gd correction refitted per budget) writes
  results/<ID>/ladder_<design>.csv        seed-mean R2 / MAE / RMSE / per-seed R2 / min seed at k = 3, 5, 6
  results/<ID>/levels_ladder_<design>.csv chemotype-bootstrap 95 % intervals of the R2 level at each k
  results/<ID>/contrasts_k<k>_<design>.csv the correction vs PWL_R, PWL_Z and HBK@ML on the same reads, k = 5, 6
  results/R24/contrasts_k5_placement_<design>.csv  PWL_R@POL_ERR(5) - PWL_R@DOPT_Q(5) on identical C6 rows
All bootstraps use r3.metrics defaults (2000 replicates, gen20k seed), the same as every other round-3 contrast.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_ladders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import metrics as MT, paths  # noqa: E402

LADDERS = {"R20": "COR[GD]@k", "R24": "COR[Dnc]@k"}
ARMS = ["PWL_R", "PWL_Z", "HBK@ML", "RIDGE2"]


def main() -> int:
    for cid, col in LADDERS.items():
        out = paths.RES / cid
        for design in ("BP", "BPG"):
            t = pd.read_parquet(paths.PRED / f"{cid}_{design}.parquet")
            arms = [col] + ARMS
            lad, lev = [], []
            for k in (3, 5, 6):
                s = MT.summary(t, arms, K=6, k=k, subsets=False)
                s.insert(0, "design", design)
                lad.append(s)
                lv = MT.levels_ci(t, arms, K=6, ks=(k,))
                lv.insert(0, "design", design)
                lev.append(lv)
                if k in (5, 6):
                    c = MT.contrasts(t, [(col, r) for r in ARMS], K=6, k=k)
                    c.insert(0, "k", k)
                    c.insert(0, "design", design)
                    c.to_csv(out / f"contrasts_k{k}_{design}.csv", index=False)
            pd.concat(lad).to_csv(out / f"ladder_{design}.csv", index=False)
            if cid == "R24":                       # placement alone at k = 5: PWL_R@POL_ERR(5) vs PWL_R@DOPT_Q(5) (R20)
                key = ["split_seed", "fold", "cell_id", "A", "B", "k"]
                d20 = pd.read_parquet(paths.PRED / f"R20_{design}.parquet")[key + ["y", "PWL_R"]]
                m = t.merge(d20.rename(columns={"PWL_R": "PWL_R@DOPT_Q", "y": "y20"}), on=key)
                assert len(m) == len(t) and (m.y == m.y20).all()
                m = m.rename(columns={"PWL_R": "PWL_R@POL_ERR"})
                c = MT.contrasts(m, [("PWL_R@POL_ERR", "PWL_R@DOPT_Q")], K=6, k=5)
                c.insert(0, "design", design)
                c.to_csv(out / f"contrasts_k5_placement_{design}.csv", index=False)
            pd.concat(lev).to_csv(out / f"levels_ladder_{design}.csv", index=False)
            print("written", cid, design, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
