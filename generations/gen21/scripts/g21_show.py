#!/usr/bin/env python
"""Print the chat/report table of one attempt from its result files (no recomputation).

    .venv/Scripts/python.exe generations/gen21/scripts/g21_show.py --attempt 2 [--k 3] [--arms A,B]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from g21 import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    d = paths.RESULTS / f"attempt{a.attempt}"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    for design in ("BP", "BPG"):
        p = d / f"levels_{design}.csv"
        if not p.exists():
            continue
        lv = pd.read_csv(p)
        lv = lv[(lv.cohort == "C3") & (lv.k == a.k)]
        if a.arms:
            lv = lv[lv.arm_base.isin(a.arms.split(","))]
        print(f"== attempt {a.attempt} {design} C3 k={a.k}")
        print(lv[["arm", "R2", "R2_ci_low", "R2_ci_high", "MAE", "RMSE", "MAE_macro_extractant", "n_rows", "n_cells"]]
              .sort_values("R2", ascending=False).to_string(index=False, float_format=fmt))
        c = d / f"contrasts_{design}_C3.csv"
        if c.exists():
            ct = pd.read_csv(c)
            ct = ct[ct.name.str.endswith(f"k{a.k}") & ct.statistic.isin(["R2", "MAE"])]
            print(ct[["name", "statistic", "point", "ci_low", "ci_high", "p_two_sided"]].to_string(index=False, float_format=fmt))
            ps = pd.read_csv(d / f"contrasts_per_seed_{design}_C3.csv")
            ps = ps[ps.k == a.k].groupby(["candidate", "reference"]).agg(
                seeds_R2_better=("dR2", lambda v: int((v > 0).sum())), seeds_MAE_better=("dMAE_ref_minus_cand", lambda v: int((v > 0).sum())))
            print(ps.to_string())
        s = d / f"subsets_{design}_C3_k3.csv"
        if s.exists():
            st = pd.read_csv(s)
            if a.arms:
                st = st[st.arm.isin(a.arms.split(","))]
            print(st[["subset", "arm", "R2", "MAE", "RMSE", "n_rows_per_seed", "n_cells_per_seed"]].to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
