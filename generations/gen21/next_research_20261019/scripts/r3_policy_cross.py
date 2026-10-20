#!/usr/bin/env python
"""Direction D cross-comparison (brief §9): every predictor under DOPT_Q (config B0 or the method's own
DOPT_Q config) and under POL_ERR (config R09 / R12 / R14), on identical cells and query pairs at k = 3.
Writes results/policy_cross/{cross,contrasts}_<design>.csv.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_policy_cross.py --pol R09 --dopt B0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from r3 import metrics as MT  # noqa: E402
from r3 import paths  # noqa: E402

KEY = ["design", "split_seed", "fold", "cell_id", "extractant", "chemotype", "publication", "n_metals", "model", "k_max",
       "A", "B", "dZ", "regime", "y", "draw"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", default="R09")
    ap.add_argument("--dopt", default="B0")
    ap.add_argument("--arms", default="PWL_R,PWL_Z,HBK@ML")
    ap.add_argument("--replicates", type=int, default=2000)
    a = ap.parse_args()
    out = paths.RES / "policy_cross"
    out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    for d in ("BP", "BPG"):
        tp = pd.read_parquet(paths.PRED / f"{a.pol}_{d}.parquet")
        td = pd.read_parquet(paths.PRED / f"{a.dopt}_{d}.parquet")
        tp, td = tp[(tp.k == 3) & (tp.k_max >= 3)], td[(td.k == 3) & (td.k_max >= 3)]
        m = td[KEY + ["k", "policy", "a_read", "b_read"] + arms].merge(
            tp[KEY + ["a_read", "b_read"] + arms], on=KEY, suffixes=("@DOPT_Q", "@POL_ERR"), validate="one_to_one")
        if len(m) != len(td) or len(m) != len(tp):
            raise SystemExit(f"row sets differ: {len(td)} DOPT_Q, {len(tp)} POL_ERR, {len(m)} matched")
        cols = [f"{x}@DOPT_Q" for x in arms] + [f"{x}@POL_ERR" for x in arms]
        s = MT.summary(m.assign(a_read=m["a_read@DOPT_Q"], b_read=m["b_read@DOPT_Q"]), cols, K=3, k=3, subsets=False)
        s.insert(0, "design", d)
        s.to_csv(out / f"cross_{d}_{a.pol}.csv", index=False)
        pairs = [(f"{x}@POL_ERR", f"{x}@DOPT_Q") for x in arms] + [(f"{x}@POL_ERR", "PWL_R@DOPT_Q") for x in arms if x != "PWL_R"]
        ct = MT.contrasts(m.assign(a_read=m["a_read@DOPT_Q"], b_read=m["b_read@DOPT_Q"]), pairs, K=3, k=3, replicates=a.replicates)
        ct.insert(0, "design", d)
        ct.to_csv(out / f"contrasts_{d}_{a.pol}.csv", index=False)
        share = float((m["a_read@POL_ERR"] | m["b_read@POL_ERR"]).mean())
        print(d, f"one-end-read share: DOPT_Q {float((m['a_read@DOPT_Q'] | m['b_read@DOPT_Q']).mean()):.3f}, POL_ERR {share:.3f}")
        print(s[["arm", "R2", "MAE", "RMSE", "R2_min_seed"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(ct[ct.statistic.isin(["R2", "MAE"])][["name", "statistic", "point", "ci_low", "ci_high", "seeds_R2_better",
                                                    "seeds_MAE_better"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
