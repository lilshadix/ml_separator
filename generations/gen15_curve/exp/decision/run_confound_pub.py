"""Confound control for the design question: rank extractants *inside one publication*.

The pooled cross-extractant ranking of Q3 compares systems measured in different laboratories under
different acids, diluents and concentrations, so part of the observed spread in log SF is the
laboratory, not the ligand.  Restricting the ranking to (metal pair, publication) blocks with at
least five extractants removes that confound completely -- everything being ranked was measured by
one group under one protocol -- at the price of a much narrower corpus (4 publications).

Held-out status is preserved: every prediction is still out-of-fold under the stated design.
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

from gen15 import valuebench as V   # noqa: E402
import decmetrics as M              # noqa: E402
from run_metrics import DESIGNS, ARMS, REF, load_design  # noqa: E402

ARMSET = ARMS + REF + ["G14_TIED"]
MIN_EXT = 5


def within_publication(tab: pd.DataFrame, arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["split_seed", "A", "B", "publication_id"]
    g = tab.groupby(key + ["extractant"], sort=False)[["y"] + arms].median().reset_index()
    recs = []
    for (seed, a, bm, pub), blk in g.groupby(key, sort=False):
        n = len(blk)
        if n < MIN_EXT:
            continue
        obs = blk["y"].to_numpy()
        if np.ptp(obs) == 0:
            continue
        hi, lo = obs.max(), obs.min()
        bhi = (obs >= hi - 1e-12).astype(float)
        blo = (obs <= lo + 1e-12).astype(float)
        for arm in arms + ["_RANDOM"]:
            p = np.zeros(n) if arm == "_RANDOM" else blk[arm].to_numpy()
            recs.append({"split_seed": seed, "A": a, "B": bm, "publication_id": pub,
                         "n_units": n, "obs_spread": float(hi - lo), "arm": arm,
                         "spearman": M._spearman(p, obs),
                         "regret": 0.5 * ((hi - M._tie_expected_value(p, obs))
                                          + (M._tie_expected_value(-p, obs) - lo)),
                         "top1": 0.5 * (M._tie_expected_top1(p, bhi)
                                        + M._tie_expected_top1(-p, blo))})
    per = pd.DataFrame(recs)
    rows = []
    for arm, blk in per.groupby("arm"):
        rec = {"arm": arm, "n_tasks": int(len(blk)),
               "n_publications": int(blk["publication_id"].nunique()),
               "median_n_units": float(blk["n_units"].median()),
               "obs_spread": float(blk["obs_spread"].mean())}
        for col in ("spearman", "top1", "regret"):
            per_seed = blk.groupby("split_seed")[col].mean()
            rec[col] = float(per_seed.mean())
            rec[col + "_sd"] = float(per_seed.std(ddof=1))
        rows.append(rec)
    return pd.DataFrame(rows), per


def main() -> None:
    bench = V.load()
    pub = bench.frame[["cell_id", "publication_id"]]
    out, longs = [], []
    for d in DESIGNS:
        tab = load_design(bench, d).merge(pub, on="cell_id", validate="many_to_one")
        s, per = within_publication(tab, ARMSET)
        s.insert(0, "design", d)
        per.insert(0, "design", d)
        out.append(s)
        longs.append(per)
        print(f"[{d}] {int(s['n_tasks'].iat[0])} within-publication ranking tasks", flush=True)
    pd.concat(out, ignore_index=True).to_csv(HERE / "results" / "confound_within_pub.csv",
                                             index=False)
    pd.concat(longs, ignore_index=True).to_csv(HERE / "results" / "confound_within_pub_long.csv",
                                               index=False)
    print("written")


if __name__ == "__main__":
    main()
