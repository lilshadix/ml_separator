"""Q3 in its magnitude-only form: 'which extractant gives the largest separation, either way'.

The main Q3 table asks the directional question -- maximise log SF(A over B), or maximise
log SF(B over A) -- and averages the two, because a chemist wants a named metal in the raffinate.
This variant asks the weaker question the brief also names: rank by |predicted log SF| and score
against |observed log SF|, so a system that separates the pair strongly in the *wrong* direction
still counts as a good separator.
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

from gen15 import valuebench as V                      # noqa: E402
import decmetrics as M                                  # noqa: E402
from run_metrics import DESIGNS, ARMS, REF, load_design  # noqa: E402

ARMSET = ARMS + REF + ["G14_TIED"]
MIN_EXT = M.MIN_EXT_RANK


def main() -> None:
    bench = V.load()
    out, longs = [], []
    for d in DESIGNS:
        tab = load_design(bench, d)
        key = ["split_seed", "A", "B"]
        g = tab.groupby(key + ["extractant"], sort=False)[["y"] + ARMSET].median().reset_index()
        recs = []
        for (seed, a, bm), blk in g.groupby(key, sort=False):
            n = len(blk)
            if n < MIN_EXT:
                continue
            obs = np.abs(blk["y"].to_numpy())
            if np.ptp(obs) == 0:
                continue
            hi = obs.max()
            best = (obs >= hi - 1e-12).astype(float)
            for arm in ARMSET + ["_RANDOM"]:
                p = np.zeros(n) if arm == "_RANDOM" else np.abs(blk[arm].to_numpy())
                recs.append({"split_seed": seed, "A": a, "B": bm, "n_units": n, "arm": arm,
                             "obs_spread": float(hi - obs.min()),
                             "spearman_abs": M._spearman(p, obs),
                             "top1_abs": M._tie_expected_top1(p, best),
                             "regret_abs": hi - M._tie_expected_value(p, obs)})
        per = pd.DataFrame(recs)
        per.insert(0, "design", d)
        longs.append(per)
        rows = []
        for arm, blk in per.groupby("arm"):
            rec = {"design": d, "arm": arm, "n_tasks": int(len(blk))}
            for col in ("spearman_abs", "top1_abs", "regret_abs"):
                ps = blk.groupby("split_seed")[col].mean()
                rec[col] = float(ps.mean())
                rec[col + "_sd"] = float(ps.std(ddof=1))
            rows.append(rec)
        out.append(pd.DataFrame(rows))
        print(f"[{d}] done", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(HERE / "results" / "q3_absolute.csv", index=False)
    pd.concat(longs, ignore_index=True).to_csv(HERE / "results" / "q3_absolute_long.csv",
                                               index=False)
    for m in ("spearman_abs", "top1_abs", "regret_abs"):
        print(m)
        print(res.pivot(index="arm", columns="design", values=m)[list(DESIGNS)].round(4)
              .to_string())


if __name__ == "__main__":
    main()
