"""Turn the permutation replicates into a null distribution and an empirical p for every headline.

The real arms' headline numbers are recomputed from the cached pair tables with exactly the
function the replicates used, so the comparison is like for like.  ``p_emp`` is the one-sided
fraction of null replicates at least as good as the observed value, with the usual +1/+1
correction, so the smallest reportable p is 1/(R+1).
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
from run_metrics import DESIGNS, load_design  # noqa: E402

LOWER_BETTER = {"macro_mae", "cross_regret"}
PAIRS = {"G14": "G14_PERM", "G13_FULL": "G13_PERM"}
METRICS = ["macro_mae", "sign_acc", "cell_spearman", "cell_top1",
           "cross_spearman", "cross_top1", "cross_regret", "calib_slope"]


def main() -> None:
    bench = V.load()
    real_path = HERE / "results" / "headline_real.csv"
    if real_path.exists():
        real = pd.read_csv(real_path)
    else:
        rows = []
        for d in DESIGNS:
            tab = load_design(bench, d)
            for arm in ("FLAT", "MEAN_CURVE", "HEAVIER_ALWAYS", "G14", "G14_TIED",
                        "G13_FULL", "O_BOTH"):
                h = M.headline(tab, arm)
                h["design"] = d
                rows.append(h)
            print(f"[{d}] real headline done", flush=True)
        real = pd.DataFrame(rows)
        real.to_csv(real_path, index=False)

    out = []
    for d in DESIGNS:
        p = HERE / "perm" / f"null_{d}.csv"
        if not p.exists():
            continue
        # two copies of the runner were briefly live at once; the permutation is deterministic in
        # (seed, fold, replicate), so the repeated rows are identical to 1e-10 and one is dropped
        null = pd.read_csv(p).drop_duplicates(["rep", "arm"], keep="first")
        for arm, parm in PAIRS.items():
            n = null[null["arm"] == parm]
            if n.empty:
                continue
            r = real[(real["design"] == d) & (real["arm"] == arm)]
            if r.empty:
                continue
            r = r.iloc[0]
            for m in METRICS:
                v, nv = float(r[m]), n[m].to_numpy(dtype=float)
                nv = nv[np.isfinite(nv)]
                better = (nv <= v).sum() if m in LOWER_BETTER else (nv >= v).sum()
                out.append({"design": d, "arm": arm, "metric": m, "observed": v,
                            "null_mean": float(nv.mean()), "null_sd": float(nv.std(ddof=1)),
                            "null_min": float(nv.min()), "null_max": float(nv.max()),
                            "n_rep": int(len(nv)),
                            "z_vs_null": float((v - nv.mean()) / nv.std(ddof=1))
                            if nv.std(ddof=1) > 0 else np.nan,
                            "p_emp": float((better + 1) / (len(nv) + 1))})
    df = pd.DataFrame(out)
    df.to_csv(HERE / "results" / "permutation_null.csv", index=False)
    print(df[df.design == "BP"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
