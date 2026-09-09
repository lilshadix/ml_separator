"""Gen15 step 3c: is the measurement-assisted prediction's own uncertainty honest?

A separation factor is acted on, not admired: a laboratory needs to know when the number can be
trusted.  The BLUP already carries a posterior covariance, and nothing in this programme has ever
used it.  This script asks the only questions that matter about it:

  * **coverage** -- do nominal 50 / 80 / 90 / 95 % intervals contain the observed log SF that often?
  * **resolution** -- does a larger predicted sd actually mark a larger error, or is it flat?
  * **decision value** -- on the pairs where the model is confident about the *sign*, how often is
    the sign right, and what fraction of pairs is that?  That is the statement a chemist can use:
    "for this fraction of pairs the model tells you which lanthanide goes into the organic phase,
    and it is right this often."

Usage:  python gen15_curve/scripts/g15_uncertainty.py [designs] [--arm G14]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen15 import arms as A  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
DESIGNS = argv[0].split(",") if argv else ["BP"]
ARM = "G14"
for i, a in enumerate(sys.argv):
    if a == "--arm" and i + 1 < len(sys.argv):
        ARM = sys.argv[i + 1]

KS = (0, 1, 2, 3)
LEVELS = (0.50, 0.80, 0.90, 0.95)


def coverage_table(t: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = []
    for k in KS:
        pred, sd = t[f"{arm}@k{k}"], t[f"sd@k{k}"]
        err = (t["y"] - pred).abs()
        rec = {"k": k, "n_pairs": len(t), "mae": float(err.mean()),
               "mean_sd": float(sd.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
               "z_sd": float((err / sd).std()), "median_abs_z": float((err / sd).median())}
        for lv in LEVELS:
            z = stats.norm.ppf(0.5 + lv / 2)
            rec[f"cover{int(lv * 100)}"] = float((err <= z * sd).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def resolution_table(t: pd.DataFrame, arm: str, k: int, bins: int = 5) -> pd.DataFrame:
    d = pd.DataFrame({"sd": t[f"sd@k{k}"], "err": (t["y"] - t[f"{arm}@k{k}"]).abs(),
                      "y": t["y"].abs()})
    d["bin"] = pd.qcut(d.sd, bins, duplicates="drop")
    return (d.groupby("bin", observed=True)
            .agg(n=("err", "size"), mean_sd=("sd", "mean"), mae=("err", "mean"),
                 mean_abs_y=("y", "mean")).reset_index())


def decision_table(t: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Sign accuracy as a function of how confident the model is that the sign is not zero."""
    rows = []
    for k in KS:
        pred, sd = t[f"{arm}@k{k}"], t[f"sd@k{k}"]
        z = (pred.abs() / sd)
        for thr in (0.0, 1.0, 1.64, 1.96, 2.58):
            m = z >= thr
            if m.sum() < 30:
                continue
            strong = m & (t["y"].abs() >= 0.3)
            rows.append({"k": k, "z_threshold": thr, "coverage": float(m.mean()),
                         "n": int(m.sum()),
                         "sign_acc": float((np.sign(t.loc[m, "y"]) == np.sign(pred[m])).mean()),
                         "sign_acc_strong_pairs": float(
                             (np.sign(t.loc[strong, "y"]) == np.sign(pred[strong])).mean())
                         if strong.sum() >= 20 else np.nan,
                         "mae": float((t.loc[m, "y"] - pred[m]).abs().mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    pd.set_option("display.width", 220)
    out = {}
    for d in DESIGNS:
        r = FS.evaluate(bench, {"G14": A.g14, "FLAT": A.flat}, d, ks=KS, how="widest",
                        mask_publication=True, verbose=False)
        t = r.pairs
        out[d] = t
        cov = coverage_table(t, ARM); cov.insert(0, "design", d)
        dec = decision_table(t, ARM); dec.insert(0, "design", d)
        res = resolution_table(t, ARM, 1); res.insert(0, "design", d)
        V.RESULTS.mkdir(parents=True, exist_ok=True)
        cov.to_csv(V.RESULTS / f"g15_uncertainty_coverage_{d}.csv", index=False)
        dec.to_csv(V.RESULTS / f"g15_uncertainty_decision_{d}.csv", index=False)
        res.to_csv(V.RESULTS / f"g15_uncertainty_resolution_{d}.csv", index=False)
        print(f"\n=== {d}: interval coverage of {ARM} (nominal vs actual) ===")
        print(cov.round(4).to_string(index=False))
        print(f"\n=== {d}: resolution at k=1 -- does a bigger predicted sd mark a bigger error? ===")
        print(res.round(4).to_string(index=False))
        print(f"\n=== {d}: decision value -- sign accuracy vs how much of the corpus is kept ===")
        print(dec.round(4).to_string(index=False))
    print(f"\n[uncertainty] total {time.time() - t0:.0f}s")
