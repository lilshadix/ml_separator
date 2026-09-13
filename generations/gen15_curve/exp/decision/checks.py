"""Self-checks on the decision metrics.  Run before trusting anything in REPORT.md."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from gen15 import valuebench as V                          # noqa: E402
from gen13sep.metrics import heavier_always_sign_accuracy  # noqa: E402
import decmetrics as M                                     # noqa: E402
from run_metrics import load_design                        # noqa: E402


def check_topk_expectation() -> None:
    """The closed-form tie expectation must match a Monte-Carlo draw of consistent orderings."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(4, 12))
        score = rng.integers(0, 3, size=n).astype(float)     # deliberately heavy ties
        best = np.zeros(n)
        best[rng.choice(n, size=int(rng.integers(1, 3)), replace=False)] = 1.0
        closed = M._tie_expected_topk(score, best, 3)
        hits = 0
        for _ in range(4000):
            order = np.lexsort((rng.random(n), -score))
            hits += best[order[:3]].sum() > 0
        worst = max(worst, abs(closed - hits / 4000))
    assert worst < 0.04, worst
    print(f"  tie-expected top-3 matches Monte Carlo to {worst:.4f}")


def check_top1_expectation() -> None:
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(3, 10))
        score = rng.integers(0, 2, size=n).astype(float)
        best = np.zeros(n)
        best[rng.integers(0, n)] = 1.0
        closed = M._tie_expected_top1(score, best)
        hits = 0
        for _ in range(4000):
            order = np.lexsort((rng.random(n), -score))
            hits += best[order[0]]
        worst = max(worst, abs(closed - hits / 4000))
    assert worst < 0.04, worst
    print(f"  tie-expected top-1 matches Monte Carlo to {worst:.4f}")


def main() -> None:
    bench = V.load()
    print("checks:")
    check_top1_expectation()
    check_topk_expectation()

    tab = load_design(bench, "BP")
    arms = ["FLAT", "HEAVIER_ALWAYS", "G14"]

    sb = M.sign_bands(tab, arms)
    mine = float(sb[(sb.band == "all") & (sb.arm == "HEAVIER_ALWAYS")].sign_acc.iat[0])
    repo = heavier_always_sign_accuracy(tab)
    assert abs(mine - repo) < 1e-9, (mine, repo)
    print(f"  heavier-always sign accuracy matches gen13's own function: {mine:.6f}")

    wc, cells = M.within_cell(tab, arms)
    f = wc[wc.arm == "FLAT"].iloc[0]
    r = wc[wc.arm == "_RANDOM"].iloc[0]
    for col in ("spearman", "kendall", "top1", "top3"):
        assert abs(float(f[col]) - float(r[col])) < 1e-12, col
    print("  FLAT reproduces _RANDOM exactly on every within-cell rank statistic")

    ce, _ = M.cross_extractant(tab, arms)
    for arm in ("FLAT", "HEAVIER_ALWAYS"):
        a = ce[ce.arm == arm].iloc[0]
        b = ce[ce.arm == "_RANDOM"].iloc[0]
        for col in ("spearman", "top1", "regret"):
            assert abs(float(a[col]) - float(b[col])) < 1e-12, (arm, col)
    print("  FLAT and HEAVIER_ALWAYS reproduce _RANDOM exactly on cross-extractant selection")

    # the oracle must be perfect on its own coefficients, and the pair table must be consistent
    coef = bench.coef
    ids = {c: i for i, c in enumerate(bench.frame["cell_id"])}
    r = bench.basis
    row = tab.iloc[0]
    i = ids[row["cell_id"]]
    la = list(bench.frame.columns[bench.frame.columns.str.startswith("logD__")])
    metals = [c.split("__")[1] for c in la]
    ia, ib = metals.index(row["A"]), metals.index(row["B"])
    want = float(coef[i] @ (r[:, ia] - r[:, ib]))
    assert abs(want - float(row["O_BOTH"])) < 1e-9, (want, row["O_BOTH"])
    print("  O_BOTH column equals the cell's own coefficients times the basis difference")

    slope, intercept, _, _ = M.calibration_slope(tab.assign(_perfect=tab["y"]), "_perfect")
    assert abs(slope - 1.0) < 1e-9 and abs(intercept) < 1e-9, (slope, intercept)
    print("  calibration slope of a perfect predictor is exactly 1.0")
    print("all checks passed")


if __name__ == "__main__":
    main()
