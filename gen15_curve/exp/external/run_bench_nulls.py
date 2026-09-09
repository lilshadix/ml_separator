"""Null arms shaped exactly like LOGK_DIR, to price its apparent gain.

``LOGK_PERM_k``  the same logistic fitted on the 273 logK ligands with their labels PERMUTED.
                 Same estimator, same feature block, same external structures, no external label
                 information.  Five seeds.
``RAND_LIGHT``   calls a fixed random 20 % of extractants light -- LOGK_DIR's own prevalence with
                 no structure at all.  Prices the "hedging" explanation of a gain over ALWAYS_HEAVY.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))

from gen15 import valuebench as V, arms as A       # noqa: E402
from gen15.valuebench import Ctx                   # noqa: E402
import extarms as X                                # noqa: E402


@lru_cache(maxsize=8)
def _perm_model(seed: int):
    Xk, yk, _ = X._external()
    rng = np.random.default_rng(seed)
    m = X._pipe()
    m.fit(Xk, rng.permutation(yk), logisticregression__sample_weight=np.ones(len(yk)))
    return m


def logk_perm(seed: int):
    def f(ctx: Ctx) -> np.ndarray:
        p = _perm_model(seed).predict_proba(ctx.X[:, ctx.fs["TOPO39"]][ctx.test])[:, 1]
        return np.c_[np.where(p >= 0.5, -1.0, 1.0) * ctx.train_mean_magnitude(),
                     np.full(len(ctx.test), ctx.train_mean_curvature())]
    return f


def rand_light(seed: int, frac: float = 0.195):
    """A fixed random subset of *extractants* called light; deterministic in the extractant."""
    def f(ctx: Ctx) -> np.ndarray:
        ext = ctx.bench.frame.extractant.to_numpy()
        uniq = np.array(sorted(set(ext)))
        rng = np.random.default_rng(seed)
        light = set(rng.choice(uniq, size=int(round(frac * len(uniq))), replace=False))
        sign = np.array([1.0 if e in light else -1.0 for e in ext[ctx.test]])
        return np.c_[sign * ctx.train_mean_magnitude(),
                     np.full(len(ctx.test), ctx.train_mean_curvature())]
    return f


ARMS = {"FLAT": A.flat, "ALWAYS_HEAVY": X.always_heavy, "G14": A.g14,
        "LOGK_DIR": X.logk_dir()}
ARMS.update({f"LOGK_PERM_{k}": logk_perm(k) for k in range(5)})
ARMS.update({f"RAND_LIGHT_{k}": rand_light(100 + k) for k in range(3)})

COMPS = {f"LOGK_DIR_vs_PERM_{k}": (f"LOGK_PERM_{k}", "LOGK_DIR") for k in range(5)}
COMPS.update({f"LOGK_DIR_vs_RAND_{k}": (f"RAND_LIGHT_{k}", "LOGK_DIR") for k in range(3)})


def main() -> None:
    bench = V.load()
    B, C, _ = V.score(bench, ARMS, ["B", "BR", "BQ", "A", "BP"], comps=COMPS)
    B.to_csv(HERE / "bench_nulls_board.csv", index=False)
    C.to_csv(HERE / "bench_nulls_contrasts.csv", index=False)
    for value in ("macro_mae_extractant", "macro_sign_acc_strong", "macro_pair_spearman"):
        w = V.wide(B, value)
        print(f"\n=== {value} ===")
        print(w.round(4).to_string())
        w.round(6).to_csv(HERE / f"nulls_wide_{value}.csv")
    print("\n=== contrasts ===")
    keep = ["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
            "loco_sign_stable", "units_improved"]
    print(C[keep].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
