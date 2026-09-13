"""The block's best possible reading: single descriptors chosen with full-corpus hindsight.

``run_bench.py`` refits every choice inside the training fold, which is the honest thing and the
only thing that can be deployed.  This script does the opposite on purpose: it takes the four
descriptors that the *whole corpus* says are most associated with the targets --

    E_rms_A    residual rms of the composition-blocked energy fit -> b   (rho -0.317; the only
               column that clears the 28-column family-wise permutation bar at the chemotype level)
    d_spread   std of the Ln-donor distances, the preorganisation/strain proxy -> b   (rho -0.315)
    n_block    metals in the constant-composition block -> |a|  (rho +0.416, but partial rho given
               the number of metals the experiment measured is only +0.144)
    E_rms_B    residual rms of the count-controlled energy fit -> |a|  (rho +0.227)

-- and fits a one-dimensional monotone map from each to its target inside every fold.  Because the
column identity was picked on the full corpus, these scores are an OPTIMISTIC UPPER BOUND, not a
deployable result.  If the upper bound does not beat G14, the block is dead for that target.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))
from gen15 import arms as A, valuebench as V  # noqa: E402
from gen15.arms import _curve, _logistic_sign, _fit_magnitude_1d  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402
import arms_phys3d as P  # noqa: E402


def _col(ctx: Ctx, name: str) -> np.ndarray:
    P.cell_matrix(ctx, "P3_ALL")            # warms the per-bench cache
    return P._CACHE[id(ctx.bench)][name]


def mag_one(name: str, how: str = "isotonic"):
    def f(ctx: Ctx) -> np.ndarray:
        tr, w = ctx.rich_train(), ctx.rich_weights()
        v = _col(ctx, name)
        mag = _fit_magnitude_1d(v[tr], np.abs(ctx.amp[tr]), w, v[ctx.test], how)
        return _curve(_logistic_sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def curv_one(name: str):
    """Monotone (isotonic, both directions tried on the training loss) map from one column to b."""
    from sklearn.isotonic import IsotonicRegression

    def f(ctx: Ctx) -> np.ndarray:
        tr, w = ctx.rich_train(), ctx.rich_weights()
        v, b = _col(ctx, name), ctx.cur
        ok = np.isfinite(v[tr])
        if ok.sum() < 20:
            return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(),
                          ctx.train_mean_curvature(), len(ctx.test))
        fill = float(np.median(v[tr][ok]))
        vte = np.where(np.isfinite(v[ctx.test]), v[ctx.test], fill)
        best, best_loss = None, np.inf
        for inc in (True, False):
            iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
            iso.fit(v[tr][ok], b[tr][ok], sample_weight=w[ok])
            loss = float(np.average(np.abs(iso.predict(v[tr][ok]) - b[tr][ok]), weights=w[ok]))
            if loss < best_loss:
                best, best_loss = iso, loss
        lo, hi = np.percentile(b[tr], [5, 95])
        return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(),
                      np.clip(best.predict(vte), lo, hi), len(ctx.test))
    return f


ARMS = {
    "FLAT": A.flat,
    "G14": A.g14,
    "BEST_CURV_ERMSA": curv_one("E_rms_A"),
    "BEST_CURV_DSPREAD": curv_one("d_spread"),
    "BEST_MAG_NBLOCK": mag_one("n_block"),
    "BEST_MAG_ERMSB": mag_one("E_rms_B"),
}

comps = {f"{k}_vs_G14": ("G14", k) for k in ARMS if k not in ("G14", "FLAT")}
comps["G14_vs_FLAT"] = ("FLAT", "G14")

t0 = time.time()
bench = V.load()
B, C, tables = V.score(bench, ARMS, ["B", "BR", "BQ", "A", "BP"], comps=comps)
print(f"\ntotal {time.time() - t0:.0f}s")
for col in ("macro_mae_extractant", "macro_sign_acc_strong", "macro_pair_spearman"):
    print(f"\n=== {col} ===")
    print(V.wide(B, col).round(4).to_string())
print("\n=== chemotype-blocked paired bootstrap ===")
print(C.round(4).to_string(index=False))
B.to_csv(HERE / "board_bestshot.csv", index=False)
C.to_csv(HERE / "contrasts_bestshot.csv", index=False)
