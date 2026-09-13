"""L5 diagnostic: the conditioning of every covariance estimator on the REAL residual rows.

The measured-mode BLUP never inverts Sigma; it solves ``D Sigma D' + noise I``.  That is safe while
Sigma is positive semi-definite and *unsafe* when it is not, because greedy D-optimal selection
maximises ``(Sigma d)'(Sigma d) / (d' Sigma d + noise)`` -- a criterion that goes to +infinity
exactly along the directions where ``d' Sigma d -> -noise``.  An indefinite Sigma therefore does not
merely add noise: the support chooser actively seeks its worst direction.

This script measures, per design and per estimator, the eigenvalue spectrum of the covariance the
deployed route actually builds (leave-chemotype-out, publication-masked residual curves of the G14
arm) and the fraction of measurable pair contrasts ``d = e_a - e_b`` with ``d' Sigma d <= 0`` or
``<= -noise_var``.  It scores nothing and enters no contrast; it explains a number.

Usage:  .venv/Scripts/python.exe generations/gen16_leads/scripts/l5_condition.py [designs]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l5_cov as LC  # noqa: E402
from gen16 import l5_fewshot as L5  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen15.fewshot import NOISE_VAR, N_LN  # noqa: E402

OUT = bootstrap.RESULTS / "L5"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)

PAIRS = [(a, b) for a in range(N_LN) for b in range(a + 1, N_LN)]
D = np.zeros((len(PAIRS), N_LN))
for i, (a, b) in enumerate(PAIRS):
    D[i, a], D[i, b] = 1.0, -1.0


def main() -> None:
    t0 = time.time()
    bench = V.load()
    rows = []
    for d in DESIGNS:
        blocks = L5.residual_rows(bench, A.g14, d, mask_publication=True)
        usable = [(k, r, g) for k, r, g in blocks if r is not None]
        for name, fn in LC.ESTIMATORS.items():
            mins, maxs, negfrac, badfrac, traces, nrows = [], [], [], [], [], []
            for key, r, g in usable:
                C = fn(r, g)
                ev = np.linalg.eigvalsh(C)
                q = np.einsum("ij,jk,ik->i", D, C, D)
                mins.append(float(ev.min()))
                maxs.append(float(ev.max()))
                traces.append(float(np.trace(C)))
                negfrac.append(float((q <= 0).mean()))
                badfrac.append(float((q <= -NOISE_VAR).mean()))
                nrows.append(len(r))
            rows.append({"design": d, "estimator": name, "n_cov_keys": len(usable),
                         "median_rows_per_cov": float(np.median(nrows)),
                         "median_min_eig": float(np.median(mins)),
                         "worst_min_eig": float(np.min(mins)),
                         "frac_keys_indefinite": float(np.mean(np.asarray(mins) < 0)),
                         "median_max_eig": float(np.median(maxs)),
                         "median_trace": float(np.median(traces)),
                         "mean_frac_pairs_var_nonpositive": float(np.mean(negfrac)),
                         "mean_frac_pairs_var_below_minus_noise": float(np.mean(badfrac))})
        print(f"  [{d}] {len(usable)} covariance keys, {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "covariance_conditioning.csv", index=False)
    pd.set_option("display.width", 260)
    print("\n=== conditioning of each estimator on the real leave-chemotype-out rows ===")
    print(T.round(5).to_string(index=False))
    print(f"\n[l5-condition] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
