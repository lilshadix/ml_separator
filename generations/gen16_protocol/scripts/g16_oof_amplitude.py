"""Out-of-fold predicted amplitude of the deployed gen14 arm, per cell, under every design.

Why this file exists
--------------------
``g16_variance.py`` can compute a variance-component table without it, but the *headline* it is
meant to produce -- the Nakagawa-Schielzeth **marginal** R2 of the deployed model on the cell
amplitude -- needs one column that the repository does not currently store anywhere: the model's
own out-of-fold prediction for each cell.  Every existing artefact stores either a *pairwise* MAE
(``g14_value_*.csv``, ``g16_bp1_board.csv``) or a *direction hit* (``unit_hits``), never the
predicted coefficient itself.

The distinction the table is for
--------------------------------
* ``R2_marginal``     variance of the *fixed* part (the model's prediction) over the total.  This is
  the fraction of amplitude variance that transfers to a new extractant in a new laboratory, and it
  is the honest answer to "does this model predict lanthanide separation".
* ``R2_conditional``  fixed + random.  This is what a design that lets the extractant, the
  publication or the diluent appear on both sides of the split is really reporting -- i.e. what
  design A measures.  Reporting the pair makes the leak a number instead of a caveat.
  (Nakagawa & Schielzeth, Methods Ecol. Evol. 4(2):133-142, 2013, doi:10.1111/j.2041-210x.2012.00261.x;
  Nakagawa, Johnson & Schielzeth, J. R. Soc. Interface 14(134):20170213, 2017.)

Two R2s are written, and they are **not** the same number:

``r2_oof_raw``    ``1 - SSE/SST`` of the raw out-of-fold prediction against the observed amplitude.
                  No parameter is fitted after the fact; this is what the deployed model actually
                  achieves and it can be negative.
``R2_marginal``   Nakagawa's, computed by ``g16_variance.py`` from a REML fit with the prediction as
                  a *fixed effect*, i.e. after an intercept and a slope have been refitted on the
                  full sample.  It is therefore an upper bound on ``r2_oof_raw`` -- the best any
                  affine recalibration of this prediction could do.  Quote both or neither.

Because the deployed arm is ``sign(logistic) x constant magnitude``, its prediction takes exactly
two values per fold, so its marginal R2 is by construction the variance explained by the *direction
bit alone*.  That is the point: gen14 says the bit is all that transfers, and this is that claim
stated on the variance scale rather than the MAE scale.

Usage:  python gen16_protocol/scripts/g16_oof_amplitude.py [designs...]
        (default: BP; pass e.g. ``BP A B BR BQ`` for the whole board)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen16_protocol"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen14 import models as M                                    # noqa: E402
from gen14.dirbench import DESIGNS, decision, feature_sets, load, run  # noqa: E402

OUT = ROOT / "gen16_protocol" / "results"
OUT.mkdir(parents=True, exist_ok=True)

#: the deployed arm and the two references the report already quotes next to it
ARMS = {
    "G14":         M.candidate(M.dir_logistic(), M.amp_constant),
    "DIR_ORACLE":  None,          # filled in below: true sign x the same constant magnitude
    "MEAN_CURVE":  None,          # the corpus mean amplitude (a single number, no direction)
}


def main(designs: tuple[str, ...]) -> None:
    bench = load()
    fs = feature_sets(bench)
    amp = bench.coef[:, 0]
    frames = []
    for design in designs:
        oof = run(bench, "G14", ARMS["G14"], features=fs["TOPO39"], design=design)
        c = oof.cells.copy()
        c["pred_amp"] = decision(c.p.to_numpy(), c.mag.to_numpy(), "hard")
        # DIR_ORACLE and MEAN_CURVE re-use the same folds and the same fold-mean magnitude, so the
        # three columns are directly comparable row by row.
        c["pred_amp_oracle_dir"] = np.where(c.amp.to_numpy() < 0, -1.0, 1.0) * c.mag.to_numpy()
        frames.append(c)

        per_cell = (c.groupby("cell_id")
                     .agg(pred_amp=("pred_amp", "mean"),
                          pred_amp_oracle_dir=("pred_amp_oracle_dir", "mean"),
                          p_heavy=("p", "mean"), mag=("mag", "mean"),
                          amp=("amp", "first"), n_metals=("n_metals", "first"),
                          extractant=("extractant", "first"),
                          chemotype=("chemotype", "first"),
                          n_pred=("pred_amp", "size"))
                     .reset_index())
        per_cell.to_csv(OUT / f"g16_oof_amplitude_{design}.csv", index=False)

        rich = per_cell.n_metals.to_numpy() >= 5
        y = per_cell.amp.to_numpy()[rich]
        # Two denominators, because they answer different questions and the programme should quote
        # both.  ``sst_mean`` is the statistical one: the reference predictor is the corpus mean
        # amplitude, i.e. the MEAN_CURVE arm, so R2 = 0 means "no better than the mean curve".
        # ``sst_zero`` is the chemical one: the reference is a *flat* curve, a = 0, no separation
        # at all.  The corpus mean amplitude is -0.25, not 0, so the two differ a lot here and a
        # single unlabelled "R2" would be ambiguous.
        sst_mean = float(((y - y.mean()) ** 2).sum())
        sst_zero = float((y ** 2).sum())
        rows = []
        for col, nm in (("pred_amp", "G14 (predicted direction x constant magnitude)"),
                        ("pred_amp_oracle_dir", "oracle direction x constant magnitude")):
            p = per_cell[col].to_numpy()[rich]
            sse = float(((y - p) ** 2).sum())
            # the affine-recalibrated ceiling for this same prediction: this, not r2_oof_raw, is
            # the quantity Nakagawa's R2_marginal reports, because REML refits intercept and slope
            A = np.c_[np.ones(len(p)), p]
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse_cal = float(((y - A @ beta) ** 2).sum())
            rows.append({"design": design, "arm": nm, "n_cells": int(rich.sum()),
                         "r2_vs_mean_curve": 1.0 - sse / sst_mean,
                         "r2_vs_flat_curve": 1.0 - sse / sst_zero,
                         "r2_affine_recalibrated": 1.0 - sse_cal / sst_mean,
                         "mae_amplitude": float(np.abs(y - p).mean()),
                         "sd_amplitude": float(y.std()),
                         "slope_refit": float(beta[1])})
        rows.append({"design": design, "arm": "corpus mean amplitude (MEAN_CURVE reference)",
                     "n_cells": int(rich.sum()), "r2_vs_mean_curve": 0.0,
                     "r2_vs_flat_curve": 1.0 - sst_mean / sst_zero,
                     "r2_affine_recalibrated": 0.0,
                     "mae_amplitude": float(np.abs(y - y.mean()).mean()),
                     "sd_amplitude": float(y.std()), "slope_refit": np.nan})
        rows.append({"design": design, "arm": "flat curve (a = 0, no separation)",
                     "n_cells": int(rich.sum()),
                     "r2_vs_mean_curve": 1.0 - sst_zero / sst_mean,
                     "r2_vs_flat_curve": 0.0, "r2_affine_recalibrated": 0.0,
                     "mae_amplitude": float(np.abs(y).mean()),
                     "sd_amplitude": float(y.std()), "slope_refit": np.nan})
        tab = pd.DataFrame(rows)
        print(f"\n=== {design} ===")
        print(tab.round(4).to_string(index=False))
        tab.to_csv(OUT / f"g16_oof_amplitude_r2_{design}.csv", index=False)

    pd.concat(frames, ignore_index=True).to_parquet(OUT / "g16_oof_amplitude_folds.parquet",
                                                    index=False)
    print(f"\nwrote g16_oof_amplitude_<design>.csv for {', '.join(designs)}")
    print("now re-run:  python gen16_protocol/scripts/g16_variance.py 300")


if __name__ == "__main__":
    args = tuple(a for a in sys.argv[1:] if a in DESIGNS) or ("BP",)
    main(args)
