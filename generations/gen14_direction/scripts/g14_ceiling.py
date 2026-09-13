"""Gen14 step 6: how much of the remaining direction error can any model remove?

Three ceilings, all measurable on the frozen cohort:

1. **Label noise.**  A cell's direction is the sign of a coefficient estimated from its own
   measured metals.  The data audit puts the replicate noise floor of a single ``log D`` at a
   median sd of 0.237, and 38 % of cells have |amplitude| < 0.1, so for a large minority of cells
   the *observed* label is a coin flip and no model can agree with it better than chance.  The
   standard error of the coefficient is propagated through the same ridge that defines it, and the
   probability that a perfect model disagrees with the observed label is ``Phi(-|a| / se)``.
2. **Extractant consistency.**  Ten of the 21 extractants with more than one well-determined cell
   contain cells of both signs; the features are constant within an extractant, so no ligand-only
   model can be right on all of them.  The best any such model can do is the majority within each
   extractant.
3. **Confidence.**  What the model is worth if it is allowed to abstain -- accuracy against
   coverage, ordered by |2p - 1|, which is the form the result would be deployed in.

Usage:  python generations/gen14_direction/scripts/g14_ceiling.py [design]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))
from gen14 import dirbench as db
from gen14 import models as M
from gen13sep.basis import DEFAULT_RIDGE

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
REPLICATE_SD = 0.237        # DATA_AUDIT section 3: median within-replicate sd of log D

bench = db.load()
FS = db.feature_sets(bench)
AMP = bench.coef[:, 0]
RICH = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
BASIS = bench.basis
frame = bench.frame


def coefficient_se() -> pd.DataFrame:
    """Standard error of each cell's radius coefficient, from the ridge that defines it.

    ``a = (B'B + lam I)^-1 B' y`` on the cell's observed metals, so
    ``var(a) = sigma^2 (B'B + lam I)^-1 B'B (B'B + lam I)^-1``.  Two values of ``sigma``: the
    cell's own fit residual (which also absorbs any misfit of the two-term basis) and the corpus
    replicate floor.
    """
    rows = []
    centred = bench.Y - np.nanmean(bench.Y, axis=1, keepdims=True)
    for i in range(len(frame)):
        obs = np.flatnonzero(~np.isnan(centred[i]))
        if len(obs) < 3:
            rows.append({"se_fit": np.nan, "se_floor": np.nan, "sigma_fit": np.nan})
            continue
        B = BASIS[:, obs].T
        G = B.T @ B
        A = np.linalg.inv(G + DEFAULT_RIDGE * np.eye(BASIS.shape[0]))
        cov_unit = A @ G @ A
        resid = centred[i, obs] - B @ bench.coef[i]
        dof = max(len(obs) - BASIS.shape[0], 1)
        sigma_fit = float(np.sqrt((resid ** 2).sum() / dof))
        rows.append({"se_fit": float(sigma_fit * np.sqrt(cov_unit[0, 0])),
                     "se_floor": float(REPLICATE_SD * np.sqrt(cov_unit[0, 0])),
                     "sigma_fit": sigma_fit})
    return pd.DataFrame(rows, index=frame.index)


se = coefficient_se()
cells = pd.DataFrame({"extractant": frame.extractant, "chemotype": frame.chemotype,
                      "n_metals": frame.n_metals, "amp": AMP,
                      "se_fit": se.se_fit, "se_floor": se.se_floor})[RICH].copy()
for tag in ("fit", "floor"):
    cells[f"p_flip_{tag}"] = norm.cdf(-np.abs(cells.amp) / cells[f"se_{tag}"])

print(f"well-determined cells {len(cells)}  extractants {cells.extractant.nunique()}")
print(f"median se(a): fit {cells.se_fit.median():.3f}  replicate-floor {cells.se_floor.median():.3f}"
      f"   median |amp| {cells.amp.abs().median():.3f}")

# --- ceiling 1: a perfect model still disagrees with a noisy label
ceil = {}
for tag in ("fit", "floor"):
    per_unit = (1 - cells[f"p_flip_{tag}"]).groupby(cells.extractant).mean()
    ceil[tag] = float(per_unit.mean())
    print(f"label-noise ceiling ({tag:5s} sigma): macro {ceil[tag]:.4f}  "
          f"(cells with p_flip > 0.25: {(cells[f'p_flip_{tag}'] > 0.25).mean():.1%})")

# --- ceiling 2: any model constant within an extractant
maj = cells.assign(heavy=(cells.amp < 0).astype(float)).groupby("extractant")["heavy"].mean()
ceil_ext = float(np.maximum(maj, 1 - maj).mean())
print(f"extractant-constancy ceiling: macro {ceil_ext:.4f}  "
      f"({int((maj.between(0.001, 0.999)).sum())} of {len(maj)} extractants are sign-mixed)")

# --- the model, banded by how well determined the label is and by its own confidence
oof = db.run(bench, "LOGIT_TOPO39", M.candidate(M.dir_logistic()), features=FS["TOPO39"],
             design=DESIGN)
b = oof.cells[oof.cells.n_metals >= db.MIN_METALS].copy()
b["hit"] = ((b.p >= 0.5).astype(int) == b.y).astype(float)
b["conf"] = (2 * b.p - 1).abs()
b = b.merge(cells.reset_index()[["extractant", "amp", "p_flip_fit", "p_flip_floor"]]
            .drop_duplicates(["extractant", "amp"]), on=["extractant", "amp"], how="left")

bands = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 99)]
rows = []
for lo, hi in bands:
    m = (b.amp.abs() >= lo) & (b.amp.abs() < hi)
    if not m.any():
        continue
    unit = b[m].groupby(["extractant", "chemotype"])["hit"].mean()
    rows.append({"band": f"{lo}-{hi}", "n_cells": int(m.sum()),
                 "n_extractants": int(b[m].extractant.nunique()),
                 "macro_accuracy": float(unit.mean()),
                 "ceiling_fit": float(1 - b[m].p_flip_fit.mean()),
                 "ceiling_floor": float(1 - b[m].p_flip_floor.mean())})
band = pd.DataFrame(rows)
print(f"\n=== {DESIGN}: accuracy against how well the label is determined ===")
print(band.round(4).to_string(index=False))

rows = []
for cov in (1.0, 0.9, 0.75, 0.5, 0.25):
    thr = b.conf.quantile(1 - cov)
    m = b.conf >= thr
    unit = b[m].groupby(["extractant", "chemotype"])["hit"].mean()
    rows.append({"coverage": cov, "conf_threshold": float(thr), "n_cells": int(m.sum()),
                 "n_extractants": int(b[m].extractant.nunique()),
                 "macro_accuracy": float(unit.mean()),
                 "pooled_accuracy": float(b[m].hit.mean()),
                 "mean_abs_amp": float(b[m].amp.abs().mean())})
cover = pd.DataFrame(rows)
print(f"\n=== {DESIGN}: accuracy if the model may abstain (ordered by |2p-1|) ===")
print(cover.round(4).to_string(index=False))

summary = pd.DataFrame([{"design": DESIGN, "model_macro": float(
    b.groupby(["extractant", "chemotype"])["hit"].mean().mean()),
    "ceiling_label_noise_fit": ceil["fit"], "ceiling_label_noise_floor": ceil["floor"],
    "ceiling_extractant_constancy": ceil_ext,
    "n_cells": len(b), "n_extractants": int(b.extractant.nunique())}])
print()
print(summary.round(4).to_string(index=False))
db.RESULTS.mkdir(parents=True, exist_ok=True)
band.to_csv(db.RESULTS / f"g14_ceiling_bands_{DESIGN}.csv", index=False)
cover.to_csv(db.RESULTS / f"g14_ceiling_coverage_{DESIGN}.csv", index=False)
summary.to_csv(db.RESULTS / f"g14_ceiling_summary_{DESIGN}.csv", index=False)
