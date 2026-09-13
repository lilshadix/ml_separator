"""Gen14 step 2: does anything beat the plain logistic on 39 topology columns?

Five families of candidate, each a single change against ``LOGIT_TOPO39`` so the change is
attributable:

* **estimator** -- penalty strength chosen honestly inside the fold; a blend with gen13's trees.
* **label noise** -- 38 % of cells have a direction that is barely determined (|amp| < 0.1, a whole
  La->Lu contrast under 0.32 log units against a replicate sd of 0.113).  ``confidence`` weights
  those cells down without touching any chemotype's share of the fit.
* **training unit** -- 61 of the 82 scored extractants contribute one cell and one contributes 63;
  collapsing the training fold to one row per extractant matches the unit the score is computed on.
* **target** -- a ridge on the signed amplitude, squashed into a probability, uses the magnitude
  information a classifier discards.
* **representation** -- topology plus the donor-element counts, the gen6 census, the architecture
  family, and the mass-action columns (the last is the only way conditions can enter, and the only
  candidate that could explain the ten extractants whose own cells disagree about the direction).

Every candidate runs under all five designs; a candidate counts only if the sign is consistent.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen14 import dirbench as db
from gen14 import models as M

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(db.DESIGNS)
bench = db.load()
FS = db.feature_sets(bench)

CANDIDATES = {
    # yardsticks
    "ALWAYS_HEAVY":       (M.candidate(M.dir_always_heavy), "TOPO39"),
    "DONORS13_ET":        (M.candidate(M.dir_extratrees()), "DONORS13"),
    "G13_ET_TOPO39":      (M.candidate(M.dir_extratrees()), "TOPO39"),
    "LOGIT_TOPO39":       (M.candidate(M.dir_logistic()), "TOPO39"),
    # estimator
    "LOGIT_CV_TOPO39":    (M.candidate(M.dir_logistic_cv()), "TOPO39"),
    "LOGIT_C03_TOPO39":   (M.candidate(M.dir_logistic(C=0.3)), "TOPO39"),
    "LOGIT_C3_TOPO39":    (M.candidate(M.dir_logistic(C=3.0)), "TOPO39"),
    "BLEND_LOGIT_ET":     (M.candidate(M.dir_blend({"l": (M.dir_logistic(), 1.0),
                                                    "e": (M.dir_extratrees(), 1.0)})), "TOPO39"),
    # label noise
    "LOGIT_CONF_TOPO39":  (M.candidate(M.dir_logistic(weights="confidence")), "TOPO39"),
    "ET_CONF_TOPO39":     (M.candidate(M.dir_extratrees(weights="confidence")), "TOPO39"),
    # training unit
    "LOGIT_EXTUNIT":      (M.candidate(M.dir_logistic(units="extractant")), "TOPO39"),
    # target
    "RIDGESIGN_TOPO39":   (M.candidate(M.dir_from_regression()), "TOPO39"),
    "RIDGESIGN_CONF":     (M.candidate(M.dir_from_regression(weights="confidence")), "TOPO39"),
    # representation
    "LOGIT_TOPO_ELEM":    (M.candidate(M.dir_logistic()), "TOPO_ELEM"),
    "LOGIT_TOPO_DONORS":  (M.candidate(M.dir_logistic()), "TOPO_DONORS"),
    "LOGIT_TOPO_ARCH":    (M.candidate(M.dir_logistic()), "TOPO_ARCH"),
    "LOGIT_TOPO_MASSACT": (M.candidate(M.dir_logistic()), "TOPO_MASSACT"),
    "LOGIT_COORD114":     (M.candidate(M.dir_logistic()), "COORD114"),
}

REFERENCES = ("ALWAYS_HEAVY", "DONORS13_ET", "G13_ET_TOPO39", "LOGIT_TOPO39")

boards, gains, t0 = [], [], time.time()
for design in DESIGNS:
    oofs = []
    for name, (fn, fs) in CANDIDATES.items():
        o = db.run(bench, name, fn, features=FS[fs], design=design)
        oofs.append(o)
        print(f"  {design:3s} {name:20s} {o.seconds:5.1f}s", flush=True)
    boards.append(db.direction_board(oofs, baseline="ALWAYS_HEAVY"))
    by = {o.name: o for o in oofs}
    for ref in REFERENCES:
        for cand in CANDIDATES:
            if cand != ref:
                gains.append(db.paired_gain(by[ref], by[cand]))
    print(f"[{design}] {time.time() - t0:.0f}s", flush=True)

board = pd.concat(boards, ignore_index=True)
gain = pd.DataFrame(gains)
db.RESULTS.mkdir(parents=True, exist_ok=True)
board.to_csv(db.RESULTS / "g14_sweep_board.csv", index=False)
gain.to_csv(db.RESULTS / "g14_sweep_gains.csv", index=False)

piv = board.pivot_table(index="model", columns="design", values="macro_accuracy")
cols = [d for d in db.DESIGNS if d in piv.columns]
piv = piv[cols]
piv["min"] = piv.min(axis=1)
print("\n=== macro accuracy, five designs ===")
print(piv.sort_values("min", ascending=False).round(4).to_string())

vs = gain[gain.reference == "LOGIT_TOPO39"]
if not vs.empty:
    p2 = vs.pivot_table(index="candidate", columns="design", values="gain")[cols]
    p2["worst"] = p2.min(axis=1)
    print("\n=== gain over LOGIT_TOPO39 (positive = better), five designs ===")
    print(p2.sort_values("worst", ascending=False).round(4).to_string())
print(f"\ntotal {time.time() - t0:.0f}s")
