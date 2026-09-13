"""Gen14 step 5: the second round of direction candidates -- loss, transform, label, training set.

Round one settled the estimator family (a plain L2 logistic on the 39 topology columns, ahead of
gen13's trees under every design).  Round two asks whether anything is left in the *statistical*
choices around it, all of which are free:

* **training set** -- the classifier currently sees only well-determined cells (>= 5 metals).  The
  other 232 cells are noisier but they are data.
* **label** -- a cell's own direction, or the direction of its extractant's mean amplitude, which
  denoises the 38 % of cells whose coefficient is inside the measurement noise.
* **transform** -- the 39 columns take only 22 distinct values over 82 extractants, so a rank
  transform or a PCA in front of the linear model may match that resolution better than a z-score.
* **loss** -- hinge instead of log; L1 instead of L2 (which of 39 correlated columns survive?).
* **balancing** -- one vote per chemotype (locked) or one vote per extractant.

Usage:  python generations/gen14_direction/scripts/g14_sweep2.py [design,design,...]
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

#: name -> (candidate, feature set, train on well-determined cells only?)
CANDIDATES = {
    "ALWAYS_HEAVY":      (M.candidate(M.dir_always_heavy), "TOPO39", True),
    "G13_ET_TOPO39":     (M.candidate(M.dir_extratrees()), "TOPO39", True),
    "LOGIT_TOPO39":      (M.candidate(M.dir_logistic()), "TOPO39", True),
    "LOGIT_ALLCELLS":    (M.candidate(M.dir_logistic()), "TOPO39", False),
    "LOGIT_EXTLABEL":    (M.candidate(M.dir_logistic(label="extractant")), "TOPO39", True),
    "LOGIT_EXTLABEL_ALL": (M.candidate(M.dir_logistic(label="extractant")), "TOPO39", False),
    "LOGIT_L1":          (M.candidate(M.dir_logistic(penalty="l1")), "TOPO39", True),
    "LOGIT_L1_C03":      (M.candidate(M.dir_logistic(penalty="l1", C=0.3)), "TOPO39", True),
    "LOGIT_RANK":        (M.candidate(M.dir_logistic(transform="rank")), "TOPO39", True),
    "LOGIT_PCA5":        (M.candidate(M.dir_logistic(transform="pca", n_components=5)), "TOPO39", True),
    "LOGIT_PCA8":        (M.candidate(M.dir_logistic(transform="pca", n_components=8)), "TOPO39", True),
    "SVM_TOPO39":        (M.candidate(M.dir_svm()), "TOPO39", True),
    "LOGIT_C3":          (M.candidate(M.dir_logistic(C=3.0)), "TOPO39", True),
    "LOGIT_C10":         (M.candidate(M.dir_logistic(C=10.0)), "TOPO39", True),
}

boards, gains, t0 = [], [], time.time()
for design in DESIGNS:
    oofs = []
    for name, (fn, fs, rich) in CANDIDATES.items():
        o = db.run(bench, name, fn, features=FS[fs], design=design, rich_only_train=rich)
        oofs.append(o)
    boards.append(db.direction_board(oofs, baseline="ALWAYS_HEAVY"))
    by = {o.name: o for o in oofs}
    for ref in ("ALWAYS_HEAVY", "G13_ET_TOPO39", "LOGIT_TOPO39"):
        for cand in CANDIDATES:
            if cand != ref:
                gains.append(db.paired_gain(by[ref], by[cand]))
    print(f"[{design}] {time.time() - t0:.0f}s", flush=True)

board = pd.concat(boards, ignore_index=True)
gain = pd.DataFrame(gains)
board.to_csv(db.RESULTS / "g14_sweep2_board.csv", index=False)
gain.to_csv(db.RESULTS / "g14_sweep2_gains.csv", index=False)

cols = [d for d in db.DESIGNS if d in set(board.design)]
piv = board.pivot_table(index="model", columns="design", values="macro_accuracy")[cols]
piv["min"] = piv.min(axis=1)
print("\n=== macro accuracy, five designs ===")
print(piv.sort_values("min", ascending=False).round(4).to_string())

vs = gain[gain.reference == "LOGIT_TOPO39"]
p2 = vs.pivot_table(index="candidate", columns="design", values="gain")[cols]
p2["worst"] = p2.min(axis=1)
print("\n=== gain over LOGIT_TOPO39, five designs ===")
print(p2.sort_values("worst", ascending=False).round(4).to_string())
print(f"\ntotal {time.time() - t0:.0f}s")
