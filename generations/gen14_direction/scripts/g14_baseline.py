"""Gen14 step 1: reproduce gen13's direction result on the new bench, then run the five designs.

Two things have to be true before any gen14 candidate is believable:

* the bench reproduces the locked stage-3 number exactly -- gen13's 400 extremely randomised trees
  on the 39 donor-topology columns must score 0.7683085207475452 macro accuracy under design BP;
* every model is scored under all five hold-out designs at once (protocol Addendum 3), and against
  the *cheapest sensible alternative* (the 13-column gen6 donor census), not only against the
  constant "always heavy-selective" rule.

Usage:  python generations/gen14_direction/scripts/g14_baseline.py [design,design,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen14 import dirbench as db
from gen14 import models as M

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(db.DESIGNS)
LOCKED_BP = 0.7683085207475452          # gen13 stage-3, s3_direction_accuracy_FIVE_DESIGNS.csv

bench = db.load()
FS = db.feature_sets(bench)
print(f"cells {len(bench.frame)}  extractants {bench.frame.extractant.nunique()}  "
      f"chemotypes {bench.frame.chemotype.nunique()}", flush=True)

CANDIDATES = {
    "ALWAYS_HEAVY":    (M.candidate(M.dir_always_heavy), "TOPO39"),
    "TRAIN_MAJORITY":  (M.candidate(M.dir_train_majority), "TOPO39"),
    "DONORS13_ET":     (M.candidate(M.dir_extratrees()), "DONORS13"),
    "DONORS13_LOGIT":  (M.candidate(M.dir_logistic()), "DONORS13"),
    "BITE6_LOGIT":     (M.candidate(M.dir_logistic()), "BITE6"),
    "G13_ET_TOPO39":   (M.candidate(M.dir_extratrees()), "TOPO39"),
    "LOGIT_TOPO39":    (M.candidate(M.dir_logistic()), "TOPO39"),
    "LEAN209_ET":      (M.candidate(M.dir_extratrees()), "LEAN209"),
    "CHEM137_ET":      (M.candidate(M.dir_extratrees()), "CHEM137"),
}

boards, gains, t0 = [], [], time.time()
for design in DESIGNS:
    oofs = []
    for name, (fn, fs) in CANDIDATES.items():
        o = db.run(bench, name, fn, features=FS[fs], design=design)
        oofs.append(o)
        print(f"  {design:3s} {name:16s} {o.seconds:5.1f}s", flush=True)
    board = db.direction_board(oofs, baseline="ALWAYS_HEAVY")
    boards.append(board)
    by = {o.name: o for o in oofs}
    for ref in ("ALWAYS_HEAVY", "DONORS13_ET", "DONORS13_LOGIT", "G13_ET_TOPO39"):
        for cand in CANDIDATES:
            if cand != ref:
                gains.append(db.paired_gain(by[ref], by[cand]))
    print(f"\n--- design {design} ---")
    print(board[["model", "macro_accuracy", "ci_low", "ci_high", "gain", "gain_lo", "gain_hi",
                 "p_two_sided"]].round(4).to_string(index=False), flush=True)
    if design == "BP":
        got = float(board.loc[board.model == "G13_ET_TOPO39", "macro_accuracy"].iat[0])
        ok = abs(got - LOCKED_BP) < 1e-9
        print(f"\n[reproduction] G13_ET_TOPO39 @ BP = {got!r}  locked {LOCKED_BP!r}  "
              f"{'MATCH' if ok else 'MISMATCH'}", flush=True)

db.RESULTS.mkdir(parents=True, exist_ok=True)
pd.concat(boards, ignore_index=True).to_csv(db.RESULTS / "g14_baseline_board.csv", index=False)
pd.DataFrame(gains).to_csv(db.RESULTS / "g14_baseline_gains.csv", index=False)
print(f"\ntotal {time.time() - t0:.0f}s -> results/g14_baseline_board.csv")
