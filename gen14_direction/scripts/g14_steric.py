"""Gen14 step 7 (EXPLORATORY, post-hoc): can a steric block fix what donor topology cannot see?

**This candidate was chosen after looking at held-out errors and is labelled accordingly.**  The
error analysis under BP found that the direction model is right on 40 of the 43 strongly directed
extractants (macro 0.920 there), and that two of the three failures are the same motif: a
diglycolamide carrying alkyl substituents on the ether backbone
(``...C(=O)[C@H](CCC)O[C@H](CCC)C(=O)N...``), which is *light*-selective at amplitude +0.30 and
+0.36 while its 39 topology columns are bit-identical to an ordinary, strongly heavy-selective DGA.
Bond counts between donor atoms cannot see a substituent hanging off the chelate ring, and steric
crowding next to the donors is a standard explanation for a reversed lanthanide preference.

Hypothesis, stated before the run: adding a small number of *steric* columns -- how much carbon
sits per donor, how branched the skeleton is, how symmetric it is -- to the topology block should
raise accuracy on strongly directed extractants without diluting the block, where the whole
25-column architecture family (tested in round 1) did dilute it.

Rules kept: all five designs, the full table reported, and the increment stated against
``LOGIT_TOPO39`` as well as against the constant baseline.  A post-hoc candidate that passes here
is a hypothesis for the next cohort, not a result.

Usage:  python gen14_direction/scripts/g14_steric.py [design,design,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))
from gen14 import dirbench as db
from gen14 import models as M
from gen13sep.amplitude_bench import LEAN_BLOCKS

DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(db.DESIGNS)
STERIC = ["coord__arch__carbon_per_donor", "coord__arch__heavy_atoms_per_donor",
          "coord__arch__mean_heavy_degree", "coord__arch__n_degree3_atoms",
          "coord__arch__n_degree4_atoms", "coord__arch__fraction_csp3",
          "coord__arch__frac_atoms_in_largest_orbit"]

bench = db.load()
FS = db.feature_sets(bench)
lean = bench.columns(LEAN_BLOCKS)
idx = {c: i for i, c in enumerate(lean)}
missing = [c for c in STERIC if c not in idx]
if missing:
    raise KeyError(f"missing steric columns: {missing}")
FS["STERIC7"] = np.array([idx[c] for c in STERIC], dtype=int)
FS["TOPO_STERIC"] = np.concatenate([FS["TOPO39"], FS["STERIC7"]])

CANDIDATES = {
    "ALWAYS_HEAVY":  (M.candidate(M.dir_always_heavy), "TOPO39"),
    "LOGIT_TOPO39":  (M.candidate(M.dir_logistic()), "TOPO39"),
    "LOGIT_STERIC7": (M.candidate(M.dir_logistic()), "STERIC7"),
    "LOGIT_TOPO_STERIC": (M.candidate(M.dir_logistic()), "TOPO_STERIC"),
    "ET_TOPO_STERIC": (M.candidate(M.dir_extratrees()), "TOPO_STERIC"),
}

boards, gains, strong, t0 = [], [], [], time.time()
for design in DESIGNS:
    oofs = [db.run(bench, n, fn, features=FS[fs], design=design) for n, (fn, fs) in CANDIDATES.items()]
    boards.append(db.direction_board(oofs, baseline="ALWAYS_HEAVY"))
    by = {o.name: o for o in oofs}
    for cand in CANDIDATES:
        if cand != "LOGIT_TOPO39":
            gains.append(db.paired_gain(by["LOGIT_TOPO39"], by[cand]))
    # the band the hypothesis is about
    for name, o in by.items():
        b = o.cells[o.cells.n_metals >= db.MIN_METALS].copy()
        b["hit"] = ((b.p >= 0.5).astype(int) == b.y).astype(float)
        u = b.groupby(["extractant", "chemotype"]).agg(hit=("hit", "mean"),
                                                       absamp=("amp", lambda v: v.abs().mean())).reset_index()
        s = u[u.absamp >= 0.2]
        strong.append({"design": design, "model": name, "n_strong_extractants": len(s),
                       "macro_accuracy_strong": float(s.hit.mean()),
                       "macro_accuracy_weak": float(u[u.absamp < 0.2].hit.mean())})
    print(f"[{design}] {time.time() - t0:.0f}s", flush=True)

board = pd.concat(boards, ignore_index=True)
gain = pd.DataFrame(gains)
st = pd.DataFrame(strong)
board.to_csv(db.RESULTS / "g14_steric_board.csv", index=False)
gain.to_csv(db.RESULTS / "g14_steric_gains.csv", index=False)
st.to_csv(db.RESULTS / "g14_steric_strong.csv", index=False)

cols = [d for d in db.DESIGNS if d in set(board.design)]
print("\n=== macro accuracy, five designs (POST-HOC candidate) ===")
print(board.pivot_table(index="model", columns="design", values="macro_accuracy")[cols].round(4).to_string())
print("\n=== gain over LOGIT_TOPO39 ===")
print(gain.pivot_table(index="candidate", columns="design", values="gain")[cols].round(4).to_string())
print("\n=== the band the hypothesis is about: mean |amplitude| >= 0.2 ===")
print(st.pivot_table(index="model", columns="design", values="macro_accuracy_strong")[cols].round(4).to_string())
print("\n=== the rest ===")
print(st.pivot_table(index="model", columns="design", values="macro_accuracy_weak")[cols].round(4).to_string())
print(f"\ntotal {time.time() - t0:.0f}s")
