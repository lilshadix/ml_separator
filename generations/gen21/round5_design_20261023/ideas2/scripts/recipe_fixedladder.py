#!/usr/bin/env python
"""Theme "minimal-measurement recipes": score the FIXED static metal ladder that the LOPO rule picked most often on
the C5-free corpus (2: La,Ho; 3: La,Nd,Ho; 4: La,Nd,Er,Lu; 5: La,Nd,Gd,Er,Lu) -- frozen here from
recipe_static_legal.log, so this is in-sample for the noC5 corpus and a transfer check only for the own-prior arm.
A cell that lacks a ladder metal falls back to HULL.  Compared with HULL and with DOPT_Q-nested sets (V1 of
recipe_doptq_var).  EXPLORATORY.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_fixedladder.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
import recipe_analyze as RA  # noqa: E402

OUT = RA.OUT
LADDER = {2: "La,Ho", 3: "La,Nd,Ho", 4: "La,Nd,Er,Lu", 5: "La,Nd,Gd,Er,Lu"}


def main() -> int:
    cells = pd.read_parquet(OUT / "recipe_cells.parquet")
    pairs = pd.read_parquet(OUT / "recipe_pairs.parquet")
    pops = {"noC5_214": cells[~cells.c5], "OWN1_noC5": cells[(cells.n_own >= 1) & ~cells.c5],
            "OWN1_C5": cells[cells.n_own >= 1], "ALL229": cells}
    rows = []
    for pk, sub in pops.items():
        P = RA.Pop(sub, pk)
        idx = sub.ci.to_numpy()
        for n, s in LADDER.items():
            z = np.load(OUT / f"recipe_k{n}.npz")
            q = {c: i for i, c in enumerate(itertools.combinations(range(N_LN), n))}[
                tuple(sorted(LANTHANIDES.index(m) for m in s.split(",")))]
            for arm in (("D7", "SH_own") if pk.startswith("OWN") else ("D7",)):
                hull = pairs[pairs.n_read == n].groupby("ci")[f"hull_{arm}"].sum().reindex(idx).fillna(0).to_numpy()
                v = z[f"sse_{arm}"][idx, q]
                has = ~np.isnan(v)
                fx = np.where(has, v, hull)
                d = P.summ(fx, hull)
                rows.append({"pop": pk, "arm": arm, "n": n, "set": s, "coverage": float(has.mean()),
                             "FIXED": d["R2"], "lo": d["lo"], "hi": d["hi"], "HULL": float(P.r2(hull)),
                             "d_vs_HULL": d["d"], "d_lo": d["d_lo"], "d_hi": d["d_hi"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "recipe_fixedladder.csv", index=False)
    with pd.option_context("display.width", 250):
        print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
