#!/usr/bin/env python
"""Skeptic B for confirm-4, check 3 (EXPLORATORY, seen rows): the non-DGA evidence for SH > B1 at 4 reads in the
whole CLEAN population (any n_sib, with and without C5-BTBP), per pub x chemotype unit, with the D-range of the
target cells (max - min finite log D over the target's metals) as the chemistry covariate that decides whether a
full La-Lu series is measurable at one condition.  Writes only ideas2/confirm4_skepB_out/nondga_units.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm4_skepB_out"
sys.path.insert(0, str(HERE.parents[3] / "round4_20261022"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from r4 import paths  # noqa: E402,F401
from r4 import donor as DN  # noqa: E402
from gen20k import data as D  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
    rng_ = {c: float(np.nanmax(y) - np.nanmin(y)) for c, y in zip(f.cell_id.astype(str), Y)}
    t = pd.read_parquet(HERE.parent / "confirm_out" / "p1" / "P1_rows.parquet")
    t = t[(t.design == "BPG") & ~t.t_v2drop & (t.n_clean >= 1)].copy()
    t["dga"] = t.extractant.astype(str).map(DN.is_dga)
    t["unit"] = t.publication + "|" + t.chemotype
    rows = []
    for u, g in t.groupby("unit"):
        s = g.assign(er=(g.B1_3 - g.y) ** 2, e4=(g.CL_SH_3 - g.y) ** 2, ec=(g.CL_COPY - g.y) ** 2).groupby("split_seed")[["er", "e4", "ec"]].sum().mean()
        cells = g.cell_id.unique()
        rows.append({"unit": u, "dga": bool(g.dga.iloc[0]), "c5": bool(g.c5.iloc[0]), "cells": len(cells),
                     "nsib_med": g.drop_duplicates("cell_id").n_clean.median(), "cls": g.drop_duplicates("cell_id").cls.mode().iloc[0],
                     "logD_range_med": float(np.median([rng_[c] for c in cells])),
                     "L_SH4": float(np.log(s.er / s.e4)), "L_COPY": float(np.log(s.er / s.ec))})
    R = pd.DataFrame(rows).sort_values(["dga", "L_SH4"])
    R.to_csv(OUT / "nondga_units.csv", index=False)
    print(R.round(3).to_string())
    for lab, q in (("nonDGA all", R[~R.dga]), ("nonDGA no C5", R[~R.dga & ~R.c5]), ("DGA", R[R.dga])):
        print(f"{lab:14s} units {len(q):2d} SH4 W/L {(q.L_SH4 > 0).sum()}/{(q.L_SH4 < 0).sum()} mean {q.L_SH4.mean():+.3f} median {q.L_SH4.median():+.3f}"
              f"  logD range median {q.logD_range_med.median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
