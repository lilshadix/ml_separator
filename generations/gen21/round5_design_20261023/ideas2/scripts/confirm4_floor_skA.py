#!/usr/bin/env python
"""Skeptic A for idea confirm-4 (C-ECT), step 5 (EXPLORATORY): V2-5 floor readings (kept in P1 and in step 3).
Which PRIMARY targets / siblings carry a V2-5 floor reading, and does dropping every scored row whose pair touches a
floor metal of the target, and every target whose siblings carry floor readings at the target's metals, change the unit
signs of SH4 vs B1@4?  Reads confirm4_skA_out/rows.parquet; writes stdout only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = HERE.parent / "confirm4_skA_out"


def main() -> int:
    t = pd.read_parquet(OUT / "rows.parquet")
    t = t[(t.design == "BPG") & (t.n_sib >= 2) & ~t.c5].copy()
    t["pubchem"] = t.publication + "|" + t.chemotype
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    pubof = dict(zip(f.cell_id.astype(str), f.publication_id.astype(str)))
    extof = dict(zip(f.cell_id.astype(str), f.extractant.astype(str)))
    fl = V2.V2_5_FLOOR
    print("floor cells -> publication, in PRIMARY targets?")
    for c, m in fl.items():
        print(c, pubof.get(c), m, "target" if c in set(t.cell_id) else "-")
    # siblings carrying floor readings: same pub & extractant as a PRIMARY target
    tgt_with_floor_sib = set()
    for c in fl:
        for tc in t.cell_id.unique():
            if tc != c and pubof[tc] == pubof[c] and extof[tc] == extof[c]:
                tgt_with_floor_sib.add(tc)
    print("PRIMARY targets whose publication x extractant contains a floor cell:", len(tgt_with_floor_sib))
    bad = t.apply(lambda r: r.cell_id in fl and (r.A in fl[r.cell_id] or r.B in fl[r.cell_id]), axis=1)
    print("scored PRIMARY rows touching a target floor metal:", int(bad.sum()), "of", len(t))
    for name, tt in (("all", t), ("drop floor rows", t[~bad]),
                     ("drop floor rows + targets with floor sibling", t[~bad & ~t.cell_id.isin(tgt_with_floor_sib)])):
        g = tt.assign(er=(tt.B1_4 - tt.y) ** 2, ec=(tt.SH_4 - tt.y) ** 2)
        a = g.groupby(["pubchem", "split_seed"])[["er", "ec"]].sum().groupby(level=0).mean()
        L = np.log(a.er / a.ec)
        print(f"{name}: units {len(L)}, W {int((L > 0).sum())} / L {int((L < 0).sum())}, mean {L.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
