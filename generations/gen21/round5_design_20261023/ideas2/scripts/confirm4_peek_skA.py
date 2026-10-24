#!/usr/bin/env python
"""Skeptic A for idea confirm-4 (C-ECT), step 1 (EXPLORATORY, already-seen 229 C3 cells / 5 dev seeds).

Read-only inspection of the confirm theme's P1 rebuild (ideas2/scripts/confirm_out/p1/P1_rows.parquet) and the V1 bench:
  - the PRIMARY units (pub x chemotype): extractant SMILES, DGA flag, crude mechanism class, change class, cpg cluster,
    cells, per-unit log SSE ratio (SH4 vs B1@4, SH3 vs B1@4, COPY vs B1@4);
  - the condition fields that differ between targets and siblings.
Writes nothing except stdout.
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
from r4 import donor as DN  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from gen20k import data as D  # noqa: E402

P1 = HERE.parent / "confirm_out" / "p1"


def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 90)
    t = pd.read_parquet(P1 / "P1_rows.parquet")
    print("rows", len(t), "cols", len(t.columns))
    print(sorted(t.columns))
    t = t[t.design == "BPG"].copy()
    prim = t[~t.t_v2drop & (t.n_clean >= 2) & ~t.c5].copy()
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    ref = M4.add_clusters(ref, None)
    cl = ref.groupby("cell_id").cluster_cpg.first()
    prim["cpg"] = prim.cell_id.map(cl)
    prim["unit"] = prim.publication + "|" + prim.chemotype
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    print(f.columns.tolist()[:80])
    ext = prim.groupby("unit").extractant.unique()
    rows = []
    for u, g in prim.groupby("unit"):
        a = g.groupby("split_seed").apply(lambda q: pd.Series({
            "er": ((q.B1_3 - q.y) ** 2).sum(), "sh4": ((q.CL_SH_3 - q.y) ** 2).sum(),
            "sh3": ((q.CL_SH_2 - q.y) ** 2).sum(), "copy": ((q.CL_COPY - q.y) ** 2).sum(),
            "sib": ((q.CL_SIB_3 - q.y) ** 2).sum(), "near": ((q.CL_NEAR_3 - q.y) ** 2).sum()})).mean()
        exs = list(ext[u])
        rows.append({"unit": u, "cells": g.cell_id.nunique(), "rows": len(g) / 5, "cpg": g.cpg.iloc[0],
                     "n_ext": len(exs), "dga": np.mean([DN.is_dga(e) for e in exs]),
                     "cls": ",".join(sorted(g.groupby("cell_id").cls.first().value_counts().index)),
                     "L_sh4": np.log(a["er"] / a["sh4"]), "L_sh3": np.log(a["er"] / a["sh3"]), "L_copy": np.log(a["er"] / a["copy"]),
                     "L_sib": np.log(a.er / a.sib), "L_near": np.log(a.er / a.near),
                     "ext0": exs[0][:80]})
    u = pd.DataFrame(rows).sort_values("L_sh4")
    print(u.round(3).to_string())
    for u_, g in prim.groupby("unit"):
        print(u_, sorted(set(g.extractant)))
    # condition columns per cell for each PRIMARY unit
    cond = [c for c in f.columns if c.startswith("cond__")]
    print(cond)
    cells = prim.cell_id.unique()
    ff = f[f.cell_id.isin(cells)].set_index("cell_id")
    for u_, g in prim.groupby("unit"):
        cc = sorted(g.cell_id.unique())
        x = ff.loc[cc, cond]
        x = x.loc[:, x.notna().any() & (x.nunique(dropna=False) > 1)]
        print("====", u_, len(cc))
        print(x.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
