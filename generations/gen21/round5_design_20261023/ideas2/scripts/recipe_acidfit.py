#!/usr/bin/env python
"""Theme "minimal-measurement recipes": ICP-MS condition skipping with a per-pair linear trend in log10[acid]
fitted over ALL of a campaign's acid-only sibling series (EXPLORATORY; already-seen cells; C5-BTBP out).

Population: C3 targets with >= 2 CLEAN own siblings that differ from the target only in acid concentration (the same
rule as recipe_extra E3).  Arms at 0 reads (no experiment at the new acid level):
  MEAN    mean of the acid-only siblings' log SF
  LIN     OLS line of log SF on log10[acid] over the siblings, evaluated at the target acid
  LINCLIP LIN, but the target acid clamped to the siblings' acid range (no extrapolation)
  LINHALF MEAN + 0.5 x (LIN - MEAN)  (slope shrunk by half)
Interior = target acid inside the siblings' range; exterior = outside.  Publication bootstrap (2,000).

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_acidfit.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC  # noqa: E402
import recipe_analyze as RA  # noqa: E402

OUT = RA.OUT


def main() -> int:
    cells = pd.read_parquet(OUT / "recipe_cells.parquet")
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    acid = f["cond__acid_concentration_M"].to_numpy(float)
    rows = []
    for r in cells.itertuples():
        if r.n_own < 2 or r.c5:
            continue
        own = json.loads(r.own_ids)
        fams = json.loads(r.own_fams)
        js = [pos[j] for j, fm in zip(own, fams) if fm == "acid_conc" and np.isfinite(acid[pos[j]]) and acid[pos[j]] > 0]
        i = pos[r.cell_id]
        if len(js) < 2 or not np.isfinite(acid[i]) or acid[i] <= 0:
            continue
        la = np.log10(acid[js])
        if np.ptp(la) < 1e-9:
            continue
        lt = math.log10(acid[i])
        inside = la.min() - 1e-9 <= lt <= la.max() + 1e-9
        Mi = [LANTHANIDES.index(s) for s in r.metals.split(",")]
        for p in range(len(Mi) - 1):
            a, b = Mi[p], Mi[p + 1]
            if Z_VEC[b] - Z_VEC[a] != 1:
                continue
            v = Y[js, a] - Y[js, b]
            y = Y[i, a] - Y[i, b]
            bb = np.polyfit(la, v, 1)
            lin = float(np.polyval(bb, lt))
            linc = float(np.polyval(bb, min(max(lt, la.min()), la.max())))
            mean = float(v.mean())
            rows.append({"cell_id": r.cell_id, "publication": r.publication, "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[b]}",
                         "inside": inside, "n_sib": len(js), "n_levels": len(np.unique(np.round(la, 6))),
                         "y": y, "MEAN": mean, "LIN": lin, "LINCLIP": linc, "LINHALF": mean + 0.5 * (lin - mean)})
    t = pd.DataFrame(rows)
    t.to_parquet(OUT / "recipe_acidfit_rows.parquet", index=False)
    out = []
    for lab, g in (("interior", t[t.inside]), ("exterior", t[~t.inside]), ("all", t),
                   ("interior, >=3 sibling acid levels", t[t.inside & (t.n_levels >= 3)])):
        if not len(g):
            continue
        cc = g.groupby("cell_id").agg(publication=("publication", "first"), N=("y", "size"), sy=("y", "sum"),
                                      syy=("y", lambda s: float((s ** 2).sum()))).reset_index()
        P = RA.Pop(cc, lab)

        def cs(col):
            return g.assign(e=(g[col] - g.y) ** 2).groupby("cell_id").e.sum().reindex(cc.cell_id).to_numpy()
        for arm in ("MEAN", "LIN", "LINCLIP", "LINHALF"):
            d = P.summ(cs(arm), cs("MEAN"))
            out.append({"subset": lab, "cells": len(cc), "pubs": cc.publication.nunique(), "rows": len(g),
                        "arm": arm, "R2": d["R2"], "lo": d["lo"], "hi": d["hi"], "d_vs_MEAN": d["d"],
                        "d_lo": d["d_lo"], "d_hi": d["d_hi"], "lopo_min": d["lopo_min"], "lopo_max": d["lopo_max"]})
    out = pd.DataFrame(out)
    out.to_csv(OUT / "recipe_acidfit.csv", index=False)
    with pd.option_context("display.width", 250):
        print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
