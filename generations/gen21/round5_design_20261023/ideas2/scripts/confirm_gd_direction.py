#!/usr/bin/env python
"""confirm theme, check 5 (EXPLORATORY): non-DGA-specific Gd question.  Corpus units (publication x chemotype, C3) split
by donor class and by the direction of selectivity (least-squares slope of log D on -r(CN8) over the cell's measured
metals; > 0 = heavy-selective).  Power of
  GDDEEP_HS : heavy-selective non-DGA units, mean winsorised chord < beta (B1 under-corrects)   one-sample t, df G-1
  HS_vs_LS  : heavy-selective non-DGA minus light-selective units < 0                            Welch t
at the G a ~13-laboratory supply could give, with size at the null boundaries (re-centred worlds).
EXPLORATORY (seen cells).  Writes only ideas2/scripts/confirm_out/gd_direction_*.
"""
from __future__ import annotations

import itertools
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parent / "confirm_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

BETA = -0.0923
CLIP = (-0.40, 0.20)
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
R = 4000


def main() -> int:
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    LN = list(D.LANTHANIDES)
    XR = interp.X_R
    t = (XR[LN.index("Gd")] - XR[LN.index("Eu")]) / (XR[LN.index("Tb")] - XR[LN.index("Eu")])
    c = Y[:, LN.index("Gd")] - ((1 - t) * Y[:, LN.index("Eu")] + t * Y[:, LN.index("Tb")])
    sl = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            sl[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    d = pd.DataFrame({"c": c, "hs": sl > 0, "c3": f.cell_id.isin(set(ref.cell_id)),
                      "dga": f.extractant.astype(str).map(DN.is_dga), "pub": f.publication_id,
                      "unit": f.publication_id.astype(str) + "|" + f.chemotype.astype(str)})
    d = d[d.c3 & np.isfinite(d.c)]
    rows, desc = [], []
    for scope, q in (("all", d), ("noC5DMD", d[~d.pub.isin([C5, DMD])])):
        u = q.groupby("unit").agg(c=("c", "mean"), hs=("hs", "mean"), dga=("dga", "mean"), n=("c", "size"))
        u["cw"] = np.clip(u.c, *CLIP)
        u["cls"] = np.where(u.dga > 0.5, "DGA", np.where(u.hs > 0.5, "nonDGA_HS", "nonDGA_LS"))
        for cls, g in u.groupby("cls"):
            desc.append({"scope": scope, "cls": cls, "units": len(g), "cells": int(g.n.sum()), "mean_w": g.cw.mean(),
                         "median": g.c.median(), "sd_w": g.cw.std(ddof=1)})
        hs = u[u.cls == "nonDGA_HS"].cw.to_numpy()
        ls = u[u.cls != "nonDGA_HS"][lambda x: x.cls == "nonDGA_LS"].cw.to_numpy()
        for world in ("EMP", "NULL"):
            for G in (4, 5, 6, 8, 10, 13):
                rng = np.random.default_rng(20261023 + zlib.crc32(f"dir|{scope}|{world}|{G}".encode()))
                X = hs[rng.integers(len(hs), size=(R, G))]
                if world == "NULL":
                    X = X - hs.mean() + BETA
                m = X.mean(1)
                se = X.std(1, ddof=1) / np.sqrt(G)
                tc = SS.t.ppf(0.975, G - 1)
                E = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
                Z = BETA - X
                p = ((Z @ E.T) >= Z.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]
                rows.append({"scope": scope, "test": "GDDEEP_HS", "world": world, "G_hs": G, "G_ls": 0,
                             "rej_t": float((((m - BETA) / se) < -tc).mean()), "rej_sf": float((p <= 0.025).mean())})
            for Gh, Gl in ((6, 3), (6, 4), (8, 4), (10, 3), (8, 6), (10, 5)):
                rng = np.random.default_rng(20261023 + zlib.crc32(f"dir2|{scope}|{world}|{Gh}|{Gl}".encode()))
                Xh = hs[rng.integers(len(hs), size=(R, Gh))]
                Xl = ls[rng.integers(len(ls), size=(R, Gl))]
                if world == "NULL":
                    Xh = Xh - hs.mean()
                    Xl = Xl - ls.mean()
                tt = SS.ttest_ind(Xh, Xl, axis=1, equal_var=False, alternative="less")
                rows.append({"scope": scope, "test": "HS_vs_LS", "world": world, "G_hs": Gh, "G_ls": Gl,
                             "rej_t": float((tt.pvalue <= 0.025).mean()), "rej_sf": np.nan})
    pd.set_option("display.width", 200)
    dd = pd.DataFrame(desc)
    rr = pd.DataFrame(rows)
    dd.to_csv(OUT / "gd_direction_desc.csv", index=False)
    rr.to_csv(OUT / "gd_direction_replay.csv", index=False)
    print(dd.round(3).to_string())
    print(rr.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
