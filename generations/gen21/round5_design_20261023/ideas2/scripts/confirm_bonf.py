#!/usr/bin/env python
"""confirm theme, check 6 (EXPLORATORY): power of the three package claims at the family-wise (Bonferroni) level
alpha = 0.025 / 3, next to 0.025, for the decisive legs:
  C-GD   heavy-selective non-DGA units: winsorised mean chord < beta  (one-sample t)          [confirm_gd_direction pools]
  C-ECT  SH (4 reads) vs B1 (4 reads), equal-weight log SSE ratio, exact sign-flip             [P1 PRIMARY pubchem units]
  C-ECT3 SH (3 reads) vs B1 (4 reads), same test
EXPLORATORY (seen cells).  Writes only ideas2/scripts/confirm_out/bonf.csv.
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

BETA = -0.0923
R = 4000
ALPHAS = (0.025, 0.025 / 3)


def main() -> int:
    rows = []
    # ---- C-GD: rebuild the heavy-selective non-DGA unit values from the check-5 inputs
    from r4 import paths  # noqa: F401
    from r4 import donor as DN
    from r3 import interp
    from gen20k import data as D
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    LN = list(D.LANTHANIDES)
    XR = interp.X_R
    tg = (XR[LN.index("Gd")] - XR[LN.index("Eu")]) / (XR[LN.index("Tb")] - XR[LN.index("Eu")])
    c = Y[:, LN.index("Gd")] - ((1 - tg) * Y[:, LN.index("Eu")] + tg * Y[:, LN.index("Tb")])
    sl = np.array([np.polyfit(XR[np.isfinite(y)], y[np.isfinite(y)], 1)[0] if np.isfinite(y).sum() >= 5 else np.nan
                   for y in Y])
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    d = pd.DataFrame({"c": c, "hs": sl > 0, "c3": f.cell_id.isin(set(ref.cell_id)),
                      "dga": f.extractant.astype(str).map(DN.is_dga), "pub": f.publication_id,
                      "unit": f.publication_id.astype(str) + "|" + f.chemotype.astype(str)})
    d = d[d.c3 & np.isfinite(d.c)]
    for scope, q in (("all", d), ("noC5DMD", d[~d.pub.isin(["pub_6a0ef8528c", "pub_8415538291"])])):
        u = q.groupby("unit").agg(c=("c", "mean"), hs=("hs", "mean"), dga=("dga", "mean"))
        hs = np.clip(u[(u.dga <= 0.5) & (u.hs > 0.5)].c.to_numpy(), -0.40, 0.20)
        for G in (5, 6, 8, 10, 13):
            rng = np.random.default_rng(20261023 + zlib.crc32(f"bonf|gd|{scope}|{G}".encode()))
            X = hs[rng.integers(len(hs), size=(R, G))]
            tstat = (X.mean(1) - BETA) / (X.std(1, ddof=1) / np.sqrt(G))
            for a in ALPHAS:
                rows.append({"claim": "C-GD (HS nonDGA, mean chord < beta)", "scope": scope, "G": G, "alpha": a,
                             "power": float((tstat < -SS.t.ppf(1 - a, G - 1)).mean())})
    # ---- C-ECT: P1 PRIMARY units
    t = pd.read_parquet(OUT / "p1" / "P1_rows.parquet")
    t = t[(t.design == "BPG") & ~t.t_v2drop & (t.n_clean >= 2) & ~t.c5].copy()
    t["unit"] = t.publication + "|" + t.chemotype
    for name, cand in (("C-ECT SH4 vs B1@4", "CL_SH_3"), ("C-ECT3 SH3 vs B1@4", "CL_SH_2")):
        g = t.assign(er=(t.B1_3 - t.y) ** 2, ec=(t[cand] - t.y) ** 2).groupby(["unit", "split_seed"])[["er", "ec"]].sum()
        g = g.groupby(level=0).mean()
        L = np.log(g.er / g.ec).to_numpy()
        for tau in (0.0, 0.065):
            for G in (5, 6, 7, 8, 9):
                rng = np.random.default_rng(20261023 + zlib.crc32(f"bonf|ect|{name}|{tau}|{G}".encode()))
                X = L[rng.integers(len(L), size=(R, G))]
                if tau > 0:
                    s = tau / np.sqrt(2)
                    X = X + 2 * np.log((1 - rng.normal(0, s, size=X.shape)) / (1 - rng.normal(0, s, size=X.shape)))
                E = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
                p = ((X @ E.T) >= X.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]
                for a in ALPHAS:
                    rows.append({"claim": name, "scope": f"PRIMARY tau={tau}", "G": G, "alpha": a,
                                 "power": float((p <= a + 1e-12).mean())})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "bonf.csv", index=False)
    print(r.pivot_table(index=["claim", "scope", "G"], columns="alpha", values="power").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
