#!/usr/bin/env python
"""Skeptic A for idea lab-2: is the dominant publication's gain specific to TODGA, or would any in-campaign series do?
EXPLORATORY (already-seen cells); BPG; prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab2_rank_skA.py

pub_126bbe7976 (one TODGA series + 14 other DGAs, one condition) carries ~80 % of the pooled OWN gain.  Every usable
series d of that publication is tried as the single "reference" (shape of d, w = 1 / 7.1) for every C3 target t != d
that is not TODGA.  For each candidate reference, rel SSE vs B1 on targets != d; TODGA's rel on the same rows.  Rank of
TODGA among all candidate references = a within-campaign permutation check.  Same for pub_4cba27f5f3 (3 TODGA series,
4 other DGAs) and pub_b9dfc468cd.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab2_refseries_skA import devs, shp, pwl, sq  # noqa: E402

PUBS = ("pub_126bbe7976", "pub_4cba27f5f3", "pub_b9dfc468cd")


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    nm = f.extractant_name.astype(str).to_numpy()
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ok = np.array([c not in v2 for c in ids])
    Dv = devs(Y)
    has = np.isfinite(Dv).any(1) & ok
    P = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    cand = {p: [j for j in np.where(has & (pub == p))[0]] for p in PUBS}
    rows = []
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            u = np.zeros(N_LN)
            u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            prow = P[(P.split_seed == seed) & (P.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg or t in v2:
                    continue
                i = pos[t]
                if pub[i] not in PUBS or nm[i] == "TODGA":
                    continue
                yi = Y[i]
                metals = list(cell["metals"])
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                B1 = pwl(yi, reads, metals, u)
                Ls = {}
                for j in cand[pub[i]]:
                    if j == i:
                        continue
                    c, w = shp(Dv, [j])
                    Ls[j] = pwl(yi, reads, metals, (1 - w) * u + w * c)
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    for j, lv in Ls.items():
                        rows.append({"pub": pub[i], "t": t, "d": ids[j], "dname": nm[j], "seed": seed,
                                     "eB": (B1[ia] - B1[ib] - r.y) ** 2, "eD": (lv[ia] - lv[ib] - r.y) ** 2})
    T = pd.DataFrame(rows)
    for p in PUBS:
        d = T[T.pub == p]
        tod_ids = sorted(set(d[d.dname == "TODGA"].d))
        out = []
        for dj, g in d.groupby("d"):
            # common rows: targets != dj; TODGA's error on exactly those rows (first TODGA donor id for reference)
            o = {"donor": dj[:16], "name": g.dname.iloc[0], "n_targets": g.t.nunique(),
                 "rel": 1 - g.eD.sum() / g.eB.sum()}
            for k, tj in enumerate(tod_ids):
                gt = d[(d.d == tj) & d.t.isin(set(g.t))]
                o[f"TODGA{k}_same_rows"] = 1 - gt.eD.sum() / gt.eB.sum() if len(gt) else np.nan
            out.append(o)
        R = pd.DataFrame(out).sort_values("rel", ascending=False).reset_index(drop=True)
        R["rank"] = np.arange(1, len(R) + 1)
        print(f"\n=== {p}: {len(R)} candidate single references; targets {d.t.nunique()} ===")
        print(R.round(4).to_string(index=False))
        nt = R[R.name != "TODGA"].rel
        print(f"non-TODGA candidates: mean rel {nt.mean():+.4f}, median {nt.median():+.4f}, share > 0: {(nt > 0).mean():.2f}; "
              f"TODGA rank(s): {R[R.name == 'TODGA']['rank'].tolist()} of {len(R)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
