#!/usr/bin/env python
"""E-LAB protocol sizing: (1) how many other-extractant series of the campaign does the level-matched fingerprint need,
(2) does it save reads (2 / 3 / 4 reads against B1 at 4 reads)?  EXPLORATORY (already-seen cells), BPG only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_ladder.py

Population: CLEAN C3 targets whose publication has >= 1 other-extractant donor series and NO same-extractant cell
(POPX0 of lab_calib.py).  Rule MX_LM10 exactly as lab_calib2.py (tol 1.0 around the target's B1 level at the same read
set, K = 6.1, quadratic-detrended donors with >= 6 metals).
(1) Donor subsampling at 4 reads, targets with >= 3 donors: exact expectation over all m-subsets (m = 1, 2, 3; at most
    200 seeded subsets), squared error averaged over subsets.
(2) Read ladder: nested DOPT_Q sets seq[:1], seq[:2], seq[:3] (2, 3, 4 reads); B1 at each and MX_LM10 at each, all on
    the rows scored at 4 reads.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
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
from lab_calib import detrend, levels, K6  # noqa: E402
from lab_calib2 import lm_shape  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    ar = np.arange(len(f))
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    rows = []
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            u = np.zeros(N_LN)
            u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            prow = par[(par.split_seed == seed) & (par.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg or t in v2drop:
                    continue
                i = pos[t]
                base = has & (ar != i) & (pub == pub[i])
                dX = np.where(base & (ext != ext[i]))[0]
                if len(dX) == 0 or (base & (ext == ext[i])).any():
                    continue
                yi = Y[i]
                metals = list(cell["metals"])
                L = {}
                for k in (1, 2, 3):
                    rd = sorted({m for p in cell["seqs"]["DOPT_Q"][:k] for m in p})
                    lb = levels(yi, rd, metals, u)
                    L[f"B1_{k}"] = lb
                    c, w = lm_shape(D2, Y, dX, lb, 1.0, K6)
                    L[f"LM_{k}"] = levels(yi, rd, metals, (1 - w) * u + w * c)
                rd = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                SUB = {}
                if len(dX) >= 3:
                    for m in (1, 2, 3):
                        combos = list(itertools.combinations(range(len(dX)), m))
                        if len(combos) > 200:
                            rng = np.random.default_rng(int(hashlib.sha256(f"{t}:{m}".encode()).hexdigest()[:8], 16))
                            combos = [tuple(sorted(rng.choice(len(dX), m, replace=False))) for _ in range(200)]
                        SUB[m] = []
                        for cmb in combos:
                            c, w = lm_shape(D2, Y, dX[list(cmb)], L["B1_3"], 1.0, K6)
                            SUB[m].append(levels(yi, rd, metals, (1 - w) * u + w * c))
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "y": r.y, "n_donor": len(dX)}
                    for k2, lv in L.items():
                        o[k2] = lv[ia] - lv[ib]
                    for m, lst in SUB.items():
                        o[f"E_SUB{m}"] = float(np.mean([(lv[ia] - lv[ib] - r.y) ** 2 for lv in lst]))
                    rows.append(o)
    T = pd.DataFrame(rows)
    print(f"POPX0 BPG: cells {T.cell_id.nunique()}, pubs {T.publication.nunique()}, rows/seed {len(T) / 5:.0f}")
    print("read ladder (R2; rel SSE vs B1 at 4 reads):")
    for k, nr in ((1, 2), (2, 3), (3, 4)):
        print(f"   {nr} reads: B1 {r2(T, f'B1_{k}'):.4f} ({rel(T, f'B1_{k}', 'B1_3'):+.3f})   MX_LM10 {r2(T, f'LM_{k}'):.4f} "
              f"({rel(T, f'LM_{k}', 'B1_3'):+.3f})")
    for k in (1, 2):
        r = infer(T, f"LM_{k}", "B1_3")
        print(f"   MX_LM10 at {k + 1} reads - B1 at 4 reads: rel {r['rel']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] up/down "
              f"{r['up']}/{r['down']} of {r['G']}, p {r['p_sf']:.3f}, top3 {r['top3']:.2f}")
    S = T[T.n_donor >= 3]
    print(f"\ndonor subsampling (targets with >= 3 donors: cells {S.cell_id.nunique()}, pubs {S.publication.nunique()}, "
          f"rows/seed {len(S) / 5:.0f}):")
    print(f"   B1 {r2(S, 'B1_3'):.4f}; m=1 {r2(S, 'E_SUB1'):.4f} ({rel(S, 'E_SUB1', 'B1_3'):+.3f}); m=2 {r2(S, 'E_SUB2'):.4f} "
          f"({rel(S, 'E_SUB2', 'B1_3'):+.3f}); m=3 {r2(S, 'E_SUB3'):.4f} ({rel(S, 'E_SUB3', 'B1_3'):+.3f}); all "
          f"{r2(S, 'LM_3'):.4f} ({rel(S, 'LM_3', 'B1_3'):+.3f})")
    r = infer(S, "E_SUB1", "B1_3")
    print(f"   one donor series - B1: up/down {r['up']}/{r['down']} of {r['G']}, rel {r['rel']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
