#!/usr/bin/env python
"""E-LAB extra checks.  EXPLORATORY (already-seen cells).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_extra.py

1. Level bands of within-publication sharing.  For same-publication, other-extractant cell pairs (SH + SO), the pooled
   r (lab_varcomp2 definition) by the band of the two cells' mean log D at that metal, both cells within 1 log of each
   other.  A mass-balance or blank-type analytical bias grows as log D falls (bias ~ eps (1 + 1/D)); a chemistry
   pattern need not.  Cross-publication pairs (other extractant) as the reference.
2. Medium-matched borrowed reference.  For the non-TODGA targets of the five TODGA publications: the TODGA shape from
   OTHER publications restricted to TODGA cells with the target's aqueous medium (same acid one-hot, |dlog10 acid| <=
   0.15, same diluent one-hot), shrunk with its own n (K = 6.1).  Tells whether a user could borrow a literature
   TODGA curve at the same medium instead of running an in-lab reference.  Targets without a matched TODGA keep B1.
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
from lab_calib import detrend, shape, levels, K6  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402

pd.set_option("display.width", 250)


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
    name = f.extractant_name.astype(str).to_numpy()
    acid = f[[c for c in f.columns if c.startswith("cond__acid__")]].fillna(0).to_numpy() > 0
    dil = f[[c for c in f.columns if c.startswith("cond__diluent__")]].fillna(0).to_numpy() > 0
    lac = np.log10(np.clip(f["cond__acid_concentration_M"].to_numpy(float), 1e-4, None))
    # ---- 1. level bands ----
    idx = np.where(has)[0]
    Dc = D2[idx] - np.nanmean(D2[idx], 0)
    Yl = Y[idx]
    I, J = np.triu_indices(len(idx), 1)
    sp = pub[idx][I] == pub[idx][J]
    se = ext[idx][I] == ext[idx][J]
    A, B = Dc[I], Dc[J]
    fin = np.isfinite(A) & np.isfinite(B) & (np.abs(Yl[I] - Yl[J]) <= 1)
    mid = 0.5 * (Yl[I] + Yl[J])
    bands = [(-9, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 9)]
    out = []
    for nm, rows in (("same pub, other extractant", sp & ~se), ("same pub, same extractant", sp & se),
                     ("other pub, other extractant", ~sp & ~se)):
        rec = {"pairs": nm}
        for lo, hi in bands:
            m = fin & rows[:, None] & (mid >= lo) & (mid < hi)
            num = (A * B)[m].sum()
            den = (0.5 * (A ** 2 + B ** 2))[m].sum()
            rec[f"[{lo},{hi})"] = f"{num / den:+.3f} ({m.sum()})" if m.sum() >= 20 else "n<20"
        out.append(rec)
    print("pooled r (metal-level pairs within 1 log) by band of mean log D; (n metal-pairs):")
    print(pd.DataFrame(out).to_string(index=False))
    # ---- 2. medium-matched borrowed TODGA ----
    ref_pubs = sorted({pub[i] for i in range(len(f)) if name[i] == "TODGA"})
    tod = np.where(has & (name == "TODGA"))[0]
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    rows = []
    nmatch = {}
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
                if pub[i] not in ref_pubs or name[i] == "TODGA":
                    continue
                own = np.where(has & (pub == pub[i]) & (name == "TODGA") & (np.arange(len(f)) != i))[0]
                if len(own) == 0:
                    continue
                medm = tod[(pub[tod] != pub[i]) & (acid[tod] == acid[i]).all(1) & (dil[tod] == dil[i]).all(1)
                           & (np.abs(lac[tod] - lac[i]) <= 0.15)]
                nmatch[t] = (len(medm), len(set(pub[medm])))
                yi = Y[i]
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                metals = list(cell["metals"])
                L = {"B1": levels(yi, reads, metals, u)}
                c, w = shape(D2, own, K6)
                L["MREF_own"] = levels(yi, reads, metals, (1 - w) * u + w * c)
                c, w = shape(D2, medm, K6)
                L["MREF_med"] = levels(yi, reads, metals, (1 - w) * u + w * c)
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "y": r.y, "has_med": len(medm) > 0}
                    for k, lv in L.items():
                        o[k] = lv[ia] - lv[ib]
                    rows.append(o)
    R = pd.DataFrame(rows)
    print(f"\nTODGA-publication non-TODGA targets: {R.cell_id.nunique()} cells; with >= 1 medium-matched other-publication "
          f"TODGA cell: {R[R.has_med].cell_id.nunique()} cells; matched TODGA cells per target (n, pubs): "
          f"{sorted(set(nmatch.values()))}")
    for nm, d in (("all reference targets", R), ("targets with a medium-matched borrowed TODGA", R[R.has_med])):
        if d.empty:
            continue
        print(f"   {nm} ({d.cell_id.nunique()} cells, {d.publication.nunique()} pubs): B1 {r2(d, 'B1'):.4f}  own TODGA "
              f"{r2(d, 'MREF_own'):.4f} ({rel(d, 'MREF_own'):+.3f})  borrowed medium-matched TODGA {r2(d, 'MREF_med'):.4f} "
              f"({rel(d, 'MREF_med'):+.3f})")
        r = infer(d, "MREF_med", "B1")
        print(f"      borrowed - B1 by publication: up {r['up']} down {r['down']} of {r['G']}; rel {r['rel']:+.3f} "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
