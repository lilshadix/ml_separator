#!/usr/bin/env python
"""Skeptic A for idea lab-1, step 3: shrinkage-only controls and shared-source checks.  EXPLORATORY (seen rows).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab1_controls_skA.py

1. Global-shrinkage control: unmatched MX (all own-publication other-extractant donors) at K in {6.1, 12, 20, 35, 60, 100}.
   If the best K (chosen on these very rows: optimistic for the control) approaches MX_LM10, level matching adds little
   beyond shrinkage.
2. Target-donor shared underlying records: overlap of experiment_series_ids / safe_exp_ids between each POPX target and
   its donors (a donor built from the same experiment rows as the target would be leakage).
3. Level-matched donor count per target metal and the fraction of target metals where LM falls back to B1.
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
from lab1_recompute_skA import qdev, interp_levels, lm_cw, r2, rel, infer, TOL  # noqa: E402

KS = (6.1, 12.0, 20.0, 35.0, 60.0, 100.0)


def toks(s):
    s = str(s)
    if s in ("", "nan", "None"):
        return set()
    return {t.strip() for t in s.replace("[", "").replace("]", "").replace("'", "").replace(";", ",").replace("|", ",").split(",") if t.strip()}


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    Dv = np.array([qdev(Y[i]) for i in range(len(Y))])
    has = np.isfinite(Dv).any(1) & okd
    ar = np.arange(len(f))
    print("example experiment_series_ids / safe_exp_ids:", f.experiment_series_ids.iloc[0], "|", f.safe_exp_ids.iloc[0])
    ES = [toks(x) for x in f.experiment_series_ids]
    SE = [toks(x) for x in f.safe_exp_ids]

    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    rows, ov, fb = [], [], []
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
                dX = np.where(has & (ar != i) & (pub == pub[i]) & (ext != ext[i]))[0]
                if not len(dX):
                    continue
                yi = Y[i]
                metals = list(cell["metals"])
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                lv = interp_levels(yi, reads, metals, u)
                L = {"B1": lv}
                c, w = lm_cw(Dv, Y, dX, lv)
                L["LM"] = interp_levels(yi, reads, metals, (1 - w) * u + w * c)
                A = Dv[dX]
                fin = np.isfinite(A)
                for k in KS:
                    cc, _ = lm_cw(Dv, Y, dX, lv, mask_override=fin)
                    ww = fin.sum(0) / (fin.sum(0) + k)
                    L[f"MX_K{k:g}"] = interp_levels(yi, reads, metals, (1 - ww) * u + ww * cc)
                if seed == paths.DEV_SEEDS[0]:
                    close = fin & (np.abs(Y[dX] - lv[None, :]) <= TOL)
                    mm = [m for m in metals if m not in reads]
                    fb.append({"cell": t, "pub": pub[i], "n_donors": len(dX), "unread_metals": len(mm),
                               "unread_with_0_matched": int(sum(close[:, m].sum() == 0 for m in mm)),
                               "mean_w_LM_unread": float(np.mean([close[:, m].sum() / (close[:, m].sum() + 6.1) for m in mm])),
                               "mean_w_MX_unread": float(np.mean([fin[:, m].sum() / (fin[:, m].sum() + 6.1) for m in mm]))})
                    for j in dX:
                        ov.append({"target": t, "donor": ids[j], "pub": pub[i], "es_overlap": len(ES[i] & ES[j]),
                                   "se_overlap": len(SE[i] & SE[j])})
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "y": r.y}
                    for k2, l2 in L.items():
                        o[k2] = l2[ia] - l2[ib]
                    rows.append(o)
    T = pd.DataFrame(rows)
    print(f"POPX rows/seed {len(T) / 5:.0f}, cells {T.cell_id.nunique()}, pubs {T.publication.nunique()}")
    arms = ["B1", "LM"] + [f"MX_K{k:g}" for k in KS]
    print(pd.DataFrame([{"arm": a, "R2": r2(T, a), "rel_vs_B1": rel(T, a), "LM_minus_arm_rel": rel(T, "LM", a)} for a in arms])
          .round(4).to_string(index=False))
    best = max(arms[2:], key=lambda a: rel(T, a))
    r = infer(T, "LM", best)
    print(f"LM - best global-K MX ({best}): rel {r['rel']:+.4f} [{r['lo95']:+.3f}, {r['hi95']:+.3f}] up/down {r['up']}/{r['down']} "
          f"p {r['p_sf']:.3f}")
    r = infer(T, best, "B1")
    print(f"{best} - B1: rel {r['rel']:+.4f} [{r['lo95']:+.3f}, {r['hi95']:+.3f}] up/down {r['up']}/{r['down']} p {r['p_sf']:.3f}")
    O = pd.DataFrame(ov)
    print(f"\ntarget-donor pairs {len(O)}: experiment_series_ids overlap > 0: {(O.es_overlap > 0).sum()}; "
          f"safe_exp_ids overlap > 0: {(O.se_overlap > 0).sum()}")
    if (O.se_overlap > 0).any() or (O.es_overlap > 0).any():
        print(O[(O.se_overlap > 0) | (O.es_overlap > 0)].head(30).to_string(index=False))
    F = pd.DataFrame(fb)
    print("\nfallback anatomy (seed 0, POPX targets):")
    print(f"  unread metals {F.unread_metals.sum()}, with 0 level-matched donors {F.unread_with_0_matched.sum()} "
          f"({F.unread_with_0_matched.sum() / F.unread_metals.sum():.2f}); mean w unread LM {F.mean_w_LM_unread.mean():.3f} "
          f"vs MX {F.mean_w_MX_unread.mean():.3f}")
    print(F.groupby("pub").agg(cells=("cell", "size"), donors=("n_donors", "median"), frac0=("unread_with_0_matched", "sum"),
                               unread=("unread_metals", "sum"), wLM=("mean_w_LM_unread", "mean"), wMX=("mean_w_MX_unread", "mean"))
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
