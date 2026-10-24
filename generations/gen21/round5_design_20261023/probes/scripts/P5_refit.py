#!/usr/bin/env python
"""P5 (idea M25), training-side sensitivity: what B1's Gd amplitude (and B1's OOF score) would be if a
provenance-pending item were never in the corpus.

EXPLORATORY (229 C3 cells / 5 dev seeds already seen).  For every dev outer fold of BPG and BP the round-4 B1 fit
(r4.refs.fit_reference, inner-CV ridge strength over round 3's grid + infinity, unchanged code) is rerun on a copy
of the fold record whose outer training index omits the item's cells (chemotype-balanced cell weights therefore
re-balance; inner DOPT_Q covariances and the read sets of the OOF queries are kept, i.e. byte-identical queries).
The OOF prediction is rebuilt exactly as B1 = B0 + beta * dphi, with dphi = (HIST_D7 - B0) / (-0.072) taken from
REF_<design>.parquet (asserted to reproduce B1 with the stored beta).  Nothing is written but stdout.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore \
      generations/gen21/round5_design_20261023/probes/scripts/P5_refit.py [BPG|BP ...]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
R4 = ROOT / "generations" / "gen21" / "round4_20261022"
sys.path.insert(0, str(R4))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

C5_PUB, DMD_PUB = "pub_6a0ef8528c", "pub_8415538291"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def scenario_cells(c3: pd.DataFrame) -> dict[str, set]:
    noloc = c3.data_location.isna()
    v2 = c3[["v2_1_ornl_reimport", "v2_2_duplicate_entry", "v2_3_hidden_merge", "v2_5_resolution_floor"]].any(axis=1)
    s = lambda m: set(c3.loc[m, "cell_id"])  # noqa: E731
    sc = {"T0 none (identity)": set(),
          "T1 C5-BTBP (15)": s(c3.publication_id == C5_PUB),
          "T2 DMDPhPDA copy B (5)": s((c3.publication_id == DMD_PUB) & c3.ornl_import_all),
          "T3 ORNL-54": s(c3.ornl_import_all & noloc),
          "T4 all no-location (79)": s(noloc),
          "T5 V2 any (23)": s(v2)}
    sc["T6 UNION C5+DMD+ORNL54+V2 (89)"] = (sc["T1 C5-BTBP (15)"] | s(c3.publication_id == DMD_PUB) | sc["T3 ORNL-54"]
                                           | sc["T5 V2 any (23)"])
    # true no-location: the 10 curated 'no-location' cells carry their location in the archive comments field
    sc["T7 true no-location ORNL-54 + C5 (69)"] = sc["T3 ORNL-54"] | sc["T1 C5-BTBP (15)"]
    return sc


def pick(sc: dict, keys) -> dict:
    """Keep T0 (identity, must run first) plus the requested scenario prefixes, e.g. T7."""
    return {k: v for k, v in sc.items() if k.startswith("T0") or any(k.startswith(p) for p in keys)}


def main():
    args = sys.argv[1:]
    designs = [a for a in args if a in ("BPG", "BP")] or ["BPG", "BP"]
    only = [a for a in args if a.startswith("T")]
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    ci_of = dict(zip(f.cell_id, f.index))
    cells = pd.read_csv(R4 / "results" / "provenance" / "cells.csv")
    c3 = cells[cells.c3]
    # exclusion applies to every V1 cell of the item that is C3 (all item cells are C3); non-C3 cells untouched
    scen = scenario_cells(c3)
    if only:
        scen = pick(scen, only)
    print("EXPLORATORY training-side refit; scenarios:", {k: len(v) for k, v in scen.items()}, flush=True)
    out_all = {}
    for design in designs:
        t = pd.read_parquet(R4 / "predictions" / f"REF_{design}.parquet")
        t = t[(t.k == 3) & (t.k_max >= 3)].reset_index(drop=True)
        dphi = (t.HIST_D7 - t.B0) / RF.HIST_D7_BETA
        fitted = json.loads((R4 / "results" / "REF" / f"fitted_{design}.json").read_text())
        betas = {k: [] for k in scen}
        lams = {k: [] for k in scen}
        pred = {k: t.B0.to_numpy().copy() for k in scen}
        t0 = time.time()
        for fo in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
            rec = C.load(fo.design, fo.seed, fo.fold)
            stored = fitted[f"{fo.seed}_{fo.fold}"]["ref:B1"]["coef"][0]
            rows = ((t.split_seed == fo.seed) & (t.fold == fo.fold)).to_numpy()
            # identity of the OOF reconstruction with the stored beta
            assert np.allclose(t.B1[rows], t.B0[rows] + stored * dphi[rows], atol=1e-9), (fo.seed, fo.fold)
            tr = np.asarray(rec["outer"]["train_index"])
            for k, excl in scen.items():          # T0 first: the orphan memo is built from the full training index
                drop = np.array([ci_of[c] for c in excl], dtype=int)
                rec2 = dict(rec)
                rec2["outer"] = dict(rec["outer"], train_index=tr[~np.isin(tr, drop)])
                fit = RF.fit_reference(bench, rec2, bench.Y, "B1")
                b = float(fit["coef"][0])
                if k.startswith("T0"):
                    assert abs(b - stored) < 1e-9, ("T0 does not reproduce the stored beta", fo.seed, fo.fold, b, stored)
                betas[k].append(b)
                lams[k].append(fit["lambda"])
                pred[k][rows] = t.B0[rows].to_numpy() + b * dphi[rows].to_numpy()
            print(f"  {design} {fo.seed}/{fo.fold}: " + " ".join(f"{k[:2]}={betas[k][-1]:+.4f}" for k in scen)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        res = []
        for k, excl in scen.items():
            out_item = ~t.cell_id.isin(excl).to_numpy()
            r_all = [r2(g.y, pred[k][g.index]) for _, g in t.groupby("split_seed")]
            r_out = [r2(g.y, pred[k][g.index]) for _, g in t[out_item].groupby("split_seed")]
            r_out_b1 = [r2(g.y, g.B1) for _, g in t[out_item].groupby("split_seed")]
            r_out_b0 = [r2(g.y, g.B0) for _, g in t[out_item].groupby("split_seed")]
            res.append({"scenario": k, "excluded cells": len(excl), "beta mean": np.mean(betas[k]),
                        "beta min": np.min(betas[k]), "beta max": np.max(betas[k]),
                        "lambda=inf folds": sum(1 for x in lams[k] if isinstance(x, float) and math.isinf(x)),
                        "R2 B1' all rows": np.mean(r_all), "R2 B1' outside item": np.mean(r_out),
                        "R2 B1(V1 fit) outside item": np.mean(r_out_b1), "R2 B0 outside item": np.mean(r_out_b0),
                        "B1'-B0 outside": np.mean(np.array(r_out) - np.array(r_out_b0)),
                        "B1'-B0 outside min seed": np.min(np.array(r_out) - np.array(r_out_b0))})
        d = pd.DataFrame(res)
        pd.set_option("display.width", 250)
        print(f"\n== {design}: training-side refit (seed means; beta over 25 folds) ==")
        print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        out_all[design] = d
    print("\nEXIT 0")


if __name__ == "__main__":
    main()
