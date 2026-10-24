#!/usr/bin/env python
"""P5 (idea M25): stakes-first source-recovery accounting audit on the round-4 OOF rows.

EXPLORATORY: every number here is computed on the 229 C3 cells / 5 dev seeds that rounds 3-4 already scored.
Nothing is fitted; nothing in V1 is edited.  Scenarios are row-level sensitivities on byte-identical queries
(the round-4 read sets and predictions are held fixed unless a scenario swaps a whole duplicate cell).

Inputs (read-only):
  round4_20261022/predictions/REF_{BPG,BP}.parquet   (k = 3, DOPT_Q, 5 dev seeds x 1,802 rows)
  round4_20261022/results/provenance/cells.csv         (provenance per V1 cell)
  gen20k.data.load_bench()                             (log D per cell, for the floor scenarios)
  dataset_all_metals/clean/master_clean.parquet        (D_raw strings for the rounding-noise bound)

Run from D:\\ml_separator_gh:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore \
      generations/gen21/round5_design_20261023/probes/scripts/P5_accounting.py
Output: stdout only (the report P5_source_recovery.md quotes it).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
R4 = ROOT / "generations" / "gen21" / "round4_20261022"
sys.path.insert(0, str(R4))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401  (puts r3 / g21 / gen20k on sys.path)
from gen20k import data as D  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 70)

LN = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
IDX = {m: i for i, m in enumerate(LN)}
ARMS = ("B0", "B1", "HIST_D7")
C5_PUB, C5_CELL = "pub_6a0ef8528c", "c950dc2982e96b83"
DMD_PUB = "pub_8415538291"
DMD_A5, DMD_B5 = "ceb21275e22d7bc5", "7054e94bc4c6833f"      # copy A (curated, Table 1) / copy B (ORNL) at 5 M HNO3
CENSUS_SEED = 20261023                                         # label-free census order (fixed before any recovery)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - p) ** 2).sum() / sst


def load():
    oof = {}
    for d in ("BPG", "BP"):
        t = pd.read_parquet(R4 / "predictions" / f"REF_{d}.parquet")
        t = t[(t.k == 3) & (t.k_max >= 3)].copy()
        assert t.groupby("split_seed").size().eq(1802).all(), d
        t["pair"] = t.A + "/" + t.B
        oof[d] = t
    cells = pd.read_csv(R4 / "results" / "provenance" / "cells.csv")
    c3 = cells[cells.c3].copy()
    c3["noloc"] = c3.data_location.isna()
    c3["v2_any"] = c3[["v2_1_ornl_reimport", "v2_2_duplicate_entry", "v2_3_hidden_merge", "v2_5_resolution_floor"]].any(axis=1)
    assert set(c3.cell_id) == set(oof["BPG"].cell_id), "cell sets differ"
    return oof, c3


def items(c3: pd.DataFrame) -> dict[str, set]:
    s = lambda m: set(c3.loc[m, "cell_id"])  # noqa: E731
    it = {
        "ALL": s(c3.c3),
        "C5-BTBP publication (pub_6a0ef8528c)": s(c3.publication_id == C5_PUB),
        "  C5-BTBP cell c950dc2982e96b83": {C5_CELL},
        "  C5-BTBP other 14 cells": s((c3.publication_id == C5_PUB) & (c3.cell_id != C5_CELL)),
        "DMDPhPDA publication (pub_8415538291), 10 cells": s(c3.publication_id == DMD_PUB),
        "  DMDPhPDA copy A (curated, Table 1)": s((c3.publication_id == DMD_PUB) & ~c3.ornl_import_all),
        "  DMDPhPDA copy B (ORNL, V2-1)": s((c3.publication_id == DMD_PUB) & c3.ornl_import_all),
        "  DMDPhPDA 5 M cells (A + B)": {DMD_A5, DMD_B5},
        "ORNL import-only, no location (54)": s(c3.ornl_import_all & c3.noloc),
        "  ORNL-54 without DMDPhPDA copy B (49)": s(c3.ornl_import_all & c3.noloc & (c3.publication_id != DMD_PUB)),
        "curated, no location (25)": s(~c3.ornl_import_all & c3.noloc),
        "  curated no-location without C5-BTBP (10)": s(~c3.ornl_import_all & c3.noloc & (c3.publication_id != C5_PUB)),
        "all no-location (79)": s(c3.noloc),
        "located (table/figure named, 150)": s(~c3.noloc),
        # the 10 Aurora Clark cells carry their location in comments_raw ("table1", "fig2 raffinate", "fig3", "fig3b")
        "  location only in archive comments (10)": s(~c3.ornl_import_all & c3.noloc & (c3.publication_id != C5_PUB)),
        "true no-location: ORNL-54 + C5-BTBP IDEaL (69)": s(c3.noloc & (c3.ornl_import_all | (c3.publication_id == C5_PUB))),
        "located incl. comment locations (160)": s(~(c3.noloc & (c3.ornl_import_all | (c3.publication_id == C5_PUB)))),
        "9 ORNL-only clusters": s(c3.source_cluster.isin(
            c3.groupby("source_cluster").apply(lambda h: bool((h.ornl_import_all & h.noloc).all()), include_groups=False)
            .loc[lambda v: v].index)),
        "round-3 V2 flag, any (23)": s(c3.v2_any),
        "  V2-1 ORNL re-import (D1 + D4)": s(c3.v2_1_ornl_reimport),
        "  V2-2 duplicate entry (D2)": s(c3.v2_2_duplicate_entry),
        "  V2-3 hidden merge (D3)": s(c3.v2_3_hidden_merge),
        "  V2-5 resolution floor (D5)": s(c3.v2_5_resolution_floor),
    }
    it["UNION C5 + DMDPhPDA + ORNL-54 + V2"] = (it["C5-BTBP publication (pub_6a0ef8528c)"] | it["DMDPhPDA publication (pub_8415538291), 10 cells"]
                                                | it["ORNL import-only, no location (54)"] | it["round-3 V2 flag, any (23)"])
    return it


def row_masks(t: pd.DataFrame, it: dict) -> dict[str, np.ndarray]:
    m = {k: t.cell_id.isin(v).to_numpy() for k, v in it.items()}
    light = t.pair.isin(["La/Ce", "Ce/Pr", "Pr/Nd"]).to_numpy()
    m["  C5-BTBP light-end rows (La/Ce, Ce/Pr, Pr/Nd)"] = m["C5-BTBP publication (pub_6a0ef8528c)"] & light
    m["  DMDPhPDA copy-B 5 M Pr/Nd row"] = (t.cell_id.eq(DMD_B5) & t.pair.eq("Pr/Nd")).to_numpy()
    return m


def accounting(t: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """Per item: rows/seed; share of B1 SSE; share of B1's gain over B0; model-agnostic share; R2 without the item."""
    out = []
    seeds = sorted(t.split_seed.unique())
    tot = {s: {a: float(((g.y - g[a]) ** 2).sum()) for a in ARMS} for s, g in t.groupby("split_seed")}
    for name, m in masks.items():
        rec = {"item": name, "rows/seed": int(m.sum() / len(seeds)), "cells": t.loc[m, "cell_id"].nunique(),
               "pubs": t.loc[m, "publication"].nunique()}
        sh, gs, shmax, mse_ratio, r2b0, r2b1, d = [], [], [], [], [], [], []
        for s in seeds:
            sel = (t.split_seed == s).to_numpy()
            g, gi = t[sel], t[sel & m]
            sse = {a: float(((gi.y - gi[a]) ** 2).sum()) for a in ARMS}
            sh.append(sse["B1"] / tot[s]["B1"])
            gs.append((sse["B0"] - sse["B1"]) / (tot[s]["B0"] - tot[s]["B1"]))
            shmax.append(max(sse[a] / tot[s][a] for a in ARMS))
            mse_ratio.append((sse["B1"] / max(len(gi), 1)) / (tot[s]["B1"] / len(g)))
            if name != "ALL":
                go = t[sel & ~m]
                b0, b1 = r2(go.y, go.B0), r2(go.y, go.B1)
            else:
                b0, b1 = r2(g.y, g.B0), r2(g.y, g.B1)
            r2b0.append(b0)
            r2b1.append(b1)
            d.append(b1 - b0)
        rec.update({"B1 SSE share": np.mean(sh), "share min": np.min(sh), "share max": np.max(sh),
                    "gain share": np.mean(gs), "gain min": np.min(gs), "gain max": np.max(gs),
                    "max-arm SSE share": np.mean(shmax), "MSE/avg": np.mean(mse_ratio),
                    "R2 B0 w/o": np.mean(r2b0), "R2 B1 w/o": np.mean(r2b1), "B1-B0 w/o": np.mean(d),
                    "B1-B0 w/o min seed": np.min(d)})
        out.append(rec)
    return pd.DataFrame(out)


def per_seed_r2(t, ycol="y", arms=("B0", "B1")):
    return {a: [r2(g[ycol], g[a]) for _, g in t.groupby("split_seed")] for a in arms}


def summarise(label, t, ycol="y", note=""):
    r = per_seed_r2(t, ycol)
    b0, b1 = np.mean(r["B0"]), np.mean(r["B1"])
    d = np.array(r["B1"]) - np.array(r["B0"])
    return {"scenario": label, "rows/seed": int(len(t) / t.split_seed.nunique()), "R2 B0": b0, "R2 B1": b1,
            "B1-B0": b1 - b0, "B1-B0 min seed": d.min(), "B1-B0 max seed": d.max(),
            "R2 B1 per seed": ";".join(f"{v:.4f}" for v in r["B1"]), "note": note}


def interval_r2(t: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, arm: str):
    """Per seed R2 with interval targets: error = distance from the prediction to [lo, hi] (0 inside);
    point rows have lo = hi = y.  SST uses the V1 y of the kept rows (placeholder for interval rows)."""
    out = []
    for s, g in t.groupby("split_seed"):
        ix = g.index.to_numpy()
        p = g[arm].to_numpy()
        e = np.where(p < lo[ix], lo[ix] - p, np.where(p > hi[ix], p - hi[ix], 0.0))
        y = g.y.to_numpy()
        out.append(1.0 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return out


def floor_scenarios(t: pd.DataFrame, bench, c5_cells: set, floor: float):
    """C5-BTBP readings below `floor` treated as censored (true log D <= floor).  Only C5-BTBP cells are touched."""
    f = bench.frame.reset_index(drop=True)
    ci = dict(zip(f.cell_id, range(len(f))))
    t = t.reset_index(drop=True)
    yA = np.array([bench.Y[ci[c], IDX[a]] for c, a in zip(t.cell_id, t.A)])
    yB = np.array([bench.Y[ci[c], IDX[b]] for c, b in zip(t.cell_id, t.B)])
    inC5 = t.cell_id.isin(c5_cells).to_numpy()
    assert np.allclose((yA - yB)[inC5], t.y.to_numpy()[inC5], atol=1e-9), "bench log D does not reproduce y"
    cA, cB = inC5 & (yA < floor), inC5 & (yB < floor)
    both, one = cA & cB, (cA ^ cB)
    lo, hi = t.y.to_numpy().copy(), t.y.to_numpy().copy()
    # A censored, B point: log SF = logD(A) - yB <= floor - yB
    lo[cA & ~cB], hi[cA & ~cB] = -np.inf, floor - yB[cA & ~cB]
    # B censored, A point: log SF >= yA - floor
    lo[cB & ~cA], hi[cB & ~cA] = yA[cB & ~cA] - floor, np.inf
    keep = ~both
    tk = t[keep]
    ib0 = interval_r2(tk, lo, hi, "B0")
    ib1 = interval_r2(tk, lo, hi, "B1")
    # readings censored in C5 cells (unique cell x metal)
    c5f = f[f.cell_id.isin(c5_cells)]
    nread = int(np.isfinite(bench.Y[c5f.index]).sum())
    ncens = int((bench.Y[c5f.index] < floor).sum())
    drop = summarise(f"C5 floor {floor:+.1f}: drop every row touching a censored reading", t[~(cA | cB)])
    inter = {"scenario": f"C5 floor {floor:+.1f}: interval targets (both-censored rows dropped)",
             "rows/seed": int(keep.sum() / t.split_seed.nunique()), "R2 B0": np.mean(ib0), "R2 B1": np.mean(ib1),
             "B1-B0": np.mean(ib1) - np.mean(ib0), "B1-B0 min seed": float(np.min(np.array(ib1) - np.array(ib0))),
             "B1-B0 max seed": float(np.max(np.array(ib1) - np.array(ib0))),
             "R2 B1 per seed": ";".join(f"{v:.4f}" for v in ib1),
             "note": f"{ncens}/{nread} C5 readings censored; rows/seed both {int(both.sum()/5)}, one {int(one.sum()/5)}"}
    return drop, inter


def dmd_swap(t: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """Replace cell `dst`'s rows (y and every prediction) by cell `src`'s rows for the same seed and pair.
    Valid because the two cells share fold, read set and metals (asserted)."""
    t = t.copy()
    for s in t.split_seed.unique():
        a = t[(t.split_seed == s) & (t.cell_id == src)].set_index("pair")
        b = t[(t.split_seed == s) & (t.cell_id == dst)]
        assert (a.read_set.iloc[0] == b.read_set.iloc[0]) and (a.fold.iloc[0] == b.fold.iloc[0]), "read set / fold differ"
        for col in ("y",) + ARMS:
            t.loc[b.index, col] = a.loc[b.pair, col].to_numpy()
    return t


def scenarios(t: pd.DataFrame, bench, masks: dict, it: dict) -> pd.DataFrame:
    rows = [summarise("V1 as is (reference)", t)]
    m = masks
    c5 = m["C5-BTBP publication (pub_6a0ef8528c)"]
    rows.append(summarise("C5: publication unverifiable -> stratum removed", t[~c5]))
    rows.append(summarise("C5: light-end rows removed (La/Ce, Ce/Pr, Pr/Nd)", t[~m["  C5-BTBP light-end rows (La/Ce, Ce/Pr, Pr/Nd)"]]))
    rows.append(summarise("C5: cell c950dc2982e96b83 removed", t[~m["  C5-BTBP cell c950dc2982e96b83"]]))
    for fl in (-4.0, -3.5):
        rows += list(floor_scenarios(t, bench, it["C5-BTBP publication (pub_6a0ef8528c)"], fl))
    # DMDPhPDA
    rows.append(summarise("DMD D-1: Figure 5 right (copy B 5 M rows := copy A 5 M rows)", dmd_swap(t, DMD_A5, DMD_B5)))
    rows.append(summarise("DMD D-2: Table 1 as printed right (copy A 5 M rows := copy B 5 M rows)", dmd_swap(t, DMD_B5, DMD_A5)))
    rows.append(summarise("DMD D-3: copy B quarantined as duplicate (V2-1 part, 5 cells)", t[~m["  DMDPhPDA copy B (ORNL, V2-1)"]]))
    rows.append(summarise("DMD D-4: publication unverifiable -> removed (10 cells)", t[~m["DMDPhPDA publication (pub_8415538291), 10 cells"]]))
    rows.append(summarise("DMD D-1 + D-3: copy B removed and Figure 5 kept for copy A", t[~m["  DMDPhPDA copy B (ORNL, V2-1)"]],
                          note="identical to D-3 (copy A already holds the Figure 5 values)"))
    # ORNL imports
    rows.append(summarise("ORNL-54 unverifiable -> removed", t[~m["ORNL import-only, no location (54)"]]))
    rows.append(summarise("ORNL-49 (without DMD copy B) removed", t[~m["  ORNL-54 without DMDPhPDA copy B (49)"]]))
    rows.append(summarise("all no-location (79) removed: located-only", t[~m["all no-location (79)"]]))
    rows.append(summarise("V2 flagged (23) removed", t[~m["round-3 V2 flag, any (23)"]]))
    rows.append(summarise("UNION removed (C5 + DMD + ORNL-54 + V2)", t[~m["UNION C5 + DMDPhPDA + ORNL-54 + V2"]]))
    # combined best case for V1 levels: C5 floor -4 drop + DMD D-1
    t2 = dmd_swap(t, DMD_A5, DMD_B5)
    d, _ = floor_scenarios(t2, bench, it["C5-BTBP publication (pub_6a0ef8528c)"], -4.0)
    d["scenario"] = "combined: C5 floor -4 (drop) + DMD D-1"
    rows.append(d)
    return pd.DataFrame(rows)


def excess_bound(t: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """Best case for a recovered stratum if all of its B1 MSE above the located-cell B1 MSE were transcription error:
    remove that excess from SSE and SST (rows kept)."""
    loc = masks["located (table/figure named, 150)"]
    out = []
    for name in ("ORNL import-only, no location (54)", "  ORNL-54 without DMDPhPDA copy B (49)", "curated, no location (25)",
                 "C5-BTBP publication (pub_6a0ef8528c)"):
        m = masks[name]
        res = []
        for s, g in t.groupby("split_seed"):
            sel = (t.split_seed == s).to_numpy()
            e = (t.y - t.B1) ** 2
            mse_loc = float(e[sel & loc].mean())
            sse_i = float(e[sel & m].sum())
            exc = max(sse_i - mse_loc * (sel & m).sum(), 0.0)
            sst = float(((g.y - g.y.mean()) ** 2).sum())
            sse = float(((g.y - g.B1) ** 2).sum())
            res.append((exc, exc / sse, (1 - (sse - exc) / (sst - exc)) - (1 - sse / sst), mse_loc, sse_i / (sel & m).sum()))
        r = np.array(res)
        out.append({"stratum": name, "excess SSE/seed": r[:, 0].mean(), "excess share of B1 SSE": r[:, 1].mean(),
                    "best-case dR2 B1": r[:, 2].mean(), "B1 MSE located": r[:, 3].mean(), "B1 MSE stratum": r[:, 4].mean()})
    return pd.DataFrame(out)


def per_document(t: pd.DataFrame, c3: pd.DataFrame) -> pd.DataFrame:
    meta = c3.groupby("publication_id").agg(doi=("doi", "first"), year=("year", "first"), cells=("cell_id", "size"),
                                            ornl=("ornl_import_all", "sum"), noloc=("noloc", "sum"), v2=("v2_any", "sum"),
                                            cluster=("source_cluster", "first"), curator=("curator", "first"),
                                            title=("title", "first"))
    seeds = sorted(t.split_seed.unique())
    tot = {s: {a: float(((g.y - g[a]) ** 2).sum()) for a in ARMS} for s, g in t.groupby("split_seed")}
    rows = []
    for p, g in t.groupby("publication"):
        sse = {a: np.mean([((h.y - h[a]) ** 2).sum() / tot[s][a] for s, h in g.groupby("split_seed")]) for a in ARMS}
        gain = np.mean([(((h.y - h.B0) ** 2).sum() - ((h.y - h.B1) ** 2).sum()) / (tot[s]["B0"] - tot[s]["B1"])
                        for s, h in g.groupby("split_seed")])
        rows.append({"publication": p, "rows/seed": len(g) // len(seeds), "B1 SSE share": sse["B1"],
                     "max-arm SSE share": max(sse.values()), "gain share": gain})
    d = pd.DataFrame(rows).set_index("publication").join(meta)
    d["class"] = np.where(d.noloc == 0, "located", np.where(d.ornl == d.cells, "import-only",
                          np.where(d.ornl == 0, "curated-noloc", "mixed")))
    d["stake"] = np.maximum(d["max-arm SSE share"], d["gain share"].abs())
    return d.sort_values("stake", ascending=False)


def per_cluster(t: pd.DataFrame, c3: pd.DataFrame) -> pd.DataFrame:
    cl = dict(zip(c3.cell_id, c3.source_cluster))
    orn = c3.assign(o=c3.ornl_import_all & c3.noloc).groupby("source_cluster").o.agg(["sum", "size"])
    t = t.assign(cluster=t.cell_id.map(cl))
    tot = {s: {a: float(((g.y - g[a]) ** 2).sum()) for a in ARMS} for s, g in t.groupby("split_seed")}
    rows = []
    for c, g in t.groupby("cluster"):
        gain = [(((h.y - h.B0) ** 2).sum() - ((h.y - h.B1) ** 2).sum()) for _, h in g.groupby("split_seed")]
        rows.append({"cluster": c, "rows/seed": len(g) // 5, "cells": g.cell_id.nunique(), "pubs": g.publication.nunique(),
                     "ORNL-54 cells": int(orn.loc[c, "sum"]), "all C3 cells": int(orn.loc[c, "size"]),
                     "B1 SSE share": np.mean([((h.y - h.B1) ** 2).sum() / tot[s]["B1"] for s, h in g.groupby("split_seed")]),
                     "gain share": np.mean([gn / (tot[s]["B0"] - tot[s]["B1"]) for gn, (s, _) in zip(gain, g.groupby("split_seed"))]),
                     "gain SSE/seed": np.mean(gain), "gain>0 seeds": int(sum(x > 0 for x in gain))})
    d = pd.DataFrame(rows).sort_values("rows/seed", ascending=False)
    d["ORNL-only"] = d["ORNL-54 cells"] == d["all C3 cells"]
    return d


def gd_contrast(t: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """Label-based descriptive: mean B0 residual on Eu/Gd minus mean on Gd/Tb (log units), and B1's gain on Gd pairs."""
    out = []
    for name in ("ALL", "located (table/figure named, 150)", "C5-BTBP publication (pub_6a0ef8528c)",
                 "DMDPhPDA publication (pub_8415538291), 10 cells", "  DMDPhPDA copy A (curated, Table 1)",
                 "ORNL import-only, no location (54)", "  ORNL-54 without DMDPhPDA copy B (49)",
                 "  curated no-location without C5-BTBP (10)", "round-3 V2 flag, any (23)"):
        m = masks[name]
        g = t[m]
        r = g.y - g.B0
        eg, gt = r[g.pair == "Eu/Gd"], r[g.pair == "Gd/Tb"]
        gd = g.pair.isin(["Eu/Gd", "Gd/Tb"])
        gain_gd = (((g.y - g.B0) ** 2 - (g.y - g.B1) ** 2)[gd]).sum() / 5
        out.append({"item": name, "Eu/Gd rows/seed": len(eg) // 5, "Gd/Tb rows/seed": len(gt) // 5,
                    "mean r(Eu/Gd)": eg.mean(), "mean r(Gd/Tb)": gt.mean(), "contrast": eg.mean() - gt.mean(),
                    "Gd-pair gain SSE/seed": gain_gd})
    return pd.DataFrame(out)


def sig_figs(s: str) -> int | None:
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v <= 0 or not math.isfinite(v):
        return None
    if len(s.replace("-", "").replace(".", "").replace("e", "").replace("E", "").replace("+", "")) > 9:
        s = f"{v:.8g}"                     # float noise from a log -> D conversion (ORNL imports)
    mant = s.lower().split("e")[0].lstrip("+-")
    digits = mant.replace(".", "").lstrip("0")
    if "." not in mant:
        digits = digits.rstrip("0") or digits
    return max(len(digits), 1)


def rounding_bound(t: pd.DataFrame, bench, masks: dict) -> pd.DataFrame:
    """Label-free in intent: expected squared rounding error of each strict-adjacent log SF from the reported
    significant figures of D (uniform rounding; a cell value averaged over n records gets var / n)."""
    a = pd.read_parquet(ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet",
                        columns=["source_record_id", "metal_symbol", "D_raw", "D_value"])
    a["sid"] = a.source_record_id.astype(int)
    a["sf"] = a.D_raw.map(sig_figs)
    v = a.D_value.astype(float)
    # half-width of the last reported digit, in log10 units
    expo = np.floor(np.log10(v.where(v > 0)))
    hw_D = 0.5 * 10.0 ** (expo - a.sf.astype(float) + 1)
    a["var_log"] = (hw_D / (v * math.log(10))) ** 2 / 3.0
    f = bench.frame.reset_index(drop=True)
    var = {}
    nsf1 = {}
    for _, r in f.iterrows():
        ids = [int(x.split(":")[-1]) for x in str(r.safe_exp_ids).split(";") if x]
        g = a[a.sid.isin(ids)]
        for m, h in g.groupby("metal_symbol"):
            if m in IDX:
                var[(r.cell_id, m)] = float(h.var_log.mean()) / len(h)
                nsf1[(r.cell_id, m)] = bool((h.sf == 1).all())
    t = t.assign(rv=[var.get((c, x), np.nan) + var.get((c, y), np.nan) for c, x, y in zip(t.cell_id, t.A, t.B)],
                 sf1=[nsf1.get((c, x), False) or nsf1.get((c, y), False) for c, x, y in zip(t.cell_id, t.A, t.B)])
    out = []
    tot = {s: float(((g.y - g.B1) ** 2).sum()) for s, g in t.groupby("split_seed")}
    for name in ("ALL", "ORNL import-only, no location (54)", "located (table/figure named, 150)", "curated, no location (25)",
                 "C5-BTBP publication (pub_6a0ef8528c)", "DMDPhPDA publication (pub_8415538291), 10 cells"):
        m = masks[name]
        g = t[m]
        rv = g.groupby("split_seed").rv.sum().mean()
        out.append({"item": name, "rows/seed": int(m.sum() / 5), "rows with a 1-s.f. reading": int(g.sf1.sum() / 5),
                    "missing var": int(g.rv.isna().sum() / 5), "expected rounding SSE/seed": rv,
                    "as share of total B1 SSE": rv / np.mean(list(tot.values())),
                    "as share of item B1 SSE": rv / (((g.y - g.B1) ** 2).sum() / 5),
                    "rounding sd of log SF (rms)": math.sqrt(rv / max(m.sum() / 5, 1))})
    return pd.DataFrame(out)


def census_order(c3: pd.DataFrame) -> dict:
    imp = c3[c3.ornl_import_all & c3.noloc]
    dois = sorted(imp.doi.unique())
    rng = np.random.default_rng(CENSUS_SEED)
    order = [dois[i] for i in rng.permutation(len(dois))]
    txt = "\n".join(order)
    return {"seed": CENSUS_SEED, "n": len(order), "order": order,
            "sha256": hashlib.sha256(txt.encode()).hexdigest(),
            "cells_per_doi": imp.groupby("doi").size().to_dict(),
            "pairs_per_doi": imp.groupby("doi").n_queries.sum().astype(int).to_dict()}


def overlaps(it: dict) -> pd.DataFrame:
    keys = ["C5-BTBP publication (pub_6a0ef8528c)", "DMDPhPDA publication (pub_8415538291), 10 cells",
            "ORNL import-only, no location (54)", "round-3 V2 flag, any (23)", "curated, no location (25)"]
    return pd.DataFrame([[len(it[a] & it[b]) for b in keys] for a in keys], index=keys, columns=[k[:18] for k in keys])


def main():
    """Sections (argv, default all): head, acc, scen, docs.  acc / scen run for BPG and BP."""
    want = set(sys.argv[1:]) or {"head", "acc", "scen", "docs"}
    oof, c3 = load()
    bench = D.load_bench()
    it = items(c3)
    print("EXPLORATORY: computed on the 229 C3 cells / 5 dev seeds already scored in rounds 3-4.\n")
    if "head" in want:
        print("== item overlaps (cells) ==")
        print(overlaps(it).to_string())
        print("\nV2 cells by rule and publication:")
        v2 = c3[c3.v2_any][["cell_id", "publication_id", "doi", "data_location", "ornl_import_all", "v2_1_ornl_reimport",
                             "v2_2_duplicate_entry", "v2_3_hidden_merge", "v2_5_resolution_floor", "n_queries",
                             "aqueous_complexant"]]
        print(v2.to_string(index=False))
    for d in ("BPG", "BP"):
        t = oof[d].reset_index(drop=True)
        masks = row_masks(t, it)
        print(f"\n================ design {d} ================")
        if "acc" in want:
            acc = accounting(t, masks)
            print("== accounting (seed means; shares are of the seed's total) ==")
            print(acc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if "scen" in want:
            print("\n== scenarios (row-level sensitivities; V1 never edited) ==")
            sc = scenarios(t, bench, masks, it)
            print(sc.drop(columns=["R2 B1 per seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("\n== best-case bound: B1 excess MSE over located cells treated as transcription error ==")
            print(excess_bound(t, masks).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if d == "BPG" and "docs" in want:
            print("\n== per document (union stake = max(max-arm SSE share, |gain share|); value-informed ranking) ==")
            pdoc = per_document(t, c3)
            show = pdoc[(pdoc["class"] != "located") | (pdoc.v2 > 0)]
            print(show.drop(columns=["title"]).to_string(float_format=lambda v: f"{v:.4f}"))
            print("\nlocated-only documents in the top 15 by stake:")
            print(pdoc.head(15)[["class", "doi", "rows/seed", "B1 SSE share", "gain share", "stake"]].to_string(
                float_format=lambda v: f"{v:.4f}"))
            print("\n== per source cluster ==")
            print(per_cluster(t, c3).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("\n== Gd contrast (B0 residual Eu/Gd minus Gd/Tb; descriptive, label-based) ==")
            print(gd_contrast(t, masks).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("\n== rounding-noise bound from reported significant figures (label-free in intent) ==")
            print(rounding_bound(t, bench, masks).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("\n== DMDPhPDA 5 M rows, seed 104729 ==")
            x = t[(t.split_seed == 104729) & t.cell_id.isin([DMD_A5, DMD_B5])]
            print(x[["cell_id", "read_set", "pair", "y", "B0", "B1"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("\n== C5-BTBP log D per cell (bench) ==")
            f = bench.frame.reset_index(drop=True)
            cc = c3[c3.publication_id == C5_PUB][["cell_id", "acid_M", "extractant_M"]]
            Y = pd.DataFrame(bench.Y[[f.index[f.cell_id == c][0] for c in cc.cell_id]], columns=LN, index=cc.cell_id)
            print(pd.concat([cc.set_index("cell_id"), Y.round(2)], axis=1).to_string())
            print("readings <= -4.0:", int((Y.to_numpy() < -4.0).sum()), " <= -3.5:", int((Y.to_numpy() < -3.5).sum()),
                  " of", int(np.isfinite(Y.to_numpy()).sum()))
            allY = bench.Y[[f.index[f.cell_id == c][0] for c in c3.cell_id]]
            print("all C3 readings < -4.0:", int((allY < -4.0).sum()), "in cells:",
                  int((np.nan_to_num(allY, nan=0) < -4.0).any(axis=1).sum()))
            print("\n== label-free census order of the import-only DOIs ==")
            print(json.dumps(census_order(c3), indent=1))


if __name__ == "__main__":
    main()
