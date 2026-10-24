#!/usr/bin/env python
"""Skeptic A for idea shift-3, third pass (EXPLORATORY; all bench cells already seen).

  F1  the three publications that have La and Nd but no Ce/Pr acid slope: raw series and windowed La/Nd slopes.
  F2  label-free 'flat-floor' cell rule: drop a cell whose measured log D span across metals is < 0.3 (all metals
      at one assay floor, e.g. pub_4cba27f5f3 at 0.1 M HCl); recount Ce/Pr and La/Pr per publication and per lab.
  F3  which publications survive the [-1.5, 2.0] window for Ce/Pr, and with how many cells.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift3_floor_skA.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lab", HERE / "shift3_labs_skA.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

pd.set_option("display.width", 250)


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float).copy()
    v123 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    Y[f.cell_id.isin(v123).to_numpy()] = np.nan
    f["acid"] = L.label(f, "cond__acid__")
    f["dil"] = L.label(f, "cond__diluent__")
    f["add"] = L.label(f, "cond__additive__")
    f["Lc"] = f.cond__extractant_concentration_M.astype(float).round(4)
    f["T"] = f.cond__temperature_C.astype(float).round(0)
    f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))
    span = np.nanmax(Y, 1) - np.nanmin(Y, 1)
    nmet = np.isfinite(Y).sum(1)
    flat = (nmet >= 4) & (span < 0.3)
    keep = f.x.notna() & ~f.cell_id.isin(v123)
    G = []
    for k, g in f[keep].groupby(["publication_id", "extractant", "acid", "dil", "add", "Lc", "T"]):
        if len(g) >= 2 and np.ptp(g.x) >= 0.1:
            G.append({"pub": k[0], "acid": k[2], "idx": g.index.to_numpy(), "x": g.x.to_numpy(),
                      "ext": str(g.extractant_name.iloc[0])[:14], "cells": list(g.cell_id)})

    print("=" * 90 + "\nF1: publications with La/Nd but no Ce/Pr\n" + "=" * 90)
    for g in G:
        if g["pub"] in ("pub_b9dfc468cd", "pub_f8a621ede4", "pub_392f966ef2"):
            la, nd = Y[g["idx"], 0], Y[g["idx"], 3]
            if np.isfinite(la - nd).sum() >= 2:
                o = np.argsort(g["x"])
                print(f"{g['pub']} {g['ext']} {g['acid']}: La/Nd slope {L.slope(g['x'], la - nd):+.3f}")
                for i in o:
                    print(f"   [acid] {10 ** g['x'][i]:.2f} M  logD {np.round(Y[g['idx'][i]], 2)}")

    print("\n" + "=" * 90 + "\nF2: flat-floor cells dropped (span across metals < 0.3, >= 4 metals)\n" + "=" * 90)
    print("flat cells in acid groups:", [(g["pub"], c, round(10 ** g["x"][i], 3)) for g in G
                                         for i, c in enumerate(g["cells"]) if flat[g["idx"][i]]])
    for a, bb, nm in ((1, 2, "Ce/Pr"), (0, 2, "La/Pr"), (0, 1, "La/Ce"), (2, 3, "Pr/Nd")):
        rows = []
        for g in G:
            ok = ~flat[g["idx"]]
            s = L.slope(g["x"][ok], (Y[g["idx"], a] - Y[g["idx"], bb])[ok])
            if np.isfinite(s):
                rows.append({"pub": g["pub"], "slope": s})
        P = pd.DataFrame(rows).groupby("pub").slope.mean()
        lv = P.groupby(P.index.map(L.LAB_JAEA_MERGED)).mean()
        lvn = lv.drop("Chalmers-Ekberg", errors="ignore")
        print(f"{nm}: pubs {len(P)} negative {int((P < 0).sum())}; labs {len(lv)} negative {int((lv < 0).sum())} "
              f"(one-sided p {L.signflip(lv.to_numpy()):.4f}); labs no C5 {len(lvn)} negative {int((lvn < 0).sum())} "
              f"(one-sided p {L.signflip(lvn.to_numpy()):.4f}); 4cba slope {P.get('pub_4cba27f5f3', np.nan):+.3f}")

    print("\n" + "=" * 90 + "\nF3: Ce/Pr in the [-1.5, 2.0] window, per publication\n" + "=" * 90)
    for g in G:
        ce, pr = Y[g["idx"], 1].copy(), Y[g["idx"], 2].copy()
        bad = (ce < -1.5) | (ce > 2) | (pr < -1.5) | (pr > 2)
        ce[bad] = np.nan
        s = L.slope(g["x"], ce - pr)
        if np.isfinite(s):
            print(f"   {g['pub']} {g['ext']:14s} {g['acid']:6s} cells in window {int(np.isfinite(ce - pr).sum())} "
                  f"slope {s:+.3f} lab {L.LAB_JAEA_MERGED[g['pub']]}")


if __name__ == "__main__":
    main()
