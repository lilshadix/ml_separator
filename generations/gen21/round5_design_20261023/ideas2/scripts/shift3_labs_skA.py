#!/usr/bin/env python
"""Skeptic A for idea shift-3, second pass (EXPLORATORY; all bench cells already seen).

  L1  laboratory units: the 11 Ce/Pr publications collapse to research groups by shared senior authors (read from
      the round-4 provenance DOI table and the P4 corpus registry author strings; the mapping is written out below
      and was fixed by author names only, before looking at any slope by lab).
  L2  raw light-end log D of the 2-cell series (to see floor ties / identical values).
  L3  La/Nd per publication (which publications have La and Nd but no Ce/Pr, and their sign).
  L4  power of the pre-registered prospective sign test (n = 8, 10, 13 new units; one-sided sign test at 0.05 and the
      '< 70 % negative' kill) under per-unit negative rates estimated from the bench (lab-level, no C5).
  L5  predictive leg variants on bench groups: prior = median of other LABS excluding C5; scored pooled and
      equal-weight per lab.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift3_labs_skA.py
"""
from __future__ import annotations

import itertools
import sys
from math import comb

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES as LN  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
C5 = "pub_6a0ef8528c"
# Fixed from author strings (sources_by_doi.csv + P4_corpus_registry.csv): senior/shared authors.
LAB = {"pub_b0b3296ec5": "JAEA-Sasaki", "pub_1605e436d7": "JAEA-Sasaki", "pub_d68ed8c8cf": "JAEA-Sasaki",
       "pub_af0c74798c": "JAEA-Sasaki", "pub_8415538291": "JAEA-Tachimori/Yaita",
       "pub_1c1b558856": "KIT/JRC-Geist", "pub_37086f6378": "KIT/JRC-Geist",
       "pub_6a0ef8528c": "Chalmers-Ekberg", "pub_19783c039b": "CN-Tian", "pub_4cba27f5f3": "An/Qi",
       "pub_0fd6ab4e40": "Moscow-Petrov/Matveev", "pub_c8195b4ca7": "Mowafy", "pub_b9dfc468cd": "Mowafy",
       "pub_f8a621ede4": "CN-Sun/Cui", "pub_0a8da089ad": "CN-Sun/Cui", "pub_bdb1564c49": "CN-Sun/Cui",
       "pub_ab5ae597b9": "Yuan/Gong", "pub_392f966ef2": "Nakamura/Miyake"}
LAB_JAEA_MERGED = {k: ("JAEA" if v.startswith("JAEA") else v) for k, v in LAB.items()}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 0.1:
        return np.nan
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def signflip(v, alt="one"):
    v = np.asarray(v, float)
    S = np.array(list(itertools.product((1, -1), repeat=len(v))))
    null = S @ v
    if alt == "two":
        return float((np.abs(null) >= abs(v.sum()) - 1e-12).mean())
    return float((null <= v.sum() + 1e-12).mean())


def label(f, pre):
    cols = [c for c in f.columns if c.startswith(pre)]
    M = f[cols].to_numpy()
    return ["+".join(c[len(pre):] for c, v in zip(cols, row) if v == 1) or "-" for row in M]


def binom_tail(n, k, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float).copy()
    v123 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    Y[f.cell_id.isin(v123).to_numpy()] = np.nan
    f["acid"] = label(f, "cond__acid__")
    f["dil"] = label(f, "cond__diluent__")
    f["add"] = label(f, "cond__additive__")
    f["Lc"] = f.cond__extractant_concentration_M.astype(float).round(4)
    f["T"] = f.cond__temperature_C.astype(float).round(0)
    f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))
    keep = f.x.notna() & ~f.cell_id.isin(v123)
    G = []
    for k, g in f[keep].groupby(["publication_id", "extractant", "acid", "dil", "add", "Lc", "T"]):
        if len(g) >= 2 and np.ptp(g.x) >= 0.1:
            G.append({"pub": k[0], "acid": k[2], "idx": g.index.to_numpy(), "x": g.x.to_numpy(),
                      "ext": str(g.extractant_name.iloc[0])[:14], "cells": list(g.cell_id)})

    def pslopes(a, bb, win=None):
        rows = []
        for gi, g in enumerate(G):
            ya, yb = Y[g["idx"], a].copy(), Y[g["idx"], bb].copy()
            if win is not None:
                bad = (ya < win[0]) | (ya > win[1]) | (yb < win[0]) | (yb > win[1])
                ya[bad] = np.nan
            s = slope(g["x"], ya - yb)
            if np.isfinite(s):
                rows.append({"g": gi, "pub": g["pub"], "slope": s})
        return pd.DataFrame(rows)

    print("=" * 90 + "\nL1: laboratory units for Ce/Pr\n" + "=" * 90)
    for win in (None, (-2, 2.5), (-1.5, 2.0)):
        P = pslopes(1, 2, win)
        pub = P.groupby("pub").slope.mean()
        for nm, lab in (("lab (JAEA split)", LAB), ("lab (JAEA merged)", LAB_JAEA_MERGED)):
            lv = pub.groupby(pub.index.map(lab)).mean()
            lvn = lv.drop("Chalmers-Ekberg", errors="ignore")
            print(f"window {win} {nm}: {len(lv)} labs, negative {int((lv < 0).sum())}; one-sided p {signflip(lv.to_numpy()):.4f}, "
                  f"two-sided {signflip(lv.to_numpy(), 'two'):.4f} | no C5: {len(lvn)} labs, negative {int((lvn < 0).sum())}, "
                  f"one-sided p {signflip(lvn.to_numpy()):.4f}, two-sided {signflip(lvn.to_numpy(), 'two'):.4f}")
            if win is None and nm.endswith("merged)"):
                print("   lab means:", lv.round(3).to_dict())
    print("publications per lab among the 11 Ce/Pr publications:",
          pd.Series([LAB_JAEA_MERGED[p] for p in pslopes(1, 2).pub.unique()]).value_counts().to_dict())

    print("\n" + "=" * 90 + "\nL2: raw light-end log D in Ce/Pr groups with <= 3 cells\n" + "=" * 90)
    for gi, g in enumerate(G):
        yy = Y[g["idx"], 1] - Y[g["idx"], 2]
        if np.isfinite(yy).sum() >= 2 and len(g["idx"]) <= 3:
            print(f"G{gi} {g['pub']} {g['ext']} {g['acid']}")
            for i, c in enumerate(g["cells"]):
                print(f"   {c} [acid] {10 ** g['x'][i]:.3f} M  La..Nd {np.round(Y[g['idx'][i], :4], 3)}  "
                      f"Yb,Lu {np.round(Y[g['idx'][i], 12:], 3)}")

    print("\n" + "=" * 90 + "\nL3: La/Nd and La/Pr per publication\n" + "=" * 90)
    LaNd = pslopes(0, 3).groupby("pub").slope.mean()
    CePr = pslopes(1, 2).groupby("pub").slope.mean()
    LaPr = pslopes(0, 2).groupby("pub").slope.mean()
    T = pd.DataFrame({"La/Nd": LaNd, "La/Pr": LaPr, "Ce/Pr": CePr})
    T["lab"] = T.index.map(LAB_JAEA_MERGED)
    print(T.round(3).to_string())
    noCP = T[T["Ce/Pr"].isna() & T["La/Nd"].notna()]
    print(f"publications with La/Nd but no Ce/Pr slope: {len(noCP)}; La/Nd negative in {int((noCP['La/Nd'] < 0).sum())}")

    print("\n" + "=" * 90 + "\nL4: power of the prospective one-sided sign test\n" + "=" * 90)
    pub = CePr.drop(C5)
    lv = pub.groupby(pub.index.map(LAB_JAEA_MERGED)).mean()
    print(f"bench (no C5) publication-level slopes: mean {pub.mean():.3f}, sd {pub.std(ddof=1):.3f}; lab-level mean "
          f"{lv.mean():.3f}, sd {lv.std(ddof=1):.3f}")
    for p in (0.95, 0.9, 0.8, 0.7, 0.6):
        out = []
        for n in (6, 8, 10, 13):
            kmin = next(k for k in range(n + 1) if binom_tail(n, k, 0.5) <= 0.05)
            k70 = int(np.ceil(0.7 * n))
            out.append(f"n={n}: need >= {kmin} neg (p<=.05) -> power {binom_tail(n, kmin, p):.2f}; "
                       f"'>= 70 %' alone ({k70}) {binom_tail(n, k70, p):.2f}")
        print(f"true per-unit P(negative) = {p}: " + " | ".join(out))

    print("\n" + "=" * 90 + "\nL5: predictive tilt, prior = median of other labs, no C5 in prior\n" + "=" * 90)
    P = pslopes(1, 2)
    pubs = P.groupby("pub").slope.mean()
    labs = pubs.groupby(pubs.index.map(LAB_JAEA_MERGED)).mean()
    rows = []
    for gi, g in enumerate(G):
        yy = Y[g["idx"], 1] - Y[g["idx"], 2]
        for i in range(len(yy)):
            oth = np.isfinite(yy) & (np.arange(len(yy)) != i)
            if not np.isfinite(yy[i]) or oth.sum() < 1:
                continue
            mylab = LAB_JAEA_MERGED[g["pub"]]
            pri = labs.drop([mylab, "Chalmers-Ekberg"], errors="ignore")
            dx = g["x"][i] - g["x"][oth].mean()
            a = yy[oth].mean()
            rows.append({"pub": g["pub"], "lab": mylab, "y": yy[i], "acopy": a, "tilt_med": a + pri.median() * dx,
                         "tilt_mean": a + pri.mean() * dx, "dx": dx, "prior_med": pri.median()})
    E = pd.DataFrame(rows)
    E = E[E.pub != C5]
    print(f"prior (median of other labs, no C5) range {E.prior_med.min():.3f} .. {E.prior_med.max():.3f}")
    for arm in ("tilt_med", "tilt_mean"):
        sa = (E.y - E.acopy) ** 2
        st = (E.y - E[arm]) ** 2
        perl = E.assign(sa=sa, st=st).groupby("lab")[["sa", "st"]].sum()
        rel = (perl.sa - perl.st) / perl.sa
        perp = E.assign(sa=sa, st=st).groupby("pub")[["sa", "st"]].sum()
        relp = (perp.sa - perp.st) / perp.sa
        print(f"{arm}: rows {len(E)}; pooled rel SSE gain {(sa.sum() - st.sum()) / sa.sum():+.3f}; labs improved "
              f"{int((rel > 0).sum())}/{len(rel)} (equal-weight mean {rel.mean():+.3f}); publications improved "
              f"{int((relp > 0).sum())}/{len(relp)}; lab rel: {rel.round(2).to_dict()}")
        # without the single biggest contributor
        contrib = (perp.sa - perp.st)
        top = contrib.idxmax()
        print(f"   largest SSE-gain publication {top} ({contrib.max():.4f} of {contrib.sum():.4f}); "
              f"pooled without it {(contrib.drop(top).sum()) / perp.sa.drop(top).sum():+.3f}")


if __name__ == "__main__":
    main()
