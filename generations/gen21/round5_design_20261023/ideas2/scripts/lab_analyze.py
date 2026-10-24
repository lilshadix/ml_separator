#!/usr/bin/env python
"""E-LAB analysis of ideas2/lab_out/lab_rows.parquet (lab_calib.py).  EXPLORATORY (already-seen rows); prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_analyze.py

R^2: pooled inside each split seed over the population's rows, then the seed mean (round-4 convention).
rel SSE: 1 - SSE(arm) / SSE(B1) on the same rows, summed over seeds (positive = better than B1).
E_* columns hold an expected squared error (random-donor placebo); they are scored as squared errors.
Inference unit = publication: per-publication Delta SSE, count improved, leave-one-publication-out (LOPO) jackknife
t interval on rel SSE, exact one-sided sign-flip over publications, top-3 share.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4 import metrics as M4  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab_out")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
LIGHTP = {"La/Ce", "Ce/Pr", "Pr/Nd"}
CEP = {"La/Ce", "Ce/Pr"}
GDP = {"Eu/Gd", "Gd/Tb"}


def sqe(d, arm):
    return d[arm].to_numpy(float) if arm.startswith("E_") else (d[arm].to_numpy(float) - d.y.to_numpy(float)) ** 2


def r2(d, arm):
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1.0 - sqe(g, arm).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def rel(d, arm, base="B1"):
    return float(1.0 - sqe(d, arm).sum() / sqe(d, base).sum())


def infer(d, arm, base="B1"):
    """Publication-unit inference on rel SSE of arm vs base."""
    d = d.assign(_a=sqe(d, arm), _b=sqe(d, base))
    g = d.groupby("publication")[["_a", "_b"]].sum()
    dl = (g._b - g._a) / 5.0                        # per-publication SSE reduction per seed
    G = len(g)
    point = 1.0 - g._a.sum() / g._b.sum()
    jk = np.array([1.0 - g._a.drop(p).sum() / g._b.drop(p).sum() for p in g.index])
    se = float(np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum())) if G > 2 else np.nan
    q = SS.t.ppf(0.975, G - 1) if G > 2 else np.nan
    v = dl.to_numpy()
    obs = v.sum()
    if G <= 18:
        sums = np.array([np.dot(s, np.abs(v)) for s in itertools.product((-1, 1), repeat=G)])
        p = float((sums >= obs - 1e-12).mean())
    else:
        rng = np.random.default_rng(0)
        sums = (rng.choice((-1, 1), size=(200000, G)) * np.abs(v)).sum(1)
        p = float((sums >= obs - 1e-12).mean())
    tot = v[v > 0].sum()
    top3 = float(np.sort(v)[::-1][:3].sum() / obs) if obs > 0 else np.nan
    return {"rel": point, "lo": point - q * se, "hi": point + q * se, "G": G, "up": int((v > 1e-12).sum()),
            "down": int((v < -1e-12).sum()), "p_sf": p, "top3": top3, "jk_min": float(jk.min()),
            "jk_min_pub": g.index[int(np.argmin(jk))], "pos_mass": tot}


def show(title, d, arms, base="B1"):
    print(f"\n--- {title}: cells {d.cell_id.nunique()}, pubs {d.publication.nunique()}, rows/seed {len(d) / 5:.0f} ---")
    out = []
    for a in arms:
        if a not in d or d[a].isna().any():
            continue
        out.append({"arm": a, "R2": r2(d, a), "relSSE_vs_" + base: rel(d, a, base)})
    print(pd.DataFrame(out).round(4).to_string(index=False))


def show_inf(title, d, pairs):
    out = []
    for a, b in pairs:
        if d[a].isna().any() or d[b].isna().any():
            continue
        r = infer(d, a, b)
        out.append({"contrast": f"{a} - {b}", "dR2": r2(d, a) - r2(d, b), **{k: r[k] for k in
                    ("rel", "lo", "hi", "G", "up", "down", "p_sf", "top3", "jk_min", "jk_min_pub")}})
    print(f"\n[{title}] publication-unit inference (rel SSE, LOPO 95 %)")
    print(pd.DataFrame(out).round(4).to_string(index=False))


def pairshare(d, arm, base="B1"):
    d = d.assign(_g=sqe(d, base) - sqe(d, arm))
    s = d.groupby("pair")._g.sum() / 5.0
    tot = s.sum()
    return s, tot


def main() -> int:
    T = pd.read_parquet(OUT / "lab_rows.parquet")
    bench = D.load_bench()
    T = M4.add_clusters(T, bench.frame)
    for design in ("BPG", "BP"):
        d0 = T[T.design == design]
        print(f"\n==================== {design} ====================")
        arms = ["B1", "MX", "MX1", "MXc", "MP", "MS", "OFFX", "OFFX1", "OFFP", "OFFP1", "MG", "MCO", "E_MRP",
                "MX_light", "MX_ce", "MX_gd", "MX_heavy", "MX_LM", "MX_LX"]
        show("ALL C3 rows (no-donor rows keep B1)", d0, arms)
        X = d0[(d0.n_MX > 0) & ~d0.t_v2drop]
        show("POPX: CLEAN targets with >= 1 other-extractant same-publication donor", X, arms)
        X0 = X[X.n_MS == 0]
        show("POPX0: ... and no same-extractant cell in the publication (pure new-extractant case)", X0, arms)
        X1 = X[X.n_MS > 0]
        show("POPX1: ... and >= 1 same-extractant cell in the publication", X1, arms)
        XL = X[~X.publication.isin(["pub_126bbe7976", "pub_1bd76c48d9"])]
        show("POPX without pub_126bbe7976 and pub_1bd76c48d9 (Skeptic B leave-outs)", XL, arms)
        XG = X[X.cluster_cpg != "g:pub_74428aeab2"]
        show("POPX outside the giant cluster g:pub_74428aeab2", XG, arms)
        show_inf("POPX", X, [("MX", "B1"), ("OFFX", "B1"), ("MX", "OFFX"), ("MX", "MG"), ("MX", "E_MRP"),
                             ("MX", "MCO"), ("MX_LM", "MX_LX"), ("MX_light", "B1"), ("MX_heavy", "B1")])
        show_inf("POPX0", X0, [("MX", "B1"), ("OFFX", "B1"), ("MX", "E_MRP")])
        if design == "BPG":
            # per publication
            rec = []
            for p, g in X.groupby("publication"):
                rec.append({"pub": p, "cells": g.cell_id.nunique(), "rows/seed": len(g) / 5, "n_ext": g.extractant.nunique(),
                            "cluster": g.cluster_cpg.iloc[0][2:14], "B1_mse": float(sqe(g, "B1").mean()),
                            "rel_MX": rel(g, "MX"), "rel_OFFX": rel(g, "OFFX"), "rel_MP": rel(g, "MP"),
                            "rel_MRP": rel(g, "E_MRP"), "rel_MX_light": rel(g, "MX_light"),
                            "rel_MX_heavy": rel(g, "MX_heavy"), "dSSE_MX/seed": float((sqe(g, "B1") - sqe(g, "MX")).sum() / 5)})
            print("\nper publication (POPX, BPG):")
            print(pd.DataFrame(rec).sort_values("dSSE_MX/seed", ascending=False).round(4).to_string(index=False))
            # pair-type decomposition
            for arm in ("MX", "OFFX", "MP"):
                s, tot = pairshare(X, arm)
                print(f"\npair-type share of the {arm} - B1 SSE gain (POPX, BPG; total {tot:.3f} SSE/seed; B1 SSE/seed "
                      f"{sqe(X, 'B1').sum() / 5:.3f}):")
                print((s / tot).round(3).to_dict())
                print(f"   light pairs {s[s.index.isin(LIGHTP)].sum() / tot:.3f}; Ce pairs {s[s.index.isin(CEP)].sum() / tot:.3f}; "
                      f"Gd pairs {s[s.index.isin(GDP)].sum() / tot:.3f}")
            nl = X[~X.pair.isin(LIGHTP)]
            show("POPX without light pairs (La/Ce, Ce/Pr, Pr/Nd)", nl, ["B1", "MX", "OFFX", "MP", "E_MRP"])
            nce = X[~X.pair.isin(CEP)]
            show("POPX without Ce pairs", nce, ["B1", "MX", "OFFX", "MP"])
            # window split (label-free: B1 predicted levels of both metals in [-2, 2])
            inw = (X.lvA.between(-2, 2)) & (X.lvB.between(-2, 2))
            show("POPX inside the level window", X[inw], ["B1", "MX", "OFFX", "MP", "E_MRP", "MX_LM", "MX_LX"])
            show("POPX outside the level window", X[~inw], ["B1", "MX", "OFFX", "MP", "E_MRP", "MX_LM", "MX_LX"])
            print(f"   share of MX gain from out-of-window rows: "
                  f"{((sqe(X[~inw], 'B1') - sqe(X[~inw], 'MX')).sum() / (sqe(X, 'B1') - sqe(X, 'MX')).sum()):.3f}; "
                  f"out-of-window share of rows {(~inw).mean():.3f}")
            # homologue vs other chemotype on rows where both exist
            H = X[(X.n_MXH > 0) & (X.n_MXO > 0)]
            show("rows with BOTH a same-chemotype and an other-chemotype other-extractant donor", H,
                 ["B1", "MX", "MXH", "MXO", "E_MRP"])
            show_inf("homologue split", H, [("MXH", "B1"), ("MXO", "B1"), ("MXH", "MXO")])
            Hh = X[(X.n_MXH > 0)]
            show("rows with a same-chemotype other-extractant donor", Hh, ["B1", "MX", "MXH", "MXO"])
            Ho = X[(X.n_MXO > 0)]
            show("rows with an other-chemotype same-publication donor", Ho, ["B1", "MX", "MXH", "MXO", "E_MRP"])
            show_inf("other-chemotype donor rows", Ho, [("MXO", "B1"), ("MXO", "E_MRP")])
            # reference-series protocol
            R = d0[d0.ref_target & ~d0.t_v2drop]
            show("REFERENCE-SERIES (TODGA in the same publication; non-TODGA targets)", R,
                 ["B1", "MREF_own", "MREF_pool", "E_MREF_oth", "OFFREF", "MX", "MP", "E_MRP", "MG"])
            show_inf("reference series", R, [("MREF_own", "B1"), ("MREF_own", "E_MREF_oth"), ("MREF_own", "MREF_pool"),
                                            ("MX", "MREF_own"), ("OFFREF", "B1")])
            print("   reference-series targets per publication:",
                  R.groupby("publication").agg(cells=("cell_id", "nunique"), n_ref=("n_ref_own", "max"),
                                               rows=("y", "size")).assign(rows=lambda q: q.rows / 5).to_dict("index"))
            for p, g in R.groupby("publication"):
                print(f"   {p}: B1 R2 {r2(g, 'B1'):.3f}  rel SSE own-TODGA {rel(g, 'MREF_own'):+.3f}  other-lab TODGA "
                      f"{rel(g, 'E_MREF_oth'):+.3f}  pooled TODGA {rel(g, 'MREF_pool'):+.3f}  all own other-extractants {rel(g, 'MX'):+.3f}")
            # clusters
            print("\nPOPX by cluster: ", X.groupby("cluster_cpg").apply(
                lambda g: pd.Series({"cells": g.cell_id.nunique(), "rows": len(g) / 5, "rel_MX": rel(g, "MX"),
                                     "rel_OFFX": rel(g, "OFFX")})).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
