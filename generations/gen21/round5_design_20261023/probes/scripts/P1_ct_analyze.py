#!/usr/bin/env python
"""Probe P1 (idea M01, estimand E-CT), analysis step on <out>/P1_rows.parquet from P1_ct_build.py.

EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds.  Prints every table of the P1 report and writes
<out>/P1_summary.json.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/probes/scripts/P1_ct_analyze.py --out <dir>

R2 convention = round 4's: pooled strict-adjacent R2 inside each split seed, then the seed mean (rows identical
across arms inside every comparison).  Arms stored as expected squared errors (RAND1, SUB<m>) enter as SSE.
Inference unit = publication x chemotype; per-unit contribution to the seed-mean R2 contrast
c_u = mean_s (SSE_ref,u,s - SSE_cand,u,s) / SST_s, so sum_u c_u is the pooled contrast exactly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4 import metrics as M4  # noqa: E402
from gen20k import data as D  # noqa: E402

K_SH = 1.78
SUMMARY: dict = {}
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)


def err2(d: pd.DataFrame, arm: str) -> np.ndarray:
    """Squared error per row (arms 'E_*' are already expected squared errors)."""
    if "_E_" in arm:
        return d[arm].to_numpy(float)
    return (d.y.to_numpy(float) - d[arm].to_numpy(float)) ** 2


def r2_seeds(d: pd.DataFrame, arm: str) -> np.ndarray:
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1.0 - err2(g, arm).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(out)


def r2(d, arm) -> float:
    return float(r2_seeds(d, arm).mean())


def mae(d, arm) -> float:
    if "_E_" in arm:
        return float("nan")
    return float((d.y - d[arm]).abs().mean())


def relsse(d, cand, ref) -> float:
    """Seed-mean relative SSE reduction of cand against ref (positive = cand better)."""
    out = []
    for _, g in d.groupby("split_seed"):
        out.append(1.0 - err2(g, cand).sum() / err2(g, ref).sum())
    return float(np.mean(out))


def table(d: pd.DataFrame, arms, ref="B1_3") -> pd.DataFrame:
    rows = []
    for a in arms:
        rs = r2_seeds(d, a)
        rows.append({"arm": a, "R2": rs.mean(), "R2_min_seed": rs.min(), "R2_max_seed": rs.max(), "MAE": mae(d, a),
                     "relSSE_vs_" + ref: relsse(d, a, ref)})
    return pd.DataFrame(rows).set_index("arm")


def describe(d: pd.DataFrame) -> str:
    return (f"cells {d.cell_id.nunique()}, publications {d.publication.nunique()}, pub x chemotype units "
            f"{d[['publication', 'chemotype']].drop_duplicates().shape[0]}, rows/seed {len(d) / d.split_seed.nunique():.1f}")


# ------------------------------------------------------------------------------------------------------------
# inference
# ------------------------------------------------------------------------------------------------------------
def unit_contrib(d: pd.DataFrame, cand: str, ref: str, unit=("publication", "chemotype")) -> pd.DataFrame:
    sst = {s: float(((g.y - g.y.mean()) ** 2).sum()) for s, g in d.groupby("split_seed")}
    n_seed = d.split_seed.nunique()
    e = d[list(unit) + ["split_seed"]].copy()
    e["ec"] = err2(d, cand)
    e["er"] = err2(d, ref)
    e["sst"] = e.split_seed.map(sst)
    e["c"] = (e.er - e.ec) / e.sst / n_seed
    g = e.groupby(list(unit)).agg(c=("c", "sum"), sse_c=("ec", "sum"), sse_r=("er", "sum"), rows=("c", "size"))
    g["rho"] = 1.0 - g.sse_c / g.sse_r
    g["logratio"] = np.log(g.sse_r / g.sse_c)
    return g.reset_index()


def signflip_p(c: np.ndarray, groups=None, n_mc=200000, seed=20261023) -> float:
    """One-sided p for sum(c) > 0 under sign symmetry; exact enumeration when 2^G <= 2^20."""
    c = np.asarray(c, float)
    if groups is None:
        groups = np.arange(len(c))
    labs, inv = np.unique(groups, return_inverse=True)
    gs = np.bincount(inv, weights=c)
    G = len(gs)
    obs = gs.sum()
    if G <= 20:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
    else:
        signs = np.random.default_rng(seed).choice((1.0, -1.0), size=(n_mc, G))
    null = signs @ gs
    return float((null >= obs - 1e-15).mean())


def im_t(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    G = len(x)
    if G < 2 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = x.mean() / (x.std(ddof=1) / np.sqrt(G))
    return float(t), float(SS.t.sf(t, G - 1))


def lopo(d: pd.DataFrame, cand: str, ref: str) -> dict:
    point = r2(d, cand) - r2(d, ref)
    pubs = sorted(d.publication.unique())
    jk = np.array([r2(d[d.publication != p], cand) - r2(d[d.publication != p], ref) for p in pubs])
    P = len(jk)
    se = float(np.sqrt((P - 1) / P * ((jk - jk.mean()) ** 2).sum()))
    q = float(SS.t.ppf(0.975, P - 1))
    i_min = int(np.argmin(jk))
    return {"point": point, "se": se, "lo": point - q * se, "hi": point + q * se, "P": P, "min": float(jk.min()),
            "min_pub": pubs[i_min], "max": float(jk.max())}


def infer(d: pd.DataFrame, cand: str, ref: str, label: str) -> dict:
    u = unit_contrib(d, cand, ref)
    c = u.c.to_numpy()
    tot = c.sum()
    srt = np.sort(c)[::-1]
    pub_u = unit_contrib(d, cand, ref, unit=("publication",))
    res = {
        "label": label, "cand": cand, "ref": ref,
        "dR2": r2(d, cand) - r2(d, ref), "R2_cand": r2(d, cand), "R2_ref": r2(d, ref),
        "dR2_seeds": (r2_seeds(d, cand) - r2_seeds(d, ref)).round(4).tolist(),
        "relSSE": relsse(d, cand, ref),
        "G_units": int(len(u)), "units_pos": int((c > 1e-6).sum()), "units_neg": int((c < -1e-6).sum()),
        "units_zero": int((np.abs(c) <= 1e-6).sum()),
        "p_signflip_unit": signflip_p(c),
        "p_signflip_chemotype_blocks": signflip_p(c, u.chemotype.to_numpy()),
        "G_chemotypes": int(u.chemotype.nunique()),
        "IMt_rho": im_t(u.rho.to_numpy()), "IMt_logratio": im_t(u.logratio.to_numpy()),
        "top3_share": float(srt[:3].sum() / tot) if tot > 0 else float("nan"),
        "eqw_pub_relSSE": float(pub_u.rho.mean()), "pubs_pos": int((pub_u.c > 0).sum()), "P": int(len(pub_u)),
        "lopo": lopo(d, cand, ref),
        "units": u.sort_values("c", ascending=False).round(5).to_dict("records"),
    }
    return res


def print_infer(r: dict) -> None:
    lo = r["lopo"]
    print(f"  [{r['label']}] {r['cand']} - {r['ref']}: dR2 {r['dR2']:+.4f} (seeds {r['dR2_seeds']}), relSSE {r['relSSE']:+.3%}; "
          f"units {r['units_pos']}+/{r['units_neg']}-/{r['units_zero']}=0 (|c|<=1e-6) of {r['G_units']}; sign-flip p(unit) {r['p_signflip_unit']:.4f}, "
          f"p(chemotype blocks, {r['G_chemotypes']}) {r['p_signflip_chemotype_blocks']:.4f}; IM-t rho t {r['IMt_rho'][0]:.2f} "
          f"p {r['IMt_rho'][1]:.4f}; IM-t log t {r['IMt_logratio'][0]:.2f} p {r['IMt_logratio'][1]:.4f}; top-3 share "
          f"{r['top3_share']:.2f}; eq-w pub relSSE {r['eqw_pub_relSSE']:+.3f} ({r['pubs_pos']}/{r['P']} pubs +); "
          f"LOPO jackknife se {lo['se']:.4f} 95% [{lo['lo']:+.4f}, {lo['hi']:+.4f}] (P={lo['P']}), LOPO min {lo['min']:+.4f} "
          f"(without {lo['min_pub']}), max {lo['max']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    T = pd.read_parquet(out / "P1_rows.parquet")
    bench = D.load_bench()
    T = M4.add_clusters(T, bench.frame)

    for design in ("BPG", "BP"):
        t = T[T.design == design]
        print("\n" + "=" * 110 + f"\nDESIGN {design}\n" + "=" * 110)
        S = {}
        S["V1_all"] = t[t.n_v1 >= 1]
        S["CL_ge1"] = t[t.n_clean >= 1]
        S["CL_ge1_noC5"] = t[(t.n_clean >= 1) & ~t.c5]
        S["PRIMARY"] = t[(t.n_clean >= 2) & ~t.c5]
        S["CL_ge2_withC5"] = t[t.n_clean >= 2]
        S["CL_eq1"] = t[t.n_clean == 1]
        S["CL_eq1_noC5"] = t[(t.n_clean == 1) & ~t.c5]
        for k, d in S.items():
            print(f"{k}: {describe(d)}")
        P = S["PRIMARY"]
        print("\nclusters (cluster_cpg) in PRIMARY:", P.groupby("cluster_cpg").size().div(5).round(1).to_dict(),
              " Kish", round(M4.kish(P[P.split_seed == P.split_seed.iloc[0]], "cluster_cpg"), 2))
        print("clusters in CL_ge2_withC5:", S["CL_ge2_withC5"].groupby("cluster_cpg").size().div(5).round(1).to_dict())

        # ---- A. main comparator table, identical rows --------------------------------------------------------
        arms = ["B1_3", "B0_3", "CL_SH_3", "CL_SIB_3", "CL_SHK2_3", "CL_OFF", "CL_OFF_ALL", "CL_NEAR_3", "CL_NEARSH_3",
                "CL_E_RAND1_raw", "CL_E_RAND1_sh", "CL_COPY", "CL_SH0", "G14Z", "CL_E_RAND1_raw_0"]
        for nm in ("PRIMARY", "CL_ge2_withC5", "CL_ge1", "CL_ge1_noC5", "CL_eq1", "CL_eq1_noC5"):
            print(f"\n[A] comparators on identical rows, {nm} ({describe(S[nm])})")
            print(table(S[nm], arms).round(4).to_string())
        # placebo blends on rows where the placebo pool exists
        for nm in ("PRIMARY", "CL_ge2_withC5"):
            d0 = S[nm]
            for pb in ("CL_PB_CHEM", "CL_PB_XEXT", "CL_PB_PUB"):
                d = d0[d0[pb].notna()]
                print(f"\n[A-placebo] {pb} on its rows inside {nm}: {describe(d)}; coverage {len(d) / len(d0):.3f} of rows")
                if len(d):
                    print(table(d, ["B1_3", "CL_SH_3", "CL_OFF", pb, pb + "_raw"]).round(4).to_string())
        # OFF K sensitivity (OFF recomputed from its stored offset sum)
        for nm in ("PRIMARY", "CL_ge2_withC5"):
            d = S[nm].copy()
            ssum = (d.CL_OFF - d.B1_par) * (d.CL_OFF_np + K_SH)
            line = []
            for K in (0.25, 0.5, 1.0, 1.78, 3.0, 5.0):
                d[f"OFFK{K}"] = d.B1_par + np.where(d.CL_OFF_np > 0, ssum / (d.CL_OFF_np + K), 0.0)
                line.append(f"K={K}: {r2(d, f'OFFK{K}'):.4f}")
            print(f"\n[A-OFF-K] {nm}: OFF R2 by K (K pinned 1.78 is primary; others = hindsight sensitivity): " + "; ".join(line))
            print(f"   rows with no OFF donor for the pair (OFF = B1): {(d.CL_OFF_np == 0).mean():.3f}; SH R2 {r2(d, 'CL_SH_3'):.4f}")

        # ---- B. read ladder ----------------------------------------------------------------------------------
        lad = [("0 reads", ["CL_COPY", "CL_SH0", "G14Z"]), ("2 reads", ["B1_1", "B0_1", "CL_SIB_1", "CL_SH_1"]),
               ("3 reads", ["B1_2", "B0_2", "CL_SIB_2", "CL_SH_2"]), ("4 reads", ["B1_3", "B0_3", "CL_SIB_3", "CL_SH_3", "CL_OFF"])]
        for nm in ("PRIMARY", "CL_ge2_withC5", "CL_ge1", "CL_eq1"):
            d = S[nm]
            print(f"\n[B] read ladder on identical (4-read scored) rows, {nm}: " + describe(d))
            for lab, arms_ in lad:
                print("   " + lab + ": " + "; ".join(f"{x} {r2(d, x):.4f}" for x in arms_))
        d = S["V1_all"]
        print("\n[B-V1] E6 reproduction (V1 population, every covering sibling): " + describe(d))
        print("   " + "; ".join(f"{x} {r2(d, x):.4f}" for x in ["V1_COPY", "V1_SIB_1", "V1_SIB_2", "V1_SIB_3", "V1_SHK2_3",
                                                              "V1_SH_3", "B1_3", "B1_1", "B1_2"]))

        # ---- C. within-publication subsampling ---------------------------------------------------------------
        for nm, d0 in (("n>=5 no C5", t[(t.n_clean >= 5) & ~t.c5]), ("n>=5 with C5", t[t.n_clean >= 5])):
            print(f"\n[C] sibling subsampling, CLEAN {nm}: {describe(d0)}")
            rr = []
            for m in (1, 2, 3, 5):
                rr.append({"m": m, "raw": r2(d0, f"CL_E_SUB{m}_raw"), "SH": r2(d0, f"CL_E_SUB{m}_sh")})
            rr.append({"m": "all", "raw": r2(d0, "CL_SIB_3"), "SH": r2(d0, "CL_SH_3")})
            print(pd.DataFrame(rr).round(4).to_string(index=False), f"\n   B1 {r2(d0, 'B1_3'):.4f}; OFF {r2(d0, 'CL_OFF'):.4f}; "
                  f"n_clean distribution {d0.drop_duplicates('cell_id').n_clean.value_counts().sort_index().to_dict()}")
            print("   per-publication cells:", d0.drop_duplicates("cell_id").groupby("publication").size().to_dict())

        # ---- D. n_sib strata (descriptive) -------------------------------------------------------------------
        print("\n[D] between-cell n_sib strata (CLEAN, no C5; descriptive only)")
        tt = t[(t.n_clean >= 1) & ~t.c5].copy()
        tt["nbin"] = pd.cut(tt.n_clean, [0, 1, 2, 4, 7, 100], labels=["1", "2", "3-4", "5-7", ">=8"])
        rr = []
        for b, d in tt.groupby("nbin", observed=True):
            rr.append({"n_sib": b, "cells": d.cell_id.nunique(), "pubs": d.publication.nunique(), "rows/seed": len(d) / 5,
                       "B1": r2(d, "B1_3"), "SH": r2(d, "CL_SH_3"), "SIB": r2(d, "CL_SIB_3"), "OFF": r2(d, "CL_OFF"),
                       "relSSE_SH_vs_B1": relsse(d, "CL_SH_3", "B1_3"), "relSSE_SH_vs_OFF": relsse(d, "CL_SH_3", "CL_OFF")})
        print(pd.DataFrame(rr).round(4).to_string(index=False))
        for nm in ("CL_eq1", "CL_eq1_noC5"):
            d = S[nm]
            print(f"   harm rule {nm}: relSSE(SH vs B1) {relsse(d, 'CL_SH_3', 'B1_3'):+.4f} (harm if < -0.05); "
                  f"SH {r2(d, 'CL_SH_3'):.4f} B1 {r2(d, 'B1_3'):.4f} SIB {r2(d, 'CL_SIB_3'):.4f}")

        # ---- E. pair-type decomposition ----------------------------------------------------------------------
        for nm in ("PRIMARY", "CL_ge2_withC5"):
            d = S[nm]
            print(f"\n[E] pair-type decomposition, {nm}")
            e = d[["pair", "split_seed"]].copy()
            for x in ("B1_3", "CL_SH_3", "CL_OFF", "CL_SIB_3"):
                e[x] = err2(d, x)
            g = e.groupby("pair", sort=False)[["B1_3", "CL_SH_3", "CL_OFF", "CL_SIB_3"]].sum() / 5
            g["rows/seed"] = e.groupby("pair", sort=False).size() / 5
            g["gain_SH_vs_B1"] = g.B1_3 - g.CL_SH_3
            g["share_gain_SH_vs_B1"] = g.gain_SH_vs_B1 / g.gain_SH_vs_B1.sum()
            g["gain_SH_vs_OFF"] = g.CL_OFF - g.CL_SH_3
            g["share_gain_SH_vs_OFF"] = g.gain_SH_vs_OFF / g.gain_SH_vs_OFF.sum()
            order = ["La/Ce", "Ce/Pr", "Pr/Nd", "Nd/Sm", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm",
                     "Tm/Yb", "Yb/Lu"]
            g = g.reindex([o for o in order if o in g.index] + [o for o in g.index if o not in order])
            print(g.round(4).to_string())
            ce = d.pair.isin(["La/Ce", "Ce/Pr"])
            gd = d.pair.isin(["Eu/Gd", "Gd/Tb"])
            for lab, dd in (("all", d), ("excl. Ce pairs", d[~ce]), ("excl. Gd pairs", d[~gd]), ("excl. Ce and Gd", d[~ce & ~gd])):
                print(f"   {lab}: rows/seed {len(dd) / 5:.1f}; " + "; ".join(
                    f"{x} {r2(dd, x):.4f}" for x in ("B1_3", "CL_SH_3", "CL_OFF", "CL_SIB_3")) +
                    f"; SH-B1 {r2(dd, 'CL_SH_3') - r2(dd, 'B1_3'):+.4f}; SH-OFF {r2(dd, 'CL_SH_3') - r2(dd, 'CL_OFF'):+.4f}")
            print(f"   artefact flag: Ce-pair share of SH-B1 SSE gain {g.loc[['La/Ce', 'Ce/Pr'], 'gain_SH_vs_B1'].sum() / g.gain_SH_vs_B1.sum():.3f}"
                  f" (flag if >= 0.50); Gd-pair share {g.loc[['Eu/Gd', 'Gd/Tb'], 'gain_SH_vs_B1'].sum() / g.gain_SH_vs_B1.sum():.3f}")

        # ---- F. condition-change type ------------------------------------------------------------------------
        for nm in ("PRIMARY", "CL_ge1_noC5"):
            d0 = S[nm]
            print(f"\n[F] by condition-change type (union of the CLEAN siblings' differing cond__ families), {nm}")
            rr = []
            for c, d in d0.groupby("cls"):
                rr.append({"class": c, "cells": d.cell_id.nunique(), "pubs": d.publication.nunique(), "rows/seed": len(d) / 5,
                           "B1": r2(d, "B1_3"), "SH": r2(d, "CL_SH_3"), "SIB": r2(d, "CL_SIB_3"), "OFF": r2(d, "CL_OFF"),
                           "COPY": r2(d, "CL_COPY"),
                           "SSEgain_SH_vs_B1/seed": float((err2(d, "B1_3") - err2(d, "CL_SH_3")).sum() / 5),
                           "SSEgain_SH_vs_OFF/seed": float((err2(d, "CL_OFF") - err2(d, "CL_SH_3")).sum() / 5)})
            ft = pd.DataFrame(rr)
            ft["share_gain_vs_B1"] = ft["SSEgain_SH_vs_B1/seed"] / ft["SSEgain_SH_vs_B1/seed"].sum()
            print(ft.round(4).to_string(index=False))
            print("   publications per class:", {c: sorted(d.publication.unique().tolist()) for c, d in d0.groupby("cls")})

        # ---- G. inference -----------------------------------------------------------------------------------
        print("\n[G] inference, unit = publication x chemotype")
        res = []
        for nm in ("PRIMARY", "CL_ge2_withC5", "CL_ge1_noC5", "CL_ge1"):
            d = S[nm]
            for cand, ref in (("CL_SH_3", "B1_3"), ("CL_SH_3", "CL_OFF"), ("CL_SH_3", "CL_OFF_ALL"), ("CL_OFF", "B1_3"),
                              ("CL_SIB_3", "B1_3"), ("CL_COPY", "B1_3"), ("CL_SH_2", "B1_3"), ("CL_SH_1", "B1_3"),
                              ("CL_SH_3", "CL_NEAR_3"), ("CL_SH_3", "CL_SIB_3")):
                r = infer(d, cand, ref, nm)
                res.append(r)
                print_infer(r)
            for pb in ("CL_PB_CHEM", "CL_PB_PUB"):
                dd = d[d[pb].notna()]
                if dd.publication.nunique() >= 3:
                    r = infer(dd, "CL_SH_3", pb, nm + f" rows with {pb}")
                    res.append(r)
                    print_infer(r)
        # single-cell influence of c950dc2982e96b83 (a C5-BTBP cell) when C5 is included
        d = S["CL_ge2_withC5"]
        dd = d[d.cell_id != "c950dc2982e96b83"]
        print(f"\n   c950dc in CL_ge2_withC5: {'c950dc2982e96b83' in set(d.cell_id)}; SH-B1 with {r2(d, 'CL_SH_3') - r2(d, 'B1_3'):+.4f}, "
              f"without {r2(dd, 'CL_SH_3') - r2(dd, 'B1_3'):+.4f}; SH-OFF with {r2(d, 'CL_SH_3') - r2(d, 'CL_OFF'):+.4f}, "
              f"without {r2(dd, 'CL_SH_3') - r2(dd, 'CL_OFF'):+.4f}")
        print("\n   PRIMARY units for SH-OFF (c = contribution to seed-mean dR2):")
        r = [x for x in res if x["label"] == "PRIMARY" and x["cand"] == "CL_SH_3" and x["ref"] == "CL_OFF"][0]
        print(pd.DataFrame(r["units"]).to_string(index=False))
        print("\n   PRIMARY units for SH-B1:")
        r = [x for x in res if x["label"] == "PRIMARY" and x["cand"] == "CL_SH_3" and x["ref"] == "B1_3"][0]
        print(pd.DataFrame(r["units"]).to_string(index=False))
        SUMMARY[design] = [{k: v for k, v in x.items() if k != "units"} for x in res]
    (out / "P1_summary.json").write_text(json.dumps(SUMMARY, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
