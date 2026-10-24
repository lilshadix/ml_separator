#!/usr/bin/env python
"""SKEPTIC A for idea confirm-2 (C-GD: heavy-selective non-DGA Gd chord deeper than B1's beta).

EXPLORATORY.  Every number comes from the 229 C3 cells (V1 bench) that rounds 1-4 and probes P1-P5 have already
seen.  Independent re-implementation (does not import the confirm_* scripts) of:
  A  the stratum table (DGA / non-DGA heavy-selective / non-DGA light-selective; pubchem units; winsorised)
  B  direction-rule edge cases (cells with < 5 metals get slope NaN -> 'light-selective' in the original)
  C  robustness of the HS mean: V2 matrix, unit = publication / copy_group / cluster_cpg, leave-one-pub/cluster-out,
     drop-k most extreme units, near-copy / bit-identical series, C5-BTBP and DMDPhPDA
  D  alternative explanation: steepness (Eu->Tb step, |slope|) instead of direction
  E  post hoc split: label-permutation of HS/LS within non-DGA
  F  does a stratum amplitude transfer?  leave-one-cluster-out / leave-one-publication-out prediction of held-out
     Gd chords (= B1's Gd error) with pooled beta vs stratum means
  G  power: own bootstrap (unit and cluster level), analytic non-central t at deflated worlds, small-unit noise
  H  reads-only direction: slope sign from the DOPT_Q k=3 read sets vs the full-series slope sign
Writes only ideas2/scripts/confirm2_skA_out/.
   .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/confirm2_gd_skA.py
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parent / "confirm2_skA_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

LN = list(D.LANTHANIDES)
iEu, iGd, iTb = LN.index("Eu"), LN.index("Gd"), LN.index("Tb")
XR = np.asarray(interp.X_R, float)
TG = (XR[iGd] - XR[iEu]) / (XR[iTb] - XR[iEu])
BETA = -0.0923
CLIP = (-0.40, 0.20)
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
R = 4000
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)


def chord(Y):
    return Y[:, iGd] - ((1 - TG) * Y[:, iEu] + TG * Y[:, iTb])


def slope(Y, min_n=5):
    s = np.full(len(Y), np.nan)
    n = np.isfinite(Y).sum(1)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= min_n:
            s[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    return s, n


def classify(u):
    return np.where(u.dga > 0.5, "DGA", np.where(u.hs > 0.5, "HS", "LS"))


def units_of(d, unit="pubchem", col="c"):
    u = d.groupby(unit).agg(c=(col, "mean"), hs=("hs", "mean"), dga=("dga", "mean"), n=(col, "size"),
                            slope=("slope", "mean"), step=("step", "mean"), pub=("pub", "first"),
                            npub=("pub", "nunique"), cl=("cluster_cpg", "first"), cg=("copy_group", "first"),
                            name=("name", lambda s: "/".join(sorted(set(map(str, s))))[:70]))
    u["cw"] = np.clip(u.c, *CLIP)
    u["cls"] = classify(u)
    return u


def strata(u):
    return u.groupby("cls").agg(units=("cw", "size"), cells=("n", "sum"), mean_w=("cw", "mean"),
                                sd_w=("cw", "std"), median=("c", "median"))


def tpower(X, beta, alpha=0.025):
    G = X.shape[1]
    t = (X.mean(1) - beta) / (X.std(1, ddof=1) / np.sqrt(G))
    return float((t < SS.t.ppf(alpha, G - 1)).mean())


def nct_power(delta, sd, G, alpha=0.025):
    """one-sample one-sided t power, true mean - beta = delta (<0), unit sd = sd."""
    ncp = -delta / (sd / np.sqrt(G))
    return float(SS.nct.sf(SS.t.ppf(1 - alpha, G - 1), G - 1, ncp))


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    b = D.load_bench()
    f = b.frame.reset_index(drop=True).copy()
    Y1 = np.asarray(b.Y, float)
    Y2 = np.asarray(V2.matrix(b), float)
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(ref.cell_id)
    cl = M4.add_clusters(ref[["cell_id", "publication", "chemotype"]].drop_duplicates("cell_id"), None)
    f = f.merge(cl[["cell_id", "cluster_cpg", "copy_group"]], on="cell_id", how="left")
    sl1, nmet = slope(Y1)
    sl_any, _ = slope(Y1, min_n=2)
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    base = pd.DataFrame({
        "cell_id": f.cell_id, "pub": f.publication_id.astype(str), "chem": f.chemotype.astype(str),
        "name": f.extractant_name.astype(str), "family": f.chem_family.astype(str),
        "c": chord(Y1), "c_v2": chord(Y2), "slope": sl1, "slope_any": sl_any, "nmet": nmet,
        "step": Y1[:, iTb] - Y1[:, iEu],
        "dga": f.extractant.astype(str).map(DN.is_dga), "c3": f.cell_id.isin(c3),
        "v2drop": f.cell_id.isin(v2drop), "cluster_cpg": f.cluster_cpg, "copy_group": f.copy_group})
    base["hs"] = base.slope > 0                      # as in the original: NaN slope -> False (= light-selective)
    base["pubchem"] = base.pub + "|" + base.chem
    d = base[base.c3 & np.isfinite(base.c)].copy()
    print(f"C3 cells with a Eu-Gd-Tb chord: {len(d)}; pubs {d.pub.nunique()}; pubchem {d.pubchem.nunique()}")

    # ---------------- A reproduce ----------------------------------------------------------------------------
    u = units_of(d)
    st = strata(u)
    print("\n[A] strata (pubchem, V1, winsorised)\n", st.round(4).to_string())
    hsu = u[u.cls == "HS"].sort_values("c")
    print("\n[A] heavy-selective non-DGA units\n",
          hsu[["n", "c", "cw", "slope", "step", "npub", "cl", "cg", "name"]].round(3).to_string())
    lsu = u[u.cls == "LS"].sort_values("c")
    print("\n[A] light-selective non-DGA units\n",
          lsu[["n", "c", "cw", "slope", "step", "cl", "cg", "name"]].round(3).to_string())
    res["A_strata"] = st.round(4).reset_index().to_dict("records")
    hsu.to_csv(OUT / "hs_units.csv")
    u.to_csv(OUT / "all_units.csv")

    # ---------------- B direction-rule edge cases ------------------------------------------------------------
    nan_sl = d[~np.isfinite(d.slope)]
    print(f"\n[B] C3 chord cells with slope NaN (< 5 metals): {len(nan_sl)}")
    if len(nan_sl):
        print(nan_sl[["cell_id", "pub", "nmet", "c", "slope_any", "dga", "name"]].round(3).to_string())
    d_b = d.copy()
    d_b["hs"] = d_b.slope_any > 0
    ub = units_of(d_b)
    print("[B] strata with slope from >= 2 metals\n", strata(ub).round(4).to_string())
    res["B_nan_slope_cells"] = int(len(nan_sl))
    # units whose cells are mixed in direction
    mixed = u[(u.hs > 0) & (u.hs < 1) & (u.dga <= 0.5)]
    print("[B] non-DGA units with mixed cell directions\n", mixed[["n", "hs", "c", "cls", "name"]].round(3).to_string())

    # ---------------- C robustness of the HS mean -------------------------------------------------------------
    rob = []

    def hsmean(dd, unit="pubchem", col="c"):
        uu = units_of(dd, unit, col)
        h = uu[uu.cls == "HS"]
        return len(h), float(h.cw.mean()), float(h.cw.std(ddof=1)) if len(h) > 1 else np.nan

    for lab, dd, unit, col in (
            ("V1 pubchem", d, "pubchem", "c"),
            ("V2 matrix pubchem", d[np.isfinite(d.c_v2)], "pubchem", "c_v2"),
            ("V1 drop V2-flagged cells", d[~d.v2drop], "pubchem", "c"),
            ("V1 unit=publication", d, "pub", "c"),
            ("V1 unit=copy_group", d, "copy_group", "c"),
            ("V1 unit=cluster_cpg", d, "cluster_cpg", "c"),
            ("no C5", d[d.pub != C5], "pubchem", "c"),
            ("no C5, no DMD", d[~d.pub.isin([C5, DMD])], "pubchem", "c")):
        n, m, s = hsmean(dd, unit, col)
        rob.append({"variant": lab, "HS_units": n, "HS_mean_w": m, "HS_sd_w": s})
    # leave-one-publication-out / cluster-out among HS units
    hs_pubs = sorted(set(hsu.pub))
    lopo = []
    for p in hs_pubs:
        n, m, s = hsmean(d[d.pub != p])
        lopo.append((p, n, m))
    lopo = pd.DataFrame(lopo, columns=["dropped_pub", "HS_units", "HS_mean_w"])
    hs_cl = sorted(set(hsu.cl))
    loco = []
    for c in hs_cl:
        n, m, s = hsmean(d[d.cluster_cpg != c])
        loco.append((c, n, m))
    loco = pd.DataFrame(loco, columns=["dropped_cluster", "HS_units", "HS_mean_w"])
    rob.append({"variant": "LOPO range (min..max)", "HS_units": int(lopo.HS_units.min()),
                "HS_mean_w": float(lopo.HS_mean_w.max()), "HS_sd_w": float(lopo.HS_mean_w.min())})
    rob.append({"variant": "LOCO range (max=least negative)", "HS_units": int(loco.HS_units.min()),
                "HS_mean_w": float(loco.HS_mean_w.max()), "HS_sd_w": float(loco.HS_mean_w.min())})
    hv = np.sort(hsu.cw.to_numpy())
    for k in (1, 2, 3, 4):
        rob.append({"variant": f"drop {k} most negative HS units", "HS_units": len(hv) - k,
                    "HS_mean_w": float(hv[k:].mean()), "HS_sd_w": float(hv[k:].std(ddof=1))})
    rob = pd.DataFrame(rob)
    print("\n[C] robustness of the HS stratum mean\n", rob.round(4).to_string())
    print("[C] LOPO\n", lopo.round(4).to_string())
    print("[C] LOCO\n", loco.round(4).to_string())
    print("[C] HS units per cluster_cpg / copy_group:\n", hsu.groupby("cl").size().to_string(), "\n",
          hsu.groupby("cg").size().to_string())
    rob.to_csv(OUT / "hs_robustness.csv", index=False)
    res["C_robustness"] = rob.round(4).to_dict("records")

    # near-copy / bit-identical series among C3 cells (cross-publication or cross-unit)
    Yc = Y1[base.c3.to_numpy()]
    ids = base.cell_id[base.c3].to_numpy()
    pubs = base.pub[base.c3].to_numpy()
    pc = base.pubchem[base.c3].to_numpy()
    nc = []
    for i in range(len(Yc)):
        for j in range(i + 1, len(Yc)):
            m = np.isfinite(Yc[i]) & np.isfinite(Yc[j])
            if m.sum() < 4:
                continue
            dd_ = Yc[i, m] - Yc[j, m]
            sdd = float(np.std(dd_, ddof=1))
            if sdd < 0.03:
                nc.append({"a": ids[i], "b": ids[j], "pub_a": pubs[i], "pub_b": pubs[j], "unit_a": pc[i], "unit_b": pc[j],
                           "shared": int(m.sum()), "sd_shift": sdd, "mean_shift": float(dd_.mean()),
                           "identical": bool(np.allclose(dd_, 0, atol=1e-9)),
                           "gd_shared": bool(m[iEu] and m[iGd] and m[iTb])})
    nc = pd.DataFrame(nc)
    hs_units_set = set(hsu.index)
    if len(nc):
        nc["a_in_HS"] = nc.unit_a.isin(hs_units_set)
        nc["b_in_HS"] = nc.unit_b.isin(hs_units_set)
        nc["cross_unit"] = nc.unit_a != nc.unit_b
        x = nc[(nc.a_in_HS | nc.b_in_HS)]
        print(f"\n[C] near-copy pairs (sd of shift < 0.03, >= 4 shared) touching HS units: {len(x)} "
              f"(cross-unit {int(x.cross_unit.sum())}; identical {int(x.identical.sum())})")
        print(x[x.cross_unit].round(4).to_string())
        nc.to_csv(OUT / "nearcopy_pairs_c3.csv", index=False)
        res["C_nearcopy_pairs_touching_HS"] = int(len(x))
        res["C_nearcopy_cross_unit_touching_HS"] = int(x.cross_unit.sum())
    # chord of the near-copies: within-unit duplicates inflate n only; cross-unit duplicates inflate G.

    # ---------------- D steepness vs direction ---------------------------------------------------------------
    nd = u[u.cls != "DGA"].copy()
    nd["is_hs"] = (nd.cls == "HS").astype(float)
    nd["abs_slope"] = nd.slope.abs()
    nd["abs_step"] = nd.step.abs()
    print("\n[D] non-DGA units: chord vs steepness")
    out_d = {}
    for xcol in ("slope", "abs_slope", "step", "abs_step"):
        ok = np.isfinite(nd[xcol])
        rho, p = SS.spearmanr(nd.loc[ok, xcol], nd.loc[ok, "cw"])
        out_d[f"spearman_cw_{xcol}"] = (float(rho), float(p))
        print(f"   Spearman(cw, {xcol}) = {rho:+.3f} (p {p:.3f}, n {ok.sum()})")
    import numpy.linalg as LA
    for xcol in ("abs_slope", "abs_step"):
        ok = np.isfinite(nd[xcol])
        X = np.column_stack([np.ones(ok.sum()), nd.loc[ok, "is_hs"], nd.loc[ok, xcol]])
        yv = nd.loc[ok, "cw"].to_numpy()
        beta_, *_ = LA.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta_
        s2 = resid @ resid / (len(yv) - 3)
        se = np.sqrt(np.diag(s2 * LA.inv(X.T @ X)))
        print(f"   cw ~ 1 + HS + {xcol}: HS coef {beta_[1]:+.3f} (se {se[1]:.3f}), {xcol} coef {beta_[2]:+.3f} (se {se[2]:.3f})")
        out_d[f"ols_HS_coef_adj_{xcol}"] = (float(beta_[1]), float(se[1]))
        out_d[f"ols_{xcol}_coef"] = (float(beta_[2]), float(se[2]))
    # relative chord: chord / |Eu->Tb step| would explode for flat series; report median by class
    nd["slope_q"] = pd.qcut(nd.abs_slope, 3, labels=["flat", "mid", "steep"])
    print(nd.groupby(["slope_q", "cls"], observed=True).cw.agg(["size", "mean"]).round(3).to_string())
    # DGA is also mostly heavy-selective: report DGA direction
    dg = u[u.cls == "DGA"]
    print(f"   DGA units heavy-selective share: {float((dg.hs > 0.5).mean()):.2f}; DGA mean |slope| {dg.slope.abs().mean():.3f}; "
          f"HS non-DGA mean |slope| {hsu.slope.abs().mean():.3f}; LS mean |slope| {lsu.slope.abs().mean():.3f}")
    res["D_steepness"] = out_d

    # ---------------- E post hoc split: permutation ----------------------------------------------------------
    rng = np.random.default_rng(20261023 + zlib.crc32(b"skA|perm"))
    lab_ = nd.is_hs.to_numpy().astype(bool)
    yv = nd.cw.to_numpy()
    obs = yv[lab_].mean() - yv[~lab_].mean()
    perm = np.array([(lambda L: yv[L].mean() - yv[~L].mean())(rng.permutation(lab_)) for _ in range(20000)])
    p_perm = float((perm <= obs + 1e-12).mean())
    welch = SS.ttest_ind(yv[lab_], yv[~lab_], equal_var=False, alternative="less")
    # chance that a random 12-of-25 non-DGA subset has mean <= observed HS mean
    sub = np.array([yv[rng.choice(len(yv), lab_.sum(), replace=False)].mean() for _ in range(20000)])
    p_sub = float((sub <= yv[lab_].mean() + 1e-12).mean())
    print(f"\n[E] HS - LS (non-DGA units): {obs:+.3f}; permutation p {p_perm:.4f}; Welch p {welch.pvalue:.4f}; "
          f"P(random {lab_.sum()}-subset mean <= HS mean) {p_sub:.4f}")
    # same contrast at publication and cluster level (each pub/cluster one value within class)
    for unit in ("pub", "cluster_cpg"):
        uu = units_of(d, unit)
        h = uu[uu.cls == "HS"].cw.to_numpy()
        lsv = uu[uu.cls == "LS"].cw.to_numpy()
        if len(h) > 1 and len(lsv) > 1:
            w = SS.ttest_ind(h, lsv, equal_var=False, alternative="less")
            print(f"   unit={unit}: HS n={len(h)} mean {h.mean():+.3f}; LS n={len(lsv)} mean {lsv.mean():+.3f}; Welch p {w.pvalue:.4f}")
    res["E_perm"] = {"diff": float(obs), "p_perm": p_perm, "welch_p": float(welch.pvalue), "p_random_subset": p_sub}

    # ---------------- F transfer of a stratum amplitude (LOCO / LOPO) ----------------------------------------
    # predict each held-out cell's chord (= B1's Gd error when beta is the amplitude) with amplitudes fitted on
    # the other groups (equal-weight pubchem unit means, winsorised as frozen).  SSE over held-out cells, and the
    # count of held-out groups that improve.
    d["cls_cell"] = d.pubchem.map(u.cls)
    rows_f = []
    for grp in ("cluster_cpg", "pub"):
        out = []
        for g, te in d.groupby(grp):
            tr = d[d[grp] != g]
            ut = units_of(tr)
            a_pool = float(ut.cw.mean())
            a_cls = ut.groupby("cls").cw.mean().to_dict()
            a_nd = float(ut[ut.cls != "DGA"].cw.mean())
            a_dga = float(ut[ut.cls == "DGA"].cw.mean())
            # held-out class from the held-out cells' own unit labels (full-series direction = label-derived;
            # the reads-only sign agrees in ~97 % of cells, see H)
            cc = te.c.to_numpy()
            k = te.cls_cell.to_numpy()
            p_cls = np.array([a_cls.get(x, a_pool) for x in k])
            p_2 = np.where(k == "DGA", a_dga, a_nd)
            out.append({"grp": g, "n": len(te), "classes": "/".join(sorted(set(k))),
                        "sse_beta": float(((cc - BETA) ** 2).sum()), "sse_pool": float(((cc - a_pool) ** 2).sum()),
                        "sse_2": float(((cc - p_2) ** 2).sum()), "sse_3": float(((cc - p_cls) ** 2).sum()),
                        "sse_0": float((cc ** 2).sum()),
                        "hs_cells": int((k == "HS").sum())})
        o = pd.DataFrame(out)
        tot = o[["sse_0", "sse_beta", "sse_pool", "sse_2", "sse_3"]].sum()
        hs_o = o[o.hs_cells > 0]
        tot_hs = hs_o[["sse_0", "sse_beta", "sse_pool", "sse_2", "sse_3"]].sum()
        rec = {"holdout": grp, "groups": len(o), "cells": int(o.n.sum()),
               **{f"all_{k}": float(v) for k, v in tot.items()},
               "rel_3_vs_beta": float(1 - tot.sse_3 / tot.sse_beta), "rel_3_vs_2": float(1 - tot.sse_3 / tot.sse_2),
               "rel_2_vs_beta": float(1 - tot.sse_2 / tot.sse_beta),
               "groups_3_better_than_beta": int((o.sse_3 < o.sse_beta - 1e-12).sum()),
               "groups_3_worse_than_beta": int((o.sse_3 > o.sse_beta + 1e-12).sum()),
               "HS_groups": len(hs_o), "HS_rel_3_vs_beta": float(1 - tot_hs.sse_3 / tot_hs.sse_beta),
               "HS_groups_3_better": int((hs_o.sse_3 < hs_o.sse_beta - 1e-12).sum())}
        rows_f.append(rec)
        o.to_csv(OUT / f"transfer_{grp}.csv", index=False)
        print(f"\n[F] hold-out by {grp}: groups {len(o)}, cells {int(o.n.sum())}")
        print("   SSE (cells):", {k: round(float(v), 3) for k, v in tot.items()})
        print(f"   3-stratum vs beta: {rec['rel_3_vs_beta']:+.3f} rel SSE; vs 2-stratum {rec['rel_3_vs_2']:+.3f}; "
              f"2-stratum vs beta {rec['rel_2_vs_beta']:+.3f}; groups better/worse {rec['groups_3_better_than_beta']}/"
              f"{rec['groups_3_worse_than_beta']}")
        print(f"   groups holding HS cells: {len(hs_o)}; their SSE 3-stratum vs beta {rec['HS_rel_3_vs_beta']:+.3f}; "
              f"better in {rec['HS_groups_3_better']}")
        print(hs_o.round(4).to_string())
    res["F_transfer"] = rows_f

    # ---------------- G power ----------------------------------------------------------------------------------
    pw = []
    hs_vals = hsu.cw.to_numpy()
    hs_vals_noc5 = units_of(d[~d.pub.isin([C5, DMD])]).query("cls == 'HS'").cw.to_numpy()
    # cluster-level HS values (each cluster_cpg one value; equal-weight over its HS units)
    hs_cl_vals = hsu.groupby("cl").cw.mean().to_numpy()
    hs_cg_vals = hsu.groupby("cg").cw.mean().to_numpy()
    print(f"\n[G] HS values: pubchem n={len(hs_vals)} mean {hs_vals.mean():+.3f} sd {hs_vals.std(ddof=1):.3f}; "
          f"copy_group n={len(hs_cg_vals)} mean {hs_cg_vals.mean():+.3f} sd {hs_cg_vals.std(ddof=1):.3f}; "
          f"cluster n={len(hs_cl_vals)} mean {hs_cl_vals.mean():+.3f} sd {hs_cl_vals.std(ddof=1) if len(hs_cl_vals) > 1 else float('nan'):.3f}")
    for lab, vals in (("boot pubchem", hs_vals), ("boot pubchem noC5DMD", hs_vals_noc5),
                      ("boot copy_group", hs_cg_vals), ("boot cluster_cpg", hs_cl_vals)):
        if len(vals) < 2:
            continue
        for G in (5, 6, 8, 10, 13):
            r_ = np.random.default_rng(20261023 + zlib.crc32(f"skA|{lab}|{G}".encode()))
            X = vals[r_.integers(len(vals), size=(R, G))]
            Xn = X - vals.mean() + BETA
            pw.append({"world": lab, "G": G, "power": tpower(X, BETA), "power_a3": tpower(X, BETA, 0.025 / 3),
                       "size": tpower(Xn, BETA)})
    # analytic worlds: deflated means, inflated sd (between-unit sd + new-unit cell noise for m = 1 or 2 cells)
    sd_b = float(hs_vals.std(ddof=1))
    for mu in (-0.26, -0.217, -0.19, -0.175, -0.15, -0.13):
        for sdl, sdv in (("sd_obs", sd_b), ("sd_obs+m2noise", float(np.sqrt(sd_b ** 2 + 0.127 ** 2 / 2))),
                         ("sd 0.15", 0.15)):
            for G in (6, 8, 10, 13):
                pw.append({"world": f"nct mu={mu} {sdl}", "G": G, "power": nct_power(mu - BETA, sdv, G),
                           "power_a3": nct_power(mu - BETA, sdv, G, 0.025 / 3), "size": np.nan})
    pw = pd.DataFrame(pw)
    pw.to_csv(OUT / "power.csv", index=False)
    print(pw.pivot_table(index="world", columns="G", values="power", sort=False).round(3).to_string())
    print("size:\n", pw.dropna(subset=["size"]).pivot_table(index="world", columns="G", values="size").round(3).to_string())
    # LOO-deflated HS mean: each HS unit's value predicted by the mean of the others; and an EB-shrunk HS mean
    nd_mean = float(nd.cw.mean())
    tau2 = max(float(np.var(nd.groupby("cls").cw.mean().to_numpy(), ddof=1)) - 0, 0.0)
    se2 = sd_b ** 2 / len(hs_vals)
    shrunk = nd_mean + (tau2 / (tau2 + se2)) * (hs_vals.mean() - nd_mean)
    print(f"   non-DGA mean {nd_mean:+.3f}; crude two-group EB-shrunk HS mean {shrunk:+.3f}")
    res["G_shrunk_hs_mean"] = shrunk

    # ---------------- H reads-only direction --------------------------------------------------------------------
    rr = ref[(ref.policy == "DOPT_Q") & (ref.k == 3)][["cell_id", "split_seed", "read_set"]].drop_duplicates()
    pos = {c: i for i, c in enumerate(base.cell_id)}
    full_sign = {}
    agree = []
    for cid, seed, rs in rr.itertuples(index=False):
        i = pos[cid]
        mets = [LN.index(m) for m in str(rs).split(",")]
        y = Y1[i, mets]
        okm = np.isfinite(y)
        if okm.sum() < 2 or not np.isfinite(base.slope.iat[i]) or not np.isfinite(base.c.iat[i]):
            continue
        s = np.polyfit(XR[np.array(mets)[okm]], y[okm], 1)[0]
        agree.append({"cell_id": cid, "seed": seed, "reads": rs, "n_reads": int(okm.sum()),
                      "read_sign": s > 0, "full_sign": base.slope.iat[i] > 0, "pubchem": base.pubchem.iat[i]})
    ag = pd.DataFrame(agree)
    ag["agree"] = ag.read_sign == ag.full_sign
    ag["cls"] = ag.pubchem.map(u.cls)
    print(f"\n[H] DOPT_Q k=3 read sets with chord (C3): {len(ag)}; n_reads {ag.n_reads.value_counts().to_dict()}; "
          f"sign agreement {ag.agree.mean():.3f}")
    print(ag.groupby("cls").agree.agg(["size", "mean"]).round(3).to_string())
    # unit classification by majority of read signs (non-DGA)
    ag["unit_rs"] = ag.groupby("pubchem").read_sign.transform("mean") > 0.5
    ur = ag.groupby("pubchem").agg(rs=("unit_rs", "first")).join(u[["cls", "cw"]], how="left")
    ur = ur[ur.cls != "DGA"]
    flips = ur[(ur.rs & (ur.cls == "LS")) | (~ur.rs & (ur.cls == "HS"))]
    print(f"   non-DGA units whose read-majority direction differs from the full-series class: {len(flips)}")
    print(flips.round(3).to_string())
    res["H_read_sign"] = {"read_sets": int(len(ag)), "agree": float(ag.agree.mean()),
                          "unit_flips_nonDGA": int(len(flips))}
    res["seconds"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"\ndone in {res['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
