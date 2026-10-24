#!/usr/bin/env python
"""Skeptic A for idea confirm-1 (CP-13 package).  EXPLORATORY: every number is on already-seen V1 / C3 cells.

Independent recompute (does not import the confirm_* scripts):
  1. Gd chord per cell (logD(Gd) - Eu-Tb interpolation on -r(CN8)), selectivity direction (LS slope over all
     measured metals), DGA flag (r4 SMARTS), C3 membership (REF_BPG), V2 drop, copy group, cluster_cpg,
     a SMILES-based mechanism class.
  2. Stratum table: heavy-selective non-DGA (HS), light-selective non-DGA (LS), DGA; per-unit listing.
  3. Independence: how many copy groups / clusters / chemotypes make up the HS stratum; leave-one-group-out.
  4. Metadata stratum (acidic organoP + chelators + aromatic N-donors) vs slope stratum.
  5. Tautology: is the chord -> B1-B0 link algebraic?  (B1 - B0 on REF rows.)
  6. C-GD power under alternative planning worlds: authors' unit bootstrap; one-unit-per-copy-group bootstrap;
     predictive power (stratum mean drawn from its own bootstrap); effect shrunk; realistic G_HS 3-5;
     size with clustered units (two chemotype units from one lab).
Writes only to ideas2/confirm1_skA_out/.
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/confirm1_recompute_skA.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm1_skA_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402
from rdkit import Chem  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import copy_group_map  # noqa: E402

BETA = -0.0923
CLIP = (-0.40, 0.20)
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
R = 6000
PAT = {
    "P_acid": Chem.MolFromSmarts("[PX4](=O)[OX2H1,OX1-]"),
    "COOH": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "arom_N": Chem.MolFromSmarts("n"),
    "P_neutral": Chem.MolFromSmarts("[PX4]=O"),
    "quatN": Chem.MolFromSmarts("[NX4+]"),
    "diketone": Chem.MolFromSmarts("[CX3](=O)[CH2,CH1][CX3](=O)"),
    "amide": Chem.MolFromSmarts("[NX3][CX3](=O)"),
    "OH_arom": Chem.MolFromSmarts("c[OX2H1]"),
}


def mech(smiles: str) -> str:
    out = []
    for s in str(smiles).split("."):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        hits = {k for k, p in PAT.items() if m.HasSubstructMatch(p)}
        if DN.is_dga(s):
            out.append("DGA")
        elif "P_acid" in hits:
            out.append("acidP")
        elif "COOH" in hits:
            out.append("carboxylic")
        elif "arom_N" in hits or "OH_arom" in hits:
            out.append("aromN")
        elif "diketone" in hits:
            out.append("diketone")
        elif "quatN" in hits:
            out.append("quatN")
        elif "P_neutral" in hits:
            out.append("neutralP")
        elif "amide" in hits:
            out.append("amide")
        else:
            out.append("other")
    return "+".join(sorted(set(out))) if out else "unparsed"


def tpow(X, mu0, alpha=0.025):
    G = X.shape[1]
    m = X.mean(1)
    se = X.std(1, ddof=1) / np.sqrt(G)
    tc = SS.t.ppf(1 - alpha, G - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((((m - mu0) / se) < -tc).mean())


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    b = D.load_bench()
    f = b.frame.reset_index(drop=True).copy()
    Y = np.asarray(b.Y, float)
    LN = list(D.LANTHANIDES)
    XR = np.asarray(interp.X_R, float)
    iEu, iGd, iTb = LN.index("Eu"), LN.index("Gd"), LN.index("Tb")
    t = (XR[iGd] - XR[iEu]) / (XR[iTb] - XR[iEu])
    f["c"] = Y[:, iGd] - ((1 - t) * Y[:, iEu] + t * Y[:, iTb])
    sl = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            sl[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    f["slope"] = sl
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    f["c3"] = f.cell_id.isin(set(ref.cell_id))
    f["v2drop"] = f.cell_id.isin(set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS))
    f["dga"] = f.extractant.astype(str).map(DN.is_dga)
    f["mech"] = f.extractant.astype(str).map(mech)
    cg = copy_group_map()
    f["copy_group"] = f.publication_id.astype(str).map(cg)
    cl = M4.add_clusters(ref[["cell_id", "publication", "chemotype"]].drop_duplicates("cell_id"), None)
    f = f.merge(cl[["cell_id", "cluster_cpg"]], on="cell_id", how="left")
    f["unit"] = f.publication_id.astype(str) + "|" + f.chemotype.astype(str)
    res = {"t_gd": float(t)}

    d = f[f.c3 & np.isfinite(f.c)].copy()
    res["cells_with_chord_C3"] = int(len(d))
    res["v2drop_among_them"] = int(d.v2drop.sum())
    u = d.groupby("unit").agg(c=("c", "mean"), hs=("slope", lambda s: float((s > 0).mean())), dga=("dga", "mean"),
                              n=("c", "size"), pub=("publication_id", "first"), chem=("chemotype", "first"),
                              cgroup=("copy_group", "first"), cluster=("cluster_cpg", "first"),
                              mech=("mech", lambda s: "/".join(sorted(set(s)))),
                              ext=("extractant", lambda s: str(s.iloc[0])[:70]))
    u["cw"] = np.clip(u.c, *CLIP)
    u["cls"] = np.where(u.dga > 0.5, "DGA", np.where(u.hs > 0.5, "nonDGA_HS", "nonDGA_LS"))
    u = u.sort_values(["cls", "c"])
    u.to_csv(OUT / "units.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 70)
    print(u[["cls", "c", "cw", "n", "hs", "mech", "cgroup", "cluster", "ext"]].round(3).to_string(), flush=True)

    # ---- 2. stratum table, unit level and lab (copy-group) level -------------------------------------------
    strat = []
    for cls, g in u.groupby("cls"):
        lab = g.groupby("cgroup").cw.mean()
        clu = g.groupby("cluster").cw.mean()
        strat.append({"cls": cls, "units": len(g), "cells": int(g.n.sum()), "pubs": g.pub.nunique(),
                      "copy_groups": g.cgroup.nunique(), "clusters": g.cluster.nunique(), "chemotypes": g.chem.nunique(),
                      "single_cell_units": int((g.n == 1).sum()),
                      "mean_w": g.cw.mean(), "sd_w": g.cw.std(ddof=1), "mean_raw": g.c.mean(),
                      "lab_mean_w": lab.mean(), "lab_sd_w": lab.std(ddof=1) if len(lab) > 1 else np.nan,
                      "cluster_mean_w": clu.mean(), "cluster_n": len(clu)})
    strat = pd.DataFrame(strat)
    print(strat.round(3).to_string(), flush=True)
    res["strata"] = strat.to_dict("records")

    hs = u[u.cls == "nonDGA_HS"].copy()
    ls = u[u.cls == "nonDGA_LS"].copy()
    # ---- 3. dependence on few groups ------------------------------------------------------------------------
    dep = []
    for key in ("pub", "cgroup", "cluster", "chem"):
        for gname, _ in hs.groupby(key):
            rest = hs[hs[key] != gname]
            dep.append({"by": key, "dropped": gname, "units_left": len(rest), "mean_w": rest.cw.mean(),
                        "sd_w": rest.cw.std(ddof=1)})
    dep = pd.DataFrame(dep)
    print(dep.round(3).to_string(), flush=True)
    dep.to_csv(OUT / "hs_leave_one_out.csv", index=False)
    s = hs.sort_values("cw")
    drops = {"no_C5": hs[hs.pub != C5], "no_C5_DMD": hs[~hs.pub.isin([C5, DMD])],
             "no_sc061_BTBP": hs[hs.chem != "sc061"], "no_top3_units": s.iloc[3:],
             "no_aromN": hs[~hs.mech.str.contains("aromN")], "aromN_only": hs[hs.mech.str.contains("aromN")],
             "acidP_only": hs[hs.mech.str.contains("acidP")]}
    res["hs_subsets"] = {k: {"units": len(v), "mean_w": float(v.cw.mean()) if len(v) else None,
                             "sd_w": float(v.cw.std(ddof=1)) if len(v) > 1 else None,
                             "units_list": list(v.index)} for k, v in drops.items()}
    print(json.dumps(res["hs_subsets"], indent=1), flush=True)

    # ---- 4. metadata stratum vs slope stratum (non-DGA) -----------------------------------------------------
    nd = u[u.cls != "DGA"].copy()
    nd["meta_hs"] = nd.mech.str.contains("acidP|carboxylic|aromN|diketone")
    ct = pd.crosstab(nd.meta_hs, nd.cls)
    print(ct, flush=True)
    res["meta_vs_slope"] = {str(k): v for k, v in ct.to_dict().items()}
    mh = nd[nd.meta_hs]
    res["meta_stratum"] = {"units": len(mh), "mean_w": float(mh.cw.mean()), "sd_w": float(mh.cw.std(ddof=1)),
                           "share_slope_hs": float((mh.cls == "nonDGA_HS").mean())}
    print("meta stratum", res["meta_stratum"], flush=True)
    print(nd.groupby("mech").agg(units=("cw", "size"), mean_w=("cw", "mean"),
                                 hs_share=("cls", lambda s: float((s == "nonDGA_HS").mean()))).round(3), flush=True)

    # ---- 5. tautology: where B1 differs from B0, and the algebraic gain identity ----------------------------
    r = ref[(ref.k == 3)].copy()
    r["dB"] = r.B1 - r.B0
    nz = r[np.abs(r.dB) > 1e-12]
    res["B1_vs_B0"] = {"rows": int(len(r)), "rows_B1_ne_B0": int(len(nz)),
                       "pairs_where_B1_ne_B0": nz.assign(p=nz.A + "/" + nz.B).p.value_counts().head(8).to_dict(),
                       "share_rows_ne_touching_Gd": float(((nz.A == "Gd") | (nz.B == "Gd")).mean())}
    # Gd-target rows with Eu and Tb read: identity gain = beta(2c - beta)?
    rr = r.copy()
    rr["g"] = (rr.B0 - rr.y) ** 2 - (rr.B1 - rr.y) ** 2
    cell_g = rr.groupby("cell_id").g.sum() / rr.split_seed.nunique()
    dd = d.set_index("cell_id").loc[lambda x: x.index.isin(cell_g.index)]
    dd = dd.assign(gain=cell_g.reindex(dd.index))
    res["B1_vs_B0"]["cells"] = int(len(dd))
    res["B1_vs_B0"]["agree_sign_rule_cells"] = float(((dd.c < BETA / 2) == (dd.gain > 0)).mean())
    res["B1_vs_B0"]["spearman"] = float(SS.spearmanr(dd.c, dd.gain)[0])
    # what a pure algebraic predictor gives: gain_pred = sum over Gd pairs of the change in squared error if
    # only the Gd log D shifts by beta; correlation with the observed gain
    print(json.dumps(res["B1_vs_B0"], indent=1, default=str), flush=True)

    # ---- 6. C-GD power under alternative planning worlds ---------------------------------------------------
    rng = np.random.default_rng(20261023)
    hsv = hs.cw.to_numpy()
    lab_hs = hs.groupby("cgroup").cw.mean().to_numpy()
    nd_all = nd.cw.to_numpy()
    pw = []

    def add(world, G, X, mu0=BETA, note=""):
        pw.append({"world": world, "G": G, "pow_025": tpow(X, mu0, 0.025), "pow_a3": tpow(X, mu0, 0.025 / 3),
                   "note": note})

    for G in (3, 4, 5, 6, 8, 10):
        # authors' world
        add("A_units_boot", G, hsv[rng.integers(len(hsv), size=(R, G))])
        # one value per lab (copy group)
        add("B_labs_boot", G, lab_hs[rng.integers(len(lab_hs), size=(R, G))], note=f"{len(lab_hs)} copy groups")
        # predictive: stratum mean uncertain -> draw a bootstrap stratum, then G units from it
        Xp = np.empty((R, G))
        for i in range(R):
            boot = hsv[rng.integers(len(hsv), size=len(hsv))]
            Xp[i] = boot[rng.integers(len(boot), size=G)]
        add("C_predictive_units", G, Xp)
        Xq = np.empty((R, G))
        for i in range(R):
            boot = lab_hs[rng.integers(len(lab_hs), size=len(lab_hs))]
            Xq[i] = boot[rng.integers(len(boot), size=G)]
        add("D_predictive_labs", G, Xq)
        # effect not HS-specific: new HS units centred at the all-non-DGA unit mean, HS spread
        add("E_center_nonDGA_mean", G, hsv[rng.integers(len(hsv), size=(R, G))] - hsv.mean() + nd_all.mean(),
            note=f"center {nd_all.mean():.3f}")
        # effect halved toward beta (winner's curse allowance)
        cen = BETA + 0.5 * (hsv.mean() - BETA)
        add("F_effect_halved", G, hsv[rng.integers(len(hsv), size=(R, G))] - hsv.mean() + cen, note=f"center {cen:.3f}")
        # no BTBP/C5: stratum without sc061
        v = hs[hs.chem != "sc061"].cw.to_numpy()
        add("G_no_sc061", G, v[rng.integers(len(v), size=(R, G))], note=f"{len(v)} units, mean {v.mean():.3f}")
        # all non-DGA spread (sd ~0.12) centred at -0.26 / -0.175 / -0.142
        for cen2 in (-0.26, -0.20, -0.175, -0.142):
            add(f"H_nonDGAspread_c{cen2}", G, nd_all[rng.integers(len(nd_all), size=(R, G))] - nd_all.mean() + cen2)
    # size with clustered units: G_HS units from fewer labs; within-lab pairs share a lab effect.
    # Estimate within-copy-group ICC of unit chords among non-DGA units with >=2 units per copy group.
    grp = nd.groupby("cgroup").cw
    multi = [g.to_numpy() for _, g in grp if len(g) >= 2]
    within = np.concatenate([x - x.mean() for x in multi]) if multi else np.array([])
    dfw = sum(len(x) - 1 for x in multi)
    var_w = float((within ** 2).sum() / dfw) if dfw else np.nan
    var_tot = float(nd.cw.var(ddof=1))
    icc = max(0.0, 1 - var_w / var_tot) if np.isfinite(var_w) else np.nan
    res["icc_within_copygroup_nonDGA"] = {"groups_with_2plus": len(multi), "var_within": var_w, "var_total": var_tot,
                                          "icc": icc}
    print("ICC", res["icc_within_copygroup_nonDGA"], flush=True)
    sd0 = float(hsv.std(ddof=1))
    for G, labs in ((6, 6), (6, 4), (6, 3), (8, 5), (8, 4)):
        for rho in (0.0, 0.3, 0.5, icc):
            # G units in `labs` labs, assigned round-robin; normal world at the null boundary
            lab_of = np.arange(G) % labs
            L = rng.normal(0, np.sqrt(rho) * sd0, size=(R, labs))
            E = rng.normal(0, np.sqrt(1 - rho) * sd0, size=(R, G))
            X = BETA + L[:, lab_of] + E
            pw.append({"world": f"SIZE_clustered_rho{rho:.2f}_labs{labs}", "G": G, "pow_025": tpow(X, BETA, 0.025),
                       "pow_a3": tpow(X, BETA, 0.025 / 3), "note": "size at null"})
    pw = pd.DataFrame(pw)
    pw.to_csv(OUT / "cgd_power_worlds.csv", index=False)
    print(pw.round(3).to_string(), flush=True)
    res["seconds"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("seconds", res["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
