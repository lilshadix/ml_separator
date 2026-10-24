#!/usr/bin/env python
"""SKEPTIC A for idea confirm-3 (C-B1: B1 transport decided by the model-free Gd chord, A1 -> A2 -> A3).

EXPLORATORY: every number here comes from the 229 already-seen C3 cells / 5 dev seeds.  Independent recompute:
own chord code (Shannon CN8 radii checked), own unit tables, own B1/B0 unit SSEs from REF_BPG, own resampling.
Checks: (1) chord + link reproduction, algebraic identity of the link; (2) V2 / floor readings / bit-identical
series / near-copies; (3) publication clustering (pseudo-replication of lab x chemotype units); (4) LOPO and
C5 / DMD dependence; (5) A1/A2/A3 chain power, iid units vs lab-clustered draws, A3 on chord units vs all units;
(6) operating characteristics of the kill rules in null and B1-favourable worlds; (7) winsorisation vs estimand.
Writes only to ideas2/confirm3_skA_out/.  Reads only round-4 predictions, the bench and r3/r4 helpers.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm3_skA_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

LN = list(D.LANTHANIDES)
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
BETA = -0.0923
CLIP = (-0.40, 0.20)
R = 4000
SHANNON_CN8 = {"Eu": 1.066, "Gd": 1.053, "Tb": 1.040}


def log(*a):
    print(*a, flush=True)


def chord(Y):
    iE, iG, iT = (LN.index(m) for m in ("Eu", "Gd", "Tb"))
    t = (SHANNON_CN8["Eu"] - SHANNON_CN8["Gd"]) / (SHANNON_CN8["Eu"] - SHANNON_CN8["Tb"])
    tX = (interp.X_R[iG] - interp.X_R[iE]) / (interp.X_R[iT] - interp.X_R[iE])
    c = Y[:, iG] - ((1 - t) * Y[:, iE] + t * Y[:, iT])
    return c, t, float(tX)


def flips(G):
    return np.array(list(itertools.product((1.0, -1.0), repeat=G)))


def ttest_lower(X, c0, a=0.025):
    G = X.shape[1]
    m = X.mean(1)
    se = X.std(1, ddof=1) / np.sqrt(G)
    tc = SS.t.ppf(1 - a, G - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rej = (m - c0) / se < -tc
        lo = m - tc * se
        hi = m + tc * se
    return rej, lo, hi


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True).copy()
    Y = np.asarray(bench.Y, float)
    Yv2 = np.asarray(V2.matrix(bench), float)          # V2: whole-cell quarantines + V2-5 floor readings removed
    c, t_sh, t_x = chord(Y)
    cv2, _, _ = chord(Yv2)
    f["c"], f["cv2"] = c, cv2
    res["t_shannon"], res["t_XR"] = t_sh, t_x
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(ref.cell_id)
    f = f[f.cell_id.isin(c3)].copy()
    f["dga"] = f.extractant.astype(str).map(DN.is_dga)
    f["pubchem"] = f.publication_id.astype(str) + "|" + f.chemotype.astype(str)
    fc = f.dropna(subset=["c"]).copy()
    res["c3_cells"] = int(len(f))
    res["c3_cells_with_chord"] = int(len(fc))
    res["cells_chord_changed_by_V2"] = int((~np.isclose(fc.c, fc.cv2, equal_nan=False)).sum())
    res["cells_chord_lost_by_V2"] = fc.loc[fc.cv2.isna(), "cell_id"].tolist()

    # ---------------- (1) reproduce unit chords --------------------------------------------------------------
    def units(df, col="c"):
        g = df.dropna(subset=[col]).groupby("pubchem")
        return pd.DataFrame({"c": g[col].mean(), "n": g[col].size(), "dga": g.dga.mean() > 0.5,
                             "pub": g.publication_id.first()})
    U = units(fc)
    Uv2 = units(fc, "cv2")
    rep = {}
    for nm, UU in (("V1", U), ("V2", Uv2)):
        for grp, q in (("all", UU), ("nonDGA", UU[~UU.dga]), ("DGA", UU[UU.dga]),
                       ("nonDGA_noC5DMD", UU[~UU.dga & ~UU.pub.isin([C5, DMD])])):
            rep[f"{nm}|{grp}"] = {"units": int(len(q)), "pubs": int(q.pub.nunique()), "mean": float(q.c.mean()),
                                  "sd": float(q.c.std(ddof=1)), "median": float(q.c.median()),
                                  "mean_wins": float(np.clip(q.c, *CLIP).mean()),
                                  "n_clipped": int(((q.c < CLIP[0]) | (q.c > CLIP[1])).sum()),
                                  "single_cell_units": int((q.n == 1).sum())}
    res["unit_chords"] = rep
    log(json.dumps(rep, indent=1))

    # ---------------- (2) bit-identical series / near-copies among chord cells ---------------------------------
    dup = []
    Yc = Y[fc.index.to_numpy()]
    ids = fc.cell_id.to_numpy()
    for i in range(len(fc)):
        for j in range(i + 1, len(fc)):
            sh = np.isfinite(Yc[i]) & np.isfinite(Yc[j])
            if sh.sum() < 3:
                continue
            d = Yc[i, sh] - Yc[j, sh]
            if np.max(np.abs(d - d.mean())) < 1e-6:     # identical up to a constant (incl. exact copies)
                dup.append({"a": ids[i], "b": ids[j], "pub_a": fc.publication_id.iat[i], "pub_b": fc.publication_id.iat[j],
                            "chem_a": fc.chemotype.iat[i], "chem_b": fc.chemotype.iat[j], "shared": int(sh.sum()),
                            "offset": float(d.mean()), "c_a": float(fc.c.iat[i]), "c_b": float(fc.c.iat[j])})
    dup = pd.DataFrame(dup)
    dup.to_csv(OUT / "dup_series.csv", index=False)
    res["dup_pairs"] = int(len(dup))
    res["dup_pairs_cross_unit"] = int(0 if dup.empty else ((dup.pub_a + dup.chem_a) != (dup.pub_b + dup.chem_b)).sum())
    log("duplicates (identical up to a constant, >=3 shared metals):")
    log(dup.to_string() if len(dup) else "none")

    # ---------------- (3) link and algebraic identity on REF_BPG ----------------------------------------------
    t = ref[(ref.k_max >= 3) & (ref.k == 3)].copy()
    t["gd_read"] = t.read_set.str.split(",").map(lambda s: "Gd" in s)
    t["gdpair"] = (t.A.isin(["Eu", "Gd"]) & t.B.isin(["Gd", "Tb"]))
    t["g"] = (t.B0 - t.y) ** 2 - (t.B1 - t.y) ** 2
    cm = fc.set_index("cell_id").c
    t["c"] = t.cell_id.map(cm)
    res["rows"] = int(len(t))
    res["share_rows_gd_read"] = float(t.groupby(["split_seed", "cell_id"]).gd_read.first().mean())
    # identity: Gd hidden, Eu/Gd and Gd/Tb pairs: d = B1 - B0 = +beta_fold on (Eu,Gd), -beta_fold on (Gd,Tb)
    h = t[~t.gd_read & t.gdpair & t.c.notna()].copy()
    h["d"] = h.B1 - h.B0
    # per (seed, fold, cell, read_set) Gd-pair gain vs 2*(c^2 - (c - b)^2) with b = the fold's beta
    eg = h[h.A == "Eu"].set_index(["split_seed", "cell_id", "read_set", "policy", "draw"])
    gt = h[h.B == "Tb"].set_index(["split_seed", "cell_id", "read_set", "policy", "draw"])
    jj = eg[["d", "g", "c"]].join(gt[["d", "g"]], rsuffix="_t", how="inner")
    jj["b"] = jj.d
    jj["g_pair"] = jj.g + jj.g_t
    jj["g_alg"] = 2 * (jj.c ** 2 - (jj.c - jj.b) ** 2)
    res["identity"] = {"n": int(len(jj)), "d_eu_plus_d_tb_maxabs": float((jj.d + jj.d_t).abs().max()),
                       "beta_fold_range": [float(jj.b.min()), float(jj.b.max())],
                       "max_abs_gain_minus_algebra": float((jj.g_pair - jj.g_alg).abs().max())}
    log(json.dumps(res["identity"], indent=1))
    ns = t.split_seed.nunique()
    cg = t.groupby("cell_id").g.sum() / ns
    cgh = t[~t.gd_read].groupby("cell_id").g.sum() / ns
    d = pd.DataFrame({"gain": cg}).join(cm, how="inner")
    d["pubchem"] = fc.set_index("cell_id").pubchem.reindex(d.index)
    d["win"] = d.gain > 0
    d["rule"] = d.c < BETA / 2
    uu = d.groupby("pubchem").agg(c=("c", "mean"), gain=("gain", "sum"))
    res["link"] = {"cells": int(len(d)), "spearman": float(SS.spearmanr(d.c, d.gain)[0]),
                   "rule_acc_cells": float((d.win == d.rule).mean()),
                   "rule_acc_units": float(((uu.c < BETA / 2) == (uu.gain > 0)).mean()), "units": int(len(uu)),
                   "share_gain_from_gd_hidden_rows": float(cgh.sum() / cg.sum())}
    log(json.dumps(res["link"], indent=1))

    # ---------------- unit SSE table for A3 (all pubchem units, chord or not) ----------------------------------
    t["pubchem"] = t.publication.astype(str) + "|" + t.chemotype.astype(str)
    g = t.assign(er=(t.B0 - t.y) ** 2, ec=(t.B1 - t.y) ** 2).groupby(["pubchem", "split_seed"])[["er", "ec"]].sum()
    us = g.groupby(level=0).mean()
    us["dga"] = t.assign(dga=t.extractant.astype(str).map(DN.is_dga)).groupby("pubchem").dga.mean().reindex(us.index) > 0.5
    us["pub"] = t.groupby("pubchem").publication.first().reindex(us.index)
    us["c"] = U.c.reindex(us.index)
    us["L"] = np.log(us.er / us.ec)
    nd = us[~us.dga]
    res["nonDGA_units_all"] = {"units": int(len(nd)), "with_chord": int(nd.c.notna().sum()),
                               "no_chord_W_L_T": [int((nd[nd.c.isna()].L > 1e-12).sum()),
                                                  int((nd[nd.c.isna()].L < -1e-12).sum()),
                                                  int((nd[nd.c.isna()].L.abs() <= 1e-12).sum())],
                               "chord_W_L": [int((nd[nd.c.notna()].L > 0).sum()), int((nd[nd.c.notna()].L < 0).sum())]}
    log(json.dumps(res["nonDGA_units_all"], indent=1))
    us.to_csv(OUT / "unit_table.csv")

    # ---------------- (4) publication clustering of chord units ------------------------------------------------
    P = U[~U.dga].copy()
    pubm = P.groupby("pub").c.agg(["mean", "size"])
    # one-way ANOVA ICC among pubchem units within publication (pubs with >= 2 units)
    multi = P[P.pub.map(P.pub.value_counts()) >= 2]
    k_ = multi.groupby("pub").size()
    msb = (k_ * (multi.groupby("pub").c.mean() - multi.c.mean()) ** 2).sum() / (len(k_) - 1)
    msw = ((multi.c - multi.pub.map(multi.groupby("pub").c.mean())) ** 2).sum() / (len(multi) - len(k_))
    n0 = (len(multi) - (k_ ** 2).sum() / len(multi)) / (len(k_) - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    res["clustering_nonDGA"] = {"units": int(len(P)), "pubs": int(len(pubm)), "pubs_multi_unit": int(len(k_)),
                                "units_in_multi_pubs": int(len(multi)), "icc_within_pub": float(icc),
                                "pub_level_mean": float(pubm["mean"].mean()), "pub_level_sd": float(pubm["mean"].std(ddof=1))}
    log(json.dumps(res["clustering_nonDGA"], indent=1))

    # ---------------- (5) LOPO / C5 / top-k dependence (analytic power, noncentral t) -------------------------
    def npow(mu, sd, G, c0):
        dlt = (c0 - mu) / (sd / np.sqrt(G))
        return float(SS.nct.sf(SS.t.ppf(0.975, G - 1), G - 1, dlt))
    lo = []
    for p in sorted(P.pub.unique()):
        q = np.clip(P[P.pub != p].c, *CLIP)
        lo.append({"drop": p, "units": len(q), "mean_w": q.mean(), "sd_w": q.std(ddof=1),
                   "A1_G8": npow(q.mean(), q.std(ddof=1), 8, 0), "A2_G8": npow(q.mean(), q.std(ddof=1), 8, BETA / 2),
                   "A2_G13": npow(q.mean(), q.std(ddof=1), 13, BETA / 2)})
    q = np.clip(P.c, *CLIP)
    lo.append({"drop": "none", "units": len(q), "mean_w": q.mean(), "sd_w": q.std(ddof=1),
               "A1_G8": npow(q.mean(), q.std(ddof=1), 8, 0), "A2_G8": npow(q.mean(), q.std(ddof=1), 8, BETA / 2),
               "A2_G13": npow(q.mean(), q.std(ddof=1), 13, BETA / 2)})
    top3 = pubm.sort_values("mean").index[:3].tolist()
    q = np.clip(P[~P.pub.isin(top3)].c, *CLIP)
    lo.append({"drop": "top3_most_negative:" + ",".join(top3), "units": len(q), "mean_w": q.mean(), "sd_w": q.std(ddof=1),
               "A1_G8": npow(q.mean(), q.std(ddof=1), 8, 0), "A2_G8": npow(q.mean(), q.std(ddof=1), 8, BETA / 2),
               "A2_G13": npow(q.mean(), q.std(ddof=1), 13, BETA / 2)})
    lo = pd.DataFrame(lo)
    lo.to_csv(OUT / "lopo.csv", index=False)
    log(lo.round(3).to_string())

    # ---------------- (6) chain power: iid chord units vs all units for A3 vs lab-clustered draws -------------
    ndc = nd[nd.c.notna()].copy()
    rows = []
    E = {G: flips(G) for G in (6, 8, 10, 13)}

    def chain(Xc, Lx, sr, sc, Gm):
        """Xc (R,G) chords with NaN for units lacking a chord; Lx log ratios.  A1/A2 on chord units only."""
        out = {}
        a1 = np.zeros(len(Xc), bool)
        a2 = np.zeros(len(Xc), bool)
        fut = np.zeros(len(Xc), bool)
        for r in range(len(Xc)):
            x = Xc[r][np.isfinite(Xc[r])]
            if len(x) < 3:
                continue
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            tc = SS.t.ppf(0.975, len(x) - 1)
            a1[r] = m / se < -tc
            a2[r] = (m - BETA / 2) / se < -tc
            fut[r] = m - tc * se > BETA / 2
        Ee = E[Gm] if Gm in E else flips(Gm)
        pL = ((Lx @ Ee.T) >= Lx.sum(1)[:, None] - 1e-12).sum(1) / Ee.shape[0]
        a3 = (pL <= 0.025 + 1e-12) & (sr.sum(1) > sc.sum(1))
        out.update(A1=a1.mean(), A2=a2.mean(), A3=a3.mean(), A1A2=(a1 & a2).mean(), chain=(a1 & a2 & a3).mean(),
                   A1_kill=(~a1).mean(), A2_futility=fut.mean())
        return out

    for G in (8, 13):
        for tau in (0.0, 0.065):
            for design in ("iid_chord_units", "iid_all_nonDGA_units_A3_all"):
                rng = np.random.default_rng(20261023 + zlib.crc32(f"skA|{design}|{G}|{tau}".encode()))
                PP = ndc if design == "iid_chord_units" else nd
                ui = rng.integers(len(PP), size=(R, G))
                Xc = np.clip(PP.c.to_numpy(), *CLIP)[ui]
                sr, sc = PP.er.to_numpy()[ui].copy(), PP.ec.to_numpy()[ui].copy()
                if tau > 0:
                    s = tau / np.sqrt(2)
                    sr *= (1 - rng.normal(0, s, sr.shape)) ** 2
                    sc *= (1 - rng.normal(0, s, sc.shape)) ** 2
                rec = {"design": design, "G": G, "tau": tau}
                rec.update(chain(Xc, np.log(sr / sc), sr, sc, G))
                rows.append(rec)
                log(f"{time.time() - t0:6.1f}s {rec}")
    chain_df = pd.DataFrame(rows)

    # lab-clustered: draw G_lab publications, keep ALL their chord units; chord t-test on all units (pseudo-
    # replication) -- size at re-centred null boundaries and power, versus one unit per lab
    pubs = sorted(ndc.pub.unique())
    byp = {p: np.clip(ndc[ndc.pub == p].c.to_numpy(), *CLIP) for p in pubs}
    mu_units = np.clip(ndc.c.to_numpy(), *CLIP).mean()
    crow = []
    for G in (6, 8, 13):
        for world, cen in (("NULL_A1_0", 0.0), ("NULL_A2_halfbeta", BETA / 2), ("EMP", None), ("WORLD_beta", BETA)):
            for mode in ("all_units_of_lab", "one_unit_per_lab"):
                rng = np.random.default_rng(20261023 + zlib.crc32(f"skAclu|{G}|{world}|{mode}".encode()))
                rej0 = np.zeros(R, bool)
                rejh = np.zeros(R, bool)
                kill = np.zeros(R, bool)
                fut = np.zeros(R, bool)
                nun = np.zeros(R)
                pi = rng.integers(len(pubs), size=(R, G))
                for r in range(R):
                    if mode == "all_units_of_lab":
                        x = np.concatenate([byp[pubs[k]] for k in pi[r]])
                    else:
                        x = np.array([rng.choice(byp[pubs[k]]) for k in pi[r]])
                    if cen is not None:
                        x = x - mu_units + cen
                    nun[r] = len(x)
                    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
                    tc = SS.t.ppf(0.975, len(x) - 1)
                    rej0[r] = m / se < -tc
                    rejh[r] = (m - BETA / 2) / se < -tc
                    kill[r] = not rej0[r]
                    fut[r] = m - tc * se > BETA / 2
                crow.append({"G_labs": G, "world": world, "mode": mode, "mean_units": nun.mean(),
                             "A1_rej": rej0.mean(), "A2_rej": rejh.mean(), "A1_kill(not rejected)": kill.mean(),
                             "A2_futility": fut.mean()})
                log(f"{time.time() - t0:6.1f}s {crow[-1]}")
    clu = pd.DataFrame(crow)
    clu.to_csv(OUT / "clustered_draws.csv", index=False)

    # ---------------- (7) kill-rule operating characteristics in shifted worlds (iid chord units) ------------
    krow = []
    base = np.clip(ndc.c.to_numpy(), *CLIP)
    for G in (6, 8, 13):
        for cen in (0.0, BETA / 2, BETA, -0.142, -0.217, -0.26):
            rng = np.random.default_rng(20261023 + zlib.crc32(f"skAkill|{G}|{cen}".encode()))
            X = base[rng.integers(len(base), size=(R, G))] - base.mean() + cen
            a1, lo1, hi1 = ttest_lower(X, 0.0)
            a2, lo2, _ = ttest_lower(X, BETA / 2)
            krow.append({"G": G, "true_mean_chord": cen, "B1_helps_on_avg": cen < BETA / 2,
                         "P_A1_pass": a1.mean(), "P_A1_kill_rule(UB>0 or est>=0)": (~a1).mean(),
                         "P_A1_est_ge_0": (X.mean(1) >= 0).mean(), "P_A2_pass": (a1 & a2).mean(),
                         "P_A2_futility": (lo2 > BETA / 2).mean(),
                         "P_inconclusive_after_A1": (a1 & ~a2 & ~(lo2 > BETA / 2)).mean()})
    kill = pd.DataFrame(krow)
    kill.to_csv(OUT / "kill_rules.csv", index=False)
    log(kill.round(3).to_string())

    # ---------------- (8) winsorisation vs the linear-gain estimand --------------------------------------------
    allu = U.copy()
    res["winsor"] = {"units_clipped_all": allu[(allu.c < CLIP[0]) | (allu.c > CLIP[1])].index.tolist(),
                     "corpus_min_unit": float(allu.c.min()), "corpus_max_unit": float(allu.c.max()),
                     "gain_raw_all": float((2 * BETA * (2 * allu.c - BETA)).mean()),
                     "gain_wins_all": float((2 * BETA * (2 * np.clip(allu.c, *CLIP) - BETA)).mean())}
    log(json.dumps(res["winsor"], indent=1))

    chain_df.to_csv(OUT / "chain_power.csv", index=False)
    log(chain_df.round(3).to_string())
    log(clu.round(3).to_string())
    res["seconds"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
