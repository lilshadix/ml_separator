#!/usr/bin/env python
"""Skeptic B check of idea recipe-3 (sign-first triage from an own prior series).  EXPLORATORY: the 229 C3 cells of
the V1 bench have been seen by rounds 1-4 and by every round-5 probe.  Training-free; reads only existing outputs
(recipe_out/*.parquet, round-4 REF_BPG.parquet) and the V1 bench.  Writes only under ideas2/recipe3_skepB_out/.

Questions
 Q1  Are the comparators scored on the SAME selected rows (|COPY| >= 0.1)?  always-heavier, pair-type majority
     (leave-own-publication-out), chemotype x pair-type majority (LOPO), G14 zero-shot (round-4 OOF), B1 at 4 reads
     (round-4 OOF, DOPT_Q), D7 with 2 / 4 HULL reads, SH_own with 2 HULL reads (recipe-1's own option), xpub COPY.
 Q2  Coverage-matched comparators (each comparator's own |pred| threshold at COPY's coverage).
 Q3  Three-way (lighter / heavier / inseparable at 0.1) accuracy.
 Q4  Publication-level accuracy, lower bounds, the kill criterion replayed on the corpus, and its power at 11 families.
 Q5  Near-replicate sign agreement (same publication and extractant, only contact time / metal conc / nothing
     differs) as the 're-measure' reliability ceiling.
 Q6  Confident wrong calls; accuracy by condition-change type.
 Q7  What the triage saves: metals needed to re-measure the flagged pairs vs recipe-1's 2 reads.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe3_skepB_signcheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

BASE = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2")
RIN = BASE / "recipe_out"
OUT = BASE / "recipe3_skepB_out"
K_SH = 1.78
GD = CO.GD
X = interp.X_R
THR = 0.1
FLAG_PAIRS = {"Er/Tm", "Ho/Er", "La/Ce"}
TRIVIAL = {"contact_time", "metal_conc"}
rng = np.random.default_rng(20261024)


def fam(c: str) -> str:
    if c.startswith("cond__acid__"):
        return "acid_type"
    if c == "cond__acid_concentration_M":
        return "acid_conc"
    if c.startswith("cond__additive__"):
        return "additive"
    if c.startswith("cond__diluent__"):
        return "diluent"
    return {"cond__contact_time_min": "contact_time", "cond__extractant_concentration_M": "extr_conc",
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}[c]


def pwl(read_idx, r, x):
    """PWL along x with constant continuation from the read indices."""
    ri = sorted(read_idx)
    return np.interp(x, x[ri], r[ri])


def acc(pred, y, rows):
    m = rows & np.isfinite(pred)
    return float((np.sign(pred[m]) == np.sign(y[m])).mean()) if m.sum() else np.nan, int(m.sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols].to_numpy(dtype=float)
    pubA = f["publication_id"].astype(str).to_numpy()
    extA = f["extractant"].astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)

    cells = pd.read_parquet(RIN / "recipe_cells.parquet")
    pairs = pd.read_parquet(RIN / "recipe_pairs.parquet")
    p0 = pairs[pairs.n_read == 0].merge(
        cells[["ci", "publication", "chemotype", "cluster", "c5", "n_own", "n_xpub", "own_fams", "own_ids",
               "extractant"]], on="ci")
    allc3 = p0.copy()  # all 229 C3 cells, 1802 rows
    # ---------- comparators that need other publications (LOPO majority) ----------
    noc5 = allc3[~allc3.c5]
    pt_lopo, ch_lopo = [], []
    for _, r in allc3.iterrows():
        o = noc5[(noc5.pair == r.pair) & (noc5.publication != r.publication)]
        pt_lopo.append(np.sign(o.y.median()) if len(o) else np.nan)
        oc = o[o.chemotype == r.chemotype]
        ch_lopo.append(np.sign(oc.y.median()) if len(oc) >= 3 else (np.sign(o.y.median()) if len(o) else np.nan))
    allc3["pred_PTMAJ"] = pt_lopo
    allc3["pred_CHMAJ"] = ch_lopo
    allc3["pred_AH"] = -1.0
    # ---------- round-4 OOF: B1 (4 reads, DOPT_Q) and G14 zero-shot, per seed ----------
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    ref["pair"] = ref.A + "/" + ref.B
    seeds = sorted(ref.split_seed.unique())
    for s_i, s in enumerate(seeds):
        rr = ref[ref.split_seed == s][["cell_id", "pair", "B1", "G14@ZERO"]].rename(
            columns={"B1": f"pred_B1_s{s_i}", "G14@ZERO": f"pred_G14_s{s_i}"})
        allc3 = allc3.merge(rr, on=["cell_id", "pair"], how="left")
    # ---------- D7 and SH_own at 2 / 4 HULL reads (recomputed) ----------
    extra = {k: np.full(len(allc3), np.nan) for k in ("pred_D7_2", "pred_D7_4", "pred_SHown_2", "pred_SHown_4")}
    allc3 = allc3.reset_index(drop=True)
    for t, g in allc3.groupby("cell_id"):
        i = pos[t]
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x = X[Mi]
        yv = Y[i, Mi]
        u = np.zeros(m)
        if GD in Mi:
            u[Mi.index(GD)] = HIST_D7_BETA
        order = [0, m - 1]
        while len(order) < m:
            rest = [q for q in range(m) if q not in order]
            dmin = [min(abs(x[q] - x[o]) for o in order) for q in rest]
            order.append(rest[int(np.argmax(dmin))])
        own = json.loads(g.own_ids.iloc[0])
        T = Y[np.ix_([pos[j] for j in own], Mi)].mean(axis=0) if own else None
        w = len(own) / (len(own) + K_SH) if own else 0.0
        name = {q: LANTHANIDES[Mi[q]] for q in range(m)}
        idx_of = {name[q]: q for q in range(m)}
        for n in (2, 4):
            rd = order[:n]
            lev = u + pwl(rd, yv - u, x)
            levS = (T + pwl(rd, yv - T, x)) if T is not None else None
            for ridx, row in g.iterrows():
                a, b = row.pair.split("/")
                qa, qb = idx_of[a], idx_of[b]
                d7 = lev[qa] - lev[qb]
                extra[f"pred_D7_{n}"][ridx] = d7
                if levS is not None:
                    extra[f"pred_SHown_{n}"][ridx] = w * (levS[qa] - levS[qb]) + (1 - w) * d7
    for k, v in extra.items():
        allc3[k] = v

    P = allc3[(allc3.n_own >= 1) & ~allc3.c5].reset_index(drop=True)
    y = P.y.to_numpy()
    cp = P.pred_SIB_own.to_numpy()
    big = np.abs(y) >= THR
    sel = np.abs(cp) >= THR
    allr = np.ones(len(P), bool)
    say(f"population own>=1 noC5: rows {len(P)}, cells {P.cell_id.nunique()}, pubs {P.publication.nunique()}, "
        f"clusters {P.cluster.nunique()}; |y|>=0.1 rows {big.sum()}; |COPY|>=0.1 rows {sel.sum()}")
    R["n_rows"] = len(P)

    # seed-averaged accuracy for B1 / G14
    def acc_seeded(prefix, rows):
        a = [acc(P[f"{prefix}_s{s}"].to_numpy(), y, rows)[0] for s in range(len(seeds))]
        return float(np.mean(a)), int(rows.sum())

    comps = {"COPY_own": "pred_SIB_own", "always_heavier": "pred_AH", "pairtype_majority_LOPO": "pred_PTMAJ",
             "chemotype_pairtype_majority_LOPO": "pred_CHMAJ", "D7_HULL_2": "pred_D7_2", "D7_HULL_4": "pred_D7_4",
             "SH_own_HULL_2": "pred_SHown_2", "SH_own_HULL_4": "pred_SHown_4", "COPY_xpub": "pred_SIB_xpub"}
    # ---------------- Q1: same-row comparison ----------------
    say("\n=== Q1 sign accuracy on identical rows (binary sign; ties/NaN excluded for that arm)")
    tab = []
    for lab, rows in (("all rows", allr), ("|y|>=0.1", big), ("|COPY|>=0.1, all rows", sel),
                      ("|COPY|>=0.1 & |y|>=0.1", sel & big), ("|COPY|<0.1, all rows", ~sel),
                      ("|COPY|<0.1 & |y|>=0.1", (~sel) & big)):
        rec = {"rows": lab, "n": int(rows.sum())}
        for k, c in comps.items():
            rec[k] = acc(P[c].to_numpy(), y, rows)[0]
            if k == "COPY_xpub":
                rec["n_xpub"] = acc(P[c].to_numpy(), y, rows)[1]
        rec["B1_4reads_DOPTQ"] = acc_seeded("pred_B1", rows)[0]
        rec["G14_zeroshot"] = acc_seeded("pred_G14", rows)[0]
        tab.append(rec)
    T1 = pd.DataFrame(tab)
    T1.to_csv(OUT / "Q1_same_rows.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        say(T1.round(3).to_string(index=False))
    R["Q1"] = tab

    # ---------------- Q2: coverage-matched selective comparators ----------------
    say("\n=== Q2 coverage-matched: each arm keeps its own top-|pred| rows at COPY's coverage")
    tab2 = []
    for lab, elig in (("all rows", allr), ("|y|>=0.1", big)):
        cov = float((sel & elig).sum() / elig.sum())
        rec = {"rows": lab, "coverage": cov}
        for k, c in [("COPY_own", "pred_SIB_own"), ("D7_HULL_2", "pred_D7_2"), ("D7_HULL_4", "pred_D7_4"),
                     ("SH_own_HULL_2", "pred_SHown_2"), ("SH_own_HULL_4", "pred_SHown_4")]:
            v = P[c].to_numpy()
            mag = np.where(elig & np.isfinite(v), np.abs(v), -1)
            thr = np.quantile(mag[elig & np.isfinite(v)], 1 - cov)
            rows = elig & np.isfinite(v) & (np.abs(v) >= thr)
            rec[k] = acc(v, y, rows)[0]
        bs = []
        for s in range(len(seeds)):
            v = P[f"pred_B1_s{s}"].to_numpy()
            thr = np.quantile(np.abs(v[elig]), 1 - cov)
            bs.append(acc(v, y, elig & (np.abs(v) >= thr))[0])
        rec["B1_4reads_DOPTQ"] = float(np.mean(bs))
        tab2.append(rec)
    T2 = pd.DataFrame(tab2)
    T2.to_csv(OUT / "Q2_coverage_matched.csv", index=False)
    say(T2.round(3).to_string(index=False))
    R["Q2"] = tab2

    # ---------------- Q3: three-way decision ----------------
    say("\n=== Q3 three-way (heavier y<=-0.1 / lighter y>=0.1 / inseparable) accuracy on all rows")

    def cls(v):
        return np.where(v >= THR, 1, np.where(v <= -THR, -1, 0))

    yc = cls(y)
    # majority three-way class per pair type, LOPO, no C5
    noc5c = noc5.assign(c=cls(noc5.y.to_numpy()))
    maj3 = []
    for _, r in P.iterrows():
        o = noc5c[(noc5c.pair == r.pair) & (noc5c.publication != r.publication)]
        maj3.append(o.c.value_counts().idxmax())
    maj3 = np.array(maj3)
    t3 = {"class shares (H/I/L)": [float((yc == -1).mean()), float((yc == 0).mean()), float((yc == 1).mean())],
          "COPY_own": float((cls(cp) == yc).mean()),
          "pairtype_modal_class_LOPO": float((maj3 == yc).mean()),
          "D7_HULL_2": float((cls(P.pred_D7_2.to_numpy()) == yc).mean()),
          "D7_HULL_4": float((cls(P.pred_D7_4.to_numpy()) == yc).mean()),
          "SH_own_HULL_2": float((cls(P.pred_SHown_2.to_numpy()) == yc).mean()),
          "SH_own_HULL_4": float((cls(P.pred_SHown_4.to_numpy()) == yc).mean()),
          "B1_4reads": float(np.mean([(cls(P[f'pred_B1_s{s}'].to_numpy()) == yc).mean() for s in range(5)])),
          "always_heavier_3way": float((yc == -1).mean())}
    # severe errors: called a direction and truth is the opposite separable direction
    for k, v in (("COPY_own", cp), ("D7_HULL_2", P.pred_D7_2.to_numpy()), ("SH_own_HULL_2", P.pred_SHown_2.to_numpy())):
        c = cls(v)
        t3[f"{k}_opposite_call_rate"] = float(((c * yc) == -1).mean())
    for k, v in t3.items():
        say(f"  {k}: {np.round(v, 3)}")
    R["Q3"] = t3

    # ---------------- Q4: publication-level, kill criterion replay, power ----------------
    say("\n=== Q4 publication-level accuracy at |COPY|>=0.1 (all rows), and the kill rule replayed")
    P["ok"] = (np.sign(cp) == np.sign(y)).astype(float)
    P["okAH"] = (y < 0).astype(float)
    S = P[sel]
    per = S.groupby("publication").agg(n=("ok", "size"), acc=("ok", "mean"), accAH=("okAH", "mean"),
                                       cluster=("cluster", "first")).sort_values("n", ascending=False)
    per.to_csv(OUT / "Q4_per_publication.csv")
    say(per.round(3).to_string())
    pooled = float(S.ok.mean())
    pubs = per.index.to_numpy()
    bs = []
    for _ in range(4000):
        dr = rng.choice(pubs, len(pubs), replace=True)
        sub = pd.concat([S[S.publication == q] for q in dr])
        bs.append(sub.ok.mean())
    eq = per.acc.to_numpy()
    from scipy.stats import t as tdist
    eqw_lb = float(eq.mean() - tdist.ppf(0.95, len(eq) - 1) * eq.std(ddof=1) / np.sqrt(len(eq)))
    big_cl = P.cluster.value_counts().idxmax()
    R["Q4"] = {"pooled_acc": pooled, "pooled_pubboot_5pct": float(np.percentile(bs, 5)),
               "pooled_pubboot_2.5pct": float(np.percentile(bs, 2.5)), "n_pubs": int(len(pubs)),
               "eqweight_mean": float(eq.mean()), "eqweight_t_lb95_1sided": eqw_lb,
               "AH_same_rows": float(S.okAH.mean()),
               "acc_outside_giant_cluster": float(S[S.cluster != big_cl].ok.mean()),
               "n_outside_giant_cluster": int((S.cluster != big_cl).sum()),
               "AH_outside_giant_cluster": float(S[S.cluster != big_cl].okAH.mean()),
               "giant_cluster_share_selected": float((S.cluster == big_cl).mean())}
    for k, v in R["Q4"].items():
        say(f"  {k}: {np.round(v, 3)}")
    # power at 11 families: resample 11 publications (n >= 5 selected rows) as stand-in families
    elig_p = per[per.n >= 5].index.to_numpy()
    surv_lb, surv_mean, surv_ah = 0, 0, 0
    NS = 4000
    for _ in range(NS):
        dr = rng.choice(elig_p, 11, replace=True)
        a = per.loc[dr, "acc"].to_numpy()
        ah = per.loc[dr, "accAH"].to_numpy()
        lb = a.mean() - 1.812 * a.std(ddof=1) / np.sqrt(11)  # t(0.95, 10 df)
        surv_lb += lb >= 0.90
        surv_mean += a.mean() >= 0.90
        surv_ah += (a.mean() - ah.mean()) >= 0.05
    R["Q4_power11"] = {"P(eqw family LB >= 0.90)": surv_lb / NS, "P(mean >= 0.90)": surv_mean / NS,
                       "P(mean - AH >= 0.05)": surv_ah / NS, "n_pubs_used": int(len(elig_p))}
    say(f"  power at 11 families (corpus publications as families, n>=5 rows): {R['Q4_power11']}")
    # per pair-type precision of the selected rows
    pt = S.groupby("pair").agg(n=("ok", "size"), acc=("ok", "mean"), accAH=("okAH", "mean"))
    pt["binom_se"] = np.sqrt(pt.acc * (1 - pt.acc) / pt.n)
    pt.to_csv(OUT / "Q4_per_pair_selected.csv")
    say("  per pair type at |COPY|>=0.1 (all rows):")
    say(pt.round(3).to_string())

    # ---------------- Q5: near-replicate sign agreement ----------------
    say("\n=== Q5 near-replicate sign agreement (same pub+extractant, differ only in contact time/metal conc/nothing)")
    c3ids = set(allc3.cell_id)
    rep = []
    for t in c3ids:
        if t in v2drop:
            continue
        i = pos[t]
        for j in np.where((pubA == pubA[i]) & (extA == extA[i]))[0]:
            if j == i or ids[j] in v2drop:
                continue
            dfm = set()
            for q, c in enumerate(cond_cols):
                a, b = F[i, q], F[j, q]
                if np.isnan(a) and np.isnan(b):
                    continue
                if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                    dfm.add(fam(c))
            if dfm - TRIVIAL:
                continue
            for p in range(N_LN - 1):
                if Z_VEC[p + 1] - Z_VEC[p] != 1:
                    continue
                yi = Y[i, p] - Y[i, p + 1]
                yj = Y[j, p] - Y[j, p + 1]
                if np.isfinite(yi) and np.isfinite(yj):
                    rep.append({"t": t, "j": ids[j], "pub": pubA[i], "pair": f"{LANTHANIDES[p]}/{LANTHANIDES[p + 1]}",
                                "yi": yi, "yj": yj, "dif": ",".join(sorted(dfm)) or "none"})
    rep = pd.DataFrame(rep)
    if len(rep):
        rb = rep[rep.yi.abs() >= THR]
        rs = rep[(rep.yi.abs() >= THR) & (rep.yj.abs() >= THR)]
        R["Q5"] = {"n_rows": int(len(rep)), "n_pubs": int(rep.pub.nunique()), "n_cells": int(rep.t.nunique()),
                   "agree_all": float((np.sign(rep.yi) == np.sign(rep.yj)).mean()),
                   "agree_|yi|>=0.1": float((np.sign(rb.yi) == np.sign(rb.yj)).mean()), "n_big": int(len(rb)),
                   "agree_both>=0.1": float((np.sign(rs.yi) == np.sign(rs.yj)).mean()), "n_both": int(len(rs)),
                   "sd_diff": float((rep.yi - rep.yj).std() / np.sqrt(2)),
                   "pubs": rep.pub.value_counts().to_dict(), "dif": rep.dif.value_counts().to_dict()}
        rep.to_csv(OUT / "Q5_near_replicates.csv", index=False)
    else:
        R["Q5"] = {"n_rows": 0}
    say(f"  {R['Q5']}")

    # ---------------- Q6: confident wrong calls, condition type ----------------
    say("\n=== Q6 confident wrong calls (|COPY|>=0.1, sign wrong)")
    W = P[sel & (P.ok == 0)][["cell_id", "publication", "cluster", "pair", "y", "pred_SIB_own", "own_fams"]]
    W = W.assign(sep=W.y.abs() >= THR)
    W.to_csv(OUT / "Q6_confident_wrong.csv", index=False)
    say(f"  n wrong {len(W)} of {int(sel.sum())}; separable wrong {int(W.sep.sum())}; by pub: "
        f"{W.publication.value_counts().to_dict()}; by pair: {W.pair.value_counts().to_dict()}")

    def ctype(s):
        fs = set()
        for e in json.loads(s):
            fs |= set(e.split(","))
        if fs == {"acid_conc"}:
            return "acid_conc_only"
        if "diluent" in fs:
            return "diluent_involved"
        if fs <= {"acid_conc", "acid_type"}:
            return "acid_type(+conc)"
        if fs <= {"extr_conc", "acid_conc"}:
            return "extr_conc(+acid)"
        return "mixed_other"
    P["ctype"] = P.own_fams.map(ctype)
    ct = []
    for k, g in P.groupby("ctype"):
        gb = g[g.y.abs() >= THR]
        gs = g[g.pred_SIB_own.abs() >= THR]
        ct.append({"ctype": k, "cells": g.cell_id.nunique(), "pubs": g.publication.nunique(), "rows": len(g),
                   "acc_|y|>=0.1": float(gb.ok.mean()) if len(gb) else np.nan,
                   "AH_|y|>=0.1": float(gb.okAH.mean()) if len(gb) else np.nan,
                   "acc_sel": float(gs.ok.mean()) if len(gs) else np.nan,
                   "AH_sel": float(gs.okAH.mean()) if len(gs) else np.nan, "cov_sel": len(gs) / len(g)})
    ct = pd.DataFrame(ct)
    ct.to_csv(OUT / "Q6_by_condition_type.csv", index=False)
    say(ct.round(3).to_string(index=False))
    R["Q6_ctype"] = ct.to_dict("records")

    # ---------------- Q7: what the triage saves ----------------
    say("\n=== Q7 metals needed to re-measure flagged pairs (|COPY|<0.1 or Er/Tm, Ho/Er, La/Ce)")
    sv = []
    for t, g in P.groupby("cell_id"):
        nm = len([m for m in range(N_LN) if np.isfinite(Y[pos[t], m])])
        fl = g[(g.pred_SIB_own.abs() < THR) | g.pair.isin(FLAG_PAIRS)]
        fl2 = g[(g.pred_SIB_own.abs() < THR)]
        mets = set()
        for pq in fl.pair:
            mets |= set(pq.split("/"))
        mets2 = set()
        for pq in fl2.pair:
            mets2 |= set(pq.split("/"))
        sv.append({"cell": t, "n_metals": nm, "n_pairs": len(g), "flag_pairs": len(fl), "flag_metals": len(mets),
                   "flag_metals_thr_only": len(mets2)})
    sv = pd.DataFrame(sv)
    sv.to_csv(OUT / "Q7_savings.csv", index=False)
    R["Q7"] = {"mean_metals": float(sv.n_metals.mean()), "mean_pairs": float(sv.n_pairs.mean()),
               "mean_flag_pairs": float(sv.flag_pairs.mean()), "mean_flag_metals": float(sv.flag_metals.mean()),
               "mean_flag_metals_thr_only": float(sv.flag_metals_thr_only.mean()),
               "share_cells_needing_>=2_metals": float((sv.flag_metals >= 2).mean()),
               "share_cells_needing_>=4_metals": float((sv.flag_metals >= 4).mean()),
               "share_cells_needing_all_but_<=2": float((sv.n_metals - sv.flag_metals <= 2).mean()),
               "metals_saved_fraction": float(1 - sv.flag_metals.sum() / sv.n_metals.sum())}
    say(f"  {R['Q7']}")
    # triage policy direction accuracy on |y|>=0.1 rows: re-measured pairs taken as right (optimistic),
    # vs SH_own@2 (2 metals, full coverage)
    tri = np.where(sel & ~P.pair.isin(FLAG_PAIRS).to_numpy(), P.ok.to_numpy(), 1.0)
    R["Q7_triage_dir_acc_|y|>=0.1_if_remeasured_perfect"] = float(tri[big].mean())
    say(f"  triage (re-measured pairs counted right) dir acc on |y|>=0.1: {tri[big].mean():.3f}; "
        f"SH_own@2 on same rows: {acc(P.pred_SHown_2.to_numpy(), y, big)[0]:.3f}")

    (OUT / "recipe3_skepB_summary.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    (OUT / "recipe3_skepB_signcheck.log").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
