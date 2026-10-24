#!/usr/bin/env python
"""Skeptic A for idea recipe-3 ("sign-first triage of target pairs from an own prior series").

EXPLORATORY: every number here is computed on the 229 already-seen C3 cells of the V1 bench.  Nothing is a test.

Independent recomputation: the pools, COPY, D7 / SH HULL and every comparator are rebuilt here from
gen20k.data.load_bench() and the round-4 REF_BPG cell list; the recipe_* outputs are only read at the end
to compare numbers.  No existing file is modified.  Writes only under ideas2/recipe3_skA_out/.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe3_signfirst_skA.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4.metrics import add_clusters  # noqa: E402
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe3_skA_out")
REC = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe_out")
C5 = "pub_6a0ef8528c"
GD = LANTHANIDES.index("Gd")
K_SH = 1.78
X = interp.X_R
TRIV = {"contact_time", "metal_conc"}
RNG = np.random.default_rng(777)


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
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}.get(c, c)


def pwl(xr, yr, xq):
    """PWL interpolation with constant continuation (xr increasing)."""
    return np.interp(xq, xr, yr)


def hull_order(x):
    m = len(x)
    order = [0, m - 1]
    while len(order) < m:
        rest = [q for q in range(m) if q not in order]
        dmin = [min(abs(x[q] - x[o]) for o in order) for q in rest]
        order.append(rest[int(np.argmax(dmin))])
    return order


def boot_acc(df, unit, col_ok, n=2000, seed=11):
    units = df[unit].unique()
    g = df.groupby(unit)[col_ok].agg(["sum", "count"]).reindex(units)
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    r = np.random.default_rng(seed)
    W = r.multinomial(len(units), np.ones(len(units)) / len(units), size=n).astype(float)
    a = (W @ s) / np.maximum(W @ c, 1)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(np.percentile(a, 5.0))


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols].to_numpy(dtype=float)
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okp = np.array([c not in v2drop for c in ids])
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = sorted(par.cell_id.unique())
    clus = add_clusters(par[["cell_id", "chemotype", "publication"]].drop_duplicates(), bench.frame)
    cl_of = dict(zip(clus.cell_id, clus.cluster_cpg))
    P("C3 cells", len(c3), "bench cells", len(ids))

    def dfam(i, j):
        s = set()
        for q, c in enumerate(cond_cols):
            a, b = F[i, q], F[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                s.add(fam(c))
        return s

    rows = []       # one row per (target cell, strict-adjacent pair)
    sibrows = []    # one row per (target cell, own sibling, pair): single-sibling COPY
    nearcopy = []   # per (target, own sibling): series similarity
    for t in c3:
        i = pos[t]
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x = X[Mi]
        yv = Y[i, Mi]
        pa = np.array([p for p in range(m - 1) if Z_VEC[Mi[p + 1]] - Z_VEC[Mi[p]] == 1])
        pb = pa + 1
        if len(pa) == 0:
            continue
        ysf = yv[pa] - yv[pb]
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        cover[i] = False
        tv2 = t in v2drop
        same = np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
        own = [int(j) for j in same if (not tv2) and okp[j] and bool(dfam(i, j) - TRIV)]
        own_any = [int(j) for j in same]   # no V2 / no condition rule (sensitivity)
        pools = {"own": own,
                 "xext": [] if tv2 else np.where((pub == pub[i]) & (ext != ext[i]) & cover & okp)[0].tolist(),
                 "xpub": [] if tv2 else np.where((ext == ext[i]) & (pub != pub[i]) & cover & okp)[0].tolist(),
                 "chem": [] if tv2 else np.where((chem == chem[i]) & (pub != pub[i]) & cover & okp)[0].tolist()}
        copy = {}
        for k, js in pools.items():
            if js:
                T = Y[np.ix_(js, Mi)].mean(axis=0)
                copy[k] = T[pa] - T[pb]
        # D7 and SH_own under HULL reads
        u = np.zeros(m)
        if GD in Mi:
            u[Mi.index(GD)] = HIST_D7_BETA
        order = hull_order(x)
        d7 = {}
        sh = {}
        for n in (2, 3, 4, 5):
            if n > m:
                continue
            S = sorted(order[:n])
            lev = pwl(x[S], (yv - u)[S], x) + u
            d7[n] = lev[pa] - lev[pb]
            if own:
                T = Y[np.ix_(own, Mi)].mean(axis=0)
                sib = T + pwl(x[S], (yv - T)[S], x)
                w = len(own) / (len(own) + K_SH)
                levs = w * sib + (1 - w) * lev
                sh[n] = levs[pa] - levs[pb]
        meas = {n: np.isin(pa, order[:n]) & np.isin(pb, order[:n]) for n in d7}
        for q in range(len(pa)):
            r = {"cell_id": t, "pub": pub[i], "ext": ext[i], "chem": chem[i], "cluster": cl_of[t],
                 "c5": pub[i] == C5, "v2": tv2,
                 "pair": f"{LANTHANIDES[Mi[pa[q]]]}/{LANTHANIDES[Mi[pb[q]]]}", "y": ysf[q],
                 "n_own": len(own), "n_own_any": len(own_any),
                 **{f"n_{k}": len(v) for k, v in pools.items() if k != "own"}}
            for k, v in copy.items():
                r[f"copy_{k}"] = v[q]
            for n, v in d7.items():
                r[f"d7_{n}"] = v[q]
                r[f"meas_{n}"] = bool(meas[n][q])
            for n, v in sh.items():
                r[f"sh_{n}"] = v[q]
            rows.append(r)
        # single-sibling COPY and near-copy diagnostics
        for j in own:
            sj = Y[j, Mi][pa] - Y[j, Mi][pb]
            fams = sorted(dfam(i, j) - TRIV)
            dl = Y[j, Mi] - yv
            nearcopy.append({"cell_id": t, "sib": ids[j], "pub": pub[i], "fams": ",".join(fams),
                             "max_abs_dSF": float(np.max(np.abs(sj - ysf))),
                             "sd_level_shift": float(np.std(dl)), "mean_level_shift": float(np.mean(dl)),
                             "identical_logD": bool(np.allclose(dl, 0, atol=1e-9))})
            for q in range(len(pa)):
                sibrows.append({"cell_id": t, "sib": ids[j], "pub": pub[i], "cluster": cl_of[t],
                                "c5": pub[i] == C5, "fams": ",".join(fams), "n_own": len(own),
                                "pair": f"{LANTHANIDES[Mi[pa[q]]]}/{LANTHANIDES[Mi[pb[q]]]}",
                                "y": ysf[q], "copy1": sj[q]})
    R = pd.DataFrame(rows)
    S1 = pd.DataFrame(sibrows)
    NC = pd.DataFrame(nearcopy)
    R.to_parquet(OUT / "skA_rows.parquet", index=False)
    S1.to_parquet(OUT / "skA_sibrows.parquet", index=False)
    NC.to_csv(OUT / "skA_nearcopy.csv", index=False)
    res = {}

    # ===== 1. reproduction of the headline numbers =====
    P("\n=== 1. REPRODUCTION (own pool, no C5) ===")
    own = R[(R.n_own >= 1) & ~R.c5].copy()
    big = own.y.abs() >= 0.1
    own["ok"] = (np.sign(own.copy_own) == np.sign(own.y)).astype(float)
    P("own rows", len(own), "cells", own.cell_id.nunique(), "pubs", own.pub.nunique(), "clusters", own.cluster.nunique(),
      "|y|>=0.1 rows", int(big.sum()))
    P("COPY sign acc |y|>=0.1:", round(own.ok[big].mean(), 4), "pub-boot 95%", boot_acc(own[big], "pub", "ok")[:2])
    for thr in (0.05, 0.1, 0.2):
        sel = own.copy_own.abs() >= thr
        P(f" thr {thr}: |y|>=0.1 acc {own.ok[sel & big].mean():.4f} cov {(sel & big).sum() / big.sum():.3f} n {(sel & big).sum()}"
          f" | all rows acc {own.ok[sel].mean():.4f} cov {sel.mean():.3f} n {sel.sum()}")
    res["repro"] = {"acc_big": float(own.ok[big].mean()), "n_big": int(big.sum()),
                    "acc_thr_big": float(own.ok[(own.copy_own.abs() >= 0.1) & big].mean()),
                    "acc_thr_all": float(own.ok[own.copy_own.abs() >= 0.1].mean())}
    # compare with recipe_signrule.csv
    try:
        rs = pd.read_csv(REC / "recipe_signrule.csv")
        P("recipe_signrule own rows (for comparison):\n", rs[rs.pool == "own"].round(4).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        P("could not read recipe_signrule.csv", e)
    # other pools like-for-like
    P("\nlike-for-like other pools (no C5), thr 0.1: acc on |y|>=0.1 rows / on all rows")
    for k in ("own", "xext", "xpub", "chem"):
        q = R[(R[f"n_{k}"] >= 1) & ~R.c5] if k != "own" else own
        okk = np.sign(q[f"copy_{k}"]) == np.sign(q.y)
        bg = q.y.abs() >= 0.1
        sl = q[f"copy_{k}"].abs() >= 0.1
        P(f"  {k}: overall|y|>=.1 {okk[bg].mean():.3f}  thr|y|>=.1 {okk[bg & sl].mean():.3f} (cov {(bg & sl).sum() / bg.sum():.2f})"
          f"  thr all {okk[sl].mean():.3f} (cov {sl.mean():.2f}) pubs {q.pub.nunique()}")

    # ===== 2. comparators ON THE SAME covered rows =====
    P("\n=== 2. COMPARATORS on the identical rows selected by |COPY_own| >= 0.1 ===")
    sel = own.copy_own.abs() >= 0.1
    pair_major = {}
    # pair-type majority, leave-one-publication-out, from all non-C5 C3 rows (no own-pool restriction)
    allr = R[~R.c5]
    for p in own.pub.unique():
        other = allr[(allr.pub != p) & (allr.y.abs() >= 0.1)]
        pair_major[p] = other.groupby("pair").y.apply(lambda v: np.sign(np.sign(v).sum()) or -1.0).to_dict()
    own["pairmaj"] = [pair_major[p].get(pr, -1.0) for p, pr in zip(own.pub, own.pair)]
    # chemotype x pair majority, LOPO
    cm = {}
    for p in own.pub.unique():
        other = allr[(allr.pub != p) & (allr.y.abs() >= 0.1)]
        cm[p] = other.groupby(["chem", "pair"]).y.apply(lambda v: np.sign(np.sign(v).sum()) or -1.0).to_dict()
    own["chempairmaj"] = [cm[p].get((c, pr), pair_major[p].get(pr, -1.0)) for p, c, pr in zip(own.pub, own.chem, own.pair)]
    comps = {"COPY_own (0 exp)": own.copy_own, "always-heavier": pd.Series(-1.0, index=own.index),
             "pair-type majority LOPO": own.pairmaj, "chemotype x pair majority LOPO": own.chempairmaj}
    for n in (2, 3, 4):
        comps[f"D7 HULL@{n}"] = own[f"d7_{n}"]
        comps[f"SH_own HULL@{n}"] = own[f"sh_{n}"]
    tab = []
    for lab_rows, msk in (("|y|>=0.1 & |COPY|>=0.1", big & sel), ("all & |COPY|>=0.1", sel),
                          ("|y|>=0.1 (all coverage)", big), ("|y|>=0.1 & |COPY|<0.1 (uncovered)", big & ~sel)):
        for name, pred in comps.items():
            ok = (np.sign(pred[msk]) == np.sign(own.y[msk])).astype(float)
            d = own[msk].assign(okc=ok)
            lo, hi, lo1 = boot_acc(d, "pub", "okc")
            tab.append({"rows": lab_rows, "arm": name, "n": int(msk.sum()), "acc": float(ok.mean()), "lo95": lo,
                        "hi95": hi, "lo_1sided95": lo1})
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "skA_comparators.csv", index=False)
    P(tab.round(3).to_string(index=False))
    res["comparators"] = tab.to_dict("records")
    # paired difference COPY - D7@2 etc on covered |y|>=.1 rows with pub bootstrap
    m = big & sel
    for name in ("always-heavier", "D7 HULL@2", "D7 HULL@4", "pair-type majority LOPO"):
        a = (np.sign(own.copy_own[m]) == np.sign(own.y[m])).astype(float)
        b = (np.sign(comps[name][m]) == np.sign(own.y[m])).astype(float)
        d = own[m].assign(dd=a - b, one=1.0)
        units = d.pub.unique()
        g = d.groupby("pub").agg(s=("dd", "sum"), c=("one", "sum")).reindex(units)
        W = np.random.default_rng(5).multinomial(len(units), np.ones(len(units)) / len(units), 2000)
        bd = (W @ g.s.to_numpy()) / (W @ g.c.to_numpy())
        P(f"  COPY - {name} on covered |y|>=.1 rows: {float((a - b).mean()):+.3f} [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")
    # cheap sensible alternative also needs 0 experiments: D7@2 needs 2.  Report also the |y|>=.1 base rate of
    # heavy-selective on covered rows
    P("  share of covered |y|>=.1 rows that are heavy-selective (y<0):", round(float((own.y[m] < 0).mean()), 3))

    # ===== 3. publication / cluster dependence =====
    P("\n=== 3. PUBLICATION / CLUSTER DEPENDENCE (own, no C5, |COPY|>=0.1) ===")
    for lab_rows, msk in (("|y|>=.1", big & sel), ("all", sel)):
        d = own[msk]
        g = d.groupby("pub").agg(n=("ok", "size"), acc=("ok", "mean"), cluster=("cluster", "first"),
                                 ext=("ext", "nunique"), cells=("cell_id", "nunique")).sort_values("n", ascending=False)
        g["share"] = g.n / g.n.sum()
        P(f"[{lab_rows}] per publication:\n" + g.round(3).to_string())
        P(f"  top-1 share {g.share.iloc[0]:.2f}, top-3 share {g.share.iloc[:3].sum():.2f}; equal-weight per-pub acc "
          f"{g.acc.mean():.3f}; min per-pub acc {g.acc.min():.3f}; pubs with acc<0.9: {int((g.acc < 0.9).sum())} of {len(g)}")
        lopo = {p: float(d[d.pub != p].ok.mean()) for p in g.index}
        P(f"  LOPO acc range [{min(lopo.values()):.3f}, {max(lopo.values()):.3f}]")
        top3 = list(g.index[:3])
        P(f"  without top-3 pubs: acc {d[~d.pub.isin(top3)].ok.mean():.3f} n {int((~d.pub.isin(top3)).sum())}")
        gc = d.groupby("cluster").agg(n=("ok", "size"), acc=("ok", "mean"))
        gc["share"] = gc.n / gc.n.sum()
        P("  per cluster:\n" + gc.round(3).to_string())
        bigc = gc.n.idxmax()
        P(f"  without dominant cluster {bigc}: acc {d[d.cluster != bigc].ok.mean():.3f} n {int((d.cluster != bigc).sum())}")
        # family = publication x extractant
        d = d.assign(family=d.pub + "|" + d.ext)
        gf = d.groupby("family").agg(n=("ok", "size"), acc=("ok", "mean"))
        lo, hi, lo1 = boot_acc(d, "family", "ok")
        accs = gf.acc.to_numpy()
        tlo = accs.mean() - 1.753 * accs.std(ddof=1) / np.sqrt(len(accs)) if len(accs) > 2 else np.nan
        P(f"  families {len(gf)}; pooled acc {d.ok.mean():.3f}; family-bootstrap one-sided 95% lower {lo1:.3f}; "
          f"equal-weight family mean {accs.mean():.3f}, t one-sided 95% lower {tlo:.3f}")
        res[f"pubdep_{lab_rows}"] = {"top3_share": float(g.share.iloc[:3].sum()), "eq_pub": float(g.acc.mean()),
                                     "fam_lo1": lo1, "fam_t_lo": float(tlo), "n_fam": int(len(gf))}

    # ===== 4. near-copies / bit-identical SF series =====
    P("\n=== 4. NEAR-COPIES among own siblings (no C5 targets) ===")
    nc = NC[NC.pub != C5]
    P("target-sibling links", len(nc), "targets", nc.cell_id.nunique())
    P("  identical logD series:", int(nc.identical_logD.sum()))
    for tol in (1e-9, 0.005, 0.02, 0.05):
        P(f"  links with max|dSF| <= {tol}: {int((nc.max_abs_dSF <= tol).sum())} ({(nc.max_abs_dSF <= tol).mean():.3f})")
    sfsame = nc[nc.max_abs_dSF <= 1e-6]
    if len(sfsame):
        P("  SF-identical links by condition family and pub:\n" + sfsame.groupby(["pub", "fams"]).size().to_string())
    # rows where COPY_own == y (to 1e-6)
    eq = (own.copy_own - own.y).abs() <= 1e-6
    P(f"  own rows with COPY == y (1e-6): {int(eq.sum())} of {len(own)}; among covered |y|>=.1 rows: "
      f"{int((eq & big & sel).sum())} of {int((big & sel).sum())}")
    # accuracy excluding targets with any SF-identical (<=0.005) sibling
    near_t = set(nc[nc.max_abs_dSF <= 0.005].cell_id)
    d = own[big & sel & ~own.cell_id.isin(near_t)]
    P(f"  covered |y|>=.1 acc without targets having a near-identical (<=0.005) sibling: {d.ok.mean():.3f} n {len(d)} "
      f"(removed {len(near_t)} targets)")
    res["nearcopy"] = {"links": int(len(nc)), "sf_identical_1e9": int((nc.max_abs_dSF <= 1e-9).sum()),
                       "le_0.005": int((nc.max_abs_dSF <= 0.005).sum())}

    # ===== 5. single prior series (what a chemist actually has) & condition-change strata =====
    P("\n=== 5. SINGLE-SIBLING COPY (each own sibling separately; no C5) ===")
    s1 = S1[~S1.c5].copy()
    s1["ok"] = (np.sign(s1.copy1) == np.sign(s1.y)).astype(float)
    s1["big"] = s1.y.abs() >= 0.1
    s1["sel"] = s1.copy1.abs() >= 0.1
    # weight each target-pair equally across its siblings
    s1["w"] = 1.0 / s1.groupby(["cell_id", "pair"]).sib.transform("size")
    for lab_rows, msk in (("|y|>=.1 all cov", s1.big), ("|y|>=.1 & |COPY1|>=.1", s1.big & s1.sel), ("all & |COPY1|>=.1", s1.sel)):
        d = s1[msk]
        P(f"  [{lab_rows}] target-weighted acc {np.average(d.ok, weights=d.w):.3f}  (links {len(d)}, targets "
          f"{d.cell_id.nunique()}, pubs {d.pub.nunique()})")
    P(f"  coverage among |y|>=.1 (target-weighted): {np.average(s1.sel[s1.big], weights=s1.w[s1.big]):.3f}")
    # worst single sibling per target (pessimistic: the chemist's one prior series is the least similar one)
    d = s1[s1.big & s1.sel]
    P(f"  min over siblings of per-(target,pair) correctness, covered rows: "
      f"{d.groupby(['cell_id', 'pair']).ok.min().mean():.3f}")
    # strata by condition change family
    s1["fclass"] = np.where(s1.fams == "acid_conc", "acid_conc only",
                            np.where(s1.fams == "extr_conc", "extr_conc only",
                                     np.where(s1.fams.str.contains("diluent"), "diluent (any)",
                                              np.where(s1.fams.str.contains("acid"), "acid + other", "other"))))
    st = []
    for fc, d in s1.groupby("fclass"):
        for lab_rows, msk in (("|y|>=.1", d.big), ("|y|>=.1&sel", d.big & d.sel), ("all&sel", d.sel)):
            dd = d[msk]
            if len(dd) == 0:
                continue
            st.append({"class": fc, "rows": lab_rows, "links": len(dd), "targets": dd.cell_id.nunique(),
                       "pubs": dd.pub.nunique(), "acc": float(np.average(dd.ok, weights=dd.w)),
                       "cov_big": float(np.average(d.sel[d.big], weights=d.w[d.big])) if d.big.any() else np.nan})
    st = pd.DataFrame(st)
    st.to_csv(OUT / "skA_condclass.csv", index=False)
    P(st.round(3).to_string(index=False))
    # nearest vs farthest sibling (by number of differing families then by |dacid|)
    res["single_sib"] = st.to_dict("records")

    # ===== 6. C5 dependence =====
    P("\n=== 6. C5-BTBP dependence ===")
    allown = R[R.n_own >= 1].copy()
    allown["ok"] = (np.sign(allown.copy_own) == np.sign(allown.y)).astype(float)
    b2, s2 = allown.y.abs() >= 0.1, allown.copy_own.abs() >= 0.1
    P(f"  with C5: |y|>=.1 acc {allown.ok[b2].mean():.3f} n {int(b2.sum())}; covered {allown.ok[b2 & s2].mean():.3f}; "
      f"all covered {allown.ok[s2].mean():.3f}; C5 rows {int(allown.c5.sum())}")
    c5 = allown[allown.c5]
    if len(c5):
        P(f"  C5 alone: |y|>=.1 acc {c5.ok[c5.y.abs() >= .1].mean():.3f} n {int((c5.y.abs() >= .1).sum())}; covered "
          f"{c5.ok[(c5.y.abs() >= .1) & (c5.copy_own.abs() >= .1)].mean():.3f}")

    # ===== 7. threshold & eligibility robustness (post-hoc 0.1) =====
    P("\n=== 7. THRESHOLD ROBUSTNESS (own, no C5) ===")
    thr_rows = []
    for ey in (0.05, 0.1, 0.15, 0.2):
        for thr in (0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2):
            bg = own.y.abs() >= ey
            sl = own.copy_own.abs() >= thr
            thr_rows.append({"elig_|y|": ey, "thr": thr, "acc": float(own.ok[bg & sl].mean()),
                             "cov": float((bg & sl).sum() / bg.sum()), "n": int((bg & sl).sum()),
                             "acc_allrows": float(own.ok[sl].mean()), "cov_allrows": float(sl.mean()),
                             "always_heavier_same_rows": float((own.y[bg & sl] < 0).mean())})
    thr_rows = pd.DataFrame(thr_rows)
    thr_rows.to_csv(OUT / "skA_threshold.csv", index=False)
    P(thr_rows.round(3).to_string(index=False))

    # ===== 8. three-way decision (lighter / heavier / inseparable at |y| < 0.1) =====
    P("\n=== 8. THREE-WAY DECISION (own rows, no C5) ===")

    def cls(v, t=0.1):
        return np.where(v >= t, "L", np.where(v <= -t, "H", "I"))
    truth = cls(own.y.to_numpy())
    tw = []
    for name, pred in [("COPY_own", own.copy_own), ("D7 HULL@2", own.d7_2), ("D7 HULL@4", own.d7_4),
                       ("SH_own HULL@2", own.sh_2), ("SH_own HULL@4", own.sh_4)]:
        pc = cls(pred.to_numpy())
        acc = float((pc == truth).mean())
        # among pairs called separable, wrong direction (L<->H) rate; among truly separable, called I rate
        called = pc != "I"
        flip = float(((pc == "L") & (truth == "H") | (pc == "H") & (truth == "L")).sum() / max(called.sum(), 1))
        miss = float(((pc == "I") & (truth != "I")).sum() / max((truth != "I").sum(), 1))
        false_sep = float(((pc != "I") & (truth == "I")).sum() / max(called.sum(), 1))
        tw.append({"arm": name, "acc3": acc, "called_sep": int(called.sum()), "dir_flip_among_called": flip,
                   "called_sep_but_truly_I": false_sep, "truly_sep_called_I": miss})
    tw.append({"arm": "always H", "acc3": float((truth == "H").mean())})
    tw.append({"arm": "always I", "acc3": float((truth == "I").mean())})
    tw = pd.DataFrame(tw)
    tw.to_csv(OUT / "skA_threeway.csv", index=False)
    P(tw.round(3).to_string(index=False))

    # ===== 9. per pair at |COPY| >= 0.1 =====
    P("\n=== 9. PER PAIR (own, no C5): overall |y|>=.1 and covered ===")
    pp = []
    for pr, d in own.groupby("pair"):
        b, s = d.y.abs() >= 0.1, d.copy_own.abs() >= 0.1
        pp.append({"pair": pr, "n_big": int(b.sum()), "acc_big": float(d.ok[b].mean()) if b.any() else np.nan,
                   "n_cov": int((b & s).sum()), "acc_cov": float(d.ok[b & s].mean()) if (b & s).any() else np.nan,
                   "n_all_cov": int(s.sum()), "acc_all_cov": float(d.ok[s].mean()) if s.any() else np.nan,
                   "always_heavier_cov": float((d.y[b & s] < 0).mean()) if (b & s).any() else np.nan,
                   "pubs_big": int(d[b].pub.nunique()), "share_top_pub_big": float(d[b].pub.value_counts(normalize=True).iloc[0]) if b.any() else np.nan})
    pp = pd.DataFrame(pp)
    pp.to_csv(OUT / "skA_perpair.csv", index=False)
    P(pp.round(3).to_string(index=False))

    # ===== 10. literature leg (xpub) dependence =====
    P("\n=== 10. LITERATURE (xpub) leg dependence (no C5) ===")
    xp = R[(R.n_xpub >= 1) & ~R.c5].copy()
    xp["ok"] = (np.sign(xp.copy_xpub) == np.sign(xp.y)).astype(float)
    bx, sx = xp.y.abs() >= 0.1, xp.copy_xpub.abs() >= 0.1
    d = xp[bx & sx]
    g = d.groupby("pub").agg(n=("ok", "size"), acc=("ok", "mean")).sort_values("n", ascending=False)
    g["share"] = g.n / g.n.sum()
    P(g.round(3).to_string())
    P(f"  pooled {d.ok.mean():.3f}; equal-weight per pub {g.acc.mean():.3f}; extractants {d.ext.nunique()}: "
      f"{d.ext.value_counts().to_dict()}")
    P(f"  always-heavier on same rows {(d.y < 0).mean():.3f}; D7@2 on same rows "
      f"{(np.sign(d.d7_2) == np.sign(d.y)).mean():.3f}")
    gc = d.groupby("cluster").ok.agg(["size", "mean"])
    P("  clusters:\n" + gc.round(3).to_string())

    (OUT / "skA_summary.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "skA_log.txt").write_text("\n".join(log), encoding="utf-8")
    P(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
