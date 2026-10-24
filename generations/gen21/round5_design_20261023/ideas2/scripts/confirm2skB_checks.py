#!/usr/bin/env python
"""Skeptic B on confirm-2 (C-GD: heavy-selective non-DGA Gd chord deeper than B1's beta).  EXPLORATORY: every number
here comes from the 229 already-seen C3 cells and the round-4 OOF parquets.  Nothing is confirmed by it.

Checks
  U  unit table: which corpus units make up the heavy-selective (HS) non-DGA stratum (names, family, mechanism)
  M  is "heavy-selective" the mechanism?  chord vs read/full slope inside DGA and inside non-DGA (unit level)
  O  does a stratum-specific Gd amplitude help B1 out of fold on the corpus?  B1's prediction is B0 + beta * f with
     f = (HIST_D7 - B0) / -0.072 (HIST_D7 is B1's form with fixed beta); per (seed, fold) we refit, on the other
     folds' rows, (a) one pooled beta and (b) one beta per stratum (DGA / nonDGA-HS / nonDGA-LS, HS by the slope of
     the row's own read metals), and score the held-out fold.  Upper bound on what the stratification is worth.
  P  analytic power of the one-sample t (one-sided 0.025) for worlds the corpus bootstrap does not cover
Writes only ideas2/confirm2skB_out/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm2skB_out"
OUT.mkdir(exist_ok=True)
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r4 import slots_a as SA  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

BETA = -0.0923
CLIP = (-0.40, 0.20)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    LN = list(D.LANTHANIDES)
    XR = interp.X_R
    ig, ie, it = LN.index("Gd"), LN.index("Eu"), LN.index("Tb")
    t = (XR[ig] - XR[ie]) / (XR[it] - XR[ie])
    c = Y[:, ig] - ((1 - t) * Y[:, ie] + t * Y[:, it])
    sl = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            sl[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(ref.cell_id)
    mech = SA._mechanisms()
    from rdkit import Chem

    def ik(s):
        try:
            return Chem.MolToInchiKey(Chem.MolFromSmiles(s))
        except Exception:
            return ""
    comp = pd.read_csv(paths.ROOT / "generations/gen19_chem_transfer/descriptors/extractant_components.csv",
                       usecols=["inchikey", "mechanism", "family", "acidity_class", "canonical_name"])
    fam = dict(zip(comp.inchikey.astype(str), comp.family.astype(str)))
    acid = dict(zip(comp.inchikey.astype(str), comp.acidity_class.astype(str)))
    d = pd.DataFrame({"cell_id": f.cell_id, "c": c, "slope": sl, "hs": sl > 0, "c3": f.cell_id.isin(c3),
                      "dga": f.extractant.astype(str).map(DN.is_dga), "pub": f.publication_id,
                      "chemotype": f.chemotype.astype(str), "name": f.extractant_name.astype(str),
                      "ik": f.extractant.astype(str).map(ik),
                      "unit": f.publication_id.astype(str) + "|" + f.chemotype.astype(str)})
    d["mech"] = d.ik.map(mech).fillna("?")
    d["fam"] = d.ik.map(fam).fillna("?")
    d["acid"] = d.ik.map(acid).fillna("?")
    d = d[d.c3 & np.isfinite(d.c)].copy()
    u = d.groupby("unit").agg(c=("c", "mean"), slope=("slope", "mean"), hs=("hs", "mean"), dga=("dga", "mean"),
                              n=("c", "size"), name=("name", lambda x: ";".join(sorted(set(x)))[:38]),
                              fam=("fam", lambda x: ";".join(sorted(set(x)))), mech=("mech", lambda x: ";".join(sorted(set(x)))),
                              acid=("acid", lambda x: ";".join(sorted(set(x)))))
    u["cw"] = np.clip(u.c, *CLIP)
    u["cls"] = np.where(u.dga > 0.5, "DGA", np.where(u.hs > 0.5, "nonDGA_HS", "nonDGA_LS"))
    u.to_csv(OUT / "U_units.csv")
    say("== U: units by stratum (EXPLORATORY) ==")
    for cls in ("nonDGA_HS", "nonDGA_LS"):
        g = u[u.cls == cls].sort_values("cw")
        say(f"-- {cls}: {len(g)} units, {int(g.n.sum())} cells, mean_w {g.cw.mean():+.3f}, sd {g.cw.std(ddof=1):.3f}")
        say(g[["n", "cw", "slope", "hs", "name", "fam", "mech", "acid"]].round(3).to_string())
    say("-- mechanism tally in HS stratum:", u[u.cls == "nonDGA_HS"].mech.value_counts().to_dict())
    say("-- acidity tally in HS stratum:", u[u.cls == "nonDGA_HS"].acid.value_counts().to_dict())
    say("-- DGA units: HS share", (u[u.cls == "DGA"].hs > 0.5).mean().round(3), "of", (u.cls == "DGA").sum())
    # ---- M: is heavy-selectivity the mechanism?
    say("\n== M: chord vs slope (unit level, winsorised chord) ==")
    for name, g in (("DGA", u[u.cls == "DGA"]), ("nonDGA", u[u.cls != "DGA"]),
                    ("nonDGA SOFT_N", u[(u.cls != "DGA") & u.mech.str.contains("SOFT_N")]),
                    ("nonDGA not SOFT_N", u[(u.cls != "DGA") & ~u.mech.str.contains("SOFT_N")])):
        if len(g) >= 4:
            r = SS.spearmanr(g.slope, g.cw)
            say(f"{name:22s} n={len(g):3d} spearman(slope, chord)={r.statistic:+.3f} p={r.pvalue:.3f}  "
                f"HS mean {g[g.hs > 0.5].cw.mean():+.3f} (n={int((g.hs > 0.5).sum())})  "
                f"LS mean {g[g.hs <= 0.5].cw.mean():+.3f} (n={int((g.hs <= 0.5).sum())})")
    for mk, g in u[u.cls != "DGA"].groupby(u[u.cls != "DGA"].mech):
        say(f"  nonDGA mech={mk:28s} units {len(g):2d} mean_w {g.cw.mean():+.3f} HS share {(g.hs > 0.5).mean():.2f}")
    # ---- O: stratified amplitude out of fold
    say("\n== O: stratum-specific Gd amplitude, OOF on round-4 REF parquets (EXPLORATORY upper bound) ==")
    ycell = dict(zip(f.cell_id, Y))
    dga_cell = dict(zip(d.cell_id, d.dga))
    full_hs = dict(zip(d.cell_id, d.hs))
    for design in ("BPG", "BP"):
        r = pd.read_parquet(paths.PRED / f"REF_{design}.parquet")
        r = r[r.k == 3].copy()
        r["f"] = (r.HIST_D7 - r.B0) / -0.072
        rs_slope = {}
        for rsx in r.read_set.unique():
            rs_slope[rsx] = [LN.index(m) for m in rsx.split(",")]
        def rslope(row):
            yv = ycell[row.cell_id]
            idx = [i for i in rs_slope[row.read_set] if np.isfinite(yv[i])]
            if len(idx) < 2:
                return np.nan
            return np.polyfit(XR[idx], yv[idx], 1)[0]
        key = r[["cell_id", "read_set"]].drop_duplicates()
        key["rsl"] = key.apply(rslope, axis=1)
        r = r.merge(key, on=["cell_id", "read_set"], how="left")
        r["dga"] = r.cell_id.map(lambda x: bool(dga_cell.get(x, DN.is_dga(str(r.loc[r.cell_id == x, 'extractant'].iloc[0])))))
        r["cls"] = np.where(r.dga, "DGA", np.where(r.rsl > 0, "nonDGA_HS", "nonDGA_LS"))
        r["cls_full"] = np.where(r.dga, "DGA", np.where(r.cell_id.map(full_hs).fillna(False), "nonDGA_HS", "nonDGA_LS"))
        r["gdpair"] = r.A.isin(["Eu", "Gd"]) & r.B.isin(["Gd", "Tb"])
        out = []
        for (seed, fold), te in r.groupby(["split_seed", "fold"]):
            tr = r[(r.split_seed == seed) & (r.fold != fold)]
            z = tr.y - tr.B0
            bp = (tr.f * z).sum() / (tr.f ** 2).sum()
            pred_pool = te.B0 + bp * te.f
            pred_s = te.B0.copy()
            pred_sf = te.B0.copy()
            bs = {}
            for col, pr in (("cls", pred_s), ("cls_full", pred_sf)):
                for cl in ("DGA", "nonDGA_HS", "nonDGA_LS"):
                    q = tr[tr[col] == cl]
                    be = (q.f * (q.y - q.B0)).sum() / max((q.f ** 2).sum(), 1e-12)
                    bs[(col, cl)] = be
                    mm = te[col] == cl
                    pr[mm] = te.B0[mm] + be * te.f[mm]
            o = te[["split_seed", "fold", "cell_id", "publication", "cls", "cls_full", "gdpair", "y", "B0", "B1"]].copy()
            o["POOL"] = pred_pool
            o["STRAT"] = pred_s
            o["STRAT_FULL"] = pred_sf
            o["b_pool"] = bp
            o["b_hs"] = bs[("cls", "nonDGA_HS")]
            o["b_ls"] = bs[("cls", "nonDGA_LS")]
            o["b_dga"] = bs[("cls", "DGA")]
            out.append(o)
        o = pd.concat(out)
        o.to_parquet(OUT / f"O_rows_{design}.parquet")
        bb = o.groupby(["split_seed", "fold"])[["b_pool", "b_hs", "b_ls", "b_dga"]].first()
        say(f"-- {design}: fold betas mean (sd): pool {bb.b_pool.mean():+.3f} ({bb.b_pool.std():.3f})  "
            f"HS {bb.b_hs.mean():+.3f} ({bb.b_hs.std():.3f})  LS {bb.b_ls.mean():+.3f} ({bb.b_ls.std():.3f})  "
            f"DGA {bb.b_dga.mean():+.3f} ({bb.b_dga.std():.3f})")
        for scope, q in (("all rows", o), ("HS rows (reads)", o[o.cls == "nonDGA_HS"]),
                         ("HS rows (reads), Gd pairs", o[(o.cls == "nonDGA_HS") & o.gdpair]),
                         ("LS rows (reads)", o[o.cls == "nonDGA_LS"]), ("DGA rows", o[o.cls == "DGA"])):
            res = []
            for seed, g in q.groupby("split_seed"):
                sst = ((g.y - g.y.mean()) ** 2).sum()
                res.append({m: 1 - ((g.y - g[m]) ** 2).sum() / sst for m in ("B0", "B1", "POOL", "STRAT", "STRAT_FULL")}
                           | {f"sse_{m}": ((g.y - g[m]) ** 2).sum() for m in ("B1", "POOL", "STRAT", "STRAT_FULL")})
            rr = pd.DataFrame(res)
            say(f"   {scope:28s} rows/seed {len(q) // max(q.split_seed.nunique(), 1):5d}  R2 B0 {rr.B0.mean():.4f} "
                f"B1 {rr.B1.mean():.4f} POOL {rr.POOL.mean():.4f} STRAT {rr.STRAT.mean():.4f} STRAT_FULL {rr.STRAT_FULL.mean():.4f}"
                f"  relSSE STRAT/POOL {(rr.sse_STRAT / rr.sse_POOL).mean() - 1:+.3%}  STRAT/B1 {(rr.sse_STRAT / rr.sse_B1).mean() - 1:+.3%}")
        # per-publication win/loss for STRAT vs POOL on HS-read rows
        q = o[o.cls == "nonDGA_HS"]
        pp = q.assign(e0=(q.y - q.POOL) ** 2, e1=(q.y - q.STRAT) ** 2).groupby("publication")[["e0", "e1"]].sum()
        say(f"   HS-read rows by publication: STRAT better in {(pp.e1 < pp.e0 - 1e-12).sum()} / worse in "
            f"{(pp.e1 > pp.e0 + 1e-12).sum()} of {len(pp)}; top-2 publications' share of the SSE gain "
            f"{((pp.e0 - pp.e1).sort_values(ascending=False).head(2).sum() / max((pp.e0 - pp.e1).sum(), 1e-12)):.2f}")
        say("   per-publication gain (POOL-STRAT SSE):", (pp.e0 - pp.e1).sort_values().round(4).to_dict())
    # ---- P: analytic power of the one-sided t at alpha 0.025 for other worlds
    say("\n== P: noncentral-t power, H1 mean < beta, one-sided 0.025 ==")
    rows = []
    for mu in (-0.26, -0.217, -0.18, -0.15, -0.12):
        for sd in (0.06, 0.10, 0.15):
            for G in (6, 8, 10, 13):
                ncp = (BETA - mu) / (sd / np.sqrt(G))
                tc = SS.t.ppf(0.975, G - 1)
                pw = float(SS.nct.sf(tc, G - 1, ncp))      # T = (beta - xbar)/(s/sqrt G) > tc
                kill = float(SS.nct.cdf(0.0, G - 1, ncp))  # xbar >= beta  <=>  T <= 0
                rows.append({"mu": mu, "sd": sd, "G": G, "power": pw, "P_kill": kill})
    pw = pd.DataFrame(rows)
    pw.to_csv(OUT / "P_power.csv", index=False)
    say(pw.pivot_table(index=["mu", "sd"], columns="G", values="power").round(2).to_string())
    say("P(kill: point estimate >= beta):")
    say(pw.pivot_table(index=["mu", "sd"], columns="G", values="P_kill").round(3).to_string())
    (OUT / "confirm2skB_checks.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
