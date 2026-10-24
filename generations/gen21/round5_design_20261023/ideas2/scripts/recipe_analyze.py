#!/usr/bin/env python
"""Theme "minimal-measurement recipes", analysis step (reads ideas2/recipe_out from recipe_build.py).
EXPLORATORY: 229 already-seen C3 cells; every population cut below (C5-BTBP out, n_own >= 1, ...) was known from
earlier probes.  Intervals: publication-level percentile bootstrap (2,000 draws, rng 20261023) and leave-one-
publication-out (LOPO) range; with 2-6 source clusters per population these are descriptive, not inference.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_analyze.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe_out")
NMAX = 6
B = 2000
RES = {}


def boot_counts(n_units: int, seed: int = 20261023) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=B).astype(float)


class Pop:
    """Per-population aggregation by publication, so R2 / differences bootstrap by publication."""

    def __init__(self, cells: pd.DataFrame, name: str):
        self.c = cells.reset_index(drop=True)
        self.name = name
        self.pubs = sorted(self.c.publication.unique())
        self.pi = self.c.publication.map({p: q for q, p in enumerate(self.pubs)}).to_numpy()
        self.G = np.zeros((len(self.c), len(self.pubs)))
        self.G[np.arange(len(self.c)), self.pi] = 1.0
        self.W = boot_counts(len(self.pubs))
        self.N = self.c.N.to_numpy(float) @ self.G
        self.sy = self.c.sy.to_numpy(float) @ self.G
        self.syy = self.c.syy.to_numpy(float) @ self.G

    def r2(self, sse_cell: np.ndarray, w: np.ndarray | None = None):
        s = sse_cell @ self.G
        if w is None:
            return 1 - s.sum() / (self.syy.sum() - self.sy.sum() ** 2 / self.N.sum())
        return 1 - (w @ s) / (w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N))

    def summ(self, sse_cell, ref=None) -> dict:
        p = self.r2(sse_cell)
        bs = self.r2(sse_cell, self.W)
        lopo = []
        for q in range(len(self.pubs)):
            w = np.ones(len(self.pubs))
            w[q] = 0
            lopo.append(float(self.r2(sse_cell, w[None, :])[0]))
        d = {"R2": float(p), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
             "lopo_min": min(lopo), "lopo_max": max(lopo)}
        if ref is not None:
            dd = bs - self.r2(ref, self.W)
            d.update({"d": float(p - self.r2(ref)), "d_lo": float(np.percentile(dd, 2.5)),
                      "d_hi": float(np.percentile(dd, 97.5)), "P_d_gt0": float((dd > 0).mean())})
        return d


def fmt(d: dict, diff=False) -> str:
    s = f"{d['R2']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    if diff and "d" in d:
        s += f" (d {d['d']:+.3f} [{d['d_lo']:+.3f}, {d['d_hi']:+.3f}])"
    return s


def main() -> int:
    cells = pd.read_parquet(OUT / "recipe_cells.parquet")
    pairs = pd.read_parquet(OUT / "recipe_pairs.parquet")
    npz = {n: dict(np.load(OUT / f"recipe_k{n}.npz")) for n in range(1, NMAX + 1)}
    nc = len(cells)
    combos = {n: list(itertools.combinations(range(N_LN), n)) for n in range(1, NMAX + 1)}

    # ---------------- 0. reproduction of round 4's HIST_D7 on DOPT_Q 4-read sets (BPG) ----------------
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    gid4 = {c: q for q, c in enumerate(combos[4])}
    ci_of = dict(zip(cells.cell_id, cells.ci))
    rep = []
    for seed, g in par.groupby("split_seed"):
        rs = g.groupby("cell_id").read_set.first()
        sse = 0.0
        for cid, r in rs.items():
            q = gid4[tuple(sorted(LANTHANIDES.index(s) for s in r.split(",")))]
            sse += float(npz[4]["sse_D7"][ci_of[cid], q])
        y = g.y.to_numpy()
        sse_par = float(((g.HIST_D7 - g.y) ** 2).sum())
        rep.append({"seed": int(seed), "R2_mine": 1 - sse / ((y - y.mean()) ** 2).sum(),
                    "R2_par": 1 - sse_par / ((y - y.mean()) ** 2).sum(), "sse_diff": sse - sse_par})
    rep = pd.DataFrame(rep)
    print("REPRO HIST_D7 @ DOPT_Q 4 reads (BPG):", rep.round(5).to_dict("records"))
    print("  mean", rep.R2_mine.mean().round(4), rep.R2_par.mean().round(4))
    RES["repro"] = rep.to_dict("records")

    # ---------------- per-cell SSE per (arm, n_read, policy) ----------------
    def cell_sse(arm: str, n: int, pol: str) -> np.ndarray:
        """RND = exact expectation over all n-subsets; HULL = lightest+heaviest then max-min; ORC = per-cell best
        (hindsight).  Cells with fewer metals than n are fully measured (SSE 0)."""
        if n == 0:
            col = f"rnd_{arm}"
            s = pairs[pairs.n_read == 0].groupby("ci")[col].sum()
            return s.reindex(range(nc)).to_numpy()
        if pol in ("RND", "HULL"):
            col = ("rnd_" if pol == "RND" else "hull_") + arm
            s = pairs[pairs.n_read == n].groupby("ci")[col].sum()
            return s.reindex(range(nc)).fillna(0.0).to_numpy()
        a = npz[n][f"sse_{arm}"]
        out = np.nanmin(np.where(np.isnan(a), np.inf, a), axis=1)
        out[~np.isfinite(out)] = 0.0
        return out

    def lopo_static(arm: str, n: int, sub: pd.DataFrame, cov=0.5):
        """Static n-metal recipe chosen on the other publications (rank sets by pooled SSE/SST over training cells
        that contain the set; coverage >= cov of training cells), applied to the held-out publication: each cell
        takes the best-ranked set it contains; fallback HULL.  Returns per-cell SSE and the chosen top set per pub."""
        a = npz[n][f"sse_{arm}"][sub.ci.to_numpy()]
        avail = ~np.isnan(a)
        a0 = np.nan_to_num(a)
        N, sy, syy = (sub[k].to_numpy(float) for k in ("N", "sy", "syy"))
        hull = cell_sse(arm, n, "HULL")[sub.ci.to_numpy()]
        out = np.zeros(len(sub))
        tops = {}
        for p in sub.publication.unique():
            tr = (sub.publication != p).to_numpy()
            te = ~tr
            ybar = sy[tr].sum() / N[tr].sum()
            sstc = syy - 2 * ybar * sy + N * ybar ** 2
            num = (a0[tr]).sum(0)
            den = (avail[tr] * sstc[tr, None]).sum(0)
            covg = avail[tr].mean(0)
            score = np.where((covg >= cov) & (den > 0), num / np.where(den > 0, den, 1), np.inf)
            rank = np.argsort(score)
            rank = rank[np.isfinite(score[rank])]
            tops[p] = ",".join(LANTHANIDES[q] for q in combos[n][rank[0]]) if len(rank) else ""
            for r in np.where(te)[0]:
                ok = rank[avail[r, rank]]
                out[r] = a0[r, ok[0]] if len(ok) else hull[r]
        return out, tops

    def insample_static(arm: str, n: int, sub: pd.DataFrame, top=5, cov=0.5):
        a = npz[n][f"sse_{arm}"][sub.ci.to_numpy()]
        avail = ~np.isnan(a)
        N, sy, syy = (sub[k].to_numpy(float) for k in ("N", "sy", "syy"))
        ybar = sy.sum() / N.sum()
        sstc = syy - 2 * ybar * sy + N * ybar ** 2
        num = np.nan_to_num(a).sum(0)
        den = (avail * sstc[:, None]).sum(0)
        covg = avail.mean(0)
        sc = np.where((covg >= cov) & (den > 0), 1 - num / np.where(den > 0, den, 1), -np.inf)
        r = np.argsort(-sc)[:top]
        return [(",".join(LANTHANIDES[q] for q in combos[n][j]), round(float(sc[j]), 3), round(float(covg[j]), 2))
                for j in r if np.isfinite(sc[j])]

    # ---------------- populations ----------------
    c = cells
    pops = {
        "ALL229": c,
        "noC5_214": c[~c.c5],
        "OWN1_noC5": c[(c.n_own >= 1) & ~c.c5],
        "OWN1_C5": c[(c.n_own >= 1)],
        "OWN2_noC5": c[(c.n_own >= 2) & ~c.c5],
        "OWN_eq1_noC5": c[(c.n_own == 1) & ~c.c5],
        "XEXT_noC5": c[(c.n_xext >= 1) & ~c.c5],
        "XPUB_noC5": c[(c.n_xpub >= 1) & ~c.c5],
        "CHEM_noC5": c[(c.n_chem >= 1) & ~c.c5],
        "OWN&XPUB_noC5": c[(c.n_own >= 1) & (c.n_xpub >= 1) & ~c.c5],
        "OWN&XEXT_noC5": c[(c.n_own >= 1) & (c.n_xext >= 1) & ~c.c5],
    }
    RES["pops"] = {}
    for k, v in pops.items():
        sh = v.groupby("cluster").N.sum()
        RES["pops"][k] = {"cells": len(v), "pubs": int(v.publication.nunique()), "clusters": int(v.cluster.nunique()),
                          "largest_cluster_share": round(float(sh.max() / sh.sum()), 2), "pairs": int(v.N.sum()),
                          "var_y": float((v.syy.sum() - v.sy.sum() ** 2 / v.N.sum()) / v.N.sum())}
        print("POP", k, RES["pops"][k])

    # ---------------- 1. scenario ladders: whole-series R2 by n_read ----------------
    ladders = {}
    scen = [("ALL229", "D7", None), ("noC5_214", "D7", None),
            ("OWN1_noC5", "D7", None), ("OWN1_noC5", "SIB_own", "D7"), ("OWN1_noC5", "SH_own", "D7"),
            ("OWN1_C5", "D7", None), ("OWN1_C5", "SH_own", "D7"),
            ("OWN2_noC5", "D7", None), ("OWN2_noC5", "SH_own", "D7"),
            ("OWN_eq1_noC5", "D7", None), ("OWN_eq1_noC5", "SH_own", "D7"), ("OWN_eq1_noC5", "SIB_own", "D7"),
            ("XEXT_noC5", "D7", None), ("XEXT_noC5", "SIB_xext", "D7"), ("XEXT_noC5", "SH_xext", "D7"),
            ("XPUB_noC5", "D7", None), ("XPUB_noC5", "SIB_xpub", "D7"), ("XPUB_noC5", "SH_xpub", "D7"),
            ("CHEM_noC5", "D7", None), ("CHEM_noC5", "SIB_chem", "D7"), ("CHEM_noC5", "SH_chem", "D7"),
            ("OWN&XPUB_noC5", "SH_own", "D7"), ("OWN&XPUB_noC5", "SH_xpub", "D7"), ("OWN&XPUB_noC5", "D7", None),
            ("OWN&XEXT_noC5", "SH_own", "D7"), ("OWN&XEXT_noC5", "SH_xext", "D7"), ("OWN&XEXT_noC5", "D7", None)]
    for pk, arm, refarm in scen:
        sub = pops[pk]
        P = Pop(sub, pk)
        idx = sub.ci.to_numpy()
        for n in range(0, NMAX + 1):
            pols = ["RND"] if n == 0 else ["RND", "HULL", "LOPO", "ORC"]
            for pol in pols:
                if pol == "LOPO":
                    s, tops = lopo_static(arm, n, sub)
                    sref = lopo_static(refarm, n, sub)[0] if refarm else None
                else:
                    s = cell_sse(arm, n, pol)[idx]
                    sref = cell_sse(refarm, n, pol)[idx] if refarm else None
                d = P.summ(s, sref)
                ladders[(pk, arm, n, pol)] = d
    rows = [{"pop": k[0], "arm": k[1], "n_read": k[2], "policy": k[3], **v} for k, v in ladders.items()]
    lad = pd.DataFrame(rows)
    lad.to_csv(OUT / "recipe_ladders.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        for pk in pops:
            t = lad[lad["pop"] == pk]
            if not len(t):
                continue
            print(f"\n=== LADDER {pk} {RES['pops'][pk]}")
            for (arm, pol), g in t.groupby(["arm", "policy"], sort=False):
                print(f"  {arm:9s} {pol:4s} " + " | ".join(
                    f"n{int(r.n_read)} {r.R2:.3f} [{r.lo:.3f},{r.hi:.3f}]" + (f" d{r.d:+.3f}[{r.d_lo:+.3f},{r.d_hi:+.3f}]" if pd.notna(getattr(r, 'd', np.nan)) else "")
                    for r in g.itertuples()))

    # ---------------- 2. which metals: in-sample top static sets (exploratory) ----------------
    RES["static_top"] = {}
    for pk, arm in (("noC5_214", "D7"), ("OWN1_noC5", "SH_own"), ("OWN1_noC5", "D7")):
        for n in (2, 3, 4, 5):
            RES["static_top"][f"{pk}|{arm}|{n}"] = insample_static(arm, n, pops[pk])
            print("STATIC", pk, arm, n, RES["static_top"][f"{pk}|{arm}|{n}"])
    for pk, arm in (("OWN1_noC5", "SH_own"), ("noC5_214", "D7")):
        for n in (2, 3, 4):
            _, tops = lopo_static(arm, n, pops[pk])
            vc = pd.Series(tops).value_counts().head(4).to_dict()
            RES["static_top"][f"LOPO_choices|{pk}|{arm}|{n}"] = vc
            print("LOPO choices", pk, arm, n, vc)

    # ---------------- 3. reads saved by the own prior: D7 reads needed to match SH_own at n reads ----------------
    sub = pops["OWN1_noC5"]
    RES["equiv"] = {}
    for pol in ("RND", "HULL"):
        d7 = {n: ladders[("OWN1_noC5", "D7", n, pol if n else "RND")]["R2"] for n in range(0, NMAX + 1)}
        sh = {n: ladders[("OWN1_noC5", "SH_own", n, pol if n else "RND")]["R2"] for n in range(0, NMAX + 1)}
        eq = {}
        for n, v in sh.items():
            better = [m for m in range(0, NMAX + 1) if d7[m] >= v]
            eq[n] = better[0] if better else f">{NMAX}"
        RES["equiv"][pol] = {"D7": d7, "SH_own": sh, "D7_reads_to_match_SH": eq}
        print("EQUIV", pol, eq)

    # ---------------- 4. target pair: per pair type, 0 experiments (COPY) vs measuring the two ends ----------------
    p0 = pairs[pairs.n_read == 0].merge(cells[["ci", "publication", "c5", "n_own", "n_xext", "n_xpub", "n_chem",
                                               "cluster"]], on="ci")
    tp = []
    for pool, arm in (("own", "SIB_own"), ("xext", "SIB_xext"), ("xpub", "SIB_xpub"), ("chem", "SIB_chem")):
        q = p0[(p0[f"n_{pool}"] >= 1) & ~p0.c5].copy()
        q["pred"] = q[f"pred_{arm}"]
        for pr, g in [("ALL", q)] + list(q.groupby("pair")):
            if len(g) < 8:
                continue
            e = g.pred - g.y
            vy = ((g.y - g.y.mean()) ** 2).mean()
            big = g.y.abs() >= 0.1
            dir_acc = float((np.sign(g.pred[big]) == np.sign(g.y[big])).mean()) if big.sum() else np.nan
            # publication bootstrap for direction accuracy and R2 within pair
            pubs = sorted(g.publication.unique())
            W = boot_counts(len(pubs), 7)
            gi = g.publication.map({p: k for k, p in enumerate(pubs)}).to_numpy()
            Gm = np.zeros((len(g), len(pubs)))
            Gm[np.arange(len(g)), gi] = 1
            sse = (e.to_numpy() ** 2) @ Gm
            Nn = np.ones(len(g)) @ Gm
            sy = g.y.to_numpy() @ Gm
            syy = (g.y.to_numpy() ** 2) @ Gm
            r2b = 1 - (W @ sse) / (W @ syy - (W @ sy) ** 2 / (W @ Nn))
            ok = ((np.sign(g.pred) == np.sign(g.y)) & big).to_numpy(float) @ Gm
            nb = big.to_numpy(float) @ Gm
            accb = (W @ ok) / np.maximum(W @ nb, 1)
            tp.append({"pool": pool, "pair": pr, "rows": len(g), "pubs": len(pubs), "sd_y": float(np.sqrt(vy)),
                       "R2": float(1 - (e ** 2).mean() / vy), "R2_lo": float(np.percentile(r2b, 2.5)),
                       "R2_hi": float(np.percentile(r2b, 97.5)), "MAE": float(e.abs().mean()),
                       "P_abs_err_le_0.1": float((e.abs() <= 0.1).mean()), "n_dir": int(big.sum()),
                       "dir_acc": dir_acc, "dir_lo": float(np.nanpercentile(accb, 2.5)),
                       "dir_hi": float(np.nanpercentile(accb, 97.5))})
    tp = pd.DataFrame(tp)
    tp.to_csv(OUT / "recipe_targetpair.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print("\n=== TARGET PAIR, 0 experiments (COPY of the pool's mean series), noC5")
        print(tp.round(3).to_string(index=False))

    # ---------------- 5. sign-stability rule: COPY_own sign vs |COPY| threshold ----------------
    ss = []
    for pool in ("own", "xext", "xpub", "chem"):
        q = p0[(p0[f"n_{pool}"] >= 1) & ~p0.c5]
        pred = q[f"pred_SIB_{pool}"]
        for thr in (0.0, 0.05, 0.1, 0.2, 0.3):
            sel = pred.abs() >= thr
            for lab, msk in (("all", np.ones(len(q), bool)), ("|y|>=0.1", (q.y.abs() >= 0.1).to_numpy())):
                m = sel.to_numpy() & msk
                if m.sum() == 0:
                    continue
                acc = float((np.sign(pred[m]) == np.sign(q.y[m])).mean())
                ss.append({"pool": pool, "thr": thr, "rows": lab, "coverage": float(m.sum() / msk.sum()),
                           "n": int(m.sum()), "sign_acc": acc,
                           "pubs": int(q[m].publication.nunique())})
    ss = pd.DataFrame(ss)
    ss.to_csv(OUT / "recipe_signrule.csv", index=False)
    print("\n=== SIGN RULE (COPY sign holds?), noC5")
    print(ss.round(3).to_string(index=False))
    # comparator: D7 (no prior) direction at 4 reads HULL, and always-heavier
    q = pairs[(pairs.n_read == 0)].merge(cells[["ci", "c5", "n_own"]], on="ci")
    q = q[(q.n_own >= 1) & ~q.c5]
    big = q.y.abs() >= 0.1
    RES["always_heavier_acc_own_noC5"] = float((q.y[big] < 0).mean())
    print("always-heavier (y<0) accuracy on |y|>=0.1 own rows:", round(RES["always_heavier_acc_own_noC5"], 3))

    (OUT / "recipe_summary.json").write_text(json.dumps(RES, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
