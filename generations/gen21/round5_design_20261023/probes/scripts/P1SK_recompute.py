#!/usr/bin/env python
"""Skeptic of probe P1 (E-CT condition transfer): independent recomputation of the decision-relevant numbers.

EXPLORATORY: every number is computed on the 229 already-seen C3 cells and the 5 dev seeds.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/probes/scripts/P1SK_recompute.py

Independent of P1_ct_build.py / P1_ct_analyze.py / P1_ct_diag.py: sibling pools, the CLEAN rule, SIB / SH / OFF,
pooled R2, the LOPO jackknife and the unit contributions are re-implemented here.  B1 and y are taken directly from
the round-4 REF parquets (not recomputed); the radius coordinate comes from r3.interp (the programme's definition).
Prints only; writes nothing.

Adds sensitivities P1 did not run:
  NC   drop near-copy siblings (E6 definition: sd of (Y_t - Y_j) over the target's metals after removing an offset and
       a radius slope, < 0.02 = near copy; label-defined, a sensitivity only as IDEAS M01 step 2 asks).
  DIG  treat condition differences below 1 % relative (digitiser noise, round-3 inventory) as no difference.
  XAC  drop the suspected ORNL cross-acid copy pair 44012938119033a5 / 5169c377777d282b as mutual siblings.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES  # noqa: E402

K = 1.78
C5 = "pub_6a0ef8528c"
XR = interp._coords("R")
TRIVIAL = ("cond__contact_time_min", "cond__metal_concentration_mM")
XAC = {"44012938119033a5", "5169c377777d282b"}


def r2_seed(g, col):
    y = g.y.to_numpy(float)
    return 1.0 - ((y - g[col].to_numpy(float)) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2(d, col):
    return float(np.mean([r2_seed(g, col) for _, g in d.groupby("split_seed")]))


def relsse(d, cand, ref):
    return float(np.mean([1 - ((g.y - g[cand]) ** 2).sum() / ((g.y - g[ref]) ** 2).sum() for _, g in d.groupby("split_seed")]))


def lopo(d, cand, ref):
    pt = r2(d, cand) - r2(d, ref)
    pubs = sorted(d.publication.unique())
    jk = np.array([r2(d[d.publication != p], cand) - r2(d[d.publication != p], ref) for p in pubs])
    P = len(jk)
    se = np.sqrt((P - 1) / P * ((jk - jk.mean()) ** 2).sum())
    q = SS.t.ppf(0.975, P - 1)
    i = int(np.argmin(jk))
    return pt, pt - q * se, pt + q * se, P, float(jk.min()), pubs[i]


def units(d, cand, ref):
    sst = {s: float(((g.y - g.y.mean()) ** 2).sum()) for s, g in d.groupby("split_seed")}
    e = d[["publication", "chemotype", "split_seed"]].copy()
    e["c"] = (((d.y - d[ref]) ** 2) - ((d.y - d[cand]) ** 2)) / d.split_seed.map(sst) / d.split_seed.nunique()
    return e.groupby(["publication", "chemotype"]).c.sum()


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    cond = [c for c in f.columns if c.startswith("cond__")]
    C = f[cond].to_numpy(float)
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)

    def differs(i, j, rtol):
        out = []
        for q, c in enumerate(cond):
            a, b = C[i, q], C[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b):
                out.append(c)
                continue
            if abs(a - b) > rtol * max(abs(a), abs(b)) + 1e-12:
                out.append(c)
        return out

    def shape_sd(i, j, mets):
        d = Y[i, mets] - Y[j, mets]
        xx = XR[mets]
        A = np.c_[np.ones(len(mets)), xx]
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        r = d - A @ coef
        return float(np.std(r, ddof=0))

    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    c3 = sorted(set(refs["BPG"].cell_id))
    c3s = set(c3)

    # sibling pools, independently built
    pools = {}
    sibrec = []
    for t in c3:
        i = pos[t]
        mets = [m for m in range(len(LANTHANIDES)) if np.isfinite(Y[i, m])]
        cov = np.isfinite(Y[:, mets]).all(axis=1)
        cov[i] = False
        cand = np.flatnonzero((pub == pub[i]) & (ext == ext[i]) & cov)
        P = {"base": [], "nc": [], "dig": [], "xac": [], "all": []}
        for j in cand:
            dif = [c for c in differs(i, j, 1e-6) if c not in TRIVIAL]
            difd = [c for c in differs(i, j, 0.01) if c not in TRIVIAL]
            ok = t not in drop and ids[j] not in drop and bool(dif)
            sd = shape_sd(i, j, mets)
            sibrec.append({"t": t, "j": ids[j], "pub": pub[i], "ok": ok, "sd": sd, "dif": ",".join(dif), "difd": ",".join(difd)})
            if ok:
                P["base"].append(j)
                if sd >= 0.02:
                    P["nc"].append(j)
                if difd:
                    P["dig"].append(j)
                if not ({t, ids[j]} <= XAC):
                    P["xac"].append(j)
        pools[t] = P
    S = pd.DataFrame(sibrec)
    ok = S[S.ok]
    print(f"CLEAN sibling links: {len(ok)} (V1 links {len(S)}); near-copy links (shape sd < 0.02) among CLEAN: "
          f"{(ok.sd < 0.02).sum()}; CLEAN links whose only non-trivial differences are < 1 % relative: "
          f"{(ok.difd == '').sum()}")
    print("XAC pair links:", S[S.t.isin(XAC) & S.j.isin(XAC)][["t", "j", "ok", "sd", "dif"]].to_string(index=False))
    for t in XAC:
        print(f"  {t}: C3 {t in c3s}, pub {pub[pos[t]]}, CLEAN n {len(pools[t]['base']) if t in pools else '-'}")

    for design in ("BPG", "BP"):
        par = refs[design].copy()
        res = {(r.split_seed, r.cell_id, r.A, r.B): r.y - r.B1 for r in par[["split_seed", "cell_id", "A", "B", "y", "B1"]].itertuples(index=False)}
        rsmap = {(r.split_seed, r.cell_id): (r.read_set, r.fold) for r in par.drop_duplicates(["split_seed", "cell_id"]).itertuples(index=False)}
        out = {}
        for var in ("base", "nc", "dig", "xac"):
            SH, SIB, OFF, NN, GEOM = [], [], [], [], []
            for r in par.itertuples(index=False):
                i = pos[r.cell_id]
                pool = pools[r.cell_id][var]
                n = len(pool)
                NN.append(n)
                if n == 0:
                    SH.append(np.nan); SIB.append(np.nan); OFF.append(np.nan); GEOM.append(False)
                    continue
                ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                S_ = [LANTHANIDES.index(m) for m in str(r.read_set).split(",")]
                xs = XR[S_]
                o = np.argsort(xs)
                xs = xs[o]
                vals = []
                for j in pool:
                    rr = (Y[i, S_] - Y[j, S_])[o]
                    la = Y[j, ia] + np.interp(XR[ia], xs, rr)
                    lb = Y[j, ib] + np.interp(XR[ib], xs, rr)
                    vals.append(la - lb)
                sib = float(np.mean(vals))
                w = n / (n + K)
                SIB.append(sib)
                SH.append(w * sib + (1 - w) * r.B1)
                donors = [ids[j] for j in pool if ids[j] in c3s and (r.split_seed, ids[j], r.A, r.B) in res]
                rr_ = [res[(r.split_seed, c, r.A, r.B)] for c in donors]
                OFF.append(r.B1 + sum(rr_) / (len(rr_) + K))
                GEOM.append(all(ids[j] in c3s and rsmap.get((r.split_seed, ids[j])) == (r.read_set, r.fold)
                                and (r.split_seed, ids[j], r.A, r.B) in res for j in pool))
            d = par.copy()
            d["SH"], d["SIB"], d["OFF"], d["n"], d["geom"] = SH, SIB, OFF, NN, GEOM
            out[var] = d
        d0 = out["base"]
        P = d0[(d0.n >= 2) & (d0.publication != C5)].copy()
        print(f"\n===== {design} PRIMARY (CLEAN n>=2, no C5): cells {P.cell_id.nunique()}, pubs {P.publication.nunique()}, "
              f"units {P[['publication', 'chemotype']].drop_duplicates().shape[0]}, rows/seed {len(P) / 5:.1f}")
        for c in ("B1", "SH", "SIB", "OFF"):
            print(f"   R2 {c}: {r2(P, c):.4f}  seeds {[round(r2_seed(g, c), 4) for _, g in P.groupby('split_seed')]}")
        for cand, ref in (("SH", "B1"), ("SH", "OFF"), ("OFF", "B1")):
            pt, lo, hi, Pn, mn, mp = lopo(P, cand, ref)
            u = units(P, cand, ref)
            print(f"   {cand}-{ref}: dR2 {pt:+.4f}, relSSE {relsse(P, cand, ref):+.4f}, LOPO 95% [{lo:+.4f}, {hi:+.4f}] (P={Pn}), "
                  f"LOPO min {mn:+.4f} without {mp}; units +{(u > 1e-6).sum()} -{(u < -1e-6).sum()} 0:{(u.abs() <= 1e-6).sum()}")
        u = units(P, "SH", "OFF")
        print("   SH-OFF unit contributions (nonzero):", u[u.abs() > 1e-7].round(5).to_dict())
        g = P.geom
        print(f"   identity: rows with every CLEAN sibling a C3 donor at the target's read set and fold: {g.mean():.4f}; "
              f"max |SH-OFF| there {(P.SH - P.OFF)[g].abs().max():.2e}; rows elsewhere/seed {(~g).sum() / 5:.0f}, cells "
              f"{P[~g].cell_id.nunique()}, pubs {sorted(P[~g].publication.unique())}; on them SH {r2(P[~g], 'SH'):.4f} "
              f"OFF {r2(P[~g], 'OFF'):.4f} B1 {r2(P[~g], 'B1'):.4f}")
        nonc3 = sum(1 for t in P.cell_id.unique() for j in pools[t]["base"] if ids[j] not in c3s)
        print(f"   CLEAN siblings of PRIMARY targets that are not C3 cells: {nonc3}")
        # without pub_19783c039b
        Q = P[P.publication != "pub_19783c039b"]
        print(f"   without pub_19783c039b: SH-OFF {r2(Q, 'SH') - r2(Q, 'OFF'):+.4f}; SH-B1 {r2(Q, 'SH') - r2(Q, 'B1'):+.4f}")
        # sensitivities
        for var, lab in (("nc", "near-copy siblings dropped (shape sd < 0.02)"), ("dig", "<1 % condition differences = none"),
                         ("xac", "cross-acid copy pair dropped")):
            dv = out[var]
            Pv = dv[(dv.n >= 2) & (dv.publication != C5)]
            # same PRIMARY rows with the reduced pools (cells left with 0 siblings fall back to B1)
            Pr = dv.loc[P.index].copy()
            Pr["SH"] = Pr.SH.fillna(Pr.B1)
            Pr["OFF"] = Pr.OFF.fillna(Pr.B1)
            print(f"   [{lab}] re-derived stratum: cells {Pv.cell_id.nunique()}, pubs {Pv.publication.nunique()}, rows/seed "
                  f"{len(Pv) / 5:.0f}: B1 {r2(Pv, 'B1'):.4f} SH {r2(Pv, 'SH'):.4f} OFF {r2(Pv, 'OFF'):.4f}; SH-B1 "
                  f"{r2(Pv, 'SH') - r2(Pv, 'B1'):+.4f} LOPO [{lopo(Pv, 'SH', 'B1')[1]:+.4f}, {lopo(Pv, 'SH', 'B1')[2]:+.4f}]; "
                  f"SH-OFF {r2(Pv, 'SH') - r2(Pv, 'OFF'):+.4f}")
            print(f"        on the original PRIMARY rows: SH {r2(Pr, 'SH'):.4f} (B1 {r2(Pr, 'B1'):.4f}), SH-B1 "
                  f"{r2(Pr, 'SH') - r2(Pr, 'B1'):+.4f}; cells whose pool changed {sum(len(pools[t][var]) != len(pools[t]['base']) for t in P.cell_id.unique())}")
        # near-copy per publication
        nc = ok[(ok.sd < 0.02) & ok.t.isin(set(P.cell_id))]
        print("   near-copy CLEAN links into PRIMARY targets by publication:", nc.groupby("pub").size().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
