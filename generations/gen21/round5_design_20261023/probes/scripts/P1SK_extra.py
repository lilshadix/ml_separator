#!/usr/bin/env python
"""Skeptic of probe P1, second pass: duplicate-sibling scan, Skeptic-A OFF reconciliation, OFF_XEXT and harm rule.

EXPLORATORY (229 already-seen C3 cells, 5 dev seeds).  Prints only; writes nothing.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/probes/scripts/P1SK_extra.py

1. Duplicate scan of every CLEAN sibling link into a PRIMARY target: share of the target's metals at which the two
   log D values are bit-identical (|d| < 1e-9), and share of strict-adjacent log SF within 0.005.  Links with >= 50 %
   bit-identical log D at a different recorded condition are "suspect copies" (the round-3 flag-only cross-acid copy
   44012938119033a5 / 5169c377777d282b is 13 of 14 identical).  SH re-scored with suspect copies dropped.
2. Skeptic A's "extractant x publication offsets, 0.665-0.673 vs SH 0.682" on the V1 n_sib >= 2 no-C5 stratum:
   OFF over V1 same-extractant C3 siblings vs OFF over every other C3 cell of the publication.
3. OFF_XEXT (other extractants of the same publication) on PRIMARY rows that have such a donor.
4. Harm rule at CLEAN n_sib = 1.
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
NL = len(LANTHANIDES)


def r2(d, col):
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1.0 - ((y - g[col].to_numpy(float)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def relsse(d, cand, ref):
    return float(np.mean([1 - ((g.y - g[cand]) ** 2).sum() / ((g.y - g[ref]) ** 2).sum() for _, g in d.groupby("split_seed")]))


def lopo(d, cand, ref):
    pt = r2(d, cand) - r2(d, ref)
    pubs = sorted(d.publication.unique())
    jk = np.array([r2(d[d.publication != p], cand) - r2(d[d.publication != p], ref) for p in pubs])
    P = len(jk)
    se = np.sqrt((P - 1) / P * ((jk - jk.mean()) ** 2).sum())
    q = SS.t.ppf(0.975, P - 1)
    return pt, pt - q * se, pt + q * se, P


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
    srcfields = [c for c in f.columns if any(k in c.lower() for k in ("source", "provenance", "ornl", "origin", "curat"))]
    print("frame provenance-like columns:", srcfields[:10])

    def differs(i, j):
        out = []
        for q, c in enumerate(cond):
            a, b = C[i, q], C[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or abs(a - b) > 1e-6 * max(abs(a), abs(b)) + 1e-12:
                out.append(c)
        return [c for c in out if c not in TRIVIAL]

    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(par.cell_id)
    pools, v1pools, links = {}, {}, []
    for t in sorted(c3):
        i = pos[t]
        mets = [m for m in range(NL) if np.isfinite(Y[i, m])]
        cov = np.isfinite(Y[:, mets]).all(axis=1)
        cov[i] = False
        cand = np.flatnonzero((pub == pub[i]) & (ext == ext[i]) & cov)
        v1pools[t] = list(cand)
        cl = []
        for j in cand:
            dif = differs(i, j)
            if t in drop or ids[j] in drop or not dif:
                continue
            cl.append(j)
            dd = Y[i, mets] - Y[j, mets]
            adj = [(a, b) for a, b in zip(mets[:-1], mets[1:]) if b == a + 1]
            sfd = np.array([(Y[i, a] - Y[i, b]) - (Y[j, a] - Y[j, b]) for a, b in adj])
            links.append({"t": t, "j": ids[j], "pub": pub[i], "ident": float(np.mean(np.abs(dd) < 1e-9)),
                          "sf005": float(np.mean(np.abs(sfd) < 0.005)) if len(sfd) else np.nan,
                          "med_abs_dlogD": float(np.median(np.abs(dd))), "dif": ",".join(c.replace("cond__", "") for c in dif)})
        pools[t] = cl
    L = pd.DataFrame(links)
    nprim = {t for t in c3 if len(pools[t]) >= 2 and pub[pos[t]] != C5}
    LP = L[L.t.isin(nprim)]
    sus = LP[LP.ident >= 0.5]
    print(f"\n1. CLEAN links into PRIMARY targets: {len(LP)}; with >= 50 % bit-identical log D: {len(sus)}; "
          f"with >= 75 % of adjacent log SF within 0.005: {(LP.sf005 >= 0.75).sum()}")
    print(sus.sort_values(["pub", "t"]).to_string(index=False))
    print("   links with >= 75 % adjacent SF within 0.005, by publication:", LP[LP.sf005 >= 0.75].groupby("pub").size().to_dict())
    print("   ident distribution of PRIMARY links:", LP.ident.describe().round(3).to_dict())
    bad = set(zip(sus.t, sus.j))

    res = {(r.split_seed, r.cell_id, r.A, r.B): r.y - r.B1 for r in par[["split_seed", "cell_id", "A", "B", "y", "B1"]].itertuples(index=False)}
    PUB = par.drop_duplicates("cell_id").set_index("cell_id").publication.to_dict()
    EXT = par.drop_duplicates("cell_id").set_index("cell_id").extractant.to_dict()

    def sib_val(i, j, S_, ia, ib):
        xs = XR[S_]
        o = np.argsort(xs)
        rr = (Y[i, S_] - Y[j, S_])[o]
        return (Y[j, ia] + np.interp(XR[ia], xs[o], rr)) - (Y[j, ib] + np.interp(XR[ib], xs[o], rr))

    rows = []
    for r in par.itertuples(index=False):
        t = r.cell_id
        i = pos[t]
        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
        S_ = [LANTHANIDES.index(m) for m in str(r.read_set).split(",")]
        out = {"split_seed": r.split_seed, "cell_id": t, "publication": r.publication, "chemotype": r.chemotype, "y": r.y,
               "B1": r.B1, "n": len(pools[t]), "n_v1": len(v1pools[t])}
        for nm, pool in (("SH", pools[t]), ("SHnd", [j for j in pools[t] if (t, ids[j]) not in bad]),
                         ("SHV1", v1pools[t])):
            n = len(pool)
            if n:
                w = n / (n + K)
                out[nm] = w * np.mean([sib_val(i, j, S_, ia, ib) for j in pool]) + (1 - w) * r.B1
            else:
                out[nm] = r.B1
        out["n_nd"] = len([j for j in pools[t] if (t, ids[j]) not in bad])
        # offsets
        def off(donors):
            rr = [res[(r.split_seed, c, r.A, r.B)] for c in donors if (r.split_seed, c, r.A, r.B) in res]
            return (r.B1 + sum(rr) / (len(rr) + K)) if rr else np.nan
        out["OFF"] = off([ids[j] for j in pools[t] if ids[j] in c3])
        out["OFF_V1ext"] = off([ids[j] for j in v1pools[t] if ids[j] in c3])
        out["OFF_V1ext_all"] = off([c for c in c3 if c != t and PUB[c] == r.publication and EXT[c] == EXT[t]])
        out["OFF_PUB_V1"] = off([c for c in c3 if c != t and PUB[c] == r.publication])
        out["OFF_XEXT"] = off([c for c in c3 if c != t and c not in drop and PUB[c] == r.publication and EXT[c] != EXT[t]])
        rows.append(out)
    T = pd.DataFrame(rows)
    for c in ("OFF", "OFF_V1ext", "OFF_V1ext_all", "OFF_PUB_V1"):
        T[c + "_f"] = T[c].fillna(T.B1)
    P = T[(T.n >= 2) & (T.publication != C5)]
    Pn = T[(T.n_nd >= 2) & (T.publication != C5)]
    print(f"\n   PRIMARY: SH {r2(P, 'SH'):.4f}, B1 {r2(P, 'B1'):.4f}; suspect copies dropped, same rows: SHnd {r2(P, 'SHnd'):.4f} "
          f"(SH-B1 {r2(P, 'SHnd') - r2(P, 'B1'):+.4f}, LOPO [{lopo(P, 'SHnd', 'B1')[1]:+.4f}, {lopo(P, 'SHnd', 'B1')[2]:+.4f}]); "
          f"re-derived stratum ({Pn.cell_id.nunique()} cells, {len(Pn) / 5:.0f} rows/seed): SHnd {r2(Pn, 'SHnd'):.4f} B1 {r2(Pn, 'B1'):.4f}")
    for p in sorted(sus.pub.unique()):
        q = P[P.publication == p]
        print(f"   {p}: rows/seed {len(q) / 5:.0f}; SH-B1 SSE gain/seed {(((q.y - q.B1) ** 2) - ((q.y - q.SH) ** 2)).sum() / 5:.4f}; "
              f"with suspect copies dropped {(((q.y - q.B1) ** 2) - ((q.y - q.SHnd) ** 2)).sum() / 5:.4f}")

    V = T[(T.n_v1 >= 2) & (T.publication != C5)]
    print(f"\n2. V1 n_sib >= 2 no C5: cells {V.cell_id.nunique()}, rows/seed {len(V) / 5:.0f}; SH(V1 pool) {r2(V, 'SHV1'):.4f}; "
          f"B1 {r2(V, 'B1'):.4f}; OFF ext x pub (V1 covering C3 sibs) {r2(V, 'OFF_V1ext_f'):.4f}; OFF ext x pub (every C3 cell of "
          f"pub x ext, no coverage rule) {r2(V, 'OFF_V1ext_all_f'):.4f}; OFF publication-wide {r2(V, 'OFF_PUB_V1_f'):.4f}")

    X = P[P.OFF_XEXT.notna()]
    print(f"\n3. PRIMARY rows with an OFF_XEXT donor: {len(X) / len(P):.3f} of rows, {X.cell_id.nunique()} cells, "
          f"{X.publication.nunique()} pubs {sorted(X.publication.unique())}: OFF_XEXT {r2(X, 'OFF_XEXT'):.4f}; SH {r2(X, 'SH'):.4f}; "
          f"OFF {r2(X, 'OFF_f'):.4f}; B1 {r2(X, 'B1'):.4f}; share of SH-B1 recovered "
          f"{(r2(X, 'OFF_XEXT') - r2(X, 'B1')) / (r2(X, 'SH') - r2(X, 'B1')):.3f}")
    for p in sorted(X.publication.unique()):
        q = X[X.publication == p]
        print(f"   {p}: cells {q.cell_id.nunique()} rows/seed {len(q) / 5:.0f}; SSE/seed B1 {((q.y - q.B1) ** 2).sum() / 5:.4f} "
              f"SH {((q.y - q.SH) ** 2).sum() / 5:.4f} OFF_XEXT {((q.y - q.OFF_XEXT) ** 2).sum() / 5:.4f}")

    H = T[T.n == 1]
    print(f"\n4. CLEAN n_sib = 1: cells {H.cell_id.nunique()}, pubs {H.publication.nunique()}, C5 cells "
          f"{H[H.publication == C5].cell_id.nunique()}; SH {r2(H, 'SH'):.4f} B1 {r2(H, 'B1'):.4f}; relSSE SH vs B1 {relsse(H, 'SH', 'B1'):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
