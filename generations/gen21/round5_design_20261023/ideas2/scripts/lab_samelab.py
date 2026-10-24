#!/usr/bin/env python
"""E-LAB: laboratory or campaign?  Does the per-metal fingerprint persist across DIFFERENT publications of the SAME
laboratory?  EXPLORATORY (already-seen cells).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_samelab.py

Laboratory rule (frozen before scoring, metadata only): two publications belong to the same laboratory when their
author lists share >= 2 surnames (round-4 provenance sources_by_doi.csv maps publication_id -> DOI; P4_corpus_registry.csv
gives the author list per DOI; surname = last alphabetic token of each author entry, lower-cased, titles dropped).

1. Descriptive: pooled r of per-metal deviations (lab_varcomp definition) for cell pairs from different publications of
   the same laboratory (SL) vs different laboratories (OL), split by same / other extractant.
2. Predictive (BPG, k = 3 DOPT_Q, same machinery as lab_calib2): for C3 targets whose laboratory has >= 1 OTHER
   publication with donor series: SLP_LM10 = level-matched shape from the same laboratory's other publications (any
   extractant); OL_LM10 = the same rule from other laboratories' publications; MX_LM10 (own publication, other
   extractants) where available.
"""
from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab_calib import detrend, levels, K6  # noqa: E402
from lab_calib2 import lm_shape  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402

G21 = r"D:\ml_separator_gh\generations\gen21"
TITLES = {"dr", "prof", "mr", "ms", "mrs", "phd"}


def surnames(s: str) -> set:
    out = set()
    for a in re.split(r"[;,&]", str(s)):
        toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ\-']+", a) if t.lower().strip(".") not in TITLES]
        if toks:
            out.add(toks[-1].lower())
    return out


def main() -> int:
    src = pd.read_csv(fr"{G21}\round4_20261022\results\provenance\sources_by_doi.csv")
    reg = pd.read_csv(fr"{G21}\round5_design_20261023\probes\P4_corpus_registry.csv")
    au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
    pub2s = {}
    for r in src.itertuples(index=False):
        for p in str(r.publication_ids).split(";"):
            p = p.strip()
            if p and str(r.doi) in au:
                pub2s[p] = pub2s.get(p, set()) | surnames(au[str(r.doi)])
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    pubs = sorted(set(pub))
    print(f"publications with an author list: {sum(p in pub2s for p in pubs)} of {len(pubs)}")
    same_lab = {p: {q for q in pubs if q != p and p in pub2s and q in pub2s and len(pub2s[p] & pub2s[q]) >= 2} for p in pubs}
    links = sorted({tuple(sorted((p, q))) for p in pubs for q in same_lab[p]})
    print(f"same-laboratory publication links (>= 2 shared surnames): {len(links)}")
    for p, q in links:
        print(f"   {p} ~ {q}: {sorted(pub2s[p] & pub2s[q])[:5]}")
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    # ---- 1. descriptive ----
    idx = np.where(has)[0]
    Dc = D2[idx] - np.nanmean(D2[idx], 0)
    I, J = np.triu_indices(len(idx), 1)
    pi, pj = pub[idx][I], pub[idx][J]
    sl = np.array([b in same_lab[a] for a, b in zip(pi, pj)])
    sp = pi == pj
    se = ext[idx][I] == ext[idx][J]
    A, B = Dc[I], Dc[J]
    fin = np.isfinite(A) & np.isfinite(B)
    P = np.where(fin, A * B, 0).sum(1)
    V = np.where(fin, 0.5 * (A ** 2 + B ** 2), 0).sum(1)
    for nm, m in (("same pub", sp), ("same lab, other pub", sl & ~sp), ("other lab", ~sl & ~sp)):
        for en, e in (("same extractant", se), ("other extractant", ~se)):
            s = m & e & (V > 0)
            if s.sum():
                print(f"   r {nm:22s} {en:16s}: {P[s].sum() / V[s].sum():+.3f}  (pairs {s.sum()}, pubs "
                      f"{len(set(pi[s]) | set(pj[s]))})")
    # ---- 2. predictive ----
    ar = np.arange(len(f))
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    rows = []
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            u = np.zeros(N_LN)
            u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            prow = par[(par.split_seed == seed) & (par.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg or t in v2drop:
                    continue
                i = pos[t]
                slp = np.where(has & np.isin(pub, list(same_lab[pub[i]])))[0]
                if len(slp) == 0:
                    continue
                olp = np.where(has & (pub != pub[i]) & ~np.isin(pub, list(same_lab[pub[i]])))[0]
                dX = np.where(has & (ar != i) & (pub == pub[i]) & (ext != ext[i]))[0]
                yi = Y[i]
                metals = list(cell["metals"])
                rd = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                L = {"B1": levels(yi, rd, metals, u)}
                for nm, dn in (("SLP_LM10", slp), ("OL_LM10", olp), ("MX_LM10", dX)):
                    c, w = lm_shape(D2, Y, dn, L["B1"], 1.0, K6)
                    L[nm] = levels(yi, rd, metals, (1 - w) * u + w * c)
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "y": r.y, "has_mx": len(dX) > 0}
                    for k2, lv in L.items():
                        o[k2] = lv[ia] - lv[ib]
                    rows.append(o)
    T = pd.DataFrame(rows)
    if T.empty:
        print("no C3 target has a same-laboratory other-publication donor")
        return 0
    print(f"\nC3 targets with a same-laboratory other-publication donor: cells {T.cell_id.nunique()}, pubs "
          f"{T.publication.nunique()}, rows/seed {len(T) / 5:.0f}")
    for a in ("SLP_LM10", "OL_LM10", "MX_LM10"):
        print(f"   {a}: R2 {r2(T, a):.4f} (B1 {r2(T, 'B1'):.4f}), rel {rel(T, a):+.3f}")
    for a, b in (("SLP_LM10", "B1"), ("SLP_LM10", "OL_LM10")):
        q = infer(T, a, b)
        print(f"   {a} - {b}: rel {q['rel']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}], up/down {q['up']}/{q['down']} of {q['G']}, p {q['p_sf']:.3f}")
    S = T[T.has_mx]
    if len(S):
        print(f"   rows that also have own-publication other-extractant donors ({S.cell_id.nunique()} cells): SLP_LM10 "
              f"{rel(S, 'SLP_LM10'):+.3f}, MX_LM10 {rel(S, 'MX_LM10'):+.3f}, OL_LM10 {rel(S, 'OL_LM10'):+.3f}")
    print("   per publication rel SSE SLP_LM10 / OL_LM10:",
          {p: (round(rel(g, 'SLP_LM10'), 3), round(rel(g, 'OL_LM10'), 3), g.cell_id.nunique()) for p, g in T.groupby("publication")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
