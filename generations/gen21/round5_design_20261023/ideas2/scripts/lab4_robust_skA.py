#!/usr/bin/env python
"""Skeptic A for idea lab-4, part 2: robustness of the contrast-level and predictive checks.  EXPLORATORY (seen cells).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab4_robust_skA.py

1. Per-publication contrast statistics (log SSE ratio L): rank-based same-laboratory and same-chemotype pair correlations
   (outlier-robust), publications with >= 10 rows/seed only, permutation null; ICC by cpg cluster (M36's grouping) vs by
   laboratory, for the same contrasts.
2. B1 pair-residual same-lab sharing without the cross-publication near-copies found in lab4_recompute_skA (part E) and
   without C5-BTBP.
3. Predictive SLP_LM10 (the idea's arm, rebuilt with the idea's helper functions lm_shape / levels): R0 replicate, curated
   R2 donors, laboratory-unit inference, without C5-BTBP, without near-copy donors.
Reads only; writes only ideas2/lab4_skA_out/*.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as MT  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab_calib import detrend, levels, K6  # noqa: E402  (the idea's own helpers, imported read-only)
from lab_calib2 import lm_shape  # noqa: E402
from lab4_recompute_skA import build_rules, pair_products, C5  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab4_skA_out")
RNG = np.random.default_rng(7)


def flip_p(v):
    v = np.asarray(v, float)
    G = len(v)
    obs = v.sum()
    if G <= 18:
        sums = np.array([np.dot(s, np.abs(v)) for s in itertools.product((-1, 1), repeat=G)])
    else:
        sums = (RNG.choice((-1, 1), size=(200000, G)) * np.abs(v)).sum(1)
    return float((sums >= obs - 1e-12).mean())


def unit_infer(T, arm, base, unit):
    g = T.assign(_a=(T[arm] - T.y) ** 2, _b=(T[base] - T.y) ** 2).groupby(unit)[["_a", "_b"]].sum()
    dl = (g._b - g._a) / 5.0
    G = len(g)
    jk = np.array([1.0 - g._a.drop(p).sum() / g._b.drop(p).sum() for p in g.index])
    rel = 1.0 - g._a.sum() / g._b.sum()
    return {"G": G, "rel": rel, "p_flip": flip_p(dl.to_numpy()), "up": int((dl > 1e-12).sum()),
            "down": int((dl < -1e-12).sum()), "jk_min": float(jk.min()), "jk_max": float(jk.max()),
            "top": g.index[int(np.argmin(jk))]}


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    pubs = sorted(set(pub))
    rules, labof, links, sn = build_rules(set(pubs))
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    NC = pd.read_csv(OUT / "E_near_copies.csv")
    nc_pairs = {frozenset((a, b)) for a, b in zip(NC.cell_a, NC.cell_b) if max(0, 1) and True}
    nc_tight = {frozenset((a, b)) for a, b, m in zip(NC.cell_a, NC.cell_b, NC.max_abs_diff) if m <= 0.05}
    print(f"near-copy pairs: {len(nc_pairs)} (max|d| <= 0.05: {len(nc_tight)})")

    # ------------------------------------------------------------------------------------------- 1. robust contrast
    print("\n=== 1. per-publication log SSE ratio: rank correlation over same-lab / same-chemotype pairs ===")
    base_cols = ["split_seed", "cell_id", "read_set", "A", "B", "draw", "k", "policy"]
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    ref = ref[ref.k == 3].copy()
    contrasts = {"B1-B0": ("B0", "B1"), "HIST_D7-B1": ("B1", "HIST_D7"), "B2-B1": ("B1", "B2"), "HBK@ML-B1": ("B1", "HBK@ML")}
    for s in ("S01", "S02", "S05", "S06"):
        sl = pd.read_parquet(paths.PRED / f"{s}_BPG.parquet")
        sl = sl[sl.k == 3]
        new = [c for c in sl.columns if c not in ref.columns]
        ref = ref.merge(sl[base_cols + new], on=base_cols, how="left", validate="one_to_one")
        contrasts[f"{new[0]}-B1"] = ("B1", new[0])
    ref["publication"] = ref.publication.astype(str)
    cl = MT.add_clusters(ref[ref.split_seed == ref.split_seed.iloc[0]], f)
    cpg = cl.groupby("publication").cluster_cpg.first()
    dom_chem = ref.groupby("publication").chemotype.agg(lambda s: s.value_counts().index[0])
    nrow = ref.groupby("publication").size() / 5
    out = []
    for minrows in (0, 10):
        up = sorted(p for p in ref.publication.unique() if nrow[p] >= minrows)
        G = len(up)
        PA = [(a, b) for i, a in enumerate(up) for b in up[i + 1:]]
        ia = np.array([up.index(a) for a, _ in PA])
        ib = np.array([up.index(b) for _, b in PA])
        masks = {f"SL_{rn}": np.array([rules[rn](a, b) for a, b in PA]) for rn in ("R0", "R2", "R3")}
        masks["same_chem"] = np.array([dom_chem[a] == dom_chem[b] for a, b in PA])
        masks["same_chem_otherlab_R2"] = masks["same_chem"] & ~masks["SL_R2"]
        masks["SL_R2_otherchem"] = masks["SL_R2"] & ~masks["same_chem"]
        masks["same_cpg"] = np.array([cpg[a] == cpg[b] for a, b in PA])
        masks["same_cpg_otherlab_R2"] = masks["same_cpg"] & ~masks["SL_R2"]
        perms = np.argsort(RNG.random((20000, G)), axis=1)
        for cn, (r_, c_) in contrasts.items():
            d = ref[ref.publication.isin(up)].assign(sr=lambda t: (t[r_] - t.y) ** 2, sc=lambda t: (t[c_] - t.y) ** 2)
            g = d.groupby(["publication", "split_seed"])[["sr", "sc"]].sum().groupby("publication").mean().loc[up]
            L = np.log(g.sr / g.sc).to_numpy()
            rk = pd.Series(L).rank().to_numpy()
            z = (rk - rk.mean()) / rk.std()
            zp = z[perms]
            line = []
            for mn, m in masks.items():
                if m.sum() == 0:
                    continue
                obs = (z[ia][m] * z[ib][m]).mean()
                p = float(((zp[:, ia[m]] * zp[:, ib[m]]).mean(1) >= obs - 1e-12).mean())
                out.append({"minrows": minrows, "G": G, "contrast": cn, "pairs": mn, "n_pairs": int(m.sum()),
                            "rank_r": obs, "perm_p": p})
                line.append(f"{mn} {obs:+.2f} (p {p:.2f}, {int(m.sum())})")
            print(f"  [rows>={minrows:2d}, G {G}] {cn:12s}: " + "; ".join(line))
    pd.DataFrame(out).to_csv(OUT / "R1_contrast_rank_corr.csv", index=False)

    # ------------------------------------------------------------------------------------------- 2. B1 residual sharing
    print("\n=== 2. B1 pair-residual sharing, same-lab other-pub, without near-copies / C5-BTBP ===")
    par = ref.copy()
    par["e"] = par.y - par.B1
    par["pt"] = par.A + "-" + par.B
    em = par.groupby(["cell_id", "pt"]).e.mean().unstack()
    c3cells = [c for c in em.index if c not in v2drop]
    E = em.loc[c3cells].to_numpy(float)
    cpos = np.array([pos[c] for c in c3cells])
    M = np.full((len(f), E.shape[1]), np.nan)
    M[cpos] = E - np.nanmean(E, 0)
    I, J, P, V = pair_products(M, cpos)
    ci, cj = cpos[I], cpos[J]
    pi, pj = pub[ci], pub[cj]
    se = ext[ci] == ext[cj]
    isnc = np.array([frozenset((ids[a], ids[b])) in nc_pairs for a, b in zip(ci, cj)])
    noc5 = (pi != C5) & (pj != C5)
    for rn in ("R0", "R2"):
        sl = np.array([rules[rn](a, b) for a, b in zip(pi, pj)]) & (pi != pj)
        for nm, keep in (("all", np.ones_like(sl)), ("no near-copies", ~isnc), ("no C5", noc5), ("neither", ~isnc & noc5)):
            for en, e in (("same_ext", se), ("other_ext", ~se)):
                s = sl & e & keep & (V > 0)
                o = ~sl & (pi != pj) & e & keep & (V > 0)
                print(f"  {rn} {nm:15s} {en:9s}: SL r {P[s].sum() / V[s].sum():+.3f} ({s.sum()} pairs, "
                      f"{len(set(pi[s]) | set(pj[s]))} pubs, near-copies in class {int((sl & e & isnc).sum())}) | "
                      f"other-lab r {P[o].sum() / V[o].sum():+.3f}")

    # ------------------------------------------------------------------------------------------- 3. predictive SLP
    print("\n=== 3. predictive SLP_LM10 (idea's arm) under rule variants and laboratory units ===")
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    ar = np.arange(len(f))
    parB = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    same_lab = {rn: {p: {q for q in pubs if q != p and rules[rn](p, q)} for p in pubs} for rn in ("R0", "R2")}
    nc_cells = {}
    for a, b in zip(NC.cell_a, NC.cell_b):
        nc_cells.setdefault(a, set()).add(b)
        nc_cells.setdefault(b, set()).add(a)
    rows = []
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            u = np.zeros(N_LN)
            u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            prow = parB[(parB.split_seed == seed) & (parB.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg or t in v2drop:
                    continue
                i = pos[t]
                yi = Y[i]
                metals = list(cell["metals"])
                rd = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                L = {"B1": levels(yi, rd, metals, u)}
                flags = {}
                for rn in ("R0", "R2"):
                    slp = np.where(has & np.isin(pub, list(same_lab[rn][pub[i]])))[0]
                    flags[rn] = len(slp) > 0
                    if not flags[rn]:
                        continue
                    c, w = lm_shape(D2, Y, slp, L["B1"], 1.0, K6)
                    L[f"SLP_{rn}"] = levels(yi, rd, metals, (1 - w) * u + w * c)
                    slp2 = np.array([j for j in slp if ids[j] not in nc_cells.get(t, set())], int)
                    c, w = lm_shape(D2, Y, slp2, L["B1"], 1.0, K6)
                    L[f"SLPnc_{rn}"] = levels(yi, rd, metals, (1 - w) * u + w * c)
                for r in pcg[t].itertuples(index=False):
                    ia_, ib_ = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "y": r.y,
                         "lab_R0": labof["R0c"][pub[i]], "lab_R2": labof["R2"][pub[i]],
                         "has_R0": flags["R0"], "has_R2": flags["R2"]}
                    for k2, lv in L.items():
                        o[k2] = lv[ia_] - lv[ib_]
                    rows.append(o)
    T = pd.DataFrame(rows)
    T.to_parquet(OUT / "R3_slp_rows.parquet")
    for rn in ("R0", "R2"):
        S = T[T[f"has_{rn}"]]
        for nm, SS in (("all", S), ("no C5 target", S[S.publication != C5])):
            for arm in (f"SLP_{rn}", f"SLPnc_{rn}"):
                for unit in ("publication", f"lab_{rn}"):
                    q = unit_infer(SS, arm, "B1", unit)
                    print(f"  {rn} {nm:13s} {arm:10s} unit {unit:11s}: cells {SS.cell_id.nunique():3d} rel {q['rel']:+.3f} "
                          f"G {q['G']:2d} up/down {q['up']}/{q['down']} p_flip {q['p_flip']:.3f} "
                          f"LOO-unit [{q['jk_min']:+.3f}, {q['jk_max']:+.3f}] (min without {q['top']})")
    print("\nEXPLORATORY: all numbers are on already-seen cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
