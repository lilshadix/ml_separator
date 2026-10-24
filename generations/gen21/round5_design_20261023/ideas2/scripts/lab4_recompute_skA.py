#!/usr/bin/env python
"""Skeptic A for idea lab-4 ("campaign, not laboratory: recalibrate the confirmation unit").

EXPLORATORY.  Every number below comes from the 229 already-seen C3 cells / 521 bench cells and the 5 dev seeds.
Nothing is sealed; the curated laboratory map (R2/R3) is a desk annotation written by this skeptic AFTER reading the
author lists (post hoc by construction) and is used only as a sensitivity of the frozen author-overlap rule (R0).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab4_recompute_skA.py

Reads only (never modifies): bench (gen20k), round-3 datav2 V2 lists, round-4 provenance sources_by_doi.csv,
probes/P4_corpus_registry.csv, round-4 REF/S* OOF parquets.  Writes only ideas2/lab4_skA_out/*.

Parts
  A  laboratory rules: R0 (author lists share >= 2 surnames, pairwise links, the idea's frozen rule, re-implemented),
     R0c (connected components of R0 = what a "laboratory unit" would have to be), R2 (curated groups), R3 (curated
     institutions / consortia: JAEA merged, EU partitioning consortium merged).
  B  descriptive pooled r of per-metal deviations from each cell's own quadratic in the CN8 radius coordinate, corpus mean
     removed (the idea's definition), by relation class; pair-weighted (replicate), publication-pair-weighted,
     lab-cluster bootstrap CI, leave-one-lab-out, without C5-BTBP, with V2 cells kept.
  B2 the same classes for B1's seed-mean pair residuals (y - B1 per cell x pair), i.e. the error of the contrast arm.
  C  per-publication contrast statistics (B1-B0 and near-null contrasts): same-laboratory pair correlation with a
     permutation null (the idea's step-2 kill), chemotype confound, ANOVA ICC by laboratory.
  D  type-I of publication-unit tests under a laboratory component of the per-unit contrast statistic, with the corpus's
     laboratory sizes and with a prospective "3 papers per lab" design; and the power of the idea's step-2 kill test.
  E  cross-publication near-copy scan (V2 cells kept) and whether near-copies are same-laboratory.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

G21 = r"D:\ml_separator_gh\generations\gen21"
OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab4_skA_out")
C5 = "pub_6a0ef8528c"
TITLES = {"dr", "prof", "mr", "ms", "mrs", "phd", "jr"}
RNG = np.random.default_rng(20261023)

# ---------------------------------------------------------------- curated laboratory map (post hoc desk annotation)
CUR = {
    "JAEA_SASAKI": ["1605e436d7", "d68ed8c8cf", "af0c74798c", "5a68dc5665", "b4273fb42a", "2c76a46f18", "bef16e560c",
                    "b0b3296ec5", "8dafaf94bc"],
    "JAEA_TACHIMORI_YAITA": ["74428aeab2", "8415538291", "b8175dc3a1"],
    "MOWAFY": ["1bd76c48d9", "51c0693f9c", "9a3afbee00", "b9dfc468cd", "c8195b4ca7", "f6d85bdb79"],
    "JINAN_CUI_SUN": ["0a8da089ad", "4829d176ba", "6c8ed9481c", "bdb1564c49", "f8a621ede4"],
    "ORNL": ["a49146c8fd", "126bbe7976", "dfa96a5118"],
    "JULICH": ["0af6736cb1", "2e9f2dbea9", "56338b6e47", "aed382b25f", "401a06b4b7"],
    "KIT": ["1845fd6492", "37086f6378", "af7a1cd919", "1c1b558856"],
    "CHALMERS_READING": ["723ceb6c2c", "6a0ef8528c", "c649e663f3"],
    "BABAIN": ["3e0f6bb1b5", "e9799b75f3"],
    "MSU": ["3f3b6220e0", "0fd6ab4e40"],
    "CEA_HILL": ["40c355ae86", "457a2adba4"],
}
CONSORT = {"JAEA": ["JAEA_SASAKI", "JAEA_TACHIMORI_YAITA"], "EU": ["JULICH", "KIT", "CHALMERS_READING", "CEA_HILL"]}


def surnames(s: str) -> set:
    out = set()
    s = str(s).replace("‐", "-").replace("‑", "-").replace("’", "'")
    for a in re.split(r";|,|&|\band\b", s):
        toks = [t.lower() for t in re.findall(r"[^\W\d_][^\W\d_\-'’‐]*(?:[\-‐'’][^\W\d_]+)*", a)]
        toks = [t for t in toks if t.strip(".") not in TITLES]
        if toks:
            out.add(toks[-1])
    return out


def components(nodes, links):
    par = {n: n for n in nodes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in links:
        par[find(a)] = find(b)
    return {n: find(n) for n in nodes}


def build_rules(pubs):
    src = pd.read_csv(fr"{G21}\round4_20261022\results\provenance\sources_by_doi.csv")
    reg = pd.read_csv(fr"{G21}\round5_design_20261023\probes\P4_corpus_registry.csv")
    au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
    sn = {}
    for r in src.itertuples(index=False):
        for p in str(r.publication_ids).split(";"):
            p = p.strip()
            if p in pubs and str(r.doi) in au:
                sn[p] = sn.get(p, set()) | surnames(au[str(r.doi)])
    links = sorted({(p, q) for p in pubs for q in pubs if p < q and p in sn and q in sn and len(sn[p] & sn[q]) >= 2})
    r0 = {p: {q for a, b in links for q in (a, b) if p in (a, b) and q != p} for p in pubs}
    comp = components(pubs, links)
    cur = {}
    for lab, ps in CUR.items():
        for p in ps:
            cur["pub_" + p] = lab
    r2 = {p: cur.get(p, p) for p in pubs}
    inv = {s: k for k, v in CONSORT.items() for s in v}
    r3 = {p: inv.get(r2[p], r2[p]) for p in pubs}
    unknown = sorted(set(cur) - set(pubs))
    if unknown:
        print("WARNING curated pubs not in bench:", unknown)
    rules = {
        "R0": lambda a, b: b in r0[a],
        "R0c": lambda a, b: comp[a] == comp[b],
        "R2": lambda a, b: r2[a] == r2[b],
        "R3": lambda a, b: r3[a] == r3[b],
    }
    labof = {"R0": comp, "R0c": comp, "R2": r2, "R3": r3}
    return rules, labof, links, sn


def detrend(Y, X, deg=2, min_n=6):
    Dv = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        m = np.isfinite(Y[i])
        if m.sum() >= min_n:
            Dv[i, m] = Y[i, m] - np.polyval(np.polyfit(X[m], Y[i, m], deg), X[m])
    return Dv


def pair_products(M, cells):
    """All unordered pairs of rows of M (cells): P = sum of products over shared finite entries, V = mean of squares."""
    A0 = M[cells]
    I, J = np.triu_indices(len(cells), 1)
    A, B = A0[I], A0[J]
    fin = np.isfinite(A) & np.isfinite(B)
    P = np.where(fin, A * B, 0).sum(1)
    V = np.where(fin, 0.5 * (A ** 2 + B ** 2), 0).sum(1)
    return I, J, P, V


def class_table(tag, cells, pub, ext, P, V, I, J, rules, labof, out_rows, boot=True, verbose=True):
    pi, pj = pub[cells][I], pub[cells][J]
    se = ext[cells][I] == ext[cells][J]
    sp = pi == pj
    ok = V > 0
    for rn, rule in rules.items():
        sl = np.array([rule(a, b) for a, b in zip(pi, pj)]) & ~sp
        for cn, m in (("same_pub", sp), ("same_lab_other_pub", sl), ("other_lab", ~sl & ~sp)):
            for en, e in (("same_ext", se), ("other_ext", ~se)):
                s = m & e & ok
                if not s.any():
                    continue
                r_pair = P[s].sum() / V[s].sum()
                # publication-pair weighting: each unordered (pub_i, pub_j) counts once (per-pair means)
                g = pd.DataFrame({"a": np.minimum(pi[s], pj[s]), "b": np.maximum(pi[s], pj[s]), "P": P[s], "V": V[s]})
                gm = g.groupby(["a", "b"])[["P", "V"]].mean()
                r_pp = gm.P.sum() / gm.V.sum()
                lo = hi = np.nan
                loo_min = loo_max = np.nan
                if boot and cn == "same_lab_other_pub":
                    lab = np.array([labof[rn][a] for a in pi[s]])
                    agg = pd.DataFrame({"lab": lab, "P": P[s], "V": V[s]}).groupby("lab")[["P", "V"]].sum()
                    L = len(agg)
                    if L >= 2:
                        idx = RNG.integers(0, L, size=(5000, L))
                        bs = agg.P.to_numpy()[idx].sum(1) / agg.V.to_numpy()[idx].sum(1)
                        lo, hi = np.percentile(bs, [2.5, 97.5])
                        loo = [(agg.P.sum() - agg.P[k]) / (agg.V.sum() - agg.V[k]) for k in agg.index]
                        loo_min, loo_max = min(loo), max(loo)
                row = {"set": tag, "rule": rn, "relation": cn, "extractant": en, "r_pair": r_pair, "r_pubpair": r_pp,
                       "pairs": int(s.sum()), "pubs": len(set(pi[s]) | set(pj[s])), "pubpairs": len(gm),
                       "boot_lo": lo, "boot_hi": hi, "loo_lab_min": loo_min, "loo_lab_max": loo_max}
                out_rows.append(row)
                if verbose:
                    print(f"  {tag:10s} {rn:4s} {cn:19s} {en:9s} r_pair {r_pair:+.3f}  r_pubpair {r_pp:+.3f}  pairs {s.sum():5d} "
                          f"pubs {row['pubs']:2d} pubpairs {len(gm):4d}" +
                          (f"  lab-boot95 [{lo:+.3f}, {hi:+.3f}] LOO-lab [{loo_min:+.3f}, {loo_max:+.3f}]"
                           if np.isfinite(lo) else ""))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    pubs = sorted(set(pub))
    X = np.asarray(interp._coords("R"), float)

    # ------------------------------------------------------------------------------------------------- A. rules
    print("=== A. laboratory rules ===")
    rules, labof, links, sn = build_rules(set(pubs))
    print(f"R0 links (>= 2 shared surnames): {len(links)} (idea: 51)")
    comp = labof["R0c"]
    for rn in ("R0c", "R2", "R3"):
        lab = labof[rn]
        sizes = pd.Series([lab[p] for p in pubs]).value_counts()
        print(f"{rn}: {len(sizes)} labs over {len(pubs)} pubs; multi-pub labs {int((sizes > 1).sum())}; sizes>1 "
              f"{sorted(sizes[sizes > 1].tolist(), reverse=True)}")
    big = pd.Series([comp[p] for p in pubs]).value_counts()
    for root in big[big > 1].index:
        ms = sorted(p for p in pubs if comp[p] == root)
        print(f"   R0 component ({len(ms)}): {ms}")
    # R0 pairs that R2 considers different labs, and R2 pairs R0 misses
    r0, r2 = rules["R0"], rules["R2"]
    fp = [(a, b) for a, b in links if not r2(a, b)]
    fn = [(a, b) for a in pubs for b in pubs if a < b and r2(a, b) and not r0(a, b)]
    print(f"R0 links that the curated map R2 calls different labs: {len(fp)}: "
          f"{[(a, b, sorted(sn[a] & sn[b])) for a, b in fp]}")
    print(f"same-lab pairs under R2 that R0 misses: {len(fn)}")
    c3pubs = set(pd.read_parquet(paths.PRED / "REF_BPG.parquet", columns=["publication"]).publication.astype(str))
    for a, b in fn:
        if a in c3pubs or b in c3pubs:
            print(f"   missed (C3 involved): {a}{'*' if a in c3pubs else ''} ~ {b}{'*' if b in c3pubs else ''}  "
                  f"shared surnames {sorted(sn.get(a, set()) & sn.get(b, set()))}")

    # ------------------------------------------------------------------------------------------------- B. descriptive
    print("\n=== B. per-metal deviation sharing (detrended quadratic, corpus mean removed) ===")
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, X)
    rows = []
    for tag, keep in (("V2clean", okd), ("V2clean-C5", okd & (pub != C5)), ("withV2", np.ones(len(f), bool))):
        has = np.isfinite(D2).any(1) & keep
        cells = np.where(has)[0]
        Dc = D2.copy()
        Dc[cells] = D2[cells] - np.nanmean(D2[cells], 0)
        I, J, P, V = pair_products(Dc, cells)
        print(f" [{tag}] cells {len(cells)}, pubs {len(set(pub[cells]))}")
        class_table(tag, cells, pub, ext, P, V, I, J, rules, labof, rows)
        if tag == "V2clean":
            # per-lab breakdown of the SL class under R0 and R2
            pi, pj = pub[cells][I], pub[cells][J]
            for rn in ("R0", "R2"):
                sl = np.array([rules[rn](a, b) for a, b in zip(pi, pj)]) & (pi != pj) & (V > 0)
                lab = np.array([labof[rn][a] for a in pi])
                t = pd.DataFrame({"lab": lab[sl], "P": P[sl], "V": V[sl], "se": (ext[cells][I] == ext[cells][J])[sl]})
                agg = t.groupby("lab").agg(P=("P", "sum"), V=("V", "sum"), n=("P", "size"), n_se=("se", "sum"))
                agg["r"] = agg.P / agg.V
                agg["share_V"] = agg.V / agg.V.sum()
                print(f"   per-lab SL breakdown ({rn}):\n" + agg.sort_values("V", ascending=False).round(3).to_string())
            # metal-restricted: is the per-metal SL sharing concentrated where the within-pub sharing is (Pr-Dy)?
    pd.DataFrame(rows).to_csv(OUT / "B_descriptive_classes.csv", index=False)

    # ------------------------------------------------------------------------------------------------- B2. B1 residuals
    print("\n=== B2. B1 pair-residual sharing (seed-mean y - B1 per cell x (A,B)) ===")
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    par = par[(par.k == 3)]
    par["e"] = par.y - par.B1
    par["pt"] = par.A + "-" + par.B
    em = par.groupby(["cell_id", "pt"]).e.mean().unstack()
    c3cells = [c for c in em.index if c not in v2drop]
    E = em.loc[c3cells].to_numpy(float)
    cpos = np.array([pos[c] for c in c3cells])
    rows2 = []
    for center in (False, True):
        Ec = E - np.nanmean(E, 0) if center else E
        M = np.full((len(f), Ec.shape[1]), np.nan)
        M[cpos] = Ec
        I, J, P, V = pair_products(M, cpos)
        tag = "B1res_c" if center else "B1res"
        print(f" [{tag}] C3 cells {len(cpos)}")
        class_table(tag, cpos, pub, ext, P, V, I, J, {k: rules[k] for k in ("R0", "R2")}, labof, rows2)
    pd.DataFrame(rows2).to_csv(OUT / "B2_B1residual_classes.csv", index=False)

    # ------------------------------------------------------------------------------------------------- C. contrasts
    print("\n=== C. per-publication contrast statistics: same-laboratory correlation (permutation null) ===")
    base_cols = ["split_seed", "cell_id", "read_set", "A", "B", "draw", "k", "policy"]
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    ref = ref[ref.k == 3].copy()
    contrasts = {"B1-B0": ("B0", "B1"), "HIST_D7-B1": ("B1", "HIST_D7"), "B2-B1": ("B1", "B2"),
                 "HBK@ML-B1": ("B1", "HBK@ML"), "PWL_Z-B0": ("B0", "PWL_Z")}
    for s in ("S01", "S02", "S05", "S06", "S11"):
        sl = pd.read_parquet(paths.PRED / f"{s}_BPG.parquet")
        sl = sl[sl.k == 3]
        new = [c for c in sl.columns if c not in ref.columns]
        if len(new) == 1:
            ref = ref.merge(sl[base_cols + new], on=base_cols, how="left", validate="one_to_one")
            contrasts[f"{new[0]}-B1"] = ("B1", new[0])
        else:
            print(f"   slot {s}: new columns {new} -> skipped")
    ref["publication"] = ref.publication.astype(str)
    dom_chem = ref.groupby("publication").chemotype.agg(lambda s: s.value_counts().index[0])
    upubs = sorted(ref.publication.unique())
    G = len(upubs)
    pairs_all = [(a, b) for i, a in enumerate(upubs) for b in upubs[i + 1:]]
    stats = {}
    for cn, (r_, c_) in contrasts.items():
        d = ref.assign(sr=(ref[r_] - ref.y) ** 2, sc=(ref[c_] - ref.y) ** 2)
        g = d.groupby(["publication", "split_seed"])[["sr", "sc"]].sum().groupby("publication").mean()
        st = pd.DataFrame({"dSSE": g.sr - g.sc, "L": np.log(g.sr / g.sc), "rel": 1 - g.sc / g.sr, "n": d.groupby("publication").size() / 5})
        stats[cn] = st.loc[upubs]
    out_c = []
    PERMS = np.argsort(RNG.random((20000, G)), axis=1)
    for rn in ("R0", "R2", "R3"):
        SL = np.array([rules[rn](a, b) for a, b in pairs_all])
        SC = np.array([dom_chem[a] == dom_chem[b] for a, b in pairs_all])
        ia = np.array([upubs.index(a) for a, _ in pairs_all])
        ib = np.array([upubs.index(b) for _, b in pairs_all])
        nl = int(SL.sum())
        npubs_sl = len({p for (a, b), s in zip(pairs_all, SL) if s for p in (a, b)})
        print(f" rule {rn}: same-lab C3 publication pairs {nl} among {npubs_sl} publications; of them same dominant "
              f"chemotype {int((SL & SC).sum())}; other-lab same-chemotype pairs {int((~SL & SC).sum())}")
        for cn, st in stats.items():
            for metric in ("L", "dSSE"):
                x = st[metric].to_numpy(float)
                if not np.isfinite(x).all() or np.std(x) == 0:
                    continue
                z = (x - x.mean()) / x.std()
                obs = (z[ia][SL] * z[ib][SL]).mean()
                obs_sc = (z[ia][~SL & SC] * z[ib][~SL & SC]).mean() if (~SL & SC).any() else np.nan
                zp = z[PERMS]
                null = (zp[:, ia[SL]] * zp[:, ib[SL]]).mean(1)
                p = float((null >= obs - 1e-12).mean())
                # ANOVA ICC(1) by laboratory (C3 pubs)
                labs = pd.Series([labof[rn][u] for u in upubs])
                df = pd.DataFrame({"lab": labs, "x": x})
                k = df.groupby("lab").x.agg(["size", "mean"])
                N, Kl = len(df), len(k)
                msb = (k["size"] * (k["mean"] - x.mean()) ** 2).sum() / max(Kl - 1, 1)
                msw = ((df.x - df.lab.map(k["mean"])) ** 2).sum() / max(N - Kl, 1)
                n0 = (N - (k["size"] ** 2).sum() / N) / max(Kl - 1, 1)
                icc = (msb - msw) / (msb + (n0 - 1) * msw)
                out_c.append({"rule": rn, "contrast": cn, "metric": metric, "r_samelab": obs, "perm_p": p,
                              "r_otherlab_samechem": obs_sc, "icc_anova": icc, "n_sl_pairs": nl})
                if metric == "L" or cn == "B1-B0":
                    print(f"   {cn:14s} {metric:4s}: r_SL {obs:+.3f} (perm p {p:.3f})  r_OL-same-chem {obs_sc:+.3f}  "
                          f"ICC_lab {icc:+.3f}")
    C_df = pd.DataFrame(out_c)
    C_df.to_csv(OUT / "C_contrast_samelab.csv", index=False)
    st = stats["B1-B0"].assign(lab_R0=[labof["R0"][u] for u in upubs], lab_R2=[labof["R2"][u] for u in upubs],
                               chem=dom_chem.loc[upubs].to_numpy())
    st.to_csv(OUT / "C_B1B0_per_pub.csv")
    print("   B1-B0 per publication in multi-publication R2 labs:")
    m = st.groupby("lab_R2").filter(lambda g: len(g) > 1).sort_values("lab_R2")
    print(m[["lab_R2", "chem", "n", "rel", "L", "dSSE"]].round(3).to_string())

    # ------------------------------------------------------------------------------------------------- D. type-I sim
    print("\n=== D. type-I of publication-unit tests under a laboratory component (synthetic, structure from C3) ===")
    nrep, nflip = 4000, 2000

    def flip_p(x):  # one-sided sign-flip p for H1: mean > 0; x shape (reps, G)
        s = RNG.choice((-1.0, 1.0), size=(nflip, x.shape[1]))
        obs = x.sum(1)
        null = np.abs(x) @ s.T  # (reps, nflip)
        return ((null >= obs[:, None] - 1e-12).sum(1) + 1) / (nflip + 1)

    def sim(lab_ids, w, rho, heavy=False):
        labs, inv = np.unique(lab_ids, return_inverse=True)
        if heavy:
            a = RNG.standard_t(3, size=(nrep, len(labs))) / np.sqrt(3)
            e = RNG.standard_t(3, size=(nrep, len(lab_ids))) / np.sqrt(3)
        else:
            a = RNG.standard_normal((nrep, len(labs)))
            e = RNG.standard_normal((nrep, len(lab_ids)))
        x = w[None, :] * (np.sqrt(rho) * a[:, inv] + np.sqrt(1 - rho) * e)
        p_pub = flip_p(x)
        # laboratory units: one aggregated value per lab
        xl = np.zeros((nrep, len(labs)))
        np.add.at(xl.T, inv, x.T)
        p_lab = flip_p(xl)
        return float((p_pub <= 0.05).mean()), float((p_pub <= 0.025).mean()), float((p_lab <= 0.05).mean())

    out_d = []
    designs = {}
    for rn in ("R0c", "R2", "R3"):
        designs[f"corpusC3_{rn}_equal"] = (np.array([labof[rn][u] for u in upubs]), np.ones(G))
        designs[f"corpusC3_{rn}_rowweighted"] = (np.array([labof[rn][u] for u in upubs]),
                                                 np.sqrt(stats["B1-B0"].n.to_numpy(float)))
    designs["prospective_7labs_x3"] = (np.repeat(np.arange(7), 3), np.ones(21))
    designs["prospective_4labs_x5"] = (np.repeat(np.arange(4), 5), np.ones(20))
    designs["prospective_20labs_x1"] = (np.arange(20), np.ones(20))
    for dn, (lab_ids, w) in designs.items():
        _, cnt = np.unique(lab_ids, return_counts=True)
        mbar = (cnt ** 2).sum() / cnt.sum()
        for rho in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.44):
            for heavy in (False, True):
                a05, a025, l05 = sim(lab_ids, w, rho, heavy)
                out_d.append({"design": dn, "rho_lab": rho, "heavy_t3": heavy, "size_pub_0.05": a05,
                              "size_pub_0.025": a025, "size_lab_0.05": l05, "m_bar": mbar})
        r = [o for o in out_d if o["design"] == dn and not o["heavy_t3"]]
        print(f" {dn:30s} m_bar {mbar:.2f}  size(pub units, nominal 0.05) by rho_lab 0/.05/.1/.15/.2/.3/.44: "
              + " ".join(f"{o['size_pub_0.05']:.3f}" for o in r) + "  | lab units at 0.3: "
              + f"{[o['size_lab_0.05'] for o in r if o['rho_lab'] == 0.3][0]:.3f}")
    pd.DataFrame(out_d).to_csv(OUT / "D_typeI_sim.csv", index=False)

    # power of the idea's step-2 kill test (r_SL >= 0.3 and permutation p < 0.05) given the corpus link graph
    print("\n power of the step-2 kill (r_SL >= 0.3 & perm p < 0.05) on the corpus C3 same-lab graph:")
    for rn in ("R0", "R2"):
        SL = np.array([rules[rn](a, b) for a, b in pairs_all])
        ia = np.array([upubs.index(a) for a, _ in pairs_all])[SL]
        ib = np.array([upubs.index(b) for _, b in pairs_all])[SL]
        labs = np.array([labof[rn][u] for u in upubs])
        # for R0 (non-transitive links) generate the lab effect on R0 components
        if rn == "R0":
            labs = np.array([labof["R0c"][u] for u in upubs])
        _, inv = np.unique(labs, return_inverse=True)
        perms = np.array([RNG.permutation(G) for _ in range(2000)])
        res = []
        for rho in (0.0, 0.1, 0.2, 0.3, 0.5):
            hit = 0
            rs = []
            for _ in range(400):
                a = RNG.standard_normal(inv.max() + 1)
                x = np.sqrt(rho) * a[inv] + np.sqrt(1 - rho) * RNG.standard_normal(G)
                z = (x - x.mean()) / x.std()
                obs = (z[ia] * z[ib]).mean()
                zp = z[perms]
                p = ((zp[:, ia] * zp[:, ib]).mean(1) >= obs).mean()
                rs.append(obs)
                hit += (obs >= 0.3) and (p < 0.05)
            res.append(f"rho {rho:.1f}: P(kill) {hit / 400:.2f}, median r_SL {np.median(rs):+.2f} [{np.percentile(rs, 5):+.2f}, {np.percentile(rs, 95):+.2f}]")
        print(f"   {rn} ({len(ia)} SL pairs): " + "; ".join(res))

    # ------------------------------------------------------------------------------------------------- E. near copies
    print("\n=== E. cross-publication near-copy scan (all bench cells, V2 kept) ===")
    cells = np.where(np.isfinite(Y).sum(1) >= 5)[0]
    I, J = np.triu_indices(len(cells), 1)
    A, B = Y[cells][I], Y[cells][J]
    fin = np.isfinite(A) & np.isfinite(B)
    nsh = fin.sum(1)
    dlt = np.where(fin, A - B, np.nan)
    with np.errstate(all="ignore"):
        mad = np.nanmax(np.abs(dlt), 1)
        sdd = np.nanstd(dlt, 1)
    ci, cj = cells[I], cells[J]
    cross = pub[ci] != pub[cj]
    near = cross & (nsh >= 5) & ((mad <= 0.05) | (sdd <= 0.02))
    rr = []
    for a, b, k_, mx, sd in zip(ci[near], cj[near], nsh[near], mad[near], sdd[near]):
        rr.append({"cell_a": ids[a], "cell_b": ids[b], "pub_a": pub[a], "pub_b": pub[b], "n_shared": int(k_),
                   "max_abs_diff": mx, "sd_diff": sd, "same_ext": ext[a] == ext[b], "same_lab_R0": rules["R0"](pub[a], pub[b]),
                   "same_lab_R2": rules["R2"](pub[a], pub[b]), "v2_a": ids[a] in v2drop, "v2_b": ids[b] in v2drop,
                   "c3_a": pub[a] in c3pubs, "c3_b": pub[b] in c3pubs})
    NC = pd.DataFrame(rr)
    NC.to_csv(OUT / "E_near_copies.csv", index=False)
    print(f" cross-publication cell pairs with >= 5 shared metals: {int((cross & (nsh >= 5)).sum())}; near-copies "
          f"(max|d| <= 0.05 or sd(d) <= 0.02): {len(NC)}")
    if len(NC):
        print(NC.groupby(["same_lab_R0", "same_lab_R2"]).size().to_string())
        print(NC.round(3).to_string())
    print("\nEXPLORATORY: all numbers are on already-seen cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
