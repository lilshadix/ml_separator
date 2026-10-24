#!/usr/bin/env python
"""Skeptic B check for idea lab-4 (campaign, not laboratory). EXPLORATORY: all 229 C3 cells are already seen.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab4skB_contrast_icc.py

Question: lab-4 infers "publication units are near-independent within a laboratory" from per-METAL residual sharing
(0.05). Type-I error of a paired confirmation test depends instead on how per-source CONTRAST statistics (log SSE
ratio B0/B1 etc.) co-vary between sources. This script measures that directly on the stored P2 per-unit statistics.

1. Layout: among C3 publications, how many have a same-lab C3 partner (lab-4's frozen rule: >= 2 shared surnames),
   and how many of those pairs already share a cpg cluster / a chemotype (i.e. M34's cluster unit already aggregates them).
2. Pairwise shared fraction of the per-publication log SSE ratio L (and per publication x chemotype unit) for:
   same lab (other pub); same cpg cluster, other lab; same chemotype, other lab; unrelated. Permutation nulls:
   (a) L permuted over all publications; (b) L permuted within cpg cluster (lab effect beyond cluster).
"""
from __future__ import annotations

import re
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4.metrics import add_clusters  # noqa: E402

G21 = r"D:\ml_separator_gh\generations\gen21"
TITLES = {"dr", "prof", "mr", "ms", "mrs", "phd"}
RNG = np.random.default_rng(20261023)
NPERM = 4000


def surnames(s: str) -> set:
    out = set()
    for a in re.split(r"[;,&]", str(s)):
        toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ\-']+", a) if t.lower().strip(".") not in TITLES]
        if toks:
            out.add(toks[-1].lower())
    return out


def lab_links(pubs):
    src = pd.read_csv(fr"{G21}\round4_20261022\results\provenance\sources_by_doi.csv")
    reg = pd.read_csv(fr"{G21}\round5_design_20261023\probes\P4_corpus_registry.csv")
    au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
    p2s = {}
    for r in src.itertuples(index=False):
        for p in str(r.publication_ids).split(";"):
            p = p.strip()
            if p and str(r.doi) in au:
                p2s[p] = p2s.get(p, set()) | surnames(au[str(r.doi)])
    return {p: {q for q in pubs if q != p and p in p2s and q in p2s and len(p2s[p] & p2s[q]) >= 2} for p in pubs}


def shared(L, I, J):
    """pooled pairwise shared fraction sum(a b) / sum((a^2 + b^2) / 2) of centred values."""
    a, b = L[I], L[J]
    v = 0.5 * (a ** 2 + b ** 2)
    return float((a * b).sum() / v.sum()) if v.sum() > 0 else np.nan


def analyse(units, L, pubs_of, chem_of, clus_of, same_lab, label):
    m = np.isfinite(L)
    units = [u for u, k in zip(units, m) if k]
    L = L[m]
    Lc = L - L.mean()
    n = len(units)
    I, J = np.triu_indices(n, 1)
    pu = np.array([pubs_of[u] for u in units])
    ch = np.array([chem_of[u] for u in units], dtype=object)
    cl = np.array([clus_of[u] for u in units])
    samepub = pu[I] == pu[J]
    sl = np.array([pu[j] in same_lab[pu[i]] for i, j in zip(I, J)]) & ~samepub
    sc = np.array([len(ch[i] & ch[j]) > 0 for i, j in zip(I, J)])
    scl = cl[I] == cl[J]
    classes = {
        "same pub, other chemotype (pubchem only)": samepub,
        "same lab, other pub": sl,
        "  same lab, other pub, shared chemotype": sl & sc,
        "  same lab, other pub, no shared chemotype": sl & ~sc,
        "same cpg cluster, other lab": scl & ~sl & ~samepub,
        "shared chemotype, other lab": sc & ~sl & ~samepub,
        "unrelated (other lab, other cluster)": ~scl & ~sl & ~samepub,
    }
    out = {}
    print(f"\n  [{label}] units {n}, mean L {L.mean():+.4f}, sd {L.std():.4f}")
    # permutation nulls
    perm_all = {k: [] for k in classes}
    perm_cl = {k: [] for k in classes}
    cl_idx = {c: np.where(cl == c)[0] for c in set(cl)}
    for _ in range(NPERM):
        x = Lc[RNG.permutation(n)]
        y = Lc.copy()
        for c, ix in cl_idx.items():
            y[ix] = Lc[RNG.permutation(ix)]
        for k, s in classes.items():
            if s.sum():
                perm_all[k].append(shared(x, I[s], J[s]))
                perm_cl[k].append(shared(y, I[s], J[s]))
    for k, s in classes.items():
        if not s.sum():
            continue
        r = shared(Lc, I[s], J[s])
        pa = (np.sum(np.array(perm_all[k]) >= r) + 1) / (NPERM + 1)
        pc = (np.sum(np.array(perm_cl[k]) >= r) + 1) / (NPERM + 1)
        nu = len(set(I[s]) | set(J[s]))
        print(f"    {k:46s} r {r:+.3f}  pairs {int(s.sum()):4d} units {nu:3d}  p_perm(all) {pa:.3f}  p_perm(within cpg) {pc:.3f}")
        out[k] = r
    return out


def main() -> int:
    ref = pd.read_parquet(fr"{G21}\round4_20261022\predictions\REF_BPG.parquet")
    cells = ref[["cell_id", "publication", "chemotype"]].drop_duplicates()
    cells = add_clusters(cells, None)
    pubs = sorted(cells.publication.astype(str).unique())
    same_lab = lab_links(pubs)
    pub_chem = cells.groupby("publication").chemotype.apply(lambda s: set(s.astype(str))).to_dict()
    pub_clus = cells.groupby("publication").cluster_cpg.agg(lambda s: sorted(set(s))[0]).to_dict()
    multi_clu = cells.groupby("publication").cluster_cpg.nunique()
    print(f"C3 publications {len(pubs)}; publications spanning >1 cpg cluster: {int((multi_clu > 1).sum())}")
    links = sorted({tuple(sorted((p, q))) for p in pubs for q in same_lab[p]})
    inlab = sorted({p for l in links for p in l})
    print(f"same-lab links among C3 publications: {len(links)}; publications involved: {len(inlab)}")
    n_scl = sum(pub_clus[a] == pub_clus[b] for a, b in links)
    n_sch = sum(len(pub_chem[a] & pub_chem[b]) > 0 for a, b in links)
    print(f"  of these links: same cpg cluster {n_scl}, shared chemotype {n_sch}")
    for a, b in links:
        print(f"   {a} ~ {b}: cluster {'same' if pub_clus[a] == pub_clus[b] else 'DIFF'}; chem {sorted(pub_chem[a])} / {sorted(pub_chem[b])}")
    # lab components (connected) among C3 pubs
    par = {p: p for p in pubs}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in links:
        par[f(a)] = f(b)
    comp = pd.Series({p: f(p) for p in pubs})
    sizes = comp.value_counts()
    print(f"lab components (transitive): {len(sizes)} for {len(pubs)} publications; multi-pub labs: "
          f"{int((sizes > 1).sum())} holding {int(sizes[sizes > 1].sum())} pubs; sizes {sizes[sizes > 1].tolist()}")
    # how many pub x chemotype units collapse under lab x chemotype
    pc = cells[["publication", "chemotype"]].drop_duplicates().copy()
    pc["lab"] = pc.publication.map(comp)
    print(f"units: publication x chemotype {len(pc)}; lab x chemotype {pc[['lab', 'chemotype']].drop_duplicates().shape[0]}; "
          f"cpg clusters {cells.cluster_cpg.nunique()}")

    for design in ("BPG", "BP"):
        us = pd.read_csv(fr"{G21}\round5_design_20261023\probes\P2_out\P2_unit_stats_{design}.csv")
        print(f"\n===== {design} =====")
        for con in ("B1-B0", "HBK@ML-B0", "HIST_D7-B1", "S02-B1", "S05-B1", "S06-B1"):
            for ut in ("publication", "pubchem"):
                d = us[(us.unit_type == ut) & (us.contrast == con)].copy()
                for minrows in (0, 10):
                    dd = d[d.n_rows_per_seed >= minrows]
                    units = dd.unit.astype(str).tolist()
                    if ut == "publication":
                        pubs_of = {u: u for u in units}
                        chem_of = {u: pub_chem[u] for u in units}
                    else:
                        pubs_of = {u: u.split("|")[0] for u in units}
                        chem_of = {u: {u.split("|")[1]} for u in units}
                    clus_of = {u: pub_clus[pubs_of[u]] for u in units}
                    analyse(units, dd.log_SSE_ratio.to_numpy(float), pubs_of, chem_of, clus_of, same_lab,
                            f"{con} | {ut} | n_rows>={minrows}")
            if design == "BP":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
