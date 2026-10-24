#!/usr/bin/env python
"""Skeptic B check for idea lab-4, part 2. EXPLORATORY (all bench cells already seen; labels used descriptively).

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab4skB_gddip_lab.py

B1 - B0 differs only by the Gd amplitude, so the dependence that matters for its type-I error is how the TRUE Gd dip
(the effect B1 exploits) is shared between sources, not generic per-metal residual sharing. Gd dip of a cell:
  dip = logD_Gd - (logD_Eu + logD_Tb) / 2   (cells with Eu, Gd, Tb finite).
Also the B0 OOF residual Gd chord per C3 cell (mean y - B0 on Eu/Gd minus on Gd/Tb), which is exactly what B1 corrects.
Sharing r = sum(a b) / sum((a^2 + b^2) / 2) on centred values, for cell pairs and for publication means, by relation
class (lab rule of lab-4: author lists share >= 2 surnames). Permutation null for publication means.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401  (adds gen20k to sys.path)
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES  # noqa: E402
from lab4skB_contrast_icc import lab_links  # noqa: E402

G21 = r"D:\ml_separator_gh\generations\gen21"
RNG = np.random.default_rng(7)
NPERM = 4000


def sh(a, b):
    v = 0.5 * (a ** 2 + b ** 2)
    return float((a * b).sum() / v.sum()) if v.sum() > 0 else np.nan


def classes_for(pa, pb, ea, eb, ca, cb, same_lab):
    sp = pa == pb
    sl = np.array([q in same_lab[p] for p, q in zip(pa, pb)]) & ~sp
    se = ea == eb
    sc = ca == cb
    return {
        "same pub, same extractant": sp & se,
        "same pub, other extractant": sp & ~se,
        "same lab other pub, same extractant": sl & se,
        "same lab other pub, same chemotype, other extr": sl & sc & ~se,
        "same lab other pub, other chemotype": sl & ~sc,
        "other lab, same extractant": ~sp & ~sl & se,
        "other lab, same chemotype, other extr": ~sp & ~sl & sc & ~se,
        "other lab, other chemotype": ~sp & ~sl & ~sc,
    }


def cell_level(v, pub, ext, chem, same_lab, label):
    m = np.isfinite(v)
    v, pub, ext, chem = v[m], pub[m], ext[m], chem[m]
    c = v - v.mean()
    I, J = np.triu_indices(len(v), 1)
    cl = classes_for(pub[I], pub[J], ext[I], ext[J], chem[I], chem[J], same_lab)
    print(f"\n[{label}] cells {len(v)}, pubs {len(set(pub))}, mean {v.mean():+.3f}, sd {v.std():.3f}")
    for k, s in cl.items():
        if s.sum():
            print(f"   {k:48s} r {sh(c[I[s]], c[J[s]]):+.3f}  pairs {int(s.sum()):6d}  pubs {len(set(pub[I[s]]) | set(pub[J[s]])):3d}")


def pub_level(v, pub, chem, same_lab, label):
    df = pd.DataFrame({"v": v, "pub": pub, "chem": chem}).dropna()
    g = df.groupby("pub").agg(v=("v", "mean"), n=("v", "size"),
                              chem=("chem", lambda s: s.value_counts().index[0])).reset_index()
    x = g.v.to_numpy() - g.v.mean()
    p = g.pub.to_numpy()
    ch = g.chem.to_numpy()
    I, J = np.triu_indices(len(g), 1)
    sl = np.array([b in same_lab[a] for a, b in zip(p[I], p[J])])
    sc = ch[I] == ch[J]
    cls = {"same lab": sl, "  same lab, same main chemotype": sl & sc, "  same lab, other main chemotype": sl & ~sc,
           "other lab, same main chemotype": ~sl & sc, "other lab, other main chemotype": ~sl & ~sc}
    perm = {k: [] for k in cls}
    for _ in range(NPERM):
        y = x[RNG.permutation(len(x))]
        for k, s in cls.items():
            if s.sum():
                perm[k].append(sh(y[I[s]], y[J[s]]))
    print(f"\n[{label}] publication means: pubs {len(g)}, sd {g.v.std():.3f}")
    for k, s in cls.items():
        if s.sum():
            r = sh(x[I[s]], x[J[s]])
            pv = (np.sum(np.array(perm[k]) >= r) + 1) / (NPERM + 1)
            print(f"   {k:40s} r {r:+.3f}  pairs {int(s.sum()):4d}  p_perm {pv:.3f}")
    # within-lab ICC (one-way ANOVA over multi-pub labs, publication means)
    return g


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    pubs = sorted(set(pub))
    same_lab = lab_links(pubs)
    iE, iG, iT = (LANTHANIDES.index(m) for m in ("Eu", "Gd", "Tb"))
    dip = Y[:, iG] - 0.5 * (Y[:, iE] + Y[:, iT])
    print(f"bench cells {len(f)}; cells with Eu, Gd, Tb: {int(np.isfinite(dip).sum())}")
    cell_level(dip, pub, ext, chem, same_lab, "TRUE Gd dip, all bench cells")
    pub_level(dip, pub, chem, same_lab, "TRUE Gd dip")

    # B0 OOF residual Gd chord on C3 cells (what B1 corrects), BPG, mean over seeds
    for design in ("BPG", "BP"):
        ref = pd.read_parquet(fr"{G21}\round4_20261022\predictions\REF_{design}.parquet")
        ref = ref[(ref.k == 3)]
        ref["res0"] = ref.y - ref.B0
        ref["pair"] = ref.A.astype(str) + "/" + ref.B.astype(str)
        a = ref[ref.pair == "Eu/Gd"].groupby("cell_id").res0.mean()
        b = ref[ref.pair == "Gd/Tb"].groupby("cell_id").res0.mean()
        chord = (a - b).dropna()
        meta = ref.drop_duplicates("cell_id").set_index("cell_id")
        meta = meta.loc[chord.index]
        cell_level(chord.to_numpy(), meta.publication.astype(str).to_numpy(), meta.extractant.astype(str).to_numpy(),
                   meta.chemotype.astype(str).to_numpy(), same_lab, f"B0 OOF residual Gd chord, C3, {design}")
        pub_level(chord.to_numpy(), meta.publication.astype(str).to_numpy(), meta.chemotype.astype(str).to_numpy(),
                  same_lab, f"B0 OOF residual Gd chord, {design}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
