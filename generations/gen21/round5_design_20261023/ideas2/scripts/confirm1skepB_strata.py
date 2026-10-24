#!/usr/bin/env python
"""confirm-1 Skeptic B (EXPLORATORY, seen cells).
(1) Who is in the heavy-selective non-DGA corpus stratum (extractant names, family, acid, n cells)?
(2) Variance components of the unit Gd chord by chemotype (is the 'lab' unit the right unit for a chemistry claim?).
(3) Size of C-GD's lab-level one-sample t when new labs cluster in a few chemotypes (null: class mean = beta).
Writes only ideas2/confirm1skepB_out/.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm1skepB_out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parents[3] / "round4_20261022"))
import numpy as np
import pandas as pd
from scipy import stats as SS
from r4 import paths
from r4 import donor as DN
from r3 import interp
from gen20k import data as D

BETA = -0.0923
CLIP = (-0.40, 0.20)
LN = list(D.LANTHANIDES)
XR = interp.X_R


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    t = (XR[LN.index("Gd")] - XR[LN.index("Eu")]) / (XR[LN.index("Tb")] - XR[LN.index("Eu")])
    c = Y[:, LN.index("Gd")] - ((1 - t) * Y[:, LN.index("Eu")] + t * Y[:, LN.index("Tb")])
    sl = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            sl[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    acid = [col for col in f.columns if col.startswith("cond__acid__")]
    acid_name = f[acid].idxmax(axis=1).where(f[acid].max(axis=1) > 0, "none").str.replace("cond__acid__", "")
    d = pd.DataFrame({"c": c, "slope": sl, "c3": f.cell_id.isin(set(ref.cell_id)),
                      "dga": f.extractant.astype(str).map(DN.is_dga), "pub": f.publication_id.astype(str),
                      "chemotype": f.chemotype.astype(str), "fam": f.chem_family.astype(str),
                      "name": f.extractant_name.astype(str), "acid": acid_name,
                      "acidM": f.cond__acid_concentration_M})
    d["unit"] = d.pub + "|" + d.chemotype
    d = d[d.c3 & np.isfinite(d.c)]
    u = d.groupby("unit").agg(c=("c", "mean"), hs=("slope", lambda s: (s > 0).mean()), slope=("slope", "median"),
                              dga=("dga", "mean"), n=("c", "size"), chemotype=("chemotype", "first"),
                              fam=("fam", "first"), name=("name", "first"), acid=("acid", "first"),
                              pub=("pub", "first"))
    u["cw"] = np.clip(u.c, *CLIP)
    u["cls"] = np.where(u.dga > 0.5, "DGA", np.where(u.hs > 0.5, "nonDGA_HS", "nonDGA_LS"))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
    tab = u.sort_values(["cls", "c"])[["cls", "c", "n", "slope", "chemotype", "fam", "acid", "name"]]
    tab.to_csv(OUT / "strata_units.csv")
    print(tab.round(3).to_string())
    print("\nHS by family:")
    print(u[u.cls == "nonDGA_HS"].groupby("fam").agg(units=("c", "size"), mean_cw=("cw", "mean"),
                                                      cells=("n", "sum")).round(3))
    # (2) variance components: chemotypes with >= 2 publication units (all classes)
    vc = u.groupby("chemotype").filter(lambda g: len(g) >= 2)
    print("\nchemotypes with >=2 pub units:", vc.chemotype.nunique(), "units", len(vc))
    for nm, q in (("all", vc), ("noOutlier", vc[vc.cw < 0.6])):
        grp = [g.cw.to_numpy() for _, g in q.groupby("chemotype")]
        k = len(grp); N = sum(len(x) for x in grp); gm = np.concatenate(grp).mean()
        ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in grp); ssw = sum(((x - x.mean()) ** 2).sum() for x in grp)
        msb, msw = ssb / (k - 1), ssw / (N - k)
        n0 = (N - sum(len(x) ** 2 for x in grp) / N) / (k - 1)
        s2b = max((msb - msw) / n0, 0)
        print(f"  {nm}: k={k} N={N} msb={msb:.4f} msw={msw:.4f} s2b={s2b:.4f} sd_between_chemotype={np.sqrt(s2b):.3f} "
              f"sd_within={np.sqrt(msw):.3f} ICC={s2b/(s2b+msw):.2f}")
    # all-unit chemotype ICC on non-DGA + DGA using the winsorised chord
    # (3) size of lab-level t under a chemotype-clustered null
    rng = np.random.default_rng(7)
    rows = []
    for sdw in (0.05, 0.08, 0.10):
        for icc in (0.3, 0.5, 0.69):
            sdb = np.sqrt(icc / (1 - icc)) * sdw
            for comp in ((2, 2, 2), (3, 2, 1), (4, 2), (3, 3), (2, 2, 2, 2), (3, 2, 2, 1), (4, 3, 2, 1, 1, 1, 1)):
                G = sum(comp); R = 20000
                cb = rng.normal(0, sdb, size=(R, len(comp)))
                X = np.concatenate([cb[:, [j]] + rng.normal(0, sdw, size=(R, m)) for j, m in enumerate(comp)], 1) + BETA
                tt = (X.mean(1) - BETA) / (X.std(1, ddof=1) / np.sqrt(G))
                size = float((tt < -SS.t.ppf(0.975, G - 1)).mean())
                rows.append({"sd_within": sdw, "icc": icc, "sd_between": round(sdb, 3), "comp": str(comp), "G": G,
                             "size_lab_t": size})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "clustered_null_size.csv", index=False)
    print("\nsize of lab-level one-sample t (nominal 0.025) when labs share chemotypes:")
    print(s.pivot_table(index=["comp", "G"], columns=["sd_within", "icc"], values="size_lab_t").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
