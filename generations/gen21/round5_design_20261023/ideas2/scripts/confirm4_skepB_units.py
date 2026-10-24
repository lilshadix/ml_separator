#!/usr/bin/env python
"""Skeptic B for confirm-4 (C-ECT), check 1 (EXPLORATORY, already-seen 229 C3 cells, 5 dev seeds).
Per-unit table of the P1 PRIMARY stratum with donor class, mechanism tag (cation exchanger vs neutral, from SMILES
only), cpg cluster, change class and the per-unit log SSE ratios used by confirm-4; plus a bench-wide census of
families (publication x extractant, >= 3 cells) by mechanism and by recorded change type, to see whether the
T0 / T1-T3 split has any corpus rehearsal.
Reads P1 rows rebuilt by confirm (ideas2/scripts/confirm_out/p1).  Writes only ideas2/confirm4_skepB_out/.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm4_skepB_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from r4 import paths  # noqa: E402,F401
from r4 import donor as DN  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from gen20k import data as D  # noqa: E402
from rdkit import Chem  # noqa: E402

ACID_SMARTS = {"P_acid": "[PX4](=[O,S])[OX2H1,SX2H1]", "COOH": "[CX3](=O)[OX2H1]", "diketone": "[CX3](=O)[CH2,CH1][CX3](=O)",
               "SO3H": "S(=O)(=O)[OH]", "oxine": "[OH]c1cccc2cccnc12"}


def mech(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "unparsable"
    hits = [k for k, s in ACID_SMARTS.items() if m.HasSubstructMatch(Chem.MolFromSmarts(s))]
    if "." in smi:
        return "mixture:" + ("+".join(hits) if hits else "neutral")
    return ("cation_exch:" + "+".join(hits)) if hits else "neutral"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    t = pd.read_parquet(HERE.parent / "confirm_out" / "p1" / "P1_rows.parquet")
    t = t[t.design == "BPG"].copy()
    t = M4.add_clusters(t, f)
    t["dga"] = t.extractant.astype(str).map(DN.is_dga)
    t["mech"] = t.extractant.astype(str).map(mech)
    t["pubchem"] = t.publication + "|" + t.chemotype
    P = t[~t.t_v2drop & (t.n_clean >= 2) & ~t.c5].copy()

    def L(g, cand, ref="B1_3"):
        g = g.dropna(subset=[cand, ref])
        if g.empty:
            return np.nan
        s = g.assign(er=(g[ref] - g.y) ** 2, ec=(g[cand] - g.y) ** 2).groupby("split_seed")[["er", "ec"]].sum().mean()
        return float(np.log(s.er / s.ec))
    rows = []
    for u, g in P.groupby("pubchem"):
        rows.append({"unit": u, "cells": g.cell_id.nunique(), "rows_seed": len(g) / 5, "n_ext": g.extractant.nunique(),
                     "dga": g.dga.mean(), "mech": "|".join(sorted(g.mech.unique())), "cluster": g.cluster_cpg.iloc[0],
                     "cls": "|".join(f"{k}:{v}" for k, v in g.drop_duplicates("cell_id").cls.value_counts().items()),
                     "n_clean_med": g.drop_duplicates("cell_id").n_clean.median(),
                     "L_SH4": L(g, "CL_SH_3"), "L_SH3": L(g, "CL_SH_2"), "L_SH2": L(g, "CL_SH_1"),
                     "L_COPY": L(g, "CL_COPY"), "L_PBPUB": L(g, "CL_PB_PUB"), "L_SH4_vs_PBPUB": L(g, "CL_SH_3", "CL_PB_PUB"),
                     "L_SH4_vs_NEARSH": L(g, "CL_SH_3", "CL_NEARSH_3"), "L_SH4_vs_OFF": L(g, "CL_SH_3", "CL_OFF")})
    U = pd.DataFrame(rows).sort_values("L_SH4")
    U.to_csv(OUT / "units_primary.csv", index=False)
    print(U.round(3).drop(columns=["unit"]).assign(unit=U.unit.str.slice(0, 22)).to_string())
    print("\nPRIMARY extractants by mech:", P.drop_duplicates("cell_id").groupby(["mech", "dga"]).cell_id.size().to_dict())
    print("clusters rows/seed:", (P.groupby("cluster_cpg").size() / 5).to_dict())
    for lab, q in (("nonDGA", U[U.dga < 0.5]), ("DGA", U[U.dga >= 0.5]), ("giant_cluster", U[U.cluster == U.cluster.value_counts().idxmax()]),
                   ("not_giant", U[U.cluster != U.cluster.value_counts().idxmax()])):
        print(f"{lab:14s} units {len(q):2d}  SH4 W/L {(q.L_SH4 > 0).sum()}/{(q.L_SH4 < 0).sum()} mean {q.L_SH4.mean():+.3f}"
              f"  SH3 {(q.L_SH3 > 0).sum()}/{(q.L_SH3 < 0).sum()} mean {q.L_SH3.mean():+.3f}  COPY {(q.L_COPY > 0).sum()}/{(q.L_COPY < 0).sum()}")
    # ---- bench-wide family census by mechanism, all 521 cells: families with >= 3 cells (pub x extractant)
    cond = [c for c in f.columns if c.startswith("cond__")]
    f2 = f.assign(mech=f.extractant.astype(str).map(mech), dga=f.extractant.astype(str).map(DN.is_dga))
    Y = np.asarray(bench.Y, float)
    LN = list(D.LANTHANIDES)
    fam_rows = []
    for (p, e), g in f2.groupby(["publication_id", "extractant"]):
        if len(g) < 3:
            continue
        varying = [c for c in cond if g[c].astype(str).nunique() > 1]
        nmet = np.isfinite(Y[g.index]).sum(1)
        fam_rows.append({"pub": p, "mech": g.mech.iloc[0], "dga": g.dga.iloc[0], "cells": len(g), "chemotype": g.chemotype.iloc[0],
                         "full_ge10": int((nmet >= 10).sum()), "varying": ",".join(v.replace("cond__", "") for v in varying)})
    FR = pd.DataFrame(fam_rows)
    FR.to_csv(OUT / "bench_families.csv", index=False)
    print("\nbench families (pub x extractant, >= 3 cells):", len(FR))
    print(FR.groupby(["mech", "dga"]).agg(fams=("pub", "size"), pubs=("pub", "nunique"), cells=("cells", "sum"),
                                          full=("full_ge10", "sum")).to_string())
    print(FR[FR.mech != "neutral"].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
