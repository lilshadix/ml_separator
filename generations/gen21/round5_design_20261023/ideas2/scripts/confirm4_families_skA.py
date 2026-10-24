#!/usr/bin/env python
"""Skeptic A for idea confirm-4 (C-ECT), step 2 (EXPLORATORY; read-only on the V1 bench).

1. Which publication x extractant families in the whole 521-cell bench have >= 3 CLEAN conditions (siblings by the
   P1 rule, re-implemented here), by chem_family / chemotype / DGA flag; which are C3; which would be T0 (cation
   exchanger, acid-concentration / extractant-concentration-only change) under confirm-4's own metadata rule.
2. Near-copy diagnostics on every CLEAN target-sibling pair of C3 targets: max |dlogD|, sd of dlogD across shared metals
   (sd ~ 0 = the same SF pattern shifted by a constant = a mass-action-like / copy sibling), bit-identical series.
Writes nothing except stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4 import donor as DN  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

TRIV = ("cond__contact_time_min", "cond__metal_concentration_mM")


def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 70)
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond = [c for c in f.columns if c.startswith("cond__")]
    nontriv = [c for c in cond if c not in TRIV]
    C = f[cond].astype(float).to_numpy()
    ci = {c: k for k, c in enumerate(cond)}
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(ref.cell_id)
    print("chem_family values:")
    print(f.chem_family.value_counts().to_string())
    f["dga"] = f.extractant.astype(str).map(DN.is_dga)
    f["c3"] = f.cell_id.isin(c3)
    f["v2drop"] = f.cell_id.isin(drop)

    def differs(i, j, cols):
        out = []
        for c in cols:
            x, y = C[i, ci[c]], C[j, ci[c]]
            if np.isnan(x) and np.isnan(y):
                continue
            if np.isnan(x) or np.isnan(y) or not np.isclose(x, y):
                out.append(c)
        return out

    # ---- families (publication x extractant) with >= 3 non-dropped cells that differ in a non-trivial field
    fam = []
    for (p, e), g in f[~f.v2drop].groupby(["publication_id", "extractant"]):
        if len(g) < 3:
            continue
        idx = g.index.to_numpy()
        fields = set()
        for a in range(len(idx)):
            for c in range(a + 1, len(idx)):
                for col in differs(idx[a], idx[c], nontriv):
                    fields.add(col.split("__")[1] if col.count("__") >= 2 else col)
        fam.append({"pub": p, "chem_family": g.chem_family.iloc[0], "chemotype": g.chemotype.iloc[0],
                    "dga": bool(g.dga.iloc[0]), "cells": len(g), "c3_cells": int(g.c3.sum()),
                    "nmet_med": float(np.median(np.isfinite(Y[idx]).sum(1))),
                    "fields": ",".join(sorted(fields)), "ext": str(e)[:60]})
    fam = pd.DataFrame(fam).sort_values(["chem_family", "pub"])
    print(f"\n{len(fam)} pub x extractant families with >= 3 non-V2-dropped cells")
    print(fam.to_string())
    print("\nby chem_family (families / cells / C3 cells):")
    print(fam.groupby("chem_family").agg(fams=("pub", "size"), pubs=("pub", "nunique"), cells=("cells", "sum"),
                                         c3=("c3_cells", "sum")).to_string())

    # ---- near-copy diagnostics on CLEAN target-sibling pairs (targets = C3 cells)
    pos = {c: k for k, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    recs = []
    for t in sorted(c3):
        i = pos[t]
        met = np.where(np.isfinite(Y[i]))[0]
        cover = np.all(np.isfinite(Y[:, met]), axis=1)
        cover[i] = False
        for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]:
            dfm = differs(i, j, nontriv)
            clean = (t not in drop) and (ids[j] not in drop) and bool(dfm)
            d = Y[j, met] - Y[i, met]
            sf_i = np.diff(Y[i, met])
            sf_j = np.diff(Y[j, met])
            recs.append({"target": t, "sib": ids[j], "pub": pub[i], "clean": clean, "fields": ",".join(dfm),
                         "maxabs_dlogD": float(np.max(np.abs(d))), "sd_dlogD": float(np.std(d)),
                         "maxabs_dSF": float(np.max(np.abs(sf_j - sf_i))) if len(met) > 1 else np.nan,
                         "identical": bool(np.array_equal(Y[i, met], Y[j, met])),
                         "sib_c3": ids[j] in c3})
    r = pd.DataFrame(recs)
    rc = r[r.clean]
    print(f"\nCLEAN target-sibling pairs: {len(rc)} (targets {rc.target.nunique()})")
    print("bit-identical CLEAN pairs:", int(rc.identical.sum()))
    print("CLEAN pairs with maxabs dlogD < 0.02:", int((rc.maxabs_dlogD < 0.02).sum()))
    print("CLEAN pairs with sd dlogD < 0.02 (same SF pattern up to a constant):", int((rc.sd_dlogD < 0.02).sum()))
    print("CLEAN pairs with sd dlogD < 0.05:", int((rc.sd_dlogD < 0.05).sum()))
    print(rc.sort_values("sd_dlogD").head(40).round(4).to_string())
    print("\nper publication: CLEAN pairs, median sd_dlogD, share sd<0.05")
    print(rc.groupby("pub").agg(pairs=("sd_dlogD", "size"), med_sd=("sd_dlogD", "median"),
                                min_sd=("sd_dlogD", "min"),
                                share_lt05=("sd_dlogD", lambda x: float((x < 0.05).mean()))).round(3).to_string())
    # V1 (non-clean) identical pairs for reference
    print("\nnon-CLEAN pairs that are bit-identical:")
    print(r[~r.clean & r.identical][["target", "sib", "pub", "fields"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
