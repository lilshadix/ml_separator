#!/usr/bin/env python
"""E-LAB (laboratory fingerprint calibration), check 0: inventory of the other-extractant same-publication channel.

EXPLORATORY: metadata of the 229 already-seen C3 cells (no label is used here except coverage).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_inventory.py

Prints, per publication with >= 2 extractants among its bench cells: C3 cells, C3 extractants, bench cells and
extractants, chemotypes, extractant names, source cluster (cluster_cpg) and whether it is C5-BTBP.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 200)


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True).copy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(par.cell_id)
    t = par.drop_duplicates("cell_id")[["cell_id", "publication", "chemotype", "extractant"]]
    t = M4.add_clusters(t, f)
    clu = dict(zip(t.cell_id, t.cluster_cpg))
    f["c3"] = f.cell_id.isin(c3)
    f["v2drop"] = f.cell_id.isin(v2drop)
    rows_per_cell = par[par.split_seed == par.split_seed.iloc[0]].groupby("cell_id").size().to_dict()
    out = []
    for p, g in f.groupby("publication_id"):
        gc = g[g.c3]
        if gc.empty:
            continue
        names = sorted(set(g.extractant_name.astype(str)))
        out.append({"pub": p, "c3_cells": len(gc), "c3_ext": gc.extractant.nunique(), "c3_clean": int((~gc.v2drop).sum()),
                    "bench_cells": len(g), "bench_ext": g.extractant.nunique(), "chemotypes": ",".join(sorted(set(gc.chemotype))),
                    "cluster": clu[gc.cell_id.iloc[0]], "rows_seed0": int(sum(rows_per_cell.get(c, 0) for c in gc.cell_id)),
                    "names": "; ".join(n[:22] for n in names[:6])})
    o = pd.DataFrame(out).sort_values(["bench_ext", "c3_cells"], ascending=False)
    print(o.to_string(index=False))
    mx = o[o.bench_ext >= 2]
    print(f"\npublications with C3 cells: {len(o)}; with >= 2 extractants in the bench: {len(mx)} "
          f"({mx.c3_cells.sum()} C3 cells, {mx.rows_seed0.sum()} rows in one seed of {o.rows_seed0.sum()}); with >= 2 C3 extractants: "
          f"{(o.c3_ext >= 2).sum()} ({o[o.c3_ext >= 2].c3_cells.sum()} cells)")
    print("clusters among multi-extractant publications:", mx.groupby("cluster").c3_cells.sum().to_dict())
    # per C3 cell: does it have another-extractant donor in the same publication (bench, not V2-dropped)?
    pub = f.publication_id.to_numpy()
    ext = f.extractant.to_numpy()
    Y = np.asarray(bench.Y, float)
    rec = []
    for i in np.where(f.c3)[0]:
        donors = np.where((pub == pub[i]) & (ext != ext[i]) & ~f.v2drop.to_numpy())[0]
        same = np.where((pub == pub[i]) & (ext == ext[i]) & ~f.v2drop.to_numpy() & (np.arange(len(f)) != i))[0]
        rec.append({"cell": f.cell_id[i], "pub": pub[i], "n_xext": len(donors), "n_xext_c3": int(f.c3.to_numpy()[donors].sum()),
                    "n_same": len(same), "n_metals": int(np.isfinite(Y[i]).sum()),
                    "same_chem_xext": int((f.chemotype.to_numpy()[donors] == f.chemotype[i]).sum())})
    r = pd.DataFrame(rec)
    print(f"\nC3 cells with >= 1 other-extractant same-publication donor: {(r.n_xext > 0).sum()} of {len(r)} "
          f"(of them with a same-chemotype homologue donor: {(r.same_chem_xext > 0).sum()}); pubs {r[r.n_xext > 0].pub.nunique()}")
    print(f"... with >= 1 other-extractant donor AND >= 1 same-extractant sibling: {((r.n_xext > 0) & (r.n_same > 0)).sum()}")
    print(f"... with other-extractant donor but NO same-extractant cell in the publication: {((r.n_xext > 0) & (r.n_same == 0)).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
