#!/usr/bin/env python
"""P5SK extra archive checks (EXPLORATORY; reads only, prints to stdout).

  - non-C5 C3 readings exactly at log D = -4.0 (is the "< -4" cut a knife edge?)
  - DMDPhPDA copy-A curator note on the 5 M La / Ce / Pr values (F8)
  - size / log-D range of the whole "updated ORNL" import channel (is it the ORNL ML compilation of JACS Au 2022?)
  - DOIs of the "updated ORNL" channel vs the 22 import DOIs
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
R4 = ROOT / "generations" / "gen21" / "round4_20261022"
sys.path.insert(0, str(R4))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from gen20k import data as D  # noqa: E402

LN = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")


def main():
    cells = pd.read_csv(R4 / "results" / "provenance" / "cells.csv")
    c3 = cells[cells.c3]
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    idx = {c: i for i, c in enumerate(f.cell_id)}
    Y = b.Y[[idx[c] for c in c3.cell_id]]
    nonc5 = (c3.publication_id != "pub_6a0ef8528c").to_numpy()
    eq = np.isclose(np.nan_to_num(Y, nan=0), -4.0, atol=1e-6) & nonc5[:, None]
    print("non-C5 C3 readings at log D == -4.0:", int(eq.sum()), " cells:", c3.cell_id[eq.any(1)].tolist(),
          " pubs:", c3.publication_id[eq.any(1)].tolist(), " metals:", [LN[j] for j in np.where(eq)[1]])
    print("non-C5 readings in (-4.0, -3.5]:", int(((Y > -4.0) & (Y <= -3.5) & nonc5[:, None]).sum()))
    a = pd.read_parquet(ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet")
    # F8
    for cid in ("ceb21275e22d7bc5", "7054e94bc4c6833f"):
        ids = [int(x.split(":")[-1]) for x in str(f.loc[idx[cid], "safe_exp_ids"]).split(";") if x]
        r = a[a.source_record_id.astype(int).isin(ids)]
        print(f"\nDMD cell {cid}: curator {r.entry_author.unique().tolist()} data_location {r.data_location.unique().tolist()}")
        print(r[r.metal_symbol.isin(["La", "Ce", "Pr", "Nd"])][["metal_symbol", "D_raw", "comments_raw"]].to_string(index=False))
    # ORNL channel
    o = a[a.comments_raw.astype(str).str.contains("updated ORNL", na=False)]
    print("\n'updated ORNL' archive rows:", len(o), " canonical:", int(o.is_canonical_row.sum()),
          " DOIs:", o.doi_primary.nunique(), " Ln rows:", int(o.is_lanthanide.sum()),
          " log D range:", float(o.log_D.min()), float(o.log_D.max()))
    oc = o[o.is_canonical_row & o.is_lanthanide]
    print("canonical Ln rows:", len(oc), " distinct extractant keys:", oc.extractant_system_key.nunique(),
          " n extractants per record:", oc.n_organic_extractants.value_counts().to_dict())
    print("entry authors:", o.entry_author.value_counts().to_dict(), " dates:",
          pd.to_datetime(o.addition_date).dt.date.astype(str).value_counts().to_dict())
    imp = set(c3[c3.ornl_import_all & c3.data_location.isna()].doi.str.lower())
    od = set(o.doi_primary.astype(str).str.lower())
    print("22 import DOIs all in ORNL channel:", imp <= od, len(imp & od))
    yr = pd.to_numeric(o.publication_year, errors="coerce")
    print("ORNL channel years:", yr.min(), yr.max())
    at4 = c3[c3.cell_id.isin(["7ad5ac5e8cdc02e6", "cc095c60c1836798", "f457e9a3d6532cc8"])]
    print("cells at exactly -4.0:", at4[["cell_id", "publication_id", "ornl_import_all", "data_location", "acid_M"]]
          .to_dict("records"))
    print("ORNL rows with log_D == -4.0 (whole channel):", int(np.isclose(o.log_D.astype(float), -4.0).sum()),
          " <= -3.9:", int((o.log_D.astype(float) <= -3.9).sum()))
    non = a[~a.comments_raw.astype(str).str.contains("updated ORNL|IDEaL", na=False) & a.is_lanthanide.astype(bool)]
    print("non-ORNL non-IDEaL Ln rows: log D min", float(non.log_D.astype(float).min()),
          " rows at exactly -4.0:", int(np.isclose(non.log_D.astype(float), -4.0).sum()))
    i = a[a.comments_raw.astype(str).str.fullmatch("IDEaL", na=False)]
    print("\nIDEaL rows:", len(i), " canonical:", int(i.is_canonical_row.sum()), " DOIs:", i.doi_primary.nunique(),
          " log D range:", float(i.log_D.min()), float(i.log_D.max()))


if __name__ == "__main__":
    main()
