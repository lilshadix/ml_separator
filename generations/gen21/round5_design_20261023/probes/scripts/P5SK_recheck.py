#!/usr/bin/env python
"""P5SK (skeptic of P5): independent recomputation of P5's headline accounting numbers and provenance facts.

EXPLORATORY: all numbers are on the 229 C3 cells / 5 dev seeds already scored in rounds 3-4.
Written from scratch (does not import P5_accounting.py).  Reads only; prints to stdout.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore \
      generations/gen21/round5_design_20261023/probes/scripts/P5SK_recheck.py
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

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 90)
LN = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
C5 = "pub_6a0ef8528c"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    cells = pd.read_csv(R4 / "results" / "provenance" / "cells.csv")
    c3 = cells[cells.c3].copy()
    noloc = c3.data_location.isna()
    S = {
        "ornl54": set(c3.cell_id[c3.ornl_import_all & noloc]),
        "c5": set(c3.cell_id[c3.publication_id == C5]),
        "noloc79": set(c3.cell_id[noloc]),
        "located150": set(c3.cell_id[~noloc]),
    }
    S["true69"] = S["ornl54"] | S["c5"]
    S["comment10"] = S["noloc79"] - S["true69"]
    print({k: len(v) for k, v in S.items()})
    print("ornl54 & c5 overlap:", len(S["ornl54"] & S["c5"]))
    print("ornl_import_all & located (C3):", int((c3.ornl_import_all & ~noloc).sum()))

    print("\n== A. shares (independent) ==")
    for d in ("BPG", "BP"):
        t = pd.read_parquet(R4 / "predictions" / f"REF_{d}.parquet")
        t = t[t.k == 3]
        assert t.groupby("split_seed").size().eq(1802).all()
        for name in ("true69", "ornl54", "c5", "noloc79", "comment10", "located150"):
            sh, gs, rows = [], [], []
            for s, g in t.groupby("split_seed"):
                m = g.cell_id.isin(S[name])
                e1 = (g.y - g.B1) ** 2
                e0 = (g.y - g.B0) ** 2
                sh.append(e1[m].sum() / e1.sum())
                gs.append((e0[m].sum() - e1[m].sum()) / (e0.sum() - e1.sum()))
                rows.append(m.sum())
            print(f"{d} {name:11s} rows/seed {np.mean(rows):7.1f}  SSE share {np.mean(sh):.4f} [{min(sh):.4f},{max(sh):.4f}]"
                  f"  gain share {np.mean(gs):.4f} [{min(gs):.4f},{max(gs):.4f}]")
        # located-160 rows: B0/B1 R2 per seed
        out = []
        for s, g in t.groupby("split_seed"):
            g2 = g[~g.cell_id.isin(S["true69"])]
            out.append((s, r2(g2.y, g2.B0), r2(g2.y, g2.B1), r2(g.y, g.B0), r2(g.y, g.B1)))
        o = pd.DataFrame(out, columns=["seed", "B0_loc160", "B1_loc160", "B0_all", "B1_all"])
        o["d_loc160"] = o.B1_loc160 - o.B0_loc160
        print(o.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(o.mean(numeric_only=True).round(4).to_dict())
        # gain per seed in SSE units
        tot = [((g.y - g.B0) ** 2).sum() - ((g.y - g.B1) ** 2).sum() for _, g in t.groupby("split_seed")]
        print("B0-B1 SSE gain per seed:", np.round(tot, 3), " B1 SSE per seed:",
              np.round([((g.y - g.B1) ** 2).sum() for _, g in t.groupby("split_seed")], 2))

    print("\n== B. floor readings ==")
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    idx = {c: i for i, c in enumerate(f.cell_id)}
    Y3 = bench.Y[[idx[c] for c in c3.cell_id]]
    low = np.nan_to_num(Y3, nan=0.0) < -4.0
    print("C3 readings < -4:", int(low.sum()), " cells:", int(low.any(1).sum()))
    pubs = c3.publication_id.to_numpy()
    print("publications of those cells:", pd.Series(pubs[low.any(1)]).value_counts().to_dict())
    c5i = (c3.publication_id == C5).to_numpy()
    print("C5 finite readings:", int(np.isfinite(Y3[c5i]).sum()), " <-4:", int(low[c5i].sum()),
          " <-3.5:", int((np.nan_to_num(Y3[c5i], nan=0) < -3.5).sum()))
    print("non-C5 min log D:", float(np.nanmin(Y3[~c5i])))
    # all bench cells (not only C3)
    lowall = np.nan_to_num(bench.Y, nan=0.0) < -4.0
    print("all 521 bench cells: readings < -4:", int(lowall.sum()), " pubs:",
          f.publication_id[lowall.any(1)].value_counts().to_dict())

    # floor drop scenario, BPG
    for d in ("BPG", "BP"):
        t = pd.read_parquet(R4 / "predictions" / f"REF_{d}.parquet")
        t = t[t.k == 3].reset_index(drop=True)
        yA = np.array([bench.Y[idx[c], LN.index(a)] for c, a in zip(t.cell_id, t.A)])
        yB = np.array([bench.Y[idx[c], LN.index(b)] for c, b in zip(t.cell_id, t.B)])
        assert np.allclose(yA - yB, t.y, atol=1e-9)
        inC5 = t.cell_id.isin(S["c5"]).to_numpy()
        cens = inC5 & ((yA < -4) | (yB < -4))
        r = [(r2(g.y, g.B0), r2(g.y, g.B1)) for _, g in t[~cens].groupby("split_seed")]
        r = np.array(r)
        print(f"{d} floor -4 drop: rows/seed {(~cens).sum()/5:.0f}  B0 {r[:,0].mean():.4f}  B1 {r[:,1].mean():.4f}"
              f"  d {np.mean(r[:,1]-r[:,0]):+.4f} (min {np.min(r[:,1]-r[:,0]):+.4f})")
        # same number of rows dropped at random from non-C5 cells? (cheap control: drop the rows with largest B1 error)
        rr = []
        for s, g in t.groupby("split_seed"):
            n = int(cens[g.index].sum())
            e = (g.y - g.B1) ** 2
            keep = ~g.index.isin(e.sort_values(ascending=False).index[:n])
            rr.append((r2(g.y[keep], g.B0[keep]), r2(g.y[keep], g.B1[keep])))
        rr = np.array(rr)
        print(f"   control: drop the same number of rows with the largest B1 error (any source): B1 {rr[:,1].mean():.4f}")

    print("\n== C. archive fields ==")
    a = pd.read_parquet(ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet")
    a["sid"] = a.source_record_id.astype(int)

    def recs(cell_ids):
        ids = []
        for s in f.set_index("cell_id").loc[list(cell_ids), "safe_exp_ids"]:
            ids += [int(x.split(":")[-1]) for x in str(s).split(";") if x]
        return a[a.sid.isin(ids)]

    # F1: comment-located 10 cells
    r = recs(S["comment10"])
    print("F1 comment10: n records", len(r))
    m = f.set_index("cell_id").loc[list(S["comment10"])]
    for cid in sorted(S["comment10"]):
        rr = recs([cid])
        print("  ", cid, c3.set_index("cell_id").loc[cid, "publication_id"], "| curator:",
              rr.entry_author.unique().tolist(), "| comments_raw:", rr.comments_raw.unique().tolist()[:4],
              "| data_location:", rr.data_location.unique().tolist())
    # F2/F3: C5 records
    r = recs(S["c5"])
    print("\nF2 C5 cells: records", len(r), " entry_author:", r.entry_author.value_counts().to_dict())
    print("  addition_date:", r.addition_date.astype(str).value_counts().head(5).to_dict())
    for col in ("comments_raw", "reference_other", "data_location", "sub_source_file", "source_file", "flags",
                "temperature_C", "contact_time_min", "phase_ratio_org_aq", "solvent_key", "doi_primary",
                "duplicate_group_size", "duplicate_class"):
        print(f"  {col}:", r[col].astype(str).value_counts().head(4).to_dict())
    doi = r.doi_primary.iloc[0]
    alld = a[a.doi_primary == doi]
    print(f"  all archive rows with DOI {doi}:", len(alld), " unique (metal, acid, ext conc, D):",
          alld[["metal_symbol", "acid_concentration_M", "extractant_primary_concentration_M", "D_raw"]]
          .drop_duplicates().shape[0],
          " canonical rows:", int(alld.is_canonical_row.sum()),
          " metals:", sorted(alld.metal_symbol.unique()))
    print("  dup group size:", alld.duplicate_group_size.value_counts().to_dict(),
          " class:", alld.duplicate_class.value_counts().to_dict())
    print("  solvents:", alld.solvent_key.value_counts().to_dict())
    # tags anywhere mentioning IDEaL
    hit = a[a.astype(str).apply(lambda col: col.str.contains("IDEaL", case=False, na=False)).any(axis=1)]
    print("  archive rows mentioning IDEaL:", len(hit), " DOIs:", hit.doi_primary.value_counts().head(5).to_dict())
    cols_hit = [c for c in a.columns if a[c].astype(str).str.contains("IDEaL", case=False, na=False).any()]
    print("  columns containing 'IDEaL':", cols_hit, hit[cols_hit].astype(str).drop_duplicates().head(3).to_dict("records"))
    # ORNL tag
    o = recs(S["ornl54"])
    cols_o = [c for c in a.columns if o[c].astype(str).str.contains("ORNL", case=False, na=False).any()]
    print("\nORNL-54 records:", len(o), " cols mentioning ORNL:", cols_o, " entry_author:",
          o.entry_author.value_counts().to_dict(), " addition dates:", o.addition_date.astype(str).value_counts().head(3).to_dict())
    print("  ORNL tag values:", o[cols_o].astype(str).drop_duplicates().head(3).to_dict("records") if cols_o else None)
    # F5: Foreman DOI in cohort?
    dois = set(c3.doi.dropna().str.lower())
    print("\nF5: C3 cohort DOIs:", len(dois), " Foreman 2005 in C3:", "10.1080/07366290500230735" in dois,
          " in archive:", (a.doi_primary.astype(str).str.lower() == "10.1080/07366290500230735").sum())
    print("   C5 pub DOI:", c3[c3.publication_id == C5].doi.unique(), " extractants:",
          c3[c3.publication_id == C5].extractant_name.unique(), " chemotype:", c3[c3.publication_id == C5].chemotype.unique())
    # F6
    for pub in ("pub_84113c17ad", "pub_1927dff570", "pub_40c355ae86"):
        cc = c3[c3.publication_id == pub]
        rr = recs(cc.cell_id)
        print(f"\nF6/F7 {pub}: doi {cc.doi.unique()} title {cc.title.unique()} journal {cc.journal.unique()} year {cc.year.unique()}")
        print("   cells:", cc[["cell_id", "extractant_name", "solvent", "aqueous_complexant", "acid_M", "extractant_M"]].to_string(index=False))
        print("   raw extractant:", rr.extractant_name_raw.unique().tolist()[:6], " raw complexant:",
              rr.complexant_name_raw.unique().tolist()[:6], " solvent raw:", rr.solvent_name_raw.unique().tolist()[:4])
        print("   Dy D_raw:", rr[rr.metal_symbol == "Dy"][["sid", "D_raw"]].to_dict("records"))


if __name__ == "__main__":
    main()
