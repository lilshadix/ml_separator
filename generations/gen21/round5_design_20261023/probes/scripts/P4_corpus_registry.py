"""P4 (literature supply probe): build a label-blind DOI / author registry of the existing corpus.

Reads ONLY bibliographic / metal-identity metadata columns of the gen19 archive
(dataset_all_metals/clean/master_clean.parquet) and the round-4 cohort table
(round4_20261022/results/provenance/sources_by_doi.csv). No D / log D column is read.

Writes probes/P4_corpus_registry.csv (one row per archive DOI or non-DOI reference) and
probes/P4_corpus_surnames.csv (author surname -> number of archive DOIs), used to decide
whether a literature candidate's laboratory is already represented in the corpus.

Run from D:/ml_separator_gh:
    .venv/Scripts/python.exe generations/gen21/round5_design_20261023/probes/scripts/P4_corpus_registry.py
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
ARCH = ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet"
COH = ROOT / "generations" / "gen21" / "round4_20261022" / "results" / "provenance" / "sources_by_doi.csv"
OUT = Path(__file__).resolve().parents[1]

META_COLS = [
    "doi_primary_corrected", "doi_all", "reference_other", "publication_title", "reference_title",
    "publication_authors", "reference_authors", "publication_year", "reference_year",
    "reference_journal", "metal_symbol", "is_lanthanide", "entry_author", "extractant_primary_name",
]
FORBIDDEN = {"D_raw", "D_value", "log_D"}
assert not FORBIDDEN & set(META_COLS)


def norm_doi(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x)
    return x


def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def surnames(auth: str) -> set[str]:
    out = set()
    if not isinstance(auth, str):
        return out
    for a in re.split(r";|,| and ", auth):
        a = strip_acc(a).strip()
        if not a:
            continue
        toks = [t for t in re.split(r"[\s.]+", a) if t]
        if not toks:
            continue
        last = toks[-1]
        if len(last) <= 1:  # 'Mowafy E.A.' style
            last = toks[0]
        out.add(last.lower())
    return out


def main() -> None:
    df = pd.read_parquet(ARCH, columns=META_COLS)
    df["doi"] = df["doi_primary_corrected"].map(lambda v: norm_doi(v) if isinstance(v, str) and v.strip() else None)
    df["key"] = df["doi"].fillna("REF:" + df["reference_other"].fillna("unknown").astype(str).str[:80])
    ln = df["is_lanthanide"].astype(str).str.lower().eq("true")

    rows = []
    for key, g in df.groupby("key", sort=True):
        title = g["publication_title"].dropna().astype(str)
        title = title.iloc[0] if len(title) else (g["reference_title"].dropna().astype(str).iloc[0] if g["reference_title"].notna().any() else "")
        auth = "; ".join(sorted(set(g["reference_authors"].dropna().astype(str)) | set(g["publication_authors"].dropna().astype(str))))
        yr = g["reference_year"].dropna().astype(str)
        yr = yr.iloc[0] if len(yr) else (g["publication_year"].dropna().astype(str).iloc[0] if g["publication_year"].notna().any() else "")
        jr = g["reference_journal"].dropna().astype(str)
        lnm = sorted(set(g.loc[ln.loc[g.index], "metal_symbol"].dropna().astype(str)))
        rows.append({
            "key": key,
            "year": yr,
            "journal": jr.iloc[0] if len(jr) else "",
            "title": re.sub(r"\s+", " ", title)[:200],
            "authors": auth[:400],
            "n_rows": len(g),
            "n_ln_metals": len(lnm),
            "ln_metals": " ".join(lnm),
            "curators": "; ".join(sorted(set(g["entry_author"].dropna().astype(str)))),
            "extractants": "; ".join(sorted(set(g["extractant_primary_name"].dropna().astype(str))))[:200],
        })
    reg = pd.DataFrame(rows)

    coh = pd.read_csv(COH)
    coh["doi"] = coh["doi"].map(norm_doi)
    reg = reg.merge(coh[["doi", "c3_cells", "cells"]].rename(columns={"doi": "key", "cells": "v1_cells"}), on="key", how="left")
    reg["in_v1_cohort"] = reg["v1_cells"].notna()
    reg.to_csv(OUT / "P4_corpus_registry.csv", index=False)

    cnt: dict[str, int] = {}
    for a in reg["authors"]:
        for s in surnames(a):
            cnt[s] = cnt.get(s, 0) + 1
    sn = pd.DataFrame(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])), columns=["surname", "n_archive_refs"])
    sn.to_csv(OUT / "P4_corpus_surnames.csv", index=False)

    print("archive keys:", len(reg), "| with DOI:", reg["key"].str.startswith("10.").sum(),
          "| with >=4 Ln metals:", (reg["n_ln_metals"] >= 4).sum(),
          "| in V1 cohort:", int(reg["in_v1_cohort"].sum()), "| C3 DOIs:", int((reg["c3_cells"].fillna(0) > 0).sum()))
    print("cohort DOIs not matched in archive:", sorted(set(coh["doi"]) - set(reg["key"])))
    print("distinct surnames:", len(sn))


if __name__ == "__main__":
    main()
