"""Skeptic A for idea lab-4: author lists, C3 counts, chemotypes per publication (read-only). EXPLORATORY."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import pandas as pd, numpy as np
from r4 import paths
from gen20k import data as D
par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
f = D.load_bench().frame
src = pd.read_csv(r"D:\ml_separator_gh\generations\gen21\round4_20261022\results\provenance\sources_by_doi.csv")
reg = pd.read_csv(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\probes\P4_corpus_registry.csv")
au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
c3n = par[par.split_seed == par.split_seed.iloc[0]].groupby("publication").cell_id.nunique()
ncell = f.groupby("publication_id").size()
chem = f.groupby("publication_id").chemotype.agg(lambda s: ",".join(sorted(set(s))))
nm = f.groupby("publication_id").extractant_name.agg(lambda s: ",".join(sorted(set(map(str, s))))[:60])
for r in src.sort_values("doi").itertuples(index=False):
    for p in str(r.publication_ids).split(";"):
        p = p.strip()
        if p in ncell.index:
            print(f"{p} c3={int(c3n.get(p, 0)):2d} n={ncell[p]:2d} {r.year} {chem[p][:30]:30s} {nm[p]:60s} | {au.get(str(r.doi), 'NA')[:300]}")
