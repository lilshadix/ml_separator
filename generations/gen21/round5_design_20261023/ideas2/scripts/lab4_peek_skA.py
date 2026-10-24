"""Skeptic A for idea lab-4: peek at inputs (read-only). EXPLORATORY."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import pandas as pd, numpy as np
from r4 import paths
from r3 import datav2 as V2
from gen20k import data as D
par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
print(par.columns.tolist()); print(par.head(3).T)
print(len(par), par.split_seed.unique(), par.cell_id.nunique())
print("V2 sizes", len(V2.V2_1_CELLS), len(V2.V2_2_DROP), len(V2.V2_3_CELLS))
print([n for n in dir(V2) if n.isupper()])
b = D.load_bench(); f = b.frame
print(f.shape, f.columns.tolist()[:40])
print(f.publication_id.nunique())
src = pd.read_csv(r"D:\ml_separator_gh\generations\gen21\round4_20261022\results\provenance\sources_by_doi.csv")
print(src.columns.tolist()); print(src.head(3).T)
from r4 import metrics
print([n for n in dir(metrics) if 'clus' in n.lower()])

# --- authors of the 58 bench publications ---
reg = pd.read_csv(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\probes\P4_corpus_registry.csv")
au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
pubs = set(f.publication_id.astype(str))
multi = src[src.publication_ids.astype(str).str.contains(";")]
print("DOIs mapping to >1 publication_id:", len(multi)); print(multi[["doi", "publication_ids"]].to_string())
c3 = set(par.publication)
for r in src.itertuples(index=False):
    ps = [p.strip() for p in str(r.publication_ids).split(";")]
    if any(p in pubs for p in ps):
        print(ps, "C3" if any(p in c3 for p in ps) else "  ", r.year, "|", au.get(str(r.doi), "NA")[:160])
fr = metrics.add_clusters(par[par.split_seed == par.split_seed.iloc[0]].copy()) if hasattr(metrics, "add_clusters") else None
print(fr.columns.tolist() if fr is not None else None)
print(metrics.CLUSTERINGS)
