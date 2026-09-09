import pandas as pd, numpy as np
t = pd.read_parquet("D:/ml_separator_gh/gen13_separation/analysis/stage3/s3_direction_predictions.parquet")
print(t.shape, t.columns.tolist())
print(t.groupby(["design","model"]).size())
bp = t[(t.design=="BP")&(t.model=="donor_geometry")]
print("BP cells", bp.cell_id.nunique(), "ext", bp.extractant.nunique(), "chem", bp.chemotype.nunique())
print("folds per seed:", bp.groupby("split_seed")["fold"].nunique().to_dict())
print("rows per seed:", bp.groupby("split_seed").size().to_dict())
print("cells per seed:", bp.groupby("split_seed")["cell_id"].nunique().to_dict())
print("ext per seed:", bp.groupby("split_seed")["extractant"].nunique().to_dict())
# how many times is each cell predicted
print("times each cell predicted:", bp.groupby("cell_id").size().value_counts().to_dict())
for d in ("A","B","BP"):
    s = t[(t.design==d)&(t.model=="donor_geometry")]
    print(d, "folds:", s.groupby("split_seed")["fold"].nunique().to_dict(), "cells", s.cell_id.nunique())
print("\nalways_heavy p values by design:")
for d in ("A","B","BP"):
    s = t[(t.design==d)&(t.model=="always_heavy")]
    print(d, s.p.describe().round(4).to_dict(), " frac p>=0.5:", float((s.p>=0.5).mean()))
