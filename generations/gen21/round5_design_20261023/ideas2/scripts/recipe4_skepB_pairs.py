"""Skeptic B, recipe-4: where the interior LINHALF - MEAN gain sits (pair type; Ce pairs; top publications).
EXPLORATORY (already-seen cells). Reuses recipe4_skepB_check builders; prints only."""
import sys
from pathlib import Path
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import recipe4_skepB_check as S
from gen20k import data as D
cells = pd.read_parquet(S.OUT_IN / "recipe_cells.parquet")
t = S.add_arms(S.build_rows(cells, D.load_bench()))
ti = t[t.inside].copy()
ti["g_LH"] = (ti.MEAN - ti.y) ** 2 - (ti.LINHALF - ti.y) ** 2
ti["g_PM"] = (ti.MEAN - ti.y) ** 2 - (ti.PAIRMEAN - ti.y) ** 2
tot = ti.g_LH.sum()
print("total gain", round(tot, 4))
print(ti.groupby("pair")[["g_LH", "g_PM"]].sum().assign(share=lambda d: d.g_LH / tot).round(3).sort_values("g_LH").to_string())
ce = ti.pair.isin(["La/Ce", "Ce/Pr"])
for lab, g in (("no Ce pairs", ti[~ce]), ("Ce pairs", ti[ce])):
    print(lab, "rows", len(g), "R2 MEAN", round(S.r2(g, "MEAN"), 3), "LINHALF", round(S.r2(g, "LINHALF"), 3),
          "PAIRMEAN", round(S.r2(g, "PAIRMEAN"), 3))
g = ti[~ti.publication.isin(["pub_19783c039b", "pub_8415538291"])]
print("without the top-2 pubs: rows", len(g), "R2 MEAN", round(S.r2(g, "MEAN"), 3), "LINHALF", round(S.r2(g, "LINHALF"), 3),
      "PAIRMEAN", round(S.r2(g, "PAIRMEAN"), 3), "NEAR", round(S.r2(g, "NEAR"), 3))
g = ti[~ti.publication.isin(["pub_1605e436d7"])]
print("without pub_1605e436d7: MEAN", round(S.r2(g, "MEAN"), 3), "NEAR", round(S.r2(g, "NEAR"), 3), "PAIRMEAN", round(S.r2(g, "PAIRMEAN"), 3))
# extractants and acid levels of the top-2 pubs
for p in ("pub_19783c039b", "pub_8415538291", "pub_1605e436d7"):
    gg = ti[ti.publication == p]
    print(p, gg.extractant.unique()[:3], "target acids", sorted(gg.target_acid_M.unique()), "sib levels", sorted(set(np.round(10 ** np.concatenate([np.asarray(x) for x in gg.la]), 4))))
# exterior: is LINCLIP's gain = copying the end level?
te = t[~t.inside].copy()
print("exterior NEAR (end level) R2", round(S.r2(te, "NEAR"), 3), "LINCLIP", round(S.r2(te, "LINCLIP"), 3), "MEAN", round(S.r2(te, "MEAN"), 3))
gx = te.assign(g=(te.MEAN - te.y) ** 2 - (te.LINCLIP - te.y) ** 2).groupby("publication").g.sum()
print("exterior LINCLIP-MEAN gain by pub", gx.round(4).to_dict())
