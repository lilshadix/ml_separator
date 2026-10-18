#!/usr/bin/env python
"""Step 1 of the brief: reproduce gen20's metal-model DOPT_Q baselines on BP (dev seeds) from the
cache and compare with gen20's published levels CSV."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from g21 import evaluate as E  # noqa: E402
from g21 import paths  # noqa: E402
from gen20k import data as D  # noqa: E402

bench = D.load_bench()
methods = E.gen20_baselines(bench)
t = E.outer_table(bench, "BP", paths.DEV_SEEDS, methods, ks=(0, 1, 2, 3))
E.assert_no_known_pair(t)
lv, wide = E.cohort_levels(t, list(methods), K=3, replicates=1000)
ref = pd.read_csv(paths.ROOT / "generations/gen20_kshot/results/discovery/levels_BP_metal_C3_DOPT_Q.csv")
ref = ref.set_index("arm")
out = []
for _, r in lv.iterrows():
    a = r["arm"]
    g = ref.loc[a] if a in ref.index else None
    out.append({"arm": a, "n_rows": r["n_rows"], "n_cells": r["n_cells"], "R2_g21": r["R2"], "R2_g20": None if g is None else g["R2"],
                "MAE_g21": r["MAE"], "MAE_g20": None if g is None else g["MAE"],
                "macro_g21": r["MAE_macro_extractant"], "macro_g20": None if g is None else g["MAE_macro_extractant"]})
o = pd.DataFrame(out)
paths.RESULTS.mkdir(parents=True, exist_ok=True)
o.to_csv(paths.RESULTS / "reproduction_gen20_BP_C3_DOPT_Q.csv", index=False)
pd.set_option("display.width", 200)
print(o.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
