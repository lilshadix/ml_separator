#!/usr/bin/env python
"""Round-2 diagnoses around attempt 6, from saved predictions and fitted parameters only.
Output: results/diag/diag_a6.txt

1. before attempt 6 (its motivation): per-pair-type linear calibration of HBK@ML at k = 3 (BP, C3):
   intercept, slope, in-sample share of squared error an intercept + slope would remove
2. after attempt 6: stability of the fitted metal effects alpha over the 25 BP folds, and per-pair
   MAE of HBK@ML against HBKM[mean] / HBKM@BIC
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g21 import paths  # noqa: E402

L = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
out = io.StringIO()
with redirect_stdout(out):
    t = pd.read_parquet(paths.PREDICTIONS / "attempt4_BP.parquet")
    t = t[(t.k_max >= 3) & (t.k == 3)]
    rows = []
    for (a, b), g in t.groupby(["A", "B"], sort=False):
        x, y = g["HBK@ML"].to_numpy(), g.y.to_numpy()
        X = np.c_[np.ones_like(x), x]
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append({"pair": f"{a}/{b}", "n": len(g), "intercept": c[0], "slope": c[1],
                     "sse_share_removed_in_sample": 1 - ((y - X @ c) ** 2).sum() / ((y - x) ** 2).sum()})
    print("1. HBK@ML (attempt 4), k=3, C3, BP: per-pair calibration (in-sample, diagnosis only)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    f = pd.read_csv(paths.RESULTS / "attempt6" / "fitted_BP.csv")
    A = np.array([json.loads(x) for x in f["HBKM[mean].alpha_metal"]])
    print("\n2. fitted metal effects alpha (HBKM[mean]) over the 25 BP folds")
    print(pd.DataFrame({"mean": A.mean(0), "sd": A.std(0), "min": A.min(0), "max": A.max(0)}, index=L).round(3).T.to_string())
    print("BIC choice:", f.bic_variant.value_counts().to_dict())
    pp = pd.read_csv(paths.RESULTS / "attempt6" / "per_pair_BP_C3_k3.csv")
    print("\nper-pair MAE at k=3 (BP, C3)")
    print(pp.pivot_table(index="pair", columns="arm", values="MAE", sort=False)[["HBK@ML", "HBKM[mean]", "HBKM@BIC"]].round(4).to_string())
(paths.RESULTS / "diag" / "diag_a6.txt").write_text(out.getvalue(), encoding="utf-8")
print(out.getvalue())
