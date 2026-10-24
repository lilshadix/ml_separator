"""shift theme, step 9 (EXPLORATORY, threshold HINDSIGHT-adjacent: 0.30 decade = a factor of 2 in acid, picked after
seeing the step-8 bins): zero-read acid-only copy vs B1 with 4 reads when the target acid is within a factor of
2 of the acid-only siblings' (geometric) mean acid; sensitivity at 0.15 and 0.45 decades."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "shift_analyze.py"); sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
T = pd.read_parquet(sa.OUT / "shift_rows.parquet")
for design in ("BPG", "BP"):
    t = T[(T.design == design) & (T.n_AS >= 1)].copy()
    print(f"\n===== {design}")
    for thr in (0.15, 0.30, 0.45):
        for nm, d0 in (("no C5", t[~t.c5]), ("with C5", t)):
            d = d0[d0.dx_mean.abs() <= thr + 1e-9]
            far = d0[d0.dx_mean.abs() > thr + 1e-9]
            print(f"thr {thr} {nm}: near {sa.describe(d)} | far {sa.describe(far)}; near: ACOPY_0 {sa.r2(d, 'ACOPY_0'):.3f} "
                  f"B1@4 {sa.r2(d, 'B1_3'):.3f} SH@4 {sa.r2(d, 'SH_3'):.3f}; far: ACOPY_0 {sa.r2(far, 'ACOPY_0'):.3f} "
                  f"B1@4 {sa.r2(far, 'B1_3'):.3f} SH@4 {sa.r2(far, 'SH_3'):.3f}")
            if thr == 0.30:
                sa.infer(d, "ACOPY_0", "B1_3", f"near<=0.30 {nm}")
                sa.infer(far, "ACOPY_0", "B1_3", f"far>0.30 {nm}")
                sa.infer(d, "SH_3", "ACOPY_0", f"near<=0.30 {nm}")
