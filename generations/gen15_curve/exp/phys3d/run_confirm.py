"""Independent re-run of the phys3d headline rows under all five designs.

Six arms only -- the two references plus the four phys3d arms that came closest -- so the whole
five-design table is reproduced from scratch in minutes rather than the 36 min the 17-arm main
run costs.  Any disagreement with ``board_main.csv`` would mean the pipeline is not deterministic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))
from gen15 import arms as A, valuebench as V  # noqa: E402
import arms_phys3d as P  # noqa: E402

ARMS = {
    "FLAT": A.flat,
    "G14": A.g14,
    "P3_CURV_1D": P.curv_1d("P3_ALL"),
    "P3_MAG_1D": P.mag_1d("P3_ALL"),
    "P3_CURV_ALL": P.curv_arm("P3_ALL"),
    "P3_DIR_TOPO": P.dir_arm("TOPO39+P3_ALL"),
}

comps = {f"{k}_vs_G14": ("G14", k) for k in ARMS if k not in ("G14", "FLAT")}
comps["G14_vs_FLAT"] = ("FLAT", "G14")

t0 = time.time()
bench = V.load()
B, C, tables = V.score(bench, ARMS, ["B", "BR", "BQ", "A", "BP"], comps=comps)
print(f"\ntotal {time.time() - t0:.0f}s")
for col in ("macro_mae_extractant", "macro_sign_acc_strong", "macro_pair_spearman"):
    print(f"\n=== {col} ===")
    print(V.wide(B, col).round(4).to_string())
print("\n=== chemotype-blocked paired bootstrap ===")
print(C.round(4).to_string(index=False))
B.to_csv(HERE / "board_confirm.csv", index=False)
C.to_csv(HERE / "contrasts_confirm.csv", index=False)
