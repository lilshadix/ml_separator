"""Score the phys3d block on the gen15 curve bench under all five designs.

Usage:  run_bench.py [smoke|main|attrib]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))
from gen15 import arms as A, valuebench as V  # noqa: E402
import arms_phys3d as P  # noqa: E402


def arm_set(which: str) -> dict:
    if which == "smoke":
        return {"FLAT": A.flat, "G14": A.g14,
                "P3_DIR_ALL": P.dir_arm("P3_ALL"),
                "P3_DIR_TOPO": P.dir_arm("TOPO39+P3_ALL"),
                "P3_CURV_ALL": P.curv_arm("P3_ALL")}
    if which == "main":
        return {
            "FLAT": A.flat,
            "G14": A.g14,
            # --- direction -------------------------------------------------------------
            "P3_DIR_ALL": P.dir_arm("P3_ALL"),
            "P3_DIR_PHYS": P.dir_arm("P3_PHYS"),
            "P3_DIR_TOPO": P.dir_arm("TOPO39+P3_ALL"),
            # --- magnitude -------------------------------------------------------------
            "P3_MAG_ALL": P.mag_arm("P3_ALL"),
            "P3_MAG_TOPO": P.mag_arm("TOPO39+P3_ALL"),
            "TOPO_MAG": P.mag_arm("TOPO39"),
            "P3_MAG_1D": P.mag_1d("P3_ALL"),
            # --- curvature -------------------------------------------------------------
            "P3_CURV_ALL": P.curv_arm("P3_ALL"),
            "P3_CURV_TOPO": P.curv_arm("TOPO39+P3_ALL"),
            "TOPO_CURV": P.curv_arm("TOPO39"),
            "P3_CURV_1D": P.curv_1d("P3_ALL"),
            # --- everything at once ----------------------------------------------------
            "P3_COMP_ALL": P.composite("P3_ALL"),
            "P3_COMP_TOPO": P.composite("TOPO39+P3_ALL"),
            "P3_TREE_ALL": P.tree_arm("P3_ALL"),
            "P3_TREE_TOPO": P.tree_arm("TOPO39+P3_ALL"),
        }
    if which == "attrib":
        return {
            "FLAT": A.flat,
            "G14": A.g14,
            "P3_DIR_STATIC": P.dir_arm("P3_STATIC"),
            "P3_DIR_RESP": P.dir_arm("P3_RESP"),
            "P3_DIR_ENERGY": P.dir_arm("P3_ENERGY"),
            "P3_DIR_SUPPORT": P.dir_arm("P3_SUPPORT"),
            "P3_CURV_STATIC": P.curv_arm("P3_STATIC"),
            "P3_CURV_RESP": P.curv_arm("P3_RESP"),
            "P3_CURV_ENERGY": P.curv_arm("P3_ENERGY"),
            "P3_MAG_STATIC": P.mag_arm("P3_STATIC"),
            "P3_MAG_RESP": P.mag_arm("P3_RESP"),
            "P3_MAG_ENERGY": P.mag_arm("P3_ENERGY"),
        }
    raise SystemExit(f"unknown arm set {which!r}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    designs = ["B", "BR", "BQ", "A", "BP"] if which != "smoke" else ["BP"]
    arms = arm_set(which)
    comps = {f"{k}_vs_G14": ("G14", k) for k in arms if k not in ("G14", "FLAT")}
    comps["G14_vs_FLAT"] = ("FLAT", "G14")
    if which == "smoke":
        comps.update({f"{k}_vs_FLAT": ("FLAT", k) for k in arms if k not in ("G14", "FLAT")})
    t0 = time.time()
    bench = V.load()
    B, C, tables = V.score(bench, arms, designs, comps=comps)
    print(f"\ntotal {time.time() - t0:.0f}s")
    for col in ("macro_mae_extractant", "macro_sign_acc_strong", "macro_pair_spearman"):
        print(f"\n=== {col} ===")
        print(V.wide(B, col).round(4).to_string())
    print("\n=== chemotype-blocked paired bootstrap (positive delta = the arm is better) ===")
    print(C.round(4).to_string(index=False))
    B.to_csv(HERE / f"board_{which}.csv", index=False)
    C.to_csv(HERE / f"contrasts_{which}.csv", index=False)
    for d, (table, pe) in tables.items():
        pe.to_csv(HERE / f"per_extractant_{which}_{d}.csv", index=False)


if __name__ == "__main__":
    main()
