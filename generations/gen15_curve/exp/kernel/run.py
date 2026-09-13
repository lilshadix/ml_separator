"""Run one named batch of kernel arms through all five designs and save the board.

    python run.py <batch>            # dir | mag | cur | joint | final

Every batch carries FLAT and G14 so each board is anchored to the two references the protocol
requires (the honest floor, and the cheapest sensible alternative already in the repo).  Boards and
the per-extractant frames are written to ``gen15_curve/exp/kernel/results/`` so any paired contrast
can be recomputed later without refitting.
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))

from gen15 import valuebench as V, arms as A          # noqa: E402
import kernels as KK                                   # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DESIGNS = ["B", "BR", "BQ", "A", "BP"]

REF = {"FLAT": A.flat, "G14": A.g14}

# --- ridge / penalty grids, fixed in advance and reported in full ----------------------
KRR_ALPHAS = [0.01, 0.1, 1.0, 10.0]
KLR_LAMS = [0.1, 1.0, 10.0, 100.0]
SVC_CS = [0.1, 1.0, 10.0]
LOGIT_CS = [0.01, 0.1, 1.0]
KS = [1, 3, 5, 10]


def batch_dir() -> dict:
    a = dict(REF)
    for lam in KLR_LAMS:
        a[f"D_TAN_KLR_l{lam:g}"] = KK.dir_kernel_logistic("TAN", lam)
    for C in SVC_CS:
        a[f"D_TAN_SVC_C{C:g}"] = KK.dir_kernel_svc("TAN", C)
    for al in KRR_ALPHAS:
        a[f"D_TAN_KRR_a{al:g}"] = KK.dir_kernel_ridge("TAN", al)
    for C in LOGIT_CS:
        a[f"D_ECFPz_LOGIT_C{C:g}"] = KK.dir_ecfp_logistic(C, scale=True)
        a[f"D_ECFPraw_LOGIT_C{C:g}"] = KK.dir_ecfp_logistic(C, scale=False)
    for k in KS:
        a[f"D_KNN_TAN_k{k}"] = KK.dir_knn(k, "TAN")
    return a


def batch_dir2() -> dict:
    """Descriptor kernels and the ECFP-plus-topology combination for the direction."""
    a = dict(REF)
    for lam in KLR_LAMS:
        a[f"D_RBF_KLR_l{lam:g}"] = KK.dir_kernel_logistic("RBF", lam)
        a[f"D_COMBO_KLR_l{lam:g}"] = KK.dir_kernel_logistic("COMBO", lam)
    for lam in (1.0, 10.0):
        a[f"D_MAT_KLR_l{lam:g}"] = KK.dir_kernel_logistic("MAT", lam)
        a[f"D_TOPOK_KLR_l{lam:g}"] = KK.dir_kernel_logistic("TOPO", lam)
    for al in (0.1, 1.0, 10.0):
        a[f"D_RBF_KRR_a{al:g}"] = KK.dir_kernel_ridge("RBF", al)
    for al in (1.0, 10.0, 100.0, 1000.0):
        a[f"D_LIG2D_RIDGE_a{al:g}"] = KK.dir_ridge_lig2d(al)
    return a


def batch_mag() -> dict:
    a = dict(REF)
    for al in KRR_ALPHAS:
        a[f"M_TAN_KRR_a{al:g}"] = KK.mag_kernel_ridge("TAN", al, log=True)
        a[f"M_TANlin_KRR_a{al:g}"] = KK.mag_kernel_ridge("TAN", al, log=False)
    for al in (0.1, 1.0, 10.0):
        a[f"M_RBF_KRR_a{al:g}"] = KK.mag_kernel_ridge("RBF", al, log=True)
    a["M_TAN_GP"] = KK.mag_gp("TAN")
    a["M_RBF_GP"] = KK.mag_gp("RBF")
    a["M_MAT_GP"] = KK.mag_gp("MAT")
    for k in KS:
        a[f"M_KNN_TAN_k{k}"] = KK.mag_knn(k, "TAN")
    for al in (10.0, 100.0, 1000.0):
        a[f"M_LIG2D_RIDGE_a{al:g}"] = KK.mag_ridge_lig2d(al)
    return a


def batch_joint() -> dict:
    """Both coefficients from the kernel, and the curvature-only arms."""
    a = dict(REF)
    for al in KRR_ALPHAS:
        a[f"J_TAN_KRR_a{al:g}"] = KK.joint_kernel_ridge("TAN", al)
    for al in (0.1, 1.0, 10.0):
        a[f"J_RBF_KRR_a{al:g}"] = KK.joint_kernel_ridge("RBF", al)
    a["J_MAT_KRR_a1"] = KK.joint_kernel_ridge("MAT", 1.0)
    a["J_TAN_GP"] = KK.joint_gp("TAN")
    a["J_RBF_GP"] = KK.joint_gp("RBF")
    for k in KS:
        a[f"J_KNN_TAN_k{k}"] = KK.joint_knn(k, "TAN")
    for al in (0.1, 1.0, 10.0):
        a[f"C_TAN_KRR_a{al:g}"] = KK.cur_kernel_ridge("TAN", al)
    a["C_RBF_KRR_a1"] = KK.cur_kernel_ridge("RBF", 1.0)
    a["C_TAN_GP"] = KK.cur_gp("TAN")
    for k in (3, 5, 10):
        a[f"C_KNN_TAN_k{k}"] = KK.cur_knn(k, "TAN")
    return a


BATCHES = {"dir": batch_dir, "dir2": batch_dir2, "mag": batch_mag, "joint": batch_joint}


def main(name: str, design: str) -> None:
    """One batch, ONE design, one process.

    The machine has 8 GB shared between four agents and a five-design run in a single process
    died on a 4 MB allocation, losing three completed designs with it.  A process per design caps
    the peak and makes the run resumable: a design whose board already exists is skipped.
    """
    bcsv = OUT / f"board_{name}_{design}.csv"
    if bcsv.exists():
        print(f"[{name}/{design}] already done, skipping", flush=True)
        return
    arms = BATCHES[name]()
    print(f"[{name}/{design}] {len(arms)} arms", flush=True)
    bench = V.load()
    t0 = time.time()
    B, _, tables = V.score(bench, arms, [design], verbose=True)
    B.to_csv(bcsv, index=False)
    with (OUT / f"perext_{name}_{design}.pkl").open("wb") as fh:
        pickle.dump(tables[design][1], fh, protocol=5)
    print(V.wide(B).round(4).to_string())
    print(f"[{name}/{design}] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
