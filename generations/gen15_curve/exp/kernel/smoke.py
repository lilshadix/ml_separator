"""Smoke test: one design, a handful of arms, plus sanity facts about the two blocks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen15 import valuebench as V, arms as A          # noqa: E402
import kernels as KK                                   # noqa: E402

bench = V.load()
print("cells", len(bench.frame), "extractants", bench.frame.extractant.nunique())

# --- Gram sanity ---------------------------------------------------------------------
class _Fake:
    pass
fake = _Fake(); fake.bench = bench
T = KK.tanimoto_gram(fake)
w = np.linalg.eigvalsh(T)
print("Tanimoto Gram: min eig %.3e  max %.3f  offdiag mean %.3f" %
      (w.min(), w.max(), T[np.triu_indices(len(T), 1)].mean()))

arms = {
    "FLAT": A.flat,
    "G14": A.g14,
    "D_TAN_KLR_l1": KK.dir_kernel_logistic("TAN", 1.0),
    "D_TAN_SVC_C1": KK.dir_kernel_svc("TAN", 1.0),
    "D_ECFP_LOGIT_C0.1": KK.dir_ecfp_logistic(0.1),
    "M_TAN_GP": KK.mag_gp("TAN"),
    "J_KNN_TAN_k5": KK.joint_knn(5, "TAN"),
    "J_RBF_KRR_a1": KK.joint_kernel_ridge("RBF", 1.0),
}
B, C, tables = V.score(bench, arms, ["B"], verbose=True)
print(V.wide(B).round(4).to_string())
print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
