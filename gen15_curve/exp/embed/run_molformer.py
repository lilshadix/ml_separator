"""The five-design MAE board for MoLFormer-XL, once its weights finally downloaded.

Same arm roles as ``run_score.py``'s ``main`` batch, with MoLFormer's mean-pooled vector as the
representation: direction at C in {0.01, 0.1, 1} and at 16 principal components, magnitude by ridge
and by the two kernels, curvature by ridge and by a sign logistic, the composite, and the
concatenation with gen14's own 39 topology columns.  FLAT and G14 ride along as the reproducibility
check.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arms_embed as E                                      # noqa: E402
from gen15 import arms as A, valuebench as V                 # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

PRIMARY = "molformer__mean"


def batch() -> dict:
    d = {"FLAT": A.flat, "G14": A.g14}
    d["DIR_mf_mean_C1"] = E.direction_arm(PRIMARY, C=1.0)
    d["DIR_mf_mean_C0.1"] = E.direction_arm(PRIMARY, C=0.1)
    d["DIR_mf_mean_C0.01"] = E.direction_arm(PRIMARY, C=0.01)
    d["DIR_mf_mean_P16"] = E.direction_arm(PRIMARY, C=1.0, n_pca=16)
    d["DIR_mf_cls_C1"] = E.direction_arm("molformer__cls", C=1.0)
    d["MAG_mf_mean_ridge"] = E.magnitude_arm(PRIMARY, how="ridge", alpha=100.0)
    d["MAG_mf_mean_rbf"] = E.magnitude_arm(PRIMARY, how="rbf", alpha=1.0)
    d["MAG_mf_mean_cos"] = E.magnitude_arm(PRIMARY, how="cosine", alpha=1.0)
    d["CUR_mf_mean_ridge"] = E.curvature_arm(PRIMARY, how="ridge", alpha=100.0)
    d["CUR_mf_mean_sign"] = E.curvature_arm(PRIMARY, how="sign", alpha=1.0)
    d["COMP_mf_mean"] = E.composite_arm(PRIMARY, C=1.0, how="ridge", alpha=100.0)
    d["CONCAT_mf_mean_P16"] = E.concat_arm(PRIMARY, C=1.0, n_pca=16)
    return d


def main() -> None:
    designs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)
    arms = batch()
    comps = {f"{k}_vs_G14": ("G14", k) for k in arms if k not in ("G14", "FLAT")}
    comps.update({f"{k}_vs_FLAT": ("FLAT", k) for k in arms if k not in ("G14", "FLAT")})
    bench = V.load()
    t0 = time.time()
    B, C, _ = V.score(bench, arms, designs, comps=comps)
    B.to_csv(OUT / "board_molformer.csv", index=False)
    if len(C):
        C.to_csv(OUT / "contrast_molformer.csv", index=False)
    print(f"\n=== molformer  extractant-macro MAE  ({time.time() - t0:.0f}s) ===")
    print(V.wide(B).round(4).to_string())
    print("\n--- macro_sign_acc_strong ---")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    if len(C):
        g = C[C.comparison.str.endswith("_vs_G14")].copy()
        p = g.pivot(index="candidate", columns="design", values="point")
        print("\n--- paired gain vs G14 (positive = better) ---")
        print(p[[d for d in V.DESIGNS if d in p.columns]].round(4).to_string())


if __name__ == "__main__":
    main()
