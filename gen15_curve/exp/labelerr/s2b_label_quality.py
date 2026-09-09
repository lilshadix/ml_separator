"""Step 2b: is the de-noised label actually a *better estimate of the cell's own curve*?

The endpoint test in s2s3 answers "does training on de-noised labels help the deployed model".  It
does not answer the prior question: is the hierarchical estimate closer to the truth than the ridge
estimate at all?  That question can be settled without any model, and without any assumption about
sigma, by hold-out inside the cell:

    drop the two metals of the pair being scored, refit the cell's coefficients on what is left --
    once by the frozen per-cell ridge (penalty 0.5) and once by the hierarchical posterior with a
    prior estimated from the rest of the corpus -- and score both on the dropped pair.

If the two-stage label really is attenuated and noisy, the shrunk refit must win here.  The prior
is estimated on the whole corpus, which includes the cell being predicted; that is a small leak and
it favours the hierarchical arm, so a loss for it is conclusive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import RIDGE, shrink, sigma_table  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import _pair_frame  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bench = V.load()
    Y, basis = bench.Y, bench.basis
    C = Y - np.nanmean(Y, axis=1, keepdims=True)
    n, k = len(Y), basis.shape[0]
    se = sigma_table(bench, "resid_const")
    idx = np.arange(n)

    priors = {}
    for grp in ("extractant", "none"):
        _, info = shrink(bench, idx, se, group=grp)
        gcol = (np.full(n, "ALL") if grp == "none"
                else bench.frame[grp].astype(str).to_numpy())
        priors[grp] = (np.linalg.inv(info["T"]), info["mu"], gcol)
        print(f"prior fitted, group={grp}: T = {np.round(info['T'], 4).tolist()}, "
              f"converged={info['converged']}")

    t = _pair_frame(bench.frame, Y, idx, 0, 0)
    ia, ib, loc = t["ia"].to_numpy(), t["ib"].to_numpy(), t["cell_local"].to_numpy()
    out = t.drop(columns=["ia", "ib", "cell_local"]).copy()
    r1, r2 = basis[0], basis[1]
    d1, d2 = r1[ia] - r1[ib], r2[ia] - r2[ib]

    cache: dict[tuple, dict] = {}
    cols = {"RIDGE_LOPO": np.empty((len(out), k)), "EB_LIG_LOPO": np.empty((len(out), k)),
            "EB_GLOB_LOPO": np.empty((len(out), k))}
    for j in range(len(out)):
        key = (int(loc[j]), int(ia[j]), int(ib[j]))
        if key not in cache:
            i, a_, b_ = key
            o = ~np.isnan(Y[i])
            o[a_] = False
            o[b_] = False
            m = int(o.sum())
            if m == 0:
                cache[key] = {c: np.zeros(k) for c in cols}
            else:
                B = basis[:, o].T
                yc = C[i, o]
                ridge_c = np.linalg.solve(B.T @ B + RIDGE * np.eye(k), B.T @ yc)
                Mc = np.eye(m) - np.ones((m, m)) / m
                Bc, ycc = Mc @ B, Mc @ Y[i, o]
                w = 1.0 / np.maximum(se[i, o] ** 2, 1e-8)
                P = Bc.T @ (Bc * w[:, None])
                rr = Bc.T @ (ycc * w)
                d = {"RIDGE_LOPO": ridge_c}
                for name, grp in (("EB_LIG_LOPO", "extractant"), ("EB_GLOB_LOPO", "none")):
                    Ti, mu, gcol = priors[grp]
                    d[name] = np.linalg.solve(P + Ti, rr + Ti @ mu[gcol[i]])
                cache[key] = d
        for c in cols:
            cols[c][j] = cache[key][c]

    for c, v in cols.items():
        out[c] = v[:, 0] * d1 + v[:, 1] * d2
    out["FLAT"] = 0.0
    names = ["RIDGE_LOPO", "EB_LIG_LOPO", "EB_GLOB_LOPO", "FLAT"]
    sm = summarise(per_extractant(out, names), out, names)
    sm.to_csv(OUT / "s2b_label_quality.csv", index=False)
    print("\n=== which first-stage estimator predicts the cell's own held-out pair better? ===")
    print(sm[["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
              "macro_sign_acc_strong", "macro_pair_spearman"]].round(4).to_string(index=False))

    out["band"] = pd.cut(out["n_metals"], [1, 2, 4, 8, 14], labels=["2", "3-4", "5-8", "9-14"])
    rows = []
    for bnd, blk in out.groupby("band", observed=True):
        s = summarise(per_extractant(blk, names), blk, names)
        for _, rr in s.iterrows():
            rows.append({"band": str(bnd), "arm": rr["arm"],
                         "macro_mae": rr["macro_mae_extractant"]})
    bd = pd.DataFrame(rows).pivot(index="band", columns="arm", values="macro_mae")
    bd.to_csv(OUT / "s2b_label_quality_by_band.csv")
    print("\nby n_metals band (the thin cells are where shrinkage is supposed to pay)")
    print(bd.round(4).to_string())


if __name__ == "__main__":
    main()
