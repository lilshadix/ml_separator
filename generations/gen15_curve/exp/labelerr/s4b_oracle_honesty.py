"""Step 4b: how much of every own-cell oracle is the oracle fitting its own measurement noise?

``O_AMP``, ``O_CURV`` and ``O_BOTH`` are all fitted to the same log D values they are then scored
against, so part of their advantage is the coefficient absorbing the noise of the very pair being
scored.  The clean correction needs no assumption about sigma at all: refit the cell's coefficients
*without the two metals of the pair being scored*, and score that.  What survives is the part of
the oracle a model could in principle reach.

Six arms on identical rows, three matched pairs:

    SIGN_CONSTB        perfect direction, constant magnitude, constant curvature   (~ O_SIGN)
    SIGN_OWNB          ... with the cell's own b, fitted in sample                 (~ O_CURV)
    SIGN_OWNB_LOPO     ... with the cell's own b, refitted without the scored pair
    OWNA_CONSTB        the cell's own a, in sample                                 (~ O_AMP)
    OWNA_LOPO          the cell's own a, refitted without the scored pair
    OWNBOTH / _LOPO    both coefficients                                           (~ O_BOTH)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import RIDGE  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import _pair_frame, cell_weights  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
MIN_METALS = 5


def _fit(row, basis, drop=()):
    m = ~np.isnan(row)
    for j in drop:
        m[j] = False
    if m.sum() == 0:
        return np.zeros(basis.shape[0])
    B = basis[:, m].T
    return np.linalg.solve(B.T @ B + RIDGE * np.eye(basis.shape[0]), B.T @ row[m])


def main() -> None:
    bench = V.load()
    Y, basis = bench.Y, bench.basis
    C = Y - np.nanmean(Y, axis=1, keepdims=True)
    n = len(Y)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    wr = cell_weights(bench.groups[rich], bench.n_obs[rich])
    w_all = cell_weights(bench.groups, bench.n_obs)
    MEANMAG = float(np.average(np.abs(bench.coef[rich, 0]), weights=wr))
    BCONST = float(np.average(bench.coef[:, 1], weights=w_all))
    print(f"global constants used by every arm: mean|a| (rich, chemotype-balanced) = {MEANMAG:.4f}, "
          f"mean b = {BCONST:.4f}")

    full = np.vstack([_fit(C[i], basis) for i in range(n)])
    sign = np.where(full[:, 0] < 0, -1.0, 1.0)

    t = _pair_frame(bench.frame, Y, np.arange(n), 0, 0)
    ia, ib, loc = t["ia"].to_numpy(), t["ib"].to_numpy(), t["cell_local"].to_numpy()
    out = t.drop(columns=["ia", "ib", "cell_local"]).copy()

    lopo = np.empty((len(out), 2))
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for k in range(len(out)):
        key = (int(loc[k]), int(ia[k]), int(ib[k]))
        if key not in cache:
            cache[key] = _fit(C[key[0]], basis, drop=(key[1], key[2]))
        lopo[k] = cache[key]

    r1, r2 = basis[0], basis[1]
    d1 = r1[ia] - r1[ib]
    d2 = r2[ia] - r2[ib]
    a_full, b_full = full[loc, 0], full[loc, 1]
    a_lopo, b_lopo = lopo[:, 0], lopo[:, 1]
    s = sign[loc]

    out["FLAT"] = 0.0
    out["SIGN_CONSTB"] = s * MEANMAG * d1 + BCONST * d2
    out["SIGN_OWNB"] = s * MEANMAG * d1 + b_full * d2
    out["SIGN_OWNB_LOPO"] = s * MEANMAG * d1 + b_lopo * d2
    out["OWNA_CONSTB"] = a_full * d1 + BCONST * d2
    out["OWNA_LOPO"] = a_lopo * d1 + BCONST * d2
    out["OWNBOTH"] = a_full * d1 + b_full * d2
    out["OWNBOTH_LOPO"] = a_lopo * d1 + b_lopo * d2
    names = ["FLAT", "SIGN_CONSTB", "SIGN_OWNB", "SIGN_OWNB_LOPO", "OWNA_CONSTB", "OWNA_LOPO",
             "OWNBOTH", "OWNBOTH_LOPO"]

    pe = per_extractant(out, names)
    sm = summarise(pe, out, names)
    sm.to_csv(OUT / "s4b_oracle_honesty.csv", index=False)
    print("\n=== own-cell oracles, in sample vs with the scored pair held out ===")
    print(sm[["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
              "macro_sign_acc_strong", "macro_pair_spearman"]].round(4).to_string(index=False))

    g = sm.set_index("arm")["macro_mae_extractant"]
    print("\nself-fitting inflation of each own-cell oracle (macro MAE, in sample -> honest)")
    for a_, b_, lab in ((("SIGN_OWNB"), "SIGN_OWNB_LOPO", "own curvature b  (the O_CURV claim)"),
                        ("OWNA_CONSTB", "OWNA_LOPO", "own amplitude a  (the O_AMP claim)"),
                        ("OWNBOTH", "OWNBOTH_LOPO", "both             (the O_BOTH ceiling)")):
        print(f"  {lab:38s} {g[a_]:.4f} -> {g[b_]:.4f}   inflation {g[b_] - g[a_]:+.4f}")
    print(f"\n  reference on the same rows: FLAT {g['FLAT']:.4f}, "
          f"perfect direction + constants {g['SIGN_CONSTB']:.4f}")

    # restrict to well-determined cells, where dropping two metals still leaves a curve
    sub = out[out["n_metals"] >= 8]
    pe2 = per_extractant(sub, names)
    sm2 = summarise(pe2, sub, names)
    sm2.to_csv(OUT / "s4b_oracle_honesty_m8plus.csv", index=False)
    print("\n=== same, cells with >= 8 measured metals only "
          f"({sub['cell_id'].nunique()} cells, {len(sub)} pairs) ===")
    print(sm2[["arm", "macro_mae_extractant", "pooled_mae"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
