"""Step 4c: the endpoint's measurement-noise floor as a function of sigma, and the sigma at which
that floor equals the numbers the programme quotes.

The floor is very nearly linear in sigma, so a grid plus a linear read-off answers the question the
programme actually needs: *is the 0.181 "representation ceiling" above or below the noise?*
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import replicate_arrays, sigma_table  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import _pair_frame  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
GRID = [0.05, 0.10, 0.13, 0.15, 0.172, 0.201, 0.25, 0.334, 0.50]
N_DRAWS = 5
SEED = 424242


def floor_at(bench, se: np.ndarray, obs: np.ndarray, truth: np.ndarray, rng, draws: int) -> dict:
    parts = []
    for d in range(draws):
        e = rng.normal(size=bench.Y.shape) * np.nan_to_num(se)
        Yp = np.where(obs, truth + e, np.nan)
        b2 = type(bench)(frame=bench.frame, Y=Yp, basis=bench.basis, coef=bench.coef,
                         n_obs=bench.n_obs, frames=bench.frames, groups=bench.groups)
        t = _pair_frame(bench.frame, Yp, np.arange(len(Yp)), d, 0)
        ia, ib, loc = t["ia"].to_numpy(), t["ib"].to_numpy(), t["cell_local"].to_numpy()
        tt = t.drop(columns=["ia", "ib", "cell_local"]).copy()
        cur = np.nan_to_num(truth)
        tt["ORACLE"] = cur[loc, ia] - cur[loc, ib]
        tt["FLAT"] = 0.0
        parts.append(tt)
    tab = pd.concat(parts, ignore_index=True)
    sm = summarise(per_extractant(tab, ["ORACLE", "FLAT"]), tab, ["ORACLE", "FLAT"]).set_index("arm")
    return {"floor_macro": float(sm.loc["ORACLE", "macro_mae_extractant"]),
            "floor_pooled": float(sm.loc["ORACLE", "pooled_mae"]),
            "flat_macro": float(sm.loc["FLAT", "macro_mae_extractant"])}


def main() -> None:
    bench = V.load()
    obs = ~np.isnan(bench.Y)
    nr, _ = replicate_arrays(bench)
    rep = np.where(np.isfinite(nr), np.maximum(nr, 1.0), 1.0)
    truth = np.where(obs, bench.coef @ bench.basis, np.nan)
    rng = np.random.default_rng(SEED)
    rows = []
    for s in GRID:
        se = np.where(obs, s / np.sqrt(rep), np.nan)
        r = floor_at(bench, se, obs, truth, rng, N_DRAWS)
        rows.append({"sigma": s, "sigma_kind": "global constant", **r})
        print(f"  sigma {s:.3f}  floor(macro) {r['floor_macro']:.4f}", flush=True)
    # per-cell sigma from each cell's own residual
    for mode in ("resid_cell",):
        se = sigma_table(bench, mode)
        r = floor_at(bench, se, obs, truth, rng, N_DRAWS)
        rows.append({"sigma": float(np.nanmedian(se)), "sigma_kind": mode, **r})
        print(f"  {mode:>12s} (median se {np.nanmedian(se):.3f})  floor(macro) {r['floor_macro']:.4f}")
    fl = pd.DataFrame(rows)
    fl.to_csv(OUT / "s4c_floor_grid.csv", index=False)
    print("\n=== measurement-noise floor of the extractant-macro MAE vs sigma ===")
    print(fl.round(4).to_string(index=False))

    g = fl[fl.sigma_kind == "global constant"]
    slope = float(np.polyfit(g.sigma, g.floor_macro, 1)[0])
    icpt = float(np.polyfit(g.sigma, g.floor_macro, 1)[1])
    print(f"\nfloor(macro) = {slope:.4f} * sigma + {icpt:.4f}  (R^2 "
          f"{np.corrcoef(g.sigma, g.floor_macro)[0, 1] ** 2:.5f})")
    for target, label in ((0.181, "O_BOTH, the quoted representation ceiling"),
                          (0.2736, "OWNBOTH_LOPO, the honest own-cell oracle"),
                          (0.322, "O_AMP"), (0.427, "O_CURV"), (0.500, "G14, deployed")):
        print(f"  sigma that would make the floor equal {target:.3f} "
              f"({label}): {(target - icpt) / slope:.3f}")


if __name__ == "__main__":
    main()
