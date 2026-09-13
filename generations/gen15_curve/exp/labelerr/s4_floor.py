"""Step 4: the measurement-noise floor of the endpoint, by simulation.

The programme quotes 0.500 (gen14) against a "representation ceiling" of 0.181 (the cell's own two
coefficients).  Both are MAEs against *observed* log D.  Observed log D carries measurement noise,
so even a model that knew the true curve exactly could not score zero -- and 0.181 is an *in-sample*
number, fitted to the very values it is scored on, so it is not a ceiling a real model could reach
either.  This script measures both corrections.

Three quantities, all on the frozen pair rows and the frozen extractant-macro aggregation:

``FLOOR_noise``   truth is the fitted quadratic itself; observations are that curve plus fresh
                  per-metal noise.  A model that predicts the truth exactly still scores this.
                  It is the lowest extractant-macro MAE *any* model can reach.
``FLOOR_total``   truth is the observed cell means; observations are those plus fresh noise.  Adds
                  the basis misfit to the noise floor (and double counts the original noise inside
                  the misfit, so it is an over-estimate).
``O_BOTH_LOPO``   no simulation: for each held-out pair the cell's own coefficients are refitted
                  *without the two metals of that pair*.  The honest version of "the model knows
                  this cell's curve", and the gap to O_BOTH is how much of 0.181 is self-fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import RIDGE, pooled_residual_sigma, replicate_arrays, sigma_table  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import _pair_frame  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
N_DRAWS = 5
SEED = 20260909


def _table(bench, cols: dict[str, np.ndarray], draw: int) -> pd.DataFrame:
    """Pair table over every cell, with one prediction column per supplied 521x14 curve matrix."""
    idx = np.arange(len(bench.Y))
    t = _pair_frame(bench.frame, bench.Y, idx, draw, 0)
    ia, ib, loc = t["ia"].to_numpy(), t["ib"].to_numpy(), t["cell_local"].to_numpy()
    out = t.drop(columns=["ia", "ib", "cell_local"]).copy()
    for name, curve in cols.items():
        out[name] = curve[loc, ia] - curve[loc, ib]
    return out


def _fit(centred_row, basis, ridge=RIDGE, drop=()):
    m = ~np.isnan(centred_row)
    for j in drop:
        m[j] = False
    if m.sum() == 0:
        return np.zeros(basis.shape[0])
    B = basis[:, m].T
    return np.linalg.solve(B.T @ B + ridge * np.eye(basis.shape[0]), B.T @ centred_row[m])


def leave_pair_out(bench) -> pd.DataFrame:
    """O_BOTH refitted without the pair being scored, plus O_BOTH itself, on identical rows."""
    Y = bench.Y
    C = Y - np.nanmean(Y, axis=1, keepdims=True)
    idx = np.arange(len(Y))
    t = _pair_frame(bench.frame, Y, idx, 0, 0)
    ia, ib, loc = t["ia"].to_numpy(), t["ib"].to_numpy(), t["cell_local"].to_numpy()
    out = t.drop(columns=["ia", "ib", "cell_local"]).copy()
    full = np.vstack([_fit(C[i], bench.basis) for i in range(len(Y))]) @ bench.basis
    out["O_BOTH"] = full[loc, ia] - full[loc, ib]
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    pred = np.empty(len(out))
    for k in range(len(out)):
        i, a, b = int(loc[k]), int(ia[k]), int(ib[k])
        key = (i, a, b)
        if key not in cache:
            cache[key] = _fit(C[i], bench.basis, drop=(a, b)) @ bench.basis
        cv = cache[key]
        pred[k] = cv[a] - cv[b]
    out["O_BOTH_LOPO"] = pred
    # a milder version: drop only the lighter metal of the pair
    cache1: dict[tuple[int, int], np.ndarray] = {}
    pred1 = np.empty(len(out))
    for k in range(len(out)):
        i, a, b = int(loc[k]), int(ia[k]), int(ib[k])
        if (i, a) not in cache1:
            cache1[(i, a)] = _fit(C[i], bench.basis, drop=(a,)) @ bench.basis
        cv = cache1[(i, a)]
        pred1[k] = cv[a] - cv[b]
    out["O_BOTH_LOO1"] = pred1
    out["FLAT"] = 0.0
    return out


def main() -> None:
    bench = V.load()
    nr, _ = replicate_arrays(bench)
    obs = ~np.isnan(bench.Y)
    res_pool = pooled_residual_sigma(bench)
    inter = pd.read_csv(OUT / "s1b_summary.csv").pooled_interaction_sd.iat[0] \
        if (OUT / "s1b_summary.csv").exists() else 0.334

    sigmas = {"optimistic_0.10": 0.10,
              f"residual_bound_{res_pool:.3f}": res_pool,
              f"replicate_interaction_{inter:.3f}": float(inter),
              "naive_replicate_0.760": 0.760}

    truth_quad = np.where(obs, (bench.coef @ bench.basis), np.nan)
    truth_obs = bench.Y.copy()

    rng = np.random.default_rng(SEED)
    rows = []
    for label, s in sigmas.items():
        se = np.where(obs, s / np.sqrt(np.where(np.isfinite(nr), np.maximum(nr, 1.0), 1.0)), np.nan)
        for kind, truth in (("FLOOR_noise", truth_quad), ("FLOOR_total", truth_obs)):
            parts = []
            for d in range(N_DRAWS):
                e = rng.normal(size=bench.Y.shape) * np.nan_to_num(se)
                Yp = np.where(obs, truth + e, np.nan)
                if kind == "FLOOR_total":
                    Yp = np.where(obs, truth_obs + e, np.nan)
                    pred = np.where(obs, truth_obs, np.nan)
                else:
                    pred = np.where(obs, truth_quad, np.nan)
                b2 = type(bench)(frame=bench.frame, Y=Yp, basis=bench.basis, coef=bench.coef,
                                 n_obs=bench.n_obs, frames=bench.frames, groups=bench.groups)
                parts.append(_table(b2, {"ORACLE": np.nan_to_num(pred), "FLAT": np.zeros_like(pred)}, d))
            tab = pd.concat(parts, ignore_index=True)
            pe = per_extractant(tab, ["ORACLE", "FLAT"])
            sm = summarise(pe, tab, ["ORACLE", "FLAT"])
            rec = {"sigma_label": label, "sigma": s, "kind": kind}
            for _, r in sm.iterrows():
                rec[f"{r['arm']}_macro_mae"] = r["macro_mae_extractant"]
                rec[f"{r['arm']}_pooled_mae"] = r["pooled_mae"]
            rows.append(rec)
            print(f"  {label:>28s} {kind:>12s}  oracle macro {rec['ORACLE_macro_mae']:.4f} "
                  f"(flat on the same noisy labels {rec['FLAT_macro_mae']:.4f})", flush=True)
    fl = pd.DataFrame(rows)
    fl.to_csv(OUT / "s4_floor.csv", index=False)
    print("\n=== simulated measurement-noise floor of the endpoint ===")
    print(fl.round(4).to_string(index=False))

    # ---------------- the honest within-cell oracle ---------------------------------------
    print("\ncomputing the leave-pair-out oracle ...", flush=True)
    lp = leave_pair_out(bench)
    names = ["O_BOTH", "O_BOTH_LOO1", "O_BOTH_LOPO", "FLAT"]
    pe = per_extractant(lp, names)
    sm = summarise(pe, lp, names)
    sm.to_csv(OUT / "s4_lopo.csv", index=False)
    print("\n=== O_BOTH is fitted to the values it is scored on; here is the honest version ===")
    print(sm[["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
              "macro_sign_acc_strong"]].round(4).to_string(index=False))

    # by n_metals band, because a 2-metal cell has nothing left after dropping the pair
    lp["band"] = pd.cut(lp["n_metals"], [1, 2, 4, 8, 14], labels=["2", "3-4", "5-8", "9-14"])
    band = []
    for bnd, blk in lp.groupby("band", observed=True):
        pe_b = per_extractant(blk, names)
        sm_b = summarise(pe_b, blk, names)
        for _, r in sm_b.iterrows():
            band.append({"band": str(bnd), "arm": r["arm"], "n_pairs": len(blk),
                         "macro_mae": r["macro_mae_extractant"]})
    bd = pd.DataFrame(band).pivot(index="band", columns="arm", values="macro_mae")
    bd.to_csv(OUT / "s4_lopo_by_band.csv")
    print("\nby n_metals band")
    print(bd.round(4).to_string())


if __name__ == "__main__":
    main()
