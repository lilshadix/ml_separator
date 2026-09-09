"""Step 1c: is the residual about the fitted quadratic white noise, or real non-quadratic chemistry?

The whole noise accounting rests on sigma.  The corpus's binding handle on sigma is the residual of
a well-measured cell about its own fitted quadratic (dof-pooled 0.201; 0.172 on cells with >= 13
metals), but that is only an *upper* bound: any real curve feature the two-term basis cannot
express -- a tetrad kink, the Gd break -- lands in the residual too.

Real chemistry is smooth along the lanthanide series; measurement noise is not.  So compare the
observed lag-1 autocovariance of the residual along Z with what pure white noise would produce
*through the same ridge projection* (which itself induces a small negative correlation, exactly as
OLS residuals do).  Excess positive autocovariance is the systematic part; what is left is the
measurement noise, and it is a two-sided estimate rather than a bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import RIDGE, N_BASIS_DOF  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bench = V.load()
    Y, basis = bench.Y, bench.basis
    C = Y - np.nanmean(Y, axis=1, keepdims=True)
    obs = ~np.isnan(Y)
    k = basis.shape[0]

    rows = []
    for min_m in (8, 10, 13, 14):
        num_obs = den_obs = 0.0     # observed lag-1 autocovariance and variance
        num_exp = den_exp = 0.0     # the same two quantities for unit white noise
        ncell = 0
        for i in range(len(Y)):
            o = obs[i]
            m = int(o.sum())
            if m < min_m:
                continue
            idx = np.flatnonzero(o)
            B = basis[:, o].T
            Mc = np.eye(m) - np.ones((m, m)) / m
            Bc = Mc @ B
            A = np.eye(m) - Bc @ np.linalg.solve(Bc.T @ Bc + RIDGE * np.eye(k), Bc.T)
            R = A @ Mc                                   # residual operator on the raw row
            r = R @ Y[i, o]
            Sig = R @ R.T                                # covariance of r under unit white noise
            adj = np.flatnonzero(np.diff(idx) == 1)      # residual entries that are Z-adjacent
            if len(adj) == 0:
                continue
            num_obs += float((r[adj] * r[adj + 1]).sum())
            den_obs += float((r ** 2).sum())
            num_exp += float(Sig[adj, adj + 1].sum())
            den_exp += float(np.trace(Sig))
            ncell += 1
        if ncell == 0:
            continue
        # solve for sigma^2 assuming the systematic part is perfectly smooth over adjacent Z
        # (so it contributes equally to the variance and to the lag-1 covariance, up to the
        #  same projection): num = s2*num_exp + v_sys*num_exp_sys, den = s2*den_exp + v_sys*den_sys
        # with the crude but conservative choice num_exp_sys = den_sys (perfect lag-1 correlation).
        s2_upper = den_obs / den_exp                     # all residual is noise -> the upper bound
        # two equations, treating the systematic part as contributing r_adj*v to both
        Adet = num_exp * 1.0 - den_exp * (num_obs / den_obs if den_obs else 0.0)
        # explicit 2x2 solve
        Mx = np.array([[num_exp, 1.0], [den_exp, 1.0]])
        rhs = np.array([num_obs, den_obs])
        try:
            s2, v_sys = np.linalg.solve(Mx, rhs)
        except np.linalg.LinAlgError:
            s2, v_sys = s2_upper, 0.0
        rows.append({
            "min_metals": min_m, "n_cells": ncell,
            "obs_lag1_autocorr": num_obs / den_obs if den_obs else np.nan,
            "white_lag1_autocorr": num_exp / den_exp,
            "sigma_upper_all_noise": float(np.sqrt(max(s2_upper, 0.0))),
            "sigma_white_part": float(np.sqrt(max(s2, 0.0))),
            "systematic_var": float(max(v_sys, 0.0)),
            "frac_residual_systematic": float(np.clip(1 - s2 * den_exp / den_obs, 0, 1))
            if den_obs else np.nan,
        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s1c_residual_whiteness.csv", index=False)
    print("=== is the residual white noise or real curve structure? ===")
    print("(lag-1 autocorrelation of the residual along Z; 'white' is what the ridge projection\n"
          " alone produces from pure white noise, so equality means the residual IS noise)")
    print(d.round(4).to_string(index=False))
    print(f"\nunused dof check: N_BASIS_DOF = {N_BASIS_DOF}")


if __name__ == "__main__":
    main()
