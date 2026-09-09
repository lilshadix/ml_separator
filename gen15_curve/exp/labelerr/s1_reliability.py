"""Step 1: how much of the spread of (a, b) across cells is sampling noise?

Writes results/s1_*.csv and prints the tables that go into REPORT.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import (N_BASIS_DOF, cell_residual_sigma, coef_cov, pooled_replicate_sigma,  # noqa: E402
                   pooled_residual_sigma, reliability, replicate_arrays, sigma_table)
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import cell_weights  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
BANDS = [(2, 2), (3, 4), (5, 8), (9, 13), (14, 14)]
MODES = ["white_0.137", "resid_const", "resid_cell", "inter_0.334", "rep_cell"]
CONSTS = {"white_0.137": 0.137, "inter_0.334": 0.334}


def _se(bench, mode):
    """The whiteness test (s1c) leaves 0.137 as measurement noise; 0.201 is the residual upper
    bound; 0.334 is the replicate series x metal interaction; rep_cell is the naive replicate sd."""
    if mode in CONSTS:
        return sigma_table(bench, "const", sigma=CONSTS[mode])
    return sigma_table(bench, mode)


def main() -> None:
    bench = V.load()
    f = bench.frame
    m = f.n_metals.to_numpy(dtype=float)
    a, b = bench.coef[:, 0], bench.coef[:, 1]
    w = cell_weights(bench.groups, bench.n_obs)

    # ---------------- sigma ---------------------------------------------------------------
    nr, rsd = replicate_arrays(bench)
    rep_mask = np.isfinite(rsd) & (nr >= 2)
    rep_pool, rep_med = pooled_replicate_sigma(bench)
    res_pool = pooled_residual_sigma(bench)
    cs = cell_residual_sigma(bench)
    rows = [
        {"handle": "replicates (dof-pooled RMS)", "sigma": rep_pool,
         "n_units": int(rep_mask.sum()), "note": "216 (cell,metal) means with nrep>=2, in 41 cells"},
        {"handle": "replicates (median sd)", "sigma": rep_med, "n_units": int(rep_mask.sum()),
         "note": "median of the 216 within-(cell,metal) sds"},
        {"handle": "residual about own quadratic (dof-pooled, m>=5)", "sigma": res_pool,
         "n_units": int(np.isfinite(cs).sum()),
         "note": "UPPER bound: contains real non-quadratic chemistry too"},
        {"handle": "residual, m>=13 only", "n_units": int(np.isfinite(cs[m >= 13]).sum()),
         "sigma": float(np.sqrt(np.sum((m[np.isfinite(cs) & (m >= 13)] - N_BASIS_DOF)
                                       * cs[np.isfinite(cs) & (m >= 13)] ** 2)
                               / np.sum(m[np.isfinite(cs) & (m >= 13)] - N_BASIS_DOF))),
         "note": "the best-determined cells"},
    ]
    sig = pd.DataFrame(rows)
    sig.to_csv(OUT / "s1_sigma_handles.csv", index=False)
    print("\n=== sigma: the two handles the corpus gives ===")
    print(sig.round(4).to_string(index=False))

    resid_by_m = pd.DataFrame({"n_metals": m, "resid_sigma": cs}).dropna()
    rbm = resid_by_m.groupby("n_metals").resid_sigma.agg(["count", "median", "mean"])
    rbm.to_csv(OUT / "s1_residual_sigma_by_nmetals.csv")
    print("\nper-cell residual sigma by n_metals")
    print(rbm.round(4).to_string())

    # ---------------- reliability ---------------------------------------------------------
    recs, percell = [], {}
    for mode in MODES:
        se = _se(bench, mode)
        cov = coef_cov(bench, se)
        va, vb = cov[:, 0, 0], cov[:, 1, 1]
        percell[mode] = (va, vb)
        for name, val, var in (("a", a, va), ("b", b, vb)):
            for wname, ww in (("unweighted", None), ("chemotype_balanced", w)):
                r = reliability(val, var, ww)
                recs.append(dict(sigma_mode=mode, coef=name, weighting=wname, band="all", **r))
            for lo, hi in BANDS:
                sel = (m >= lo) & (m <= hi)
                if sel.sum() < 5:
                    continue
                r = reliability(val[sel], var[sel], None)
                recs.append(dict(sigma_mode=mode, coef=name, weighting="unweighted",
                                 band=f"{lo}-{hi}", **r))
    rel = pd.DataFrame(recs)
    rel.to_csv(OUT / "s1_reliability.csv", index=False)

    print("\n=== reliability (ICC = 1 - E[sampling var] / Var observed) ===")
    piv = rel[rel.band == "all"].pivot_table(index=["coef", "weighting"], columns="sigma_mode",
                                             values="reliability")[MODES]
    print(piv.round(3).to_string())
    print("\nby n_metals band (unweighted)")
    pb = rel[(rel.band != "all")].pivot_table(index=["coef", "band"], columns="sigma_mode",
                                              values="reliability")[MODES]
    print(pb.round(3).to_string())

    print("\nvariance decomposition, chemotype-balanced, primary sigma = resid_const"
          f" ({res_pool:.3f})")
    sub = rel[(rel.sigma_mode == "resid_const") & (rel.band == "all")
              & (rel.weighting == "chemotype_balanced")]
    print(sub[["coef", "var_observed", "var_sampling", "var_true", "reliability"]]
          .round(4).to_string(index=False))
    sub2 = rel[(rel.sigma_mode == "rep_cell") & (rel.band == "all")
               & (rel.weighting == "chemotype_balanced")]
    print("\nsame under the pessimistic replicate sigma (rep_cell)")
    print(sub2[["coef", "var_observed", "var_sampling", "var_true", "reliability"]]
          .round(4).to_string(index=False))

    # ---------------- how often is the direction label a coin flip? -----------------------
    lines = []
    for mode in MODES:
        va, vb = percell[mode]
        sa, sb = np.sqrt(va), np.sqrt(vb)
        rich = m >= 5
        lines.append({
            "sigma_mode": mode,
            "median_se_a": float(np.median(sa)), "median_se_a_rich": float(np.median(sa[rich])),
            "median_se_b": float(np.median(sb)), "median_se_b_rich": float(np.median(sb[rich])),
            "frac_|a|<se": float(np.mean(np.abs(a) < sa)),
            "frac_|a|<se_rich": float(np.mean(np.abs(a[rich]) < sa[rich])),
            "frac_|b|<se": float(np.mean(np.abs(b) < sb)),
            "frac_|b|<se_rich": float(np.mean(np.abs(b[rich]) < sb[rich])),
        })
    cf = pd.DataFrame(lines)
    cf.to_csv(OUT / "s1_coinflip.csv", index=False)
    print("\n=== per-cell precision and how often the label is inside its own noise ===")
    print(cf.round(4).to_string(index=False))

    # ---------------- what the direction label costs -------------------------------------
    #   probability the *observed* sign of a differs from the true sign, given the se
    from scipy.stats import norm
    rows = []
    for mode in MODES:
        va, _ = percell[mode]
        se_a = np.sqrt(va)
        # P(sign flip) for a cell whose true a equals the shrunk estimate |a|
        pflip = norm.cdf(-np.abs(a) / np.maximum(se_a, 1e-9))
        rich = m >= 5
        rows.append({"sigma_mode": mode, "mean_P_signflip_all": float(pflip.mean()),
                     "mean_P_signflip_rich": float(pflip[rich].mean()),
                     "mean_P_signflip_rich_wtd": float(np.average(
                         pflip[rich], weights=cell_weights(bench.groups[rich], bench.n_obs[rich])))})
    sf = pd.DataFrame(rows)
    sf.to_csv(OUT / "s1_signflip.csv", index=False)
    print("\n=== expected label-flip rate of the direction target ===")
    print(sf.round(4).to_string(index=False))

    np.save(OUT / "s1_var_a_resid_const.npy", percell["resid_const"][0])
    np.save(OUT / "s1_var_b_resid_const.npy", percell["resid_const"][1])
    np.save(OUT / "s1_var_a_rep_cell.npy", percell["rep_cell"][0])
    np.save(OUT / "s1_var_b_rep_cell.npy", percell["rep_cell"][1])


if __name__ == "__main__":
    main()
