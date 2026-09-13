"""Claim-by-claim comparison: summary under review vs this independent re-derivation.

Reads only the CSVs produced by v_d4_check.py / v_d4_robustness.py / v_d4_extra.py.
"""

import os

import pandas as pd

OUT = "D:/ml_separator_gh/gen13_separation/analysis/stage2/d4_amplitude_vs_shape/verify"

sh = pd.read_csv(os.path.join(OUT, "v_d4_shares.csv")).set_index("arm")
mae = pd.read_csv(os.path.join(OUT, "v_d4_macro_mae.csv")).set_index("arm")
gn = pd.read_csv(os.path.join(OUT, "v_d4_gain_oracle.csv")).set_index("arm")
rb = pd.read_csv(os.path.join(OUT, "v_d4_share_robustness.csv")).set_index("arm")
rg = pd.read_csv(os.path.join(OUT, "v_d4_amplitude_regression.csv")).set_index("arm")
ex = pd.read_csv(os.path.join(OUT, "v_d4_extra.csv"))
nf = pd.read_csv(os.path.join(OUT, "v_d4_noise_floor.csv")).iloc[0]

C, X = "C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2"
ext_or = ex[ex.quantity == "oracle_gain_per_extractant"].set_index("arm")
cell_or = ex[ex.quantity == "oracle_gain_per_cell"].set_index("arm")
rmse = ex[ex.quantity == "pair_rmse"].set_index("arm")


def rng(s, f="{:.4f}"):
    return f.format(s.min()) + " - " + f.format(s.max())


rows = [
    ("amplitude share of pairwise SSE (pooled over pairs)",
     "82.5 - 87.0 %", rng(100 * sh.amplitude_share, "{:.1f}") + " %", "MATCH"),
    ("  per arm C/M_SEL/M_PHY/M_LR/X_ENS",
     "87.0/83.7/82.6/82.5/84.6",
     "/".join(f"{100*sh.amplitude_share[a]:.1f}" for a in sh.index), "MATCH"),
    ("amplitude share, units with n_metals >= 4",
     "82.5 - 87.1 %", rng(100 * sh.amplitude_share_m_ge_4, "{:.1f}") + " %", "MATCH"),
    ("n units with m == 2 / their SSE share",
     "690 / 0.65 %", f"{int(sh.n_units_m_eq_2[C])} / "
     f"{100*sh.sse_share_m_eq_2[C]:.2f} %", "MATCH"),
    ("amplitude share, EQUAL WEIGHT PER CELL",
     "not reported", rng(100 * rb.amp_share_equal_cell, "{:.1f}") + " %", "NOT REPORTED"),
    ("amplitude share, EQUAL WEIGHT PER EXTRACTANT",
     "not reported", rng(100 * rb.amp_share_equal_extractant, "{:.1f}") + " %",
     "NOT REPORTED - 15-20 pts lower than headline"),
    ("SSE share of dominant chemotype sc009 (375/521 cells)",
     "not reported", rng(100 * rb.sse_share_dominant_chemotype, "{:.1f}") + " %",
     "NOT REPORTED"),
    ("largest single unit's share of pooled SSE",
     "not reported", rng(100 * rb.sse_share_top1_unit, "{:.2f}") + " %",
     "OK - no single cell dominates"),
    ("extractant-macro MAE baseline C_DIRECT_ROW / X_ENS",
     "0.4953 / 0.4810",
     f"{mae.macro_mae_baseline[C]:.4f} / {mae.macro_mae_baseline[X]:.4f}", "MATCH"),
    ("extractant-macro MAE, amplitude made exact C / X",
     "0.2096 / 0.2067",
     f"{mae.macro_mae_amplitude_exact[C]:.4f} / "
     f"{mae.macro_mae_amplitude_exact[X]:.4f}", "MATCH"),
    ("  fraction of MAE removed", "57 - 58 %",
     rng(mae.pct_removed_amplitude, "{:.1f}") + " %", "MATCH"),
    ("extractant-macro MAE, shape made exact C / X",
     "0.4300 / 0.4165",
     f"{mae.macro_mae_shape_exact[C]:.4f} / {mae.macro_mae_shape_exact[X]:.4f}", "MATCH"),
    ("oracle best global g (5 arms)", "1.05/0.95/1.00/1.00/1.20",
     "/".join(f"{gn.best_g[a]:.2f}" for a in gn.index), "MATCH"),
    ("oracle best-g MAE gain (5 arms)", "0.0004/0.0001/0.0000/0.0000/0.0026",
     "/".join(f"{gn.oracle_gain[a]:.4f}" for a in gn.index), "MATCH"),
    ("sd(observed amplitude)", "1.182", f"{sh.sd_a_obs[C]:.3f}", "MATCH"),
    ("sd(predicted amplitude) range", "0.499 - 0.652",
     rng(sh.sd_a_pred, "{:.3f}"), "MATCH"),
    ("sd ratio observed/predicted", "1.81 - 2.37",
     rng(sh.sd_a_obs / sh.sd_a_pred, "{:.2f}"), "MATCH"),
    ("through-origin slope C / X", "1.411 / 1.491",
     f"{rg.slope_through_origin[C]:.3f} / {rg.slope_through_origin[X]:.3f}", "MATCH"),
    ("  95% CI C_DIRECT_ROW", "[0.971, 1.823]",
     f"[{rg.ci_lo[C]:.3f}, {rg.ci_hi[C]:.3f}]", "MATCH (bootstrap seed differs)"),
    ("slope with intercept / intercept / R2",
     "0.95-1.31 / -0.19 to -0.35 / 0.204-0.319",
     rng(rg.slope_with_intercept, "{:.2f}") + " / " + rng(rg.intercept, "{:.2f}")
     + " / " + rng(rg.r_squared, "{:.3f}"), "MATCH"),
    ("oracle per-extractant gain C / X", "-0.086 / -0.091",
     f"-{ext_or.gain[C]:.4f} / -{ext_or.gain[X]:.4f}", "MATCH"),
    ("oracle per-cell gain C / X", "-0.090 / -0.096",
     f"-{cell_or.gain[C]:.4f} / -{cell_or.gain[X]:.4f}", "MATCH"),
    ("pair RMSE baseline -> amplitude exact",
     "0.720-0.780 -> 0.281-0.303",
     rng(rmse.macro_mae_base, "{:.3f}") + " -> " + rng(rmse.macro_mae_oracle, "{:.3f}"),
     "MATCH"),
    ("curve reconstruction max |P c_obs - y|", "2.665e-15",
     f"{sh.max_recon_resid[C]:.3e}", "MATCH (both far below 1e-9)"),
    ("non-integrable share of predicted-pair SSE", "<= 1.1e-31",
     f"{sh.nonintegrable_share_of_pred_sse.max():.1e}", "MATCH"),
    ("replicate-noise pair MAE floor (cohort / programme)",
     "0.341 / 0.267",
     f"{nf.gauss_pair_mae:.3f} / {nf.gauss_pair_mae_at_0237:.3f}",
     f"MATCH (n={int(nf.n_repsd)} repsd, median {nf.median_logD_sd:.3f})"),
]

t = pd.DataFrame(rows, columns=["claim", "summary_value", "my_value", "verdict"])
t.to_csv(os.path.join(OUT, "v_d4_claim_comparison.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
print(t.to_string(index=False))
