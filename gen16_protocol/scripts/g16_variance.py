"""Variance-component table for the cell amplitude, with two independent kinds of interval,
plus the marginal / conditional R2 of the deployed model, plus an order-free Shapley split that
replaces the order-dependent nested ANOVA.

Usage:  python gen16_protocol/scripts/g16_variance.py [boot_reps]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen16_protocol"):
    sys.path.insert(0, str(p))

from gen14.dirbench import load                                        # noqa: E402
from gen16.variance import (indicator, lmg_bootstrap, lmg_shapley,     # noqa: E402
                            nakagawa_r2, parametric_bootstrap_ci, profile_ci, reml_fit)

BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = ROOT / "gen16_protocol" / "results"
OUT.mkdir(parents=True, exist_ok=True)

bench = load()
f = bench.frame.reset_index(drop=True)
rich = f.n_metals.to_numpy() >= 5
amp = bench.coef[:, 0]


def onehot_label(prefix: str) -> np.ndarray:
    cols = [c for c in f.columns if c.startswith(prefix)]
    A = f[cols].to_numpy(dtype=float)
    lab = np.array([cols[j].replace(prefix, "") if A[i].max() > 0 else "none"
                    for i, j in enumerate(A.argmax(axis=1))])
    return lab


f = f.assign(diluent=onehot_label("cond__diluent__"), acid=onehot_label("cond__acid__"))
sub = f[rich].copy()
y = amp[rich]
print(f"n = {len(y)} well-determined cells; sd(amplitude) = {y.std():.4f}")

Z = {
    "chemotype": indicator(sub.chemotype),
    "extractant": indicator(sub.extractant),
    "publication": indicator(sub.publication_id),
    "diluent": indicator(sub.diluent),
    "acid": indicator(sub.acid),
}
X1 = np.ones((len(y), 1))

fit = reml_fit(y, X1, Z)
print("\n=== REML variance components (intercept only) ===")
rows = []
for nm, v in zip(fit.names, fit.var):
    rows.append({"component": nm, "variance": v, "sd": np.sqrt(v), "share": v / fit.total})
tab = pd.DataFrame(rows)

prof = [profile_ci(y, X1, Z, fit, nm) for nm in fit.names]
tab["profile_low"] = [p["profile_low"] for p in prof]
tab["profile_high"] = [p["profile_high"] for p in prof]

pb = parametric_bootstrap_ci(y, X1, Z, fit, reps=BOOT)
tab["boot_var_low"] = pb["var_low"]
tab["boot_var_high"] = pb["var_high"]
tab["share_low"] = pb["share_low"]
tab["share_high"] = pb["share_high"]
print(tab.round(5).to_string(index=False))
tab.to_csv(OUT / "g16_variance_components.csv", index=False)

# ICC at chemotype level, with its interval, next to the stage-2 point estimate 0.72
icc = (fit.var[0] + fit.var[1]) / fit.total
print(f"\nICC(extractant within chemotype + chemotype) = {icc:.3f}")
print(f"ICC(chemotype alone)                        = {fit.var[0] / fit.total:.3f} "
      f"[{tab.share_low[0]:.3f}, {tab.share_high[0]:.3f}]")

# ------------------------------------------------------------------------------------
# marginal vs conditional R2, with the deployed model's own prediction as the fixed effect
# ------------------------------------------------------------------------------------
print("\n=== Nakagawa-Schielzeth R2, intercept-only fixed part ===")
print({k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in nakagawa_r2(y, X1, Z, fit).items() if k != "sigma2_random"})

pred_path = OUT / "g16_oof_amplitude_BP.csv"
if pred_path.exists():
    oof = pd.read_csv(pred_path).groupby("cell_id")["pred_amp"].mean()
    p = sub.cell_id.map(oof).to_numpy(dtype=float)
    ok = np.isfinite(p)
    X2 = np.c_[np.ones(int(ok.sum())), p[ok]]
    Z2 = {k: v[ok] for k, v in Z.items()}
    fit2 = reml_fit(y[ok], X2, Z2)
    r2 = nakagawa_r2(y[ok], X2, Z2, fit2)
    print("\n=== R2 with the gen14 BP out-of-fold predicted amplitude as the fixed effect ===")
    print(f"  R2_marginal    (transfers to a new extractant in a new laboratory) = {r2['R2_marginal']:.4f}")
    print(f"  R2_conditional (what a design that reuses the extractant reports)  = {r2['R2_conditional']:.4f}")
    pd.DataFrame([{"R2_marginal": r2["R2_marginal"], "R2_conditional": r2["R2_conditional"],
                   "sigma2_fixed": r2["sigma2_fixed"], "n": int(ok.sum())}]
                 ).to_csv(OUT / "g16_r2_marginal_conditional.csv", index=False)
else:
    print(f"\n[skip] no {pred_path.name}; run g16_oof_amplitude.py first for the marginal R2")

# ------------------------------------------------------------------------------------
# Shapley (LMG) replacement for the order-dependent nested ANOVA
# ------------------------------------------------------------------------------------
print("\n=== Shapley / LMG split of R2 (order-free) ===")
d6 = ROOT / "gen13_separation/analysis/stage2/d6_condition_law/d6_cell_amplitudes.csv"
core = sub
if d6.exists():
    keep = pd.read_csv(d6)
    key = "cell_id" if "cell_id" in keep.columns else None
    if key is not None:
        core = sub[sub.cell_id.isin(set(keep[key]))]
        print(f"  restricted to the d6 cohort: {len(core)} cells")
yc = amp[rich][sub.index.isin(core.index)] if len(core) != len(sub) else y
ci = core.reset_index(drop=True)

n_metals_block = indicator(ci.n_metals.astype(str))
blocks = {
    "extractant identity": indicator(ci.extractant),
    "measured metal set": indicator(ci.metals.astype(str)) if "metals" in ci.columns else n_metals_block,
    "acid concentration": np.log10(np.clip(ci["cond__acid_concentration_M"].to_numpy(dtype=float), 1e-3, None))[:, None],
    "acid identity": indicator(ci.acid),
    "extractant concentration": np.log10(np.clip(ci["cond__extractant_concentration_M"].to_numpy(dtype=float), 1e-4, None))[:, None],
    "temperature": np.nan_to_num(ci["cond__temperature_C"].to_numpy(dtype=float), nan=25.0)[:, None],
    "diluent identity": indicator(ci.diluent),
    "publication_id": indicator(ci.publication_id),
}
sh = lmg_shapley(yc, blocks)
bs = lmg_bootstrap(yc, blocks, ci.chemotype.to_numpy(), reps=200)
srows = []
for i, k in enumerate(bs["keys"]):
    srows.append({"term": k, "shapley_r2": sh["shapley"][k],
                  "share_of_model_r2": sh["share_of_r2"][k],
                  "boot_low": bs["low"][i], "boot_high": bs["high"][i]})
S = pd.DataFrame(srows).sort_values("shapley_r2", ascending=False)
print(f"  R2 of the full model = {sh['r2_full']:.4f}  (Shapley shares sum to {sh['check_sums_to_r2']:.4f})")
print(S.round(4).to_string(index=False))
S.to_csv(OUT / "g16_shapley_r2.csv", index=False)
