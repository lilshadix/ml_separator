"""REFUTER lens A (round 2), part 3 for L1NULL: matched competitor, the composition-position
descriptor against the registered bar, and the scale audit of the naive slope."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_l1null_a4 import OUT, PHYS3D, rho, prho  # noqa: E402
from refute_l1null_a5 import blocked_boot, fisher_ci  # noqa: E402

S = {}
sl = pd.read_csv(OUT / "A4_slopes.csv")
comp = pd.read_csv(OUT / "A4_composition_descriptors.csv")
tgt = pd.read_parquet(PHYS3D / "extractant_targets.parquet")[
    ["extractant", "a", "b", "abs_a", "n_metals", "chemotype"]]
cft = pd.read_csv(OUT / "A6_competitor_raw.csv", index_col=0)["frac_donor_pairs_within_3"]

base = sl[sl["model"] == "NAIVE"][["extractant", "n_computed_metals", "n_ligs_varies"]].copy()
base = base.merge(tgt, on="extractant", how="inner")
base["S8"] = base["n_computed_metals"] >= 8
base["S14"] = base["n_computed_metals"] == 14
base["S3"] = base["n_computed_metals"] >= 3
base["competitor"] = base["extractant"].map(cft)
base = base.merge(comp[["extractant", "nligs_range", "frac_at_max_nligs", "n_comp",
                        "step_r", "mean_nligs"]], on="extractant", how="left")
for m in sl["model"].unique():
    base[f"slope_{m}"] = base["extractant"].map(
        sl[sl["model"] == m].set_index("extractant")["slope"])

rowsout = []
for name in ("competitor", "nligs_range", "frac_at_max_nligs", "n_comp", "mean_nligs",
             "slope_NAIVE", "slope_SPECIES", "slope_CYCLE_ADD"):
    for lbl, d in (("S8", base[base["S8"]]), ("S14", base[base["S14"]]), ("S3", base[base["S3"]]),
                   ("S8_no_sc009", base[base["S8"] & (base["chemotype"] != "sc009")]),
                   ("all82", base)):
        x = d[name].to_numpy(float)
        y = d["a"].to_numpy(float)
        r, p, n = rho(x, y)
        pr = prho(x, y, d["n_metals"].to_numpy(float))
        ch = d["chemotype"].to_numpy()
        lv = [rho(x[ch != c], y[ch != c])[0] for c in np.unique(ch)]
        lv = [v for v in lv if np.isfinite(v)]
        lo, hi, nb = blocked_boot(x, y, ch)
        rowsout.append({"descriptor": name, "set": lbl, "n": n, "rho_vs_a": r, "p": p,
                        "partial_rho_n_metals": pr,
                        "loco_min": float(np.min(lv)) if lv else np.nan,
                        "loco_max": float(np.max(lv)) if lv else np.nan,
                        "loco_sign_stable": bool(lv and np.all(np.sign(lv) == np.sign(r))),
                        "ci_lo": lo, "ci_hi": hi, "n_boot": nb,
                        "ci_excludes_zero": bool(np.isfinite(lo) and lo * hi > 0),
                        "passes_registered_bar": bool(np.isfinite(r) and abs(r) >= 0.40
                                                      and np.isfinite(pr) and abs(pr) >= 0.40
                                                      and lv and np.all(np.sign(lv) == np.sign(r))
                                                      and np.isfinite(lo) and lo * hi > 0)})
pd.DataFrame(rowsout).to_csv(OUT / "A6_descriptor_bar.csv", index=False)

# cross-correlations: is the composition step the same thing as the naive slope / the 2D bite?
d8 = base[base["S8"]]
d14 = base[base["S14"]]
cross = []
for A, B in (("slope_NAIVE", "nligs_range"), ("slope_NAIVE", "frac_at_max_nligs"),
             ("slope_NAIVE", "competitor"), ("nligs_range", "competitor"),
             ("frac_at_max_nligs", "competitor"), ("slope_SPECIES", "nligs_range"),
             ("slope_CYCLE_ADD", "nligs_range"), ("slope_NAIVE", "slope_CYCLE_ADD"),
             ("slope_NAIVE", "slope_SPECIES")):
    for lbl, d in (("S8", d8), ("S14", d14)):
        r, p, n = rho(d[A].to_numpy(float), d[B].to_numpy(float))
        cross.append({"x": A, "y": B, "set": lbl, "n": n, "rho": r, "p": p})
pd.DataFrame(cross).to_csv(OUT / "A6_cross.csv", index=False)

# scale audit: can an eV-scale interaction term move the naive slope's ranks at all?
sc = []
for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "CYCLE_ADD"):
    v = base.loc[base["S8"], f"slope_{m}"].to_numpy(float)
    v = v[np.isfinite(v)]
    sc.append({"model": m, "n": len(v), "slope_sd_eV_per_unit_r": float(np.std(v)),
               "slope_iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
               "median_abs_slope": float(np.median(np.abs(v)))})
pd.DataFrame(sc).to_csv(OUT / "A6_scale.csv", index=False)
S["amplitude_a_sd"] = float(base.loc[base["S8"], "a"].std())

# partial: does the naive slope add anything to the composition count?
d = base[base["S8"]]
S["partial_rho_slopeNAIVE_vs_a_given_nligs_range_S8"] = prho(
    d["slope_NAIVE"].to_numpy(float), d["a"].to_numpy(float), d["nligs_range"].to_numpy(float))
S["partial_rho_nligs_range_vs_a_given_slopeNAIVE_S8"] = prho(
    d["nligs_range"].to_numpy(float), d["a"].to_numpy(float), d["slope_NAIVE"].to_numpy(float))
d = base[base["S14"]]
S["partial_rho_slopeNAIVE_vs_a_given_frac_at_max_S14"] = prho(
    d["slope_NAIVE"].to_numpy(float), d["a"].to_numpy(float), d["frac_at_max_nligs"].to_numpy(float))
S["partial_rho_frac_at_max_vs_a_given_slopeNAIVE_S14"] = prho(
    d["frac_at_max_nligs"].to_numpy(float), d["a"].to_numpy(float), d["slope_NAIVE"].to_numpy(float))
S["fisher_ci_nligs_range_S8"] = fisher_ci(-0.499533, 62)
with open(OUT / "A6_summary.json", "w", encoding="utf-8") as fh:
    json.dump(S, fh, indent=2, default=float)
pd.set_option("display.width", 220)
print(pd.DataFrame(rowsout).to_string(index=False))
print()
print(pd.DataFrame(cross).to_string(index=False))
print()
print(pd.DataFrame(sc).to_string(index=False))
print(json.dumps(S, indent=2, default=float))
