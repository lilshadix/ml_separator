"""REFUTER lens B, part 4 -- identification of the nuisance coefficients, composition-position
descriptors (attack d), matched competitor, permutation nulls with a fresh seed, determinism."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa
from gen16 import l1_cycle as L  # noqa

OUT = L.RESULTS.parent / "refutation" / "L1NULL" / "B"
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


df = pd.read_pickle(OUT / "_df.pkl")
rows, count_cols = L.load_energy_rows()
fin = np.isfinite(rows["complex_total_energy_eV"].to_numpy(float))
fitted = rows[fin].reset_index(drop=True).copy()
grp = fitted["series"].to_numpy()
met = fitted["metal_symbol"].to_numpy()
y0 = fitted["complex_total_energy_eV"].to_numpy(float)
G_NO3, SE_NO3 = -431.970550, 1.386140
G_H2O, SE_H2O = -138.875323, 0.309589

# ============================================ identification sweep over the fill coefficients
log("=== gamma identification sweep (SPECIES with gamma pinned, delta still free) ===")
Cd, nd = L.species_design(fitted)                      # gammas + deltas
delta_only = [i for i, n in enumerate(nd) if n.startswith("delta::")]
Cdelta = Cd[:, delta_only]
ndelta = [nd[i] for i in delta_only]
sweep = []
for kn in (-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2):
    for kw in (-1, 0, 1):
        gn = G_NO3 + kn * SE_NO3
        gw = G_H2O + kw * SE_H2O
        y = y0 - gn * fitted["n_NO3"].to_numpy(float) - gw * fitted["n_H2O"].to_numpy(float)
        r, _c, *_ = L.two_way_fit(y, grp, met, cov=Cdelta, cov_names=ndelta)
        f2 = fitted.copy()
        f2["resid_SWEEP"] = r
        sl = L.slope_table(f2, ("SWEEP",), g_all=rows)
        d = L.attach_sets(sl)
        d8 = d[d.S8]
        d14 = d[d.S14]
        sweep.append(dict(k_nitrate=kn, k_water=kw, gamma_no3=gn, gamma_h2o=gw,
                          rho_S8=rho(d8.slope.to_numpy(float), d8.a.to_numpy(float))[0],
                          p_S8=rho(d8.slope.to_numpy(float), d8.a.to_numpy(float))[1],
                          rho_S14=rho(d14.slope.to_numpy(float), d14.a.to_numpy(float))[0],
                          rms_median=float(np.nanmedian(d8.rms))))
sweep = pd.DataFrame(sweep)
sweep.to_csv(OUT / "checkD_gamma_sweep.csv", index=False)
print(sweep.round(4).to_string(index=False))

# ============================================ (d) composition-position descriptors
log("=== (d) composition-position descriptors vs the amplitude ===")
sel = df[df.model == "NAIVE"][["extractant", "series", "a", "b", "abs_a", "n_metals", "chemotype",
                              "S8", "S14", "S3", "slope"]].rename(columns={"slope": "naive_slope"})
comp = []
for _, r0 in sel.iterrows():
    s = rows[(rows.series == r0.series) & fin].sort_values("r")
    if len(s) < 3:
        comp.append({})
        continue
    rr = s["r"].to_numpy(float)
    rec = {}
    for col in ("n_ligs", "n_NO3", "n_H2O", "n_fill", "coreCN"):
        if col not in s.columns:
            continue
        v = s[col].to_numpy(float)
        A = np.c_[np.ones(len(rr)), rr - rr.mean()]
        bb, *_ = np.linalg.lstsq(A, v, rcond=None)
        rec[f"slope_{col}_vs_r"] = float(bb[1])
        rec[f"range_{col}"] = float(v.max() - v.min())
        rec[f"mean_{col}"] = float(v.mean())
        # position of the step: mean radius weighted by whether the value is above its own median
        hi = v > np.median(v)
        rec[f"steppos_{col}"] = float(rr[hi].mean() - rr[~hi].mean()) if hi.any() and (~hi).any() else np.nan
    rec["n_compositions"] = int(s["composition"].nunique())
    rec["is_nitrate"] = float(s["fill_ligand"].iloc[0] == "nitrate")
    comp.append(rec)
comp = pd.DataFrame(comp, index=sel.index)
sel = pd.concat([sel, comp], axis=1)
sel.to_csv(OUT / "checkD_composition_descriptors.csv", index=False)

drows = []
desc_cols = [c for c in sel.columns if c.startswith(("slope_", "range_", "mean_", "steppos_"))] + \
            ["n_compositions", "is_nitrate", "naive_slope"]
for c in desc_cols:
    for s_ in ("S8", "S14"):
        d = sel[sel[s_]]
        x = d[c].to_numpy(float)
        drows.append(dict(descriptor=c, set=s_, n=int(np.isfinite(x).sum()),
                          rho_a=rho(x, d.a.to_numpy(float))[0], p_a=rho(x, d.a.to_numpy(float))[1],
                          rho_abs_a=rho(x, d.abs_a.to_numpy(float))[0],
                          rho_naive_slope=rho(x, d.naive_slope.to_numpy(float))[0]))
drows = pd.DataFrame(drows).sort_values(["set", "rho_a"], key=lambda s: s.abs() if s.name == "rho_a" else s,
                                        ascending=[True, False])
drows.to_csv(OUT / "checkD_composition_stats.csv", index=False)
print(drows.round(4).to_string(index=False))

# ============================================ matched cheapest competitor
log("=== matched competitor: coord__dist__frac_donor_pairs_within_3 ===")
from gen15 import valuebench as V  # noqa
bench = V.load()
bf = bench.frame
BITE = "coord__dist__frac_donor_pairs_within_3"
from gen13sep.amplitude_bench import LEAN_BLOCKS  # noqa
lean_cols = bench.columns(LEAN_BLOCKS)
j = list(lean_cols).index(BITE)
M = bench.matrix(LEAN_BLOCKS)[:, j]
per = pd.Series(M, index=bf["extractant"].to_numpy()).groupby(level=0).mean()
print("BITE per-extractant available:", per is not None, per.shape)
if per is not None:
    t = L.targets().set_index("extractant")
    t["bite"] = per.reindex(t.index)
    setm = df[df.model == "NAIVE"].set_index("extractant")[["S8", "S14", "S3"]]
    t = t.join(setm)
    crows = []
    for s_ in ("all82", "S8", "S14", "S3"):
        d = t if s_ == "all82" else t[t[s_].fillna(False)]
        for tgt in ("a", "abs_a", "b"):
            r, p, n = rho(d["bite"].to_numpy(float), d[tgt].to_numpy(float))
            crows.append(dict(subset=s_, target=tgt, n=n, rho=r, p=p))
    crows = pd.DataFrame(crows)
    crows.to_csv(OUT / "checkE_competitor.csv", index=False)
    print(crows.round(4).to_string(index=False))
log("part4 done")
