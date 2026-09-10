"""REFUTER lens B, part 7 -- model-free reliability of the per-series slope (split-half and
jackknife), and the rho-vs-reliability correspondence."""
from __future__ import annotations
import sys, time
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


df = pd.read_pickle(OUT / "_df.pkl")
rows, count_cols = L.load_energy_rows()
fitted, _ = L.fit_models(rows, count_cols)
series_of = L.choose_series(rows)

MODELS = ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "SPECIES_NFILLCOL")


def lin_slope(rr, yy):
    if len(rr) < 3 or np.std(rr) == 0:
        return np.nan
    A = np.c_[np.ones(len(rr)), rr - rr.mean()]
    b, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return float(b[1])


out = []
for m in MODELS:
    recs = []
    for smi, ser in series_of.items():
        s = fitted[fitted["series"] == ser].sort_values("r")
        rv = s["r"].to_numpy(float)
        yv = s[f"resid_{m}"].to_numpy(float)
        ok = np.isfinite(rv) & np.isfinite(yv)
        rr, yy = rv[ok], yv[ok]
        if len(rr) < 8:
            continue
        h1 = np.arange(len(rr)) % 2 == 0
        s1, s2 = lin_slope(rr[h1], yy[h1]), lin_slope(rr[~h1], yy[~h1])
        jk = [lin_slope(np.delete(rr, i), np.delete(yy, i)) for i in range(len(rr))]
        jk = np.array(jk, float)
        jse = np.sqrt((len(rr) - 1) / len(rr) * np.nansum((jk - np.nanmean(jk)) ** 2))
        recs.append(dict(extractant=smi, half1=s1, half2=s2, full=lin_slope(rr, yy),
                         jack_se=float(jse), n=int(len(rr))))
    R = pd.DataFrame(recs).merge(df[df.model == m][["extractant", "S8", "S14", "a", "chemotype"]],
                                 on="extractant", how="inner")
    for s_ in ("S8", "S14"):
        d = R[R[s_].fillna(False)].dropna(subset=["half1", "half2"])
        if len(d) < 8:
            continue
        rr_half = float(stats.spearmanr(d.half1, d.half2).statistic)
        sb = 2 * rr_half / (1 + rr_half) if rr_half > -1 else np.nan
        var_obs = float(np.var(d.full, ddof=1))
        var_jk = float(np.mean(d.jack_se ** 2))
        out.append(dict(model=m, set=s_, n=len(d),
                        split_half_rho=rr_half, spearman_brown=sb,
                        slope_sd=float(np.sqrt(var_obs)), jack_se_mean=float(np.sqrt(var_jk)),
                        reliability_jack=max(0.0, 1 - var_jk / var_obs),
                        max_rho_jack=float(np.sqrt(max(0.0, 1 - var_jk / var_obs))),
                        rho_obs=float(stats.spearmanr(d.full, d.a).statistic)))
out = pd.DataFrame(out)
out.to_csv(OUT / "checkJ_reliability_modelfree.csv", index=False)
print(out.round(4).to_string(index=False))

r1 = out[out.set == "S8"]
print()
print("S8: Spearman(rho_obs, spearman_brown) over the five models =",
      round(float(stats.spearmanr(r1.rho_obs.abs(), r1.spearman_brown).statistic), 4))
print("S8: Spearman(rho_obs, reliability_jack) over the five models =",
      round(float(stats.spearmanr(r1.rho_obs.abs(), r1.reliability_jack).statistic), 4))
log("part7 done")
