"""REFUTER lens B, part 6 -- reliability of the corrected slope (is the null informative?),
byte-identical scored units, determinism."""
from __future__ import annotations
import sys, time, subprocess
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
fitted, coefs = L.fit_models(rows, count_cols)
series_of = L.choose_series(rows)

log("=== byte-identical scored units across the compared arms ===")
sets = {}
for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_NFILLCOL"):
    d = df[(df.model == m) & df.S8]
    sets[m] = tuple(sorted(d.extractant))
print("S8 extractant set identical for NAIVE/ELEM/SPECIES/NFILLCOL:",
      len(set(sets.values())) == 1, {k: len(v) for k, v in sets.items()})
sc = tuple(sorted(df[(df.model == "SPECIES_CONST") & df.S8].extractant))
print("SPECIES_CONST S8 (19) is a subset of the 62:", set(sc) <= set(sets["SPECIES"]))
nrows = {m: int(np.isfinite(fitted[f"resid_{m}"]).sum()) for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST")}
print("rows entering each fit:", nrows)

log("=== reliability of the per-series slope: is the null informative? ===")
rel = []
for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "SPECIES_NFILLCOL"):
    recs = []
    for smi, ser in series_of.items():
        s = fitted[fitted["series"] == ser].sort_values("r")
        rv = s["r"].to_numpy(float)
        yv = s[f"resid_{m}"].to_numpy(float)
        ok = np.isfinite(rv) & np.isfinite(yv)
        if ok.sum() < 5:
            continue
        rr, yy = rv[ok], yv[ok]
        A = np.c_[np.ones(len(rr)), rr - rr.mean(), (rr - rr.mean()) ** 2]
        b, *_ = np.linalg.lstsq(A, yy, rcond=None)
        res = yy - A @ b
        dof = max(1, len(rr) - 3)
        s2 = (res ** 2).sum() / dof
        cov = np.linalg.pinv(A.T @ A) * s2
        recs.append(dict(extractant=smi, slope=float(b[1]), se=float(np.sqrt(max(cov[1, 1], 0))), n=int(ok.sum())))
    R = pd.DataFrame(recs)
    R = R.merge(df[(df.model == m)][["extractant", "S8", "S14", "a"]], on="extractant", how="left")
    for s_ in ("S8", "S14"):
        d = R[R[s_].fillna(False)]
        if len(d) < 8:
            continue
        var_obs = float(np.var(d.slope, ddof=1))
        var_err = float(np.mean(d.se ** 2))
        reliab = max(0.0, 1 - var_err / var_obs)
        rel.append(dict(model=m, set=s_, n=len(d), slope_sd=float(np.sqrt(var_obs)),
                        mean_se=float(np.sqrt(var_err)), reliability=reliab,
                        max_attainable_rho=float(np.sqrt(reliab)),
                        rho_obs=float(stats.spearmanr(d.slope, d.a).statistic)))
rel = pd.DataFrame(rel)
rel.to_csv(OUT / "checkI_reliability.csv", index=False)
print(rel.round(4).to_string(index=False))

log("=== determinism: rerun the model fit in a fresh subprocess and compare ===")
code = (
    "import sys;sys.path.insert(0,r'%s');"
    "from gen16 import bootstrap;from gen16 import l1_cycle as L;import numpy as np;"
    "rows,cc=L.load_energy_rows();f,_=L.fit_models(rows,cc);"
    "sl=L.slope_table(f,L.MODELS+('SPECIES_NFILLCOL',),g_all=rows);d=L.attach_sets(sl);"
    "print(repr(float(np.nansum(d['slope'].to_numpy(float)))));"
    "print(repr(float(d[(d.model=='SPECIES')&d.S8].slope.sum())))"
) % str(Path(__file__).resolve().parents[1])
outs = []
for _ in range(2):
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    outs.append(p.stdout.strip())
print("subprocess run 1:", outs[0].replace("\n", " | "))
print("subprocess run 2:", outs[1].replace("\n", " | "))
print("identical across processes:", outs[0] == outs[1])
insess = repr(float(np.nansum(df["slope"].to_numpy(float)))) + "\n" + \
    repr(float(df[(df.model == "SPECIES") & df.S8].slope.sum()))
print("matches in-session:", insess == outs[0])
log("part6 done")
