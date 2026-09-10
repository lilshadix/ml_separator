"""REFUTER lens B, part 5 -- the decisive mirror-image test: can the SPECIES correction RECOVER a
signal that is really there?  Plus permutation nulls on a fresh seed, decision-rule recomputation,
BH accounting and determinism."""
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
base = rows[fin].reset_index(drop=True).copy()

# scale references
sp8 = df[(df.model == "SPECIES") & df.S8]
log("SPECIES S8 slope sd = %.4f  median rms = %.4f" % (sp8.slope.std(), sp8.rms.median()))
nv8 = df[(df.model == "NAIVE") & df.S8]
log("NAIVE   S8 slope sd = %.4f  median rms = %.4f" % (nv8.slope.std(), nv8.rms.median()))

# ============================================ signal-injection recovery
log("=== signal injection: add a KNOWN linear-in-radius trend, see what each model recovers ===")
targ = L.targets().set_index("extractant")
series_of = L.choose_series(rows)
ser_to_ext = {v: k for k, v in series_of.items()}
all_series = sorted(base["series"].unique())
rng = np.random.default_rng(20260910)
# c_i is a perfect rank image of a_i where the extractant has one; random normal elsewhere
z = {}
avals = {}
for s in all_series:
    e = ser_to_ext.get(s)
    avals[s] = float(targ.loc[e, "a"]) if (e is not None and e in targ.index) else np.nan
known = [s for s in all_series if np.isfinite(avals[s])]
rk = stats.rankdata([avals[s] for s in known])
rk = (rk - rk.mean()) / rk.std()
for s, v in zip(known, rk):
    z[s] = float(v)
for s in all_series:
    if s not in z:
        z[s] = float(rng.normal())
zvec = base["series"].map(z).to_numpy(float)
rvec = base["r"].to_numpy(float)
r_ctr = rvec - pd.Series(rvec).groupby(base["series"].to_numpy()).transform("mean").to_numpy()

inj = []
for kappa in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
    work = base.copy()
    work["E_inj"] = work["complex_total_energy_eV"].to_numpy(float) + kappa * zvec * r_ctr
    fit, _c = L.fit_models(work, count_cols, energy="E_inj")
    sl = L.slope_table(fit, L.MODELS + ("SPECIES_NFILLCOL",), g_all=rows)
    d = L.attach_sets(sl)
    d["z_true"] = d["series"].map(z)
    for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST"):
        for s_ in ("S8", "S14"):
            dd = d[(d.model == m) & d[s_]]
            x = dd.slope.to_numpy(float)
            y = dd.a.to_numpy(float)
            zt = dd.z_true.to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(zt)
            # attenuation: OLS of recovered slope on the injected coefficient
            att = np.nan
            if ok.sum() > 5 and kappa > 0:
                A = np.c_[np.ones(ok.sum()), kappa * zt[ok]]
                bb, *_ = np.linalg.lstsq(A, x[ok], rcond=None)
                att = float(bb[1])
            inj.append(dict(kappa=kappa, model=m, set=s_, n=int(np.isfinite(x).sum()),
                            rho_slope_vs_a=rho(x, y)[0], p=rho(x, y)[1],
                            rho_slope_vs_injected=rho(x, zt)[0],
                            recovery_coef=att, slope_sd=float(np.nanstd(x)),
                            rms_median=float(np.nanmedian(dd.rms))))
    log("kappa", kappa, "done")
inj = pd.DataFrame(inj)
inj.to_csv(OUT / "checkF_injection.csv", index=False)
piv = inj.pivot_table(index=["kappa"], columns=["model", "set"], values="rho_slope_vs_a")
print(piv.round(3).to_string())
print()
piv2 = inj.pivot_table(index=["kappa"], columns=["model", "set"], values="recovery_coef")
print("recovery coefficient (1.0 = the injected trend is returned intact):")
print(piv2.round(3).to_string())

# ============================================ permutation null, fresh seed
log("=== permutation null with a fresh seed (check 5) ===")
W = L.wide(df, "slope")
fam_cols = [f"slope__{m}__{s}" for m in L.MODELS for s in L.SETS]
out = []
for seed in (20260909, 991, 13577):
    rng2 = np.random.default_rng(seed)
    null, obs = L.permutation_null(W, fam_cols, rng2)
    null["seed"] = seed
    out.append(null)
perm = pd.concat(out, ignore_index=True)
perm.to_csv(OUT / "checkG_perm_null_seeds.csv", index=False)
print(perm.round(4).to_string(index=False))

# ============================================ decision rule recomputed from the LEAD's own csv
log("=== decision rule recomputed from the lead's stage1_stats.csv (check 7) ===")
st = pd.read_csv(L.RESULTS / "stage1_stats.csv")
dec = L.decision(st, "SPECIES", "S8")
dec2 = L.decision(st, "SPECIES_CONST", "S8")
con = pd.read_csv(L.RESULTS / "contrasts_stage1.csv")
fam = con.groupby("family").size().to_dict()
reg = con[con.family == "registered"][["comparison", "point", "p_two_sided", "p_bh_within_family", "n_units"]]
expl = con[con.family == "exploratory"].nsmallest(6, "p_two_sided")[
    ["comparison", "point", "p_two_sided", "p_bh_within_family", "n_units"]]
print(json.dumps({k: v for k, v in dec.items() if k != "rule"}, indent=1, default=str))
print(json.dumps({k: v for k, v in dec2.items() if k != "rule"}, indent=1, default=str))
print("family sizes:", fam)
print(reg.round(5).to_string(index=False))
print(expl.round(6).to_string(index=False))
# BH recomputed independently over the exploratory family
p = con.loc[con.family == "exploratory", "p_two_sided"].to_numpy(float)
mine = L.bh(p)
print("BH exploratory: max |mine - lead| =",
      float(np.nanmax(np.abs(mine - con.loc[con.family == "exploratory", "p_bh_within_family"].to_numpy(float)))))
pd.DataFrame([dec, dec2]).to_csv(OUT / "checkH_decision.csv", index=False)
log("part5 done")
