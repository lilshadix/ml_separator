"""REFUTER lens B, part 3 -- the attacks on the null (a)-(d), matched competitor, nulls, stats."""
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
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def fisher_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n < 6:
        return (np.nan, np.nan)
    se = 1.06 / np.sqrt(n - 3)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


df = pd.read_pickle(OUT / "_df.pkl")
rows, count_cols = L.load_energy_rows()
log("rows loaded")

# ==================================================================== (a) power at n = 19
log("=== (a) power / CI width at n = 19 (SPECIES_CONST on S8) ===")
pw = []
for n, r_obs in ((19, 0.1667), (11, 0.1545), (33, 0.1985), (62, -0.0837), (39, -0.0482), (43, 0.0005)):
    lo, hi = fisher_ci(r_obs, n)
    se = 1.06 / np.sqrt(n - 3)
    zc = stats.norm.ppf(0.975)
    # power to reject rho=0 at alpha=.05 when the truth is rho_true
    powers = {}
    for rt in (0.3, 0.4, 0.5, 0.6, 0.644):
        zt = np.arctanh(rt)
        powers[f"power_vs_{rt}"] = float(1 - stats.norm.cdf(zc - zt / se) + stats.norm.cdf(-zc - zt / se))
    # two-sided test that the observed differs from rho_true
    pdiff = {}
    for rt in (0.4, 0.5, 0.6, 0.644):
        z = (np.arctanh(rt) - np.arctanh(r_obs)) / se
        pdiff[f"p_obs_ne_{rt}"] = float(2 * (1 - stats.norm.cdf(abs(z))))
    pw.append(dict(n=n, rho_obs=r_obs, fisher_lo=lo, fisher_hi=hi, ci_width=hi - lo, se_z=se, **powers, **pdiff))
pw = pd.DataFrame(pw)
pw.to_csv(OUT / "checkA_power.csv", index=False)
print(pw.round(4).to_string(index=False))

# bootstrap CI for the n=19 row with the lead's construction but MIN_BOOT_ROWS lowered (my own
# exploratory relaxation, declared: the lead refused it as a post-hoc change to a frozen module)
W = L.wide(df, "slope")
chem = W["chemotype"].to_numpy()
uch = np.unique(chem)
idx_of = {c: np.flatnonzero(chem == c) for c in uch}
relax = []
for seed in (20260909, 771, 4242):
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([idx_of[c] for c in rng.choice(uch, size=len(uch), replace=True)])
             for _ in range(2000)]
    for col in ("slope__SPECIES_CONST__S8", "slope__SPECIES__S8", "slope__NAIVE__S8", "slope__NAIVE__S14",
                "slope__SPECIES_CONST__S14"):
        x = W[col].to_numpy(float)
        y = W["a"].to_numpy(float)
        vals = []
        for b in boots:
            bb = b[np.isfinite(x[b]) & np.isfinite(y[b])]
            if len(bb) < 8 or len(np.unique(x[bb])) < 5:
                continue
            v = stats.spearmanr(x[bb], y[bb]).statistic
            if np.isfinite(v):
                vals.append(v)
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) > 100 else (np.nan, np.nan))
        relax.append(dict(seed=seed, column=col, min_rows=8, n_boot=len(vals), ci_lo=float(lo), ci_hi=float(hi)))
relax = pd.DataFrame(relax)
relax.to_csv(OUT / "checkA_bootstrap_relaxed.csv", index=False)
print(relax.round(4).to_string(index=False))

# ==================================================================== (b) fixed-coefficient cycle
log("=== (b) fixed-coefficient cycle: the correction that cannot absorb the trend ===")
els = [c for c in count_cols if c[2:] in L.KNOWN_ELEMENTS]
el_names = [c[2:] for c in els]
fin = np.isfinite(rows["complex_total_energy_eV"].to_numpy(float))
varying = set(L.varying_series(rows[fin]))
comp_var = rows.groupby("series")["composition"].nunique()
const_comp_series = set(comp_var.index[comp_var == 1])

from rdkit import Chem, RDLogger  # noqa
RDLogger.DisableLog("rdApp.*")
lig_formula = {s: L.ligand_formula(s) for s in rows["canonical_smiles"].unique()}


def elem_fit(mask):
    X = rows.loc[mask, els].to_numpy(float)
    y = rows.loc[mask, "complex_total_energy_eV"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(zip(el_names, beta))


def cycle_energy(eps, gamma=None):
    """dE = E - n_ligs*Ehat(L) - n_NO3*Ehat(NO3) - n_H2O*Ehat(H2O); all coefficients FIXED."""
    e_no3 = gamma["nitrate"] if gamma else eps.get("N", 0.0) + 3 * eps.get("O", 0.0)
    e_h2o = gamma["water"] if gamma else 2 * eps.get("H", 0.0) + eps.get("O", 0.0)
    lig_e = {s: sum(eps.get(e, 0.0) * c for e, c in f.items() if e in eps) for s, f in lig_formula.items()}
    return (rows["complex_total_energy_eV"].to_numpy(float)
            - rows["n_ligs"].to_numpy(float) * rows["canonical_smiles"].map(lig_e).to_numpy(float)
            - rows["n_NO3"].to_numpy(float) * e_no3
            - rows["n_H2O"].to_numpy(float) * e_h2o)


masks = {
    "ALL": fin,
    "CONSTCOMP": fin & rows["series"].isin(const_comp_series).to_numpy(),
    "CONSTNLIGS": fin & (~rows["series"].isin(varying)).to_numpy(),
}
# gamma from the lead's SPECIES_CONST fit (identified only on constant-n_ligs series)
cf = pd.read_csv(L.RESULTS / "stage1_coefficients.csv")
gc = cf[(cf.model == "SPECIES_CONST") & cf.term.str.startswith("gamma::")]
gamma_const = {"nitrate": float(gc[gc.term == "gamma::n_NO3"].coef.iloc[0]),
               "water": float(gc[gc.term == "gamma::n_H2O"].coef.iloc[0])}
log("gamma from SPECIES_CONST:", gamma_const)

extra = {}
for name, mk in masks.items():
    eps = elem_fit(mk)
    extra[f"FIXCYC_{name}"] = cycle_energy(eps)
eps_cn = elem_fit(masks["CONSTNLIGS"])
extra["FIXCYC_CONSTNLIGS_GAMMACONST"] = cycle_energy(eps_cn, gamma=gamma_const)

work = rows.copy()
for k, v in extra.items():
    work[f"E_{k}"] = v

# NAIVE two-way on each fixed-cycle energy; plus SPECIES variants with restricted delta freedom
fitted = rows[fin].reset_index(drop=True).copy()
grp = fitted["series"].to_numpy()
met = fitted["metal_symbol"].to_numpy()
res_cols = {}
for k in extra:
    y = work.loc[fin, f"E_{k}"].to_numpy(float)
    r, _c, *_ = L.two_way_fit(y, grp, met)
    res_cols[k] = r

y0 = fitted["complex_total_energy_eV"].to_numpy(float)
# SPECIES with no delta at all (gamma only)
C, nm_ = L.species_design(fitted, ligand_terms=False)
res_cols["SPECIES_NODELTA"], _c, *_ = L.two_way_fit(y0, grp, met, cov=C, cov_names=nm_)
# SPECIES with a single GLOBAL n_ligs coefficient instead of 56 free per-series deltas
nl = fitted["n_ligs"].to_numpy(float)
C2 = np.column_stack([fitted["n_NO3"].to_numpy(float), fitted["n_H2O"].to_numpy(float), nl])
res_cols["SPECIES_GLOBALDELTA"], _c, *_ = L.two_way_fit(y0, grp, met, cov=C2,
                                                        cov_names=["g_no3", "g_h2o", "d_global"])
# SPECIES with one delta per LIGAND (shared across that ligand's series), still free
lig = fitted["canonical_smiles"].to_numpy()
cols3 = [fitted["n_NO3"].to_numpy(float), fitted["n_H2O"].to_numpy(float)]
names3 = ["g_no3", "g_h2o"]
nl_c = nl - pd.Series(nl).groupby(lig).transform("mean").to_numpy()
for s in sorted(set(lig[pd.Series(nl).groupby(lig).transform("nunique").to_numpy() > 1])):
    cols3.append(np.where(lig == s, nl_c, 0.0))
    names3.append(f"dlig::{s}")
res_cols["SPECIES_LIGDELTA"], _c, *_ = L.two_way_fit(y0, grp, met, cov=np.column_stack(cols3), cov_names=names3)
# delta fixed at the element-regression ligand energy (a fixed scalar per ligand, no freedom):
# gamma free for the fill, ligand term subtracted
lig_e_cn = {s: sum(eps_cn.get(e, 0.0) * c for e, c in f.items() if e in eps_cn) for s, f in lig_formula.items()}
y_fixlig = y0 - nl * pd.Series(lig).map(lig_e_cn).to_numpy(float)
Cf, nmf = L.species_design(fitted, ligand_terms=False)
res_cols["SPECIES_FIXLIG"], _c, *_ = L.two_way_fit(y_fixlig, grp, met, cov=Cf, cov_names=nmf)

for k, v in res_cols.items():
    fitted[f"resid_{k}"] = v
newmods = tuple(res_cols.keys())
sl2 = L.slope_table(fitted, newmods, g_all=rows)
d2 = L.attach_sets(sl2)
d2.to_csv(OUT / "checkB_fixedcycle_slopes.csv", index=False)

b_rows = []
for m in newmods:
    for s in ("S8", "S14", "S3"):
        d = d2[(d2.model == m) & d2[s]]
        x = d.slope.to_numpy(float)
        y = d.a.to_numpy(float)
        r, p, n = rho(x, y)
        d1 = d[d.chemotype != "sc009"]
        r1, p1, n1 = rho(d1.slope.to_numpy(float), d1.a.to_numpy(float))
        vc = [rho(d[d.chemotype != ch].slope.to_numpy(float), d[d.chemotype != ch].a.to_numpy(float))[0]
              for ch in sorted(d.chemotype.unique())]
        lo, hi = fisher_ci(r, n)
        b_rows.append(dict(model=m, set=s, n=n, rho=r, p=p, fisher_lo=lo, fisher_hi=hi,
                           partial_n_metals=L._partial_rho(x, y, d.n_metals.to_numpy(float)),
                           rho_no_sc009=r1, n_no_sc009=n1,
                           loco_min=float(np.nanmin(vc)), loco_max=float(np.nanmax(vc)),
                           rho_abs_a=rho(x, np.abs(y))[0], rho_b=rho(x, d.b.to_numpy(float))[0]))
b = pd.DataFrame(b_rows)
b.to_csv(OUT / "checkB_fixedcycle_stats.csv", index=False)
print(b.round(4).to_string(index=False))

# ==================================================================== (c) set / statistic sensitivity
log("=== (c) Spearman vs Pearson vs Kendall, set changes ===")
c_rows = []
allm = list(("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "SPECIES_NFILLCOL")) + list(newmods)
big = pd.concat([df, d2], ignore_index=True)
for m in allm:
    for s in ("S8", "S14", "S3"):
        d = big[(big.model == m) & big[s]]
        if len(d) < 8:
            continue
        x = d.slope.to_numpy(float)
        y = d.a.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        c_rows.append(dict(model=m, set=s, n=int(ok.sum()),
                           spearman=stats.spearmanr(x[ok], y[ok]).statistic,
                           pearson=stats.pearsonr(x[ok], y[ok]).statistic,
                           pearson_log=stats.pearsonr(np.sign(x[ok]) * np.log1p(np.abs(x[ok])), y[ok]).statistic,
                           kendall=stats.kendalltau(x[ok], y[ok]).statistic))
c = pd.DataFrame(c_rows)
c.to_csv(OUT / "checkC_statistics.csv", index=False)
print(c.round(4).to_string(index=False))

# S14's 39 evaluated under every model, and S8 restricted to the S14 extractants
log("--- the 39-vs-62 set change ---")
s14_ext = set(df[(df.model == "NAIVE") & df.S14].extractant)
sc = []
for m in allm:
    d = big[(big.model == m)]
    for lab, dd in (("S8 all 62", d[d.S8]), ("S8 minus S14 (23)", d[d.S8 & ~d.extractant.isin(s14_ext)]),
                    ("S14 (39)", d[d.S14])):
        r, p, n = rho(dd.slope.to_numpy(float), dd.a.to_numpy(float))
        sc.append(dict(model=m, subset=lab, n=n, rho=r, p=p))
sc = pd.DataFrame(sc)
sc.to_csv(OUT / "checkC_setchange.csv", index=False)
print(sc.round(4).to_string(index=False))
log("part3 done")
