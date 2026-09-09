"""D6 step 2: per-extractant regression of the rank-1 amplitude on log10 acid
concentration and log10 extractant concentration.  Is there a mass-action law?"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from d6_common import load_cohort, cell_amplitudes, collapse_onehot, OUT, FIG, MIN_RZ_SPAN

df = load_cohort()
amp = cell_amplitudes(df)
cell = df.merge(amp, on="cell_id")
cell = cell[cell.rz_span >= MIN_RZ_SPAN].copy()
cell["log_acid"] = np.log10(cell["cond__acid_concentration_M"].astype(float))
cell["log_ext"] = np.log10(cell["cond__extractant_concentration_M"].astype(float))
cell["diluent_id"] = collapse_onehot(cell, "cond__diluent__", "diluent_id")
cell["acid_id"] = collapse_onehot(cell, "cond__acid__", "acid_id")


def ols_slope(x: np.ndarray, y: np.ndarray):
    """slope, se, t, p, r2, n for a simple OLS y ~ 1 + x."""
    n = len(x)
    if n < 3 or np.std(x) < 1e-9:
        return dict(slope=np.nan, se=np.nan, t=np.nan, p=np.nan, r2=np.nan, n=n)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    if dof < 1:
        return dict(slope=float(beta[1]), se=np.nan, t=np.nan, p=np.nan, r2=np.nan, n=n)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    t = float(beta[1] / se) if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else np.nan
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else np.nan
    return dict(slope=float(beta[1]), se=se, t=t, p=p, r2=r2, n=n)


def multi_slopes(g: pd.DataFrame):
    """joint fit amp ~ 1 + log_acid + log_ext (both slopes at once)."""
    s = g.dropna(subset=["log_acid", "log_ext"])
    if len(s) < 4 or s.log_acid.nunique() < 2 or s.log_ext.nunique() < 2:
        return dict(joint_acid_slope=np.nan, joint_ext_slope=np.nan, joint_n=len(s))
    X = np.column_stack([np.ones(len(s)), s.log_acid, s.log_ext])
    if np.linalg.matrix_rank(X) < 3:
        return dict(joint_acid_slope=np.nan, joint_ext_slope=np.nan, joint_n=len(s))
    b, *_ = np.linalg.lstsq(X, s.amp.to_numpy(), rcond=None)
    return dict(joint_acid_slope=float(b[1]), joint_ext_slope=float(b[2]),
                joint_n=len(s))


# --- extractants that qualify: >=3 amplitude cells spanning >=2 acid concentrations ---
rows = []
for k, g in cell.groupby("extractant"):
    ga = g.dropna(subset=["log_acid"])
    if len(ga) < 3 or ga.log_acid.nunique() < 2:
        continue
    ra = ols_slope(ga.log_acid.to_numpy(), ga.amp.to_numpy())
    ge = g.dropna(subset=["log_ext"])
    re_ = ols_slope(ge.log_ext.to_numpy(), ge.amp.to_numpy()) if \
        (len(ge) >= 3 and ge.log_ext.nunique() >= 2) else \
        dict(slope=np.nan, se=np.nan, t=np.nan, p=np.nan, r2=np.nan, n=len(ge))
    rec = dict(extractant_name=g.extractant_name.iloc[0], chemotype=g.chemotype.iloc[0],
               n_cells=len(g), n_acid_cells=len(ga),
               acid_levels=int(ga.log_acid.nunique()),
               acid_M_min=float(10 ** ga.log_acid.min()),
               acid_M_max=float(10 ** ga.log_acid.max()),
               amp_mean=float(g.amp.mean()), amp_sd=float(g.amp.std(ddof=1)),
               acid_slope=ra["slope"], acid_se=ra["se"], acid_t=ra["t"],
               acid_p=ra["p"], acid_r2=ra["r2"],
               n_ext_cells=re_["n"], ext_levels=int(ge.log_ext.nunique()),
               ext_M_min=float(ge["cond__extractant_concentration_M"].min()),
               ext_M_max=float(ge["cond__extractant_concentration_M"].max()),
               ext_slope=re_["slope"], ext_se=re_["se"], ext_t=re_["t"],
               ext_p=re_["p"], ext_r2=re_["r2"],
               n_publications=int(g.publication_id.nunique()),
               n_diluents=int(g.diluent_id.nunique()),
               n_acid_types=int(g.acid_id.nunique()))
    rec.update(multi_slopes(g))
    rows.append(rec)
sl = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
sl.to_csv(OUT + "/d6_step2_per_extractant_slopes.csv", index=False)

print("=== per-extractant slope of rank-1 amplitude vs log10 concentration ===")
print("(amplitude unit: logD per standardised-radius unit; slope unit: same per decade)")
cshow = ["extractant_name", "chemotype", "n_acid_cells", "acid_levels", "acid_M_min",
         "acid_M_max", "acid_slope", "acid_se", "acid_p", "acid_r2",
         "n_ext_cells", "ext_levels", "ext_slope", "ext_se", "ext_p", "ext_r2",
         "joint_acid_slope", "joint_ext_slope", "n_publications", "n_diluents"]
print(sl[cshow].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

qual = sl.dropna(subset=["acid_slope"])
print(f"\nextractants qualifying (>=3 amplitude cells, >=2 acid concentrations): {len(qual)}")


def summarise(s: pd.DataFrame, scol: str, pcol: str, label: str):
    v = s.dropna(subset=[scol])
    if len(v) == 0:
        print(f"\n{label}: no extractant qualifies")
        return None
    sig = v[v[pcol] < 0.05]
    pos, neg = int((v[scol] > 0).sum()), int((v[scol] < 0).sum())
    sp = stats.binomtest(max(pos, neg), len(v), 0.5).pvalue if len(v) > 0 else np.nan
    w = stats.wilcoxon(v[scol]) if len(v) >= 6 else None
    tt = stats.ttest_1samp(v[scol], 0.0) if len(v) >= 3 else None
    print(f"\n--- {label} (n = {len(v)} extractants) ---")
    print(f"  slopes: median {v[scol].median():.3f}, mean {v[scol].mean():.3f}, "
          f"sd {v[scol].std(ddof=1):.3f}, IQR [{v[scol].quantile(.25):.3f}, "
          f"{v[scol].quantile(.75):.3f}], range [{v[scol].min():.3f}, {v[scol].max():.3f}]")
    print(f"  sign: {pos} positive / {neg} negative -> sign test p = {sp:.3f}")
    print(f"  significantly non-zero at p<0.05: {len(sig)} of {len(v)} "
          f"({', '.join(sig.extractant_name.tolist()) if len(sig) else 'none'})")
    if len(sig):
        print(f"     of those, {(sig[scol] > 0).sum()} positive / "
              f"{(sig[scol] < 0).sum()} negative")
    if tt is not None:
        print(f"  one-sample t-test that the mean slope is 0: t = {tt.statistic:.2f}, "
              f"p = {tt.pvalue:.3f}")
    if w is not None:
        print(f"  Wilcoxon signed-rank vs 0: p = {w.pvalue:.3f}")
    # inverse-variance weighted pooled slope (random signs would cancel)
    vv = v.dropna(subset=[scol.replace("slope", "se")])
    secol = scol.replace("slope", "se")
    vv = vv[vv[secol] > 0]
    if len(vv) >= 2:
        wgt = 1.0 / vv[secol] ** 2
        pooled = float((wgt * vv[scol]).sum() / wgt.sum())
        pse = float(np.sqrt(1.0 / wgt.sum()))
        q = float((wgt * (vv[scol] - pooled) ** 2).sum())
        pq = float(stats.chi2.sf(q, len(vv) - 1))
        print(f"  fixed-effect pooled slope {pooled:+.3f} +/- {pse:.3f} "
              f"(z = {pooled/pse:.2f}); heterogeneity Q = {q:.1f} on {len(vv)-1} df, "
              f"p = {pq:.2e} -> {'slopes are NOT a common constant' if pq < 0.05 else 'consistent with one constant'}")
    print("  by chemotype:")
    for ct, gg in v.groupby("chemotype"):
        print(f"    {ct}: n = {len(gg)}, median {gg[scol].median():+.3f}, "
              f"{int((gg[scol] > 0).sum())} pos / {int((gg[scol] < 0).sum())} neg")
    return v


va = summarise(qual, "acid_slope", "acid_p", "amplitude vs log10 [acid] (M)")
ve = summarise(qual, "ext_slope", "ext_p", "amplitude vs log10 [extractant] (M)")
vj_a = qual.dropna(subset=["joint_acid_slope"])
if len(vj_a):
    print(f"\n--- joint fit amp ~ log10[acid] + log10[extractant] "
          f"(n = {len(vj_a)} extractants) ---")
    print(f"  acid slope: median {vj_a.joint_acid_slope.median():+.3f}, "
          f"{int((vj_a.joint_acid_slope > 0).sum())} pos / "
          f"{int((vj_a.joint_acid_slope < 0).sum())} neg")
    print(f"  ext  slope: median {vj_a.joint_ext_slope.median():+.3f}, "
          f"{int((vj_a.joint_ext_slope > 0).sum())} pos / "
          f"{int((vj_a.joint_ext_slope < 0).sum())} neg")

# --- confound check: same-publication, same-diluent, same-acid-type subsets -----------
print("\n=== confound-controlled slopes: within (extractant x publication x diluent x "
      "acid type x metal set) strata with >=3 cells and >=2 acid levels ===")
strat = []
for keys, g in cell.groupby(["extractant", "publication_id", "diluent_id", "acid_id",
                             "metal_set"]):
    ga = g.dropna(subset=["log_acid"])
    if len(ga) < 3 or ga.log_acid.nunique() < 2:
        continue
    r = ols_slope(ga.log_acid.to_numpy(), ga.amp.to_numpy())
    strat.append(dict(extractant_name=g.extractant_name.iloc[0],
                      chemotype=g.chemotype.iloc[0], publication_id=keys[1],
                      diluent=keys[2], acid=keys[3], n_metals=g.n_metals.iloc[0],
                      n_cells=len(ga), acid_levels=int(ga.log_acid.nunique()),
                      acid_slope=r["slope"], acid_se=r["se"], acid_p=r["p"],
                      acid_r2=r["r2"], amp_mean=float(ga.amp.mean())))
st = pd.DataFrame(strat)
st.to_csv(OUT + "/d6_step2_stratified_acid_slopes.csv", index=False)
if len(st):
    print(st.drop(columns=["publication_id"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  {len(st)} strata over {st.extractant_name.nunique()} extractants and "
          f"{st.chemotype.nunique()} chemotypes")
    print(f"  slopes: median {st.acid_slope.median():+.3f}, mean "
          f"{st.acid_slope.mean():+.3f}, sd {st.acid_slope.std(ddof=1):.3f}")
    print(f"  sign: {int((st.acid_slope > 0).sum())} pos / "
          f"{int((st.acid_slope < 0).sum())} neg, sign-test p = "
          f"{stats.binomtest(int((st.acid_slope > 0).sum()), len(st), 0.5).pvalue:.3f}")
    print(f"  significant at p<0.05: {int((st.acid_p < 0.05).sum())} of {len(st)}")
    ts = stats.ttest_1samp(st.acid_slope.dropna(), 0.0)
    print(f"  t-test mean slope = 0: t = {ts.statistic:.2f}, p = {ts.pvalue:.4f}")

# --- one common slope with stratum fixed effects, cluster-robust by extractant -------
def common_slope(data: pd.DataFrame, xcol: str, strata: list[str], cluster: str):
    """Single global slope of amp on xcol after absorbing a fixed effect per stratum.
    SE is clustered on `cluster` (the extractant), which is the unit of independence."""
    d = data.dropna(subset=[xcol, "amp"]).copy()
    d["_s"] = d[strata].astype(str).agg("|".join, axis=1)
    ok = (d.groupby("_s")[xcol].transform("nunique") >= 2) & \
         (d.groupby("_s")[xcol].transform("size") >= 2)
    d = d[ok]
    if len(d) < 4:
        return None
    xd = d[xcol].astype(float) - d.groupby("_s")[xcol].transform("mean").astype(float)
    yd = d.amp - d.groupby("_s").amp.transform("mean")
    xd, yd = xd.to_numpy(), yd.to_numpy()
    den = float(xd @ xd)
    b = float(xd @ yd) / den
    r = yd - b * xd
    meat = 0.0
    for _, g in pd.DataFrame({"c": d[cluster].to_numpy(), "x": xd, "r": r}).groupby("c"):
        meat += float((g.x * g.r).sum()) ** 2
    se = float(np.sqrt(meat) / den)
    nclust = d[cluster].nunique()
    t = b / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), max(nclust - 1, 1)))
    return dict(slope=b, se_cluster=se, t=t, p=p, n_cells=len(d),
                n_strata=int(d["_s"].nunique()), n_clusters=int(nclust))


print("\n=== one common slope, stratum fixed effects, SE clustered on extractant ===")
common_rows = []
for xcol, lab in [("log_acid", "log10 [acid]"), ("log_ext", "log10 [extractant]")]:
    for strata, slab in [(["extractant"], "extractant"),
                         (["extractant", "publication_id"], "extractant x publication"),
                         (["extractant", "publication_id", "diluent_id", "acid_id",
                           "metal_set"], "extractant x pub x diluent x acid x metalset")]:
        r = common_slope(cell, xcol, strata, "extractant")
        if r is None:
            continue
        r.update(predictor=lab, strata=slab)
        common_rows.append(r)
        print(f"  amp ~ {lab:22s} | {slab:48s} slope {r['slope']:+.3f} "
              f"+/- {r['se_cluster']:.3f} (p = {r['p']:.4f}), {r['n_cells']} cells, "
              f"{r['n_strata']} strata, {r['n_clusters']} extractant clusters")
pd.DataFrame(common_rows).to_csv(OUT + "/d6_step2_common_slopes.csv", index=False)

# --- reference: how big is one decade of acid vs the noise floor ---------------------
if va is not None:
    med_abs = va.acid_slope.abs().median()
    print(f"\nreference scale: |median acid slope| = {med_abs:.3f} amplitude units per "
          f"decade of [acid]; within-extractant amplitude sd = "
          f"{cell.groupby('extractant').amp.transform('std').median():.3f}; "
          f"median amplitude standard error = {cell.amp_se.median():.3f}")
print("\nwrote d6_step2_per_extractant_slopes.csv, d6_step2_stratified_acid_slopes.csv")
