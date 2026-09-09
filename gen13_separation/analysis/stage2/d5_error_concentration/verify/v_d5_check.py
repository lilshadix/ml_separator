"""Independent re-derivation of the numbers the D5 conclusion rests on.

Written from scratch without reading the D5 scripts. Run from repo root:
    .venv/Scripts/python.exe gen13_separation/analysis/stage2/d5_error_concentration/verify/v_d5_check.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "gen13_separation"
OUT = os.path.join(ROOT, "analysis", "stage2", "d5_error_concentration", "verify")
os.makedirs(OUT, exist_ok=True)

BEST = "X_ENS_DIRECT+LOWRANK_K2"
INCUMBENT = "C_DIRECT_ROW"
MEANCURVE = "B1_MEAN_CURVE"
LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

results = {}


def load(arm):
    return pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary", f"{arm}.parquet"))


# ---------------------------------------------------------------- 0. integrity
def check_integrity(df, cohort):
    """y must be an exact difference of two OBSERVED logD values in that cell."""
    rep = {}
    long = cohort.set_index("cell_id")
    lg = long[[f"logD__{m}" for m in LN]]
    lg.columns = LN
    # map (cell, metal) -> logD
    stacked = lg.stack(future_stack=True).dropna()
    stacked.index.names = ["cell_id", "metal"]
    d = stacked.to_dict()
    key_a = list(zip(df.cell_id, df.A))
    key_b = list(zip(df.cell_id, df.B))
    va = np.array([d.get(k, np.nan) for k in key_a])
    vb = np.array([d.get(k, np.nan) for k in key_b])
    rep["pairs_with_missing_endpoint"] = int(np.isnan(va).sum() + np.isnan(vb).sum())
    diff = va - vb
    err = np.abs(diff - df.y.values)
    rep["max_abs_dev_y_vs_logD_difference"] = float(np.nanmax(err))
    rep["n_pairs_y_mismatch_gt_1e-9"] = int(np.nansum(err > 1e-9))
    # dZ consistency
    z = {m: i for i, m in enumerate(LN)}  # index order == atomic number order (Pm absent)
    # true atomic numbers
    ZTRUE = dict(zip(LN, [57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]))
    dz = np.array([ZTRUE[b] - ZTRUE[a] for a, b in zip(df.A, df.B)])
    rep["n_dZ_mismatch"] = int((dz != df.dZ.values).sum())
    rep["n_dZ_nonpositive"] = int((df.dZ.values <= 0).sum())
    # every cell appears in exactly one fold per seed  -> held out once
    g = df.groupby(["split_seed", "cell_id"])["fold"].nunique()
    rep["cells_in_more_than_one_fold_per_seed"] = int((g > 1).sum())
    # chemotype grouping: a chemotype must not straddle folds within a seed
    gc = df.groupby(["split_seed", "chemotype"])["fold"].nunique()
    rep["chemotypes_straddling_folds_per_seed"] = int((gc > 1).sum())
    rep["n_chemotypes"] = int(df.chemotype.nunique())
    # pair uniqueness: each (seed, cell, A, B) once
    rep["duplicate_seed_cell_pair_rows"] = int(df.duplicated(["split_seed", "cell_id", "A", "B"]).sum())
    # every cell held out in every seed?
    rep["cells_per_seed"] = df.groupby("split_seed")["cell_id"].nunique().to_dict()
    rep["cells_total"] = int(df.cell_id.nunique())
    # pairs per cell == C(n_metals,2)?
    npair = df[df.split_seed == df.split_seed.iloc[0]].groupby("cell_id").size()
    nm = cohort.set_index("cell_id")["n_metals"]
    expect = (nm * (nm - 1) / 2).astype(int)
    common = npair.index.intersection(expect.index)
    rep["cells_with_pair_count_ne_choose2"] = int((npair.loc[common] != expect.loc[common]).sum())
    return rep


# --------------------------------------------- 1. extractant-macro MAE (anchor)
def macro_mae(df, subset_extractants=None):
    """Per-extractant MAE within a seed, mean over extractants, mean over seeds."""
    d = df
    if subset_extractants is not None:
        d = d[d.extractant.isin(subset_extractants)]
    per = d.assign(ae=(d.y - d.prediction).abs()).groupby(["split_seed", "extractant"])["ae"].mean()
    per_seed = per.groupby("split_seed").mean()
    return float(per_seed.mean()), per_seed


def per_extractant_table(df):
    """MAE per (seed, extractant) then averaged over the 5 seeds."""
    d = df.assign(ae=(df.y - df.prediction).abs())
    per = d.groupby(["split_seed", "extractant"])["ae"].mean().unstack(0)
    return per


# --------------------------------------------------------------------- 3. ICC
def icc1(mat):
    """mat: n_extractants x k_seeds. One-way random-effects ICC(1) via ANOVA."""
    x = mat.values.astype(float)
    n, k = x.shape
    row = x.mean(axis=1)
    grand = x.mean()
    msb = k * ((row - grand) ** 2).sum() / (n - 1)
    msw = ((x - row[:, None]) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)), float(msb), float(msw)


def var_ratio_icc(mat):
    """The naive 'between var / (between var + within var)' form."""
    x = mat.values.astype(float)
    vb = float(np.var(x.mean(axis=1), ddof=1))
    vw = float(np.mean(np.var(x, axis=1, ddof=1)))
    return vb / (vb + vw), vb, vw


# --------------------------------------------------- 4. split-half of residual
def split_half_residual(df, rng_seed=0, n_splits=20, min_shared=6):
    """For extractants with >=2 cells: mean signed residual per (A,B) pair over one
    random half of the cells vs the other half, Pearson r on shared pairs."""
    d = df.assign(res=df.y - df.prediction)
    # average the residual over the 5 seeds first: one residual per (cell, pair)
    cellpair = d.groupby(["extractant", "cell_id", "A", "B"])["res"].mean().reset_index()
    cellpair["pair"] = cellpair.A + "_" + cellpair.B
    rng = np.random.default_rng(rng_seed)
    rows = []
    for ext, g in cellpair.groupby("extractant"):
        cells = g.cell_id.unique()
        if len(cells) < 2:
            continue
        piv = g.pivot_table(index="cell_id", columns="pair", values="res")
        rs = []
        for _ in range(n_splits):
            perm = rng.permutation(len(cells))
            h1 = cells[perm[: len(cells) // 2]]
            h2 = cells[perm[len(cells) // 2:]]
            if len(h1) == 0 or len(h2) == 0:
                continue
            m1 = piv.loc[h1].mean(axis=0)
            m2 = piv.loc[h2].mean(axis=0)
            ok = m1.notna() & m2.notna()
            if ok.sum() < min_shared:
                continue
            a, b = m1[ok].values, m2[ok].values
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        if rs:
            rows.append({"extractant": ext, "n_cells": len(cells),
                         "n_shared_pairs_max": int(piv.notna().all(axis=0).sum()),
                         "n_valid_splits": len(rs), "mean_r": float(np.mean(rs))})
    return pd.DataFrame(rows)


# ------------------------------------------------------- 5. cell-domination test
def macro_mae_cellbalanced(df, subset=None):
    """Same macro convention but with cells given equal weight inside an extractant."""
    d = df if subset is None else df[df.extractant.isin(subset)]
    d = d.assign(ae=(d.y - d.prediction).abs())
    per_cell = d.groupby(["split_seed", "extractant", "cell_id"])["ae"].mean()
    per_ext = per_cell.groupby(["split_seed", "extractant"]).mean()
    return float(per_ext.groupby("split_seed").mean().mean())


def main():
    cohort = pd.read_parquet(os.path.join(ROOT, "manifests", "cohort_exact.parquet"))
    best = load(BEST)
    inc = load(INCUMBENT)
    mc = load(MEANCURVE)

    print("=" * 72)
    print("0. INTEGRITY / OUT-OF-SAMPLE CHECKS")
    integ = check_integrity(best, cohort)
    for k, v in integ.items():
        print(f"   {k}: {v}")
    results["integrity"] = integ

    print("=" * 72)
    print("1. ANCHOR: extractant-macro MAE")
    for name, d in [(BEST, best), (INCUMBENT, inc), (MEANCURVE, mc)]:
        m, per_seed = macro_mae(d)
        pooled = float((d.y - d.prediction).abs().mean())
        print(f"   {name:28s} macro={m:.4f}  pooled={pooled:.4f}  "
              f"per-seed={np.round(per_seed.values, 4).tolist()}")
        results[f"macro_{name}"] = m
        results[f"pooled_{name}"] = pooled
    print(f"   n_extractants={best.extractant.nunique()}  "
          f"pairs/seed={len(best)//best.split_seed.nunique()}  total={len(best)}")

    print("=" * 72)
    print("2. CONCENTRATION: worst-10 share and macro after removal")
    tab = per_extractant_table(best)
    tab["mae_mean"] = tab.mean(axis=1)
    tab = tab.sort_values("mae_mean", ascending=False)
    tot = tab["mae_mean"].sum()
    for k in (5, 10, 20):
        share = tab["mae_mean"].head(k).sum() / tot
        print(f"   worst {k:2d}: share of summed per-extractant MAE = {share*100:.2f}%")
        results[f"worst{k}_share"] = float(share)
    med = tab["mae_mean"].median()
    exc = (tab["mae_mean"] - med).clip(lower=0)
    results["worst10_excess_share"] = float(exc.head(10).sum() / exc.sum())
    print(f"   worst 10 share of excess above median extractant = "
          f"{results['worst10_excess_share']*100:.1f}%")
    # Gini
    v = np.sort(tab["mae_mean"].values)
    n = len(v)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))
    results["gini_per_extractant_mae"] = gini
    print(f"   Gini of per-extractant MAE = {gini:.4f}")
    print(f"   median={med:.4f} p90={np.percentile(tab['mae_mean'],90):.4f} "
          f"min={tab['mae_mean'].min():.4f} max={tab['mae_mean'].max():.4f}")

    for k in (10, 20):
        drop = set(tab.index[:k])
        keep = [e for e in tab.index if e not in drop]
        mb, _ = macro_mae(best, keep)
        mi, _ = macro_mae(inc, keep)
        print(f"   drop worst {k:2d}: best-arm macro={mb:.4f}  incumbent macro={mi:.4f} "
              f"(n kept={len(keep)})")
        results[f"macro_after_drop_worst{k}"] = mb
        results[f"macro_after_drop_worst{k}_incumbent"] = mi
    tab.to_csv(os.path.join(OUT, "v_per_extractant_mae.csv"))

    print("=" * 72)
    print("3. ICC of per-extractant MAE across the 5 split seeds")
    mat = tab.drop(columns=["mae_mean"])
    icc, msb, msw = icc1(mat)
    vr, vb, vw = var_ratio_icc(mat)
    print(f"   ANOVA ICC(1) = {icc:.4f}   (MSB={msb:.4f}, MSW={msw:.4f})")
    print(f"   naive var-ratio form = {vr:.4f}  (var_between={vb:.4f}, "
          f"mean var_within={vw:.4f})")
    results["icc1_anova"] = icc
    results["icc_var_ratio"] = vr
    results["var_between"] = vb
    results["var_within"] = vw

    print("=" * 72)
    print("4. Split-half reproducibility of the signed residual")
    sh = split_half_residual(best)
    print(f"   n extractants entering = {len(sh)} "
          f"(of {best.groupby('extractant')['cell_id'].nunique().gt(1).sum()} with >=2 cells)")
    print(f"   mean r = {sh.mean_r.mean():.4f}   median r = {sh.mean_r.median():.4f}")
    print(f"   frac r > 0.5 = {(sh.mean_r > 0.5).mean():.3f}   frac r > 0 = "
          f"{(sh.mean_r > 0).mean():.3f}")
    results["splithalf_mean_r"] = float(sh.mean_r.mean())
    results["splithalf_median_r"] = float(sh.mean_r.median())
    results["splithalf_n_extractants"] = int(len(sh))
    results["splithalf_frac_gt_0.5"] = float((sh.mean_r > 0.5).mean())
    # seed-sensitivity of the split-half number
    alt = [float(split_half_residual(best, rng_seed=s).mean_r.mean()) for s in (1, 2, 3)]
    print(f"   re-run with other RNG seeds: {np.round(alt,4).tolist()}")
    results["splithalf_mean_r_other_rng"] = alt
    sh.to_csv(os.path.join(OUT, "v_split_half_residual.csv"), index=False)

    print("=" * 72)
    print("5. ROBUSTNESS: does one big cell dominate an extractant?")
    mb_cell = macro_mae_cellbalanced(best)
    mi_cell = macro_mae_cellbalanced(inc)
    print(f"   pair-mean-within-extractant (convention) best={results['macro_'+BEST]:.4f}")
    print(f"   cell-balanced-within-extractant           best={mb_cell:.4f}  "
          f"incumbent={mi_cell:.4f}")
    results["macro_cellbalanced_best"] = mb_cell
    # how lopsided are extractants?
    one = best[best.split_seed == best.split_seed.iloc[0]]
    npc = one.groupby(["extractant", "cell_id"]).size()
    frac = npc.groupby("extractant").max() / npc.groupby("extractant").sum()
    multi = best.groupby("extractant")["cell_id"].nunique()
    multi_ext = multi[multi > 1].index
    print(f"   among {len(multi_ext)} multi-cell extractants, median share of pairs held "
          f"by the largest cell = {frac.loc[multi_ext].median():.3f}, "
          f"max = {frac.loc[multi_ext].max():.3f}")
    results["median_largest_cell_pair_share_multicell"] = float(frac.loc[multi_ext].median())
    # pairs per seed per extractant, min/max
    pps = one.groupby("extractant").size()
    print(f"   pairs/seed per extractant: min={pps.min()}, median={pps.median():.0f}, "
          f"max={pps.max()}")
    results["pairs_per_extractant_min"] = int(pps.min())
    results["pairs_per_extractant_max"] = int(pps.max())

    print("=" * 72)
    print("6. dZ=1 adjacent-pair macro gain over the mean curve")
    def macro_on(d, mask_fn):
        dd = d[mask_fn(d)]
        per = dd.assign(ae=(dd.y - dd.prediction).abs()).groupby(
            ["split_seed", "extractant"])["ae"].mean()
        return float(per.groupby("split_seed").mean().mean()), dd
    m1b, dd = macro_on(best, lambda d: d.dZ == 1)
    m1i, _ = macro_on(inc, lambda d: d.dZ == 1)
    m1m, _ = macro_on(mc, lambda d: d.dZ == 1)
    print(f"   dZ=1 macro: best={m1b:.4f} incumbent={m1i:.4f} meancurve={m1m:.4f} "
          f"gain={m1m-m1b:.4f}")
    print(f"   dZ=1 pairs/seed = {len(dd)//5}, extractants with dZ=1 = "
          f"{dd.extractant.nunique()}")
    results["dz1_macro_best"] = m1b
    results["dz1_macro_incumbent"] = m1i
    results["dz1_macro_meancurve"] = m1m

    with open(os.path.join(OUT, "v_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 72)
    print("wrote", os.path.join(OUT, "v_results.json"))


if __name__ == "__main__":
    main()
