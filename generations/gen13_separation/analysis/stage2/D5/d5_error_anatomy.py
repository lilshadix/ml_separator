"""D5 - where the held-out error lives, and can it be predicted without the label.

Run from repo root:
    .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/D5/d5_error_anatomy.py

Writes CSVs into gen13_separation/analysis/stage2/D5/ and figures into
gen13_separation/figures/stage2/ (filenames prefixed d5_).
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "generations", "gen13_separation"))
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER  # noqa: E402

OUT = os.path.join(ROOT, "generations", "gen13_separation", "analysis", "stage2", "D5")
FIG = os.path.join(ROOT, "generations", "gen13_separation", "figures", "stage2")
PRED = os.path.join(ROOT, "generations", "gen13_separation", "predictions", "B_primary")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

BEST = "X_ENS_DIRECT+LOWRANK_K2"
INCUMBENT = "C_DIRECT_ROW"
MEANCURVE = "B1_MEAN_CURVE"


def load(arm: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(PRED, f"{arm}.parquet"))
    df["err"] = df["y"] - df["prediction"]
    df["ae"] = df["err"].abs()
    return df


def macro(df: pd.DataFrame, col: str = "ae", by=()) -> pd.DataFrame:
    """extractant-macro: mean within (extractant[,by]) -> mean over extractants -> mean over seeds."""
    by = list(by)
    g = df.groupby(["split_seed", "extractant"] + by, observed=True)[col].mean().reset_index()
    g = g.groupby(["split_seed"] + by, observed=True)[col].mean().reset_index()
    return g.groupby(by, observed=True)[col].mean().reset_index() if by else pd.DataFrame({col: [g[col].mean()]})


def macro_scalar(df: pd.DataFrame, col: str = "ae") -> float:
    return float(
        df.groupby(["split_seed", "extractant"], observed=True)[col].mean()
        .groupby("split_seed").mean().mean()
    )


def cut_table(df: pd.DataFrame, key) -> pd.DataFrame:
    """MAE / mean|y| / ratio per level of `key`, pooled and extractant-macro."""
    keys = [key] if isinstance(key, str) else list(key)
    d = df.copy()
    d["absy"] = d["y"].abs()
    pooled = d.groupby(keys, observed=True).agg(
        n_pairs=("ae", "size"),
        n_extractants=("extractant", "nunique"),
        n_cells=("cell_id", "nunique"),
        mae_pooled=("ae", "mean"),
        mean_abs_y=("absy", "mean"),
        mae_pooled_incumbent=("ae_inc", "mean"),
        mae_pooled_meancurve=("ae_mc", "mean"),
        frac_beats_meancurve=("beats_mc", "mean"),
    ).reset_index()
    mm = macro(d, "ae", keys).rename(columns={"ae": "mae_macro"})
    my = macro(d, "absy", keys).rename(columns={"absy": "mean_abs_y_macro"})
    mc = macro(d, "ae_mc", keys).rename(columns={"ae_mc": "mae_macro_meancurve"})
    out = pooled.merge(mm, on=keys).merge(my, on=keys).merge(mc, on=keys)
    out["ratio_pooled"] = out["mae_pooled"] / out["mean_abs_y"]
    out["ratio_macro"] = out["mae_macro"] / out["mean_abs_y_macro"]
    out["n_pairs_per_seed"] = out["n_pairs"] / d["split_seed"].nunique()
    return out


def main() -> None:
    best = load(BEST)
    inc = load(INCUMBENT)
    mcv = load(MEANCURVE)

    key = ["split_seed", "cell_id", "A", "B"]
    best = best.merge(inc[key + ["prediction", "ae"]].rename(
        columns={"prediction": "pred_inc", "ae": "ae_inc"}), on=key, how="left")
    best = best.merge(mcv[key + ["prediction", "ae"]].rename(
        columns={"prediction": "pred_mc", "ae": "ae_mc"}), on=key, how="left")
    assert best[["ae_inc", "ae_mc"]].notna().all().all()
    best["beats_mc"] = (best["ae"] < best["ae_mc"]).astype(float)
    best["beats_inc"] = (best["ae"] < best["ae_inc"]).astype(float)

    # cell metadata
    coh = pd.read_parquet(os.path.join(ROOT, "generations", "gen13_separation", "manifests", "cohort_exact.parquet"))
    meta_cols = ["cell_id", "extractant_name", "publication_id", "n_rows", "replicate_sd_median",
                 "chem_family", "ecfp_cluster", "cond__acid_concentration_M",
                 "cond__extractant_concentration_M", "cond__temperature_C",
                 "recipe__DENTATE", "recipe__coreCN"]
    best = best.merge(coh[meta_cols], on="cell_id", how="left")

    sim = pd.read_parquet(os.path.join(PRED, "similarity.parquet"))
    best = best.merge(sim[["split_seed", "cell_id", "max_train_tanimoto", "band"]],
                      on=["split_seed", "cell_id"], how="left")
    print("similarity coverage:", best["max_train_tanimoto"].notna().mean())

    # chemotype size (cells per chemotype in the cohort)
    ct_size = coh.groupby("chemotype").size().rename("chemotype_n_cells")
    best = best.merge(ct_size, left_on="chemotype", right_index=True, how="left")

    best["absy"] = best["y"].abs()
    best["Z_A"] = best["A"].map(ATOMIC_NUMBER)
    best["Z_B"] = best["B"].map(ATOMIC_NUMBER)

    n_seeds = best["split_seed"].nunique()
    macro_best = macro_scalar(best, "ae")
    macro_inc = macro_scalar(best, "ae_inc")
    macro_mc = macro_scalar(best, "ae_mc")
    print(f"\n=== HEADLINE === pairs={len(best)} ({len(best)//n_seeds}/seed) cells={best.cell_id.nunique()} "
          f"extractants={best.extractant.nunique()} seeds={n_seeds}")
    print(f"macro MAE  best={macro_best:.4f}  incumbent={macro_inc:.4f}  meancurve={macro_mc:.4f}")
    print(f"pooled MAE best={best.ae.mean():.4f}  incumbent={best.ae_inc.mean():.4f}  meancurve={best.ae_mc.mean():.4f}")
    print(f"mean|y| pooled={best.absy.mean():.4f}   frac pairs beating mean curve={best.beats_mc.mean():.4f}")

    # ---------------- 1. per-extractant ----------------
    per_ex = (best.groupby(["split_seed", "extractant"], observed=True)
              .agg(mae=("ae", "mean"), mae_inc=("ae_inc", "mean"), mae_mc=("ae_mc", "mean"),
                   mean_abs_y=("absy", "mean"), beats_mc=("beats_mc", "mean"))
              .reset_index()
              .groupby("extractant", observed=True)
              .agg(mae_best=("mae", "mean"), mae_incumbent=("mae_inc", "mean"),
                   mae_meancurve=("mae_mc", "mean"), mean_abs_y=("mean_abs_y", "mean"),
                   frac_beats_meancurve=("beats_mc", "mean"),
                   mae_best_sd_over_seeds=("mae", "std"))
              .reset_index())
    static = (best.drop_duplicates(["extractant", "cell_id"])
              .groupby("extractant", observed=True)
              .agg(chemotype=("chemotype", "first"), n_cells=("cell_id", "nunique"),
                   extractant_name=("extractant_name", "first"),
                   mean_n_metals=("n_metals", "mean"), max_n_metals=("n_metals", "max"),
                   replicate_sd_median=("replicate_sd_median", "median"),
                   n_publications=("publication_id", "nunique")))
    npairs = (best[best.split_seed == best.split_seed.iloc[0]]
              .groupby("extractant", observed=True).size().rename("n_pairs_per_seed"))
    per_ex = per_ex.merge(static, on="extractant").merge(npairs, on="extractant")
    per_ex["ratio_mae_over_absy"] = per_ex["mae_best"] / per_ex["mean_abs_y"]
    per_ex["gain_vs_incumbent"] = per_ex["mae_incumbent"] - per_ex["mae_best"]
    per_ex["gain_vs_meancurve"] = per_ex["mae_meancurve"] - per_ex["mae_best"]
    # pairwise replicate-noise floor
    per_ex["noise_floor_pairwise"] = np.sqrt(2.0) * per_ex["replicate_sd_median"]
    per_ex["at_or_below_floor"] = per_ex["mae_best"] <= per_ex["noise_floor_pairwise"]
    per_ex = per_ex.sort_values("mae_best", ascending=False).reset_index(drop=True)
    per_ex.to_csv(os.path.join(OUT, "d5_per_extractant.csv"), index=False)

    q = per_ex["mae_best"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95])
    print("\n=== 1. per-extractant MAE distribution (best arm, n=%d) ===" % len(per_ex))
    print(q.round(4).to_string())
    print("incumbent:", per_ex["mae_incumbent"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(4).to_string())

    tot = per_ex["mae_best"].sum()
    for k in (5, 10, 15, 20):
        share = per_ex["mae_best"].head(k).sum() / tot
        # excess above the median extractant
        med = per_ex["mae_best"].median()
        excess = (per_ex["mae_best"].head(k) - med).sum() / (per_ex["mae_best"] - med).clip(lower=0).sum()
        print(f"worst {k:2d} extractants: {share:.3f} of the summed per-extractant MAE "
              f"({k/len(per_ex)*100:.1f}% of extractants); {excess:.3f} of the total excess-above-median")
    macro_wo10 = per_ex["mae_best"].iloc[10:].mean()
    macro_wo10_inc = per_ex["mae_incumbent"].sort_values(ascending=False).iloc[10:].mean()
    print(f"macro MAE excluding the worst 10 extractants: best={macro_wo10:.4f} (vs {macro_best:.4f}), "
          f"incumbent={macro_wo10_inc:.4f}")

    cols = ["extractant_name", "chemotype", "n_cells", "n_pairs_per_seed", "mean_n_metals",
            "mean_abs_y", "mae_best", "mae_incumbent", "mae_meancurve", "ratio_mae_over_absy",
            "replicate_sd_median"]
    worst15 = per_ex.head(15)[cols]
    best15 = per_ex.tail(15)[cols].iloc[::-1]
    worst15.to_csv(os.path.join(OUT, "d5_worst15_extractants.csv"), index=False)
    best15.to_csv(os.path.join(OUT, "d5_best15_extractants.csv"), index=False)
    print("\n--- 15 WORST extractants ---")
    print(worst15.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- 15 BEST extractants ---")
    print(best15.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- 5. noise floor ----------------
    nf = per_ex.dropna(subset=["replicate_sd_median"])
    print(f"\n=== 5. replicate-noise floor === extractants with a replicate_sd_median: {len(nf)}/{len(per_ex)}")
    print(f"at or below sqrt(2)*sd floor (best arm): {int(nf.at_or_below_floor.sum())} "
          f"({nf.at_or_below_floor.mean()*100:.1f}% of {len(nf)})")
    inc_floor = nf["mae_incumbent"] <= nf["noise_floor_pairwise"]
    print(f"same for incumbent C_DIRECT_ROW: {int(inc_floor.sum())} ({inc_floor.mean()*100:.1f}%)")
    print(f"pairs covered by at-floor extractants: "
          f"{nf.loc[nf.at_or_below_floor, 'n_pairs_per_seed'].sum():.0f} of {npairs.sum():.0f} per seed")
    print(f"median replicate_sd_median over extractants = {nf.replicate_sd_median.median():.4f} "
          f"-> pairwise floor {np.sqrt(2)*nf.replicate_sd_median.median():.4f}")
    # cohort-level floor
    cohort_sd = coh["replicate_sd_median"].median()
    print(f"cohort median replicate_sd_median = {cohort_sd:.4f} -> pairwise floor {np.sqrt(2)*cohort_sd:.4f}; "
          f"macro MAE / floor = {macro_best/(np.sqrt(2)*cohort_sd):.2f}x")
    nf[["extractant_name", "chemotype", "n_cells", "mae_best", "replicate_sd_median",
        "noise_floor_pairwise", "at_or_below_floor"]].to_csv(
        os.path.join(OUT, "d5_noise_floor_by_extractant.csv"), index=False)

    # ---------------- 2/3. cuts ----------------
    best["pair_type"] = best["A"] + "-" + best["B"]

    def band_of(z):
        if z <= 60:
            return "light(La-Nd)"
        if z <= 65:
            return "mid(Sm-Tb)"
        return "heavy(Dy-Lu)"

    best["group_pair"] = best["Z_A"].map(band_of) + " -> " + best["Z_B"].map(band_of)
    best["crosses_Gd"] = np.where((best["Z_A"] < 64) & (best["Z_B"] > 64), "crosses Gd",
                                  np.where(best["Z_B"] <= 64, "both light of Gd", "both heavy of Gd"))
    best["adjacent"] = np.where(best["dZ"] == 1, "adjacent (dZ=1)", "non-adjacent (dZ>=2)")
    best["rep_sd_bin"] = pd.cut(best["replicate_sd_median"],
                                [-0.001, 0.05, 0.15, 0.30, 0.60, 100],
                                labels=["<=0.05", "0.05-0.15", "0.15-0.30", "0.30-0.60", ">0.60"])
    best["nm_bin"] = pd.cut(best["n_metals"], [1, 3, 5, 8, 11, 14],
                            labels=["2-3", "4-5", "6-8", "9-11", "12-14"])
    best["tani_bin"] = pd.cut(best["max_train_tanimoto"], [-0.01, 0.4, 0.55, 0.7, 0.85, 1.01],
                              labels=["<0.40", "0.40-0.55", "0.55-0.70", "0.70-0.85", ">0.85"])
    best["acid_bin"] = pd.cut(best["cond__acid_concentration_M"], [-0.01, 0.1, 1.0, 3.0, 6.0, 100],
                              labels=["<=0.1M", "0.1-1M", "1-3M", "3-6M", ">6M"])

    cuts = {
        "dZ": "dZ",
        "n_metals": "n_metals",
        "n_metals_bin": "nm_bin",
        "chemotype": "chemotype",
        "replicate_sd_bin": "rep_sd_bin",
        "group_pair": "group_pair",
        "crosses_Gd": "crosses_Gd",
        "adjacency": "adjacent",
        "max_train_tanimoto_bin": "tani_bin",
        "similarity_band": "band",
        "acid_concentration_bin": "acid_bin",
        "chem_family": "chem_family",
    }
    for name, col in cuts.items():
        t = cut_table(best, col)
        t.insert(0, "cut", name)
        t = t.rename(columns={col: "level"})
        t.to_csv(os.path.join(OUT, f"d5_cut_{name}.csv"), index=False)
        print(f"\n--- cut: {name} ---")
        show = ["level", "n_pairs_per_seed", "n_extractants", "mae_pooled", "mae_macro",
                "mean_abs_y", "ratio_pooled", "mae_pooled_meancurve", "frac_beats_meancurve"]
        print(t[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # per-metal (metal appears as either member of the pair)
    rows = []
    for m in LANTHANIDES:
        sub = best[(best["A"] == m) | (best["B"] == m)]
        if len(sub) == 0:
            continue
        rows.append(dict(metal=m, Z=ATOMIC_NUMBER[m], n_pairs_per_seed=len(sub) / n_seeds,
                         n_extractants=sub["extractant"].nunique(),
                         mae_pooled=sub["ae"].mean(), mae_macro=macro_scalar(sub, "ae"),
                         mean_abs_y=sub["absy"].mean(),
                         mae_pooled_meancurve=sub["ae_mc"].mean(),
                         mae_pooled_incumbent=sub["ae_inc"].mean(),
                         frac_beats_meancurve=sub["beats_mc"].mean()))
    per_metal = pd.DataFrame(rows)
    per_metal["ratio_pooled"] = per_metal["mae_pooled"] / per_metal["mean_abs_y"]
    per_metal.to_csv(os.path.join(OUT, "d5_cut_per_metal.csv"), index=False)
    print("\n--- cut: per metal involved ---")
    print(per_metal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # top pair types
    pt = cut_table(best, "pair_type").rename(columns={"pair_type": "level"})
    pt = pt.sort_values("mae_pooled", ascending=False)
    pt.to_csv(os.path.join(OUT, "d5_cut_pair_type.csv"), index=False)
    print("\n--- worst 10 / best 10 individual metal pairs (pooled, min 50 pairs/seed) ---")
    ptf = pt[pt["n_pairs_per_seed"] >= 50]
    print(pd.concat([ptf.head(10), ptf.tail(10)])[
        ["level", "n_pairs_per_seed", "mae_pooled", "mean_abs_y", "ratio_pooled",
         "frac_beats_meancurve"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- 4. can the error be predicted ----------------
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    feat_num = ["n_metals", "n_rows", "replicate_sd_median", "max_train_tanimoto",
                "chemotype_n_cells", "dZ", "abs_pred", "cond__acid_concentration_M",
                "cond__extractant_concentration_M", "n_pairs_in_cell"]
    d = best.copy()
    d["abs_pred"] = d["prediction"].abs()
    d["n_pairs_in_cell"] = d.groupby(["split_seed", "cell_id"])["ae"].transform("size")
    d["target"] = d["ae"]
    X = d[feat_num].copy()
    med = X.median()
    X = X.fillna(med)
    d[feat_num] = X

    oof_tree = np.full(len(d), np.nan)
    oof_lin = np.full(len(d), np.nan)
    idx = np.arange(len(d))
    for seed in sorted(d["split_seed"].unique()):
        for fold in sorted(d.loc[d.split_seed == seed, "fold"].unique()):
            te = (d.split_seed == seed) & (d.fold == fold)
            tr = (d.split_seed == seed) & (d.fold != fold)
            Xt, yt = d.loc[tr, feat_num].values, d.loc[tr, "target"].values
            Xe = d.loc[te, feat_num].values
            t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=0).fit(Xt, yt)
            oof_tree[idx[te.values]] = t.predict(Xe)
            sc = StandardScaler().fit(Xt)
            r = Ridge(alpha=10.0).fit(sc.transform(Xt), yt)
            oof_lin[idx[te.values]] = r.predict(sc.transform(Xe))
    d["pred_ae_tree"] = oof_tree
    d["pred_ae_lin"] = oof_lin

    print("\n=== 4. predicting |error| without the label (out-of-fold within each seed) ===")
    res = []
    for nm, col in [("shallow_tree_d3", "pred_ae_tree"), ("ridge_linear", "pred_ae_lin"),
                    ("abs_prediction_alone", "abs_pred"), ("replicate_sd_median_alone", "replicate_sd_median"),
                    ("dZ_alone", "dZ")]:
        rho = spearmanr(d[col], d["target"]).statistic
        pear = np.corrcoef(d[col], d["target"])[0, 1]
        row = dict(model=nm, spearman=rho, pearson=pear, mae_of_ae_model=np.abs(d[col] - d["target"]).mean())
        # abstention: drop the worst decile by predicted error
        for frac in (0.05, 0.10, 0.20, 0.30):
            thr = d[col].quantile(1 - frac)
            keep = d[col] < thr
            row[f"pooled_mae_keep{int((1-frac)*100)}"] = d.loc[keep, "ae"].mean()
            row[f"macro_mae_keep{int((1-frac)*100)}"] = macro_scalar(d.loc[keep], "ae")
            row[f"frac_kept_{int((1-frac)*100)}"] = keep.mean()
        res.append(row)
    abst = pd.DataFrame(res)
    abst.insert(1, "pooled_mae_all", d["ae"].mean())
    abst.insert(2, "macro_mae_all", macro_best)
    abst.to_csv(os.path.join(OUT, "d5_error_predictability.csv"), index=False)
    print(abst.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # oracle ceiling: abstain on the true worst decile
    thr_or = d["ae"].quantile(0.9)
    print(f"ORACLE (abstain on true worst decile): pooled MAE {d.loc[d.ae < thr_or, 'ae'].mean():.4f}, "
          f"macro {macro_scalar(d.loc[d.ae < thr_or], 'ae'):.4f}")
    # decile table for the tree
    d["dec_tree"] = pd.qcut(d["pred_ae_tree"], 10, labels=False, duplicates="drop")
    dec = d.groupby("dec_tree").agg(n=("ae", "size"), pred_ae=("pred_ae_tree", "mean"),
                                    actual_mae=("ae", "mean"), mean_abs_y=("absy", "mean"),
                                    mean_dZ=("dZ", "mean"), mean_rep_sd=("replicate_sd_median", "mean"),
                                    mean_abs_pred=("abs_pred", "mean")).reset_index()
    dec.to_csv(os.path.join(OUT, "d5_error_prediction_deciles.csv"), index=False)
    print("\n--- deciles of predicted |error| (shallow tree) ---")
    print(dec.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # abstain at the EXTRACTANT level instead: drop worst-10 extractants by predicted error
    ex_pred = d.groupby("extractant")["pred_ae_tree"].mean().sort_values(ascending=False)
    drop10 = set(ex_pred.head(10).index)
    kept = d[~d["extractant"].isin(drop10)]
    print(f"abstain on 10 extractants with the highest predicted error: "
          f"macro MAE {macro_scalar(kept, 'ae'):.4f} over {kept.extractant.nunique()} extractants "
          f"(oracle worst-10 removal gives {macro_wo10:.4f})")
    hit = len(drop10 & set(per_ex.head(10)["extractant"])) if "extractant" in per_ex else 0
    print(f"overlap of predicted-worst-10 with actual-worst-10 extractants: {hit}/10")

    # ---------------- 3. normalised dZ view, explicit ----------------
    dz = cut_table(best, "dZ").rename(columns={"dZ": "level"})
    dz["mae_ratio_model_over_meancurve"] = dz["mae_pooled"] / dz["mae_pooled_meancurve"]
    dz.to_csv(os.path.join(OUT, "d5_cut_dZ.csv"), index=False)
    print("\n=== 3. normalised dZ view ===")
    print(dz[["level", "n_pairs_per_seed", "mae_pooled", "mean_abs_y", "ratio_pooled",
              "mae_pooled_meancurve", "mae_ratio_model_over_meancurve",
              "frac_beats_meancurve"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    adj = best[best.dZ == 1]
    print(f"adjacent pairs (dZ=1): n={len(adj)//n_seeds}/seed  macro MAE model={macro_scalar(adj,'ae'):.4f} "
          f"meancurve={macro_scalar(adj,'ae_mc'):.4f} incumbent={macro_scalar(adj,'ae_inc'):.4f} "
          f"mean|y|={adj.absy.mean():.4f} frac beat mean curve={adj.beats_mc.mean():.4f}")
    # paired bootstrap over extractants on adjacent pairs
    rng = np.random.default_rng(0)
    ex_list = adj["extractant"].unique()
    per_ex_adj = (adj.groupby(["split_seed", "extractant"])[["ae", "ae_mc"]].mean()
                  .groupby("extractant").mean())
    diffs = []
    for _ in range(2000):
        s = rng.choice(len(ex_list), len(ex_list), replace=True)
        sub = per_ex_adj.iloc[s]
        diffs.append(sub["ae_mc"].mean() - sub["ae"].mean())
    diffs = np.array(diffs)
    print(f"adjacent-pair macro gain over mean curve = {per_ex_adj['ae_mc'].mean()-per_ex_adj['ae'].mean():.4f} "
          f"[95% CI {np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}] "
          f"(bootstrap over {len(ex_list)} extractants)")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    pe = per_ex.sort_values("mae_best").reset_index(drop=True)
    ax[0].bar(range(len(pe)), pe["mae_best"], color="#4477aa", width=0.9)
    ax[0].plot(range(len(pe)), pe["noise_floor_pairwise"], "r.", ms=3, label="sqrt(2)*replicate sd")
    ax[0].axhline(macro_best, color="k", ls="--", lw=1, label=f"macro MAE {macro_best:.3f}")
    ax[0].set_xlabel("extractant (sorted)"); ax[0].set_ylabel("MAE (log SF units)")
    ax[0].set_title("per-extractant MAE, best arm"); ax[0].legend(fontsize=7)
    cum = pe["mae_best"].sort_values(ascending=False).cumsum() / pe["mae_best"].sum()
    ax[1].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100, "-o", ms=3)
    ax[1].plot([0, 100], [0, 100], "k:", lw=1)
    ax[1].set_xlabel("% of extractants (worst first)"); ax[1].set_ylabel("% of summed per-extractant MAE")
    ax[1].set_title("concentration of error")
    ax[2].plot(dz["level"], dz["mae_pooled"], "-o", label="best arm")
    ax[2].plot(dz["level"], dz["mae_pooled_meancurve"], "-s", label="corpus mean curve")
    ax[2].plot(dz["level"], dz["mean_abs_y"], "-^", label="mean |y|")
    ax[2].set_xlabel("dZ"); ax[2].set_ylabel("log SF units"); ax[2].legend(fontsize=8)
    ax[2].set_title("error vs signal by dZ")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d5_error_anatomy.png"), dpi=140)
    plt.close(fig)

    summary = dict(macro_mae_best=macro_best, macro_mae_incumbent=macro_inc,
                   macro_mae_meancurve=macro_mc, pooled_mae_best=float(best.ae.mean()),
                   mean_abs_y=float(best.absy.mean()),
                   n_extractants=int(per_ex.shape[0]), n_cells=int(best.cell_id.nunique()),
                   n_pairs_per_seed=int(len(best) / n_seeds),
                   worst10_share_of_summed_mae=float(per_ex["mae_best"].head(10).sum() / tot),
                   macro_mae_without_worst10=float(macro_wo10),
                   n_extractants_at_or_below_noise_floor=int(nf.at_or_below_floor.sum()),
                   n_extractants_with_floor=int(len(nf)))
    with open(os.path.join(OUT, "d5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
