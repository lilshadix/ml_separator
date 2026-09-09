"""D5 part 2 - is the residual error noise or signal, and is abstention worth anything.

Run from repo root:
    .venv/Scripts/python.exe gen13_separation/analysis/stage2/D5/d5_signal_vs_noise.py
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gen13_separation"))
from gen13sep.metals import LANTHANIDES  # noqa: E402

OUT = os.path.join(ROOT, "gen13_separation", "analysis", "stage2", "D5")
FIG = os.path.join(ROOT, "gen13_separation", "figures", "stage2")
PRED = os.path.join(ROOT, "gen13_separation", "predictions", "B_primary")
BEST = "X_ENS_DIRECT+LOWRANK_K2"


def macro_scalar(df, col="ae"):
    return float(df.groupby(["split_seed", "extractant"], observed=True)[col].mean()
                 .groupby("split_seed").mean().mean())


def main():
    d = pd.read_parquet(os.path.join(PRED, f"{BEST}.parquet"))
    d["err"] = d["y"] - d["prediction"]
    d["ae"] = d["err"].abs()
    d["absy"] = d["y"].abs()
    mc = pd.read_parquet(os.path.join(PRED, "B1_MEAN_CURVE.parquet"))
    mc["ae_mc"] = (mc["y"] - mc["prediction"]).abs()
    k = ["split_seed", "cell_id", "A", "B"]
    d = d.merge(mc[k + ["ae_mc"]], on=k, how="left")
    coh = pd.read_parquet(os.path.join(ROOT, "gen13_separation", "manifests", "cohort_exact.parquet"))
    d = d.merge(coh[["cell_id", "n_rows", "replicate_sd_median", "publication_id",
                     "cond__acid_concentration_M"]], on="cell_id", how="left")
    sim = pd.read_parquet(os.path.join(PRED, "similarity.parquet"))
    d = d.merge(sim[["split_seed", "cell_id", "max_train_tanimoto"]], on=["split_seed", "cell_id"], how="left")
    n_seeds = d["split_seed"].nunique()
    macro_all = macro_scalar(d)
    print(f"macro MAE all = {macro_all:.4f}; pooled = {d.ae.mean():.4f}; mean|y| = {d.absy.mean():.4f}")

    lines = []

    # ---- A. concentration of error: Gini + share ----
    pex = (d.groupby(["split_seed", "extractant"])["ae"].mean().groupby("extractant").mean()
           .sort_values(ascending=False))
    x = np.sort(pex.values)
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
    print(f"\n[A] Gini of per-extractant MAE over {n} extractants = {gini:.3f}")
    for k_ in (5, 10, 20, 45):
        print(f"    worst {k_:2d} carry {pex.head(k_).sum()/pex.sum()*100:5.1f}% of the summed per-extractant MAE; "
              f"macro without them = {pex.iloc[k_:].mean():.4f}")
    lines.append(dict(stat="gini_per_extractant_mae", value=gini, n=n))

    # ---- B. is a big MAE just a big signal? ----
    pes = (d.groupby(["split_seed", "extractant"])[["ae", "absy", "ae_mc"]].mean()
           .groupby("extractant").mean())
    lg = np.log10(pes["ae"].clip(lower=1e-3))
    lz = np.log10(pes["absy"].clip(lower=1e-3))
    A = np.vstack([lz, np.ones_like(lz)]).T
    coef, *_ = np.linalg.lstsq(A, lg, rcond=None)
    pred = A @ coef
    r2 = 1 - ((lg - pred) ** 2).sum() / ((lg - lg.mean()) ** 2).sum()
    rho = spearmanr(pes["ae"], pes["absy"]).statistic
    print(f"\n[B] per-extractant log10 MAE ~ log10 mean|y|: slope={coef[0]:.3f} intercept={coef[1]:.3f} "
          f"R2={r2:.3f}; Spearman(MAE, mean|y|)={rho:.3f} over {n} extractants")
    pes["log_resid"] = lg - pred
    pes["skill_vs_meancurve"] = pes["ae_mc"] - pes["ae"]
    pes = pes.sort_values("log_resid", ascending=False)
    stat = (d.drop_duplicates(["extractant", "cell_id"]).groupby("extractant")
            .agg(chemotype=("chemotype", "first"), n_cells=("cell_id", "nunique"),
                 max_n_metals=("n_metals", "max")))
    pes = pes.join(stat)
    pes.to_csv(os.path.join(OUT, "d5_extractant_size_adjusted_difficulty.csv"))
    print("    5 hardest after size adjustment (MAE far above what mean|y| predicts):")
    print(pes.head(5)[["ae", "absy", "log_resid", "n_cells", "chemotype"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print("    5 easiest after size adjustment:")
    print(pes.tail(5)[["ae", "absy", "log_resid", "n_cells", "chemotype"]].to_string(float_format=lambda v: f"{v:.3f}"))
    lines.append(dict(stat="r2_logMAE_on_logmeanabsy_per_extractant", value=float(r2), n=n))

    # ---- C. does the error repeat across split seeds (systematic vs sampling) ----
    pe_seed = d.groupby(["split_seed", "extractant"])["ae"].mean().unstack(0)
    within = pe_seed.var(axis=1, ddof=1).mean()
    between = pe_seed.mean(axis=1).var(ddof=1)
    icc = between / (between + within)
    print(f"\n[C] per-extractant MAE across the 5 split seeds: between-extractant var={between:.4f}, "
          f"mean within-extractant var={within:.4f}, ICC={icc:.3f} "
          f"(1.0 = the same extractants are hard in every seed)")
    lines.append(dict(stat="icc_extractant_mae_across_seeds", value=float(icc), n=n))

    # ---- D. is the residual reproducible across independent cells of the same extractant? ----
    # signed residual, averaged over seeds, per (extractant, cell, pair); split cells of an
    # extractant into two halves and correlate the mean residual per (pair) between halves.
    r = d.groupby(["extractant", "cell_id", "A", "B"])["err"].mean().reset_index()
    rows = []
    rng = np.random.default_rng(0)
    for ex, g in r.groupby("extractant"):
        cells = g["cell_id"].unique()
        if len(cells) < 2:
            continue
        for rep in range(20):
            perm = rng.permutation(cells)
            h1, h2 = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
            a = g[g.cell_id.isin(h1)].groupby(["A", "B"])["err"].mean()
            b = g[g.cell_id.isin(h2)].groupby(["A", "B"])["err"].mean()
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if len(j) >= 6:
                rows.append(dict(extractant=ex, rep=rep, n_pairs=len(j),
                                 pearson=np.corrcoef(j["a"], j["b"])[0, 1]))
    rep_df = pd.DataFrame(rows)
    if len(rep_df):
        per_ex_rep = rep_df.groupby("extractant")["pearson"].mean()
        print(f"\n[D] split-half reproducibility of the SIGNED residual across independent cells of the "
              f"same extractant: mean r={per_ex_rep.mean():.3f} (median {per_ex_rep.median():.3f}) "
              f"over {len(per_ex_rep)} extractants with >=2 cells and >=6 shared pairs, 20 random splits each")
        print(f"    fraction of those extractants with r>0.5: {(per_ex_rep>0.5).mean():.2f}; r>0: {(per_ex_rep>0).mean():.2f}")
        per_ex_rep.to_csv(os.path.join(OUT, "d5_residual_split_half_reproducibility.csv"))
        lines.append(dict(stat="split_half_signed_residual_pearson_mean", value=float(per_ex_rep.mean()),
                          n=int(len(per_ex_rep))))
    else:
        print("\n[D] not enough multi-cell extractants for a split-half residual test")

    # ---- E. pair-level replicate floor from repsd__ columns ----
    rs = coh[["cell_id"] + [f"repsd__{m}" for m in LANTHANIDES]].set_index("cell_id")
    sdA = d.set_index("cell_id").apply(lambda _: np.nan, axis=1) if False else None
    lookup = rs.stack().dropna()
    lookup.index = lookup.index.set_names(["cell_id", "col"])
    lk = lookup.reset_index()
    lk["metal"] = lk["col"].str.replace("repsd__", "", regex=False)
    lk = lk.rename(columns={0: "sd"})
    lk.columns = ["cell_id", "col", "sd", "metal"]
    mp = lk.set_index(["cell_id", "metal"])["sd"]
    dA = pd.MultiIndex.from_arrays([d["cell_id"], d["A"]])
    dB = pd.MultiIndex.from_arrays([d["cell_id"], d["B"]])
    d["sdA"] = mp.reindex(dA).values
    d["sdB"] = mp.reindex(dB).values
    both = d.dropna(subset=["sdA", "sdB"]).copy()
    both["pair_floor"] = np.sqrt(both["sdA"] ** 2 + both["sdB"] ** 2)
    print(f"\n[E] pairs where BOTH metals have a measured replicate sd: {len(both)//n_seeds} per seed "
          f"({len(both)/len(d)*100:.1f}% of all pairs), from {both.cell_id.nunique()} cells / "
          f"{both.extractant.nunique()} extractants")
    if len(both):
        print(f"    median pair floor sqrt(sdA^2+sdB^2) = {both.pair_floor.median():.3f}; "
              f"model MAE on these pairs = {both.ae.mean():.3f}; "
              f"fraction of pairs with |error| <= its own floor = {(both.ae<=both.pair_floor).mean():.3f}")
        gg = both.groupby("extractant").agg(mae=("ae", "mean"), floor=("pair_floor", "median"),
                                            n=("ae", "size"))
        gg["at_floor"] = gg["mae"] <= gg["floor"]
        print(f"    extractants at or below their own pair floor: {int(gg.at_floor.sum())}/{len(gg)}")
        gg.to_csv(os.path.join(OUT, "d5_pair_level_noise_floor.csv"))
        lines.append(dict(stat="frac_pairs_within_own_replicate_floor", value=float((both.ae<=both.pair_floor).mean()),
                          n=int(len(both))))

    # ---- F. abstention with honest controls ----
    from sklearn.tree import DecisionTreeRegressor
    feats = ["n_metals", "n_rows", "replicate_sd_median", "max_train_tanimoto", "dZ",
             "abs_pred", "cond__acid_concentration_M", "chemo_n"]
    d["abs_pred"] = d["prediction"].abs()
    d["chemo_n"] = d["chemotype"].map(coh.groupby("chemotype").size())
    F = d[feats].fillna(d[feats].median())
    d[feats] = F
    oof = np.full(len(d), np.nan)
    oof_nodz = np.full(len(d), np.nan)
    ix = np.arange(len(d))
    feats_nodz = [f for f in feats if f not in ("dZ", "abs_pred")]
    for s in sorted(d.split_seed.unique()):
        for f in sorted(d.loc[d.split_seed == s, "fold"].unique()):
            te = ((d.split_seed == s) & (d.fold == f)).values
            tr = ((d.split_seed == s) & (d.fold != f)).values
            m = DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=0)
            m.fit(d.loc[tr, feats].values, d.loc[tr, "ae"].values)
            oof[ix[te]] = m.predict(d.loc[te, feats].values)
            m2 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=0)
            m2.fit(d.loc[tr, feats_nodz].values, d.loc[tr, "ae"].values)
            oof_nodz[ix[te]] = m2.predict(d.loc[te, feats_nodz].values)
    d["pred_ae"] = oof
    d["pred_ae_nodz"] = oof_nodz
    print(f"\n[F] Spearman(predicted |err|, actual |err|): full tree {spearmanr(d.pred_ae, d.ae).statistic:.3f}; "
          f"tree without dZ/|pred| {spearmanr(d.pred_ae_nodz, d.ae).statistic:.3f}; "
          f"dZ alone {spearmanr(d.dZ, d.ae).statistic:.3f}; |pred| alone {spearmanr(d.abs_pred, d.ae).statistic:.3f}")
    # within-dZ Spearman (does anything beyond dZ predict the error?)
    rows = []
    for dz, g in d.groupby("dZ"):
        if len(g) < 200:
            continue
        rows.append(dict(dZ=dz, n=len(g),
                         rho_tree=spearmanr(g.pred_ae, g.ae).statistic,
                         rho_abspred=spearmanr(g.abs_pred, g.ae).statistic,
                         rho_tree_nodz=spearmanr(g.pred_ae_nodz, g.ae).statistic))
    wdz = pd.DataFrame(rows)
    wdz.to_csv(os.path.join(OUT, "d5_within_dZ_error_predictability.csv"), index=False)
    print("    within-dZ Spearman (predicted vs actual |error|):")
    print(wdz.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"    n-weighted mean within-dZ rho: tree={np.average(wdz.rho_tree, weights=wdz.n):.3f}, "
          f"|pred|={np.average(wdz.rho_abspred, weights=wdz.n):.3f}, "
          f"tree-without-dZ={np.average(wdz.rho_tree_nodz, weights=wdz.n):.3f}")
    lines.append(dict(stat="within_dZ_weighted_spearman_tree", value=float(np.average(wdz.rho_tree, weights=wdz.n)),
                      n=int(wdz.n.sum())))

    # abstention table with normalised column and random control
    rows = []
    rng = np.random.default_rng(1)
    for name, col in [("tree", "pred_ae"), ("tree_no_dZ", "pred_ae_nodz"),
                      ("dZ_only", "dZ"), ("abs_pred_only", "abs_pred"), ("oracle_true_ae", "ae")]:
        for frac in (0.10, 0.20):
            thr = d[col].quantile(1 - frac)
            keep = d[col] < thr
            kd = d[keep]
            rows.append(dict(rule=name, abstain_frac=frac, kept_frac=float(keep.mean()),
                             pooled_mae=kd.ae.mean(), macro_mae=macro_scalar(kd),
                             mean_abs_y_kept=kd.absy.mean(),
                             normalised=kd.ae.mean() / kd.absy.mean(),
                             mae_meancurve_kept=kd.ae_mc.mean()))
    for frac in (0.10, 0.20):
        vals = []
        for _ in range(50):
            keep = rng.random(len(d)) > frac
            vals.append(d.loc[keep, "ae"].mean())
        rows.append(dict(rule="random", abstain_frac=frac, kept_frac=1 - frac,
                         pooled_mae=float(np.mean(vals)), macro_mae=np.nan,
                         mean_abs_y_kept=d.absy.mean(), normalised=float(np.mean(vals)) / d.absy.mean(),
                         mae_meancurve_kept=d.ae_mc.mean()))
    rows.append(dict(rule="none(all pairs)", abstain_frac=0.0, kept_frac=1.0, pooled_mae=d.ae.mean(),
                     macro_mae=macro_all, mean_abs_y_kept=d.absy.mean(),
                     normalised=d.ae.mean() / d.absy.mean(), mae_meancurve_kept=d.ae_mc.mean()))
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(OUT, "d5_abstention_with_controls.csv"), index=False)
    print("\n    abstention with controls (normalised = pooled MAE / mean|y| among kept pairs):")
    print(ab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    with open(os.path.join(OUT, "d5_signal_vs_noise_summary.json"), "w") as f:
        json.dump(lines, f, indent=2)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].loglog(pes["absy"], pes["ae"], "o", ms=5, alpha=0.75)
    xs = np.logspace(np.log10(max(pes["absy"].min(), 1e-2)), np.log10(pes["absy"].max()), 50)
    ax[0].loglog(xs, 10 ** coef[1] * xs ** coef[0], "r-", lw=1.5,
                 label=f"slope {coef[0]:.2f}, R2={r2:.2f}")
    ax[0].loglog(xs, xs, "k:", lw=1, label="MAE = mean|y|")
    ax[0].set_xlabel("mean |log SF| of the extractant"); ax[0].set_ylabel("MAE")
    ax[0].set_title("hard vs large (90 extractants)"); ax[0].legend(fontsize=8)
    sub = ab[ab.rule.isin(["tree", "dZ_only", "oracle_true_ae", "random"])]
    for rule, g in sub.groupby("rule"):
        ax[1].plot(g["abstain_frac"], g["normalised"], "-o", label=rule)
    ax[1].axhline(d.ae.mean() / d.absy.mean(), color="k", ls=":", lw=1, label="no abstention")
    ax[1].set_xlabel("fraction of pairs abstained on"); ax[1].set_ylabel("pooled MAE / mean|y| of kept pairs")
    ax[1].set_title("abstention buys nothing once normalised"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d5_signal_vs_noise.png"), dpi=140)
    plt.close(fig)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
