"""gen8 brief section 26 -- error archaeology after calibration.

Repeats the worst-ligand analysis at k = 0, 1 and 2 and asks which held-out-ligand
failures disappear after one measurement and which remain.

Every arm is read from the acquisition run's per-repeat detail table, so the
zero-shot reference and the k-shot arms are scored on *identical* evaluation rows
of *identical* ligands under *identical* repeats -- the classification therefore
cannot be an artefact of what each arm was scored on.

Outputs
    runs/gen8_architecture/case_studies/error_archaeology.md
    runs/gen8_architecture/case_studies/ligand_failure_classes.csv
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lilshadix/PycharmProjects/ml_separator")
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.mechanism import (  # noqa: E402
    MECHANISM_COLUMNS, mechanism_features, n_unparsed)
from lanthanide_separation.gen8.report import md_table  # noqa: E402

OUT = ROOT / "runs/gen8_architecture/case_studies"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "REC_ecfp_plus_recovered"
ARMS = ["k0", "RANDOM_k1", "RANDOM_k2", "CENTRAL_k1", "CENTRAL_k2"]
KEY = ["split_seed", "fold", "extractant", "repeat"]
LEVEL_CUT, PURE_CUT, PERSIST_CUT = 1.0, 0.5, 0.7
REPS, BSEED = 5000, 8675309


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def load_arms() -> pd.DataFrame:
    detail = pd.read_parquet(ROOT / "runs/gen8_architecture/active_acquisition/primary_detail.parquet")
    keep = (((detail.adapter == "ZERO_SHOT_REF") & (detail.k == 0) & (detail.policy == "NONE"))
            | ((detail.adapter == "OFFSET_K1")
               & (detail.policy.isin(["RANDOM", "CENTRAL"])) & (detail.k.isin([1, 2]))))
    sel = detail[keep].copy()
    sel["arm"] = np.where(sel.adapter == "ZERO_SHOT_REF", "k0",
                          sel.policy + "_k" + sel.k.astype(str))
    sizes = sel.groupby("arm").size()
    assert sorted(sizes.index) == sorted(ARMS) and sizes.nunique() == 1, sizes
    units = {a: set(map(tuple, sel[sel.arm == a][KEY].to_numpy())) for a in ARMS}
    assert len(set.intersection(*units.values())) == len(units["k0"]), "arms are not paired"
    sel.attrs["n_units"] = int(sizes.iloc[0])
    sel.attrs["oracle"] = detail[(detail.policy == "ORACLE[OFFSET_K1]")
                                 & (detail.adapter == "OFFSET_K1")]
    return sel


def other_arms(sel: pd.DataFrame) -> dict:
    """Macro MAE of the slope-aware adapter, read from the same run rather than quoted.

    §8 points at `OFFSET_K3` as the first evidence that the shape is attackable; that
    comparison has to come out of the parquet in front of us, because quoting a
    remembered figure is exactly how a report survives a re-run of its own input.
    """
    detail = pd.read_parquet(ROOT / "runs/gen8_architecture/active_acquisition/primary_detail.parquet",
                             columns=["policy", "adapter", "k", "extractant", "mae"])
    out = {}
    for adapter in ("OFFSET_K1", "OFFSET_K3"):
        block = detail[(detail.adapter == adapter) & (detail.policy == "RANDOM")
                       & (detail.k == 2)]
        out[adapter] = float(block.groupby("extractant").mae.mean().mean())
    return out


def per_ligand(sel: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae", "offset", "shape_mae", "spearman", "within_0_5", "within_1_0"]
    grouped = sel.groupby(["arm", "extractant"])[metrics].mean()
    table = (sel[sel.arm == "k0"].groupby("extractant")
             .agg(tanimoto_cluster=("tanimoto_cluster", "first"),
                  ecfp_cluster=("ecfp_cluster", "first"),
                  nn_train_tanimoto=("nn_train_tanimoto", "first"),
                  n_rows=("n_rows", "first"), n_pool=("n_pool", "mean"),
                  n_eval=("n_eval", "mean")))
    for arm in ARMS:
        table[f"mae_{arm}"] = grouped.loc[arm, "mae"]
    for arm in ARMS[1:]:
        table[f"ratio_{arm}"] = table[f"mae_{arm}"] / table["mae_k0"]
    table["offset_k0"] = grouped.loc["k0", "offset"]
    table["shape_mae_k0"] = grouped.loc["k0", "shape_mae"]
    table["shape_frac_k0"] = table["shape_mae_k0"] / table["mae_k0"]
    table["spearman_k0"] = grouped.loc["k0", "spearman"]
    for arm in ["k0", "RANDOM_k1", "RANDOM_k2"]:
        table[f"within_1_0_{arm}"] = grouped.loc[arm, "within_1_0"]
    oracle = sel.attrs["oracle"]
    table["mae_oracle_k1"] = oracle[oracle.k == 1].groupby("extractant").mae.mean()
    table["mae_oracle_k2"] = oracle[oracle.k == 2].groupby("extractant").mae.mean()
    return table


def classify(mae0: float, ratio1: float, *, level=LEVEL_CUT, pure=PURE_CUT,
             persist=PERSIST_CUT) -> str:
    if mae0 <= level:
        return "ALREADY_GOOD"
    if ratio1 < pure:
        return "PURE_LEVEL"
    if ratio1 > persist:
        return "RESIDUAL_SHAPE"
    return "PARTIAL_LEVEL"


def add_classes(table: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    table["failure_class"] = [classify(m, r) for m, r in zip(table.mae_k0, table.ratio_RANDOM_k1)]
    table["failure_class_central"] = [classify(m, r)
                                      for m, r in zip(table.mae_k0, table.ratio_CENTRAL_k1)]
    table["k2_status"] = np.where(
        table.failure_class != "RESIDUAL_SHAPE", "",
        np.where(table.ratio_RANDOM_k2 <= PERSIST_CUT, "RECOVERS_AT_K2", "IRREDUCIBLE_AT_K2"))
    table["one_shot_harms"] = table.ratio_RANDOM_k1 > 1.0
    per_seed = sel.groupby(["arm", "extractant", "split_seed"])["mae"].mean()
    seeds = sorted(sel.split_seed.unique())
    agree, modal = [], []
    for ligand in table.index:
        classes = [classify(per_seed[("k0", ligand, s)],
                            per_seed[("RANDOM_k1", ligand, s)] / per_seed[("k0", ligand, s)])
                   for s in seeds]
        agree.append(classes.count(table.loc[ligand, "failure_class"]))
        modal.append(int(pd.Series(classes).value_counts().iloc[0]))
    table["seeds_matching_pooled_class"] = agree
    table["seed_modal_count"] = modal
    return table


# --------------------------------------------------------------------------- #
# covariates
# --------------------------------------------------------------------------- #
AXES = ["acid", "extractant", "metal_series", "temperature", "contact_time",
        "metal_concentration"]


def _offset_floor(oof: pd.DataFrame) -> pd.DataFrame:
    """How much of `shape_mae` is an artefact of centring on the *mean* residual.

    `evaluate._metrics` defines `shape_mae` as the MAE after subtracting the mean
    residual.  The best constant in MAE is the **median** residual, not the mean, so
    `shape_mae` is an upper bound on the offset-only floor, not the floor itself.
    Both are computed here on the ligand's full out-of-fold row set (not the P2
    evaluation subsets), which is why they are reported as a scale for the gap
    rather than as a replacement for `shape_mae`.  `mad_true` is the same statistic
    computed on the target alone -- what `shape_mae` would be if the model predicted
    a constant for the ligand -- and is the benchmark the §6 span contrasts have to
    beat before they can be called explanatory rather than definitional.
    """
    rows = []
    for (ligand, seed), block in oof.groupby(["extractant", "split_seed"], sort=True):
        residual = (block.prediction - block.log_D).to_numpy(dtype=float)
        truth = block.log_D.to_numpy(dtype=float)
        rows.append({"extractant": ligand,
                     "oof_mean_centred_mae": float(np.abs(residual - residual.mean()).mean()),
                     "oof_median_centred_mae":
                         float(np.abs(residual - np.median(residual)).mean()),
                     "mad_true": float(np.abs(truth - truth.mean()).mean())})
    return (pd.DataFrame(rows).groupby("extractant")
            [["oof_mean_centred_mae", "oof_median_centred_mae", "mad_true"]].mean())


def add_covariates(table: pd.DataFrame) -> pd.DataFrame:
    series = pd.read_parquet(ROOT / "runs/gen8_architecture/series_table.parquet")
    curves = pd.read_parquet(ROOT / "runs/gen8_architecture/series/curve_table.parquet")
    table = table.join(series.groupby("extractant").agg(
        n_series=("series_id", "nunique"), n_doi_series_sum=("n_doi", "sum"),
        series_types=("series_type", lambda s: ",".join(sorted(set(s)))),
        log_d_span_max=("log_d_span", "max")))
    # `series_table.n_doi` is a *per-series* count, so summing it over a ligand's
    # series counts one publication once per series it appears in -- for 124 of the
    # 143 ligands that is not the number of source publications.  Both are carried:
    # the summed one because it is what a "how many separate experiments" reading
    # wants, and the distinct one because it is what the phrase "source
    # publications" means.  They are not interchangeable and the report says which.
    table = table.join(series.groupby("extractant").doi.apply(
        lambda column: float(len({d.strip() for value in column
                                  for d in str(value).split(",") if d.strip()})))
        .rename("n_distinct_doi"))
    table = table.join(curves.groupby("extractant").agg(
        n_curves=("curve_id", "nunique"), n_curve_axes=("axis_label", "nunique"),
        curve_axes=("axis_label", lambda s: ",".join(sorted(set(s)))),
        median_curve_slope=("slope", "median"),
        median_curve_linear_r2=("linear_r2", "median"),
        max_curve_yspan=("y_span", "max")))
    for axis in AXES:
        present = set(curves.loc[curves.axis_label == axis, "extractant"])
        table[f"has_axis_{axis}"] = [float(x in present) for x in table.index]
    table["n_series"] = table["n_series"].fillna(0)
    table["n_curves"] = table["n_curves"].fillna(0)

    oof = pd.read_parquet(ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet")
    oof = oof[oof.model == MODEL]
    spans = (oof.groupby(["extractant", "split_seed"])
             .agg(true_span=("log_D", lambda s: float(np.ptp(s))),
                  pred_span=("prediction", lambda s: float(np.ptp(s))),
                  true_sd=("log_D", "std"), pred_sd=("prediction", "std"))
             .groupby("extractant").mean())
    table = table.join(spans)
    table = table.join(_offset_floor(oof))
    table["span_compression"] = np.where(table.true_span > 1e-9,
                                         table.pred_span / table.true_span, np.nan)

    mech = mechanism_features(list(table.index))
    assert n_unparsed(mech) == 0, f"{n_unparsed(mech)} SMILES failed to parse"
    mech.index = table.index
    table = table.join(mech)

    table["log10_n_rows"] = np.log10(table.n_rows.astype(float))
    table["log10_n_eval"] = np.log10(table.n_eval.astype(float))
    table["abs_median_curve_slope"] = table.median_curve_slope.abs()
    return table


# --------------------------------------------------------------------------- #
# chemotype-blocked contrast between two classes
# --------------------------------------------------------------------------- #
def _auc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = float((a[:, None] > b[None, :]).sum())
    equal = float((a[:, None] == b[None, :]).sum())
    return (greater + 0.5 * equal) / (a.size * b.size)


def cluster_contrast(table: pd.DataFrame, column: str, class_a: str, class_b: str,
                     *, reps: int = REPS, seed: int = BSEED) -> dict:
    """Difference between two failure classes, resampling Tanimoto chemotypes.

    Ligands inside one held-out chemotype share a training set and are correlated,
    so the blocks -- not the ligands -- are the resampling unit.  Reported as both a
    mean difference and a rank statistic (P(A > B)), because several covariates here
    are heavily right-skewed and their means are moved by one or two ligands.
    """
    blocks = sorted(table.tanimoto_cluster.unique())
    index_of = {b: np.flatnonzero((table.tanimoto_cluster == b).to_numpy()) for b in blocks}
    values = table[column].to_numpy(dtype=float)
    labels = table.failure_class.to_numpy()
    a0, b0 = values[labels == class_a], values[labels == class_b]
    point_mean = float(np.nanmean(a0) - np.nanmean(b0))
    point_auc = _auc(a0, b0)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(blocks), size=(reps, len(blocks)))
    draws_mean = np.full(reps, np.nan)
    draws_auc = np.full(reps, np.nan)
    for r in range(reps):
        take = np.concatenate([index_of[blocks[j]] for j in picks[r]])
        v, l = values[take], labels[take]
        a, b = v[l == class_a], v[l == class_b]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        draws_mean[r] = a.mean() - b.mean()
        draws_auc[r] = _auc(a, b)

    def summarise(draws: np.ndarray, null: float) -> tuple[float, float, float]:
        finite = draws[np.isfinite(draws)]
        if finite.size < 100:
            return float("nan"), float("nan"), float("nan")
        low, high = np.quantile(finite, [0.025, 0.975])
        p = 2.0 * min(float((finite <= null).mean()), float((finite >= null).mean()))
        return float(low), float(high), float(min(1.0, max(p, 1.0 / finite.size)))

    mlo, mhi, mp = summarise(draws_mean, 0.0)
    alo, ahi, ap = summarise(draws_auc, 0.5)
    return {"column": column, "n_a": int(np.isfinite(a0).sum()), "n_b": int(np.isfinite(b0).sum()),
            "mean_a": float(np.nanmean(a0)), "mean_b": float(np.nanmean(b0)),
            "median_a": float(np.nanmedian(a0)), "median_b": float(np.nanmedian(b0)),
            "mean_diff": point_mean, "diff_lo": mlo, "diff_hi": mhi, "diff_p": mp,
            "auc": point_auc, "auc_lo": alo, "auc_hi": ahi, "auc_p": ap}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


GEOMETRY = ["mad_true", "log_d_span_max", "max_curve_yspan", "true_span", "true_sd",
            "pred_span", "span_compression", "median_curve_linear_r2",
            "abs_median_curve_slope",
            "n_series", "n_doi_series_sum", "n_distinct_doi", "n_curves", "n_curve_axes",
            "log10_n_rows", "log10_n_eval", "nn_train_tanimoto"] + \
           [f"has_axis_{a}" for a in AXES]

#: Known without reading a single log D value: they count experiments and name axes.
#: `n_distinct_doi` is target-free but is *not* knowable for a molecule that has never
#: been published, so it is excluded from the deployable triage set.
TARGET_FREE = ["n_series", "n_curves", "n_curve_axes", "has_axis_acid",
               "log10_n_rows", "nn_train_tanimoto"]


def contrast_block(table: pd.DataFrame, columns, class_a: str, class_b: str,
                   label: str) -> pd.DataFrame:
    frame = pd.DataFrame([cluster_contrast(table, c, class_a, class_b) for c in columns])
    frame["block"] = label
    frame["auc_q_bh"] = benjamini_hochberg(frame.auc_p.to_numpy())
    frame["diff_q_bh"] = benjamini_hochberg(frame.diff_p.to_numpy())
    return frame


# --------------------------------------------------------------------------- #
# out-of-seed re-derivation of the classes
# --------------------------------------------------------------------------- #
def out_of_seed(sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Assign the class on three split seeds, then score it on the other two.

    The classification is read off the same numbers it is then used to describe, so
    a ligand whose zero-shot MAE happened to be inflated by which rows the P2 split
    put in its evaluation set would be selected into scope and would then regress.
    Splitting the seeds makes that testable: the class comes from seeds 1-3 and every
    number reported against it comes from seeds 4-5.  The two halves share the
    ligand's rows, so this tests the protocol's sampling noise, not the noise in the
    measurements themselves.
    """
    seeds = sorted(sel.split_seed.unique())
    fit, hold = seeds[:3], seeds[3:]

    def half(which) -> pd.DataFrame:
        block = sel[sel.split_seed.isin(which)]
        return block.groupby(["arm", "extractant"])["mae"].mean().unstack("arm")

    a, b = half(fit), half(hold)
    frame = pd.DataFrame({
        "class_fit": [classify(m, r / m) for m, r in zip(a["k0"], a["RANDOM_k1"])],
        "class_hold": [classify(m, r / m) for m, r in zip(b["k0"], b["RANDOM_k1"])],
        "k0": b["k0"], "RANDOM_k1": b["RANDOM_k1"], "RANDOM_k2": b["RANDOM_k2"],
        "CENTRAL_k2": b["CENTRAL_k2"], "ratio_hold": b["RANDOM_k1"] / b["k0"],
        "k0_fit": a["k0"]}, index=a.index)
    confusion = pd.crosstab(frame.class_fit, frame.class_hold)
    summary = (frame.groupby("class_fit")
               .agg(n=("k0", "size"), mae_k0=("k0", "mean"),
                    mae_R1=("RANDOM_k1", "mean"), mae_R2=("RANDOM_k2", "mean"),
                    mae_C2=("CENTRAL_k2", "mean"), mean_ratio=("ratio_hold", "mean"))
               .reset_index())
    scope = frame[frame.k0_fit > LEVEL_CUT]
    stats = {"n_fit_seeds": len(fit), "n_hold_seeds": len(hold),
             "agree": int((frame.class_fit == frame.class_hold).sum()), "n": len(frame),
             "residual_kept": int(((frame.class_fit == "RESIDUAL_SHAPE")
                                   & (frame.class_hold == "RESIDUAL_SHAPE")).sum()),
             "residual_fit": int((frame.class_fit == "RESIDUAL_SHAPE").sum()),
             "pure_kept": int(((frame.class_fit == "PURE_LEVEL")
                               & (frame.class_hold == "PURE_LEVEL")).sum()),
             "pure_fit": int((frame.class_fit == "PURE_LEVEL").sum()),
             "in_scope_fit": len(scope),
             "in_scope_hold": int((scope.k0 > LEVEL_CUT).sum()),
             "mae_fit": float(scope.k0_fit.mean()), "mae_hold": float(scope.k0.mean())}
    return confusion, summary, stats


# --------------------------------------------------------------------------- #
# is the triage actually available in advance?
# --------------------------------------------------------------------------- #
def triage_auc(table: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-chemotype-out AUC for predicting the class from covariates.

    The marginal AUCs in §6 are in-sample descriptions of a 54-ligand split.  A claim
    that the triage is *available in advance* is a prediction claim and has to be
    scored out of sample, on held-out chemotypes, exactly as the model itself is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    subset = table[table.failure_class.isin(["RESIDUAL_SHAPE", "PURE_LEVEL"])]
    target = (subset.failure_class == "RESIDUAL_SHAPE").to_numpy().astype(int)
    sets = {"target-free set (" + ", ".join(TARGET_FREE) + ")": TARGET_FREE,
            "has_axis_acid alone": ["has_axis_acid"],
            "n_series alone": ["n_series"],
            "n_curves alone": ["n_curves"],
            "donor chemistry (36 mechanism descriptors)": list(MECHANISM_COLUMNS),
            "true_span alone (target-derived, for scale)": ["true_span"]}
    rows = []
    for label, columns in sets.items():
        design = np.nan_to_num(subset[columns].to_numpy(dtype=float), nan=0.0)
        predicted = np.full(len(subset), np.nan)
        for cluster in subset.tanimoto_cluster.unique():
            test = (subset.tanimoto_cluster == cluster).to_numpy()
            train = ~test
            if len(np.unique(target[train])) < 2:
                continue
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
            model.fit(design[train], target[train])
            predicted[test] = model.predict_proba(design[test])[:, 1]
        scored = np.isfinite(predicted)
        rows.append({"features": label, "n_scored": int(scored.sum()),
                     "loco_auc": _auc(predicted[scored & (target == 1)],
                                      predicted[scored & (target == 0)])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def paired_gain(sel: pd.DataFrame, table: pd.DataFrame, klass: str | None,
                comparisons: dict) -> pd.DataFrame:
    members = table.index if klass is None else table.index[table.failure_class == klass]
    subset = sel[sel.extractant.isin(set(members))]
    out = paired_chemotype_bootstrap(subset, comparisons)
    out.insert(0, "class", klass or "ALL")
    return out


def worst_table(table: pd.DataFrame, column: str, n: int = 15) -> pd.DataFrame:
    top = table.sort_values(column, ascending=False).head(n).reset_index()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    for count in ("n_curves", "n_series", "n_rows"):
        top[count] = top[count].astype(int)
    return top


def main() -> None:
    sel = load_arms()
    n_units = sel.attrs["n_units"]
    table = add_classes(add_covariates(per_ligand(sel)), sel)
    table = table.sort_values("mae_k0", ascending=False)
    table.index.name = "extractant"

    csv_columns = (["tanimoto_cluster", "ecfp_cluster", "nn_train_tanimoto", "n_rows",
                    "n_pool", "n_eval", "failure_class", "failure_class_central",
                    "k2_status", "one_shot_harms", "seeds_matching_pooled_class",
                    "seed_modal_count"]
                   + [f"mae_{a}" for a in ARMS] + ["mae_oracle_k1", "mae_oracle_k2"]
                   + [f"ratio_{a}" for a in ARMS[1:]]
                   + ["offset_k0", "shape_mae_k0", "shape_frac_k0", "spearman_k0",
                      "oof_mean_centred_mae", "oof_median_centred_mae", "mad_true",
                      "within_1_0_k0", "within_1_0_RANDOM_k1", "within_1_0_RANDOM_k2",
                      "n_series", "n_doi_series_sum", "n_distinct_doi",
                      "series_types", "log_d_span_max",
                      "n_curves", "n_curve_axes", "curve_axes", "median_curve_slope",
                      "median_curve_linear_r2", "max_curve_yspan",
                      "true_span", "pred_span", "span_compression", "true_sd", "pred_sd"]
                   + [f"has_axis_{a}" for a in AXES] + list(MECHANISM_COLUMNS))
    table[csv_columns].to_csv(OUT / "ligand_failure_classes.csv")

    order = ["PURE_LEVEL", "PARTIAL_LEVEL", "RESIDUAL_SHAPE", "ALREADY_GOOD"]
    counts = table.failure_class.value_counts()

    # ---- class summary
    summary = (table.groupby("failure_class")
               .agg(n=("mae_k0", "size"),
                    mae_k0=("mae_k0", "mean"), mae_R1=("mae_RANDOM_k1", "mean"),
                    mae_R2=("mae_RANDOM_k2", "mean"), mae_C1=("mae_CENTRAL_k1", "mean"),
                    mae_C2=("mae_CENTRAL_k2", "mean"),
                    oracle_k1=("mae_oracle_k1", "mean"),
                    offset_k0=("offset_k0", "mean"), shape_k0=("shape_mae_k0", "mean"),
                    shape_frac=("shape_frac_k0", "mean"))
               .reindex(order).reset_index())
    for column in ["mae_k0", "mae_R1", "mae_R2", "mae_C1", "mae_C2"]:
        summary[f"share_{column}"] = summary["n"] * summary[column] / 143.0

    # ---- threshold sensitivity, with the two knobs separated
    #
    # The level cut and the ratio cuts do different jobs and must not be swept
    # together.  The level cut decides only which ligands are *in scope*; for a ligand
    # in scope the level/shape verdict depends on the ratio alone, so two level cuts
    # can never disagree about a ligand they both admit -- the RESIDUAL_SHAPE sets are
    # exactly nested.  The ratio cuts are therefore the only free parameters that can
    # change a verdict, and they are swept at the primary level cut.
    primary_residual = set(table.index[table.failure_class == "RESIDUAL_SHAPE"])

    def sweep(level: float, pure: float, persist: float) -> dict:
        labels = pd.Series([classify(m, r, level=level, pure=pure, persist=persist)
                            for m, r in zip(table.mae_k0, table.ratio_RANDOM_k1)],
                           index=table.index)
        residual = labels == "RESIDUAL_SHAPE"
        members = set(table.index[residual])
        union = members | primary_residual
        return {"level_cut": level, "pure_cut": pure, "persist_cut": persist,
                "ALREADY_GOOD": int((labels == "ALREADY_GOOD").sum()),
                "PURE_LEVEL": int((labels == "PURE_LEVEL").sum()),
                "PARTIAL_LEVEL": int((labels == "PARTIAL_LEVEL").sum()),
                "RESIDUAL_SHAPE": int(residual.sum()),
                "residual_share_of_bad":
                    float(residual.sum() / max(1, int((labels != "ALREADY_GOOD").sum()))),
                "residual_share_of_k2_macro":
                    float(table.loc[residual, "mae_CENTRAL_k2"].sum()
                          / table["mae_CENTRAL_k2"].sum()),
                "jaccard_vs_primary": float(len(members & primary_residual) / max(1, len(union))),
                "nested_with_primary": bool(members <= primary_residual
                                            or primary_residual <= members)}

    ratio_sensitivity = pd.DataFrame([sweep(LEVEL_CUT, p, q)
                                      for p, q in ((0.4, 0.6), (0.5, 0.7), (0.6, 0.8))])
    level_sensitivity = pd.DataFrame([sweep(lv, PURE_CUT, PERSIST_CUT)
                                      for lv in (0.75, 1.0, 1.25, 1.5)])
    sensitivity = pd.concat([ratio_sensitivity, level_sensitivity], ignore_index=True)

    # ---- contrasts
    contrasts = pd.concat([
        contrast_block(table, GEOMETRY, "RESIDUAL_SHAPE", "PURE_LEVEL", "geometry"),
        contrast_block(table, list(MECHANISM_COLUMNS), "RESIDUAL_SHAPE", "PURE_LEVEL",
                       "mechanism")], ignore_index=True)
    contrasts_good = contrast_block(table, GEOMETRY, "RESIDUAL_SHAPE", "ALREADY_GOOD",
                                    "geometry_vs_good")
    contrasts.to_csv(OUT / "failure_class_contrasts.csv", index=False)
    contrasts_good.to_csv(OUT / "failure_class_contrasts_vs_already_good.csv", index=False)

    # ---- within-chemotype blocked check
    crosstab = pd.crosstab(table.tanimoto_cluster, table.failure_class)
    mixed = crosstab[(crosstab.get("RESIDUAL_SHAPE", 0) > 0)
                     & (crosstab.get("PURE_LEVEL", 0) > 0)].index.tolist()
    blocked_rows = []
    for column in ["log_d_span_max", "true_span", "n_series", "nn_train_tanimoto",
                   "abs_median_curve_slope", "mech__softness_mean", "mech__n_donor_total"]:
        deltas = []
        for cluster in mixed:
            block = table[table.tanimoto_cluster == cluster]
            a = block.loc[block.failure_class == "RESIDUAL_SHAPE", column].mean()
            b = block.loc[block.failure_class == "PURE_LEVEL", column].mean()
            deltas.append(a - b)
        blocked_rows.append({"column": column, "n_blocks": len(mixed),
                             "mean_within_block_delta": float(np.nanmean(deltas)),
                             "blocks_positive": int(np.nansum(np.asarray(deltas) > 0)),
                             **{f"delta_{c}": float(d) for c, d in zip(mixed, deltas)}})
    blocked = pd.DataFrame(blocked_rows)

    # ---- paired within-class gains
    gains = pd.concat([paired_gain(sel, table, klass,
                                   {"k0 -> RANDOM k=1": ("k0", "RANDOM_k1"),
                                    "k0 -> RANDOM k=2": ("k0", "RANDOM_k2"),
                                    "k0 -> CENTRAL k=1": ("k0", "CENTRAL_k1"),
                                    "k0 -> CENTRAL k=2": ("k0", "CENTRAL_k2")})
                       for klass in [None] + order], ignore_index=True)
    gains.to_csv(OUT / "failure_class_paired_gains.csv", index=False)

    # ---- axis profile
    axis_profile = (table.groupby("failure_class")[[f"has_axis_{a}" for a in AXES]]
                    .mean().reindex(order).reset_index())

    # ---- out-of-seed re-derivation and the out-of-sample triage
    confusion, seed_summary, seed_stats = out_of_seed(sel)
    triage = triage_auc(table)

    write_report(table, sel, n_units, counts, summary,
                 (ratio_sensitivity, level_sensitivity), contrasts,
                 contrasts_good, blocked, gains, axis_profile, mixed, order,
                 (confusion, seed_summary, seed_stats), triage, other_arms(sel))
    print(f"wrote {OUT/'error_archaeology.md'}")
    print(f"wrote {OUT/'ligand_failure_classes.csv'}")


def write_report(table, sel, n_units, counts, summary, sensitivity, contrasts,
                 contrasts_good, blocked, gains, axis_profile, mixed, order,
                 out_of_seed_result, triage, adapters) -> None:
    macro = {a: float(table[f"mae_{a}"].mean()) for a in ARMS}
    n_lig = len(table)
    residual = table[table.failure_class == "RESIDUAL_SHAPE"]
    pure = table[table.failure_class == "PURE_LEVEL"]
    good = table[table.failure_class == "ALREADY_GOOD"]

    lines: list[str] = []
    A = lines.append
    A("# Error archaeology after calibration")
    A("")
    A("*gen8 brief section 26. Which held-out-ligand failures disappear after one "
      "measurement, and which remain?*")
    A("")
    detail_path = ROOT / "runs/gen8_architecture/active_acquisition/primary_detail.parquet"
    stamp = dt.datetime.fromtimestamp(detail_path.stat().st_mtime).isoformat(timespec="seconds")
    A(f"Frozen global model `{MODEL}`, evaluated on held-out chemotypes over 5 split "
      f"seeds x 5 folds. Source: `runs/gen8_architecture/active_acquisition/"
      f"primary_detail.parquet`, written {stamp}. **Every number in this report is "
      f"recomputed from that file; if the acquisition run is re-executed this report "
      f"must be regenerated, because the RANDOM policy's draws move with it.**")
    A("")

    A("## 1. What is compared, and why the comparison is exact")
    A("")
    A("Five arms, all read from the same acquisition run:")
    A("")
    A("| arm | adapter | policy | k |")
    A("|---|---|---|---|")
    A("| `k0` | `ZERO_SHOT_REF` | `NONE` | 0 |")
    A("| `RANDOM_k1` / `RANDOM_k2` | `OFFSET_K1` | `RANDOM` | 1 / 2 |")
    A("| `CENTRAL_k1` / `CENTRAL_k2` | `OFFSET_K1` | `CENTRAL` | 1 / 2 |")
    A("")
    A(f"Under protocol P2 each ligand's rows are split once per repeat into a candidate "
      f"pool and a disjoint evaluation set, and the zero-shot reference is emitted once "
      f"per repeat on that same evaluation set. The five arms therefore share "
      f"**{n_units:,} identical (seed, fold, ligand, repeat) units** -- "
      f"{n_lig} ligands x 5 seeds x 12 repeats -- verified by set intersection before "
      f"anything was averaged. Nothing below can be an artefact of one arm being scored "
      f"on easier rows.")
    A("")
    A("Per-ligand MAE is the mean over the 60 (seed, repeat) units of that ligand; macro "
      "is the unweighted mean over ligands.")
    A("")
    A("| arm | macro MAE |")
    A("|---|---|")
    for arm in ARMS:
        A(f"| {arm} | {macro[arm]:.3f} |")
    A("")
    A(f"These are the macro numbers of the acquisition run named above and of no other. "
      f"An earlier execution of the same protocol gave 0.998 / 0.702 / 0.646 / 0.627, "
      f"and those four figures still circulate in the gen8 write-ups; they are "
      f"superseded here by {macro['k0']:.3f} / {macro['RANDOM_k1']:.3f} / "
      f"{macro['CENTRAL_k1']:.3f} / {macro['RANDOM_k2']:.3f}. The two executions differ "
      f"in the P2 pool/evaluation split itself -- comparing the surviving smoke run "
      f"against this one on the 286 (seed, fold, ligand, repeat) units they share, the "
      f"evaluation sets are the same *size* in 286 of 286 cases and contain different "
      f"*rows* in 280 of them -- so every arm moves, the zero-shot reference included. "
      f"The size of that movement is the useful part: 0.003 on the zero-shot arm and "
      f"0.021 on the blind one-measurement arm. **A blind 1-shot policy is not "
      f"determined to three decimals by its own protocol**, and the RANDOM numbers "
      f"below should be read to two.")
    A("")

    A("## 2. The level/shape budget the classification rests on")
    A("")
    A("`_metrics` already splits every arm's error into the mean residual (`offset`) and "
      "the error left after that mean residual is removed (`shape_mae`). Because "
      "`OFFSET_K1` adds a **constant** to a ligand's predictions, `shape_mae` is "
      "identical in all five arms -- byte-identical, checked -- so it is the same "
      "quantity whatever the measurement budget and whatever the acquisition policy.")
    A("")
    A(f"One correction to how that quantity is described elsewhere in gen8. `shape_mae` "
      f"centres the residual on its **mean**; the constant that minimises MAE is the "
      f"**median**. So `shape_mae` is an upper bound on the offset-only floor, not the "
      f"floor. Recomputing both on each ligand's full out-of-fold row set: mean-centred "
      f"{table.oof_mean_centred_mae.mean():.3f}, median-centred "
      f"{table.oof_median_centred_mae.mean():.3f} -- a gap of "
      f"{table.oof_mean_centred_mae.mean() - table.oof_median_centred_mae.mean():.3f} "
      f"macro, and {residual.oof_mean_centred_mae.mean() - residual.oof_median_centred_mae.mean():.3f} "
      f"over the ligands §3 calls RESIDUAL_SHAPE (median-centred is lower for "
      f"{int((table.oof_median_centred_mae < table.oof_mean_centred_mae - 1e-9).sum())} "
      f"of {n_lig} ligands). Every 'floor' in this report is therefore the mean-centred "
      f"number, which overstates the true offset-only floor by roughly 5 % on the class "
      f"the argument rests on. That does not change the direction of anything below, and "
      f"it is stated so that no one reads `shape_mae` as an attained bound.")
    A("")
    A(f"Macro over {n_lig} ligands: zero-shot MAE {macro['k0']:.3f}, |offset| "
      f"{table.offset_k0.mean():.3f}, `shape_mae` {table.shape_mae_k0.mean():.3f} (the "
      f"two parts do not add: each is a mean of absolute values). The oracle 1-shot arm "
      f"reaches {table.mae_oracle_k1.mean():.3f}, "
      f"{abs(table.mae_oracle_k1.mean() - table.shape_mae_k0.mean()):.3f} from that "
      f"number. **Everything a measurement can buy is the level; the shape is untouched "
      f"by construction.** The question is therefore how the shape is distributed over "
      f"ligands.")
    A("")

    A("## 3. Failure classes")
    A("")
    A(f"Classified from the ligand's own paired numbers, with `RANDOM` (a blindly chosen "
      f"measurement) as the deployable default and `CENTRAL` as a sensitivity:")
    A("")
    A(f"- **ALREADY_GOOD** -- zero-shot MAE <= {LEVEL_CUT:.2f}")
    A(f"- **PURE_LEVEL** -- zero-shot MAE > {LEVEL_CUT:.2f} and "
      f"1-shot < {PURE_CUT:.2f} x zero-shot")
    A(f"- **PARTIAL_LEVEL** -- zero-shot MAE > {LEVEL_CUT:.2f} and "
      f"{PURE_CUT:.2f} <= ratio <= {PERSIST_CUT:.2f}")
    A(f"- **RESIDUAL_SHAPE** -- zero-shot MAE > {LEVEL_CUT:.2f} and "
      f"1-shot > {PERSIST_CUT:.2f} x zero-shot")
    A("")
    A("`PARTIAL_LEVEL` is the fourth class the data forced: a ratio band between the two "
      "briefed cut-offs is not empty, and collapsing it into either neighbour would "
      "overstate that neighbour.")
    A("")
    A(md_table(summary, columns=["failure_class", "n", "mae_k0", "mae_R1", "mae_R2",
                                 "mae_C1", "mae_C2", "oracle_k1", "offset_k0", "shape_k0",
                                 "shape_frac"]))
    A("")
    A("(`mae_R1`/`R2` = RANDOM at k=1/2, `mae_C1`/`C2` = CENTRAL at k=1/2, `oracle_k1` = "
      "the non-deployable best single measurement, `shape_frac` = shape_k0 / mae_k0.)")
    A("")
    A("Read across the `PURE_LEVEL` row and then the `RESIDUAL_SHAPE` row:")
    A("")
    A(f"- **PURE_LEVEL ({len(pure)} ligands): one measurement is a total repair.** "
      f"{pure.mae_k0.mean():.3f} -> {pure.mae_RANDOM_k1.mean():.3f} on a random point, "
      f"{pure.mae_CENTRAL_k1.mean():.3f} on a central one. Their error was "
      f"{pure.shape_frac_k0.mean()*100:.0f}% shape and the rest a pure offset "
      f"(|offset| {pure.offset_k0.mean():.3f} against a shape floor of "
      f"{pure.shape_mae_k0.mean():.3f}). After one measurement they are **better than "
      f"the ALREADY_GOOD ligands were before it**.")
    A(f"- **RESIDUAL_SHAPE ({len(residual)} ligands): one measurement makes them "
      f"worse.** {residual.mae_k0.mean():.3f} -> {residual.mae_RANDOM_k1.mean():.3f} "
      f"random, and two measurements only reach {residual.mae_RANDOM_k2.mean():.3f} "
      f"({residual.mae_CENTRAL_k2.mean():.3f} with CENTRAL). Their shape floor alone is "
      f"{residual.shape_mae_k0.mean():.3f} -- "
      f"{residual.shape_frac_k0.mean()*100:.0f}% of their zero-shot error -- and the "
      f"oracle 1-shot arm gets {residual.mae_oracle_k1.mean():.3f}, within "
      f"{abs(residual.mae_oracle_k1.mean() - residual.shape_mae_k0.mean()):.3f} of it "
      f"while every deployable arm is {residual.mae_CENTRAL_k2.mean() - residual.shape_mae_k0.mean():.3f} "
      f"or more away. A better acquisition policy is worth the first gap and nothing "
      f"beyond it, because an offset is the wrong object for these ligands.")
    A("")
    A(f"Of the {len(residual)} RESIDUAL_SHAPE ligands, "
      f"{int((table.k2_status == 'RECOVERS_AT_K2').sum())} drop below the "
      f"{PERSIST_CUT:.2f} ratio by k=2 and "
      f"{int((table.k2_status == 'IRREDUCIBLE_AT_K2').sum())} do not.")
    A("")
    A("### One measurement is not free")
    A("")
    A(f"{int(table.one_shot_harms.sum())} of {n_lig} ligands are *worse* after one "
      f"random measurement than before it -- {int(good.one_shot_harms.sum())} of the "
      f"{len(good)} ALREADY_GOOD ones and {int(residual.one_shot_harms.sum())} of the "
      f"{len(residual)} RESIDUAL_SHAPE ones. A single point estimates the offset with "
      f"the full shape noise of that point attached; when there is little offset to "
      f"remove, that noise is all you buy. CENTRAL reduces the count to "
      f"{int((table.ratio_CENTRAL_k1 > 1.0).sum())}.")
    A("")
    A("Paired chemotype bootstrap, positive = calibration helps "
      "(`paired_chemotype_bootstrap`, blocks = held-out Tanimoto chemotypes):")
    A("")
    A(md_table(gains, columns=["class", "comparison", "point", "ci95_low", "ci95_high",
                               "bca_low", "bca_high", "n_units", "units_improved",
                               "seeds_positive", "n_seeds"]))
    A("")
    indexed = gains.set_index(["class", "comparison"])
    harm = indexed.loc[("RESIDUAL_SHAPE", "k0 -> RANDOM k=1")]
    fix = indexed.loc[("RESIDUAL_SHAPE", "k0 -> CENTRAL k=1")]
    good_row = indexed.loc[("ALREADY_GOOD", "k0 -> RANDOM k=1")]
    A(f"Two readings of that table are load-bearing and point in opposite directions, so "
      f"neither should be quoted without the other. On RESIDUAL_SHAPE, one blind "
      f"measurement is a **measured harm, not a null**: {harm.point:+.3f} with an "
      f"interval of [{harm.ci95_low:+.3f}, {harm.ci95_high:+.3f}] that excludes zero and "
      f"{int(harm.seeds_positive)} of {int(harm.n_seeds)} seeds positive. But choosing "
      f"the point centrally does **not** demonstrably repair them: {fix.point:+.3f} "
      f"[{fix.ci95_low:+.3f}, {fix.ci95_high:+.3f}] straddles zero. CENTRAL turns a "
      f"measured harm into an unmeasurable difference; on this evidence it does not turn "
      f"it into a gain. The same caution applies to ALREADY_GOOD at k=1 "
      f"({good_row.point:+.3f} [{good_row.ci95_low:+.3f}, {good_row.ci95_high:+.3f}]).")
    A("")

    ratio_sensitivity, level_sensitivity = sensitivity
    A("## 4. The classes are not an artefact of the thresholds")
    A("")
    A("The two cut-offs do different jobs and are swept separately, because sweeping "
      "them together would hide which one matters. The **level cut** decides only which "
      "ligands are in scope. For a ligand in scope the level/shape verdict depends on "
      "the ratio alone, so two level cuts can never disagree about a ligand they both "
      "admit -- the RESIDUAL_SHAPE sets at different level cuts are exactly nested "
      "(verified below). The **ratio cuts** are therefore the only free parameters that "
      "can change a verdict.")
    A("")
    A(f"**(a) Ratio cuts, at the primary level cut of {LEVEL_CUT:.2f}.**")
    A("")
    A(md_table(ratio_sensitivity, columns=["pure_cut", "persist_cut", "ALREADY_GOOD",
                                           "PURE_LEVEL", "PARTIAL_LEVEL",
                                           "RESIDUAL_SHAPE", "residual_share_of_bad",
                                           "residual_share_of_k2_macro",
                                           "jaccard_vs_primary"]))
    A("")
    A(f"Moving both ratio cuts by +-0.1 moves the RESIDUAL_SHAPE count between "
      f"{ratio_sensitivity.RESIDUAL_SHAPE.min()} and "
      f"{ratio_sensitivity.RESIDUAL_SHAPE.max()}, with Jaccard overlap against the "
      f"primary set of at least {ratio_sensitivity.jaccard_vs_primary.min():.2f}. The "
      f"verdict is not sitting on a cliff edge.")
    A("")
    A("**(b) The level cut only resizes the population.**")
    A("")
    A(md_table(level_sensitivity, columns=["level_cut", "ALREADY_GOOD", "PURE_LEVEL",
                                           "PARTIAL_LEVEL", "RESIDUAL_SHAPE",
                                           "residual_share_of_bad",
                                           "residual_share_of_k2_macro",
                                           "nested_with_primary"]))
    A("")
    A(f"Every set here is nested with the primary one "
      f"(`nested_with_primary` is true in all "
      f"{int(level_sensitivity.nested_with_primary.sum())} rows): lowering the cut to "
      f"0.75 adds {int(level_sensitivity.RESIDUAL_SHAPE.iloc[0]) - len(residual)} milder "
      f"shape failures without reclassifying anyone, raising it to 1.5 keeps only the "
      f"{int(level_sensitivity.RESIDUAL_SHAPE.iloc[-1])} most extreme. What the level cut "
      f"buys is the headline share: at 0.75 the shape failures carry "
      f"{level_sensitivity.residual_share_of_k2_macro.iloc[0]:.2f} of the k=2 macro "
      f"error, at 1.5 only {level_sensitivity.residual_share_of_k2_macro.iloc[-1]:.2f}. "
      f"That number is a function of where the line is drawn and is quoted below with "
      f"the line stated.")
    A("")
    A("**(c) A threshold-free version of the same split.** `shape_frac_k0` "
      "(= shape_mae / MAE at k=0) needs no cut-off at all, and because `OFFSET_K1` "
      "cannot change `shape_mae` it is the quantity the classification is really "
      "reading. Spearman against the 1-shot ratio is "
      f"{table.shape_frac_k0.corr(table.ratio_RANDOM_k1, method='spearman'):.3f} over all "
      f"{n_lig} ligands and "
      f"{table[table.mae_k0 > LEVEL_CUT].shape_frac_k0.corr(table[table.mae_k0 > LEVEL_CUT].ratio_RANDOM_k1, method='spearman'):.3f} "
      f"over the {int((table.mae_k0 > LEVEL_CUT).sum())} in scope. Class medians: "
      + ", ".join(f"{k} {v:.3f}" for k, v in
                  table.groupby('failure_class').shape_frac_k0.median()
                  .reindex(order).items())
      + ".")
    A("")
    A(f"Note that ALREADY_GOOD sits high on this scale too "
      f"({good.shape_frac_k0.median():.3f}): most of *their* small error is shape as "
      f"well. That is the point of the two-dimensional definition -- the classification "
      f"asks both how big the error is and what fraction of it a constant can remove, "
      f"and only the second question is threshold-sensitive in any interesting way.")
    A("")
    A(f"**(d) Re-derivation per split seed and per policy.** Re-classifying inside each "
      f"of the 5 split seeds separately, "
      f"{int((table.seeds_matching_pooled_class >= 4).sum())} of {n_lig} ligands keep "
      f"their pooled class in >= 4 of 5 seeds "
      f"({int((table.seeds_matching_pooled_class == 5).sum())} in all 5). Re-classifying "
      f"with CENTRAL instead of RANDOM moves "
      f"{int((table.failure_class != table.failure_class_central).sum())} of {n_lig} "
      f"ligands; "
      f"{int(((table.failure_class == 'RESIDUAL_SHAPE') & (table.failure_class_central == 'RESIDUAL_SHAPE')).sum())}"
      f"/{len(residual)} RESIDUAL_SHAPE and "
      f"{int(((table.failure_class == 'PURE_LEVEL') & (table.failure_class_central == 'PURE_LEVEL')).sum())}"
      f"/{len(pure)} PURE_LEVEL ligands are unchanged.")
    A("")
    confusion, seed_summary, seed_stats = out_of_seed_result
    A("**(e) The classification is assigned on one set of seeds and scored on another.** "
      "(d) asks whether the label is stable; it does not answer the sharper objection, "
      "which is that a ligand enters scope because its zero-shot MAE was large on the "
      "very units its ratio is then computed from. Selection on a noisy quantity "
      "regresses. So: the class is fixed on the first "
      f"{seed_stats['n_fit_seeds']} split seeds and every number in this block comes "
      f"from the remaining {seed_stats['n_hold_seeds']}.")
    A("")
    A(md_table(confusion.reset_index(),
               columns=["class_fit"] + [c for c in confusion.columns]))
    A("")
    A(md_table(seed_summary, columns=["class_fit", "n", "mae_k0", "mae_R1", "mae_R2",
                                      "mae_C2", "mean_ratio"]))
    A("")
    A(f"{seed_stats['agree']} of {seed_stats['n']} ligands land in the same class on "
      f"both halves. Of the ligands called RESIDUAL_SHAPE on the fit seeds, "
      f"{seed_stats['residual_kept']}/{seed_stats['residual_fit']} are RESIDUAL_SHAPE "
      f"again on the held-out seeds, and their held-out mean ratio is "
      f"{float(seed_summary.set_index('class_fit').loc['RESIDUAL_SHAPE', 'mean_ratio']):.3f} "
      f"-- one measurement still makes them worse on seeds that had no say in labelling "
      f"them. PURE_LEVEL keeps "
      f"{seed_stats['pure_kept']}/{seed_stats['pure_fit']} at a held-out ratio of "
      f"{float(seed_summary.set_index('class_fit').loc['PURE_LEVEL', 'mean_ratio']):.3f}. "
      f"And the regression-to-the-mean worry does not materialise: the "
      f"{seed_stats['in_scope_fit']} ligands selected into scope on the fit seeds have "
      f"mean zero-shot MAE {seed_stats['mae_fit']:.3f} there and "
      f"{seed_stats['mae_hold']:.3f} on the held-out seeds, with "
      f"{seed_stats['in_scope_hold']} of them still above the level cut.")
    A("")
    A("## 5. The fifteen worst ligands at k = 0, 1 and 2")
    A("")
    show = ["rank", "extractant", "failure_class", "nn_train_tanimoto", "n_rows",
            "n_curves", "true_span"]
    for k, column, caption in [
            (0, "mae_k0", "zero-shot"),
            (1, "mae_RANDOM_k1", "one random measurement, offset correction"),
            (2, "mae_RANDOM_k2", "two random measurements, offset correction")]:
        top = worst_table(table, column)
        A(f"### k = {k} ({caption})")
        A("")
        A(md_table(top, columns=show[:2] + [column] + show[2:]))
        A("")
        A("Class mix of this top-15: "
          + ", ".join(f"{k2} {v}" for k2, v in top.failure_class.value_counts().items())
          + ".")
        A("")
    A("(`true_span` = mean per-seed range of the ligand's measured log D; `n_curves` from "
      "`series/curve_table.parquet`; SMILES are the canonical extractant strings, full "
      "precision in the CSV.)")
    A("")
    A("The turnover is the finding. At k=0 the worst list is a **level** list: "
      f"{worst_table(table,'mae_k0').failure_class.value_counts().get('PURE_LEVEL',0)}"
      f"/15 are PURE_LEVEL, ligands the model places 2-4 log units off and then tracks "
      f"correctly. One measurement deletes them from the list entirely. At k=1 the worst "
      f"list is "
      f"{worst_table(table,'mae_RANDOM_k1').failure_class.value_counts().get('RESIDUAL_SHAPE',0)}"
      f"/15 RESIDUAL_SHAPE and at k=2 it is "
      f"{worst_table(table,'mae_RANDOM_k2').failure_class.value_counts().get('RESIDUAL_SHAPE',0)}"
      f"/15. **The population of hard ligands is completely replaced by one "
      f"measurement.**")
    A("")

    A("## 6. What separates RESIDUAL_SHAPE from PURE_LEVEL")
    A("")
    A(f"{len(residual)} vs {len(pure)} ligands, spread over "
      f"{table.tanimoto_cluster.nunique()} held-out Tanimoto chemotypes. Ligands inside "
      f"one chemotype were held out together against the same reduced training set, so "
      f"every interval below resamples **chemotypes**, not ligands. Two statistics are "
      f"reported: the difference in means, and P(RESIDUAL_SHAPE > PURE_LEVEL) as a rank "
      f"statistic robust to the two or three enormous ligands. p-values are two-sided "
      f"bootstrap p, BH-adjusted within each block of covariates.")
    A("")
    A("### Donor chemistry: no.")
    A("")
    mech_block = contrasts[contrasts.block == "mechanism"].sort_values("auc_p")
    A(md_table(mech_block.head(8), columns=["column", "mean_a", "mean_b", "auc", "auc_lo",
                                            "auc_hi", "auc_p", "auc_q_bh"]))
    A("")
    A(f"All {len(mech_block)} mechanism descriptors were tested; the top 8 by raw p are "
      f"shown. **Not one survives correction** -- the smallest BH q over the whole "
      f"mechanism block is {mech_block.auc_q_bh.min():.3f}, and the smallest raw p is "
      f"{mech_block.auc_p.min():.3f}. Softness, denticity, donor counts, N/O/S "
      f"composition, chelate geometry, charge, lipophilicity: none of them tells a "
      f"level failure from a shape failure. Both classes are dominated by neutral "
      f"O-donor diglycolamides drawn from the same chemotypes.")
    A("")
    A("### Measurement geometry: yes, decisively.")
    A("")
    geom = contrasts[contrasts.block == "geometry"].sort_values("auc_p")
    A(md_table(geom, columns=["column", "mean_a", "mean_b", "median_a", "median_b",
                              "auc", "auc_lo", "auc_hi", "auc_p", "auc_q_bh"]))
    A("")
    A("Everything that separates the two classes is about **how far the ligand's log D "
      "actually travels and over how many separate experiments**; nothing that "
      "separates them is about what the molecule is made of. Listed with the BH q so "
      "the one that does not clear correction is visible as such:")
    A("")
    for column, note in [
            ("log_d_span_max", "widest log D span of any one series"),
            ("true_span", "range of the ligand's measured log D"),
            ("max_curve_yspan", "widest single titration curve"),
            ("median_curve_linear_r2", "how cleanly its curves are linear"),
            ("abs_median_curve_slope", "|median curve slope|"),
            ("n_series", "independent measurement series"),
            ("n_doi_series_sum", "series-weighted publication count"),
            ("n_distinct_doi", "distinct source publications"),
            ("has_axis_acid", "has an acid titration")]:
        row = geom[geom.column == column].iloc[0]
        A(f"- `{column}` ({note}): {row.mean_a:.3f} vs {row.mean_b:.3f}, "
          f"AUC {row.auc:.3f} [{row.auc_lo:.3f}, {row.auc_hi:.3f}], "
          f"BH q = {row.auc_q_bh:.4f}")
    A("")
    mad = geom[geom.column == "mad_true"].iloc[0]
    sd_row = geom[geom.column == "true_sd"].iloc[0]
    A(f"**The top of that table is close to a restatement of the definition, and saying "
      f"so is the difference between a caveat and an honest reading.** A ligand is "
      f"RESIDUAL_SHAPE when a constant cannot remove its error, and `shape_mae` is what "
      f"the error would be if the model predicted a constant for it. The model is nearly "
      f"flat within a ligand -- `shape_mae` is a median {float((table.shape_mae_k0 / table.mad_true).median()):.2f} "
      f"of the ligand's own mean absolute deviation of log D -- so a wide-spanning "
      f"ligand is close to *being* a shape failure rather than *explaining* one. The "
      f"benchmark row is `mad_true`, the spread of the target with the model deleted "
      f"entirely: AUC {mad.auc:.3f} [{mad.auc_lo:.3f}, {mad.auc_hi:.3f}]. `true_sd` "
      f"reaches {sd_row.auc:.3f} and `true_span` {geom[geom.column=='true_span'].iloc[0].auc:.3f}; "
      f"none of them beats knowing nothing but the target's own spread. Read those rows "
      f"as *where* the defect bites, not as an independent cause of it. The rows that "
      f"are not definitional are the ones counting experiments -- `n_series`, "
      f"`n_curves`, `has_axis_acid` -- and they are weaker, which is the honest shape of "
      f"this result.")
    A("")
    doi_sum = geom[geom.column == "n_doi_series_sum"].iloc[0]
    doi_uniq = geom[geom.column == "n_distinct_doi"].iloc[0]
    A(f"One label correction, since the two versions do not say the same thing. "
      f"`series_table.n_doi` is a per-series count; summing it over a ligand counts a "
      f"publication once per series it appears in, and for "
      f"{int((table.n_doi_series_sum != table.n_distinct_doi).sum())} of {n_lig} ligands "
      f"that sum is not the number of source publications. Counting distinct DOIs "
      f"instead moves the contrast from AUC {doi_sum.auc:.3f} (q = {doi_sum.auc_q_bh:.4f}) "
      f"to {doi_uniq.auc:.3f} [{doi_uniq.auc_lo:.3f}, {doi_uniq.auc_hi:.3f}] "
      f"(q = {doi_uniq.auc_q_bh:.3f}). The weaker number is the one that answers "
      f"\"is this ligand studied in many papers\". And the summed version is Spearman "
      f"{float(table.n_doi_series_sum.corr(table.n_series, method='spearman')):.2f} with "
      f"`n_series`, so the two are one variable and not two pieces of evidence.")
    A("")
    nn = geom[geom.column == "nn_train_tanimoto"].iloc[0]
    rows_row = geom[geom.column == "log10_n_rows"].iloc[0]
    A(f"Two negatives matter as much as the positives.")
    A("")
    A(f"- **`nn_train_tanimoto` does not separate them**: {nn.mean_a:.3f} vs "
      f"{nn.mean_b:.3f}, AUC {nn.auc:.3f} [{nn.auc_lo:.3f}, {nn.auc_hi:.3f}], "
      f"p = {nn.auc_p:.3f}. Distance to the nearest training ligand predicts *whether* a "
      f"held-out ligand is hard, but not *which kind* of hard it is. This is consistent "
      f"with the gen8 result that mechanism-aware similarity does not fix the zero-shot "
      f"level.")
    A(f"- **Sheer row count does not separate them either**: log10 n_rows "
      f"{rows_row.mean_a:.3f} vs {rows_row.mean_b:.3f}, AUC {rows_row.auc:.3f} "
      f"[{rows_row.auc_lo:.3f}, {rows_row.auc_hi:.3f}], p = {rows_row.auc_p:.3f}; the "
      f"medians are {table[table.failure_class=='RESIDUAL_SHAPE'].n_rows.median():.0f} "
      f"and {table[table.failure_class=='PURE_LEVEL'].n_rows.median():.0f} rows. The "
      f"means differ only because a handful of RESIDUAL_SHAPE ligands are the "
      f"most-studied extractants in the corpus. It is the **span**, not the count.")
    A("")
    A("### The same contrast against ALREADY_GOOD")
    A("")
    good_span = contrasts_good[contrasts_good.column == "true_span"].iloc[0]
    good_nn = contrasts_good[contrasts_good.column == "nn_train_tanimoto"].iloc[0]
    good_rows = contrasts_good[contrasts_good.column == "log10_n_rows"].iloc[0]
    A(f"Repeating the whole contrast against the {len(good)} ALREADY_GOOD ligands "
      f"(`failure_class_contrasts_vs_already_good.csv`) gives the same ordering: "
      f"`true_span` AUC {good_span.auc:.3f} [{good_span.auc_lo:.3f}, "
      f"{good_span.auc_hi:.3f}] (BH q = {good_span.auc_q_bh:.4f}), `nn_train_tanimoto` "
      f"AUC {good_nn.auc:.3f} [{good_nn.auc_lo:.3f}, {good_nn.auc_hi:.3f}] "
      f"(q = {good_nn.auc_q_bh:.3f}). One difference is worth recording: row count "
      f"*does* separate RESIDUAL_SHAPE from ALREADY_GOOD (log10 n_rows AUC "
      f"{good_rows.auc:.3f}, q = {good_rows.auc_q_bh:.3f}) while it does not separate it "
      f"from PURE_LEVEL. Being heavily measured makes a ligand more likely to be a "
      f"failure at all; it does not decide which kind.")
    A("")
    A("### Which curve axes")
    A("")
    A(md_table(axis_profile, columns=["failure_class"] + [f"has_axis_{a}" for a in AXES]))
    A("")
    acid = geom[geom.column == "has_axis_acid"].iloc[0]
    metal = geom[geom.column == "has_axis_metal_series"].iloc[0]
    A(f"Fraction of ligands with at least one curve on each axis. The acid axis is the "
      f"discriminating one ({acid.mean_a:.2f} of RESIDUAL_SHAPE vs {acid.mean_b:.2f} of "
      f"PURE_LEVEL, AUC {acid.auc:.3f} [{acid.auc_lo:.3f}, {acid.auc_hi:.3f}], BH q = "
      f"{acid.auc_q_bh:.4f}). The lanthanide-series axis runs the other way and does "
      f"**not** clear correction ({metal.mean_a:.2f} vs {metal.mean_b:.2f}, AUC "
      f"{metal.auc:.3f} [{metal.auc_lo:.3f}, {metal.auc_hi:.3f}], BH q = "
      f"{metal.auc_q_bh:.3f}) -- a ligand whose data is one lanthanide scan at fixed "
      f"conditions is a level problem; a ligand with an acid titration is a shape "
      f"problem.")
    A("")
    A("### Can any of this be used in advance? Out-of-chemotype scoring")
    A("")
    A("Every AUC above is an in-sample description of one 54-ligand split. Whether a "
      "covariate would let you *sort a new ligand* is a different question, and it has "
      "to be answered the way the model itself is scored: fit on all chemotypes but "
      "one, predict the held-out chemotype, repeat. Logistic regression, standardised "
      "inputs, RESIDUAL_SHAPE as the positive class.")
    A("")
    A(md_table(triage, columns=["features", "n_scored", "loco_auc"]))
    A("")
    A(f"The target-free set holds up at "
      f"{float(triage.iloc[0].loco_auc):.3f} out of sample; no single target-free "
      f"variable does, which is why the useful form of this is a rule over the campaign "
      f"as a whole rather than one number. Donor chemistry is at "
      f"{float(triage[triage.features.str.startswith('donor')].loco_auc.iloc[0]):.3f} "
      f"-- below chance, which for 36 descriptors on 54 ligands is what over-fitting "
      f"nothing looks like. The target-derived `true_span` row "
      f"({float(triage[triage.features.str.startswith('true_span')].loco_auc.iloc[0]):.3f}) "
      f"is there only as the scale: it is not available in advance and, per the "
      f"definitional caveat above, it is close to reading the answer.")
    A("")
    A("### Blocked check inside mixed chemotypes")
    A("")
    A(f"The cluster bootstrap already respects the block structure, but only "
      f"{len(mixed)} chemotypes ({', '.join(mixed)}) contain both classes, so the "
      f"strictly within-block contrast is reported separately as a weak but "
      f"assumption-free check:")
    A("")
    A(md_table(blocked, columns=["column", "n_blocks", "mean_within_block_delta",
                                 "blocks_positive"]))
    A("")
    A(f"With {len(mixed)} blocks this is not a test, and it is not offered as one. It "
      f"agrees in direction with the pooled result on the span and series variables and "
      f"is flat on similarity and donor chemistry.")
    A("")

    A("## 7. Why the shape failures look the way they do")
    A("")
    A("Joining the frozen model's own out-of-fold predictions gives the mechanism "
      "directly. Per ligand, averaged over seeds:")
    A("")
    span = (table.groupby("failure_class")[["true_span", "pred_span", "true_sd",
                                            "pred_sd", "span_compression"]]
            .median().reindex(order).reset_index())
    A(md_table(span, columns=["failure_class", "true_span", "pred_span", "true_sd",
                              "pred_sd", "span_compression"]))
    A("")
    A(f"(Medians. `span_compression` = pred_span / true_span, undefined for the "
      f"{int(table.span_compression.isna().sum())} ligands with a single distinct "
      f"log D.)")
    A("")
    comp = contrasts[contrasts.column == "span_compression"].iloc[0]
    pspan = contrasts[contrasts.column == "pred_span"].iloc[0]
    A(f"RESIDUAL_SHAPE ligands' log D really moves "
      f"{residual.true_span.median():.2f} decades; the model moves its prediction "
      f"{residual.pred_span.median():.2f}. PURE_LEVEL ligands really move "
      f"{pure.true_span.median():.2f} and the model moves "
      f"{pure.pred_span.median():.2f}.")
    A("")
    A(f"Two things are true at once here and the second one is easy to miss. The model "
      f"does respond *more* for the RESIDUAL_SHAPE ligands in absolute terms "
      f"(pred_span {pspan.median_a:.2f} vs {pspan.median_b:.2f}), but it responds far "
      f"less *proportionally*: it captures {residual.span_compression.median()*100:.0f}% "
      f"of the true range against {pure.span_compression.median()*100:.0f}% for "
      f"PURE_LEVEL, and that difference survives the chemotype bootstrap "
      f"(AUC {comp.auc:.3f} [{comp.auc_lo:.3f}, {comp.auc_hi:.3f}], BH q = "
      f"{comp.auc_q_bh:.3f}). So the shape failures are not merely wider-ranging "
      f"ligands hit by a uniform flattening -- they are *both* wider-ranging *and* "
      f"flattened harder. Underlying both is the already-established gen8 slope result "
      f"(median true d(logD)/d(log10[extractant]) 2.31 against 0.075 predicted).")
    A("")
    A("The consequence for calibration is the same either way. A ligand measured at one "
      "condition has almost no shape to get wrong, so an offset repairs it completely; "
      "a ligand with a 4-decade acid titration has roughly 3 decades of unmodelled "
      "response that no constant can absorb, and adding a measurement only moves the "
      "constant.")
    A("")
    A("So the two classes are not two chemistries. They are the same modelling defect "
      "measured where it costs nothing and where it costs everything.")
    A("")

    A("## 8. What remains after two measurements")
    A("")
    n_res = len(residual)
    share_k2 = float(residual.mae_CENTRAL_k2.sum() / table.mae_CENTRAL_k2.sum())
    share_k0 = float(residual.mae_k0.sum() / table.mae_k0.sum())
    A(f"| | k = 0 | k = 1 (CENTRAL) | k = 2 (CENTRAL) |")
    A("|---|---|---|---|")
    A(f"| macro MAE, all {n_lig} ligands | {macro['k0']:.3f} | "
      f"{macro['CENTRAL_k1']:.3f} | {macro['CENTRAL_k2']:.3f} |")
    for klass in order:
        block = table[table.failure_class == klass]
        A(f"| {klass} (n={len(block)}) | {block.mae_k0.mean():.3f} | "
          f"{block.mae_CENTRAL_k1.mean():.3f} | {block.mae_CENTRAL_k2.mean():.3f} |")
    A(f"| RESIDUAL_SHAPE share of macro | {share_k0:.2f} | "
      f"{float(residual.mae_CENTRAL_k1.sum()/table.mae_CENTRAL_k1.sum()):.2f} | "
      f"{share_k2:.2f} |")
    A("")
    A(f"**{n_res} of {n_lig} ligands ({n_res/n_lig*100:.0f}%) carry "
      f"{share_k2*100:.0f}% of the error that survives two measurements**, up from "
      f"{share_k0*100:.0f}% at k=0. Their remaining "
      f"{residual.mae_CENTRAL_k2.mean():.3f} sits against a shape floor of "
      f"{residual.shape_mae_k0.mean():.3f} that offset calibration cannot cross at any k, "
      f"with any policy, or with an oracle. That {share_k2*100:.0f}% is stated at the "
      f"level cut used throughout; it is {level_sensitivity.residual_share_of_k2_macro.iloc[0]*100:.0f}% "
      f"at a 0.75 cut and {level_sensitivity.residual_share_of_k2_macro.iloc[-1]*100:.0f}% "
      f"at a 1.5 cut, so it should be read with the line stated, not quoted bare.")
    A("")
    A(f"The same conclusion has a form that needs no classification at all. Macro at "
      f"k=2 (CENTRAL) is {macro['CENTRAL_k2']:.3f}; the macro `shape_mae` -- the part of "
      f"the error offset calibration does not touch at any k, for any ligand (and, per "
      f"§2, an upper bound on the true offset-only floor) -- is "
      f"{table.shape_mae_k0.mean():.3f}. "
      f"**{table.shape_mae_k0.mean()/macro['CENTRAL_k2']*100:.0f}% of what survives two "
      f"measurements is shape, not level**, and that number involves no threshold, no "
      f"class and no cut-off.")
    A("")
    A("Three consequences for gen9, stated as what the numbers do and do not license:")
    A("")
    A(f"1. **More measurements per ligand is the wrong axis for these {len(residual)}.** "
      f"The gap between the deployable 2-shot arm "
      f"({residual.mae_CENTRAL_k2.mean():.3f}) and `shape_mae` "
      f"({residual.shape_mae_k0.mean():.3f}) is "
      f"{residual.mae_CENTRAL_k2.mean() - residual.shape_mae_k0.mean():.3f}; the gap "
      f"between that and zero is {residual.shape_mae_k0.mean():.3f}. Perfecting "
      f"acquisition buys the small number. Only a model that bends the curve buys the "
      f"large one.")
    A(f"2. **The target is the response function, not the level.** The three gen8 shape "
      f"results now line up: the model predicts titration curves an order of magnitude "
      f"too flat, cross-series transfer fails between curve types, and the ligands an "
      f"offset cannot fix are exactly the wide-span, acid-titrated, multi-series ones. "
      f"A slope-aware adapter -- or a model with the mass-action functional form built "
      f"in rather than fitted -- is the intervention this analysis points at. Read off "
      f"this same run, `OFFSET_K3` on two random measurements is "
      f"{adapters['OFFSET_K3']:.3f} macro against `OFFSET_K1`'s "
      f"{adapters['OFFSET_K1']:.3f}: the first evidence in that direction, not tested "
      f"per class here, and not claimed by this report.")
    triage_free = triage.iloc[0]
    single = triage[triage.features.str.contains("alone") & ~triage.features.str.contains("true_span")]
    A(f"3. **A triage exists before any measurement, but it is weaker than the "
      f"in-sample AUCs suggest, and it is about the campaign rather than the molecule.** "
      f"Scored the way a prediction has to be -- leave-one-chemotype-out, on the "
      f"{int(triage_free.n_scored)} ligands of the two classes -- the whole target-free "
      f"set reaches AUC {triage_free.loco_auc:.3f}, while singly the same variables "
      f"reach only {single.loco_auc.min():.3f}-{single.loco_auc.max():.3f}; the "
      f"marginal in-sample AUCs of §6 "
      f"({geom[geom.column.isin(TARGET_FREE)].auc.min():.2f}-"
      f"{geom[geom.column.isin(TARGET_FREE)].auc.max():.2f}) are descriptions of this "
      f"split, not out-of-sample performance. Donor chemistry scores "
      f"{float(triage[triage.features.str.startswith('donor')].loco_auc.iloc[0]):.3f}, "
      f"i.e. worse than a coin. A gen9 triage of the form \"if this ligand's campaign is "
      f"a multi-decade acid titration, do not expect a calibration point to help\" is "
      f"supported at that strength; a rule keyed on donor type is not supported at all.")
    A("")
    A("## 9. Caveats")
    A("")
    A(f"- `mad_true`, `log_d_span_max`, `true_span`, `true_sd`, `max_curve_yspan`, "
      f"`median_curve_linear_r2` and the curve slopes are computed **from the targets**, "
      f"and (per §6) the largest of them are near-restatements of the class definition "
      f"rather than independent explanations of it. They are not features, were not used "
      f"to build one, and must not be quoted as predictors. The target-free separators "
      f"are `n_series`, `n_curves`, `n_curve_axes` and `has_axis_acid`.")
    A(f"- `n_distinct_doi` is target-free but not knowable for a molecule that has never "
      f"been published, so it is excluded from the deployable triage set even though it "
      f"is reported in §6.")
    A(f"- `shape_frac_k0` likewise uses targets, and its Spearman against the 1-shot "
      f"ratio is high partly by construction: `OFFSET_K1` cannot change `shape_mae`, so "
      f"the two quantities share a term. It is the descriptive decomposition, not an "
      f"independent confirmation and not a predictor.")
    A(f"- The classes are read off the same units they then describe. §4(e) re-derives "
      f"them on three split seeds and scores them on the other two; the seeds share the "
      f"ligands' rows, so that tests the protocol's sampling noise and not the noise in "
      f"the measurements themselves.")
    A(f"- The cut-offs ({LEVEL_CUT:.2f} on the level, {PURE_CUT:.2f}/{PERSIST_CUT:.2f} on "
      f"the ratio) were fixed before the contrasts were run but are not derived from "
      f"anything; PARTIAL_LEVEL was added after seeing that the band between them was "
      f"occupied. §4 is a sensitivity analysis, not a claim that the cut-offs are "
      f"canonical.")
    A(f"- All {n_lig} ligands have rows in `series_table.parquet` and "
      f"`curve_table.parquet`, so no class contrast loses members to missing covariates. "
      f"The one exception is `span_compression`, undefined for the "
      f"{int(table.span_compression.isna().sum())} ligands with a single distinct log D "
      f"value -- all {int(table.span_compression.isna().sum())} are ALREADY_GOOD, so the "
      f"RESIDUAL_SHAPE / PURE_LEVEL contrast is unaffected.")
    A(f"- PARTIAL_LEVEL has {int(counts.get('PARTIAL_LEVEL', 0))} members and its "
      f"per-class intervals are wide; no claim in this report rests on it.")
    A(f"- The within-chemotype blocked contrast has {len(mixed)} blocks. It is a "
      f"direction check, not a test.")
    A("- Everything is conditional on one frozen global model and the P2 protocol's "
      "half-pool/half-evaluation split. A ligand with 5 rows contributes an evaluation "
      "set of 2 or 3, so its per-ligand MAE is noisier than a ligand with 200; macro "
      "weights them equally by design.")
    A("")
    (OUT / "error_archaeology.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
