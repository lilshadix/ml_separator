#!/usr/bin/env python
"""Phase 1: choose the level definition, on training rows only, before any model.

Reads ``config/level_definition_rules.json`` — written and frozen first — and applies
its decision rule to the development fold's training rows.  Writes
``analysis/level_definition_study.md`` and ``manifests/level_definition_choice.json``.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_level_definition_study.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import levels, paths  # noqa: E402
from gen12eu import splits  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402


def stable_half(row_id: str, extractant: str) -> int:
    """Deterministic 0/1 half-assignment of a cell.  Independent of process and order."""
    digest = hashlib.blake2b(f"{extractant}|{row_id}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") % 2


def alpha_of(frame: pd.DataFrame, definition: str, blocks: dict, seed: int) -> pd.Series:
    """The definition applied to one block of rows, with the block acting as its own
    training set.  Used only for the reliability diagnostics, never for a model."""
    targets = levels.build_level_targets(frame, frame, blocks, definition=definition, seed=seed)
    return targets.train


def main() -> int:
    rules = json.loads((paths.CONFIG_DIR / "level_definition_rules.json").read_text())
    development = rules["development_fold"]
    cohort = build_cohort()
    frame = cohort.frame
    folds = [f for f in splits.all_folds(frame, design=development["design"])
             if f.seed == development["split_seed"] and f.fold == development["fold"]]
    if len(folds) != 1:
        raise SystemExit("development fold not found")
    fold = folds[0]
    train = frame.iloc[fold.train_index].reset_index(drop=True)
    test = frame.iloc[fold.test_index].reset_index(drop=True)
    seed = fold.model_seed
    report: list[str] = []
    out: dict = {"rules_digest": paths.sha256_of(paths.CONFIG_DIR / "level_definition_rules.json"),
                 "development_fold": development,
                 "n_train_rows": int(len(train)),
                 "n_train_extractants": int(train["extractant"].nunique()),
                 "n_train_chemotypes": int(train["chemotype"].nunique())}

    # ---- the five definitions on the development-fold training rows --------- #
    alphas: dict[str, pd.Series] = {}
    diagnostics: dict[str, dict] = {}
    for definition in levels.DEFINITIONS:
        targets = levels.build_level_targets(train, test, cohort.blocks, definition=definition,
                                             seed=seed, inner_seed=seed)
        alphas[definition] = targets.train
        diagnostics[definition] = targets.diagnostics
    table = pd.DataFrame(alphas)
    counts = train["extractant"].astype(str).value_counts().reindex(table.index)
    table["n_cells"] = counts.to_numpy()

    summary = pd.DataFrame({
        "mean": table[list(levels.DEFINITIONS)].mean(),
        "sd": table[list(levels.DEFINITIONS)].std(),
        "min": table[list(levels.DEFINITIONS)].min(),
        "max": table[list(levels.DEFINITIONS)].max(),
        "defined_for_all": table[list(levels.DEFINITIONS)].notna().all(),
        "defined_for_one_cell": table.loc[table["n_cells"] == 1,
                                          list(levels.DEFINITIONS)].notna().all(),
    })
    out["summary"] = summary.to_dict(orient="index")

    correlation = table[list(levels.DEFINITIONS)].corr(method="spearman")
    out["spearman_between_definitions"] = correlation.round(4).to_dict()

    # ---- criterion 2: split-half reliability -------------------------------- #
    eligible = table.index[table["n_cells"] >= 4]
    halves = train.assign(half=[stable_half(r, e) for r, e in
                                zip(train["row_id"], train["extractant"])])
    reliability: list[dict] = []
    for definition in levels.DEFINITIONS:
        values = {}
        for side in (0, 1):
            block = halves[halves["half"] == side]
            keep = block["extractant"].value_counts()
            keep = keep[keep >= 2].index
            block = block[block["extractant"].isin(keep)].reset_index(drop=True)
            if block.empty:
                continue
            values[side] = alpha_of(block, definition, cohort.blocks, seed)
        if len(values) < 2:
            continue
        shared = sorted(set(values[0].index) & set(values[1].index) & set(eligible))
        a, b = values[0].reindex(shared), values[1].reindex(shared)
        reliability.append({
            "definition": definition, "n_extractants": len(shared),
            "spearman": float(a.corr(b, method="spearman")),
            "pearson": float(a.corr(b, method="pearson")),
            "mean_abs_half_difference": float((a - b).abs().mean()),
            "median_abs_half_difference": float((a - b).abs().median()),
        })
    reliability = pd.DataFrame(reliability)
    out["split_half_reliability"] = reliability.to_dict(orient="records")

    # ---- criterion 3: condition-sampling distortion ------------------------- #
    shift = (table["LVL_MEAN"] - table["LVL_COND_RESIDUAL"]).abs()
    distortion = {
        "n_extractants": int(len(shift)),
        "median_abs_shift": float(shift.median()),
        "mean_abs_shift": float(shift.mean()),
        "frac_above_0_5_decades": float((shift > 0.5).mean()),
        "frac_above_1_0_decades": float((shift > 1.0).mean()),
        "spearman_mean_vs_residual": float(
            table["LVL_MEAN"].corr(table["LVL_COND_RESIDUAL"], method="spearman")),
        "shift_vs_n_cells_spearman": float(
            pd.Series(shift.to_numpy()).corr(pd.Series(table["n_cells"].to_numpy()),
                                             method="spearman")),
    }
    out["condition_sampling_distortion"] = distortion

    # ---- addendum 1: is the adjustment removing level rather than conditions? --- #
    g_train, _ = levels.condition_only_predictions(train, test, cohort.blocks, seed=seed,
                                                   inner_seed=seed)
    extractants = train["extractant"].astype(str)
    y_train = train[levels.TARGET].to_numpy(dtype=float)
    mean_y = pd.Series(y_train).groupby(extractants.to_numpy()).mean()
    mean_g = pd.Series(g_train).groupby(extractants.to_numpy()).mean()
    absorption = {
        "pooled_r2_of_condition_model": float(
            1.0 - ((y_train - g_train) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()),
        "between_variance_of_level": float(mean_y.var()),
        "between_variance_of_condition_prediction": float(mean_g.var()),
        "between_variance_share": float(mean_g.var() / mean_y.var()),
        "spearman_level_vs_condition_prediction": float(mean_y.corr(mean_g, method="spearman")),
    }
    counts_by_extractant = extractants.value_counts()
    shift_by_n = []
    for low, high, label in ((1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"),
                             (7, 20, "7-20"), (21, 10 ** 6, "21+")):
        keep = counts_by_extractant[(counts_by_extractant >= low)
                                    & (counts_by_extractant <= high)].index
        block = shift.reindex(sorted(set(keep) & set(shift.index)))
        if len(block):
            shift_by_n.append({"n_cells": label, "n_extractants": int(len(block)),
                               "median_abs_shift": float(block.median())})
    absorption["shift_by_cell_count"] = shift_by_n
    out["condition_model_absorption"] = absorption

    # ---- the level-reliable cohort threshold -------------------------------- #
    stats = levels.shrinkage_ratio(train, train[levels.TARGET].to_numpy(dtype=float))
    tau = float(np.sqrt(stats["tau2"]))
    sigma = float(np.sqrt(stats["sigma2"]))
    rule = rules["level_reliable_cohort_rule"]
    threshold = None
    ladder = []
    for n in range(1, 21):
        se = sigma / np.sqrt(n)
        ladder.append({"n_cells": n, "se_of_mean": round(se, 4),
                       "se_over_tau": round(se / tau, 4)})
        if threshold is None and se <= 0.25 * tau:
            threshold = n
    threshold = int(min(max(threshold or 20, 2), 7))
    out["level_reliable_cohort"] = {
        "sigma_within": sigma, "tau_between": tau, "ladder": ladder,
        "threshold_cells": threshold, "rule": rule["rule"],
    }

    # ---- apply the frozen decision rule ------------------------------------- #
    best = reliability.sort_values("spearman", ascending=False).iloc[0]
    by_definition = reliability.set_index("definition")
    mean_row = by_definition.loc["LVL_MEAN"]
    residual_row = by_definition.loc["LVL_COND_RESIDUAL"]
    override_1 = bool(best["spearman"] - mean_row["spearman"] > 0.05)
    shift_large = bool(distortion["frac_above_0_5_decades"] > 0.10)
    # Addendum 1: the adjustment must not itself be removing level.  See
    # config/level_definition_rules.json -> addendum_1 for the evidence and the date.
    adjustment_is_clean = bool(absorption["between_variance_share"] < 0.10)
    reliability_holds = bool(mean_row["spearman"] - residual_row["spearman"] <= 0.02)
    override_2 = bool(shift_large and adjustment_is_clean and reliability_holds)
    if override_2:
        primary, why = "LVL_COND_RESIDUAL", "override_2 (amended): condition-sampling distortion"
    elif override_1:
        primary, why = str(best["definition"]), "override_1: split-half reliability"
    else:
        primary, why = "LVL_MEAN", "default: no override triggered under the amended rule"
    secondary = "LVL_COND_RESIDUAL" if primary == "LVL_MEAN" else "LVL_MEAN"
    out["decision"] = {"primary_level_definition": primary, "reason": why,
                       "secondary_level_definition": secondary,
                       "override_1_triggered": override_1,
                       "override_2_shift_large": shift_large,
                       "override_2_adjustment_is_clean": adjustment_is_clean,
                       "override_2_reliability_holds": reliability_holds,
                       "override_2_triggered": override_2,
                       "original_rule_would_have_chosen":
                           "LVL_COND_RESIDUAL" if shift_large else "LVL_MEAN",
                       "best_reliability_definition": str(best["definition"]),
                       "best_reliability_spearman": float(best["spearman"]),
                       "lvl_mean_reliability_spearman": float(mean_row["spearman"])}

    (paths.MANIFEST_DIR / "level_definition_choice.json").write_text(
        json.dumps(out, indent=1, default=float))

    report.append("# Level-definition study — training rows only\n")
    report.append(f"Development fold: design {development['design']}, split seed "
                  f"{development['split_seed']}, fold {development['fold']}. "
                  f"{out['n_train_rows']} training rows, {out['n_train_extractants']} training "
                  f"extractants, {out['n_train_chemotypes']} training chemotypes. No held-out "
                  "row and no model output entered any number below.\n")
    report.append("## The five candidate definitions\n")
    report.append(summary.round(4).to_markdown() + "\n")
    report.append("## Agreement between definitions (Spearman)\n")
    report.append(correlation.round(3).to_markdown() + "\n")
    report.append("## Criterion 2 — split-half reliability "
                  "(extractants with at least four cells)\n")
    report.append(reliability.round(4).to_markdown(index=False) + "\n")
    report.append("## Criterion 3 — condition-sampling distortion\n")
    report.append(pd.Series(distortion).to_frame("value").round(4).to_markdown() + "\n")
    report.append("### Is the adjustment removing conditions, or removing level?\n")
    report.append(pd.Series({k: v for k, v in absorption.items()
                             if k != "shift_by_cell_count"}).to_frame("value")
                  .round(4).to_markdown() + "\n")
    report.append(pd.DataFrame(absorption["shift_by_cell_count"]).to_markdown(index=False) + "\n")
    report.append("## Level-reliable cohort threshold\n")
    report.append(pd.DataFrame(ladder).head(10).to_markdown(index=False) + "\n")
    report.append(f"\nsigma_within = {sigma:.4f}, tau_between = {tau:.4f}, "
                  f"threshold = **{threshold} cells**.\n")
    report.append("## Decision\n")
    report.append(pd.Series(out["decision"]).to_frame("value").to_markdown() + "\n")
    (paths.ANALYSIS_DIR / "level_definition_study.md").write_text("\n".join(report))

    print(summary.round(4).to_string())
    print("\nsplit-half reliability:")
    print(reliability.round(4).to_string(index=False))
    print("\ncondition-sampling distortion:", json.dumps(distortion, indent=1))
    print(f"\nsigma_within {sigma:.4f}  tau_between {tau:.4f}  level-reliable threshold {threshold}")
    print("\nDECISION:", json.dumps(out["decision"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
