#!/usr/bin/env python
"""Every table the decision report quotes, rendered from the artefacts.

No number in ``DECISION_REPORT.md`` is typed by hand, and every table carries its regime
in its columns: cohort, definition, learner, statistic, band.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_headline_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402

OUT = paths.HEADLINE_DIR


def write(name: str, table: pd.DataFrame, floats: int = 4) -> pd.DataFrame:
    table = table.copy()
    for column in table.select_dtypes(include=[float]).columns:
        table[column] = table[column].round(floats)
    table.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(table.to_markdown(index=False) + "\n")
    return table


def t1_level_leaderboard() -> pd.DataFrame:
    rows = []
    for directory in sorted((paths.METRIC_DIR / "level").glob("*")):
        path = directory / "leaderboard.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        parts = directory.name.split("_")
        table.insert(0, "learner", directory.name.split("_")[-1]
                     if directory.name.endswith(("xgboost", "catboost")) else "extratrees")
        table.insert(0, "definition", "LVL_COND_RESIDUAL"
                     if "COND_RESIDUAL" in directory.name else "LVL_MEAN")
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    keep = ["definition", "learner", "cohort", "arm", "ablation_letter", "level_mae",
            "level_mae_chemotype_macro", "level_signed_bias", "between_extractant_r2",
            "spearman", "dispersion_ratio", "frac_within_1_0", "inner_validation",
            "n_extractants", "n_chemotypes"]
    return write("t1_level_leaderboard", table[[c for c in keep if c in table.columns]])


def t2_primary_contrast() -> pd.DataFrame:
    rows = []
    for path in sorted(paths.BOOTSTRAP_DIR.glob("level_*_pairwise.csv")):
        table = pd.read_csv(path)
        tag = path.stem.replace("level_", "").replace("_pairwise", "")
        table.insert(0, "learner", tag.split("_")[-1]
                     if tag.endswith(("xgboost", "catboost")) else "extratrees")
        table.insert(0, "definition", "LVL_COND_RESIDUAL" if "COND_RESIDUAL" in tag else "LVL_MEAN")
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    keep = ["definition", "learner", "cohort", "statistic", "contrast", "reference",
            "candidate", "reference_value", "candidate_value", "point_delta", "ci95_low",
            "ci95_high", "bca_low", "bca_high", "bca_excludes_zero", "p_two_sided",
            "bootstrap_se", "improvement_direction", "seeds_favouring_candidate",
            "units_improved", "units_total", "bootstrap_blocks"]
    return write("t2_level_contrasts", table[[c for c in keep if c in table.columns]])


def t3_bands() -> pd.DataFrame:
    rows = []
    for path in sorted(paths.BOOTSTRAP_DIR.glob("level_*_by_band.csv")):
        table = pd.read_csv(path)
        table.insert(0, "task", "level")
        rows.append(table)
    path = paths.BOOTSTRAP_DIR / "full_by_band.csv"
    if path.exists():
        table = pd.read_csv(path)
        table.insert(0, "task", "full_logD")
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    keep = ["task", "band", "comparison", "point_delta", "ci95_low", "ci95_high", "bca_low",
            "bca_high", "bca_excludes_zero", "p_two_sided", "units_total", "n_blocks"]
    return write("t3_bands", table[[c for c in keep if c in table.columns]])


def t4_power() -> pd.DataFrame:
    rows = []
    for path in sorted(paths.BOOTSTRAP_DIR.glob("*_power.csv")):
        table = pd.read_csv(path)
        table.insert(0, "family", path.stem.replace("_power", ""))
        rows.append(table)
    return write("t4_power", pd.concat(rows, ignore_index=True))


def t5_multi_arm() -> pd.DataFrame:
    path = paths.BOOTSTRAP_DIR / "level_LVL_MEAN_subgroup.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    signed = [c for c in table.columns if c.startswith("signed_")]
    absolute = [c for c in table.columns if c.startswith("absolute_")]
    long = []
    for _, row in table.drop_duplicates("subgroup").iterrows():
        for column in signed:
            long.append({"subgroup": row["subgroup"], "arm": column[len("signed_"):],
                         "signed_level_error": row[column],
                         "absolute_level_error": row[f"absolute_{column[len('signed_'):]}"],
                         "n_extractants": row["n_extractants"],
                         "n_chemotypes": row["n_chemotypes"],
                         "n_eff_chemotype": row["n_eff_chemotype"]})
    return write("t5_multi_arm_subgroup", pd.DataFrame(long))


def t6_full_leaderboard() -> pd.DataFrame:
    path = paths.METRIC_DIR / "full" / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    keep = ["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae", "pooled_r2",
            "pooled_spearman", "offset_mae", "shape_mae", "frac_within_1_log",
            "inner_validation", "n_extractant_units"]
    return write("t6_full_leaderboard", table[[c for c in keep if c in table.columns]])


def t7_direct_vs_decomposed() -> pd.DataFrame:
    path = paths.BOOTSTRAP_DIR / "full_pairwise.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    keep = ["comparison", "statistic", "reference", "candidate", "point_delta", "ci95_low",
            "ci95_high", "bca_low", "bca_high", "bca_excludes_zero", "p_two_sided",
            "bootstrap_se", "seeds_favouring_candidate", "units_improved", "units_total",
            "bootstrap_blocks"]
    return write("t7_direct_vs_decomposed", table[[c for c in keep if c in table.columns]])


def t8_one_shot() -> pd.DataFrame:
    path = paths.METRIC_DIR / "one_shot" / "one_shot_gain.csv"
    if not path.exists():
        return pd.DataFrame()
    return write("t8_one_shot", pd.read_csv(path))


def t9_importance() -> pd.DataFrame:
    path = paths.METRIC_DIR / "level" / "grouped_permutation_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return write("t9_grouped_importance", pd.read_csv(path))


def t10_influence() -> pd.DataFrame:
    rows = []
    for path in sorted(paths.BOOTSTRAP_DIR.glob("*_influence.csv")):
        table = pd.read_csv(path)
        table.insert(0, "source", path.stem)
        rows.append(table.reindex(table["influence"].abs().sort_values(
            ascending=False).index).head(6))
    if not rows:
        return pd.DataFrame()
    return write("t10_influence", pd.concat(rows, ignore_index=True))


def t11_caveats() -> pd.DataFrame:
    path = paths.BOOTSTRAP_DIR / "statistical_caveats.csv"
    if not path.exists():
        return pd.DataFrame()
    return write("t11_statistical_caveats", pd.read_csv(path))


def main() -> int:
    for name, builder in (("level leaderboard", t1_level_leaderboard),
                          ("level contrasts", t2_primary_contrast),
                          ("bands", t3_bands), ("power", t4_power),
                          ("multi-arm subgroup", t5_multi_arm),
                          ("full leaderboard", t6_full_leaderboard),
                          ("direct vs decomposed", t7_direct_vs_decomposed),
                          ("one shot", t8_one_shot),
                          ("grouped importance", t9_importance),
                          ("influence", t10_influence),
                          ("statistical caveats", t11_caveats)):
        table = builder()
        print(f"{name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
