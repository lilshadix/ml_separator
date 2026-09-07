#!/usr/bin/env python
"""Every table the decision report quotes, rendered from the artefacts.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_headline_tables.py

Writes ``headline_tables/*.csv`` and a markdown rendering of each, so no number
in ``DECISION_REPORT.md`` is typed by hand and every one carries its regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import paths  # noqa: E402

OUT = paths.HEADLINE_DIR


def _write(name: str, table: pd.DataFrame, floats: int = 4) -> pd.DataFrame:
    table = table.copy()
    for column in table.select_dtypes(include=[float]).columns:
        table[column] = table[column].round(floats)
    table.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(table.to_markdown(index=False) + "\n")
    return table


def t1_zero_shot() -> pd.DataFrame:
    rows = []
    for design in ("B", "A"):
        path = paths.METRIC_DIR / design / "leaderboard.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "design", design)
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    keep = ["design", "arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
            "pooled_rmse", "median_abs_error", "pooled_r2", "pooled_spearman",
            "calibration_slope", "offset_mae", "shape_mae", "frac_within_1_log",
            "inner_validation", "n_extractant_units", "n_chemotypes"]
    return _write("t1_zero_shot_leaderboard", table[[c for c in keep if c in table.columns]])


def t2_bands() -> pd.DataFrame:
    rows = []
    for design in ("B", "A"):
        path = paths.METRIC_DIR / design / "bands.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "design", design)
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    keep = ["design", "arm", "band", "macro_mae_extractant", "macro_mae_chemotype",
            "pooled_mae", "offset_mae", "shape_mae", "n_rows", "n_extractant_units",
            "n_chemotypes"]
    return _write("t2_bands", table[[c for c in keep if c in table.columns]])


def t3_pairwise() -> pd.DataFrame:
    rows = []
    for name, path in (("zero_shot_chemotype_blocked", paths.BOOTSTRAP_DIR / "B_pairwise.csv"),
                       ("zero_shot_doi_blocked", paths.BOOTSTRAP_DIR / "B_pairwise_doi_blocked.csv"),
                       ("zero_shot_by_band", paths.BOOTSTRAP_DIR / "B_pairwise_by_band.csv"),
                       ("ablation", paths.BOOTSTRAP_DIR / "B_ablation_pairwise.csv"),
                       ("multi_lanthanide", paths.BOOTSTRAP_DIR / "B_multiln_pairwise.csv"),
                       ("few_shot", paths.BOOTSTRAP_DIR / "B_fewshot_pairwise.csv")):
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "family", name)
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    keep = ["family", "statistic", "band", "k", "reference", "candidate", "point_delta",
            "ci95_low", "ci95_high", "bca_low", "bca_high", "bca_excludes_zero",
            "bootstrap_se", "units_improved", "units_total", "bootstrap_blocks"]
    return _write("t3_pairwise_comparisons", table[[c for c in keep if c in table.columns]])


def t4_power() -> pd.DataFrame:
    rows = []
    for name, path in (("zero_shot", paths.BOOTSTRAP_DIR / "B_power.csv"),
                       ("multi_lanthanide", paths.BOOTSTRAP_DIR / "B_multiln_power.csv")):
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "family", name)
        rows.append(table)
    return _write("t4_power", pd.concat(rows, ignore_index=True))


def t5_fewshot() -> pd.DataFrame:
    path = paths.METRIC_DIR / "B" / "fewshot_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    keep = ["cohort", "arm", "adapter", "k", "mae", "offset_abs", "shape_mae",
            "mae_sd_over_seeds", "n_extractants", "n_seeds"]
    return _write("t5_fewshot_curve", table[keep])


def t6_multiln() -> pd.DataFrame:
    path = paths.METRIC_DIR / "B_multiln" / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    keep = ["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
            "offset_mae", "shape_mae", "n_extractant_units"]
    return _write("t6_multi_lanthanide", table[keep])


def t7_ablation() -> pd.DataFrame:
    path = paths.METRIC_DIR / "B_ablation" / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    keep = ["arm", "macro_mae_extractant", "macro_mae_chemotype", "offset_mae", "shape_mae"]
    return _write("t7_ablation", table[keep])


def t8_influence() -> pd.DataFrame:
    path = paths.BOOTSTRAP_DIR / "B_influence.csv"
    if not path.exists():
        return pd.DataFrame()
    return _write("t8_influence", pd.read_csv(path).head(12))


def main() -> int:
    for name, builder in (("zero-shot leaderboard", t1_zero_shot), ("bands", t2_bands),
                          ("pairwise", t3_pairwise), ("power", t4_power),
                          ("few-shot", t5_fewshot), ("multi-lanthanide", t6_multiln),
                          ("ablation", t7_ablation), ("influence", t8_influence)):
        table = builder()
        print(f"{name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
