"""Emit the report's markdown tables from the artifact CSVs, so no number is retyped.

Appends to (or prints) the leaderboard, ablation, ensemble, k-shot and error-mechanism
tables. Every value comes from a file written by a run; nothing here recomputes a
metric, so the report and the artifacts cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import leaderboard  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen7_architecture"


def md_table(frame: pd.DataFrame, columns, headers=None, digits=4) -> str:
    columns = [c for c in columns if c in frame.columns]
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            cells.append(f"{v:.{digits}f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title: str, body: str) -> str:
    return f"\n### {title}\n\n{body}\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append-to", type=Path, default=None)
    args = parser.parse_args(argv)
    parts: list[str] = ["\n## COMPLETE LEADERBOARD\n"]

    board_path = OUT / "leaderboard_all.csv"
    if board_path.exists():
        board = pd.read_csv(board_path)
        parts.append(section(
            "Every arm, every suite, identical test rows",
            md_table(board, ["model", "suite", "macro_mae", "macro_mae_sd", "offset_mae",
                             "shape_mae", "shape_r2", "pooled_mae",
                             "nn_lt_0_4__macro_mae", "n_seeds"],
                     ["model", "suite", "macro MAE", "sd", "offset", "shape", "shape R²",
                      "pooled", "nn<0.4 macro", "seeds"])))

    finalists = OUT / "finalists" / "scores_by_seed.csv"
    if finalists.exists():
        board = leaderboard(pd.read_csv(finalists))
        parts.append(section(
            "Finalists — five seeds, identical folds",
            md_table(board, ["model", "macro_mae", "macro_mae_sd", "offset_mae", "shape_mae",
                             "shape_r2", "pooled_mae", "nn_lt_0_4__macro_mae"],
                     ["model", "macro MAE", "sd", "offset", "shape", "shape R²", "pooled",
                      "nn<0.4 macro"])))

    ablation = OUT / "ablations" / "leaderboard.csv"
    bootstrap = OUT / "ablations" / "ablation_bootstrap.csv"
    if ablation.exists():
        parts.append(section("Ablation matrix — leave-one-component-out",
                             md_table(pd.read_csv(ablation),
                                      ["model", "macro_mae", "macro_mae_sd", "offset_mae",
                                       "shape_mae"],
                                      ["arm", "macro MAE", "sd", "offset", "shape"])))
    if bootstrap.exists():
        table = pd.read_csv(bootstrap)
        table = table[table["statistic"] == "mae"]
        parts.append(section(
            "Ablation deltas vs the full model (positive = the component helped)",
            md_table(table, ["candidate", "point_delta", "ci95_low", "ci95_high",
                             "block_macro_delta", "units_improved", "units_total"],
                     ["ablated arm", "delta", "CI low", "CI high", "block macro",
                      "units better", "units"])))

    ensemble = OUT / "ensemble" / "ensemble_leaderboard.csv"
    if ensemble.exists():
        parts.append(section("Ensemble and its members",
                             md_table(pd.read_csv(ensemble),
                                      ["arm", "macro_mae", "offset_mae", "shape_mae",
                                       "pooled_mae", "nn_lt_0_4__macro_mae"],
                                      ["arm", "macro MAE", "offset", "shape", "pooled",
                                       "nn<0.4 macro"])))
    correlations = OUT / "ensemble" / "residual_correlations.csv"
    if correlations.exists():
        table = pd.read_csv(correlations, index_col=0)
        numeric = table.select_dtypes("number").drop(columns=["split_seed"], errors="ignore")
        parts.append(section("Mean residual correlation between ensemble members",
                             numeric.groupby(level=0).mean().round(3).to_markdown()))

    kshot = OUT / "kshot" / "kshot_summary.csv"
    if kshot.exists():
        parts.append(section(
            "Few-shot calibration on a new chemotype (per-ligand mean MAE)",
            md_table(pd.read_csv(kshot),
                     ["model", "k", "zero_shot", "offset_only", "no_model", "affine",
                      "n_ligands"],
                     ["model", "k", "zero-shot", "offset-only", "no-model null", "affine",
                      "ligands"])))

    mechanisms = OUT / "error_analysis" / "mechanism_summary.json"
    if mechanisms.exists():
        payload = json.loads(mechanisms.read_text())
        rows = []
        for model, entry in payload.items():
            for mechanism, count in entry.get("worst_mechanism_counts", {}).items():
                rows.append({"model": model, "mechanism": mechanism, "worst_20_count": count,
                             "error_share": entry["worst_mechanism_error_share"].get(mechanism)})
        if rows:
            parts.append(section("Failure mechanism of the twenty worst held-out ligands",
                                 md_table(pd.DataFrame(rows),
                                          ["model", "mechanism", "worst_20_count", "error_share"],
                                          ["model", "mechanism", "count in worst 20",
                                           "share of their error"], digits=3)))

    ceiling = OUT / "ceiling" / "ceiling.json"
    if ceiling.exists():
        payload = json.loads(ceiling.read_text())
        rows = [{"grouping": k, "R2_of_log_D": v}
                for k, v in payload.get("variance_explained_by_grouping", {}).items()
                if v is not None]
        parts.append(section("Variance of `log D` explained by each grouping",
                             md_table(pd.DataFrame(rows), ["grouping", "R2_of_log_D"],
                                      ["grouping", "R² of log D"], digits=3)))
        parts.append(section("Batch-offset variance components and the offset floor",
                             "```json\n" + json.dumps(
                                 {"batch_offset": payload.get("batch_offset"),
                                  "offset_mae_floor": payload.get("offset_mae_floor")},
                                 indent=2) + "\n```"))

    text = "\n".join(parts)
    if args.append_to:
        with args.append_to.open("a") as handle:
            handle.write(text)
        print(f"appended {len(text)} chars to {args.append_to}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
