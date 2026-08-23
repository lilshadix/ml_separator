"""Where should the one measurement be taken? (brief §17) and does it transfer? (§18)

Both questions are answered exhaustively rather than by sampling, because under
offset-only calibration the one-shot score has a closed form.  If the frozen model's
residual on a held-out ligand is ``r = y - yhat``, then measuring row *i* and applying
the mean-residual correction gives

.. code-block:: text

    MAE(i) = mean_{j != i} | r_j - r_i |

so *every* candidate row can be scored, for every ligand, for free.  There is no draw
to average over and no sampling noise: the number below is the exact expected error of
choosing that row.

Two analyses come out of it.

**Geography (§17)** — tag each candidate row by *where it sits* (low / middle / high of
the ligand's acid range, central vs extreme lanthanide, low / middle / high extractant
concentration, interior vs endpoint of its titration curve) and report the mean one-shot
MAE per stratum.  This produces an experimental recommendation that needs no model at
deployment time: "measure here".

**Cross-series transfer (§18)** — restrict the *scored* rows to one curve type and the
*measured* row to another, giving a calibration-type x target-type matrix.  If one acid
measurement only fixes the acid series, the deployment advice is narrow; if it fixes the
level and helps every series, it is powerful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.report import md_table  # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
OOF = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
MEMBERSHIP = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "cross_series"

ACID = "massact__log10_cond__acid_concentration_M"
EXTR = "massact__log10_cond__extractant_concentration_M"


def _tercile(values: np.ndarray) -> np.ndarray:
    """low / mid / high by rank within the ligand.  Constant column -> 'single'."""
    out = np.full(len(values), "single", dtype=object)
    finite = np.isfinite(values)
    if finite.sum() < 3 or np.ptp(values[finite]) <= 0:
        return out
    ranks = pd.Series(values).rank(pct=True, method="average").to_numpy()
    out[finite & (ranks <= 1 / 3)] = "low"
    out[finite & (ranks > 1 / 3) & (ranks <= 2 / 3)] = "mid"
    out[finite & (ranks > 2 / 3)] = "high"
    out[~finite] = "missing"
    return out


def _extremity(values: np.ndarray) -> np.ndarray:
    """central vs extreme by absolute deviation from the ligand's own median."""
    out = np.full(len(values), "single", dtype=object)
    finite = np.isfinite(values)
    if finite.sum() < 3 or np.ptp(values[finite]) <= 0:
        return out
    deviation = np.abs(values - np.nanmedian(values[finite]))
    cut = np.nanmedian(deviation[finite])
    out[finite & (deviation <= cut)] = "central"
    out[finite & (deviation > cut)] = "extreme"
    out[~finite] = "missing"
    return out


def build_scores(oof: pd.DataFrame, cohort: pd.DataFrame, membership: pd.DataFrame,
                 *, min_rows: int) -> pd.DataFrame:
    """One row per (model, seed, ligand, candidate row): its exact 1-shot MAE and its strata."""
    axis_columns = [c for c in (ACID, EXTR, "lanthanide_index", "cond__temperature_C")
                    if c in cohort.columns]
    work = oof.merge(cohort[["row_id"] + axis_columns], on="row_id", how="left",
                     validate="many_to_one")
    assert len(work) == len(oof), "axis join changed the row count"

    # A row's *primary* curve axis: the axis of the longest curve it belongs to.  A row
    # at the crossing of an acid titration and a metal series belongs to both; the
    # transfer matrix below uses the full membership, this column only labels the row.
    if membership is not None and len(membership):
        best = membership.sort_values("n_points", ascending=False).drop_duplicates("row_id")
        primary = best.set_index("row_id")["axis_label"]
        work["primary_axis"] = work["row_id"].map(primary).fillna("none")
    else:
        work["primary_axis"] = "none"

    records: list[pd.DataFrame] = []
    for (model, seed), frame in work.groupby(["model", "split_seed"], sort=True):
        for ligand, block in frame.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < min_rows:
                continue
            residual = block["log_D"].to_numpy(float) - block["prediction"].to_numpy(float)
            diff = np.abs(residual[:, None] - residual[None, :])
            mae = diff.sum(axis=1) / (n - 1)
            acid = block[ACID].to_numpy(float) if ACID in block.columns else np.full(n, np.nan)
            extr = block[EXTR].to_numpy(float) if EXTR in block.columns else np.full(n, np.nan)
            metal = block["lanthanide_index"].to_numpy(float) if "lanthanide_index" in block.columns \
                else np.full(n, np.nan)
            prediction = block["prediction"].to_numpy(float)
            records.append(pd.DataFrame({
                "model": model, "split_seed": int(seed), "extractant": ligand,
                "tanimoto_cluster": block["tanimoto_cluster"], "row_id": block["row_id"],
                "n_rows": n, "one_shot_mae": mae,
                "zero_shot_mae": float(np.abs(residual).mean()),
                "oracle_level_mae": float(np.abs(residual - np.median(residual)).mean()),
                "acid_position": _tercile(acid), "extractant_position": _tercile(extr),
                "metal_position": _extremity(metal), "prediction_position": _tercile(prediction),
                "residual_position": _extremity(residual),   # diagnostic only, not deployable
                "primary_axis": block["primary_axis"],
            }))
    return pd.concat(records, ignore_index=True)


def stratum_table(scores: pd.DataFrame, column: str) -> pd.DataFrame:
    """Macro mean (one ligand, one vote) of the one-shot MAE per stratum."""
    per_ligand = scores.groupby([column, "extractant"])["one_shot_mae"].mean().reset_index()
    out = per_ligand.groupby(column).agg(
        one_shot_mae=("one_shot_mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
    rows = scores.groupby(column).size().rename("n_candidate_rows")
    return out.merge(rows, on=column).sort_values("one_shot_mae")


def transfer_matrix(oof: pd.DataFrame, cohort: pd.DataFrame, membership: pd.DataFrame,
                    *, min_rows: int) -> pd.DataFrame:
    """Measure on a row of curve type A, score only rows of curve type B."""
    if membership is None or not len(membership):
        return pd.DataFrame()
    label = membership.sort_values("n_points", ascending=False).drop_duplicates("row_id")
    label = label.set_index("row_id")["axis_label"]
    records: list[dict] = []
    for (model, seed), frame in oof.groupby(["model", "split_seed"], sort=True):
        for ligand, block in frame.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < min_rows:
                continue
            residual = block["log_D"].to_numpy(float) - block["prediction"].to_numpy(float)
            axis = block["row_id"].map(label).fillna("none").to_numpy()
            diff = np.abs(residual[:, None] - residual[None, :])
            np.fill_diagonal(diff, np.nan)
            for source in np.unique(axis):
                source_mask = axis == source
                for target in np.unique(axis):
                    target_mask = axis == target
                    sub = diff[np.ix_(source_mask, target_mask)]
                    if not np.isfinite(sub).any():
                        continue
                    records.append({
                        "model": model, "split_seed": int(seed), "extractant": ligand,
                        "calibration_axis": source, "target_axis": target,
                        "mae": float(np.nanmean(sub)),
                        "zero_shot": float(np.abs(residual[target_mask]).mean()),
                        "n_calibration_rows": int(source_mask.sum()),
                        "n_target_rows": int(target_mask.sum())})
    return pd.DataFrame(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=OOF)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--membership", type=Path, default=MEMBERSHIP)
    parser.add_argument("--model", default="REC_ecfp_plus_recovered")
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(args.oof)
    oof = oof[oof["model"] == args.model]
    if oof.empty:
        raise SystemExit(f"model {args.model!r} not in {args.oof}")
    cohort = pd.read_parquet(args.cohort)
    membership = pd.read_parquet(args.membership) if args.membership.exists() else None

    scores = build_scores(oof, cohort, membership, min_rows=args.min_rows)
    scores.to_parquet(args.out / "one_shot_candidate_scores.parquet", index=False)

    strata = {}
    for column in ("acid_position", "extractant_position", "metal_position",
                   "prediction_position", "primary_axis", "residual_position"):
        table = stratum_table(scores, column)
        strata[column] = table
        table.to_csv(args.out / f"stratum_{column}.csv", index=False)

    transfer = transfer_matrix(oof, cohort, membership, min_rows=args.min_rows)
    transfer.to_parquet(args.out / "transfer_detail.parquet", index=False)
    if not transfer.empty:
        per_ligand = transfer.groupby(
            ["calibration_axis", "target_axis", "extractant"])[["mae", "zero_shot"]].mean().reset_index()
        matrix = per_ligand.groupby(["calibration_axis", "target_axis"]).agg(
            mae=("mae", "mean"), zero_shot=("zero_shot", "mean"),
            n_ligands=("extractant", "nunique")).reset_index()
        matrix["gain"] = matrix["zero_shot"] - matrix["mae"]
        matrix.to_csv(args.out / "transfer_matrix.csv", index=False)
    else:
        matrix = pd.DataFrame()

    lines = ["# Calibration geography — where to spend the one measurement", "",
             f"*Model `{args.model}`, 5 split seeds, "
             f"{scores['extractant'].nunique()} held-out ligands, "
             f"{len(scores):,} candidate rows, each scored exactly (no sampling).*", ""]
    reference = scores.groupby("extractant")[["zero_shot_mae", "one_shot_mae", "oracle_level_mae"]].mean().mean()
    lines += [f"Reference on this population: zero-shot **{reference['zero_shot_mae']:.3f}**, "
              f"one-shot averaged over every candidate (= RANDOM) **{reference['one_shot_mae']:.3f}**, "
              f"oracle level **{reference['oracle_level_mae']:.3f}**.", ""]
    titles = {
        "acid_position": "Where in the acid range the measurement sits",
        "extractant_position": "Where in the extractant-concentration range it sits",
        "metal_position": "Central vs extreme lanthanide",
        "prediction_position": "Where in the model's own predicted range it sits",
        "primary_axis": "Which kind of series the measured point belongs to",
        "residual_position": "DIAGNOSTIC ONLY (not deployable): central vs extreme residual",
    }
    for column, table in strata.items():
        lines += [f"## {titles[column]}", "", md_table(table), ""]
    if not matrix.empty:
        lines += ["## Cross-series transfer matrix", "",
                  "Rows: the curve type the *measured* point belongs to. Columns: the curve type "
                  "of the rows being *predicted*. Values are one-shot MAE (lower is better); "
                  "`gain` is the improvement over zero-shot on the same target rows.", "",
                  md_table(matrix), ""]
    (args.out / "calibration_geography.md").write_text("\n".join(lines) + "\n")

    pd.set_option("display.width", 220)
    for column, table in strata.items():
        print(f"\n--- {column}"); print(table.to_string(index=False))
    if not matrix.empty:
        print("\n--- transfer matrix"); print(matrix.to_string(index=False))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
