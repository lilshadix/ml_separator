"""Score gen11 arms against the control, with the decomposition §11 requires.

Reuses gen9's ``curve_shape_table`` / ``summarise_shape`` and gen6's paired unit
bootstrap unchanged, so a gen11 shape number and a gen10 shape number are the
same arithmetic on the same curves.  Nothing here re-implements a metric.

The comparison baseline is **not** the frozen gen10 arm.  Every auxiliary arm runs
under the general metal representation and the annotation-safe MASSACTION block,
because the frozen ones are unusable on auxiliary rows (98.4 % missing) and
provenance-revealing respectively.  Comparing an auxiliary arm to the frozen
control would therefore measure the design change and the transfer together.  The
baseline is the control at the *same* design corner; the frozen arm is reported
alongside so the design change is visible as its own quantity.

Usage::

    PYTHONPATH=src python scripts/gen11_metrics.py --arms runs/gen11_transfer/arms
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen9.metrics import macro_mae  # noqa: E402
from lanthanide_separation.gen9.metrics import curve_shape_table, summarise_shape  # noqa: E402
from lanthanide_separation.gen9.train import load_membership  # noqa: E402
from lanthanide_separation.gen10.runner import prepared_cohort  # noqa: E402
from lanthanide_separation.gen11 import analysis  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen11_transfer" / "metrics"

#: The design corner every auxiliary arm uses; its control is the honest baseline.
BASELINE = "A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1"
#: The bit-for-bit gen10 anchor, reported but never used as the paired reference.
FROZEN_ANCHOR = "A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1"


def load_arms(arms_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(arms_dir.glob("oof_*.parquet")):
        frame = pd.read_parquet(path)
        for name, block in frame.groupby("model"):
            frames[str(name)] = block.reset_index(drop=True)
    return frames


def leaderboard(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        for seed, block in frame.groupby("split_seed"):
            error = (block["log_D"] - block["prediction"]).abs()
            residual = block["prediction"] - block["log_D"]
            level = residual.groupby(block["extractant"]).transform("mean")
            rows.append({
                "model": name, "split_seed": int(seed),
                "macro_mae": macro_mae(block),
                "pooled_mae": float(error.mean()),
                "offset_mae": float(level.abs().groupby(block["extractant"]).first().mean()),
                "shape_mae": float((residual - level).abs().mean()),
            })
    per_seed = pd.DataFrame(rows)
    board = per_seed.groupby("model").agg(
        macro_mae=("macro_mae", "mean"), macro_sd=("macro_mae", "std"),
        pooled_mae=("pooled_mae", "mean"), offset_mae=("offset_mae", "mean"),
        shape_mae=("shape_mae", "mean"), n_seeds=("split_seed", "nunique"),
    ).sort_values("macro_mae").reset_index()
    return per_seed, board


def shape_by_axis(frames: dict[str, pd.DataFrame], membership: pd.DataFrame) -> pd.DataFrame:
    rows, tables = [], []
    for name, frame in frames.items():
        for seed, block in frame.groupby("split_seed"):
            table = curve_shape_table(block.drop_duplicates("row_id"), membership)
            if table.empty:
                continue
            table["model"], table["split_seed"] = name, seed
            tables.append(table)
            summary = summarise_shape(table)
            summary.insert(0, "split_seed", seed)
            summary.insert(0, "model", name)
            rows.append(summary)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    shape = pd.concat(rows, ignore_index=True)
    mean = shape.groupby(["model", "axis_label"]).mean(numeric_only=True).reset_index()
    return mean, pd.concat(tables, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path,
                        default=REPO_ROOT / "runs" / "gen11_transfer" / "arms")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = load_arms(args.arms)
    if not frames:
        raise SystemExit(f"no OOF parquets in {args.arms}")
    print(f"loaded {len(frames)} arms: {sorted(frames)}")

    per_seed, board = leaderboard(frames)
    per_seed.to_csv(args.out / "scores_by_seed.csv", index=False)
    board.to_csv(args.out / "leaderboard.csv", index=False)
    print("\n=== leaderboard (macro MAE, lower is better) ===")
    print(board.round(4).to_string(index=False))

    membership = load_membership()
    shape_mean, curves = shape_by_axis(frames, membership)
    if not shape_mean.empty:
        shape_mean.to_csv(args.out / "shape_by_axis.csv", index=False)
        curves.to_parquet(args.out / "curve_shape.parquet", index=False)

    baseline = args.baseline if args.baseline in frames else sorted(frames)[0]
    candidates = [n for n in frames if n != baseline]
    if candidates:
        boot = analysis.compare(frames, reference=baseline, candidates=candidates,
                                block="tanimoto_cluster", replicates=args.replicates)
        boot.to_csv(args.out / "paired_bootstrap_chemotype.csv", index=False)
        print(f"\n=== paired bootstrap vs {baseline} (chemotype-blocked) ===")
        show = [c for c in ("comparison", "statistic", "mean", "ci_low", "ci_high",
                            "p_value", "n_units") if c in boot.columns]
        print(boot[show].round(4).to_string(index=False) if show
              else boot.round(4).to_string(index=False))

        directions, decomps, moved = [], [], {}
        aux_extractants = _aux_extractants()
        for candidate in candidates:
            per = analysis.per_seed_direction(frames, reference=baseline, candidate=candidate)
            per.insert(0, "candidate", candidate)
            directions.append(per)
            dec = analysis.decompose(frames, reference=baseline, candidate=candidate,
                                     aux_extractants=aux_extractants)
            dec.insert(0, "candidate", candidate)
            decomps.append(dec)
            moved[candidate] = analysis.ligands_moved(frames, reference=baseline,
                                                      candidate=candidate)
        pd.concat(directions, ignore_index=True).to_csv(
            args.out / "per_seed_direction.csv", index=False)
        pd.concat(decomps, ignore_index=True).to_csv(
            args.out / "decomposition.csv", index=False)
        (args.out / "ligands_moved.json").write_text(json.dumps(moved, indent=2, default=float))

    (args.out / "baseline.json").write_text(json.dumps(
        {"baseline": baseline, "frozen_anchor": FROZEN_ANCHOR,
         "arms": sorted(frames)}, indent=2))
    return 0


def _aux_extractants() -> list[str]:
    path = REPO_ROOT / "runs" / "gen11_transfer" / "featurizer" / "aux_features.parquet"
    if not path.exists():
        return []
    return sorted(set(pd.read_parquet(
        path, columns=["extractant_primary_smiles"])["extractant_primary_smiles"].dropna()))


if __name__ == "__main__":
    raise SystemExit(main())
