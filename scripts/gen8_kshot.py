"""Run the gen8 k-shot protocols on frozen out-of-fold predictions (brief §7-§9, §16-§18).

The global model is never refitted here: it was fitted once, under the gen6/gen7
chemotype-blocked fold plan, and its out-of-fold predictions for held-out ligands
are the input.  That is what makes every number in this study *paired* — the same
model, the same rows, the same folds, differing only in how many measurements the
calibration was given and which ones.

    .venv/bin/python scripts/gen8_kshot.py --protocol p1
    .venv/bin/python scripts/gen8_kshot.py --protocol p2 --repeats 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.kshot import DEFAULT_RIDGE, POLICY_AXES  # noqa: E402
from lanthanide_separation.gen8.protocols import (  # noqa: E402
    K_VALUES, MODES, POOL_CAP, evaluate_p1, evaluate_p2,
)
from lanthanide_separation.gen8.report import md_table  # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
OOF = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "kshot"

#: Models the k-shot study runs on.  The first is gen7's best zero-shot arm; the
#: second is the arm gen7's own k-shot table used, kept so the numbers are
#: comparable digit for digit; the third is the no-ligand floor.
DEFAULT_MODELS: tuple[str, ...] = (
    "REC_ecfp_plus_recovered", "TREE_MC_ecfp_massaction",
    "TREE_MC_lig2d_ext_massaction", "NULL_metal_cond")
#: Models whose spread defines MAX_MODEL_DISAGREEMENT.  Chosen for representational
#: diversity, not for accuracy, and fixed here so the policy cannot be tuned.
DISAGREEMENT_MODELS: tuple[str, ...] = (
    "TREE_MC_ecfp_massaction", "TREE_MC_donors", "TREE_MC_lig2d_ext_massaction",
    "REC_ecfp_plus_recovered", "GEN7_everything_EXPLORATORY")


def load_blocks(models, oof_path: Path, cohort_path: Path) -> pd.DataFrame:
    """OOF predictions joined to the condition axes, with uncertainty and disagreement."""
    oof = pd.read_parquet(oof_path)
    cohort = pd.read_parquet(cohort_path)
    axis_columns = [c for c in POLICY_AXES if c in cohort.columns]
    keep = ["row_id"] + axis_columns
    disagreement = (oof[oof["model"].isin(DISAGREEMENT_MODELS)]
                    .groupby(["row_id", "split_seed"])["prediction"].std().rename("disagreement"))
    work = oof[oof["model"].isin(set(models))].copy()
    before = len(work)
    work = work.merge(cohort[keep], on="row_id", how="left", validate="many_to_one")
    assert len(work) == before, "axis join changed the row count"
    work = work.merge(disagreement, on=["row_id", "split_seed"], how="left")
    assert len(work) == before, "disagreement join changed the row count"
    work["uncertainty"] = work.get("extra__prediction_sd", pd.Series(np.nan, index=work.index))
    return work


def run_p1(work: pd.DataFrame, *, penalty: float, draws: int, seed: int,
           min_rows: int) -> pd.DataFrame:
    records: list[dict] = []
    rng = np.random.default_rng(seed)
    for (model, split_seed), frame in work.groupby(["model", "split_seed"], sort=True):
        for ligand, block in frame.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            if len(block) < min_rows:
                continue
            for record in evaluate_p1(block, penalty=penalty, draws=draws, rng=rng):
                record.update({"model": model, "split_seed": int(split_seed)})
                records.append(record)
    return pd.DataFrame(records)


def run_p2(work: pd.DataFrame, *, penalty: float, repeats: int, seed: int, min_rows: int,
           models: list[str], pool_cap: int) -> pd.DataFrame:
    records: list[dict] = []
    for (model, split_seed), frame in work.groupby(["model", "split_seed"], sort=True):
        if model not in models:
            continue
        for ligand, block in frame.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            if len(block) < min_rows:
                continue
            for record in evaluate_p2(block, penalty=penalty, repeats=repeats, seed=seed,
                                      pool_cap=pool_cap):
                record.update({"model": model, "split_seed": int(split_seed)})
                records.append(record)
    return pd.DataFrame(records)


def summarise(detail: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Ligand first (one ligand, one vote — the macro convention), then everything else."""
    per_ligand = detail.groupby(keys + ["extractant"])["mae"].mean().reset_index()
    out = per_ligand.groupby(keys).agg(mae=("mae", "mean"), n_ligands=("extractant", "nunique"))
    return out.reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["p1", "p2", "both"], default="both")
    parser.add_argument("--oof", type=Path, default=OOF)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--p2-models", nargs="*", default=["REC_ecfp_plus_recovered"])
    parser.add_argument("--penalty", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--pool-cap", type=int, default=POOL_CAP)
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    work = load_blocks(args.models, args.oof, args.cohort)
    print(f"{len(work):,} oof rows / {work['model'].nunique()} models / "
          f"{work['extractant'].nunique()} ligands", flush=True)

    if args.protocol in ("p1", "both"):
        started = time.time()
        p1 = run_p1(work, penalty=args.penalty, draws=args.draws, seed=args.seed,
                    min_rows=args.min_rows)
        p1.to_parquet(args.out / "p1_detail.parquet", index=False)
        summary = summarise(p1, ["model", "mode", "policy", "k"])
        summary.to_csv(args.out / "p1_summary.csv", index=False)
        print(f"\nP1 ({time.time() - started:.0f}s) — exhaustive, gen7-compatible")
        show = summary[summary["model"] == args.models[0]].sort_values(["mode", "k", "policy"])
        print(show.to_string(index=False))

    if args.protocol in ("p2", "both"):
        started = time.time()
        p2 = run_p2(work, penalty=args.penalty, repeats=args.repeats, seed=args.seed,
                    min_rows=args.min_rows, models=args.p2_models, pool_cap=args.pool_cap)
        p2.to_parquet(args.out / "p2_detail.parquet", index=False)
        summary = summarise(p2, ["model", "mode", "policy", "k"])
        summary.to_csv(args.out / "p2_summary.csv", index=False)
        print(f"\nP2 ({time.time() - started:.0f}s) — acquisition-fair")
        pivot = summary[summary["mode"] == "K1"].pivot_table(
            index="policy", columns="k", values="mae")
        print(pivot.to_string())

    (args.out / "run.json").write_text(json.dumps({
        "oof": str(args.oof), "models": args.models, "p2_models": args.p2_models,
        "penalty": args.penalty, "draws": args.draws, "repeats": args.repeats,
        "pool_cap": args.pool_cap, "min_rows": args.min_rows, "seed": args.seed,
        "k_values": list(K_VALUES), "modes": list(MODES),
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
