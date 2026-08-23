"""Evaluate the slope-restoration repair against plain offset calibration (brief §10, §19)."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.adapters import default_adapters  # noqa: E402
from lanthanide_separation.gen8.evaluate import evaluate_fewshot, summarise  # noqa: E402
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.slope_restore import build_slope_adapters  # noqa: E402
import importlib
run = importlib.import_module("gen8_run") if False else None
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gen8_run import DISAGREEMENT_MODELS, SEEDS, make_fold_trainer  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen8_architecture" / "functional"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--model", default="REC_ecfp_plus_recovered")
    p.add_argument("--strengths", type=float, nargs="*", default=[0.5, 0.8, 1.0])
    p.add_argument("--policies", nargs="*", default=["RANDOM", "MEDOID"])
    p.add_argument("--tag", default="slope_restore")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[: max(1, min(5, args.seeds))]
    oof = pd.read_parquet(REPO_ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet")
    disagreement = (oof[oof["model"].isin(DISAGREEMENT_MODELS)]
                    .groupby(["row_id", "split_seed"])["prediction"].std().rename("disagreement"))
    work = oof[(oof["model"] == args.model) & (oof["split_seed"].isin(seeds))].merge(
        disagreement, on=["row_id", "split_seed"], how="left")
    cohort = pd.read_parquet(REPO_ROOT / "runs/gen7_architecture/cache/cohort.parquet")

    adapters = default_adapters() + build_slope_adapters(strengths=args.strengths)
    print(f"{len(adapters)} adapters: {', '.join(a.name for a in adapters)}", flush=True)
    started = time.time()
    detail = evaluate_fewshot(work, cohort, adapters, policies=args.policies,
                              with_oracle_for=(), repeats=args.repeats,
                              fold_trainer=make_fold_trainer(cohort))
    print(f"{len(detail):,} records in {time.time() - started:.0f}s")
    detail.to_parquet(args.out / f"{args.tag}_detail.parquet", index=False)

    summary = summarise(detail, keys=("adapter", "policy", "k"))
    for metric in ("offset", "shape_mae", "spearman"):
        summary = summary.merge(
            summarise(detail, keys=("adapter", "policy", "k"), metric=metric).drop(columns="n_ligands"),
            on=["adapter", "policy", "k"])
    summary.to_csv(args.out / f"{args.tag}_summary.csv", index=False)
    pd.set_option("display.width", 240)
    print("\nMAE (RANDOM selection):")
    print(summary[summary.policy == "RANDOM"].pivot_table(index="adapter", columns="k", values="mae").to_string())
    print("\nSHAPE MAE (RANDOM selection):")
    print(summary[summary.policy == "RANDOM"].pivot_table(index="adapter", columns="k", values="shape_mae").to_string())

    rows = []
    for k in (0, 1, 2, 3, 5):
        sub = detail[(detail["k"] == k) & (detail["policy"] == "RANDOM")]
        if sub.empty:
            continue
        base = "ZERO_SHOT" if k == 0 else "OFFSET_K1"
        contrasts = {}
        for name in sorted(sub["adapter"].unique()):
            if name.startswith("SLOPE_RESTORE"):
                contrasts[f"k={k}: {name} vs {base}"] = (base, name)
        if not contrasts:
            continue
        boot = paired_chemotype_bootstrap(sub.rename(columns={"adapter": "arm"}), contrasts,
                                          arm_column="arm")
        boot_shape = paired_chemotype_bootstrap(sub.rename(columns={"adapter": "arm"}), contrasts,
                                                arm_column="arm", value_column="shape_mae")
        boot["statistic"] = "mae"; boot_shape["statistic"] = "shape_mae"
        rows += [boot, boot_shape]
    if rows:
        allboot = pd.concat(rows, ignore_index=True)
        allboot.to_csv(args.out / f"{args.tag}_bootstrap.csv", index=False)
        print("\npaired chemotype bootstrap (positive = slope restoration better):")
        print(allboot.to_string(index=False))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
