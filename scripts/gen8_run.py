"""The gen8 primary run: every acquisition policy x every calibration adapter x k.

One invocation produces the paired table §21 asks for.  Everything is scored on the
same held-out ligands, the same repeats, the same candidate pools and the same
evaluation rows, so every difference in it is attributable to exactly one of:

* **measurement** — k = 0 against k > 0;
* **architecture** — one adapter against another at the same k and policy;
* **acquisition** — one policy against another at the same k and adapter.

Extra adapters (CNP, physics-latent, curve baselines) are imported by name if their
modules exist, and because the pool/evaluation split is a deterministic function of
(seed, repeat, ligand, n_rows) alone, a separate later run of those adapters is
*still* paired with this one row for row.

    .venv/bin/python scripts/gen8_run.py --seeds 5 --repeats 12
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen8.adapters import default_adapters  # noqa: E402
from lanthanide_separation.gen8.evaluate import evaluate_fewshot, summarise  # noqa: E402
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.kshot import POLICIES  # noqa: E402
from lanthanide_separation.gen8.report import md_table  # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
OOF = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "active_acquisition"
SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
DISAGREEMENT_MODELS = ("TREE_MC_ecfp_massaction", "TREE_MC_donors",
                       "TREE_MC_lig2d_ext_massaction", "REC_ecfp_plus_recovered",
                       "GEN7_everything_EXPLORATORY")


def make_fold_trainer(cohort: pd.DataFrame, n_splits: int = 5):
    """(split_seed, fold) -> (train frame, train target, model seed), gen5's plan exactly."""
    groups = cohort["tanimoto_cluster"].astype(str).to_numpy()
    target = cohort["log_D"].to_numpy(dtype=float)
    cache: dict[tuple[int, int], tuple] = {}

    def trainer(split_seed: int, fold: int):
        key = (int(split_seed), int(fold))
        if key not in cache:
            for k, (train_index, _) in enumerate(seeded_group_kfold(groups, n_splits, split_seed)):
                cache[(int(split_seed), k)] = (
                    cohort.iloc[train_index], target[train_index],
                    int(42 + k * 1009 + 9_999_991))
        return cache[key]
    return trainer


def load(model: str, oof_path: Path, cohort_path: Path, seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_parquet(oof_path)
    disagreement = (oof[oof["model"].isin(DISAGREEMENT_MODELS)]
                    .groupby(["row_id", "split_seed"])["prediction"].std().rename("disagreement"))
    work = oof[(oof["model"] == model) & (oof["split_seed"].isin(seeds))].copy()
    if work.empty:
        raise SystemExit(f"no rows for model {model!r}")
    before = len(work)
    work = work.merge(disagreement, on=["row_id", "split_seed"], how="left")
    assert len(work) == before
    return work, pd.read_parquet(cohort_path)


def extra_adapters(names) -> list:
    """Import optional adapter modules if they have been built; never fail on absence."""
    out: list = []
    for module_name, builder in names:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - an absent module is a normal state here
            print(f"  (skipping {module_name}: {type(error).__name__}: {error})", flush=True)
            continue
        if not hasattr(module, builder):
            print(f"  (skipping {module_name}: no {builder})", flush=True)
            continue
        try:
            out.extend(getattr(module, builder)())
        except Exception as error:  # noqa: BLE001
            print(f"  (skipping {module_name}.{builder}: {type(error).__name__}: {error})", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="REC_ecfp_plus_recovered")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--policies", nargs="*", default=sorted(POLICIES))
    parser.add_argument("--with-extra", action="store_true",
                        help="import cnp / physics_latent / curve_baselines adapters if present")
    parser.add_argument("--oof", type=Path, default=OOF)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="primary")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[: max(1, min(5, args.seeds))]
    oof, cohort = load(args.model, args.oof, args.cohort, seeds)
    adapters = default_adapters()
    if args.with_extra:
        adapters += extra_adapters([
            ("lanthanide_separation.gen8.curve_baselines", "build_curve_adapters"),
            ("lanthanide_separation.gen8.physics_latent", "build_physics_adapters"),
            ("lanthanide_separation.gen8.cnp", "build_cnp_adapters"),
        ])
    print(f"model {args.model} / {len(seeds)} seeds / {len(adapters)} adapters "
          f"({', '.join(a.name for a in adapters)}) / {len(args.policies)} policies", flush=True)

    started = time.time()
    detail = evaluate_fewshot(
        oof, cohort, adapters, policies=args.policies,
        with_oracle_for=("OFFSET_K1",), repeats=args.repeats, min_rows=args.min_rows,
        fold_trainer=make_fold_trainer(cohort))
    print(f"\n{len(detail):,} records in {time.time() - started:.0f}s", flush=True)
    detail.to_parquet(args.out / f"{args.tag}_detail.parquet", index=False)

    summary = summarise(detail, keys=("adapter", "policy", "k"))
    for metric in ("offset", "shape_mae", "spearman", "within_0_5", "within_1_0"):
        extra = summarise(detail, keys=("adapter", "policy", "k"), metric=metric)
        summary = summary.merge(extra.drop(columns="n_ligands"), on=["adapter", "policy", "k"])
    summary.to_csv(args.out / f"{args.tag}_summary.csv", index=False)

    pd.set_option("display.width", 240)
    pivot = summary[summary["adapter"] == "OFFSET_K1"].pivot_table(
        index="policy", columns="k", values="mae")
    print("\nOFFSET_K1 by policy and k:")
    print(pivot.to_string())
    pivot2 = summary[summary["policy"] == "RANDOM"].pivot_table(
        index="adapter", columns="k", values="mae")
    print("\nRANDOM selection, by adapter and k:")
    print(pivot2.to_string())

    (args.out / f"{args.tag}_run.json").write_text(json.dumps({
        "model": args.model, "seeds": [int(s) for s in seeds], "repeats": args.repeats,
        "min_rows": args.min_rows, "policies": list(args.policies),
        "adapters": [a.name for a in adapters], "n_records": int(len(detail)),
        "seconds": time.time() - started,
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
