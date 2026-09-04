#!/usr/bin/env python
"""PHASE 1 — is gen9's relative-context mechanism robust to the question asked?

Fits each arm once per (seed, fold), then asks the *same fitted model* about the
same absolute points inside many different candidate designs.  Nothing is refitted
between designs, so a difference in the answer is a difference in the question.

The frozen model is carried as a positive control: it reads no design context, so
every shift it reports must be exactly zero.  If it is not, the harness is
measuring itself — which is what gen9's acquisition study was doing before its
issue 1 was found, and the reason that control is not optional.

Outputs, all under ``runs/gen10_final/query_consistency``:

``rows.parquet``           one record per (arm, seed, fold, ligand, curve, variant)
``summary_by_arm.csv``     the distribution per arm and variant family
``summary_by_axis.csv``    the same, split by curve type
``by_publication.csv``     by source publication, for the Phase 8 clustering
``by_width.csv``           by window-width quartile
``worst_ligands.csv``      the tail: which chemistry is least stable
``control.json``           the frozen-model control, which must be zero
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, FoldContext, build_folds,
)
from lanthanide_separation.gen10.architectures import (  # noqa: E402
    LevelShapeModel, MonolithModel, RecomposedModel, ResidualShapeModel, prime_raw_cache,
)
from lanthanide_separation.gen10.consistency import (  # noqa: E402
    build_variants, compare_variant, eligible_curves, shift_for_shifted_window,
)
from lanthanide_separation.gen10.runner import (  # noqa: E402
    DETERMINISM_TOLERANCE, prepared_cohort,
)
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen10_final" / "query_consistency"

#: The arms Phase 1 compares.  ``FROZEN`` is the control and must report zero.
ARMS: dict[str, callable] = {
    "FROZEN": lambda: MonolithModel(name="FROZEN", representation_name="NONE"),
    "REL_MONOLITH": lambda: MonolithModel(
        name="REL_MONOLITH", representation_name="GEN9", gen9_compat=True),
    "SHAPE_RECOMPOSED": lambda: RecomposedModel(
        name="SHAPE_RECOMPOSED", representation_name="GEN9", gen9_compat=True),
    "LEVEL_SHAPE": lambda: LevelShapeModel(
        name="LEVEL_SHAPE", representation_name="GEN9"),
    "RESIDUAL_SHAPE": lambda: ResidualShapeModel(
        name="RESIDUAL_SHAPE", representation_name="GEN9"),
    "REL_HYBRID": lambda: MonolithModel(
        name="REL_HYBRID", representation_name="HYBRID"),
    "RECOMPOSED_HYBRID": lambda: RecomposedModel(
        name="RECOMPOSED_HYBRID", representation_name="HYBRID"),
    "RECOMPOSED_RANK": lambda: RecomposedModel(
        name="RECOMPOSED_RANK", representation_name="RANK"),
    "RECOMPOSED_LOCAL": lambda: RecomposedModel(
        name="RECOMPOSED_LOCAL", representation_name="LOCAL"),
    "RECOMPOSED_ALL": lambda: RecomposedModel(
        name="RECOMPOSED_ALL", representation_name="ALL"),
    "RECOMPOSED_GEN9_PLUS": lambda: RecomposedModel(
        name="RECOMPOSED_GEN9_PLUS", representation_name="GEN9_PLUS"),
}


def run(arms: list[str], seeds: list[int], max_ligands: int | None,
        out_dir: Path) -> pd.DataFrame:
    cohort = prepared_cohort()
    # Thousands of small queries against a 2,160-column design; without this the
    # per-call `pd.to_numeric` sweep dominates the whole benchmark.
    prime_raw_cache(cohort)
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    records: list[dict] = []

    for seed in seeds:
        folds = build_folds(frame, seed)
        for fold in folds:
            train = frame.iloc[fold.train_index]
            test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
            context = FoldContext(fold=fold, cohort=cohort, feature_columns=(),
                                  model_seed=fold.model_seed)
            y = target[fold.train_index]
            fitted = {}
            for name in arms:
                started = time.time()
                fitted[name] = ARMS[name]().fit(train, y, context).set_predict_jobs(1)
                print(f"  seed {seed} fold {fold.fold} fit {name} "
                      f"({time.time() - started:.0f}s)", flush=True)

            ligands = list(dict.fromkeys(test["extractant"].astype(str)))
            if max_ligands:
                ligands = ligands[:max_ligands]
            for ligand in ligands:
                rows = test[test["extractant"].astype(str) == ligand].reset_index(drop=True)
                # Never Python's salted `hash`: gen8 lost cross-run pairing to it.
                rng = np.random.default_rng([int(seed), int(fold.fold)])
                n_all = eligible_curves(rows, max_curves=None)["curve_id"].nunique()
                curves = eligible_curves(rows, rng=rng)
                if curves.empty:
                    continue
                for curve_id, block in curves.groupby("curve_id", sort=True):
                    axis = str(block["axis"].iloc[0])
                    label = str(block["axis_label"].iloc[0])
                    variants = build_variants(rows, str(curve_id), axis, curves, rng=rng)
                    if not variants:
                        continue
                    reference = variants[0]
                    assert reference.family == "REFERENCE"
                    base = {name: pd.Series(
                        model.predict(reference.frame),
                        index=reference.frame["row_id"].astype(str).to_numpy())
                        for name, model in fitted.items()}
                    for variant in variants[1:]:
                        for name, model in fitted.items():
                            prediction = pd.Series(
                                model.predict(variant.frame),
                                index=variant.frame["row_id"].astype(str).to_numpy())
                            if variant.family == "SHIFT":
                                stats = shift_for_shifted_window(
                                    base[name], prediction,
                                    variant.detail["paired_with"], variant.compare_ids)
                            else:
                                stats = compare_variant(base[name], prediction,
                                                        variant.compare_ids)
                            records.append({
                                "arm": name, "split_seed": int(seed), "fold": int(fold.fold),
                                "extractant": ligand, "curve_id": str(curve_id),
                                "axis_label": label, "family": variant.family,
                                "variant": variant.label,
                                "n_curve_rows": int(len(block)),
                                "n_eligible_curves_ligand": int(n_all),
                                "n_curves_used_ligand": int(curves["curve_id"].nunique()),
                                "window_width": float(np.ptp(
                                    block["axis_value"].to_numpy(dtype=float))),
                                **{f"detail__{k}": v for k, v in variant.detail.items()
                                   if not isinstance(v, list)},
                                **stats})
    table = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "rows.parquet", index=False)
    return table


def summarise(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [c for c in ("shift_median", "shift_p90", "shift_p95", "shift_max",
                           "shape_shift_median", "shape_shift_max", "level_shift")
               if c in table.columns]
    rows = []
    for key, block in table.groupby(keys, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, values))
        record["n_comparisons"] = int(len(block))
        record["n_ligands"] = int(block["extractant"].nunique())
        for column in numeric:
            series = block[column].dropna()
            if series.empty:
                continue
            record[f"{column}__median"] = float(series.median())
            record[f"{column}__p90"] = float(series.quantile(0.90))
            record[f"{column}__p95"] = float(series.quantile(0.95))
            record[f"{column}__max"] = float(series.max())
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--arms", nargs="*", default=["FROZEN", "REL_MONOLITH",
                                                      "SHAPE_RECOMPOSED"])
    parser.add_argument("--max-ligands", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summarise-only", action="store_true",
                        help="regenerate every summary from an existing rows.parquet")
    args = parser.parse_args(argv)

    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; have {sorted(ARMS)}")
    seeds = list(DEFAULT_SEEDS[:max(1, min(5, args.seeds))])
    started = time.time()
    if args.summarise_only:
        table = pd.read_parquet(args.out / "rows.parquet")
        args.arms = sorted(table["arm"].unique())
    else:
        table = run(args.arms, seeds, args.max_ligands, args.out)
    if table.empty:
        raise SystemExit("no eligible curves — nothing was measured")

    non_shift = table[table["family"] != "SHIFT"]
    summarise(non_shift, ["arm", "family"]).to_csv(
        args.out / "summary_by_arm.csv", index=False)
    summarise(non_shift, ["arm", "axis_label", "family"]).to_csv(
        args.out / "summary_by_axis.csv", index=False)
    summarise(table[table["family"] == "SHIFT"], ["arm", "axis_label"]).to_csv(
        args.out / "shifted_window.csv", index=False)

    # Quartiles of the *curve* width distribution, one value per curve, so a ligand
    # with many variants does not pull the cut points toward itself.
    per_curve = non_shift.drop_duplicates("curve_id")["window_width"]
    edges = np.unique(np.quantile(per_curve, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(edges) >= 3:
        labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
        width = non_shift.assign(width_quartile=pd.cut(
            non_shift["window_width"], bins=edges, labels=labels, include_lowest=True))
        summarise(width, ["arm", "width_quartile"]).to_csv(
            args.out / "by_width.csv", index=False)

    worst = (non_shift.groupby(["arm", "extractant"])["shift_p95"].median()
             .reset_index().sort_values(["arm", "shift_p95"], ascending=[True, False]))
    worst.to_csv(args.out / "worst_ligands.csv", index=False)

    control = non_shift[non_shift["arm"] == "FROZEN"]
    exact = non_shift[non_shift["family"].isin(["CONTEXT", "PERMUTE"])]
    exact_max = float(exact["shift_max"].max()) if len(exact) else 0.0
    payload = {
        "exact_invariance_families": ["CONTEXT", "PERMUTE"],
        "exact_invariance_max_shift": exact_max,
        "exact_invariance_tolerance": DETERMINISM_TOLERANCE,
        "exact_invariance_holds": bool(exact_max <= DETERMINISM_TOLERANCE),
        "seeds": seeds, "arms": args.arms,
        "n_comparisons": int(len(table)),
        "n_ligands": int(table["extractant"].nunique()),
        "n_curves": int(table["curve_id"].nunique()),
        "elapsed_s": round(time.time() - started, 1),
    }
    if len(control):
        payload["frozen_control_max_shift"] = float(control["shift_max"].max())
        payload["frozen_control_is_zero"] = bool(
            control["shift_max"].max() <= DETERMINISM_TOLERANCE)
    (args.out / "control.json").write_text(json.dumps(payload, indent=2))

    print("\n=== query-set consistency, shift in log units ===")
    board = summarise(non_shift, ["arm", "family"])
    print(board.to_string(index=False))
    # "Exactly zero" means the ExtraTrees thread-order floor (~1e-15), not literal
    # zero: a forest sums its trees in completion order, so two predictions of the
    # same row in different batches differ in the last bit.  DETERMINISM_TOLERANCE
    # is three orders of magnitude above that floor and nine below any effect.
    if len(control) and control["shift_max"].max() > DETERMINISM_TOLERANCE:
        raise SystemExit(
            f"the frozen control moved by {control['shift_max'].max():.3g} — it reads no "
            "design context, so the benchmark is measuring its own plumbing")
    if exact_max > DETERMINISM_TOLERANCE:
        raise SystemExit(
            f"a CONTEXT/PERMUTE variant moved a prediction by {exact_max:.3g}; these are "
            "exact invariances of every arm and a value above the float floor is a bug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
