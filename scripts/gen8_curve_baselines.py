"""Run the gen8 §3 series-aware curve baselines against offset correction.

    .venv/bin/python scripts/gen8_curve_baselines.py --seeds 104729 --repeats 6
    .venv/bin/python scripts/gen8_curve_baselines.py --repeats 12          # all five seeds

Nothing is refitted here that gen7 froze.  The global model's out-of-fold
predictions are read from disk; the fold plan is re-derived from
``seeded_group_kfold`` over the Tanimoto chemotype and **asserted equal** to the
``fold`` column of those predictions, so a trainable baseline's training rows are
exactly the rows the frozen model was trained on for that fold.  Every baseline is
scored by :func:`.gen8.evaluate.evaluate_fewshot` on the same ligands, repeats,
pools, selected rows and evaluation rows as ``OFFSET_K1``.
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

from lanthanide_separation.gen6.cohorts import seeded_group_kfold          # noqa: E402
from lanthanide_separation.gen8.adapters import AdaptContext, default_adapters  # noqa: E402
from lanthanide_separation.gen8.curve_baselines import (                   # noqa: E402
    AXIS_SOURCE, CURVE_COLUMN, FAMILY_COLUMN, SURROGATE_FEATURES, X_COLUMN,
    build_curve_adapters, curve_feature_columns, family_difference_test, massaction_priors,
    prepare_curve_columns,
)
from lanthanide_separation.gen8.evaluate import evaluate_fewshot, summarise  # noqa: E402
from lanthanide_separation.gen8.kshot import POLICY_AXES, stable_hash        # noqa: E402, stable_hash
from lanthanide_separation.gen8.protocols import POOL_CAP, make_p2_split    # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
OOF = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
MEMBERSHIP = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "baselines"

MODEL = "REC_ecfp_plus_recovered"
ALL_SEEDS = (104729, 130363, 155921, 196613, 262147)
#: The frozen gen5/gen6/gen7 model seed formula.  Fixed at 42 + fold*1009, offset by
#: a constant, and deliberately independent of the split seed (gen7 harness contract).
MODEL_SEED = lambda fold: 42 + fold * 1009 + 9999991      # noqa: E731

GP_EXTRA = ("massact__log10_cond__acid_concentration_M",
            "massact__log10_cond__extractant_concentration_M", "lanthanide_index")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load(cohort_path: Path, oof_path: Path, membership_path: Path):
    cohort = pd.read_parquet(cohort_path)
    membership = pd.read_parquet(membership_path)
    prepared = prepare_curve_columns(cohort, membership)
    oof = pd.read_parquet(oof_path)
    oof = oof[oof["model"] == MODEL].copy()
    return prepared, oof


def slim_features(prepared: pd.DataFrame) -> list[str]:
    """Only the columns an adapter or a policy actually reads, plus ``row_id``.

    Passing the whole 2,492-column cohort into the driver would materialise a
    half-gigabyte merge for no purpose; more importantly, a narrow frame makes the
    set of things an adapter *could* have read auditable by eye.
    """
    wanted = ["row_id", *POLICY_AXES, *GP_EXTRA, *curve_feature_columns()]
    seen, out = set(), []
    for name in wanted:
        if name in prepared.columns and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def make_fold_trainer(prepared: pd.DataFrame, oof: pd.DataFrame, *, verify: bool = True):
    """``(split_seed, fold) -> (train rows, y, model_seed)`` on the frozen fold plan."""
    groups = prepared["tanimoto_cluster"].astype(str).to_numpy()
    row_id = prepared["row_id"].to_numpy()
    plan: dict[tuple[int, int], np.ndarray] = {}
    for split_seed in sorted(oof["split_seed"].unique()):
        for fold, (train_index, test_index) in enumerate(
                seeded_group_kfold(groups, 5, int(split_seed))):
            plan[(int(split_seed), fold)] = train_index
            if verify:
                mine = set(row_id[test_index])
                theirs = set(oof.loc[(oof["split_seed"] == split_seed) & (oof["fold"] == fold),
                                     "row_id"])
                assert mine == theirs, (
                    f"fold plan disagrees with the OOF frame at seed {split_seed} fold {fold}: "
                    f"{len(mine)} vs {len(theirs)} test rows, {len(mine ^ theirs)} differing")

    columns = ["extractant", "log_D", "tanimoto_cluster", *SURROGATE_FEATURES,
               *slim_features(prepared)]
    columns = list(dict.fromkeys(c for c in columns if c in prepared.columns))
    table = prepared[columns]

    def fold_trainer(split_seed: int, fold: int):
        index = plan[(int(split_seed), int(fold))]
        train = table.iloc[index].reset_index(drop=True)
        return train, train["log_D"].to_numpy(dtype=float), MODEL_SEED(int(fold))

    return fold_trainer


# --------------------------------------------------------------------------- #
# Axis-resolved shape diagnostic
# --------------------------------------------------------------------------- #

def axis_shape_diagnostic(prepared, oof, adapters, fold_trainer, seeds, *,
                          repeats: int, seed: int, min_rows: int, pool_cap: int) -> pd.DataFrame:
    """Within-curve, per-axis shape error of each adapter's **k = 0** surface.

    Total MAE mixes the level (which one measurement fixes) with the shape (which
    it cannot).  This measures the shape alone and resolves it by swept axis, on
    exactly the evaluation rows the main table scores — so a baseline that fixes
    the extractant axis and does nothing to the metal axis is visible as such
    rather than averaged into a single flat verdict.
    """
    feature_columns = [c for c in slim_features(prepared) if c != "row_id"]
    merged = oof.merge(prepared[["row_id", *feature_columns]], on="row_id", how="left",
                       validate="many_to_one")
    merged = merged[merged["split_seed"].isin(seeds)]
    shape_only = [a for a in adapters if getattr(a, "adapt", None) == "none"]
    records: list[dict] = []
    for (split_seed, fold), fold_block in merged.groupby(["split_seed", "fold"], sort=True):
        train, y_train, model_seed = fold_trainer(int(split_seed), int(fold))
        for adapter in shape_only:
            adapter.fit_fold(train, y_train, split_seed=int(split_seed), fold=int(fold),
                             model_seed=int(model_seed))
        for ligand, block in fold_block.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < min_rows:
                continue
            truth = block["log_D"].to_numpy(dtype=float)
            prediction = block["prediction"].to_numpy(dtype=float)
            context = AdaptContext(split_seed=int(split_seed), fold=int(fold), extractant=ligand)
            surfaces = {"ZERO_SHOT": prediction}
            for adapter in shape_only:
                surfaces[adapter.name] = np.asarray(
                    adapter.predict(block, prediction, np.array([], dtype=int),
                                    np.array([]), context), dtype=float)
            evaluated = np.zeros(n, dtype=bool)
            for repeat in range(repeats):
                rng = np.random.default_rng((seed, repeat, stable_hash(ligand)))
                split = make_p2_split(n, rng, pool_cap=pool_cap)
                evaluated[split.evaluation] = True
            for axis in AXIS_SOURCE:
                curve = block[CURVE_COLUMN.format(axis=axis)].to_numpy(dtype=object)
                on_axis = (curve != "") & evaluated
                if on_axis.sum() < 3:
                    continue
                keys = curve[on_axis]
                frame = pd.DataFrame({"curve": keys, "truth": truth[on_axis]})
                centred_truth = (frame["truth"]
                                 - frame.groupby("curve")["truth"].transform("mean")).to_numpy()
                for name, surface in surfaces.items():
                    values = pd.Series(surface[on_axis])
                    centred = (values - values.groupby(keys).transform("mean")).to_numpy()
                    records.append({
                        "split_seed": int(split_seed), "fold": int(fold), "extractant": ligand,
                        "axis": axis, "surface": name, "n_rows": int(on_axis.sum()),
                        "shape_mae": float(np.abs(centred - centred_truth).mean()),
                        "flat_shape_mae": float(np.abs(centred_truth).mean())})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=[104729])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--policies", nargs="*", default=["RANDOM", "CENTRAL"])
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--pool-cap", type=int, default=POOL_CAP)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--skip-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("seed1" if len(args.seeds) == 1 else "all")

    prepared, oof = load(COHORT, OOF, MEMBERSHIP)
    oof = oof[oof["split_seed"].isin(args.seeds)].copy()
    print(f"{len(oof):,} oof rows / {oof['extractant'].nunique()} ligands / "
          f"{len(args.seeds)} split seeds", flush=True)
    for axis in AXIS_SOURCE:
        on = (prepared[CURVE_COLUMN.format(axis=axis)] != "").sum()
        print(f"  axis {axis:6s} on {on:5,} of {len(prepared):,} cohort rows")
    print(prepared[FAMILY_COLUMN].value_counts().to_string(), flush=True)

    fold_trainer = make_fold_trainer(prepared, oof)
    print("fold plan verified against the OOF fold column", flush=True)

    adapters = build_curve_adapters(penalty=args.penalty) + default_adapters()
    print("adapters: " + ", ".join(a.name for a in adapters), flush=True)

    started = time.time()
    detail = evaluate_fewshot(
        oof, prepared[slim_features(prepared)], adapters,
        policies=args.policies, repeats=args.repeats, seed=args.seed,
        min_rows=args.min_rows, pool_cap=args.pool_cap, fold_trainer=fold_trainer)
    print(f"evaluate_fewshot: {time.time() - started:.0f}s, {len(detail):,} records", flush=True)
    detail.to_parquet(args.out / f"detail_{tag}.parquet", index=False)

    fits = pd.DataFrame([record for adapter in adapters
                         for record in getattr(adapter, "fit_log", [])])
    if not fits.empty:
        fits.to_csv(args.out / f"fits_{tag}.csv", index=False)
        print("\n=== what each baseline learned per fold (training rows only) ===")
        if "shrink" in fits.columns:
            shrink = fits.dropna(subset=["shrink"])
            print(shrink.groupby(["adapter", "axis"]).agg(
                shrink_mean=("shrink", "mean"), shrink_min=("shrink", "min"),
                shrink_max=("shrink", "max"), n_train_ligands=("n_ligands", "mean"),
                n_curves=("n_curves", "mean")).round(3).to_string())
        if "length_scale" in fits.columns:
            gp_fits = fits.dropna(subset=["length_scale"])
            print(gp_fits.groupby(["adapter", "length_scale", "amplitude", "noise"])
                  .size().rename("folds").to_string())

    mae = summarise(detail, keys=("adapter", "policy", "k"), metric="mae")
    shape = summarise(detail, keys=("adapter", "policy", "k"), metric="shape_mae")
    table = mae.merge(shape.drop(columns=["n_ligands"]), on=["adapter", "policy", "k"])
    table.to_csv(args.out / f"summary_{tag}.csv", index=False)

    for policy in args.policies:
        sub = table[table["policy"].isin([policy, "NONE"])]
        print(f"\n=== MAE, policy {policy} (one ligand one vote) ===")
        print(sub.pivot_table(index="adapter", columns="k", values="mae").to_string())
        print(f"\n=== SHAPE MAE, policy {policy} ===")
        print(sub.pivot_table(index="adapter", columns="k", values="shape_mae").to_string())

    # paired difference against OFFSET_K1 on identical (ligand, repeat, policy, k) cells
    keys = ["split_seed", "fold", "extractant", "repeat", "policy", "k"]
    reference = detail[detail["adapter"] == "OFFSET_K1"].set_index(keys)["mae"]
    paired = []
    for name, sub in detail.groupby("adapter"):
        if name in ("OFFSET_K1", "ZERO_SHOT_REF"):
            continue
        joined = sub.set_index(keys)["mae"].to_frame("mine").join(
            reference.rename("offset"), how="inner").dropna()
        if joined.empty:
            continue
        joined = joined.reset_index()
        joined["delta"] = joined["mine"] - joined["offset"]
        per_ligand = joined.groupby(["k", "extractant"])["delta"].mean().reset_index()
        for k, group in per_ligand.groupby("k"):
            values = group["delta"].to_numpy(dtype=float)
            rng = np.random.default_rng(args.seed)
            draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
            paired.append({"adapter": name, "k": int(k), "n_ligands": int(len(values)),
                           "delta_mae": float(values.mean()),
                           "ci_lo": float(np.percentile(draws, 2.5)),
                           "ci_hi": float(np.percentile(draws, 97.5)),
                           "win_rate": float((values < 0).mean())})
    paired = pd.DataFrame(paired)
    paired.to_csv(args.out / f"paired_vs_offset_k1_{tag}.csv", index=False)
    print("\n=== paired delta vs OFFSET_K1 (negative = better), ligand bootstrap ===")
    print(paired.sort_values(["k", "delta_mae"]).to_string(index=False))

    # mass-action priors and whether the families genuinely differ
    priors, tests = [], []
    for axis in ("ext", "acid", "metal"):
        priors.append(massaction_priors(prepared, prepared["log_D"].to_numpy(dtype=float), axis))
        test = family_difference_test(prepared, prepared["log_D"].to_numpy(dtype=float), axis)
        if test:
            tests.append(test)
    priors = pd.concat([p for p in priors if not p.empty], ignore_index=True)
    priors.to_csv(args.out / "massaction_priors.csv", index=False)
    print("\n=== mass-action priors per donor family (ligand bootstrap, whole cohort) ===")
    print(priors.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))

    if not args.skip_diagnostic:
        started = time.time()
        diagnostic = axis_shape_diagnostic(
            prepared, oof, adapters, fold_trainer, args.seeds, repeats=args.repeats,
            seed=args.seed, min_rows=args.min_rows, pool_cap=args.pool_cap)
        diagnostic.to_parquet(args.out / f"axis_shape_{tag}.parquet", index=False)
        print(f"\n=== within-curve SHAPE MAE by swept axis (k = 0 surfaces, "
              f"{time.time() - started:.0f}s) ===")
        per_ligand = diagnostic.groupby(["axis", "surface", "extractant"]).agg(
            shape_mae=("shape_mae", "mean"), flat=("flat_shape_mae", "mean")).reset_index()
        summary = per_ligand.groupby(["axis", "surface"]).agg(
            shape_mae=("shape_mae", "mean"), flat=("flat", "mean"),
            n_ligands=("extractant", "nunique")).reset_index()
        print(summary.to_string(index=False))
        summary.to_csv(args.out / f"axis_shape_summary_{tag}.csv", index=False)

    (args.out / f"run_{tag}.json").write_text(json.dumps({
        "model": MODEL, "seeds": args.seeds, "repeats": args.repeats,
        "policies": args.policies, "min_rows": args.min_rows, "pool_cap": args.pool_cap,
        "seed": args.seed, "penalty": args.penalty,
        "adapters": [a.name for a in adapters],
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
