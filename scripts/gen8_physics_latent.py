#!/usr/bin/env python
"""Run the gen8 physics-latent model against offset correction on identical rows.

The whole point of the run is the pairing.  ``PHYS_*`` and ``OFFSET_K1`` are scored
on the same held-out ligand, the same repeat, the same candidate pool, the same
selected rows and the same evaluation rows, so a difference between the two
columns of the output table is a difference in method and nothing else.

Fold plan
---------
Frozen: ``seeded_group_kfold`` over ``tanimoto_cluster``, 5 folds, and the run
asserts that the test rows of fold *f* in that plan are byte-identical to the rows
the OOF frame marks ``fold == f``.  ``model_seed = 42 + fold*1009 + 9999991``.

One disclosed caveat
--------------------
``PHYS_residual*`` models the residual of the frozen global model, so ``fit_fold``
needs a prediction for its *training* rows too.  The only one available is the
frozen model's own out-of-fold prediction at the same split seed, which for a
training row comes from a different fold's model -- a model that did see the
held-out ligand.  The information path is long (through a global tree ensemble's
fit) but it is not zero, so the residual arms are *mildly optimistic*.  They are
reported anyway because the interesting outcome is whether they lose even with
that thumb on the scale.

Usage
-----
    python scripts/gen8_physics_latent.py --seeds 104729 --repeats 6
    python scripts/gen8_physics_latent.py --all-seeds --repeats 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "runs/gen7_architecture/cache/cohort.parquet"
OOF = ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet"
CURVES = ROOT / "runs/gen8_architecture/series/curve_table.parquet"
OUT = ROOT / "runs/gen8_architecture/physics_latent"
MODEL = "REC_ecfp_plus_recovered"


def build_fold_plan(cohort: pd.DataFrame, oof: pd.DataFrame, seeds) -> dict:
    """The frozen plan, asserted equal to the OOF frame's own fold labels."""
    from lanthanide_separation.gen6.cohorts import seeded_group_kfold

    groups = cohort["tanimoto_cluster"].astype(str).to_numpy()
    row_id = cohort["row_id"].to_numpy()
    plan = {}
    for seed in seeds:
        marked = oof[oof["split_seed"] == seed]
        for fold, (train, test) in enumerate(seeded_group_kfold(groups, 5, int(seed))):
            theirs = set(marked.loc[marked["fold"] == fold, "row_id"])
            assert set(row_id[test]) == theirs, (
                f"fold plan disagrees with the OOF frame at seed {seed} fold {fold}")
            plan[(int(seed), int(fold))] = train
    return plan


def slope_report(cohort: pd.DataFrame) -> dict:
    """Stage-1 sanity check: fitted exponents against the measured curve slopes."""
    from lanthanide_separation.gen8.physics_latent import (
        build_design_spec, fit_hierarchical, unshrunk_slopes, within_series_slopes)

    y = cohort["log_D"].to_numpy(dtype=float)
    spec = build_design_spec(cohort, metal_basis="rbf", n_basis=4)
    hierarchy = fit_hierarchical(cohort, y, spec)
    theta = pd.DataFrame(np.vstack([hierarchy.theta[k] for k in sorted(hierarchy.theta)]),
                         columns=list(spec.names))
    pooled = unshrunk_slopes(cohort, y, spec)
    within = within_series_slopes(cohort, y)

    curves = pd.read_parquet(CURVES)
    per_ligand = {
        "n_ext": curves[curves.axis == "cond__extractant_concentration_M"]
                 .groupby("extractant")["slope"].median(),
        "m_acid": curves[curves.axis == "cond__acid_concentration_M"]
                  .groupby("extractant")["slope"].median(),
    }
    out = {
        "design": list(spec.names),
        "noise_var": float(hierarchy.noise),
        "mu": dict(zip(spec.names, np.round(hierarchy.mu, 4).tolist())),
        "sigma_diag": dict(zip(spec.names, np.round(np.diag(hierarchy.sigma), 4).tolist())),
        "shrunk_theta_quantiles": {
            c: {str(q): round(float(v), 3) for q, v in
                theta[c].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()}
            for c in ("intercept", "n_ext", "m_acid")},
        "identifiable": {},
    }
    for label in ("n_ext", "m_acid"):
        mine = within.set_index("extractant")[label].dropna()
        reference = per_ligand[label]
        joined = pd.concat([mine.rename("fitted"), reference.rename("curve")],
                           axis=1).dropna()
        out["identifiable"][label] = {
            "n_ligands_pooled_ols": int(pooled[label].notna().sum()),
            "n_ligands_within_series": int(len(mine)),
            "n_ligands_total": int(cohort["extractant"].nunique()),
            "median_fitted_within_series": float(mine.median()),
            "median_curve_table_per_ligand": float(reference.median()),
            "median_curve_table_per_curve": float(
                curves[curves.axis == ("cond__extractant_concentration_M" if label == "n_ext"
                                       else "cond__acid_concentration_M")]["slope"].median()),
            "spearman_vs_curve_table": float(
                joined.corr(method="spearman").iloc[0, 1]) if len(joined) > 3 else float("nan"),
            "n_overlap": int(len(joined)),
        }
    return out


def leakage_selftest(physics, cohort, oof, fold_trainer, seed: int) -> None:
    """Prove the adapters cannot read an unmeasured target, rather than asserting it.

    The block is passed twice: once as the driver would build it, and once with
    every target destroyed (``log_D`` set to NaN) and the row order of that column
    scrambled.  Identical output is the only evidence that matters -- a code review
    can miss a column read, a byte-for-byte comparison cannot.
    """
    from lanthanide_separation.gen8.adapters import AdaptContext

    feature_columns = [c for c in cohort.columns if c != "log_D"]
    merged = oof[oof["split_seed"] == seed].merge(cohort[feature_columns], on="row_id",
                                                  how="left", validate="many_to_one",
                                                  suffixes=("", "__cohort"))
    fold = int(merged["fold"].iloc[0])
    train, y_train, model_seed = fold_trainer(seed, fold)
    for adapter in physics:
        adapter.fit_fold(train, y_train, split_seed=seed, fold=fold, model_seed=model_seed)

    block = merged[merged["fold"] == fold]
    ligand = block["extractant"].value_counts().index[0]
    block = block[block["extractant"] == ligand].reset_index(drop=True)
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    selected = np.array([0, 1], dtype=int)
    context = AdaptContext(split_seed=seed, fold=fold, extractant=ligand)

    poisoned = block.copy()
    poisoned["log_D"] = np.nan
    for adapter in physics:
        adapter._cache.clear()
        clean = adapter.predict(block, prediction, selected, truth[selected], context)
        adapter._cache.clear()
        blind = adapter.predict(poisoned, prediction, selected, truth[selected], context)
        assert np.array_equal(np.asarray(clean), np.asarray(blind)), (
            f"{adapter.name} changed its answer when the targets were destroyed")
        assert np.isfinite(np.asarray(clean)).all(), f"{adapter.name} returned non-finite"
    print(f"leakage self-test passed for {len(physics)} adapters on {len(block)} rows",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[104729])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-extras", action="store_true",
                        help="only the two core physics arms plus the references")
    args = parser.parse_args()

    from lanthanide_separation.gen8.adapters import default_adapters
    from lanthanide_separation.gen8.evaluate import evaluate_fewshot, summarise
    from lanthanide_separation.gen8.physics_latent import build_physics_adapters

    cohort = pd.read_parquet(COHORT)
    oof = pd.read_parquet(OOF)
    oof = oof[oof["model"] == MODEL].reset_index(drop=True)
    seeds = sorted(oof["split_seed"].unique().tolist()) if args.all_seeds else args.seeds
    oof = oof[oof["split_seed"].isin(seeds)].reset_index(drop=True)

    plan = build_fold_plan(cohort, oof, seeds)
    print(f"fold plan verified against the OOF frame for seeds {seeds}", flush=True)

    diagnostics = slope_report(cohort)
    print(json.dumps(diagnostics, indent=2, default=float), flush=True)

    frozen = {int(s): oof.loc[oof["split_seed"] == s].set_index("row_id")["prediction"]
              for s in seeds}

    def fold_trainer(split_seed: int, fold: int):
        index = plan[(int(split_seed), int(fold))]
        train = cohort.iloc[index].copy()
        train["prediction"] = train["row_id"].map(frozen[int(split_seed)]).to_numpy(dtype=float)
        assert np.isfinite(train["prediction"]).all(), "missing frozen prediction on a train row"
        model_seed = 42 + int(fold) * 1009 + 9999991
        return train, train["log_D"].to_numpy(dtype=float), model_seed

    physics = build_physics_adapters(include_gbm=not args.no_extras,
                                     include_mlp=not args.no_extras,
                                     include_categorical=not args.no_extras)
    adapters = physics + default_adapters()
    print("adapters:", [a.name for a in adapters], flush=True)

    leakage_selftest(physics, cohort, oof, fold_trainer, seeds[0])

    started = time.time()
    detail = evaluate_fewshot(oof, cohort, adapters, repeats=args.repeats,
                              fold_trainer=fold_trainer, verbose=True)
    print(f"evaluated in {time.time() - started:.0f}s -> {len(detail):,} records", flush=True)

    summary = summarise(detail)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summary.sort_values(["adapter", "policy", "k"]).to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("allseeds" if args.all_seeds else "_".join(map(str, seeds)))
    detail.to_parquet(OUT / f"physics_latent_detail_{tag}.parquet", index=False)
    summary.to_csv(OUT / f"physics_latent_summary_{tag}.csv", index=False)
    fold_state = pd.DataFrame([h for a in physics for h in a.fitter.history])
    fold_state.drop_duplicates(subset=["split_seed", "fold", "target", "metal_basis",
                                       "map_kind"]).to_json(
        OUT / f"physics_latent_foldstate_{tag}.json", orient="records", indent=2)
    (OUT / f"physics_latent_slopes_{tag}.json").write_text(
        json.dumps(diagnostics, indent=2, default=float))
    print("wrote", OUT / f"physics_latent_detail_{tag}.parquet", flush=True)


if __name__ == "__main__":
    main()
