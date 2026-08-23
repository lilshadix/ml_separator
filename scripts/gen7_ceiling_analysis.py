"""How much of the offset error is learnable at all?

The brief forbids "collect more data" as a conclusion, and rightly.  But it also
asks which world we live in, and one of the candidate worlds is that a slice of
the level error is not a property of the chemistry at all — it is a property of
*which experiment happened*, and no representation of a molecule can predict it.

This script measures that slice.  The unit is the **batch**: one (DOI, table or
figure) pair recovered from the upstream SAFE exports — 694 over 115 publications in the
raw bundle, 541 over 111 inside the evaluation cohort — and the finest
experimental-session proxy the record supports.

The model is

.. code-block:: text

    y = level(ligand) + response(conditions, metal | ligand) + batch offset + noise

and the quantity of interest is the standard deviation of the batch offset *within
a ligand*.  It is estimated by fitting a flexible within-ligand model of the
conditions and the metal, then taking the batch means of its residuals.

That estimate is biased upward on its own, because batch means of *any* noisy
residual have spread.  So every number is reported against a **permutation null**
that shuffles the batch labels within each ligand, preserving the batch size
distribution exactly.  The null absorbs the residual noise, the fitting bias, and
the batch-size imbalance; what survives it is a real batch effect.

The final translation is a simulation: given the fitted batch-offset scale and the
actual batch composition of each held-out ligand, what offset MAE would a model
with *perfect* knowledge of the ligand's true level still incur?  That number is
the floor the gen7 leaderboard is being measured against.
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

from lanthanide_separation.gen7.harness import load_cohort  # noqa: E402

MIN_ROWS_PER_BATCH = 2
MIN_BATCHES_PER_LIGAND = 2
MIN_ROWS_PER_LIGAND = 10


def within_ligand_residuals(frame: pd.DataFrame, feature_columns, *, seed: int) -> pd.Series:
    """Residuals of a flexible per-ligand model of conditions + metal.

    Cross-fitted (5-fold, within the ligand) so a residual is never taken from a
    row the model was fitted on — otherwise a flexible learner drives residuals to
    zero and destroys the very batch signal being measured.  Ligands too small to
    cross-fit fall back to a ridge, and ligands too small even for that are dropped
    and counted.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, block in frame.groupby("extractant", sort=False):
        if len(block) < MIN_ROWS_PER_LIGAND:
            continue
        x = block[list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        keep = ~np.all(np.isnan(x), axis=0)
        x = x[:, keep]
        if x.shape[1] == 0:
            continue
        median = np.nan_to_num(np.nanmedian(x, axis=0))
        x = np.where(np.isfinite(x), x, median)
        y = block["log_D"].to_numpy(float)
        prediction = np.full(len(y), np.nan)
        splitter = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=seed)
        for fit_idx, hold_idx in splitter.split(x):
            if len(fit_idx) < 4:
                continue
            if len(fit_idx) < 20:
                model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
            else:
                model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                            random_state=seed, n_jobs=-1)
            model.fit(x[fit_idx], y[fit_idx])
            prediction[hold_idx] = model.predict(x[hold_idx])
        out.loc[block.index] = y - prediction
    return out


def batch_offset_scale(frame: pd.DataFrame, residuals: pd.Series, *,
                       n_permutations: int = 200, seed: int = 20260819) -> dict:
    """Observed vs permuted spread of within-ligand batch means."""
    rng = np.random.default_rng(seed)
    work = frame.assign(_residual=residuals).dropna(subset=["_residual"])
    observed: list[float] = []
    null: list[float] = []
    per_ligand: list[dict] = []
    for name, block in work.groupby("extractant", sort=False):
        sizes = block.groupby("nuisance__batch").size()
        usable = sizes[sizes >= MIN_ROWS_PER_BATCH]
        if len(usable) < MIN_BATCHES_PER_LIGAND:
            continue
        sub = block[block["nuisance__batch"].isin(usable.index)]
        values = sub["_residual"].to_numpy(float)
        labels = sub["nuisance__batch"].to_numpy()
        means = pd.Series(values).groupby(labels).mean()
        spread = float(means.std(ddof=1))
        draws = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(values)
            draws.append(float(pd.Series(shuffled).groupby(labels).mean().std(ddof=1)))
        null_mean = float(np.mean(draws))
        observed.append(spread)
        null.append(null_mean)
        per_ligand.append({
            "extractant": name, "n_rows": int(len(sub)), "n_batches": int(len(usable)),
            "batch_spread": spread, "permutation_null": null_mean,
            "excess": spread - null_mean,
            "p_value": float((1 + np.sum(np.asarray(draws) >= spread)) / (1 + n_permutations)),
            "residual_sd": float(values.std(ddof=1)),
        })
    table = pd.DataFrame(per_ligand)
    if table.empty:
        return {"table": table, "summary": {}}
    # Variance-components estimate: observed² = true² + null², so true² = obs² − null²
    variance = np.maximum(table["batch_spread"] ** 2 - table["permutation_null"] ** 2, 0.0)
    summary = {
        "n_ligands": int(len(table)),
        "n_rows_covered": int(table["n_rows"].sum()),
        "median_observed_batch_spread": float(table["batch_spread"].median()),
        "median_permutation_null": float(table["permutation_null"].median()),
        "median_excess": float(table["excess"].median()),
        "sigma_batch_variance_components": float(np.sqrt(np.average(
            variance, weights=table["n_rows"]))),
        "frac_ligands_p_below_0_05": float((table["p_value"] < 0.05).mean()),
        "frac_ligands_positive_excess": float((table["excess"] > 0).mean()),
    }
    return {"table": table, "summary": summary}


def offset_floor_simulation(frame: pd.DataFrame, sigma_batch: float, *,
                            n_simulations: int = 2000, seed: int = 20260819) -> dict:
    """Offset MAE a model with perfect ligand-level knowledge would still incur.

    For each ligand, draw one offset per batch from ``N(0, sigma_batch²)``, average
    them with the ligand's real row weights, and take the absolute value.  The mean
    over ligands is the floor of ``offset_mae`` — the metric is defined as the mean
    over held-out ligands of ``|mean residual|``, so this is exactly comparable.
    """
    rng = np.random.default_rng(seed)
    weights = []
    for _, block in frame.groupby("extractant", sort=False):
        counts = block.groupby("nuisance__batch").size().to_numpy(float)
        weights.append(counts / counts.sum())
    draws = np.empty((n_simulations, len(weights)))
    for s in range(n_simulations):
        for k, w in enumerate(weights):
            draws[s, k] = abs(float(np.dot(w, rng.normal(0.0, sigma_batch, size=len(w)))))
    per_simulation = draws.mean(axis=1)
    return {
        "sigma_batch": sigma_batch,
        "n_ligands": len(weights),
        "offset_mae_floor_mean": float(per_simulation.mean()),
        "offset_mae_floor_p05": float(np.quantile(per_simulation, 0.05)),
        "offset_mae_floor_p95": float(np.quantile(per_simulation, 0.95)),
        "median_batches_per_ligand": float(np.median([len(w) for w in weights])),
        "frac_single_batch_ligands": float(np.mean([len(w) == 1 for w in weights])),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "ceiling")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    frame = cohort.frame
    if "nuisance__batch" not in frame.columns:
        raise SystemExit("cohort lacks nuisance__batch; build the recovered cell table first")
    features = list(cohort.block_columns(("METAL", "COND", "MASSACTION")))
    print(f"cohort {len(frame)} rows, {frame.extractant.nunique()} ligands, "
          f"{frame.nuisance__batch.nunique()} batches", flush=True)

    residuals = within_ligand_residuals(frame, features, seed=20260819)
    print(f"cross-fitted residuals for {int(residuals.notna().sum())} rows", flush=True)

    result = batch_offset_scale(frame, residuals, n_permutations=args.permutations)
    table, summary = result["table"], result["summary"]
    table.to_csv(args.out / "batch_offset_per_ligand.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)

    sigma = summary.get("sigma_batch_variance_components", float("nan"))
    floor = offset_floor_simulation(frame, sigma) if np.isfinite(sigma) else {}
    print(json.dumps(floor, indent=2), flush=True)

    # what fraction of total variance each level explains, for the report's table
    def grouped_r2(keys) -> float:
        grouped = frame.groupby(keys)["log_D"].transform("mean")
        total = float(((frame["log_D"] - frame["log_D"].mean()) ** 2).sum())
        return float(1.0 - ((frame["log_D"] - grouped) ** 2).sum() / total)

    variance_table = {
        "ligand": grouped_r2(["extractant"]),
        "publication": grouped_r2(["nuisance__doi"]),
        "batch": grouped_r2(["nuisance__batch"]),
        "metal": grouped_r2(["metal_symbol"]),
        "ligand_x_metal": grouped_r2(["extractant", "metal_symbol"]),
        "ligand_x_batch": grouped_r2(["extractant", "nuisance__batch"]),
        "series": grouped_r2(["series_id"]) if "series_id" in frame.columns else None,
    }
    payload = {"variance_explained_by_grouping": variance_table,
               "batch_offset": summary, "offset_mae_floor": floor,
               "n_rows": int(len(frame)), "n_ligands": int(frame.extractant.nunique()),
               "n_batches": int(frame.nuisance__batch.nunique()),
               "n_publications": int(frame.nuisance__doi.nunique())}
    (args.out / "ceiling.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(variance_table, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
