#!/usr/bin/env python
"""Turn the physics-latent detail frame into a verdict.

Three things a summary table on its own cannot say:

* **is the gap real?**  Every adapter is scored on the same ligand, the same
  repeat and the same evaluation rows, so the comparison is paired and the right
  uncertainty is a bootstrap over *ligands* -- one ligand, one vote, the macro
  convention this repo has used since gen2, because one extractant is 43% of the
  pairs and a pooled bootstrap would be a referendum on that one molecule;
* **where does the gap come from?**  Splitting the held-out ligands by whether
  they carry a real concentration titration says whether the win is the physics
  or just a better-regularised offset;
* **what did the model choose to be?**  The per-fold ``trust`` and ``spread`` say
  whether the chemistry prior was used at all.

Usage
-----
    python scripts/gen8_physics_latent_report.py --tag allseeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gen8_architecture/physics_latent"
COHORT = ROOT / "runs/gen7_architecture/cache/cohort.parquet"
X_EXT = "massact__log10_cond__extractant_concentration_M"
X_ACID = "massact__log10_cond__acid_concentration_M"


def per_ligand(detail: pd.DataFrame, policy: str) -> pd.DataFrame:
    frame = detail[(detail["policy"] == policy) | (detail["policy"] == "NONE")]
    return frame.groupby(["adapter", "k", "extractant"])["mae"].mean().reset_index()


def paired_bootstrap(table: pd.DataFrame, a: str, b: str, k: int, *, draws: int = 4000,
                     seed: int = 20260820) -> dict:
    """Bootstrap the macro-MAE difference ``a - b`` over the ligands both were scored on."""
    left = table[(table.adapter == a) & (table.k == k)].set_index("extractant")["mae"]
    right = table[(table.adapter == b) & (table.k == k)].set_index("extractant")["mae"]
    common = left.index.intersection(right.index)
    if len(common) < 8:
        return {"n_ligands": int(len(common))}
    difference = (left.loc[common] - right.loc[common]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(difference), size=(draws, len(difference)))
    means = difference[index].mean(axis=1)
    return {
        "a": a, "b": b, "k": int(k), "n_ligands": int(len(common)),
        "mae_a": float(left.loc[common].mean()), "mae_b": float(right.loc[common].mean()),
        "delta": float(difference.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_a_better": float((means < 0).mean()),
        "n_lig_a_better": int((difference < 0).sum()),
    }


def titration_flags(cohort: pd.DataFrame) -> pd.DataFrame:
    """Per ligand: does it carry the titration that makes ``n_L`` / ``m_L`` identifiable?"""
    def distinct(series: pd.Series) -> int:
        return int(series.dropna().round(6).nunique())

    grouped = cohort.groupby("extractant")
    return pd.DataFrame({
        "n_rows": grouped.size(),
        "ext_levels": grouped[X_EXT].apply(distinct),
        "acid_levels": grouped[X_ACID].apply(distinct),
        "n_series": grouped["series_id"].nunique(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="allseeds")
    parser.add_argument("--policy", default="RANDOM")
    parser.add_argument("--method", default="PHYS_residual")
    args = parser.parse_args()

    detail = pd.read_parquet(OUT / f"physics_latent_detail_{args.tag}.parquet")
    cohort = pd.read_parquet(COHORT)
    table = per_ligand(detail, args.policy)

    order = ["ZERO_SHOT_REF", "PHYS_prior", "PHYS_residual_prior", "NO_MODEL",
             "NEAREST_OBSERVED", "OFFSET_K1", "OFFSET_K2", "OFFSET_K3", "PHYS_map",
             "PHYS_map_metalcat", "PHYS_residual", "PHYS_residual_metalcat",
             "PHYS_residual_hypernet", "PHYS_residual_gbm"]
    grid = (table.groupby(["adapter", "k"])["mae"].mean().reset_index()
            .pivot(index="adapter", columns="k", values="mae"))
    counts = (table.groupby(["adapter", "k"])["extractant"].nunique().reset_index()
              .pivot(index="adapter", columns="k", values="extractant"))
    grid = grid.reindex([a for a in order if a in grid.index])
    print(f"\n=== macro MAE by k, policy={args.policy}, tag={args.tag} ===")
    print(grid.round(4).to_string())
    print("\n=== ligands scored ===")
    print(counts.reindex(grid.index).to_string())

    print(f"\n=== paired bootstrap over ligands: {args.method} vs the references ===")
    rows = []
    for reference in ("OFFSET_K1", "OFFSET_K3", "NO_MODEL", "NEAREST_OBSERVED"):
        for k in (1, 2, 3, 5):
            result = paired_bootstrap(table, args.method, reference, k)
            if result.get("n_ligands", 0) >= 8:
                rows.append(result)
    for other in ("PHYS_residual_gbm", "PHYS_residual_hypernet", "PHYS_residual_metalcat",
                  "PHYS_map"):
        for k in (1, 3):
            result = paired_bootstrap(table, other, args.method, k)
            if result.get("n_ligands", 0) >= 8:
                rows.append(result)
    bootstrap = pd.DataFrame(rows)
    print(bootstrap.round(4).to_string(index=False))

    zero = []
    for arm in ("PHYS_prior", "PHYS_residual_prior"):
        left = table[(table.adapter == arm) & (table.k == 1)].set_index("extractant")["mae"]
        right = table[table.adapter == "ZERO_SHOT_REF"].set_index("extractant")["mae"]
        common = left.index.intersection(right.index)
        zero.append({"arm": arm, "mae": float(left.loc[common].mean()),
                     "zero_shot": float(right.loc[common].mean()),
                     "delta": float((left.loc[common] - right.loc[common]).mean()),
                     "n_ligands": int(len(common))})
    print("\n=== zero-shot (k=0) arms against the frozen global model ===")
    print(pd.DataFrame(zero).round(4).to_string(index=False))

    print("\n=== stratified by whether the ligand can identify its own exponents ===")
    flags = titration_flags(cohort)
    strata = []
    for k in (1, 3):
        for label, mask in (("acid titration >=3 levels", flags.acid_levels >= 3),
                            ("no acid titration", flags.acid_levels < 3),
                            ("extractant titration >=3 levels", flags.ext_levels >= 3),
                            ("no extractant titration", flags.ext_levels < 3),
                            ("single series", flags.n_series == 1),
                            ("multi series", flags.n_series > 1)):
            keep = set(flags.index[mask])
            sub = table[table.extractant.isin(keep)]
            result = paired_bootstrap(sub, args.method, "OFFSET_K1", k)
            if result.get("n_ligands", 0) >= 8:
                strata.append({"stratum": label, **result})
    strata_frame = pd.DataFrame(strata)
    print(strata_frame.round(4).to_string(index=False))

    print("\n=== identifiability of the held-out ligands (the honest failure) ===")
    scored = set(table.extractant.unique())
    sub = flags.loc[[i for i in flags.index if i in scored]]
    print(pd.DataFrame({
        "n_ligands_scored": [len(sub)],
        "with_ext_titration_ge3": [int((sub.ext_levels >= 3).sum())],
        "with_acid_titration_ge3": [int((sub.acid_levels >= 3).sum())],
        "with_both": [int(((sub.ext_levels >= 3) & (sub.acid_levels >= 3)).sum())],
        "with_neither": [int(((sub.ext_levels < 3) & (sub.acid_levels < 3)).sum())],
        "single_series": [int((sub.n_series == 1).sum())],
    }).to_string(index=False))

    state_path = OUT / f"physics_latent_foldstate_{args.tag}.json"
    if state_path.exists():
        state = pd.DataFrame(json.loads(state_path.read_text()))
        print("\n=== what the inner loop chose, per fold ===")
        chosen = (state.groupby(["target", "map_kind", "metal_basis"])
                  .agg(trust=("trust", "mean"), spread=("spread", "median"),
                       noise=("noise", "mean"), folds=("fold", "size")).reset_index())
        print(chosen.round(3).to_string(index=False))
        rbf = state[(state.metal_basis == "rbf") & (state.map_kind == "ridge")]
        names = rbf["names"].iloc[0]
        slopes = np.vstack(rbf["map_slope"].to_list())[:, :len(names)]
        print("\n=== chemistry -> theta map: out-of-sample skill per coefficient ===")
        print("(a recalibration slope of 0 means the map had no out-of-sample skill and "
              "the prior fell back to the population value)")
        print(pd.DataFrame({"coefficient": names,
                            "mean_recalibration_slope": slopes.mean(axis=0),
                            "folds_with_zero_slope": (slopes <= 1e-9).sum(axis=0),
                            "n_folds": len(slopes)}).round(3).to_string(index=False))

    bootstrap.to_csv(OUT / f"physics_latent_bootstrap_{args.tag}.csv", index=False)
    strata_frame.to_csv(OUT / f"physics_latent_strata_{args.tag}.csv", index=False)
    grid.to_csv(OUT / f"physics_latent_grid_{args.tag}.csv")
    print("\nwrote", OUT / f"physics_latent_bootstrap_{args.tag}.csv")


if __name__ == "__main__":
    main()
