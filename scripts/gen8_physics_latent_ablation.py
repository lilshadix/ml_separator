#!/usr/bin/env python
"""Attribution: which part of the physics-latent arm is doing the work?

``PHYS_residual`` beats offset correction, but it differs from ``OFFSET_K3`` in
three ways at once, and a win that cannot be attributed is not a finding.  Each
arm below removes exactly one ingredient, on identical rows:

``NOTRUST``
    the chemistry -> theta prior is switched off (``trust_grid = (0.0,)``); the
    inner replay still chooses the prior *strength*.  If this ties the full arm,
    the compact ligand representation contributes nothing and the win is the
    physical design plus a learned regularisation strength.
``FIXED``
    no inner replay at all: no chemistry prior, prior covariance at its estimated
    scale.  Isolates what the ``(trust, spread)`` search buys.
``LEVELONLY`` / ``NOMETAL`` / ``NOMASSACTION``
    the design is cut back to the intercept alone, to the mass-action terms only,
    and to the lanthanide basis only.  ``LEVELONLY`` is the physics-latent spelling
    of ``OFFSET_K1``: whatever it beats ``OFFSET_K1`` by is the learned shrinkage,
    and whatever the full arm beats *it* by is the response law.

Usage
-----
    python scripts/gen8_physics_latent_ablation.py --seeds 104729 130363 --repeats 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "runs/gen8_architecture/physics_latent"
MODEL = "REC_ecfp_plus_recovered"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[104729, 130363])
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()

    from gen8_physics_latent import COHORT, OOF, build_fold_plan
    from lanthanide_separation.gen8.adapters import RidgeOffset, ZeroShot
    from lanthanide_separation.gen8.evaluate import evaluate_fewshot, summarise
    from lanthanide_separation.gen8.physics_latent import PhysicsAdapter, PhysicsLatentFitter

    cohort = pd.read_parquet(COHORT)
    oof = pd.read_parquet(OOF)
    oof = oof[(oof["model"] == MODEL)
              & (oof["split_seed"].isin(args.seeds))].reset_index(drop=True)
    plan = build_fold_plan(cohort, oof, args.seeds)
    frozen = {int(s): oof.loc[oof["split_seed"] == s].set_index("row_id")["prediction"]
              for s in args.seeds}

    def fold_trainer(split_seed: int, fold: int):
        index = plan[(int(split_seed), int(fold))]
        train = cohort.iloc[index].copy()
        train["prediction"] = train["row_id"].map(frozen[int(split_seed)]).to_numpy(dtype=float)
        return train, train["log_D"].to_numpy(dtype=float), 42 + int(fold) * 1009 + 9999991

    adapters = [
        PhysicsAdapter(PhysicsLatentFitter(target="residual"), "PHYS_residual"),
        PhysicsAdapter(PhysicsLatentFitter(target="residual", trust_grid=(0.0,)),
                       "PHYS_residual_NOTRUST"),
        PhysicsAdapter(PhysicsLatentFitter(target="residual", tune=False, trust=0.0,
                                           spread=1.0), "PHYS_residual_FIXED"),
        PhysicsAdapter(PhysicsLatentFitter(target="residual", n_basis=0, terms=(),
                                           trust_grid=(0.0,)), "PHYS_residual_LEVELONLY"),
        PhysicsAdapter(PhysicsLatentFitter(target="residual", n_basis=0,
                                           trust_grid=(0.0,)), "PHYS_residual_NOMETAL"),
        PhysicsAdapter(PhysicsLatentFitter(target="residual", terms=(),
                                           trust_grid=(0.0,)), "PHYS_residual_NOMASSACTION"),
        ZeroShot(), RidgeOffset(mode="K1"), RidgeOffset(mode="K3"),
    ]
    detail = evaluate_fewshot(oof, cohort, adapters, repeats=args.repeats,
                              fold_trainer=fold_trainer, verbose=True)
    summary = summarise(detail)
    table = (summary[summary.policy == "RANDOM"]
             .pivot(index="adapter", columns="k", values="mae"))
    print(table.round(4).to_string())
    detail.to_parquet(OUT / "physics_latent_ablation.parquet", index=False)
    print("wrote", OUT / "physics_latent_ablation.parquet")


if __name__ == "__main__":
    main()
