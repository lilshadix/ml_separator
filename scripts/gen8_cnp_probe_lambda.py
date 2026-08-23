#!/usr/bin/env python
"""Read the shrinkage schedule ``CNPRES_shrinkage`` actually learned, off the network.

The arm's whole claim is that it learned *how much to trust k measurements*.  Its
head sees only ``[log1p(k)/2, mean, sd]``, so the function it computes can be read
directly: feed a synthetic context summary and see what correction comes out.  A
pure shrinkage is ``mu = lambda_k * mean`` — a line through the origin whose slope
depends on k.  Anything else (a non-zero intercept, a slope above 1, a dependence
on sd) is not shrinkage.

Reported next to it: the shrinkage a two-parameter grid search finds on the same
training rows, and the James-Stein value implied by the fold's own residual
variance decomposition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen8.cnp import CNPAdapter, _summary_stats  # noqa: E402
from gen8_cnp_verify import ShrinkAnalytic, ShrinkOffset, load_cohort  # noqa: E402

CACHE = ROOT / "runs" / "gen7_architecture" / "cache"
FINALISTS = ROOT / "runs" / "gen7_architecture" / "finalists"
MODEL = "REC_ecfp_plus_recovered"


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 104729
    folds = [int(f) for f in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2500

    oof = pd.read_parquet(FINALISTS / "oof_predictions.parquet")
    oof = oof[(oof["model"] == MODEL) & (oof["split_seed"] == seed)]
    cohort = load_cohort()
    identity = pd.read_parquet(CACHE / "cohort.parquet",
                               columns=["row_id", "extractant", "tanimoto_cluster", "log_D"])
    groups = identity["tanimoto_cluster"].astype(str).to_numpy()
    prediction = oof.set_index("row_id")["prediction"]

    plan = list(seeded_group_kfold(groups, 5, seed))
    rows = []
    for fold in folds:
        index = plan[fold][0]
        frame = cohort.iloc[index].copy()
        frame["extractant"] = identity["extractant"].to_numpy()[index]
        frame["tanimoto_cluster"] = identity["tanimoto_cluster"].to_numpy()[index]
        frame["prediction"] = prediction.reindex(frame["row_id"]).to_numpy()
        y = identity["log_D"].to_numpy(dtype=float)[index]
        model_seed = 42 + fold * 1009 + 9999991

        adapter = CNPAdapter(name="CNPRES_shrinkage", use_ligand=False, shrinkage_only=True,
                             residual=True, anchor=True, relative=True, steps=steps, threads=1)
        adapter.fit_fold(frame, y, split_seed=seed, fold=fold, model_seed=model_seed)
        fitted = ShrinkOffset()
        fitted.fit_fold(frame, y, split_seed=seed, fold=fold, model_seed=model_seed)
        analytic = ShrinkAnalytic()
        analytic.fit_fold(frame, y, split_seed=seed, fold=fold, model_seed=model_seed)

        # Read the head: mu(k, m) for a grid of mean residuals m, sd fixed at the
        # training within-ligand spread.  Slope of mu against m is the network's
        # effective lambda; the intercept is the unconditional drift it added.
        scale = adapter.y_scale
        m_grid = np.linspace(-2.0, 2.0, 41)
        for k in (0, 1, 2, 3, 5):
            summary = np.stack([_summary_stats(np.full(max(k, 1), m / scale))
                                if k else np.zeros(3, dtype=np.float32) for m in m_grid])
            if k:
                summary[:, 0] = np.log1p(k) / 2.0
                summary[:, 2] = float(np.sqrt(analytic.s2_within)) / scale if k > 1 else 0.0
            with torch.no_grad():
                out = adapter.net.decoder(torch.from_numpy(summary.astype(np.float32)))
                mu = out[:, 0].numpy() + (summary[:, 1] if adapter.net.use_anchor else 0.0)
            correction = mu * scale
            slope, intercept = np.polyfit(m_grid, correction, 1) if k else (0.0, float(correction.mean()))
            rows.append({"fold": fold, "k": k, "cnp_lambda": float(slope),
                         "cnp_intercept": float(intercept),
                         "fitted_lambda": fitted.lam.get(k, np.nan) if k else np.nan,
                         "analytic_lambda": (analytic.s2_level /
                                             (analytic.s2_level + analytic.s2_within / k))
                         if k else np.nan,
                         "best_step": adapter.history[0]})
        print(f"fold {fold}: best_step={adapter.history[0]} "
              f"fitted_lambda={fitted.lam} "
              f"s2_level={analytic.s2_level:.3f} s2_within={analytic.s2_within:.3f}", flush=True)

    table = pd.DataFrame(rows)
    print()
    print(table.round(4).to_string(index=False))
    print("\nmean over folds:")
    print(table.groupby("k")[["cnp_lambda", "cnp_intercept", "fitted_lambda",
                              "analytic_lambda"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
