#!/usr/bin/env python
"""Phase 13: grouped permutation importance on the level task.

**Interpretation only.**  The feature set was fixed by the pre-registration and nothing
here selects anything; the pre-registered ablation contrasts in
``bootstrap/level_*_pairwise.csv`` are what carry the evidence.  Individual correlated
descriptors are not interpreted — the unit is the descriptor family.

The model is refitted per fold exactly as the ladder fits it, then each family's columns
are permuted **among the held-out extractants of that fold** and the increase in level MAE
recorded.  Permuting within the held-out set keeps the marginal distribution intact.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_importance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, levelmodels, levelrunner, paths  # noqa: E402
from gen12eu import splits  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402
from gen12eu.preprocess import FoldPreprocessor  # noqa: E402

N_REPEATS = 10
ARM = "L7_ALL"


def main() -> int:
    cohort = build_cohort()
    frame = cohort.frame
    folds = splits.all_folds(frame, design="B")
    table = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    structure, columns = levelrunner.structure_frame(frame, table, cohort.blocks)
    targets = levelrunner.level_targets_by_fold(frame, folds, cohort.blocks,
                                                definition="LVL_MEAN")

    groups: dict[str, tuple[str, ...]] = {
        "ECFP": tuple(columns["ECFP"]),
        "generic_rdkit_lig2d": tuple(columns["LIG2D"]),
        "generic_physchem": tuple(columns["PHYSCHEM"]),
        "gen12_donor_census": tuple(columns["DONORS"]),
    }
    for family, cols in coordination.columns_by_family(tuple(columns["COORD"])).items():
        groups[f"coord_{family}"] = cols

    arm = levelmodels.TreeLevel(name=ARM, blocks=levelmodels.LADDER_BLOCKS[ARM])
    all_columns = []
    for block in arm.blocks:
        all_columns.extend(columns[block])
    all_columns = list(dict.fromkeys(all_columns))

    records: list[dict] = []
    for fold in folds:
        outer, inner = targets[(fold.seed, fold.fold)]
        train_names, test_names = list(outer.train.index), list(outer.test.index)
        context = levelmodels.LevelContext(
            columns=columns, model_seed=fold.model_seed,
            inner_train=structure.loc[list(inner.train.index)],
            inner_validation=structure.loc[list(inner.test.index)],
            alpha_inner_train=inner.train.to_numpy(dtype=float),
            alpha_inner_validation=inner.test.to_numpy(dtype=float),
            split_seed=fold.seed, fold=fold.fold)
        arm.fit_predict(structure.loc[train_names], outer.train.to_numpy(dtype=float),
                        structure.loc[test_names], context)
        model, pre = arm.model_, arm.preprocessor_
        test_block = structure.loc[test_names].copy()
        truth = outer.test.reindex(test_names).to_numpy(dtype=float)
        base = float(np.abs(model.predict(pre.transform(test_block)) - truth).mean())
        rng = np.random.default_rng(fold.model_seed)
        for name, cols in groups.items():
            present = [c for c in cols if c in all_columns]
            if not present:
                continue
            scores = []
            for _ in range(N_REPEATS):
                shuffled = test_block.copy()
                order = rng.permutation(len(shuffled))
                shuffled[present] = shuffled[present].to_numpy()[order]
                scores.append(float(np.abs(model.predict(pre.transform(shuffled))
                                           - truth).mean()))
            records.append({"split_seed": fold.seed, "fold": fold.fold, "group": name,
                            "n_columns": len(present), "base_level_mae": base,
                            "permuted_level_mae": float(np.mean(scores)),
                            "importance": float(np.mean(scores) - base)})
    detail = pd.DataFrame(records)
    summary = (detail.groupby("group")
               .agg(n_columns=("n_columns", "first"),
                    base_level_mae=("base_level_mae", "mean"),
                    permuted_level_mae=("permuted_level_mae", "mean"),
                    importance=("importance", "mean"),
                    importance_sd_over_folds=("importance", "std"),
                    folds_positive=("importance", lambda s: int((s > 0).sum())),
                    n_folds=("importance", "size"))
               .sort_values("importance", ascending=False).reset_index())
    out = paths.METRIC_DIR / "level"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "grouped_permutation_detail.csv", index=False)
    summary.to_csv(out / "grouped_permutation_importance.csv", index=False)
    (out / "grouped_permutation_meta.json").write_text(json.dumps(
        {"arm": ARM, "n_repeats": N_REPEATS, "definition": "LVL_MEAN",
         "note": "interpretation only; no feature set was selected from this table"}, indent=1))
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
