#!/usr/bin/env python
"""POST-HOC: do four descriptor-definition defects change the primary contrast?

An adversarial review of the descriptor block, run **after** the locked evaluation, found
four rule-level defects: aromatic ring oxygens and aromatic-carbon carbonyls were invisible
to the donor classes, the lactam ring test did not require the carbonyl carbon to be in the
ring, the pyridine-like ring class counted nitrogen by element rather than as ``[nX2]``, and
the named-chelating-unit sum double-counted malonamides while omitting the
P(=S)–CH2–P(=S) motif.

This script rebuilds the block from ``config/coordination_smarts_posthoc.json``, re-runs the
level ladder arms that carry the primary contrast on the **same frozen folds**, and reports
whether the answer moves.

**Nothing here replaces a pre-registered number.** The frozen v1.1.0 matrix still reproduces
bit-identically (hash ``da609be8fcd5f260``) and is what every headline figure is computed
from. This is a sensitivity, labelled as one everywhere it appears.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_descriptor_defect_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, levelmetrics, levelmodels, levelrunner, paths  # noqa: E402
from gen12eu import inference, splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402

ARMS = ("L1_ECFP", "L3_ECFP_GENERIC", "L4_COORD", "L5_ECFP_COORD", "L7_ALL")
CONTRASTS = {"PRIMARY_G_vs_D": ("L3_ECFP_GENERIC", "L7_ALL"),
             "E_vs_A": ("L1_ECFP", "L5_ECFP_COORD"),
             "C_vs_D": ("L3_ECFP_GENERIC", "L4_COORD")}


def main() -> int:
    cohort = build_cohort()
    frame = cohort.frame
    folds = splits.all_folds(frame, design="B")
    similarity = pd.read_parquet(paths.GEN12_PREDICTIONS / "B" / "similarity.parquet")
    similarity["band"] = band_of(similarity["max_train_tanimoto"]).to_numpy()

    frozen = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    if paths.blake2b_of_frame(coordination.build_table(sorted(frozen.index))) != \
            json.loads((paths.FEATURE_DIR / "coordination_audit.json").read_text())[
                "feature_matrix_blake2b"]:
        raise SystemExit("the frozen v1.1.0 matrix no longer reproduces; refusing to proceed")

    coordination.use_spec(paths.CONFIG_DIR / "coordination_smarts_posthoc.json")
    corrected = coordination.build_table(sorted(set(frame["extractant"].astype(str))))
    corrected.to_parquet(paths.FEATURE_DIR / "coordination_descriptors_posthoc.parquet")
    changed = [c for c in corrected.columns if c in frozen.columns
               and not corrected[c].equals(frozen.reindex(corrected.index)[c])]
    added = [c for c in corrected.columns if c not in frozen.columns]

    structure, columns = levelrunner.structure_frame(frame, corrected, cohort.blocks)
    targets = levelrunner.level_targets_by_fold(frame, folds, cohort.blocks,
                                                definition="LVL_MEAN")
    out = paths.PREDICTION_DIR / "level" / "LVL_MEAN_posthoc_spec"
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in ARMS:
        contender = levelmodels.TreeLevel(name=name, blocks=levelmodels.LADDER_BLOCKS[name])
        predictions, selection = levelrunner.run_level_contender(
            contender, frame, structure, columns, folds, similarity,
            definition="LVL_MEAN", blocks=cohort.blocks, targets_by_fold=targets)
        predictions.to_parquet(out / f"{name}.parquet", index=False)
        selection.to_csv(out / f"{name}__selection.csv", index=False)
        frames[name] = predictions
        print(f"{name:20s} level MAE {levelmetrics.summarise(predictions)['level_mae']:.4f}",
              flush=True)

    units = {a: levelmetrics.per_extractant(f) for a, f in frames.items()}
    block = inference.unit_table(units, statistic="mae")
    table = inference.paired_bootstrap(block, CONTRASTS, statistic="mae")
    table.to_csv(paths.BOOTSTRAP_DIR / "posthoc_spec_pairwise.csv", index=False)

    frozen_pairwise = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_pairwise.csv")
    frozen_pairwise = frozen_pairwise[(frozen_pairwise["cohort"] == "full")
                                      & (frozen_pairwise["statistic"] == "mae")]
    comparison = []
    for _, record in table.iterrows():
        prior = frozen_pairwise[frozen_pairwise["contrast"] == record["comparison"]]
        comparison.append({
            "contrast": record["comparison"],
            "frozen_delta": float(prior["point_delta"].iloc[0]) if len(prior) else np.nan,
            "frozen_bca": [float(prior["bca_low"].iloc[0]),
                           float(prior["bca_high"].iloc[0])] if len(prior) else None,
            "frozen_p": float(prior["p_two_sided"].iloc[0]) if len(prior) else np.nan,
            "posthoc_delta": float(record["point_delta"]),
            "posthoc_bca": [float(record["bca_low"]), float(record["bca_high"])],
            "posthoc_ci95": [float(record["ci95_low"]), float(record["ci95_high"])],
            "posthoc_p": float(record["p_two_sided"]),
            "sign_unchanged": bool(np.sign(record["point_delta"])
                                   == np.sign(prior["point_delta"].iloc[0])) if len(prior) else None,
        })
    summary = {
        "status": "POST-HOC sensitivity; does not replace any pre-registered number",
        "frozen_spec_sha256": paths.sha256_of(paths.COORDINATION_SMARTS_JSON),
        "posthoc_spec_sha256": paths.sha256_of(
            paths.CONFIG_DIR / "coordination_smarts_posthoc.json"),
        "n_columns_changed": len(changed), "n_columns_added": len(added),
        "columns_added": added,
        "level_mae": {a: float(levelmetrics.summarise(f)["level_mae"])
                      for a, f in frames.items()},
        "contrasts": comparison,
    }
    (paths.ANALYSIS_DIR / "descriptor_defect_sensitivity.json").write_text(
        json.dumps(summary, indent=1, default=float))
    lines = ["# Post-hoc: do the descriptor-definition defects change the answer?\n",
             "*Found by adversarial review **after** the locked evaluation. The frozen "
             "v1.1.0 matrix still reproduces bit-identically and remains the basis of every "
             "headline number; this page is a sensitivity and is labelled as one.*\n",
             f"The corrections change **{len(changed)} of 114** columns and add "
             f"{len(added)} ({', '.join(added)}).\n",
             "## Level MAE, zero-shot, design B, full cohort\n",
             pd.DataFrame({"posthoc_spec": summary["level_mae"]}).round(4).to_markdown() + "\n",
             "## The contrasts, frozen against post-hoc\n",
             pd.DataFrame(comparison).round(4).to_markdown(index=False) + "\n"]
    (paths.ANALYSIS_DIR / "descriptor_defect_sensitivity.md").write_text("\n".join(lines))
    coordination.use_spec(paths.COORDINATION_SMARTS_JSON)
    print(pd.DataFrame(comparison).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
