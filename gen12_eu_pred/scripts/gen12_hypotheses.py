#!/usr/bin/env python
"""The pre-registered hypotheses, each tested against the control it actually needs.

One correction is applied here and reported rather than absorbed.  The Tier-0
condition-only baseline ``B1_COND_ONLY`` uses histogram gradient boosting while
the Tier-1 champion uses extremely randomised trees, so the contrast between them
confounds **information** with **learner**.  The matched control is
``ABL_A_CONDITIONS``: the same ExtraTrees on the same folds with the molecular
blocks removed.  Every H1/H2 number below is computed against that, and the
unmatched contrast is reported beside it so the size of the confound is visible.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_hypotheses.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import inference, paths  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402

CANDIDATES = {
    "T1_EXTRATREES": paths.PREDICTION_DIR / "B" / "T1_EXTRATREES.parquet",
    "ABL_D_PLUS_LIG2D": paths.PREDICTION_DIR / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet",
    "T3_DMPNN_COND": paths.PREDICTION_DIR / "B" / "T3_DMPNN_COND.parquet",
    "T3_DMPNN_COND_DESC": paths.PREDICTION_DIR / "B" / "T3_DMPNN_COND_DESC.parquet",
    "T3_DMPNN_GRAPH_ONLY": paths.PREDICTION_DIR / "B" / "T3_DMPNN_GRAPH_ONLY.parquet",
    "B2_NN_CHEMICAL": paths.PREDICTION_DIR / "B" / "B2_NN_CHEMICAL.parquet",
}
CONTROLS = {
    "ABL_A_CONDITIONS": paths.PREDICTION_DIR / "B_ablation" / "ABL_A_CONDITIONS.parquet",
    "B1_COND_ONLY": paths.PREDICTION_DIR / "B" / "B1_COND_ONLY.parquet",
    "B0_GLOBAL_MEAN": paths.PREDICTION_DIR / "B" / "B0_GLOBAL_MEAN.parquet",
}
#: Pre-registered: an effect below this is indistinguishable from a change of machine.
FLOOR = 0.01
EQUIVALENCE_MARGIN = 0.02


def main() -> int:
    frames = {a: pd.read_parquet(p) for a, p in {**CONTROLS, **CANDIDATES}.items() if p.exists()}
    units = {a: per_extractant(f) for a, f in frames.items()}
    rows: list[dict] = []

    for control in [c for c in CONTROLS if c in units]:
        for band in ("overall", "far", "mid", "near"):
            subset = ({a: t for a, t in units.items()} if band == "overall"
                      else {a: t[t["band"] == band] for a, t in units.items()})
            shared = set.intersection(*[set(t["extractant"]) for t in subset.values()])
            subset = {a: t[t["extractant"].isin(shared)] for a, t in subset.items()}
            if min(len(t) for t in subset.values()) < 5:
                continue
            block = inference.unit_table(subset, statistic="mae")
            comparisons = {a: (control, a) for a in subset if a != control}
            table = inference.paired_bootstrap(block, comparisons, statistic="mae")
            mde = {a: inference.minimum_detectable_effect(block, control, a)["mde"]
                   for a in comparisons}
            macro = {a: t.groupby("split_seed")["mae"].mean().mean() for a, t in subset.items()}
            per_seed = {a: t.groupby("split_seed")["mae"].mean() for a, t in subset.items()}
            for _, record in table.iterrows():
                candidate = record["candidate"]
                delta_by_seed = per_seed[control] - per_seed[candidate]
                rows.append({
                    "control": control, "candidate": candidate, "band": band,
                    "control_macro_mae": macro[control],
                    "candidate_macro_mae": macro[candidate],
                    "point_delta": record["point_delta"],
                    "bca_low": record["bca_low"], "bca_high": record["bca_high"],
                    "bca_excludes_zero": record["bca_excludes_zero"],
                    "bootstrap_se": record["bootstrap_se"],
                    "mde_80pct_power": mde[candidate],
                    "seeds_favouring_candidate": int((delta_by_seed > 0).sum()),
                    "n_seeds": int(len(delta_by_seed)),
                    "units_improved": record["units_improved"],
                    "units_total": record["units_total"],
                    "bootstrap_blocks": record["bootstrap_blocks"],
                    "above_reproducibility_floor": bool(abs(record["point_delta"]) > FLOOR),
                    "resolvable": bool(abs(record["point_delta"]) > mde[candidate]),
                    "equivalence_supported": bool(
                        record["bca_low"] > -EQUIVALENCE_MARGIN
                        and record["bca_high"] < EQUIVALENCE_MARGIN),
                })
    table = pd.DataFrame(rows)
    table.to_csv(paths.BOOTSTRAP_DIR / "B_hypotheses.csv", index=False)
    (paths.HEADLINE_DIR / "t9_hypotheses.md").write_text(
        table.round(4).to_markdown(index=False) + "\n")

    view = table[table["control"] == "ABL_A_CONDITIONS"]
    print("=== H1/H2 against the matched no-chemistry control (same learner) ===")
    print(view[["band", "candidate", "control_macro_mae", "candidate_macro_mae", "point_delta",
                "bca_low", "bca_high", "bca_excludes_zero", "mde_80pct_power",
                "seeds_favouring_candidate", "equivalence_supported"]].round(4).to_string(index=False))
    print("\n=== the same contrast against the UNMATCHED control (different learner) ===")
    print(table[(table["control"] == "B1_COND_ONLY") & (table["band"] == "overall")]
          [["candidate", "point_delta", "bca_low", "bca_high", "bca_excludes_zero"]]
          .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
