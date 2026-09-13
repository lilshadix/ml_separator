#!/usr/bin/env python
"""Phase 14: the statistical caveats that qualify every significance claim here.

Three things a table of BCa intervals does not say, and that this study must:

* **How many tests were run.** One primary endpoint was pre-registered and is protected by
  that. Everything else is unadjusted, and the count is published rather than left implicit.
* **Whether the BCa interval and the percentile interval agree.** They come from the same
  10,000 draws. Where they disagree, the bias-correction and acceleration terms are doing
  the work, and the two-sided bootstrap p-value is the honest summary.
* **How much a significant estimate is inflated by having been selected for significance.**
  At the realised power, a significant effect is systematically larger than the truth
  (Gelman and Carlin's Type-M error). It is computed here for the primary endpoint.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_statistical_caveats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402

ALPHA = 0.05


def type_m(effect: float, se: float, *, draws: int = 2_000_000, seed: int = 8675309) -> dict:
    """Expected exaggeration of an estimate conditional on clearing significance.

    Assume the truth is the observed effect and the sampling sd is the observed bootstrap
    SE.  Simulate estimates, keep those that clear a two-sided alpha test, and report the
    mean ratio of |estimate| to |truth| and the probability the surviving estimate has the
    wrong sign.
    """
    from scipy.stats import norm
    if not np.isfinite(effect) or not np.isfinite(se) or se <= 0:
        return {"power": np.nan, "type_m": np.nan, "type_s": np.nan}
    critical = norm.ppf(1 - ALPHA / 2) * se
    rng = np.random.default_rng(seed)
    sample = rng.normal(effect, se, draws)
    significant = np.abs(sample) > critical
    if not significant.any():
        return {"power": 0.0, "type_m": np.nan, "type_s": np.nan}
    kept = sample[significant]
    return {"power": float(significant.mean()),
            "type_m": float(np.abs(kept).mean() / abs(effect)),
            "type_s": float((np.sign(kept) != np.sign(effect)).mean())}


def main() -> int:
    rows, tests, flagged = [], 0, 0
    disagreements = []
    # This script's own outputs live in bootstrap/ and carry a ``bca_excludes_zero``
    # column, so scanning the directory naively counts them and the total grows on every
    # re-run.  They are excluded by name.
    own = {"statistical_caveats.csv", "interval_disagreements.csv"}
    for path in sorted(paths.BOOTSTRAP_DIR.glob("*.csv")):
        if path.name in own:
            continue
        table = pd.read_csv(path)
        if "bca_excludes_zero" not in table.columns:
            continue
        tests += len(table)
        flagged += int(table["bca_excludes_zero"].sum())
        if {"ci95_low", "ci95_high"} <= set(table.columns):
            block = table[table["bca_excludes_zero"]
                          & (table["ci95_low"] <= 0) & (table["ci95_high"] >= 0)].copy()
            if len(block):
                block["source"] = path.name
                disagreements.append(block)
    disagreement = (pd.concat(disagreements, ignore_index=True) if disagreements
                    else pd.DataFrame())

    primary = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_pairwise.csv")
    primary = primary[(primary["statistic"] == "mae")]
    for _, record in primary.iterrows():
        magnitude = type_m(record["point_delta"], record["bootstrap_se"])
        rows.append({
            "contrast": record["contrast"], "cohort": record["cohort"],
            "point_delta": record["point_delta"], "bootstrap_se": record["bootstrap_se"],
            "ci95_low": record["ci95_low"], "ci95_high": record["ci95_high"],
            "bca_low": record["bca_low"], "bca_high": record["bca_high"],
            "bca_excludes_zero": record["bca_excludes_zero"],
            "percentile_excludes_zero": bool(record["ci95_low"] > 0
                                             or record["ci95_high"] < 0),
            "p_two_sided": record["p_two_sided"],
            "mde_80pct": 2.80 * record["bootstrap_se"],
            **magnitude,
        })
    table = pd.DataFrame(rows)
    table["intervals_agree"] = table["bca_excludes_zero"] == table["percentile_excludes_zero"]
    table.to_csv(paths.BOOTSTRAP_DIR / "statistical_caveats.csv", index=False)

    summary = {
        "n_bootstrap_rows_emitted": int(tests),
        "n_flagged_bca_excludes_zero": int(flagged),
        "multiple_comparison_note": (
            "One primary endpoint was pre-registered (PRIMARY_G_vs_D, level MAE, full cohort) "
            "and is protected by that pre-registration. Every other contrast in this "
            "directory is unadjusted and is reported as exploratory."),
        "n_contrasts_where_bca_and_percentile_disagree": int(len(disagreement)),
        "primary": table[(table["contrast"] == "PRIMARY_G_vs_D")
                         & (table["cohort"] == "full")].to_dict(orient="records"),
    }
    (paths.BOOTSTRAP_DIR / "statistical_caveats.json").write_text(
        json.dumps(summary, indent=1, default=float))
    if len(disagreement):
        disagreement.to_csv(paths.BOOTSTRAP_DIR / "interval_disagreements.csv", index=False)

    show = ["contrast", "cohort", "point_delta", "ci95_low", "ci95_high", "bca_low",
            "bca_high", "p_two_sided", "intervals_agree", "power", "type_m", "mde_80pct"]
    print(table[show].round(4).to_string(index=False))
    print(f"\n{tests} bootstrap rows emitted across bootstrap/, {flagged} flagged as "
          f"excluding zero, {len(disagreement)} of them with a percentile interval that "
          "includes zero.")
    print(json.dumps({k: v for k, v in summary.items() if k != "primary"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
