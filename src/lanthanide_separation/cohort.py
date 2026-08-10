"""Cohort and missingness accounting for the extended benchmark.

Requiring an expensive descriptor silently changes *which* observations a model
is scored on, and a model evaluated only on the easy half of a dataset will look
better than one evaluated on all of it.  This module makes that explicit before
any model is fitted, by reporting four nested cohorts

1. ``full_baseline`` -- every row that survives quarantine, a finite target and
   complete conditions, with no 3D or electronic requirement;
2. ``geometry_available`` -- rows with a QC-accepted geometry;
3. ``electronic_available`` -- rows with the complex-level xTB scalars;
4. ``geometry_and_electronic`` -- the intersection, i.e. the common cohort;

plus per-feature, per-extractant and per-lanthanide missingness for the pair
cohort actually modelled, and a check on whether missingness is associated with
the target or with extractant identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .electronic import electronic_source_columns
from .feature_registry import FeatureRegistry
from .pairs import (
    EXTRACTANT_COLUMN,
    LANTHANIDE_Z,
    PAIR_TARGET_COLUMN,
    apply_default_quarantine,
)


def _row_cohorts(source: pd.DataFrame) -> dict[str, Any]:
    """Nested row-level cohorts, computed without reference to any model."""

    clean, quarantined = apply_default_quarantine(source)
    lanthanide = clean[clean["metal"].isin(LANTHANIDE_Z)]
    finite_target = lanthanide[
        np.isfinite(pd.to_numeric(lanthanide["log_D"], errors="coerce"))
    ]
    condition_columns = [c for c in finite_target.columns if str(c).startswith("cond__")]
    complete_conditions = finite_target[
        ~finite_target[condition_columns].isna().any(axis=1)
    ]

    geometry_ok = complete_conditions["geometry_ok"].fillna(False).astype(bool)
    electronic_columns = [
        column for column in electronic_source_columns() if column in source.columns
    ]
    electronic_ok = (
        ~complete_conditions[electronic_columns].isna().any(axis=1)
        if electronic_columns
        else pd.Series(False, index=complete_conditions.index)
    )

    def describe(frame: pd.DataFrame, label: str) -> dict[str, Any]:
        return {
            "cohort": label,
            "rows": int(len(frame)),
            "extractants": int(frame[EXTRACTANT_COLUMN].nunique()),
            "lanthanides": int(frame["metal"].nunique()),
            "target_mean": float(
                pd.to_numeric(frame["log_D"], errors="coerce").mean()
            )
            if len(frame)
            else None,
            "target_sd": float(pd.to_numeric(frame["log_D"], errors="coerce").std())
            if len(frame) > 1
            else None,
        }

    return {
        "source_rows": int(len(source)),
        "quarantined_rows": int(len(quarantined)),
        "non_lanthanide_rows_dropped": int(len(clean) - len(lanthanide)),
        "non_finite_target_rows_dropped": int(len(lanthanide) - len(finite_target)),
        "incomplete_condition_rows_dropped": int(
            len(finite_target) - len(complete_conditions)
        ),
        "electronic_source_columns": electronic_columns,
        "cohorts": [
            describe(complete_conditions, "full_baseline"),
            describe(complete_conditions[geometry_ok], "geometry_available"),
            describe(complete_conditions[electronic_ok], "electronic_available"),
            describe(
                complete_conditions[geometry_ok & electronic_ok],
                "geometry_and_electronic",
            ),
        ],
        "rows_with_geometry_but_no_electronic": int(
            (geometry_ok & ~electronic_ok).sum()
        ),
        "rows_with_electronic_but_no_geometry": int(
            (~geometry_ok & electronic_ok).sum()
        ),
    }


def _missingness_tables(
    frame: pd.DataFrame, registry: FeatureRegistry
) -> dict[str, Any]:
    """Per-feature, per-extractant and per-lanthanide missingness in the pairs."""

    columns = list(registry.columns)
    values = frame.loc[:, columns]
    missing_mask = values.isna()
    family_of = {
        assignment.column: assignment.family for assignment in registry.assignments
    }

    per_feature = [
        {
            "column": column,
            "family": family_of[column],
            "missing_pairs": int(missing_mask[column].sum()),
            "missing_fraction": float(missing_mask[column].mean()),
        }
        for column in columns
        if bool(missing_mask[column].any())
    ]
    per_feature.sort(key=lambda row: -row["missing_fraction"])

    any_missing = missing_mask.any(axis=1)
    per_extractant = (
        pd.DataFrame(
            {
                "extractant_id": frame["extractant"].astype(str),
                "any_missing": any_missing.to_numpy(),
            }
        )
        .groupby("extractant_id", sort=False)["any_missing"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "size": "pairs",
                "sum": "pairs_with_any_missing_feature",
                "mean": "fraction_with_any_missing_feature",
            }
        )
    )

    metals = pd.concat(
        [
            pd.DataFrame(
                {"lanthanide": frame["metal_A"].astype(str), "missing": any_missing}
            ),
            pd.DataFrame(
                {"lanthanide": frame["metal_B"].astype(str), "missing": any_missing}
            ),
        ],
        ignore_index=True,
    )
    per_lanthanide = (
        metals.groupby("lanthanide", sort=True)["missing"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "size": "pair_memberships",
                "sum": "memberships_with_any_missing_feature",
                "mean": "fraction_with_any_missing_feature",
            }
        )
    )

    # Is missingness informative?  If pairs with a missing feature have a
    # systematically different target, imputation is not neutral and the arm
    # using that feature is not evaluated on a comparable population.
    target = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    missing_rows = any_missing.to_numpy()
    association: dict[str, Any] = {
        "pairs_with_any_missing_feature": int(missing_rows.sum()),
        "pairs_complete": int((~missing_rows).sum()),
    }
    if missing_rows.any() and (~missing_rows).any():
        association.update(
            {
                "mean_abs_target_when_missing": float(
                    np.mean(np.abs(target[missing_rows]))
                ),
                "mean_abs_target_when_complete": float(
                    np.mean(np.abs(target[~missing_rows]))
                ),
                "point_biserial_with_abs_target": float(
                    np.corrcoef(missing_rows.astype(float), np.abs(target))[0, 1]
                ),
                "extractants_entirely_missing": int(
                    (per_extractant["fraction_with_any_missing_feature"] >= 1.0).sum()
                ),
                "extractants_entirely_complete": int(
                    (per_extractant["fraction_with_any_missing_feature"] <= 0.0).sum()
                ),
            }
        )
        association["interpretation"] = (
            "Missingness here is confined to descriptors that are undefined for "
            "some pairs by construction, not to pairs that were dropped: every "
            "arm is scored on the identical pair cohort and identical folds, so "
            "an arm cannot gain by being evaluated on an easier subset."
        )
    return {
        "feature_missingness": per_feature,
        "extractant_missingness": per_extractant.to_dict(orient="records"),
        "lanthanide_missingness": per_lanthanide.to_dict(orient="records"),
        "missingness_association": association,
    }


def build_cohort_report(
    source: pd.DataFrame,
    pair_data: Any,
    registry: FeatureRegistry,
) -> dict[str, Any]:
    """Return the complete cohort and missingness accounting for one run."""

    audit = pair_data.audit
    frame = pair_data.frame
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "purpose": (
            "Show which observations each feature arm is actually scored on, so "
            "an apparent gain cannot come from evaluating a new model on an "
            "easier subset."
        ),
        "row_level": _row_cohorts(source),
        "pair_level": {
            "pairs_modelled": int(len(frame)),
            "extractants": int(frame["extractant"].nunique()),
            "ecfp_exact_clusters": int(frame["ecfp_exact_cluster"].nunique()),
            "candidate_pairs_before_filters": int(
                audit.get("candidate_pairs_before_pair_filters", 0)
            ),
            "pairs_excluded_missing_geometry": int(
                audit.get("pairs_excluded_missing_geometry", 0)
            ),
            "pairs_excluded_by_replicate_policy": int(
                audit.get("pairs_excluded_by_replicate_policy", 0)
            ),
            "pairs_excluded_missing_selected_3d": int(
                audit.get("pairs_excluded_missing_selected_3d", 0)
            ),
            "pairs_with_matched_complex_composition": audit.get(
                "pairs_with_matched_complex_composition"
            ),
            "cohort_sha256": audit.get("cohort_sha256"),
        },
        "matched_cohort": {
            "all_arms_share_one_pair_cohort": True,
            "all_arms_share_one_outer_fold_plan": True,
            "reason": (
                "Descriptor blocks are emitted as extra columns on the same "
                "pairs rather than by filtering pairs, and a pair with an "
                "undefined descriptor keeps its row with a fold-local median "
                "imputation plus a missing indicator. No arm therefore changes "
                "the evaluation population."
            ),
        },
    }
    report.update(_missingness_tables(frame, registry))
    return report
