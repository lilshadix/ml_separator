#!/usr/bin/env python
"""Phase 2: build the coordination-topology feature matrix and audit it.

Structure in, descriptors out.  No target, no condition, no fold and no metal is
read by anything in this script, so the matrix it writes cannot depend on an
outcome.  The audit it prints is the target-free part of
``COORDINATION_DESCRIPTOR_SPEC.md``.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_build_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, paths  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402


def main() -> int:
    cohort = build_cohort()
    frame = cohort.frame
    structures = sorted(set(frame["extractant"].astype(str)))
    table = coordination.build_table(structures)

    # ---- determinism: rebuilding from a shuffled input must be bit-identical - #
    import random
    shuffled = list(structures)
    random.Random(20260905).shuffle(shuffled)
    again = coordination.build_table(shuffled)
    if not table.equals(again.reindex(table.index)[table.columns]):
        raise AssertionError("the coordination table depends on input order")

    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        columns = sorted({table.columns[j] for _, j in zip(*np.where(~np.isfinite(values)))})
        raise AssertionError(f"non-finite coordination descriptor: {columns}")

    constant = [c for c in table.columns if table[c].nunique(dropna=True) <= 1]
    informative = [c for c in table.columns if c not in set(constant)]
    families = coordination.columns_by_family(tuple(table.columns))

    audit = {
        "spec_version": coordination.spec()["spec_version"],
        "spec_sha256": coordination.spec_digest(),
        "n_structures": len(structures),
        "n_columns_defined": int(table.shape[1]),
        "n_columns_informative": len(informative),
        "constant_columns": constant,
        "columns_by_family": {k: len(v) for k, v in families.items()},
        "informative_by_family": {
            k: len([c for c in v if c in set(informative)]) for k, v in families.items()},
        "feature_matrix_blake2b": paths.blake2b_of_frame(table),
        "max_abs_value": float(np.abs(values).max()),
        "max_abs_by_family": {
            k: float(np.abs(table[list(v)].to_numpy(dtype=float)).max()) for k, v in families.items()},
        "dynamic_range_ratio_by_family": {},
        "cohort_fingerprint": cohort.fingerprint,
    }
    for family, columns in families.items():
        block = np.abs(table[list(columns)].to_numpy(dtype=float))
        nonzero = block[block > 0]
        audit["dynamic_range_ratio_by_family"][family] = float(
            block.max() / np.median(nonzero)) if nonzero.size else 0.0

    # ---- the heavy-tail guard Gen12 needed, applied to this block ----------- #
    ranges = {}
    for column in informative:
        series = table[column].astype(float)
        low, high = float(series.min()), float(series.max())
        spread = float(series.std(ddof=0))
        ranges[column] = {"min": low, "max": high, "sd": spread,
                          "max_over_sd": float(abs(high) / spread) if spread > 1e-12 else 0.0}
    worst = sorted(ranges.items(), key=lambda kv: -kv[1]["max_over_sd"])[:8]
    audit["largest_max_over_sd"] = {k: v for k, v in worst}
    audit["max_max_over_sd"] = float(worst[0][1]["max_over_sd"]) if worst else 0.0

    # ---- the pre-registered MULTI_ARM subgroup, defined here and frozen ------ #
    cluster_column = "coord__arm__local_donor_cluster_count"
    multi_arm = (table[cluster_column] >= 2).rename("MULTI_ARM")
    identity = (frame.drop_duplicates("extractant")
                .set_index("extractant")[["extractant_name", "chemotype", "chem_family"]])
    subgroup = pd.DataFrame({"MULTI_ARM": multi_arm}).join(identity)
    subgroup["n_cells"] = frame["extractant"].value_counts().reindex(subgroup.index).to_numpy()
    subgroup["n_donor_clusters"] = table[cluster_column].to_numpy()
    subgroup["repeated_arm_count"] = table["coord__arm__repeated_arm_count"].to_numpy()
    subgroup["potential_donor_count"] = table["coord__donor__potential_donor_count"].to_numpy()
    subgroup.to_csv(paths.MANIFEST_DIR / "multi_arm_subgroup.csv")
    audit["multi_arm"] = {
        "rule": "coord__arm__local_donor_cluster_count >= 2",
        "n_multi_arm": int(multi_arm.sum()),
        "n_single_arm": int((~multi_arm).sum()),
        "chemotypes_multi_arm": int(subgroup.loc[multi_arm, "chemotype"].nunique()),
        "chemotypes_single_arm": int(subgroup.loc[~multi_arm, "chemotype"].nunique()),
        "by_family": subgroup.groupby(["chem_family", "MULTI_ARM"]).size().unstack(
            fill_value=0).to_dict(),
        "cells_multi_arm": int(subgroup.loc[multi_arm, "n_cells"].sum()),
        "cells_single_arm": int(subgroup.loc[~multi_arm, "n_cells"].sum()),
    }

    table.to_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    (paths.FEATURE_DIR / "coordination_audit.json").write_text(json.dumps(audit, indent=1))
    (paths.FEATURE_DIR / "coordination_columns.json").write_text(json.dumps(
        {"all": list(table.columns), "informative": informative, "constant": constant,
         "by_family": {k: list(v) for k, v in families.items()}}, indent=1))

    print(f"coordination descriptors: {table.shape[0]} structures x {table.shape[1]} columns "
          f"({len(informative)} informative)")
    print("by family:", audit["informative_by_family"])
    print("matrix blake2b:", audit["feature_matrix_blake2b"], " spec:", audit["spec_sha256"][:16])
    print(f"largest |x|/sd across informative columns: {audit['max_max_over_sd']:.2f}")
    print("\nMULTI_ARM subgroup (pre-registered, structure-only):")
    print(json.dumps({k: v for k, v in audit["multi_arm"].items() if k != "by_family"}, indent=1))
    print(subgroup.groupby(["chem_family", "MULTI_ARM"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
