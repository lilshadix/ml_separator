"""Stage 6 -- row accounting, human-readable reports and the artifact manifest.

The accounting ledger is the stopping rule from the brief: every one of the
48,471 raw CSV rows has to leave this pipeline as exactly one of

* a canonical measurement,
* a member of a documented duplicate group,
* a conflict preserved as its own observation, or
* an unresolved row queued for manual review.

``row_accounting.json`` is checked to balance at each step; if it does not, the
stage raises instead of writing a report that quietly loses rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_identity as ID                                        # noqa: E402
import sae_schema as SCHEMA                                      # noqa: E402
from sae_paths import (INTERMEDIATE_DIR, CLEAN_DIR, AUDIT_DIR, REPORTS_DIR,  # noqa: E402
                       RAW_DIR, RUN_DIR, ROOT, ensure_dirs)
from sae_structures import RDKIT_VERSION                          # noqa: E402


def build_accounting(master: pd.DataFrame, mapping: pd.DataFrame,
                     fanout: dict, groups: pd.DataFrame) -> dict:
    raw_rows = int(len(mapping))
    records = int(len(master))

    readiness = master["model_readiness"].value_counts().to_dict()
    classes = master["duplicate_class"].value_counts().to_dict()

    ledger = [
        {"step": "0_raw_csv_rows", "rows": raw_rows,
         "note": f"{len(list(RAW_DIR.glob('*.csv')))} immutable per-metal CSV files"},
        {"step": "1_export_fanout_collapsed",
         "rows_removed": int(fanout["raw_rows_folded_away"]),
         "rows_remaining": records,
         "note": "one archive record appears once per metal file whose metal was present "
                 "in the system; all 37 raw columns were verified identical within each "
                 "exp_id group before collapsing"},
        {"step": "2_duplicate_classification", "rows_remaining": records,
         "breakdown": classes,
         "note": "no rows removed; every record keeps its own row"},
        {"step": "3_redundant_duplicates_marked",
         "rows_marked_non_canonical": int((~master["is_canonical_row"]).sum()),
         "rows_remaining_canonical": int(master["is_canonical_row"].sum()),
         "note": "only classes A and B collapse; C/D/E/F keep every member"},
        {"step": "4_model_readiness_tiers", "breakdown": readiness,
         "note": "tiers are labels, not filters -- no row is dropped"},
    ]

    disposition = {
        "canonical_measurement": int((master["model_readiness"] == "A_model_ready").sum()),
        "canonical_with_caveats": int((master["model_readiness"] == "B_usable_with_caveats").sum()),
        "redundant_duplicate_member": int((master["model_readiness"] == "D_redundant_duplicate").sum()),
        "unresolved_or_manual_review": int((master["model_readiness"] == "C_not_modelable").sum()),
    }
    total = sum(disposition.values())
    if total != records:
        raise AssertionError(f"record disposition sums to {total}, expected {records}")

    # Every raw row must resolve to a record, and every record must be reachable.
    unmapped = int(mapping["canonical_measurement_id"].isna().sum())
    if unmapped:
        raise AssertionError(f"{unmapped} raw rows have no canonical measurement")
    reachable = set(mapping["canonical_measurement_id"])
    missing = set(master["canonical_measurement_id"]) - reachable
    if missing:
        raise AssertionError(f"{len(missing)} records are not reachable from any raw row")

    return {
        "raw_rows": raw_rows,
        "archive_records": records,
        "ledger": ledger,
        "record_disposition": disposition,
        "raw_row_disposition": (mapping.merge(
            master[["canonical_measurement_id", "model_readiness"]],
            on="canonical_measurement_id", how="left", suffixes=("", "_m"))
            ["model_readiness_m" if "model_readiness_m" in mapping.columns else "model_readiness"]
            .value_counts().to_dict()),
        "duplicate_groups": int(len(groups)),
        "balance_check": "ok",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_schema_doc(master: pd.DataFrame) -> str:
    lines = ["# Schema -- `master_clean.parquet`", "",
             f"{len(master):,} rows x {len(master.columns)} columns. One row per archive "
             "record (`exp_id`), never per raw CSV row.", "",
             "Columns are grouped by role. **The provenance group must never be used as a "
             "model feature**: DOI, source file and archive record id are near-perfect "
             "proxies for the experiment and would let a model memorise the target.", ""]
    groups = [
        ("Provenance (never a feature)", SCHEMA.PROVENANCE_COLUMNS),
        ("Primary experimental information (model inputs)", SCHEMA.PRIMARY_COLUMNS),
        ("Target", SCHEMA.TARGET_COLUMNS),
        ("Derived reference values (reproducible lookups)", SCHEMA.DERIVED_REFERENCE_COLUMNS),
        ("Series / curve structure", SCHEMA.STRUCTURE_COLUMNS),
        ("Quality tiers", SCHEMA.QUALITY_COLUMNS),
    ]
    for title, columns in groups:
        lines += [f"## {title}", "", "| column | dtype | non-null | note |", "|---|---|---|---|"]
        for column in columns:
            if column not in master.columns:
                continue
            non_null = int(master[column].notna().sum())
            note = COLUMN_NOTES.get(column, "")
            lines.append(f"| `{column}` | {master[column].dtype} | {non_null:,} | {note} |")
        lines.append("")
    lines += ["## Multi-component representation", "",
              "`components` is a list of dicts, one per chemically active component, each with "
              "`role` (`organic_extractant` / `phase_modifier` / `aqueous_complexant` / "
              "`aqueous_holdback`), `name`, `smiles_raw`, `smiles_canonical`, "
              "`structure_source` and `concentration_M`.", "",
              "A synergistic pair is therefore *two* entries, not one flattened SMILES. "
              "`extractant_system_key` is the order-invariant join of the sorted canonical "
              "SMILES, so `\"A, B\"` and `\"B, A\"` -- which the archive uses "
              "interchangeably -- produce the same key.", ""]
    return "\n".join(lines)


COLUMN_NOTES = {
    "canonical_measurement_id": "`SAE:<exp_id>`; stable across reruns",
    "raw_row_ids": "every contributing raw CSV row, as `<file>:<line>`",
    "export_metals_queried": "metal files this record appeared in -- an export artefact, NOT the measured metal",
    "doi_primary": "first non-archive DOI; null when only the archive self-citation exists",
    "archive_citation_doi": "the archive's own DOI, kept apart from the measurement's source",
    "extractant_system_key": "order-invariant `|`-join of sorted canonical SMILES",
    "components": "list of dicts; the multi-component schema (see below)",
    "metal_symbol": "element symbol; `UO2+2` is expanded to U(VI)",
    "lanthanide_index": "Z - 56, matching gen10; null for non-lanthanides",
    "ionic_radius_cn8_A": "Shannon (1976) CN=8; null where not tabulated -- see `ionic_radius_status`",
    "log_D": "log10 of the archive's distribution ratio; null when D is missing or non-positive",
    "series_id": "gen8-compatible series hash; built without reading `log_D`",
    "model_readiness": "A_model_ready / B_usable_with_caveats / C_not_modelable / D_redundant_duplicate",
    "is_canonical_row": "false only for members of class A/B duplicate groups",
}


def main() -> None:
    ensure_dirs()
    master = pd.read_parquet(CLEAN_DIR / "master_clean.parquet")
    mapping = pd.read_parquet(AUDIT_DIR / "raw_to_canonical_mapping.parquet")
    groups = pd.read_parquet(AUDIT_DIR / "duplicate_groups.parquet")
    fanout = json.loads((AUDIT_DIR / "stage01_fanout_report.json").read_text())
    normalization = json.loads((AUDIT_DIR / "stage02_normalization_log.json").read_text())
    dedup = json.loads((AUDIT_DIR / "stage03_dedup_report.json").read_text())
    series = json.loads((AUDIT_DIR / "stage04_series_report.json").read_text())
    export = json.loads((AUDIT_DIR / "stage05_export_report.json").read_text())

    accounting = build_accounting(master, mapping, fanout, groups)
    (REPORTS_DIR / "row_accounting.json").write_text(
        json.dumps(accounting, indent=2, sort_keys=True, default=str) + "\n")
    (REPORTS_DIR / "normalization_log.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True, default=str) + "\n")
    (REPORTS_DIR / "schema.md").write_text(write_schema_doc(master) + "\n")

    payload = {"fanout": fanout, "normalization": normalization, "dedup": dedup,
               "series": series, "export": export, "accounting": accounting}
    (REPORTS_DIR / "_report_payload.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(accounting["record_disposition"], indent=2))
    print("raw rows:", accounting["raw_rows"], "records:", accounting["archive_records"])


if __name__ == "__main__":
    main()
