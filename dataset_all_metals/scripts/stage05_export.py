"""Stage 5 -- quality tiers, subsets, coverage tables and the review queue."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_schema as SCHEMA                                              # noqa: E402
from sae_paths import INTERMEDIATE_DIR, CLEAN_DIR, AUDIT_DIR, REPORTS_DIR, ensure_dirs  # noqa: E402

#: Fields a row must have before it can enter a feature matrix at all.
MODEL_REQUIRED = (
    "metal_symbol", "log_D", "extractant_primary_smiles",
    "acid_concentration_M", "extractant_primary_concentration_M", "solvent_key",
)


def assign_quality(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["has_target"] = frame["log_D"].notna()
    frame["structures_resolved"] = frame["extractant_smiles_canonical"].map(
        lambda v: bool(len(v)) and all(x is not None for x in v) if v is not None else False)
    frame["in_value_conflict"] = frame["duplicate_class"].eq("E_VALUE_CONFLICT")

    complete = np.ones(len(frame), dtype=bool)
    for column in MODEL_REQUIRED:
        complete &= frame[column].notna().to_numpy()
    # MODEL_REQUIRED names extractant_primary_*, which is just the first listed
    # component.  On its own that would let a synergistic system be called
    # "model ready" while one of its two extractants is unresolved -- exactly
    # the flattening this dataset exists to avoid.  Readiness therefore also
    # requires EVERY organic component to have a structure and a concentration.
    all_components_resolved = frame["extractant_smiles_canonical"].map(
        lambda v: bool(v is not None and len(v) and all(x is not None for x in v))).to_numpy()
    all_concentrations_known = frame["extractant_concentrations_M"].map(
        lambda v: bool(v is not None and len(v)
                       and all(x is not None and not pd.isna(x) for x in v))).to_numpy()
    complete &= all_components_resolved & all_concentrations_known

    suspect_flags = {"extractant_name_structure_conflict", "component_name_structure_mismatch",
                     "component_list_length_mismatch", "implausible_oxidation_state",
                     "rdkit_parse_failure", "non_positive_D", "implausible_extreme_D"}
    frame["has_suspect_flag"] = frame["flags"].map(
        lambda fl: bool(suspect_flags.intersection(fl)) if fl is not None else False)

    readiness = np.where(
        ~complete, "C_not_modelable",
        np.where(frame["in_value_conflict"] | frame["has_suspect_flag"],
                 "B_usable_with_caveats", "A_model_ready"))
    # A row that was folded away as an A/B duplicate is never a training row.
    readiness = np.where(frame["is_canonical_row"], readiness, "D_redundant_duplicate")
    frame["model_readiness"] = readiness
    frame["is_model_ready"] = frame["model_readiness"].eq("A_model_ready")
    return frame


def build_review_queue(frame: pd.DataFrame, groups: pd.DataFrame,
                       conflicts: pd.DataFrame) -> pd.DataFrame:
    """Cases the pipeline could not settle, most consequential first."""
    rows: list[dict] = []

    def add(priority, category, ids, explanation, evidence):
        rows.append({
            "review_priority": priority,
            "category": category,
            "n_records": len(ids),
            "canonical_measurement_ids": "; ".join(sorted(ids)),
            "why_unresolved": explanation,
            "evidence": evidence,
        })

    # 1. identical conditions, different measured value
    for _, group in groups[groups["duplicate_class"] == "E_VALUE_CONFLICT"].iterrows():
        members = frame[frame["duplicate_group_id"] == group["duplicate_group_id"]]
        values = [None if pd.isna(v) else round(float(v), 6) for v in members["log_D"]]
        add(1, "same_conditions_different_logD", list(members["canonical_measurement_id"]),
            "Every scientifically meaningful condition agrees but the measured log D "
            "does not. Averaging would invent a number that was never measured, so all "
            "values are kept and the group is flagged.",
            f"log_D={values}; spread={group['reason']}; dois={list(group['dois'])}; "
            f"figures={list(group['data_locations'])}")

    # 2. same measurement under different provenance
    for _, group in groups[groups["duplicate_class"].isin(
            ["B_SAME_MEASUREMENT_DIFF_PROV", "C_POSSIBLE_INDEPENDENT_REPLICATE"])].iterrows():
        members = frame[frame["duplicate_group_id"] == group["duplicate_group_id"]]
        add(2, "identical_record_different_publication", list(members["canonical_measurement_id"]),
            "Conditions and value agree while the citation or sub-source differs. Whether "
            "this is one measurement copied between sources or two independent repeats "
            "cannot be decided from the archive alone.",
            f"class={group['duplicate_class']}; {group['reason']}; dois={list(group['dois'])}; "
            f"sub_sources={list(group['sub_sources'])}")

    # 3. multi-component systems with unresolved structure
    unresolved = frame[frame["system_component_class"].str.endswith("STRUCTURE_UNRESOLVED", na=False)]
    for name, block in unresolved.groupby("extractant_name_raw", dropna=False):
        add(3, "multi_component_structure_unresolved", list(block["canonical_measurement_id"]),
            "A chemically active component of this system has no structure in the archive, "
            "so the system cannot be represented without dropping a component.",
            f"extractant_name_raw={name!r}; class={block['system_component_class'].iloc[0]}")

    # 4. name / structure conflicts
    conflicted = frame[frame["flags"].map(
        lambda fl: "extractant_name_structure_conflict" in fl if fl is not None else False)]
    # Group by the component that actually disagrees, not by the row's
    # first-listed name -- 210 of these were filed under "TODGA", whose own
    # structure is consistent in all of its occurrences.
    by_component: dict[str, list] = {}
    for _, row in conflicted.iterrows():
        for detail, name in zip(row["conflicting_component_detail"],
                                row["conflicting_component_names"]):
            by_component.setdefault(f"{name}\x1f{detail}", []).append(
                row["canonical_measurement_id"])
    for key, ids in sorted(by_component.items()):
        name, detail = key.split("\x1f", 1)
        add(4, "extractant_name_structure_conflict", ids,
            f"Component {name!r} carries a structure in these records that disagrees with the "
            "structure the archive gives that same name elsewhere. One sub-source stores a "
            "masking agent's name next to the organic extractant's structure, so neither the "
            "name nor the structure can be trusted on its own for this component.",
            detail)

    # 5. component name/structure ordering that could not be resolved
    mismatched = frame[frame["flags"].map(
        lambda fl: "component_name_structure_mismatch" in fl if fl is not None else False)]
    for name, block in mismatched.groupby("extractant_name_raw", dropna=False):
        add(4, "component_pairing_ambiguous", list(block["canonical_measurement_id"]),
            "This multi-component record lists names and structures in an order that does "
            "not match the archive's own consensus, so which concentration belongs to "
            "which component is undetermined.",
            f"name_raw={name!r}; smiles_raw={block['extractant_smiles_raw'].iloc[0]!r}")

    # 6. ambiguous diluent naming
    from sae_normalize import SOLVENT_AMBIGUOUS
    for key, reason in SOLVENT_AMBIGUOUS.items():
        block = frame[frame["solvent_components"].map(
            lambda v: key in list(v) if v is not None else False)]
        if len(block):
            add(5, "diluent_name_ambiguous", list(block["canonical_measurement_id"]),
                f"Diluent name {key!r} is ambiguous: {reason}. It was NOT merged with any "
                "similar name, so these rows form their own diluent category.",
                f"solvent_name_raw examples={sorted(set(block['solvent_name_raw'].dropna()))[:4]}")

    # 7. chemically impossible oxidation states
    implausible = frame[frame["metal_oxidation_state_plausible"] == False]  # noqa: E712
    for (symbol, ox), block in implausible.groupby(["metal_symbol", "metal_oxidation_state"]):
        add(1, "implausible_oxidation_state", list(block["canonical_measurement_id"]),
            f"{symbol}({int(ox)}) is not an accessible oxidation state in aqueous "
            "solvent extraction. The archive value was preserved unchanged rather than "
            "corrected, because the correct state cannot be inferred from the record.",
            f"metal={symbol}; oxidation_state={int(ox)}; dois={sorted(set(block['doi_primary'].dropna()))[:3]}")

    extreme = frame[frame["flags"].map(
        lambda fl: "implausible_extreme_D" in fl if fl is not None else False)]
    if len(extreme):
        add(1, "implausible_extreme_D", list(extreme["canonical_measurement_id"]),
            "The reported distribution ratio is beyond six orders of magnitude, which is not "
            "measurable by solvent extraction. These values are bit-identical across different "
            "extractants read off the same figure, so they are almost certainly a "
            "figure-digitisation artefact. The value is preserved exactly as recorded.",
            f"D_raw={sorted(set(extreme['D_raw'].dropna()))}; "
            f"extractants={sorted(set(extreme['extractant_primary_name'].dropna()))}; "
            f"dois={sorted(set(extreme['doi_primary'].dropna()))}")

    repaired = frame[frame["doi_correction_rule"].notna()]
    for rule, block in repaired.groupby(["doi_primary", "doi_correction_rule"]):
        doi, rule_text = rule
        add(6, "doi_repaired_verify", list(block["canonical_measurement_id"]),
            "The DOI recorded by the archive does not resolve. A corrected form does, and the "
            "archive itself records that corrected form on other records. The raw value is kept "
            "in `doi_primary`; the repair only populates `doi_primary_corrected`. Confirm the "
            "corrected reference is the intended source before relying on it.",
            f"recorded={doi!r}; rule={rule_text}; "
            f"corrected={block['doi_primary_corrected'].iloc[0]!r}; "
            f"{block['doi_correction_evidence'].iloc[0]}")

    # Structures the archive never states directly. Their provenance is weaker
    # than an archive-stated SMILES, and at least one (Br-Cosan) is known to be
    # chemically wrong in the archive itself, so every one is surfaced rather
    # than only mentioned in a source comment.
    deduced = frame[frame["components"].map(
        lambda row: any(c.get("structure_source") == "deduced_by_elimination" for c in row)
        if row is not None else False)]
    if len(deduced):
        by_name: dict[tuple, list] = {}
        for _, row in deduced.iterrows():
            for comp in row["components"]:
                if comp.get("structure_source") != "deduced_by_elimination":
                    continue
                by_name.setdefault((comp["name"], comp["smiles_canonical"]), []).append(
                    row["canonical_measurement_id"])
        for (name, smiles), ids in sorted(by_name.items(), key=lambda kv: str(kv[0])):
            note = ""
            if name == "Br-Cosan":
                note = (" KNOWN WRONG: Br-Cosan is a cobalt bis(dicarbollide) "
                        "([3,3'-Co(8-Br-1,2-C2B9H10)(1',2'-C2B9H11)]-), a boron-cluster "
                        "metallacarborane. The structure below is a bromo-methyl "
                        "chloroacetanilide and is chemically unrelated. It comes from the "
                        "archive's own two-component record, not from this pipeline, and was "
                        "preserved as recorded rather than corrected.")
            add(3, "structure_deduced_by_elimination", ids,
                "The archive never states a SMILES for this name in any single-component "
                "record. The structure was recovered by elimination from the archive's own "
                "two-component records, which is evidence from the data rather than an "
                "outside assumption -- but it is only as reliable as the archive's SMILES "
                "column, so it should be confirmed before use." + note,
                f"name={name!r}; deduced_structure={smiles!r}; "
                f"tag=structure_source='deduced_by_elimination'")

    # 8. recoverable-looking gaps
    no_metal = frame[frame["metal_symbol"].isna()]
    if len(no_metal):
        add(6, "metal_identity_missing", list(no_metal["canonical_measurement_id"]),
            "No metal is recorded in Metal_Name, ini_comp or the comments. The per-metal "
            "export file is NOT evidence of the measured metal (records measuring Nd appear "
            "in Am.csv, Ba.csv and ten other files), so the metal cannot be inferred. "
            "Every one of these records also lacks a measured D value.",
            f"export_files examples={[list(v) for v in no_metal['export_metals_queried'].head(3)]}")

    no_reference = frame[~frame["has_source_reference"]]
    if len(no_reference):
        add(6, "no_primary_source_reference", list(no_reference["canonical_measurement_id"]),
            "The only citation is the archive's own DOI, so the measurement cannot be "
            "traced to a primary publication for verification.",
            f"sub_sources={sorted(set(no_reference['sub_source_file'].dropna()))[:5]}")

    # 9. condition conflicts (same value, differing condition)
    for _, row in conflicts.iterrows():
        add(3, "same_value_conflicting_condition", list(row["canonical_measurement_ids"]),
            f"These records report an identical log D while disagreeing about "
            f"{row['differing_field']}. One of the two conditions is likely a transcription "
            "error, but the archive gives no way to say which, so neither was changed.",
            f"field={row['differing_field']}; values={list(row['distinct_values_of_field'])}; "
            f"log_D={row['shared_log_D']}")

    queue = pd.DataFrame(rows)
    if len(queue):
        queue = queue.sort_values(["review_priority", "category", "canonical_measurement_ids"])
        queue = queue.reset_index(drop=True)
        queue.insert(0, "review_id", [f"MR{i:05d}" for i in range(len(queue))])
    return queue


def _component_names(series) -> str:
    """Every component's name, not just the first-listed one.

    Reporting ``extractant_primary_name`` made a synergistic system look like a
    single extractant in the human-readable coverage table.
    """
    names = set()
    for value in series:
        if value is None:
            continue
        for name in value:
            if isinstance(name, str) and name:
                names.add(name)
    return " + ".join(sorted(names))


def coverage_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ready = frame[frame["model_readiness"].isin(["A_model_ready", "B_usable_with_caveats"])]

    metal = (frame.groupby("metal_symbol", dropna=False)
             .agg(records=("canonical_measurement_id", "count"),
                  canonical=("is_canonical_row", "sum"),
                  with_target=("has_target", "sum"),
                  model_ready=("is_model_ready", "sum"),
                  extractant_systems=("extractant_system_key", "nunique"),
                  solvents=("solvent_key", "nunique"),
                  rows_with_ionic_radius=("ionic_radius_cn8_A", "count"),
                  log_D_min=("log_D", "min"), log_D_max=("log_D", "max"),
                  log_D_mean=("log_D", "mean"), log_D_std=("log_D", "std"))
             .reset_index())
    meta = frame.drop_duplicates("metal_symbol").set_index("metal_symbol")
    metal["category"] = metal["metal_symbol"].map(meta["metal_category"])
    metal["is_lanthanide"] = metal["metal_symbol"].map(meta["is_lanthanide"])
    metal["atomic_number"] = metal["metal_symbol"].map(meta["atomic_number"])
    # Radius availability is a property of the (element, oxidation state) pair,
    # so it can differ between rows of the same metal: a row whose oxidation
    # state the archive never recorded gets no radius even when its sibling
    # rows do.  Reporting one representative row's status would conflate
    # "this element is not tabulated" with "this row lacks an oxidation state",
    # so both are reported explicitly.
    metal["ionic_radius_coverage"] = [
        "none" if with_radius == 0 else ("all" if with_radius == total else "partial")
        for with_radius, total in zip(metal["rows_with_ionic_radius"], metal["records"])
    ]
    statuses = (frame.dropna(subset=["metal_symbol"])
                .groupby("metal_symbol")["ionic_radius_status"]
                .agg(lambda s: "|".join(sorted(set(s.dropna())))))
    metal["ionic_radius_status"] = metal["metal_symbol"].map(statuses)
    metal = metal.sort_values("records", ascending=False)

    extractant = (frame.groupby("extractant_system_key", dropna=False)
                  .agg(records=("canonical_measurement_id", "count"),
                       model_ready=("is_model_ready", "sum"),
                       metals=("metal_symbol", "nunique"),
                       solvents=("solvent_key", "nunique"),
                       names=("extractant_names", _component_names),
                       n_components=("n_organic_extractants", "max"),
                       log_D_min=("log_D", "min"), log_D_max=("log_D", "max"))
                  .reset_index().sort_values("records", ascending=False))

    matrix = pd.crosstab(ready["extractant_system_key"], ready["metal_symbol"])
    return {"metal": metal, "extractant": extractant, "extractant_metal_matrix": matrix}


def main() -> None:
    ensure_dirs()
    frame = assign_quality(pd.read_parquet(INTERMEDIATE_DIR / "records_with_series.parquet"))
    groups = pd.read_parquet(AUDIT_DIR / "duplicate_groups.parquet")
    conflicts = pd.read_parquet(AUDIT_DIR / "condition_conflict_candidates.parquet")

    queue = build_review_queue(frame, groups, conflicts)
    frame["review_priority"] = None

    columns = [c for c in SCHEMA.MASTER_COLUMN_ORDER if c in frame.columns]
    extra = [c for c in frame.columns if c not in columns]
    master = frame[columns + extra]
    master.to_parquet(CLEAN_DIR / "master_clean.parquet", index=False)

    csv_view = master.copy()
    for column in SCHEMA.LIST_COLUMNS:
        if column in csv_view.columns:
            csv_view[column] = csv_view[column].map(
                lambda v: json.dumps(v.tolist() if hasattr(v, "tolist") else v, default=str)
                if v is not None else "")
    csv_view.to_csv(CLEAN_DIR / "master_clean.csv", index=False)

    lanthanide = master[master["is_lanthanide"] == True]           # noqa: E712
    non_lanthanide = master[(master["is_lanthanide"] == False) & master["metal_symbol"].notna()]
    # The conservative subset excludes anything with a *recorded* second
    # component, even one whose concentration is an explicit zero.  94 records
    # carry a masking agent at 0 M -- chemically single-component, but the
    # point of this subset is to be free of multi-component ambiguity
    # altogether, so they are held back.
    single = master[(master["system_component_class"] == "SINGLE_EXTRACTANT")
                    & master["holdback_smiles_canonical"].isna()
                    & master["complexant_smiles_canonical"].isna()
                    & master["modifier_name"].isna()
                    & (master["n_organic_extractants"] <= 1)]
    lanthanide.to_parquet(CLEAN_DIR / "lanthanide_clean.parquet", index=False)
    non_lanthanide.to_parquet(CLEAN_DIR / "non_lanthanide_clean.parquet", index=False)
    single.to_parquet(CLEAN_DIR / "single_component_clean.parquet", index=False)

    tables = coverage_tables(master)
    tables["metal"].to_csv(REPORTS_DIR / "metal_coverage.csv", index=False)
    tables["extractant"].to_csv(REPORTS_DIR / "extractant_coverage.csv", index=False)
    tables["extractant_metal_matrix"].to_csv(REPORTS_DIR / "extractant_metal_matrix.csv")
    if len(queue):
        queue.to_csv(AUDIT_DIR / "manual_review_queue.csv", index=False)

    conflict_groups = groups[groups["duplicate_class"].isin(
        ["E_VALUE_CONFLICT", "D_CONDITION_CONFLICT"])]
    conflict_groups.to_parquet(AUDIT_DIR / "conflict_groups.parquet", index=False)

    mapping = pd.read_parquet(INTERMEDIATE_DIR / "raw_row_mapping.parquet")
    mapping = mapping.merge(
        master[["canonical_measurement_id", "duplicate_group_id", "duplicate_class",
                "group_representative_id", "is_canonical_row", "model_readiness"]],
        on="canonical_measurement_id", how="left")
    mapping.to_parquet(AUDIT_DIR / "raw_to_canonical_mapping.parquet", index=False)

    summary = {
        "records": int(len(master)),
        "readiness": master["model_readiness"].value_counts().to_dict(),
        "subsets": {
            "lanthanide": int(len(lanthanide)),
            "non_lanthanide": int(len(non_lanthanide)),
            "metal_unresolved": int(master["metal_symbol"].isna().sum()),
            "single_component": int(len(single)),
            "multi_component": int((master["system_component_class"] != "SINGLE_EXTRACTANT").sum()),
        },
        "unique_metals": int(master["metal_symbol"].nunique()),
        "unique_extractant_systems": int(master["extractant_system_key"].nunique()),
        "unique_solvents": int(master["solvent_key"].nunique()),
        "review_queue_cases": int(len(queue)),
        "review_records": int(len({i for ids in queue["canonical_measurement_ids"]
                                   for i in ids.split("; ") if i})) if len(queue) else 0,
        "raw_rows_mapped": int(len(mapping)),
    }
    (AUDIT_DIR / "stage05_export_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
