"""Column groups for the cleaned master table.

The split is load-bearing, not cosmetic: ``PROVENANCE`` columns must never be
offered to a model.  Several of them (DOI, source file, archive record id) are
near-perfect proxies for the experiment itself and would let a model memorise
the answer instead of learning chemistry.
"""

from __future__ import annotations

#: Identity and traceability.  Excluded from every feature matrix.
PROVENANCE_COLUMNS = (
    "canonical_measurement_id", "source_record_id", "raw_row_ids",
    "export_source_files", "export_metals_queried", "export_fanout_size",
    "representative_raw_row_id", "source_file", "source_line_number",
    "doi_primary", "doi_source_all", "doi_all", "archive_citation_doi",
    "has_source_reference", "reference_other", "entry_author", "addition_date",
    "publication_year", "publication_title", "publication_authors",
    "data_location", "sub_source_file", "comments_raw", "ini_comp_raw",
    "reference_title", "reference_year", "reference_journal", "reference_authors",
    "reference_publisher", "reference_url", "reference_metadata_source",
    "doi_primary_corrected", "doi_correction_rule", "doi_correction_evidence",
    "aqueous_phase_metals_declared", "n_metals_declared", "n_extractants_declared",
    "extractant_name_raw", "extractant_smiles_raw", "solvent_name_raw",
    "acid_name_raw", "modifier_name_raw", "modifier_concentration_raw",
    "acid_concentration_organic_raw", "phase_ratio_raw", "complexant_name_raw",
    "complexant_smiles_raw", "complexant_concentration_raw", "holdback_smiles_raw",
    "holdback_concentration_raw", "nitrate_concentration_raw",
    "extractant_concentration_raw", "acid_concentration_raw",
    "metal_concentration_raw", "temperature_raw", "contact_time_raw",
    "shaking_time_raw", "metal_raw", "metal_oxidation_state_raw", "D_raw",
    "duplicate_group_id", "duplicate_class", "duplicate_class_reason",
    "duplicate_group_size", "group_representative_id", "is_canonical_row",
    "identity_hash", "flags", "rdkit_parse_failures",
)

#: Primary experimental information -- the chemistry and the conditions.
#: These are the model's legitimate inputs.
PRIMARY_COLUMNS = (
    # chemical system
    "components", "system_component_class", "n_chemically_active_components",
    "n_organic_extractants", "extractant_system_key", "extractant_names",
    "extractant_smiles_canonical", "extractant_concentrations_M",
    "extractant_primary_smiles", "extractant_primary_name",
    "extractant_primary_concentration_M",
    "modifier_name", "modifier_concentration_M",
    "complexant_name", "complexant_smiles_canonical", "complexant_concentration_M",
    "holdback_name", "holdback_smiles_canonical", "holdback_concentration_M",
    # metal
    "metal_symbol", "metal_species_form", "metal_oxidation_state",
    "metal_oxidation_state_source", "metal_category", "is_lanthanide",
    # aqueous phase
    "acid_names", "acid_primary", "acid_signature", "acid_anion",
    "acid_concentration_M", "acid_concentration_organic_M", "nitrate_concentration_M",
    # diluent
    "solvent_components", "solvent_fractions", "solvent_key", "solvent_primary",
    "solvent_n_components", "solvent_pattern",
    # conditions
    "metal_concentration_M", "temperature_C", "contact_time_min",
    "shaking_time_min", "phase_ratio_org_aq",
)

#: The prediction target.
TARGET_COLUMNS = ("log_D", "D_value")

#: Reference quantities that the modelling code may derive itself.  They are
#: shipped because they are pure lookups on ``metal_symbol`` and oxidation
#: state, but a model is free to recompute them.
DERIVED_REFERENCE_COLUMNS = (
    "atomic_number", "lanthanide_index", "ionic_radius_cn8_A", "ionic_radius_status",
    "metal_oxidation_state_plausible",
)

#: Series/curve structure needed by the k-shot pipeline.
STRUCTURE_COLUMNS = ("series_id",)

#: Quality tiers assigned by stage 5.
QUALITY_COLUMNS = (
    "model_readiness", "is_model_ready", "has_target", "structures_resolved",
    "in_value_conflict", "review_priority",
)

MASTER_COLUMN_ORDER = (
    PROVENANCE_COLUMNS + PRIMARY_COLUMNS + TARGET_COLUMNS
    + DERIVED_REFERENCE_COLUMNS + STRUCTURE_COLUMNS + QUALITY_COLUMNS
)

#: Columns whose values are Python lists / dicts and therefore need JSON
#: encoding before they can go into a CSV.
LIST_COLUMNS = (
    "raw_row_ids", "export_source_files", "export_metals_queried", "doi_all",
    "doi_source_all", "reference_other", "extractant_names",
    "extractant_smiles_canonical", "extractant_concentrations_M", "acid_names",
    "solvent_components", "solvent_fractions", "components", "flags",
    "rdkit_parse_failures",
)
