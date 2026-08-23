# Schema -- `master_clean.parquet`

16,770 rows x 134 columns. One row per archive record (`exp_id`), never per raw CSV row.

Columns are grouped by role. **The provenance group must never be used as a model feature**: DOI, source file and archive record id are near-perfect proxies for the experiment and would let a model memorise the target.

## Provenance (never a feature)

| column | dtype | non-null | note |
|---|---|---|---|
| `canonical_measurement_id` | str | 16,770 | `SAE:<exp_id>`; stable across reruns |
| `source_record_id` | str | 16,770 |  |
| `raw_row_ids` | object | 16,770 | every contributing raw CSV row, as `<file>:<line>` |
| `export_source_files` | object | 16,770 |  |
| `export_metals_queried` | object | 16,770 | metal files this record appeared in -- an export artefact, NOT the measured metal |
| `export_fanout_size` | int64 | 16,770 |  |
| `representative_raw_row_id` | str | 16,770 |  |
| `source_file` | str | 16,770 |  |
| `source_line_number` | int64 | 16,770 |  |
| `doi_primary` | str | 15,615 | first non-archive DOI; null when only the archive self-citation exists |
| `doi_source_all` | object | 16,770 |  |
| `doi_all` | object | 16,770 |  |
| `archive_citation_doi` | str | 9,446 | the archive's own DOI, kept apart from the measurement's source |
| `has_source_reference` | bool | 16,770 |  |
| `reference_other` | object | 16,770 |  |
| `entry_author` | str | 16,770 |  |
| `addition_date` | str | 16,770 |  |
| `publication_year` | str | 5,738 |  |
| `publication_title` | str | 5,738 |  |
| `publication_authors` | str | 5,738 |  |
| `data_location` | str | 5,738 |  |
| `sub_source_file` | str | 1,104 |  |
| `comments_raw` | str | 15,710 |  |
| `ini_comp_raw` | str | 16,770 |  |
| `reference_title` | str | 15,615 |  |
| `reference_year` | float64 | 15,615 |  |
| `reference_journal` | str | 15,615 |  |
| `reference_authors` | str | 15,615 |  |
| `reference_publisher` | str | 15,615 |  |
| `reference_url` | str | 15,615 |  |
| `reference_metadata_source` | str | 15,615 |  |
| `doi_primary_corrected` | str | 15,615 |  |
| `doi_correction_rule` | str | 2,690 |  |
| `doi_correction_evidence` | str | 2,690 |  |
| `aqueous_phase_metals_declared` | str | 1,082 |  |
| `n_metals_declared` | str | 5,738 |  |
| `n_extractants_declared` | str | 5,738 |  |
| `extractant_name_raw` | str | 16,544 |  |
| `extractant_smiles_raw` | str | 15,067 |  |
| `solvent_name_raw` | str | 16,770 |  |
| `acid_name_raw` | str | 16,769 |  |
| `modifier_name_raw` | str | 797 |  |
| `modifier_concentration_raw` | str | 792 |  |
| `acid_concentration_organic_raw` | str | 1,962 |  |
| `phase_ratio_raw` | str | 8,153 |  |
| `complexant_name_raw` | str | 991 |  |
| `complexant_smiles_raw` | str | 991 |  |
| `complexant_concentration_raw` | str | 991 |  |
| `holdback_smiles_raw` | str | 327 |  |
| `holdback_concentration_raw` | str | 1,072 |  |
| `nitrate_concentration_raw` | str | 1,104 |  |
| `extractant_concentration_raw` | str | 16,536 |  |
| `acid_concentration_raw` | str | 16,592 |  |
| `metal_concentration_raw` | str | 7,412 |  |
| `temperature_raw` | str | 15,641 |  |
| `contact_time_raw` | str | 9,488 |  |
| `shaking_time_raw` | str | 2,883 |  |
| `metal_raw` | str | 14,852 |  |
| `metal_oxidation_state_raw` | str | 13,468 |  |
| `D_raw` | str | 14,622 |  |
| `duplicate_group_id` | str | 16,770 |  |
| `duplicate_class` | str | 16,770 |  |
| `duplicate_class_reason` | str | 16,770 |  |
| `duplicate_group_size` | int64 | 16,770 |  |
| `group_representative_id` | str | 16,770 |  |
| `is_canonical_row` | bool | 16,770 | false only for members of class A/B duplicate groups |
| `identity_hash` | str | 16,770 |  |
| `flags` | object | 16,770 |  |
| `rdkit_parse_failures` | object | 16,770 |  |

## Primary experimental information (model inputs)

| column | dtype | non-null | note |
|---|---|---|---|
| `components` | object | 16,770 | list of dicts; the multi-component schema (see below) |
| `system_component_class` | str | 16,770 |  |
| `n_chemically_active_components` | int64 | 16,770 |  |
| `n_organic_extractants` | int64 | 16,770 |  |
| `extractant_system_key` | str | 16,544 | order-invariant `|`-join of sorted canonical SMILES |
| `extractant_names` | object | 16,770 |  |
| `extractant_smiles_canonical` | object | 16,770 |  |
| `extractant_concentrations_M` | object | 16,770 |  |
| `extractant_primary_smiles` | str | 16,544 |  |
| `extractant_primary_name` | str | 16,544 |  |
| `extractant_primary_concentration_M` | float64 | 16,536 |  |
| `modifier_name` | str | 797 |  |
| `modifier_concentration_M` | float64 | 792 |  |
| `complexant_name` | str | 991 |  |
| `complexant_smiles_canonical` | str | 991 |  |
| `complexant_concentration_M` | float64 | 991 |  |
| `holdback_name` | object | 0 |  |
| `holdback_smiles_canonical` | str | 327 |  |
| `holdback_concentration_M` | float64 | 1,072 |  |
| `metal_symbol` | str | 14,852 | element symbol; `UO2+2` is expanded to U(VI) |
| `metal_species_form` | str | 16 |  |
| `metal_oxidation_state` | float64 | 13,484 |  |
| `metal_oxidation_state_source` | str | 13,484 |  |
| `metal_category` | str | 14,852 |  |
| `is_lanthanide` | object | 14,852 |  |
| `acid_names` | object | 16,770 |  |
| `acid_primary` | str | 16,769 |  |
| `acid_signature` | str | 16,770 |  |
| `acid_anion` | str | 16,769 |  |
| `acid_concentration_M` | float64 | 16,592 |  |
| `acid_concentration_organic_M` | float64 | 1,962 |  |
| `nitrate_concentration_M` | float64 | 1,104 |  |
| `solvent_components` | object | 16,770 |  |
| `solvent_fractions` | object | 16,770 |  |
| `solvent_key` | str | 16,770 |  |
| `solvent_primary` | str | 16,770 |  |
| `solvent_n_components` | int64 | 16,770 |  |
| `solvent_pattern` | str | 16,770 |  |
| `metal_concentration_M` | float64 | 7,412 |  |
| `temperature_C` | float64 | 15,641 |  |
| `contact_time_min` | float64 | 9,488 |  |
| `shaking_time_min` | float64 | 2,883 |  |
| `phase_ratio_org_aq` | float64 | 8,153 |  |

## Target

| column | dtype | non-null | note |
|---|---|---|---|
| `log_D` | float64 | 14,622 | log10 of the archive's distribution ratio; null when D is missing or non-positive |
| `D_value` | float64 | 14,622 |  |

## Derived reference values (reproducible lookups)

| column | dtype | non-null | note |
|---|---|---|---|
| `atomic_number` | float64 | 14,852 |  |
| `lanthanide_index` | float64 | 8,053 | Z - 56, matching gen10; null for non-lanthanides |
| `ionic_radius_cn8_A` | float64 | 12,664 | Shannon (1976) CN=8; null where not tabulated -- see `ionic_radius_status` |
| `ionic_radius_status` | str | 16,770 |  |
| `metal_oxidation_state_plausible` | object | 13,484 |  |

## Series / curve structure

| column | dtype | non-null | note |
|---|---|---|---|
| `series_id` | str | 16,770 | gen8-compatible series hash; built without reading `log_D` |

## Quality tiers

| column | dtype | non-null | note |
|---|---|---|---|
| `model_readiness` | str | 16,770 | A_model_ready / B_usable_with_caveats / C_not_modelable / D_redundant_duplicate |
| `is_model_ready` | bool | 16,770 |  |
| `has_target` | bool | 16,770 |  |
| `structures_resolved` | bool | 16,770 |  |
| `in_value_conflict` | bool | 16,770 |  |
| `review_priority` | object | 0 |  |

## Multi-component representation

`components` is a list of dicts, one per chemically active component, each with `role` (`organic_extractant` / `phase_modifier` / `aqueous_complexant` / `aqueous_holdback`), `name`, `smiles_raw`, `smiles_canonical`, `structure_source` and `concentration_M`.

A synergistic pair is therefore *two* entries, not one flattened SMILES. `extractant_system_key` is the order-invariant join of the sorted canonical SMILES, so `"A, B"` and `"B, A"` -- which the archive uses interchangeably -- produce the same key.

