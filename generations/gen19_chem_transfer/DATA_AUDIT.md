# Gen19 data audit (brief §28, Phase A)

**Scope.** This document covers brief §28 Phase A, items 1–14, in order. No model was trained or fitted.

**Citations.**
- Every number is copied from a file written by a Gen19 script. The bracket after it gives the file and the key or column.
- Paths are relative to `generations/gen19_chem_transfer/`. A bare file name means `data_audit/<file>`.
- A number marked **†** is a row tally of the cited table under the stated filter. It is not a stored summary field.

**Terms.**
- **MODEL** means `g19_tier == "MODEL"`: finite log D, resolved metal, canonical row, model readiness A or B (`gen19ct/data/load.py`).
- **"X(?)"** is a metal whose oxidation state is not recorded.
- **"System"** is an `extractant_system_key`: the order-invariant join of the canonical SMILES of every organic extractant.

**Where the numbers come from.** The five manifests below were produced at git head 40f6a75 against archive sha256 `7b32979383c5…9b01`. When this document was written, every output hash they list matched the file on disk.

| Script (`scripts/`) | Main outputs | Manifest |
|---|---|---|
| `g19_audit_corpus.py` | `counts.json`, `sparsity.json`, `dataset_hashes.csv`, `columns.csv`, `matrix_*.csv`, `*_coverage.csv`, F01/F02/F04/F05/F06 | `manifests/g19_audit_corpus.json` |
| `g19_build_metals.py` | `metal_alias_audit.csv`, `metal_descriptor_coverage.csv`, `descriptors/metals.csv` | `manifests/g19_build_metals.json` |
| `g19_build_extractants.py` | `family_coverage.csv`, `ligand_alias_collisions.csv`, `named_extractant_presence.csv`, `descriptors/extractant_*.csv` | `manifests/g19_build_extractants.json` |
| `g19_audit_leakage.py` | `leakage_*.csv`, `leakage_summary.json`, `metadata_availability.csv` | `manifests/g19_audit_leakage.json` |
| `g19_feasibility.py` | `feasibility.json`, `feasibility_*.csv`, F03 | `manifests/g19_feasibility.json` |

**To rerun,** start from the repo root with `PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/<script>.py`.
- Run `g19_build_metals`, then `g19_build_extractants`, then `g19_audit_leakage`, then `g19_feasibility`. The last one reads outputs of the first three.
- `g19_audit_corpus` does not depend on the others.

---

## 0. What this corpus is

The expanded corpus is the SAFE ("Separation Archive for Elements") export in `dataset_all_metals/`. It holds 16,770 archive records [counts.json › rows.total], of which 12,411 are MODEL [rows.by_tier.MODEL].

It is a **nuclear-separation corpus**, built around nitrate media and neutral solvating ligands, especially diglycolamides:
- **Acid:** HNO3 is the primary acid on 11,204 MODEL rows [counts.json › acid_molarity_semantics.below_1e-3_model_by_acid_primary, key HNO3, model_rows_with_acid].
- **Mechanism:** neutral-solvating systems carry 10,711 MODEL rows [feasibility_mechanisms.csv › NEUTRAL_SOLVATING.n_model_rows].
- **Family:** diglycolamide-family systems carry 7,621 MODEL rows, or 7,616 without the one name–structure-conflict system [feasibility_families.csv › diglycolamide.n_model_rows, n_model_rows_excl_name_structure_conflict].
- **Metals:** lanthanides have 6,495 MODEL rows and actinides 5,420 [counts.json › rows.by_metal_category_model].
- **Concentration:** TODGA alone is 0.225 of MODEL rows with a known oxidation state [sparsity.json › system_x_metal_state.share_rows_top1_system].

Rare-earth hydrometallurgy extractants barely appear:
- **Acidic cation exchange:** 14 MODEL rows in 3 systems [feasibility_mechanisms.csv › ACIDIC_CATION_EXCHANGE]. Three systems whose only recorded "extractant" is an aqueous chelator or hydrophilic acid (HEDTA, CDTA, thiodiglycolic acid under the name TDGA; 29 rows) are UNKNOWN, not acidic or chelating (FEASIBILITY Q4).
- **Phosphoric acids:** one structure with 10 MODEL rows [family_coverage.csv › system_family phosphoric_acid].
- **Phosphonic and phosphinic acids:** no structure at all [named_extractant_presence.csv › PC88A:SUBSTRUCTURE_FAMILY and Cyanex 272:SUBSTRUCTURE_FAMILY, verdict ABSENT].

pH, saponification, organic loading, reported uncertainty and a "D reported vs reconstructed" flag are never recorded (§14).

## 1. Location of the expanded dataset

- **Row-level table:** `dataset_all_metals/clean/master_clean.parquet`. It has 134 archive columns; `load.py` adds 9 derived `g19_*` columns [counts.json › dataset.n_archive_columns, dataset.n_g19_columns]. There is one row per archive `exp_id`.
- **Raw files:** 41 per-metal export CSVs in `dataset_all_metals/raw/` [dataset_hashes.csv, group == raw †].
  - These files are an **export fan-out**: a record appears in the file of every metal the export queried.
  - On 7,334 MODEL rows the export file is named after a different metal from the one measured [leakage_summary.json › metal_alias.n_model_rows_export_file_is_other_metal].
- **Other archive folders:** `audit/`, `intermediate/`, `reports/`, `scripts/`, `tests/`, `cache/`. All are hashed in `dataset_hashes.csv`.
- **Frozen 14-lanthanide bundle:** `dataset with 3D structures/dataset.parquet`. It is a strict subset of the archive (§15).

## 2. Dataset hashes (binary and CRLF→LF-normalised)

| File | Bytes | Bytes (LF) | sha256 (binary) | sha256 (LF-normalised) | Check |
|---|---|---|---|---|---|
| `clean/master_clean.parquet` (headline) | 2,389,792 | n/a (binary) | `7b32979383c5246f22d36ade6246231a347ea22490f4589a76e189af07b09b01` | n/a | equals pinned value and archive `manifest.json:artifacts` (binary) |
| `clean/master_clean.csv` | 36,567,890 | 36,537,128 | `cd6991a29cabb355c676dc3397bd9c40289fbbfc0b3332836eb4963511a8e936` | `e69469839124eced6bb188a858c1996af02f5fe7386564e273965309064c4cfa` | archive manifest records the LF digest (lf_normalised) |
| `reports/schema.md` | 7,924 | 7,756 | `829ea86dad714f4eb1e31ac572f11fe6e95fd30569362262b7ee8920223784a2` | `83447015f8e62f9999cd2babb87fc0b373b33b8a12f8e1d5c56880231fad2829` | lf_normalised |
| `reports/quality_report.md` | 17,191 | 16,917 | `daca16c7426c778c2c06ce8621922531706d810c461faccd5740358da120ebea` | `3275827e1f2b18b443456bdf33e20c201eebc3a3d0fcadaf9563237edeae39b7` | lf_normalised |
| `intermediate/raw_rows.parquet` | 1,311,911 | n/a | `7b55a06adb3d8a288cff227e7b2e1455166113f6f47353abd915086b422be6db` | n/a | no recorded digest |
| bundle `dataset.parquet` | 2,802,776 | n/a | `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd` | n/a | matches pinned prefix `fefbefc6` |

[dataset_hashes.csv › bytes, bytes_lf, sha256, sha256_lf, manifest_match; `is_headline_dataset_hash` is True only for master_clean.parquet]

**Verification summary.** `dataset_hashes.csv` has 98 rows: 97 archive files plus the bundle.
- 63 rows have a recorded digest, and all 63 verify [has_recorded_digest == True †; verified == True †]:
  - 51 match as binary.
  - 11 match only after CRLF→LF normalisation.
  - 1 matches a pinned prefix (the bundle).

  [manifest_match value counts †]
- 35 rows have no recorded digest to compare against.
- The 41 raw CSVs also match `raw_checksums.sha256` [raw_checksum_match == binary †].

Text files carry both digests, so a CRLF checkout does not look like a reproducibility failure (brief §24). One small gap: `dataset_all_metals/.gitignore` has no LF digest, because `paths.TEXT_SUFFIXES` does not list `.gitignore`.

## 3. Columns

`columns.csv` lists all 143 columns (134 archive + 9 derived). For each it gives the `schema.md` section, role, dtype, list-valued flag, fill fractions over all rows and MODEL rows, and distinct counts.

| `schema_section` | Columns † | Content |
|---|---|---|
| Provenance (never a feature) | 69 | ids, export fan-out, DOIs, references, curator, raw strings (`D_raw`, `metal_raw`, `extractant_name_raw` …), duplicate bookkeeping, `flags` |
| Primary experimental information (model inputs) | 43 | components, extractant system/SMILES/concentrations, modifier, complexant, holdback, metal, acid, solvent, metal concentration, temperature, contact/shaking time, O/A |
| Target | 2 | `log_D`, `D_value` |
| Derived reference values | 5 | `atomic_number`, `lanthanide_index`, `ionic_radius_cn8_A`, `ionic_radius_status`, `metal_oxidation_state_plausible` |
| Series / curve structure | 1 | `series_id` |
| Quality tiers | 6 | `model_readiness`, `is_model_ready`, `has_target`, `structures_resolved`, `in_value_conflict`, `review_priority` |
| NOT_IN_SCHEMA_MD | 8 | complexant list forms, `conflicting_component_*`, `has_suspect_flag` (roles INFERRED) |
| g19_derived | 9 | `g19_publication_id`, `g19_publication_status`, `g19_publication_refs`, `g19_study_id`, `g19_metal`, `g19_ox`, `g19_metal_state`, `g19_tier`, `g19_bundle_exp_id` |

- 20 columns are list-valued. Under pandas 3 they are numpy arrays [list_valued == True †].
- The non-null count matches `schema.md` for all 126 columns that `schema.md` documents [nonnull_matches_schema_md == True †].
- **Provenance guard list.** `load.PROVENANCE_COLUMNS` covers 76 of the 143 columns [in_PROVENANCE_COLUMNS == True †]: all 69 columns of the schema's Provenance section, the target-derived `in_value_conflict` (Quality tiers), `series_id` and 5 derived `g19_*` columns [schema_section tally of in_PROVENANCE_COLUMNS == True †]. No column with schema role provenance is missing from it [schema_role == provenance and in_PROVENANCE_COLUMNS == False †: 0].
  - The list therefore includes `D_raw`, the raw target string, as well as `metal_raw`, `duplicate_class`, `duplicate_class_reason`, `is_canonical_row` and `flags`.
  - An earlier build of this table reported 43 names and 32 schema-provenance columns missing; the shared loader has since been extended, and `tests/test_manifest.py::test_provenance_guard_list_covers_schema_provenance_and_target_derived_columns` now passes without an xfail marker.

## 4. Row counts and tiers

| Quantity | All rows | MODEL | Source |
|---|---|---|---|
| Rows | 16,770 | 12,411 | counts.json › rows.total, rows.by_tier |
| Tier TARGET_ONLY / NO_TARGET | 2,211 / 2,148 | – | rows.by_tier |
| Lanthanide / actinide rows | 8,053 / 6,302 | 6,495 / 5,420 | rows.by_metal_category_all / _model |
| Alkaline earth / transition / rare-earth non-Ln / post-transition | 285 / 109 / 59 / 44 | 284 / 109 / 59 / 44 | same |
| No metal recorded | 1,918 | 0 | rows.by_metal_category_all["<no metal>"] |
| Publication status DOI / REPORT_SUBSOURCE / REPORT | 15,615 / 1,104 / 51 | 11,289 / 1,096 / 26 | rows.by_publication_status_* |
| SINGLE_EXTRACTANT / +AQUEOUS_AGENT / +MODIFIER / SYNERGISTIC / NO_EXTRACTANT | 14,206 / 933 / 797 / 608 / 226 | 10,457 / 864 / 616 / 474 / 0 | rows.by_system_component_class_* |
| Rows with unknown oxidation state | – | 1,362 | rows.model_rows_unknown_oxidation_state |
| Distinct condition keys (17 fields, 6 s.f.) | 7,858 | 5,856 | condition_keys.n_condition_keys_* |
| Distinct experiment keys | 9,145 | 7,104 | condition_keys.n_experiment_keys_* |

**Flags recorded in `counts.json`.** Values are kept unaltered; nothing is dropped.
- **Implausible state:** the archive marks Sr(III) implausible on 38 rows, all MODEL, in 2 publications [oxidation_state_plausibility].
- **Out-of-range values:** 1 MODEL row has acid > 16 M (SAE:4624, Fe, 24.0244 M) and 1 has extractant > 5 M (SAE:4554, Fe, 5.81 M) [unit_sanity].
- **Acid values that may not be molarities (INFERRED):**
  - 302 MODEL rows have an acid molarity on a 0.01 log10 grid with more than 3 significant figures [acid_molarity_semantics.acid_M_log10_grid_model]. That pattern fits a back-converted pH or a log-axis digitisation. About 28.94 such rows would be expected by chance [chance_false_positive_model.expected_chance_hits_model].
  - 187 of the 302 come from pub_e287124c22 and 98 from pub_1605e436d7 [log10_grid_model_by_publication].
  - 160 MODEL rows have acid < 1e-3 M [acid_M_below_1e-3_model].

## 5. Unique counts

| Entity | All rows | MODEL | Source / note |
|---|---|---|---|
| Metals (element symbols) | 40 | 40 | counts.json › unique.*.metals_symbol |
| Metal states, known oxidation state | 43 | 43 | unique.*.metal_states_known_ox |
| Metal state labels incl. "X(?)" | 74 | 74 | unique.*.metal_states_incl_unknown_ox; 31 elements have unknown-state rows [metals_with_unknown_ox_rows] |
| Distinct oxidation-state values | 6 | 6 | unique.*.oxidation_states |
| Metal categories | 6 | 6 | unique.*.metal_categories |
| Extractant systems (`extractant_system_key`) | 262 | 259 | unique.*.extractant_system_keys |
| Primary-extractant SMILES | 246 | 243 | unique.*.extractant_primary_smiles |
| Primary-extractant name strings | 284 | 280 | unique.*.extractant_primary_names (names are unreliable, §9) |
| Extractant component structures (descriptor table) | 283 structures + 11 name-only | – | manifests/g19_build_extractants.json › summary.n_structures, n_name_only |
| Extractant families | not in archive | 26 system-family labels (structure-derived by Gen19) | counts.json › extractant_families.status = NOT_IN_ARCHIVE; feasibility.json › F03.n_families_plotted; per-structure counts in manifests/g19_build_extractants.json › summary.structure_family_counts (diglycolamide 89, monoamide 37, hydrophilic_aqueous_agent 36, pyridine_carboxamide 30, n_heterocyclic 26, …) |
| Structures with two unrelated family rules firing | 14 | – | summary.family_rules_status_counts.AMBIGUOUS |
| Structures whose canonical name fails a name–structure check | 5 (Br-Cosan, TPDGA malonamide, TDGA, TBADIPIC, NDDIPIC) | – | summary.name_structure_conflict_structures; `family_status = NAME_STRUCTURE_CONFLICT` in `descriptors/extractant_components.csv` |
| Publications (`g19_publication_id`) | 169 | 157 | unique.*.publications |
| Studies (`g19_study_id`) | 263 | 251 | unique.*.studies |
| Acids (`acid_primary`) | 8 | 8 | unique.*.acid_primary; `acid_signature` 9 / 8 |
| Diluents: `solvent_key` / primary solvent / components / family | 83 / 49 / 51 / 9 | 81 / 47 / 49 / 9 | unique.*.diluent_* (diluent-family rules for trade names are INFERRED, counts.json › diluent_family_rules) |
| Temperatures (°C values) | 28 | 26 | unique.*.temperatures_C |

## 6. Metal × extractant matrix

Three matrices were written, all over MODEL rows:
- `matrix_rows_system_x_metal.csv`: 259 systems × 74 state labels, row counts per cell.
- `matrix_pubs_system_x_metal.csv`: the same layout, publication counts per cell.
- `matrix_rows_primary_extractant_x_metal.csv`: 243 primary SMILES × state labels.

Figure: [F01_observation_matrix.png](figures/F01_observation_matrix.png) asks which metal states each system has been measured with. It is truncated to the top 80 systems by rows, which hold 88% of rows (stated in the figure title).

## 7. Sparsity

[sparsity.json › system_x_metal_state; known-state MODEL rows, 11,049 rows]

| Measure | Value |
|---|---|
| Shape (systems × metal states) | 258 × 43 = 11,094 cells |
| Non-zero cells / density | 1,520 / 0.137 |
| Cells with ≥5 / ≥10 / ≥20 rows | 500 / 291 / 123 |
| Cells with ≥2 / ≥3 publications | 200 / 94 |
| Systems spanning ≥3 / ≥5 / ≥10 metal states | 126 / 101 / 66 |
| Metal states spanning ≥3 / ≥5 / ≥10 systems | 30 / 25 / 23 |
| Gini of rows over non-zero cells | 0.718 |
| Row share of top system (TODGA, 2,491 rows) / top 5 systems | 0.225 / 0.380 |
| Row share of top publication / top 5 publications | 0.099 / 0.304 |

Other layouts:
- **Unknown states as their own columns:** 259 × 74, 1,610 non-zero cells, density 0.084 [system_x_metal_state_incl_unknown_ox].
- **States pooled per element:** 259 × 40, 1,492 non-zero cells, density 0.144. Here 93 systems span ≥5 elements and 21 elements span ≥5 systems [system_x_metal_symbol].

## 8. Pr/Nd coverage

| | Pr | Nd | Source |
|---|---|---|---|
| MODEL rows | 369 | 685 | feasibility.json › Q9_Q10_prnd.*.model_rows |
| Known state (III) / unknown state | 348 / 21 | 433 / 252 | rows_known_III, rows_unknown_state |
| Systems / publications / families | 80 / 43 / 12 | 87 / 67 / 13 | systems, publications, families |
| Rows by acid | HNO3 226, HCl 137, malonic 6 | HNO3 576, HCl 67, H2SO4 23, malonic 8, lactic 4, tartaric 4, citric 3 | Q9_Q10_prnd.*.acids |
| Rows in diglycolamide systems | 271 (38 systems, 22 publications) | 539 (42 systems, 41 publications) | feasibility_prnd.csv, breakdown family |
| Rows in acidic cation-exchange systems | 1 | 1 | feasibility_prnd.csv, breakdown mechanism |

Systems measuring both:
- 95 systems have Pr or Nd, and 72 have both [feasibility.json › Q11_ln_neighbours.n_systems_with_pr_or_nd; Q9_Q10_prnd.systems_with_both_pr_and_nd].
- **Comparable Pr/Nd pairs** (same publication, identical condition key): 277 row pairs in 202 condition groups, across 72 systems and 38 publications [feasibility.json › V6_prnd_double_cell].
- By acid medium the 277 pairs are HNO3 189, HCl 85 and malonic acid 3 [feasibility.json › V6_prnd_double_cell.total_comparable_prnd_row_pairs_by_acid].
- **TODGA** supplies 83 of those pairs: HCl 54, HNO3 26, malonic acid 3 [feasibility_v6_systems.csv › TODGA.comparable_prnd_row_pairs, comparable_prnd_row_pairs_by_acid].
- Only 6 systems have two or more publications that measured both metals [prnd_coverage.csv, publications_with_both ≥ 2 †].

The 252 unknown-state Nd rows sit in 9 systems and 21 publications [metal_coverage.csv › Nd(?)]. They are the largest alias block among the lanthanides (§9).

## 9. Lanthanide coverage

- Of the 15 Ln(III) states, all 14 except Pm(III) have ≥201 MODEL rows. Pm(III) has 1 row [metal_coverage.csv › model_rows].
- Eu(III) is the best covered: 1,692 rows in 201 systems and 73 publications. Tm(III) is the thinnest after Pm: 201 rows in 55 systems [metal_coverage.csv].
- `lanthanide_coverage.csv` lists 209 lanthanide-bearing systems, one row each. Of these, 89 have ≥3 Ln(III) states [n_ln_iii ≥ 3 †] and 53 have ≥14 Ln(III) states [n_ln_iii ≥ 14 †].
- Acid variety is mainly a lanthanide feature. Ln(III) states other than Pm(III) have 3–7 acids each (HNO3, HCl, H2SO4, HClO4 for Eu, and citric, lactic, malonic and tartaric acids); Pm(III) has HNO3 only. Among actinides only Am(III) has more than HNO3 (HCl and HClO4) [feasibility_metal_states.csv › n_acids, acids].
- 66 systems carry a contiguous Ce–Pr–Nd–Sm run of Ln(III) inside a single publication. By family: diglycolamide 34, pyridine_carboxamide 14, n_heterocyclic 5 and others [feasibility.json › Q11_ln_neighbours].

Figure: [F04_lanthanide_coverage.png](figures/F04_lanthanide_coverage.png).

## 10. Actinide coverage

| State | MODEL rows | Systems | Publications | Source |
|---|---|---|---|---|
| Am(III) | 1,733 | 129 | 39 | metal_coverage.csv |
| U(VI) | 1,036 | 49 | 42 | same |
| Th(IV) | 773 | 46 | 22 | same |
| Pu(IV) | 580 | 54 | 40 | same |
| Np(IV) / Np(V) / Np(VI) | 216 / 187 / 67 | 16 / 13 / 11 | 8 / 5 / 6 | same |
| Cm(III) | 135 | 12 | 12 | same |
| Pu(III) / Pu(VI) | 118 / 107 | 9 / 13 | 6 / 8 | same |
| Pa(V), Am(VI), U(IV), Cf(III) | 43, 27, 15, 9 | 1, 3, 3, 2 | 2, 1, 3, 2 | same |
| Unknown state Am(?) / U(?) / Th(?) / Pu(?) / Cm(?) | 274 / 46 / 29 / 16 / 9 | – | – | same |

**Overlap with lanthanides:**
- `actinide_coverage.csv` lists 171 actinide-bearing systems, one row each. 122 of them also have lanthanide MODEL rows [feasibility.json › Q12_actinide_lanthanide_overlap.n_systems_with_actinide_and_lanthanide_model_rows], and 117 have both Am and Eu [actinide_coverage.csv, has_am_and_eu †].
- Actinide–lanthanide rows with identical conditions in the same publication: 1,207 row pairs in 110 systems and 20 publications. 802 of the pairs are Am(III)/Eu(III), and 73 pair an actinide with Pr or Nd [Q12_actinide_lanthanide_overlap].
- Actinide chemistry here is nitric-acid chemistry. Every known-state actinide except Am(III) was measured in HNO3 only; Am(III) adds HCl and HClO4 [feasibility_metal_states.csv › acids].
- The monoamide family is essentially actinide-only: 2,144 of its 2,154 MODEL rows are actinide rows [feasibility_families.csv › monoamide].

Figure: [F05_actinide_coverage.png](figures/F05_actinide_coverage.png). Density by metal is in [F02_density_by_metal.png](figures/F02_density_by_metal.png), where the implausible Sr(III) state is marked.

## 11. Alias collisions

### 11a. Ligand aliases

The check summary rows of [ligand_alias_collisions.csv] report:

| Check | Checked | Member rows written | Groups † |
|---|---|---|---|
| One name → several structures | 307 names | 83 | 34 |
| One structure → several names | 283 structures | 107 | 24 |
| Spelling variants (normalised name keys) | 306 keys | 2 | 1 (Cy5S-Me4-BTBP; merge allowed because both spellings sit on one structure) |
| Raw vs canonical SMILES disagreement | 286 pairs | 0 | 0 |
| Same parent structure, different canonical SMILES (salt / charge / stereo forms kept apart by the archive) | 275 parent keys | 15 | 7 (CDTA, DTPA, HEDTA, p-TODGA syn/anti, Et-Me-TDDGA syn/anti, p-TDDGA syn/anti, Me2-TODGA cis/trans) |

Largest structure → names groups [ligand_alias_collisions.csv › n_group_members]:
- **TWE-18 structure:** 24 names.
- **TODGA structure:** 22 names. Masking-agent names were stored in the extractant name field, and 24 name rows on various structures are rejected as `MASKING_AGENT_NAME` [rejection_reason †].
- **TWE-20:** 10 names. **TWE-19:** 8 names.

Consequences:
- Extractant identity must be keyed on structure (`extractant_system_key`), never on name.
- Name-plausibility checks correct the chemistry of rows whose recorded name fails a check on the recorded structure [descriptors/extractant_components.csv › name_structure_checks_failed, known_name_override; descriptors/extractant_systems.csv › effective_component_basis]. Br-Cosan is stored with a boron-free structure (a known name override); 2-bromodecanoic acid is stored on the N-DPP structure and the name TPDGA on a tetrapropylmalonamide (name-implied structures). Every family label downstream of the rows carries the corrections, with the structure-only value kept beside it in a `*_structural` column or level. The component table changes only under the known override, because a name-implied correction depends on the row's name, not on the structure:
  - The Br-Cosan component row reads family `metallacarborane_anion`, mechanism ACIDIC_CATION_EXCHANGE and acidity ACIDIC. The structure-derived values (monoamide, NEUTRAL_SOLVATING, NEUTRAL) are kept in `family_structural`, `mechanism_structural` and `acidity_class_structural`.
  - The system family follows the corrected row resolution: `metallacarborane_anion+pyridine_carboxamide` for the two Br-Cosan mixtures, `carboxylic_acid+n_heterocyclic` for the two N-DPP mixtures and `diglycolamide` for the malonamide recorded as TPDGA [descriptors/extractant_systems.csv › system_family, system_family_structural].
  - The per-component labels of those 5 systems follow too: `component_families` / `component_core_families` carry the corrected family of each component (voted over the system's rows), and `component_families_structural` / `component_core_families_structural` keep the structure-only families [descriptors/extractant_systems.csv, system_id sys_ee61c4e87b, sys_e78807fde5, sys_b36b4f9fca, sys_7af8395f08, sys_7a6c952561].
  - In `family_coverage.csv` the levels `system_family` and `component_family` use the corrected per-row families, and the levels `system_family_structural` and `component_family_structural` use the structure-only ones. The `mechanism` level was already corrected.
- **Name–structure conflicts.** Five structures carry a canonical name that fails a check [descriptors/extractant_components.csv › family_status = NAME_STRUCTURE_CONFLICT]:
  - Br-Cosan (no B);
  - "TPDGA" on tetrapropylmalonamide (not a diglycolamide);
  - "TDGA" on thiodiglycolic acid (the diglycolamide check now requires two amide carbonyls);
  - "TBADIPIC" and "NDDIPIC" on 2,6-bis(aminoacetyl)pyridines ("DIPIC" read as dipicolinamide, INFERRED).

  These systems stay in training but are never V3/V4 units.
- **Stereo and salt forms.** The SAME_PARENT rows above are not linked by the exact-SMILES "shares a component" rule. The stereo-free Me2-TODGA structure (64 MODEL rows) and cMe2-/tMe2-TODGA (16 + 16) are one constitution. Whether the stereo-free record is a mixture or an unrecorded isomer is an open curation item; hiding through the parent structure is a registered sensitivity [ligand_alias_collisions.csv, SAME_PARENT_DIFFERENT_CANONICAL_SMILES; feasibility_v5_grid.csv › n_cells_hidden_rows_in_system_sharing_parent_structure].

**Open (INFERRED): DMDOHEMA.** The name DMDOHEMA is accepted on a structure with 161 MODEL rows and rejected on another with 82 MODEL rows [ligand_alias_collisions.csv, name == DMDOHEMA].
- The 82-row structure's systematic name is `2-(2-hexoxyethyl)-N,N-dimethyl-N,N-dioctylpropanediamide`, which is the chemical meaning of DMDOHEMA.
- The accepted label may therefore be on the wrong structure. This has not been resolved.

### 11b. Metal aliases

[leakage_summary.json › metal_alias; metal_alias_audit.csv]

- **Unknown states:** 1,362 MODEL rows have no oxidation state. States are never imputed.
- **Mixed known and unknown:** 26 elements have MODEL rows with both unknown and known state (Am, Ba, Bi, Ca, Ce, Cm, Dy, Er, Eu, Gd, Ho, La, Lu, Nd, Pb, Pr, Pu, Sm, Sr, Tb, Th, Tm, U, Y, Yb, Zr).
  - Largest unknown-state blocks: Am 274, Nd 252, Eu 214, La 86 and Sr 86 MODEL rows [metal_alias_audit.csv, audit_section state_mix_by_symbol].
- **Several known states:** six symbols have two or more [issues contains mixed_states]:
  - Sr (II/III), Pd (II/IV), U (IV/VI), Np (IV/V/VI), Pu (III/IV/VI), Am (III/VI).
- **Shared near-duplicate keys:** 269 MODEL rows sit in near-duplicate base keys (6 s.f., oxidation state wildcarded, within one publication) shared across state labels of one element [n_model_rows_in_condition_keys_shared_across_state_labels; the key is `leakage.near_duplicate_key(include_metal_state=False)` grouped by (publication, element)]. This is not the 17-field condition key used elsewhere in this document.
- **Bundle labels:** the bundle labels +3 on 797 rows where the archive has no state, 796 of them MODEL [leakage_summary.json › bundle_overlap.bundle_ox_where_archive_ox_unknown, tier_where_archive_ox_unknown].
- **Uranyl:** the archive expands `UO2+2` to U(VI). `normalize_metal` agrees on every label combination it resolves [metal_alias_audit.csv › normalize_agrees].

**Consequence for leave-metal-out folds.** A fold that hides only "Nd(III)" leaves 252 Nd(?) rows in training. Folds must hide every row of the element: all known states plus X(?).

## 12. Duplicates

[leakage_summary.json › exact_duplicates; leakage_exact_duplicates.csv]

| Archive `duplicate_class` | Multi-row groups (all rows) | Rows in them | MODEL groups / rows |
|---|---|---|---|
| A_EXACT_DATABASE_DUPLICATE | 716 | 2,609 | 0 / 0 |
| B_SAME_MEASUREMENT_DIFF_PROV | 9 | 20 | 0 / 0 |
| C_POSSIBLE_INDEPENDENT_REPLICATE | 10 | 20 | 10 / 20 |
| E_VALUE_CONFLICT | 307 | 1,040 | 305 / 1,028 |
| F_INSUFFICIENT_INFORMATION | 86 | 381 | 0 / 0 |

- **Consistency checks:** all 13 checks on the archive's canonical-row bookkeeping pass, 2 of them informational [exact_duplicates.checks †]. The checks include:
  - 1,904 non-canonical rows, all inside A/B groups.
  - One canonical row per A/B group (725 groups).
  - Zero spread in log D inside A/B groups.
  - No non-canonical row in the MODEL tier.
  - `identity_hash` one-to-one with groups (13,828).
- **Value conflicts in MODEL rows:** 305 groups keep several rows with identical conditions and different log D. The largest spread is 2.63 log units, and 4 of these groups span publications; the stored total of 5 also counts one C_POSSIBLE_INDEPENDENT_REPLICATE group [model_rows.max_log_D_spread_in_group, n_multi_row_groups_spanning_publications_by_class, n_multi_row_groups_spanning_publications].
  - A duplicate group that spans publications must not be split by a leave-publication-out fold. The V1 grouping now merges such publications (§13a).
  - These are not leakage, but they set a floor on attainable error.
  - The within-condition replicate SD is given in FEASIBILITY.md, Q13.

## 13. Near-duplicates

**Key.** Pairs share the base key (system, metal, state, acid, solvent, acid M, extractant M, temperature) at 6 or 3 significant figures [leakage_summary.json › near_duplicates.key_fields_base]. The "strict" key adds modifier, complexant, holdback, metal M, O/A and nitrate.

| MODEL rows | 6 s.f. | 3 s.f. |
|---|---|---|
| Pairs | 12,883 | 13,271 |
| Pairs across publications | 926 | 1,020 |
| Publications in cross-publication pairs | 50 | 57 |
| Pairs with strict key equal (of which cross-publication) | 7,478 (35) | 7,674 (55) |
| Pairs with \|Δlog D\| ≤ 0.005 / > 0.3 | 1,595 / 3,826 | 1,665 / 3,919 |
| Rows in pairs | 2,903 | 3,190 |

[near_duplicates.model_rows.sig6, sig3; the full pair list is leakage_near_duplicates.csv, not truncated: pair_listing_truncation]

**Fold-group burden (V1).** At 6 s.f. with any value, 470 MODEL rows in 50 publications sit in cross-publication near-duplicate pairs [v1_near_duplicate_burden_model_rows.key_sig6_any_value].
- Merging those publications gives 12 merged groups holding 4,157 rows, down to 119 publication groups. The largest group spans 18 publications and 2,190 rows.
- Counting only value-matched pairs (\|Δlog D\| ≤ 0.005) gives 178 rows, 8 merged groups, 1,544 rows in merged groups, and 144 publication groups [key_sig6_delta_le_0.005].
- The cumulative merge ladder (corrected DOI → primary-source DOI → archive duplicate group → value copy → near-duplicate key → compilation DOI) is stored per publication in `leakage_publication_components.csv` and summarised in leakage_summary.json › publication_groups.
  - The archive-duplicate-group rung links publications that hold MODEL rows of one `duplicate_group_id` (any class).
  - It was added after verification: two E_VALUE_CONFLICT groups (a La(III) pair and a Pu(IV) triple) spanned copy groups, so 5 of the registered V1 folds failed `fold_isolation_check` on `duplicate_group_shared`.
  - With the rung, `group_cross_publication_copy` has 139 groups with MODEL rows (103 with ≥20 rows). All 104 registered folds pass [feasibility_v1_groups.csv; tests/test_registered_folds.py].

### 13a. Cross-publication copies and double digitisation

**Copies across publications.** A copy is a pair across publications with \|Δlog D\| ≤ 0.005 on the 6 s.f. key. In MODEL rows there are 93 such pairs, covering 178 rows, 21 publications and 14 publication pairs [cross_publication_copies.model_rows_sig6].
- 47 of the pairs are between pub_4a1871d85d and pub_c2d786b979 [top_publication_pairs_model_rows]. Both ids carry DOI 10.13182/nt01-a3198 [leakage_publication_components.csv › g19_publication_refs].
- So this is one source entered under two ids, not two independent measurements.

**Same DOI, different ids** [leakage_summary.json › doi_multiplicity]:
- `load.py` builds 169 publication ids, which correct to 167 [n_g19_publication_ids, n_corrected_publication_ids].
- 4 ids differ only by DOI spelling: a stray trailing character, or PDF page-footer text appended to the DOI. They hold 334 rows, 303 of them MODEL.
- 3 primary DOIs are split across two ids each (risk V1_SPLIT_SAME_SOURCE):
  - 10.13182/nt01-a3198: 114 MODEL rows.
  - 10.1081/sei-100103812: 67 MODEL rows.
  - 10.1002/slct.202202610: 34 MODEL rows.
- The compilation DOI 10.1021/acssuschemeng.4c06166 is co-cited across 34 ids holding 2,066 MODEL rows (COMPILATION_FANOUT).

**Double digitisation** (pairs at 3 s.f., tolerance 0.02) [double_digitisation.model_rows]:
- 2,503 MODEL pair candidates in 54 publications.
- 1,277 of them have identical log D. 1,134 have identical log D and an equal strict key, covering 434 rows.
- By location: 339 pairs are at the same `data_location`, 350 at different locations, and 1,814 have no location recorded.
- In 1,683 of the pairs both rows are in archive E_VALUE_CONFLICT groups [duplicate_class_pairs].
- All 1,180 rows with an explicit digitiser tag (12 publications) are NO_TARGET [explicit_digitiser_tag_rows]. No MODEL pair has two different digitiser tags.

**Does source metadata leak the target?** No signal was found [metadata_target]:
- The best text match of the target inside comments, `data_location`, `ini_comp` or sub-source file names hits 9 rows [max_rows_matched_any_text_check].
- The best provenance grouping predicts log D no better than the best chemistry grouping: leave-one-out group-mean R² 0.485 vs 0.484 on MODEL rows [group_predictability_model].
- Consecutive rows are strongly correlated within a publication (lag-1 r 0.726) and weakly across publications (0.101) [serial_lag1_r]. This is the curve structure, not leakage, but it rules out random row splits as evidence (brief V0).

### 13b. Copies the near-duplicate key cannot see (added 2026-09-15, verification finding VL-04)

The key above, the copy grouping and `fold_isolation_check` all contain `g19_ox` and `extractant_system_key`. A value-matched record that differs only in the state token or in the structure key therefore passes every registered guard. `scripts/g19_build_folds.py` now reports them with `leakage.wildcard_copy_pairs` on MODEL rows (6 s.f. key, \|Δlog D\| ≤ 0.005) [folds/INDEX.json › leakage_sensitivity_wildcard_copies; pair list folds/wildcard_copy_pairs.csv]:

| MODEL rows | state token wildcarded, different state label | structure key dropped, different system |
|---|---|---|
| Pairs | 63 | 202 |
| Rows involved | 111 | 363 |
| Pairs inside one publication | 54 | 178 |
| Pairs with bit-identical, non-decade `D_raw` (of which same figure/table token) | 42 (41) | 60 (23) |
| "Strict" pairs (every state-wildcard pair; structure-wildcard pairs with bit-identical non-decade `D_raw`) | 63 | 60 |
| Pairs across copy groups (of which strict) | 1 (1) | 14 (1) |
| Copy groups with MODEL rows if the cross-group pairs linked publications (now 139) | 138 | 131 |

[STATE_WILDCARD.*, STRUCTURE_WILDCARD.*, n_copy_groups_with_model_rows; linking only the strict cross-group pairs gives 137: n_copy_groups_if_strict_cross_group_pairs_linked]

- A structure-wildcard pair that is not strict may be two independent extractants with the same D at the same conditions; that it is a copy is INFERRED, never asserted. Decade values (e.g. log D −2) are excluded from "strict" because detection floors repeat them (`feasibility.json` › log_D_censoring_candidates).
- **Crossings of the registered folds** (a scored row whose wildcard partner stays in the fold's training rows) [crossings_by_design; row list folds/wildcard_copy_crossings.csv]:
  - V1 exact (copy groups): 20 scored rows in 11 folds (1 state-wildcard, 19 structure-wildcard); 1 strict row in 1 fold.
  - V5-primary exact: 134 scored rows in 67 folds, all structure-wildcard; 26 strict rows in 22 folds.
  - V2 (element and state level): none.
- They are reported, not guarded. The pre-seal run prints exploratory V1 and V5 scores with these rows excluded (`scoring_filter` `wildcard_copies_excluded_scoring` and `wildcard_copies_strict_excluded_scoring` in tables/preseal_summary.csv). Whether they become a registered sensitivity or are merged into the publication groups is an open decision in `preregistration_draft.md` §2.

## 14. §1.4 metadata availability

[metadata_availability.csv, section == field, source == primary; fill = fraction of MODEL rows filled (`fill_model`); PROXY rows are the supplementary text-scan evidence (source == supplement)]

| # | Brief §1.4 field | Status | MODEL fill | Archive columns | Note |
|---|---|---|---|---|---|
| 1 | publication/source ID | DERIVED_BY_GEN19 | 1.000 | `g19_publication_id`, `g19_study_id` | built from uncorrected DOIs (§13a) |
| 2 | DOI | PRESENT | 0.910 | `doi_primary_corrected`, `doi_primary`, `doi_all` | non-DOI rows carry a CORDIS/INIS/thesis reference |
| 3 | table/figure/page | PRESENT | 0.443 | `data_location` | one curator's comments only; no page numbers; PROXY with comment tokens 0.644 |
| 4 | metal | PRESENT | 1.000 | `metal_symbol` | export file ≠ measured metal (§1) |
| 5 | oxidation state | PRESENT | 0.890 | `metal_oxidation_state` | never imputed; Ln 0.877, An 0.931, other 0.615 |
| 6 | extractant | PRESENT | 1.000 | `extractant_names`, `extractant_primary_name` | names untrustworthy (§11a) |
| 7 | extractant family | **ABSENT** | 0 | – | Gen19 derives it from structure (`descriptors/family_rules.json`) |
| 8 | extractant structure identifier | PRESENT | 1.000 | `extractant_system_key`, SMILES | 1,781 rows (675 MODEL) carry a component deduced by elimination (note of the supplement row; 675 = its 12,411 MODEL rows − 11,736 filled) |
| 9 | diluent | PRESENT | 1.000 | `solvent_key` | modifiers often sit inside the diluent string |
| 10 | acid | PRESENT | 1.000 | `acid_primary`, `acid_signature` | |
| 11 | aqueous composition | PROXY | 0.208 | nitrate, complexant, holdback, declared co-metals | "none" cannot be told from "not recorded" |
| 12 | extractant concentration | PRESENT | 1.000 | `extractant_primary_concentration_M` | initial (formal) value |
| 13 | initial metal concentration | PRESENT | 0.557 | `metal_concentration_M` | Ln 0.697, An 0.394; tracer runs often unnumbered |
| 14 | equilibrium/final pH | **ABSENT** | 0 | – | 5 free-text "pH" mentions, all non-MODEL |
| 15 | acidity | PRESENT | 1.000 | `acid_concentration_M` | nominal initial molarity; not pH; see §4 log-grid flag |
| 16 | O/A | PRESENT | 0.588 | `phase_ratio_org_aq` | 8,092 of 8,153 filled values are 1.0; missing ≠ 1 |
| 17 | temperature | PRESENT | 0.940 | `temperature_C` | |
| 18 | contact time | PRESENT | 0.863 | `contact_time_min` or `shaking_time_min` | |
| 19 | saponification | **ABSENT** | 0 | – | 0 free-text mentions |
| 20 | modifiers | PRESENT | 0.050 | `modifier_name` | PROXY via multi-component diluent 0.175 |
| 21 | complexants | PRESENT | 0.091 | complexant / holdback columns | empty is ambiguous |
| 22 | loading | DERIVED_BY_GEN19 | 0.376 | metal M ÷ (extractant M × O/A) | an upper bound from initial values, not a measured loading |
| 23 | measured D | PRESENT | 1.000 | `D_value`, `log_D`, `D_raw` | |
| 24 | D directly reported vs reconstructed | **ABSENT** | 0 | – | PROXY: value located in a figure 0.555; `D_raw` with ≥15 significant digits 0.581 (INFERRED machine-computed) |
| 25 | uncertainty if reported | **ABSENT** | 0 | – | 0 "±" / error mentions |
| 26 | data-status / quality flag | PRESENT | 1.000 | `model_readiness`, `duplicate_class`, `flags`, `in_value_conflict`, `g19_tier` | |

**Totals:** 18 PRESENT, 1 PROXY, 2 DERIVED_BY_GEN19 and 5 ABSENT fields [leakage_summary.json › metadata_availability.status_counts_primary].

**Conflicting fills in `counts.json`.** `counts.json › metadata_availability_section_1_4` gives an earlier, simpler status table from the corpus script. It disagrees in two places:
- **Complexants:** 0.065 there, because it counts `complexant_name` only.
- **Loading:** ABSENT there, because it does not count the derived upper bound.

The table above follows `metadata_availability.csv`.

**Condition values** [condition_coverage.csv]:
- **Phase ratio:** takes 4 distinct values in MODEL rows and only 1 in lanthanide MODEL rows [phase_ratio_org_aq.n_distinct_model, n_distinct_model_lanthanide]. It varies within a group in 2 systems [n_systems_varying].
- **Acid molarity** varies within a group in 158 systems.
- **Extractant molarity** varies within a group in 70 systems.
- **Metal concentration** varies within a group in 19 systems.
- **Not available:** pH, saponification degree, organic loading and ionic strength have fill 0, with a reason in the `note` column.

Figure: [F06_condition_coverage.png](figures/F06_condition_coverage.png).

## 15. Bundle ↔ archive relationship

[leakage_summary.json › bundle_overlap; leakage_bundle_overlap.csv]

- **Join:** all 5,992 bundle rows join the archive 1:1 through `safe_exp_id` = "<stem>_SAFE:<exp_id>" (join rate 1.0).
- **Agreement with the archive:**
  - Metal agrees on 5,992 rows.
  - log D agrees to a maximum \|Δ\| of 5.6e-14.
  - Acid M, extractant M and temperature agree on 5,992 rows.
  - The bundle SMILES equals the archive primary SMILES on 5,992 rows.
- **Oxidation state** agrees on 5,195 rows. On 797 rows the bundle says +3 where the archive has no state.
- **Tiers:** 5,980 bundle rows are MODEL and 12 TARGET_ONLY.
- **Duplicate classes:** 5,326 UNIQUE, 381 E_VALUE_CONFLICT and 275 A_EXACT_DATABASE_DUPLICATE. No bundle row is a non-canonical archive row.
- **Publications:** the bundle rows come from 105 archive publications. The Gen19 publication partition equals the gen6/gen18 partition one-to-one: 5,991 ids identical, at most 1 id per id in both directions.
- **Coverage:** 515 of the archive's 6,495 lanthanide MODEL rows are not in the bundle [archive_model_lanthanide_rows_not_in_bundle].
- **Stem ≠ metal:** the bundle stem is the export file, not the metal. For example, `Am_SAFE:10321` is a Eu(III) row [leakage_bundle_overlap.csv, first row].

## 16. Known traps

1. **"DEHPA" is an amide, not D2EHPA.**
   - The archive's DEHPA is a propanamide, SMILES `CCCCC(CC)CN(CC(CC)CCCC)C(=O)CC`. It appears on 53 rows (25 MODEL, Th/U) [named_extractant_presence.csv › DEHPA, verdict NAME_TRAP_DIFFERENT_STRUCTURE_FAMILY].
   - HDEHP (18 rows, 14 MODEL, Eu/Am) appears only as a name, with no structure, always in the phase-modifier slot [HDEHP, NAME_ONLY_NO_STRUCTURE, roles_on_matching_components]. Chemically it is an **acidic co-extractant** there, not a modifier, so those 14 rows (TODGA 4, DOHyA 10) are synergistic chemistry. `extractant_systems.csv` counts them apart (`n_model_rows_acidic_coextractant_modifier`), and the system label is voted without them.
   - Titles name bis(2-ethylhexyl)phosphoric acid on 48 rows (39 MODEL) in 3 publications whose recorded structures are DOHyA, TEHDGA, HEDTA and CDTA [D2EHPA:FULL_NAME_OR_TITLE_TEXT]. A structure or short-name search misses these.
   - A name search for D2EHPA therefore returns wrong chemistry, and a structure search alone misses the title-level evidence.
2. **TODGA's structure carries 22 names** [ligand_alias_collisions.csv › cmp_a9ed8f70c7:TODGA]. Masking-agent names sit in the extractant name field. Structures are trustworthy; names are not. Key everything on `extractant_system_key`.
3. **Export fan-out.** `raw/<Metal>.csv` and the bundle stem name the queried export, not the measured metal. On 7,334 MODEL rows they differ (§1). Never infer the metal from the file name.
4. **The CORDIS ACSEPT project counts as one publication.**
   - pub_e287124c22 (`https://cordis.europa.eu/project/id/211267`, status REPORT_SUBSOURCE) fronts ~100 study files. It holds 1,096 MODEL rows, the largest single group [leakage_publication_components.csv; feasibility_v1_groups.csv › largest_group_rows].
   - Splitting it by study file would leak one project across V1 folds.
5. **Leave-metal-out must key on the element, including X(?).** See §11b: 26 elements mix unknown and known states, and 6 have several known states. Hiding "Nd(III)" alone leaves 252 Nd(?) rows in training.
6. **Two publication ids can be one paper.** `g19_publication_id` hashes the uncorrected DOI set (§13a). Use a merged group column from `leakage_publication_components.csv` for V1, not the raw id.
7. **Acid molarity may encode pH.** 302 MODEL rows sit on a 0.01 log10 grid, mostly in two publications (§4, INFERRED). Do not treat these values as molarity without a check, and never convert molarity to pH.
8. **Sr(III)** has 38 MODEL rows that the archive itself marks implausible (§4). The tier rule does not exclude them; a policy decision is needed.
9. **The monoamide family is actinide-only in practice** (§10). A "family" effect for monoamides is confounded with the actinide series.
10. **Fingerprints do not tell long-chain homologues apart.** Morgan r=2 Tanimoto is 1.0 between TODGA and TDdDGA, DHD2DGA, TDDGA and THDGA [feasibility_named_analogues.csv, reference TODGA]. Ligand-distance support scores built on this fingerprint are weak inside the diglycolamide family.
11. **An aqueous agent can sit in the extractant slot on its own.**
    - HEDTA, CDTA and thiodiglycolic acid ("TDGA") are the only recorded extractants of 29 MODEL rows. The papers' titles or ligand sets point to TEHDGA + HDEHP and to an S-bridged diglycolamide (INFERRED).
    - Their systems are UNKNOWN (`AQUEOUS_AGENT_RECORDED_AS_EXTRACTANT`), not chelating or acidic [descriptors/extractant_systems.csv › mechanism_flags].
12. **Detection floors look like data.**
    - 188 MODEL rows sit at an exact-decade log D (−4, −3, −2) that is the minimum of their (study, system) group and is shared by ≥2 rows. Another 7 sit at such a ceiling.
    - 156 of the 195 are in the CORDIS project [feasibility.json › log_D_censoring_candidates].
    - They are kept as recorded (INFERRED censoring), flagged per row by `g19_feasibility`, and excluded from scoring in a registered sensitivity.
13. **Acid media are pooled inside some cells.** 38 of the 224 primary V5 cells hold rows from more than one acid; for example, Nd(III) × TODGA mixes HNO3, HCl and four organic acids [feasibility_v5_grid.csv, medium all, k10 p1 m3 › n_cells_multi_acid; feasibility_v5_cells.csv › acids]. Do not assume one response surface across media (brief §4.3).
