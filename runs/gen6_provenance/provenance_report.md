# Provenance audit

* rows: 5992, extractants: 190
* upstream tables: /Users/lilshadix/PycharmProjects/lanthanide_dataset_builder/raw_data (5992/5992 rows joined, 100.00 %)

| identifier | status | definition |
|---|---|---|
| `experiment_id` | **reconstructed** | safe_exp_id — the upstream row id, unique on every row |
| `publication_id` | **reconstructed** | canonical set of DOIs from the upstream SAFE export, excluding the SAFE database self-citation 10.1021/jacs.5c19738, hashed to pub_<sha1[:10]> |
| `experiment_series_id` | **reconstructed** | (publication_id, comments_description) from the upstream export. The description is free text such as 'Fig 2 …' or 'Table 4 …', so a series is only as well defined as the paper's own figure labelling; rows without a description fall back to the bundle surrogate. |
| `replicate_id` | **surrogate** | SURROGATE only. Identical conditions WITH identical D were collapsed upstream (the dedup key includes D), so a cell that repeats here differs in something. Read the two counts separately: repeated_cells_with_a_PHYSICAL_hidden_axis is the defensible 'different experiment' count; repeated_cells_differing_only_in_free_text differ only in the paper's own figure/table caption, which a genuine replicate reported twice also does. Neither count licenses a claim about the measurement noise floor on its own -- see repeated_cell_log_d_range_by_subset, where the cells with NO recoverable difference scatter more than the ones with one. |

## Counts
* publications: 105
* series: 690  |  cells: 5302  |  experiments: 5992
* rows_per_publication: median 26, q25–q75 8–54, max 459 (n=105)
* extractants_per_publication: median 1, q25–q75 1–3, max 58 (n=105)
* metals_per_publication: median 3, q25–q75 1–14, max 14 (n=105)
* rows_per_series: median 1, q25–q75 1–8, max 200 (n=690)
* conditions_per_series: median 1, q25–q75 1–3, max 75 (n=690)
* metals_per_series: median 1, q25–q75 1–2, max 14 (n=690)
* extractants appearing in more than one publication: 28 (3710 rows)

## How much of the modelling structure crosses study boundaries
| grouping | groups | groups spanning >1 publication | rows affected |
|---|---|---|---|
| extractant x condition x metal (averaged in gen5) | 5302 | 3 | 6 (0.10 %) |
| levels.series_labels — the CV group gen5's unseen_series used | 344 | 27 | 1577 (26.32 %) |
| the pair-cohort key (log_SF) | 2459 | 8 | 61 (1.02 %) |
| the CV group of unseen_ligand (same ligand, several papers) | 190 | 28 | 3710 (61.92 %) |

## Are the repeated cells replicates?
* repeated cells: 313 holding 1003 rows
* their log D range: median 0.398, mean 0.817, p90 2.352, max 5.591; 45.4 % exceed 0.5 log units and only 3.8 % are exactly zero
* cells differing in a **physical** variable the bundle drops: **79** of 313
* cells differing **only in the paper's own figure/table caption**: 208 — a genuine replicate reported in two figures looks exactly like this, so these are NOT evidence of a different experiment
* cells with no recoverable difference at all: 26 (counting only physical axes, 234 would qualify)
* per-axis breakdown (cells in which the upstream column varies):
    * `Solvent_Name`: 36 cells (11.5 %)
    * `f_Solvent_Name`: 36 cells (11.5 %)
    * `Phase_Modifier_Concentration_M`: 6 cells (1.9 %)
    * `Shaking_Time_min`: 2 cells (0.6 %)
    * `Metal_Oxidation_state`: 14 cells (4.5 %)
    * `ini_comp`: 58 cells (18.5 %)
    * `comments_description`: 233 cells (74.4 %)

| subset | cells | median log D range | max |
|---|---|---|---|
| physical axis varies | 79 | 0.342 | 5.591 |
| free text only | 208 | 0.368 | 4.111 |
| no recoverable difference | 26 | 0.875 | 1.978 |

  **How far this goes, and no further.** A quarter of the repeated cells differ in a recoverable *physical* variable, so for those the published noise floor is partly missing-feature error rather than measurement error. It does not follow that the floor is wrong overall: most repeated cells differ only in a caption, and the cells with no recoverable difference at all scatter *more* than the ones with a hidden axis. The defensible statement is that the floor is contaminated and its true value is unknown, not that it is an artefact.

## Dataset variants
* `legacy_compatible`: 5992 rows (everything).
* `provenance_strict`: 5992 rows (0 dropped). rows with an identifiable primary publication, with the averaging cell keyed by (extractant, condition, metal, publication_id) instead of (extractant, condition, metal). Materialised as the `strict_cell_id` column; a cohort builder that groups on it never averages two studies into one target value.
* averaging cells: legacy 5302 -> strict 5305 (**3** cells split apart because they mixed two studies)
* strict == legacy: **False**
