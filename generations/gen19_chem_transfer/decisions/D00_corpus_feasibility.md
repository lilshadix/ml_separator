# D00: Corpus feasibility (end of Phase A/B)

Format follows brief §29. Numbers are copied from the files cited; paths are relative to `generations/gen19_chem_transfer/`. The full tables are in `DATA_AUDIT.md` and `FEASIBILITY.md`. No model was trained or fitted.

## Question

Does the expanded SAFE corpus support the transfer tests Gen19 is built to run? Specifically:
1. Is the metal–extractant graph connected?
2. Does it support V1 (publication), V2 (metal) and V5 (missing cell) hold-outs?
3. For which chemistry?
4. Can the applied Pr/Nd case (PC88A / Cyanex 272 / D2EHPA / TODGA, brief §19) be tested?

## Evidence

**Inputs**
- **Dataset:** `dataset_all_metals/clean/master_clean.parquet`, sha256 `7b32979383c5246f22d36ade6246231a347ea22490f4589a76e189af07b09b01`. It equals the pinned value, and all 63 files with a recorded digest verify [data_audit/dataset_hashes.csv].
- **Manifests:** `manifests/g19_audit_corpus.json`, `g19_build_metals.json`, `g19_build_extractants.json`, `g19_audit_leakage.json` and `g19_feasibility.json`, all at git head 40f6a75. Their listed output hashes matched the files on disk when this decision was written.

**Machine-readable evidence**
- `data_audit/counts.json` and `sparsity.json`
- `data_audit/leakage_summary.json` and `metadata_availability.csv`
- `data_audit/named_extractant_presence.csv` and `family_coverage.csv`
- `data_audit/feasibility.json` and the `feasibility_*.csv` tables

**Figures**
- `figures/F01_observation_matrix.png`
- `figures/F02_density_by_metal.png`
- `figures/F03_density_by_family.png`
- `figures/F04_lanthanide_coverage.png`
- `figures/F05_actinide_coverage.png`
- `figures/F06_condition_coverage.png`

## Metrics

| Metric | Value | Source |
|---|---|---|
| MODEL rows / systems / publications | 12,411 / 259 / 157 | feasibility.json › population |
| Known-state metal states / rows without state | 43 / 1,362 | same |
| Matrix density (258 × 43, known states) | 0.137 (1,520 non-zero cells) | sparsity.json › system_x_metal_state |
| Graph components, all edges / ≥5-row edges | 1 / 1; ≥5-row backbone holds 0.881 of known-state rows | feasibility.json › graph_connectivity |
| V1 groups (cross-publication-copy merge, archive duplicate groups included) with ≥20 rows; largest group share | 103; 0.088 | feasibility_v1_groups.csv |
| V2 metal states at ≥100 rows and ≥5 systems | 23 (14 Ln(III), 9 An) | feasibility.json › V2_leave_metal_out.grid |
| V5 eligible cells: loose / proposed (k10, p1, m3) / strict | 359 / 224 / 70 | feasibility_v5_grid.csv |
| V5 cells on HNO3 rows only: loose / proposed / strict; proposed cells pooling >1 acid | 353 / 218 / 57; 38 | feasibility_v5_grid.csv (medium HNO3_only › n_cells; medium all, k10 p1 m3 › n_cells_multi_acid) |
| V5 proposed cells whose component-aware hiding removes hidden-state or X(?) rows in a component-sharing system (parent-structure link) | 36 (40) | feasibility_v5_grid.csv › n_cells_hidden_rows_in_system_sharing_component, n_cells_hidden_rows_in_system_sharing_parent_structure |
| V5 acidic, chelating or synergistic cells | 0 of 224 † | feasibility_v5_cells.csv (mechanism tally) |
| V6 Pr/Nd systems (≥5 rows each, ≥2 other Ln); with ≥2 common publications; comparable pairs (HNO3) | 13; 4; 209 (142) | feasibility.json › V6_prnd_double_cell.grid |
| TODGA Pr/Nd comparable pairs by medium: HCl / HNO3 / malonic | 54 / 26 / 3 | feasibility_v6_systems.csv › comparable_prnd_row_pairs_by_acid |
| V3 systems (≥50 rows, ≥3 states, ≥2 peers) / V4 families (≥3 systems, ≥200 rows, name–structure conflicts excluded) | 30 / 4 | Q4_V3_V4_experts |
| V7 system × acid units (20% region, ≥10 rows in and out): acid M / extractant M | 65 / 33 | feasibility_v7_grid.csv |
| MODEL rows by expert: E1 acidic / E2 neutral / E3 ion-pair / E4 chelating / E5 synergistic / UNKNOWN | 14 / 12,273 / 0 / 3 / 92 / 29 | feasibility_mechanisms.csv |
| PC88A / Cyanex 272 / D2EHPA / TODGA direct MODEL rows | 0 / 0 / 0 / 3,424 | feasibility_candidate_extractants.csv |
| Metal effect at fixed conditions vs replicate spread (pooled SD) | 0.658 vs 0.299 | feasibility_q13_variance.csv |
| Censoring candidates: floor / ceiling MODEL rows | 188 / 7 | feasibility.json › log_D_censoring_candidates |
| Condition-matched cross-publication groups | 2 | feasibility_q13_variance.csv › publication_effect_within_system_metal_condition, key_variant full_key, n_groups |
| §1.4 fields PRESENT / PROXY / DERIVED / ABSENT | 18 / 1 / 2 / 5 | leakage_summary.json › metadata_availability |
| Actinide–lanthanide comparable row pairs (Am/Eu; with Pr/Nd) | 1,207 (802; 73) | feasibility.json › Q12 |

## Verdict: null / supported / ambiguous

Each claim gets a separate verdict.

| Claim | Verdict |
|---|---|
| The corpus graph is connected | **supported** (CONNECTED at all four variants) |
| V1, V2 and V5 are feasible | **supported**, restricted to neutral-solvating chemistry; mostly nitrate media, with HCl and organic-acid rows pooled in 38 of the 224 proposed V5 cells (scored per medium, HNO3-only sensitivity) |
| V3, V4, V6 and V7 are feasible | **ambiguous**: feasible only at small unit counts, inside the diglycolamide/monoamide/soft-N block |
| Acidic-extractant transfer, mechanism-expert routing across brief experts, and the PC88A / Cyanex 272 / D2EHPA applied case | **null**: no rows, no same-family structure (PC88A, Cyanex 272), or one 10-row analogue (D2EHPA; HDEHP otherwise appears only as a co-extractant name and in 3 paper titles) |
| Source (publication) effects can be estimated from matched replicates | **null**: 2 matched groups |

## Decision

1. **Proceed to Phase C with a registered scope restriction.** Transfer claims will be made only for neutral-solvating chemistry (brief expert E2) in the media the corpus covers.
   - Any statement about acidic organophosphorus extractants is out of scope and labelled UNSUPPORTED.
   - This is a scope decision, not a stop: the graph is connected and the three first-run tests (V1, V2, V5) have enough units.
2. **Name Nd(III) × TODGA as the V5 test cell and TODGA Pr(III)+Nd(III) as the primary V6 target**, in place of PC88A × Nd. They replace it only as the named test cells for the transfer machinery. They carry no information about PC88A-type cation-exchange Pr/Nd processing, because the acid dependence has the opposite sign and PC88A needs pH and saponification. The other multi-publication V6 systems are DMDODGA, C5BTBP and DMDPDGA. For a nitrate process case the TODGA V6 evidence is its 26 HNO3 pairs, not all 83.
3. **Fold rules to carry into the pre-registration** (FEASIBILITY.md V-table, all PROPOSED):
   - **V1:** groups come from `group_cross_publication_copy` (now including archive duplicate-group links), not the raw `g19_publication_id`; all 104 registered folds pass the guard.
   - **V2:** hide every row of the element, including X(?) rows.
   - **V5:** k≥10, p≥1, m≥3. Hide, **in the cell's own system and in every system sharing a component**, the rows of the hidden metal state and the element's X(?) rows (36 proposed cells have such rows in a component-sharing system). Other known states of the element stay in training: they are a different species, and element-level transfer is V2's question. Sensitivity settings at 359 and 70 cells; HNO3-only cells; parent-structure hiding; censoring candidates excluded from scoring.
   - **V6 carve-out:** `V6_TARGET_ROWS` (580 Pr/Nd rows of the 13 V6 systems and of component-sharing systems) are never scored before the confirmation run.
   - **Selection / confirmation halves** of V5 systems (105 / 105 scored cells), V1 folds (52 / 52) and V2 states (12 / 11) are fixed now (`data_audit/feasibility_halves.csv`); ladder decisions and claim freezing use the selection half only.
   - **Pairs:** generated after fold assignment.
4. **Do not build mechanism experts beyond E2.** The §31 "+ mechanism" and §32 routing arms are registered as within-E2 sub-groupings (O-donor vs soft-N vs mixed), INFERRED, or dropped. The genuinely acidic rows number 14; the HEDTA, CDTA and TDGA systems (29 rows) are UNKNOWN because their recorded "extractant" is an aqueous agent.
5. **Process layer.** Gen18 may consume Gen19 predictions only for TODGA-type systems in nitrate media (the HNO3 rows and pairs; the HCl and organic-acid TODGA rows are another response surface). PC88A, Cyanex 272 and D2EHPA remain literature-placeholder cases in Gen18 and must not be relabelled as data-supported.

## Next action

**Phase C.**
1. **Build folds** under `folds/`: V1, V2 and V5 per the proposed definitions, using `gen19ct.folds.registered` (V6 carve-out, halves) and `support_graph.hide_cell(component_aware=True)`. Each fold gets a fold hash, plus a test asserting that every hidden cell and element is absent from training (brief §27); `tests/test_registered_folds.py` already runs the registered V1, V2, V5 and V6 splits through the guard.
2. **Implement baselines B0–B8** (brief §13), all evaluated on those folds:
   - B0: global mean.
   - B1: metal mean.
   - B2: extractant mean.
   - B3: Gen18 nearest-condition lookup.
   - B4: the same lookup with the source excluded.
   - B5: a tree model on descriptors.
   - B6: metal × extractant factorisation.
   - B7: mass action, neutral-solvating systems only.
   - B8: the previous best Gen-series model.
3. **Seal the draft pre-registration** before any B5, B6, B8 or M outer score exists: pre-seal Phase C runs only B0–B4 (with B3x, B3i, B3l, B4x) and B7. The success margins are already fixed numerically in the draft. *As run (disclosed in pre-registration §0, order of work item 1):* the pre-seal run used a fuller arm set. It also ran B4l, the pair yardsticks FLAT and HEAVIER, split-conformal intervals and V0, none of them a learned model (`manifests/g19_run_preseal.json` → `arms`, `pair_yardsticks`, `learned_arms_fitted_or_scored` = `[]`). The pre-registration should record:
   - the scope restriction;
   - the V5 threshold, with its sensitivity grid;
   - the §30 actinide ablation (paired cluster bootstrap on the same lanthanide cells);
   - the replicate floor (pooled SD 0.299), below which differences carry no weight.

**Blocking items to settle before fold hashes are frozen** (shared code is owned by the orchestrator):
1. **Publication ids.** `gen19ct/data/load.py` builds `g19_publication_id` from uncorrected DOIs. 4 ids differ only by DOI spelling, holding 303 MODEL rows [leakage_summary.json › doi_multiplicity]. The workaround is the merged group column.
2. **Provenance guard list: resolved.** `load.PROVENANCE_COLUMNS` now lists all 69 columns of the schema's Provenance section, including `D_raw`, the raw target string, plus the target-derived `in_value_conflict`; no schema-provenance column is missing [data_audit/columns.csv, in_PROVENANCE_COLUMNS †]. `tests/test_manifest.py::test_provenance_guard_list_covers_schema_provenance_and_target_derived_columns` passes without an xfail marker.
3. **Policy decisions:**
   - Sr(III) (38 implausible MODEL rows);
   - the 302 log-grid acid rows;
   - the DMDOHEMA name–structure assignment;
   - the five name–structure-conflict structures;
   - the stereo-free vs stereo-specified Me2-TODGA identity.
