# Gen19 feasibility (brief §28, Phase B)

**Scope.** This document covers brief §28 Phase B, questions 1–13. It then gives:
- the V0–V7 support table;
- the feasibility of the §30 actinide, §31 architecture and §32 priors questions;
- the §19 applied Pr/Nd case.

No model was trained or fitted.

**Citations.** The conventions are those of `DATA_AUDIT.md`:
- Every number is copied from a Gen19 output. Paths are relative to `generations/gen19_chem_transfer/`, and a bare file name means `data_audit/<file>`.
- **†** marks a row tally of the cited table under the stated filter.
- The population is the 12,411 MODEL rows. Of these, 11,049 have a known oxidation state and 1,362 do not; they span 259 extractant systems and 157 publications [feasibility.json › population].
- Threshold grids are recorded in `manifests/g19_feasibility.json › arguments`.

Definitions proposed below are marked **PROPOSED**. They are inputs to the pre-registration and are not sealed.

---

## Verdict in brief

- **The metal–extractant graph is CONNECTED.** The corpus supports:
  - leave-publication-out (V1);
  - leave-metal-out (V2);
  - missing-cell transfer (V5).

  All three hold **only inside neutral-solvating chemistry**: diglycolamides, monoamides and soft N-donors.
- **The media are mostly, not only, nitrate.** At the headline setting 39 of 359 V5 cells pool more than one acid, and 696 of their 8,184 rows are not HNO3 (HCl and organic acids) [feasibility_v5_grid.csv, medium all › n_cells_multi_acid, n_rows_non_hno3]. On HNO3 rows alone the grid keeps 353 / 218 / 57 cells at the loose / primary / strict settings [medium HNO3_only › n_cells]. Different acid systems are not assumed to share a response surface (brief §4.3): V5 is scored per acid medium as well, with an HNO3-only sensitivity.
- **Limited:** V3 (leave-extractant-out), V4 (leave-family-out), V6 (Pr/Nd double hold-out) and V7 (condition extrapolation).
- **Not supported:**
  - any claim about acidic organophosphorus extractants: PC88A, Cyanex 272, D2EHPA;
  - mechanism-expert routing across the brief's five experts;
  - any condition axis involving pH, saponification or measured loading.
- **Named test cells in place of PC88A × Nd:** Nd(III) × TODGA in V5 and TODGA Pr(III)+Nd(III) in V6. They replace PC88A × Nd only as the named test cells for the transfer machinery. They carry no information about PC88A-type cation-exchange Pr/Nd processing: TODGA extracts neutral metal nitrates, with log D rising with acidity, whereas PC88A exchanges H+ at pH 1–4 and depends on saponification (chemistry, not a corpus number).

## Graph connectivity

[feasibility.json › graph_connectivity; feasibility_graph_components.csv]

**Graph.** The bipartite graph links metal states to extractant systems, with MODEL rows as edges.

| Variant | Components | Giant: states / systems / rows | Share of population rows on the giant | Verdict |
|---|---|---|---|---|
| Known state, all edges | 1 | 43 / 258 / 11,049 | 1.000 | CONNECTED |
| Known state, edges with ≥5 rows | 1 | 41 / 131 / 9,732 | 0.881 | CONNECTED |
| Including X(?) states, all edges | 1 | 74 / 259 / 12,411 | 1.000 | CONNECTED |
| Including X(?) states, edges with ≥5 rows | 1 | 65 / 133 / 11,019 | 0.888 | CONNECTED |

- **Threshold.** The verdict rule is that the giant component holds ≥0.95 of rows [connected_share_threshold].
- **Where the ≥5-row backbone is thin.** On that backbone, 127 systems have no ≥5-row edge, and neither do Cd(II) and Pm(III) [known_state_edges_ge5.n_systems_without_any_qualifying_edge, metal_states_without_any_qualifying_edge].
- **Full support multigraph.** It adds the family, publication and condition-regime layers. It is one component of 601 nodes: 74 metal states, 259 systems, 26 families, 157 publications and 85 condition regimes [support_multigraph].

**Reading.** Connectivity is necessary but not sufficient.
- The graph is held together by a neutral-solvating hub. TODGA alone is 0.225 of known-state rows [sparsity.json › system_x_metal_state.share_rows_top1_system].
- The acidic, chelating and synergistic systems hang off that hub through thin edges (Q4). A connected graph does not mean that transfer into those blocks is supported.

---

## Q1. How many metal × extractant cells have enough data for V5 testing?

**Definition** [feasibility.json › Q1_V5_cells.definition]:
- A cell is (`g19_metal_state`, `extractant_system_key`) over MODEL rows.
- A cell is eligible when it has ≥k rows and ≥p publications, the metal keeps ≥m other systems, the system keeps ≥m other metal states, and the cell edge is not a bridge.
- **Hiding (PROPOSED, the registered reading)** applies one state-level rule in the cell's own system **and in every other system that shares a component structure** (`support_graph.hide_cell(component_aware=True)`): it removes the rows of the hidden metal state and the element's X(?) rows there. Other known states of the element stay in training everywhere (for example Pu(IV) rows under DHOA|TODGA when Pu(VI) × TODGA is hidden): they are a different species, and element-level transfer is what V2 tests. Every `hidden_*` column of `feasibility_v5_cells.csv` is computed under this hiding. For every cell the script asserts that no row of the hidden state and no X(?) row of its element is left in the own or a component-sharing system, and that the hiding removed exactly the cell, its own-system X(?) rows and `hidden_rows_in_other_systems_sharing_a_component`.

**Answer:** from 70 to 359 cells, depending on thresholds. Of the 1,520 non-empty cells, 500 have ≥5 rows [n_cells_ge1_row, n_cells_ge_kmin_rows].

**Threshold sensitivity** (publication basis `g19_publication_id`, medium `all` unless stated) [feasibility_v5_grid.csv]:

| k rows | p pubs | m other | Cells | Systems | Metal states | Ln cells | Pr/Nd cells | Non-diglycolamide cells | Cells with X(?) rows | Cells with hidden-state or X(?) rows in a component-sharing system | Cells pooling >1 acid | Cells on HNO3 rows only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 2 | **359** | 55 | 30 | 185 | 30 | 157 | 51 | 41 | 39 | 353 |
| 5 | 1 | 5 | 304 | 39 | 24 | 179 | 30 | 127 | 48 | 35 | 39 | 298 |
| 5 | 2 | 2 | 157 | 34 | 26 | 84 | 11 | 62 | 48 | 24 | 33 | 150 |
| 10 | 1 | 2 | 236 | 43 | 27 | 123 | 18 | 107 | 49 | 36 | 38 | 230 |
| **10** | **1** | **3** | **224** | 37 | 26 | 123 | 18 | 100 | 49 | 36 | 38 | 218 |
| 10 | 2 | 3 | 132 | 27 | 26 | 65 | 9 | 55 | 46 | 24 | 32 | 125 |
| 20 | 1 | 2 | 105 | 32 | 25 | 43 | 6 | 28 | 38 | 21 | 35 | 91 |
| 20 | 2 | 5 | **70** | 17 | 22 | 34 | 5 | 21 | 34 | 19 | 29 | 57 |

The component-sharing column counts cells whose registered hiding removes rows outside the cell's own system: rows of the hidden state or X(?) rows of its element in a component-sharing system [n_cells_hidden_rows_in_system_sharing_component]. Counted over every row of the element, other known states included (the scope of the element-level rule this replaces), it reads 46 / 39 at the headline / primary settings [n_cells_element_in_system_sharing_component]. In 10 headline / 7 primary cells, rows of another known state of the element stay in a component-sharing system [n_cells_other_known_state_kept_in_system_sharing_component]. The HNO3-only column recomputes rows, publications, partners and bridges on HNO3 rows [medium HNO3_only › n_cells]; the HNO3-only grid rows recompute every per-cell attribute (X(?) rows, component sharing, acids, flags, support features) on HNO3 rows alone [feasibility_v5_cells.csv › `*_hno3` columns].

**Robustness checks:**
- **Publication basis:** merging publications by primary-source DOI (`group_primary_source_doi`) gives identical counts at all 18 settings [feasibility_v5_grid.csv].
- **Bridge condition:** it never removes a cell. `n_cells_failing_only_connectivity` is 0 everywhere.
- **Partners after the registered hiding:** every primary cell's metal keeps ≥3 other systems once the component-sharing systems are removed too [feasibility_v5_cells.csv › other_systems_for_metal_after_registered_hiding †].

**Composition at k=10, p=1, m=3 (224 cells)** [feasibility_v5_cells.csv, eligible__g19_publication_id__k10_p1_m3 †]:
- **By family:** diglycolamide 124, monoamide 37, pyridine_carboxamide 16, podand_ether 14, n_heterocyclic 13, diglycolamide+malonamide 11, diglycolamide+monoamide 4, amino_polyamide 3, diglycolamide+neutral_organophosphate 2.
- **By mechanism:** NEUTRAL_SOLVATING 178, SOFT_N_DONOR 29, MIXED_NEUTRAL 17.
- **No cell** is acidic, chelating or synergistic.

**Caveats that shape the fold definition:**
- **Partial leaks through component-sharing systems.**
  - At the headline setting, 41 cells (36 at k=10, p=1, m=3) have rows of the hidden state or X(?) rows of its element in another system that shares a component [feasibility_v5_grid.csv › n_cells_hidden_rows_in_system_sharing_component]. For example, hiding Nd(III) × TODGA in its own system only would leave 18 Nd(III) rows in 3 systems that contain TODGA [feasibility_v5_cells.csv › hidden_rows_in_other_systems_sharing_a_component, n_other_systems_sharing_a_component_with_metal], and Nd(III) × DMDOHEMA|TODGA would leave 161 Nd rows, of which 101 are Nd(III) and 60 Nd(?) [same row › metal_rows_in_other_systems_sharing_a_component, unknown_state_rows_in_other_systems_sharing_a_component].
  - X(?) rows of the element sit in component-sharing systems for 20 headline cells (17 primary) [n_cells_unknown_state_in_system_sharing_component]. The registered hiding therefore removes them with the hidden state there, exactly as it removes the element's X(?) rows under the cell's own system.
  - Rows of another known state of the element stay in a component-sharing system for 10 headline cells (7 primary) [n_cells_other_known_state_kept_in_system_sharing_component]. Under the primary setting these are five Pu cells, one Am(III) and one U(VI) cell, with 134 such rows in total (for example, 54 Pu(III) or Pu(VI) rows in systems sharing a component with DHOA|TODGA stay when Pu(IV) × DHOA|TODGA is hidden) [feasibility_v5_cells.csv › other_known_state_rows_in_other_systems_sharing_a_component †]. They stay in training by the state-level rule.
  - 51 cells carry X(?) rows under their own system, which are hidden together with the cell.
  - **Parent-structure sensitivity.** Linking systems whose components share a stereo-free, salt-free parent structure (p-TODGA-Syn/-Anti, stereo-free vs cis/trans Me2-TODGA, p-TDDGA and Et-Me-TDDGA Syn/Anti) raises the count to 51 headline cells (10 more) and 40 primary cells (4 more) under the same state-level rule [n_cells_hidden_rows_in_system_sharing_parent_structure, n_cells_extra_under_parent_structure_rule].
- **Acid media are pooled in some cells.** 38 primary cells hold rows from more than one acid; for example, Nd(III) × TODGA is HNO3 62, HCl 12, malonic 8, lactic 4, tartaric 4 and citric 3 rows [feasibility_v5_cells.csv › acids].
- **Censoring candidates.** 10 primary cells (32 rows) hold rows whose log D sits at an exact-decade floor or ceiling of its (study, system) group (see V-table, V5) [n_cells_with_censoring_candidate_rows, n_rows_log_D_censoring_candidate].
- **Weak ligand distance.** After the registered hiding, the remaining nearest ligand has Morgan r2 Tanimoto 1.0 in 158 of 359 headline cells and in 107 of 224 cells at k=10, p=1, m=3 [hidden_nearest_ligand_tanimoto == 1.0 †]. Fingerprint distance cannot rank support inside the diglycolamide homologue series.

## Q2. How many extractants span ≥5 metals?

[feasibility.json › Q2_Q3_spans.systems_spanning_metal_states; unit = extractant system]

| Span threshold | Any cell ≥1 row: known states / elements | Cells ≥5 rows: known states / elements |
|---|---|---|
| ≥3 | 126 / 126 | 43 / 43 |
| **≥5** | **101 / 93** | **27 / 26** |
| ≥10 | 66 / 64 | 14 / 12 |

**Answer:**
- 101 systems touch ≥5 known metal states.
- Only 27 have ≥5 rows in each of ≥5 states.
- Spans computed for a single primary structure (ignoring co-extractants) were not computed. The matrix `matrix_rows_primary_extractant_x_metal.csv` exists but has no span summary.

## Q3. How many metals span ≥5 extractants?

[feasibility.json › Q2_Q3_spans.metal_states_spanning_systems]

| Span threshold | Any cell ≥1 row: known states / elements | Cells ≥5 rows: known states / elements |
|---|---|---|
| ≥3 | 30 / 24 | 28 / 23 |
| **≥5** | **25 / 21** | **24 / 20** |
| ≥10 | 23 / 20 | 16 / 16 |

**Answer:** 25 known metal states span ≥5 systems, and 24 do so with ≥5 rows per cell.

At the V2 thresholds (≥100 MODEL rows and ≥5 systems), 23 states qualify [feasibility.json › V2_leave_metal_out.grid]:
- **14 lanthanides:** La(III)–Lu(III) except Pm(III).
- **9 actinide states:** Th(IV), U(VI), Np(IV), Np(V), Pu(III), Pu(IV), Pu(VI), Am(III), Cm(III).

No transition, post-transition, alkaline-earth or Sc/Y state reaches ≥100 rows. Sr(II) is the largest of these, with 94 rows [feasibility_metal_states.csv].

## Q4. Which extractant families have enough coverage?

**Families** (system-level, structure-derived by Gen19, with the builder's name corrections: a known name override and a name-implied structure count toward the family) [feasibility_families.csv; feasibility.json › Q4_V3_V4_experts.v4_grid]:

| Family | Systems (excl. name–structure conflicts) | MODEL rows (excl. conflicts) | Metal states | Ln(III) / An states | Publications | V4-eligible at ≥3 systems and ≥200 rows, conflicts excluded |
|---|---|---|---|---|---|---|
| diglycolamide | 96 (95) | 7,621 (7,616) | 40 | 15 / 11 | 80 | yes |
| monoamide | 34 (34) | 2,154 (2,154) | 12 | 1 / 11 | 36 | yes |
| pyridine_carboxamide | 30 (30) | 513 (513) | 15 | 14 / 1 | 11 | yes |
| n_heterocyclic | 25 (23) | 713 (709) | 15 | 14 / 1 | 9 | yes |
| malonamide | 12 (12) | 195 (195) | 16 | 14 / 2 | 8 | no (5 rows short) |
| podand_ether | 2 | 278 | 16 | 14 / 2 | 1 | no (2 systems) |
| amino_polyamide | 8 | 186 | 20 | 14 / 3 | 5 | no (<200 rows) |
| carboxylic_acid | 2 (1) | 16 (3) | 5 | 3 / 0 | 2 | no |
| phosphoric_acid | 1 | 10 | 10 | 10 / 0 | 1 | no |
| dithiophosphinic_acid | 1 | 1 | 1 | 1 / 0 | 1 | no |
| phosphonic / phosphinic acid | 0 | 0 | – | – | – | absent (Q5–Q6) |

[n_systems / n_model_rows with n_systems_excl_name_structure_conflict / n_model_rows_excl_name_structure_conflict in brackets; v4_eligible_excl_conflicts__systems3_rows200]

**What the corrections moved** (compared with the first Phase B build):
- The tetrapropyl-malonamide structure recorded under the name TPDGA (5 rows) fails the diglycolamide name check, so it is a **name–structure conflict**. The builder's name-implied structure puts it with the diglycolamides; either way it is excluded from V3/V4 units, which leaves malonamide at 12 systems and 195 rows, below the 200-row threshold.
- EsPyTri (10 rows) is a bis-triazolylpyridine N3 donor whose CH2COOH sits on the pyridine 4-position, where it cannot chelate with the ring N atoms. It is now n_heterocyclic / SOFT_N_DONOR, not carboxylic_acid (`family_rules.json › n_heterocyclic.subsumes_when`).
- The two Br-Cosan mixtures (30 rows) now carry the metallacarborane-anion family of their name override rather than the monoamide family of the boron-free structure; the N-DP(DOM)P|N-DPP and 3-Me-N-DP(DOM)P|N-DPP mixtures (20 rows), whose second component is 2-bromodecanoic acid by name, are carboxylic_acid+n_heterocyclic.
- TDGA (13 rows) is a name–structure conflict (see E1 below) and TBADIPIC / NDDIPIC (2 rows each) fail the new dipicolinamide name check [descriptors/extractant_components.csv › family_status = NAME_STRUCTURE_CONFLICT].

**Threshold sensitivity** [v4_grid; the counts are the same with and without conflicts]:
- At ≥2 systems and ≥100 rows, 7 families qualify.
- At ≥2 systems and ≥200 rows, 5 qualify (podand_ether is the fifth).
- At ≥3 systems and ≥200 rows, 4 families qualify: diglycolamide, monoamide, n_heterocyclic and pyridine_carboxamide. The same 4 qualify at ≥500 rows (with ≥2, ≥3 or ≥5 systems).

Every qualifying family is neutral or soft-N.

**Mechanisms / brief §5 experts** [feasibility_mechanisms.csv; Q4_V3_V4_experts.expert_grid]:

| Brief expert | MODEL rows | Systems | Metal states | Publications |
|---|---|---|---|---|
| E1 acidic cation exchange | 14 | 3 | 11 | 3 |
| E2 neutral solvating (includes SOFT_N_DONOR 1,226 rows and MIXED_NEUTRAL 336) | 12,273 | 243 | 43 | 153 |
| E3 ion-pair / basic | 0 | 0 | 0 | 0 |
| E4 chelating | 3 | 1 | 3 | 1 |
| E5 synergistic | 92 | 9 | 2 | 1 |
| no expert: UNKNOWN | 29 | 3 | 10 | 2 |

- **E1** is dihexyl hydrogen phosphate (10 rows), 2-(dibutylcarbamoyl)benzoic acid (3) and a dithiophosphinic acid (1) [descriptors/extractant_systems.csv, mechanism ACIDIC_CATION_EXCHANGE †].
- **UNKNOWN** holds the three systems whose only recorded "extractant" is a hydrophilic acid or chelator that cannot be the organic-phase extractant (flag `AQUEOUS_AGENT_RECORDED_AS_EXTRACTANT`) [extractant_systems.csv › mechanism_flags]:
  - HEDTA (8 rows) and CDTA (8 rows), both from one paper in dodecane at 3 M HNO3 whose title names tetra(2-ethylhexyl)diglycolamide and bis(2-ethylhexyl)phosphoric acid as the solvent (INFERRED from the title; the solvent composition is not recorded) [named_extractant_presence.csv › D2EHPA:FULL_NAME_OR_TITLE_TEXT];
  - thiodiglycolic acid under the name TDGA (13 Tc/Pd rows). The paper studies diglycolamide-type ligands in dodecane, so TDGA is most likely the S-bridged diglycolamide (INFERRED); its structure is unresolved. Seven of the rows are Tc(VII), the TcO4− anion [descriptors/metals.csv › species_charge], for which cation exchange is impossible.
- **E4** is quercetin (3 rows), an O,O chelator newly covered by the `hydroxyketone_catechol` rule. Before these corrections E4 held the 16 HEDTA/CDTA rows and E1 held 37 rows including TDGA and EsPyTri.
- Grouping SOFT_N_DONOR and MIXED_NEUTRAL under E2 is a Gen19 choice (INFERRED) [expert_mapping_note].
- At every grid setting (≥100/200/500 rows, ≥3 systems, ≥5 states) only **one** expert qualifies: E2 [expert_grid].
- The E5 rows are Eu/Am only.
- **Acidic co-extractants in the modifier slot.** 14 MODEL rows of neutral systems carry HDEHP recorded as a phase modifier: TODGA 4 (Eu(?)) and DOHyA 10 (Am 5, Eu 5) [feasibility.json › acidic_coextractant_modifier_rows]. Those rows are synergistic chemistry. The system label is voted over the rows without them, and no eligible V5 cell contains any [feasibility_v5_grid.csv › n_cells_with_acidic_coextractant_rows = 0].
- Different basis: `family_coverage.csv` counts SYNERGISTIC as 11 systems and 106 rows, because it uses per-row labels (co-extractant rows included) rather than the system majority label.

**Answer.**
- Four families have enough coverage for a family-level hold-out at ≥3 systems and ≥200 rows, all of them neutral or soft-N chemistry.
- No acidic or chelating extractant family has enough coverage for anything; the genuinely acidic rows number 14.

## Q5–Q8. Are PC88A, Cyanex 272, D2EHPA and TODGA present?

**Search scope.** Name text in the master table and in all 41 raw CSVs, the exact canonical SMILES, the connectivity key, a family substructure search, and — added after verification — the full chemical names in the master text columns plus the short aliases and full names in `reference_title` / `publication_title`. The verdicts do not depend on any threshold. [named_extractant_presence.csv; feasibility.json › Q5_Q8_named_extractants; feasibility_candidate_extractants.csv]

| Extractant | Verdict | Name hits (rows / MODEL) | Exact / connectivity structure | Same-family structure in corpus | Notes |
|---|---|---|---|---|---|
| **Q5 PC88A** (also PC-88A, P507, EHEHPA, HEH[EHP], Ionquest 801) | **ABSENT** | 0 / 0 for every alias; full name or title 0 | ABSENT / ABSENT | phosphonic-acid monoester: ABSENT | — |
| **Q6 Cyanex 272** (also Cyanex272) | **ABSENT** | 0 / 0; full name or title 0 | ABSENT / ABSENT | phosphinic acid: ABSENT | — |
| **Q7 D2EHPA** (also P204, HDEHP, DEHPA) | **ABSENT_AS_STRUCTURE_NAME_ONLY_ROWS** | D2EHPA 0, P204 0; HDEHP 18 / 14, always in the phase-modifier slot (Eu/Am) — chemically an **acidic co-extractant**, not a modifier; DEHPA 53 / 25 (**name trap: an amide**, Th/U); **title or full name 48 / 39 rows in 3 publications** whose recorded structures are DOHyA, TEHDGA, HEDTA and CDTA | ABSENT / ABSENT | phosphoric acid PRESENT: dihexyl hydrogen phosphate, 10 MODEL rows, 1 publication, La–Ho, Tanimoto to D2EHPA 0.469 [feasibility_named_analogues.csv] | support class SAME_FAMILY_ONLY |
| **Q8 TODGA** | **PRESENT** | 3,978 / 3,311 | exact structure on 4,098 rows / 3,424 MODEL | diglycolamide: 7,928 MODEL rows in 99 systems | 57 publications; 37 known metal states; support class DIRECT_PR_AND_ND |

- **The D2EHPA title rows** [named_extractant_presence.csv › D2EHPA:FULL_NAME_OR_TITLE_TEXT, row_extractant_structures_model_rows]. Three papers name bis(2-ethylhexyl)phosphoric acid in their title:
  - DOHyA with HDEHP (synergic extraction of Am/Eu);
  - TEHDGA with HDEHP as a "reactive" phase modifier;
  - TEHDGA with HDEHP for Am(III) separation — its rows record HEDTA or CDTA in the extractant slot (the UNKNOWN systems of Q4), including 2 Pr and 2 Nd MODEL rows.

  None of these rows carries an HDEHP structure, and the organic composition of the last paper is not recorded. They are therefore evidence that D2EHPA chemistry was measured, not usable D2EHPA data.
- For comparison, TOPO, Aliquat and HTTA are also ABSENT. TBP is PRESENT: 1,164 rows carry its structure, 118 of them MODEL. CMPO is PRESENT on 17 MODEL rows.
- Nearest analogues for PC88A by Tanimoto, over any family, are neutral amides: DEHBA 0.400, TWE-2 0.395, DEHHA 0.390 [feasibility_named_analogues.csv]. Structural similarity to these points at the wrong mechanism.

## Q9–Q10. How much Pr and Nd data exists?

[feasibility.json › Q9_Q10_prnd, V2_leave_metal_out.ln_focus; feasibility_prnd.csv]

| | Pr | Nd |
|---|---|---|
| MODEL rows (known III + X(?)) | 369 (348 + 21) | 685 (433 + 252) |
| Systems / publications / families | 80 / 43 / 12 | 87 / 67 / 13 |
| Systems with ≥5 rows of the (III) state | 17 | 18 |
| Rows by acid | HNO3 226, HCl 137, malonic 6 | HNO3 576, HCl 67, H2SO4 23, malonic 8, lactic 4, tartaric 4, citric 3 |
| HCl rows: systems / publications | 26 / 7 | 24 / 9 |
| Diglycolamide rows (systems / publications) | 271 (38 / 22) | 539 (42 / 41) |
| Acidic cation-exchange rows | 1 | 1 |
| V2-eligible (≥100 rows and ≥5 systems) | yes | yes |

**Pairs.** 72 systems have both Pr and Nd. Comparable Pr/Nd row pairs (same publication and identical condition key) number 277, in 202 condition groups, 72 systems and 38 publications [V6_prnd_double_cell]. By acid medium: HNO3 189, HCl 85, malonic acid 3 [total_comparable_prnd_row_pairs_by_acid]. A condition key includes the acid, so no pair mixes media.

**Sensitivity.** Dropping metal concentration from the condition key leaves the pair count unchanged at 277 [total_comparable_prnd_row_pairs_nm].

**Other V2 focus lanthanides** (III rows / publications / systems / X(?) rows) [ln_focus]:
- La 344 / 48 / 87 / 86
- Ce 293 / 39 / 71 / 21
- Sm 323 / 42 / 80 / 40
- Eu 1,692 / 73 / 201 / 214
- Gd 375 / 40 / 76 / 37

All seven focus lanthanides are V2-eligible.

## Q11. Which neighbouring lanthanides exist under the same extractants?

[feasibility.json › Q11_ln_neighbours; feasibility_ln_neighbours.csv; Q1 grid]

- **Presence under Pr/Nd systems.** Of the 95 systems that carry Pr or Nd, the (III) state is present for Ce in 71, Pr in 80, Nd in 85, Sm in 79 and Pm in 1.
- **Contiguous runs.** 66 systems carry a contiguous Ce–Pr–Nd–Sm(III) run inside one publication. By family: diglycolamide 34, pyridine_carboxamide 14, n_heterocyclic 5, carbamoylmethylphosphine_oxide 3, aminopolycarboxylic_acid 2, malonamide 2, podand_ether 2, and 1 each for amino_polyamide, diglycolamide+malonamide, phosphine_oxide and phosphoric_acid.
- **V5 lanthanide cells with a Z±1 neighbour under the same system:**
  - 171 of 185 at the headline setting, with 150 bracketed on both sides within ±2;
  - 111 of 123 at k=10, p=1, m=3, with 100 bracketed;
  - 34 of 34 at the strictest setting, with 28 bracketed.
- **Same-charge radius brackets.** Under the registered hiding, 118 of the 210 primary cells scored in discovery have a same-charge metal with a smaller and one with a larger Shannon radius under the same system — the two ends a radius interpolation needs. Among the 109 scored lanthanide cells, 89 are bracketed [feasibility.json › Q1_V5_cells.radius_interpolation_supply_primary_scored › n_cells, n_cells_radius_bracketed_same_charge, n_lanthanide_cells, n_lanthanide_cells_radius_bracketed_same_charge]. Before the V6 carve-out, 103 of the 123 primary lanthanide cells are bracketed [feasibility_v5_grid.csv, medium all, k10 p1 m3 › n_lanthanide_cells_radius_bracketed_same_charge].
- **Pm(III) gap.** Pm(III) has 1 MODEL row. Nd–Sm neighbours are therefore bracketed across a missing Pm.
- **TODGA.** Twelve other Ln(III) states sit beside Pr(III)/Nd(III): La, Ce, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu [feasibility_v6_systems.csv › TODGA.other_ln_iii].

**Answer.** Series neighbours are well covered, which makes a series-smoothness prior testable (§32).

## Q12. Which actinide systems overlap with the same extractants?

[feasibility.json › Q12_actinide_lanthanide_overlap; feasibility_an_ln_overlap.csv]

- **Overlapping systems.** 122 systems have both actinide and lanthanide MODEL rows. 110 of them have comparable actinide–lanthanide pairs (same publication, identical condition key).
- **Pairs.** There are 1,207 row pairs in 765 condition groups, from 20 publications.
  - By pair type: Am(III)/Eu(III) 802; U(VI)/Eu(III) 40; Cm(III)/Eu(III) 37; Pu(IV)/Eu(III) 30.
  - Am(III) with La/Gd/Nd/Sm/Pr/Ce: 28/27/26/25/24/22.
  - Actinide with Pr or Nd, in total: 73.
- **By family.** Diglycolamide 724, n_heterocyclic 142, diglycolamide+malonamide 60, diglycolamide+neutral_organophosphate 42, sulfur_donor_other 40, pyridine_carboxamide 36, carboxylic_acid+n_heterocyclic 31, amino_polyamide 30, and others [families_with_an_ln_pairs_full_key].
- **Key sensitivity.** Dropping metal concentration from the key raises the pairs from 1,207 to 1,940 [n_row_pairs_an_ln_key_excl_metal_conc].
- **Media and family confounds.**
  - Actinide states other than Am(III) are HNO3-only [feasibility_metal_states.csv › acids].
  - The monoamide family is 2,144 actinide rows out of 2,154 [feasibility_families.csv].

**Answer.**
- Structural overlap is broad: 122 systems have both series.
- Condition-matched overlap is narrow, dominated by Am/Eu, and thin for Pr/Nd.

## Q13. Are conditions rich enough to separate metal effects from condition effects?

[feasibility.json › Q13_metal_vs_conditions; feasibility_q13_variance.csv (DESCRIPTIVE: pooled one-way variance of level means, no model)]

| Component | Grouping | Groups | Pooled SD of log D |
|---|---|---|---|
| Metal effect at fixed publication, system and conditions | (publication, system, condition key); levels = metal states | 1,601 | 0.658 |
| Replicate spread | (publication, system, condition key, metal state) | 293 | 0.299 |
| Publication effect at fixed system, metal and conditions | (system, metal state, condition key); levels = publications | **2** | 0.543 |
| same, with metal concentration dropped from the key | | 9 | 0.386 |
| Publication effect, conditions differ (CONFOUNDED) | (system, metal state) | 200 | 0.846 |
| Metal effect, conditions and publications differ (CONFOUNDED) | (system) | 227 | 0.744 |

**Association between design factors** [association_cramers_v]. Cramér's V, bias-corrected in brackets:
- metal state vs acid: 0.171 (0.160);
- metal state vs system: 0.290 (0.247);
- metal state vs publication: 0.418 (0.406);
- system vs publication: 0.730 (0.718).

**Other design measures:**
- **Single-publication systems:** 0.826 of systems come from one publication; they hold 0.294 of MODEL rows [fraction_systems_in_one_publication, fraction_model_rows_in_single_publication_systems].
- **Single-publication metal states:** 0.209 of metal states [fraction_metal_states_in_one_publication].
- **Shared condition groups:** consider the 227 systems with ≥2 metal states (10,713 rows) [n_systems_with_ge2_metal_states, rows_in_those_systems]. Of their rows, 0.595 sit in a (publication, condition key) group measured for ≥2 metal states, or 0.658 without metal concentration in the key [fraction_rows_in_shared_condition_groups_weighted, _nm].
- **Cross-publication condition matches:** only 2 systems have any, or 3 without metal concentration [n_systems_with_any_cross_publication_condition_match].

**Condition axes** [condition_coverage.csv; V7 below]:
- Acid molarity varies within groups in 158 systems, extractant molarity in 70, and metal concentration in 19.
- O/A varies in 2, and lanthanide rows have a single O/A value.
- pH, saponification, loading and ionic strength are absent.

**Answer: partly.**
- **Metal effects: yes, within a publication.** 1,601 groups compare metal states at identical conditions. The spread between metals (0.658) is about twice the replicate spread (0.299). So metal effects can be separated from conditions in paired, within-publication designs, and the pairwise task (brief §8) has 23,551 comparable row pairs to work with [feasibility.json › pairwise_supply.total_row_pairs].
  - A comparable pair is two known metal states measured in the same publication, system and condition key.
  - The pairs span 216 systems and 87 publications; 21,264 of them are lanthanide–lanthanide [pairwise_supply].
  - Pairs must be generated after fold assignment (brief §12).
- **Source effects: no.** Only 2 condition-matched cross-publication groups exist. The publication offset S_s in brief §3 cannot be estimated from matched replicates, and it is confounded with system (Cramér's V 0.73).
- **Consequence.** Any source-hierarchy term is identified only through model structure, not through replicated measurements.

---

## V0–V7 support table

| Test | Verdict | Governing numbers (source) | PROPOSED concrete definition |
|---|---|---|---|
| **V0** random split | **supported (diagnostic only)** | 12,411 MODEL rows. Lag-1 log D correlation between consecutive rows is 0.726 within a publication and 0.101 across (leakage_summary.json › metadata_target.serial_lag1_r), so a random split will be optimistic | 5-fold random split of MODEL rows with a fixed seed, reported only as a debugging diagnostic and never compared against a strict fold |
| **V1** leave-publication-out | **supported** | See the grouping table below (feasibility_v1_groups.csv). Raw ids split the same source: 303 MODEL rows sit in ids that differ only by DOI spelling, and 3 primary DOIs are split across 2 ids (leakage_summary.json › doi_multiplicity). Two archive E_VALUE_CONFLICT groups (a La(III) pair and a Pu(IV) triple) spanned copy groups; the copy grouping now merges publications sharing a `duplicate_group_id` first, and the 104 registered folds all pass `fold_isolation_check(level="V1", near_dup_value_tol=0.005)` (tests/test_registered_folds.py). Before that merge 5 folds failed on `duplicate_group_shared` | Leave-one-group-out over `group_cross_publication_copy` (leakage_publication_components.csv). This grouping merges corrected DOIs, primary-source DOIs, archive duplicate groups and value-matched copies. The 103 groups with ≥20 MODEL rows are folds; the rest are pooled into one remainder fold. Sensitivity arm: `group_near_duplicate_key`. Folds must pass `fold_isolation_check` with a near-duplicate value tolerance. Selection / confirmation halves: 52 folds each (6,206 / 6,205 rows) (feasibility_halves.csv; feasibility.json › selection_confirmation_halves.V1) |
| **V2** leave-metal-out | **supported** | 23 states qualify at ≥100 rows and ≥5 systems (14 Ln(III), 9 An), including La, Ce, Pr, Nd, Sm, Eu and Gd. Sensitivity: 25 states at ≥50 rows / ≥3 systems, 19 at ≥200 rows. 26 elements mix X(?) and known states (feasibility.json › V2_leave_metal_out; leakage_summary.json › metal_alias) | For each qualifying state, remove **every row of its element** from training: all known states and X(?). Test on the target state's rows. Report the seven focus lanthanides first. Pr/Nd rows in `V6_TARGET_ROWS` are not scored before confirmation. Halves: 12 selection / 11 confirmation states, stratified focus-Ln / other Ln / An (feasibility.json › V2_leave_metal_out.v2_halves_rows100_systems5). Sensitivity: element hold-out crossed with its publication groups (row counts not computed) |
| **V3** leave-extractant-out | **limited** | 30 systems at ≥50 rows, ≥3 metal states and ≥2 same-family peers, all diglycolamide or monoamide (7 non-diglycolamide). Sensitivity: 51 systems in 6 families at ≥20 rows and ≥1 peer; 16 at ≥100 rows (Q4_V3_V4_experts.v3_grid). TODGA and its C10/C12 homologues are Tanimoto 1.0 (feasibility_named_analogues.csv) | Unit = extractant system at the 50/3/2 setting; a system with a name–structure-conflict component is never a unit. Hide the system's rows and the rows of every system sharing a component with it (`n_rows_other_systems_sharing_a_component`, feasibility_systems.csv; guard level V3). Identity is keyed on structure, never on name. Sensitivity: link systems through the stereo-free parent structure (`n_rows_other_systems_sharing_a_parent_structure`) |
| **V4** leave-family-out | **limited** | 4 families at ≥3 systems and ≥200 rows once name–structure-conflict systems are excluded, all neutral or soft-N; malonamide falls to 195 rows (Q4). Rows removed when hiding systems that contain a member (resolved component families): diglycolamide 7,933, monoamide 2,235, n_heterocyclic 799, pyridine_carboxamide 543 (feasibility_families.csv › n_rows_any_system_containing_a_member_family). The brief's phosphinic-acid example has 0 rows | Hide each of the 4 families in turn, including mixed-family systems that contain a member (guard level V4). Report as a boundary test inside neutral chemistry only; malonamide (356 rows touched) as a sensitivity. Acidic-family transfer is not testable |
| **V5** missing cell | **supported, neutral-solvating chemistry only** | 70–359 cells across the grid (Q1). At k=10, p=1, m=3: 224 cells, 37 systems, 26 states, 123 Ln cells, 18 Pr/Nd cells, 0 acidic cells. 36 cells need component-aware hiding beyond their own system (40 under the parent-structure link), 49 carry X(?) rows, 38 pool more than one acid; 218 cells remain eligible on HNO3 rows alone. 10 cells (32 rows) hold censoring candidates: 188 MODEL rows sit at an exact-decade floor (−4: 50, −3: 122, −2: 16) and 7 at a ceiling of their (study, system) group, 156 of them in the CORDIS project (feasibility.json › log_D_censoring_candidates). After the V6 carve-out 210 cells are scored, split 105 / 105 into selection and confirmation halves by system (feasibility.json › Q1_V5_cells.v6_carve_out, selection_halves_primary_scored) | Cell = (metal state, system), eligible at **k≥10 rows, p≥1 publication, m≥3** (224 cells). Hide, **in the cell's own system and in every system sharing a component**, the rows of the hidden metal state and the element's X(?) rows; other known states of the element stay in training (`hide_cell(component_aware=True)`; guard `fold_isolation_check(level="V5", component_aware=True)`, which raises on a hidden-state or X(?) row left in any of those systems). Score per acid medium as well as pooled. Sensitivities: k5/p1/m2 (359), k20/p2/m5 (70), HNO3-only cells (218), parent-structure hiding, cell-only hiding, censoring candidates excluded from scoring. The test asserts that the hidden pair is absent from training (brief §27) |
| **V6** Pr/Nd double hold-out | **limited** | 13 systems have ≥5 Pr(III) and ≥5 Nd(III) rows in common publications plus ≥2 other Ln(III): 5 non-diglycolamide, 209 comparable pairs (**142 in HNO3**), only **4** span ≥2 common publications. At ≥10 rows: 7 systems, 168 pairs (101 HNO3), 3 multi-publication. TODGA gives 83 of the 277 corpus-wide pairs: **HCl 54 (in 6 condition groups), HNO3 26 (26 groups), malonic 3**. `V6_TARGET_ROWS` (Pr/Nd rows of the 13 systems and of systems sharing a component with them) = 580 rows, 8 of them outside the 13 systems (V6_prnd_double_cell.grid, v6_target_rows__rows5_otherln2; feasibility_v6_systems.csv) | For each of the 13 systems, hide Pr(III) and Nd(III) together under the V5 state-level rule: the Pr(III), Nd(III), Pr(?) and Nd(?) rows of the system and of every component-sharing system (guard level V6). Pr and Nd have no other recorded state, so these are all Pr and Nd rows there (feasibility_metal_states.csv › other_known_states_same_element). Score D_Pr, D_Nd, the sign and magnitude of logSF(Nd/Pr) on comparable pairs, and interval coverage, with a system-level cluster bootstrap; report per acid medium, and the HNO3 pairs as the nitrate-process evidence. Report TODGA separately. `V6_TARGET_ROWS` are never scored before the confirmation run |
| **V7** condition extrapolation | **limited** | Unit = system × acid, region = top or bottom 20% of the unit's own log10 range, ≥10 rows inside and outside: acid M 65 units (55 systems), extractant M 33 (31), metal M 28 (27). At ≥20 rows: 29 / 23 / 15 units. 302 acid values are log-grid flagged. Loading and pH are absent (feasibility_v7_grid.csv; feasibility.json › V7_condition_regions) | Contiguous top-20% and bottom-20% regions of acid M and extractant M within system × acid, at the ≥10/≥10 setting. Run with and without the 302 log-grid-flagged rows. "High loading" is not a supported region, because only a derived upper bound exists, on 0.376 of MODEL rows |

**V1 grouping options** [feasibility_v1_groups.csv]:

| Grouping | Groups | Groups with ≥20 rows (rows in them) | Largest group (share) | Top-5 share |
|---|---|---|---|---|
| `g19_publication_id` (raw) | 157 | 116 (12,073) | 1,096 (0.088) | 0.276 |
| `group_primary_source_doi` | 153 | 114 (12,075) | 1,096 (0.088) | 0.276 |
| **`group_cross_publication_copy`** (includes archive duplicate-group links) | 139 | 103 (12,102) | 1,096 (0.088) | 0.304 |
| `group_near_duplicate_key` | 118 | 86 (12,144) | 2,190 (0.176) | 0.391 |
| `group_compilation_doi` | 100 | 70 (12,171) | 2,190 (0.176) | 0.510 |

---

## §30 Actinide question: feasibility

**Question.** Does including actinide rows improve hidden-lanthanide prediction?

**The ablation is feasible:**
- The same V5 lanthanide cells (123 at the proposed setting) and V2 lanthanide folds can be scored with and without the 5,420 actinide MODEL rows in training [counts.json › rows.by_metal_category_model.actinide].
- The test set and fold groups do not change between arms.

**The evidence it can use is narrow:**
- 1,207 condition-matched An/Ln row pairs, of which 802 are Am(III)/Eu(III) and only 73 involve Pr or Nd (Q12).
- Actinide rows other than Am(III) are HNO3-only.
- The monoamide family is actinide-only in practice (2,144 of 2,154 rows). "With actinides" therefore also means "with a family the lanthanide targets barely have".

**Consequences for the design:**
- **Prior result.** Gen11 ran this question on a smaller setting and returned INCONCLUSIVE (`runs/gen11_transfer/GEN11_DECISION_REPORT.md`).
- **Statistical method.** A paired, cluster-level (cell or system) bootstrap and a pre-registered minimum effect are required. A single MAE difference will not settle it.
- **Negative transfer.** It must be checked separately for the Am/Eu-heavy diglycolamide cells and the monoamide-only systems.
- **Power.** The minimum detectable effect is not computed.

## §31 Architecture question: feasibility

**Question.** Does factorising metals and extractants beat flat categorical modelling under V5?

**Feasible.** Every eligible V5 cell keeps ≥m other systems for its metal and ≥m other states for its system. That is the precondition for any of the four arms to predict the hidden cell:
- flat categorical;
- descriptor model;
- factorised interaction;
- factorised + mechanism.

**Descriptor coverage on MODEL rows** [data_audit/metal_descriptor_coverage.csv, section coverage, metal_category ALL]:

| Metal descriptor | Coverage |
|---|---|
| oxidation state | 0.890 |
| CN6 radius | 0.885 |
| HSAB class | 0.882 |
| CN8 radius | 0.831 |
| Pauling electronegativity | 0.805 |
| CN9 radius | 0.542 |
| polarizability | 0.000 |

- **Ligand descriptors** exist for 283 structures [manifests/g19_build_extractants.json › summary.n_structures]. The Morgan r2 fingerprint is degenerate for chain homologues (Q1 caveat). A descriptor arm needs a chain-length-aware descriptor, for example the `mw`, `mol_logp` and `rotatable_bonds` columns in `descriptors/extractant_components.csv`.
- **The "+ mechanism" arm cannot use the brief's five experts.** Every eligible cell is E2 (Q1 composition). The only mechanism contrast available inside V5 is NEUTRAL_SOLVATING (178 cells) vs SOFT_N_DONOR (29) vs MIXED_NEUTRAL (17) at the proposed setting †. That is a Gen19 sub-grouping (INFERRED), and it should be registered as such.
- **Pb(II) electronegativity** is Allred's +2-state value 1.87; the element value 2.33 is the +4 state. The Fe, Cr and Mo element values carry an unverified oxidation state [descriptors/metals.csv › electronegativity_pauling_source].

## §32 Priors question: feasibility

**Question.** Do mechanism and physics priors improve out-of-distribution transfer, under V1, V2, V5 and V6?

| Prior | Testable? | Governing numbers |
|---|---|---|
| Mechanism routing | **no** (across brief experts) | Only E2 qualifies at every expert threshold; E3 has 0 rows, E1 14 and E4 3 (Q4). A within-E2 routing (O-donor vs soft-N) is the only testable variant |
| Monotonic penalties | **limited** | Needs contiguous condition ranges: acid M 65 and extractant M 33 system × acid units at the V7 setting. Must not force monotonicity where the chemistry can be non-monotonic (brief §33). The 302 log-grid acid rows must be handled first |
| Source hierarchy | **limited** | V1 folds are feasible (103 groups with ≥20 rows), but the matched publication effect rests on 2 condition groups (9 without metal concentration), and system–publication Cramér's V is 0.73 (Q13). Expect the hierarchy to be weakly identified; report the shrinkage |
| Series smoothness | **yes** (lanthanides) | 66 systems carry a contiguous Ce–Pr–Nd–Sm run in one publication; 111 of 123 lanthanide V5 cells have a Z±1 neighbour (Q11). V6 coverage of the prior is limited to 13 systems |

---

## §19 Applied Pr/Nd case

[feasibility_candidate_extractants.csv; feasibility.json › section19_candidates; feasibility_todga_by_metal_state.csv]

| Brief §19 field | PC88A / P507 | Cyanex 272 | D2EHPA / P204 | TODGA |
|---|---|---|---|---|
| Direct data count (MODEL rows) | 0 | 0 | 0 | 3,424 |
| Direct publications | 0 | 0 | 0 | 57 |
| Same-family data count (rows / systems) | 0 / 0 | 0 / 0 | 10 / 1 (dihexyl hydrogen phosphate) | 7,928 / 99 |
| Pr direct support (rows) | 0 | 0 | 0 | 97 (Pr(III) 76 + Pr(?) 21, any system containing TODGA) |
| Nd direct support (rows) | 0 | 0 | 0 | 171 (Nd(III) 111 + Nd(?) 60) |
| Neighbouring Ln(III) states | 0 | 0 | 0 | 12 |
| Actinide support (rows) | 0 | 0 | 0 | 1,118 |
| Condition coverage | – | – | – | 8 acids; acid 2.63e-7 to 7.0 M (lowest value log-grid flagged) |
| Support class | UNSUPPORTED_NO_DIRECT_NO_FAMILY | UNSUPPORTED_NO_DIRECT_NO_FAMILY | SAME_FAMILY_ONLY | DIRECT_PR_AND_ND |
| Domain status / uncertainty | not computed (Phase G) | not computed | not computed | not computed |
| Best robust process result | not computed (Phase H) | not computed | not computed | not computed |

**Testability.**
- **TODGA is testable.** Nd(III) × TODGA has 93 rows in 14 publications, and Pr(III) × TODGA has 66 rows in 10. Both cells are eligible at all 36 grid settings [feasibility_v5_cells.csv › n_grid_settings_eligible].
- **D2EHPA is not testable.** It has no structure. Its single phosphoric-acid relative has 1 Pr row and 1 Nd row in one publication. The archive's "DEHPA" is an amide. HDEHP appears only as a co-extractant name in the modifier slot (14 MODEL rows, Eu/Am) and in the titles of 3 papers whose rows record other structures (Q7).
- **PC88A and Cyanex 272 are not testable.** Neither they nor their families have any rows.

**Named test cells that replace PC88A × Nd for the transfer machinery.** They carry no information about PC88A-type cation-exchange Pr/Nd processing (acid dependence of opposite sign, pH and saponification control; see the verdict):
- **V5.** The central named cell is **Nd(III) × TODGA** (93 rows: HNO3 62, HCl 12, malonic 8, lactic 4, tartaric 4, citric 3). Hiding it must also remove:
  - the 60 Nd(?) rows under TODGA [feasibility_v5_cells.csv, Nd(III) × TODGA row › alias_unknown_state_rows_hidden_with_cell];
  - the 18 Nd(III) rows in 3 other systems that contain TODGA; no Nd(?) row sits in those systems [same row › hidden_rows_in_other_systems_sharing_a_component, unknown_state_rows_in_other_systems_sharing_a_component, n_other_systems_sharing_a_component_with_metal].

  After this hiding, the nearest remaining ligand measured with Nd(III) is DODDdDGA at Tanimoto 1.0, and 196 same-family Nd rows remain [feasibility_v5_cells.csv, Nd(III) × TODGA row › hidden_nearest_ligand_system_label, hidden_n_same_family_other_system_rows_for_metal].
- **V5, the other Pr/Nd cells.** At k=10, p=1, m=3 there are 18 Pr/Nd cells, all listed in `feasibility_v5_cells.csv`. Besides the two TODGA cells, their systems are:
  - TDdDGA, DMDODGA, TBDGA, C5BTBP, TEHDGA, DHD2DGA, DBD1MHDGA, DOODA (C12), DMDOHEMA|TODGA;
  - N,N'-dimethyl-N,N'-diphenylpyridine-2,6-dicarboxamide.

  C5BTBP and the dicarboxamide are the soft-N representatives.
- **V6.** The primary double-cell target is **TODGA Pr(III)+Nd(III)**: 66 and 63 rows inside 10 common publications, 83 comparable pairs.
  - **By medium the 83 pairs are HCl 54 (6 condition groups), HNO3 26 (26 groups) and malonic acid 3 (3 groups)** [feasibility_v6_systems.csv › comparable_prnd_row_pairs_by_acid, comparable_prnd_condition_groups_by_acid]. For a nitrate process case the V6 evidence is 26 pairs, not 83.
  - The other systems spanning ≥2 common publications are DMDODGA (3 publications, 25 pairs, 15 of them HNO3), C5BTBP (2, 16, all HNO3) and DMDPDGA (2, 5, all HNO3).
  - Each of the remaining 9 of the 13 V6 systems has a single publication that measured both Pr and Nd [feasibility_v6_systems.csv › n_common_publications]; all their pairs are HNO3.

**What the process layer can honestly consume.**
- **Available now: TODGA-type Pr/Nd predictions** for diglycolamide systems in nitrate media — the HNO3 rows and pairs only; the HCl and organic-acid TODGA rows are a different response surface.
  - Label them IN_DOMAIN or INTERPOLATION where the cell is measured.
  - Label them CROSS_LIGAND_TRANSFER for another diglycolamide, and only after V5/V6 have scored that transfer.
  - Gen18 already runs a TODGA Pr/Nd case (`generations/gen18_process/cases/todga_prnd_feed.json`).
- **Condition range.** Predictions are confined to measured acid molarities.
  - Loading can only be bounded: the derived upper bound covers 0.475 of lanthanide MODEL rows [metadata_availability.csv › loading, fill_model_lanthanide].
  - O/A in lanthanide rows has a single value.
- **Not available: PC88A, Cyanex 272 and D2EHPA** (UNSUPPORTED). Gen18's own summary (`generations/gen18_process/SUMMARY.md`) states that it runs this case on literature placeholders. The expanded archive does not change that: it holds no rows for these extractants or their families.
  - Under brief §18, an UNSUPPORTED prediction must not become a winning recipe.
  - Acidic-extractant cascades are controlled by pH and saponification, and the corpus records neither.
- **Not available: HCl-medium Pr/Nd recipes from data.** HCl Pr and Nd rows exist (137 Pr rows / 7 publications; 67 Nd rows / 9 publications). Which systems they belong to, and whether they form a V5 cell, is not tallied in a summary file.

---

## Open items that Phase C must settle

1. **V1 grouping.** Use a merged group column, not `g19_publication_id` (DATA_AUDIT §13a). A fix to `load.py` belongs to the orchestrator.
2. **Sr(III).** Decide policy for its 38 implausible MODEL rows. They do not reach any V2 threshold at ≥100 rows.
3. **Acid values.** Decide how the 302 log-grid acid rows enter V7 and the monotonic-prior arms.
4. **DMDOHEMA.** Resolve the name–structure assignment (DATA_AUDIT §11a) before V3 folds are keyed.
5. **Ligand descriptor.** Choose a descriptor that separates chain homologues before any support score uses ligand distance.
6. **Name–structure conflicts** (Br-Cosan, TPDGA malonamide, TDGA, TBADIPIC, NDDIPIC): they stay in training but are excluded from V3/V4 units. Resolving their structures is a curation item.
7. **Stereo-free vs stereo-specified identity** (Me2-TODGA 64 rows vs cMe2-/tMe2-TODGA 16 + 16; p-TODGA, p-TDDGA and Et-Me-TDDGA Syn/Anti): open curation item; the parent-structure hiding is registered as a sensitivity.
8. **Censoring candidates** (188 floor and 7 ceiling rows): kept as recorded, with a registered scoring sensitivity that excludes them.
