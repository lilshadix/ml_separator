# Gen19 — pre-registration of the chemistry-transfer evaluation

> Status: sealed by scripts/g19_seal_prereg.py on 2026-09-15, after DATA_AUDIT.md, FEASIBILITY.md and the
> pre-seal Phase C run of B0–B4 (with B3x, B3i, B3l, B4x, B4l), B7, the pair yardsticks FLAT and HEAVIER and
> split-conformal intervals on them; no outer-fold score of B5, B6, B6r0, B8, FLAT_CAT or any M model existed at
> sealing, and nothing had been run on V6.

*Drafted 2026-09-15 for brief §21, after the Phase A audit and the Phase B feasibility outputs and
before any predictive model was fitted. Every corpus number below is copied from a file written by
a Gen19 script. The bracket after the number names the file and key; paths are relative to
`generations/gen19_chem_transfer/`, and a bare file name means `data_audit/<file>`. A number marked
**†** is a row tally of the cited table under the stated filter, not a stored summary field (the
`DATA_AUDIT.md` convention). Anything not written here is exploratory and is labelled so in every
table.*

## 0. Sealing, order of work, and what is committed to

| | |
|---|---|
| brief | `GEN19_CLAUDE_CODE_INSTRUCTIONS.md` (commit 9d155dc); §11–§14, §21–§22 and §30–§32 are implemented here |
| frozen inputs | archive `dataset_all_metals/clean/master_clean.parquet`, SHA-256 `7b32979383c5246f22d36ade6246231a347ea22490f4589a76e189af07b09b01` (`gen19ct/paths.py`, `dataset_hashes.csv`); bundle `dataset with 3D structures/dataset.parquet` (prefix `fefbefc6`), read only with `columns=[...]`; gen18 read-only |
| population | MODEL tier (`gen19ct.data.load.load_model_rows`): 12,411 rows, of which 11,049 have a known oxidation state and 1,362 do not; 43 known metal states, 259 extractant systems, 157 publications (`feasibility.json` → `population`) |
| data graph | metal-state × system bipartite graph of MODEL rows is **CONNECTED**: one component. The ≥ 5-row backbone is also one component and holds 0.881 of known-state rows; 127 systems and Cd(II), Pm(III) have no ≥ 5-row edge (`feasibility.json` → `graph_connectivity`) |
| code relied on | `gen19ct/paths.py`, `gen19ct/data/load.py`, `gen19ct/manifest.py`, `gen19ct/data/leakage.py` (`fold_isolation_check`, `pair_isolation_check`), `gen19ct/chemistry/support_graph.py` (`SupportIndex`, `hide_cell`, `hide_cells`), `gen19ct/folds/registered.py` (`v6_target_mask`, `assert_not_scored`, `assign_halves`); the registered splits are exercised by `tests/test_registered_folds.py` |
| regime of every number | hold-out design, threshold setting, averaging unit, cluster unit, seed set (discovery or confirmation), and parameter status (registered / exploratory) — a table without these fields is not quoted |

**Order of work.**

1. **Before sealing**, Phase C runs only the lookup and closed-form baselines B0–B4 (with B3x, B3i,
   B3l, B4x, B4l) and B7 (§5), the pair yardsticks FLAT and HEAVIER (§4), and split-conformal
   intervals on those arms (§12), on V1, V2 and V5, including every V5 variant in §3.1, V5-P and
   V5-PAIR. It may also run them on V3, V4 and V7, and on V0 as a diagnostic (§3.6). **B5, B6, B8 and every M model are not run on any outer fold before sealing**, so
   no learned model's outer score can inform a threshold. Phase C never runs anything on V6 (§3.4)
   and never scores a `V6_TARGET_ROWS` row (§2). It writes the descriptive difficulty numbers of §9
   (the success margins are already fixed numbers) and the fold and batch counts that §3 leaves open.
   *Disclosure (2026-09-15):* the first draft of this item named B0–B4 with B3x, B3i, B3l, B4x and B7
   on V1, V2 and V5 only. The pre-seal run (`scripts/g19_run_preseal.py`) also ran B4l, FLAT, HEAVIER,
   split-conformal intervals and V0 (`manifests/g19_run_preseal.json` → `arms`, `pair_yardsticks`;
   `evaluation/preseal/predictions/V0__rows.parquet`); none of them is a learned model
   (`manifests/g19_run_preseal.json` → `learned_arms_fitted_or_scored` = `[]`). The item is amended to
   name what ran; nothing else changed.
2. Before any B5, B6, B8 or M fit, every bracketed placeholder in this file is replaced from those outputs.
   `--commit-seeds` draws the confirmation seeds and their digest is quoted in §15.
   `g19_seal_prereg.py --seal` then refuses unless:
   - `DATA_AUDIT.md` and `FEASIBILITY.md` exist;
   - no placeholder marker and no draft banner remain;
   - no checklist box of §20 is left unticked (added 2026-09-15 after verification finding M1: before,
     the gate passed with unticked boxes once the two marked placeholders were filled);
   - every section brief §21 requires has a heading;
   - the seed digest is quoted in the text.
3. Discovery runs the baselines B5, B6, B8 and the M ladder on the discovery seeds, scoring the
   **selection half** of the V5 systems, V1 folds and V2 states (§3, `data_audit/feasibility_halves.csv`).
   At most five claims are frozen from selection-half results. They are then confirmed once on the
   withheld seeds and the **confirmation half**, together with the single V6 run (§15).

**After sealing.** The text above the footer is never edited. A change to the registered analysis
is a section appended **below** the footer, headed
`## POST-HOC addendum <n> (<YYYY-MM-DD>, <author>; results seen: <yes|no>)`. It states what
changed, why, and whether any outcome had been seen. `--check` accepts only such addenda below the
footer and fails if anything above it changes. Fit scripts call `--check` and refuse to run when it
fails. The footer digest is computed on LF-normalised text, so CRLF and LF checkouts agree (brief §24).

## 1. Questions and the primary hypothesis

**Primary hypothesis H1 (brief §2, §7, §11 V5).** A factorised metal × extractant model reconstructs
log D in a metal × extractant cell whose every measurement is hidden. It must do so better than the
cheapest sensible alternative: the gen18-style nearest-condition lookup transferred across metals
inside the same extractant system, either the nearest-radius metal (B3x) or a radius interpolation
between the bracketing same-charge metals (B3i), whichever is stronger (§5). It must also beat the
constant baseline. The factorised model shares a learned metal representation across extractants,
an extractant representation across metals, and an explicit interaction `I(m, l) = e_mᵀ W e_l`.
Primary candidate: **M2** (§6). Primary contrast: **M2 vs the lookup comparator on V5-primary**,
macro MAE of log D over hidden cells (discovery on the selection half, confirmation on the
confirmation half).

**Secondary questions (registered).**

* **H1b.** The classical factorisation B6 vs the lookup comparator on V5. If B6 passes and M2 does not, the
  report says "factorised structure helps; the learned encoder does not".
* **H2 (V6, applied).** With the Pr and Nd cells of a system hidden, does the model recover
  D_Pr, D_Nd, the sign and magnitude of logSF_Nd/Pr, and calibrated intervals?
* **H3 (§30).** Does training with actinide rows improve prediction of hidden lanthanide cells?
* **H4 (§31).** Under V5, does factorising outperform flat categorical modelling and descriptor
  modelling? Does adding mechanism routing help further?
* **H5 (§32).** Do mechanism routing, the extractant-concentration monotonicity penalty, the
  source hierarchy and lanthanide-series smoothness each improve out-of-distribution transfer?
* **H6 (§10, §34.8).** Are the intervals calibrated at 50/80/95 %, overall and per domain-status
  category? Does predicted uncertainty rank the errors?
* **H7 (§17–§18, conditional).** Is the robust Gen18 process recommendation for Pr/Nd stable under
  propagated D uncertainty? This is evaluated only if S1 (§9) passes.

## 2. Data rules for every design

* **Rows.**
  - Only MODEL rows are fitted or scored.
  - The target is `log_D` as recorded; no transformation, no imputation, no synthetic target.
  - The one synthetic quantity is the injected signal of the §8 power check, which is never a
    result.
  - **Scored rows (every design, V0 and V1 included).** A hidden row is scored only if it has a known
    metal state, is not Sr(III), is not in `V6_TARGET_ROWS`, and is not an acidic co-extractant row;
    X(?) rows are hidden with their unit and never scored. Every calibration set (§12) and inner
    validation set uses the same population. This is the reading the fold builder and the pre-seal
    run applied before any score (`folds/INDEX.json` → `readings.scored_rows`; 10,585 MODEL rows meet the
    first three conditions, `population.scorable_rows`, before the acidic co-extractant rows are removed at
    scoring; `evaluation/preseal/difficulty.json` → `readings.scoring_exclusions`,
    `readings.calibration_population`). Under it, 21 of the 104 V1 folds hold no scored row
    (`evaluation/preseal/guard_log.csv`, job `V1__copy`, status `no_scored_rows_not_fitted`).
    **Resolved (orchestrator, 2026-09-15; pre-seal baseline scores seen, no learned-model score existed):**
    X(?) rows stay unscored in every design, V0 and V1 included, and stay out of every calibration and
    inner validation set. Reason: one population across designs, and no state is ever imputed (§2
    "Metal identity"); not score-driven.
* **Censoring candidates.**
  - A row is flagged when its log D is an exact decade that is the minimum (≤ −2) or the maximum
    (≥ 2) of its (`g19_study_id`, `extractant_system_key`) group, and ≥ 2 rows share that value.
  - That gives 188 floor rows (log D −4: 50, −3: 122, −2: 16) and 7 ceiling rows in 9 publications,
    156 of them in the CORDIS project (`feasibility.json` → `log_D_censoring_candidates`). They look
    like detection limits (INFERRED).
  - They are kept as recorded and are scored. Registered sensitivity: every design is re-scored with
    the flagged rows excluded from scoring, still in training (§8, R19 item 6). Floors favour
    lookups and FLAT, so a claim must survive this sensitivity.
* **`V6_TARGET_ROWS` (the applied test is untouched until confirmation).**
  - Defined once: the Pr and Nd rows (known state and X(?)) of the 13 V6 systems (§3.4) and of every
    system sharing a component with one of them. That is 580 MODEL rows: 572 in the V6 systems and
    8 elsewhere (`feasibility.json` → `V6_prnd_double_cell.v6_target_rows__rows5_otherln2`;
    `gen19ct.folds.registered.v6_target_mask`).
  - They stay in training everywhere.
  - They are **excluded from every scoring set before the confirmation run**: V0, V1, V2, V3, V4, V5
    and every variant, V7, the H3 test sets, every inner validation set (inner V5 cells or cell pairs,
    inner V2 metals, inner V1/V3/V4/V7 folds), calibration sets, and the Pr/Nd residual-correlation
    estimate of §14.
  - Every scoring index passes `registered.assert_not_scored` before a metric is written.
* **Chemistry flags from the extractant builder** (`descriptors/extractant_systems.csv`).
  - The 14 MODEL rows with an acidic co-extractant in the modifier slot (HDEHP with TODGA or DOHyA)
    are synergistic chemistry. They stay in training, are hidden with their cell, and are never
    scored as rows of a neutral cell. None lies in an eligible V5 cell.
  - Systems whose only recorded "extractant" is an aqueous agent (HEDTA, CDTA, TDGA; mechanism
    UNKNOWN, 29 rows) and systems with a name–structure-conflict component (5 structures) stay in
    training. They are never V3/V4 units.
* **Metal identity.** `g19_metal_state`, e.g. "Nd(III)".
  - An unknown-state row (`X(?)`, 1,362 MODEL rows) is never imputed to a state.
  - Whenever a design hides a metal state under a system (or everywhere), it also hides that
    element's unknown-state rows under the same scope (`support_graph.hide_cell`; the
    `fold_isolation_check` alias rule).
  - Those rows are hidden but not scored.
* **Sr(III).** The 38 MODEL rows carry an oxidation state the archive itself marks implausible
  (`metal_coverage.csv`). They stay in training as recorded and are excluded from every scoring
  set. Sensitivity: they are dropped from training.
* **Publication groups.**
  - Every "publication" unit used for folds, clusters, inner folds and B4 is
    `group_cross_publication_copy` of `leakage_publication_components.csv`, as `FEASIBILITY.md`
    proposes for V1.
  - This is the merge of `g19_publication_id` groups linked by corrected DOIs, primary-source DOIs,
    archive duplicate groups (any class, MODEL rows) and value-matched cross-publication copies.
  - It is needed because `g19_publication_id` is built from uncorrected DOIs: 303 MODEL rows sit in
    ids that differ only by DOI spelling (`leakage_summary.json` → `doi_multiplicity`).
  - The groupings are nested (g19 id ⊂ corrected DOI ⊂ primary-source DOI ⊂ archive duplicate group
    ⊂ copy ⊂ near-duplicate key ⊂ compilation DOI; the ladder is cumulative in
    `leakage.publication_link_components`).
  - The copy grouping has 139 groups with MODEL rows: 103 of them have ≥ 20 rows (12,102 rows), the
    largest holds 1,096 rows (0.088), and the top-5 share is 0.304 (`feasibility_v1_groups.csv`).
    Every archive duplicate group lies inside one copy group (`tests/test_registered_folds.py`).
  - Sensitivities: `group_near_duplicate_key` (118 groups, largest 2,190 rows, share 0.176) and
    `group_compilation_doi` (100 groups, top-5 share 0.510).
  - **Copies the key cannot see.** The near-duplicate key contains `g19_ox` and
    `extractant_system_key`, so a value-matched record that differs only in the state token (Zr(IV) vs
    Zr(?)) or in the structure key passes every guard above and links no publications. The fold
    builder reports them (`folds/INDEX.json` → `leakage_sensitivity_wildcard_copies`,
    `readings.wildcard_copies`; `folds/wildcard_copy_pairs.csv`, `folds/wildcard_copy_crossings.csv`),
    and the pre-seal run prints two exploratory scoring filters for V5-primary and V1
    (`wildcard_copies_excluded_scoring`, `wildcard_copies_strict_excluded_scoring`). **Resolved
    (orchestrator, 2026-09-15; the exploratory filters' baseline values were seen, all moves < 0.01 log
    D):** `wildcard_copies_excluded_scoring` (any partner) becomes a **registered R19 item 6
    sensitivity of V1 and V5** (rows excluded from scoring, kept in training); the strict filter stays
    exploratory. The publication groups, V1 folds and halves are not rebuilt (they were fixed before any
    score, and a rebuild would re-open the halves). Reason: the conservative option that leaves every
    registered unit untouched.
* **Comparable pairs (brief §8).**
  - Two rows form a pair only if they have the same publication group, the same
    `extractant_system_key` and an identical `gen19ct.data.normalize.condition_key`, which includes
    metal concentration, and different metal states.
  - Sensitivity: the key without metal concentration. On An/Ln pairs this raises the count from
    1,207 to 1,940 (`feasibility.json` → `Q12_actinide_lanthanide_overlap`).
  - Pairs are generated **after** fold assignment, separately inside the training rows and inside
    the test rows. `pair_isolation_check` must pass before any pair enters a loss or a metric.
* **Features.**
  - No column of `gen19ct.data.load.PROVENANCE_COLUMNS` is ever a feature.
  - Neither are publication or study ids, DOI, data location, row ids or free-text comments.
  - Every fitted preprocessing step is fitted on the training rows of the fold it serves: scaling,
    imputation, fingerprint PCA, embeddings, calibration and source priors (brief §12).
* **Acid-molarity semantics.**
  - 302 MODEL rows carry acid molarities on a 0.01 log grid consistent with a back-converted pH
    (INFERRED; `feasibility.json` → `V7_condition_regions.acid_M_log10_grid_flag_rows`).
  - They are kept.
  - Every registered contrast is re-run without them as a sensitivity. A sign change is reported as
    a failure of robustness.
* **Mechanism labels.**
  - The system-level majority vote of `descriptors/extractant_systems.csv`.
  - The brief's five experts map to the builder's labels as in `feasibility_mechanisms.csv`. That
    mapping is a Gen19 grouping and is INFERRED.
* **Leakage guard.**
  - Every outer and inner split of V1, V2, V3, V4, V5 and V6 passes `fold_isolation_check` at its
    design level before a model is fitted: `level="V1"`; `"V2", element_level=True`; `"V3"`;
    `"V4"` with the hidden families and the component-family map; `"V5", component_aware=True`;
    `"V6"` (component-aware by construction).
  - *As run pre-seal (disclosure):* every outer split and every V1, V2 and V0 inner calibration split
    was checked directly. The V5 inner calibration splits used the conformal wrapper's
    `nested_certificate` mode: one check per inner cell on a superset split (MODEL rows minus the cell's
    registered hiding, test = every row of the cell), with the inner split asserted to lie inside it on
    both sides. It is sound only because every violation class of `fold_isolation_check` needs a training
    row and a test row, so removing rows from either side cannot create one (INFERRED from
    `gen19ct/data/leakage.py`); it was cross-checked against checking every split on 6 outer folds
    (`evaluation/preseal/inner_guard_verification.json`, `folds[*].residuals_and_quantiles_identical`).
    **Resolved (orchestrator, 2026-09-15):** `nested_certificate` counts as "passes
    `fold_isolation_check`" for inner splits of every design, on the monotonicity argument above
    (removing rows from either side of a split that passes cannot create a violation). Safeguard: at the
    start of discovery, `every_split` is also run on 20 outer folds per design drawn with
    `numpy.random.default_rng(19)` and stratified by metal category (Ln / An / other) where the design
    has cells; any difference in residuals, quantiles or guard verdicts makes `every_split` mandatory
    for that design. Not score-driven.
  - The settings are `near_dup_sig=6` and `near_dup_value_tol=0.005`, the leakage module's
    `COPY_TOLERANCE_LOG_D`, i.e. brief §12's "identical conditions and values".
  - V7 has no guard level. Its fold builder runs the always-on checks (index, duplicate group,
    near-duplicate) and separately asserts that no training row of the unit lies inside the hidden
    region.
  - A violation aborts the run. The registered V1 (104 folds), V2 (23), V5 primary (224 cells) and
    V6 (13) splits pass on the corpus (`tests/test_registered_folds.py`).
  - The value-blind near-duplicate check (`near_dup_value_tol=None`) is run and reported as a
    sensitivity, together with the `group_near_duplicate_key` folds.
  - Names are never identity: systems are keyed on structure. The unresolved DMDOHEMA name–structure
    assignment (`DATA_AUDIT.md` §11a) therefore affects labels, not folds.
  - "Shares a component" means an identical canonical SMILES. Registered sensitivity for V3, V5 and
    V6: the stereo-free, salt-free parent structure (`registered.parent_component_map` passed as
    `component_map`), which also links p-TODGA-Syn/-Anti, the stereo-free and cis/trans Me2-TODGA,
    p-TDDGA and Et-Me-TDDGA isomers.
  - A test asserts that each hidden metal × extractant cell has zero training rows (brief §27).

## 3. Hold-out designs

### 3.1 V5 — missing metal × extractant cell (PRIMARY)

**Unit.**
- A cell is `(g19_metal_state, extractant_system_key)` over MODEL rows.
- A cell is **eligible** when all of these hold:
  - it has ≥ k rows and ≥ p publications;
  - after the cell is hidden, its metal keeps ≥ m other systems and its system keeps ≥ m other
    metal states;
  - the cell edge is not a bridge of the all-edge bipartite graph
    (`feasibility.json` → `Q1_V5_cells.definition`).
- **Hiding** (the `FEASIBILITY.md` definition; `support_graph.hide_cell(component_aware=True)`) applies
  one **state-level** rule in the cell's own system **and in every other system that shares a
  component structure** with it. In each of these systems it removes:
  - the rows of the hidden metal state (in the own system, the cell's rows);
  - the same element's unknown-state X(?) rows.
- Other known states of the element stay in training everywhere (e.g. Pu(IV) rows under DHOA|TODGA
  when Pu(VI) × TODGA is hidden), because they are a different chemical species, not a measurement
  of the hidden cell, and element-level transfer is what V2 tests.
- Counts at the primary setting (`feasibility_v5_grid.csv`, `pub_basis = g19_publication_id`):
  - 49 primary cells carry X(?) rows under their own system (`n_cells_with_alias_rows`);
  - 36 cells have rows of the hidden state or X(?) rows of its element in a component-sharing system
    (`n_cells_hidden_rows_in_system_sharing_component`); in 17 of them there are X(?) rows
    (`n_cells_unknown_state_in_system_sharing_component`);
  - in 7 cells, rows of another known state of the element stay in a component-sharing system
    (`n_cells_other_known_state_kept_in_system_sharing_component`).
  - For example, when Nd(III) × TODGA is hidden, the 18 Nd(III) rows of the three systems sharing the
    TODGA structure (DEHAA|TODGA, DHOA|TODGA, DMDOHEMA|TODGA; `descriptors/extractant_systems.csv`,
    `component_canonical_names`) are hidden with it; none of those systems holds an Nd(?) row
    (`feasibility_v5_cells.csv`, row Nd(III) × TODGA: `n_other_systems_sharing_a_component_with_metal` = 3,
    `hidden_rows_in_other_systems_sharing_a_component` = 18,
    `unknown_state_rows_in_other_systems_sharing_a_component` = 0).
- `fold_isolation_check(level="V5", component_aware=True)` enforces exactly this rule: a train row of
  the hidden state or an X(?) row of its element in a component-sharing system is a violation
  (`V5_hidden_state_in_component_sharing_system`); other known states there are only counted as a
  warning.
- This reads brief V5's "remove every direct measurement of that combination" conservatively.
  After this hiding every primary cell's metal still has ≥ 3 other systems
  (`feasibility_v5_cells.csv`, `other_systems_for_metal_after_registered_hiding` †).
- **Acid media.** 38 primary cells pool rows from more than one acid
  (`feasibility_v5_grid.csv`, medium `all`, `n_cells_multi_acid`). Cells are hidden whole, whatever the medium.
  Scores are also reported per acid medium, and the HNO3-only cells are a registered sensitivity
  (brief §4.3).

**Registered threshold settings** (`feasibility_v5_grid.csv`, `pub_basis = g19_publication_id`;
the primary-source-DOI basis gives identical counts):

| setting | k, p, m | cells | rows | systems | metal states | families | Ln cells | An cells | Pr/Nd cells | non-DGA cells |
|---|---|---|---|---|---|---|---|---|---|---|
| **primary** | 10, 1, 3 | **224** | 7,174 | 37 | 26 | 9 | 123 | 97 | 18 | 100 |
| loose | 5, 1, 2 | 359 | 8,184 | 55 | 30 | 10 | 185 | 166 | 30 | 157 |
| strict | 20, 2, 5 | 70 | 4,534 | 17 | 22 | 5 | 34 | 36 | 5 | 21 |
| V5-P base | 10, 2, 3 | 132 | 5,681 | 27 | 26 | 8 | 65 | 64 | 9 | 55 |
| HNO3-only (sensitivity) | 10, 1, 3 on HNO3 rows | 218 | 6,480 | 36 | 26 | 9 | 117 | 97 | 17 | 100 |

**Why the primary thresholds.**
- **k = 10.** A cell mean then rests on ten rows against a replicate SD of 0.299
  (`feasibility_q13_variance.csv`).
- **m = 3.** The metal and the system each keep at least three other partners, so both sides of a
  rank-1 interaction rest on at least three measured partners. This is a design choice, not a derived
  minimum.
- **p = 1.** With p = 2 the cell count falls to 132 and Pr/Nd cells to 9.
- The loose and strict settings exist so that a claim must hold its sign across them (§8, R19 item 6).

**V6 carve-out.**
- Every cell whose rows are in `V6_TARGET_ROWS` (§2) is **excluded from V5 scoring before
  confirmation**. That covers the Pr/Nd cells of the 13 V6 systems and of the systems sharing a
  component with them. Those rows stay in training for every other cell.
- At the primary setting 14 of the 18 Pr/Nd cells are carved out; the component-sharing extension
  adds no primary cell. 210 primary cells in 37 systems remain scored, 109 of them lanthanide
  (`feasibility.json` → `Q1_V5_cells.v6_carve_out`; `feasibility_v5_cells.csv`,
  `scored_primary_discovery`).

**Selection and confirmation halves (fixed now, before any score).**
- The systems holding the 210 scored cells are split into a selection half (S) and a confirmation
  half (C) by `registered.assign_halves`, stratified by diglycolamide / other and balanced on scored
  cells (`feasibility_halves.csv`, design `V5_system`; `feasibility_v5_cells.csv`, `selection_half`).
- S holds 105 cells in 18 systems (59 Ln, 46 non-DGA); C holds 105 cells in 19 systems (50 Ln,
  46 non-DGA) (`feasibility.json` → `Q1_V5_cells.selection_halves_primary_scored`).
- Discovery, the ladder and claim freezing score S only. Confirmation scores C (§15). Batching and
  every other V5 rule apply within a half.

**Fold construction.**
- Exact leave-one-cell-out for the deterministic and cheap arms: B0–B4, B6, B7.
- For B5, B8 and M0–M7, cells are **batched**. A batch is a colour class of a greedy colouring of
  the conflict graph, where two cells conflict when they share a metal state or a system. Vertex
  order is drawn by the run seed.
- Each batch is one outer fit. Every eligibility condition is re-checked with all of the batch's
  cells hidden; a cell failing it moves to a singleton batch.
- **Registered check:** on B6, batched and exact macro MAE must differ by < 0.01 log D. Otherwise
  every arm uses exact leave-one-cell-out.
- Batches per discovery seed, both halves together, seeds 104729 / 130363 / 155921 / 196613 / 262147
  (`folds/INDEX.json` → `placeholders.V5_batches_per_discovery_seed.<variant>.<seed>.total`):
  - primary: 38 / 37 / 37 / 37 / 37; for seed 104729, 22 selection-half and 16 confirmation-half
    batches (`.S`, `.C`);
  - loose: 41 / 41 / 41 / 41 / 41;
  - strict: 23 / 23 / 23 / 23 / 23;
  - HNO3-only: 37 / 37 / 37 / 37 / 37;
  - cell-only: 37 / 37 / 37 / 37 / 37;
  - parent-structure: 38 / 37 / 37 / 37 / 37.

**Sensitivity V5-cell-only.** Component-sharing rows stay in training. This is the laxer reading,
in which a mixture containing the extractant counts as a different system.

**V5-P (publication-masked, brief B4 analogue; secondary).**
- Base: the V5-P base setting (132 cells).
- For each scored cell and each of its publication groups, the rows of that group are scored with
  training that excludes the whole cell **and** every row of that publication group.
- A cell stays eligible only if, after this masking, its system keeps ≥ 3 other metal states and its
  metal ≥ 3 other systems.
- Eligible after masking: 101 cells in 15 systems (selection half 56, confirmation half 45), scored in
  420 cell × publication-group folds, out of 125 base cells scored after the V6 carve-out
  (`folds/INDEX.json` → `placeholders.V5P_eligible_after_masking`: `n_eligible_cells`,
  `n_eligible_systems`, `n_eligible_cells_by_half`, `n_folds`, `n_base_scored_cells`; counted under
  the fold builder's reading that every group-masked training set of a cell must pass, `readings.v5p`).
  V5-P is dropped, and the report says so, if fewer than 30 cells in ≥ 5 systems remain; at 101 cells
  in 15 systems it is retained (`retained`).

**V5-PAIR (secondary; carries the logSF and direction endpoints of H1).**
- Unit: an unordered pair of primary-eligible cells `(A × S, B × S)` under one system.
- The two cells must share ≥ 3 comparable row pairs. Each cell pair is labelled by its category class
  (Ln–Ln, An–Ln, An–An, other), and results are reported per class.
- Both cells are hidden together, plus their alias rows. Eligibility is re-checked with both hidden.
- logSF and direction are scored on the comparable test–test pairs only.
- Supply: 23,551 comparable row pairs over 216 systems, of which Ln|Ln 21,264, An|Ln 1,207 and
  An|An 933 (`feasibility.json` → `pairwise_supply`). 111 of the 123 primary Ln cells have a ±1
  series neighbour under the same system (`feasibility_v5_grid.csv`).
- Cell pairs touching `V6_TARGET_ROWS` are excluded, as in the carve-out. Cell pairs are scored in
  the half of their system.
- Eligible cell pairs by category class: Ln–Ln 405, An–Ln 31, An–An 31, other 5; in total 472 cell
  pairs in 29 systems (selection half 309 in 16 systems, confirmation half 163 in 13) with 11,770
  comparable test–test row pairs (`folds/INDEX.json` → `placeholders.V5PAIR_eligible_cell_pairs`:
  `n_eligible_cell_pairs_by_class`, `n_eligible_cell_pairs`, `n_eligible_systems`, `by_half`,
  `n_test_test_row_pairs`). V5-PAIR is dropped, and the report says so, if fewer than 30 cell pairs in
  ≥ 10 systems remain; at 472 cell pairs in 29 systems it is retained (`retained`).
- **Resolved (orchestrator, 2026-09-15; compute-driven, no learned-model score existed):**
  - *Closed-form arms* (B0–B4 variants, B6, B6r0, B7, FLAT, HEAVIER) run the unbatched outer folds of
    V5-P and V5-PAIR, as pre-seal. Exception: for S1(c), B3x and B3i are also re-fitted on exactly the
    batched V5-PAIR folds M2 is fitted on (§9 S1(c), "Same fitted folds").
  - *Heavy arms* (B5, B8, FLAT_CAT, M0–M7) run **batched** outer folds of V5-P and V5-PAIR, built before
    discovery by the §3.1 colouring rule on seed 104729 only: V5-P units are cell × publication-group
    folds that conflict when they share a metal state, a system or a publication group; V5-PAIR units are
    cell pairs that conflict when they share a metal state or a system. Eligibility is re-checked with
    every unit of a batch hidden; a failing unit moves to a singleton batch. Exception: for S1(c) at
    confirmation, batched V5-PAIR folds are also built by the same colouring rule with each withheld seed,
    and M2, B3x and B3i are fitted on them (§9 S1(c), "Same fitted folds").
  - *Consistency correction (task X, 2026-09-15; not score-driven): the two bullets above had no
    exception for S1(c); the §9 S1(c) "Same fitted folds" resolution requires B3x and B3i re-fitted on the
    batched V5-PAIR folds and, at confirmation, batched V5-PAIR folds built with each withheld seed. The
    exceptions add nothing beyond that resolution.*
  - No inner cell-pair design is built. Heavy-arm fits on V5-P and V5-PAIR are tuned with the V5 inner
    design of §7 (inner hidden cells recomputed on the outer training rows); logSF and direction are
    derived from the same fitted model's row predictions.
  - V5-P and V5-PAIR heavy-arm runs happen only for contrasts that are candidates for freezing (§15),
    i.e. that passed R19 items 1–5 on V5-primary. (Exception: the S1(c) selection-half counterweight of
    §9 needs M2 on the seed-104729 batched V5-PAIR folds; that run is part of the H1 evaluation.)
  - The batch re-check recomputes every eligibility condition of a unit's cell with the other units of
    the batch hidden, including the cell's own row and publication counts (k, p) (resolved by the
    orchestrator, 2026-09-15: the literal and stricter reading the fold builder applied; it moves 17
    confirmation-half V5-P units to singleton batches; not score-driven).
  - The nested-certificate safeguard (§2) is drawn for every fold file whose inner splits use a
    certificate (`folds/INDEX.json` → `nested_certificate_safeguard_sample`); the V1 and V2 draws are
    kept and labelled vacuous (their inner splits are checked directly).

### 3.2 V1 — leave-publication-out (secondary)

- **Groups:** `group_cross_publication_copy` (§2).
- **Exact design** (`FEASIBILITY.md`): leave-one-group-out over the 103 groups with ≥ 20 MODEL rows,
  plus one remainder fold pooling the groups below 20 rows (104 folds). The copy grouping includes
  archive duplicate-group links: without them 5 folds split an E_VALUE_CONFLICT group and fail the
  guard.
- **Halves.** The 104 folds are split 52 / 52 (6,206 / 6,205 MODEL rows) into selection and
  confirmation folds (`feasibility_halves.csv`, design `V1`).
  - The halves were balanced with the pooled remainder fold as ONE unit (`feasibility_halves.csv`,
    unit `REMAINDER`, weight 309, half S), while the pre-seal run scored the publication group (§4 as
    read before the resolution below), so the remainder fold's groups entered the selection half as
    separate units and the confirmation half had none. Under the scored-row rule of §2, 83 of the 104
    folds hold scored rows (S 39, C 44; `evaluation/preseal/guard_log.csv`, job `V1__copy`, status
    `fitted`). As run, the scored units were S 65 / C 44 publication groups
    (`evaluation/preseal/difficulty.json` → `V1.n_units_selection`;
    `evaluation/preseal/difficulty_resolved.json` → `V1.confirmation_descriptive_exploratory_publication_group.n_units`),
    of which 27 selection units came from the remainder fold
    (`V1.unit_sensitivities_exploratory.n_selection_units_from_remainder_fold`). Under the resolution
    below the scored units are the outer folds, S 39 / C 44 (`evaluation/preseal/difficulty_resolved.json`
    → `V1.selection.n_units`, `V1.confirmation_descriptive.n_units`). *Consistency correction (task X,
    2026-09-15; not score-driven): this bullet read "the scored unit is the publication group (§4)" and
    "the scored units are S 65 / C 44 publication groups", which the resolution below superseded; the
    as-run counts are kept, labelled as run, and the C 44 citation `tables/preseal_summary.csv` (job
    `V1__copy`, half `confirmation`, arm `B3`, `n_units`), which now holds a row for each unit reading,
    is replaced by the per-reading keys of `difficulty_resolved.json`.*
  - **Resolved (orchestrator, 2026-09-15; the pre-seal baseline V1 scores under all three unit readings
    had been seen — `V1.unit_sensitivities_exploratory`: B3 1.262 / 1.033 / 1.026, B0 1.164 / 1.142 /
    1.146 — and no learned-model score existed):** the V1 scoring unit is the **outer fold**: each of the
    103 single-group folds is one unit and the pooled remainder fold is **one** unit. Reasons, neither of
    which is the baseline values: (i) it is the unit the halves were balanced on, so both halves are
    scored on the unit they were built from; (ii) every unit then holds ≥ 20 MODEL rows, the threshold
    that defined the folds, instead of 27 selection units below it. Its consequence is disclosed: under
    this reading the V1 comparator B3 is *stronger* (1.033 vs 1.262 as run), which makes a V1 claim
    against it harder, not easier. The per-publication-group reading is printed beside it, labelled
    exploratory. §4 is read accordingly for V1.
- **Which arms run how.**
  - B0–B4, B6 and B7 run the exact design.
  - B5, B8 and M0–M7 run ten grouped outer folds (seeded greedy balance of MODEL row counts, each
    group whole).
  - **Registered check:** on B6, the ten-fold and exact macro MAE must differ by < 0.01 log D.
    Otherwise every arm runs the exact design.
- Scored unit: the outer fold, the pooled remainder fold one unit (the §3.2 resolution above; §4 is read
  accordingly). *Consistency correction (task X, 2026-09-15; not score-driven): this line read "the publication
  group (§4)", which the resolution superseded.*
- `fold_isolation_check(level="V1", near_dup_value_tol=0.005)` must pass. All 104 folds do
  (`tests/test_registered_folds.py`). Raw `g19_publication_id` folds fail it.
- Rows in `V6_TARGET_ROWS` are not scored.
- **Sensitivities:** `group_near_duplicate_key` folds with the value-blind check;
  `group_compilation_doi` folds.

### 3.3 V2 — leave-metal-out (secondary)

- **Units.** The 23 metal states with ≥ 100 MODEL rows and ≥ 5 systems (`feasibility.json` →
  `V2_leave_metal_out.grid`):
  - 14 Ln(III) (La–Lu without Pm);
  - Th(IV), U(VI), Np(IV), Np(V), Pu(III), Pu(IV), Pu(VI), Am(III), Cm(III).
- **Hiding.** One fold per state. **Every row of its element** is removed from training: all known
  states and X(?) (`FEASIBILITY.md`; `fold_isolation_check(level="V2", element_level=True)`). For
  Nd that is 433 known-state plus 252 unknown-state rows (`feasibility.json` →
  `V2_leave_metal_out.ln_focus`).
- **Scoring.** Only the target state's rows are scored.
- **Sensitivity (state-level hiding).** Other known states of the element stay in training. This
  changes only U, Np, Pu and Am.
- **Summaries.**
  - Primary summary: the seven focus lanthanides of brief V2 (La, Ce, Pr, Nd, Sm, Eu, Gd).
  - Secondary summaries: all 23 states, all 14 Ln(III), and all 9 actinide states.
- **V6 carve-out.** Rows in `V6_TARGET_ROWS` are excluded from V2 scoring before confirmation.
- **Halves.** Stratified by focus lanthanide / other lanthanide / actinide: selection Ce, Pr, Nd, Gd,
  Tb, Ho, Lu(III), Th(IV), U(VI), Np(IV), Pu(IV), Cm(III); confirmation La, Sm, Eu, Dy, Er, Tm,
  Yb(III), Np(V), Pu(III), Pu(VI), Am(III) (`feasibility.json` →
  `V2_leave_metal_out.v2_halves_rows100_systems5`). The focus-7 summary is reported for each half.

### 3.4 V6 — Pr/Nd double-cell hold-out (secondary; the applied test; run ONCE)

- **Systems.** The `FEASIBILITY.md` set: systems with ≥ 5 Pr(III) and ≥ 5 Nd(III) rows inside
  publications that measured both, plus ≥ 2 other Ln(III) (`feasibility.json` →
  `V6_prnd_double_cell.grid`, setting `min_pr_and_nd_rows_in_common_publications = 5`,
  `min_other_ln_iii = 2`).
- That gives **13 systems** and 209 comparable Pr/Nd row pairs, 142 of them in HNO3. 5 of the
  systems are not pure diglycolamide, and only 4 span ≥ 2 common publications.
- The systems, with comparable pairs and common publications (`feasibility_v6_systems.csv`,
  `v6_eligible__rows5_otherln2`):

  | system | pairs | common publications |
  |---|---|---|
  | TODGA | 83 | 10 |
  | DMDODGA | 25 | 3 |
  | C5BTBP | 16 | 2 |
  | DOODA (C12) | 16 | 1 |
  | TDdDGA | 12 | 1 |
  | DMDO-HPyranDGA | 12 | 1 |
  | N,N'-dimethyl-N,N'-diphenylpyridine-2,6-dicarboxamide | 10 | 1 |
  | DOODA (C8) | 8 | 1 |
  | DMDOHEMA\|TODGA | 6 | 1 |
  | TEDGA | 6 | 1 |
  | DMDPDGA | 5 | 2 |
  | TPrDGA | 5 | 1 |
  | mTDDGA | 5 | 1 |

- TODGA is the only brief §19 candidate with direct Pr and Nd support: 83 pairs in 35 condition
  groups, split **HCl 54 (6 groups), HNO3 26 (26 groups), malonic acid 3 (3 groups)**
  (`feasibility_v6_systems.csv`, `comparable_prnd_row_pairs_by_acid`). Only DMDODGA also has
  non-nitrate pairs (HCl 10 of 25). TODGA is reported on its own as well.
- **Hiding.** Per system S, the V5 state-level rule of §3.1 is applied to the cells Pr(III) × S and
  Nd(III) × S together (`support_graph.hide_cells(component_aware=True)`). In S and in every system
  sharing a component with S this removes:
  - the Pr(III) and Nd(III) rows;
  - the Pr(?) and Nd(?) rows.

  Pr and Nd have no other recorded state (`feasibility_metal_states.csv`,
  `other_known_states_same_element` empty for Pr(III) and Nd(III)), so these are every Pr and Nd row
  of those systems. Pr and Nd stay in every other system.
- **Scoring.**
  - log D of the hidden Pr and Nd rows;
  - logSF_Nd/Pr and its direction on test–test comparable pairs;
  - interval coverage.
  - Every V6 number is reported pooled and per acid medium. The nitrate process case (§14) rests on
    the HNO3 pairs only.
- **Once only.** V6 is not run in Phase C or discovery. It runs once, with the frozen configurations
  and the withheld seeds (§15).
- **Sensitivity.** The 7-system setting (≥ 10 Pr and Nd rows, ≥ 5 other Ln(III); 168 pairs) is
  scored in the same single run.

### 3.5 V3, V4, V7 (secondary, boundary-defining)

**V3 — leave-extractant-out.**
- 30 systems with ≥ 50 rows, ≥ 3 metal states and ≥ 2 family peers with ≥ 50 rows: 23
  diglycolamide, 7 monoamide (`feasibility.json` → `Q4_V3_V4_experts.v3_grid`).
- Systems with a name–structure-conflict component are never units; none of the 30 has one.
- Hidden: the system plus every row of any system sharing a component structure with it (guard level
  V3). Sensitivity: components linked through the stereo-free parent structure.
- Scored: the system's rows outside `V6_TARGET_ROWS`. Summary: macro over systems.
- Limitation: Morgan r = 2 fingerprints give Tanimoto 1.0 between long-chain DGAs such as TODGA and
  TDdDGA. The nearest remaining ligand scores 1.0 for 107 of the 224 primary V5 cells †
  (`feasibility_v5_cells.csv`, `hidden_nearest_ligand_tanimoto`).
- Every ligand-distance use in this registration therefore combines Tanimoto with a chain-length-aware
  distance: `d_desc` = Euclidean distance in (`mw`, `mol_logp`, `rotatable_bonds`) of
  `descriptors/extractant_components.csv`, standardised over outer-training systems. This applies to
  B3l ties (§5) and support component s4 (§13), and answers `FEASIBILITY.md` open item 5.

**V4 — leave-family-out.**
- Four families with ≥ 3 systems and ≥ 200 rows once systems with a name–structure-conflict component
  are excluded (`feasibility_families.csv`, `v4_eligible_excl_conflicts__systems3_rows200`):

  | family | systems | rows |
  |---|---|---|
  | diglycolamide | 95 | 7,616 |
  | monoamide | 34 | 2,154 |
  | N-heterocyclic | 23 | 709 |
  | pyridine carboxamide | 30 | 513 |

- Malonamide (12 systems, 195 rows) falls below the threshold and is scored as a sensitivity only.
- Hidden: every system containing a member of the family (resolved component families), mixed
  systems included (guard level V4). For diglycolamide that is 7,933 rows.
- Scored: rows of the family's pure, conflict-free systems outside `V6_TARGET_ROWS`.
- All four are neutral or soft-N-donor families; the brief's phosphinic-acid example has zero rows.
- No success threshold. V4 defines the boundary and is reported with the full baseline ladder.

**V7 — condition extrapolation.**
- Unit: system × acid identity.
- Per variable, hide the top 20 %, and separately the bottom 20 %, of the unit's own log10 range.
  A unit qualifies with ≥ 10 rows inside and ≥ 10 outside the region (`feasibility_v7_grid.csv`,
  `unit = system_x_acid`, `region_fraction 0.2`, `min_rows 10`).
- Registered variables:
  - acid molarity: 65 units / 55 systems (61 top, 24 bottom);
  - extractant molarity: 33 units / 31 systems.
- Metal concentration (28 units / 27 systems) is exploratory: it only bounds loading.
- Loading and pH regions cannot be formed (no columns).
- Units in different systems are batched.
- The acid variable is run with and without the 302 acid-grid-flagged rows.
- The fold builder asserts that no training row of a unit lies inside its hidden region (V7 has no
  guard level, §2). Rows in `V6_TARGET_ROWS` are not scored.

### 3.6 V0 — random split (diagnostic only)

- Five-fold row-level random split per discovery seed, over all 12,411 MODEL rows. Every row is hidden
  in exactly one fold per seed; only the rows of §2's scored-row rule are scored (X(?), Sr(III),
  `V6_TARGET_ROWS` and acidic co-extractant rows are hidden with their fold, trained on in the other
  four, and never scored; correction of the earlier text, which named only `V6_TARGET_ROWS`;
  `folds/INDEX.json` → `readings.scored_rows`, `readings.v6_rows_in_a_hidden_unit`).
- Reported in every table as `V0 (diagnostic)`.
- As run pre-seal, the V0 averaging unit was the publication group (`group_cross_publication_copy`),
  per seed and as the mean over the 5 discovery seeds, and its calibration design was the grouped
  3-fold of §7 V1 (`evaluation/preseal/difficulty.json` → `readings.V0`, `readings.V0_inner_design`).
  **Resolved (orchestrator, 2026-09-15; not score-driven):** the as-run reading is registered. V0
  averaging unit = publication group (`group_cross_publication_copy`) within each seed; seed aggregation
  = mean over the 5 discovery seeds of the per-seed macro MAE; F1's 95 % interval = percentile interval
  of a publication-group cluster bootstrap (10,000 resamples, seed 19) of the seed-mean statistic, with
  the same resampled groups applied to every seed. V0 has no secondary cluster.
- Never used for selection, tuning or any claim (brief §1.1). Used only for failure condition F1
  (§10).

## 4. Metrics

**Averaging unit (named for every design).**

| design | unit of scoring | primary aggregation |
|---|---|---|
| V5, V5-P | hidden cell | mean of \|ŷ − y\| over the cell's scored rows → **equal-weight mean over cells** |
| V5-PAIR | hidden cell pair | mean over the pair's test–test comparable row pairs → equal-weight mean over cell pairs |
| V1 | outer fold (§3.2 resolution: each single-group fold; the pooled remainder fold is one unit) | mean over the fold's scored rows → mean over folds; the per-publication-group reading is printed beside it, exploratory |
| V2 | metal state | mean over the state's scored rows → mean over states (focus-7, all-23, Ln-14, An-9) |
| V6 | system | mean over hidden Pr and Nd rows (each metal weight ½) → mean over the 13 systems |
| V3 / V4 / V7 | system / family / unit × direction | as named |

Row-pooled and system-macro versions are printed beside each primary aggregate, labelled, and
never used for a decision. V5 is also reported by stratum: Ln / An / other,
diglycolamide / non-DGA, acid medium of the scored rows (HNO3 / HCl / organic acids), and
domain-status category (§13). Every table names the half (selection or confirmation) it was scored on.

**log D.**
- MAE is the primary metric.
- Also reported: RMSE, R², and the Spearman ρ between ŷ and y across the design's scored rows. For
  V1 these are the brief's list.
- Within-cell Spearman ρ over cells with ≥ 5 rows spanning ≥ 3 distinct condition keys, macro over
  cells.
- **Rank accuracy:** over row pairs inside one scoring unit, in the same publication group, with
  |Δy| ≥ 0.1, the fraction ordered correctly by ŷ. A tie in ŷ counts ½.

**logSF and direction.**
- logSF_A/B = log D_A − log D_B on comparable test–test pairs (§2). The predicted value is ŷ_A − ŷ_B
  from the same fitted model.
- Metrics:
  - logSF MAE;
  - **direction accuracy**: sign agreement on pairs with |observed logSF| ≥ 0.3, plus, for V6
    only, on pairs with |observed logSF| ≥ 0.1. Predicted zero counts ½.
- Every direction number is printed with its n.
- **FLAT floor:** predict no separation, logSF = 0. Its MAE is mean |observed logSF| and its
  direction accuracy is ½.
- **HEAVIER yardstick:** for Ln–Ln pairs the heavier lanthanide is predicted more extracted. It is
  not defined for pairs with an actinide. As computed pre-seal it is defined only for Ln(III)–Ln(III)
  row pairs (`gen19ct.evaluation.pairs.heavier_direction`); on every other pair (An–Ln, An–An, other)
  it is undefined, and undefined pairs are excluded from its accuracy and counted, never scored ½
  (`evaluation/preseal/difficulty.json` → `readings.DIR5`).
- Pair metrics are quoted against FLAT and against the cheapest sensible alternative (§5 table).

**Intervals.**
- Central 50 %, 80 % and 95 % predictive intervals per row.
- Empirical coverage per scoring unit (fraction of its rows covered), aggregated as the table above.
  Pooled coverage is printed beside it.
- Also: mean interval width, Gaussian CRPS (closed form, implemented by hand), and Spearman ρ of
  |error| with predicted SD. §12 has the full evaluation.

## 5. Baselines (exact definitions)

Every baseline is fitted on the outer-training MODEL rows of its fold only. Each baseline records
per prediction the fallback level it used, and the counts are reported.

* **B0 — global mean.** Row-weighted mean of training `log_D`.
* **B1 — metal mean.** Mean over training rows with the same `g19_metal_state`. If there are none,
  use the element's unknown-state rows; if none, B0.
* **B2 — extractant mean.** Mean over training rows with the same `extractant_system_key`. If none,
  the same `system_family`; then the same brief expert (mechanism group); then B0.
* **B3 — gen18 nearest-condition lookup.** This is the `NearestConditionD` form used by gen18's R1
  evaluation: raw 1-NN, no depletion term, no tracer lift.
  - **Candidates:** training rows with the same `extractant_system_key`, the same
    `g19_metal_state` and the same `acid_anion`. If none share the anion, drop the anion
    (flagged).
  - **Distance:** Euclidean in (log10 acid M, log10 extractant M), unstandardised.
  - **Ties:** smaller |Δ log10 acid|, then smaller `canonical_measurement_id`. Replicate rows at
    identical conditions therefore contribute one row, not their mean (gen18 averages each replicate
    group before its lookup; `gen19ct/models/baselines.py` module docstring, "Relation to gen18").
    **Resolved (orchestrator, 2026-09-15; L5, δ5 and the comparator choice had been computed under
    it):** the one-row tie rule is confirmed for B3 and every lookup built on it, exactly as implemented
    before any score. The replicate-mean reading is not run. Reason: it is the implemented, tested
    definition that produced every §9 difficulty number; replacing it now would re-open those numbers.
  - **Missing coordinates:** candidates missing a coordinate are used only when every candidate
    misses it; the prediction is then their mean.
  - **Undefined** when the (system, metal) pair has no training rows. The cross-cell variants below
    then apply:
  * **B3x — cross-metal lookup.**
    - Within the same system (and anion), replace the metal by the nearest-radius training metal
      state. "(and anion)" restricts the pool itself: the pool is the training metal states of the
      query's system that have a training row with the query's acid anion; the anion is dropped (flag
      `radius_pool_anion_dropped`) only when no other metal state of the system has one. This is the
      reading implemented before any score (`gen19ct/models/baselines.py`,
      `REGISTRATION_CHOICES["B3x_pool_anion"]`; `evaluation/preseal/difficulty.json` → `readings.B3x_B3i_anion_pool`),
      recorded here as a clarification of the text, not chosen by a score. The same pool applies to B3i
      and B4x. The pre-seal numbers under the other plain reading (pool over every anion of the system,
      anion applied only in the B3 step) were computed by a verifier and have been seen.
      Inside that set the charge pool comes first: same formal charge and same aqueous-species charge (an
      actinyl borrows an actinyl), else same formal charge, else every metal. Inside the pool, the
      smallest |Δ Shannon radius| on the CN8 basis when the query and a pool member have CN8 radii,
      else CN6; ties by label order. This is `SupportIndex._nearest_radius` applied to that anion-restricted
      pool (`gen19ct/models/baselines.py` `nearest_metal`); it equals `SupportIndex.features` →
      `nearest_radius_metal` only when every metal state of the system has a training row with the query's
      anion (U(VI) borrows Np(VI) or Pu(VI), not Pu(IV) or Zr(IV)).
    - Then apply the B3 rule within that (system, metal).
  * **B3i — radius interpolation between lookups.**
    - Within the same system (and anion, read as for B3x: the anion restricts the pool), take the nearest
      same-charge training metal state with a smaller radius and the nearest with a larger radius, on the B3x basis
      (the bracket rule of `radius_bracket_lower_metal` / `radius_bracket_upper_metal`, applied to the
      anion-restricted pool; equal to those `SupportIndex.features` fields only when every metal state of
      the system has a training row with the query's anion).
    - Apply the B3 rule to each, then interpolate linearly in radius to the hidden metal's radius.
    - Where the metal is not bracketed (`radius_bracketed_same_charge` false), B3i equals B3x. Of the
      210 discovery-scored primary cells, 118 are bracketed
      (`feasibility.json` → `Q1_V5_cells.radius_interpolation_supply_primary_scored`).
  * B3x and B3i together are the cheapest sensible alternative for V5, V5-PAIR, V2 and V6 (table below).
  * **B3l — cross-ligand lookup.**
    - Within the same metal state, replace the system by the training system with the highest
      Morgan-r2 Tanimoto on the primary-extractant SMILES (`nearest_ligand_system`).
    - Ties (Tanimoto 1.0 is common among homologues, §3.5): smaller `d_desc`, then same
      `system_family`, then more training rows of that metal, then key order.
    - Then apply the B3 rule.
  * When neither variant is defined, B2 applies.
* **B4 — same-source-excluded lookup.** B3 with every training row of the target row's publication
  group removed from the candidate set. **B4x** and **B4l** are B3x and B3l with the same removal,
  and the nearest-radius metal or nearest system is re-chosen after the removal. Under V1 it is identical to B3 by construction
  and is printed once, labelled.
* **B5 — CatBoost on engineered descriptors (= M0).**
  - **Metal block** (`descriptors/metals.csv`): Z, formal charge, CN6/CN8/CN9 radii, f- and
    d-electron counts, category/series, HSAB class, Pauling electronegativity. Missing values stay
    NaN. Polarizability is excluded (all NA).
  - **Extractant block** (`descriptors/extractant_systems.csv`, `extractant_components.csv`):
    - family, mechanism, number of components;
    - donor-site type counts (`donor_*`), `n_donor_sites`, `denticity_proxy`;
    - per-component `mw`, `mol_logp`, `tpsa` and `rotatable_bonds`, weighted by concentration
      share;
    - a 16-component PCA of Morgan count fingerprints, fitted on outer-training systems.
  - **Condition block** (`gen19ct.data.normalize.condition_vector`):
    - log10 acid M, acid anion, log10 extractant M, log10 metal M;
    - O/A, temperature, contact time, diluent family;
    - modifier and complexant presence and log concentrations;
    - the acid-grid flag.
  - **Settings:** loss RMSE, eval metric MAE; depth ∈ {4, 6, 8}; learning rate 0.05; `l2_leaf_reg`
    ∈ {3, 10}; ≤ 3,000 iterations with early stopping after 200 rounds on the inner validation fold;
    `one_hot_max_size` 10; `thread_count` 2.
* **B6 — metal × extractant factorisation with side information (ALS / ridge).**
  - Model: `y = μ + a_m + b_s + u_mᵀ v_s + βᵀ x_cond + ε`.
  - `u_m = A z_m + δ_m` and `v_s = B z_s + δ_s`, where z_m and z_s are the standardised numeric
    metal and extractant blocks of B5 (training medians imputed).
  - Main effects follow the same form: `a_m = α_aᵀ z_m + a'_m` and `b_s = α_bᵀ z_s + b'_s`.
  - Fitted by alternating ridge least squares over rows: ≤ 50 sweeps, relative tolerance 1e-6,
    seeded initialisation.
  - A metal or system absent from training gets its free parts (δ, a′, b′) set to 0.
  - Rank k ∈ {0, 1, 2, 3} and penalties λ ∈ {0.1, 1, 10} are chosen in inner folds.
  - **B6r0** (k = 0, the additive "flat" model) is always reported as well.
* **B7 — mechanism-aware mass action (gen18 M1 form).**
  - Defined for systems labelled NEUTRAL_SOLVATING, SOFT_N_DONOR or MIXED_NEUTRAL.
  - Per (system, acid anion): `log D = a_m + n·log10[L] + p_eff·log10[acid] + δ_pub`. Slopes are
    shared across metals; publication effects are sum-to-zero for groups with ≥ 2 rows; WLS,
    unpenalised.
  - An axis with < 3 distinct training levels takes the gen18 prior n0 = 3.0 or p0 = 2.0 (status
    `assumed`, flagged).
  - A hidden metal's intercept is the intercept of its B3x nearest-radius metal. A new publication
    gets δ = 0.
  - Acidic cation exchangers (3 systems, 14 MODEL rows; `feasibility_mechanisms.csv`) have no pH
    column, so their slope is not identifiable: B7 falls back to B3x. With the system absent, B7
    falls back to B2. Both fallbacks are counted.
* **B8 — previous best Gen-series models, re-fitted inside Gen19 folds.**
  - **Level: MONO_ET architecture** (gen6/gen7 protocol; `src/lanthanide_separation/gen7`).
    - ExtraTrees, 400 trees, `max_features` 0.30, `min_samples_leaf` 2.
    - Sample weights balanced over ECFP clusters of the primary extractant, clustered on
      outer-training systems. Predictions clipped to 1.5× the training range.
    - Features are Gen19 equivalents of METAL (Z, lanthanide index, CN8 radius), COND (the B5
      condition block), LIG2D_EXT (2,048-bit ECFP4 of the primary extractant plus component count)
      and MASSACTION (log10 concentrations and n0·log10[L], recomputed).
  - **Direction: the gen14 G14 rule, for Ln(III)–Ln(III) pairs.**
    - An L2 logistic (C = 1, standardised, median-imputed) on the 39 TOPO39 donor-topology columns
      predicts the sign of the lanthanide-axis radius coefficient.
    - It is **re-fitted inside every outer training fold** on training systems with ≥ 5 Ln(III).
      The frozen gen14 model is never used, because its training cells include V6 targets.
    - Magnitude: the training-fold mean |logSF| of the pair's ΔZ class.
    - TOPO39 exists only for bundle extractants. Systems without it get no B8 direction; this
      coverage is reported.
* **FLAT_CAT (reference for H4).** CatBoost on categorical `g19_metal_state` and
  `extractant_system_key` plus the condition block. No descriptors, no factorisation; B5 settings.

**Cheapest sensible alternative per design** (the second comparator of gen13 Addendum 3; every claim
is quoted against it **and** against B0, or FLAT for pair metrics):

| design | log D comparator | pair comparator |
|---|---|---|
| V0 | B3 | B3-derived logSF |
| V1 | B3 (≡ B4) | B3-derived logSF |
| V2 | stronger of B3x and B3i | B3x-derived logSF, HEAVIER |
| V3 | B3l | B3l-derived logSF |
| V4 | B3l across families | FLAT |
| **V5 / V5-PAIR** | **stronger of B3x and B3i** (the one with the lower selection-half macro MAE, fixed before any learned model is scored; B3l, B4x printed beside; "lookup oracle" = per-cell min over B3x/B3i/B3l/B4x, labelled optimistic) | **HEAVIER, B3x-derived and B3i-derived** for direction, each paired with the candidate on identical pairs (§9 S1(c) as redefined; their max, `DIR5`, is descriptive only); FLAT and the chosen lookup for MAE |
| V5-P | B4x | B4x-derived logSF |
| V6 | stronger of B3x and B3i (as chosen for V5) | FLAT, HEAVIER, B3x- and B3i-derived, B8 |
| V7 | B3 (nearest condition at the region boundary) and B7 | — |

Every baseline, B3x included, also carries split-conformal intervals (§12), so calibration too is
compared with a cheap alternative.

## 6. Allowed model families and the ablation ladder

**Allowed:**
- means and lookups (B0–B4);
- gradient-boosted trees (CatBoost, XGBoost; LightGBM is excluded because its DLL is broken on this
  machine);
- ExtraTrees (B8);
- ridge/ALS factorisation (B6);
- WLS mass action (B7);
- small factorised neural models in PyTorch on CPU: ≤ 200 k parameters, embeddings ≤ 16 dimensions,
  ≤ 2 hidden layers of ≤ 128 units;
- ensembles, heteroscedastic heads, quantile heads, conjugate Bayesian ridge heads;
- split and cross-conformal calibration, implemented by hand (no mapie or properscoring);
- MC dropout, as a secondary uncertainty baseline only.

**Not allowed** (repository nulls or brief §33):
- pretrained chemical language models;
- graph neural networks over molecular graphs;
- similarity kernels or GPs as a primary arm (a design-B mirage in gen15);
- TabPFN;
- label de-noising;
- 3D or xTB descriptors (the gen15 composition trap);
- any prediction of gen15 or another model used as a label or pseudo-label;
- any external D value;
- hyperparameter search outside the grids of §5–§7.

**Ladder (brief §14).**
- Each step is compared under V5-primary, V1 and V2 with its **retained predecessor**, i.e. the last
  step that was kept, on the **selection half** of each design only (§3).
- A step is **kept only if** it passes R19 (§8) against its predecessor on the V5-primary selection
  half **and** is non-inferior on the V1 and V2 selection halves (TOST, margin ε = 0.05).
- Because keep/remove decisions use these outer scores, selection-half ladder results are
  optimistically biased. They are reported as such and never quoted as confirmed.
- **If a component adds no strict-fold benefit, it is removed**, and the next step builds on the
  predecessor.
- M7 is judged on calibration (§12), with non-inferior MAE.

| step | adds |
|---|---|
| M0 | B5 (descriptor CatBoost) |
| M1 | metal embedding `e_m = e_shared(z_m) + e_series + e_ox + e_element` (brief §15) and ligand embedding `e_l = W_l z_l + δ_l`, with a condition encoder; head is an MLP on `[e_m, e_l, h(x_cond)]` |
| **M2** | M1 + explicit bilinear interaction `I(m, l) = e_mᵀ W e_l`, rank(W) ≤ 8 — **the H1 candidate** |
| M3 | mechanism experts: separate `W` and condition heads for NEUTRAL_SOLVATING, SOFT_N_DONOR and MIXED_NEUTRAL (≥ 200 training rows each); one pooled expert for ACIDIC (14 rows), CHELATING (3), SYNERGISTIC (92) and UNKNOWN (29); routing by the known label, no learned gate (`feasibility_mechanisms.csv`). Every V5 cell is E2; inside V5-primary the testable split is NEUTRAL_SOLVATING 178 / SOFT_N_DONOR 29 / MIXED_NEUTRAL 17 cells † (`feasibility_v5_cells.csv`, `mechanism`) |
| M4 | source hierarchy: random publication-group effect `b_g ~ N(0, τ²)`, τ tuned in inner folds; population prior 0 for a new group (brief §9 model C; model B, a fixed embedding, is exploratory) |
| M5 | pairwise loss L_pair on training comparable pairs, generated after fold assignment |
| M6 | physics penalties: (a) hinge on −∂log D/∂log10[extractant] for neutral/soft-N systems, only inside each system × acid unit's training range of extractant concentration, never evaluated on acid-grid-flagged rows; (b) smoothness Σ‖e_Z − e_Z+1‖² over adjacent Ln(III) and An(III) element embeddings. No acid monotonicity (non-monotone for solvating extractants; no pH for acidic ones); no loading term (organic loading not recorded) |
| M7 | uncertainty ensemble: 5 members of the retained configuration (seeds and publication-group bootstrap) + heteroscedastic head + split-conformal calibration fitted in inner folds |

Composite-loss weights: λ_pair ∈ {0, 0.3, 1}, λ_phys ∈ {0, 0.1, 1}, λ_src ∈ {0.1, 1}; weight decay
∈ {1e-4, 1e-3, 1e-2}; embedding dimension ∈ {4, 8, 16}; early stopping on inner validation, at most
500 epochs.

**M-model training settings (resolved by the orchestrator, 2026-09-15, before any M fit; not
score-driven).**
- **Fixed for every M model:** AdamW, learning rate 3·10⁻³ with a 10-epoch linear warm-up, then
  constant; mini-batch 512 rows, shuffled per epoch by the model seed; float32; `torch.set_num_threads(2)`;
  target standardised with the outer-training mean and SD; Huber loss (δ = 1 in standardised units) for
  L_D, the same for L_pair; gradient-norm clip 5. Early stopping on the inner validation macro MAE,
  evaluated every epoch, patience 30 epochs, at most 300 epochs (this replaces "at most 500"), restoring
  the best epoch. The outer refit on all outer-training rows runs for the median best-epoch count of the
  inner folds.
- **Architecture constants:** metal and ligand encoders are linear maps from their descriptor blocks
  plus free per-unit offsets (§6 M1); condition encoder 2 layers × 32 units, SiLU; prediction head
  2 layers × 64 units, SiLU; dropout 0.1 in the head only.
- **Searched, per ladder step (never jointly):** each step searches only the hyperparameters its
  component introduces, with every earlier hyperparameter fixed at the value retained by the previous
  step (per outer fold):
  - M1: embedding dimension ∈ {4, 8, 16} × weight decay ∈ {1e-4, 1e-3, 1e-2} (9 configurations);
  - M2: rank(W) ∈ {2, 4, 8} (3);
  - M3: no new hyperparameter (routing by the known label);
  - M4: τ ∈ {0.1, 0.3, 1.0} log D (3), with λ_src fixed at 1 (the λ_src grid is dropped: τ carries the
    prior strength);
  - M5: λ_pair ∈ {0.3, 1} (2; λ_pair = 0 is M4);
  - M6: λ_phys ∈ {0.1, 1} (2; 0 is M5);
  - M7: no search (5 members at the retained configuration).
- Grids are not widened after results (§7).

**Architecture question (brief §31, H4).**
- Under V5-primary with identical batches, seeds and scored cells, compare:
  - flat: FLAT_CAT and B6r0;
  - descriptor: M0;
  - factorised: M2;
  - factorised + mechanism: M3.
- Registered contrasts: M2 − FLAT_CAT, M2 − M0, M3 − M2, and the linear analogue B6 − B6r0.

**Chemistry-priors question (brief §32, H5).**
- Four registered contrasts, each the component on vs off with everything else at the retained
  configuration: mechanism routing (M3 vs M2), source hierarchy (M4 on/off), monotonicity penalty
  (M6a on/off), series smoothness (M6b on/off).
- They are run under V1, V2 and V5 in discovery, on the selection halves.
- V6 enters only if the component is among the ≤ 5 frozen claims (§15).

## 7. Tuning procedure

* **Inner folds only.** Hyperparameters, early stopping, composite-loss weights, conformal scales and
  source priors are chosen inside each outer training set.
* **Outer scores inform exactly two kinds of choice:** ladder keep/remove decisions (§6) and the
  ≤ 5 frozen claims (§15). Both use the **selection half only**, and those results are labelled
  optimistically biased. The confirmation half and V6 inform no choice. **V6 is touched once**
  (§3.4).
* **Inner validation never scores `V6_TARGET_ROWS`** (§2). They stay in inner training.
* **Inner grouping is by publication group** (§2), with 3 inner folds.
  - **V1:** grouped 3-fold on publication groups.
  - **V5 (also for the V5-P and V5-PAIR heavy-arm fits, which are tuned with this V5 inner design; no inner
    cell-pair design is built, §3.1 resolution):** inner hidden cells are the training cells that are eligible
    under the same thresholds, recomputed on the outer training rows, excluding cells in
    `V6_TARGET_ROWS` (the 14 V6 Pr/Nd cells are never inner validation cells). They are split into 3 inner
    folds by the publication group holding each cell's majority of rows, with at most 30 cells per
    inner fold (seeded subsample). Batching follows the outer rule.
  - **V2:** leave-one-metal-out over 3 seeded training metal states that are themselves V2-eligible;
    an inner Pr(III) or Nd(III) fold scores only its rows outside `V6_TARGET_ROWS`.
  - **Assignment rule and random streams (one implementation per design).** Every random draw of an
    inner design comes from `numpy.random.SeedSequence([seed, tag, ...])` with the tags of
    `gen19ct.folds.io.STREAM_TAGS`, and the fold builder and every calibration or tuning split use the
    same functions (`tests/test_inner_designs.py` asserts identical inner units for one outer fold and
    seed):
    - V1 (and the V0 calibration design): `source_holdout.inner_group_assignment` — groups in the random
      order of stream `[seed, 11]`, each to the inner fold with the fewest outer-training rows so far
      (ties to the lower fold index), each group whole;
    - V5: `cell_holdout.inner_cell_majority` (eligible, no `V6_TARGET_ROWS` row, at least one scored
      row; majority group ties to the smallest group label) and `cell_holdout.inner_cell_assignment` —
      majority groups, weighted by their number of cells, in the random order of stream `[seed, 15]`,
      each to the inner fold with the fewest cells so far; a fold above 30 cells keeps 30 drawn by
      stream `[seed, 16, k]`; inner batches by stream `[seed, 17, k]`;
    - V2: `metal_holdout.inner_state_pick` — 3 of the sorted V2-eligible states drawn without
      replacement by stream `[seed, 12]`.
    This records the fold builder's rule (`folds/INDEX.json` → `readings.inner_v1`, `readings.inner_v5`,
    `readings.inner_v2`), which already applied the seeded greedy balance that §3.2 registers for the
    grouped outer folds. The pre-seal conformal wrapper had used a second implementation (groups
    heaviest first, `default_rng(seed)`); it was replaced by these functions because two
    implementations of one registered design disagreed (verification finding VL-01), before the
    intervals were recomputed and without comparing intervals under the two.
  - **V6:** V5-style inner cells.
  - **V3 and V4:** leave-one-system-out over 3 seeded eligible training systems.
  - **V7:** hide the same-direction region of 3 seeded training units.
* **Selection criterion.** Inner macro MAE with the design's own averaging. Configurations within
  0.005 of the best are resolved toward the smaller configuration: lower rank, fewer parameters,
  stronger penalty.
* **Compute plan (resolved by the orchestrator, 2026-09-15; compute-driven, no learned-model score
  existed).** The machine is a 12-thread CPU with 8 GB RAM; at most 2 worker processes. Selection-half
  outer folds: V5-primary batched 106 over the 5 seeds, the five refit sensitivities 516, V5-P 288 and
  V5-PAIR 309 unbatched (`folds/INDEX.json` → `designs.<stem>.by_half.S.n_folds`); the closed-form
  pre-seal V5-primary job took 2,468 s on one worker (`manifests/run_info/g19_run_preseal_jobs.json`).
  1. **Tuning is nested per outer fold on every seed; no hyperparameter is carried across seeds or
     folds.** On seed 104729 the full §7 inner design is used (3 inner folds). On the other four
     discovery seeds, and on the withheld seeds at confirmation, tuning uses the first of the 3 inner
     folds only (one fit per configuration), then one outer refit.
  2. **Ladder keep/remove decisions (§6) are taken on seed 104729** (R19 items 1–3, 5 and the
     scoring-filter sensitivities of item 6). R19 item 4 (seed consistency) and the refit sensitivities
     of item 6 (loose, strict, HNO3-only, cell-only, parent-structure, Sr(III)-dropped) are evaluated
     only for contrasts that are candidates for freezing (§15) and for the registered H1, H1b and H4
     contrasts; refit sensitivities run on seed 104729 only. Scoring-filter sensitivities (non-DGA
     stratum, censoring-candidate, acid-grid and wildcard-copy exclusions) re-score existing predictions
     and apply to every contrast.
  3. **Order of discovery:** B6/B6r0 (and the batched-vs-exact check) → B5 = M0, FLAT_CAT, B8 → M1 → M2 →
     M3 → M4 → M5 → M6 → M7.
  4. **Stop rule (brief §28 Phase D→E, "only after Phase D shows signal").** If neither M2 nor B6 passes
     R19 items 1–3 and 5 against the V5 comparator B3i on the V5-primary selection half (seed 104729),
     M3–M6 are not run as registered steps (they may run as exploratory, labelled so), M7 is run once on
     the retained configuration for H6, and H7 (process) is not run because S1 cannot pass.
  5. **Budget:** 60 hours of wall-clock discovery compute. If it is exhausted, steps are demoted to
     exploratory (not run, or reported as incomplete) in the order M7 → M6 → M5 → M4 → M3; H1, H1b and
     H4 are never demoted. Elapsed compute per step is recorded in `manifests/run_info/`.
  6. **B6 batched-vs-exact check fails** (|Δ macro MAE| ≥ 0.01): batches are re-coloured with at most 4
     cells per batch (same seeds) and the check is repeated once. If it still fails, every heavy-arm V5
     result is labelled "batched (check failed)" and S1 is reported UNDECIDED, never passed.
  7. V5-P and V5-PAIR heavy-arm folds follow §3.1 (batched, seed 104729, only for freezing candidates;
     the S1(c) exceptions of §3.1 and §9: M2 on the seed-104729 batched V5-PAIR folds for the
     selection-half counterweight, and batched V5-PAIR folds built with each withheld seed at
     confirmation, with B3x and B3i re-fitted on the same folds). *Consistency correction (task X,
     2026-09-15; not score-driven): this item read "(batched, seed 104729, only for freezing
     candidates)" without the exceptions the §9 S1(c) "Same fitted folds" resolution requires.*
  8. V1 heavy arms use the ten grouped outer folds per seed (§3.2); V2 heavy arms use the 12
     selection-half metal states per seed.
* **No outer-informed choice.** Any choice made after an outer score has been seen is exploratory and
  requires a POST-HOC addendum. Grids are not widened after results.

## 8. Statistical procedure and decision rule R19

**Bootstrap.**
- Paired cluster bootstrap: 10,000 resamples, `numpy.random.default_rng(19)`. Clusters are resampled
  with replacement, and the scored units (held-out cells, V1 outer folds, metal states) nested in
  them enter as they fall. The design's macro statistic is recomputed on each resample.
  *Consistency correction (task X, 2026-09-15; not score-driven): the V1 unit here read "publication
  groups", which the §3.2 resolution (V1 scoring unit = outer fold) superseded.*
- An unclustered bootstrap over scored units is printed for reference and never decides.
- Percentile and BCa 95 % intervals, two-sided bootstrap p, and `mde_80 = 2.80 × bootstrap SD` are
  reported for every contrast.
- Cluster units:

  | design | primary cluster | secondary cluster |
  |---|---|---|
  | V5 / V5-P / V5-PAIR | `extractant_system_key` (37 systems scored at the primary setting: 18 in the selection half, 19 in the confirmation half) | publication group of each cell's majority rows |
  | V1 | publication group (`group_cross_publication_copy`) | — |
  | V2 | metal state | — |
  | V3 / V4 / V7 | system | — |

- V6 has 13 systems. A system-cluster bootstrap is computed (percentile only; BCa is unreliable at 13
  clusters), and S2 (§9) states where it is used. Exact sign counts and a per-system table are
  reported; a pair-level bootstrap over condition groups within systems is descriptive only.

**Decision rule R19.** A registered contrast Δ = metric(comparator) − metric(candidate) (sign flipped
for higher-is-better metrics) **passes** only if all of the following hold:

1. the point estimate Δ ≥ the design's margin δ (§9);
2. the percentile and BCa 95 % intervals exclude 0 under **every** registered cluster unit;
3. the two-sided p < 0.05 under **every** registered cluster unit, as in item 2 (resolved by the
   orchestrator, 2026-09-15: the conservative reading; not score-driven);
4. Δ > 0 in ≥ 4 of the 5 discovery seeds on the selection half (5 of 5 withheld seeds on the
   confirmation half at confirmation);
5. removing any single cluster (system, publication group or metal state) does not make Δ ≤ 0;
6. Δ > 0 in every registered sensitivity of the design. For V5 these are:
   - the loose and strict settings, V5-P and V5-cell-only;
   - the non-DGA stratum and the HNO3-only cells;
   - the parent-structure hiding;
   - the acid-grid-excluded rows, the censoring-candidate-excluded scoring (§2) and the
     Sr(III)-dropped training;
   - the wildcard-copy-excluded scoring (`wildcard_copies_excluded_scoring`, §2 resolution), which is
     also a registered sensitivity of V1, V5-P and V5-PAIR (resolved by the orchestrator, 2026-09-15,
     extending the §2 resolution to the V5 variants that have crossings: 12 crossings on 11 scored
     entries per V5-P design, 371 on 359 per V5-PAIR design, `folds/INDEX.json`; not score-driven).

   For every other design the censoring-candidate-excluded scoring applies too. Under the V1
   outer-fold unit the pooled remainder fold is also one publication-group cluster. This is gen13
   Addendum 3: score under every hold-out design.

Deterministic arms have one seed value; item 4 is then vacuous and is marked so.

**Equivalence and non-inferiority.**
- Margin ε = 0.05 log D. Below that, the process consequence sits inside the replicate floor, the
  gen18 rationale.
- TOST with a 90 % interval under the primary cluster.
- "No difference" is claimed only when that interval lies inside ±ε. Otherwise the verdict is
  UNDECIDED, never "no effect".

**Multiplicity.**
- Benjamini–Hochberg is applied separately to the registered family (§19) and to the exploratory
  family. Raw and adjusted p are printed side by side.
- A registered contrast is judged by R19. Its adjusted p is shown, not used.

**Signal-injection power check (a null must be informative).**
- Before a failed H1, H1b or H3 contrast is reported as a null, the pipeline is re-run on injected
  targets `y' = y + κ·s`, κ ∈ {0.1, 0.25, 0.5, 1.0} log D.
- For H1: `s_row = u_m v_s`, with u ~ N(0, 1) per metal state and v ~ N(0, 1) per system, seeded and
  standardised. A cross-metal lookup cannot exploit this signal; a factorisation can.
- For H3: every An(III) state shares the u of its nearest-CN8-radius Ln(III).
- X(?) rows are **dropped from the injected refits** of every arm in the contrast (resolved by the
  orchestrator, 2026-09-15; not score-driven). Reason: they have no state-level `u`, giving them none
  would plant an inconsistent signal inside an element, and dropping them from both arms keeps the
  paired comparison fair. The same scored rows as the un-injected run are used.
- The injection is applied to training and test rows alike. Models are refitted at their selected
  hyperparameters without re-tuning. The report gives κ_min, the smallest κ at which R19 passes.
- A null with κ_min > 0.25 log D is reported as UNDECIDED (underpowered).
- Injected values exist only inside this check and are never data (brief §33).

**Reliability before correlation.**
- Scope: any derived per-unit quantity whose correlation or trend is interpreted, namely
  - B7 slopes n and p_eff;
  - factor loadings u_m and v_s;
  - learned embeddings (brief figures 10–11);
  - per-system logSF amplitudes;
  - support-score components.
- Each reports reliability before any correlation is read:
  - split-half by publication group where the unit has ≥ 4 groups: 20 seeded random halves,
    Spearman–Brown corrected;
  - otherwise jackknife-by-publication reliability, between-unit variance / (between-unit variance
    + mean within-unit SE²).
- **Floor 0.3.** A quantity below it may neither support nor close a correlation-based claim. Its
  correlations are printed as UNDECIDED (unreliable).
- Embedding figures carry their Procrustes-aligned bootstrap stability.

## 9. Success thresholds (relative to baseline difficulty)

**Difficulty numbers from the pre-seal Phase C run** (B0–B4 and B7 only, V5-primary selection half
unless stated; none involves a learned model):
- `L5` = macro MAE of the stronger of B3x and B3i;
- `C5` = macro MAE of B0;
- `SF5_FLAT` = FLAT logSF MAE on V5-PAIR;
- `DIR5` = max(HEAVIER, B3x-derived, B3i-derived) direction accuracy on V5-PAIR pairs with |observed logSF| ≥ 0.3;
- `A5` = macro MAE of B6r0, computed after sealing and descriptive only;
- noise floor `N0` = 0.299 × √(2/π) = 0.239 log D, the expected absolute deviation at the pooled
  replicate SD (`feasibility_q13_variance.csv`, replicate component, full key).

**Pre-seal values** (filled 2026-09-15 from `evaluation/preseal/difficulty.json`, SHA-256
`bbdb72acff5be40b6de0bbd06f512719ea0aa7b05bebc1a9184709df9f28f1d2` as recorded in
`manifests/g19_run_preseal.json` → `outputs`; selection half; exact leave-one-cell-out on 105 cells in 18
systems, `V5.n_cells_selection`, `V5.n_systems_selection`). They are quoted here so that sealing freezes
them; the formulas above were fixed before any of them was computed.

| quantity | value | key |
|---|---|---|
| V5 log D comparator (also V5-PAIR and V6, §5 table) | **B3i** (B3x 0.8269, B3i 0.7670) | `V5.lookup_comparator` |
| `L5` | 0.7670 | `V5.L5` |
| `C5` | 1.5262 | `V5.C5` |
| `N0` | 0.2386 | `V5.N0` |
| `δ5` = max(0.05, 0.20 × (L5 − N0)) | 0.1057 (L5 − N0 = 0.5284) | `V5.delta5`, `V5.L5_minus_N0` |
| `SF5_FLAT` | 0.7986 | `V5_PAIR.SF5_FLAT` |
| lookup-derived (B3i) logSF MAE on V5-PAIR | 0.3847 | `V5_PAIR.chosen_lookup_derived_logsf_mae` |
| `DIR5` (reading of §4; descriptive only since the S1(c) redefinition, enters no rule) | 0.9498 (HEAVIER on 242 Ln–Ln cell pairs; B3x 0.8538 and B3i 0.8876 on 280 cell pairs) | `V5_PAIR.DIR5`, `V5_PAIR.DIR5_components` |
| V2 log D comparator | **B3i** (focus-7 selection-half macro MAE 0.6032; B0 1.1012) | `V2.comparator_choice`, `V2.L2_focus7_selection`, `V2.C2_B0_focus7_selection` |
| `A5` | not computed pre-seal | `V5.A5` |

These values are unchanged by the verification corrections of 2026-09-15, which changed only intervals,
labels and printed sensitivities: the `V5`, `V5_PAIR`, `V2` and `V0_diagnostic` blocks of the rerun's
`difficulty.json` are identical to those of the first run (compared with a copy of the first run's file
taken before the rerun; the copy is not kept in the repository).

**S1 — scientific success (H1, V5).** All of:

* **(a) Margin.** M2 vs the V5 lookup comparator (§5) passes R19 with margin
  `δ5 = max(0.05, ρ5 × (L5 − N0))`, where **ρ5 = 0.20** is fixed now: the fraction of the lookup's
  above-noise error the candidate must remove. The same δ5 applies to H1b (B6 vs the lookup).
* **(b) Constant baseline.** M2 also passes R19 against B0 and B6r0 with margin 0.05.
* **(c) Selectivity direction** (brief §22 "substantially above chance"). On V5-PAIR, M2's direction
  accuracy exceeds the yardsticks by ≥ **γ5 = 0.05** (fixed before any baseline score) under the paired
  rule below, and its logSF MAE is below both FLAT and the lookup-derived logSF MAE by ≥ **η5 = 0.02**.
  If V5-PAIR is dropped (§3.1), S1(c) is untestable and S1 is reported as UNDECIDED, never as passed.
  (The earlier wording "exceeds `DIR5`" is superseded by the redefinition below.)
  - *Disclosure (2026-09-15, results seen: yes).* DIR5 was computed under the reading that follows §4:
    each yardstick's cell-pair-macro direction accuracy over the pairs where it is defined, HEAVIER on
    Ln(III)–Ln(III) pairs only, and the max of the three. The value, the alternative with undefined
    HEAVIER pairs counted ½, the Ln–Ln-only alternative and the confirmation-half HEAVIER accuracy have
    all been seen (`evaluation/preseal/difficulty.json` → `V5_PAIR.DIR5`,
    `V5_PAIR.DIR5_alternative_undefined_counts_half`, `V5_PAIR.DIR5_alternative_ln_ln_pairs_only`;
    `tables/preseal_pair_summary.csv`, half `confirmation`, `yardstick_reading` HEAVIER, stratum `all`,
    aggregation `unit_macro`). The registered text is kept; no reading is chosen here by its value.
  - **S1(c) redefined (orchestrator, 2026-09-15; results seen: yes — DIR5 0.9498 / 0.889 / 0.950 in the
    selection half and HEAVIER 0.561 in the confirmation half; no learned-model score existed).** The
    fixed-number reading is incoherent: it compares a confirmation-half M2 accuracy with a selection-half
    DIR5 and maxes over yardsticks defined on different pair sets. Brief §22 prescribes finalising
    thresholds after baseline difficulty and before final training; this is that step. The registered
    direction criterion is a **paired contrast on identical pairs of the half being judged**:
    - for each yardstick Y ∈ {HEAVIER, B3x-derived, B3i-derived}, Δ_Y = direction accuracy of M2 −
      direction accuracy of Y, both computed on the same comparable test–test pairs where Y is defined
      and |observed logSF| ≥ 0.3, cell-pair macro, same half, same fitted folds. **"Same fitted folds"
      (resolved by the orchestrator, 2026-09-15; not score-driven):** the B3x and B3i yardsticks are
      re-fitted on exactly the batched V5-PAIR folds M2 is fitted on (closed-form, cheap), not taken
      from the unbatched pre-seal folds; HEAVIER needs no fit. In discovery these are the seed-104729
      batched folds of §3.1; at confirmation the batched V5-PAIR folds are built by the same colouring
      rule with each withheld seed, and every arm of the contrast is fitted on them;
    - **seed combination at confirmation:** each Δ_Y is the mean over the 5 withheld seeds of the per-seed
      cell-pair-macro Δ_Y; its interval is the percentile interval of a system-cluster bootstrap
      (10,000 resamples, seed 19) of that seed mean, with the same resampled systems applied to every
      seed; in addition Δ_Y > 0 in 5 of 5 withheld seeds (as R19 item 4 at confirmation);
    - **at confirmation** (confirmation half, withheld seeds): min_Y Δ_Y ≥ γ5 = 0.05, and every Δ_Y's
      system-cluster percentile 95 % interval excludes 0;
    - **counterweight, because the confirmation-half HEAVIER value was seen and is low:** on the
      selection half (discovery seed 104729) min_Y Δ_Y ≥ −0.02 (non-inferiority against the yardsticks
      where HEAVIER scored 0.950). S1(c) passes only if both hold.
    - The logSF MAE part is unchanged in form but is also paired: M2's logSF MAE is below FLAT's and below
      the B3i-derived logSF MAE by ≥ η5 = 0.02 on the identical pair set of the confirmation half.
    - `DIR5` stays in the §9 table as a descriptive difficulty number and enters no rule.
* **(d) Calibration** (§12), macro over V5-primary cells:
  - 50 % coverage ∈ [0.40, 0.60];
  - 80 % coverage ∈ [0.70, 0.90];
  - 95 % coverage ∈ [0.88, 0.99];
  - in every domain-status category with ≥ 20 scored cells, 80 % coverage ∈ [0.65, 0.92].
* **(e) Error grows with support distance.** The Spearman ρ between cell MAE and the cell's
  `support_score` (§13) is ≤ −0.10, with the system-cluster bootstrap interval excluding 0.
  The support-score components pass the reliability floor.

**S2 — strong success (H2, V6; evaluated in the single confirmation run).** The formulas are fixed
now. The difficulty numbers are the V6 baselines computed in that same run, so V6 is not touched
earlier.

* **(a) Sign of logSF_Nd/Pr.** The sign of the per-system median predicted logSF equals the sign of
  the observed median in **≥ 11 of the 13 systems** (fixed now). Pair-level direction accuracy
  (|observed| ≥ 0.1) must also beat every yardstick by **+ 0.05** under the paired rule of S1(c):
  for each Y ∈ {HEAVIER, B3x-derived, B3i-derived, B8}, Δ_Y = accuracy(M2) − accuracy(Y) on the
  identical V6 test–test pairs where Y is defined (B8 only in systems with TOPO39 columns), with B3x,
  B3i and B8 fitted on the same V6 folds and withheld seeds as M2 and the same seed combination;
  min_Y Δ_Y ≥ 0.05, in pooled pairs and in the HNO3 pairs (142 of 209). (Resolved by the
  orchestrator, 2026-09-15, replacing "≥ max(HEAVIER, B3x-derived, B3i-derived, B8) + 0.05", which
  maxed over yardsticks defined on different pair sets; no V6 score of any arm exists, so this is not
  score-informed.)
* **(b) Useful magnitude.** logSF MAE ≤ FLAT **− 0.02** (fixed now), with the 13-system percentile
  bootstrap interval of the gain over FLAT excluding 0. It must also be ≤ the lookup-derived logSF
  MAE. The macro log D MAE of the hidden Pr and Nd rows is ≤ the V5 lookup comparator.
* **(c) Registered coverage.** Over the 209 V6 pairs, pooled logSF interval coverage lies in
  [0.70, 0.90] at 80 % and is ≥ 0.88 at 95 %.
* **(d) Stable process recommendation** (only if Phase H runs; §14). The top recipe keeps its
  identity in ≥ 0.80 of the D-draw bootstrap re-rankings.

S1 is the scientific claim of Gen19. S2 is the applied claim. S2 without S1 is reported as
"Pr/Nd reconstructed in the V6 systems; not established as general transfer".

## 10. Failure conditions (brief §22)

Gen19 is **unsuccessful** if any of the following holds. Each is evaluated mechanically and printed
verbatim in `decisions/`.

* **F1 — gains only on random split.**
  - On V0, M2 beats B3 with the 95 % interval excluding 0;
  - yet on V5-primary S1(a) fails, and on V1 the M2 − B3 interval includes 0 or Δ ≤ 0.
* **F2 — strict transfer folds collapse.** Under V5-primary, V1 or V2, the primary-cluster 95 %
  percentile interval of the deployed predictor's gain over B1, or over B2, includes 0 or lies below
  it. The predictor then cannot beat a marginal mean.
* **F3 — uncertainty badly miscalibrated.** For the deployed predictor on V5-primary or V1:
  80 % coverage outside [0.60, 0.95], or 95 % coverage < 0.85.
* **F4 — negative actinide transfer.** The configuration deployed for lanthanide prediction was
  trained with actinide rows, and the WITHOUT arm of §11 beats it on the Ln test set (95 % interval
  excluding 0).
* **F5 — process optimum depends on unsupported predictions** (only if Phase H runs).
  - (i) Any reported recommendation uses an UNSUPPORTED D. This is a code defect and also a failure.
  - (ii) Or, for the TODGA Pr/Nd case, no recipe reaches P(both targets) ≥ 0.5 when predictions
    labelled UNSUPPORTED or CONDITION_EXTRAPOLATION are barred, while one does when they are allowed.
* **F6 — complexity adds no strict-fold benefit.**
  - M2 fails S1(a), and no step M1–M6 is kept by the ladder rule.
  - The deployed predictor is then the best-passing baseline, and the report says the architecture
    claim failed.

A clean null is a result of this generation, not a failure of the programme. It is reported, not
hidden. When H1 fails, gen18's B1 nearest-condition lookup remains the process chain's default D
source.

## 11. Actinide ablation (brief §30, H3)

**Prior.** Gen11 (`runs/gen11_transfer/GEN11_DECISION_REPORT.md`) added archive actinide rows to
lanthanide prediction under leave-chemotype-out and was **INCONCLUSIVE**:

| arm | effect | BCa interval | note |
|---|---|---|---|
| all-actinide | +0.049 | [−0.0005, +0.119] at 5,000 replicates | — |
| non-actinide rows | −0.053 | — | — |
| size-matched comparison | +0.014 | [−0.034, +0.085] | MDE 0.087 |

About 60 % of archive rows share a fingerprint-identical ligand with the lanthanide cohort. The
expectation registered here is therefore "small or none", and power is the first thing reported.

**Corpus support** (`feasibility.json` → `Q12_actinide_lanthanide_overlap`):
- 5,420 actinide MODEL rows (`counts.json` → `rows.by_metal_category_model.actinide`).
- 122 systems have both An and Ln MODEL rows; 110 of them have comparable An/Ln pairs.
- 1,207 comparable row pairs from 20 publications; 802 of them are Am(III)/Eu(III), and only 73
  pair an actinide with Pr or Nd.
- Monoamide is actinide-only in practice (2,144 of its 2,154 rows; `feasibility_families.csv`),
  so WITHOUT also removes a family the lanthanide targets barely have (`FEASIBILITY.md` §30).

**Design.**
- **Arms.**
  - Same architecture: the retained ladder configuration, plus B6 and B5 as transparent references.
  - Same folds and batches, computed once on the full corpus.
  - Same seeds, same Ln test set.
  - **WITH:** all MODEL training rows. **WITHOUT:** every training row with metal category
    actinide removed, unknown-state actinide rows included.
- **Ln test set.**
  - The Ln(III) cells of V5-primary after the V6 carve-out (selection half in discovery);
  - the Ln(III) folds of V2 in the selection half;
  - Ln rows of the V1 selection folds;
  - rows in `V6_TARGET_ROWS` are excluded from all three;
  - V6, at confirmation only.
  - Cells whose eligibility depends on actinide partners are scored in both arms and reported as a
    separate stratum.
- **Controls.**
  - `ACT_PERMUTED`: actinide rows kept, `log_D` permuted within (system, publication group). This
    keeps coverage and destroys chemistry.
  - `ACT_METAL_SHUFFLED`: actinide metal-state labels shuffled among actinide rows within a system.
  - WITH − PERMUTED isolates actinide chemistry from row count.
- **Deltas.** Each is reported WITH − WITHOUT and WITH − PERMUTED, with R19 statistics and TOST
  (ε = 0.05):
  - Δ macro MAE log D;
  - Δ rank accuracy;
  - Δ logSF MAE (V5-PAIR Ln pairs; V6 at confirmation);
  - Δ calibration: |cov80 − 0.80|, |cov95 − 0.95| and CRPS.
- **Verdicts.**
  - *helps*: WITH beats WITHOUT and PERMUTED under R19 on V5 Ln cells and is non-inferior on V1/V2;
  - *hurts*: WITHOUT beats WITH under R19;
  - *equivalent*: TOST inside ±0.05;
  - otherwise UNDECIDED, with κ_min from §8.
  - Actinide rows enter the deployed Ln configuration only on *helps*.
- **Negative-transfer investigation** (run if WITHOUT is better, point estimate or passed):
  - separate deltas for the Am/Eu-heavy diglycolamide cells and for Ln cells whose systems share a
    component with a monoamide-only system;
  - per-family and per-mechanism deltas;
  - per-cell delta against the number of actinide rows in the cell's system, and against the
    system's median |An/Ln logSF| (soft N-donors separate An from Ln strongly);
  - the WITH arm re-run with a shared-only metal embedding (no `e_series`, `e_ox`) against the full
    §15 embedding.

## 12. Uncertainty evaluation (brief §10, H6)

* **Methods compared.** All produce a predictive mean, SD and 50/80/95 % intervals.
  - bootstrap ensembles over publication groups (20 members for B5/B6);
  - deep ensembles (M7, 5 members);
  - heteroscedastic Gaussian head;
  - CatBoost quantile loss;
  - split-conformal and CV+ conformal with publication-grouped calibration folds;
  - Mondrian conformal conditional on domain-status category;
  - conjugate Bayesian ridge head (B6);
  - MC dropout (secondary only).
* **Implementation and fitting.** Conformal and CRPS are implemented by hand. Calibration is fitted
  in inner folds only, never on the outer hold-out.
  - The calibration rows of an inner split are the rows that split would score under §2 (known metal
    state, not Sr(III), not `V6_TARGET_ROWS`, not an acidic co-extractant row), so the calibrated
    population is the scored population; the pre-seal V1 and V0 calibration had first included X(?)
    rows, and was corrected before the intervals were recomputed (`gen19ct/models/interface.py`,
    `CONFORMAL_CHOICES["not_scored_in_calibration"]`; `evaluation/preseal/difficulty.json` →
    `readings.calibration_population`; verification finding VL-03).
* **Metrics.**
  - coverage at 50/80/95 % with the §4 averaging;
  - mean width (sharpness) and CRPS;
  - Spearman of |error| with predicted SD, plus a binned reliability curve (10 equal-count bins);
  - width against transfer distance (`support_score` and each raw §16 support quantity);
  - coverage and width by domain-status category, printed with each category's cell count.
* **Designs.** Reported for V5, V1 and V2; for V6 at confirmation.
* **Pass bands.** S1(d) (§9) and F3 (§10).
* **"Knows when it does not know"** (brief §34.8) is established only if both hold:
  - Spearman(|error|, SD) > 0 with the system-cluster interval excluding 0;
  - coverage in the UNSUPPORTED and FAMILY_EXTRAPOLATION categories is not lower than in IN_DOMAIN
    by more than 0.10.
* **Preferred model.** A model with worse MAE but better calibration may be preferred (brief §10)
  only if its MAE is non-inferior (TOST, ε = 0.05) to the MAE-best model. This choice uses the
  selection half only.

## 13. Domain status and support score (brief §1.2, §16)

**Inputs.** Every prediction gets its labels from `SupportIndex(train_rows).features(metal_state,
system, conditions)`, fitted on the rows of its fold. Two further training-set counts are computed by
the evaluation code: `family_rows` (training rows in systems of the query's family) and
`mechanism_rows` (the same for its mechanism group).

**Thresholds (fitted per outer fold on training rows only, brief §12).**
- For each outer fold, the fold's `SupportIndex` (whose condition standardisation is fitted on that
  fold's training rows) computes the leave-one-row-out `condition_distance_pair` for every training
  row whose (metal state, system) pair has ≥ 2 training rows.
- τ_in, τ_ext and τ_max are the 50th, 95th and 99.5th percentiles of those distances. They are
  target-free, never use a test row, and are on the same standardised scale as the distances they
  bin.
- The fold builder writes the per-fold values beside the fold hash (`folds/<design>.json` →
  `folds[*].support_tau`, computed by `gen19ct.evaluation.support.fold_tau` on MODEL rows minus the
  fold's hidden rows; `folds/INDEX.json` → `readings.support_tau`), and every evaluation run recomputes
  and checks them. Inner folds recompute them on inner training rows. The first pre-seal run had
  computed them only inside `scripts/g19_run_preseal.py` (verification finding VL-06).

**Labels.** First match wins:

| # | status | operational definition |
|---|---|---|
| 1 | `UNSUPPORTED` | `mechanism_rows` = 0 or mechanism UNKNOWN; **or** metal absent from training (`n_systems_for_metal` = 0) with no same-charge metal under the system (`nearest_radius_same_charge` false) and no ±1 series neighbour; **or** system absent (`n_publications_system` = 0), `family_rows` = 0 and metal absent; **or** condition distance to the nearest training row of the system (of the family when the system is absent) > τ_max; **or** the query's acid anion is never seen with the system or family in training |
| 2 | `FAMILY_EXTRAPOLATION` | system absent and `family_rows` = 0 (mechanism present) |
| 3 | `CROSS_METAL_LIGAND_TRANSFER` | `exact_pair_rows` = 0, with system and metal both present in training (the V5 cell); also system and metal both absent with family present (flag `both_nodes_new`) |
| 4 | `CROSS_LIGAND_TRANSFER` | system absent, `family_rows` > 0, metal present |
| 5 | `CROSS_METAL_TRANSFER` | system present, metal absent, with a same-charge or ±1 series neighbour under the system |
| 6 | `CONDITION_EXTRAPOLATION` | `exact_pair_rows` ≥ 1 and `condition_distance_pair` > τ_ext |
| 7 | `IN_DOMAIN` | `exact_pair_rows` ≥ 5 and `condition_distance_pair` ≤ τ_in |
| 8 | `INTERPOLATION` | every other request with `exact_pair_rows` ≥ 1 (including missing condition coordinates, flag `conditions_unknown`) |

Categories 3–5 also carry `condition_extrapolated = condition_distance_system > τ_ext`.

**support_score v1.** Registered now, target-free, not fitted to errors. It is the mean of the
available components, each mapped to [0, 1]:

| component | definition |
|---|---|
| s1 | min(1, log1p(`exact_pair_rows`)/log1p(20)) |
| s2 | min(1, `n_neighbour_metals_same_charge`/5) |
| s3 | exp(−`nearest_radius_distance_A`/0.05), or 0 if there is no neighbour |
| s4 | max over training systems measured with the metal of Tanimoto × exp(−`d_desc`/2) (chain-length-aware, §3.5; 0 if none). Resolved (orchestrator, 2026-09-15; not score-driven): the query's own system counts when it has training rows of the query's metal state (legitimate in-domain support; under V5 the hidden cell has none, so it never counts there); a system with an undefined `d_desc` is skipped (conservative: undefined similarity is no support) |
| s5 | min(1, log1p(`n_same_family_rows_for_metal`)/log1p(100)) |
| s6 | min(1, `n_publications_system`/5) |
| s7 | exp(−CD/τ_ext), where CD = `condition_distance_pair` if the pair exists, else `condition_distance_system`; 0.5 if CD is undefined |
| s8 | 1 if `series_bracketed`; 0.5 if `n_series_neighbours_pm1` ≥ 1; 0 otherwise (Ln and An only; omitted for other categories) |

A learned support score is exploratory. `support_score`, `domain_status`, `nearest_support`
(nearest radius metal and nearest ligand system) and the §1.3 fields (`mean_logD`, `std_logD`,
`lower_95`, `upper_95`) are part of every prediction record passed to Gen18.

## 14. Process evaluation (brief §17–§18, H7; conditional)

**Gate.**
- Phase H runs only if S1 passes for the deployed predictor.
- If S2 fails or V6 has not run, every process result is labelled `transfer-unsupported`, and none
  can be a headline recommendation.
- If S1 fails, no Gen19 process evaluation is run, and gen18's chain keeps its B1 lookup default.

**Case.**
- The TODGA nitrate system: the only §19 candidate with direct Pr and Nd support
  (`feasibility_candidate_extractants.csv`). Only the HNO3 rows and pairs count as evidence for it:
  26 of TODGA's 83 comparable Pr/Nd pairs are HNO3, 54 are HCl and 3 are malonic acid
  (`feasibility_v6_systems.csv`, `comparable_prnd_row_pairs_by_acid`).
- Feed: gen18 `cases/todga_prnd_feed.json`. Specification grid: gen18 `cases/prnd_spec.json`,
  purity ∈ {0.95, 0.97, 0.99} × recovery ∈ {0.80, 0.85, 0.90}. Every value in both files is an
  assumed placeholder and stays flagged.
- PC88A, Cyanex 272 and D2EHPA are not Gen19 predictions (§17). Gen18's literature-parameterised
  case for them is quoted only as literature.

**Monte Carlo.**
- 64 joint draws per operating point, seeded.
- One draw is one ensemble member plus a residual drawn from the calibrated conformal scale.
- The Pr and Nd residuals are drawn jointly, with the correlation estimated from comparable-pair
  residuals in inner folds. No `V6_TARGET_ROWS` row is used for this. Independent draws would
  inflate logSF uncertainty.
- Each draw goes through the gen18 cascade: read-only import (`paths.add_gen18_to_path()`), cascade
  code unmodified.
- The operating-point design space is gen18 `config/design_spaces.json`.
- Outputs per operating point:
  - median and 5/50/95 % purity;
  - median and 5/50/95 % recovery;
  - P(purity ≥ target), P(recovery ≥ target), P(both);
  - P(phase and loading constraints satisfied), taken from gen18's `THIRD_PHASE_RISK` and loading
    flags;
  - reagent use and stage count.

**Objective** (brief §18). A lexicographic ranking:

0. support rank. Recipes using any UNSUPPORTED D are **ineligible**. Recipes using
   CONDITION_EXTRAPOLATION or FAMILY_EXTRAPOLATION D rank below every recipe that uses neither;
1. P(both targets);
2. P(feasible);
3. reagent consumption;
4. stage count;
5. throughput.

**Checks.**
- Stability: the fraction of 20 bootstrap re-rankings of the 64 draws that keep the top recipe
  (identical stage counts ± 1 and O/A within 10 %). This is S2(d).
- Reproducibility test: the same seed gives a byte-identical process table (brief §27).
- The F5 audit (§10) re-ranks with the support gate lifted and reports whether the winner changes.

## 15. Seeds and confirmation

* **Discovery seeds (public).** 104729, 130363, 155921, 196613, 262147 (the programme's split seeds).
  - They drive V0/V1 fold assignment, V5 batching order, inner folds, model initialisation,
    ensembles and conformal splits.
  - Model seed rule: `42 + fold·1009 + 9,999,991`.
  - Bootstrap seed: 19.
  - Deterministic arms (B0–B4, B7) do not depend on seeds; their confirmation values equal their
    discovery values, and this is printed. This holds for their point predictions only: their
    split-conformal intervals draw the inner calibration folds from a seed (as run pre-seal: seed 104729
    for V1, V2 and V5, the fold's own seed for V0; `evaluation/preseal/difficulty.json` →
    `readings.conformal_seed`). **Resolved (orchestrator, 2026-09-15; not score-driven):** in discovery
    the deterministic arms' conformal inner folds are drawn with each of the 5 discovery seeds and the
    coverage and width metrics are the mean over those seeds (seed 104729 alone where a learned arm's
    comparison is on seed 104729 only, §7 compute plan); at confirmation they are drawn with each withheld
    seed and averaged the same way. Point predictions are unaffected.
* **Confirmation seeds (withheld).**
  - Five seeds in [100000, 999999], excluding the discovery seeds, drawn with Python `secrets` by
    `scripts/g19_seal_prereg.py --commit-seeds --seed-store <path outside the repository>`.
  - The store holds the seeds and a random 256-bit salt. It lives in the session scratchpad and is
    never committed.
  - Only `sha256(canonical JSON{n, salt, schema, seeds})` is written, to
    `manifests/confirmation_seeds_sha256.txt`.
  - Commitment: `65e8ae8ceb8e92947a1da27f42e2689e74b04cc7b820d2b6df26889ceabf5f82` (`manifests/confirmation_seeds_sha256.txt`, written by `--commit-seeds` on 2026-09-15).
  - Unlike gen16's public rule, these seeds cannot be recomputed from the text. No discovery result
    can have been computed on them.
  - They are revealed in `decisions/CONFIRMATION.md` together with the `--verify-seeds` verdict.
* **Confirmation.**
  - At the end of discovery, at most **5 claims** are frozen in `decisions/CONFIRMATION_PLAN.md`.
    Each is a registered contrast from §19 that passed R19 in discovery; the plan gives the arm,
    comparator, metric and designs.
  - One run on the withheld seeds scores those claims on the **confirmation half** of each design
    (`feasibility_halves.csv`) and runs **V6 once**, including the S2 evaluation and the §11 V6 deltas.
  - The confirmation half contributed to no ladder decision, claim or preferred-model choice, so
    this run removes both seed luck and selection optimism.
  - A claim is **confirmed** only if it passes R19 with 5 of 5 confirmation seeds.
  - Whatever confirmation returns is the result. No sixth claim is added and nothing is re-run.

## 16. Experiment tracking (brief §23)

Every run is wrapped in `gen19ct.manifest.Run`.

**Deterministic manifest** (`manifests/<name>.json`):
- git commit;
- dataset hash (archive SHA-256, plus the bundle prefix where it is read);
- **fold hash**: SHA-256 of the LF-written fold/batch assignment CSV, sorted by
  `canonical_measurement_id`;
- **feature version**: SHA-256 of `descriptors/*.csv` and of the feature-code files;
- **model version**: class name plus SHA-256 of the model code file;
- random seeds;
- hyperparameters, including the inner-selected values per outer fit;
- metrics files, with binary and LF-normalised digests of every input and output.

**Volatile manifest** (`manifests/run_info/<name>.json`): training date, hardware, package versions,
runtime.

Metrics are written as CSV/JSON through `write_csv`/`write_json` before any markdown. Every number
quoted in a decision file must exist in such a file.

**Decision files** (brief §29):

| file | contents |
|---|---|
| `D01_transfer_signal.md` | Phase C, V1/V2/V5 baselines |
| `D02_factorization.md` | H1, H4, ladder |
| `D03_actinide_transfer.md` | H3 |
| `D04_mechanism_experts.md` | H5 |
| `D05_uncertainty.md` | H6 |
| `D06_process_integration.md` | H7 |

Each contains: question, evidence, metrics, verdict (null / supported / ambiguous), decision, next
action.

## 17. Deviations from the brief forced by the data

1. **PC88A/P507, Cyanex 272, D2EHPA/P204 absent.** None is present as a structure (exact or
   connectivity match), and no phosphonic- or phosphinic-acid structure exists at all
   (`named_extractant_presence.csv`; `feasibility.json` → `Q5_Q8_named_extractants`).
   - The only phosphoric acid is dihexyl hydrogen phosphate: 10 MODEL rows, one publication,
     Tanimoto 0.47 to D2EHPA (`feasibility_named_analogues.csv`).
   - "HDEHP" occurs only as a name in the phase-modifier slot (14 MODEL rows, Eu/Am). Chemically it is
     an acidic co-extractant there, and those rows are handled as synergistic chemistry (§2).
   - Three paper titles name bis(2-ethylhexyl)phosphoric acid (48 rows, 39 MODEL), but their rows record
     DOHyA, TEHDGA, HEDTA or CDTA as the extractant (`named_extractant_presence.csv`,
     `D2EHPA:FULL_NAME_OR_TITLE_TEXT`).
   - "DEHPA" in the archive is a monoamide (25 MODEL rows): a name trap.
   - Consequences: the brief's examples PC88A × Nd (V5), PC88A (V3) and phosphinic acids (V4)
     cannot be run. The §19 comparison reduces to TODGA (`DIRECT_PR_AND_ND`). PC88A and Cyanex 272
     are `UNSUPPORTED_NO_DIRECT_NO_FAMILY`; D2EHPA is `SAME_FAMILY_ONLY` and is treated as
     UNSUPPORTED for process ranking (`feasibility_candidate_extractants.csv`).
2. **No pH, saponification, organic-loading, reported-uncertainty, extractant-family or
   "D reported vs reconstructed" column** (`metadata_availability.csv`).
   - The acidic mass-action term n_H·pH (§6) is not identifiable, and neither is any acidic-slope
     constraint.
   - Saponification is not modelled.
   - Loading enters only as the Gen19-derived loading bound, and gen18's depletion correction is
     exploratory.
   - Noise is anchored to the replicate SD 0.299 rather than to reported uncertainties.
   - The reconstructed-D heuristic flag (INFERRED) is a sensitivity only.
   - Extractant family is the Gen19 structural classification.
3. **Mechanism experts (§5).**

   | expert | MODEL rows | systems | note |
   |---|---|---|---|
   | E2 neutral solvating (incl. soft N-donor, mixed neutral) | 12,273 | 243 | — |
   | E1 acidic cation exchange | 14 | 3 | dihexyl hydrogen phosphate, a carbamoylbenzoic acid, a dithiophosphinic acid |
   | E3 ion-pair/basic | 0 | — | — |
   | E4 chelating | 3 | 1 | quercetin |
   | E5 synergistic | 92 | 9 | all Eu/Am, one publication |
   | UNKNOWN | 29 | 3 | HEDTA, CDTA, TDGA: an aqueous agent recorded as the only extractant |

   Source: `feasibility_mechanisms.csv`. The five-expert mixture is not testable; M3 routes among
   sub-labels of E2 and pools the rest.
4. **V3 and V4 are narrow.** V3 covers only 30 diglycolamide/monoamide systems; V4 only four
   neutral or soft-N families once name–structure-conflict systems are excluded (malonamide, at 195
   rows, is a sensitivity).
5. **V5 is neutral-solvating chemistry**, dominated by diglycolamides: 100 of 224 primary cells are
   non-DGA. V6 has 13 systems, only 4 of them with ≥ 2 common publications, so it supports only a
   weak 13-cluster percentile bootstrap (§8). Its decisions rest on sign counts and margins (§9).
6. **Publication effects cannot be estimated directly.** Only 2 exact cross-publication condition
   matches exist (`feasibility.json` → `Q13_metal_vs_conditions`), so M4 is judged only by
   strict-fold error, never by its offsets. Metal effect within (publication, system, condition):
   pooled SD 0.658 over 1,601 groups, against a replicate SD of 0.299.
7. **Publication ids.** Folds use `group_cross_publication_copy` rather than the shared loader's
   `g19_publication_id`, which splits a DOI across ids because it is built from uncorrected DOIs
   (§2; reported to the orchestrator).
8. **Metal descriptors.** Polarizability is NA everywhere (source inaccessible). Pauling
   electronegativity is NA for Pm, Eu, Tb and Yb, and HSAB classes are partly INFERRED
   (`descriptors/metals_sources.md`). Redox descriptors were not built. No such value is imputed
   from memory.
9. **Loading correction (§6 "retain Gen18's loading correction").** It is not a registered model
   component: organic loading is not recorded. It is re-evaluated only in exploratory arms where
   metal-concentration series exist.
10. **Code layout.** The package is `gen19ct/` (repository convention) instead of the brief's `src/`.
    The §26 module layout maps onto `gen19ct/{data,chemistry,folds,models,losses,evaluation,process}`.
11. **Preregistration file.** It is drafted as `preregistration_draft.md` and sealed into
    `preregistration.md` (brief §20/§21) by the script, never by hand.
12. **Acid media are pooled inside some V5 cells** (38 primary cells). Cells are hidden whole. Scores
    are also reported per medium, and the HNO3-only cells are a registered sensitivity (brief §4.3).
13. **Aqueous agents recorded as the only extractant** (HEDTA, CDTA, thiodiglycolic acid; 29 rows)
    and **name–structure conflicts** (Br-Cosan, TPDGA malonamide, TDGA, TBADIPIC, NDDIPIC) are kept
    in training with their flags and are never V3/V4 units. Their mechanism labels differ
    (`descriptors/extractant_systems.csv`, `mechanism`, `n_rows_MODEL`):
    - HEDTA (8 rows), CDTA (8) and TDGA (13) are UNKNOWN and belong to no expert;
    - the two Br-Cosan mixtures (16 + 14 rows) are SYNERGISTIC through the Br-Cosan name override (E5);
    - TPDGA (5 rows) is NEUTRAL_SOLVATING, and TBADIPIC and NDDIPIC (2 + 2 rows) are SOFT_N_DONOR, so
      these three sit inside the E2 expert and its M3 sub-experts.
14. **Censoring.** No detection-limit or saturation flag exists in the archive. The exact-decade
    floor/ceiling rule of §2 is a Gen19 heuristic (INFERRED) with a registered scoring sensitivity.
15. **Selection / confirmation halves** replace a purely seed-based confirmation. Every discovery
    decision and claim uses the selection half, so confirmed claims rest on about half of each
    design's scored units (§3, §15).

## 18. What is not claimed and will not be done

**Not claimed:**
- no random-split claim;
- no statement about PC88A, Cyanex 272, D2EHPA or any phosphorus acid from Gen19 predictions;
- no zero-shot claim for a family outside the four V4 families (malonamide only as a sensitivity);
- no process recommendation from UNSUPPORTED chemistry;
- no validation of the gen18 literature case;
- no interpretation of an embedding or slope below the reliability floor.

**Will not be done:**
- no Gen15 prediction used as data;
- no tuning or selection on V6, and no second V6 run;
- no `V6_TARGET_ROWS` row scored before the confirmation run, and no choice made on the confirmation half;
- no cell, system, metal, threshold, margin, metric, seed or baseline changed after an outer result
  has been seen, except by a POST-HOC addendum;
- no merge into `main` before review;
- no edit to gen18, the archive or the bundle.

## 19. Comparison accounting

**Registered family.**

| group | contrasts |
|---|---|
| primary | M2 vs the V5 lookup comparator, the stronger of B3x and B3i (V5 log D MAE) |
| H1b | B6 vs the V5 lookup comparator (V5) |
| S1(b) | M2 vs B0; M2 vs B6r0 (V5) |
| S1(c) | paired direction contrasts Δ_Y = M2 − Y for Y ∈ {HEAVIER, B3x-derived, B3i-derived}, judged on min_Y Δ_Y (§9 S1(c) as redefined); M2 vs FLAT and vs the B3i-derived logSF MAE on the identical pair set (V5-PAIR) |
| H4 | M2 − FLAT_CAT; M2 − M0; M3 − M2; B6 − B6r0 (V5) |
| H5 | M3 vs M2; M4 on/off; M6a on/off; M6b on/off (V1, V2, V5) |
| H3 | WITH − WITHOUT; WITH − ACT_PERMUTED (V5 Ln, V2 Ln, V1 Ln) |
| ladder | each kept or removed step vs its predecessor (V5, V1, V2; selection halves) |
| secondary designs | M2 vs the design comparator on V1, V2, V3, V4, V7 |
| confirmation | S2(a)–(c) (V6) |

**Exploratory.** Everything else, with every evaluated arm written to
`evaluation/contrasts_*.csv` with a `family` column. `decisions/` prints arms and contrasts per
family, with BH-adjusted p beside raw p. A contrast added later enters the exploratory family with a
POST-HOC addendum.

## 20. Pre-seal checklist (every open item above)

- [x] §0 banner replaced by the sealed-status line.
- [x] §3.1 number of V5 batches per discovery seed; V5-P eligible count after masking;
      V5-PAIR eligible cell-pair counts by category class.
- [x] §9 ρ5 = 0.20, γ5 = 0.05, η5 = 0.02 (S1); S2 sign count 11 of 13, direction margin 0.05, magnitude
      margin 0.02 — fixed numerically in this draft, before any baseline outer score.
- [x] §13 τ_in, τ_ext, τ_max — a per-fold rule on training rows, so no pre-seal number is needed.
- [x] The pre-seal Phase C run used only B0–B4 (with B3x, B3i, B3l, B4x, B4l), B7, the pair yardsticks
      FLAT and HEAVIER, and split-conformal intervals on them (§0 item 1 as amended); its manifest lists
      no B5, B6, B8 or M output (`manifests/g19_run_preseal.json` → `arms`, `pair_yardsticks`,
      `learned_arms_fitted_or_scored` = `[]`; `evaluation/preseal/run_checks.json` →
      `forbidden_arm_names_found` = `{}`, `v6_target_rows_scored` = 0; checked 2026-09-15 on the rerun after
      the verification corrections; re-check if the run is repeated before `--seal`).
- [x] Every placeholder marker added on 2026-09-15 for the verification findings (X(?) scoring,
      the wildcard copies, the inner guard mode, V5-PAIR / V5-P inner and batched folds, the V1 scoring unit,
      the V0 unit, the B3 tie rule, the M-model settings, the compute plan, R19 item 3, the power check on
      X(?) rows, S1(c), s4, the interval seeds) resolved by the orchestrator, each resolution stating
      whether outcomes had been seen (2026-09-15; only baseline outcomes existed).
- [x] The code consequences of those resolutions are implemented and re-verified before `--seal`:
      batched V5-P / V5-PAIR heavy-arm folds; the V1 outer-fold scoring unit and the wildcard-copy
      sensitivity in the scorer; the V0 F1 interval; the s4 reading; multi-seed conformal intervals of the
      deterministic arms; the paired S1(c) evaluator; the nested-certificate safeguard sample; dropping X(?)
      rows in injected refits; R19 item 3 under every registered cluster unit (`transfer.r19`); the §7
      compute-plan item-6 re-colouring cap of 4 cells per batch (`cell_holdout` `max_cells_per_batch`).
- [x] §15 confirmation-seed commitment digest.
- [x] `DATA_AUDIT.md` and `FEASIBILITY.md` present; `g19_seal_prereg.py --check` reports no
      blocker other than this box, the draft banner and the absent sealed file; then `--seal`.

---
Sealed SHA-256 of everything above this line: `135842499a86eb3d478ece01a45718ac5b9673a4134bb4acda550f975bb45641`

## POST-HOC addendum 1 (2026-09-15, orchestrator Claude Code with the user's approval; results seen: no)

**Results seen:** no learned-model outcome of any kind exists. No B5, B6, B6r0, B8, FLAT_CAT or M model
has been fitted on an outer fold and scored; nothing has run on V6. The only outcomes that exist are the
pre-seal baseline outcomes already disclosed above the footer, and fit-only timings on training rows
(`evaluation/discovery/benchmark/cost_estimate.json`, `measurements_*.json`; no test row predicted).

**Why.** The benchmark prices the sealed §7 compute plan at 1,535.7 CPU-hours, 767.9 h of wall clock on
the registered 2 workers, against the registered 60 h budget (`cost_estimate.md`). Even the H1 path
(B6 plus the seed-104729 main designs of B5, FLAT_CAT, M1 and M2) is 124.6 h of wall clock. The cost is
almost entirely V5 inner tuning: each outer fold re-tunes over about 21 batched inner splits × 6–12
configurations. §7 item 5's demotion order cannot help, because H1, H1b and H4 are never demoted. The
plan is therefore infeasible on the registered machine. The user chose a reduced plan over a
~32-day run or other hardware. Every change below is compute-driven.

**What changes (discovery only unless stated).**
1. **V5 inner tuning (every learned arm and B6, every V5 design and variant).** Each of the registered
   inner folds of §7 is ONE inner fit per configuration. All inner hidden cells of that inner fold
   (≤ 30) are hidden simultaneously, each with its registered hiding. Eligibility is re-checked with all
   of them hidden, and a cell that fails stays hidden but is dropped from the inner score. The inner
   batching of §3.1/§7 is not used for tuning. V1 and V2 inner designs are unchanged, since they were
   already one fit per inner fold.
2. **Inner folds per seed.** Three inner folds on every seed that is run, with no "first inner fold only"
   reading. Configuration selection is the mean over the three inner folds of the design's unit-macro
   MAE, with §7's 0.005 tie rule. Conformal residuals of inner fold j come from the configuration
   selected without fold j, reusing the fits already made (cross-fitted; no extra fits). Outer refit
   iterations or epochs are the median best count over the three inner folds.
3. **Seeds in discovery.** Learned arms (B5 = M0, FLAT_CAT, B6, B6r0, B8, M1–M7) run discovery seed
   **104729 only**. R19 item 4 is NOT_EVALUATED in discovery, and every discovery table says so. At
   confirmation R19 item 4 (5 of 5 withheld seeds) is unchanged and required, and each withheld seed uses
   the tuning of items 1–2. The deterministic comparators' conformal intervals use seed 104729 in
   discovery, since the comparisons are on that seed only (§15 resolution, parenthesis).
4. **R19 item 6 refit sensitivities for learned arms.** Only the V5 strict and V5 HNO3-only refits are
   run, on seed 104729, for H1 (M2), H1b (B6), H4 (M0, FLAT_CAT, M2, B6, B6r0) and any freezing
   candidate. For learned arms the loose, cell-only, parent-structure and Sr(III)-dropped V5 refits, the
   heavy-arm V5-P runs, and the V1/V2 refit sensitivities (state-level V2, Sr(III)-dropped,
   near-duplicate-key and compilation-DOI groupings) are **not run**. R19 item 6 for a learned-arm
   contrast is evaluated over the run refits plus every registered scoring-filter sensitivity (non-DGA,
   censoring-candidate, acid-grid and wildcard-copy exclusions), labelled "reduced sensitivity set
   (addendum 1)". Such a contrast may still be frozen, and the report states which sensitivities were not
   run. The closed-form arms (B0–B4 variants, B7) keep the full registered sensitivity set.
5. **V5-PAIR.** Learned arms run on the seed-104729 batched V5-PAIR folds only for M2 and for B3x/B3i
   re-fitted on the same folds (the S1(c) selection-half counterweight), and for any freezing candidate
   with a pair endpoint.
6. **Readings of the sealed text that the built runner implements** (verification of the runner,
   2026-09-15; recorded here so that nothing is silent):
   - (a) *Order.* The §7 item 3 arm order applies within each pass. The passes are: B6/B6r0; the
     seed-104729 main designs; the seed-104729 refit sensitivities. Exhausting item 5's budget demotes
     only M3–M7.
   - (b) *Model seed rule (§15).* "fold" is the position of the (discovery seed, outer fold) pair in the
     design enumerated in registered order. This holds for every learned arm, B6 initialisation included.
   - (c) *M-model target standardisation.* Inner fits standardise the target on their inner-training
     rows; the outer refit uses the outer-training rows. No row outside a fit's training set is used.
   - (d) *M2 builds on M1 per outer fold.* M2 keeps that outer fold's retained M1 hyperparameters,
     including in its cross-fitted calibration.
   - (e) *Uncertainty metrics needing a predictive SD.* CRPS, Spearman(|error|, SD) and the "knows when it
     does not know" test are NOT_RUN for arms whose only uncertainty is conformal (every arm before M7),
     and are computed from M7's heteroscedastic ensemble.
   - (f) *Multiplicity.* BH-adjusted p is also printed over all 60 registered discovery contrasts of §19,
     with contrasts not run entered as p = 1.
   - (g) *Feature readings* (`gen19ct/models/features.py` → `REGISTRATION_CHOICES`):
     - scaling and imputation statistics are row-weighted, while the fingerprint PCA and d_desc weight
       each training system once;
     - the PCA and donor counts use the primary extractant;
     - X(?) rows carry element-level metal values;
     - the complexant block is the aqueous-complexant role only;
     - B8's lanthanide index is NaN for non-lanthanides;
     - TOPO39 comes from the frozen coordination table only (198 of 259 systems).
7. **Budget and demotion.** The 60 h budget and the demotion order M7 → M6 → M5 → M4 → M3 are unchanged.
   The expected wall clock of items 1–5 for B5, FLAT_CAT, B6/B6r0, B8, M1, M2, the S1(c) V5-PAIR run and
   the strict/HNO3 refits is about 25–35 h on 2 workers (INFERRED from the fit-only timings). The runner's
   re-priced estimate is written to `evaluation/discovery/benchmark/cost_estimate_addendum1.json` before
   launch.

**What does not change.** Designs, folds, halves, hiding and guards; the V6 carve-out and single V6 run;
metrics and averaging units; comparators; margins ρ5, γ5, η5 and ε; S1, S2 and F1–F6; R19 items 1–3 and 5;
the stop rule; the grids of §5 and §6; confirmation on the withheld seeds.

## POST-HOC addendum 2 (2026-09-19, orchestrator Claude Code; results seen: yes - the B6 fold-scheme check verdicts and the B6 values entering them, no contrast)

**Results seen.** No registered contrast has been scored: `scripts/g19_score_discovery.py` has not run, no arm has been compared with a comparator on any design, and nothing has run on V6. Predictions of B6, B6r0, B5, FLAT_CAT, B8, M1 or M2 may exist on disk from the running discovery; none has been read. What has been seen: (i) the pre-seal baseline outcomes disclosed above the footer; (ii) the verdicts of the registered B6 fold-scheme checks, without which the plan cannot proceed — the §3.1 / §7 item 6 batched-vs-exact check on seed 104729 FAILED (|Δ macro MAE| ≥ 0.01; `evaluation/discovery/decisions/b6_checks.json`, state `recolour`), the heavy-arm batches were re-coloured with at most 4 cells per batch on the same seeds and vertex-order rule (`scripts/g19_build_folds_max4.py`, scheme `batched_max4`), and the check was repeated once. Verdicts, from `b6_checks.json`: the repeated V5 check PASSED (`passed_after_recolour`), so the heavy-arm V5 scheme is `batched_max4` and S1 remains attainable; the §3.2 V1 ten-fold check FAILED, so the heavy arms run V1 on the exact design (`PlanState.heavy_v1_scheme` = `exact`; the 39 selection-half folds with scored rows). These verdicts are differences of two B6 macro MAEs; no value is quoted here, and no B6 macro MAE has been compared with any comparator. The orchestrator read the B6 macro MAE values that enter the checks (exact, batched and re-coloured on V5; exact and ten-fold on V1) together with the verdicts; they are B6-only values, no comparator value was read beside them, and no contrast was formed. The heading says "yes" for that reason; addendum 1's "no" predates the checks.

**Why.** The post-discovery runners and the addendum-1 implementation record, in the `REGISTRATION_CHOICES` / `READINGS` dictionaries written with every output, the reading they implement where the sealed text is silent, and several are flagged "needs a POST-HOC addendum" (`report.readings_needing_addenda` lists them). This addendum states them so that nothing is silent, and separates five items that do not fill a gap but narrow or contradict a sealed sentence or addendum 1. Every item was fixed in code before any registered contrast was scored (commits 358c802 of 2026-09-16, 3343d3f of 2026-09-18, daa0469 of 2026-09-19); the daa0469 readings (ladder, H3, power, process, figures, report) postdate the first B6 check verdict, which enters none of them. Each "Resolved:" sentence marks a choice taken here by the orchestrator.

**What changes** (each compute- or implementation-driven; none score-driven).

1. **Cross-fitted calibration refits** (addendum 1 item 2 said "reusing the fits already made (cross-fitted; no extra fits)"; `discovery.READINGS['calibration_refits']`, `boosted.REGISTRATION_CHOICES['conformal']`, `neural.REGISTRATION_CHOICES['cross_fit']`). For B6 / B6r0 the sentence holds (`B6TunedConformal.fit_from_tuning`: every configuration is fitted on every split, so the fits are reused). For B5, FLAT_CAT, M1 and M2 the inner fit of fold j chose its tree count or epoch by early stopping on fold j's own validation rows, so its residuals on fold j are selection-touched; the runner therefore REFITS, on fold j's inner training rows and for a fixed count, the configuration and count selected without fold j (`CrossFitResidualConformal`; three refits per outer fold, no early stopping). B8 has no selection step: its level is refitted once per inner split and the three splits' absolute residuals are pooled. This is the statistically conservative reading of item 2's purpose (no calibration residual from a fit that saw its row) and a change to its literal "no extra fits" (priced in the addendum-1 estimate as one calibration fit per split). Fixed at 3343d3f.

2. **A failed B6 batched-vs-exact check: §7 item 6 governs, not §3.1.** The sealed text holds two rules for the same event: §3.1 fold construction ("Otherwise every arm uses exact leave-one-cell-out") and §7 item 6 (re-colour with at most 4 cells per batch, same seeds, repeat once; if it still fails, label every heavy-arm V5 result "batched (check failed)" and report S1 UNDECIDED, never passed). The code implements §7 item 6 — the later, specific compute-plan rule of 2026-09-15 — and reports §3.1's sentence beside it (`factorized.REGISTRATION_CHOICES['batched_check']`, fixed at 358c802 before any check ran; `discovery.PlanState`: `recolour` → `passed_after_recolour` | `failed`, both giving the heavy V5 scheme `batched_max4`; `discovery.READINGS['s1_check_failed']`). Now that the first check has failed, this resolution decides how the heavy arms run, so it is recorded as a change to §3.1's sentence rather than a reading. The exact design for B5, B8 and M0–M7 would be one outer fit per scored cell per configuration (105 selection-half cells against 22 seed-104729 batches, §3.1), which the addendum-1 pricing already found infeasible for batched folds.

3. **§12 uncertainty methods: only what is built is compared.** §12 lists eight methods. Built: cross-fitted split-conformal intervals for every arm (§12 as read by addendum 1 item 2); for M7 the 5-member deep ensemble with publication-group bootstrap, the heteroscedastic Gaussian head and normalised-by-SD split conformal (`models.ladder`). NOT built: bootstrap ensembles over publication groups for B5 / B6 (20 members), CatBoost quantile loss, CV+ conformal, Mondrian conformal by domain status, the conjugate Bayesian ridge head of B6 (`factorized.REGISTRATION_CHOICES['variance']`: std_logD is NaN), MC dropout (`discovery.UNCERTAINTY_NOT_RUN`). Consequently CRPS, Spearman(|error|, SD), the SD-binned reliability curve and the "knows when it does not know" test are NOT_RUN for every arm before M7 and computed from M7 alone (addendum 1 reading 6(e)); H6 / D05 rests on M7 and on the conformal coverage of the other arms. A narrowing of §12's "methods compared", implementation-driven.

4. **§11 shared-only-embedding re-run deferred.** The last bullet of §11's negative-transfer investigation (the WITH arm re-run with a shared-only metal embedding, no `e_series`, `e_ox`) is not implemented: `neural.FactorisedNet` has no switch for the offsets (`h3.READINGS['shared_only_embedding']`). It is conditional on WITHOUT being better. Resolved: it is implemented in a NEW module (`gen19ct/models/shared_only.py`, a subclass of `neural.FactorisedNet` with the `e_series` and `e_ox` offsets held at zero; wired into the H3 module and runner, which lie outside the discovery run's import closure; no file of that closure was edited) before `scripts/g19_run_h3.py` runs, and executes only under §11's condition that WITHOUT beats WITH; otherwise D03 reports it as not run.

5. **Gate constants and record verification** (`discovery.READINGS['addenda_digest_gate']`). Every runner refuses unless the below-footer text holds exactly `N_ADDENDA_EXPECTED` addenda and digests to `REGISTERED_ADDENDA_SHA256`, and every discovery fold record's digest carries that constant. Appending this addendum requires updating both constants; the discovery records were written under addendum 1's digest and are valid under it. Resolved: a digest registry (`manifests/digest_registry.json`) records, per stage, the below-footer digest and the code digest under which that stage's records were written (discovery: the addendum-1 digest and the discovery code digest); a record is verified against the registry entry of its stage, never against the live text or live code; no record is rewritten; the report prints every registry entry. Any later edit to an existing module is logged in the registry with its commit and reason and can neither validate nor invalidate a record written earlier. The discovery runner has one registered pass after the scorer (the freezing-candidate refits and pair runs of §3.1 / §7, stage `09_candidates`); it is registered as its own stage `discovery_candidates`, its records are written and verified under that stage's digests, and no record of the earlier discovery stages is refitted. The gate constants are replaced by the registry when this addendum is appended - a code change made after the discovery run has completed and before any post-discovery runner starts.

**Readings** (the sealed text is silent; each was fixed in code before any registered contrast was scored).

**1. Inner design and tuning** (addendum 1 items 1–2; §3.1, §7, §12, §15; `models.inner_design`, `boosted`, `factorized`, `neural`, `previous_gen`, `features`, `evaluation.discovery`).
- `recheck` / `inner_recheck`: "re-checked with all of them hidden" counts every condition of a cell, own row and publication counts (k, p) included, with the fold's other cells hidden and nothing restored — the reading §3.1 registered for the identical sentence of the batch re-check; a cell emptied by a same-fold cell's component-aware hiding stays hidden but unscored (`InnerSplit.meta['dropped_cells']`). `inner_cells`, `hiding`: the inner cells and folds are the §7 functions (streams [seed, 15] and [seed, 16, k]); only the hiding changed, to cumulative. `certificate`: the nested certificate is the whole table with every cell of the fold hidden, so `nested_certificate` stays valid but is per fold. `empty_fold`: an inner fold without a scorable surviving cell raises (three inner folds on every seed). `calibration_rows` / `validation_population`: inner validation and calibration rows are the §2 scored population (known state, not Sr(III), not `V6_TARGET_ROWS`, not an acidic co-extractant row). `inner_guard`: every inner split passes `isolation_check` before any fit.
- `inner_wildcard_copies`: a §2 wildcard copy can sit in an inner split's training rows while its partner is a calibration row; the registration is silent on inner validation, so the selection score is unchanged and the count is recorded per fold. Resolved: no - the registered filter is an outer scoring sensitivity; inner selection is unchanged and the per-fold count is diagnostic (conservative for the registration: selection scores are exactly as registered).
- Units and ties (`b6_inner_units`, `inner_units`, `v1_inner_unit`, `inner_macro_mae`, `validation_unit`): every learned arm and B6 selects on the design's §4 unit of the inner rows (V5 hidden cell; V1 publication group, or REMAINDER below 20 outer-training rows; V2 metal state), the mean over the three inner folds. Within 0.005: CatBoost lower depth then larger `l2_leaf_reg` (`tie_rule_order`); B6 smaller rank then larger λ (`tie_rule`); M1 / M2 smaller rank, fewer parameters, larger weight decay (`tie_order`); ladder stronger penalty, fewer parameters, grid position.
- Stopping and refits: CatBoost early-stops on row-pooled MAE while selection uses the unit-macro MAE (`early_stopping_metric`); the outer refit uses the median kept tree count or best epoch over the inner fits, rounded half up (`outer_iterations`, `refit_epochs`); neural early stopping keeps the first best epoch on ties (`early_stopping`). `heavy_arm_intervals`: M2's cross-fitted calibration keeps the outer fold's retained M1 values, which were selected with every inner fold — reading 6(d), a second-order reuse recorded as such.
- B6 (`factorized`): `side_information` (the numeric B5 blocks only), `x_cond`, `statistics_unit` (re-fitted for every fit, inner splits included), `row_level_z_s`, `unknown_state_rows` (X(?) rows are their own `element(?)` unit, never scored), `unseen_tokens`, `penalty` (one λ, μ unpenalised, unscaled residual sum), `convergence` (50 sweeps, flag `b6_converged`), `initialisation` (cold start, `INIT_SCALE` chosen on synthetic data), `b6r0_tuning`, `fold_subset` (diagnostic only).
- M1 / M2 (`neural`): `inputs`, `linear_maps`, `e_series` (Ln / An / other; brief §15's five categories are not an offset), `unknown_ox_offset`, `offset_init`, `weight_decay_scope`, `warmup`, `batches`, `loss_weights`, `system_embedding_table`, `m2_rank` (rank(W) ≤ min(rank, emb_dim)); `target_scaling` is reading 6(c), `model_seed` reading 6(b), `cross_fit` change 1. B5 / FLAT_CAT (`boosted`): `no_sample_weights`, `catboost_defaults`, `target_hidden_from_features`.
- B8 (`previous_gen`): `clip_range` (the gen6 rule: half the training span beyond either end of the range), `imputation`, `design_column_order`, `massaction_duplicates`, `threads`, `row_order`, `variance_model`, `cell_key`, `direction_unit`, `direction_weights`, `direction_training_features`, `pm_excluded`, `magnitude_pairs`, `magnitude_fallback`, `direction_undefined`.
- Features beyond reading 6(g) (`features`): `pca_components_short`, `donor_counts`, `share_weights`, `condition_raw_scale`, `modifier`, `categorical_native`, `flat_cat_unknown_state`, `b8_massaction`, `b8_fp_missing`, `b6_preset`, `ecfp_cluster`, `system_static_source`.
- Scorer (`discovery`): `selection_half_only` — every scored row's half is re-derived; V0 is not run in discovery and F1 is evaluated at the report stage. `b6_batched_check`: identical scored cells, both sides tuned on the simultaneous inner design, so it compares outer batching alone; `v1_tenfold_check`: identical scored rows, V1 outer-fold unit, both sides on the unchanged V1 inner design. `margins`: δ5 on V5; 0.05 for S1(b) and for V1 / V2 (§9 names none; the floor of the δ5 formula). `decision_scopes`: stop rule = R19 items 1, 2, 3, 5; ladder and freezing screen add the scoring-filter sensitivities. `r19_item4_availability`, `refit_sensitivities_not_run`: a missing seed or refit is NOT_EVALUATED / UNTESTABLE, so a learned-arm full verdict is at best UNDECIDED in discovery. `wildcard_filter_pairing`: in a paired contrast a row dropped for either arm is dropped for both. `bh_p`, `full_family_60`: BH per family on the primary-cluster p; reading 6(f)'s 60 = the 57 discovery contrasts of §19 plus the 3 confirmation-only S2 contrasts entered as p = 1, with m = 57 printed beside. Resolved: confirmed - reading 6(f)'s 60 is the 57 discovery contrasts plus the 3 confirmation-only S2 contrasts entered as p = 1. `v2_summary`: V2 contrasts are judged on the §3.3 focus-7 states present in the selection half (Ce, Pr, Nd, Gd). `comparator_intervals`: no comparator-interval job runs in discovery (addendum 1 item 3's parenthesis). `record_verification`: a record is scored only when its digest, fold hash and fold ids equal what the current code, fold file and plan state produce; M2 requires M1's verified record of the same fold. `safeguard_probe`: the §2 safeguard compares `nested_certificate` with `every_split` on B3x per drawn fold, and a difference makes `every_split` mandatory for that file. `budget`: `--max-hours` is an operator pause, never a demotion.

**2. Ladder M3–M7** (`models.ladder`, `scripts/g19_run_ladder.py`; §6, §7 items 3–6, §12, §15).
- `training_settings` (imported from `models.neural`). `experts_scope`, `expert_source`: per-expert bilinear terms and condition encoders, shared encoders and head; an expert of the three named mechanisms exists only with ≥ 200 training rows of the fit, else its rows take the pooled expert; routing by the row's mechanism column, else the systems table, else UNKNOWN. `predecessor_chain`: each step inherits the fold's retained predecessor and varies only its own field.
- `source_penalty`, `group_column`: b_g on the standardised target, weight 1 / τ_std² with τ_std = τ / y_sd, the MAP sum divided by n_train; λ_src = 1; weight decay also on b_g; an unseen group takes the prior 0; group = `group_cross_publication_copy`, else `pub_group`.
- `pairs`, `pair_loss_units`: comparable pairs generated inside the fit's training rows after fold assignment; pair mini-batches are consecutive chunks of a seeded permutation, cycling; Huber on the standardised difference.
- `hinge`: autograd relu(−∂ŷ_std / ∂log10[L]) rescaled by the column's SD, on training rows of NEUTRAL_SOLVATING / SOFT_N_DONOR / MIXED_NEUTRAL systems whose (system, `acid_primary`) unit spans at least two distinct extractant concentrations, never on acid-grid-flagged rows. `smoothness`: the sum over adjacent-Z trivalent Ln and An states present in training of ‖e_m(Z) − e_m(Z+1)‖² on the full e_m; a sum, not a mean. `phys_weight`: one λ_phys for both terms. Vacuity (`component_activity`): a term with no eligible row or adjacent pair on every fold is `vacuous`, and an M6 decision or H5 toggle on it is labelled so.
- H5 (`h5_toggle_config`, `H5_ABLATION_ARMS`): M3 vs M2 and M4 on / off are the ladder-step contrasts of M3 and M4 against their retained predecessor; M6a and M6b need toggle arms — the retained configuration with the hinge or the smoothness term switched off when on (the other term keeps its weight) or on at the smallest registered weight when off; comparator M6 when kept, else its retained predecessor; none when M0 is retained.
- M7: `member_seed` = `registered_model_seed(fold) + k`, k = 0..4 (§15 names one seed per fold; §6 says members differ by seeds). `bootstrap`: publication groups resampled with replacement, multiplicity as row weight, undrawn groups absent from that member's rows, pairs and hinge set. `heteroscedastic_head`: sd = softplus(MLP on the detached trunk) + `SD_FLOOR_STD`, Gaussian NLL with the mean detached, so the mean network equals the retained configuration's. `ensemble`: law-of-total-variance SD. `m7_epochs`: the retained predecessor's epoch count. `conformal`: §12 normalised-by-SD split conformal on cross-fitted inner folds, centre and SD from the outer refit.
- Runner. Steps run on the seed-104729 main designs the plan gave M2 (V5-primary, V1, V2 at the plan's schemes). Keep = R19 items 1–3, 5 plus the scoring-filter sensitivities on V5-primary against the retained predecessor, and TOST non-inferiority (ε = 0.05) on V1 and V2 (`discovery.ladder_step`); a step whose V5 ladder-scope verdict is not PASS, or whose V1 or V2 TOST is not non-inferior, is removed and the next step builds on the predecessor (§6). A keep decision is undecidable only when a design's contrast result is missing; the ladder then STOPS (resumable) instead of treating the step as removed, since §6 removes a step only on evidence. With neither M1 nor M2 kept the retained configuration is M0 (CatBoost) and M3–M7 are `not_run` (§6's components belong to the factorised model) — under the stop rule too, so §7 item 4's "M7 is run once for H6" holds only when M1 or M2 is retained. Under the stop rule M3–M6 are `exploratory_not_run`. M7 is kept only if its MAE is TOST non-inferior on V5, V1 and V2 and the S1(d) bands hold on V5-primary; undecided while an input is missing. After M7 has been judged, never under the stop rule, the H5 toggles and, when M3 is kept, its V5 strict / HNO3-only refits run; the M3 vs M2 rows are re-evaluated with them and the keep decision is untouched (asserted).
- Budget. §7 item 5 says "60 hours of wall-clock discovery compute", §0 item 3 counts the M ladder within discovery, and addendum 1 item 7 prices the discovery items against that budget with no separate ladder figure; the runner implemented one ledger (discovery wall clock plus ladder wall clock). The discovery run alone is priced at about 57 h of wall clock because two registered fallbacks fired (the §7 item 6 re-colouring: 28 rather than 22 selection-half batches; the §3.2 V1 exact design: 39 rather than 10 folds), so under one ledger M3-M7 - the uncertainty ensemble included - would be demoted before the first step, and brief §34 questions 5, 7 and 8 could not be answered from registered runs. Resolved - a compute-driven change to §7 item 5, chosen by the user on 2026-09-19 before any registered contrast was scored: the ladder has its own budget of 40 h of wall clock (`evaluation/ladder/decisions/wall_clock.json`), checked before each step; the demotion order M7 -> M6 -> M5 -> M4 -> M3 and every other rule of §7 items 3-6 are unchanged; the discovery ledger stays as registered for the discovery stages. The runner's one-ledger reading is replaced accordingly (a change to `scripts/g19_run_ladder.py`, a new file, made before the ladder starts).

**3. H3** (`evaluation.h3`, `scripts/g19_run_h3.py`; §11, §10 F4, §8).
- `deployed_rule`: "the retained ladder configuration" is the highest kept ladder step M7 > … > M3 from a registered (stop rule false) ladder run; else M2, then M1, when kept by the scorer's ladder decisions; else M2 when it passes the stop-rule scope against B3i; else B6 when it does (H1b); else B3i (`FALLBACK_DEPLOYED`). The last three read §10 F6's "best-passing baseline" as the registered H1 / H1b arms and then the lookup comparator; B5, B7 and B8 are not candidates. Resolved: M2 (and B6 for H1b) is deployed by the fallback only when its contrast passes the stop-rule scope (items 1, 2, 3, 5) AND item 6 over the reduced sensitivity set of addendum 1 - no evaluated item may FAIL; item 4 NOT_EVALUATED is allowed in discovery - i.e. the full discovery verdict is PASS or undecided by item 4 alone, never FAIL. Undecidable — and the H3, figures and report runners refuse — until every step M3–M7 is done or skipped and no ladder decision is pending.
- `ladder_arm_refit`, `tuning`: WITH is the recorded run; WITHOUT, ACT_PERMUTED and ACT_METAL_SHUFFLED are refitted at the WITH record's SELECTED configuration, stopping count and model seed (M7: the five member seeds and the group bootstrap), never re-tuned, with cross-fitted calibration at the WITH record's selections; a ladder code or decision change stales the H3 record through its digest.
- `comparator_folds`: a closed-form arm is fitted on the exact leave-one-cell-out folds of §3.1, never on the heavy arm's batched folds; on V5-PAIR every arm uses the seed-104729 batched folds. `calibration_guard`: a refit whose inner folds differ from the WITH record's is recorded `not_calibrated_inner_folds_differ_from_the_with_record`.
- X(?) rows: `discovery.actinide_rows` classifies by element, so §11's "unknown-state actinide rows included" removes X(?) rows of actinide elements in WITHOUT, and the two controls act on the same element-classified rows; X(?) rows are never scored. `ln_test_set`: the Ln(III) scored rows of the selection-half folds (V5 Ln(III) cells; every Ln(III) V2 state, focus-7 beside; Ln(III) rows of the V1 selection folds).
- `r19_scope`, `margins`, `equivalent`, `helps_non_inferior`, `f4_designs`, `actinide_dependent`, `crps`: discovery verdicts on the freezing-screen scope; δ5 on V5 Ln cells, 0.05 on V1 / V2; TOST on V5 Ln cells for equivalence and on V1 / V2 for non-inferiority; F4 per design, holding when ANY design shows it; actinide-dependent = eligibility failing once every actinide row is removed; CRPS NOT_RUN without a predictive SD.

**4. Power check and reliability** (`evaluation.power`, `scripts/g19_run_power.py`; §8).
- `needs_power`: a contrast needs the check when its family is primary (H1), H1b, S1(b) or H3 and its full R19 verdict is not PASS; S1(b) is read as part of H1. Under addendum 1 item 3 a learned-arm contrast is at best UNDECIDED, so the check is also reported on the discovery scope (`r19_scope`).
- `injection_seed`: one signal per contrast, drawn with the decision seed 104729, scaled by every κ. `h3_u_share`: nearest Shannon CN8 radius, ties by label; a state without a radius keeps its own u. `refit`: every arm at its SELECTED hyperparameters; closed-form comparators on the exact folds. `scored_rows`: X(?) rows dropped from every refit, scored in no design. `no_persisted_values`: metrics, R19 rows and κ_min only.
- `per_unit_branches`: §8's rule per UNIT — split-half (`split_half_correlation`: Pearson, Spearman–Brown, Spearman beside) for units with ≥ 4 groups, delete-one-publication jackknife for 2–3 groups (`jackknife_se`), single-group units counted, never scored; the quantity's reliability is the MINIMUM over the computed branches, both printed. Resolved: confirmed - the minimum over the computed branches (a quantity is reliable only if every estimable branch says so).
- `embedding_stability`: Procrustes-aligned bootstrap stability is upward-biased, so a label-permuted null is printed beside and a value inside the null band is flagged; the 0.3 gate stays on the raw value. `reliability_not_fitted`: B7 slopes, support-score components and logSF amplitudes by default; learned quantities only with `--include-learned`. No `V6_TARGET_ROWS` row enters any estimate; the logSF amplitude is estimated on the selection half only.

**5. Process** (`process.*`, `scripts/g19_run_process.py`; §14, §13, §9 S2(d), §10 F5).
- `draw_residual`, `member_choice`: one draw = one M7 member (uniform from a second seeded stream, the same member for Pr and Nd because a member is one model) plus, per metal, one Gaussian residual of scale std_logD · conformal_q95 / 1.96 — the Gaussian whose central 95 % interval equals the calibrated conformal one — common over the grid, Pr and Nd correlated with the §14 ρ through a Gaussian copula; §14 names no residual distribution. `fallback`: a table without `member_logD_k` / `conformal_q95` is refused in registered mode and drawn as a truncated Gaussian only under `--exploratory`, flagged.
- `PROVENANCE_CONVENTION`: gen18 has no model-predicted provenance status, so Gen19 predictions enter as ASSUMED with the placeholder label, `range` = the table-wide 95 % interval, `source.kind` model naming the arm, `model_id` the deployed configuration, the prediction file's digest and a MODEL_DERIVED note; `model_derived = True` on the Gen19 side.
- Support: a table with an UNSUPPORTED cell is trimmed to its `supported_box` (the largest rectangle free of UNSUPPORTED cells in every metal layer; untrimmed only for the F5(ii) allowed run); a stage evaluating outside the grid is `OUTSIDE_TABLE`, treated as UNSUPPORTED; every evaluation records the statuses it used (`statuses_used`), and `rank_recipes` REFUSES a converged candidate that records none.
- `objective_key_1`: keys 1–2 follow the sealed §14 order, P(both targets) then P(feasible); brief §18's joint P(both and feasible) is printed beside and does not rank. `f5_ii_per_cell`: per spec cell; the case fails when any cell does. `constraint_flags`: THIRD_PHASE_RISK, LOADING_CAP_HIT, HIGH_LOADING; PHASE_BEHAVIOUR_UNKNOWN reported as unknown. `design_count`: gen18's default of 1,000 LHS points (unregistered; §14 registers only the 64 draws). `design_space_intersection`: acid and ligand axes intersected with the supported box. `case_fixed_variables`: no saponification, complexant, salting anion, dilution or bleed.
- Gate: refused unless the discovery run is complete and the stop rule decided; refused when the stop rule fired unless `--exploratory`; refused always when the confirmation run's `confirmation.json` (schema `gen19.confirmation.v1`) says S1 failed, and when it is absent or undecided unless `--exploratory`; every output labelled `transfer-unsupported` unless S1, S2 and V6.run all hold; the ρ file and the deployed arm's own table are required in registered mode.

**6. Report and figures** (`evaluation.report`, `evaluation.figures` and their scripts; §16, §17, §19, §4, brief §34).
- Ledger: every printed number carries its raw value, source path and key (`tables/report_numbers.csv`) and is re-resolved after writing; the addenda digest is compared with `REGISTERED_ADDENDA_SHA256` and printed only from a file holding the literal. A missing input prints `not computed (input missing: <path>)` — never a placeholder or estimate — with status `not run`; a learned-arm discovery contrast carries "reduced sensitivity set (addendum 1)"; the deployed predictor is `h3.deployed_configuration`, as for H3.
- `classify_support` (Q10): the worst §13 status among the D values a recipe used decides — unsupported ← UNSUPPORTED, OUTSIDE_TABLE; speculative ← FAMILY_EXTRAPOLATION, CONDITION_EXTRAPOLATION, CROSS_LIGAND_TRANSFER; transfer-supported ← CROSS_METAL_LIGAND_TRANSFER, CROSS_METAL_TRANSFER; directly supported ← IN_DOMAIN, INTERPOLATION; an empty set is unclassified, an unknown token raises. A reading of brief §34 item 10, labelled so.
- Direction thresholds (F13): |observed logSF| ≥ 0.3 on every design as primary, plus ≥ 0.1 for V6 only, one panel per (design, threshold) — the §4 rule. Every figure names its inputs and writes a companion CSV; an absent frame yields `skipped` with the reason. The report lists every runner reading still naming a further addendum.

**What does not change.** Designs, folds, halves, hiding and guards (the re-colouring is §7 item 6 as registered); the V6 carve-out and single V6 run; metrics and averaging units; comparators; margins ρ5, γ5, η5 and ε; S1, S2 and F1–F6; R19 items 1–6 and the reduced set of addendum 1 item 4; the stop rule; the grids of §5 and §6; confirmation on the withheld seeds.

## POST-HOC addendum 3 (2026-09-22, orchestrator Claude Code; results seen: yes - the discovery selection-half scores and verdicts)

**Results seen.** The discovery run is complete and scored on the selection half, seed 104729
(`evaluation/discovery/decisions/{decisions,stop_rule,plan_state}.json`, `tables/discovery_*`): the stop
rule does not fire (M2 vs B3i on V5-primary passes items 1, 2, 3 and 5 with the point estimate above
delta5); B6 vs B3i fails; the ladder keeps M0 and drops M1 and M2, because M0 - the descriptor CatBoost
of section 5 / section 6 - has the lower V5 macro MAE; every full R19 verdict is UNDECIDED because item 4
is NOT_EVALUATED in discovery by addendum 1 item 3. Nothing has run on V6; no confirmation-half row has
been scored; the withheld seeds are untouched. The two items below are forced by that state and are
resolved now, before the ladder, H3, the power check or the confirmation run produce anything.

**1. What "passed R19 in discovery" means when freezing a claim (section 15).** Section 15 freezes at
most five claims that "passed R19 in discovery". Addendum 1 item 3 makes R19 item 4 NOT_EVALUATED in
discovery for every learned arm, so no contrast can satisfy the literal reading and the confirmation run
- and with it the single V6 run of section 3.4, S2 and the whole applied Pr/Nd question of brief section
34 - could never happen. **Resolved:** a contrast is eligible for freezing when every item R19 evaluates
in discovery is PASS and no item is FAIL; item 4, NOT_EVALUATED by addendum 1 item 3, does not
disqualify it, and is evaluated at confirmation exactly as registered (5 of 5 withheld seeds). This is
the same reading addendum 2 gives F6 for deployment and would have been required whatever the scores
were; the scorer's `passed_items_1_5_v5_primary` flag is renamed and recomputed accordingly. Everything
else about section 15 is unchanged: at most five claims, named in `decisions/CONFIRMATION_PLAN.md`
before the run, one run on the withheld seeds and the confirmation half, V6 once.

**2. The deployed configuration when the ladder keeps M0 (section 11, brief section 10 F6).** Addendum 2
resolves "the retained ladder configuration" to the highest kept ladder step, then M2 or M1 when kept,
then M2 when it passes the stop-rule scope, then B6, then B3i - and states that B5, B7 and B8 are not
candidates. That list was written when M0 could not be the retained configuration. The ladder has now
kept M0 and dropped M1 and M2, so the rule as written would deploy M2 for H3 and the process layer while
the registered ladder decision retains a different model that is better on the primary design.
**Resolved:** the deployed configuration is the ladder's retained configuration, which includes **M0
(= B5, the registered descriptor arm of section 5 and the M0 row of the section 6 ladder)** when neither
M1 nor M2 is kept; the rest of addendum 2's order is unchanged and applies only when the ladder has no
retained step. H3 (section 11) therefore trains its WITH / WITHOUT / control arms on M0, at M0's
selected hyperparameters per fold, and every report sentence about "the deployed predictor" names M0.
The factorised M2 is still the H1 candidate and is reported as such: H1 is about M2 against the lookup,
and section 31's architecture question is answered by the M2 - M0 contrast, not by what is deployed.

**3. Consequences that follow from the registered text and are recorded here, not changed.** The ladder
steps M3-M7 are `not_run` because neither M1 nor M2 is retained (addendum 2, section 2: the M3-M7
components belong to the factorised model), independently of the discovery ledger's budget demotion; the
ladder's own 40 h budget of addendum 2 is therefore not consumed. The signal-injection power check of
section 8 runs before any failed H1, H1b or H3 contrast is reported as a null. S1 stays UNDECIDED in
discovery by construction, so the process layer of section 14 runs only in the exploratory mode
addendum 2 describes, with every output labelled transfer-unsupported.

**What does not change.** Designs, folds, halves, hiding and guards; the V6 carve-out and the single V6
run; metrics and averaging units; comparators; margins rho5, gamma5, eta5 and epsilon; S1, S2 and
F1-F6; R19 items 1-6 and the reduced set of addendum 1 item 4; the stop rule; the grids of sections 5
and 6; confirmation on the withheld seeds.

## POST-HOC addendum 4 (2026-09-23, orchestrator Claude Code; results seen: yes - the discovery scores and the H3 V5 leg)

**Results seen.** Everything addendum 3 lists, plus: the H3 V5 leg of the deployed arm M0 is complete
(WITH from discovery, WITHOUT and ACT_PERMUTED refits, 28 batches each) and its deltas are computed; the
section 8 power check and the reliability report have run. V6, the confirmation half and the withheld
seeds remain untouched.

**1. ACT_METAL_SHUFFLED is not run (section 11 Controls).** Section 11 lists two controls but registers
deltas only for **WITH - WITHOUT** and **WITH - PERMUTED**; no registered delta, verdict or failure
condition uses ACT_METAL_SHUFFLED. Completing it would cost about a third of the H3 budget - the ablation
refits the deployed CatBoost and its conformal calibration per fold, measured at about 620 s per fold -
for a quantity nothing reads. **Resolved, compute-driven:** ACT_METAL_SHUFFLED is not run; the single
fold already fitted is labelled exploratory and is not scored. WITH - PERMUTED, which section 11 uses to
separate actinide chemistry from row count, is unchanged and runs on every design.

**2. H3 design scope and its compute cap.** Section 11's Ln test set is the V5-primary Ln(III) cells, the
V2 Ln(III) folds and the Ln rows of the V1 selection folds, and its arms are the retained ladder
configuration plus B6 and B5 as transparent references; here the retained configuration IS B5 (= M0), so
the arm set is B5 and B6. That is registered and is run in full. Section 11 registers no compute budget;
**resolved:** H3 gets a cap of 20 h of wall clock on 2 workers, recorded in
`evaluation/h3/decisions/wall_clock.json`. If the cap is reached, the priority order is V5 (done), then
V2, then V1, and any design not reached is reported NOT_RUN with its reason - no verdict is taken from a
partial design.

**3. Records written under a superseded code digest are refitted, not accepted.** Addendum 2 change 5
verifies a record against the digests registered for its stage. While H3 was running, its module was
corrected (the deployed-arm alias fix and the F6 guard), which superseded the `h3` entry; 13
ACT_PERMUTED folds were written after that point. **Resolved:** those records are deleted and refitted
under the current registered `h3` digest before any H3 delta is scored; records written before the
supersession keep their entry and verify as registered. No discovery record is touched.

**4. What the power check may conclude.** Section 8 requires the injection check before a failed H1, H1b
or H3 contrast is reported as a null. The check is implemented and has run; three contrasts it was asked
about are not nulls at all (their point estimates favour the comparator, so the registered wording
"null" never applies) and are reported as **POWERED_NOT_A_NULL**; B6 vs B3i, the one genuine failure,
has no kappa at which R19 passes and is therefore **UNDECIDED (underpowered)**, never "no effect". A
contrast with no registered power check is reported UNDECIDED, never null. This is the registered rule
made explicit, not a change to it.

**What does not change.** Designs, folds, halves, hiding and guards; the V6 carve-out and the single V6
run; metrics and averaging units; comparators; margins rho5, gamma5, eta5 and epsilon; S1, S2 and
F1-F6; R19 items 1-6 and the reduced set of addendum 1 item 4; the stop rule; the grids of sections 5
and 6; confirmation on the withheld seeds; the deployed configuration of addendum 3 item 2.

## POST-HOC addendum 5 (2026-09-23, orchestrator Claude Code with the user's approval; results seen: yes - discovery, the ladder, the power check and H3)

**Results seen.** Everything addenda 3 and 4 list, plus the scored H3 ablation on the deployed M0 (V5, V2
and the WITHOUT leg of V1), the power-check and reliability outputs, and the confirmation plan's costing.
V6, the confirmation half and the 5 withheld seeds are untouched.

**1. The confirmation run is the core run, not the full one - compute-driven, chosen by the user.** The
plan's full registered version costs about 174 h of serial compute, 89 h of wall clock at 2 workers, of
which the section 11 V6 actinide deltas (about 30 h) and a section 8 power check over the H3 contrasts
(about 436 h of injected refits for the full inventory, about 50 h for the subset the plan costed) are
the dominant lines and carry no frozen claim. **Resolved:** the single confirmation run evaluates
- the frozen claims of `decisions/CONFIRMATION_PLAN.md` (at most five) on the confirmation half with the
  5 withheld seeds, R19 item 4 exactly as registered (5 of 5 seeds);
- the single V6 run of section 3.4 with S2(a), S2(b) and S2(c);
- S1(c) under its addendum-2 confirmation rule and S1(d) calibration.
It does NOT run the section 11 V6 actinide deltas nor any section 8 power check; both are reported
NOT_RUN with their cost and their consequence (a contrast with no power check is UNDECIDED, never a
null - addendum 4 item 4). R19 item 6 refits at confirmation run on seed 104729 only, as in discovery
(addendum 1 item 4); every other registered rule of sections 8, 9, 14 and 15 is unchanged.

**2. The value comparison of the fold-isolation guard reads the recorded log D, not an arm's permuted
target.** ACT_PERMUTED (section 11 Controls) permutes training log D within (system, publication group);
the section 2 guard's near-duplicate VALUE comparison (`near_dup_value_tol` = 0.005) then reads permuted
values and fails folds the recorded data never had - diagnosed on `pub_97510df3a0`, where a permuted
value lands 0.00201 from another row's value while the recorded nearest is 0.115576
(`evaluation/h3/decisions/isolation_guard_diagnosis.json`). **Resolved:** for an arm whose training
target is permuted by construction, the value comparison reads the corpus's recorded log D; every other
level of the guard - publication group, archive duplicate group, and the near-duplicate key at six
significant figures - is value-independent and unchanged, and the value-blind mode stays what section 2
registers it as, a sensitivity. The 14 unfitted V1 ACT_PERMUTED folds are completed under this rule, so
no registered contrast is left INCOMPLETE.

**3. F4 is evaluated under both registered intervals, and it holds on V2.** Section 8 registers the
percentile and the BCa interval for every contrast, so F4 - "the WITHOUT arm beats the deployed
configuration on the Ln test set, 95 % interval excluding 0" - is read under both, and the conservative
outcome governs. On V2 the BCa interval of WITHOUT vs WITH excludes zero (delta +0.0264,
[+0.0031, +0.0783]) while the percentile interval does not; V5 and V1 show nothing. **F4 therefore
HOLDS**, and the report says so plainly, together with the fact that on the primary design V5 the WITH
arm is the better one (+0.0669 and +0.0782 against the permuted control, both intervals excluding zero,
both under delta5). Section 11's own consequence is recorded with it: on a verdict that is not *helps*,
"actinide rows enter the deployed Ln configuration only on helps", so the deployed lanthanide
configuration is the WITHOUT-actinide fit of M0, under which F4's first clause is false. Both readings -
F4 holding against a WITH deployment, and F4 not holding against the registered WITHOUT deployment - are
reported side by side, with the V5 point-estimate cost of the registered choice stated. No verdict is
softened and no interval is chosen by its outcome.

**4. The power-check debt is inventoried, not discharged.** 20 contrasts owe a section 8 check
(6,272 injected refits, about 436 h of serial compute at the measured per-fold cost); none is run. Every
one of them is reported UNDECIDED with the words addendum 4 item 4 fixes, never as a null, and the
inventory with its cost is part of the report so the omission is visible rather than implicit.

**What does not change.** Designs, folds, halves, hiding and the guard's value-independent levels; the
V6 carve-out and the single V6 run; metrics and averaging units; comparators; margins rho5, gamma5,
eta5 and epsilon; S1, S2, F1-F3, F5 and F6; R19 items 1-6 and the reduced set of addendum 1 item 4; the
stop rule; the grids of sections 5 and 6; the deployed configuration rule of addendum 3 item 2 as
qualified by item 3 above; confirmation on the withheld seeds.
