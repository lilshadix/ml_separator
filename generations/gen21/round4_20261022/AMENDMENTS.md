# Amendments (append-only)

Each entry records:
- the date (UTC);
- the sealed artifact that changed and why;
- **every result already seen** at the time;
- the effect on confirmatory status;
- the new seal.

Silent changes to sealed artifacts are not allowed. A change to sealed code makes `r4_run.py` refuse to run until the change is recorded here and resealed (`r4_seal.py --seal --reason`, which archives the old manifest).

## A1 (2026-10-22): stage 1 block A, provenance and power; supersedes stage-0 seal 1a09964dd60b

### What changed

**Documents**
- **PROTOCOL.md:**
  - §0 / §7: block membership. S21 moves into block A, and S22–S24 become their own block.
  - §10, new: block A's models, slots, run order, conditional routes, claim rules, leakage chain, synthetic discrimination and budget.
- **DATA_CONTRACT.md:** "Block A features" (donor class only; publication-fingerprint check).
- **RESOURCE_PLAN.md:** block A costs.
- **EXPERIMENT_REGISTRY.csv:** S01–S04 and S21 specified (status SEALED-A1). S21 is redefined from the brief's "donor-interaction ablation" to the chemistry block-shuffle control of the selected route: a stricter test of the same claim with capacity held fixed. Statuses updated for the R3-* and REF rows.
- **New documents:** PROVENANCE.md, POWER.md and DATA_REQUEST.md.
- **Changes from the pre-seal review** (`wf_37a82383-576`; 24 of 32 findings confirmed, `results/blockA_review.json`):
  - S03 now carries the same class-specific Gd term as S02, and S04 permutes only the electron-count shapes. Before this, a donor-dependent Gd dip alone could pass S02's electronic-specificity rule.
  - `r4_poison.py` writes `poison_<config id>.json`. A colon in an NTFS filename would have hidden the S21 evidence in an alternate data stream.
  - PROTOCOL §10 was rewritten, with:
    - the β-only nesting wording;
    - control rules as inequalities required on BPG and BP;
    - the 50% rule applied to S21 only;
    - finalist eligibility (controls never eligible);
    - downgrade semantics;
    - route ties.
  - A sealed decision evaluator (`report/r4_decide.py`) implements criterion 1, the harm rule, routing and control rules.
  - The synthetic study was rerun (run 2) and the power simulation was rerun (run 3).
  - Docstrings were corrected (P1 / linear content; D7's in-sample origin).
  - The registry marks S12, S17 and S18 NOT RUN (gates failed).

**Code**
- `r4/slots_a.py` (new);
- `r4/registry.py` (slot configurations);
- `scripts/r4_run.py` (slot fitted-info and the per-fold slot identity check);
- `scripts/r4_seal.py` (stage label; documents, evidence and inputs extended; `report/` swept);
- new scripts: `r4_provenance.py`, `r4_power.py`, `r4_synthetic_a.py`;
- `report/r4_table.py` (reporting only);
- `report/r4_decide.py` (sealed decision evaluator);
- `report/r4_synthetic_summary.py` (reporting only);
- `tests/test_slots_a.py` (6 tests).
- `draft/slots_b.py` exists **outside** the sealed set. It is block B's draft; it is not registered or used, and will be sealed by a later amendment.

**New sealed inputs:** gen19 `extractant_components.csv` (LF digest) and gen13's physics-basis vectors.

**Unchanged:** B0, B1, B2, metrics, decision margins, the leakage chain and all stage-0 code paths of the REF configuration. Stage-0 REF outputs stay valid. `r4_run.py` changed only by adding slot handling, so REF's checkpoint key would change under the new seal; REF is not rerun.

### Results seen before this amendment was sealed

- **Every stage-0 result.**
  - B0 0.4180 / 0.4179, B1 0.4521 / 0.4519, B2 0.4299 / 0.4246 (BPG / BP).
  - B1 − B0 +0.0341, B2 − B1 −0.0222, HIST_D7 − B1 −0.0010.
  - The full contrast, subgroup and per-pair tables (`results/REF/TABLE.md`, `per_pair_*`, `per_publication_*`, `levels_*`, `calibration_*`).
  - Identity and intervention passed; fit poisoning REF passed 50/50.
- **Provenance** (`PROVENANCE.md`, `results/provenance/`). This includes the finding that the archive has no independent extra lanthanide chemistry, and the An(III) counts.
- **Power simulation, four runs** (`POWER.md`; runs 1–3 archived in `results/power/run1`, `run2` and `run3_badboundary`; run 4 is current).
  - Optimistic bound: about 40 new sources for 80% power.
  - Heterogeneous bound: about 1 in 3 for a B1-sized gain even with 80 sources.
  - The boundary-null size is about 0.05, and the intervals undercover.
  - The balanced-MAE margin binds.
- **The publication-fingerprint check** of the donor class (0.183 vs 0.087).
- **Technical smokes** on one fold:
  - by me: S01, S04 and S21:S02 (BP);
  - by the reviewers: S02, S03 and S21:S01 (BPG).

  These are fit-side outputs only (λ, coefficients, inner scores, support exactness); no outer metric was computed.
- **The pre-seal review's 32 findings and verdicts** (`results/blockA_review.json`).
- **Tests of the decision evaluator** on stand-in columns built from REF predictions (HIST_D7 as a stand-in reproduced HIST_D7 − B1 = −0.0010; a constructed passing slot and a constructed Pr/Nd harm; nothing of block A).
- **The label-free geometry coverage** of C3 queries (89.0% with geometry at both ends; `results/provenance/geometry_coverage.json`).
- **The synthetic discrimination study** (below).
- **The source-recovery desk research** (below).
- **Unit tests:** 19 passed.

**No block-A slot has been scored on outer folds.**

### Effect on confirmatory status

- **None.** Every number remains exploratory development.
- **Adaptivity (disclosed):** block A was designed after the REF results and the per-pair tables were seen. Its shapes and donor classes come from earlier analyses (gen13 D7, gen13 physics basis, gen19 curated mechanism, the stage-0 DGA rule), not from round 4's REF residuals.
  - However, S01's shapes (e_Gd, e_Nd) and its donor contrast were selected from **gen13 D7's in-sample residual analysis**. D7 used 268 cells, including 224 of the 229 scored C3 cells.
  - So S01 is an exploratory refit of a pattern found on the same labels, not an independent test or replication of D7. Its amplitudes are refit within outer training only.
- **Margins unchanged** (+0.01 R² and 0.005 balanced MAE vs B1), although the power study shows that the balanced-MAE margin effectively requires a B1-sized gain.

### Synthetic discrimination study

Run 1 (below, as first recorded; superseded) used one BP fold, noise 0.03 and a CHEM_D7 world with D7's Nd direction reversed. Run 2 is summarised after it.

#### Run 1 (superseded)

Source: `results/synthetic_a/SUMMARY.md` and `*.json`. The specificity rules are evaluated only for a slot that passes criterion 1, as PROTOCOL §10 applies them.

| world | reps | crit1 S01 | crit1 S02 | crit1 S03 | electronic claim | donor claim S01 | donor claim S02 | elec. rule (uncond.) | mean gain S01 | S02 | S03 | S04 draws | S21:S01 draws | S21:S02 draws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NULL | 10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | -0.0002 | -0.0002 | -0.0001 | -0.0003 | -0.0001 | -0.0000 |
| CHEM_D7 | 10 | 0.30 ± 0.14 | 0.00 ± 0.00 | 0.10 ± 0.09 | 0.00 ± 0.00 | 0.20 ± 0.13 | 0.00 ± 0.00 | 0.00 ± 0.00 | +0.0153 | +0.0019 | +0.0030 | +0.0048 | +0.0073 | -0.0002 |
| ELEC | 10 | 0.00 ± 0.00 | 0.80 ± 0.13 | 0.00 ± 0.00 | 0.80 ± 0.13 | 0.00 ± 0.00 | 0.80 ± 0.13 | 1.00 ± 0.00 | -0.0000 | +0.0294 | +0.0008 | +0.0059 | -0.0001 | +0.0000 |
| SMOOTH | 10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.10 ± 0.09 | -0.0007 | -0.0002 | +0.0009 | -0.0001 | -0.0004 | +0.0000 |

**Reading**
- **No false positives.** Every rule fires 0/10 in the NULL world; the electronic and donor claims also fire 0/10 in SMOOTH.
- **The electronic route is well discriminated.** In ELEC, S02 passes criterion 1 8/10 times and the full electronic claim fires 8/10, while S03 and S04 stay near zero.
- **The donor route has low power at D7-sized effects on one fold.**
  - In CHEM_D7, S01 gains +0.015 R² on average, but criterion 1 passes only 3/10 times because the 0.005 balanced-MAE margin binds.
  - The S21 shuffle keeps about half of S01's gain, because the shared part of the dips survives shuffling.
  - So the donor claim fires 2/10.
  - A negative S01 result therefore does not rule out a D7-sized donor effect.
- **Smooth class effects are mostly absorbed by interpolation** (SMOOTH): S03 gains about 0. S03 is informative as a capacity comparison for S02, not as a detector of smooth chemistry.
- The controls are not changed after this study.

#### Run 2 (after the review; `results/synthetic_a/SUMMARY.md`, `*.json`)

- **Setup:**
  - two BPG folds (seed 130363, folds 2 and 3) with synthetic targets;
  - class-agnostic per-chemotype nuisance amplitudes on every slot shape;
  - noise sd 0.11, which gives B1 a synthetic pair RMSE of 0.17–0.19, against about 0.16 for real data;
  - 8 replicates per world, with exact 95% intervals;
  - the corrected S03 / S04, and D7's actual Nd direction.
- **Caveat:** the rules evaluated are an upper bound of the sealed ones (vs B1 only, no harm rule, no B2, one fold, no seed mean).

| world | B1 RMSE | crit1 S01 | crit1 S02 | crit1 S03 | ΔR²≥0.01 S01 / S02 | electronic claim | donor claim S01 | donor claim S02 | mean ΔR² S01 / S02 / S03 | S04 / S21:S01 / S21:S02 draws |
|---|---|---|---|---|---|---|---|---|---|---|
| NULL | 0.172 | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 0 / 0 of 8 | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | +0.0000 / -0.0003 / +0.0006 | -0.0003 / -0.0003 / -0.0014 |
| CHEM_D7 | 0.168 | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 1 / 0 of 8 | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | +0.0041 / -0.0021 / -0.0093 | -0.0000 / +0.0009 / -0.0001 |
| GDCLASS | 0.179 | 1/8 [0.00, 0.53] | 1/8 [0.00, 0.53] | 0/8 [0.00, 0.37] | 6 / 5 of 8 | 0/8 [0.00, 0.37] | 1/8 [0.00, 0.53] | 1/8 [0.00, 0.53] | +0.0270 / +0.0208 / +0.0233 | +0.0200 / +0.0003 / -0.0002 |
| ELEC | 0.185 | 0/8 [0.00, 0.37] | 4/8 [0.16, 0.84] | 0/8 [0.00, 0.37] | 0 / 5 of 8 | 4/8 [0.16, 0.84] | 0/8 [0.00, 0.37] | 4/8 [0.16, 0.84] | -0.0002 / +0.0336 / +0.0013 | +0.0032 / -0.0000 / -0.0032 |
| PROJ | 0.190 | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 4/8 [0.16, 0.84] | 1 / 1 of 8 | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | +0.0034 / +0.0032 / +0.0388 | +0.0007 / +0.0001 / -0.0057 |
| UNIVERSAL | 0.180 | 0/8 [0.00, 0.37] | 3/8 [0.09, 0.76] | 0/8 [0.00, 0.37] | 1 / 5 of 8 | 3/8 [0.09, 0.76] | 0/8 [0.00, 0.37] | 0/8 [0.00, 0.37] | +0.0008 / +0.0160 / +0.0019 | +0.0012 / +0.0001 / +0.0118 |

**Reading (run 2)**
- **No false claims in any world built to test one.**
  - Structured NULL: 0/8 on every rule.
  - GDCLASS (a donor-dependent Gd dip that S01, S02 and S03 all fit): 0/8 electronic claims. The corrected S03 / S04 controls work.
  - PROJ (a non-electronic class shape): 0/8 electronic claims; S03 picks it up instead (criterion 1 4/8).
  - UNIVERSAL (a class-free gain): 0/8 donor claims, because S21's shuffled draws keep the gain.
  - Each exact upper bound is 0.37 with 8 replicates.
- **Power is moderate where the signal is clear.**
  - ELEC: the electronic claim fires 4/8.
  - UNIVERSAL: S02's (correct, class-free) electronic claim fires 3/8.
  - The 0.005 balanced-MAE margin binds: ΔR² ≥ 0.01 is reached more often than criterion 1.
- **gen13 D7's donor contrast is not detectable at realistic noise on one fold.** CHEM_D7 gives S01 only +0.004 ΔR² (0/8). A negative S01 result in the real data would therefore be uninformative about a D7-sized donor effect; it rules out only larger effects.
- These numbers come from one fold per replicate and one design. The sealed rules use seed-pooled BPG and BP results over 25 folds, so these power figures are indicative only.

### Source recovery

Workflow `wf_07655a20-edb`: 3 researchers, public web pages only, 28 agents including skeptic re-checks (`results/source_recovery.json`; PROVENANCE.md "Source recovery").
- **C5-BTBP:** no detection limit can be verified (closed access). Same-group secondary evidence: "no extraction" stated as D < 10⁻⁴. The cells' attribution to Nilsson 2006 is doubtful; Foreman et al. 2005 is a hypothesis.
- **DMDPhPDA:** unresolved. The likely source is Shimada's open PhD thesis PDF, not downloaded without the user's permission.
- **D4GC01146E** is already in the cohort.
- **SAFE** needs a login.
- **Open sources outside the corpus:** fewer than 10 verified open-access ≥ 10-Ln sources.

**Gate decisions recorded now:**
- **G-V3 fails** (S17, S18 NOT RUN).
- **G-transfer fails for extra Ln** (S12 NOT RUN).
- **G-transfer passes the count thresholds for An(III)** (S13–S16, S24 still need the inner gate and a chemotype assignment).

### New seal

(written by `r4_seal.py` into RUN_RECORD; this entry names the superseded stage-0 seal 1a09964dd60b)

## A2 (2026-10-23): stage 1 block B (strain-architecture adaptation); supersedes seal 3e7ca14186c8

### Block A outcome (before this amendment; seal 3e7ca141)

**Results**
- **S01:** ΔR² vs B1 −0.0001 (BPG) / −0.0005 (BP); Δbalanced-MAE −0.0006 / −0.0008.
- **S02:** −0.0240 / −0.0277; +0.0003 / +0.0001. Its losses sit in sc009 / DGA (R² 0.446 → 0.384), Eu/Gd (0.422 → 0.320) and Yb/Lu.
- **Criterion 1 fails for both.** By the sealed routes, S03, S04 and S21 are NOT RUN, and their slots are not consumed.
- **Checks:**
  - interventions passed (99,110 predictions per run, max change 0);
  - slot identity passed (λ = ∞ in 7–11 of 25 folds);
  - fit poisoning for S01 and S02: **50/50 folds bit-identical each** (`results/leakage/poison_S01.json`, `poison_S02.json`). It finished after the block-B review began and before this seal.
- Files: `results/blockA_decision.*`, `results/S01/`, `results/S02/`.

**Reading.** Donor-class conditioning of the Gd / Nd dips, and electron-count shapes × donor class, add nothing over B1's single Gd amplitude. Synthetic run 2 shows that only effects much larger than gen13 D7's would have been detectable.

### What changed

**Documents**
- **PROTOCOL.md:**
  - §0 / §7: block membership. S22 moves into block B; S23–S24 become their own block.
  - §11, new: block B's model, slots, conditional routes, encoder gate, claim rules and synthetic study.
- **DATA_CONTRACT.md:** "Block B features" (features, the geometry join through the archive metal, the fingerprint check).
- **RESOURCE_PLAN.md:** block B costs.
- **EXPERIMENT_REGISTRY.csv:**
  - S05, S06, S09 and S22 sealed;
  - S07 / S10 gated and S08 conditional;
  - block-A statuses updated.

**Code**
- `r4/slots_b.py` (new; drafted in `draft/` while block A ran). Two fixes were made before this seal:
  - the metal comes from the archive record, not the safe_exp_id prefix;
  - S22's per-metal means are nested per inner block.
- `r4/registry.py` (block-B configurations);
- `scripts/r4_run.py` (fitted-info and identity for block-B parts: 0 trees = B1 exactly);
- `report/r4_decide.py` (block-B decision, `B` argument);
- `scripts/r4_synthetic_b.py`;
- `tests/test_slots_b.py` (4 tests).
- **Changes from the block-B pre-seal review** (`wf_c270fdb6-f07`; 39 of 42 findings confirmed; `results/blockB_review.json`):
  - **S09 rebuilt.** It is now a derangement, takes the source per (cell, metal) from another chemotype at the same metal with a seeded donor fallback, and keeps S06's missingness and coverage flag exactly (tested). The old version left 19–75% of query ends without geometry and had fixed points.
  - **S22** keeps S06's missingness and uses training-modal compositions.
  - **Composition.** The S06 anomaly feature is computed only between complexes of the same composition (ligand count, fill, inner-sphere anion; brief §6). Composition enters explicitly as features.
  - **Encoder gate.** The G-geo inner gate compares S06's positive-tree inner score with its own 0-tree (B1) score and with B2's. The old version passed by B1 alone.
  - **The geometry claim** also requires criterion 2's vs-B2 part, and the coverage-matched subset is reported as a secondary diagnostic.
  - **§7's stage-0 rules** are amended explicitly in §11, before any block-B score: G-geo's inner condition applies to the encoder only, S09 controls S06, and the S07 / S08 / S10 seal timing is declared as a deviation.
  - **Declared deviations from brief §6:** the pointwise level-residual training target, and B1-convention row weights.
  - **Complexity order** B1 < S05 < S06 for the §6.6 tie rule.
  - **Seal.** It now digests the geometry parquet, `accepted_geometries.csv`, the archive and the CatBoost version, and builds the new manifest before archiving the old one.
  - **Geometry fingerprint check** added.
  - **Resource plan** corrected (measured costs; at most 3 parallel jobs).

### Results seen before this amendment was sealed

- **Everything listed in A1**, plus:
  - all block-A results above, including the per-pair, subgroup and per-publication tables of S01 / S02 (`results/S01/`, `results/S02/`), and `blockA_decision.*`;
  - the fit-poisoning logs of S01 / S02 as far as they had run.
- **Block-B technical smokes** on one BPG fold: S05, S06, S09 and S22. These are fit-side only: tree counts, inner scores, support exactness.
- **Block-B unit tests:** 4 passed, including bit-identical fit poisoning on one fold.
- **Decision-evaluator tests** on stand-in columns derived from REF.
- **The block-B fingerprint checks:**
  - descriptors: 0.140 excluding the same extractant;
  - geometry: 0.275 excluding the same extractant;
  - baseline 0.087.
- **The block-B synthetic study** (below).
- **The pre-seal review of block B** (below).

**No block-B slot has been scored on outer folds.**

### Effect on confirmatory status

- **None.** Every number remains exploratory development.
- **Adaptivity (disclosed):**
  - block B was designed after block A's null results;
  - its feature set follows the brief's strain-architecture section;
  - its geometry columns come from the existing 3D dataset, not from any residual analysis;
  - the tree grid, CatBoost parameters and features were fixed before any block-B outer score.

### Block-B synthetic discrimination study

Run 1 (`results/synthetic_b/run1/`) used the uncorrected S09 / S22. **Run 2** uses the corrected controls and the composition-matched features.

**Setup**
- Block A run 2's synthetic structure: two BPG folds, chemotype nuisance, noise 0.11.
- 6 replicates per world.
- **Worlds:**
  - NULL;
  - GEO: 0.08 × each complex's own Ln–donor distance anomaly;
  - MISSING: +0.10 at metals with no geometry; the geometry claim must not fire.
- **Caveat:** an upper bound of the sealed rule (vs B1 only, one fold).

| world | crit1 S05 | crit1 S06 | geometry rule | geometry claim | mean dR2 S05 / S06 / S22 / S09 draws |
|---|---|---|---|---|---|
| NULL | 1/6 [0.00, 0.64] | 2/6 [0.04, 0.78] | 0/6 [0.00, 0.46] | 0/6 [0.00, 0.46] | +0.0240 / +0.0233 / +0.0217 / +0.0174 |
| GEO | 0/6 [0.00, 0.46] | 1/6 [0.00, 0.64] | 3/6 [0.12, 0.88] | 0/6 [0.00, 0.46] | +0.0055 / +0.0124 / +0.0048 / +0.0027 |
| MISSING | 2/6 [0.04, 0.78] | 3/6 [0.12, 0.88] | 1/6 [0.00, 0.64] | 1/6 [0.00, 0.64] | +0.0185 / +0.0271 / +0.0241 / +0.0232 |

**Reading (run 2)**
- **False positives.**
  - NULL: 0/6.
  - MISSING: 1/6 (exact upper bound 0.64). S06, S22 and the S09 draws all exploit the missingness signal. On one fold, their gaps of about 0.005–0.01 R² can cross the rule's 0.005 margins by chance.
- **Power.** With the corrected controls, the geometry claim has essentially no power: GEO gives 0/6. S06's gain over S05 (+0.012 vs +0.006) is real in GEO, but criterion 1's margins are rarely met at this signal size.
- **Consequence for the real run.** A positive geometry claim would need a large, consistent S06 advantage. A negative one is uninformative about geometry signals of this size.
- **The rule is not changed after this study.**

### Pre-seal review of block B

`wf_c270fdb6-f07`: 4 lenses and 42 findings; 39 confirmed by skeptics and 3 refuted. All confirmed findings are addressed as listed under "What changed". The block-B synthetic study was rerun (run 2) after the S09 / S22 / composition changes; run 1 is kept in `results/synthetic_b/run1/`.

## A3 (2026-10-23): close-out of the stage-1 campaign; supersedes seal 45c4b335591c

**No design or rule changes.** This amendment records the final statuses (EXPERIMENT_REGISTRY.csv) and seals REPORT.md and the block-B results as evidence.

### Block B outcome (seal 45c4b335)

- **Results**
  - **S05:** ΔR² vs B1 −0.0062 (BPG) / −0.0033 (BP); Δbalanced-MAE −0.0012 / −0.0009.
  - **S06:** −0.0056 / −0.0110; −0.0012 / −0.0012.
  - **Coverage-matched subset** (89% of queries): the same sign.
  - Inner CV chose 0 trees (= B1) in 16–21 of 25 folds.
  - S06's positive-tree inner score beat both B1 and B2 in only 9 of 50 fold-fits (gate G-geo needs 30).
- **Statuses.** Criterion 1 fails for both. By the sealed routes, S07, S08, S09, S10 and S22 are NOT RUN, and their slots are not consumed.
- **Checks.** Interventions passed (99,110 predictions per run, max change 0), slot identity passed, and fit poisoning was 50/50 bit-identical for S05 and S06.

### Campaign accounting

- **Slots consumed: 4 of 24** (S01, S02, S05, S06).
- **NOT RUN by sealed routes or gates:** S03, S04, S07, S08, S09, S10, S12, S17, S18, S19, S20, S21, S22, S23.
- **NOT REACHED:** block C (S11, S13–S16, S24), which is the An(III) transfer route. Its count gate passes, but its multitask design, the chemotype assignment for non-cohort extractants, and its controls were not written in this campaign.
- **Finalist:** none among the slots. Under §6.6 the finalist is the stronger reference, **B1** (BPG R² 0.4521), reported as a reference (series / read-set prior), not as learned chemistry.
- **Confirmation:** not run. No source-independent data exist. POWER.md shows that confirming even a B1-sized gain needs about 40 or more new independent sources.

### Results seen

Everything in A1 and A2, plus all block-B results (`results/S05/`, `results/S06/`, `results/blockB_decision.*`, `results/leakage/poison_S05.json`, `poison_S06.json`) and REPORT.md.

## A4 (2026-10-23): stage 1 block C (An(III) transfer); supersedes seal 7c8b2f043ede

After the A3 close-out, the user asked for block C: design it (An(III) transfer, chemotype assignment for non-cohort extractants, the Ln-only multitask scaffold and its controls), review and seal it, then run it. A3 had recorded block C as NOT REACHED. This amendment reopens the campaign for block C only. Blocks A and B stay closed, and their results are unchanged. The work was paused once by the user (laptop shutdown, 06:24–08:18 UTC); nothing ran during the pause.

### What changed

- **Documents.**
  - PROTOCOL §12 is new: data, chemotype links, union exclusions, model, selection, slots, the seal-bound G-transfer, routes enforced in code, claim rules with sub-labels and the synthetic-power consequence, leakage and the synthetic study.
  - §0 is headed as the stage-0 state.
  - §0 / §7 are annotated in place: S24 moves to block C; "independent publications" are copy groups; G-transfer's inner part compares S13 with S11 and B2; S24 has its own count gate.
  - DATA_CONTRACT has a new "Block C data" section, and its "not yet frozen" list is updated.
  - RESOURCE_PLAN has a new block-C section: measured memory, a parallelism cap and a 40 wall-hour ceiling.
  - PROVENANCE.md is annotated: its An(III) inventory and gate figures are superseded for block C.
  - The registry rows S11, S13–S16 and S24 are updated.
- **New code:**
  - `r4/an_data.py`: the An(III) filter, the direct Tanimoto-0.7 chemotype links, and the outer / inner union exclusions;
  - `r4/slots_c.py`: the shared/private-head network, the auxiliary data per arm, inner selection, refit and net_state;
  - `scripts/r4_gate_c.py`: label-free `--counts` plus the isolation table; seal-bound shards and combine;
  - `scripts/r4_synthetic_c.py` and `report/r4_synthetic_c_summary.py`;
  - `tests/test_slots_c.py` (9 tests).
- **Changed code:**
  - `r4/leakage.py`: `deep_equal` compares torch tensors by value. Without this, fit poisoning could not see block-C networks (review finding).
  - `r4/registry.py`: block-C configurations ("c:" parts with a draw index).
  - `scripts/r4_run.py`:
    - block-C fitted_info;
    - `block_c_route`, which refuses an off-route or stale-seal run;
    - the slot identity extended to "an arm with outer choice w_X = 0 on S11's network equals S11";
    - **predictions are written only after the intervention and identity checks pass** (`.pending`, then rename).
  - `report/r4_decide.py`:
    - `run_valid` / `load` count a run only if its intervention and identity checks passed. Every A and B run passes, so their decisions are unchanged.
    - `block_c()` and the `C` entry point: G-transfer enforced under the current seal; "untested" while on-route controls have not run; the §12 label wording; the synthetic-power consequence; S14 fallback counts.
  - `scripts/r4_poison.py`: the poisoned fit also sees the fold-excluded An records with randomised log D.
  - `scripts/r4_seal.py`:
    - digests the block-C code (including the gen19 loader's directory) and the An record and link tables;
    - skips modules with a relative `__file__`;
    - new A4 evidence and stage label.
- **Line endings.** All edited text files were normalised to LF. A Windows text-mode rewrite had turned some into CRLF, which put a literal line break inside a string in r4_seal.py (review finding).

### How the design changed before the seal (no block-C outer score existed at any point)

1. **First design** (RUN_RECORD 04:28 onwards).
   - The main head had a raw-unit loss, with weight decay 10⁻³ and a w_Ln-only grid.
   - S24 was a global cyclic relabel of the metal one-hot.
   - S16 sampled targets from the whole source block, with 3 draws.
   - S15 matched publications and row counts only.
   - The An filter was looser: 1,839 rows.
2. **Label-free counts** showed that the post-exclusion An set is 83–92% Am. Three changes followed:
   - **S24:** the global relabel is a pure reparametrisation of the private head. S24 became the within-series metal cycle behind a ≥ 10% count gate. The threshold was fixed with the counts known (label-free), and the gate fails.
   - **S16:** now samples at the same metal.
   - **S15:** now tops up extractant diversity across Ln publications.

   Synthetic run 1 was stopped at this point, before producing any output.
3. **Synthetic run 2** (12 replicates, NULL / AN_SIGNAL) showed that every arm chose the zero correction (= B1) in inner CV, except one. The cause was a design flaw:
   - the main loss was in raw units (about 0.02), while the auxiliary losses were variance-normalised (about 1);
   - weight decay dominated the main task.

   The main loss is now variance-normalised too.
4. **Synthetic-only diagnostics** (`results/synthetic_c/design_diagnostics.json`; synthetic targets on folds BPG 130363/2 and /1):
   - The pointwise head at WD 10⁻³ overfits: its inner pair loss is 17–39% above B1's.
   - At WD 0.03 or 0.3 it tracks or beats B1 (−3% to −17%).
   - A support-consistent shape-amplitude head θ(h)·(s − I_S s) was worse than B1 at every WD tried (+35% to +128%), and was dropped.

   WD ∈ {0.03, 0.3} therefore enters S11's inner grid.
5. **The first pre-seal review** (`wf_9800444f-48b`: 40 findings, 38 confirmed; `results/blockC_review.json`) led to:
   - tensor-aware `deep_equal`, plus net_state, plus a leak positive control;
   - the r4_seal.py parse fix;
   - S14's Ln head always shared (weight max(w_Ln, 0.3) when w_X > 0);
   - the descriptor scale floor and ±5 clip (a constant descriptor had saturated the encoder in 9 inner blocks);
   - stricter An filters (gen19 MODEL tier, no value conflict, one extractant: 1,574 rows);
   - copy-group counting, and B2 added to the gate's inner part;
   - 5 S16 draws;
   - routes enforced in code;
   - exclusion tests on all 50 fold-fits and 198 inner blocks, plus the isolation table;
   - the S14 selectivity sub-label;
   - paired synthetic worlds with a SMOOTH_NULL world;
   - declared deviations, tie rules and corrected counts.
6. **Synthetic run 3** (`results/synthetic_c/run3/`; the 1,574-row An set; paired, 6 replicates × 3 worlds; signal 0.15):
   - the transfer claim fired 0/6 in every world;
   - S13 − S11 ≥ 0.005 held in 0/6;
   - the paired AN_SIGNAL − NULL difference in S13 − S11 was +0.0007 on average (max +0.0024);
   - S11 beat B1 by about +0.02 in every world (generic Ln structure), and passed criterion 1 in 1/6.
7. **The verification review** (`wf_756372ce-bc9`: 26 findings, 23 confirmed; `results/blockC_review2.json`) led to:
   - An filters for active or unknown-concentration aqueous hold-back agents (102 rows), acidic co-extractants recorded as modifiers (5) and suspect flags (5), giving **1,462 rows**;
   - the gate and decision bound to the seal;
   - predictions written only after the checks;
   - the S15 50% reproduction rule;
   - the pre-declared synthetic-power consequence;
   - the §12 label wording;
   - "untested" while controls are pending;
   - a test that rebuilds the link map from scratch;
   - documentation fixes.
8. **Synthetic run 4** (the sealed code) is summarised below.

### Results seen before this amendment was sealed

- **Everything in A1–A3:** every block-A and block-B result, the decisions, the poisoning results and REPORT.md.
- **Label-free An(III) counts** (`results/blockC_counts.json`):
  - **The whole set:** 1,462 rows / 38 publications / 35 copy groups / 112 extractants (Am 1,288, Pu 95, Cm 78, Cf 1).
  - **Per fold-fit:** 557–1,369 rows, 13–35 publications, 12–32 copy groups, 77–106 extractants, Am share 0.83–0.93.
  - **S24 changed fraction:** 0.0031–0.075.
  - **S15 diversity:** 35–69 distinct extractants, against a training-Ln count of 36–74.
  - **Chemotype links:** 43 of 112 extractants are linked; 21 (241 rows) reach sc009, sc016 or sc036 only through An–An chains.
  - **An isolation:** 0 overlaps in 248 checks (`results/leakage/isolation_an.csv`).
  - **Earlier filters:** 1,839 rows and then 1,574.
- **The technical smoke of the first design** on real data (04:22 UTC; BPG 130363 fold 2; `SC.fit_slot` S11 and S13 with inner selection and outer refit; no held-out cell predicted; in `design_diagnostics.json`):
  - S11's inner losses: B1 0.03322; w_Ln 0 / 0.3 / 1 gave 0.03485 / 0.03776 / 0.04026.
  - S13's inner losses: w_X 0.3 / 1 gave 0.03312 / 0.03410. The choice was w_X = 0.3.
- **Synthetic runs:** run 1 (stopped, no output), run 2 (`results/synthetic_c/run2/`), run 3 (`run3/`) and run 4 (`results/synthetic_c/`, below).
- **The synthetic-only design diagnostics** above.
- **Both review records,** with verdicts and resolutions.
- **Unit tests** (`results/logs/tests_a4.log`, 32 passed on the final code): synthetic or training-side checks only. They include real-data inner and refit computations on fold BPG 130363/2 with a 15-epoch schedule.
- **`report/r4_decide.py C`**, exercised on stand-in tables during development (no block-C table existed).

### Synthetic run 4 (the sealed code; `results/synthetic_c/SUMMARY.md`, `summary.json`)

Six paired replicates on two BPG folds, three worlds, signal 0.15. The rules are an upper bound of the sealed ones: vs B1 only, one fold, no harm rule, no B2.

| world | transfer claim | S13 − S11 ≥ 0.005 | criterion 1 (S11 / S13) | gate-equivalent | S13 chose w_X > 0 | mean ΔR² vs B1: S11 / S13 / S14 / S15 / S16 | paired (S13 − S11) minus NULL's |
|---|---|---|---|---|---|---|---|
| NULL | 0/6 | 0/6 | 1/6 / 1/6 | 0/6 | 2/6 | +0.0189 / +0.0189 / +0.0189 / +0.0189 / +0.0189 | 0 |
| SMOOTH_NULL | 0/6 | 0/6 | 1/6 / 1/6 | 0/6 | 3/6 | +0.0189 / +0.0194 / +0.0195 / +0.0189 / +0.0189 | +0.0005 (max +0.0016) |
| AN_SIGNAL | 0/6 | 0/6 | 1/6 / 1/6 | 0/6 | 3/6 | +0.0189 / +0.0196 / +0.0190 / +0.0189 / +0.0189 | +0.0007 (max +0.0020) |

**What the table shows.**
- The revised scaffold S11 learns generic Ln structure: +0.019 ΔR² over B1 on synthetic tasks.
- An(III) information transfers only weakly. Even when the An targets carry the Ln-relevant property (AN_SIGNAL), S13 adds at most +0.002 over S11. That is below the 0.005 leg, so the transfer claim cannot fire at this effect size.
- The S15 and S16 controls sit at S11, so the null worlds never fire.

**Consequence** (PROTOCOL §12, applied by `r4_decide.py C`): the transfer claim fired in 0 of 6 AN_SIGNAL replicates, which is fewer than half.
- A non-positive S13 transfer result is therefore labelled "control power not shown". On this corpus it is uninformative about An signals of the simulated size.
- No null world fired, so a positive label would carry no null-firing caveat.
- The gate-equivalent inner condition never held, in any world (0/6). If the same holds on real data, G-transfer's inner part fails and S13 is NOT RUN. That is the sealed route, and it costs only S11's slot.

### Effect on confirmatory status

None. Block C is exploratory development on the same 229 C3 cells, like blocks A and B. No confirmation data exist (A3, POWER.md).

### New seal

(written by `r4_seal.py` into RUN_RECORD; this entry names the superseded seal 7c8b2f043ede)

## A5 (2026-10-23): close-out of block C; supersedes seal b4c2eea09af7

**No design or rule changes.**
- **What A5 records:** block C's outcome and the final statuses (EXPERIMENT_REGISTRY.csv), with REPORT.md and the block-C results sealed as evidence.
- **What changed in r4_seal.py:** only the evidence list and the stage label.

### Block C outcome (seal b4c2eea0)

- **G-transfer** (`results/gate_c.json`, computed under the seal):
  - **Count part:** passes (min 557 rows, 12 copy groups, 77 extractants).
  - **Inner part:** fails. S13's best positive-w_X inner score was below both S11's selected and B2's best inner score in 3 of 50 fold-fits (4 of 50 against S11 alone; 30 needed).
  - **S11's best network in the gate** (the (w_Ln, WD) that S13 inherits) had WD 0.3 in 42 of 50 fold-fits. S11 itself selected a network over B1 in only 3 of 50.
- **S11** (both designs):
  - ΔR² vs B1 −0.0005 (BPG) / −0.0017 (BP); Δbalanced-MAE −0.0000 / −0.0020; no harm.
  - On BP it is slightly worse than B1: every balanced-MAE interval excludes 0, and no seed favours S11. On BPG it equals B1.
  - Criterion 1 fails.
  - Inner CV chose B1 (the zero correction) in 24 / 25 (BPG) and 23 / 25 (BP) folds.
- **Statuses** (sealed routes):
  - S13 NOT RUN (G-transfer). `r4_run.py` refused an attempted S13 run and logged it (RUN_RECORD 10:05:07).
  - S14, S15 and S16 NOT RUN (route closed). S24 NOT RUN (count gate).
  - Their slots are not consumed.
- **Mechanical labels** (`results/blockC_decision.md`): S11 "no meaningful gain over B1 (criterion 1 failed)"; S13 NOT RUN.
- **Reading** (not a mechanical label; `r4_decide.py` issues the power caveat only for a run S13): in synthetic run 4, the gate-equivalent condition and the transfer claim held in 0/6 AN_SIGNAL replicates. The gate failure therefore reads as "not detected; control power not shown", not "refuted".
- **Checks:**
  - interventions passed (99,110 predictions per design, max change 0);
  - the slot identity passed (S11 equals B1 exactly where its inner choice was zero);
  - fit poisoning: 50 of 50 fold-fits bit-identical (`results/leakage/poison_S11.json`, combined from 3 shards). Network weights were compared by value, and the fold-excluded An records were poisoned.

### Campaign accounting

- **Slots consumed: 5 of 24** (S01, S02, S05, S06, S11).
- **NOT RUN by sealed routes or gates:** S03, S04, S07, S08, S09, S10, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24.
- **Finalist:** none among the slots. Under §6.6 the finalist is the stronger reference, **B1** (BPG R² 0.4521), reported as a reference (series / read-set prior), not as learned chemistry.
- **Confirmation:** not run. No source-independent data exist (POWER.md).

### Results seen

Everything in A1–A4, plus all block-C results and the close-out audit (workflow `wf_139b7d25-8db`: 14 of 16 findings confirmed; wording and numbers in REPORT.md, RESUME.md, the registry and this entry corrected before the seal):
- `results/gate_c.json` and `results/gate_c/`;
- `results/S11/`;
- `results/blockC_decision.*`;
- `results/leakage/poison_S11.json`;
- REPORT.md.

### New seal

(written by `r4_seal.py` into RUN_RECORD; this entry names the superseded seal b4c2eea09af7)
