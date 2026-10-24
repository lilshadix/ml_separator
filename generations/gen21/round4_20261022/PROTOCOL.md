# Gen21 round 4: protocol (stage 0, strict references)

Brief: `../round4_design_20261022/CLAUDE_CODE_PROMPT.md`. Evidence: `../round4_design_20261022/RESEARCH_IDEAS.md` and `EVIDENCE_INVENTORY.md`.

Companion documents are sealed with this one by `scripts/r4_seal.py` (`manifests/stage0_seal.txt`):
- `DATA_CONTRACT.md`
- `METRICS.md`
- `RESOURCE_PLAN.md`
- `EXPERIMENT_REGISTRY.csv`
- `README.md`
- the pre-seal evidence files

**Records.**
- `RUN_RECORD.md` logs every run start and end, in UTC. It is append-only from the seal on. Its header was prepended once before the seal, which is disclosed there.
- `AMENDMENTS.md` records every change after the seal, with every result already seen. A reseal archives the previous manifest and seal.

## 0. Status, and what was seen before this seal

*This section records the state at the stage-0 seal (1a09964dd60b). What was seen before each later seal is listed in amendments A1–A4 and in §10–§12; slot statuses are in EXPERIMENT_REGISTRY.csv.*

Stage 0 freezes the task, the metrics, the decision rules and the strict references B0, B1 and B2, before B1 and B2 are scored on outer development folds. The campaign slots S01–S24 (section 7) are PLANNED.

**Seen before this seal** (all exploratory development on the same 229 C3 cells):
- every round 1–3 result, including round 3's single final check, whose seeds are spent and revealed;
- the task-identity check (`results/task_identity.json`, which compares rows and targets only) and the isolation tables;
- the exact reproductions of round 3's B0, R14 and R02 (`results/reproduction.json`);
- the non-training cache poisoning (`results/leakage/cache_poison_nontrain_*.json`);
- stage-0 unit tests, which use synthetic data, training-side fits, or known round-3 R14 rows for a technical check of the metric code;
- the pre-seal review (workflow `wf_85dca07b-85f`: 35 findings, 30 confirmed, all addressed before this seal).

B2 has never been scored on outer folds.

**Adaptivity of stage 1 (declared).** The details of S01–S24 (feature lists, grids, architectures, controls) will be written **after** the B0 / B1 / B2 development results are seen. Stage 1 is therefore adaptive to those results. To bound this:
1. All slots of one stage block are sealed together in one amendment, before any of them is scored. The blocks are S01–S04 + S21 (block A; A1 moved S21 here, see §10), S05–S10 + S22 (block B; A2 moved S22 here, see §11), S11–S16 + S24 (block C; A4 moved S24 here, see §12), S17–S18, S19–S20 and S23.
2. Each amendment lists every result already seen, including the per-pair and per-subgroup tables.
3. The finalist rule, stage gates, conditional routes and confirmatory family are fixed **now** (§6, §7). They cannot be written after slot results.

## 1. Task (frozen)

- **Target.** `logSF(A/B) = log10 D(A) - log10 D(B)`, where A is the lighter metal and `|Z_A - Z_B| = 1`. That gives 12 pair types; Nd/Sm is not adjacent.
- **Primary regime P0.**
  - V1 data;
  - C3 cohort (`k_max >= 3`);
  - `k = 3`, meaning **four measured log D values** per held-out cell;
  - read sets from the original DOPT_Q policy (cached nested sequences);
  - the original fixed query set.
- **Counts (recomputed; `results/task_identity.json`).** Every seed of both designs has 229 cells and 1,802 queries.
  - Row identities and targets are bit-identical to round 3's B0 rows.
  - Every query has at least one hidden end, and no query has both ends read.
  - The input digests equal round 3's.
- **Designs.** These are written out in DATA_CONTRACT "Splits".
  - BP and BPG as implemented, with the design mask re-applied inside every inner block.
  - **BPG is the deciding design**: it isolates gen19 copy groups. BP does not, so copy groups cross train/test in 10 of 25 BP outer folds and in 23 BP inner blocks (`results/leakage/isolation_BP.csv`).
  - BP and BPG are never treated as independent replications.
- **Other regimes** each carry their own label, and none of them replaces P0:
  - zero-shot;
  - other budgets;
  - POL_ERR or new policies;
  - geometry-covered subsets;
  - V3;
  - external confirmation.
  - Budget comparisons use the same eligible cells and query denominator.

## 2. Evaluation labels

Every number carries one of these labels:
- **technical smoke**;
- **exploratory development**: the 5 dev seeds × 5 outer folds over the 229 C3 cells. Every stage-0 number is of this kind;
- **partition stability**: new seeds over the same cells. These are never blind;
- **independent confirmation**: source-independent data that arrive after the freeze.

Round 3's final seeds and strain's fresh444 pairs are consumed.

## 3. Information contract

See `DATA_CONTRACT.md`. In brief:
- A predictor sees the four read metals' log D, the star support, the metal identities, the query set, and quantities fitted on the outer training fold.
- Inner selection uses only the outer training fold.

## 4. Strict references (frozen; code `r4/refs.py`)

The common support-preserving form, with legal read set S, the PWL_R operator `I_S` (piecewise linear along −Shannon CN8 radius, constant beyond the extreme reads) and per-metal potentials u:

```
level(m) = I_S[y_S - u_S](m) + u(m)  =  PWL_R(m) + sum_j u_j * (1[m = j] - w_j(m))
```

`w_j(m)` is read metal j's PWL_R weight at m. Measured supports are exact, and pair predictions are differences of one curve.

**B0.** PWL_R, with u = 0.

**B1.** One Gd amplitude, `u = beta * 1[Gd]`; this is round 3's `gd_dip`.
- **Training situations.** All outer-training cells with `k_max >= 3`, each masked at k = 3 with:
  - its nested DOPT_Q set;
  - its UNIFORM set;
  - 4 seeded RANDOM legal sets.
- **Weights.** Every training row (one hidden strict-adjacent query under one mask) carries the cell's gen13 chemotype-balanced weight divided by the number of masks.
  - Weights are computed over all outer-training cells, so a cell's total weight also grows with its number of hidden queries. This is R14's convention, kept so that B1 reproduces R14.
  - The evaluation's balanced MAE uses a different weighting (METRICS.md).
- **Estimator.** Weighted ridge with column standardisation and a trace-scaled λ; the penalty multiplier is 10.
- **Inner selection.**
  - λ comes from {0.003, 0.03, 0.3, 3, 30} **plus λ = ∞, which gives β = 0** and returns B0.
  - The criterion is the weighted squared error on each inner block's validation cells under their nested-DOPT_Q masks. The inner-fit masks are rebuilt with the block's own covariance.
  - Ties go to the stronger penalty. Round 3 broke ties toward the weaker one; exact ties have not occurred.
- **Disclosure (inherited, unchanged):** each inner block's noise variance includes gen20's noise factor. gen20 chose that factor on inner block 0's validation labels. So λ selection is nested for the covariance but not for that factor. This stays inside outer training.
- **Orphan rule (round-4 change, declared before scoring).**
  - The problem: the g21 cache skips an inner block whose masked inner-training set has fewer than 20 cells. In BP and BPG seed 155921 fold 2, this leaves 148 of the fold's 183 C3 training cells in no inner block, and round 3 / R14 dropped them from every fit.
  - The rule: round 4 builds their cell records label-free, with DOPT_Q and UNIFORM sequences from the outer-training covariance. They join every inner fit whose training set contains them (masks recomputed with that block's covariance) and the final refit. They cannot be inner-validation cells.
- **Identity check.** B1 = R14 holds exactly in every fold where λ is finite and there are no orphans; when λ = ∞, B1 = B0 exactly. The check runs per fold (`results/REF/identity_<design>.json`) and stops the run on any violation.

**B2.** A universal, chemistry-free series / read-set prior with the same training situations, weights, nesting, grid (including ∞) and tie rule as B1:
- coherent potentials for all 14 metals (penalty multiplier 10);
- round 3's 6 read-geometry features of the unread metal (penalty multiplier 1).
  - These depend on the read positions **and on the read values**: data curvature and edge slopes.
  - So B2 is chemistry-free, but not support-value-free.

**B2 gauge.** The 14 potentials are identified only modulo span{1, x}. The x direction trades against the dL / dR geometry columns, and predictions are unchanged under that trade. Stored coefficients are the ridge minimum-penalty representative and are not per-metal effects. B2 is frozen; no variant may be selected.

**HIST_D7 (diagnostic only).**
- What it is: B1's form with the fixed −0.072 Gd residual from the full-corpus gen13 D7 diagnostic.
- Why it is not a reference: that diagnostic used 268 cells, including 224 of the 229 C3 cells. It is not fold-independent and never a reference, candidate or "external" prior.

**Historical diagnostics** scored on the same rows: PWL_Z, HBK@ML, RIDGE2, LINE, G14@ZERO, FLAT.

**Reference contrasts** (dev, both designs): B1−B0, B2−B0, B2−B1, HIST_D7−B1, B1−R14_REPRO, B0−HBK@ML, B0−PWL_Z.

**Reproductions** (not slots): round 3's B0, R14 and R02. All six are bit-identical (`results/reproduction.json`).

## 5. Metrics and uncertainty

See `METRICS.md`.
- **Primary.** Per-seed pooled strict-adjacent R², then the seed mean.
- **Balanced MAE.** Chemotype-macro over extractant-macro of |error|, inside each seed; then the seed mean.
- **Clusterings.**
  - `chemotype`: 38 nominal, Kish about 3.85.
  - `cluster_cpg`, the declared inferential clustering: connected components of chemotype ∪ publication ∪ gen19 copy group. It has 13 nominal clusters, Kish about 2.24, and the largest holds about 65% of rows.
- **Paired contrasts** report:
  - percentile and BCa 95% cluster-bootstrap intervals (2,000 draws, shared);
  - a cluster-jackknife t(G−1) 95% interval;
  - leave-one-cluster-out extremes, per-seed signs, and nominal / Kish cluster counts.
- **Small samples.** With so few effective clusters, every interval is approximate. The jackknife t is the small-cluster sensitivity. Seed standard errors are never read as precision.

## 6. Decision rules (sealed now)

**Design clause.** Every criterion is judged on **BPG**, the deciding design, and must also hold on **BP** (same sign and margin) as a robustness conjunction. BP alone never promotes.

1. **Meaningful improvement over B1** (exploratory, dev). Both of the following must hold on BPG and on BP:
   - point ΔR² ≥ **+0.01**;
   - balanced-MAE reduction ≥ **0.005** log10 units.
2. **Evidence for cell-conditioned learning** (exploratory, dev). Criterion 1 must also hold against **B2**, and the claim's falsification controls (§8.6) must fail to reproduce the gain. A control reproduces the gain when its ΔR² vs B1 is ≥ 50% of the candidate's.
3. **Source-independent confirmation.**
   - **Scope:** one frozen finalist, run once on data that arrive after the freeze.
   - **Paired intervals:** for **both** endpoints (ΔR² and Δbalanced-MAE vs B1, and vs B2 for a cell-conditioned claim), the percentile 95%, the BCa 95% **and** the cluster-jackknife t 95% intervals must all exclude zero in the favourable direction. A NaN interval counts as not excluding.
   - **Clustering:** `cluster_cpg`, extended to new publications by the frozen provenance links (DATA_CONTRACT).
   - **Decision rule:** intersection–union, α = 0.05.
   - **Family:** the confirmatory family is this one finalist with its two endpoints. A second claim needs Holm and an amendment before its labels arrive.
   - **Floor:** if the confirmation data's Kish-effective cluster count is < 3, the result is labelled INCONCLUSIVE (underpowered), whatever the intervals show.
   - **Harm:** no material subgroup harm (item 4).
4. **Material subgroup harm.**
   - **Comparators:** B1 for criterion 1; B1 and B2 for criterion 2.
   - **Declared subgroups:**
     - each of the 12 pair types (Pr/Nd, Eu/Gd and Gd/Tb named);
     - sc009 and non-sc009;
     - donor-defined DGA and non-DGA (`r4/donor.py`, frozen);
     - one end read and both hidden.
   - **Dev harm:** the subgroup row MAE is worse by > **0.010** log10 units in the seed mean **and** in ≥ 4 of 5 seeds, on BPG.
   - **Confirmation harm** (single run): the subgroup MAE is worse by > 0.010, on a subgroup with ≥ 20 rows.
   - **Where 0.010 comes from:** twice the planning margin, fixed a priori without test information. The brief's "training-only" derivation was not used: no training quantity was judged more defensible than this fixed number, and this deviation is declared.
   - A failed Pr/Nd slice is always reported.
5. **The R² ≥ 0.50 level** is an aspiration. It is judged on unrounded values, with its cluster-bootstrap interval (`levels_*.csv`), and never substitutes for criterion 1 or 3.
6. **Finalist selection rule.**
   - **Eligible:** candidates that pass criterion 1 on both designs, every leakage check (§8), and their claim's falsification controls.
   - **Choice:** the largest BPG point ΔR² vs B1.
   - **Ties:** differences < 0.005 go to fewer fitted parameters, then to the lower slot number.
   - **No eligible candidate:** the finalist is the stronger of B1 and B2 by BPG point R². The report names it as a reference, not as learned chemistry.
7. **Labelling.** A dev gain is **exploratory**, and an unresolved interval is **INCONCLUSIVE**. Margins are never relaxed after a result.
8. **Power and MDE.** Before any confirmation design, the power to reject a non-positive gain and the probability of passing the full gate are simulated:
   - by effect injection into training / development tasks under the `cluster_cpg` structure;
   - at zero, at the margins, and at 2× the margins;
   - with Monte Carlo uncertainty.

   If power is below 80%, the study is labelled underpowered, together with the number of independent sources needed.

## 7. Campaign budget, stage gates and conditional routes (brief §7)

**Cap.** At most **24 outer-scored configurations** (S01–S24), counting every outer-viewed variant, control, combination, policy and data version, including failed or off-registry runs. References and reproductions cost nothing.

**Allocation** (PLANNED; sealed per stage block, §0):
- S01–S04: residual / electronic;
- S05–S10: strain adaptation;
- S11–S16: data / transfer;
- S17–S18: source quality;
- S19–S20: combination / policy crossing;
- S21: the chemistry block-shuffle control of block A (amendment A1, §10);
- S22: the radius-only geometry control of block B (amendment A2, §11);
- S23: additional control;
- S24: the metal-state-mapping control of block C (amendment A4, §12; NOT RUN on its label-free count gate).

**Stage gates** (numeric, inner-only, fixed now). A gate that fails marks its slots NOT RUN, with the reason.
- **G-geo (S06, S07, S09, S10, S22).** Two conditions:
  - geometry for both ends of ≥ 70% of C3 queries (label-free count);
  - the slot's inner-CV loss (weighted SSE on inner-validation DOPT_Q masks) below B2's in ≥ 30 of the 50 outer fold-fits.
  - Encoder slots (S07, S10) additionally need S06 to pass this inner gate.
- **G-transfer (S12–S16, S24).** After the Ln ∪ auxiliary exclusions of every outer fold, the auxiliary data must keep:
  - ≥ 5 independent publications (operationalised in A4 as gen19 copy groups);
  - ≥ 10 extractants;
  - ≥ 100 rows (minimum over folds).
  - The model must also beat the Ln-only scaffold S11 in inner CV in ≥ 30 of 50 fold-fits (A4, §12: S13, against S11 and B2).
  - S24 additionally has its own label-free count gate (A4, §12).
- **G-V3 (S17, S18).** At least one C3 cell's value or censoring status must change on documented source evidence (no residual-based rule).

**Conditional routes** (fixed now; inner-only evidence):
- **S09** controls whichever geometry route (S06 or S07) has the larger mean inner-CV improvement over B2.
- **S19** combines only components that each passed criterion 1 vs B1 on both designs, and whose single-component ablations exist.
- **S23** (training-fitted rescaling) is applied to the slot with the largest BPG ΔR² vs B1.

## 8. Leakage and controls

Every reference and candidate must pass these on every outer fold of both designs.

1. **Isolation.**
   - `assert_test_isolation` on every fold, plus `results/leakage/isolation_<design>.csv` (outer and every inner block).
   - Overlap must be zero for cells, chemotypes and publications, and for copy groups under BPG. Inner blocks must lie inside outer training.
   - BP's copy-group crossings are reported, not required to be zero.
2. **Fit poisoning.**
   - All label fields of every non-training cell are randomised: log D, gen13 coefficients, and the frame's `logD__*`, `repsd__*` and `replicate_sd_median`.
   - The whole fitted chain must be bit-identical, with a private inner cache (`scripts/r4_poison.py`, all 50 folds).
3. **Cache poisoning.**
   - Every fold cache is rebuilt with the labels of **all non-training cells** randomised (held-out and design-excluded), and every cached field must be identical.
   - Passed before the seal on 50 / 50 folds (`results/leakage/cache_poison_nontrain_*.json`). Stage 0 adds no cache.
4. **Prediction intervention.**
   - Every unread held-out value is replaced, and every method is rebuilt by the registry on the scrambled bench copy. Predictions and read sequences must be bit-identical.
   - Two positive controls, a `block["Y"]` peek and a `bench.Y` closure, are caught (tests).
   - The frame's label columns and coef are not scrambled in this check. They are covered by item 2 for fits, and no predictor reads them.
5. **Support consistency.** Exact supports, a fixed denominator, no pair with both ends measured, correct sign and adjacency, and the per-fold B1 identity check.
6. **Claim-specific falsification controls.**
   - Declared per slot in the stage-1 amendment. Each must first show on synthetic null and known-signal tasks that it detects its intended claim.
   - Never the round-3 within-pair shuffle as a universal gate.
7. **Calibration control.** S23 is a training-only rescaling / shrinkage competitor. The test-fitted optimal scale is reported only as an ORACLE.

## 9. Git and publication

Nothing is committed or pushed without the user's separate request. The seal is local, not public preregistration.


## 10. Stage 1, block A: does donor class modulate the series shape beyond B1? (amendment A1)

Code: `r4/slots_a.py` and `r4/registry.py`. The decision evaluator is `report/r4_decide.py`.

These rules were sealed before any of S01–S04 or S21 was scored on outer folds. The pre-seal review (workflow `wf_37a82383-576`) confirmed 24 of 32 findings, and they are folded in here.

### Model

Every slot is B1 plus a support-consistent correction:

```
level(m) = B1(m) + sum_s theta_s(cell) * (s(m) - I_S[s](m))
```

- **Amplitudes.** `theta_s(cell) = shared_s + dev_{s, class(cell)}`. Deviations are one-hot over all three classes, so ridge shrinks every class toward the shared value.
- **Fit.**
  - Training situations, masks and weights are exactly B1's (§4, DATA_CONTRACT).
  - The target is the hidden log SF minus B1's prediction.
  - **Inner CV.** Inside every inner block, B1's β is refitted on that block's training rows. λ_B1, however, is the outer-fold selection, which used every inner block's validation cells. So β is nested but λ_B1 is not. This stays inside outer training and is conservative for the slot. The inherited gen20 noise-factor disclosure (§4) also applies.
  - The slot λ comes from {0.003, 0.03, 0.3, 3, 30, ∞}: ∞ returns B1 exactly, and ties go to the stronger penalty. Every column has penalty multiplier 10.
- **Donor class** (label-free; DATA_CONTRACT "Block A features"): DGA (diglycolamide SMARTS), SOFT_N (gen19 curated mechanism SOFT_N_DONOR), or OTHER.

### Slots

| slot | shared shapes | class-deviation shapes | columns | role |
|---|---|---|---|---|
| **S01** | e_Nd | e_Gd, e_Nd | 7 | hierarchical residual: the gen13 D7 donor contrast of the Gd and Nd dips |
| **S02** | gd_break, tetrad_e1, tetrad_e3 (gen13 physics basis) | e_Gd, gd_break, tetrad_e1, tetrad_e3 | 15 | electronic shapes × donor class |
| **S03** | P2, P3, P4 (orthonormal polynomials in −radius) | **e_Gd**, P2, P3, P4 | 15 | matched-capacity smooth control of S02: the same class-specific Gd term, with smooth shapes in place of the electron-count shapes |
| **S04** | S02 with the metal map of its **electron-count shapes** (gd_break, e1, e3) permuted; the class-specific Gd term is kept in place | | 15 × 5 draws | electronic-map control of S02 (fixed seeds 20261023–27; each permutation differs from identity and from the others) |
| **S21** | the route's shapes, with each chemotype's majority donor class permuted across chemotypes | | 5 draws | chemistry block-shuffle control (seeds 20261123–27; class counts over chemotypes preserved; every extractant of a chemotype gets one class) |

- **Why S03 and S04 keep the Gd term.** With the class-specific Gd term in place, a donor-dependent Gd dip alone (S01's hypothesis) cannot pass S02's electronic-specificity rule.
- **Why S21 replaces the brief's "donor-interaction ablation".** The shuffle tests the same donor-interaction claim with capacity and shapes held fixed.
- **S03's shapes, P2–P4,** leave out degree 1. Degree 1 is annihilated by I_S only inside the read hull. S02's shapes carry almost no linear content, so a null S03 does not show the absence of a smooth class effect at the series edges.

### Run order and conditional routes (labelled-dev rules, declared now)

1. S01 and S02, on BPG and then BP. Then `report/r4_decide.py`.
2. **S03 and S04 run only if S02 passes criterion 1** (both designs, plus the harm rule). Otherwise they are NOT RUN, and their slots are not consumed.
3. **S21 runs only if S01 or S02 passes criterion 1.** Its route is the passing slot with the larger BPG ΔR² vs B1. A difference < 0.005 goes to fewer parameters (S01 has 7, S02 has 15), then to the lower slot number. Otherwise S21 is NOT RUN.

### Claim rules (`report/r4_decide.py` applies them mechanically)

- **Criterion 1 (exploratory gain).** All of:
  - ΔR² ≥ +0.01 and Δbalanced-MAE ≥ 0.005 vs B1, on BPG **and** on BP;
  - no material harm vs B1 on BPG. Harm is a declared subgroup whose row MAE is worse by > 0.010 in the seed mean and in ≥ 4 of 5 seeds. The subgroups are the 12 pair types, sc009 / non-sc009, DGA / non-DGA donor, and one-end-read / both-hidden.
- **Criterion 2 (donor-conditioned learning, route only).** All of:
  - criterion 1 vs B1, plus the same margins and harm rule vs B2;
  - **on BPG and on BP**: route − max(S21 draws) > 0;
  - **on BPG and on BP**: mean(S21 draws) < 0.5 × route.

  All quantities are ΔR² vs B1. This is §6.2's 50% reproduction rule, applied to the draw mean, and it applies to S21 only.
- **Electronic specificity (S02 only; on BPG and on BP).** Both:
  - S02 − S03 ≥ 0.005;
  - S02 − max(S04 draws) > 0.

  All quantities are ΔR² vs B1. Otherwise any S02 gain is attributed to flexible shape capacity.
- **What failing a control does.** It downgrades the attribution, not criterion-1 eligibility. A slot that passes criterion 1 but fails its control is labelled an "exploratory gain over B1" with no donor or electronic attribution.
- **A non-route slot** that passes criterion 1 is labelled "exploratory gain; donor claim untested".
- **Finalist eligibility in block A (§6.6).**
  - Only S01 and S02 can be finalists, on criterion 1 plus every §8 leakage check.
  - S03, S04 and S21 are controls (brief §7): they are never finalist-eligible, whatever their own ΔR², and their repeated draws are never selectable.

### Leakage

Every slot run passes the same chain as REF:
- isolation;
- fit poisoning on all 50 folds (`r4_poison.py <ID>`; output `poison_<config id>.json`);
- the prediction intervention with methods rebuilt;
- support exactness;
- the per-fold slot identity: a slot with λ = ∞ equals B1 exactly.

### Discriminating power (brief §8.8; shown before sealing)

`scripts/r4_synthetic_a.py`, run 2, is summarised in AMENDMENTS A1; run 1 is kept.
- **Setup:** two BPG folds' real structure with synthetic targets, realistic noise and class-agnostic chemotype nuisance structure.
- **Worlds:** NULL (structured), CHEM_D7, GDCLASS, ELEC, PROJ and UNIVERSAL, 8 replicates each.
- **Reporting:** exact binomial intervals.
- **Caveat:** the study evaluates an upper bound of these rules (vs B1 only, no harm rule, no B2, one fold, no seed mean).

### Budget

Block A uses at most 5 of the 24 slots (S01, S02, S03, S04, S21); S03, S04 and S21 are conditional.

Gate decisions already recorded (PROVENANCE.md): S12, S17 and S18 are NOT RUN, because their gates failed on counts or source evidence.


## 11. Stage 1, block B: strain-architecture adaptation (amendment A2)

Code: `r4/slots_b.py` and `r4/registry.py`. The decision evaluator is `report/r4_decide.py B`.

These rules were sealed after block A was scored (its results are listed in A2) and before any block-B slot was scored on outer folds. The pre-seal review (workflow `wf_c270fdb6-f07`) confirmed 39 of 42 findings, and they are folded in here.

**Question.** Does a small learned correction of B1's level residual at unread metals add skill beyond B1?
- **S05:** tabular information only.
- **S06:** tabular information plus compact 3D geometry of the cell's own complexes.

This ports the strain architecture's level / shape idea in support-preserving form (brief §6):
- B1 plays the level and anchor;
- a CatBoost model supplies the scalar shape correction;
- read metals keep their exact readings.

### Model

- **Target.** For each of B1's training situations and each unread metal m: `t = y(m) − B1_S(m)`.
- **Prediction.** `level(m) = B1_S(m) + f(features(m))` at unread metals, and `level = y` at read metals.
- **Estimator.** CatBoostRegressor (depth 4, learning rate 0.05, l2 3, RMSE, seed 0, 4 threads). CatBoost is deterministic here; the fit-poisoning test is bit-identical.
  - The number of trees comes from {0 (= B1), 100, 300, 1000} by inner chemotype-block **pair** loss, using staged predictions of one 1,000-tree fit per inner block. Ties go to fewer trees.
  - B1's β is refitted per inner block at λ_B1 (as in block A: β nested, λ_B1 the outer selection).
  - S22's per-metal means and modal compositions come from each inner block's own training cells.
- **Declared deviations from brief §6:**
  - The model is fitted on pointwise level residuals at unread metals. Only its tree count is chosen by pair loss; it is not trained on pair differences directly.
  - Row weights follow B1's convention (cell weight / masks for each unread-metal row), not the brief's per-cell normalisation.
- **Features** (label-free for the held-out cell; DATA_CONTRACT "Block B features"):
  - **S05:** metal and read-set geometry, support summaries, B1's coherent Gd feature, round 3's read-geometry features, donor class, and 8 RDKit descriptors.
  - **S06:** S05's features, plus for the cell's own complex at metal m:
    - mean Ln–donor distance, physical coordination number and metal partial charge;
    - **the complex's composition:** ligand count, fill-ligand count and inner-sphere anion (nitrate / water / other);
    - the anomaly g(m) − I_S[g(S)](m), **only when every read metal used at m has a complex of the same composition**, otherwise missing. Brief §6 requires a physically comparable composition, and 195 of 229 C3 cells change composition across metals;
    - a coverage flag and a composition-match flag.
  - **Coverage:** 89.0% of C3 queries have the cell's own geometry at both ends. The full 229-cell denominator is always scored, and missing geometry takes CatBoost's missing branch.
  - **Coverage-matched subset:** reported as a secondary diagnostic (`report/r4_decide.py B`).

### Slots

| slot | content | role |
|---|---|---|
| S05 | tabular correction | does a flexible support-summary / chemistry correction add to B1? |
| S06 | S05 + compact geometry (with composition) | does geometry add? |
| S09 | **geometry identity shuffled, missingness kept** (5 seeded draws, seeds 20261223–27) | geometry block-shuffle control of S06 |
| S22 | S06 with each target's geometry replaced by the training-fitted per-metal mean and modal composition, **missingness kept** | radius-only (series-only) geometry control; moved into block B by A2 |
| S07, S10 | distance encoder; random-encoder control | gated below |
| S08 | nested scalar blend of B1 / S05 / S06 | conditional below |

**How S09 builds a draw.**
- **Derangement.** π is a derangement over the chemotypes that carry geometry (rejection sampling with the sealed seed), so no chemotype keeps its own geometry.
- **Source choice.** A target (cell, metal) that has its own geometry receives the geometry and composition of a seeded random cell of π(c) at the same metal.
- **Fallback.** If π(c) has no geometry at that metal, the source is a seeded donor chemotype ≠ c. This happens for 287–1,130 of 3,106 targets, depending on the draw.
- **Missing stays missing.** A target without its own geometry stays missing, so the coverage pattern equals S06's exactly (tested).

### Run order and conditional routes (labelled-dev rules, declared now)

1. S05 and S06, on BPG and on BP. Then `report/r4_decide.py B`.
2. **S09 and S22 run only if S06 passes criterion 1.**
3. **S08 runs only if S05 and S06 both pass criterion 1.** Its design (convex weights on inner out-of-fold pair predictions, grid step 0.1 including 0) is then sealed in a further amendment before scoring.
4. **The encoder slots S07 / S10** need all of:
   - S06 passing criterion 1;
   - gate G-geo (inner, amended in A2): S06's best inner score over **positive** tree counts is below both its own 0-tree score (= B1) and B2's best inner score, in ≥ 30 of the 50 outer fold-fits;
   - their own amendment and review before scoring (fold-local CPU encoder).

   Otherwise S07 / S10 are NOT RUN.

### Amendment of the stage-0 §7 rules (declared before any block-B score)

- **G-geo coverage** (≥ 70% of C3 queries with geometry at both ends) **passes** at 89.0%.
- **G-geo's inner condition** applies to the **encoder** slots only (S07, S10). The compact-geometry route (S06, with its controls S09 and S22) is the brief's first, cheap geometry test, and is always run or routed as above.
- **§7's conditional route for S09** ("the geometry route with the larger inner improvement") reduces to S06, because S07 is not run unless its gate passes.
- **Block membership.** §0's "all slots of a block sealed together" is met for S05, S06, S09 and S22. S07, S08 and S10 are sealed only if triggered, each by its own amendment before scoring. This deviation is declared.

### Claim rules (on BPG and on BP)

- **Criterion 1 and the harm rule** as in §10.
- **Geometry claim (S06; cell-conditioned).** All of:
  - criterion 1 vs B1, plus criterion 2's vs-B2 part;
  - S06 − S05 ≥ 0.005;
  - S06 > max(S09 draws);
  - mean(S09 − S05) < 0.5 × (S06 − S05);
  - S06 − S22 ≥ 0.005.

  All quantities are ΔR² vs B1. Otherwise any S06 gain carries "geometry attribution not shown".
- **S05 passing criterion 1** is labelled "exploratory gain; chemistry attribution untested". Its gain could come from support summaries rather than chemistry (block-B synthetic NULL world).
- **Finalist eligibility:** S05 and S06 (S08 if run). S09, S22 and S10 are controls and never eligible.
- **§6.6 tie rule, complexity order:** B1 < S05 < S06.

### Discriminating power

`scripts/r4_synthetic_b.py`, run 2, uses block A run 2's structure with three worlds:
- NULL;
- GEO: the signal follows each complex's own Ln–donor distance anomaly;
- MISSING: the signal follows geometry missingness only, so the geometry claim must not fire.

It is summarised in A2; run 1 is kept.


## 12. Stage 1, block C: An(III) transfer through a common Ln-only multitask scaffold (amendment A4)

Code: `r4/an_data.py`, `r4/slots_c.py`, `r4/registry.py`, `scripts/r4_gate_c.py`. The decision evaluator is `report/r4_decide.py C`.

These rules were sealed after blocks A and B were scored and closed out (A3; every result listed in A4), and before any block-C configuration was scored on outer folds. Before this seal, only the following block-C quantities were seen:
- label-free counts (`results/blockC_counts.json`, `results/leakage/isolation_an.csv`);
- training-side inner scores of one fold from a technical smoke;
- unit tests;
- the synthetic studies, and the synthetic-only design diagnostics.

A4 lists all of these, together with the design changes they caused. The pre-seal review (workflow `wf_9800444f-48b`) confirmed 38 of 40 findings, and they are folded in here.

**Question** (brief §6, "Actinide transfer"). Do verified An(III) extraction records, used only as training-side auxiliary data, improve held-out lanthanide adjacent log SF beyond the same model trained without them? The comparison is against a common Ln-only multitask scaffold (S11), with the same Ln rows, targets, weights and optimisation budget in every arm.

### Data (label-free selection; DATA_CONTRACT "Block C data")

- **Source.** The gen19 archive through `gen19ct.data.load.load_archive`, which adds the gen19 publication ids. Copy groups are the project's `copy_group_map`.
- **Eligible An(III) records:**
  - metal category actinide; metal ∈ {Am, Cm, Pu, Cf}; oxidation state 3 (An(VI) and unknown states excluded);
  - `acid_anion == nitrate`: nitric acid, the cohort's majority regime;
  - no aqueous complexant, and no active (or unknown-concentration) aqueous hold-back agent: SANEX-type aqueous complexation reverses the selectivity mechanism. Organic-phase soft-N extractants are kept, because the cohort includes SOFT_N chemotypes;
  - no ionic-liquid diluent (a solvent key containing `[`);
  - finite log D; an extractant that RDKit can parse;
  - gen19 tier MODEL: this drops duplicate members and not-modelable rows;
  - no gen19 value conflict, and no suspect flag;
  - exactly one organic extractant, and no acidic co-extractant recorded as a modifier (HDEHP-type): synergistic mixtures are dropped, because the encoder sees one extractant. Alcohol phase modifiers are kept.
- **The set** holds 1,462 rows, 38 publications (35 copy groups) and 112 extractants: Am 1,288, Pu 95, Cm 78, Cf 1.
- **This supersedes for block C** the stage-1 An(III) inventory in PROVENANCE.md, which used a looser definition (2,009 rows; "minimum 19 publications / 95 extractants / 918 rows").
- **No An curve target is fabricated** (brief §6). The An data enter only as raw log D per (extractant, metal, conditions) record. There is no An adjacent-curve target.
- **Conditions are not inputs.** Acid concentration, extractant concentration and diluent are not model inputs, so the auxiliary head learns an extractant-level affinity averaged over conditions inside the compatible regime. This simplification is declared.

### Chemotype assignment of non-cohort extractants

The project's rule is `TANIMOTO_CLUSTER_THRESHOLD = 0.7`, with **direct (one-hop) links** as in gen11 `pools.chemotype_assignment`. There is no An–An transitive closure.
- An An extractant is **linked** to every cohort chemotype that has a member at Morgan (radius 2, 2,048 bits) Tanimoto ≥ 0.7.
- 43 of the 112 An extractants have a direct link.
- No An extractant is assigned to a new chemotype, and links are never fitted to labels.

**Declared, label-free consequence.** 21 An extractants (241 rows, 12 publications) reach further cohort chemotypes (sc009, sc016, sc036) only through An–An ≥ 0.7 chains. 18 of them have no direct link (164 rows). These stay in training when those chemotypes are held out.

### Union exclusions (applied before any pretraining, fitting or weight choice)

**Outer fold.** Every An record is dropped that shares any of the following with any cell outside the fold's training set (held-out cells of every class, and design-excluded cells):
- (a) the gen19 publication;
- (b) the copy group;
- (c) the exact extractant (canonical SMILES);
- (d) a linked chemotype.

**Inner block.** The same rule is applied again against the outer-training cells outside the block's inner-training set.

**What survives.** Over the 50 dev fold-fits (`results/blockC_counts.json`):
- 557–1,369 rows;
- 13–35 publications (12–32 copy groups);
- 77–106 extractants;
- an Am share of 0.83–0.93.

**Check.** `r4_gate_c.py --counts` recomputes the four clauses from the held cells without calling `excluded_by`, for every outer fold and all 198 inner blocks. It finds 0 overlaps in 248 checks (`results/leakage/isolation_an.csv`).
- It reuses the sealed link map, canonicaliser and copy-group map.
- The unit tests repeat the check on every fold and inner block, and rebuild the link map from scratch: a fresh fingerprint generator against the frozen bench frame.

### Model (all arms identical in architecture)

**Main head (Ln selectivity residual).** Its target is block B's: for each of B1's training situations and each unread metal m, `t = y(m) − B1_S(m)`, with B1's masks and weights.
- **Prediction:** `level(m) = B1_S(m) + correction(m)` at unread metals; read metals keep their exact readings.
- **Inputs** are `[h(extractant), r]`. Here r is block B's metal / read-set / support-summary feature set **without** its chemistry columns (16 features, standardised on the fit's training rows, missing → 0).
- **Network:** Linear(24, 16) – tanh – Linear(16, 1).

**Shared encoder.** `h = E(z)`, where z is the 8 RDKit descriptors plus the donor class.
- **Descriptor scaling:** standardised on the fit's training Ln extractants; a scale below 10⁻⁶ is replaced by 1; standardised values are clipped to ±5.
- **Network:** Linear(11, 16) – tanh – Linear(16, 8) – tanh.

**Ln auxiliary head (raw log D; present in every arm).** Inputs are `[h, x(m), q(m)]`. Its targets are the raw log D of training Ln cells at measured metals, with row weight = the chemotype-balanced cell weight / the measured metal count.
- **Network:** Linear(10, 16) – tanh – Linear(16, 1).

**Private auxiliary head (the auxiliary domain's own head; present in every network).** S13, S15, S16 and S24 use it when w_X > 0. S11 never uses it, and S14 routes its An rows through the Ln head instead. Inputs are `[h, metal one-hot (4)]`. Its targets are the auxiliary raw log D, with row weight = 1 / (rows of the record's publication), so each publication weighs the same.
- **Network:** Linear(12, 16) – tanh – Linear(16, 1).

**Loss.** Every term is in units of its own target variance:
```
main weighted MSE / var(main) + w_Ln · Ln-aux weighted MSE / var(Ln-aux) + w_X · aux weighted MSE / var(aux)
```

**Training.**
- torch, CPU, float64, deterministic algorithms, one thread.
- Adam (lr 0.01; weight decay WD, applied to every parameter), 400 full-batch epochs.
- 3 initialisation seeds (0, 1, 2), with predictions averaged.

**Declared deviations from brief §6** (as in §11, plus one for block C):
- The main head is fitted on pointwise level residuals at unread metals. Only the selection uses pair loss.
- Row weights follow B1's convention (cell weight / masks), not the brief's per-cell normalisation.
- S13 tunes one more knob (w_X) than S11, so the brief's "preserve the tuning budget" is not met literally. Every control tunes the same knob, and the S15 / S16 rules are there to catch a gain from that knob alone.

### Selection (inner chemotype blocks; pair loss on inner-validation DOPT_Q masks; B1's β refitted per inner block, as in block B)

- **S11.**
  - **Candidates:** B1 (zero correction), plus the network for every w_Ln ∈ {0, 0.3, 1} × WD ∈ {0.03, 0.3}, with w_X = 0.
  - **Ties** go to B1, then to the smaller w_Ln, then to the larger WD.
- **S13 and the controls.**
  - **Candidates:** B1, plus the network with (w_Ln, WD) fixed at S11's best network and w_X ∈ {0, 0.3, 1}.
  - **Ties** go to B1, then to the smaller w_X.
  - **w_X = 0 is exactly S11's best network.** It is reused, not retrained: this is bit-identical by construction and tested. So S13 tunes one additional knob on identical Ln rows.
  - Every control tunes the same knob with the same grid, and its inner tuning is rerun for each arm and each randomised draw.
- **Outer fit.** The chosen configuration is refitted on all outer-training situations, with the fold's An set (outer exclusions only).

### Slots

| slot | auxiliary data and how it enters | role |
|---|---|---|
| **S11** | none (Ln auxiliary head only) | the common Ln-only multitask scaffold |
| **S13** | the fold's An(III) records through the **private** head | selective An transfer |
| **S14** | the same An records through the **Ln** auxiliary head as pseudo-lanthanides of equal f count (Pu→Sm, Am→Eu, Cm→Gd, Cf→Dy), with no private head; when w_X > 0 the Ln raw-log D term enters with weight max(w_Ln, 0.3), so the head is always shared with Ln data. Where S11's best network has w_Ln = 0, S14 therefore carries an Ln term that S13 lacks (declared; `r4_decide.py C` reports how often this happens and how often S14 falls back to S11's network) | deliberately naive shared-head control |
| **S15** | training-Ln raw log D rows through the private head, matched to the An set (below) | source / diversity / weight-matched auxiliary control |
| **S16** | An targets block-permuted across extractants (**5** seeded draws, seeds 20261323–27) | An-label permutation control |
| **S24** | An metal labels cycled inside each measurement series | metal-state-mapping shuffle |

**S12** (extra independent Ln) is NOT RUN: G-transfer's count gate failed, with 0 independent series (PROVENANCE.md).

**How the S15 rows are matched** (seed 20261423; training Ln only):
- the same number of weight units as the An set has publications;
- the same per-unit row counts, and equal unit weights;
- per unit, as many distinct extractants as the paired An publication has. Units are topped up from the next-ranked Ln publications when one publication has too few.
- The metal one-hot becomes four Ln metal bins.
- **Diversity is bounded by the fold's training-Ln extractant count** (36–74), which is below the An set's (77–106). S15 reaches 35–69 distinct extractants.
- What S15 cannot match is **novelty**: An extractants are mostly absent from the Ln training set. That novelty is the information S13 is meant to use.

**How an S16 draw is built.**
- Extractant blocks are deranged.
- Each An record takes a target sampled, with replacement, from its source block **at the same metal** where that metal exists, and otherwise from the whole source block.
- The metal composition and the per-metal target distribution are kept, and the extractant–target association is broken.

**How S24 is built.** Inside each measurement series (same publication, extractant, diluent, acid and extractant concentration), the series' own metal set is cycled in Am, Cm, Pu, Cf order, so each record of a multi-metal series carries another metal of the same series.
- **Why not a global relabelling.** A global relabelling of the one-hot (the brief's literal "mapping shuffle") is a pure reparametrisation of the private head and cannot act as a control.
- **S24's count gate.** S24 runs only if the cycle changes ≥ 10% of the fold's An rows in every one of the 50 fold-fits.
- **The threshold was fixed after the label-free counts were known.** No outcome information was used, because the counts are label-free.
- **The counts:** min 0.31%, median 4.0%, max 7.5%. The gate fails, and **S24 is NOT RUN**: the eligible An(III) set carries almost no metal-state information (83–93% Am per fold). Metal-state specificity is reported as **untested**.

### Gate G-transfer (§7; evaluated before any block-C outer score)

`scripts/r4_gate_c.py` uses training-side inner scores only, and predicts no held-out cell. It has two parts:
- **Count part:** the minimum over the 50 dev fold-fits of the post-exclusion An set must keep ≥ 5 independent publications (gen19 copy groups), ≥ 10 extractants and ≥ 100 rows. This **passes before sealing** (min 12 copy groups / 77 extractants / 557 rows).
- **Seal binding.** Shards and the combine step run only under the verified seal and carry its key. The combine step refuses shards that are mixed or incomplete (not exactly the 50 fold-fits). `r4_run.py` and `r4_decide.py` accept the gate and decision files only under the current seal.
- **Inner part:** in ≥ 30 of the 50 fold-fits, S13's best inner score over **positive** w_X must be below both:
  - S11's selected inner score (the minimum over B1 and the Ln-only networks);
  - B2's best inner score (as G-geo, §11).

If G-transfer fails, S13–S16 are NOT RUN and S11 is the only block-C slot scored.

### Run order and conditional routes (enforced in code: `r4_run.py` refuses an off-route block-C run; `r4_decide.py C` never makes an off-route S13 eligible; a run counts only if its intervention and identity checks passed, and `r4_run.py` writes its predictions only after both)

1. `r4_gate_c.py` (3 shards, then `--combine 3`) → `results/gate_c.json`.
2. **S11** on BPG and BP, always (the scaffold is a candidate in its own right).
3. **S13** on BPG and BP, only if G-transfer passes. Then run `report/r4_decide.py C`.
4. **Controls S14, S15 and S16** (both designs) run only if both:
   - S13 passes criterion 1;
   - S13 − S11 ≥ 0.005 (ΔR² vs B1) on BPG and on BP.

   S24 additionally needs its count gate, which fails (above).
5. **Fit poisoning** for every scored configuration (`r4_poison.py`), including the An poisoning below.

### Claim rules (on BPG and on BP; all quantities ΔR² vs B1 unless stated)

- **Criterion 1 and the harm rule** as in §10, for S11 and S13.
- **S11 passing criterion 1** is labelled "exploratory gain over B1 (Ln-only multitask scaffold; chemistry attribution untested)". The scaffold also carries support summaries, as in block B.
- **An(III) transfer claim (S13).** All of:
  - criterion 1 vs B1, plus criterion 2's vs-B2 part;
  - S13 − S11 ≥ 0.005;
  - S13 − S15 ≥ 0.005 and (S15 − S11) < 0.5 × (S13 − S11), §6.2's 50% reproduction rule as for S16;
  - S13 > max(S16 draws);
  - mean(S16 − S11) < 0.5 × (S13 − S11).

  Otherwise any S13 gain carries "An attribution not shown". While the on-route controls have not run yet, it carries "An attribution untested".
- **The label of a passing claim** reads "An(III) data improve held-out Ln log SF beyond Ln-only, matched and permuted auxiliaries (exploratory; a smooth-auxiliary regulariser is not excluded)".
- **Sub-labels, given the transfer claim:**
  - **private-head selectivity over naive sharing:** S13 − S14 > 0 on both designs;
  - **metal-state specificity:** S13 − S24 > 0 on both designs. S24 is NOT RUN, so this is "untested".
- **What the controls exclude, and what they do not.**
  - S15 excludes gains reproducible by reusing training-Ln targets with the An set's source structure.
  - S16 excludes gains from An targets whose extractant association is broken, which also covers tuning luck from the extra knob, since S16 tunes the same knob.
  - **Neither excludes a gain from fitting *any* smooth-in-descriptor auxiliary target at novel extractants** (a regulariser). The synthetic SMOOTH_NULL world measures how often that fires. A transfer label is therefore "An(III) data improve held-out Ln log SF beyond Ln-only, matched and permuted auxiliaries", not proof that An chemistry specifically transfers.
- **With 5 S16 draws,** S13 > max(S16) passes by chance with probability 1/6 per design under exchangeability.
- **Synthetic-power consequence** (pre-declared; `r4_decide.py C` reads the sealed `results/synthetic_c/summary.json`):
  - If the transfer claim fired in fewer than half of the synthetic AN_SIGNAL replicates, a non-positive S13 transfer result carries "control power not shown", with the synthetic count. That result is uninformative about An signals of the simulated size.
  - If the claim fired in any NULL or SMOOTH_NULL replicate, a positive label carries "not distinguishable from an irrelevant / smooth auxiliary".
- **An gains on An records are never evidence.** Only held-out Ln log SF counts (brief §6).
- **Finalist eligibility:** S11, and S13 when on route. S14, S15, S16 and S24 are controls and never eligible.
  - Failing a transfer control downgrades S13's label, not its criterion-1 eligibility. This is §10's reading of §6.6.
  - The §6.6 complexity order is B1 < S11 < S13.

### Amendment of the stage-0 §0 / §7 statements (declared before any block-C score; annotated in place)

- **Block membership.** S11 and S13–S16 are sealed together. S24 joins block C as its metal-state control, and is NOT RUN on its label-free count. S12 is NOT RUN on its gate.
- **G-transfer.**
  - "Independent publications" are gen19 copy groups.
  - The inner part's "model" is S13, compared against S11 and B2.
  - The controls are not gated separately: they run only on S13's route.

### Leakage

The chain is the same as REF's:
- isolation, plus the An isolation table above;
- fit poisoning on all 50 folds;
- the prediction intervention with methods rebuilt;
- support exactness;
- the per-fold slot identity.

**What fit poisoning does in block C.**
- The poisoned fit also sees the An record set with log D randomised for every record the fold's exclusions remove.
- `deep_equal` now compares torch tensors by value. The A4 review found that it previously treated any two networks as equal.
- The fit stores every network parameter as numpy as well.
- A unit test shows that a refit that ignores the exclusions is caught (positive control).

**What the slot identity checks.**
- A configuration whose inner choice is zero must equal B1 exactly.
- An arm whose outer choice is w_X = 0 on S11's chosen network must equal the S11 column exactly.

**Tests** (`tests/test_slots_c.py`) cover:
- the union exclusions on all 50 fold-fits and every inner block, copy groups included;
- identical Ln auxiliary rows in every arm;
- the control constructions;
- w_X = 0 reproducing S11 bit for bit;
- the descriptor-scale floor and clip;
- tensor-aware `deep_equal`;
- the outer refit's poison invariance with a leak positive control;
- determinism and support exactness.

### Discriminating power (brief §8.8; shown before sealing)

`scripts/r4_synthetic_c.py` (run 4, the sealed code) uses two BPG folds' real structure and the **real** An record set (rows, publications, metals, exclusions), with synthetic targets for both domains. Run 4's firing counts (`results/synthetic_c/summary.json`) drive the synthetic-power consequence above. It is summarised in A4.
- **Earlier runs** (record in A4): runs 1 and 2 used the pre-review scaffold. Run 3 used this design with the 1,574-row An set and without the S15 50% rule.
- **Latent properties** (standardised RDKit descriptors over cohort and An extractants; principal components scaled to unit sd):
  - u = tanh(1.5 · PC1) · sign(PC2);
  - s = tanh(1.5 · PC3).
- **The Ln signal** is 0.15 · u · tetrad_e3 / sd(tetrad_e3), on top of block A's nuisance structure and noise (sd 0.11).
- **The worlds are paired:** the same Ln targets in each replicate. An log D = 1.5 · level + metal offset (Am 0, Cm 0.1, Pu −0.3, Cf 0.2) + noise (sd 0.3), where the level is:
  - **AN_SIGNAL:** u;
  - **SMOOTH_NULL:** s, which is smooth in descriptors but irrelevant to the Ln task;
  - **NULL:** an independent per-extractant level.

### Budget

- **Slots:** block C uses at most 5 more of the 24 slots (S11, S13, S14, S15, S16); S24 is NOT RUN.
- **Campaign total:** at most 9 of 24.
- **Costs** are in RESOURCE_PLAN.md.
