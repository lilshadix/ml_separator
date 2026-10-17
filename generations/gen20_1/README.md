# gen20_1: absolute log D versus adjacent log SF on identical leakage-safe splits

**SEALED AND RUN — protocol sealed 2026-10-16 (S4); stages S5–S12 completed 2026-10-17.**

**Read this README, then `GEN20_1_REPORT.md` (results, verdicts by the sealed rules), `PRE_REGISTRATION.md`
(sealed hypotheses and decision rules), `IMPLEMENTATION_PLAN.md` (frozen plan), `ADDENDA.md` (dated operational
deviations) and `results/AUDIT_S12.md` (independent audit).**

**Status (2026-10-17).** Drafted 2026-10-15; on 2026-10-16 the user instructed to run without answering the open
items, so O1–O27 hold at their defaults (S2). S3 label-free counts reproduced every documented value; the
pre-registration was sealed (SEALED-SHA256 `f34a1b97…`, `manifests/seal_g201.json`); S5 target statistics, S6
byte-identical re-run, S7 gen20 anchors, S8 smoke, S9 discovery (6 designs × 25 units × primary and COH-NODMD,
300 units), S10 report, S11 one-shot confirmation (design BP, 5 new seeds) and S12 audit (no blocker; every number
reproduced independently) are complete. Nothing in this folder is committed to git.

## Results in one table (design BP, mode Z = zero-shot, gen20's C3 cells; seed means, discovery / confirmation)

| | absolute log D (variant A) | adjacent log SF from A (variant B) | adjacent log SF, direct (variant C) |
|---|---|---|---|
| target spread on the scored rows | sd 1.650 (85 % of variance between cells) | sd 0.222 | sd 0.222 |
| pooled R² | **0.258 / 0.233** | −0.010 / −0.011 | −0.016 / −0.042 |
| within-cell R² (log D only) | 0.275 / 0.282 | — | — |
| extractant-macro MAE (primary metric) | 1.042 (baselines 1.054–1.139) | 0.1665 (FLAT 0.1731, G14_ZERO 0.1612) | 0.1691 |
| R² ≥ 0.4 | not reached (both seed sets) | ruled out | ruled out |
| R² ≥ 0.5 | ruled out | ruled out | ruled out |
| beats every pre-named baseline | H1 INCONCLUSIVE (UNDERPOWERED) | H2 INCONCLUSIVE | H3 INCONCLUSIVE |

B and C are equivalent within ±0.02 log10 (H4 EQUIVALENT, confirmed). No H1–H3 verdict is confirmed. The pre-written
reading that applies is PAT-5: none of A, B or C was shown to beat its training-only baselines under design BP; this
does not show that either target is unpredictable. Why log D's R² does not carry over: 85 % of A's squared error is a
cell-level offset (MET-17 λ 0.854) that cancels exactly in log SF, and A's adjacent residuals correlate at 0.988, so
the level skill contributes nothing to adjacent selectivity. Under weaker hold-outs absolute log D reaches R² 0.352
(design B, no publication mask) and 0.410 (design A, extractant hold-out with near analogues), while adjacent log SF
stays at 0.03–0.09. Dropping the 5 corrupt DMDPhPDA cells changes no verdict. Full tables: `results/discovery/`,
`results/confirmation/`; every number is reproduced in `results/AUDIT_S12.md`.

Sources: path:line (or key), or **[CNT]** = design-time label-free counts made 2026-10-15 from the frozen inputs
(cell/metal/pair/fold membership and NaN masks only; no target value read); the counting script is not part of this
draft; every [CNT] value is recomputed at S3 by `scripts/g201_counts.py`, and a mismatch is a STOP. **[DMD]** (the
DMDPhPDA corrupt-copy rule applied to all 521 cells; every match is a C3 cell) and **[INNER]** (inner-fold
sizes), used in the other two documents, are design-time label-free counts of the same kind, recomputed at S3 by
the same script under the same STOP rule. "levels csv" =
`generations/gen20_kshot/results/discovery/levels_BP_metal_C3_DOPT_Q.csv`; RTF:n = line n of the brief.
Every ID used here (arms, baselines, cohorts COH-*, metrics MET-*, identities I*, rules RULE-*, claims H* and
T*, outcomes OUT*, stages S*, tests LT-*, tables TAB-*, open items O*) is defined in `IMPLEMENTATION_PLAN.md`
and used identically in all three documents; the decision rules are stated in full in `PRE_REGISTRATION.md`.

## Goal and research question

Which is easier and more reliable to predict on the same honest splits: absolute experimental log D or
adjacent log SF(a/b) = log D(a) − log D(b) (RTF:12-15)? Q1: how well and how reliably is log D per (cell,
metal) predicted for unseen chemotypes and publications? Q2: the same for strict-adjacent log SF (B, C).

| brief | English digest of the Russian brief, `generations/gen20.1/GEN20_1 INSTRUCTIONS.rtf` (commit 44df763; the RTF is authoritative) |
|---|---|
| RTF:17 | gen20's best main strict-adjacent R² was about 0.31 (RIDGE2, a k = 3 measured-mode value, see below); values of about 0.4 and above appeared only in modified settings and do not show 0.4–0.5 on the same strict task. Can an honest absolute-log-D setting reach 0.4–0.5, and what would that mean for predicting selectivity? |
| RTF:28-33 | Modify no gen20, gen13 or other frozen result; invent no measurements; never present synthetic data as experimental evidence; the brief gives the audit counts as known data (5,992 raw rows; 5,860 after the 129-row TODGA quarantine and 3 sentinel rows; 3,871 raw rows in ≥ 2-metal cells; 3,359 (cell, metal) observations; 521 cells; restated with sources below) and asks that its count of "normal" values not be called normal before its origin is checked; keep the chemotype hold-out and design BP; a high pooled log D R² can come from predicting the overall extraction level and does not by itself show good prediction of differences between metals; R² depends on target variance, so MAE, RMSE, target variance/sd, test size and the matching simple baseline stand beside it. |
| RTF:35-40 | Write only README.md, IMPLEMENTATION_PLAN.md and PRE_REGISTRATION.md in `generations/gen20_1/`, as unsealed drafts; seal and run nothing; state that results are not computed. |
| RTF:42-47 | On identical held-out data: A predicts absolute log D per (cell, metal); B derives adjacent log SF from A's out-of-fold predictions; C, direct contrast training if the code base allows it correctly, is a separate variant, never mixed with B. Same outer folds, seeds, test cells and observations; preprocessing, tuning, early stopping and model selection inside the training fold only. |
| RTF:49-65 | Define a primary metric per target; secondary metrics (pooled and macro MAE, RMSE, direction accuracy, pooled R², balanced/within-cell R²); baselines trained without test labels; how to compare R² across target variances; the bootstrap unit; replicates and noise; NA/negative R² rules; separate R² ≥ 0.4 / ≥ 0.5 reporting for log D and log SF. Do not tune the protocol to the threshold; distinguish in advance (1) the threshold on log D, (2) the threshold on log SF, (3) gain over baselines under the chemotype/publication hold-out. Flag conflicts and missing definitions with the most conservative solution. |

## What gen20 established, and why its 0.31 is measured-mode context only

| item | fact (source) |
|---|---|
| question | do k measured separation factors make an unseen extractant's adjacent log SF predictable? gen20 never predicted or scored absolute log D: a read metal reveals only within-cell differences (`generations/gen20_kshot/gen20k/design.py:423-428`) |
| verdict | measurements help; no model beats a straight line through them by the registered margin; the pre-registered neural arm is significantly worse than that line (`generations/gen20_kshot/README.md:10-12`) |
| the 0.31, exactly | RIDGE2 (a Model-3 baseline, exploratory even in gen20; gen20 `PRE_REGISTRATION.md:86-87,146`), metal model, k = 3 (4 read metals, 3 revealed differences; `GEN20_REPORT.md:51`), DOPT_Q supports, design BP, cohort C3, regime Q: seed-mean pooled R² 0.310, 95 % chemotype-blocked percentile CI [0.222, 0.401], 1,802 strict-adjacent pairs per seed on 229 cells (levels csv:73; `GEN20_REPORT.md:175`); 0.321 on the withheld confirmation seeds (`generations/gen20_kshot/results/confirmation/levels_BP_metal_C3_DOPT_Q.csv:73`) |
| its role in gen20.1 | gen20.1 registers only k = 0 (mode Z), so this k = 3 value (CTX_RIDGE2_K3) is measured-mode context: never a threshold reference or a comparator; in the report it appears only in TAB-14, never in a threshold sentence. The like-for-like k = 0 anchors on the identical COH-SF-C3 rows are G14@ZERO 0.0529, MEAN_CURVE@ZERO −0.1156 and FLAT −0.1347 (R²; levels csv:18, 30, 6) |
| values ≥ 0.35 | every one came from a changed measured-mode task, e.g. Nd–Sm added to the query (RIDGE2 k = 3: 0.43; `GEN20_REPORT.md:184`); the strict registered-setting maximum is 0.326 (RIDGE2, C6, k = 5; `generations/gen20_kshot/results/discovery/levels_BP_metal_C6_DOPT_Q.csv:108`); a CI upper bound is not attainment, and ΔR² values such as +0.426 (`GEN20_REPORT.md:154`) are not R²; TAB-14 lists representative values by task change (1,220 rows of gen20's `results/*/levels_*.csv` have R² ≥ 0.35, all measured mode, none in the strict-adjacent BP/metal/DOPT_Q setting) |

## Research area and scope

| item | gen20.1 choice (source) |
|---|---|
| research area | lanthanide solvent extraction: D is a metal's distribution ratio, SF(a/b) = D(a)/D(b). Adjacent metals differ by one in Z, so adjacent log SF is small against noise: sd (ddof 0) 0.222 log10 over the 1,802 C3 pairs (levels csv:73, `y_sd`) against a median replicate sd of 0.302 log D on 216 replicated (cell, metal) entries (gen20 `AUDIT.md:116`; a cited log D replicate statistic, not a log SF noise estimate: gen20.1 measures noise only at S5) |
| population | the frozen gen13 exact cohort: 521 cells, 90 extractants, 45 chemotypes, 58 publications (`generations/gen13_separation/DATA_AUDIT.md:39-41`); 3,359 replicate-mean (cell, metal) observations (`generations/gen13_separation/analysis/stage2/D5/FINDINGS.md:44`); provenance chain below |
| primary scored rows | fixed label-free before any fit, on the 229 C3 cells, 79 extractants, 38 chemotypes, 39 publications (`GEN20_REPORT.md:116`); C3 = gen20's cells with a non-empty strict-adjacent query and metal-model k_max ≥ 3, which at k = 0 only restricts scoring to metal-rich cells (5–14 measured metals each, [CNT]): COH-D-C3 = 2,425 (cell, metal) observations per seed ([CNT]); COH-SF-C3 = 1,802 strict-adjacent pairs per seed (levels csv:73). About four effective chemotypes: pair-weighted Kish 3.85 (levels csv:73); the largest chemotype (sc009) holds 0.49 of the pairs (`GEN20_REPORT.md:116`) |
| folds | gen20's `gen20k.splits.folds_for`: design BP (chemotype 5-fold; training cells sharing a publication with a held-out cell dropped) × split seeds 104729, 130363, 155921, 196613, 262147 (gen20 `PRE_REGISTRATION.md:12`), asserted before anything is scored against fold-plan SHA-256 `7046c640…` (`GEN20_REPORT.md:102`) and the further fold-identity gates of `IMPLEMENTATION_PLAN.md` §4 (train+test and per-design digests, gen20's 25 BP fold sizes, a new outer-plus-inner digest); any failure stops the run (`INCOMPARABLE_FOLDS`). Designs B, BR, BQ, BPG and A are sensitivities scored on the same rows (each cell is tested once per seed in every design; design A deals folds by extractant, so its rows are paired per (seed, cell, metal/pair), never per fold) |
| selection | nested per outer fold on a 4-fold inner chemotype cross-fit, with the publication mask (design BP) or copy-group mask (design BPG) re-applied inside each inner fold; designs B, BR, BQ and A have no inner mask; LEAN209 is fixed from gen13 (O26); preprocessing, weights and configuration choice never see an outer-test cell |
| metric, baselines, claims | primary for both targets: MET-01, extractant-macro MAE (log10). Baselines, all trained without test labels: BD_MEAN, BD_METAL, BD_COND, BD_NN (log D); FLAT, PAIRMEAN, G14_ZERO (log SF). Registered claims H1–H4 (Bonferroni K = 4, 98.75 % deciding intervals); threshold classes T1, T1w, T2, T3 (RULE-T); chemotype bootstrap (deciding) plus publication bootstrap (required) |
| out of scope | registered k > 0 analyses; new data or single-metal cells (O23); architectures beyond the fixed grid; changing cohorts, queries, metrics or thresholds, or dropping cells, after any result or S5 statistic; oracles or noise ceilings as ceilings; cross-target R² ranking; a registered Pr/Nd claim; real-world separation claims. Re-splitting the frozen corpus is not independent validation. No file outside this folder is modified |

## Variants and modes

| ID | what it is | trained on | primary rows | status |
|---|---|---|---|---|
| Variant A (arm `A`) | absolute log D per observed (cell, metal), uncentred: RowForest(target = raw), ExtraTrees on LEAN209 cell features + METAL_BASIC2; config K1–K4 per outer fold by inner MET-01 (grid TO BE FINALISED, O9) | all observed entries of the outer-training cells after the design mask (COH-TRAIN) | COH-D-C3 | registered: T1, T1w, H1 |
| Variant B (arm `B`) | adjacent log SF = Â(c, a) − Â(c, b), lighter metal a first, from A's out-of-fold predictions of the same fold model | derived from A: no refit, no re-selection, no recalibration | COH-SF-C3 | registered: T2, H2, H4 |
| Variant C (arm `C`) | direct contrast training: RowForest(target = centred) with A's learner, rows, X, weights, random_state and grid; y = log D − mean over the training cell's observed metals; config by inner MET-20; pair = ĉ(c, a) − ĉ(c, b) | COH-TRAIN, centred target | COH-SF-C3 | registered: T3, H3, H4; reported separately from B, never pooled |
| Mode Z | zero-shot, k = 0: no label of any outer-test cell enters any arm, baseline, selection, preprocessing, noise estimate or covariance | — | as above | the only registered mode |
| Mode M | measured mode, k = 3: gen20's metal model with gen20-identical DOPT_Q supports (4 read metals); a read metal reveals the test cell's level, so "log D is easier" would be an artefact | — | unread metals only | OFF; switched on only by the user's S2 answer to O4 (entered in the drafts before the seal) or, after the seal, by a dated addendum before any run; never pooled with Z; outside all claims |

## How results will be read

"R² ≥ 0.4 on log D" means only T1 = REACHED(0.4): A's seed-mean pooled R² (MET-06, about the scored rows'
unweighted mean) on COH-D-C3 is ≥ 0.4 **and** the 95 % chemotype percentile and BCa lower bounds, the 95 %
publication lower bound (O7), at least 4 of 5 per-seed values and R2_balanced all reach 0.4 (RULE-T). A
point alone is POINT_ONLY; CI upper bounds, ΔR², k-shot or secondary-population values never count. For scale:
gen20's 95 % chemotype interval (RIDGE2, k = 3) on the same log SF rows was [0.222, 0.401] around 0.310 (levels
csv:73), a half-width of about 0.09; at a similar width, REACHED(0.4) on log SF would need a point estimate of about
0.49 or more (arithmetic). The interval width for log D is unknown until results exist. T1 = REACHED(0.4)
would mean that, zero-shot under design BP, A explains at least 40 % of the variance of log D about its
mean on those rows for unseen chemotypes and publications. It would not mean selectivity is predicted:

| point | rule (source) |
|---|---|
| the level dominates (I1) | R²_pooled = s_b·R²_level + s_w·R²_within_cell. Every source agrees the between-cell level dominates log D variance (gen13 audit ratio 89/11 on mixed populations; gen6 92/8 on another cohort); the share on the gen20.1 rows is unmeasured (gen13 `DATA_AUDIT.md:63-64`; `docs/results/gen6_phase2_results_20260819.md:60`). With R²_within ≤ 0, a pooled τ needs R²_level ≥ τ/(1 − π) |
| the level cancels in log SF (I5) | B's pair error is r̃_a − r̃_b, so A's cell-level error cancels exactly and level skill adds nothing to log SF. Precedent: gen6's ORACLE_LEVEL, which reads the test cell's true level, had pooled log D R² 0.935 (split seed 104729; `runs/gen6_expC_5seed/arm_metrics.csv:8`), yet its derived pair macro MAE (0.503469, log10) is identical to that of the same shape model with a predicted level, C2_TRUECENTRE (0.503469; `runs/gen6_expC_5seed/decision_report.md:532,534`; `src/lanthanide_separation/gen6/hierarchical.py:496,504`): the level cancels exactly and bought nothing on pairs (MONO_ET 0.5189 at :528 for scale) |
| how T1 is printed | always with T1w (MET-08), the I1 decomposition, n, y_sd0/y_var0, MAE, RMSE and BD_METAL's values on the same rows; labelled "level-attributable" when BD_LEVEL's class ≥ A's (class order REACHED > POINT_ONLY > NOT_REACHED > RULED_OUT; if either class is NA, this clause prints "not evaluable (NA)" and the other two still apply), or R2_within_cell(A) ≤ 0, or ≤ BD_LEVEL_CURVE's; "reached also by <baseline>: the threshold alone carries no evidence of model skill" when any H1 baseline is REACHED; "log D prediction" is said only if H1 = PASS |
| three outcomes | OUT1 (T1 with T1w), OUT2 (T2 via B, T3 via C) and OUT3 (H1, H2, H3; H4 beside) are separate; OUT2 is never inferred from OUT1, and separation claims rest on the log SF table only |
| R² across targets | never ranked: R² = 1 − RMSE²/var0(y), so one RMSE gives very different R² at different target variance; each target gets a side-by-side card (TAB-07: n, y_sd0, y_var0, MAE, RMSE, named baselines, Skill_MSE; log D on COH-D-C3 and on the matched COH-D-C3M, O27). On COH-SF-C3, where FLAT scores R² −0.1347 (levels csv:6), R²_pooled 0.4 ⇔ R2_0 ≈ 0.471 and 0.5 ⇔ ≈ 0.559 (I3, arithmetic on that value); "easier" is stated only by the verdict-based sentences of `PRE_REGISTRATION.md` §1, and "more reliable" is never a verdict |
| expectations | pre-stated, no rule depends on them: T1 NOT_REACHED or RULED_OUT at 0.4, RULED_OUT at 0.5; T1w below 0.4; T2/T3 RULED_OUT at 0.4; H1 INCONCLUSIVE; H2/H3 do not PASS against G14_ZERO; H4 EQUIVALENT or NOT_ESTABLISHED. A null is a result |

## Layout (created during S3–S11)

```
gen201/       package: bootstrap.py (first), RowForest, baselines, metrics, inference, seal helpers
scripts/      g201_{counts (S3), describe (S5), discovery (S9), report (S10), confirm (S11)}.py,
              plus anchor (S7) and smoke (S8) entry points
tests/        LT-01..LT-30 leakage, validity and injection tests (synthetic arrays live only here)
manifests/    fold_plan_g201.json (S3), confirmation_seeds_sha256.txt (S4), run manifests
predictions/  float64 long tables per design and label, smoke*/ (gitignored; digests in manifests)
results/      result tables, confirmation/RUN_ONCE.json (S11)
GEN20_1_REPORT.md   written by the S10 report script, re-rendered by S11 with the confirmation
ADDENDA.md    dated operational deviations after the seal; results/AUDIT_S12.md: the S12 audit record
.gitignore    per-generation: predictions/**/*.parquet, results/**/*.parquet, smoke*/
```

## Conventions

| topic | rule |
|---|---|
| orientation | y(a/b) = log10 D(a) − log10 D(b), a the lighter metal (heavier-preferred ⇔ y < 0); heavier-first sources (recovery plan, gen19) are converted explicitly; a Nd/Pr ↔ Pr/Nd flip test checks invariance |
| adjacency | strict \|ΔZ\| = 1, 12 pair types, Pm absent (gen20 `README.md:51`); gen13's "adjacent" adds Nd–Sm and appears only as the labelled pass COH-SF-NDSM, never pooled with or compared against strict numbers |
| R² | per split seed, headline = seed mean; never clipped, floored, replaced by corr² or averaged across folds, extractants or pair types; NA with `R2_reason` when n < 2 or SST ≤ 0; the seed mean is NA if any seed is NA (`n_seeds_defined` printed); a non-finite prediction is a run failure, not an NA |
| direction | MET-14, log SF only, at \|y\| > 0, ≥ 0.1 and ≥ 0.3; a zero prediction scores 0.5; n_t and the number of exact-zero predictions are printed |
| units | log D, log SF, MAE and RMSE in log10; MSE and variance in log10². Every count names its unit (raw rows, (cell, metal) observations, cells, pairs per seed). Designs are always written "design BP", "design B", …, never confused with variants |

## The count of "normal values" mentioned in the brief

The brief asks that a count of "normal" values not be called normal before its origin is checked. That
count is not written anywhere in this repository except the brief itself
(`generations/gen20.1/GEN20_1 INSTRUCTIONS.rtf`, line 30; searched in md/py/txt/json/log/rtf files with
comma, space, no-break and thin-space separators). The nearest documented populations count different
things: (i) **4,881** averaged (extractant, 64-column condition, metal) level observations of the
gen5/gen6 BASE91 cohort, keyed without publication and without the TODGA-SMILES quarantine
(`docs/HANDOFF_FOR_CHATGPT.md:178`; 5,564 rows averaged to 4,881:
`runs/gen5_levels_20260818T211105Z/summary.json:10,14,15`; key (extractant, condition_id, metal_symbol):
`src/lanthanide_separation/levels.py:446-457`, `runs/gen7_architecture/ambiguity_audit.md:307`; quarantine never
called: same file :310); (ii) **4,746** rows of the external lanthanidestrain table whose 3D-geometry QC class is
"OK". This is a geometry flag, not a log D quality flag, and the count cannot be
reproduced here because that repository is not on this machine
(`generations/gen20_kshot/IMPLEMENTATION_PLAN.md:120`); (iii) **4,691** gen7 level rows in the gen10 "A
consistent" data-quality stratum (`runs/gen10_final/GEN10_ERROR_CEILING.md:72`). None of these is the
gen20.1 population. None is defined on the gen13 cell key with publication, and none is described here as
"normal". gen20.1 uses the frozen gen13 exact cohort: 5,992 raw bundle rows → 5,860 after the 129-row
TODGA-SMILES/name-mismatch quarantine and 3 sentinel rows (log D ≤ −6) → 521 cells (extractant SMILES ×
publication × 64-column condition key) with at least two lanthanides, holding 3,871 raw rows averaged into
3,359 replicate-mean (cell, metal) observations (`generations/gen13_separation/DATA_AUDIT.md:37-42`;
`generations/gen13_separation/analysis/stage2/D5/FINDINGS.md:44`;
`generations/gen13_separation/manifests/cohort_exact.parquet`). The primary scored subset is the 229 C3
cells: 2,425 (cell, metal) observations and 1,802 strict-adjacent pairs per seed. Open item O1: where did
the brief's figure come from, and what does it count?

## Open items (all resolved at their defaults at S2, 2026-10-16; full list in `IMPLEMENTATION_PLAN.md` §11 and `PRE_REGISTRATION.md` §13)

**The most consequential items** (resolved at the defaults shown; any later change would be a dated addendum):

1. **O4** mode M (k = 3, gen20's measured mode). Default: OFF; only k = 0 is registered, so gen20's 0.31 stays context.
2. **O13 + O14 + O27** primary rows. Default: C3 for both targets; TAB-07 also prints log D on the matched COH-D-C3M.
3. **O7** RULE-T bounds at 95 % or 98.75 %. Default: 95 % (less strict than 98.75 %); 98.75 % printed.
4. **O8** G14_ZERO inside the log SF baseline set of H2/H3. Default: inside.
5. **O10 + O11** definition of C. Default: the centred-row arm, selected on its own target (inner MET-20).
6. **O5** confirmation. Default: one-shot on new seeds, design BP only.
7. **O26** LEAN209 disclosure. Default: keep LEAN209 fixed; every verdict carries the disclosure label.
8. **O1 / O2** origin of the brief's "normal values" figure (default: never used); language (default: English).

The full list of O1–O27 follows.

- [x] **O1** Origin and meaning of the brief's "normal values" count. Default: never used.
- [x] **O2** English or Russian documents. Default: English plus the digest above.
- [x] **O3** Keep `gen20_1/` beside `gen20.1/`, add a topic suffix or move the RTF; when to add index rows? Default:
  keep both; index rows after approval.
- [x] **O4** Include mode M (k = 3, DOPT_Q, C3) and its RIDGE2 k = 3 anchor. Default: OFF; switched on only by the
  user's S2 answer to O4 (entered in the drafts before the seal) or, after the seal, by a dated addendum before any run.
- [x] **O5** One-shot confirmation on new seeds; BP only or six designs. Default: registered, BP only.
- [x] **O6** Margin δ_Y = max(0.02, 0.10·MET-01(Y)) instead of gen20's fixed 0.02. Default: max().
- [x] **O7** RULE-T bounds at 95 % or 98.75 %; its co-conditions; the class printed when POINT_ONLY and
  RULED_OUT both apply. Default: 95 %, all six (less strict than the 98.75 % alternative); 98.75 % printed; that
  precedence TO BE FINALISED (conservative proposal: POINT_ONLY, never counted as attainment, both percentiles printed).
- [x] **O8** G14_ZERO and PAIRMEAN inside the H2/H3 IUT or reported only. Default: inside.
- [x] **O9** Learner grid. Default: ExtraTrees, LEAN209 + METAL_BASIC2, K1–K4 (TO BE FINALISED); LEAN209's origin: O26.
- [x] **O10** Select C on its own target (inner MET-20) or on inner adjacent log SF. Default: own target.
- [x] **O11** Primary C: centred row (tested in gen13) or the never-run pair arm. Default: centred row.
- [x] **O12** Inner selection: 4-fold inner cross-fit or gen20's single split. Default: 4-fold.
- [x] **O13** Primary log SF rows: C3 or all strict-adjacent pairs. Default: C3.
- [x] **O14** Primary log D rows: C3 entries or every entry of the 521-cell cohort. Default: C3.
- [x] **O15** Drop the DMDPhPDA corrupt-copy cells from the primary. Default: keep; COH-NODMD refit (keeps gen20
  comparability; biases log D results, and the 5 M Pr/Nd pair, toward larger error).
- [x] **O16** Run-paired replicates for log SF noise. Default: checked at S3, else "not estimable".
- [x] **O17** Descriptive R² noise ceilings. Default: not computed.
- [x] **O18** Compute budget against the six-design requirement. Default: per-design runs; never drop one; a
  design never run gives INSUFFICIENT_COMPUTE when RULE-C (a)–(g) hold on design BP, otherwise (h) counts as failed.
- [x] **O19** Seal and addenda mechanism; whether `IMPLEMENTATION_PLAN.md` is frozen with the pre-registration.
  Default: gen20-style last-line digest (LF), ADDENDA.md; plan freeze TO BE FINALISED (conservative proposal: the
  plan's LF-normalised digest in the seal manifest, later changes only by dated addendum).
- [x] **O20** Estimand-consistency clauses required or annotations only. Default: required (a failure gives
  ESTIMAND_INCONSISTENT).
- [x] **O21** Wild-cluster-t threshold 0.0125 or gen20's 0.05. Default: 0.0125.
- [x] **O22** A separate registered Pr/Nd claim. Default: descriptive (MET-24) plus a PASS annotation.
- [x] **O23** Single-metal training augmentation for A. Default: not run.
- [x] **O24** Estimand and mode meant by the 0.4 / 0.5 thresholds. Default: k = 0, pooled + within R².
- [x] **O25** Read local gitignored gen13 predictions for continuity. Default: if present; LT-13 gates.
- [x] **O26** LEAN209 was chosen in gen13 on outer BP scores over these same folds
  (`generations/gen13_separation/analysis/stage2/STAGE2_REPORT.md:95-110`). Keep it fixed, or select block sets inside
  each training fold? Default: keep LEAN209 fixed (no gen20.1 outer choice, no added degrees of freedom); every H and T
  verdict and the confirmation label carry "feature set fixed from gen13 outer results on these folds".
- [x] **O27** Log D rows for cross-target statements: all C3 entries (COH-D-C3, 2,425) or only those underlying the
  scored log SF pairs (COH-D-C3M, 2,208 [CNT]), per the brief's matched held-out data (RTF:42)? Default: both printed
  in TAB-07; H1, T1 and T1w stay on COH-D-C3.

## Relationship to other folders

- `generations/gen20.1/`: only the brief (commit 44df763), left untouched; this folder is `gen20_1/`, the
  path the brief names (RTF:35). Index files are not edited now (O3).
- `generations/gen20_kshot/` (frozen): outer folds, strict-adjacent query, k = 0 anchors and inference
  code are imported read-only, never its writers. `generations/gen13_separation/`: cohort manifest and
  learner designs, read-only. The recovery plan allows R²(log D) only in an appendix
  (`docs/protocols/lanthanide_separation_recovery_plan_20260927.md:192`); the later brief governs, so
  variant A gets its own table ("how predictable is absolute log D"), never the separation-success table.
