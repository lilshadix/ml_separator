# gen20_1 — pre-registration: absolute log D versus adjacent log SF on identical leakage-safe splits

**SEALED PROTOCOL (S4) — approved for launch (S2); sealed after the S3 label-free counts and before any target
statistic (S5) or outcome-bearing run.**

*Written 2026-10-15 at HEAD `44df763` (main) as draft 1, after `IMPLEMENTATION_PLAN.md` (draft 1). **Approval
(S2), 2026-10-16:** the user instructed in chat to run the comparison without answering O1–O27, so every open item
is resolved at its default as written in §13; the values the draft had left open (O5 seed values, O7 precedence, O9 grid, O16 feasibility, O18 reduction details, O19 mechanism, the exploratory-family size)
are stated in place below. **Sealing mechanism (O19, default):** the last line of this file is `SEALED-SHA256:`
followed by the SHA-256 of the LF-normalised UTF-8 text of every line above it; `manifests/seal_g201.json` also
records the LF-normalised digest of `IMPLEMENTATION_PLAN.md`, which is frozen with this file; after sealing
neither file is edited, and any change is a dated entry in `ADDENDA.md` stating what changed, why, and whether S5
statistics, smoke output or results had been seen. No gen20.1 target statistic, prediction or result existed when
this file was sealed; only label-free counts ([CNT], [DMD], [INNER], recomputed at S3) had been made.*

**Sources.** *Brief* = `generations/gen20.1/GEN20_1 INSTRUCTIONS.rtf` (commit 44df763), "brief :N" = RTF
line N. *[LEV]* = `generations/gen20_kshot/results/discovery/levels_BP_metal_C3_DOPT_Q.csv` (gen20: design
BP, metal model, C3, regime Q, 5-seed mean, 1,802 strict-adjacent pairs per seed). *[CNT]* = design-time
label-free counts made 2026-10-15 from the frozen inputs (cell/metal/pair/fold membership and NaN masks only; no
target value read) of rows, cells, pairs, clusters and fold sizes; *[DMD]* = the same kind of count of the cells,
entries and strict-adjacent pairs matched by the §2 DMDPhPDA rule; *[INNER]* = the same kind of count of masked
inner-training cells per inner fold (§2). The counting script is not part of this draft; every [CNT], [DMD] and
[INNER] value is recomputed at S3 by `scripts/g201_counts.py`, and a mismatch is a STOP. `gen20k/`,
`gen13sep/` = the gen20_kshot and gen13_separation packages; "recovery plan" =
`docs/protocols/lanthanide_separation_recovery_plan_20260927.md`. IDs are shared with `README.md` and
`IMPLEMENTATION_PLAN.md` (implementation detail); "design X" = hold-out design.

## 0. Seeds

| use | value | source |
|---|---|---|
| outer split seeds (discovery) | 104729, 130363, 155921, 196613, 262147: gen20's, hence gen20's folds | gen13sep/splits.py:24; generations/gen20_kshot/PRE_REGISTRATION.md:12 |
| model seed of every tree fit | `fold.model_seed` = 42 + fold·1009 + 9,999,991 (fold index only, so A, C and every RowForest share it); A_PRIME uses + 1 | gen13sep/splits.py:27-29,58-60 |
| inner chemotype split | `inner_seed` = seed·31 + fold·7919 + 17 | gen20k/splits.py:31-33 |
| chemotype bootstrap | 10,000 draws, seed 8675309 | gen20k/inference.py:29-30 |
| publication bootstrap | 10,000 draws, seed 20260930 (the recovery plan's seed; its 2,000 draws raised, §12) | recovery plan :101, :302 |
| wild-cluster-t | 9,999 reps, seed 8675309 | gen13sep/wildcluster.py:207-208 |
| confirmation split seeds | 215593, 276871, 348239, 385670, 915723 (rule below; computed at S4, commitment sha256 `a2bb89f77fdc4be9f0e7ac186517d9d92cb6e9c406bc9c207bdb2850865f23bd`, design-BP outer+inner fold digest `f6d7bd70460f6b35da9b60d68723016e3883578800c2352a100b86c043148841`) | O5; `manifests/confirmation_seeds.json` |

No other random generator is used: BD_NN averages ties, and C_PAIR does not subsample (max_pairs_per_cell
None). **Confirmation seed rule (new; registered by default, open item O5).** gen20's
confirmation seeds 148569, 228670, 253534, 561551, 978178 are spent
(generations/gen20_kshot/results/confirmation/RUN_ONCE.json:2; GEN20_REPORT.md:320) and never reused.
Proposed: for i = 1, 2, …, s = int(sha256("gen20_1-confirmation-seed-{i}")[:8], 16) % 900000 + 100000,
skipping discovery seeds, gen16 confirmation seeds (137171, 506411, 721632, 937310, 939478;
generations/gen20_kshot/scripts/g20_confirm.py:29), gen20's seeds (re-derived at S4 from its string
"gen20-confirmation-seed-{i}", g20_confirm.py:34-41, never importing that script) and accepted seeds; the
first five, sorted, are computed at S4 (not now) and committed in `manifests/confirmation_seeds_sha256.txt`.

## 1. Questions

**Primary** (brief :12, :42-45): which of absolute log D and strict-adjacent log SF is easier and more reliable
to predict, on the frozen gen13 exact cohort, gen20's outer folds, design BP (chemotype hold-out; training cells
sharing a publication with a held-out cell removed), mode Z:
- **Q1, absolute log D.** How accurately and reliably does variant A predict log D of every observed (cell,
  metal) of a held-out cell; does it beat every pre-named log D baseline (H1); does it reach R² 0.4 / 0.5
  (T1 pooled, T1w within-cell)?
- **Q2, strict-adjacent log SF.** How accurately is y(a/b) predicted by differencing A's out-of-fold
  predictions (variant B) and by direct contrast training (variant C), each against every pre-named log SF
  baseline (H2, H3) and against τ (T2, T3); do B and C differ (H4)?
- **"Easier and more reliable"** (brief :12) is answered only by these pre-stated sentences, never by ranking,
  differencing or normalising R² or any other cross-target statistic (§5). "Absolute log D was shown to be
  predictable beyond its pre-named baselines and strict-adjacent log SF was not" is written only when H1 = PASS
  and neither H2 nor H3 = PASS; the converse only when H2 or H3 = PASS and H1 ≠ PASS; in every other case
  "neither target was shown to be more predictable than the other beyond its baselines". Each such sentence
  prints all verdicts in full and states that the baseline sets differ: the log SF set contains the prior model
  G14_ZERO, the log D set no prior model. "More reliable" is never a verdict; the MET-23 panels of both targets
  are printed side by side with no ranking.
- **Secondary** (exploratory): error cancellation from A to B (MET-17, I5); level and within-cell shares of
  A's R² (I1, MET-09); selection vs target effects (EX-01..EX-04); the publication-leak price R2_pooled(A)
  under design B minus under design BP; six-design sign agreement; EX-09, EX-10, MET-24.

## 2. Cohort, target, hold-out

- **Cohort (COH-FROZEN)**, loaded by `gen20k.data.load_bench(check=True)` with the plan's assertions
  (equality with `generations/gen13_separation/manifests/cohort_exact.parquet`, sha256 `3f11ac4c…`; bundle
  `fefbefc6…`; a failure is a STOP): 5,992 raw bundle rows → 5,860 after the 129-row
  TODGA-SMILES/name-mismatch quarantine and 3 sentinel rows with log D ≤ −6
  (generations/gen13_separation/DATA_AUDIT.md:37); 521 cells with ≥ 2 lanthanides, cell = canonical SMILES ×
  publication_id × 64-column condition key (:38-39); 90 extractants, 45 chemotypes, 58 publications
  (:40-41); 3,871 raw rows in those cells (:42), averaged into 3,359 replicate-mean (cell, metal)
  observations (generations/gen13_separation/analysis/stage2/D5/FINDINGS.md:44). The brief's "normal
  values" figure is not used as any population (O1; `README.md`).
- **Targets.** log D = frozen arithmetic replicate mean of log10 D per (cell, metal). y(a/b) = log D(a) −
  log D(b), a the lighter metal, one cell (gen20k/data.py:88-92); heavier-preferred ⇔ y < 0. Strict
  adjacency |ΔZ| = 1 (12 pair types, Pm absent); Q_adj = every strict-adjacent pair of a cell, lighter first.
- **Primary scored populations** (label-free, seed-invariant; row manifests at S3; every arm must cover
  exactly these keys with n_pred_finite == n_rows, else the run fails):
  - **COH-D-C3** (log D): every observed (cell, metal) entry of the 229 C3 cells: 2,425 entries per seed;
    79 extractants, 38 chemotypes, 39 publications; Kish (chemotypes, rows) 3.430; 5–14 metals per cell [CNT].
  - **COH-SF-C3** (log SF): Q_adj pairs of the same cells: 1,802 pairs per seed; Kish (chemotypes, pairs)
    3.8511214634053776 ([LEV]:6); sc009 holds 0.489 of the pairs ([CNT];
    generations/gen20_kshot/results/discovery/cohort_counts_BP_metal.csv:5, largest_chemotype_share;
    GEN20_REPORT.md:116, :126-127).
  - C3 = cells with non-empty Q_adj and metal-model k_max ≥ 3 (gen20k/report.py:69); at k = 0 this only
    restricts scoring to metal-rich cells. It is primary for continuity with gen20's headline metal-model k = 0
    anchors ([LEV]:6,18,30,50,62). gen20 also has k = 0 anchors on the pair-model C3 cohort (275 cells, 1,980
    strict-adjacent pairs; generations/gen20_kshot/results/discovery/levels_BP_pair_C3_DOPT_Q.csv:6,18) and on
    the metal-model C5/C6 subsets; it has none on COH-SF-ANY or COH-SF-C1, and none on absolute log D (O13,
    O14). Both tables share the chemotype (38) and publication (39) lists: one draw matrix per unit (LT-17).
- **Secondary populations and sensitivities** (identical for all arms, labelled; none decides a claim,
  fulfils OUT1–OUT3 or replaces the primary; counts [CNT] unless stated): COH-D-C3M, C3 entries whose metal
  is in ≥ 1 scored Q_adj pair (2,208 entries); COH-D-ALL, all 3,359 entries (cannot fulfil OUT1);
  COH-SF-ANY, all strict-adjacent pairs (2,114 pairs, 378 cells); COH-SF-C1, k_max ≥ 1 (2,067 pairs, 331
  cells; GEN20_REPORT.md:114); COH-SF-NDSM, gen20's Nd–Sm pass: gen13-adjacency pairs (strict + Nd–Sm) on the
  cells whose `'adjacent_gen13'` query (gen20k/design.py:62-63) is non-empty with metal-model k_max ≥ 3 under
  it, hence not the 229 COH-SF-C3 cells (1,984 pairs, 236 cells, 39 chemotypes;
  generations/gen20_kshot/results/discovery_nd_sm/levels_BP_metal_C3_DOPT_Q.csv:2; GEN20_REPORT.md:182;
  overlap with COH-SF-C3 counted at S3), own cluster lists, a labelled pass never compared numerically with
  strict numbers; COH-SC009 / COH-NOSC009 (129 cells, 1,270 entries, 882 pairs / 100 cells, 1,155 entries, 920
  pairs); COH-NODMD, minus the 5 DMDPhPDA corrupt-copy cells, a REFIT under all six designs on the same Fold
  objects (folds never re-dealt; the 5 cells removed from every outer and inner training set and from scoring)
  (C3: 224 cells, 2,355 entries, 1,742 pairs; arithmetic on [CNT], [DMD]); COH-REPCAP, scoring only, dropping
  entries with nrep ≥ 2 and repsd > 0.5 and pairs containing one (definition sealed; its count is
  label-derived and produced at S5, after the seal).
- **Training data (COH-TRAIN)**: all observed (cell, metal) entries of the outer-training cells after the
  design mask, non-C3 cells included, identical for A, C and every RowForest baseline (design BP: 76–492 cells,
  725–3,084 log D rows per fold [CNT]); no single-metal cells (gen13sep/cohort.py:51; COH-AUG not run, O23).
- **Hold-out.** Design BP alone selects, decides and is headlined; designs B, BR, BQ, BPG and A run the full
  arm and baseline set with their own nested selection on the same scored rows (RULE-C (h)). One
  `gen20k.splits.folds_for` call per design, one Fold object for every arm, and the plan's fold-identity gates
  (gen16 digests `7046c640…`, `9a24b6e9…`; generations/gen16_leads/results/anchors/anchors.json:31-40; 25/25
  BP fold sizes equal gen20's `folds_BP.json` and design B/BR/BQ/BPG test sets equal design BP's [CNT]; a new
  outer and inner digest) at S3 and in every outcome-bearing script; a failure is STOP `INCOMPARABLE_FOLDS`.
  Variants A, B and C and every baseline therefore share the outer folds, the split and model seeds (§0), the
  test cells and the available training observations (brief :47); arms and baselines of the same target share
  identical scored rows. A is scored on every observed entry of the 229 C3 cells (COH-D-C3, 2,425); B and C on
  their strict-adjacent pairs (COH-SF-C3, 1,802), built from exactly 2,208 of those entries (COH-D-C3M,
  TAB-09(a)); 217 log D entries have no scored log SF counterpart (O14, O27). B adds no fit of its own.
- **Inner data (COH-INNER).** Per outer fold, `seeded_group_kfold(chemotype[train_f], 4, inner_seed(seed,
  fold))` (gen20's split, gen20k/cov.py:75-90) with `gen20k.cov.inner_mask` re-applied, in every design
  (design A included, regrouped by chemotype; its stored extractant-grouped `Fold.inner_*` is never used). An
  inner fold with < 20 masked inner-training cells is skipped and logged (gen20k/cov.py:84): under BP 1 of
  100 (seed 155921, fold 2, inner fold 3, 18 cells), under design B none (minimum 59) [INNER]. If every
  inner fold of an outer fold is skipped, K1 is used and the case is logged.
- **Inputs, weights, DMDPhPDA.** LEAN209 cell features and METAL_BASIC2 = (Z, CN8 Shannon radius), behind
  the gen13 allow-list and the plan's deny list; a test cell contributes only its feature row, and its
  measured metals only select scored rows. Row weight w_c/n_obs(c), w_c chemotype-balanced over each fit's
  own training cells (gen13sep/models.py:257-261). The DMDPhPDA corrupt copy stays in the primary (O15),
  identified by SMILES `CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1` AND `cond__diluent__ch3cl == 1` (5
  cells, 70 entries, 60 strict-adjacent pairs, all in C3 [DMD]).

## 3. Modes

- **Mode Z (k = 0): primary and only registered mode.** No label of any outer-test cell (log D, contrast or
  replicate statistic) enters any arm, baseline, selection, preprocessing, noise estimate or covariance
  (LT-11, masked-label invariance).
- **Mode M (k = 3; gen20 metal model, DOPT_Q, 4 read metals): OFF** (O4). A read metal reveals the cell's
  level, so an offset update raises A's log D R² mechanically and leaves B unchanged on pairs without a read
  endpoint ("log D is easier" would be an artefact). Switched on only by the user's S2 answer to O4 (entered in
  the drafts before the seal) or, after the seal, by a dated addendum before any run (O4 in
  `IMPLEMENTATION_PLAN.md`); gated on `RIDGE2|k3` = 0.30991136191292135 ([LEV]:73); never pooled; no claim (EX-15).
- **CTX_RIDGE2_K3**, the brief's motivating number (brief :17): RIDGE2, metal model, k = 3 (4 read metals),
  DOPT_Q, design BP, C3, regime Q: pooled R² 0.30991136191292135, 95 % chemotype CI [0.2221, 0.4006]
  ([LEV]:73); 0.3209589052649169 on gen20's confirmation seeds
  (generations/gen20_kshot/results/confirmation/levels_BP_metal_C3_DOPT_Q.csv:73). Context only (TAB-14):
  never a comparator, threshold reference or T sentence, never without its k = 3 measured-mode label.

## 4. Variants and arms

**4.1 Learner and grid (O9 default, fixed at S4).** gen13 `tree_pipeline`: SimpleImputer(median,
keep_empty_features=True) → ExtraTreesRegressor(400 trees, random_state = `fold.model_seed`), fitted on
training rows only (gen13sep/models.py:36-42); features LEAN209 + METAL_BASIC2. Configurations, shared by A,
C and BD_COND, in the simplicity order that breaks exact ties (|Δcriterion| < 1e-12 → lowest K): K1
(min_samples_leaf 8, max_features 0.3), K2 (8, 0.5), K3 (2, 0.3), K4 (2, 0.5; gen13 default). No other learner.

**4.2 Registered arms.** C is direct contrast training by identity I6 (its centred squared loss = all-pairs
within-cell contrast loss / n_c + n_c·(cell mean of ĉ)², a penalty on the cell mean of the predictions; that
mean drops out of every contrast, but the penalty still shapes the fit of a learner that cannot set each cell's
mean freely, so C is contrast training with a mean-zero regulariser; C_PAIR, EX-05, is the pure pairwise form).
It differs from B in its target (centred vs raw log D) and in its selection criterion (inner MET-20 vs A's
inner log D MET-01; O10), so its per-fold configuration may differ; C_ACFG (EX-04) isolates the target, and
EX-01..EX-03 separate target and criterion. C is reported separately and never pooled with B. If C fails its
smoke test it is "not run", H3 and H4 are `NA`, and no substitute is chosen after outcomes (O11).

| arm | variant | target; scored rows | fit | selection (per outer fold, inner data only) |
|---|---|---|---|---|
| `A` | A | absolute log D; COH-D-C3 | RowForest(target = raw) on COH-TRAIN, uncentred y | K1–K4 by inner MET-01 of log D over the pooled inner OOF rows; argmin; refit on the outer training fold |
| `B` | B | strict-adjacent log SF; COH-SF-C3 | none: Â_f[c, a] − Â_f[c, b] from A's selected configuration and fitted model | inherited from A; no re-selection, recalibration, shrinkage or offset (LT-15) |
| `C` | C | strict-adjacent log SF; COH-SF-C3 | RowForest(target = centred): A's rows, X, weights, random_state, grid; ỹ = y − mean over the training cell's observed metals; pair = ĉ[c, a] − ĉ[c, b] | K1–K4 by inner MET-20 (within-cell MAE of centred log D) (O10) |

**4.3 Selection rules fixed in advance.** Per outer fold: fit K1–K4 on each inner fold (masked inner-training
cells only), score the criterion on the pooled inner OOF rows, take the argmin, refit on the outer training
fold. No cross-fold vote (gen20's 13/25 majority nominated NNC@SELECTED, the worse of its two candidates and
the worst arm at k = 3 apart from the arithmetic-only PROJ: R² −0.128, extractant-macro MAE 0.174 against
0.135–0.157 for the others; GEN20_REPORT.md:130-133, :170-180), no 1-SE rule, no selection on R². All
configurations are refit on the outer fold only for the "not selectable" appendix (A_K*, B_K*, C_K*; TAB-13);
selection records are MET-22. Nothing (arm, configuration, baseline, population, metric, threshold) is chosen
on a gen20.1 outer score; the report code has no override. Disclosure: the fixed feature set LEAN209 is not
blind to these test cells. gen13 preferred the lean block set to COND+MASSACT and to all seven blocks on outer
extractant-macro MAE of all-pair log SF (C_DIRECT_ROW, M_SELECTED) under design BP, on these five split seeds
and folds (generations/gen13_separation/analysis/stage2/STAGE2_REPORT.md:95-110, :296-297). gen20.1 keeps
LEAN209 fixed and adds no feature-set choice (O26). The median imputer and the row weights are fitted on the
fit's own training (or masked inner-training) rows only. No learner uses early stopping: the tree fits have
fixed tree counts (400; C_PAIR 200), and gen20's NNC, the only gen20k learner that early-stops
(generations/gen20_kshot/gen20k/nnc.py:119), is not used; the brief's requirement that early stopping stay
inside the training fold (brief :47) therefore holds trivially.

**4.4 Baselines** (training-only; on the candidate's rows; fallback counts printed, > 50 % labelled; the
test mean is not a predictor, recovery plan :165). Also HEAVIER_ALWAYS (MET-14 yardstick, PAIRMEAN sign
beside), BL_D_INNER (per fold, the H1 IUT member with the lowest inner MET-01; BD_MEAN, BD_METAL and BD_NN
are refit on each masked inner-train set; BD_COND's inner score is the pooled inner-OOF MET-01 of the
configuration it selects in that outer fold, an optimistic best-of-four, stated) and BL_SF_INNER (per fold,
FLAT or PAIRMEAN, whichever has the lower inner log SF MET-01); both only in EX-11.

| ID | target | definition | role |
|---|---|---|---|
| BD_MEAN | log D | weighted mean of training log D (A's row weights) | H1 IUT |
| BD_METAL | log D | per-metal weighted training mean; metal absent from training → BD_MEAN | H1 IUT |
| BD_COND | log D | RowForest(raw) on COND64 + MASSACT8 + METAL_BASIC2 (no ligand blocks), K1–K4 nested by inner MET-01; K4 ≡ gen13 `V2_DIRECT@cond` | H1 IUT |
| BD_NN | log D | nearest training extractant(s) by ECFP Tanimoto (`frames["ECFP"] > 0.5`, ties averaged; gen20k/arms.py:45-76): its weighted mean at m, else its mean offset from BD_METAL + BD_METAL(m) | H1 IUT |
| BD_LEVEL | log D | A's per-fold configuration refit without metal columns; constant within a cell | level-attribution diagnostic; EX-08 |
| BD_LEVEL_CURVE | log D | BD_LEVEL(c) + MC(m), MC = training chemotype-balanced mean centred curve | level-attribution diagnostic; EX-07 |
| FLAT | log SF | ŷ = 0 | H2/H3 IUT; anchor |
| PAIRMEAN | log SF | gen13 `PairMeanArm` (chemotype-balanced) at strict-adjacent types; absent type → chemotype-balanced training strict-adjacent mean | H2/H3 IUT |
| G14_ZERO | log SF | gen20 G14 refit in every outer fold (`cov.make_ctx`, `arms.standard_arms()['G14']`, c[a] − c[b]) | H2/H3 IUT (O8); anchor |
| MEAN_CURVE_REF, NN_TANIMOTO_REF | log SF | gen20 MEAN_CURVE@ZERO, NN_TANIMOTO@ZERO refit in fold | anchors only, never gain references |

**Anchor gate (S7, before any gen20.1 outcome)**, COH-SF-C3, design BP: FLAT@ZERO R² −0.13471207313225375,
pooled MAE 0.16425914006377063 ([LEV]:6); G14@ZERO R² 0.05287231149042522, pooled MAE 0.1468610842819455
([LEV]:18); MEAN_CURVE@ZERO R² −0.11561984257982724 ([LEV]:30); NN_TANIMOTO@ZERO R² −0.3860763898872736
([LEV]:62). The gate applies to the seed-mean R2, MAE and MAE_macro_extractant columns of [LEV]:6,18,30,62
for all four anchors (MAE_macro_extractant: FLAT 0.1730711262478135, G14 0.16124988000838458, MEAN_CURVE
0.17608615746610712, NN_TANIMOTO 0.20993132119515084; `IMPLEMENTATION_PLAN.md` §9, S7); n_rows and n_cells must
match exactly. Seed means (full precision in [LEV]): ≤ 1e-9 "reproduced", ≤ 1e-6 "reproduced (float32 storage
or platform drift)", beyond that STOP; per-seed values are checked against [LEV]'s 5-decimal per-seed strings
at 6e-6 (half a unit of the 5th decimal plus the 1e-6 drift allowance; beyond that STOP and diagnose, as for the
seed means; gen20 cast predictions to float32 before scoring, gen20k/runner.py:335-337; per-seed strings,
gen20k/inference.py:225). LT-13 (gen13 continuity to 1e-12) is a hard gate; a failing leakage test
(LT-01..LT-30) invalidates the experiment.

**4.5 Exploratory and diagnostic arms.** `B_selSF`, `C_selSF` (raw / centred, selected on inner
strict-adjacent log SF MET-01), `C_ACFG` (centred, A's configuration), `C_PAIR` (gen13 `AntisymmetricPairArm`,
designs BP and B, after a passing smoke test; planned in gen13 "at reduced cost ... where time allows"
(generations/gen13_separation/PRE_REGISTRATION.md:82-83) but never run: no gen13 run artifact or metrics
table names it, computed: `grep -rl C_PAIR_ANTISYM generations/gen13_separation`),
`A_PRIME` (NC1), `BD_METAL_SF`, `BD_COND_SF`, `BD_NN_SF`, `BD_LEVEL_CURVE_SF`. No neural model, no new features.

## 5. Metrics

Per split seed s, headlined as the 5-seed mean with per-seed values; e = y − ŷ; l = extractant (L, n_l rows).

- **Primary metric for absolute log D and, separately, for adjacent log SF: MET-01, extractant-macro MAE
  (log10 units, seed mean)**: MAE_macro_ext,s = (1/L)·Σ_l (1/n_l)·Σ_{i∈l} |e_{i,s}| over COH-D-C3 entries
  (log D) or COH-SF-C3 pairs (log SF). Inference unit: per-extractant seed-mean MAE_l; Δ_l = MAE_l(Y) −
  MAE_l(X). It decides H1–H4 and A's selection: gen20's registered estimand
  (generations/gen20_kshot/PRE_REGISTRATION.md:125-133), one vote per extractant (limits sc009), robust to
  decade-level replicate errors; gen20 has FLAT 0.1730711262478135 and G14@ZERO 0.16124988000838458 on
  COH-SF-C3 ([LEV]:6,18); WCR on per-extractant deltas held 0.044–0.048 at nominal 0.05 in gen13's simulated
  BP_all profile (45 chemotype clusters, Kish 11.67, ICC 0.72;
  generations/gen13_separation/metrics/BP_all/inference_calibration.csv:9,27); it was not simulated at 0.0125
  or on C3's 38 chemotypes (Kish 3.43 rows / 3.85 pairs), so the size of the deciding test here is unverified.
- **Secondary** (MET-02..MET-24, formulas in `IMPLEMENTATION_PLAN.md`): pooled and chemotype-macro MAE,
  MAE_bal/RMSE_bal (MET-04, w_i = 1/(L·P_l·C_lp·N_lpc); P_l = publications of extractant l among the scored
  rows, C_lp = cells of l in publication p (a gen20.1 cell is the recovery plan's "series", :128-131), N_lpc =
  scored rows (log D entries or log SF pairs) of cell c; all recomputed per subset and draw), pooled RMSE, the R²
  family, direction accuracy (MET-14, log SF; |y| > 0, ≥ 0.1, ≥ 0.3; zero prediction = 0.5), within 1.5×/2×.
- **R² family.** MET-06 R2_pooled,s = 1 − Σ_i e_{i,s}² / Σ_i (y_i − ȳ)², ȳ the unweighted mean of the scored
  rows (gen20k `metrics.r2`; the estimand of CTX_RIDGE2_K3); seed mean = 1 − mean_s(SSE_s)/SST (I4).
  MET-07 R2_balanced = 1 − Σw_i e_i² / Σw_i (y_i − ȳ_w)² (MET-04 weights; within-cell variant on ỹ, ŷ̃).
  MET-08 R2_within_cell = 1 − Σ_c Σ_{i∈c}(ỹ_i − ŷ̃_i)² / Σ_c Σ_{i∈c} ỹ_i², y and ŷ centred over the cell's
  scored rows after prediction; BD_MEAN and BD_LEVEL score 0 up to rounding (|R2_within_cell| ≤ 1e-12;
  LT-24). MET-09 R2_level = 1 − Σ_c n_c(ȳ_c − ŷ̄_c)² / Σ_c n_c(ȳ_c − ȳ)², s_b = SST_between/SST, s_w = π =
  1 − s_b. MET-10 R2_0 = 1 − Σe²/Σy² (log SF skill vs FLAT). MET-11 Skill_MSE(X|Y) = 1 − SSE(X)/SSE(Y) on
  identical rows, a gain over a named baseline, not an R². MET-20 = MET-01 on |ỹ − ŷ̃|.
- **NA and negative R².** NA with reason when n < 2 or SST ≤ 0; every table and JSON carries `R2_reason`
  and `n_seeds_defined`. n_pred_finite == n_rows is asserted first; a non-finite prediction is a run
  failure, not an NA (gen20k/metrics.py:19-21 drops such rows silently). The seed mean is NA if any seed is
  NA. NA bootstrap draws are kept (−∞ for lower, +∞ for upper bounds, unfavourable in contrasts, counted);
  above 1 % NA draws the interval is NA ("degenerate") and conditions using it fail. R² is never clipped,
  floored, averaged across folds, extractants or types, or replaced by corr²; negative values stand.
  Skill_MSE is NA when the baseline SSE is 0.
- **R² across targets of different variance.** No cross-target test, ΔR²(A − B), variance normalisation or
  ranking (brief :33). R² = 1 − RMSE²/var0(y), so one RMSE gives very different R² on targets of different
  variance, and the reference matters (the same G14 predictions score 0.117 or 0.386; gen16 amplitude R², 289
  cells, design BP; generations/gen16_protocol/results/g16_oof_amplitude_r2_BP.csv:2). Each target gets a
  side-by-side card (TAB-07): the log D side is printed on COH-D-C3 and on COH-D-C3M (the entries underlying the
  scored log SF pairs), and any cross-target sentence (§1) cites both (O27); MET-13 (n, cells, extractants,
  chemotypes, publications, Kish, y_mean, y_sd0, y_var0 with ddof 0, y_rms0 for log SF, s_b and s_w for log D),
  R² with `R2_reason`, MET-01/02/03/05, Skill_MSE vs every named baseline, MET-12, MET-23, MET-17 with its CI,
  and the identities I1 (R²_pooled = s_b·R²_level + s_w·R²_within_cell), I2 (R²_within_cell is a
  FLAT-referenced all-pairs shape score, not adjacent), I3 (1 − R²_pooled = (1 − R²_0)(1 − R²_FLAT)), I4
  (seed-mean R² = 1 − mean_s(SSE_s)/SST) and I5 (B's pair error r̃_a − r̃_b cancels A's level error).
  Separation claims rest on the log SF table only. gen20's "y sd 0.236" (GEN20_REPORT.md:163) is y_rms0 (FLAT
  RMSE 0.23644, [LEV]:6); y_sd0 is 0.22196 ([LEV]:6). gen20.1 target statistics exist only from S5 on. Every
  printed R² or RULE-T class carries, on the same rows, n, y_sd0, y_var0, MET-01, MET-02, MET-05 and the simple
  baseline's R², MAE and RMSE (BD_METAL for log D, FLAT for log SF) (brief :33).

## 6. Inference

- **Units and draws**: chemotype (38 on C3; deciding) and publication (39; required co-condition), never
  extractant, cell or row, seeds not resampled; 10,000 draws per unit (gen20k/inference.py:102-108), one
  matrix per (unit, cluster list) shared by every arm, baseline, target and seed; each statistic recomputed
  per draw from per-cluster sums about the draw's own mean (gen20k/inference.py:34-62, applied per draw at :171
  and :215; the per-extractant delta means at :282-307).
- **Intervals**: chemotype percentile and BCa (gen20k/inference.py:65-86), publication percentile, each at
  95 % and 98.75 %. COH-SC009: chemotype interval NA ("one cluster"), publication interval (15 publications
  [CNT]). Fewer than 10 clusters → "preliminary" (recovery plan :304).
- **WCR**: `gen13sep.wildcluster.wild_cluster_t` (CR2, null imposed, Rademacher weights; Webb 6-point weights
  only when G < 12, which arises only in exploratory subsets; 9,999 reps, seed 8675309) on per-extractant
  seed-mean deltas clustered by chemotype (gen20k/inference.py:229-249); p_wcr decides. gen13's old
  pairs-percentile bootstrap over-rejected in the same simulation (0.071 at nominal 0.05;
  generations/gen13_separation/metrics/BP_all/inference_calibration.csv:2).
- **LOCO, seeds, MDE**: LOCO = the seed-mean statistic without each chemotype (contrasts need every finite
  value positive; NA fails); per-seed values and sign counts always printed (seed sd never an SE); mde80_95
  = 2.80·se_boot (gen20k/inference.py:145), mde80_dec = (z_0.99375 + z_0.80)·se_boot = 3.34·se_boot, mde80_cr
  = `detectable(se_cr2, dof_BM, alpha = 0.0125)` (gen13sep/wildcluster.py:396-403); UNDERPOWERED uses the
  larger of mde80_dec and mde80_cr.
- **Multiplicity**: K = 4 claims (H1–H4), Bonferroni α_K = 0.05/4 = 0.0125, deciding intervals 98.75 % (95 %
  descriptive except in RULE-T); intersection–union within each H without further correction (Berger 1982).
  T1/T1w/T2/T3 are classifications outside K. Exploratory p values get Benjamini–Hochberg q = 0.05.
  BASELINE_BETTER is a union over a claim's comparators and lies outside this error statement. Each named-Y
  test is one-sided at 0.00625 (two-sided 98.75 % bound and p_wcr < 0.0125), so a false BASELINE_BETTER for
  one H has probability at most |Y|·0.00625 (H1: 0.025; H2, H3: 0.01875).

## 7. Threshold reporting (τ = 0.4 and τ = 0.5, each separately)

- **RULE-T**, applied to T1 = A's seed-mean MET-06 and T1w = A's seed-mean MET-08 (log D, COH-D-C3), and T2
  = B's, T3 = C's seed-mean MET-06 (log SF, COH-SF-C3); thresholds are reported, not targeted (brief :60).
  `REACHED(τ)` iff (i) seed-mean point ≥ τ; (ii) chemotype 2.5th percentile of the 10,000
  seed-mean draws ≥ τ; (iii) chemotype 95 % BCa lower bound ≥ τ (NA fails); (iv) publication 2.5th
  percentile ≥ τ; (v) ≥ 4 of 5 per-seed values ≥ τ; (vi) seed-mean R2_balanced (within-cell variant for T1w)
  ≥ τ, required so that a crossing carried only by the pooled sc009 rows (0.489 of the C3 pairs, 0.524 of the
  C3 log D entries [CNT]) does not count as attainment. `POINT_ONLY`: (i) holds, any of (ii)–(vi) fails or is
  NA. `RULED_OUT`: chemotype AND publication 97.5th percentiles < τ. `NOT_REACHED`: (i) fails, not RULED_OUT.
  `NA`: seed mean NA. When (i) holds and both 97.5th percentiles are below τ, POINT_ONLY and RULED_OUT both
  apply; the class printed then is POINT_ONLY (O7 default, fixed at S4), never counted as attainment, with
  both 97.5th percentiles printed. Class order, used only by the level-attributable clause:
  REACHED > POINT_ONLY > NOT_REACHED > RULED_OUT; a case where POINT_ONLY and RULED_OUT both apply is ranked
  as the class printed under O7; if A's or BD_LEVEL's class is NA, the class clause is not evaluated ("not
  evaluable (NA)" is printed for this clause) and the two R2_within_cell clauses still apply. Sealed code
  classifies unrounded floats with no override; a value equal to τ satisfies "≥" (LT-23). The same class is
  printed for every baseline, reference and exploratory arm on the same rows; only A, B, C carry T-IDs. A CI
  upper bound, a ΔR², a k-shot value or a secondary-population value never establishes attainment. Bounds at
  95 % (O7), 98.75 % printed.
- **Mandatory annotations** (never change a class): first, the §5 context on the same rows (n, y_sd0, y_var0,
  MET-01, MET-02, MET-05 and the simple baseline's R², MAE and RMSE; brief :33); 98.75 % chemotype and
  publication lower bounds; LOCO minimum; COH-SC009 point; COH-NOSC009 point and interval; COH-NODMD point;
  τ + 1.96·se_boot. For T1: the I1 decomposition (s_b, s_w, R2_level, R2_within_cell) in the same sentence;
  "reached also by <baseline>: the threshold alone carries no evidence of model skill" when an H1 IUT member
  is REACHED; **level-attributable** when BD_LEVEL's class ≥ A's (class order and NA rule above), or
  R2_within_cell(A) ≤ 0, or ≤ R2_within_cell(BD_LEVEL_CURVE); "log D prediction" only if H1 = PASS. With
  R²_within ≤ 0 a pooled τ needs R²_level ≥ τ/(1 − π); π on gen20.1 rows is unmeasured until S5 (gen13's
  89/11 split, DATA_AUDIT.md:63-64, mixes two populations). For T2/T3: R2_0 and the I3 equivalents (with
  R²_FLAT = −0.1347 on these rows, R²_pooled 0.4 ⇔ R²_0 ≈ 0.471 and 0.5 ⇔ ≈ 0.559; arithmetic on [LEV]:6).
- **The three outcomes, distinguished in advance** (brief :60-63); none implies another. **OUT1, threshold
  on absolute log D** = T1 at each τ, always with T1w and I1 (a high pooled log D R² can come from the
  extraction level alone; brief :32). **OUT2, threshold on adjacent log SF** = T2 (via B) and T3 (via C),
  separately, never inferred from OUT1. **OUT3, improvement over baselines under the honest chemotype +
  publication hold-out** = H1 (log D), H2 (B), H3 (C), with H4 (C vs B) beside them.
- **Pre-written sentences** (other combinations: verdict by verdict, "pattern not anticipated"):

| ID | condition | sentence |
|---|---|---|
| PAT-1 | H1 PASS; H2, H3 not PASS | "Variant A was shown to beat every pre-named log D baseline for unseen chemotypes and publications (H1 PASS). B and C were not shown to beat every log SF baseline (H2: <verdict>, H3: <verdict>); this is not evidence that strict-adjacent selectivity is unpredictable." |
| PAT-2 | H1 and H2 PASS | "The log-D-then-subtract route predicts strict-adjacent selectivity beyond its baselines." |
| PAT-3 | H1 not PASS; H2 PASS | "Derived log SF (B) beats every pre-named log SF baseline although A was not shown to beat every log D baseline (H1: <verdict>); A's cell-level error cancels in the difference (I5)." |
| PAT-4 | H3 PASS; H2 not PASS | "Direct contrast training (C) predicts strict-adjacent log SF beyond its baselines; the derived route (B) was not shown to (H2: <verdict>)." (with H4) |
| PAT-5 | none of H1–H3 PASS (the pattern expected in §9) | "None of A, B or C (<learner, features and grid as sealed in §4.1>) was shown to beat its pre-named training-only baselines under design BP (H1: <verdict>, H2: <verdict>, H3: <verdict>). This does not show that either target is unpredictable." |
| PAT-6 | T1 REACHED; T2, T3 not REACHED | "Absolute log D reaches pooled R² ≥ τ with A (T1 REACHED; T1w: <class>; <level-attributable / not level-attributable>). Strict-adjacent log SF was not shown to reach τ with B or C (T2: <class>, T3: <class>), so the log D result is not evidence that selectivity is predicted." |
| H4 | C_BETTER / B_BETTER / EQUIVALENT / NOT_ESTABLISHED | "direct contrast training beats log D followed by subtraction on strict-adjacent log SF" / the converse / "the two routes are equivalent within ±0.02 log10" / "no difference could be established"; EX-01..EX-04 beside |

## 8. Registered claims and the exploratory family

**8.1 Registered claims** (K = 4; design BP, mode Z, seed mean; Δ positive favours the candidate). Registered
classifications outside K (RULE-T): T1 (A, MET-06, COH-D-C3), T1w (A, MET-08, COH-D-C3), T2 (B, MET-06,
COH-SF-C3), T3 (C, MET-06, COH-SF-C3), each at τ = 0.4 and 0.5.

| id | contrast | cohort | estimand | comparators (IUT) | decision rule |
|---|---|---|---|---|---|
| H1 | A vs every log D baseline | COH-D-C3 (2,425 entries per seed) | Δ_Y = MET-01(Y) − MET-01(A) | BD_MEAN, BD_METAL, BD_COND, BD_NN | RULE-C (a)–(i), every Y |
| H2 | B vs every log SF baseline | COH-SF-C3 (1,802 pairs per seed) | Δ_Y = MET-01(Y) − MET-01(B) | FLAT, PAIRMEAN, G14_ZERO | RULE-C (a)–(k), every Y |
| H3 | C vs every log SF baseline (separate from B) | COH-SF-C3 | Δ_Y = MET-01(Y) − MET-01(C) | FLAT, PAIRMEAN, G14_ZERO | RULE-C (a)–(k), every Y |
| H4 | C vs B, two-sided | COH-SF-C3 | Δ = MET-01(B) − MET-01(C) | — | RULE-H4 |

- **RULE-C** (component X vs Y; Δ = MET-01(Y) − MET-01(X)), all required: (a) Δ ≥ δ_Y = max(0.02,
  0.10·MET-01(Y)), the comparator's seed-mean outer value (O6; 0.02 for log SF against FLAT and G14_ZERO,
  as 0.10 × 0.1731 = 0.0173 and 0.10 × 0.1612 = 0.0161, arithmetic on [LEV]:6,18); (b) chemotype 98.75 %
  percentile lower bound above 0 (NA draws count ≤ 0); this condition decides. The two-sided percentile p (NA
  draws counted unfavourably) is printed; it can disagree with (b) only within one draw of the boundary; (c)
  chemotype 98.75 % BCa lower bound above 0 (NA fails); (d) p_wcr < 0.0125 with a positive mean (O21); (e)
  publication 98.75 % percentile lower bound above 0; (f) Δ_s > 0 in ≥ 4 of 5 seeds; (g) every LOCO Δ > 0;
  (h) point Δ > 0 under each of designs B, BR, BQ, BPG and A, each refitted with its own selection (gen13
  Addendum 3 rule 1, generations/gen13_separation/PRE_REGISTRATION.md:189-193); estimand consistency (recovery
  plan :270-277; O20): (i) point ΔMAE_bal > 0; (j) log SF only: point ΔR2_balanced ≥ 0.05 with chemotype and
  publication 98.75 % percentile lower bounds above 0 and the candidate's own R2_balanced above 0; (k) log SF
  only: Skill_MSE(X | FLAT) above 0.
- **Verdict for H1–H3**, in order: `NA` (candidate not run); `BASELINE_BETTER` (for a named Y the chemotype
  98.75 % upper bound < 0 AND p_wcr < 0.0125 with a negative mean); `INSUFFICIENT_COMPUTE` (a design needed
  for (h) not run while (a)–(g) hold for every Y on design BP; if some (a)–(g) fails on BP, that (h) counts as
  failed and the order continues to `INCONCLUSIVE` with its sub-labels); `PASS` (every applicable condition
  for every Y); `ESTIMAND_INCONSISTENT` (MET-01 gain established on design BP: (a)–(h) hold for every Y, but
  (i), (j) or (k) fails for some Y); `INCONCLUSIVE` otherwise, sub-labelled `ENCOURAGING` when every BP point
  Δ_Y ≥ δ_Y (recovery plan :277) and `UNDERPOWERED` when a failing component has max(mde80_dec, mde80_cr) >
  δ_Y; both sub-labels are printed when both apply (e.g. `INCONCLUSIVE (ENCOURAGING; UNDERPOWERED)`).
- **RULE-H4**, in order: `NA` (C not run); `INSUFFICIENT_COMPUTE` (a design needed for (h) not run while
  (a)–(g) hold on design BP for Δ with X = C or for −Δ with X = B; otherwise that (h) counts as failed, and
  `EQUIVALENT`, which uses design BP only, and `NOT_ESTABLISHED` are assigned as below); `C_BETTER` (RULE-C
  (a)–(k) for Δ, X = C, Y = B, δ = max(0.02, 0.10·MET-01(B))); `B_BETTER` ((a)–(k) for −Δ, X = B, Y = C,
  δ = max(0.02, 0.10·MET-01(C))); `EQUIVALENT` (the chemotype 98.75 % percentile interval and the publication
  98.75 % percentile interval of Δ both inside (−0.02, +0.02), i.e. TOST at 0.00625 per side); otherwise
  `NOT_ESTABLISHED`, `UNDERPOWERED` if max(mde80_dec, mde80_cr) > 0.02. `NOT_ESTABLISHED` is the H4 label only.
- **Fixed annotations** (never change a verdict): pooled MAE Δ, MAE_macro_chem Δ, Skill_MSE with its
  98.75 % lower bound, ΔR2_pooled; "passed at low power; effect likely overestimated" on a PASS where
  max(mde80_dec, mde80_cr) > δ_Y for some component; "not robust to known corrupt level data" if the COH-NODMD
  verdict differs; for an H2/H3 PASS the recovery-plan item-4 Pr/Nd status (:275): "verified" (98.75 % upper
  bound of MAE_PrNd(X) − MAE_PrNd(Y) ≤ 0.02 for every Y), "failed" (point > 0.02) or "not verified", only
  "verified" allowing "full success"; the confirmation status (§11); on every H and T verdict and on the
  confirmation label, "feature set fixed from gen13 outer results on these folds" (O26). **NC1**: A vs A_PRIME
  on COH-D-C3 through RULE-C (b)–(g), without the margin (a), in both directions. A seed-only change cannot
  plausibly move MET-01 by δ_Y, so (a) would make the control vacuous. If either direction meets (b)–(g),
  every PASS of H1–H4 carries "inference flagged by negative control". NC1's Δ, bounds and p_wcr are printed.
- **Exploratory family** (fixed at S4: EX-15 is not run because O4 is OFF and is excluded; one test per (EX,
  population, named comparator), so m = 83: EX-01..EX-05, EX-07 and EX-08 one each (7); EX-06 three; EX-09 40
  (H1's 4 components on COH-D-C3M and on COH-D-ALL; H2's 3, H3's 3 and H4's 1 on each of COH-SF-ANY, COH-SF-C1
  and COH-SF-NDSM; all 11 on COH-NOSC009); EX-10 22 (all 11 under each of COH-NODMD and COH-REPCAP); EX-11
  three; EX-12 two (gains over the COND pair and over the NN pair); EX-13 two (both directions); EX-14 four;
  including every EX whose arm depends on an S8 smoke test (EX-01, EX-03, EX-04 and the C contrasts within EX-09, EX-10
  and EX-11 on the centred learner; EX-05, EX-06 on C_PAIR); an EX whose arm is "not run" after S8 is reported
  NA and enters BH with p = 1, m unchanged; never promoted):
  EX-01 C_selSF vs B; EX-02 B_selSF vs B; EX-03 C_selSF vs B_selSF; EX-04 C_ACFG vs B; EX-05 C_PAIR vs B;
  EX-06 C_PAIR vs FLAT, PAIRMEAN, G14_ZERO; EX-07 A vs BD_LEVEL_CURVE on MET-20; EX-08 A vs BD_LEVEL on
  MET-20; EX-09 H1–H4 contrasts on COH-D-C3M, COH-D-ALL, COH-SF-ANY, COH-SF-C1, COH-SF-NDSM, COH-NOSC009;
  EX-10 the same under COH-NODMD and COH-REPCAP; EX-11 A vs BL_D_INNER, B and C vs BL_SF_INNER; EX-12
  Spearman of per-extractant gains of A (over BD_COND/BD_NN) and B (over BD_COND_SF/BD_NN_SF), read only if
  each has Spearman–Brown split-half reliability ≥ 0.3; EX-13 NC1; EX-14 BD_*_SF vs FLAT; EX-15 MODE_M.
- **Compute rule.** One process, designs in the order BP, B, BR, BQ, BPG, A, resumable. The planning
  estimate (`IMPLEMENTATION_PLAN.md` §9: the primary fits plus the COH-NODMD refit of every design, ≈ 280–300 s
  per (design, seed, fold), ≈ 12 h for the 150 units and ≈ 2 h for design BP alone, arithmetic on gen13
  single-fit times taken without gen20.1's 2-thread cap) is re-measured at S8; if the measured two-pass total exceeds 2× it, the reduction rule
  applies: designs run in later invocations with the procedure unchanged; nothing is dropped; a design never
  run gives `INSUFFICIENT_COMPUTE` when (a)–(g) hold on design BP, otherwise that (h) counts as failed (§8.1);
  the estimate is at the margin of the recovery plan's 12 h cap (:313), and any excess is a recorded deviation
  (O18). Details fixed at S4 (O18 default): every (design, seed, fold) unit, primary and COH-NODMD, is a
  resumable unit; designs are started in the order BP, B, BR, BQ, BPG, A; at most two processes run at once, each
  on whole designs, which cannot change any number (fixed seeds, single-threaded data handling, fits independent
  of n_jobs); no design, unit or arm is dropped or shortened; the report is produced from complete designs only,
  and a design not complete when the user stops the run gives `INSUFFICIENT_COMPUTE` under the rule above.

## 9. Expected results (stated before any gen20.1 number; they change no rule)

- **T1**: NOT_REACHED or RULED_OUT at 0.4, RULED_OUT at 0.5. Pooled level R² under chemotype hold-out
  without a publication mask, other cohorts: 0.2583 (gen5 BASE91; docs/results/gen5_levels_results_20260818.md:433),
  0.2721 (gen7; docs/results/gen7_architecture_results_20260819.md:254), 0.174–0.303 per seed (gen6
  MONO_ET; runs/gen6_expC_5seed/arm_metrics.csv:2,10,18,26,34); 0.42–0.43 only with near analogues (gen5
  unseen_ligand 0.4253, same file :432) or single-metal Eu full zero-shot log D from an exploratory
  decomposition arm (DEC_L4_S1, pooled R² 0.421, the highest in that table; gen12.2's pre-registered selection
  0.240; generations/gen12_2_eu_pred/DECISION_REPORT.md:318-319). A repository search on 2026-10-15 found no
  level R² under design BP or on the 521-cell cohort; gen13's level model persisted only centred curves
  (gen13sep/models.py:269). Each seed's giant BP fold trains on 76–98 cells and predicts 52–64 % of the C3
  pairs [CNT]. **T1w**: < 0.4 (gen6 within-cell shape R² 0.30, chemotype folds, no mask, another key;
  docs/results/gen6_phase2_results_20260819.md:111).
- **T2 / T3**: RULED_OUT at 0.4, point near 0–0.1. On these exact rows (R2_pooled): G14@ZERO 0.0529, FLAT
  −0.1347, MEAN_CURVE@ZERO −0.1156, NNC@ZERO −0.3218 ([LEV]:18,6,30,50). gen13 C_DIRECT_ROW
  (level-then-subtract) adjacent extractant-macro MAE 0.2005 under BP
  (generations/gen13_separation/metrics/BP_v2/leaderboard.csv:8) vs 0.2069 for zero
  (generations/gen13_separation/metrics/B_primary/leaderboard.csv:15), gen13 adjacency incl. Nd–Sm.
- Here "passes / fails against Y" means that the Y component meets / does not meet RULE-C (a)–(k); H-level
  verdicts are those of §8.1.
- **H1**: INCONCLUSIVE overall. Prior evidence comes from another cohort, with chemotype folds and no
  publication mask. Ligand features beat a metal + conditions ExtraTrees (gen7 NULL_metal_cond, the BD_COND
  analogue; runs/gen7_architecture/block_a_diagnosis.md:163) by pooled level R² 0.0515 → 0.2721 and macro
  MAE 1.0882 → 0.9909, a relative gain of 8.9 % (docs/results/gen7_architecture_results_20260819.md:249,
  :254). A is therefore expected to beat BD_MEAN and BD_METAL on the point estimate. Against BD_COND the
  expected relative gain is below δ_Y = 10 % of the comparator's MET-01, so (a) is expected to fail, even
  though the conditions-only arm is weaker under BP (gen13 all-pair log SF extractant-macro MAE 0.471 → 0.625;
  generations/gen13_separation/PRE_REGISTRATION.md:176-177). Against BD_NN no prior exists. There is no
  in-cohort prior: gen13 saved no absolute predictions.
- **H2 / H3**: not PASS (INCONCLUSIVE, ESTIMAND_INCONSISTENT or BASELINE_BETTER), failing against G14_ZERO:
  G14's gain over FLAT was +0.012 macro MAE (p 0.036; GEN20_REPORT.md:189); C_DIRECT_ROW beat zero by ≈ 0.005
  all-pair extractant-macro MAE under BP (0.5834 vs 0.5885; same leaderboards); gen19's derived Pr/Nd log SF
  MAE 0.159 vs FLAT 0.137 (generations/gen19_chem_transfer/GEN19_REPORT.md:126; another corpus and hold-out).
- **H4**: EQUIVALENT or NOT_ESTABLISHED: centred-row vs row +0.008, not significant under BP
  (generations/gen13_separation/analysis/stage2/STAGE2_REPORT.md:294); adjacent macro MAE 0.2011 vs 0.2005
  (BP_v2 leaderboard :7-8). A null is a result.

## 10. Replicates and noise

- **Target**: the frozen replicate mean; `nrep__`, `repsd__` (ddof 1) kept; no nrep weighting; replicate
  readings never cross-multiplied into pairs (recovery plan :83). 41 of 521 cells and 216 of 3,359 entries
  carry replicates (D5/FINDINGS.md:44; 161 in COH-D-C3 [CNT]); 12 % of replicated readings differ by whole
  decades (GEN20_REPORT.md:402). Sensitivity: COH-REPCAP (§2).
- **log D noise** (descriptive, S5, never in a rule), per outer (seed, fold) from masked training cells:
  σ̂²_pool = Σ_e(n_e − 1)s_e² / Σ_e(n_e − 1), the same with s_e ≤ 0.5, and gen20's median s_e ≤ 0.5
  (gen20k/cov.py:101-114; gen20.1's own caveat, not gen20's: a median of per-entry replicate sd is biased low
  for σ when n_e is small, and the 0.5 cap truncates it further); median and range over the 25 BP folds.
- **log SF noise**: only from run-paired replicated adjacent pairs (COH-REPPAIR; O16); fewer than 20 pairs or 5
  chemotypes → "not estimable", fewer than 10 chemotypes → "preliminary"; √2·σ̂ appears only as a labelled upper
  bound, never as an estimate. S3 result (label-free, `manifests/counts_g201.json`): the only run key is
  `experiment_series_id`, giving 115 run-paired strict-adjacent candidate pairs in 20 cells from only 2 chemotypes;
  0 of the 25 BP training folds reach 20 pairs from 5 chemotypes, so log SF noise is "not estimable" and only the
  labelled √2·σ̂ bound is printed.
- **Cited log D replicate sd values, sensitivities only**: medians 0.237 (pre-quarantine; DATA_AUDIT.md:76),
  0.302 (frozen cohort, 216 entries; DATA_AUDIT.md:79), 0.225 (282 groups; generations/gen18_process/DESIGN.md:42),
  0.333 (generations/gen17_pairdiff/results/g17_summary.txt:18); gen19's noise-floor sd 0.299
  (generations/gen19_chem_transfer/preregistration.md:1070). No noise ceiling is computed or quoted (O17).

## 11. Confirmation

Default: registered (O5). After S10, `scripts/g201_confirm.py` runs once (`RUN_ONCE.json` guard) on the §0
seeds, design BP only, identical procedure. An H is "confirmed" only if it PASSes (H4: same verdict) on both
seed sets, (a)–(g) and (i)–(k) re-evaluated, (h) from discovery (a stated choice, O5); a threshold class is
final REACHED (RULED_OUT) only if so on both sets, otherwise both are shown. Label: "partition-robustness
check on the same corpus, not independent validation" (recovery plan :113-117). If confirmation is not
registered, every verdict carries "discovery only, not confirmed".

## 12. Deviations from gen20 / gen13 / recovery-plan conventions

| # | convention (source) | gen20.1 | reason |
|---|---|---|---|
| 1 | absolute log D a non-goal (recovery plan :10, :61, :192) | A has its own table and primary metric; never in the separation-success table or evidence for separation | the later user brief governs |
| 2 | MAE_bal primary; rival = best baseline on inner MAE_bal; the log-D-then-subtract rival selected and tuned by separation error (recovery plan :123, :270, :201) | MET-01 primary, MAE_bal sign required (RULE-C (i)); IUT over every named baseline (inner-chosen form = EX-11); B inherits A's inner log D selection, the SF-tuned form is the exploratory B_selSF (EX-02) | gen20 continuity; no outer choice; the brief defines B from A's out-of-fold predictions (brief :44) |
| 3 | 2,000 draws; grouping seeds 20260927-29 (recovery plan :101, :302) | 10,000 + 10,000 draws; gen20's split seeds | Monte Carlo error; identical folds (brief :47) |
| 4 | heavier-first orientation; extractant-equal PAIRMEAN (recovery plan :43, :199) | lighter first with flip test (LT-20); chemotype-balanced PAIRMEAN | gen20 continuity; A's weighting |
| 5 | Pr/Nd a separate success; folder `gen20_separation/`; 12 h cap, ≤ 2 processes (recovery plan §9.2, :335, :313) | MET-24 plus the item-4 PASS annotation; `generations/gen20_1/`; one process, six designs, excess recorded | not asked by the brief (O22); the brief's path (O3); O18 |
| 6 | single inner split `select_on_inner` (gen20k/runner.py:100-104); cross-fold majority vote (generations/gen20_kshot/PRE_REGISTRATION.md:90-94) | 4-fold inner chemotype cross-fit of the same split (design A: regrouped by chemotype, a deviation from its stored extractant-grouped inner split, gen13sep/splits.py:30,105-107); per-fold nested argmin | 10–11-cell validation sets in 4 BP folds [CNT] (O12); 13/25 near tie |
| 7 | WCR p < 0.05 at 98.75 %; margin 0.02; mde_80 = 2.80·se (generations/gen20_kshot/PRE_REGISTRATION.md:110-118, :114; gen20k/inference.py:145) | p_wcr < 0.0125; δ = max(0.02, 0.10·MET-01(Y)); decision on max(mde80_dec, mde80_cr) | level consistency (O21); log D scale (O6); gen13sep/wildcluster.py:396-403 |
| 8 | nanmean over seeds and draws; no R2_reason; float32 (gen20k/inference.py:178-180; gen20k/runner.py:335-337) | NA propagation; mandatory R2_reason; float64 | §5 |
| 9 | confirmation string "gen20-confirmation-seed-{i}"; last-line seal gate (gen20k/manifest.py:53) vs gen18/19 addenda; gen19 `preregistration_draft.md` | "gen20_1-confirmation-seed-{i}", gen20's seeds excluded; no footer now, mechanism at S4; one `PRE_REGISTRATION.md` with a DRAFT banner | gen20's seeds spent; O19; the brief names the file (brief :38) |
| 10 | gen13 Addendum 3: five designs; zero prediction scored wrong in gen13/14 code (generations/gen20_kshot/AUDIT.md:157) | six (+ BPG), sign agreement required for PASS; a zero prediction scores 0.5 | gen20 added BPG; recovery plan :187 |
| 11 | DATA_AUDIT §3 statistics as context; √2·sd as SF noise (DATA_AUDIT.md:63-64, :77) | never R² context, π at S5; run-paired SF noise | different populations; recovery plan :87 |
| 12 | pre-existing: `generations/verify_relocation.py` fails on this Windows clone (CRLF) | reported; not fixed; not attributed to gen20_1 | out of scope |

## 13. Open items requiring the user's decision before sealing

Resolved at S2 (2026-10-16): the user instructed to run without answering, so every item below is fixed at its
default; any later change is a dated addendum.

| ID | question | why open / what the default costs | conservative default |
|---|---|---|---|
| O1 | Where does the brief's "normal values" count come from, and what does it count? | the figure occurs in no repository file except the brief; the nearest documented counts are other populations (`README.md`) | never used; populations are 3,359 obs / 521 cells, and 2,425 entries / 1,802 pairs on C3 |
| O2, O3 | English or Russian documents? Keep `gen20_1/` beside `gen20.1/`, add a topic suffix or move the RTF; when to add index rows? | the brief is in Russian; it names `generations/gen20_1/` while the RTF sits in `gen20.1/` | O2: English, with an English digest of the brief and a pointer to the RTF; O3: `gen20_1/` exactly, `gen20.1/` untouched, index rows after approval |
| O4 | Include measured mode M and the `RIDGE2\|k3` anchor? | a read metal reveals the test cell's level; a measurement model the brief does not request | OFF; switched on only by the user's S2 answer to O4 (entered in the drafts before the seal) or, after the seal, by a dated addendum before any run; outside all claims |
| O5 | Register a one-shot confirmation; the new seed rule; BP only or six designs? | not asked by the brief; re-splits the same corpus, so not independent validation (§11) | registered; §0 rule; BP only; (h) from discovery |
| O6 | Margin max(0.02, 0.10·MET-01(Y)) instead of gen20's fixed 0.02? | differs from gen20's fixed 0.02; equals 0.02 for log SF | max(); still 0.02 for log SF against FLAT and G14_ZERO |
| O7 | Threshold bounds at 95 % or 98.75 %; R2_balanced and publication bound as co-conditions ((vi) blocks a crossing carried by sc009 alone; the publication bound makes the publication hold-out co-decide)? Which class is printed when POINT_ONLY and RULED_OUT both apply (§7)? | T classes are classifications outside K, not tests; 95 % is less strict than the 98.75 % alternative, which would make both REACHED and RULED_OUT harder and match the deciding level | 95 % with all six RULE-T conditions; 98.75 % printed; precedence: POINT_ONLY, never counted as attainment, both percentiles printed (resolved at the default, S2/S4) |
| O8 | G14_ZERO (and PAIRMEAN) inside the H2/H3 IUT, or reported only? | G14 is a model, not a trivial baseline; inside the IUT, B/C must beat its MET-01 0.1612 by 0.02 | inside; component verdicts printed separately |
| O9 | Learner grid (§4.1) | not given by the brief; ALL7 would add ECFP and LIG2D input columns (§14); LEAN209's origin is O26 | K1–K4 as listed; alternative: feature-set grid (LEAN + rich7; LEAN + basic; ALL7 + basic; LEAN + rich7 leaf 8) |
| O10, O11 | Select C on its own target (inner MET-20) or on inner strict-adjacent log SF MAE? Primary C learner: centred-row or the never-run antisymmetric pair arm? | the criterion changes C's per-fold configuration (§4.2); C_PAIR was planned in gen13 but never run (§4.5) | O10: own target, C_selSF printed beside; O11: centred-row, C_PAIR exploratory after a smoke test |
| O12 | 4-fold inner cross-fit or gen20's single inner split? | gen20's single split gives 10–11-cell validation sets in 4 BP folds [CNT] (§12 row 6) | 4-fold cross-fit; inner sizes printed |
| O13, O14 | Primary log SF rows: C3 (229 cells / 1,802 pairs) or all strict-adjacent (378 / 2,114)? Primary log D rows: C3 (2,425 entries) or all 3,359? (The matched COH-D-C3M rows, 2,208 entries, are O27.) | O13: k_max ≥ 3 has no meaning at k = 0, but C3 carries gen20's headline metal-model k = 0 anchors (§2); O14: all entries is the natural absolute log D set; C3 gives matched cells and shared draws | C3 for both; the others co-reported; COH-D-ALL cannot fulfil OUT1 |
| O15 | Exclude the 5 DMDPhPDA corrupt-copy cells from the primary? | keeping preserves gen20 comparability (the 1,802-pair population and its k = 0 anchors); it corrupts absolute log D | keep; COH-NODMD refit with flip reporting (biases log D results, and the 5 M Pr/Nd pair, toward larger error) |
| O16 | Can replicate readings be paired by run for log SF noise? | feasibility unknown until S3; √2·σ̂ is only an upper bound (§10) | checked at S3; else "not estimable" plus the labelled bound |
| O17 | Compute descriptive R² noise ceilings? | the replicate sd estimates are biased low and partly decade-corrupted (§10) | not computed |
| O18 | Compute budget: 12 h cap vs six designs | six designs with the COH-NODMD refit are estimated at ≈ 12 h, at the cap (§8, compute rule) | per-design invocations; excess recorded; a design never run gives INSUFFICIENT_COMPUTE when RULE-C (a)–(g) hold on design BP, otherwise that (h) counts as failed (§8.1) |
| O19 | Seal footer and addenda mechanism; is `IMPLEMENTATION_PLAN.md` (cited here for MET-02..MET-24 formulas, fold gates, the deny list, the MODE_M specification and the compute estimate) frozen with this file at S4? | the brief forbade sealing in the drafting iteration (brief :40) | resolved at the default (S2/S4): gen20-style last-line digest over LF-normalised text; dated `ADDENDA.md`; the plan is frozen with this file (its LF-normalised digest in `manifests/seal_g201.json`); no `.gitattributes` change (the digest is LF-normalised, so line endings cannot change it) |
| O20, O21 | RULE-C (i)–(k) required, or annotations? WCR threshold 0.0125 or gen20's 0.05? | O20: stacked requirements lower power (Kish about 4); O21: deviation from gen20 | O20: required (a failure gives ESTIMAND_INCONSISTENT); O21: 0.0125 |
| O22 | A separate registered Pr/Nd claim? | not asked by the brief; would enlarge K | no; MET-24 plus the PASS annotation |
| O23 | Single-metal training augmentation for A? | changes the available observations versus C; needs a cohort rebuild | not run |
| O24 | Which estimand and mode were 0.4 / 0.5 meant for? (the repository's only R² target is the recovery plan's measured-mode log SF R2_balanced(k = 3) ≥ 0.50, :287) | the brief names neither (brief :17) | descriptive, at k = 0, on R2_pooled (log D, log SF) and R2_within_cell (log D), via RULE-T |
| O25 | May local, gitignored gen13 prediction files be read for continuity checks? | such a check is not reproducible from the repository | read if present; the hard gate is the in-process refit (LT-13) |
| O26 | LEAN209 was chosen in gen13 on outer BP scores over these same folds. Keep it fixed, or select block sets inside each training fold? | LEAN209 is not blind to these test cells (§4.3) | keep LEAN209 fixed (no gen20.1 outer choice, no added degrees of freedom); every H and T verdict and the confirmation label carry "feature set fixed from gen13 outer results on these folds" |
| O27 | Log D rows for cross-target statements: all C3 entries (COH-D-C3, 2,425) or only those underlying the scored log SF pairs (COH-D-C3M, 2,208), per the brief's matched held-out data (brief :42)? | 217 C3 log D entries have no scored log SF counterpart (§2) | both printed in TAB-07; H1, T1 and T1w stay on COH-D-C3 |

## 14. What will not be done

- In this iteration no experiment, model fit or prediction is run, nothing is sealed, hashed or committed,
  and no gen20.1 code exists in the repository (only the label-free [CNT], [DMD], [INNER] counts were made); no
  gen20.1 metric, prediction, verdict or target statistic is computed before approval (S2), the seal (S4) and,
  for target statistics, S5. No k > 0 analysis is registered; mode M stays off unless switched on by the user's
  S2 answer to O4 (entered in the drafts before the seal) or, after the seal, by a dated addendum before any run.
- No data are added (dataset_all_metals, SAFE, external sets); no single-metal cells in the primary; no
  synthetic data as evidence. Nothing is chosen, redefined or dropped after any outer score or S5 statistic;
  no design is dropped silently; smoke runs compute no outer metric and tune nothing.
- No architecture beyond the fixed grid and no neural model; under the default grid no RESP3D, ECFP or LIG2D
  block is an input column of A, C, BD_COND, BD_LEVEL or any RowForest (O9's alternative ALL7 would add ECFP and
  LIG2D and is decided before the seal); ECFP fingerprints enter only the BD_NN and NN_TANIMOTO_REF
  nearest-neighbour search; no R² ranking across targets; no 89/11 split as a property of the gen20.1 rows;
  no 0.310 without its k = 3 label; no oracle or noise ceiling as a ceiling; no registered Pr/Nd claim; no
  real-world separation claim; re-splitting the frozen corpus is internal re-evaluation, not independent
  validation. No file outside `generations/gen20_1/` is modified (brief :28).
SEALED-SHA256: f34a1b97433b79b37024ecd8fd5349eeb0d4853b5d7cad60ed21470f1be2d232
