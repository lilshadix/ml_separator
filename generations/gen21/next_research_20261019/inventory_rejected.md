# Inventory of tested and rejected ideas for strict-adjacent lanthanide log SF (gen13 to gen21)

Repository: `D:\ml_separator_gh`, branch `gen21-logsf-target`, HEAD `9be774c`. I modified no repository file: `git status` before and after shows only the untracked `generations/gen21/next_research_20261019/`, which existed before I started. Scratch files are in `C:/Users/Bandai/AppData/Local/Temp/claude/D--ml-separator-gh/05bd353e-2298-45ac-ba07-8e09a6694436/scratchpad/r3inv/` (`pwl_check.py`, `pwl_score.py`, `pwl_rows.parquet`).

## 0. Things to log in the round-3 ATTEMPTS.md

1. **I looked at dev results myself.** I computed PWL(Z) and PWL(radius) on gen21's cached DOPT_Q supports for the five dev seeds, BP and BPG, C3 at k = 0–3 and C6 at k = 3/5/6. The headline values match numbers that `INSTRUCTIONS FOR NEW IDEAS ROUND 3.md` already quotes. The following breakdowns are **new dev looks**:
   - per pair type;
   - one read end vs neither end read;
   - sc009 vs the other chemotypes;
   - the correlation between the HBK@ML and PWL errors;
   - a post-hoc 50/50 HBK@ML + PWL_R blend (R² 0.424). This one bears directly on round-3 registry rows R13 and R22 (STACK).
2. **Label-free checks.** I counted which metals DOPT_Q reads at k = 3, and how often its first read pair is the cell's two end metals. Neither count uses any target value.
3. **The withheld-seeds directory.** I ran one existence check, `ls C:/Users/Bandai/gen21_withheld` with all output discarded. I opened no file in it and saw no listing. It should not have been touched.

## 1. Regimes: why numbers across generations cannot be compared directly

| generation(s) | metric | pair set | information |
|---|---|---|---|
| gen13–gen16 | extractant-macro MAE of log SF (log10) | **all pairs**; "adjacent" in gen13 = dZ = 1 **plus Nd–Sm** (`gen13_separation/PRE_REGISTRATION.md:138`) | zero-shot, or the **pair model** (measured separation factors). gen15 scored rows whose value the supports already fix by arithmetic: 24.8 % of adjacent pairs (`gen20_kshot/GEN20_REPORT.md:105-108`) |
| gen19 | log D MAE, plus V5-PAIR / V6 log SF MAE | other corpus (`dataset_all_metals`), heavier-first orientation | hidden metal × extractant cells |
| gen20, gen20.1, gen21 | **pooled R² of strict-adjacent log SF** (seed mean), cohort C3 (229 cells, 1,802 pairs per seed) | strict \|ΔZ\| = 1, 12 pair types | **metal model**: k + 1 metals read, so k = 3 means 4 metals; no scored pair has both ends read |

Target spread on C3 (verified by recomputation): mean −0.0815, sd about the mean 0.2220, RMS about 0 = 0.2364, variance 0.0493. Under this convention, R² 0.40 corresponds to R²₀ ≈ 0.471 and R² 0.50 to R²₀ ≈ 0.559 (`gen20_1/README.md:112`).

## 2. The PWL baseline and the gen15 "naive line"

### 2.1 What exists in the repository (verified)

- **PWL appears in no committed gen13–gen21 report.** A grep for `PWL|piecewise` over `generations/**/*.md` hits only `generations/gen21/INSTRUCTIONS FOR NEW IDEAS ROUND 3.md:31-40`, which quotes:
  - C3 k = 3: PWL by Z ≈ 0.412, PWL by radius ≈ 0.418;
  - C6 k = 6: PWL ≈ 0.527 (BP).

  Its source is `generations/gen21/AUDIT_2026-10-19.md` (cited at line 17), and **that file does not exist** (`ls generations/gen21/AUDIT*` finds nothing). The only PWL code is the in-progress `generations/gen21/next_research_20261019/r3/interp.py:30-43`, which is untracked.
- **LINE / NAIVE_LINE is not PWL.**
  - Definition: `gen15_curve/gen15/fewshot.py:140-149` and `gen20_kshot/gen20k/adapters.py:56-63`. It fits one least-squares slope on the radius ramp through the star supports.
  - Because it has one parameter, it saturates: gen15 0.240 / 0.237 / 0.236 macro MAE at k = 1/2/3 (`GEN15_REPORT.md:205`); gen20 strict-adjacent R² 0.168 / 0.237 / 0.236 at k = 1/2/3 (`GEN20_REPORT.md:170-174`).
- **Closest earlier relatives of PWL:**
  - gen13 stage 2: predicting a metal from its two flanking neighbours by OLS on centred log D gives mean R² 0.929 and MAE 0.083. This is in-sample, leave-one-cell-out, and scores centred log D, not log SF (`gen13_separation/analysis/stage2/d2_metal_dependency/FINDINGS.md:30`).
  - gen19 `B3i`: linear interpolation in radius between the bracketing metals' nearest-condition lookups within the same extractant system (`gen19_chem_transfer/preregistration.md:691-700`). With Pr/Nd hidden in V6, the lookup-derived log SF MAE is 0.315 against FLAT 0.137 (`GEN19_REPORT.md:126`).
- **gen20's arithmetic-only arm is not PWL either.** `PROJ` is the minimum-norm curve (`adapters.py:100-108`) and scores R² −4.55 at k = 3 (`GEN20_REPORT.md:180`). gen20's sentence "the measurements alone, without a curve prior, do not predict the unmeasured adjacent contrasts" (`GEN20_REPORT.md:210-211`) rests on that arm.

### 2.2 Verification by recomputation

Scratch scripts, identical rows to `predictions/attempt4_{BP,BPG}.parquet`: all 61,040 rows per design matched, `a_read`/`b_read` flags identical, and RIDGE2 reproduces at 0.3099.

PWL definition: `np.interp` of the read log D values along Z, or along −(Shannon CN8 radius); constant beyond the extreme reads; read metals keep their readings. Cells: R² / MAE, seed mean.

| design | cohort, k | PWL_Z | PWL_R | HBK@ML | HB2+GP | RIDGE2 | LINE |
|---|---|---|---|---|---|---|---|
| BP | C3 k = 1 | 0.181 / 0.1348 | **0.194** / 0.1330 | 0.099 / 0.1456 | 0.076 | −0.008 | 0.168 |
| BP | C3 k = 2 | 0.351 / 0.1147 | **0.360** / 0.1131 | 0.344 / 0.1151 | 0.334 | 0.279 | 0.237 |
| BP | **C3 k = 3** | **0.4118** / 0.1063 | **0.4179** / 0.1055 | **0.4142** / 0.1054 | 0.4034 | 0.3099 | 0.2356 |
| BP | C6 k = 5 | 0.4930 | 0.4953 | 0.4986 | 0.4892 | 0.3260 | 0.2300 |
| BP | C6 k = 6 | **0.5272** / 0.0919 | 0.5259 / 0.0922 | 0.5329 / 0.0928 | 0.5214 | 0.3233 | 0.2283 |
| BPG | **C3 k = 3** | 0.4107 / 0.1069 | 0.4180 / 0.1058 | 0.4188 / 0.1050 | 0.4072 | 0.3225 | 0.2368 |
| BPG | C6 k = 6 | 0.5217 | 0.5216 | 0.5310 | 0.5196 | 0.3255 | 0.2279 |

- **HBK@ML − PWL_R per seed at C3 k = 3:**
  - BP: −0.0118, +0.0136, −0.0203, +0.0038, −0.0040 (mean −0.004; better in 2 of 5 seeds);
  - BPG: −0.0082, +0.0137, −0.0063, +0.0039, +0.0009.
- **The two predictors are almost the same.** The correlation of their errors is **0.971**, and of their predictions 0.965. The post-hoc 50/50 blend scores 0.424 on BP. That is a dev look, not a method.
- **Row splits at C3 k = 3, BP (R² / MAE):**
  - one end read (≈ 1,203 rows per seed): PWL_R 0.441 / 0.1027; HBK@ML 0.433 / 0.1029; RIDGE2 0.312;
  - neither end read (≈ 599 rows per seed): PWL_R 0.368; HBK@ML 0.372; RIDGE2 0.286;
  - sc009: PWL_R 0.431, HBK@ML 0.432;
  - other chemotypes: PWL_R 0.398, HBK@ML 0.391.
- **Per pair type at C3 k = 3, BP (PWL_Z / PWL_R / HBK@ML):**
  - La/Ce .371 / .366 / .351; Ce/Pr .418 / .413 / .417; **Pr/Nd .437 / .421 / .366**; Sm/Eu .418 / .415 / .423;
  - Eu/Gd .266 / .288 / .287; Gd/Tb .290 / .330 / .353; Tb/Dy .329 / .328 / .343; Dy/Ho .400 / .421 / .482;
  - Ho/Er .134 / .162 / .224; Er/Tm .238 / .253 / .208; Tm/Yb .038 / .056 / .007; **Yb/Lu −.019 / .020 / −.140**.
- **Why HB ≈ PWL.**
  - Verified: HB2+GP's fitted residual has ρ median **0.986** (range 0.973–0.9997) and τ² median 0.98 over 25 BP folds (`gen21/results/attempt2/fitted_BP.csv`, columns `HB2+GP.rho` / `HB2+GP.tau2`). That is close to a random walk in Z.
  - Interpretation (mathematical fact plus inference): the conditional mean of a random walk with a free level, given reads, is piecewise-linear interpolation with constant extrapolation. gen21 itself found that "the gain comes from the smooth Z-correlated residual" (`ATTEMPTS.md:143-144`).
- **DOPT_Q's training dependence.** DOPT_Q reads are chosen with the G14 covariance fitted on the training fold (`gen21/g21/cache.py:52,69,79`). So "PWL has no training" holds for the predictor, not for the read choice.
- **Label-free read counts.** Share of C3 cell instances (BP, 1,145) whose four k = 3 reads include each metal:
  - La .71, Gd .58, Er .38, Nd .33, Pr .32, Dy .31, Ho .29, Yb .28;
  - Tm .23, Eu .23, Lu .17, Sm .14, Ce .04, Tb .005.

## 3. Inventory by theme

"Strict-adj R²" means C3 pooled R² unless stated. All verdicts are those of the source documents.

### A. Zero-shot structure models (k = 0)

| idea | gen | key number(s) | file:line | verdict |
|---|---|---|---|---|
| G14: L2 logistic direction on 39 donor-topology columns × training-fold magnitude and curvature | 14 | all-pairs macro MAE 0.500 BP vs FLAT 0.589; +0.088, p 0.047 (WCR p 0.059) | `GEN14_REPORT.md:71-97`; `GEN15_REPORT.md:48-51`; `gen16_protocol/README.md:12` | deployed; the direction bit is "spent" (oracle direction +0.014) |
| G14 zero-shot on strict adjacent | 20 | R² **0.053**, MAE 0.147, macro 0.161 vs FLAT −0.135 / 0.164; confirmation seeds R² 0.04 | `GEN20_REPORT.md:168,338` | best zero-shot reference, still ≈ FLAT (+0.012 macro, below margin) |
| 12 magnitude priors (loss-fitted, ridge, extra trees, direction-conditional, …) | 14–15 | all within ≈ 0.007 of a constant | `GEN14_REPORT.md:111-128`; `GEN15_REPORT.md:380` | closed ("estimator axis closed") |
| Curvature from 2D (sign(b) logistic, shape alphabets SHAPE3/4/6) | 15 | 0.5496–0.6258 vs G14 0.5001; oracle alphabet 0.444 | `GEN15_REPORT.md:115-141` | null (headroom exists, not predictable) |
| Leave-pair-out curvature gate (gen16 L2) | 16 | headroom +0.0287 [−0.0013, +0.0532]; minimum detectable 0.039 | `gen16_leads/DECISION_REPORT.md:377-401` | fails by rule; underpowered |
| ECFP / 206 RDKit / LEAN209 trees; gen13 curve bases (rank-k, physics, tetrad, Gd-break) | 13 | bases interchangeable within 0.006; ECFP and RDKit dilute the signal; under BP lean blocks +0.055 to +0.096 over conditions-only; strict-style "adjacent" (with Nd–Sm) model 0.190 vs B0_ZERO 0.207 macro MAE | `gen13 DECISION_REPORT.md:103-122,180-208` | adjacent "at the noise floor"; not strict-adjacent evidence |
| Conditions-only models | 13 | 0.471 (B) → **0.625 (BP)**, worse than the mean curve | `gen13 DECISION_REPORT.md:41-48,216-221` | laboratory fingerprint; select under BP only |
| 3D: RESP3D, xTB energies, phys3d, cycle-corrected xTB slope (L1) | 13, 15, 16 | RESP3D add −0.006; xTB slope +0.644 → +0.195 after the composition control; L1 ρ −0.084 | `gen13 DECISION_REPORT.md:237-260`; `GEN15_REPORT.md:145-178`; `gen16_leads/README.md:10` | dead (programme-wide negative) |
| External aqueous logK prior | 13, 15 | −0.000 to −0.002; logK slope sign 0.440 vs 0.722 constant | `gen13 DECISION_REPORT.md:245-251`; `GEN15_REPORT.md:381` | dead |
| Pretrained chemical LM embeddings; 85 similarity kernels; TabPFN / CatBoost / monotone XGBoost / GAM / symbolic search | 15 | embed 0.5029 (direction 0.569 vs 0.821); kernel 0.4980; tabpfn 0.4914 | `GEN15_REPORT.md:377-380`; `gen16_leads/START_HERE.md:161-174` | dead |
| Anchor regression (publication anchor) | 16 (side study) | best 0.5715 vs G14 0.5001; all 41 contrasts negative | `gen16_anchor/README.md:10,31-36` | null (unreviewed) |
| Within-publication pair differences (PADRE / DeepDelta) | 17 | every arm worse than predicting zero Δa (+0.013 to +0.033) | `gen17_pairdiff/README.md:138-143` | null (unreviewed, LOPO) |
| ExtraTrees zero-shot: log D, then difference (B) vs direct centred (C) | 20.1 | strict-adj R² **B −0.010 / C −0.016** (confirmation −0.011 / −0.042); log D R² 0.258 is level-driven (85 % of its error is cell level, cancels in SF) | `gen20_1/README.md:16-35`; `GEN20_1_REPORT.md:27-38` | H2 and H3 INCONCLUSIVE; H4 EQUIVALENT; R² ≥ 0.4 RULED_OUT |
| Reference repository's CatBoost / 3D blend | — | +0.326 adjacent R² under leave-extractant-out (not chemotype or publication hold-out) | `gen20_kshot/IMPLEMENTATION_PLAN.md:126-134`; `AUDIT.md:69-70` | not reproduced; leaky regime |
| Structure prior added to HB residual | 21 | regressing the HB2+GP residual on (G14@ZERO − HB2+GP): slope 0.08, 0.5 % of SSE in-sample | `gen21/ATTEMPTS.md:304-309` | not pursued |
| Structure after supports (G14@BLUP20 − FLAT@BLUP20) | 20 | +0.017 (k = 0) → +0.006 (k = 3, p 0.08) macro | `GEN20_REPORT.md:204-206,369-371` | marginal, vanishes with reads |

### B. Curve priors and interpolators (k-shot)

| idea | gen | strict-adj numbers (C3 unless noted) | file:line | verdict |
|---|---|---|---|---|
| LINE (one radius slope, no corpus) | 15, 20, 21 | k = 1 / 2 / 3: R² 0.168 / 0.237 / 0.236 | `GEN20_REPORT.md:170-174` | baseline; saturates |
| RIDGE2 (radius + radius² shrunk to the training prior) | 20 | k = 3 R² **0.310** [0.22, 0.40], MAE 0.121; confirmation 0.32; beats LINE by +0.008 macro (< 0.02 margin) | `GEN20_REPORT.md:154,175,334-335`; `gen21/PROTOCOL.md:113-119` | reference; superseded by HB and PWL |
| MEAN_CURVE as baseline | 15 | 0.622, worse than FLAT 0.589 (all pairs) | `GEN15_REPORT.md:44-46` | never use as floor |
| LINE_PLUS_CURV (line + training direction-conditional curvature) | 15 | 0.243 / 0.243 / 0.239 vs LINE 0.240 / 0.237 / 0.236 | `GEN15_REPORT.md:206,224-226` | worse |
| RIDGE2+OBS@RESEL (observable target; noise and λ re-chosen by inner task loss) | 21 att. 1 | 0.357 BP / 0.358 BPG; RIDGE2@RESEL 0.323 | `ATTEMPTS.md:51-59,74-87` | not advanced (MAE 3/5) |
| HB2 (hierarchical θ, no δ) | 21 att. 2 | 0.348 | `ATTEMPTS.md:113` | δ is what works |
| **HB2+GP** (exp/OU residual in Z) | 21 att. 2 | **0.403** BP / 0.407 BPG; k = 1 0.076; k = 0 −0.116; C6 k = 5 / 6: 0.489 / 0.521 | `ATTEMPTS.md:111-120` | round-1 candidate |
| HB3H+GP (Gd hinge) | 21 att. 2 | 0.403 (Δ 0.000) | `ATTEMPTS.md:112,129` | adds nothing |
| **HBK@ML** (kernel exp / m32 / m52 / se chosen by training likelihood) | 21 att. 4 | **0.414** BP / 0.419 BPG; +0.011 over HB2+GP (5/5 seeds); se chosen in 19/25 folds; k = 1 0.099 < LINE | `ATTEMPTS.md:334-369` | round-2 candidate; ≈ PWL (see §2.2) |
| PWL_Z / PWL_R (no predictor training) | audit (absent) / this check | 0.412 / 0.418 BP; best at k = 1–2 | §2.2 | the strong baseline round 3 is told to beat |
| gen13 one measured pair: basis shift / rescale / no-model line (all pairs) | 13 | basis shift 0.396; no-model line 0.483; adjacent support pair buys nothing (0.483) | `gen13 DECISION_REPORT.md:267-282` | "measure a far pair" |

### C. Neural contrast networks

| idea | gen | numbers | file:line | verdict |
|---|---|---|---|---|
| NNC: TabularNet + within-cell pairwise contrast loss (BLUP20 / HEAD_FT / FULL_FT; arm M*) | 20 | strict-adj R² zero-shot −0.322; k = 3 **−0.128** (BPG −0.215; confirmation −0.29); −0.030 macro vs G14@BLUP20; P1 and P4 fail (LINE better) | `GEN20_REPORT.md:136-147,169,178,199-203,271-274,327-337` | worst arm; leak-sensitive |
| gen19 factorised neural M1 / M2 vs descriptor CatBoost M0 | 19 | M0 − M2 = −0.154 log D; ladder keeps M0 | `GEN19_SUMMARY/SUMMARY.md:19` | neural steps dropped |
| gen13 S3_HIER (extractant hierarchy with a condition head) | 13 | 0.501, worst arm | `gen13 DECISION_REPORT.md:395-399` | null |

### D. Covariance / BLUP measured modes

| idea | gen | numbers | file:line | verdict |
|---|---|---|---|---|
| Stage-2 BLUP with the widest pair (all pairs, pair model) | 13 | 0.541 → 0.266 (BP); line through the pair 0.272; greedy budget 0.409 → 0.211 → 0.193 → 0.167 | `gen13 DECISION_REPORT.md:284-313` | lever is the measurement, not the model |
| G14 + BLUP, D-optimal k = 3 (all pairs, pair model) | 15 | 0.170 BP; +0.065 over the naive line | `GEN15_REPORT.md:243-309` | **not reproduced on strict adjacent** (see §5 C2) |
| Prototype mixture MIX6 (per-component covariance) / MIX6meanPC | 15, 16 | full mixture loses (−0.045 at k = 1); meanPC +0.0046 to +0.0070 (< margin) | `GEN15_REPORT.md:325-366`; `gen16_leads/DECISION_REPORT.md:453,462` | null with a mechanism |
| Covariance estimators LW / LOWRANK2/3 / HIER (L5) | 16 | LW explodes (34.4 at k = 3); LOWRANK2 +0.015 only at k = 2; deployed covariance indefinite in 100 % of 430 estimates | `gen16_leads/DECISION_REPORT.md:444-477` | no claim |
| G14@BLUP20 / FLAT@BLUP20 strict adjacent | 20 | k = 3 R² 0.248 / 0.189; ties LINE (+0.003, p 0.41); BLUP_INDEP within 0.002 | `GEN20_REPORT.md:153-155,176-177,193` | no gain over the line |
| G14@BLUP20+OBS | 21 att. 1 | 0.164 (vs 0.248) | `ATTEMPTS.md:56` | worse |

### E. Read policies

| idea | gen | numbers | file:line | verdict |
|---|---|---|---|---|
| widest vs D-optimal vs random (all pairs, pair model) | 15 | D-opt − widest at k = 3 +0.033 (passes); widest − random at k = 1 +0.10; widest wins at k = 1 | `GEN15_REPORT.md:268-295` | pair model only |
| RANDOM / UNIFORM / DOPT_CURVE / DOPT_Q / UNC_SEQ (metal model, strict adjacent) | 20 | spread ≤ 0.005 macro MAE (e.g. RIDGE2 0.132–0.140); DOPT_Q not the best | `GEN20_REPORT.md:213-230` | read choice ≈ irrelevant for gen20 arms |
| SERIES_STAGGER for HB2+GP / RIDGE2 | 21 att. 3 | HB2+GP own reads 0.179 / 0.288 / 0.386 (STAGGER) vs 0.076 / 0.334 / 0.403 (DOPT_Q) at k = 1/2/3; RIDGE2 0.292 vs 0.310 | `ATTEMPTS.md:220-224,252-254` | read choice matters for HB: helps at k = 1, hurts at k = 3 |
| DOPT_HB (minimise HB posterior variance) | 21 | not run; posterior variance inverted vs error across pair types (rank −0.46) | `ATTEMPTS.md:371-383`; `ROUND2_REPORT.md:58-62` | deferred, untested |
| Chemotype acquisition order AOPT (L4) | 16 | equalled by a featureless size order (AOPT − SIZE +0.005) | `gen16_leads/DECISION_REPORT.md:403-436` | killed |

### F. Pooling across conditions, Setting B, siblings

| idea | gen | numbers | file:line | verdict |
|---|---|---|---|---|
| Same-extractant other-cell ceiling (all pairs) | 13 | 0.277 (mean of other cells) / 0.247 (nearest condition) vs 0.461 | `gen13 DECISION_REPORT.md:315-319` | ligand-specific shape exists |
| Setting B COPY (other condition, same extractant and publication) | 20 | R² 0.32, MAE 0.091 on 24 units / 1,397 pairs per seed; models add +0.0009; median cross-condition change 0.043 | `GEN20_REPORT.md:308-316`; `AUDIT.md:173-181` | exploratory; same publication, so not admissible under BP |
| HBG@POOL (setting P: series pools its condition reads), staggered reads | 21 att. 3 | 0.341 vs own reads 0.382 (−0.041 [−0.106, +0.010]); 0.294 vs 0.362 on cells with siblings | `ATTEMPTS.md:197-230,239-257` | **refuted** |
| HBG@POOL with DOPT_Q reads | 21 att. 3d | 0.406 vs 0.403 (+0.002) | `ATTEMPTS.md:203-205` | neutral |
| RIDGE2@POOL | 21 | 0.259–0.263 (pooled shape hurts, −0.03 to −0.06) | `ATTEMPTS.md:202,205,218` | worse |
| Sibling anchor (copy siblings' coefficients) | 17 | copy-mean 0.345 vs zero 0.543 (all pairs, LOPO, includes same-paper siblings) | `gen17_pairdiff/README.md:143,177` | unreviewed; contradicted by gen13 d1 |
| Within-publication conditions (Frisch–Waugh condfe) | 15 | acid Spearman +0.356, worth +0.009; inner LOPO zeroes it in 90 % of folds | `GEN15_REPORT.md:382` | null |

### G. Metal-specific effects

| idea | gen | numbers | file:line | verdict |
|---|---|---|---|---|
| Residual after per-cell quadratic: Gd **dip** | 13 stage 2 | Gd −0.072 (p 0.012); **−0.127 outside sc009, −0.025 inside**; only 7.2 % of residual variance is a common curve | `gen13 .../D7/FINDINGS.md:13-33` | descriptive; never fed back |
| Bases with Gd-break / tetrad terms | 13, 21 | interchangeable within 0.006; HB3H hinge +0.000 | `gen13 DECISION_REPORT.md:120-122`; `ATTEMPTS.md:129` | null |
| HBKM: free per-metal mean α (+ per-metal noise), BIC choice | 21 att. 6 | HBKM@BIC 0.407 vs HBK@ML 0.414; MAE worse 0/5 seeds; α_Gd −0.096 ± 0.020; helps non-DGA Eu/Gd and Gd/Tb, over-corrects DGA; α_Tm unstable | `ATTEMPTS.md:478-525` | **refuted as declared** |
| Gd-only or shrunk metal offsets | proposed | — | `ROUND2_REPORT.md:152-156` | **not run** (this is round-3 R03 / R08) |

### H. Noise models and heteroscedasticity

| idea | gen | numbers | file:line | verdict |
|---|---|---|---|---|
| Shared-reading vs independent noise (R = s²DDᵀ) | 20 | within 0.002 | `GEN20_REPORT.md:193` | immaterial |
| NOISE_VAR 0.09 (gen15 BLUP) | 15, 16 | 90 % intervals cover 98.8 % at k = 3 (RMSE 0.248) | `gen16_leads/DECISION_REPORT.md:475-477` | conservative by ≈ 2.5× |
| Observable vs latent target | 21 att. 1–2 | OBS +0.034 only after reselection; for HB ±0.002 | `ATTEMPTS.md:61-67,129-130` | immaterial for HB |
| Two-component heteroscedastic mixture HBMIX | 21 att. 5 | 0.416 vs 0.414 (2/5 seeds); regime identified (AUC 0.86) but noisy-cell MSE 0.0564 vs 0.0566; noisy cells = 37.5 % of cells, 79 % of SSE | `ATTEMPTS.md:411-441` | neutral |
| Label de-noising and precision weighting | 15, 13 | null or harmful; reliability 0.978 / 0.936 | `GEN15_REPORT.md:73-76,383`; `gen13 DECISION_REPORT.md:397-399` | dead |
| Abstention / predicting error (all pairs) | 13, 14 | worth nothing | `gen13 .../D5/FINDINGS.md:33-42`; `GEN14_REPORT.md:154-156` | dead |

### I. Data-quality cohorts and sensitivities

| item | facts | file:line | effect found |
|---|---|---|---|
| COH-NODMD (5 DMDPhPDA cells: printed mantissa without exponent; 70 entries, 60 strict-adjacent pairs) | log SF equals the curated copy except the 5 M Pr/Nd pair (+1.0) | `gen20_1/IMPLEMENTATION_PLAN.md:82-90` | zero-shot B −0.0415 / C −0.0188 vs primary −0.0098 / −0.0157; "changes no verdict" (`gen20_1/README.md:34`); **never run in gen21** |
| COH-REPCAP (drops entries with nrep ≥ 2 and repsd > 0.5) | 1,755 pairs, 224 cells | `gen20_1/PRE_REGISTRATION.md:113-115`; `tab02_targets.csv:36-37` | zero-shot B R² 0.0115 vs FLAT 0.0465 (`tab09_secondary.csv:164-172`) |
| COH-SC009 / COH-NOSC009 (882 / 920 pairs) | — | `gen20_1/PRE_REGISTRATION.md:110-111` | zero-shot B −0.0727 / +0.0154 (`GEN20_1_REPORT.md:50`); gen20 k = 3 ranking the same in both halves (`GEN20_REPORT.md:282-288`) |
| BPG copy groups (45 cells, 5 publications, 3 groups) | — | `gen20_kshot/AUDIT.md:125-126` | moves gen20 arms ≤ 0.002 except NNC (`GEN20_REPORT.md:276-280`); gen21 HB +0.004 |
| Within-series exact offset copies | 7 cell pairs (14 cells, 108 pairs per seed) | `ATTEMPTS.md:181-185,226-230` | flatter pooling only |
| Identical-log-D cell pairs split by a NaN condition | 5 | `gen20_kshot/AUDIT.md:176-177` | none reported |
| D3DODGA/Nd unit slip (×144.24); tied-D groups | 7 rows; 130 groups / 324 rows | `gen18_process/DESIGN.md:40-41` | log D only |
| Decade-level replicate disagreements | 12 % of replicated readings | `GEN20_REPORT.md:402-403` | — |
| gen13 key sensitivities (series 597 cells, relaxed 509 cells) | all pairs | `gen13 DECISION_REPORT.md:173-178` | P1 unchanged |
| Nd–Sm added to the query | RIDGE2 k = 3 0.43; Nd/Sm alone 0.66 | `GEN20_REPORT.md:182-187` | inflates R²; not strict |

## 4. Replicate-noise estimates (log D unless stated)

| estimate | value | population / definition | file:line |
|---|---|---|---|
| median within-replicate sd | **0.237** (mean 0.419); implies √2·0.237 ≈ 0.34 for adjacent SF | 313 combinations, pre-quarantine | `gen13 DATA_AUDIT.md:75-78`; pre-quarantine note `gen20_1/IMPLEMENTATION_PLAN.md:78-80` |
| median sd | **0.302** | 216 replicated (cell, metal) in the frozen cohort; 41/521 cells | `gen13 DATA_AUDIT.md:79`; `D5/FINDINGS.md:44`; `gen20_kshot/AUDIT.md:116` |
| replicated subset vs cohort floor | 0.302 vs 0.113 (pairwise 0.160); pair floor median 0.521 vs MAE 0.493 on 836 pairs per seed | biased subset | `D5/FINDINGS.md:43-51` |
| replicate SEM / raw sd | 0.205 / 0.337 | 207 cell-metals | `D7/FINDINGS.md:9-10` |
| unique per-metal sd after 3 components | max 0.119 (Tm) | complete cells | `d2_metal_dependency/FINDINGS.md:32-33` |
| (extractant × pair type) mean, in-sample | R² 0.680 (764 groups): ≥ 32 % of adjacent variance is condition or replicate residual | adjacent dZ = 1, n = 2,115 | `gen13 DATA_AUDIT.md:96` |
| half-vs-half replicate split | pairwise MAE 0.60 | 9 extractants, all pairs | `gen13 DATA_AUDIT.md:80-83` |
| gen18 replicate groups | median 0.225 | 282 groups | `gen18_process/DESIGN.md:42` |
| gen17 | 0.333; \|Δa\| floor 0.130 | — | cited at `gen20_1/PRE_REGISTRATION.md:495`; `gen17_pairdiff/README.md:140` |
| gen19 noise floor | sd 0.299 | — | cited at `gen20_1/PRE_REGISTRATION.md:496` |
| gen20 training noise s (capped at 0.5) | median **0.106** (range 0.033–0.200) | 25 BP folds | `gen20_1/results/discovery/tab02_targets.csv:184` |
| pooled σ² | 0.582 uncapped / **0.0547** capped | 25 BP folds | `tab02_targets.csv:187-188` |
| √2·σ̂ bound for log SF | median 0.331 (labelled upper bound only) | — | `tab02_targets.csv:189` |
| **log SF noise from run-paired replicates** | **"not estimable"**: 115 candidate pairs in 20 cells, 2 chemotypes | — | `gen20_1/PRE_REGISTRATION.md:487-492` |
| cross-condition change of adjacent SF | median 0.043 (91 % < 0.3) | Setting B groups | `gen20_kshot/AUDIT.md:179` |
| HB fitted white noise σ² | median 0.006 (0–0.01); not identified apart from short-range δ | ML on training curves | `gen21/ATTEMPTS.md:130-133` |
| HBMIX σ² | 0.0037 (quiet) / 0.034 (noisy); a model residual, not measured noise | — | `ATTEMPTS.md:422-429`; `ROUND2_REPORT.md:141-143` |
| gen21 consistency argument | independent sd 0.30 → adjacent noise variance 0.18 > total 0.049, so the floor must be common-mode | — | `ATTEMPTS.md:131-133` |
| my arithmetic bound (inference, assumes independent per-metal noise) | neither-read rows at C3 k = 3: MSE 0.0304 (HBK@ML), 0.0307 (PWL_R), so 2σ² ≤ 0.030 → **σ ≤ 0.12** on these cells | 599 rows per seed | scratch `pwl_score.py` |
| gen15 σ-simulation | noise-only floor 0.13–0.22 (all-pairs macro MAE) | — | `GEN15_REPORT.md:72-74` |

## 5. Contradictions and inconsistencies

1. **Noise-floor narrative vs achieved skill.** gen13 and gen20 say adjacent SF "sit inside the measurement noise" because sd 0.30 > 0.23 (`gen13 DATA_AUDIT.md:76-78`, `GEN20_REPORT.md:27-29,348-349,401`). gen21 shows that independent noise of sd 0.30 is impossible here (`ATTEMPTS.md:131-133`). HBK and PWL reach R² ≈ 0.42, which bounds independent noise at ≤ 0.12 (§4). The 0.30 median comes from a biased replicated subset that includes decade-level disagreements (`D5/FINDINGS.md:50`; capped σ² 0.055 vs uncapped 0.58). Facts verified; the mechanism (common-mode noise) is inference.
2. **gen15 "the model earns its place at D-optimal k = 3" vs gen20.** gen15 reports +0.065 over the naive line (`GEN15_REPORT.md:276,297-303`). gen20 finds G14@BLUP20 tied with LINE on strict adjacent (+0.003, p 0.41; `GEN20_REPORT.md:153,207-208`). Reasons: gen15 scored all pairs in the pair model with closure-determined rows (24.8 % of adjacent; `GEN20_REPORT.md:105-108`). gen20 governs for strict adjacent.
3. **"The first two reads are always the lightest and heaviest legal metals"** (`GEN20_REPORT.md:228-229`). Verified label-free: this is true for UNIFORM (1,145 / 1,145 C3 instances under BP) but **not for DOPT_Q** (165 / 1,145). The most common DOPT_Q first pairs are La/Eu, La/Yb, La/Ho and Gd/Yb.
4. **gen20 calls 0.236 "y sd"** (`GEN20_REPORT.md:163`). 0.236 is the RMS about 0 (= FLAT RMSE). The sd about the mean is 0.222 (`gen20_1/README.md:20`; recomputed).
5. **Sign convention of the adjacent mean.** gen13 writes "+0.089 (heavier preferred)" (`gen13 DATA_AUDIT.md:65`). Under the lighter-first convention, heavier-preferred means y < 0, and gen20 gives "mean −0.089" (`gen20_kshot/AUDIT.md:46,113`). The count 2,115 vs 2,114 is pre- vs post-quarantine (`AUDIT.md:97`).
6. **"Measurements alone … do not predict"** (`GEN20_REPORT.md:210-211`) rests on PROJ (minimum-norm, R² −4.55). Corpus-free PWL gets 0.418 at k = 3, above every gen20 arm.
7. **gen21's gain attribution.** gen21 reports +0.094 / +0.104 over RIDGE2 as a gain of the hierarchical model (`GEN21_REPORT.md:84-86`; `ROUND2_REPORT.md:104`). PWL_R gets +0.108 on the same supports, and HBK@ML − PWL_R = −0.004 (BP). The round-3 brief states this (`INSTRUCTIONS…:40`), but the cited audit is absent from the repository.
8. **Missing cited sources.** `generations/gen21/AUDIT_2026-10-19.md`, `/Users/lilshadix/Downloads/audit.pdf` and any `AGENTS.md` (searched to depth 3) are not present (`INSTRUCTIONS…:15-18`). The task files use macOS paths (`CLAUDE_TASK.md:5`); the clone is Windows.
9. **Sibling-copy studies disagree.** gen17 finds copying from another paper about as good as from the same paper; gen13 d1 finds it 0.11 worse (`gen17_pairdiff/README.md:177`).
10. **Pooling tension (not a numeric contradiction).** gen20 Setting B: copying the sibling condition's SF gives R² 0.32 (`GEN20_REPORT.md:310-316`). gen21: sibling reads at unread metals hurt (−0.041; `ATTEMPTS.md:239-251`). The information differs (the full sibling SF vs k + 1 sibling reads) and so do the cohorts (24 units vs 159 cells).
11. **Proposal later rejected.** `GEN21_REPORT.md:213-214` proposes "structure-predicted prior to lift k = 0–1". The round-2 diagnosis rejected it (`ATTEMPTS.md:306-309`), consistent with `GEN20_REPORT.md:369-371`.
12. **Stale text.** `gen20_1/README.md:14` says "Nothing in this folder is committed" (committed in e579767; `ADDENDA.md:152-155`).
13. **Undercoverage of intervals.** The percentile chemotype bootstrap rejects at 0.07–0.08 at nominal 0.05 (`gen16_protocol/README.md:8-9`; `gen20_kshot/AUDIT.md:159-160`). gen21 reports "nominal 95 % ≈ real 90 %" (`ATTEMPTS.md:7-9`).

## 6. In-progress round-3 registry vs earlier results

Untracked and in flux: `generations/gen21/next_research_20261019/EXPERIMENT_REGISTRY.csv`. The table shows which earlier result each registry row resembles.

| R3 row | nearest earlier test | status / risk |
|---|---|---|
| R02 COR_G (PWL residual corrector) | gen15 LINE_PLUS_CURV worse (`GEN15_REPORT.md:224-226`); gen13 flanking-neighbour OLS (in-sample) | new on PWL, but HB ≈ PWL already absorbs smooth structure |
| R03 COR_GP (shrunk 14-metal block) | att. 6 HBKM[mean] (0.412; over-fits heavy metals) | the shrunk version was proposed, never run |
| R04 / R05 task-loss kernel and noise choice | att. 1 @RESEL (task-loss λ / noise for RIDGE2: +0.013); gen14 loss-fitted magnitude null; att. 4 kernels differ ≤ 0.003 (m52 0.417 vs ML 0.414) | expected gain is small |
| R07 COR_GC (chemistry × curvature) | gen20 structure after supports +0.006; att. diag G14 0.5 % of SSE; gen15 condfe +0.009; gen14 mass action −0.07 to −0.10 | repeatedly null |
| R08 COR_GDC (Gd × chemistry) | gen13 D7 Gd dip −0.127 non-sc009 / −0.025 sc009; att. 6 α_Gd over-corrects DGA | mechanism already documented in gen13 D7 (gen21 does not cite it) |
| R09–R11 POL_ERR | gen20 policies ≤ 0.005; gen21 STAGGER (k = 3 −0.017 for HB); DOPT_HB variance inversion | PWL's weak pairs are Yb/Lu (Lu read only 17 %) and Tm/Yb |
| R13 / R22 STACK | **my post-hoc look**: error correlation 0.971, 50/50 blend 0.424; gen13 bags +0.014 to +0.037 (zero-shot, all pairs) | small ceiling, and this look is now contaminating |
| R17 / R18 V2 corrections | gen20.1 COH-NODMD: no verdict change; DMD affects only one log SF pair | small |
| R19 ROBUST_COR | att. 5 HBMIX neutral; gen15 labelerr null | likely null |
| R20 C6 | att. 4 C6 k = 5 / 6: 0.499 / 0.533; PWL 0.493–0.495 / 0.527 | already measured |

## 7. Short reading (inference, labelled)

- **HB models are smoothed interpolation.** Everything gen21 gained over RIDGE2 at C3 k = 3 is reproduced by interpolation of the same four reads (verified). The fitted HB residual is close to a random walk (ρ ≈ 0.986), which is why HB ≈ PWL (inference).
- **No zero-shot route remains.** gen13–gen20.1 close every zero-shot route to strict-adjacent log SF (best G14, R² 0.053).
- **Corpus-learned add-ons fail on the read cells.** Pooling, mixtures, metal vectors and the neural contrast net all fail on top of the reads.
- **The untested levers** are those the registry names:
  - a shrunk or chemistry-conditioned Gd term (gen13 D7 already shows chemotype dependence);
  - read placement for PWL's weak heavy pairs;
  - a real replicate-based measurement of the log SF noise ceiling, which nobody has estimated.


## key facts
- PWL (piecewise-linear interpolation of the read log D) is computed in NO committed gen13-gen21 report; it appears only in 'generations/gen21/INSTRUCTIONS FOR NEW IDEAS ROUND 3.md':31-40, which cites generations/gen21/AUDIT_2026-10-19.md, and that file does not exist in the repo.
- Verified by recomputation (scratch only, identical rows to gen21 attempt4 predictions, RIDGE2 reproduced at 0.3099): C3 k=3 BP PWL_Z 0.4118, PWL_R 0.4179 vs HBK@ML 0.4142; BPG PWL_Z 0.4107, PWL_R 0.4180, HBK@ML 0.4188; C6 k=6 BP PWL_Z 0.5272 vs HBK@ML 0.5329. At k=1 and k=2 PWL_R (0.194, 0.360) beats HBK@ML (0.099, 0.344).
- HBK@ML minus PWL_R at C3 k=3 BP per seed: -0.0118, +0.0136, -0.0203, +0.0038, -0.0040 (2/5 positive). Error correlation HBK@ML vs PWL_R = 0.971; a post-hoc 50/50 blend gives 0.424 (dev look, not a method).
- HB2+GP's fitted residual is near a random walk: rho median 0.986 (0.973-0.9997), tau2 median 0.98 (gen21/results/attempt2/fitted_BP.csv). This mechanistically explains HB approximately equal to PWL (inference).
- gen15/gen20 'LINE'/NAIVE_LINE is a one-parameter radius-slope least-squares fit (gen15/fewshot.py:140-149; gen20k/adapters.py:56-63), not PWL. Strict-adjacent R² 0.168/0.237/0.236 at k=1/2/3 (GEN20_REPORT.md:170-174).
- The reference RIDGE2 gives 0.310 at C3 k=3 (GEN20_REPORT.md:175; gen21/PROTOCOL.md:113-119). All gen21 'gains over RIDGE2' (+0.094 HB2+GP, +0.104 HBK@ML) are matched by PWL_R (+0.108).
- Zero-shot strict-adjacent R²: G14 0.053 (GEN20_REPORT.md:168), NNC -0.322, ExtraTrees log-D-then-difference -0.010 / direct -0.016 (gen20_1/README.md:21), HB k=0 -0.116, FLAT -0.135. Every zero-shot route is closed.
- Neural contrast net NNC@SELECTED: k=3 R² -0.128 BP, -0.215 BPG, -0.29 on confirmation; significantly worse than LINE (GEN20_REPORT.md:136-147,178,271-274,327-337).
- gen20 read policies differ by <=0.005 macro MAE for its arms (GEN20_REPORT.md:213-230). For HB, SERIES_STAGGER costs -0.017 at k=3 but gains +0.10 at k=1 (ATTEMPTS.md:220-224,252-254). DOPT_HB was never run (ATTEMPTS.md:371-383).
- Pooling a condition series' reads (HBG@POOL, setting P) was refuted: -0.041 vs own reads with staggered reads, +0.002 with DOPT_Q reads (ATTEMPTS.md:197-257). gen20 Setting B COPY (same publication) gives R² 0.32 on 24 units (GEN20_REPORT.md:310-316).
- Free per-metal effects (HBKM@BIC) were refuted: 0.407 vs 0.414, MAE worse in 0/5 seeds; the Gd break helps non-DGA chemotypes and over-corrects DGA (ATTEMPTS.md:478-525). gen13 stage 2 D7 had already found a Gd dip of -0.127 outside sc009 vs -0.025 inside (D7/FINDINGS.md:32-33); gen21 does not cite this.
- Heteroscedastic mixture HBMIX is neutral (+0.0015, 2/5 seeds). Noisy cells are 37.5% of C3 cells and carry 79% of SSE (ATTEMPTS.md:411-441).
- log SF replicate noise has never been estimated: gen20.1 found only 115 run-paired strict-adjacent candidate pairs in 20 cells from 2 chemotypes, so it is 'not estimable' (gen20_1/PRE_REGISTRATION.md:487-492). Cited log D replicate sds range 0.205-0.337 (medians 0.237/0.302/0.225/0.333/0.299). gen20 capped noise s has median 0.106; gen21 ML sigma2 has median 0.006.
- Arithmetic bound (inference): on C3 k=3 neither-end-read rows the MSE is 0.0304, so independent per-metal noise satisfies sigma <= 0.12 log D. This contradicts the 'adjacent SF inside the 0.30 replicate noise' narrative of gen13/gen20.
- GEN20_REPORT.md:228-229 says 'the first two reads are always the lightest and heaviest legal metals'. A label-free check shows this holds for UNIFORM (1145/1145 C3 instances, BP) but not for DOPT_Q (165/1145). DOPT_Q reads Lu in only 17% and Tb in 0.5% of C3 instances at k=3.
- gen15's '+0.065 model over naive line at D-opt k=3' is an all-pairs pair-model result with 24.8% of adjacent rows closure-determined (GEN20_REPORT.md:105-108). On strict adjacent, G14@BLUP20 ties LINE (+0.003, p 0.41).
- DMDPhPDA corruption (5 cells; COH-NODMD) mainly affects log D. In log SF only the 5 M Pr/Nd pair is off by +1.0 (gen20_1/IMPLEMENTATION_PLAN.md:82-90). gen20.1 found no verdict change; gen21 never ran COH-NODMD.
- The in-progress, untracked round-3 registry (generations/gen21/next_research_20261019/EXPERIMENT_REGISTRY.csv) has rows R03, R04/R05, R07, R08, R09-R11, R13, R19 and R20. Each is close to an earlier tested idea; the overlaps are mapped in section 6 of the report.
- Disclosure: this inventory computed new dev-seed breakdowns (per pair type, endpoint split, sc009 split, error correlation, 50/50 blend) that round 3 should log as dev looks. I ran one existence-only ls on C:/Users/Bandai/gen21_withheld with output discarded; no file there was opened. No repository file was modified.

## open
- Where is AUDIT_2026-10-19.md (and audit.pdf)? The round-3 brief cites PWL numbers and a DMDPhPDA 'mantissa-only' defect from it, but it is not in the repository. Should the round-3 log cite my scratch recomputation instead?
- Should round 3 treat my new dev looks as disclosed contamination before its registry is frozen? They are: the 50/50 HBK+PWL blend R² 0.424, error correlation 0.971, per-pair PWL vs HBK, and the sc009 / endpoint splits. They bear directly on R13/R22 STACK and R09 POL_ERR.
- Is any replicate whole-curve data obtainable (SAFE raw rows, other series keys) to estimate the strict-adjacent log SF noise ceiling? gen20.1 found it not estimable (115 run-paired pairs, 2 chemotypes).
- Does the gen13 D7 finding (Gd dip -0.127 outside sc009, -0.025 inside) mean a chemotype-dependent Gd term can be learned without held-out-family labels, given only ~4 effective chemotypes? R08 needs this to count as a new test.
- Is PWL's weakness on the heavy end (Yb/Lu, Tm/Yb near R² 0, with Lu read in only 17% of DOPT_Q read sets) a read-placement issue, as R09 POL_ERR assumes, or a noise issue? gen20 found heavy pairs inside noise for every arm (GEN20_REPORT.md:249-255).
- gen20's statement that 'the first two reads are always the lightest and heaviest legal metals' is false for DOPT_Q. Does the gen20 report need an erratum, and does it change gen20's policy conclusion (C.6)?