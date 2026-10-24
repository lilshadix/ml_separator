# E1 — What has been tried to predict lanthanide selectivity (gen13 → gen21 round 4), and what is still open

Written 2026-10-23 for the round-5 design. This is a read-only synthesis plus a set of **new exploratory diagnostics**
(section 3) computed on the existing round-4 out-of-fold (OOF) predictions and the frozen bench. Nothing in the
repository was modified. Every number is cited to a file (paths relative to `D:\ml_separator_gh`) or to the
section-3 computation. Tags: **[M]** = measured fact in a repository file or recomputed here; **[I]** = interpretation.

---

## 0. Read this first: the regimes are not comparable across generations

| generations | target / metric | pairs | information at prediction time |
|---|---|---|---|
| gen13–gen16 | extractant-macro MAE of pairwise log SF (log10) | **all pairs**; gen13's "adjacent" = ΔZ 1 **plus Nd–Sm** | zero-shot, or the **pair model** (measured SFs); gen15 scored rows that the supports fix by arithmetic (24.8 % of adjacent rows; `generations/gen20_kshot/GEN20_REPORT.md` C.1) |
| gen19 | log D MAE; V5-PAIR / V6 log SF MAE | other corpus (`dataset_all_metals`) | hidden metal × extractant cells |
| gen20, gen20.1, gen21 | **pooled R² of strict-adjacent log SF** (mean over 5 split seeds), cohort C3 = 229 cells / 1,802 pairs per seed | strict ΔZ = 1, 12 pair types | **metal model**: k + 1 metals read (k = 3 → four metals), never both ends of a scored pair |

C3 target: mean −0.0815, sd 0.222, variance 0.0493, RMS about 0 = 0.236; R² 0.40 / 0.50 ≈ R²₀ 0.471 / 0.559
(`generations/gen21/next_research_20261019/inventory_rejected.md` §1). Designs: B (chemotype hold-out), A (exact
extractant), BR/BQ (random controls), **BP** (chemotype + the held-out cells' publications removed), **BPG** (BP + gen19
cross-publication copy groups; deciding design in gen21 round 4).

---

## 1. Timeline in one table

| gen | question | headline (regime) | verdict |
|---|---|---|---|
| 13 | zero-shot curve model for unseen extractants | all-pairs macro MAE 0.501 (B) ≈ incumbent row trees 0.495; **0.54 under BP**; conditions-only 0.471 (B) → **0.625 (BP)** (`generations/gen13_separation/DECISION_REPORT.md` headline) | conditions = laboratory fingerprint (94 % 1-NN publication ID); transferable chemistry = **one bit** (direction), 0.768 macro accuracy BP vs 0.559 |
| 13 st.2 | where is the error | 82–87 % of pairwise squared error is amplitude; Gd dip −0.072 (D7), only 7.2 % of residual variance is a common curve (`analysis/stage2/D7/FINDINGS.md`) | one measured far pair 0.541 → 0.266 (BP); line through it 0.272 |
| 14 | direction classifier + magnitude prior | L2 logistic on 39 donor-topology columns 0.821 BP macro accuracy; direction × constant magnitude 0.500 BP, design-invariant to 0.010 (`generations/gen14_direction/GEN14_REPORT.md`) | direction exhausted (oracle +0.012–0.015); 7 magnitude priors within 0.007 of a constant |
| 15 | curve, floor, measured mode | FLAT 0.5885 (BP) < MEAN_CURVE 0.6217; **honest zero-shot gain G14 − FLAT = +0.088, p 0.047** (WCR p 0.059); measured mode G14@doptk3 0.170 (`generations/gen15_curve/GEN15_REPORT.md`) | estimator axis closed (12 priors within 0.007); xTB composition trap; oracles absorb noise (~half) |
| 16 | six leads, sealed, confirmation seeds | 1 confirmed claim (direction as screening filter, +1.909 measurements; **+0.005 within one laboratory**); L1 xTB cycle ρ −0.084 with reliability 0.80 (powered null); L2 curvature headroom +0.029, MDE 0.039; L4 killed by featureless order (`generations/gen16_leads/DECISION_REPORT.md`) | nothing beats lean blocks zero-shot |
| 16 side | anchors, protocol | anchor regression best 0.5715 vs G14 0.5001, 41/41 contrasts negative (`generations/gen16_anchor/README.md`); percentile bootstrap size 0.07–0.08 at nominal 0.05; 75 of 82 extractants measured in one publication only (`generations/gen16_protocol/README.md`) | unreviewed, exploratory |
| 17 | within-publication pair differences | cleanest contrast (same publication, identical conditions, 362 pairs) null 8/8 feature sets × estimators (memory note gen15; `generations/gen17_pairdiff/README.md`) | null |
| 18 | process chain (D model → cascade) | pooled-slope mass action 1.0904 vs nearest-condition 1-NN 1.0637 (R1 NULL); one-measurement offset worse (1.2551) (`generations/gen18_process/GEN18_REPORT.md`) | not a selectivity model |
| 19 | chemistry transfer on the multi-metal archive | descriptor CatBoost M0 beats factorised M2 by 0.154 log D; **Pr/Nd S2 FAIL** (log SF MAE 0.159 > FLAT 0.137); actinide transfer F4 "negative" holds under the conservative reading (WITHOUT better by 0.026, BCa [0.003, 0.078]) (`generations/gen19_chem_transfer/SUMMARY.md`) | log D skill ≠ log SF skill |
| 20 | k-shot, strict adjacent, metal model | RIDGE2 **0.310** at k = 3; LINE 0.236; G14@BLUP20 0.248; neural NNC **−0.128** (−0.29 on confirmation); G14 zero-shot 0.053; read policies differ ≤ 0.005 macro (`generations/gen20_kshot/GEN20_REPORT.md`) | registered neural arm significantly worse than a line |
| 20.1 | absolute log D vs log SF, zero-shot | log D R² 0.258 (85 % of its error is cell level, cancels in SF); log SF from log D −0.010, direct −0.016; G14 0.053 (`generations/gen20_1/GEN20_1_REPORT.md`) | 0.4 RULED_OUT zero-shot |
| 21 r1 | three ideas at k = 3 | HB2+GP 0.403 BP / 0.407 BPG; observable-target RIDGE2 0.357; sibling-read pooling refuted (−0.041) (`generations/gen21/GEN21_REPORT.md`) | candidate HB2+GP |
| 21 r2 | three more | HBK@ML 0.414 / 0.419; HBMIX +0.0015; free per-metal offsets (HBKM@BIC) 0.407, refuted (`generations/gen21/GEN21_ROUND2_REPORT.md`) | candidate HBK@ML |
| 21 r3 | 28 configs toward 0.50 | **PWL_R 0.418 = HBK@ML**; R14 (PWL_R + one Gd amplitude) 0.452; R23 (+ POL_ERR reads) 0.475 dev, **0.471 / 0.463 on the once-only final seeds**; rule candidate R02 0.419 (no gain) (`generations/gen21/next_research_20261019/FINAL_REPORT_RU.md`) | 0.50 needs ~6 metals (seed mean), 7 by lower bound |
| 21 r4 | sealed, pre-registered blocks A–C | **B1 0.4521 / 0.4519** unbeaten; S01 = B1, S02 −0.024, S05 −0.006, S06 −0.006/−0.011, S11 −0.0005/−0.0017; An gate 3/50; power: ~40–80 new sources to confirm a B1-sized gain (`generations/gen21/round4_20261022/REPORT.md`, `POWER.md`) | nothing beats B1; corpus cannot confirm anything |

---

## 2. Approach-by-approach catalogue

Each entry: **what** → **result** → **why it failed / worked** ([M] vs [I]) → **what was left untested or dismissed weakly**.

### 2.1 Zero-shot structure → selectivity (no measured metal)

- **Curve/row models on ligand + condition blocks (gen13).** Registered curve model 0.501 vs row trees 0.495 (B), P1 null
  (−0.006, p 0.75). Under BP the condition block collapses (0.471 → 0.625, worse than the mean curve 0.639); lean binding-site
  blocks reach 0.558 (+0.096 over conditions-only) [M] (`gen13_separation/DECISION_REPORT.md` §5a).
  *Why:* the 64 `cond__` columns identify the publication at 94 % 1-NN [M]; design B rewarded reading the laboratory [I, well supported
  by the BR/BQ size-matched controls: BR costs 0.017, BQ 0.000, BP 0.154].
- **One bit per extractant (gen13 st.3, gen14).** Direction of selectivity from donor topology: 0.768 (trees) → **0.821 BP** (L2
  logistic, 39 columns), +0.09–0.14 over the 13-column donor census [M]. Mechanism: chelate bite (`frac_donor_pairs_within_3`
  Spearman −0.53 with amplitude) [M]. *Why it worked:* 22 distinct topology vectors over 82 extractants; a linear model extrapolates [I].
  *Exhausted:* oracle direction buys +0.012–0.015 MAE [M]. *Left open:* α-alkylated DGAs are light-selective with a topology vector
  identical to ordinary DGAs (2 examples; a 7-column steric block hurt, 0.920 → 0.720) — "needs a purpose-built descriptor and a cohort with
  more than two examples" [M, memory gen14 / `GEN14_REPORT.md`].
- **Magnitude / curvature from structure (gen14–15).** 12 priors (ridge, trees, loss-fitted, TabPFN, CatBoost, monotone XGBoost, GAM,
  4,602-term symbolic search) all within ~0.007 of a constant [M]. Curvature oracle O_CURV +0.073 in-sample but **≈ +0.036 honest**
  (leave-pair-out) [M]; shape alphabets exist (SHAPE4 oracle 0.444) but are unpredictable from 2D (0.55–0.63 vs 0.500) [M].
- **Honest floor.** FLAT (predict no separation) 0.5885 beats the corpus mean curve 0.6217 (BP); G14 − FLAT = **+0.088** [0.001, 0.147]
  [M] (`GEN15_REPORT.md`). On strict adjacent pairs this shrinks to **R² 0.053** (gen20), confirmed 0.04 on withheld seeds [M].
- **Zero-shot via absolute log D (gen20.1).** log D R² 0.258 is level-driven; differencing gives log SF R² −0.010 (direct −0.016) [M].
  Under design A (extractant hold-out, near analogues allowed) log D 0.410 but log SF only 0.078/0.090 [M].
- **Status:** every zero-shot route to strict-adjacent log SF is closed at R² ≈ 0.05 [M]. *Weakly dismissed:* none important; the
  α-alkyl DGA steric question is untestable with 2 examples.

### 2.2 Representations

| representation | result | file | why |
|---|---|---|---|
| ECFP / 206 RDKit descriptors | dilute the lean blocks in both designs | `gen13 DECISION_REPORT.md` §3 | [I] high-dimensional, chemotype-confounded |
| Pretrained chemical LMs (ChemBERTa) | direction 0.38–0.57 vs 0.821 (BP), all p < 0.01 | memory gen15 / `GEN15_REPORT.md` | [M] |
| Tanimoto kernels / GP / KRR (85 arms) | 0.474 under B → 0.499–0.506 under BP (= G14) | `GEN15_REPORT.md` | design-B mirage: the chemotype hold-out is itself Tanimoto-0.7 linkage [M] |
| External aqueous logK prior (BidentateGNN SDFs) | Spearman +0.00 with amplitude over 82 extractants; add −0.000 to −0.002 | `gen13 DECISION_REPORT.md` §4 | [M] |
| RESP3D (coordination-sphere response) | −0.006; = width-matched shuffle | same | [M] |
| xTB complex energies | +0.644 Spearman → artefact: **only 1 of 41 complete series has constant inner-sphere composition**; cycle-corrected dE ρ −0.084 with reliability 0.80 (powered null) | `GEN15_REPORT.md`; `gen16_leads/DECISION_REPORT.md` §4 | composition trap [M] |
| Donor-distance spread (3D) | ρ −0.34 with amplitude, LOCO-stable; block still loses on the endpoint (0.528 vs 0.500) | `GEN15_REPORT.md` | [M] |
| Compact 3D geometry at k = 3 (round 4 S06) | −0.006 BPG / −0.011 BP vs B1; same on the 89 % geometry-covered subset; inner gate 9/50 | `round4_20261022/results/blockB_decision.md` | synthetic GEO world: geometry claim 0/6 → **no power** at plausible sizes [M] |
| Generic f-shell features (n_f, tetrad, gd_break, half-shell, mphys) | gen13 bases interchangeable within 0.006; strain mphys −0.0044; S02 −0.024 | `round4_20261022/EVIDENCE_INVENTORY.md` M1–M7, E1–E3 | for Ln(III), n_f = Z − 57 is constant across adjacent pairs [M]; such terms are bases, not physics [I] |

*Untested / weak dismissals:* (i) the **168 ligand-at-complex strain single points** from gen16 L1 were never analysed —
"untested, not refuted" (`EVIDENCE_INVENTORY.md` §2). (ii) Only **60 / 229** C3 cells keep composition constant across metals, so any
cross-metal 3D energy can only be tested there [M].

### 2.3 Estimators and architectures

- Trees vs linear vs kernel vs kNN vs L1 boosting (gen13 st.3): trees best for MAE by 0.03–0.13; for direction, logistic beats trees [M].
- **Neural contrast network NNC (gen20):** worst arm at every budget (k = 3 R² −0.128 BP, −0.215 BPG, −0.29 confirmation); best where the
  design leaks (A) and worst under BPG — the over-parameterisation signature [M] (`GEN20_REPORT.md` C.8).
- **Factorised neural M1/M2 (gen19):** dropped; M0 (descriptor CatBoost) better by 0.154 log D [M].
- **Strain repository architectures:** anchored CatBoost + distance encoder +0.0156 R² on 444 "fresh" pairs — sign-only PASS rule, no
  interval, leave-extractant-out zero-shot with a binned condition key, pairs from the **same database** (and the strain corpus file *is* the
  gen21 corpus file); attention vs distance encoder ties on the union (−0.0003); FT/block transformers lose to CatBoost [M]
  (`EVIDENCE_INVENTORY.md` S1–S18).
- **Ln-only multitask encoder at k = 3 (round 4 S11):** −0.0005 BPG / −0.0017 BP vs B1; inner CV chose B1 in 24/25 and 23/25 folds [M].
- **Stacking (round 3 R13/R22):** +0.002/+0.003; member errors correlate 0.97 [M].
- *Reading [I]:* on this corpus capacity is never the bottleneck; every flexible learner either reproduces interpolation or loses
  under BP/BPG.

### 2.4 Measured mode: curve priors and interpolators (the k-shot line of work)

| method | strict-adjacent C3 k = 3 R² (BP / BPG) | notes |
|---|---|---|
| LINE (one radius slope through reads) | 0.236 / 0.237 | saturates after k = 1 (0.168 → 0.237 → 0.236) |
| G14@BLUP20 (gen15 deployed measured mode, rebuilt) | 0.248 | ties LINE (+0.003 macro, p 0.41); covariance indefinite in 100 % of 430 estimates (gen16 L5) |
| RIDGE2 (2-parameter curve shrunk to training prior) | 0.310 / 0.323 | only gen20 arm to beat LINE (+0.008 macro, < 0.02 margin) |
| RIDGE2+OBS@RESEL (observable target) | 0.357 / 0.358 | MAE only 3/5 seeds |
| HB2+GP (hierarchical ML curve, exponential residual in Z) | 0.403 / 0.407 | residual ρ median 0.986 ≈ random walk |
| HBK@ML (kernel chosen by likelihood) | 0.414 / 0.419 | se kernel in 19/25 folds |
| **PWL_R** (piecewise-linear interpolation along −CN8 radius, constant beyond the reads; **no training**) | **0.418 / 0.418** | = HBK@ML (−0.004, error correlation 0.971) |
| PWL_Z | 0.412 / 0.411 | |
| PCHIP_R (monotone cubic) | 0.400 / 0.400 | worse than PWL (−0.018) |
| **B1 = PWL_R + one fold-fitted Gd amplitude** | **0.4521 / 0.4519** | the reference; gain at Eu/Gd (0.276 → 0.422) and Gd/Tb (0.321 → 0.546) |
| HIST_D7 (B1's form with fixed −0.072) | 0.4511 / 0.4506 | = B1 (−0.001) |
| B2 (14 coherent potentials + 6 read-geometry features, one λ) | 0.4299 / 0.4246 | **worse than B1** (−0.022, 0/5 seeds) |

Sources: `GEN20_REPORT.md` C.4; `GEN21_REPORT.md`; `GEN21_ROUND2_REPORT.md`; `next_research_20261019/results/ROUND3_TABLE.csv`;
`round4_20261022/results/REF/TABLE.md`.

*Why the hierarchical models equal PWL [M + I]:* the ML-fitted residual is a near random walk (ρ 0.986, `gen21/results/attempt2/fitted_BP.csv`);
the conditional mean of a random walk given point reads is piecewise-linear interpolation [mathematical fact]. All gen21 "gains over
RIDGE2" (+0.094, +0.104) are matched by PWL_R (+0.108) [M]. gen20's sentence "measurements alone do not predict" rested on the
minimum-norm PROJ arm (R² −4.55), not on interpolation [M] (`inventory_rejected.md` §5.6).

*Why B1 works [M]:* one shared coefficient (mean −0.094 ± 0.015 over 25 BP folds) on a read-set-and-position feature; it is the
known series dip at Gd (gen13 D7 −0.072, 268 cells incl. 224 of the 229 C3 cells). The registered within-pair-type shuffle keeps
86–90 % of the gain (CTRL14W/CTRL23W) — it learns a **pair-type mean**, not cell chemistry. The fixed historical constant does as
well (HIST_D7). **B1's learned content is one known number.**

*Why B2 lost [I, partly M]:* shrinking 20 columns under one λ diluted the single real effect; inner CV still chose finite λ in 19–20/25 folds.

### 2.5 Metal-specific / electronic terms

| attempt | result | file |
|---|---|---|
| gen13 bases with Gd step, tetrad e1/e3 | within 0.006 of radius + radius² | `gen13 DECISION_REPORT.md`; `t1_leaderboard.md` |
| gen13 D7 residual catalogue | Gd **dip** (single element) −0.072, not a step; no Eu anomaly; Gd −0.127 outside sc009 vs −0.025 inside; **Nd dip sc009-specific** (−0.109 vs −0.007); between-chemotype heterogeneity p 0.33; prize 0.001–0.003 MAE | `analysis/stage2/D7/FINDINGS.md` |
| HB3H+GP Gd hinge (r1) | +0.000 | `gen21/ATTEMPTS.md` |
| HBKM@BIC free per-metal means (r2) | 0.407 vs 0.414; MAE worse 0/5 seeds; α_Gd −0.096 ± 0.020; helps non-DGA, over-corrects DGA; α_Tm unstable | `GEN21_ROUND2_REPORT.md` |
| R03 shrunk 14-metal block (r3) | +0.009 [−0.001, +0.019], MAE 3/5 | `ROUND3_TABLE.csv` |
| **R14 / B1 single Gd amplitude** | +0.034, replicated on final seeds (+0.032 / +0.031) | `FINAL_REPORT_RU.md` |
| R08 Gd × chemistry (r3) | +0.004 [−0.016, +0.036], DGA −0.023 | same |
| **S01** donor-class (DGA / SOFT_N / OTHER) Gd and Nd dips (r4) | −0.0001 / −0.0005 vs B1 | `round4_20261022/results/blockA_decision.md` |
| **S02** electron-count shapes × donor class (r4) | −0.024 / −0.028; losses in sc009, Eu/Gd, Yb/Lu | same |

*Why [M]:* the only portable piece is the shared Gd dip. Synthetic run 2 (round 4) showed a **D7-sized donor contrast is not
detectable** with these rules (CHEM_D7 world: S01 +0.004, criterion 1 0/8), so S01's null is "not detected", not "refuted"
(`round4_20261022/AMENDMENTS.md` A1). *Not run:* S03/S04/S21 (routes closed because S02/S01 failed criterion 1).
*Interpretation guard:* a Gd-centred gain does not establish a half-filled-shell mechanism (outer-sphere nitrate/water effects are an
alternative; `round4_design_20261022/RESEARCH_IDEAS.md`).

### 2.6 Correctors on top of interpolation

| corrector | Δ vs reference | file |
|---|---|---|
| R02 read-geometry ridge (rule candidate r3) | +0.001 [−0.002, +0.005]; final seeds +0.0002 / −0.0002 | `FINAL_REPORT_RU.md` |
| R07 chemistry × curvature | +0.001 | `ROUND3_TABLE.csv` |
| R19 Huber robust corrector | +0.002 | same |
| R04/R05 task-loss kernel / noise selection | −0.005 / −0.007 | same |
| S05 CatBoost on support summaries + donor class + RDKit (r4) | −0.006 / −0.003 vs B1; inner CV chose B1 in 16–19/25 | `blockB_decision.md` |
| S06 = S05 + compact 3D | −0.006 / −0.011 | same |
| S11 multitask encoder (residual + raw log D heads) | −0.0005 / −0.0017; BP balanced-MAE −0.0020 with intervals excluding 0 | `blockC_decision.md`, `REPORT.md` |

*Why [I, supported by §3]:* after interpolation the residual is overwhelmingly **metal-local** and heavy-tailed (section 3.2–3.3), which
smooth or tabular correctors cannot predict from chemistry; what is reproducible sits at chemotype/publication level, which BP/BPG
remove by construction.

*Weak dismissal:* **solvent and extractant concentration** were deliberately left out of R07's chemistry block because "they give
away the publication" (`FINAL_REPORT_RU.md` §3). Under BP the publication is already removed from training, so the leakage argument
does not forbid testing them; the "condition × series-shape" interaction at k = 3 is genuinely untested (gen15 condfe: acid worth
+0.009 on all pairs zero-shot, inner LOPO zeroed it in 90 % of folds).

### 2.7 Read policies (which metals to measure)

| policy comparison | result | file |
|---|---|---|
| widest vs D-optimal vs random (gen15, pair model, all pairs) | D-opt − widest at k = 3 +0.033 (P1 in 5 designs); widest best at k = 1 | `GEN15_REPORT.md` |
| RANDOM / UNIFORM / DOPT_CURVE / DOPT_Q / UNC_SEQ (gen20, metal model) | spread ≤ 0.005 macro MAE; DOPT_Q not the best | `GEN20_REPORT.md` C.6 |
| SERIES_STAGGER for HB (r1) | +0.10 at k = 1, −0.017 at k = 3 | `gen21/ATTEMPTS.md` |
| POL_ERR (train-fitted PWL error table) vs DOPT_Q (r3) | PWL_R +0.019 [−0.007, +0.053]; R23 − R14 +0.023 [−0.003, +0.057] dev, +0.017 / +0.009 final | `FINAL_REPORT_RU.md` |
| DOPT_HB (minimise HB posterior variance) | **never run** — posterior variance is inverted against error across pair types (Spearman −0.71 / −0.57) | `GEN21_ROUND2_REPORT.md`; `FINAL_REPORT_RU.md` §6 |

*Facts worth carrying forward [M]:* DOPT_Q reads La in 71 %, Gd 58 %, **Ce 4 %**, **Tb 0.5 %**, Lu 17 % of C3 instances
(`inventory_rejected.md` §2.2); gen20's statement that "the first two reads are always the lightest and heaviest legal metals" is
false for DOPT_Q (165/1,145). *Untested:* a label-free policy optimised for **B1** (not PWL) error; any policy at k = 4.

### 2.8 Pooling, siblings and "Setting B"

- Same-extractant other-cell ceiling (gen13, all pairs): 0.277 / 0.247 vs 0.461 [M].
- gen20 Setting B: copying the adjacent SF of another condition of the same extractant and publication gives R² 0.32 on 24 units;
  models add +0.0009 [M] (`GEN20_REPORT.md` C.10) — not admissible under BP.
- HBG@POOL (r1, pool the series' **reads**): −0.041 with staggered reads, +0.002 with DOPT_Q reads → refuted [M].
- gen17 sibling anchor (copy siblings' coefficients) 0.345 vs zero 0.543 (all pairs, includes same-paper siblings; unreviewed) [M].
- gen16 "seeing the laboratory" (BP1/BP1X, positive = letting the model see the laboratory helps) for the zero-shot G14:
  −0.0025 (p 0.89; 42 extractants) / +0.0205 (p 0.02; 21 extractants, 12 chemotypes) [M] (`gen16_protocol/README.md`; unreviewed).
- *Untested:* a k = 3 model that uses the **full curves** of a test cell's family or laboratory siblings (a legal regime only if
  declared, e.g. "new condition of a known extractant / laboratory"). Section 3.4 bounds what it could give.

### 2.9 Noise models, heteroscedasticity, data cleaning

- Observable vs latent target, shared vs independent reading noise: immaterial (≤ 0.002) [M].
- HBMIX (two variance regimes): noisy component = 37.5 % of held-out C3 cells and **79 % of squared error**; regime identified (AUC 0.86)
  but noisy-cell MSE unchanged (0.0564 vs 0.0566) [M] (`GEN21_ROUND2_REPORT.md`).
- Label de-noising, precision weighting, abstention: null (gen13–15) [M].
- Replicate noise: median replicate sd **0.302** (gen13, used for years) is driven by hidden-condition merges; excluding them gives
  **0.049** (n = 86); 53 twice-read adjacent pairs give sd(Δ log SF) 0.086 with errors correlated 0.983 — transcription, not laboratory
  repeatability, which remains **unmeasured** [M] (`next_research_20261019/DATA_QUALITY_REPORT.md`). gen20.1 found log SF noise "not
  estimable" (115 run-paired pairs, 2 chemotypes).
- Data version V2 (quarantine DMDPhPDA copy B, stereo-merged Me2-TODGA, D2 duplicates, 17 hidden-condition merges; censor 24 floor
  readings): V2 C3 = 208 cells / 1,620 pairs; PWL_R 0.424, R18 (R23 on V2) **0.486** — a different benchmark, not a model gain [M].
- Round-4 provenance: C5-BTBP (15 cells, 148 C3 pairs) light lanthanides at log D ≈ −4.2 (suspected floor; attribution Nilsson 2006 vs
  Foreman 2005 doubtful); DMDPhPDA 5 M Pr/Nd off by exactly +1.000 decade; 79/229 cells have no data location; G-V3 failed, so
  S17/S18 NOT RUN [M] (`round4_20261022/PROVENANCE.md`).

### 2.10 Transfer from other metals and other data

- gen11 (pre-gen13): all-actinide arm +0.0491 macro log D MAE, BCa [−0.0005, +0.119] → INCONCLUSIVE; matched contrast +0.0142
  [−0.034, +0.085]; all oxidation states pooled [M] (`EVIDENCE_INVENTORY.md` G11a–d).
- gen19: H3 UNDECIDED; F4 negative transfer holds under the conservative reading [M].
- Strain Ln + Am/Cm: Ln gain +0.0021, seed sd 0.0204 [M].
- Round 4 block C: 1,462 verified An(III) rows (88 % Am, 38 publications, 112 extractants); after union exclusions ≥ 557 rows / 12 copy
  groups / 77 extractants per fold; **inner gate: S13 beat S11 and B2 in 3 of 50 fold-fits** (30 needed) → S13–S16 NOT RUN; synthetic
  AN_SIGNAL world: transfer claim 0/6 (S13 adds ≤ +0.002 over S11) → "not detected; control power not shown" [M].
- Extra lanthanide series in the archive: 317 outside the cohort, 16 with ≥ 4 metals, **all** on cohort extractants (14 from cohort
  publications); after fold exclusions the minimum is **0** independent series → S12 NOT RUN [M] (`PROVENANCE.md`). The brief's "new"
  source D4GC01146E is already in the cohort.

### 2.11 Budget (number of measured metals)

C6 cohort (133 cells / 1,349 pairs): PWL_R@DOPT_Q 0.397 / 0.495 / 0.526 at 4 / 6 / 7 metals; PWL_R + Gd @POL_ERR(k) 0.455 / **0.542**
/ **0.577**; lower 95 % bound reaches 0.50 only at 7 metals with the Gd term (0.514 / 0.518) [M] (`FINAL_REPORT_RU.md` §3). The k = 4
(five-metal) rung was never measured [M] (memory `project-gen21-logsf-target`). C6 is the harder subset of C3 (R23 on the 96 C3-only
cells 0.529) [M].

---

## 3. New exploratory diagnostics for round 5 (dev looks on existing OOF rows; computed for this note)

All on `generations/gen21/round4_20261022/predictions/REF_{BPG,BP}.parquet` (k = 3, DOPT_Q, 5 dev seeds) and the frozen bench
(`gen20k.data.load_bench()`, 521 cells, replicate-mean log D). No model was fitted to outer folds; these are descriptive and some
are **transductive upper bounds** (they use other held-out cells' labels), labelled as such. Reproduction checks passed: B0 0.4180,
B1 0.4521, B2 0.4299, HIST_D7 0.4511, HBK@ML 0.4188, RIDGE2 0.3225 on BPG (= `results/REF/TABLE.md`). BP gives the same picture
to ±0.002 throughout; BPG numbers are quoted.

### 3.1 Where B1's squared error sits [M]

| slice | share of B1 SSE | share of rows |
|---|---|---|
| La/Ce + Ce/Pr | **0.342** (0.162 + 0.180) | 0.145 |
| Pr/Nd, Sm/Eu, Eu/Gd | 0.086 / 0.108 / 0.096 | 0.070 / 0.088 / 0.088 |
| the four heaviest pairs (Ho/Er … Yb/Lu) | 0.183 | 0.326 |
| one end read / both hidden | 0.647 / 0.353 | 0.670 / 0.330 |
| sc009 (DGA supercluster) | 0.410 | 0.489 |
| source cluster g:pub_74428aeab2 (the 65 % cluster) | 0.518 | 0.649 |
| source cluster g:pub_af7a1cd919 | **0.245** | 0.094 |
| **one publication, C5-BTBP `pub_6a0ef8528c`** (15 cells) | **0.231** | 0.082 |

- 20 of 229 cells carry 50 % of B1's SSE; 66 cells carry 80 %.
- **Heavy tails:** the top 1 % of rows (~18 per seed) carry 31 % of B1's SSE, the top 5 % carry 59 %, the top 10 % 73 % (B0: 29 / 57 / 71 %).
  The target itself is heavy-tailed (top 5 % of rows = 48 % of the total sum of squares).
- B1 R² on the BTBP publication alone: 0.238; **pooled B1 R² without that one publication: 0.486** (B0 0.459).
- Per pair type, B1 R²: La/Ce 0.353, Ce/Pr 0.419, Pr/Nd 0.420, Eu/Gd 0.422, Gd/Tb 0.546, Tm/Yb 0.030, Yb/Lu 0.002
  (`results/REF/per_pair_BPG.csv`).
- [I] The "hard heavy end" is a low-variance R² artefact: the heavy pairs carry less error than their row share. **Pooled R² is
  driven by the light end, by one suspect publication and by ~5 % of rows.**

### 3.2 B1's residual is metal-local, not curve shape [M + I]

Within a (seed, cell), consecutive strict-adjacent pairs (m, m+1) and (m+1, m+2):
- residual correlation **−0.522** (n = 4,949) when the shared metal is unread; **−0.218** when it is read; lag-2 pairs (no shared metal)
  **+0.091** (n = 5,860).
- A residual made of independent per-metal deviations gives exactly −0.5 at lag 1 and 0 at lag 2; a smooth shape error gives positive
  correlations at both. [I] Solving the two-component model (white + smooth) puts the smooth part at roughly 10 % of B1's residual
  variance (lag-2 covariance 0.0020 against residual variance ~0.022–0.027). This matches gen13 D7 (7.2 % of the residual is a common
  curve) on a different estimator.

### 3.3 Information ceilings at k = 3 [M]

- **Dense-read oracle:** keep exactly the k = 3 hidden ends hidden, but read *every other* measured metal of the cell and interpolate
  (PWL along −radius). On seed 104729's rows: **B1 0.449 → dense oracle 0.461** (both-hidden rows 0.397 → 0.405; one-read rows
  0.476 → 0.491). Even 8–10 extra reads elsewhere buy ~+0.01: at fixed hidden metals the error is the hidden metals' own deviation.
- **Smooth-curve fits** to all other metals (leave-pair-out, both ends hidden): linear 0.196, quadratic 0.165, quadratic + Gd 0.212,
  cubic −0.40 — worse than dense PWL (0.230) on the same both-hidden framing; low-order global curves do not describe real series at
  the adjacent-difference level.
- **Noise-floor bound:** from second differences along −radius on the 229 C3 cells, solving var(a_m) = s_m + w²s_lo + (1−w)²s_hi for
  per-metal local variances s_m, a floor that counts only the hidden ends' s_m gives an R² ceiling of **0.65** at k = 3 with raw
  variances and **0.92** with robust (MAD) variances. The gap is the heavy tail: most cells are smooth, a minority carry large local
  deviations. Estimated raw local variance is largest at the light end (Ce ≈ La 0.042, Pr 0.029, Sm 0.020, Gd 0.018) and small for
  Tb–Er (0.005–0.007) and Yb/Lu (0.001).
- [I] 0.50 at k = 3 is not excluded by the floor, but it requires predicting or avoiding a heavy-tailed minority of metal-local
  deviations, not a better smooth curve.

### 3.4 Reproducible structure exists, but at levels the hold-out removes [M; transductive upper bounds]

Pair-type offsets of B1's residual, estimated **leave-one-cell-out within a group from the other cells' OOF residuals**, shrunk
n/(n + 10) (an oracle on group membership and on other held-out cells' labels, so optimistic):

| grouping | B1 + group offsets, R² (BPG) | without BTBP publication (base 0.486) | share of gain from BTBP rows |
|---|---|---|---|
| none (global pair-type constants) | 0.459 (+0.007) | — | — |
| donor class DGA / non-DGA | 0.468 | — | — |
| chemotype | 0.501 (+0.049) | 0.510 (+0.024) | 56 % |
| source cluster | 0.497 | 0.505 | 61 % |
| extractant | 0.501 | 0.512 | 55 % |
| **publication** (laboratory siblings) | **0.524** (+0.072) | **0.542** (+0.056) | 37 % |

The chemotype-level gain is spread over Pr/Nd (0.017 of SSE), Ce/Pr (0.017), Eu/Gd (0.013), Gd/Tb (0.012), Dy/Ho (0.008).
For the Gd amplitude alone: one global β in-sample 0.4565; per DGA class LOCO 0.460; **per chemotype LOCO 0.453 (no gain)**;
per source cluster LOCO 0.471; per extractant LOCO 0.470; per cell in-sample 0.548 (but a per-cell Gd amplitude is algebraically
inseparable from Gd's own measurement deviation, so that number is a noise-absorbing oracle).

Second differences on the bench (C3 cells, interior metals; one-way ICC over groups with ≥ 2 cells):

| metal | n | mean | sd | sc009 / non-sc009 mean | ICC extractant / publication / chemotype |
|---|---|---|---|---|---|
| Ce | 118 | +0.056 | **0.244** | +0.052 / +0.058 | 0.09 / 0.03 / 0.12 |
| Pr | 119 | +0.001 | 0.203 | +0.024 / −0.012 | 0.19 / 0.25 / 0.20 |
| Nd | 127 | −0.042 | 0.151 | **−0.123 / +0.004** | 0.29 / 0.21 / 0.29 |
| Sm | 157 | +0.026 | 0.162 | +0.020 / +0.031 | 0.00 / 0.22 / 0.07 |
| Eu | 154 | +0.050 | 0.143 | +0.037 / +0.060 | 0.30 / 0.35 / 0.22 |
| **Gd** | 127 | **−0.115** | 0.148 | **−0.039 / −0.167** | **0.69** / 0.48 / 0.41 |
| Tb | 163 | +0.068 | 0.109 | +0.060 / +0.074 | 0.21 / 0.26 / 0.11 |
| Dy, Ho, Er, Tm, Yb | 130–172 | −0.014 … +0.024 | 0.06–0.11 | ≈ equal | −0.11 … 0.38 |

- [M] The Gd dip is strongly extractant-reproducible (ICC 0.69) and differs between sc009 (−0.04) and the rest (−0.17); the Nd dip is
  an **sc009 property** (−0.12 vs +0.00), not a DGA-class property (D7 said the same).
- [I] Under chemotype hold-out, sc009 (half the rows) is predicted from the other chemotypes, so neither its weak Gd dip nor its Nd dip
  can be learned unless a descriptor bridges sc009 to other chemotypes. This explains S01 = B1 and the R08 DGA over-correction better
  than "the donor contrast is absent".
- [M] The Ce deviation is heavy-tailed (5th/50th/95th percentiles −0.19 / +0.03 / +0.43; 20 of 118 cells with |a| > 0.3 carry 87 % of
  its variance), barely reproducible within extractant (ICC 0.09), Spearman −0.34 with acid molarity and −0.22 with extractant
  concentration; publication means +0.22 (BTBP), +0.18, +0.12 for three publications. [I] It behaves like a condition- or
  sample-specific artefact (or Ce(III)/Ce(IV) chemistry); it is never read by DOPT_Q (4 %), and the two Ce pairs carry 34 % of SSE.

---

## 4. Lessons that must constrain round 5

1. **Select under BP/BPG only; report every design.** Conditions identify the publication (94 % 1-NN); design-B wins evaporate under
   BP (conditions-only 0.471 → 0.625; Tanimoto kernels 0.474 → 0.50; gen14 direction classifier trained on all 521 cells +0.006 under B / −0.046 under BP). The amplitude-only arm
   passed under one design and failed under four (`feedback-evaluation-protocol`).
2. **Compare with the cheapest sensible alternative.** FLAT, not the mean curve (worse than flat); PWL, not RIDGE2 (every gen21 "gain"
   over RIDGE2 was interpolation); **B1, not PWL**, for anything claiming chemistry; and a matched-capacity universal prior for anything
   claiming cell-conditioned learning.
3. **B1 is one known number.** HIST_D7 (−0.072 from the full corpus) = B1 (−0.001). A pair-type-level prior survives any within-pair-type
   shuffle (86–90 % kept), so such a shuffle cannot test it; controls must be matched to the claim (universal prior vs cell chemistry vs
   electronic map vs geometry vs transfer).
4. **Broad universal priors dilute** the one real effect (B2 −0.022, 0/5 seeds) and inner CV did not protect against it.
5. **Composition trap.** Cross-metal energies/geometries from complexes whose inner-sphere composition changes along the series are
   artefacts (+0.644 → −0.09). Only 60/229 C3 cells are composition-constant.
6. **Own-cell oracles absorb noise** (half of the curvature oracle; the per-cell Gd amplitude is inseparable from Gd's own deviation).
   Use leave-pair-out or group-level oracles.
7. **Reliability before correlation.** A derived descriptor (slope, per-unit coefficient) must show split-half/jackknife reliability
   before its null or its correlation is read (gen16 L1: reliability 0.000 first, 0.80 after the cycle was closed).
8. **No closure-determined rows.** gen15's measured-mode win (+0.065 over the line) did not reproduce on strict adjacent pairs because
   24.8 % of its adjacent rows were fixed by arithmetic. Metal model; never score a pair with both ends read.
9. **Small-cluster inference.** 13 source clusters, Kish 2.24, largest 65 %; percentile cluster bootstrap rejects at 0.07–0.08 for
   nominal 0.05; B1's replicated +0.034 fails the jackknife-t interval; confirming a B1-sized gain needs ~40 (optimistic) to > 80
   (heterogeneous) new independent sources; the 0.005 balanced-MAE margin ≈ a +0.034 R² gain, i.e. it binds before the R² margin.
10. **Run a synthetic discrimination study before sealing.** In round 4 the donor route (D7-sized effect: 0/8), the geometry route
    (0/6) and the An transfer route (0/6) had no power at plausible sizes; their nulls are "not detected", not "refuted". A blocked
    route says little about the hypothesis if power was not shown.
11. **Adaptive selection and seed re-use.** Round 3 scored 28 configurations against a budget of 24 and made 5 post-result registry
    changes; new split seeds re-partition the same 229 cells (stability, not replication). The round-3 final seeds and the strain
    fresh-444 pairs are spent.
12. **Do not mix regimes.** Read policy (POL_ERR vs DOPT_Q), cohort (C6 vs C3), data version (V2) and dropping a publication (BTBP:
    +0.034) each move R² by 0.02–0.07 and are not model gains.
13. **Pooled R² is outlier-dominated.** 5 % of rows = 59 % of B1's SSE; one publication = 23 %. Any gain must be checked for being a
    handful of rows (report the per-cluster and trimmed contrasts next to the pooled one), and the metric must not be changed post hoc.
14. **Capacity loses under masks.** Neural/multitask/transformer arms were worst or tied (NNC −0.128; S11 = B1; strain encoders tie).
15. **Electronic features are bases, not physics.** n_f = Z − 57 is constant for every adjacent pair; f-shell bases are interchangeable
    with smooth bases (within 0.006); a Gd-centred gain does not identify a mechanism.
16. **The design defines what is learnable.** Chemotype hold-out with ~4 effective chemotypes makes chemotype-specific structure
    (sc009's Nd dip, its weak Gd dip) unlearnable by construction; a null there is a property of the design × corpus.
17. **Provenance traps found by review:** the strain corpus file is the gen21 corpus file (strain-trained weights are contaminated);
    `safe_exp_id` prefix is the export file, not the metal; a geometry shuffle broke missingness; D4GC01146E is already in the cohort;
    the g21 cache silently dropped 148 of 183 training cells in one fold (orphan rule); Windows text writes turned files CRLF and broke
    digests.

---

## 5. Untested, or dismissed for weak reasons

| item | why it was not tested / how it was dismissed | why that reason is weak | source |
|---|---|---|---|
| Condition (solvent, extractant concentration, acid) × series shape at k = 3 | left out of R07 because the columns "give away the publication" | BP already removes the test publication from training; the leakage concern does not bar a BP-scored test | `FINAL_REPORT_RU.md` §3 |
| DOPT_HB / any B1-aware read policy | deferred because HB posterior variance is inverted vs error | that argues against DOPT_HB specifically, not against a train-fitted B1 error-table policy; Ce (34 % of SSE via its pairs) is read in 4 % of DOPT_Q sets | `GEN21_ROUND2_REPORT.md`; §3.1 |
| Five-metal budget (k = 4) | never run | the only missing rung between 0.452 (4 metals) and ~0.50 (6 metals) | memory gen21 |
| Floor-aware / censored treatment of reads (C5-BTBP) | G-V3 failed (no documented detection limit); Huber corrector null | a censored likelihood using the reads only is a model choice, not a value-based row deletion; the BTBP publication carries 23 % of SSE | `PROVENANCE.md`; §3.1 |
| Ce-specific deviation (redox / analytical) | never examined | heavy-tailed, condition-correlated (ρ −0.34 with acid), 34 % of SSE via La/Ce + Ce/Pr | §3.4 |
| Family / laboratory-sibling regime at k = 3 with full sibling curves | BP forbids it; attempt 3 pooled sibling *reads* only | a declared "known family / known lab" deployment is legitimate and the transductive bound is +0.02 to +0.07 R² | §2.8, §3.4 |
| 168 ligand-at-complex strain single points (gen16 L1) | never analysed | not refuted; testable on the 60 composition-constant cells | `EVIDENCE_INVENTORY.md` §2 |
| α-alkylated DGA steric descriptor | 2 examples; 7-column steric block hurt | untestable for lack of examples, not refuted | gen14 |
| Donor-class / electronic routes (S03, S04, S21) | routes closed when S01/S02 failed criterion 1 | the S01 null had no power at D7 size (0/8) | `AMENDMENTS.md` A1 |
| Encoder routes S07/S10, geometry controls S09/S22 | encoder gate (S06 inner 9/50) | synthetic geometry power 0/6 | `AMENDMENTS.md` A2 |
| An(III) transfer S13–S16, metal-state control S24 | inner gate 3/50; S24 count gate (83–93 % Am) | synthetic power 0/6 even with signal | `blockC_decision.md` |
| Kish-based df correction of the jackknife-t | noted, not adopted | statistical, affects every confirmation design | `POWER.md` |
| Laboratory repeatability of log SF | no replicate whole curves exist | the ceiling is unknown (bounds 0.65–0.92 in §3.3) | `DATA_QUALITY_REPORT.md` |
| DMDPhPDA 5 M Pr/Nd value (Shimada thesis), BTBP attribution, ORNL data locations | need user permission / journal access | data-quality facts, not modelling | `DATA_REQUEST.md` |

---

## 6. Open hypotheses (what is still genuinely undecided)

Each line: hypothesis — current evidence — what would decide it.

1. **H-LOCAL: B1's residual at k = 3 is dominated by hidden-metal local deviations, so the remaining levers are which metals are
   hidden and whether their deviations are predictable — not curve shape.** Evidence: lag-1 −0.52 / lag-2 +0.09; dense-read oracle
   +0.012 only; every smooth/tabular corrector ≤ B1. Decide: a registered read-policy × B1 study on identical cells, plus a k = 4 rung.
2. **H-READ: a label-free read policy fitted to B1's training-fold error (per metal and position) beats DOPT_Q by ≥ 0.01 R².** Evidence:
   POL_ERR + Gd (R23) 0.471 / 0.463 on final seeds vs R14 0.455 / 0.454 (difference not significant); Ce read in 4 % of sets.
   Decide: policy crossing with B1 fixed, BPG + BP, all subsets; the regime must be declared separately.
3. **H-FAMILY: with a declared "known family / known laboratory" regime, family- or lab-level residual offsets lift k = 3 above 0.50.**
   Transductive bound: chemotype 0.501, extractant 0.501, publication 0.524 (0.542 without BTBP). Decide: nested CV under design A or a
   BP1-style design that keeps siblings in training; this is a different deployment claim from BP.
4. **H-CHEMOTYPE-DIP: the Gd/Nd dip amplitudes are chemotype properties predictable from a structural descriptor that bridges sc009
   to other chemotypes.** Evidence: Gd ICC 0.69 (extractant) / 0.41 (chemotype); Nd dip sc009-only; donor class fails (S01 = B1).
   Not decidable on this corpus (~4 effective chemotypes, synthetic power 0/8 at D7 size); needs new independent chemotypes.
5. **H-FLOOR: a censored/floor-aware predictor (reads only, no row deletion) recovers part of the BTBP error** (23 % of SSE; B1 R² 0.24
   on it). Risk: value-derived; must be specified before scoring and must not use hidden values.
6. **H-CE: the Ce deviation is condition-driven (acid, oxidant, diluent) or a Ce(IV) artefact and can be gated by conditions.** Evidence:
   ρ −0.34 with acid molarity, heavy tail, low ICC. Confounded with publication; needs a within-publication check first.
7. **H-COND: condition × series-shape interactions add at k = 3 under BP.** Never tested; prior small (gen15 acid effect +0.009 zero-shot).
8. **H-BUDGET: five measured metals with B1 reach 0.50 on C3.** Interpolates between 0.452 (4) and 0.54 (6, C6); never measured.
9. **H-CEILING: the laboratory repeatability of adjacent log SF caps k = 3 R² well below 0.65.** Unmeasured; bounds 0.65 (raw local
   variance) to 0.92 (robust); the answer decides whether 0.50 at four metals is a modelling or a measurement problem.
10. **H-ALPHA-DGA: a steric (α-substitution) descriptor fixes the strong-band direction failures.** Two examples; needs data.
11. **H-AN: An(III) data improve Ln adjacent log SF.** Not detected; control power not shown (0/6). Needs a design where An carries
    Ln-relevant variation (e.g. matched Am/Cm/Eu series), otherwise untestable.
12. **H-STRAIN: composition-constant ligand strain energies (168 single points) carry selectivity information.** Untested; low prior after
    the powered L1 null on binding energies.
13. **H-METRIC: conclusions are robust to the outlier-dominated pooled R².** Untested; ~5 % of rows carry 59 % of B1 SSE. A pre-declared
    secondary metric (per-cluster, trimmed or per-pair-standardised) should travel with every round-5 contrast.

**Bottom line [I].** Four generations of structure modelling are worth R² ≈ 0.05 zero-shot. At four measured metals, everything
learned beyond interpolation is one known Gd number (+0.034). What is left is mostly hidden-metal local deviation with a heavy tail;
the reproducible part lives at chemotype / laboratory level, which the BP/BPG designs (correctly) remove and which ~2–4 effective
source clusters cannot support. Round-5 ideas with a real chance are about **which metals are hidden, how many are read, what regime
is declared, and how floors and outliers are treated**, or they require new independent data.

---

### Appendix: how section 3 was computed

- Rows: `REF_{BPG,BP}.parquet`, k = 3, 1,802 rows per seed; residual e = y − B1; R² = 1 − SSE/SST per seed, then seed mean.
- Lag correlations: pairs (ia, ia+1) and (ia+1, ia+2) present in the same (seed, cell); "shared unread" = `b_read` false on the first.
- Gd feature: f = (HIST_D7 − B0)/(−0.072) per row (B1 − B0 = β·f).
- Group offsets: per seed, mean of e over (group, pair type) from other cells of the group, shrunk n/(n + 10), added to B1.
- Dense oracle: bench log D of all measured metals except the scored pair's k = 3-unread end(s); `np.interp` along −Shannon CN8 radius,
  constant beyond the extremes (the PWL_R convention of `next_research_20261019/r3/interp.py`).
- Second differences: a_m = L_m − [(1 − w)L_{m−1} + wL_{m+1}] along −radius (Pm absent, so Nd's neighbours are Pr and Sm); ICC(1) by
  one-way ANOVA over groups with ≥ 2 cells; local variances by non-negative least squares on var(a_m) = s_m + w²s_lo + (1 − w)²s_hi with
  s_La = s_Ce and s_Lu = s_Yb.
- Runtime < 30 s, one process; scripts kept in the session scratchpad (`e1_analysis.py`, `e1_analysis2.py`), not in the repository.
