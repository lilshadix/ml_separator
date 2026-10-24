# E5: the round-4 design, reviewed as a design

Written 2026-10-23 for the round-5 design (`generations/gen21/round5_design_20261023`). Scope: what the round-4 brief proposed, what was run, what the gates removed, which design choices limited the chance of detecting a gain, which unrun slots deserve another test, and what round 5 should change in the evaluation.

**Labels used throughout**
- **[M]** is a measured fact. It comes either from a round-4 or round-3 file (cited) or from my own exploratory computation on the stored OOF predictions or on the V1 bench. Every computation here is *exploratory development* on the same 229 C3 cells: it is not a slot, not a confirmation, and nothing is refitted.
- **[O]** is an ORACLE: a quantity fitted on test labels, or chosen with them. It bounds headroom and is never achievable.
- **[I]** is my interpretation.

Unless stated otherwise, numbers are BPG (round 4's deciding design), seed means over the 5 dev seeds. The scripts that produced the new numbers are listed in §8. They live in the session scratchpad, not in the repository.

---

## 0. Bottom line

1. **The round-4 machinery was sound; the decision geometry was the problem.**
   - **[M]** Leakage chain, seals, routes and synthetic checks all worked: fit poisoning 50/50 for every scored configuration, interventions max change 0.0, B1 = R14 bit-identity.
   - **[I]** The campaign was nearly incapable of returning a positive at the effect sizes the corpus can support. This was not a code failure.
   - **[M]** The three limiting factors:
     - **The balanced-MAE margin of 0.005** converts to +0.034 to +0.053 R². B1's own gain over interpolation sits exactly on it: +0.0341 R² / +0.0050 balanced MAE.
     - **One fold dominates.** In every seed, the fold that holds the largest source cluster carries 58% of the scored rows, and its models are trained on only 51–69 C3 cells.
     - **The effective cluster count is about 2.2** (Kish), so nothing is confirmable on the existing corpus (POWER.md).
2. **[M]/[O] The hypothesis classes tested in blocks A–C have very little headroom on BP/BPG, even as oracles.**
   - A test-label-fitted pair-type shift of B1's residual is worth +0.0138 R² / +0.0027 balanced MAE.
   - A donor-class × pair-type shift is worth +0.0408 / +0.0004.
   - Both fail criterion 1's balanced-MAE leg *with oracle knowledge*.
   - Cross-fitted honestly (leave-own-chemotype-out), both turn negative: −0.020 and −0.041 R².
3. **[M] The information that does exist is source-local or placement-related, and BP/BPG removes it by construction.**
   - Publication × pair-type residual means, cross-fitted leave-own-cell-out, give +0.0237 R² / +0.0054 balanced MAE.
   - Read placement moves R² by about 0.2: uniform-random 4-metal reads give PWL_R 0.221 against DOPT_Q's 0.418.
   - Each extra DOPT_Q read is worth about +0.03 to +0.06 R², against the model contrasts of −0.024 to −0.0001.
4. **[M] One publication dominates the error.** The C5-BTBP cells (pub_6a0ef8528c, sc061; 8.2% of rows) hold **23.1% of B1's squared error**, 63.5% of it in La/Ce and Ce/Pr, at log D ≈ −4.2 (floor-like). The light-end pairs La/Ce and Ce/Pr hold 34.2% of B1's error on 14.5% of rows. A source-quality slot (S17/S18) was gated out, but it addresses the largest single error block in the metric.
5. **[I] Round 5 should change the evaluation before it changes the model:**
   - one margin currency, with balanced MAE as a non-inferiority guard;
   - BPG alone deciding, plus a full design table (A, B, BR, BQ, BP, BPG) as diagnostics;
   - a cluster-balanced co-primary and per-fold-type reporting;
   - a pair-difference training objective;
   - a k ladder and a policy crossing as separate registered estimands;
   - a leave-pair-type-out estimand for series priors;
   - a within-source ("known lab / known extractant") estimand;
   - synthetic power studies with ≥ 50 replicates whose nuisance structure is calibrated to the real residual structure.

---

## 1. What the round-4 brief proposed

Sources: `round4_design_20261022/CLAUDE_CODE_PROMPT.md` (§§3–9) and `RESEARCH_IDEAS.md`.

**Task and regime.**
- Strict-adjacent log SF, V1, C3, k = 3 (four read metals), the DOPT_Q reference policy and the fixed round-3 query set: 229 cells and 1,802 queries per seed.
- BPG and BP designs.
- The primary metric is per-seed pooled R², then the seed mean.

**References.**
- B0 = PWL_R.
- B1 = PWL_R + one fold-fitted Gd amplitude.
- B2 = a regularised universal series / read-set prior.
- The −0.072 historical constant is a disclosed diagnostic only.

**Model contract (brief §6).** The prediction is `base(m) + correction(m) − I_S[correction(S)](m)`:
- it is trained **directly on hidden strict-adjacent differences** under masks that reproduce the task;
- weights are normalised so that more masks or pairs do not multiply a cell's weight.

**Budget.** At most 24 outer-scored configurations in 6 slot groups (table §2), with stage gates. A synthetic null / known-signal study shows each control's discriminating power before it is used.

**Planning criteria (brief §9).**
- **Criterion 1:** ΔR² ≥ +0.01 **and** balanced-MAE reduction ≥ 0.005 vs B1.
- **Cell-conditioned claim:** also beat B2 and the falsification controls.
- **Confirmation:** percentile, BCa and jackknife intervals excluding 0 for both endpoints (intersection–union, α = 0.05).
- **Power:** at least 80% to reject a non-positive gain at the margins.
- R² ≥ 0.50 is an aspiration.
- The brief itself calls the margins "explicit proposal defaults, not measured noise/process limits" and allows other margins "if justified and sealed before any new outer result".

**Hypothesis priorities (RESEARCH_IDEAS):**
1. source quality and matched Ln coverage;
2. a support-consistent residual curve;
3. electronic × donor interactions;
4. anchored scalar 3D shape;
5. selective An(III) transfer;
6. read allocation and noise-aware conditioning.

## 2. What was run, what was gated out

**[M]** Sources: `EXPERIMENT_REGISTRY.csv`, REPORT.md, AMENDMENTS A1–A5, `results/<ID>/contrasts_*.csv`. ΔR² and Δ balanced MAE are against B1; positive means the slot is better.

| slot | brief role | protocol content | status | ΔR² BPG / BP | Δ bal-MAE BPG / BP |
|---|---|---|---|---|---|
| S01 | hierarchical residual | B1 + donor-class Gd / Nd dips (7 columns) | RUN | −0.0001 / −0.0005 | −0.0006 / −0.0008 |
| S02 | electronic × chemistry | gen13 gd_break, tetrad e1 / e3 × donor class (15 columns) | RUN | −0.0240 / −0.0277 | +0.0003 / +0.0001 |
| S03 | matched-capacity Z / radius control | P2–P4 × class | NOT RUN (route: S02 failed criterion 1) | | |
| S04 | electronic-map permutation | 5 draws | NOT RUN (route) | | |
| S05 | CatBoost correction | level residual at unread metals; trees {0, 100, 300, 1000} | RUN | −0.0062 / −0.0033 | −0.0012 / −0.0009 |
| S06 | compact geometry | S05 + xTB distance / CN / charge with composition | RUN | −0.0056 / −0.0110 | −0.0012 / −0.0012 |
| S07 | distance encoder | | NOT RUN (S06 criterion 1; G-geo inner 9/50) | | |
| S08 | nested blend | | NOT RUN (no constituents) | | |
| S09 | geometry block shuffle | | NOT RUN (route) | | |
| S10 | random-encoder control | | NOT RUN (gate) | | |
| S11 | Ln-only multitask scaffold | shared encoder; residual + raw-log D heads | RUN | −0.0005 / −0.0017 | −0.0000 / −0.0020 |
| S12 | extra independent Ln | | NOT RUN (0 independent series after fold exclusions) | | |
| S13 | An(III) shared / private | 1,462 rows, 38 publications | NOT RUN (G-transfer inner 3/50, 30 needed) | | |
| S14–S16 | transfer controls | | NOT RUN (route) | | |
| S17–S18 | V3 / censoring | | NOT RUN (G-V3: no source-documented change) | | |
| S19 | combination | | NOT RUN (nothing to combine) | | |
| S20 | new-policy crossing | | NOT RUN (no candidate; POL_ERR measured in round 3) | | |
| S21 | donor shuffle (redefined from the brief's "donor-interaction ablation", A1) | | NOT RUN (route) | | |
| S22 | radius-only geometry | | NOT RUN (route) | | |
| S23 | training-fitted rescaling | | NOT RUN (no apparent gain) | | |
| S24 | metal-state shuffle | | NOT RUN (label-free count gate: 0.31–7.5% of An rows change, 10% needed) | | |

**Accounting.** 5 of 24 slots were consumed. The cap never bound; the gates and routes did. The finalist under §6.6 is the reference B1 (BPG 0.4521 / BP 0.4519).

**Design deviations from the brief that the protocol declared (§11, §12):**
- the learned corrections were fitted on **pointwise level residuals**, and only the tree count / network was selected on pair loss;
- row weights follow B1's convention, so a cell's total weight grows with its hidden queries;
- **[M]** an undeclared-as-such structural difference: block B / C corrections are added at unread metals only (`r4/slots_b.py` docstring). The brief's projection term `− I_S[correction(S)]` is absent.
  - My diagnostic (§3.4) finds this was **not** the cause of the losses.

---

## 3. Design choices that limited the chance of a detectable gain

### 3.1 The two margins are in different currencies, and the balanced-MAE margin equals the whole reference effect

**[M] What +0.01 R² and 0.005 balanced MAE each correspond to.**
- **Relative error.** B1's SSE / SST is 0.548 (my computation: SST / SSE = 1.825). So +0.01 R² means a 1.8% reduction of pair SSE, while 0.005 balanced MAE means a 4.9% reduction of B1's balanced MAE (0.1011).
- **Exchange rate under uniform residual shrinkage** (shrinking every B1 error by a fraction f): f = 0.05 gives +0.0534 R² and +0.0051 balanced MAE. So **0.005 balanced MAE ≈ +0.053 R²** under uniform gains.
- **Under B1's own gain structure:** B1 − B0 = +0.0341 R² / +0.0050 balanced MAE (`results/REF/contrasts_BPG.csv`). POWER.md's `margins_implied.csv` gives 0.0055 balanced MAE at δ = 0.034 (empirical_B1), 0.0015 at δ = 0.01, and only 0.0004–0.0009 at δ = 0.01 for uniform or concentrated gains.
- **The strongest replicated effect in the programme** (the Gd amplitude: dev +0.034, final round-3 check +0.031 / +0.032) sits exactly on the 0.005 margin.
- **Every slot that met the R² leg in a synthetic world met criterion 1 less often** (AMENDMENTS A1 run 2):
  - GDCLASS: S01 ΔR² ≥ 0.01 in 6/8 replicates, criterion 1 in 1/8;
  - ELEC: S02 5/8 vs 4/8;
  - UNIVERSAL: S02 5/8 vs 3/8;
  - block-B GEO: S06 mean +0.0124, criterion 1 in 1/6.

**[O] Ceilings of whole hypothesis classes vs the margins** (B1 residual, BPG, seed-pooled; my computation, `part1_oof.py`, `part5`):

| hypothesis class (how the shift is fitted) | ΔR² | Δ bal-MAE | passes 0.01 / 0.005? |
|---|---|---|---|
| [O] pair-type mean shift, fitted on test labels | +0.0138 | +0.0027 | R² only |
| [O] donor class (DGA / SOFT_N / OTHER) × pair type, test labels | +0.0408 | +0.0004 | R² only |
| [O] cell-mean log SF offset (perfect per-cell slope correction) | +0.0175 | +0.0015 | R² only |
| [O] optimal per-seed linear recalibration of B1 | +0.0005 | −0.0000 | no |
| cross-fitted: pair-type mean, leave own chemotype out | −0.0198 | +0.0013 | no |
| cross-fitted: pair-type mean, leave own source cluster out | −0.0426 | −0.0031 | no |
| cross-fitted: donor × pair type, leave own chemotype out | −0.0406 | −0.0060 | no |
| cross-fitted: chemotype × pair type, leave own cell out | +0.0157 | −0.0055 | R² only (MAE worse) |
| cross-fitted: publication × pair type, leave own cell out | **+0.0237** | **+0.0054** | both, but unavailable under BP / BPG |

**[I] What follows.**
- Block A's hypothesis class (donor-conditioned series shapes) could not have passed criterion 1 even with perfect test knowledge of the donor × pair-type means, because the balanced-MAE leg is unreachable.
- Its honest cross-fitted version is strongly negative, so S01 = B1 and S02 < B1 are the expected outcomes, not unlucky draws.
- The margins were kept after POWER.md showed that the balanced-MAE margin binds (A1, "Margins unchanged"). That was formally correct (no post-result relaxation), but it left the remaining blocks with a near-zero chance of a positive.

### 3.2 One fold decides the pooled R²; the learned models were trained on a quarter of the corpus there

**[M] Fold structure** (`part1b_oof.py`, `part3_misc.py`; `results/REF/fitted_*.json`).
- **The big fold.** In every seed, the fold holding the largest `cluster_cpg` component (65% of rows) has, in BPG:
  - 147 test cells on average (min 7, median 20, max 158 test cells per fold over all folds);
  - **58.2% of the scored rows**;
  - **51–69 training C3 cells** (median 60). The other folds have 134–208.
  - BP is the same: 54–72 training cells.

**[M] Where the losses sit.** Slot losses vs B1 as ΔSSE, big fold vs the other 4 folds:

| slot | ΔSSE big fold (BPG) | ΔSSE other folds (BPG) | ΔR² within the other folds |
|---|---|---|---|
| S02 | +10.97 | −0.31 | +0.0019 |
| S05 | +1.98 | +0.76 | −0.0041 |
| S06 | +2.28 | +0.22 | −0.0009 |
| S11 | 0.00 (chose B1) | +0.24 | −0.0011 |

- **[I]** The deciding number is mostly "learn from about 60 cells, predict 150". S02's 15 class-specific columns break there and are roughly neutral elsewhere.
- This is a structural property of a 13-cluster corpus with one 65% cluster. It is not a bug, but the pooled per-seed R² hides it.
- An estimand that weights clusters (§5.1) or reports per fold type would have shown that the damage comes from data-starved fits, not from the hypothesis.

### 3.3 BP ∧ BPG is a conjunction over the same test sets

- **[M]** BP and BPG have **identical fold assignments** (100% of rows) and differ only in training masks (copy groups). Their DOPT_Q read sets coincide in 87.2% of rows; the rest differ because the training covariance differs.
- **[M]** B1's row errors correlate 0.988 between designs. The slot-minus-B1 differences correlate 0.86 (S02), 0.49 (S05), 0.43 (S06) and ≈ 0 (S11).
- **[I]** Requiring the margin "on BPG and on BP" adds training-noise variance without independent evidence; PROTOCOL §1 already says they are not replications. For a learned slot it lowers power. Example: S06 −0.0056 on BPG, −0.0110 on BP.
- **[I]** A design table of genuinely different splits (A exact extractant, B chemotype, BR random cell, BQ random publication, BP, BPG) is the project's own rule (memory: evaluation protocol, gen13 Addendum 3). It would have been more informative than the BP conjunction.

### 3.4 Pointwise training vs the pair target (a declared deviation from brief §6)

**[M]** Pointwise level errors reconstructed exactly from the pair predictions (`part1_oof.py`; supports are exact, so the chain is identified):
- adjacent unread metals' pointwise errors correlate **ρ = 0.69**;
- about **36%** of pointwise error energy is a per-cell mean offset that cancels in every both-hidden pair.

**[M] Ranking reversal on S05:**
- pointwise MSE: S05 0.03727 < B1 0.03758, so S05 is better on its training objective;
- pair MSE: S05 0.02730 > B1 0.02699, so S05 is worse on the target.
- S11: pointwise 0.03763 vs B1 0.03758, pair 0.02702 vs 0.02699.

**[M] The missing projection is not the culprit.**
- 36–51% of the S05 / S06 / S11 correction energy is a per-cell offset at unread metals.
- Removing that offset label-free (`part5_offset.py`, post hoc diagnostic) makes S05 *worse* (0.4459 → 0.4395) and leaves S06 / S11 essentially unchanged.
- **[I]** The losses come from the correction's shape at unread metals, not from level offsets. A pair-difference loss is still the right objective, because the pointwise objective rewarded S05 for changes that hurt pairs. It will not by itself rescue a correction whose information is absent (§3.1).

### 3.5 Inner-CV selection was not too conservative; it was optimistic when it switched on

**[M] Inner gaps** (positive-tree or positive-network inner pair loss vs the zero correction, relative, median over 25 BPG folds):
- S05 +1.1%, S06 +3.0%, S11 +7.0% (worse);
- B1's own inner gain over λ = ∞ is −5.8%.

**[M] Inner vs outer when trees were chosen.**
- When S05's inner CV did choose trees (9 of 25 BPG folds), it promised a mean −2.2% SSE. The same folds' outer SSE was **+3.1% worse** on average, worse in 7 of 9.
- S06: 5 of 5 chosen folds were worse (+3.1%).
- BP: S05 +4.4%, S06 +11.0% when chosen.

**[I]** "Inner CV fell back to B1" did not hide gains; if anything, inner selection let through winner's-curse choices. The median inner gap (+1.1% SSE ≈ −0.006 R² at SSE / SST 0.548) predicted the outer result (−0.0062) almost exactly. Round 5 does not need a looser selection rule; it needs hypotheses with headroom.

### 3.6 k = 3 and DOPT_Q only: the levers that matter were held fixed

**[M] DOPT_Q k ladder** on round 3's fixed query sets (`predictions/R3-B0_BPG.parquet`, plus the rule GD072 = PWL_R with the fixed −0.072 Gd potential; `part4_kladder.py`; BPG):

| reads (k + 1) | C3 PWL_R | C3 + Gd rule | C6 PWL_R | C6 + Gd rule |
|---|---|---|---|---|
| 1 | n/a (HBK −0.117; G14@ZERO zero-shot +0.053) | | | |
| 2 | 0.191 | 0.231 | 0.167 | 0.206 |
| 3 | 0.360 | 0.401 | 0.345 | 0.385 |
| 4 | 0.418 | 0.451 | 0.396 | 0.431 |
| 5 | | | 0.449 | 0.483 |
| 6 | | | 0.491 | **0.524** |
| 7 | | | 0.522 | 0.552 |

- **The Gd rule** is worth +0.030 to +0.040 at every budget.
- **One more DOPT_Q read** is worth +0.058 (3 → 4 reads, C3), then +0.054, +0.042 and +0.031 (C6).
- **0.50** is crossed at 6 reads (C6) with the rule. This matches round 3 (0.540 / 0.542 with POL_ERR).

**[M] Placement** (all 111,052 legal 4-metal read sets of the 229 cells; `part2_bench.py`):
- uniform-random placement gives PWL_R 0.221 and the Gd rule 0.260;
- DOPT_Q gives 0.418 / 0.451;
- [O] the per-cell best read set gives 0.780 and the worst −0.142.
- These are not on identical queries; they show the scale only.

**[M] What DOPT_Q actually reads.**
- Read frequency per cell: La 71%, Gd 61%, Ce 4.4%, Tb 0.8%, Lu 13.5%.
- B1's gain is +0.035 where Gd is read and +0.042 where it is not.
- **Adding a 5th metal** (B1 rule, seed-0 DOPT_Q sets, legal queries) reduces SSE per cell most for Gd (0.035, n = 33 cells where Gd is unread), then Er 0.022, Ho 0.020 and Sm 0.019, and least for Lu (0.003) and Tm (0.005).
- **[I]** Low-variance heavy pairs (Tm/Yb and Yb/Lu, sd 0.12) have R² ≈ 0 but contribute little SSE, so "DOPT_Q rarely reads Lu" is not a material loss.

**[I]** Model contrasts in round 4 were −0.024 to −0.0001, while one read is worth about 0.05 and placement about 0.2. Fixing k = 3 and DOPT_Q put all of round 4's budget on the weakest lever. The brief allowed policy and budget only as separate regimes (correctly), but no slot measured them, and S20 was gated on having a candidate predictor.

### 3.7 The boundary rule predicts no separation for 12% of queries

- **[M]** 11.9% of scored rows have a hidden end outside the read hull. B0 and B1 predict exactly 0 log SF there (constant continuation), with R² −0.015 on those rows, against 0.472 for B1 inside the hull.
- **[M]** A half-slope continuation *rule*, chosen post hoc (`part4_kladder.py`), gives:
  - B0: 0.4180 → 0.4239;
  - B1-form with −0.072: 0.4511 → 0.4568, balanced MAE 0.1011 → 0.1006;
  - +0.0047 R² over B1;
  - full linear continuation is worse (−0.005).
- **[I]** The reference's edge behaviour was never varied. A training-fitted edge-slope shrinkage is the natural form of the brief's S23 ("training-fitted rescaling / shrinkage competitor"). It is small, but the same size as most of what round 4 tested.

### 3.8 The metric is dominated by a few, possibly floor-censored, cells

**[M]** Sources: `part1b_oof.py`, `part3_misc.py`, PROVENANCE.md.
- **Concentration.** The top 10 cells (4.4%) hold 36.6% of B1's squared error, and the top 23 (10%) hold 53.5%.
- **C5-BTBP** (pub_6a0ef8528c = all of sc061's 15 C3 cells, 148 rows / seed):
  - holds 23.1% of B1's SSE, with B1 R² 0.238 inside;
  - 63.5% of that SSE is in La/Ce and Ce/Pr;
  - the light lanthanides sit at log D ≈ −4.2 to −4.5 in every cell (PROVENANCE: "may be a detection floor");
  - B1 R² excluding it is 0.486. This is a sensitivity, not a gain: rows are never dropped for test error.
- **The light end.** La/Ce and Ce/Pr are 14.5% of rows and **34.2%** of B1's SSE, and Ce is hidden in 97.3% of those rows.
- **Ce's local roughness.** Ce's deviation from its La–Pr chord has sd 0.244, the largest of any metal; heavy metals have 0.06–0.11. 23 of 118 cells have |deviation| > 0.2. They concentrate in sc009 (10), in sc061 (7) and in publications pub_6a0ef8528c (6) and pub_126bbe7976 (4). Gd's mean deviation (−0.115) is the known dip.

**[I]** The G-V3 gate (S17 / S18) was correct to refuse a value-based edit. But the evaluation had no declared floor / light-end subgroup and no censoring-aware secondary estimand, so the largest single error block was invisible to every slot's design. In real chemistry, Ce can be anomalous (Ce(IV)), so not every Ce outlier is an error. That makes this a provenance question, not a modelling one.

### 3.9 Universal priors do not transfer; the information is local to a source

- **[M]** The test-label pair-type oracle (+0.014) turns into −0.020 when cross-fitted leaving the chemotype out, and −0.043 when leaving the cluster out (§3.1). B2 (−0.022 vs B1) is consistent with this.
- **[M]** Publication × pair-type residual means, cross-fitted leave-own-cell-out, give +0.024 R² / +0.0054 balanced MAE.
- **[I]** What a learner could use lives inside a source: the same laboratory, extractant or conditions. BP and BPG remove exactly that by design. They are the right designs for "new chemistry from a new lab", but the corpus then has about 2–4 effective units to learn from. A deployment question that may matter to users, "a new condition or extractant from a laboratory already in the corpus", was not posed. It is the regime where corrections have measurable headroom.

### 3.10 Small synthetic power studies, not calibrated to the real residual

**[M] Replicates and folds.**
- Block A run 2: 8 replicates per world, two BPG folds.
- Block B run 2: 6 replicates.
- Block C run 4: 6 paired replicates.
- The exact 95% upper bounds on a 0/n rate are 0.37–0.46, so these studies cannot certify power or size.

**[M] The rules evaluated** were upper bounds of the sealed ones: vs B1 only, one fold, no harm rule, no B2.

**[M] Miscalibration.** In the synthetic worlds, the learned slots gained over B1 even in NULL: S05 +0.024 (block B NULL), S11 +0.019 (block C, every world). On the real data they lost (−0.006, −0.0005).

**[I]** The synthetic generator contained support-summary structure that the real residual does not. Power estimates from it were optimistic for "S05 / S11 beat B1" and silent about the real failure mode. A round-5 generator should be calibrated on real B1-residual statistics:
- adjacent pointwise ρ ≈ 0.69;
- cell-offset share ≈ 36%;
- cluster heterogeneity: per-cluster ΔSSE shares, the light-end variance and the Ce roughness above.

### 3.11 Confirmation geometry (POWER.md)

**[M] What POWER.md found.**
- About 13 source clusters, Kish 2.24, one holding 65% of rows.
- **B1 − B0's dev gain fails the jackknife-t interval** ([−0.009, +0.077] BPG, cluster_cpg).
- **The three-interval rule undercovers:** 5–8% vs a nominal 2.5% per side under the boundary null.
- **Run 4 (heterogeneous bound):** even G = 80 new sources gives about 1 in 3 for a B1-sized gain. The full gate at δ = 0.034 passes 0.13 / 0.23 / 0.25 at G = 20 / 40 / 80.

**[M] What averaging over read sets buys** (`part2_bench.py`, cluster bootstrap over the 13 clusters, 400 draws, contrast = Gd rule − PWL_R):
- SD 0.0144 with one random read set per cell;
- SD 0.0122 averaged over all legal read sets (about 15% narrower);
- the read-set-only SD is 0.0030.

**[I]** Multi-policy evaluation adds coverage, not much power. The cluster count is binding, as POWER.md says.

---

## 4. The never-run slots: which still deserve a test, and how

**Verdicts:**
- **RETIRE:** do not spend round-5 slots on this corpus.
- **RE-TEST:** worth a slot with the redesign named.
- **NEEDS DATA:** blocked on inputs software cannot make.

| slot(s) | verdict | evidence | what would make a test informative |
|---|---|---|---|
| S03, S04, S21 (block-A controls) | RETIRE | They control S02 / S01, which failed. [O] the donor × pair-type ceiling fails the balanced-MAE leg (+0.0004), and its cross-fit is −0.041. | Only if a future donor-conditioned slot passes. |
| S07, S10 (distance encoder, random encoder) | RETIRE on this corpus | S06 inner gate 9/50 (30 needed); synthetic GEO claim 0/6; 195 of 229 C3 cells change composition across the series; gen15's composition trap. | Constant-composition geometry for most of a series, more chemotypes, and a ≥ 50-replicate power study first. |
| S08 (blend) | RETIRE | No constituents. Round-3 stacking (R13 / R22) gave +0.002 / +0.003, with member errors correlated at 0.97. | none |
| S09, S22 (geometry controls) | RETIRE (conditional) | Controls of a failed route. | Keep the definitions; run only under a passing geometry slot. |
| S12 (extra independent Ln) | NEEDS DATA, highest scientific value | 0 independent series after fold exclusions (PROVENANCE). Under 10 open ≥ 10-Ln sources found. | A training-only learning curve vs independent source count, scored on the same C3 population. New sources are curated after the freeze. |
| S13–S16 (An(III) transfer) | RETIRE for round 5 | G-transfer inner 3/50; synthetic AN_SIGNAL claim 0/6 (S13 − S11 ≤ +0.002); 83–93% Am; gen11, gen19 and strain An results inconclusive or null. | Matched-condition An(III) series with more than Am, and a calibrated power study showing ≥ 50% firing at a plausible signal. |
| S17, S18 (V3 / censoring) | RE-TEST once sources arrive (NEEDS DATA) | The C5-BTBP cells hold 23.1% of B1's SSE, mostly La/Ce and Ce/Pr near log D −4.2. DMDPhPDA copies differ by one decade. G-V3 failed only for lack of source evidence. | Register a V3 benchmark with interval (censored) scoring defined before any value is edited. Extend source recovery to the Ce-anomaly cells (§3.8). Every baseline is refitted on V3. |
| S19 (combination) | n/a | Nothing passed. | none |
| S20 (new-policy crossing) | RE-TEST as its own estimand | Placement moves R² by about 0.2 (§3.6). Round 3: POL_ERR − DOPT_Q +0.019 [−0.007, +0.053], R23 − R14 +0.023 [−0.003, +0.057]. DOPT_Q leaves Gd unread in 39.5% of cells, and Gd is the most valuable added read. | A crossing {DOPT_Q, POL_ERR, a Gd-aware D-optimal policy} × {B0, B1} on identical queries (the union of supports excluded, gen15 rule), at budgets 2–7 reads. Label-free policies only. |
| S23 (training-fitted rescaling / shrinkage) | RE-TEST, reframed | [O] global linear recalibration of B1 is worth only +0.0005, so drop it. A post hoc half-slope edge-continuation rule gives +0.0047 vs B1 (§3.7). | A training-fitted edge-slope shrinkage λ_edge ∈ [0, 1], selected in inner CV. One parameter, cheap, and it competes with any edge-related learned gain. |
| S24 (metal-state shuffle) | RETIRE | Only 0.31–7.5% of An rows change (label-free). | none |

**Slots that were run but deserve a re-design rather than a re-run:**
- **S05 / S11 with a pair-difference loss and per-cell weight normalisation** (brief §6 as written). Only after §5's evaluation changes. The headroom on BP / BPG is small (§3.1), so expect at most a few thousandths.
- **S01-type hierarchical shrinkage, but toward source-local deviations, in a within-source regime** (§5.1 E-F). The cross-fit evidence (+0.024) says that is where the signal is.

---

## 5. What round 5 should change in the evaluation itself

Everything below must be sealed before any round-5 outer result is seen. The margins and estimands are justified here from round-4 numbers that are already public, as brief §9 allows.

### 5.1 Estimands (one primary; the rest registered as separate regimes, never substituted)

**E-A (continuity).** P0 exactly as in round 4: V1, C3, k = 3, DOPT_Q, BPG; pooled per-seed R²; balanced MAE. It keeps comparability with B1 = 0.4521.

**E-B (co-primary, cluster-balanced skill).**
- **Definition:** for each `cluster_cpg` cluster, the relative pair-SSE reduction vs B1, averaged over clusters with equal weight (or over a pre-declared √n weight).
- **Why:**
  - pooled R² weights the 65% cluster 0.65;
  - balanced MAE weights 38 chemotypes equally, which is a different population again (§3.1: S02 has ΔR² −0.024 but balanced MAE +0.0003);
  - one skill score with declared weights avoids two endpoints with opposite sampling frames.
- Report it per fold type: the big-cluster fold vs the rest (§3.2).

**E-C (read-set robustness).** The same predictor scored over a fixed sample of legal read sets per cell: all legal sets, or K = 50 seeded random sets, plus DOPT_Q and POL_ERR. Training already uses DOPT_Q + UNIFORM + 4 RANDOM masks (PROTOCOL §4), so evaluation should cover the same mixture. Expect about 15% narrower cluster intervals, not more (§3.11).

**E-D (k ladder).** 2–4 reads on C3 and 2–7 on C6, on fixed queries (the largest budget's supports excluded at every budget). Every candidate is reported at every budget. A learned prior has its best chance at low k: PWL_R is 0.19 at 2 reads, against 0.42 at 4.

**E-E (leave-pair-type-out) for series-prior claims.**
- **[M]** The Gd amplitude is identifiable without Eu/Gd and Gd/Tb pairs: in-sample uniform-policy β = −0.093 without them, vs −0.118 with all pairs. But only 4.8% of the Gd feature energy lies outside those pairs.
- So E-E is estimable but weak. It is the cleanest test that a series prior transfers to unseen pair types, as opposed to fitting pair-type means (round 3's CTRL14W kept 87% of the gain).

**E-F (within-source regime, new).** A new cell from a publication / extractant already in training: leave-one-cell-out within publication, and the gen13 A / BR / BQ designs.
- It answers the user-facing question "I have run this system before; predict a new condition".
- **[M]** Cross-fitted headroom: +0.024 R² / +0.0054 balanced MAE.
- It must never be mixed with BP / BPG claims.

**E-G (measured mode / next read).** The value of the next measured metal under each policy, scored as the realised SSE reduction on fixed queries. Round 4 (§3.6): Gd > Er ≈ Ho ≈ Sm ≫ Lu, Tm. It is a decision estimand for the experimentalist, separate from prediction accuracy.

**Secondary diagnostics (no gate).**
- Selective R² on label-free reliability flags: inside vs outside the read hull (B1 0.472 vs −0.015), and one-end-read vs both-hidden.
- A declared floor / light-end subgroup: C5-BTBP, La/Ce, Ce/Pr.

### 5.2 Margins and decision rule

- **Use one currency.** Set the improvement margin as a relative pair-SSE reduction vs B1, for example ≥ 2% (≈ +0.011 R² at SSE / SST 0.548).
- **Make balanced MAE a non-inferiority guard** (for example Δ ≥ −0.001) instead of a co-equal superiority margin.
- **If a balanced-MAE superiority margin is kept, scale it to the R² margin** under the empirical gain structure: about 0.0015 at +0.01 R² (POWER `margins_implied.csv`), not 0.005.
- **BPG alone decides.** BP is reported as a sensitivity (same test folds, error correlation 0.988; §3.3). Add the six-design table as diagnostics.
- **Keep the harm rule, but make it two-sided in information:** require ≥ 20 rows per seed and report the light-end / floor subgroup separately (§3.8).
- **Confirmation (unchanged in spirit):** one frozen finalist, new sources only. Before sealing, apply POWER.md's suggestion: a Kish-corrected df for the jackknife-t, and the BCa / percentile undercoverage.

### 5.3 Power and synthetic checks

- **At least 50 replicates per world**, so a 0/50 rate has an exact upper bound of 0.07, run on the same 25-fold pipeline (or a validated surrogate) with the sealed rule, not an upper-bound rule.
- **Calibrate the synthetic nuisance on the real B1 residual:**
  - adjacent pointwise ρ 0.69;
  - cell-offset share 36%;
  - the per-cluster heterogeneity;
  - the light-end variance;
  - a NULL world in which S05 / S11 do **not** gain (round 4's NULL worlds gave them +0.02).
- **Report prospective power for each registered contrast before sealing.** If a slot's power at its plausible effect is below about 50%, do not run it as a decision slot; run it as a labelled diagnostic, or not at all.
- **Report the dev MDE alongside each contrast** (the `mde_80` column exists in `contrasts_*.csv`): B1 − B0 0.044; S05 − B1 0.011; S11 − B1 0.003 (cluster_cpg, BPG). The MDE scales with how far a candidate departs from B1.

### 5.4 Training contract for any learned correction

- **Train on pair differences** (or on first differences of the level curve along adjacency), with per-cell weight normalisation. This is brief §6 as written; round 4 declared deviations from both.
- **Use the projection form** `f(m) − I_S[f(S)](m)` for interpretability. §3.4 shows it was not the cause of round 4's losses, so it is a cleanliness requirement, not an expected gain.
- **Keep inner selection with ties toward the reference.** §3.5 shows that inner CV predicted outer results well and, when anything, was optimistic. Add a one-standard-error rule across inner blocks rather than loosening.

### 5.5 Fold design

- Keep BPG's isolation.
- Report the big-cluster fold and the remaining folds separately for every contrast.
- Consider leave-one-cluster-out (13 folds) for E-B, so that every cluster is predicted from all the others.
- **[I]** A model that needs more than about 60 training cells cannot be judged on the deciding fold of this corpus; say so in its registration.

---

## 6. Open hypotheses this review leaves for round 5 (exploratory; none is a result)

1. **Training-fitted edge-slope shrinkage** (an S23 reframing).
   - **[M]** A post hoc half-slope rule: +0.0047 R² over B1, +0.0004 balanced MAE.
   - **[I]** Expect at most about +0.005 once fitted honestly.
2. **A Gd-aware read policy (S20 reframing).**
   - **[M]** Gd is unread in 39.5% of DOPT_Q cells, and it is the most valuable added read (0.035 SSE per cell).
   - **[I]** A policy that prefers Gd at a fixed budget may beat DOPT_Q. That needs a fixed-budget crossing, not this 5th-read diagnostic.
3. **Within-source learning (E-F).**
   - **[M]** The cross-fitted publication × pair-type headroom is +0.024 / +0.0054.
   - **[I]** It is real for "known laboratory" use, and irrelevant to "new chemistry" claims.
4. **The floor / Ce provenance.**
   - **[M]** C5-BTBP holds 23.1% of B1's SSE, the light-end pairs 34.2%, and 23 of 118 cells have |Ce deviation| > 0.2.
   - **[I]** Source recovery (DATA_REQUEST §1) may move the metric more than any model. It must enter only as a separate V3 with interval scoring.
5. **Low-k regime.**
   - **[M]** At 2 reads, PWL_R gives 0.191 and HBK@ML 0.101; the zero-shot G14 prior gives 0.053.
   - **[I]** If chemistry priors help anywhere, it is at k ≤ 1. That is untested in gen21.
6. **Leave-pair-type-out for the Gd prior (E-E).**
   - **[M]** The amplitude is identifiable from neighbour pairs: β −0.093 in-sample, from 4.8% of the feature energy.
   - **[I]** This would turn round 3's "pair-type mean" objection into a direct test.

## 7. Surprises (things I did not expect before computing)

- BP and BPG share 100% of their fold assignments. The "robustness design" re-scores the same test sets.
- The learned corrections were *better* than B1 on their own pointwise objective (S05) while worse on pairs.
- Removing the per-cell offset of the learned corrections does not help. The harm is in their shape, not their level.
- When inner CV picked a positive tree count, the outer result was worse in 12 of 14 BPG fold-fits across S05 and S06. Selection was optimistic, not conservative.
- Even with test labels, a donor-class × pair-type shift gains only +0.0004 balanced MAE.
- One publication (C5-BTBP, 8% of rows) carries 23% of the reference's error, nearly all in two light-end pairs at a floor-like log D.

## 8. Reproducibility of the new numbers

All computations are read-only and exploratory: one Python process, each under 1 minute. They run with `D:\ml_separator_gh\.venv\Scripts\python.exe -B` from `D:\ml_separator_gh`. They import `r4.paths`, `r4.metrics.add_clusters`, `r3.interp` and `gen20k.data.load_bench`, and read only `predictions/*.parquet`, `results/*/fitted_*.json` and the V1 bench. The scripts and their JSON outputs are in the session scratchpad, not the repository:

| script | what it computes |
|---|---|
| `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e5\part1_oof.py` | levels; SSE decomposition; oracles and cross-fits; exchange rate; pointwise reconstruction; inner-selection gaps |
| `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e5\part1b_oof.py` | big-fold vs other folds; per-pair; cell concentration; BP / BPG agreement |
| `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e5\part2_bench.py` | B0 / HIST_D7 reproduced (max difference 0.0 / 8.9e−16 on 1,802 rows); continuation rules; all 111,052 legal read sets; placement oracle; Gd identifiability; read-set vs cluster precision; random nested ladder; 5th-read value |
| `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e5\part3_misc.py` | training-cell counts per fold type; inner vs outer when trees were chosen; neighbour-chord anomalies |
| `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e5\part4_kladder.py` | DOPT_Q k ladder with the Gd rule; half-slope balanced MAE |
| `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e5\part5_offset.py` | label-free offset removal for S05 / S06 / S11 |

The donor × pair-type oracle and the C5-BTBP sensitivity were inline one-off computations with the same inputs; the numbers are quoted in §3.1 and §3.8.

**Caveats.**
- Every number here comes from the same 229 cells that rounds 1–4 have already used. None is confirmatory.
- The half-slope rule, the offset removal and the placement oracle were chosen after seeing results. They are leads, not candidates.
- In-sample oracles over many small groups (for example chemotype × pair type at +0.177 R²) mostly fit noise. Only the leave-own-cell-out cross-fits in §3.1 are interpretable as headroom.
