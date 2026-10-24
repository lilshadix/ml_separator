# Gen21 round 5: evidence, hypotheses, and what can still be learned

Prepared 2026-10-24. This is an analysis and an experiment design, not a trained model or a positive result. It merges
the round-5 evidence base, two idea rounds (39 + 15 ideas, two skeptics each), five probes with their skeptic re-runs,
and round 4's sealed record into one programme. Nothing here is sealed or confirmatory. Revised the same day after an
adversarial review; `REVIEW_LOG.md` lists every finding and change.

> **Status and labels.**
> - Every in-corpus number below was computed on the 229 C3 cells and the 5 dev seeds that rounds 1–4, the round-5
>   probes and both skeptic rounds have already seen. It is **exploratory ("seen")**: a lead, never a test.
> - Where a probe check corrects a probe, the corrected value is used (PROBE_CHECKS takes precedence).
> - **E-A** is round 4's estimand: BPG pooled strict-adjacent R², seed mean (V1, C3, k = 3 = four read metals, DOPT_Q);
>   the reference B1 scores **0.452**. Separate estimands and data versions (§7.1): E-D, E-CT, E-LAB, E-W, E-DEC (with
>   the E-REC own-prior row), E-KX, E-RANK, E-S, E-CENS, the window axis, V3, and the new-laboratory claims C-ECT, C-GD
>   and Gd-dip transport. Their values are compared only with B1 **on their own identical rows**.
> - E-CT, E-LAB, E-W, E-DEC, E-KX, E-RANK and V3 values are never placed next to B1 = 0.452. E-D's 4-read rung is E-A
>   itself; its 5- and 6-read rungs are a different read budget, compared with the k = 3 rung (B1 0.452) only as paired
>   contrasts on identical queries, labelled as a different read budget, never as E-A values. V3 values on other row
>   sets are never placed next to 0.452.

**Source key** (paths relative to `generations/gen21/`). EB = `round5_design_20261023/evidence/EVIDENCE_BASE.md`
(E1–E6 are its analyst files); IR = `round5_design_20261023/ideas/IDEAS_RANKED.md` (M01–M39; `critiques.json`);
I2 = `round5_design_20261023/ideas2/IDEAS2_RANKED.md` (round-2 ideas, rules 9–16); P1–P5 = `round5_design_20261023/probes/`;
PC = `probes/PROBE_CHECKS.md`; R4 = `round4_20261022/` (REPORT, PROTOCOL, POWER, AMENDMENTS, DATA_REQUEST).

---

## 1. Recommendation

**E-A is finished for model work on this corpus.** Two idea rounds, five probes and their checks found no model change
that can be shown to beat B1 (radius interpolation plus one fold-fitted Gd amplitude). Every proposed E-A correction is
below its contrast-specific dev MDE (for example 0.011 R² for S05 − B1 and 0.003 for S11 − B1; EB §3; PC P3), carried
by one or two publications, or killed by its own criterion (IR and I2, Bottom line). The corpus could not confirm one
anyway: a pooled B1-sized gain needs about 40–44 new publication-like sources at τ = 0 (SF_POOL 43, SEALED ≈ 41,
JK_KISH ≈ 40, the last not a gate) and more than 80 at τ = 0.065 (P2 §4, §6; PC P2). B1 stays the deployed reference.

**0.50 at four read metals (E-A) is not reached by any tested or proposed model, and E-A is closed to new model slots
on this corpus.** Gains below about 0.01 R² are undetectable here (EB §3). With a fifth read (E-D, a different read
budget), B1 scores 0.497 on the C3 min(4, k_max) population row, a mixture in which 15.9 % of rows keep 4 reads, and
0.480 on the like-for-like C4 cohort. The paired gain of the fifth read on identical queries is +0.045 [+0.038, +0.070]
(BPG, 5/5 seeds; P3 §2; PC P3). Six reads give 0.530 only in E3's exploratory calculation on the C3 population row
(HIST_D7 form, a post hoc max-min-twice rule, E3's looser no-look-ahead legality rule; a mixture, not recomputed under
the protocol rule; EB §6.1; E3 §6), not on the registered B1 DOPT_Q-nested ladder; the fixed C6 cohort gives 0.524–0.531
at six reads (IR M18; E6 §2). Reading every other measured metal, with the k = 3 hidden ends kept hidden, reaches only
0.504 (EB §1.3). The noise share of B1's error is known only to lie in [0.08, 0.87] (EB §1.4). On E-A itself only new or
cleaner data can move the level, and those results are reported as V3 or U4 sensitivities.

**What round 5 should do, in this order.**

*Stage 0 (hours, before any round-5 score): seal the conventions.*
1. **M35, one currency.** E-A gate = pooled relative pair-SSE reduction vs B1 ≥ 2 % (≈ +0.011 R²) with a balanced-MAE
   non-inferiority guard ≥ −0.001; BPG decides, BP is a labelled sensitivity on the same test folds; mandatory
   per-cluster, big-fold-vs-rest and C5-BTBP-in/out tables (§4.2).
2. **M34, inference.** Keep the exact sign-flip on pooled ΔSSE and the exact sign test; drop the sealed jackknife-t
   (df = G − 1, anti-conservative); equal-weight log-ratio tests are breadth diagnostics only; new-source unit =
   laboratory × chemotype; fixed-sequence gatekeeping, no Holm inside a claim at small G; G floors and power in worlds
   declared in advance (§4.3).
3. **confirm-1 (CP-13) package rules**: per-claim minimum G, an exclusion list, inconclusive zones; non-rejection never
   switches the deployed model (§4.4).
4. **Closure ledger** of the E-A routes closed on this corpus, including lab-3, with grades (§3).

*Stage 1 (software, hours to days; every number "seen").*
5. **E-CT, condition transfer within a campaign** (M01 as corrected by P1 and its check; §4.1): duplicate scan first;
   frozen shrunk own-series template SH (K = 1.78); C5-BTBP in/out; read ladder 0/2/3/4; dose-response by sibling count;
   comparators on identical rows (B1, COPY, OFF, and the mechanism comparators OFF_XEXT / PB_PUB); placebo blends;
   annexes E-LAB (lab-1) and the M03 weighting factorial.
6. **Descriptive deliverables**, never decision slots: the M09 tail-risk flag, M14 selective direction calls, the M18
   metal ladder with the k = 4 rung and the M17 policy envelope, the M13 wider-pair table, the frozen M07 sc009 Nd dip,
   and the practical guidance of §5.

*Stage 2 (data-gated; requests to the user, §6).*
7. **U1** source recovery (about 3 min for the three-minute package to 35–50 min for the highest-stake document
   package; U1-h about 4 h more) → V3 sensitivities; **U2** SAFE gap-ID export → supply gate; **U3** one SI fetch → a
   dry run of the sealed E-CT pipeline on a corpus-linked laboratory; **U4** a laboratory collaborator → re-measurement
   and reproducibility; **U5** web reading for the supply ledger and the planning amplitudes; **U6** review of the
   laboratory key; **S1** the V2-3 cohort-key sign-off. The full list (U1–U6, S1, H1) is `DATA_REQUESTS.md`. With the
   literature supply ledger these feed one frozen confirmation package (C-ECT, C-GD, Gd-dip transport), sized in worlds.

**Where the remaining value is.**
- **Condition transfer gives the largest seen gain over B1 on identical rows** (E-CT, a separate estimand; B1 itself
  scores 0.531 on these rows, so the level is not the point). On 99 V2-clean cells, SH with 4 reads scores 0.708 against
  B1's 0.531 on the same rows (+0.177); the gain is +0.168 [+0.054, +0.283] once a suspected duplicate pair is dropped
  (P1; PC P1). Its mechanism is not identified, 94 of the 99 cells are DGA, and the decisive test needs at least 7
  independent laboratories with non-trivial condition series (I2 confirm-4).
- **Source recovery carries the highest stakes.** The 69 cells with no source location hold 56.4 % of B1's SSE and
  68.4 % of its gain over B0 (P5 §3), but only a curator reply (P ≈ 0.3) reaches all of them. The 35–50-minute
  highest-stake document package covers about 26 % / 44 % (≈ 20 % / 33 % probability-weighted), the three-minute package
  about 14 % / 28 % (PC P5). B1's Gd transfer to the located documents is learned mainly from the unlocated cells under
  the current inner-CV ridge; with λ fixed at 0.03 about half of the located-row advantage survives (+0.012 of +0.024)
  (P5 §4b; PC P5).

**Stopping.** No new E-A model slot unless it passes the retire-only pre-slot decidability gate (IR M35 item 7): its own
test-label oracle ceiling ≥ 2 % pooled relative SSE vs B1; power ≥ 50 % at its plausible effect; demotion if the deciding
fold has fewer than 10 training cells sharing the test rows' donor class or chemotype. Otherwise E-A stays closed on this
corpus (§7.3).

---

## 2. What round 4 and the round-5 analyses established

### 2.1 Target and reference ladder (E-A)

| quantity | value | source |
|---|---|---|
| target y = log D(A) − log D(B), ΔZ = 1, 12 pair types | var 0.0493 (sd 0.222); median \|log SF\| 0.114 (mean 0.164); interior mean ≈ 0.17 log ≈ 1 kJ/mol | EB §1.1 |
| zero-shot structure models | G14 R² 0.053; neural NNC −0.128 (−0.29 on confirmation) | EB §1.1 |
| B0 = PWL_R (radius interpolation, no training) / HBK@ML (best learned curve) | 0.418 / 0.419 (error correlation 0.971) | EB §1.1 |
| **B1 = B0 + one fold-fitted Gd amplitude** | **0.4521 BPG / 0.4519 BP**; balanced MAE 0.1011; RMSE 0.164 | R4 REPORT §1; EB §1.1 (RMSE) |
| HIST_D7 (fixed historical −0.072) | 0.4511 (= B1 − 0.001) | R4 REPORT §1 |
| B1 source-cluster interval; SSE share to remove for 0.50 | [0.394, 0.497]; 8.8 % | R4 REPORT §1; EB §1.1 |

### 2.2 Where B1's error lives

| slice | share of rows | share of B1 SSE | source |
|---|---|---|---|
| top 1 % / 5 % / 10 % of rows (kurtosis 18.2) | | 31 % / 59 % / 73 % | EB §1.2 |
| one cell, c950dc2982e96b83 (C5-BTBP) | | 11.8 % | EB §1.2 |
| one publication, C5-BTBP pub_6a0ef8528c (15 cells) | 8.2 % | 23.1 % | EB §1.2 |
| light pairs La/Ce … Pr/Nd (La/Ce + Ce/Pr: 34.2 % on 14.5 % of rows) | 21.6 % | 42.8 % | EB §1.2 |
| Gd region Sm/Eu … Gd/Tb | 26.7 % | 27.2 % | EB §1.2 |
| either log D outside [−2, 2] (label-defined) | 26.1 % | 49.7 % | EB §1.2 |
| non-DGA donors | 37.5 % | 55.2 % | EB §1.2 |
| giant source cluster g:pub_74428aeab2 | 64.9 % | 51.8 % | EB §1.2 |

The typical row is good and the tail is not: robust residual sd 0.086 against RMSE 0.164. Error does not grow with
distance to the nearest read (MSE 0.028 at 1 Z, 0.020 at ≥ 4 Z); which metal is hidden matters. Tm/Yb and Yb/Lu sit at
R² ≈ 0 because DOPT_Q rarely reads Lu (EB §1.2; E6 §6.1).

### 2.3 The residual is metal-local and shared within a publication (campaign), not across publications

- **Spikes plus level drift.** Consecutive-pair residual correlation is −0.522 where the shared metal is unread;
  consecutive hidden level errors correlate +0.69; smooth curve-shape error is about 10 % of the residual variance.
  Per-cell offsets explain 2.6 % of the pair-residual sum of squares (EB §1.3).
- **Short-range roughness is 31–35 % of the target variance**; the corpus-mean per-metal pattern is about 4 % (EB §1.3).
- **The roughness belongs to a campaign:**

  | relation between two cells | roughness r | B1 residual r |
  |---|---|---|
  | same extractant, same publication | +0.36 | 0.33 |
  | other extractant, same publication | +0.14 | |
  | same extractant, other publication | −0.02 | 0.10–0.14 |

  Source: EB §1.3. Per-metal deviations of the same extractant in another publication share r = 0.010 [−0.23, +0.17]
  (I2 lab-3). What a laboratory's papers do share is the Gd chord (+0.37 to +0.47 on publication means; I2 rule 12).
  EB §1.3 words this as "within a laboratory", but every measured stratum is "same publication" against "other
  publication"; the corpus has no laboratory key, so "same publication" reads "same study or campaign" (IR rule 8;
  P1 §14).

### 2.4 The noise floor is bounded, not measured

| handle | implication | source |
|---|---|---|
| robust per-metal σ 0.030–0.058 / mean-square σ 0.107–0.188 | floor = 4–16 % / 56–87 % of B1's MSE | EB §1.4 |
| 15 replicated C3 cells | replicate noise = 54 % of B1's MSE on the same rows | EB §1.4 |
| Spearman(cell B1 MSE, cell roughness) | 0.88 | EB §1.4 |
| threshold for 0.50 at k = 3 | impossible if σ_SF ≥ 0.157 (single-target model) to ≈ 0.19 (E2's floor model) | EB §1.4, App. 5 |

In bulk rows B1's error variance (0.0074) is only 2–6× the robust floor; in tail cells the cell's own roughness is as
large as B1's error. Laboratory repeatability has never been measured (EB §1.4, gap G1).

### 2.5 What B1 captures, and what it rests on

- **One number.** Fold β −0.094 ± 0.015; R² is flat over β ∈ [−0.072, −0.094]. Eu/Gd (0.276 → 0.422) and Gd/Tb
  (0.321 → 0.546) carry 94 % of the gain; a within-pair-type shuffle keeps 86–90 % of it, so it is a pair-type mean,
  not cell chemistry (EB §1.5).
- **How firm.** +0.034 R² and +0.005 balanced MAE in 5/5 seeds on both designs, replicated on round 3's final seeds;
  the cluster-bootstrap intervals exclude 0, but it fails the small-cluster jackknife-t test: the interval
  [−0.009, +0.077] includes 0 (R4 REPORT §1; EB §1.5).
- **What anchors it.** C5-BTBP gives 34.9 % of the gain and DMDPhPDA 17.6 %. The 69 cells with no source location
  (54 ORNL imports + 15 C5-BTBP) hold 56.4 % of B1's SSE and 68.4 % of its gain (P5 §3). Refit without them, B1 loses
  training fit by −0.0285 R², outside all 40 random-removal draws (worst −0.0225); on the 160 located cells the refit
  then ties B0 (−0.0045, 3 of 5 seeds negative), or keeps +0.0119 with λ fixed at 0.03, a sensitivity (PC P5). B1's Gd
  transfer to located documents is learned mainly from the unlocated cells under the current inner-CV ridge; with λ
  fixed at 0.03 about half of the located-row advantage survives (+0.012 of +0.024; PC P5). The amplitude itself
  survives the refit (β mean −0.0708 without the 69, against −0.0923 with all cells; P5 §4b).

### 2.6 Why round 4 could not decide, and what the probes added

Round 4's geometry decided little (EB §2.14–2.17): BP and BPG share 100 % of fold assignments; one fold holds 58.2 % of
scored rows and trains on 51–69 C3 cells; the 0.005 balanced-MAE margin equals B1's whole balanced-MAE gain, which
corresponds to +0.034 R² under B1's gain structure (+0.053 under uniform shrinkage) (EB §2.17);
inner CV was optimistic (outer SSE worse in 12 of 14 fold-fits); pointwise training is misaligned with the pair target.
The probes (P1–P5) are summarised where they are used: §4.1, §4.3, §4.8, §4.17 and §4.13.

---

## 3. What is closed and how firmly

**Grades** (EB §3). **FIRM**: negative or null when cross-fitted, an oracle ceiling below the margin, algebraic, or
replicated. **MODERATE**: null against B1 and consistent with the ceilings, but underpowered. **WEAK**: "not detected";
control power not shown. "Closed" means "cannot beat B1 by ≥ 0.01 R² under BP/BPG on this corpus", not "chemistry
absent" (EB §3). Idea-round lines are graded here with EB §3's rule unless the source gives a grade.

### 3.1 E-A model routes closed on this corpus (the Stage-0 ledger; the executable ledger is CLAUDE_CODE_PROMPT §4.4)

| route | evidence | grade | power caveat | source |
|---|---|---|---|---|
| zero-shot structure → adjacent log SF (fingerprints, descriptors, ChemBERTa, kernels, log K prior, NNC) | R² ≈ 0.05; NNC −0.128 / −0.29 | FIRM | confirmation seeds, many designs | EB §3 |
| learned curve families vs PWL at k = 3 (RIDGE2, HB2+GP, HBK) | ≤ PWL_R; error correlation 0.971 | FIRM | random-walk algebra | EB §3 |
| smoother or higher-order interpolation | −0.020 to −0.095 | FIRM | | EB §3 |
| universal per-metal / pair-type / family priors (B2, tetrad, inclined W) | LOCO −0.018 to −0.047; B2 −0.022 (0/5 seeds) | FIRM | ceiling ≈ 0.04 even if perfect | EB §3 |
| **lab-3: per-metal priors from other publications**, unconditional or level-matched | MG_tr −4.8 % rel. SSE [−8.5, −1.2]; MG_LM10_tr −5.8 % [−10.1, −1.5] | FIRM on harm | not on the reopen bar; same-chemotype priors untestable under E-A | I2 lab-3 |
| amplitude-scaled (proportional-LFER) anomalies | LOCO −0.017 (0/5) | FIRM | | EB §3 |
| convex blends / stacking; per-cell offset / global recalibration | leave-one-cluster-out −0.019 to −0.035; [O] recalibration +0.0005 | FIRM | | EB §3 |
| static fixed read sets | LOCO 0.484 vs DOPT_Q 0.489 | FIRM | 78 cells | EB §3 |
| sibling-read pooling under BP (HBG@POOL) | −0.041 / +0.002 | FIRM for reads only | says nothing about full sibling curves | EB §3 |
| cross-metal xTB energies with changing composition | ρ −0.084 at reliability 0.80 | FIRM | powered null | EB §3 |
| floor-aware read clamps (M24) | 91 % of the gain is one cell; cross-fitted F 0.4098 | FIRM on V1; substance needs a per-source limit | undecidable without a source-stated limit | IR M24 |
| Ce-acid term (M28) | other rows −0.0034 (0/5); within-publication permutation keeps 61–76 % | closed lead (IR ledger; graded FIRM here by EB §3's rule) | a two-publication effect | IR M28 |
| abstention by difficulty scores (spread, error tables, \|B1\|) | within-pair-type Spearman ≤ 0.18 | FIRM for these scores | not for the level window | EB §3 |
| S01 donor-class Gd / Nd dips | −0.0001 / −0.0005 | MODERATE | synthetic 0/8 at D7 size; cross-fitted donor × pair ceiling −0.041; [O] ceiling +0.0004 balanced MAE | EB §3 |
| S02 electron-count shapes × donor class | −0.024 / −0.028 | MODERATE | loss sits in the 51–69-cell big fold (+0.0019 R² elsewhere) | EB §3 |
| S05 CatBoost residual corrector | −0.006 / −0.003 | MODERATE | dev MDE 0.011; inner selection optimistic | EB §3 |
| S11 Ln-only multitask encoder | −0.0005 / −0.0017 | MODERATE | reproduced B1 (row r 0.99983) | EB §3 |
| S06 compact 3D geometry | −0.006 / −0.011 | WEAK | synthetic 0/6; 195 / 229 cells change composition | EB §3 |
| An(III) transfer (S13–S16) | inner gate 3 of 50 | WEAK | synthetic 0/6; 83–93 % Am | EB §3 |
| learned value-adaptive read policy | +0.006 [−0.033, +0.019] at k = 3 | WEAK (open at high k) | 13 clusters | EB §3 |

**Power caveats for every MODERATE and WEAK line** (EB §3): Kish 2.24, with 65 % of rows in one cluster; the percentile
cluster bootstrap rejects at 0.07–0.08 for a nominal 0.05; the synthetic studies used 6–8 replicates (exact 0/n upper
bounds 0.37–0.46) and were miscalibrated (S05 and S11 gained about +0.02 over B1 in NULL worlds); the dev MDEs (80 %,
cluster_cpg) are 0.044 (B1 − B0), 0.011 (S05 − B1) and 0.003 (S11 − B1); BP is not independent of BPG. The in-sample
donor contrast (+0.04 R² [O]) is real; it is not transferable across about 4 effective chemotypes (EB §3).

### 3.2 Method, data and practice closures

| line | evidence | grade | source |
|---|---|---|---|
| M01 kill (i) "SH − OFF" is an identity where reads align | \|SH − OFF\| ≤ 4.4e-16 on 72.9 % of PRIMARY rows (BPG; 69.1 % BP) | FIRM (algebra) | PC P1 |
| The sealed jackknife-t leg (df = G − 1) is anti-conservative | real pools: mean size 0.047, max 0.096 at nominal 0.025 | FIRM | PC P2 |
| Equal-weight tests are not E-A tests | they reject HBK@ML − B0 (+0.14 % pooled BPG, BP −0.64 %; P2 §2; IR M35 rounds the same contrast to +0.13 %) in 93–100 % of cpg draws at G 20–40 at τ = 0 (51–82 % at τ = 0.065) | FIRM | P2 §5 |
| E-CENS at F_lo = −4 is the C5-BTBP exclusion | 0.487 vs 0.486 | FIRM | IR M23 |
| V2-S is null; robust losses add nothing | V2-S moves B1 by +0.001 (a data-version sensitivity on a different row set; inside the random-removal null); V2-5's 16 pairs per seed (4.1 % of SSE) carried all of round 3's V2 move; Huber β −0.1167 vs −0.1147 | FIRM; MODERATE | IR M26 |
| The edge continuation is a reference variant (B1 + edge), not a claim | +0.004 to +0.0055 cross-fitted | MODERATE | IR M28 |
| Low-D sibling irreproducibility is C5-BTBP | ratio 1.08 without it | FIRM | IR M10 |
| No Gd term in aqueous-complexant media (data-contract rule, unscored) | scoring killed by drop-one (78.5 % one paper) | rule | IR M29 |
| Cross-cluster residual share ≈ 0; publication-weighted conformal = width inflation | −0.13 % [−0.41, +1.37]; row-weighted at 0.932 equivalent | MODERATE; FIRM | IR M08, M16 |
| Replicate a read only if per-metal σ > about 0.2; otherwise add a metal | ΔMSE = 0.216 s²; break-even 0.20–0.31 | FIRM (algebra) | IR M11 |
| SAFE "+1,618 newer records" are mostly pre-existing gap IDs | 1,602 missing IDs in 11 runs; about 16–21 truly newer | FIRM | IR M33 |
| Zero-read copying of an own series is never the recommendation | COPY@0 0.655 vs SH@2 0.664; below-range flag +0.30 | MODERATE | I2 shift-1 |
| Free per-metal acid-slope regression; unshrunk acid-trend extrapolation | AREG@0 −0.279; extrapolation −0.64 R² (6 of 8 publications lose) | FIRM | I2 shift-2, recipe-4 |
| A literature series of the same extractant adds no direction information | 0.922 vs always-heavier 0.934, identical rows | MODERATE | I2 recipe-3 |
| "Spread the reads, never one metal" is definitional | HULL − random +0.197 at 2 reads | definitional | I2 recipe-2 |
| Publication units are invalid for Gd-amplitude contrasts | Gd chord shared +0.37 to +0.47 within a laboratory | MODERATE | I2 lab-4 |
| An in-run TODGA reference is not needed for prediction | any second own-campaign ligand: MX +5.9 % vs own TODGA +5.3 % | MODERATE | I2 lab-2 |

### 3.3 Not closed

- **Same-chemotype priors under E-A are untestable** (0 of 1,145 instances have a same-chemotype training donor), and the
  E-KX sc009 Nd dip is carved out (I2 lab-3; §4.11).
- **Electronic, 3D and An(III) effects are "not detected"**, with control power not shown (R4 REPORT §2).
- **Floors are undecided in substance**; only a source-stated detection limit settles them (IR M24).
- **Measurement- or model-limited?** Open until U4 (§2.4).
- **Reopen bar for per-metal priors (lab-3 route).** A pre-declared fold-legal variant must beat B1 by what 13 clusters
  can detect (about 4–6 % relative SSE, about 0.02–0.03 R²) with a positive cluster-jackknife lower bound with and
  without C5-BTBP (I2 lab-3), and pass the pre-slot gate (§7.3). Other E-A routes face only the pre-slot gate and M35's
  2 % gate (§4.2).

---

## 4. The ideas, ranked

Ranking: tier first (what round 5 can and should run), then decision value × probability it holds as corrected ÷ cost.
Round-1 ideas carry judge scores (J1–J3, IR); round-2 ideas carry skeptic priors only (I2). "Prior A / B" is each
skeptic's probability that the idea holds and passes its own kill.

| rank | id | tier · stage | what it is | action | prior A / B |
|---|---|---|---|---|---|
| 1 | M01 | 1 · S1 | E-CT condition transfer (shrunk own-series template) | run, frozen design | 0.50 / 0.55 |
| 2 | M35 | 1 · S0 | one-currency decision convention | seal | 0.30 / 0.30 |
| 3 | M34 | 1 · S0 | few-cluster-valid inference (P2 replay done) | seal | 0.20 / 0.25 |
| 4 | confirm-1 | 1 · S0 | CP-13 confirmation package rules | seal into M34 / M35 / M33 | 0.35 / 0.20 |
| 5 | M24, lab-3 | 1 · S0 | closures: floor clamps; foreign per-metal priors | ledger | 0.45 / 0.85; 0.80 / 0.85 |
| 6 | M03 | 2 · S1 | sibling weighting factorial | annex to M01 | 0.40 / 0.20 |
| 7 | lab-1 | 2 · S1 | E-LAB: new ligand calibrated from its campaign | annex to M01 | 0.30 / 0.15 |
| 8 | M18 + M17 | 2 · S1 | metal ladder with the k = 4 rung; policy envelope | descriptive | 0.75 / 0.85; M17 0.60 / 0.60 |
| 9 | M09 | 2 · S1 | level window as a tail-risk flag | descriptive | 0.15 / 0.15 (tail flag ≈ 0.7) |
| 10 | M14 | 2 · S1 | selective direction calls (E-DEC) | descriptive | 0.20 / 0.35 |
| 11 | M07 | 2 · S1 | sc009 Nd dip frozen at −0.11 | descriptive + case series | 0.15 / 0.15 |
| 12 | M13 | 2 · S1 | wider-pair table (E-W) | descriptive | 0.80 / 0.65 |
| 13 | M25 | data · U1 | stakes-first source recovery → V3 | request | 0.25 / 0.25 |
| 14 | confirm-4 (+ M05) | data · S0 seal, U3 | C-ECT external condition transfer | seal now, decide on data | 0.50 / 0.15; M05 0.12 / 0.15 |
| 15 | confirm-2 / -3 | data · S0 seal | C-GD cation-exchanger Gd break; one chord interval | stratum of M33's test | 0.55 / 0.25; confirm-3 0.40 / 0.25 |
| 16 | M33 | data · U2 | SAFE gap-ID audit (supply gate) | request | 0.05 / 0.20 |
| 17 | M22 / P4 | data | literature supply ledger | ledger | 0.08 / 0.05 |
| 18 | M11 + M12 | data · U4 | re-measurement; inter-laboratory reproducibility | request | 0.30 / 0.30; M12 0.12 / 0.10 |
| 19 | shift-3 | data | Ce/Pr acid sign in DGA (sign only) | rider | 0.35 / 0.15 |

M22 moves from IR Tier 2 (rank 8, web desk work) to data-gated here, because it needs web permission (U5). In merged rows
the first prior pair belongs to the first idea named; the second idea's priors follow it.

### Tier 1: run first (software, hours)

#### 4.1 M01: E-CT, condition transfer within a campaign (Stage 1; J 7.5 · 8 · 6.5)

- **Hypothesis.** A study (campaign) that has measured an extractant at other conditions predicts adjacent log SF at a
  new condition from the shrunk mean of its own series plus 3–4 reads better than B1 does with 4 reads. Laboratory
  wording waits for the frozen laboratory key (IR rule 8; P1 §14). Which information carries this (same-extractant
  template, campaign fingerprint, shared artefact) is a second question.
- **Evidence (seen; P1 with PC corrections).**
  - *Primary stratum* (V2-clean, n_sib ≥ 2, no C5-BTBP; 99 cells, 12 publications, 14 publication × chemotype units):
    SH 0.708 vs B1 0.531 on identical rows, +0.177, LOPO 95 % [+0.076, +0.279], 13 of 14 units better; +0.168 [+0.054,
    +0.283] without the bit-identical cross-acid pair 44012938119033a5 / 5169c377777d282b (pub_19783c039b).
  - *Read ladder*, SH / B1 at the same reads: 0.616 / 0.143 (2 reads), 0.679 / 0.467 (3), 0.708 / 0.531 (4). COPY at 0
    reads (0.497) is below B1@4. SH@3 − B1@4 +0.148 (12 of 14 units); SH@2 − B1@4 +0.085 but only 8 of 14 units, top-3
    share 0.99 (P1 §6, §12).
  - *Dose-response.* Dense-grid subsampling (62 cells, B1 0.458): raw template 0.338 → 0.641 and SH 0.546 → 0.652 from
    1 sibling to all. At n_sib = 1, SH 0.603 vs B1 0.547 (+12.2 % relative SSE); a raw single sibling 0.510 (P1 §7–8).
  - *Same campaign only.* Placebo blends at the same w: same chemotype, other publications −1.3 % vs B1; same
    extractant, other publications −2.6 %; other extractants of the same publication +13.6 % against SH's +30.0 % on the
    same 25 cells (P1 §11).
  - *Mechanism not identified.* SH − OFF is +0.0119 [−0.017, +0.041], all from pub_19783c039b, and +0.0094 on the V1
    stratum, so kill (i) flips between strata; OFF_XEXT's "47 % recovery" pools 3 disagreeing publications (PC P1).
  - *Concentration.* 3 cpg clusters, 89 % of rows in one; pub_b0b3296ec5 carries 41 % of the gain; 94 of 99 cells are
    DGA; Ce pairs carry 32.8 %, Gd pairs 23.6 %. Acid-concentration-only targets (23 cells, 5 publications), the
    broadest real condition class: B1 0.451 → SH 0.655 (P1 §9–10; I2 confirm-4).
- **Corrected test (frozen before the run).** (1) Duplicate scan first (I2 rule 15): bit-identity, constant-offset copies
  and quantised values within and across publications. Links flagged by the frozen bit-identity rule (I2 items 1, 3, 4)
  are removed from sibling and donor pools in the primary analysis; lab-1's label-free near-copy rule applies to donors.
  Constant-offset and quantised items (8, 9, 11), P1's CLEAN rule without the bit-identity check, and the other listed
  items are sensitivities and provenance questions. V1 is never edited. (2) Population
  from metadata: same publication and extractant, siblings covering the target's metals, V2 D1–D3 removed; primary
  n_sib ≥ 2 without C5-BTBP, C5-BTBP in as a sensitivity, n_sib = 1 as the harm stratum. (3) Frozen SH = w·SIB +
  (1 − w)·B1, w = n/(n + K), K = (1 − r)/r = 1.78 from r = 0.36; at 2–3 reads, shrink toward B1 on the same reads. K is
  fixed from r = 0.36 measured on these same rows (seen; P1 §15). SH is not distinguishable from SIB (9 W / 5 L, p 0.34)
  or NEAR (7 / 7), so the claim is "own-series template + 3–4 reads", not "K = 1.78" (I2 confirm-4).
  (4) Comparators on identical rows, DOPT_Q nested within seed: B1 (same k and 4 reads), B0, COPY, SIB, OFF (reported
  as the read-alignment effect), nearest-condition and random single sibling, placebo blends PB_CHEM / PB_XEXT / PB_PUB
  at the target's own w, and the mechanism comparators **OFF_XEXT and PB_PUB**. (5) Read ladder 0 / 2 / 3 / 4;
  dose-response by within-publication subsampling (m = 1, 2, 3, 5, all); decompositions by pair type (with and without
  Ce and Gd pairs) and condition-change class. (6) Unit publication × chemotype: exact sign-flip, LOPO jackknife, top-3
  share, equal-weight per-publication relative SSE; cluster inference "not estimable"; until laboratories are annotated
  the claim reads "same study or campaign". (7) Annex rows: M02 (SIB@2 vs SIB@4, COPY), M04 ("avoid floor or saturated
  reference conditions"), M06 (same campaign vs other publication), shift-1 (acid distance × range position), recipe-1
  (credit by sibling count against the best no-prior policy).
- **Kill criteria.** (i) is replaced by **SH − OFF_XEXT (or PB_PUB)** where both exist, registered before the run;
  "campaign fingerprint" wording needs ≥ 4 families with a pool and a threshold set away from the seen 43–47 %. (ii) SH
  no worse than B1 by > 5 % relative SSE at n_sib = 1 (seen +12.2 %). (iii) Artefact flag if ≥ 50 % of the gain sits on
  Ce pairs (seen 32.8 %). (iv) The prospective kill is confirm-4 (§4.14). In-corpus legs are consistency checks known
  to pass (IR rule 2).
- **Expected effect.** About +0.17 over B1 on identical rows in corpus (seen); externally about +0.05 to +0.10 for
  DGA-type laboratories with dense condition grids, unknown elsewhere (IR M01).
- **Practical value.** For a user who already holds earlier series of the same extractant, this is the largest seen gain
  over B1 at the same 3–4 reads (§5.4). The information is the user's own prior data, not learned chemistry
  (CLAUDE_CODE_PROMPT §10). If confirmed, it matters for radiotracer and subset-feed work (about one tracer saved against
  B1@4) and for curators imputing partly reported conditions. An ICP-MS laboratory changes nothing in measurement, and
  copying without reads is closed (I2 confirm-4; I2 shift-1; COPY 0.497 vs B1@4 0.531, P1 §6).

#### 4.2 M35: one-currency decision convention (Stage 0, seal; J 3.5 · 3.5 · 4)

- **Hypothesis.** A rule in relative pair-SSE against B1, with a balanced-MAE guard and BPG deciding, passes the known
  positive control and rejects every round-4 loser.
- **Evidence.** 0.005 balanced MAE ≈ +0.034 to +0.053 R², and block A failed that leg even as a test-label oracle
  (+0.0408 R² / +0.0004 balanced MAE) (EB §2.17). B1 − B0 is +5.85 % pooled relative SSE; every round-4 slot is ≤ −0.02 %;
  outside the big fold no slot reaches 2 % (best S02 +0.28 to +0.35 %). Holm over a 14-contrast family gives the
  positive control p 0.16; the equal-weight leg fires on HBK@ML − B0 (p 0.0115) although HBK@ML is +0.13 % pooled (IR M35;
  the same contrast P2 §2 reports as +0.14 % BPG, −0.64 % BP). The positive control passes the sealed gate as seen:
  B1 − B0 +5.85 % pooled relative SSE, cpg SF_POOL p 0.00317 (BPG) / 0.00269 (BP) (IR M35; PC P2).
- **Sealed text.** (1) E-A pooled R² for continuity. (2) Gate: pooled relative pair-SSE reduction vs B1 ≥ 2 %
  (≈ +0.011 R²), tested with SF_POOL on the 13 cluster_cpg units (one-sided p ≤ 0.025, exact over 2^13 sign vectors;
  CLAUDE_CODE_PROMPT §4.1). (3) Guard: balanced-MAE non-inferiority ≥ −0.001 (seed-mean point estimate). (4) BPG decides; BP is a
  labelled sensitivity (same test folds). (5) Mandatory tables: 13-row per-cluster table with a provenance column
  (ORNL-only, data_location class); big fold vs the rest; C5-BTBP in / out with leave-one-publication-out and the c950dc
  single-cell influence; the six-design table for training-free arms. (6) Equal-weight per-cluster log SSE ratio as a
  breadth diagnostic only (clusters < 3 cells or < 20 rows per seed pooled). (7) The pre-slot decidability gate
  (retire-only, M38's salvage; §7.3). (8) B1 − B0 is the positive control outside any family; inside a claim, fixed
  sequence. (9) Three-way outcomes with an explicit inconclusive zone; non-rejection never switches the deployed model
  (I2 rule 9). (10) Appendix: M08's relation-class moment table with an identifiability column.
- **Kill.** Abandon the convention only if the replay with fixed sequence fails the positive control.
- **Expected effect / practical value.** None on B1's level; consistent margins and fewer unwinnable slots.

#### 4.3 M34: inference that keeps its size with few clusters (Stage 0, seal; P2 done; J 4.5 · 4.5 · 5)

- **Hypothesis.** Exact few-cluster tests keep nominal size; the confirmation planning number depends on the declared
  unit and estimand and can be computed now.
- **Evidence (P2 with PC corrections).** SIGN, SF_LOG, SF_POOL, T_LOG and IM_REL keep nominal size in every gating cell
  (by construction under the symmetric nulls; SF_POOL reaches 0.103 under a skewed null, not gating). JK_G1, the sealed
  jackknife-t leg, is anti-conservative; JK_KISH, SEALED and WCR_POOL fail the size gate marginally. G for 80 % power on
  B1 − B0 (BPG, one-sided 0.025, τ = 0 / 0.065): publication-like SF_POOL 43 / > 80, SEALED about 41 / > 80, equal-weight
  SF_LOG 19–20 / about 26; chemotype-like pooled tests 30–58 at τ = 0; publication × chemotype 59–72; a non-DGA
  publication pool SF_LOG 12 / 17, a DGA pool more than 60. At G = 8 publications the best gate-valid power is 0.228 /
  0.236 (BPG / BP) at τ = 0.065 and SF_POOL 0.285–0.296 at τ = 0, so on the pooled estimand M34's kill (< 0.30) fires
  at both τ. All of it is optimistic for new sources: B1's effect was chosen on these cells (PC P2).
- **Sealed text.** (1) Pooled estimand: **SF_POOL**; π estimand: **SIGN**; JK_G1 dropped; JK_KISH is a comparator only,
  never a gate at any G (size flags from G = 8; real-pool maximum 0.0515; PC P2).
  (2) Equal-weight log-ratio tests answer "does the average source improve?": diagnostics only, never E-A. (3) Unit:
  **laboratory × chemotype**, from P4's frozen lab_id with the repaired key (common-surname links dropped); publications
  aggregated within a unit; copy groups merged at arrival from metadata; for Gd-amplitude contrasts, the laboratory
  across chemotypes (lab-4 salvage). (4) Fixed sequence; no Holm inside a claim at small G; G ≤ 5 cannot reject at
  0.025 (minimum p 1/32), and G = 6 needs 6 of 6. (5) G floors per claim declared before data (§4.4). (6) Size gate:
  drop a test whose Clopper–Pearson lower bound exceeds 0.035 at nominal 0.025 under i.i.d. nulls and two-stage
  (chemotype → laboratory) nulls at ICC 0.23–0.69; the laboratory unit removes within-laboratory clustering only, and
  the C-GD laboratory t already fails this gate in all 63 cells of `ideas2/confirm1skepB_out/clustered_null_size.csv`
  (0.041–0.142 at nominal 0.025), so a fallback is sealed with it (CLAUDE_CODE_PROMPT §4.3 item 2). (7) Power in worlds (corpus-DGA, halved effect, cation-exchanger /
  null-like, 25 % and 50 % null-like mixtures), with every kill rule's operating characteristics (I2 rules 9, 11). (8)
  M37's B1 − B0 ledger opens only behind the M33 supply gate.
- **Consequence.** Keep the 40–80+ source planning number for any pooled claim; every round-5 E-A contrast is a
  diagnostic.
- **Practical value.** It tells the data request how many independent laboratories to ask for; more papers from the
  same laboratory add nothing.

#### 4.4 confirm-1 (round 2): CP-13, a frozen confirmation package sized to the real supply (Stage 0, seal)

- **Hypothesis.** About 7.0–11.5 new laboratories expected with PC P4's corrections (13.5 under P4's own tier rule; 5–6
  verified) can decide only a few questions; sealing only those, each with a minimum G and with undecidable endpoints
  excluded, turns the supply into decisions or honest case series (I2 confirm-1; PC P4).
- **Evidence.** Decidable in principle: C-GD (0.62–0.74 at G = 6), A1 "a dip exists" (0.83 at G = 8), E-CT SH@4 vs
  B1@4 (0.89–0.93 at 7 families). Not decidable: a TOST on β (≤ 0.08), the window as a row flag (≤ 0.15), SH@2
  (≤ 0.06). The skeptics found a stratum mismatch (confirm-1's mechanism-metadata class, aromatic N-donors plus acidic P,
  19 units, has power 0.13 / 0.23 / 0.28 / 0.33 at G = 4 / 6 / 8 / 10), a DGA / N-donor planning chemistry against an
  acidic-OP supply, realistic heavy-selective G of
  3–5, size 0.032–0.051 with 6 units from 3–4 laboratories, and an A1 rule that would withdraw a real effect about 17 %
  of the time (I2 confirm-1).
- **Sealed text (into M34, M35, M33).** A per-claim minimum-G table with supply-scenario power, recomputed laboratory by
  laboratory from P4. Exclusion list: pooled or E-A claims, a TOST on β, the window as a row flag, "two reads plus your
  own series", the sc009 Nd dip, any DGA-only Gd test. One-sided 0.025 per claim; no global claim; three-way decisions.
  Supply-audit kill: if every claim is below its minimum G, no confirmation runs; it is evaluated at the end of Stage 0
  on the frozen supply ledger and again after each answered request (CLAUDE_CODE_PROMPT §4.6). Eligibility for the Gd
  claims (C-GD, Gd-dip transport, A3): Eu, Gd and Tb reported; values below DL / LOQ excluded; aqueous-complexant media
  excluded; frozen mechanism-class tag. C-ECT has its own eligibility (§4.14). Before sealing, state
  β_frozen (B1 refit on all 229 cells; fold β −0.030 to −0.140) and C-GD power at it (β −0.118: 0.48 at G = 6, 0.71 at
  G = 8; β −0.072: 0.89 / 0.99), with per-family planning amplitudes from primary SF tables, never from the corpus
  (sealed as PENDING-U5 if the web permission has not been given by then; CLAUDE_CODE_PROMPT §4.3 item 10).
  Chord hygiene: V2-5 censoring and a rule for D reported to one decimal (±0.05 log). New values are sealed by a
  separate agent before modellers see them.
- **Expected effect.** About 0.3 that at least one claim is decided; about 0.9 that the sealing protects round 5.
- **Practical value.** It tells the user what to collect: full La–Lu series (at least Eu, Gd, Tb), at least 3 non-T0
  conditions per extractant, from independent laboratories.

#### 4.5 M24 and lab-3: closures sealed in Stage 0

- **M24, floor-aware predictors.** The clamp curve peaks at 0.471 (F = −4.25), but 91 % of that gain is one cell,
  c950dc2982e96b83; without it +0.0012 to +0.0018 (2 of 5 seeds). The physically consistent all-reads clamp is
  negative at every F, and under consistent interval scoring the "knife-edge" disappears (IR M24). **Line:** retired as
  model claims on V1. **Reopen** only with per-source limits from M25 and a gain ≥ +0.005 R² in ≥ 4 of 5 seeds over ≥ 3
  cells with c950dc excluded.
- **lab-3, foreign per-metal priors (E-A only).** Table §3.1. Same-chemotype priors untestable; M07 carved out (the lab-3
  estimator reproduces M07's lead, Pr/Nd 0.26 → 0.54). Grade FIRM on harm, not on the reopen bar (I2 lab-3).
- **Practical value.** Do not "correct" adjacent SFs with whole per-metal patterns from other publications; never mix
  clamped values with raw sub-limit reads; report "< DL" with the per-metal limit.

### Tier 2: descriptive deliverables (Stage 1; never decision slots)

#### 4.6 M03: sibling weighting factorial (annex to M01)

- **Hypothesis / evidence.** Acid-distance or read-matched weighting beats the plain shrunk mean. Kernel gain over plain
  EB +0.038, but +0.007 without C5-BTBP (below its own +0.01 kill); 73 % of the level-rule gain is C5-BTBP; read-pattern
  matching (0.619) does as well as level matching (0.622); the level rule gains only at n_sib ≥ 5 (IR M03).
- **Test.** Plain EB frozen first; {uniform, acid kernel, read-match kernel, both} × count shrinkage, nested LOPO with a
  one-SE rule toward uniform; selection-aware permutation nulls; the 2 condition-nearest siblings, AINT and PAIRMEAN as
  competitors (I2 shift-2, recipe-4); C5-BTBP excluded from training and scoring.
- **Kill / expected.** Deploy the uniform shrunk mean if the nested gain is < +0.01 without pub_6a0ef8528c and
  pub_19783c039b, or < 7 of the remaining 11 publications improve; that is the likely outcome.

#### 4.7 lab-1: E-LAB, calibrating a new ligand from its campaign's other series (annex to M01)

- **Hypothesis.** A new amide O-donor extractant read at 4 metals, in a campaign that reports other extractants'
  series, is better predicted by mixing in their level-matched detrended deviations than by B1 alone.
- **Evidence (seen; best of about 25 arms).** 92 cells: B1 0.444 → 0.494, +9.0 % [+0.3, +17.6]; new extractants (60
  cells) 0.401 → 0.467, +11.1 %. Against a count- and chemotype-matched placebo from other campaigns +10.2 % [+1.6,
  +18.9]. But random own-campaign donors already give +7.3 % (level-driven weighting), the top 3 publications carry
  93 %, and leaving two out leaves +2.6 % (I2 lab-1).
- **Mechanism not identified (I2 lab-1).** Level-driven weighting carries most of the edge: random own-campaign donors
  give +7.3 %, and the specific level-matched donors add only +1.8 % [−0.5, +4.2]. A shared analytical error of the form
  ε(1 + 1/D) explains the low-D sharing (r 0.53 at log D < −2) as well as homologue chemistry does, and 97.7 % of the
  "other-chemotype" donors share the target's ligand class, so campaign and chemistry cannot be separated in the corpus.
  With near-copies removed by the label-free 0.15 rule the gain is +7.6 % [−1.1, +16.4] (Skeptic B).
- **Test / kill.** Near-copy donors excluded by a frozen label-free rule; placebos E_PBC / E_PCM; mechanism controls
  (random own donors, global-K shrinkage, a low-D band split); laboratory units. External kill only with ≥ 8
  laboratories (false-kill rate 0.41–0.52 at 8 units, stated beside the rule).
- **Practical value.** Interpretive: a per-metal anomaly shared by every ligand in one paper is a campaign signature
  until another campaign reproduces it.

#### 4.8 M18 + M17: metal ladder with the k = 4 rung, and the read-policy envelope (E-D)

- **Evidence (seen; P3 with PC corrections; BPG, identical queries).**

  | row | k = 3 (4 reads): E-A rung (reference; same queries) | k = 4 (5 reads) B1, DOPT_Q-nested | seeds ≥ 0.50 | k = 4 without C5-BTBP (provenance sensitivity) |
  |---|---|---|---|---|
  | C3 min(4, k_max) population row | 0.452 | **0.497** (BP 0.499) | 3/5 | 0.531 (5/5) |
  | C4 fixed cohort (178 cells that admit a 5th read) | 0.429 | **0.480** (BP 0.482) | 0/5 | 0.517 (4/5) |

  The k = 3 column is the ladder's first rung on identical queries, shown only for the paired contrast; the 5-read values
  are a different read budget and are never E-A values. The paired gain of the fifth read is +0.045 [+0.038, +0.070] on
  C3 (BPG, 5/5; P3 §2). The C3 row is a mixture: 15.9 % of its rows keep 4 reads and score 0.60. The without-C5-BTBP
  column removes the publication chosen for its SSE share, an outcome-dependent choice that also removes about 18 % of
  SST; it is a provenance sensitivity, not a budget effect (PC P3). C5-BTBP is uniquely decisive for C4, not for C3 (5
  of 39 single-publication removals cross 0.50). Label-free 5th-read policies span 0.4945–0.5036, an envelope of about
  0.009 (implied 80 % MDE about 0.020 for BISECT_R − DOPT_Q); refitting β at k = 4 moves R² by ≤ 0.001 (PC P3). At
  k = 3, BISECT trades blocks: Yb/Lu 0.002 → 0.488, Ce/Pr 0.419 → 0.274 (IR M17). Six reads give 0.530 in E3's
  exploratory HIST_D7 / max-min-twice / no-look-ahead calculation on the C3 population row (EB §6.1; E3 §6), not on the
  registered B1 DOPT_Q-nested ladder; the fixed C6 cohort gives 0.524–0.531 at six reads (IR M18; E6 §2).
- **Test.** C3 population row and fixed cohorts C4 / C5 / C6 on identical queries; β refit per fold and k, plus β = 0;
  series-length strata; MSE and relative SSE beside R²; P(R² ≥ 0.50) with LOCO ranges, with and without C5-BTBP; M17's
  frozen policy family as labelled rows with per-block ΔSSE. Density invariance is already seen (k_max 4 / 5 / 6 at
  0.444 / 0.459 / 0.484): a consistency check, not a kill (PC P3).
- **Practical value.** The honest "how many metals" answer (a different read budget, not E-A): about 0.48–0.50 at 5
  metals; about 0.53 at 6 (0.530 is E3's exploratory C3-row calculation, 0.524–0.531 the fixed C6 cohort). Spread the
  reads; at 5 reads placement chooses which block is predicted well; read an end (Lu or Tm) only if the heavy pairs
  matter; every other metal of a short series (k_max 4) still scores only 0.444 (P3 §5, §7c).

#### 4.9 M09: level window as a tail-risk flag

- **Hypothesis (narrowed).** Rows whose B1-predicted log D lies in [−2, 2] for both metals rarely carry gross errors; a
  cell- or condition-level risk flag, not a way to claim R² ≥ 0.50.
- **Evidence (seen).** 0.540 inside at 74.7 % coverage vs unstratified random abstention 0.453, but a cell-matched
  control gives 0.510; the excess +0.029 has a cluster bootstrap [−0.069, +0.055] and vanishes without the top 10 cells
  or pub_126bbe7976. P(|res| > 0.5) is 0.009 inside vs 0.043 outside; without C5-BTBP the median |res| is the same
  inside and outside (IR M09).
- **Test / kill.** A frozen coverage–MSE curve over half-widths {1, 1.5, 2, 2.5, 3, ∞} and one-sided variants;
  label-free operating points at 75 % and 90 % coverage; cell-matched control primary; M39's family-nested optimism
  table (nested 0.516 at 78.7 %). The tail flag passes if the outside/inside ratio of P(|res| > 0.5) is ≥ 3 with a CI
  excluding 1, with and without C5-BTBP; the bulk-precision claim (≥ 15 % narrower intervals) is expected to fail.

#### 4.10 M14: selective direction calls (E-DEC), with M16's interval line and recipe-3's own-prior row

- **Evidence (seen).** B1 calls 0.507–0.606 of pairs at α 0.10 depending on the procedure (0.149 for the strictly valid
  fixed-sequence one); B1 − B0 +0.047 to +0.058; false sign Eu/Gd 0.18, Tm/Yb 0.29, Yb/Lu 0.62; 57 % of wrong calls have
  |y| < 0.1 (IR M14).
- **Test.** Risk–coverage curves for B0, B1, HBK@ML; a three-way lighter / heavier / inseparable rule (false sign
  counted at |y| ≥ 0.10); valid calibration (Learn-then-Test or CRC) beside the plug-in; no calls outside the read hull;
  the own-prior row (COPY_own, SH_own@2, SH_own@4) at matched coverage. Never presented as an R² gain.

#### 4.11 M07: the sc009 (TODGA-class) Nd dip, frozen at −0.11 (E-KX; Pr/Nd warning)

- **Evidence (seen).** sc009 Pr/Nd R² 0.265 with B1, 0.419 with a universal Nd term, 0.522 with the chemotype term (46
  rows per seed); Nd chord negative in 11 of 12 sc009 publications (about −0.11); B1 implies a Nd/Pr SF of about 1.7
  where about 1.34 is observed. The top 3 publications carry 75 % of the gain, the anomaly depends on acid (ρ −0.28 to
  −0.36), and the pooled effect is about +0.002 (IR M07).
- **Test.** Freeze u_Nd(sc009) = −0.11 now; E-KX-LOPO rows, never called BP; comparators: universal Nd term, a Pr/Nd
  chemotype offset, a copied SF. Excluded from the confirmation package (§4.4), so new TODGA-class papers with Pr, Nd and
  Sm form a case series; the lead dies if the Nd chord is ≥ 0 in at least 2 of the first 4, or if Pr/Nd squared error
  against B1 does not fall in most of them (IR M07). The frozen value is the raw mean anomaly; its shrinkage is
  n/(n + 5) toward 0 (critiques.json M07 test design; CLAUDE_CODE_PROMPT §6.8).

#### 4.12 M13: wider-pair table (E-W)

- **Evidence (seen).** B1 on ΔZ = 2 pairs 0.676, Nd/Sm 0.765, ΔZ = 3–4 0.811; much of that is the larger denominator
  (var(y) 2.2×, MSE 1.30×); skill against a line is 0.28 on strict pairs and 0.26 on Nd/Sm; heavy ΔZ = 2 cuts are near 0
  (Ho/Tm 0.16, Er/Yb 0.07, Tm/Lu 0.10) (IR M13).
- **Test.** One "seen" row per cut pair; one open contrast (Nd/Sm B1 − LINE outside the giant cluster); feed-defined
  pairs (Nd/Dy, Pr/Dy, Nd/Tb, Eu/Tb); never pooled with strict pairs.

### Data-gated (Stage 2; requests to the user)

#### 4.13 M25: stakes-first source recovery (U1) → V3 sensitivities (J 5 · 5 · 3.5)

- **Evidence (P5 with PC corrections).** The 69 no-location cells hold 56.4 % of SSE and 68.4 % of the gain (§2.5).
  C5-BTBP: 23.1 % / 34.9 %, and all 44 C3 readings below log D = −4 are C5-BTBP. DMDPhPDA reduces to one row (5 M Pr/Nd,
  1.8 % of SSE). ORNL rounding is only 2.2 % of SSE; the ORNL channel is itself clipped at log D −4.0. New class:
  component identity (pub_84113c17ad records CDTA / HEDTA as the extractant of a TEHDGA + HDEHP system).
- **Effects (V3 sensitivities with V1 read sets frozen; row sets as stated; never edits to V1, never E-A results).** C5
  readings at a floor (rows touching a censored reading dropped): B1 0.4950 on 1,760 of 1,802 rows per seed (B1 − B0
  +0.035); interval-scored upper bound 0.5244 on 1,780; source unverifiable: 0.4863 on the remaining 1,654 (B1 − B0
  +0.027); verified as real: no change; combined floor + DMDPhPDA Figure 5: 0.5006 on 1,760. Each of these is a different
  row set and not comparable with 0.452. Only DMDPhPDA uses the identical 1,802 rows: 0.4565 if Figure 5 holds, 0.4478 if
  Table 1 holds (P5 §4a; PC P5). Attribution alone (Nilsson 2006 vs Foreman 2005) changes no out-of-fold number (P5 Bottom line, §2 F5).
- **Test / decision.** A seeded label-free census of the 22 import DOIs → V3-R; value-informed corrections → V3-HS (an
  upper bound only); error classes in target currency (gross = |Δ adjacent log SF| > 0.3); blind auditors. "Anchored" if
  the V3 gain on resolved documents is ≥ 50 % of their V1 gain; an "unverifiable" stratum if reachable documents cover
  < 40 % of the gain. Provenance of pub_19783c039b and pub_b0b3296ec5 first (they carry the E-CT leads; I2).

#### 4.14 confirm-4 (round 2) with M05: C-ECT, external condition transfer (seal in Stage 0; first case via U3)

- **Hypothesis.** In a new laboratory × extractant family with ≥ 3 full-series conditions, for a target that differs by
  a change that alters the SF pattern (T1 aqueous medium, T2 organic phase, T3 mixed), SH with 4 reads beats B1 with 4
  reads; secondary, SH with 3 reads. Not claimed: SH with 2 reads, or "extractant transfer" as distinct from "campaign
  fingerprint".
- **Evidence (seen).** 13 W / 1 L corpus units, mean log SSE ratio +0.66; exact power for SH@4 0.89–0.93 at G = 7, but
  in a DGA world; non-DGA CLEAN units give 4 W / 1 L, mean +0.14 (I2 confirm-4).
- **Sealed design.** Laboratory unit (families nested, one weight per laboratory); per-family statistic
  log(Σ SSE_B1@4 / Σ SSE_SH) over leave-one-condition-out targets; T0 at pair level (pH, [HA] or phase-ratio scans of an
  ideal cation exchanger are near-copies by mass action, only a data check); fixed sequence SH@4 then SH@3 at one-sided
  0.025 each; SH − OFF_XEXT as the mechanism contrast; power in three worlds (corpus DGA 0.90–0.92 at G = 7; halved
  effect 0.07–0.30 at G = 6–10; cation-exchanger or null-like 0.01–0.58 at G = 6–7).
- **Kill (laboratory level).** Confirm at exact one-sided p ≤ 0.025 with ≥ 7 laboratories; kill if the mean log ratio
  is ≤ 0 or the win share < 0.5; otherwise inconclusive; fewer than 7 laboratories → a descriptive case series.
- **Supply and expectation.** About 4 laboratories with ≥ 3 full-series conditions were reported by P4 (KU Leuven,
  Chiba, Turanov, Atanassova); 3 after PC P4 moves Turanov (P4-063) to E2; full La–Lu is verified only for KU Leuven and
  Chiba (Atanassova U; Chiba abstract-only and pH-controlled, so T0) (P4 §4, §5; PC P4). After T0 classification at most
  2–3 non-T0 verified families remain against a minimum of 7 (I2 confirm-4). First case (U3): Molecules 29:3548, Table S3
  (8 diluents), a weak case because B1 already scores 0.924 on the corpus's diluent-only cells (P1 §10). Its laboratory
  (MSU, Nenajdenko group) is already in the corpus (P4-001), so it is a dry run of the sealed pipeline and does not count
  toward C-ECT's 7 new laboratories. About 0.75–0.8 that own-series
  information beats B1@4 in new non-T0 families; 0.15–0.25 for a decisive pass.

#### 4.15 confirm-2 / confirm-3 (round 2): C-GD and one chord interval, inside M33's Gd-dip transport test

- **Gating.** The Gd-dip transport test counts units on the merged SAFE + literature ledger and runs only with ≥ 10
  laboratory × chemotype units (IR M33). C-GD is its pre-specified stratum (I2 confirm-2) and is decided only when that
  stratum has ≥ 6 laboratories from ≥ 3 families. Below the ≥ 10 trigger, C-GD is a descriptive case series.
- **Hypothesis.** In new laboratories using cation-exchange extractants (acidic organophosphorus, carboxylic acids,
  chelating exchangers; frozen mechanism metadata), the laboratory-mean Gd chord is deeper than β_frozen (the all-229
  B1 refit, computed before sealing; round-4 fold mean −0.094), so B1 under-corrects them.
- **Evidence (seen).** Winsorised chord −0.217 (sd 0.101, 12 units) in the slope-defined heavy-selective non-DGA
  stratum, −0.073 light-selective, −0.051 DGA; but 11 of those 12 units are N-donors or CMPO-type, and the bench holds
  one acidic cation-exchange cell with Eu, Gd and Tb. Outside C5-BTBP and pub_84113c17ad a stratum amplitude cuts
  heavy-selective SSE by about 4 %, about +0.002 R² pooled (I2 confirm-1, -2, reconciliation check).
- **Sealed design.** One interval of the laboratory-mean chord (winsorised to [−0.40, +0.20] as primary, raw as a
  sensitivity) read against 0, β_frozen/2 and β_frozen: upper bound
  < β_frozen with ≥ 6 laboratories and ≥ 3 families confirms C-GD; lower bound > β_frozen kills it; lower bound
  > β_frozen/2 raises the B1-harm flag and switches new chemotypes to B0 (a weak rule: it fires 0.17 at G = 8 and 0.25 at
  G = 13 when B1 truly harms); otherwise inconclusive and B1 stays. The DOPT_Q Gd-read rate of the frozen read policy is
  frozen as part of the claim (I2 confirm-3). A1 is descriptive; A3 (B1 − B0 on new data) runs alone as the prospective
  positive control. Power from chemistry worlds: HDEHP-like (about −0.26, unverified) 0.90–1.00 at G = 6; P507-like
  (about β) power ≈ size. The size gate run on clustered laboratory nulls decides whether the laboratory t-interval
  survives (CLAUDE_CODE_PROMPT §4.3 item 2).
- **Expected.** About 0.85 that acidic-OP laboratories show a deeper chord if they arrive; 0.25–0.35 for a decisive
  pass (realistic G 2–5 per I2 confirm-2, which counts Osaka; verified independent HDEHP units 1–2, abstract-level only:
  Chiba HDEHP verified, Osaka HDEHP verified only if not institution-linked, and P4-083 reports single-condition
  extraction constants, not D; PC P4).

#### 4.16 M33: SAFE gap-ID audit (U2), the label-blind supply gate

- **Evidence.** Archive record IDs run 1–18,372 with 1,602 missing in 11 runs; the "+1,618 newer records" mostly
  pre-date the archive's last batch; about 16–21 are newer (IR M33).
- **Test / kill.** The user exports only the gap IDs and IDs above 18,372; a metadata-only script (D columns asserted
  absent) classifies record type, DOI novelty, Ln coverage, chemotype, laboratory and tags (Y, rigid SMARTS, non-DGA).
  SAFE units are merged with the M22 literature count into one supply ledger before the gate is read. Scored use:
  the Gd-dip transport test with one value per laboratory (I2 rule 12). IR M33 wrote it as a t-interval against 0 and
  against −0.094 (the round-4 fold mean). Round 5 reads the laboratory-mean chord interval against β_frozen and
  β_frozen/2 with C-GD's three-way rule, a stated change to IR M33; the reading against 0 is A1 and descriptive, and the
  round-4 fold mean −0.094 is shown beside β_frozen as a reference (I2 confirm-3; CLAUDE_CODE_PROMPT §7.5). The non-DGA
  interval decides; the DGA interval is descriptive, because the package excludes any DGA-only Gd test (§4.4). C-GD is
  the pre-specified cation-exchanger stratum (§4.15). < 3 experimental multi-Ln units from outside laboratories in SAFE
  stops the SAFE route; 3–9 on the merged ledger is descriptive; ≥ 10 lets the Gd-dip test run, and it decides only with
  ≥ 10 laboratories after one-value-per-laboratory aggregation. Expected from SAFE: 0–3 units.

#### 4.17 M22 / P4: the literature supply ledger

- **Evidence (P4 with PC corrections).** 92 candidates screened label-blind. Verified new independent laboratories: 7
  reported, 6 strictly, 5 with the institution-link rule applied consistently. Probable (E1 + E2): 16 reported, 11–14
  corrected. Expected yield: 13.5 by P4's rule, 9.3–11.5 without the unmeasured U-tier conversion, 7.0 with every
  correction. 15 of the 16 probable new laboratories use chemotypes absent from the corpus (acidic organophosphorus,
  quaternary-onium ionic liquids, chelators); about 5–7 work in C3's regime; 3 have ≥ 8 Ln including Eu, Gd, Tb (2 if
  Osaka is P; PC P4).
- **Consequence.** Literature cannot carry a sealed-rule confirmation of a B1-sized gain: the sealed R² leg needs about
  38–41 publication-like units (P2 grid 39 BPG / 38.3 BP; PC P2 direct evaluation ≈ 41) or about 20 cpg units (PC P2,
  PC P4). It supports a descriptive, mostly non-DGA transfer study and the C-ECT / C-GD cases.
- **Next.** Full-text verification of E2 papers; the frozen gen6 chemotype map; re-aim the search at E-CT-capable
  laboratories (I2 confirm-1); figure digitisation (Phase B) only after acquisition, with values sealed separately.

#### 4.18 M11 + M12: label-blind re-measurement and inter-laboratory reproducibility (U4)

- **Design.** Re-measure 8–12 C3 cells at their published conditions (label-blind, stratified by chemotype and
  read-predicted window status), 3 independent runs plus an analytical duplicate; in-window, high and floor arms;
  per-metal LOQ, spike recovery, interference corrections. Optional ≥ 8-laboratory own-protocol arm (estimand V_x/V_w);
  optional one fixed-condition TODGA ring-standard tube per laboratory and an in-run TODGA reference beside any non-DGA
  ligand, kept for the M12 ring test and the M31 family-dip salvage, not for prediction (any second own-campaign ligand
  carries the same signal; I2 lab-2 closure); Y recorded (IR M11, M12).
- **Decision.** Corpus label error below σ_SF ≈ 0.157–0.19 with B1 unchanged against replicate means → a modelling or
  curation limit; repeatability lower bound at or above it → noise. Laboratory signature if V_x/V_w ≥ 5 with lower bound
  ≥ 2; none if the upper bound ≤ 2.
- **Lead now (n = 1 laboratory pair, chosen after looking).** ORNL 2020 vs Ames/Purdue 2024, 3 DGAs: cross-laboratory
  mean squared difference 0.027–0.029, about B1's MSE, concentrated in La/Ce, Ce/Pr, Pr/Nd (0.0043 without them).
- **Expected.** Repeatability σ_SF about 0.02–0.09; corpus label error larger, at the floor and the light end.

#### 4.19 shift-3 (round 2): Ce/Pr acid sign in DGA-class extractants (rider)

The Ce/Pr acid slope is negative in 11 of 11 publications (mean −0.32 per decade) and 6 of 6 laboratories (p 0.016),
partly level fanning, one pair chosen post hoc out of 13 (I2 shift-3). For every new DGA-class mineral-acid series
collected for other reasons, record the frozen Ce/Pr and La/Pr slope signs, the level-conditioned residual and the
laboratory; decide at laboratory level only with ≥ 8 eligible laboratories; never a size claim.

### Rejected (fatal flaw; the fragment kept goes to the named host)

| id | claim | fatal flaw | kept → host |
|---|---|---|---|
| M29 | Gd term off in complexant media | one paper carries 78.5 % | data-contract rule (§3.2) |
| M38 | pre-slot decidability gate | uses the same labels; retires B1 − B0 itself | retire-only gate → M35 |
| M31 | non-DGA Gd-triad acquisition | kill fires 55–60 % when true; mixed stratum | family arm → M11; tags → M33 |
| M06 | S0–S4 value-of-information ladder | S1/S2 kill cannot fire (one cluster) | annex → M01 |
| M27 | Ce reliability first | ICC gate unpowered; Ce error is mostly floor | floor → M25; ledger |
| M39 | nested re-estimation of post hoc leads | "chosen in ≥ 50 % of folds" measures flatness | optimism table → M09 |
| M02 | 2 reads + sibling, shift-plus-tilt | C5-BTBP carries 73 %; mechanism refuted | ladder row → M01 |
| M04 | choose conditions, not metals | gain vs IQR-random CI includes 0 | one line → M01 |
| M08 | variance budget and futility | not identified (one cluster 97.8 %) | moment table → M35 |
| M15 | screening = amplitude (LINE control) | post hoc thresholds; contradicted | E-RANK table |
| M36 | publication vs cluster units | publication flips anti-conservative (0.07–0.18) | aggregation rule → M34 |
| M16 | publication-balanced conformal | width inflation | interval line → M14 |
| M23 | E-CENS interval scoring | = C5-BTBP exclusion; gameable | sensitivity line |
| M26 | nested data-version ladder | premise wrong; flag fires on noise | ledger (§3.2) |
| M37 | sequential e-process ledger | supply is a stock of 3–7; power 0 at G ≤ 6 | ledger → M34 behind M33 |
| M10 | measure in the window | tail below −3 is one publication | flags → M09; low-D arm → M11 |
| M19 | allocate the 5th read between cells | +0.0019 vs a +0.002 kill; ties random | one line |
| M20 | unmeasurable-end (Pm) scenario | own kill fires; does not transfer | G8 scenario note |
| M30 | yttrium as a 4f-free probe | one TODGA dataset; power 0.01–0.06 | "record Y" rider |
| M32 | rigid / preorganised stress set | the penalty is one floor cell | SMARTS tag → M33 |
| M21 | provenance class as a lever | own kill met (p 0.12) | column → M35 |
| M28 | B1+ factorial (Ce-acid × edge) | Ce-acid is two publications | B1 + edge reference variant |
| shift-1 | acid-distance rule for zero-read copying | one publication; "distance" is range position | E-CT row; ledger |
| recipe-1 | reads-equivalent credit of an own series | one publication; weakest comparator | E-REC table → M01 |
| recipe-3 | sign-first triage from an own series | comparators on different rows | own-prior row → M14 |
| recipe-2 | fixed training-free metal ladder | false on its own data; wrong for acidic OP | static row → M18 |
| recipe-4 | ICP-MS condition skipping with an acid trend | near-copy leakage; ties PAIRMEAN | M03 arms; ledger |
| lab-4 | campaign as the confirmation unit | wrong quantity; the Gd chord is lab-shared | laboratory unit → M34 |
| lab-2 | in-run TODGA reference | any second ligand does as well | arm → M11 / M12 |
| shift-2 | bracketed per-metal slope model | the frozen arm does nothing; ties AINT | M03 arms; ledger |

Sources: IR "Rejected"; I2 "Rejected ideas".

---

## 5. Practical guidance for chemists now

Every number here is in-corpus and seen. The corpus is mostly DGA (157 of 229 C3 cells) and one source cluster holds
65 % of rows (EB §3, §6.4), so the numbers fit DGA-type extractants in nitrate media best.

**The cost frame.** With ICP-MS all 14 lanthanides come from one extraction; the cost unit is the experiment
(extractant × condition). Metal-count advice matters for radiotracers, subset feeds and literature mining (EB §6.1;
I2 rule 16). For an ICP-MS laboratory the question is usually "which conditions must I run?".

### 5.1 A new extractant, nothing measured

Adjacent log SF cannot be predicted from structure (R² ≈ 0.05). Zero-shot direction (0.688) barely beats "the heavier
lanthanide extracts more" (0.666), and zero-shot candidate ranking is at chance (EB §2.1, §2.12). Run one experiment.

### 5.2 A new extractant, 4 metals measured (E-A; use B1)

- **Predictor.** Piecewise-linear interpolation of log D in CN-8 ionic radius, constant beyond the extreme reads, plus a
  Gd dip of about −0.09 (the fixed −0.072 works equally well) (R4 PROTOCOL §4, REPORT §4). Nothing tested beats it.
- **How good.** Pooled R² 0.452; MAE about 0.10 log; typical-row residual sd 0.086 but heavy tails (RMSE 0.164).
  Direction right for 77.5 % of pairs, 87 % when |log SF| ≥ 0.1, and 94 % when the prediction itself is ≥ 0.2 log
  (24 % of pairs) (EB §1.2, §2.12).
- **Intervals.** Pair-type conformal half-width 0.152 log at 80 % (SF ×/÷ 1.42) and 0.245 at 90 % (×/÷ 1.76), about 5
  points short of nominal outside the dominant cluster (E6 §6.6). For a new source use about 95 % nominal to get about
  90 % (roughly ×/÷ 2.3 in SF) (IR M14).
- **Which pairs to trust** (B1 R², E6 §6.1): Sm/Eu … Dy/Ho 0.504 (46 % of rows; Gd/Tb 0.55); La/Ce 0.35, Ce/Pr 0.42,
  Pr/Nd 0.42 (the light end carries 43 % of the error); Ho/Er and Er/Tm 0.26; Tm/Yb 0.03 and Yb/Lu 0.00, no skill unless
  Tm or Lu is read. Group splits (ΔZ ≥ 2) show higher R² from the same reads (Nd/Sm 0.765; ΔZ = 3–4 0.811), but much of
  that comes from their 2.2× larger variance: their MSE is 1.30× that of strict pairs, skill against a straight line is
  about the same (0.26 vs 0.28), and heavy cuts are near 0 (Ho/Tm 0.16, Er/Yb 0.07, Tm/Lu 0.10) (IR M13). These R²
  values are not comparable with strict-pair R² (E-W).
- **Risk flag.** If both predicted D lie between 0.01 and 100, about 1 pair in 110 is off by more than 0.5 log; outside
  that range about 1 in 23. Re-measure, or move D into range, before trusting light-pair SFs there (IR M09). The flag
  is partly a C5-BTBP effect and does not promise precision.

### 5.3 If you can choose which metals to measure (radiotracers, subset feeds)

- **Add a metal before you replicate one**: replicating a read pays only if its per-metal σ exceeds about 0.2 log (IR M11).
- **Five metals bring adjacent R² to about 0.48–0.50, six to about 0.53** (§4.8; a different read budget from E-A; the
  0.53 is E3's exploratory C3-row calculation and the fixed C6 cohort's 0.524–0.531). Spread the reads; at 5 reads
  placement chooses which block is predicted well; read an end (Lu or Tm) only if the heavy pairs matter; every other
  metal of a short series (k_max 4) still scores only 0.444 (P3 §5, §7c).
- **Placement chooses which pairs are good.** Reading Lu lifts Yb/Lu (0.002 → 0.488 under BISECT) but costs Ce/Pr
  (0.419 → 0.274) (IR M17). If one pair matters and one of its ends cannot be measured, read the other end and its
  next-but-one neighbour: two targeted reads give per-pair R² 0.34–0.87 (Yb/Lu −0.03 → 0.63) (E6 §6.2).

### 5.4 You already have your own series of this extractant at other conditions (E-CT; separate estimand)

- **Predictor.** Average your earlier series of the same extractant, align the mean to 3–4 reads at the new condition,
  and shrink it toward B1 with weight n/(n + 1.78) for n earlier series (P1 §2).
- **How good (seen; 99 cells, 94 of them DGA, mostly one source cluster).** On the same rows SH with 4 reads scores
  0.708 against B1's 0.531 (+0.168 with the suspected duplicate pair dropped; PC P1); SH with 3 reads still beats B1
  with 4 by +0.148; with one earlier series SH gives +12.2 % relative SSE against B1 (P1 §1, §6, §8). These values
  belong to their own subset and are not yet confirmed outside the corpus (§4.14). "Earlier series" here means the same
  publication (study or campaign); the corpus has no laboratory key (IR rule 8).
- **Rules.** Read at least 2–3 metals at the new condition: copying at 0 reads is below B1 with 4 reads, and copying down
  to an acidity below your measured range is worst (I2 shift-1). Shrink, do not copy: one shrunk series already beats B1
  (+12.2 % relative SSE); one raw series is worse (0.510 vs 0.547); averaging more series helps the raw template most
  (P1 §7–8). Series of the same extractant from other publications (not yet other laboratories; the corpus has no
  laboratory key) do not substitute: −2.6 % against B1 without C5-BTBP (53 cells, 6 publications), but +10.8 % with it,
  because any BTBP-type template repairs part of C5-BTBP's residuals (P1 §11; IR rule 8). Bracket the operating acidity
  with measured series, especially from below (I2 shift-2).
- **pH or [HA] scans of an ideal cation exchanger.** log SF does not depend on pH, so the SFs carry over by mass
  action; this fails in buffered or complexant media (TALSPEAK-type), at high loading and at high acidity (I2 confirm-4).

### 5.5 Chemistry-specific warnings

- **TODGA-class DGAs, Pr/Nd.** Lead (seen; the top 3 publications carry 75 %; acid-dependent; expected to narrow to a
  universal Nd term; IR M07): interpolation overstates Pr/Nd selectivity (B1 implies a Nd/Pr SF of about 1.7 where
  about 1.34 is observed). Borrow the Nd dip (about −0.11), not another paper's SF; do not borrow Gd or Ce values from
  other laboratories (IR M07). Under ICP-MS, measure Nd.
- **Cation exchangers (HDEHP-type, chelators).** A Gd break is textbook for HDEHP (Peppard); its planning depth
  (≈ −0.26) is unverified. For P507 an unverified reading puts the break near B1's β (≈ −0.105). The corpus chord −0.217
  comes from a mostly N-donor / CMPO-type stratum (11 of 12 units) and does not describe cation exchangers; the corpus's
  one acidic cation-exchange cell has chord −0.37. The skeptics' prior that acidic-OP laboratories show a deeper chord
  is about 0.85 (I2 confirm-1, confirm-2). Read Gd, or measure the Eu–Gd–Tb triad; for established P507 / HDEHP circuits
  keep using tabulated SFs. HYPOTHESIS under C-GD.
- **Aqueous complexants (DTPA, lactate, HEDTA, CDTA).** Do not apply the Gd term; treat these cells as their own stratum
  (IR M29).
- **DGA-class extractants, Ce/Pr.** Lead: the acid-slope sign is negative in 6 of 6 laboratories (11 of 11 publications,
  5 from JAEA; one pair chosen post hoc out of 13). Level fanning explains part of it, the mechanism is undetermined, and
  the claim is the sign only, with no size claim (I2 shift-3). Check at the operating D.
- **Light end across laboratories.** Another laboratory's light-end adjacent SF may be good only to about 0.15 log
  (one laboratory pair; IR M12).

### 5.6 What to report when you publish a series

Full La–Lu series (at least Eu, Gd and Tb; Y if present); per-metal detection limits with "< DL" values flagged;
contact time and phase ratio; complexant identity and concentration; replicate whole series rather than replicate
readings (IR M24, M29; R4 DATA_REQUEST §2). Keep every series of a multi-ligand paper: together they carry the campaign
information (I2 lab-1).

---

## 6. Data acquisition that could change the result

The main requests are below. The complete ranked list for the user (U1-a … U1-i, U2–U6, S1, H1), with sub-requests,
effort and the smaller items, is `DATA_REQUESTS.md` in this folder.

| request | user effort | what it decides | stake / expected outcome | source |
|---|---|---|---|---|
| **U1** highest-stake document package: say whether you have journal access; allow one download (Shimada 2004 thesis, 23.8 MB, into a quarantine folder); look up Foreman 2005 / Nilsson 2006 (Experimental section, detection limit, contact time, the c950dc Ce value) | about 35–50 min | whether B1's Gd term and the C5-BTBP floor rest on verified values → V3 sensitivities | about 26 % of B1's SSE and 44 % of its gain (≈ 20 % / 33 % probability-weighted; P5's 28 % / 52 % double-counted a duplicate); without access, thesis only: 4.4 % / 17.6 % | P5 §6; PC P5 |
| three-minute package: access answer + thesis + reading 6 open-access import pages (no downloads) | about 3 min | the cheapest stake per minute | about 14 % SSE / 28 % gain; this counts DMDPhPDA copy B's 8.8 % of the gain, which the V2-1 quarantine settles without a document; the document-dependent part is about 12 % SSE / 19 % gain (P5 §3) | PC P5; P5 §3 |
| U1-f: one e-mail to the SAFE / LANL curator (user sends) | 15 min | all 69 unlocated cells | 56.4 % / 68.4 % if answered; P(reply) ≈ 0.3 | P5 §6 |
| U1-d: the open JACS Au 2022 SI (one download; match unverified) | 1 min | ORNL-54 only: the cited source and original log D, not the table or figure location; not independent of the import; void if the values do not match | 33.3 % / 33.5 % at most; unscored in gain-% per minute | P5 §3; PC P5 |
| **U2** SAFE export of the 1,602 gap IDs and IDs above 18,372 (metadata only; check SAFE's terms) | minutes | the M33 supply gate | expected 0–3 new units; < 3 closes the SAFE route | IR M33 |
| **U3** permission to fetch the SI of Molecules 29:3548 (Table S3, 8 diluents) | minutes | a dry run of the sealed E-CT pipeline: the laboratory (MSU, Nenajdenko group) is already in the corpus (P4-001) and does not count toward C-ECT's 7 | descriptive; a weak test (diluent-only) | IR M05; P4-001 |
| **U4** a laboratory collaborator: 8–12 C3 conditions in triplicate with in-window, high-D and floor arms; optional ≥ 8-laboratory arm | months | measurement- vs model-limited (σ_SF threshold 0.157–0.19) | repeatability σ_SF about 0.02–0.09 expected | IR M11, M12 |
| **U5** web reading for the literature supply ledger (M22 Phase A, E2 verification, a targeted E-CT search) and for the C-GD planning amplitudes | 1 min permission; agent days | each claim's achievable G; the planning amplitudes (a Stage-0 input) | 5–6 verified, about 7.0–11.5 expected new laboratories (13.5 under P4's rule); 15 of 16 off-domain | P4; PC P4 |
| **U6** review of the ambiguous links in the frozen laboratory key | short; not estimated | G for every laboratory-unit claim (a Stage-0 input) | P4's tier rule moves probable new laboratories 16 → 14; the links then give 13 or 11 | PC P4 |
| **S1** sign-off of the V2-3 cohort-key remedy (11 cells) | one decision | split by aqueous complexant, or a separate sensitivity stratum | 6.4 % of B1's SSE, gain share −9.0 % | P5 §3, §5.4 |

**Power of the packages that new data would feed** (corpus resampling where marked, therefore optimistic):

| claim | minimum G | corpus-like world | other worlds (halved effect, chemistry, τ) | realistic supply | source |
|---|---|---|---|---|---|
| C-ECT (SH@4 vs B1@4, laboratory units) | 7 laboratories | 0.90–0.92 at G = 7 (DGA) | halved 0.07–0.30; null-like 0.01–0.58 | 2–3 non-T0 verified families | I2 confirm-4 |
| C-GD (laboratory-mean chord vs β_frozen; stratum of the Gd-dip test) | 6 laboratories, 3 families | slope-defined heavy-selective stratum 0.62–0.74 at G = 6 (analytic 0.68; 0.38 at sd 0.15), but 11 of 12 units are N-donor / CMPO; confirm-1's metadata class 0.13 / 0.23 / 0.28 / 0.33 at G = 4 / 6 / 8 / 10; the corrected cation-exchanger stratum has 1 corpus cell, so no corpus power | HDEHP-like planning world (≈ −0.26, unverified) 0.90–1.00 at G = 6; P507-like ≈ size | G 2–5 (counts Osaka); verified independent units 1–2 | I2 confirm-1, confirm-2; PC P4 |
| pooled B1-sized gain (SF_POOL, publication-like) | — | G80 43 at τ = 0 | > 80 at τ = 0.065 | 5–6 verified; 7.0–11.5 expected | P2 §4; PC P2, P4 |

**What would not help** (EB §2.18, §6.4): more rows from existing publications (97.7 % of archive strict-adjacent pairs
are already in V1; relaxed cohort rules add 0 new source clusters); An, Y or Sc rows as labels; more geometry or
descriptors.

---

## 7. Evaluation and stopping

### 7.1 Estimands, labels and rules for every number

**Estimands.** E-A (primary; four read metals; closed for model work); E-D (read-budget ladder on identical queries);
E-CT and E-LAB (new condition in the same campaign; new extractant in a known campaign; B1 at 4 reads on the same rows
as comparator); E-W, E-DEC (with the E-REC own-prior row), E-RANK, E-KX (descriptive); E-S (equal-weight breadth,
diagnostic only); E-CENS (one sensitivity line); the window axis (M09); V3 (data-version sensitivities with V1 read sets
frozen, on the row sets stated); C-ECT, C-GD and Gd-dip transport (new independent laboratories, frozen package). None
substitutes for another.

**One placement rule.** E-CT, E-LAB, E-W, E-DEC, E-KX, E-RANK and V3 values are never placed next to B1 = 0.452. E-D's
4-read rung is E-A itself; its 5- and 6-read rungs are a different budget, reported as paired contrasts on E-A's
queries and never as E-A values. V3 values on other row sets are never placed next to 0.452.

Labels (R4 PROTOCOL §2): **seen / exploratory** (the 229 cells and 5 dev seeds, including every round-5 number);
**partition stability** (new seeds on the same cells, never blind); **independent confirmation** (data arriving after
the freeze). Round 3's final seeds and strain's fresh444 pairs are consumed.

Every round-5 number is reported with and without C5-BTBP, with leave-one-publication-out and the c950dc influence
(IR rule 1); after a duplicate scan (I2 rule 15); against the cheapest same-information comparator on identical rows,
with placebo weights matching the arm's (IR rule 3; I2 rules 13–14); with label-free strata, windows and near-copy
rules (IR rule 7); with its cost scenario stated (I2 rule 16). In-corpus legs whose thresholds sit just below seen
values are consistency checks, not tests (IR rule 2). "Same publication" is not "same laboratory" until the laboratory
key exists (IR rule 8). A derived descriptor (chord, slope, per-unit coefficient) reports its split-half or jackknife
reliability, with a signal-injection check, before any correlation with it is interpreted.

### 7.2 Decision and inference (sealed in Stage 0)

- **E-A:** M35's single gate (≥ 2 % pooled relative SSE vs B1 by SF_POOL, balanced-MAE guard ≥ −0.001, BPG deciding)
  with its mandatory tables (§4.2). For round 5 it replaces round 4's two-margin, three-interval rule (R4 PROTOCOL §6).
- **Inference:** M34 (§4.3).
- **Confirmation:** confirm-1's package (§4.4) with C-ECT (§4.14) and C-GD inside the Gd-dip transport test (§4.15,
  §4.16; the test runs only with ≥ 10 units on the merged SAFE + literature ledger, and C-GD is decided only with ≥ 6
  laboratories from ≥ 3 families in its stratum);
  one-sided 0.025 per claim; three-way decisions; non-rejection never switches the deployed model; nothing is promoted
  from development data.

### 7.3 Stopping

1. **E-A.** No new model slot unless it passes the retire-only pre-slot decidability gate (M38 salvage; IR M35 item 7):
   its own test-label oracle ceiling ≥ the 2 % margin, power ≥ 50 % at its plausible effect, and demotion if the
   deciding fold has fewer than 10 training cells sharing the test rows' donor class or chemotype. Otherwise E-A stays
   closed on this corpus. A slot that passes is scored under M35's 2 % gate as a diagnostic (M34's kill fired); the
   per-metal-prior route also needs §3.3's lab-3 bar.
2. **0.50-level accuracy** is pursued only through other read budgets (E-D: at 5 reads the like-for-like C4 cohort is
   0.480 and the C3 min(4, k_max) population row, with 15.9 % of rows still at 4 reads, 0.497; PC P3) or new and cleaner
   data (V3, U4), each under its own label. E-A stays at k = 3: 0.50 there is not reached by any tested or proposed model,
   and gains below about 0.01 R² are undetectable on this corpus (EB §3).
3. **Supply-audit kill.** If every sealed claim is below its minimum G after U2 and the literature ledger, no
   confirmation runs and round 5 is reported as descriptive. The audit runs at the end of Stage 0 on the frozen supply
   ledger and again after each answered request; on the current ledger every claim is below its minimum (C-ECT at most
   2–3 against 7; C-GD 2–5 against 6; SAFE 0–3 against 10).
4. **No re-looking.** No new seed batches until something is significant; no relaxed margins after a result; values
   from new sources sealed by a separate agent before modellers see them.
5. **E-CT stops at the corpus.** After Stage 1 its next information comes only from external families (C-ECT); further
   in-corpus variants are not run.

---

## 8. Handoff

- **What this document is.** The evidence and hypothesis companion for round 5, written only from the sources in the
  key above. The executable brief is `CLAUDE_CODE_PROMPT.md`, the user requests are `DATA_REQUESTS.md`, and
  `README.md` indexes the folder. Writing this file changed no other file; nothing was trained, downloaded or committed.
- **Run order.** (1) Stage 0: seal M35 (§4.2), M34 (§4.3), confirm-1 with the C-ECT and C-GD texts (§4.4, §4.14,
  §4.15) and the closure ledger (§3); compute β_frozen first. (2) Stage 1: the E-CT study (§4.1) with its annexes
  (§4.6, §4.7), then the descriptive deliverables (§4.8–§4.12) and the guidance of §5, every number labelled "seen".
  Reuse `probes/scripts/P1_ct_*.py`, the PC re-run scripts (`P1SK_*`, `P2SK_*`, `P3SK_*`, `P5SK_*`) and
  `ideas2/scripts/addendum_gd_reconcile.py`. Scripts that write to fixed paths inside this design folder (P3SK_recompute,
  P3SK_analyse and the others listed in CLAUDE_CODE_PROMPT §1.2 and README "Reproduction and process notes") are copied
  into the round-5 folder with only their output path changed, never run in place. (3) Stage 2: send the requests of
  `DATA_REQUESTS.md` (U1-a … U1-i, U2–U6, S1) in one message, maintain the supply ledger, and freeze and score the
  package only if a claim reaches its minimum G.
- **Provenance checks before any E-CT number is relied on.** pub_19783c039b (all of SH − OFF; a cross-acid copy pair; a
  possible table-plus-figure parallel series) and pub_b0b3296ec5 (41 % of the E-CT gain; the only acid-type
  publication) (I2 "Duplicate and provenance candidates").
- **Open items carried from round 4.** C5-BTBP attribution and floor; the DMDPhPDA 5 M Pr/Nd row; ORNL data locations;
  laboratory repeatability (R4 REPORT §5).
