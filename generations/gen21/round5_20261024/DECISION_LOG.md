# Round-5 decision log (gen21, strict-adjacent log SF)

Stage-0 document, written 2026-10-24 before the Stage-0 seal. **No round-5 score had been computed when it was written.**
Every in-corpus number below is EXPLORATORY (seen: rounds 1-4, the round-5 probes and their checks saw it) or a Stage-0
planning input from `results/stage0/` (label-based where stated). No number here is a round-5 result.

- **Sealing.** `scripts/r5_seal.py` digests this file up to its append-only marker, the HTML comment placed as a whole
  line just before the request-status section (`MARKER_RE`: only a whole line counts; an inline mention never
  truncates; a second marker line refuses the seal). Sections 1-7 (conventions, ledger, gate, decisions, report rules)
  are sealed. Below the marker, request status and Stage-1 outcomes are appended with a UTC date and never edited: the
  seal records the byte length and digest of that tail, and `--verify` fails when the tail no longer starts with it.
- **Sources.** "brief" = `round5_design_20261023/CLAUDE_CODE_PROMPT.md` (section numbers). Tags EB, IR, I2, P1-P5, PC, RI,
  DR and R4-x are the brief's (its header table). "S0:" = a file of `results/stage0/` in this folder. "spec" =
  `results/<ITEM>/ITEM_SPEC.md`. Row ids such as `1.1-R02` are the ids of `REGISTRY.csv`.

---

## 1. Sealed conventions (summary)

### 1.1 M35: one-currency decision convention for E-A (brief 4.1; IR M35; EB 2.17, 6.5; E5 5.2)

E-A itself is unchanged: logSF(A/B), |Z_A - Z_B| = 1, V1, C3, k = 3 (four reads), DOPT_Q, 229 cells, 1,802 queries per
seed, dev seeds 104729, 130363, 155921, 196613, 262147, designs BP and BPG; reference B1 = 0.4521 (BPG) / 0.4519 (BP)
(brief 3.1; R4-REPORT 1).

| # | sealed rule |
|---|---|
| 1 | E-A pooled R² is still reported, for continuity. |
| 2 | **One gate.** Seed-mean pooled relative pair-SSE vs B1, `1 - SSE_cand/SSE_B1` per seed then the seed mean, **≥ 0.02 on BPG** (≈ +0.011 R² at SSE/SST 0.548). Plus SF_POOL on the seed-mean per-unit ΔSSE of the 13 `cluster_cpg` units, one-sided p ≤ 0.025, H0 mean ΔSSE ≤ 0, by exact enumeration of all 2^13 = 8,192 sign vectors (if G > 13: B = 4,999 random flips, p = (1 + #{T* ≥ T})/(1 + B)). Plus the guard of row 3. A pass is labelled DIAGNOSTIC, because M34's kill fired (1.4). |
| 3 | Balanced-MAE non-inferiority guard: seed-mean Δbalanced MAE (B1 - candidate) ≥ **-0.001**. |
| 4 | BPG decides; BP is a labelled sensitivity on the same test folds (EB 2.14). |
| 5 | Mandatory tables for every E-A contrast: the 13-row per-cluster table with provenance (ORNL-only; `data_location` class); the big fold against the other folds; C5-BTBP in and out; the six-design table (A, B, BR, BQ, BP, BPG) on training-free arms only (B0, HIST_D7). The A/B/BR/BQ fold records are built with `g21.cache.build_fold` into `cache/` (item 1.0). 100 records were built pre-seal; all 100 were poisoning-identical and passed isolation (RUN_RECORD 05:35-05:44 UTC; `results/leakage/cache_poison_EA_CHECK.json`). |
| 6 | Breadth diagnostic, never a gate: equal-weight per-cluster log SSE ratio; clusters with < 3 cells or < 20 rows per seed pooled into "small sources"; ORNL-only collapse a sensitivity; publication-level relative SSE inside the giant cluster. |
| 7 | Positive control outside any family: B1 - B0 (Holm over the 14-contrast replay family would give it Holm p 0.16 and kill it). |
| 8 | Appendix: M08 relation-class moment table with an identifiability column, **PLANNED** (D7; `REGISTRY.csv` 1.0-P01). |

**The equal-weight leg is not a gate. Recorded as a decision, not a test result** (brief 4.1 allows either). The M35
worlds (size under a heterogeneous null, power for a homogeneous 2 % effect, a gain only in tiny clusters, a ≥ 3 % gain
confined to the giant cluster) were **not run in Stage 0** [PC P2], so the demotion rests on the seen evidence:
- SF_LOG fires on HBK@ML - B0 (exact sign-flip p 0.0115, close to B1 - B0's 0.0137), although HBK@ML is +0.13 % pooled
  [IR M35]; P2 2 gives +0.14 % BPG and -0.64 % BP for the same contrast.
- Under a skewed heterogeneous null SF_LOG reaches 0.111 (C-LOG, chemotype units); publication-level skewed nulls stay
  ≤ 0.041 [P2 3].
- The same skew pushes SF_POOL to 0.103 (C-POOL, cpg), so skewed nulls are non-gating for both [PC P2; P2 3].

### 1.2 M35 replay facts (seen) and M35's own kill (brief 4.1; IR M35; PC P2)

- B1 - B0: **+5.85 %** pooled relative SSE (t 1.99); +12.3 % equal-weight (IM p 0.012); +14.75 % log ratio (t 2.88).
- Every round-4 slot is ≤ -0.02 % pooled. Outside the big fold no slot reaches 2 % (the best, S02, reaches +0.28 to
  +0.35 %).
- **M35's own kill, a check (seen).** The convention is abandoned only if the fixed-sequence replay fails the positive
  control [RI 4.2]. B1 - B0 is +5.85 %, with cpg SF_POOL p **0.00317 (BPG) / 0.00269 (BP)** [PC P2]. The convention
  passes its own positive control. Item 1.0 recomputes this as a CONSISTENCY CHECK (1.0-R01, 1.0-R03; outcome pending,
  section 7).

### 1.3 M34: inference convention (brief 4.2; P2 1, 3; PC P2; I2 confirm-1, confirm-4, rules 9-12)

| test | estimand | round-5 status | evidence |
|---|---|---|---|
| SF_POOL (exact sign-flip on pooled ΔSSE; exact at G ≤ 13, else 4,999 random flips) | pooled | **primary for pooled claims** | never flagged under the gating nulls |
| SIGN (exact binomial on unit wins, ties dropped) | π, the share of units that improve | **primary for win-share claims**; conservative (mean size 0.012) | never flagged |
| SF_LOG, T_LOG, IM_REL | equal-weight breadth (E-S) | diagnostics only | answer "does the average source improve?" |
| WCR_POOL (wild cluster bootstrap, CR2, Webb) | pooled | comparator only | 1 gate flag |
| SEALED (round-4 R² leg: percentile, BCa, jk-t(G-1), Kish ≥ 3) | pooled ΔR² | comparator only | 2 gate flags |
| WCR_LOG | breadth | dropped | 6 flags, size up to 0.113 |
| JK_KISH (cluster jackknife-t, Kish df) | pooled | not a gate | real-pool max size 0.0515; flags start at G = 8 |
| JK_G1 (round 4's sealed jackknife-t, df = G - 1) | pooled | **dropped** | real-pool mean size 0.047, max 0.096 (0.1245 on synthetic pools) |

**What the exact tests guarantee.**
- SIGN, SF_LOG and SF_POOL hold their size by construction under the arm-swap and symmetric nulls (0.004-0.027 in every
  checked cell). This is exactness under a sharp, exchangeable null, not an empirical finding [PC P2].
- No test holds 0.025 under skewed heterogeneous nulls at cluster or chemotype granularity (SF_POOL 0.103, SF_LOG
  0.111). Publication-level skewed nulls stay ≤ 0.041, except SIGN under C-MED (0.106 at τ 0.065, G 80) [P2 3].
- Under laboratory clustering, family-level sign-flips have size 0.026-0.054 at ICC 0.3-0.5 [I2 confirm-4]. Two-stage
  (chemotype → laboratory) nulls give 0.041-0.049 for balanced compositions at ICC 0.3 and 0.06-0.14 otherwise
  [I2 confirm-1]. Chemotype clustering among laboratories is handled by the size gate (CP-13 item 2; D3).

**Units.**
- New sources: laboratory × chemotype, with publications aggregated within a unit. Copy groups are merged at arrival from
  metadata, never retroactively [IR M34].
- Gd-amplitude contrasts: the laboratory, aggregated across chemotypes. The Gd chord is shared across a laboratory's
  papers: +0.37 to +0.47 on publication means, +0.73 cell-level for the same extractant [I2 rule 12, lab-4].
- E-CT on new data: the laboratory, with families nested and each laboratory weighted 1; the family-level p is reported
  beside it [I2 confirm-4]. D3 changes the primary aggregation for C-ECT to chemotype units.
- In-corpus diagnostics: `cluster_cpg` for E-A; publication × chemotype for E-CT (cluster level "not estimable") [P1 12].

**Multiplicity.** One-sided 0.025 per claim. No global claim. No Holm inside a claim at small G: with exact tests, G = 6
cannot reach 0.0125, because the smallest p is 1/64 = 0.0156. Fixed-sequence gatekeeping inside a claim [I2 rules 9-10].

**G floors.** G ≤ 5 cannot reject at 0.025 with SIGN or SF (minimum p 1/32 = 0.031) [P2 3]. C-ECT needs ≥ 7
laboratories (≥ 7 chemotypes under D3 (ii)). M33 Gd-dip transport runs with ≥ 10 laboratory × chemotype units on the
merged SAFE + literature ledger and decides with ≥ 10 laboratories. C-GD needs ≥ 6 laboratories from ≥ 3 families.

**Power is stated in worlds, never as one number** [I2 rule 11]: a corpus-DGA world, a halved effect, a cation-exchanger
or null-like world, and 25 % / 50 % null-like mixtures (S0: `power_worlds.json`, 10,000 replicates per world).
Corpus-resampled power assumes new units are exchangeable with corpus units. The new supply is not: 15 of 16 probable new
laboratories use chemotypes the corpus lacks [P4].

**Planning numbers (B1 - B0 positive control, one-sided 0.025; corrected values)** [P2 4; PC P2]:

| unit | pooled tests, τ = 0 | pooled tests, τ = 0.065 | equal-weight SF_LOG |
|---|---|---|---|
| publication-like | SF_POOL ≈ 43; SEALED about 38-41 (P2 grid 39 BPG / 38.3 BP; direct ≈ 41); JK_KISH ≈ 40 (not a gate) | > 80 | 19-20 (τ 0); ≈ 26 (τ 0.065) |
| chemotype-like | 30-58 for the pooled tests (SF_POOL ≈ 28-30; WCR_POOL 40; SEALED ≈ 41; JK_KISH 58 on the P2 grid, ≈ 50-55 direct); SIGN (π) ≈ 55-58 | > 80 | never within 80 |
| publication × chemotype | 59-72 | > 80 | never within 80 |
| non-DGA publications only | SF_POOL 16 | 80 | SF_LOG 12 / 17 (SIGN 11 / 27) |
| DGA publications only | > 80 | > 80 | 62 / 80 |

- A 2 % effect (B1 scaled to the M35 gate) needs > 80 units at τ = 0.065 for every test and unit. The exception is the
  publication-level log-ratio tests, at 74-77 units [P2 4]. That pool inherits B1's consistency, so it is optimistic for
  real candidates [PC P2].
- B1's per-unit gains were measured on the cells on which B1's form was chosen. Every G80 above is therefore an
  upper-side (optimistic) estimate for genuinely new sources [PC P2].

### 1.4 M34's kill outcome (brief 4.2; PC P2; IR M34)

At G = 8 publication units, the gate-valid power for B1 - B0 is **0.228 (BPG) / 0.236 (BP)** at τ = 0.065. On the pooled
decision estimand, SF_POOL power is **0.285-0.296** at τ = 0. Both are below 0.30, so **the kill fires** on the pooled
estimand at both τ. Sealed consequences:
- the 40-80-plus-source planning number stands for any pooled E-A claim;
- every round-5 model contrast is run as a **DIAGNOSTIC**;
- no E-A slot is registered (section 3).

### 1.5 CP-13: frozen confirmation-package rules (brief 4.3; I2 confirm-1 ... confirm-4)

| item | sealed rule | Stage-0 status |
|---|---|---|
| 1 | Per-claim minimum G and a supply-scenario power table, recomputed laboratory by laboratory from the frozen key (G_HS, G_Gd, G_ECT), with the probability of reaching each minimum | lab-by-lab lists in S0 `supply_audit.json` (`claim_supply_lab_by_lab`); power in S0 `power_worlds.json` and `power_cgd_beta_sens.json`. P(reach minimum G) from S0 `supply_reach.json` (forecasts; P4 rates / PC P4 rates): C-ECT 0.2266 / 0.0453; C-GD ≤ 0.0623 / 0.0 (families not modelled); Gd-dip transport 0.8459 / 0.0879 with SAFE at 3 (0.3005 / 0.0002 with SAFE 0); SAFE route 0.0. Key PROVISIONAL (D6) |
| 2 | Size gate: a test is dropped if its size exceeds 0.035 (Clopper-Pearson 95 % lower bound) under i.i.d. or two-stage (chemotype → laboratory) nulls at ICC {0.23, 0.3, 0.5, 0.69}, with ≥ 2,000 replicates and a fixed seed | run: S0 `size_gate.json` (20,000 replicates per cell). Every statistic is flagged; the fallbacks are sealed in **D3**. The C-ECT fallback (ii) passes its fresh check (S0 `size_gate_ect_chemotype.json`) |
| 3 | One value per laboratory for every Gd and E-CT claim (laboratory-bootstrap power at G = 6 is 0.742 vs 0.736 for units) | adopted. Under D3 (ii), C-ECT's primary aggregates further, to one value per chemotype |
| 4 | Exclusion list (not decidable with the supply): pooled or E-A claims; a TOST on β (power ≤ 0.08); the window as a row flag (≤ 0.15); SH with 2 reads (≤ 0.06); the sc009 Nd dip; any DGA-only Gd test | adopted. D3 (i) adds the C-GD laboratory-mean chord interval (raw and winsorised) and the Gd-dip transport interval, as descriptive intervals |
| 5 | Three-way decisions with explicit inconclusive zones; every kill rule with P(kill \| corpus effect true) and P(pass \| null) at the realistic G | operating characteristics in S0 `power_worlds.json` (example below) |
| 6 | Non-rejection never switches the deployed model (withdrawing B1's Gd term on non-rejection would withdraw a real effect about 17 % of the time at G = 8 [I2 confirm-1], and an exactly calibrated term in 56 % of G = 8 replicates [I2 confirm-3]). Only a rejection can change the default | adopted. Under D3 (i) the C-GD interval is descriptive, so the B1-harm switch to B0 for new chemotypes cannot fire in round 5 |
| 7 | Supply-audit kill (metadata only), run at the end of Stage 0 and after each answered request | **fires** (D4): round 5 is descriptive (brief 7.6) |
| 8 | Gd-claim eligibility: the unit reports Eu, Gd and Tb; DL/LOQ values excluded; aqueous complexants excluded (M29); frozen mechanism-class tag. C-ECT: ≥ 3 full-series conditions of one extractant; T0 excluded at pair level; aqueous-complexant changes allowed as T1 | class map frozen (S0 `mechanism_class_map.csv`); every complexant, acetate included, decided now as excluded for the Gd claims (`r5/supply.py` docstring); T0 at pair level (D5) |
| 9 | β_frozen = the inner-CV B1 refit on all cells, with the sensitivities BP, λ = 0.03 and the round-4 fold mean | **-0.110809** (D2) |
| 10 | Planning amplitudes per family from primary SF tables, never from the corpus | HDEHP -0.26 and P507 -0.105 sealed **PROVISIONAL / PENDING-U5** (D6) |
| 11 | Chord hygiene: V2-5 floor censoring; a resolution rule for sources that report D to one decimal (≈ ±0.05 log of quantisation noise, comparable to β_frozen/2) | rule text in `CONFIRMATION_PACKAGE.md` §13: V2-5 censoring of Gd chords is **primary** (with source-stated DL / LOQ); the chord without it is a sensitivity; the resolution rule flags only (D9) |
| 12 | Freeze before any new value is read (predictor, β_frozen, read policy, censoring, inclusion, code, hashes). External reads: `support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2·s2_all)`; 4 reads = `seq[:3]`, 3 reads = `seq[:2]`; B1 on new data uses β_frozen | object built: `cache/ext/ext_dopt.pkl`, SHA-256 `fc261e21…0194f3`, 18,457 bytes (D1). The frozen DOPT_Q Gd-read rate (part of the C-GD claim) on corpus C3 cells with Gd measured: **60/218 at 3 reads, 95/218 at 4 reads** (S0 `ext_dopt.json`) |

Example operating characteristics. C-ECT under the laboratory-level sign-flip, τ = 0, from S0 `power_worlds.json`.
These are planning values: that statistic failed the size gate, see D3.

| world | P(confirm) at G = 6 / 7 / 8 | P(kill) at G = 6 / 7 / 8 |
|---|---|---|
| CORPUS_DGA | 0.6306 / 0.9143 / 0.9836 | 0.0005 / 0.0007 / 0.0 |
| HALF | 0.07 / 0.1305 / 0.1653 | 0.1431 / 0.2222 / 0.1253 |
| MIX25 / MIX50 | 0.3187 / 0.4417 / 0.5034 and 0.1313 / 0.194 / 0.2251 | 0.0514 / 0.0452 / 0.0275 and 0.1678 / 0.1729 / 0.1288 |
| NULLLIKE_W1_CATEX | 0.0056 / 0.0222 / 0.0331 | 0.4928 / 0.6607 / 0.5306 |
| NULL_SWAP (null) | 0.0165 / 0.0213 / 0.0223 | 0.5361 / 0.6044 / 0.5441 |

The win-share leg makes P(kill) depend on the parity of G: it kills at ≤ (G − 1)/2 wins at odd G but at ≤ G/2 − 1 at
even G. A halved effect is killed 0.222 at G = 7 and 0.195 at G = 9, against 0.125 / 0.106 at G = 8 / 10; the
sign-symmetrised null 0.604 / 0.591 against 0.544 / 0.549 (τ 0; S0 `power_worlds.csv`). At the realistic supply (2
linked chemotype units) the claim is a descriptive case series: the minimum attainable p is 1/4, so P(confirm) = 0.

---

## 2. Closure ledger (brief 4.4, in full; one line each; no slot)

"Closed" means "cannot beat B1 by ≥ 0.01 R² under BP/BPG on this corpus", not "chemistry absent" [EB 3]. Grades: FIRM
(null when cross-fitted, oracle ceiling below the margin, or replicated); MODERATE (null, but the test lacked power);
WEAK (not detected; control power not shown) [EB 3].

**E-A model routes closed on this corpus**

| route | evidence | grade | power caveat | source |
|---|---|---|---|---|
| zero-shot structure → strict-adjacent log SF (ECFP, RDKit, ChemBERTa, kernels, logK prior, log-D route, NNC) | R² ≈ 0.05 at best; NNC -0.128 / -0.29 | FIRM | confirmation seeds, many designs | EB 3 |
| learned curve families vs PWL at k = 3 (RIDGE2, HB2+GP, HBK) | ≤ PWL_R; error correlation 0.971 | FIRM | random-walk algebra | EB 3 |
| smoother interpolation (PCHIP, spline, local quadratic, global fits) | -0.020 to -0.095 | FIRM | | EB 3 |
| universal per-metal / pair-type / family priors (incl. B2) | LOCO -0.018 to -0.047; B2 -0.022 (0/5 seeds) | FIRM | ceiling ≈ 0.04 even if perfect | EB 3 |
| per-metal deviation priors from other publications (lab-3), unconditional or level-matched at 0.5 / 1 / 2 log | BPG MG_tr -4.8 % [-8.5, -1.2]; MG_LM05/10/20_tr -4.8 / -5.8 / -6.8 % relative SSE; shrinkage monotone toward 0. Same-chemotype priors are untestable under E-A (0 of 1,145 instances have a same-chemotype training donor) and are not closed; the M07 sc009 Nd dip is carved out | FIRM on significant harm, not on the reopen bar | reopen bar = what 13 clusters can detect (section 3 item 4) | I2 lab-3 |
| amplitude-scaled (proportional-LFER) anomalies | in sample -0.0025; LOCO -0.017 (0/5) | FIRM | | EB 3 |
| convex blends / stacking | leave-one-cluster-out -0.019 to -0.035 | FIRM | | EB 3 |
| per-cell offset / global recalibration | leave-one-pair-out 0.292; oracle recalibration +0.0005 | FIRM | | EB 3 |
| static fixed read sets | LOCO 0.484 vs DOPT_Q 0.489 (78 cells) | FIRM | 78 cells | EB 3 |
| cross-metal xTB energies with changing composition | ρ -0.084, reliability 0.80 | FIRM | powered null | EB 3 |
| abstention by difficulty scores (ensemble spread, error tables, \|B1\|) | within-pair-type Spearman ≤ 0.18 | FIRM for these scores, not for the level window | | EB 3 |
| sibling-read pooling under BP (HBG@POOL) | -0.041 / +0.002 | FIRM for reads only | says nothing about full sibling curves | EB 3 |
| floor-aware read clamps on V1, at every floor F | 91 % of the gain is one cell; cross-fitted F 0.4098; the consistent all-reads clamp is negative | FIRM (closure); reopen only via U1 per-source limits | undecidable without a source-stated limit | IR M24 |
| S02 electron-count shapes × donor class | -0.024 / -0.028 | MODERATE | loss sits in the 51-69-cell big fold (+0.0019 R² elsewhere) | EB 3 |
| S01 donor-class Gd / Nd dips | -0.0001 / -0.0005 | MODERATE | synthetic 0/8 at D7 size; cross-fitted donor × pair ceiling -0.041; [O] ceiling +0.0004 balanced MAE | EB 3 |
| S05 CatBoost residual corrector | -0.006 / -0.003 | MODERATE | dev MDE 0.011; inner selection optimistic | EB 3 |
| S11 Ln-only multitask encoder | -0.0005 / -0.0017 | MODERATE | reproduced B1 (row r 0.99983) | EB 3 |
| S06 compact 3D geometry | -0.006 / -0.011 | WEAK | synthetic 0/6; 195 / 229 cells change composition | EB 3 |
| An(III) transfer (S13-S16) | inner gate 3/50 | WEAK | synthetic 0/6; 83-93 % Am | EB 3 |
| Ce-acid term | a two-publication effect; within-publication slope -0.024 per decade; permutation keeps 61-76 % | closed lead | | IR ledger (M28, M27) |

General power caveats for every MODERATE and WEAK line are in brief 9.2: Kish 2.24; the percentile cluster bootstrap
rejects at 0.07-0.08 for a nominal 0.05; contrast-specific dev MDEs; BP is not independent of BPG. Two more belong here
[EB 3]. First, the synthetic studies used 6-8 replicates (exact 0/n upper bounds 0.37-0.46) and were miscalibrated: S05
and S11 gained about +0.02 over B1 in NULL worlds. Second, the in-sample donor contrast (+0.04 R² [O]) is real but does
not transfer across about 4 effective chemotypes.

**Other closure lines** [IR "Closure ledger"; I2 "Closure ledger additions"]

| line | evidence |
|---|---|
| E-CENS at F_lo = -4 is the C5-BTBP exclusion | 0.487 against 0.486; F_hi has no anchor |
| V2-S is null; V2-5 carried all of round 3's V2 move | V2-S moves B1 by +0.001 (a data-version sensitivity on a different row set; inside the random-removal null); V2-5's 16 pairs per seed (4.1 % of SSE) carried all of round 3's V2 move [IR M26] |
| robust losses add nothing | Huber β -0.1167 against -0.1147; R19 +0.002 |
| the edge continuation is a reference variant (B1 + edge), not a claim | +0.004 to +0.0055 cross-fitted, depending on the estimator |
| low-D sibling irreproducibility is C5-BTBP | 88 % of rows below -3; ratio 1.08 without it |
| the provenance class is a column, not a lever | permutation p 0.12; 3 TAB publications |
| no Gd term in aqueous-complexant media (adopted as a data-contract rule) | scoring killed by drop-one (78.5 %) |
| cross-cluster residual share ≈ 0 | -0.13 % [-0.41, +1.37] |
| publication-weighted conformal intervals are width inflation | width-matched row-weighted interval at 0.932 is equivalent |
| replicate a read only if per-metal σ > about 0.2; otherwise add a metal | ΔMSE = 0.216 s²; cost-matched break-even 0.20-0.31 |
| SAFE "+1,618 newer records" are mostly pre-existing gap IDs | 1,602 missing IDs in 11 runs; highest ID 18,372 |
| zero-read copying of one's own series is never the recommendation; worst below the measured acid range | near side COPY@0 0.655 vs SH@2 0.664 (2 of 8 publications); below-range flag +0.30 |
| free per-metal acid-slope regression, linear log-acid extrapolation and slope physics at 4 reads are retired | AREG@0 -0.279; AINTX@0 -0.32; AEBGSH@4 -4.1 % vs SH |
| never extrapolate an unshrunk acid trend beyond the measured range | -0.64 R², 6 of 8 publications lose |
| a literature series of the same extractant gives no direction information beyond "the heavier member extracts more" | 0.922 vs always-heavier 0.934 on identical covered rows |
| "spread the reads, never one metal" is definitional | one read = zero reads by construction; HULL - random +0.197 at 2 reads |
| publication units are not valid for Gd-amplitude contrasts | Gd chord shared across a laboratory's papers +0.37 to +0.47 |
| an in-run TODGA reference is not needed for prediction; any second own-campaign ligand carries the signal | MX +5.9 % vs own TODGA +5.3 % |

**Open but unscheduled.** On E-A: condition × series-shape interactions (H-COND), which would need the gate of section 3.
As separate regimes, each needing its own registration and never E-A:
- chemistry priors at 2 reads or fewer (E-D; PWL_R 0.191 at 2 reads);
- Gd-aware or learned adaptive read policies (+0.035 [-0.009, +0.057] at C6 k = 6; WEAK, not ruled out at high k);
- leave-pair-type-out (E-E) [EB 3-4; E5 6].

---

## 3. Pre-slot decidability gate (brief 4.5; sealed now; retire-only)

No new E-A model slot may be scored unless all of the following hold before any outer score of that slot. If the gate
fails, the slot is not run and E-A stays closed on this corpus.

1. **Oracle ceiling ≥ margin.** The slot's own feature set, fitted with test labels and labelled ORACLE, reduces pooled
   pair-SSE against B1 by ≥ 2 % on BPG [IR M35 item 7].
2. **Power ≥ 50 % at its plausible effect.** This comes from a synthetic study with ≥ 50 replicates per world (an
   observed 0/50 has an exact upper bound of 0.07), run through the sealed rule. The nuisance is calibrated to the real
   B1 residual (adjacent pointwise ρ 0.69, cell-offset share 36 %, per-cluster heterogeneity, light-end variance), and
   the study includes a NULL world in which the candidate does not gain [E5 5.3; EB 3].
3. **Label-free support (retire-only).** The slot is demoted when the deciding fold has < 10 training cells sharing the
   test rows' donor class or chemotype [IR M35 item 7]. The slot states its training-cell need: a model that needs more
   than about 60 cells cannot be judged on the deciding fold [E5 5.5].
4. **Reopen bar for per-metal priors (lab-3).** A pre-declared fold-legal variant must beat B1 by about 4-6 % relative SSE
   (≈ 0.02-0.03 R²), with a positive cluster-jackknife lower bound with and without C5-BTBP [I2 lab-3].

A slot that passes is sealed by amendment and scored under M35 as a DIAGNOSTIC. It is compared against B1 and against the
B1 + edge reference variant (fold-fitted damping d with inner CV plus a noise-matched linear-tail control) [IR M28; EB 4
H-EDGE].

**Proposals the gate has seen: none** (as of 2026-10-24). **E-A model slots registered: none** (`REGISTRY.csv` row
`0.EA`). A later proposal is logged below the marker with its gate outcome before any outer score.

---

## 4. Decisions D1-D11 (each with its reason and sensitivity; D9-D11 added at the pre-seal review)

### D1. External DOPT_Q covariance fitted on all 521 bench cells

- **Decision.** `C_all` and `s2_all` of the external read policy (CP-13 item 12) come from the
  `g21.cache.build_fold` outer recipe applied to a fold record whose outer training set is every bench cell (521; all 229
  C3 cells included). This is the record behind β_frozen (`cache/frozen/fold_all_BPG_104729_0.pkl`, SHA-256
  `7c862d87…e835f`; design BPG; inner seed `gen20k.splits.inner_seed(104729, 0)` = 3246616).
- **Reason.** Brief 4.3 item 12 asks for the `build_fold` outer recipe "fitted once on all C3 cells". That recipe trains
  on all outer-training bench cells, not on C3 only. D1 keeps the recipe unchanged and uses the same training set as the
  β_frozen record, so the read policy and the amplitude come from one object.
- **Values** (S0 `ext_dopt.json`). n_train 521; s2_all 0.178532; noise_var = 2·s2_all = 0.357063; noise factor 16.0
  (inner selection); C_all minimum eigenvalue 5.57e-6. The pickle was rebuilt and is bit-identical to the stored one.
- **Sensitivities.**
  - Design BP: s2_all identical; maximum covariance difference 0.0012; the same read sets on 229/229 cells (3 reads,
    4 reads and full sequence).
  - The literal C3-only fit: n_train 229; s2_all 0.050138; noise factor 4.0; relative Frobenius covariance difference
    0.579. Only 89/229 four-read sets, 98/229 three-read sets and 55/229 full sequences are the same. The choice therefore
    changes most external read sets. It is recorded, not used.
  - For context, the external four-read sets agree with the round-4 dev-fold BPG sequences on 73-114 of 229 cells per seed.

### D2. β_frozen = -0.110809 (inner-CV value)

- **Decision.** β_frozen = **-0.11080937578907438**, sealed per brief 4.3 item 9. It is
  `r4.refs.fit_reference(bench, rec, Y, 'B1', k=3)` on the all-cell BPG record of D1, default `LAMBDA_GRID`; the chosen
  λ is 0.003; 229/229 C3 cells used; 4 inner blocks; one BLAS thread (S0 `beta_frozen.json`). β_frozen/2 =
  -0.05540468789453719 (from S0 `beta_frozen.json`; `size_gate.json` and `power_worlds.json` record the same value and
  the file's LF SHA-256 `2b5d0e2d…cc6264` since their pre-seal rerun, N1).
- **Reason.** The brief seals the inner-CV value, with the sensitivities reported beside it.
- **Disclosed fragility.** λ = 0.003 beat λ = 0.03 by **7.6e-5** in inner score (0.027803 vs 0.027879). Round 4 chose
  λ = 0.03 in 23 of 25 folds in both designs [PC P5; S0 `beta_frozen.json`]. β_frozen's anchor also depends on U1. B1's
  Gd transfer is learned mainly from the 69 unlocated cells; refit without them, the β mean is -0.0708 [P5 4b; PC P5].
- **Sensitivities** (S0 `beta_frozen.json`):

  | variant | β |
  |---|---|
  | design BP (λ 0.003) | -0.1095 |
  | λ fixed at 0.03 (BPG) | -0.0878 |
  | round-4 fold mean, BPG (25 folds; range -0.1398 to -0.0299) | -0.0923 |
  | round-4 fold mean, BP (25 folds; range -0.1378 to -0.0748) | -0.0949 |

- **C-GD power at these β** (planning; CGD_W confirm rule, 10,000 replicates per world; Monte Carlo SE ≤ 0.005).
  Sources: S0 `power_worlds.json` (β_frozen, -0.118, -0.072, -0.094) and S0 `power_cgd_beta_sens.json` (the four D2
  sensitivities). The second file reproduces the 768 β_frozen rows of `power_worlds.csv` exactly (maximum difference
  0.0). Brief 7.5 states C-GD power from analytic non-central t in chemistry worlds, so the chemistry-world column
  comes first; the corpus-bootstrap column is context only (it is kept because brief 4.3 item 9 quotes these values).

  | β reference | HDEHP-like (sd 0.10), G = 6 | corpus bootstrap (seen) EMP_HS12, G = 6: context only, not the planning power (brief 7.5) | EMP_HS12, G = 8: context only |
  |---|---|---|---|
  | β_frozen -0.1108 | 0.8371 | 0.5532 | 0.7897 |
  | design BP -0.1095 | 0.843 | 0.5659 | 0.8028 |
  | λ 0.03 -0.0878 | 0.9215 | 0.7839 | 0.9552 |
  | round-4 fold mean BPG -0.0923 | 0.9081 | 0.7385 | 0.9352 |
  | round-4 fold mean BP -0.0949 | 0.9004 | 0.7162 | 0.9203 |
  | -0.094 (IR M33) | 0.9034 | 0.7237 | 0.9251 |
  | -0.072 (I2 confirm-1) | 0.956 | 0.8989 | 0.9914 |
  | -0.118 (I2 confirm-1) | 0.8014 | 0.4807 | 0.7131 |

  EMP_HS12 bootstraps the 12 seen heavy-selective non-DGA unit chords. The -0.118 and -0.072 rows reproduce I2
  confirm-1's 0.48 / 0.71 and 0.89 / 0.99. A less negative reference (closer to 0) is easier to beat, so power falls as
  β moves from -0.072 toward β_frozen. These values are planning only: the statistic failed the size gate (D3), and the
  worlds are i.i.d. laboratories. The P507-like centre (sd 0.10-0.16) gives 0.02 at every G (P507 ≈ β_frozen).

### D3. Size-gate fallbacks (brief 4.3 item 2): every chord statistic is flagged

Size-gate results (S0 `size_gate.json`): flag = CP 95 % lower bound > 0.035 at nominal 0.025; 20,000 replicates per cell.

| statistic | claim / role | flagged gating rows | max size (cell) | max two-stage size by ICC 0.23 / 0.3 / 0.5 / 0.69 (confirm rule) | max i.i.d. size |
|---|---|---|---|---|---|
| CGD_W | C-GD primary (winsorised [-0.40, +0.20]) | 328 / 496 (322 two-stage, 6 i.i.d.) | 0.1471 (ICC 0.69, sdw 0.1, comp (4, 2)) | 0.05495 / 0.0671 / 0.10705 / 0.1471 | 0.02775 confirm; 0.05065 harm / kill |
| CGD_R | C-GD sensitivity (raw) | 328 / 496 (322, 6) | 0.1434 | 0.05425 / 0.0671 / 0.1043 / 0.1434 | 0.0277 confirm; 0.05065 harm / kill |
| GDT_R | Gd-dip transport (M33 laboratory-mean t-interval) | 322 / 480 (48 at a decidable G) | 0.1434 | 0.05425 / 0.0671 / 0.1043 / 0.1434 | 0.0277 confirm; 0.0313 harm / kill |
| ECT_SF | C-ECT laboratory-level exact sign-flip | 21 / 43 (11 at a decidable G) | 0.0979 (ICC 0.69, comp (4, 2)) | 0.04535 / 0.05065 / 0.0701 / 0.0979 | 0.0265 |
| ECT_SIGN | C-ECT win share | 11 / 43 (1 at a decidable G) | 0.0979 | 0.0344 / 0.04505 / 0.06415 / 0.0979 | 0.0169 |

The C-ECT gating denominators are 43 (not 48): the seen-pool skewed nulls emp_P1 and its mirror are descriptive rows
(below). The flag is the lower limit of the two-sided 95 % Clopper–Pearson interval (one-sided 97.5 %; P2's
convention); under a one-sided 95 % bound 332 / 332 / 326 rows flag for CGD_W / CGD_R / GDT_R (12 more cells: comp
(2,2,2), ICC 0.23), with the same verdicts.

The reproduction check against `ideas2/confirm1skepB_out/clustered_null_size.csv` covered 63 cells: maximum |difference|
0.0069, maximum |z| 2.46.

- **Decision.**
  - **C-GD laboratory-mean chord interval (raw and winsorised) and the Gd-dip transport interval → fallback (i).** They
    move to the CP-13 item-4 exclusion list as **descriptive intervals**. C-GD's three-way rule (confirm / kill / B1-harm
    / helps) is read descriptively. No switch of the deployed model can follow (CP-13 item 6).
  - **C-ECT → fallback (ii).** C-ECT unit = a connected component of the laboratory–chemotype graph of eligible
    families (chemotypes from the frozen gen6 map, DATA_CONTRACT §8.3), so that no laboratory contributes to two units;
    unit value = unweighted mean of its laboratories' values; exact one-sided sign-flip over units; minimum G = 7 units.
    With no spanning laboratory this is exactly one value per chemotype. The test needs a fresh size check, which uses
    the two-stage nulls of `results/stage0` at the ICC grid with ≥ 2,000 replicates and is written to
    `results/stage0/size_gate_ect_chemotype.json`. The laboratory-level sign-flip and SIGN are reported as descriptive
    beside it.
- **Reason.** Brief 4.3 item 2 requires one sealed fallback for every statistic that fails. A t-type laboratory-mean
  interval cannot be repaired by aggregation inside the realistic G. For C-ECT, aggregation to the clustering level
  removes the chemotype → laboratory dependence that the two-stage null models. The link rule is part of the unit: a
  laboratory spanning chemotypes would otherwise put one laboratory into two units that the sign-flip treats as
  independent, and the fresh size check simulates only nested compositions (each laboratory in one chemotype), so it
  does not cover a spanning laboratory.
- **Fresh size check of fallback (ii)** (S0 `size_gate_ect_chemotype.json`; `scripts/s0_sizegate_ect_chemotype.py`;
  20,000 replicates per cell; flag rule as above).
  - **ECT_CHEM_SF passes on every gating cell.** 0 of 80 gating rows flagged; 56 of them at a decidable G_chem ≥ 7.
  - Maximum size 0.0268 (two-stage, ICC 0.3, 13 one-laboratory chemotypes); CP lower bound 0.0246.
  - Maximum two-stage size by ICC 0.23 / 0.3 / 0.5 / 0.69: 0.02455 / 0.0268 / 0.02475 / 0.02635.
  - i.i.d. maxima: normal 0.02485; t(3) 0.02375; swap_P1 0.02615.
  - **Skewed seen-pool nulls (descriptive, never gating; sealed reading).** The P1 family log ratios are right-skewed;
    a one-sided sign-flip is endangered by the mirrored, left-skewed orientation. Chemotype-level size at G = 7 / 8 /
    10 / 13: emp_P1 (right-skewed) 0.00445 / 0.00825 / 0.0073 / 0.00815; emp_P1_mirror (left-skewed) 0.0723 / 0.0579 /
    0.0614 / 0.0573. Laboratory-level ECT_SF at G = 6 / 7 / 8 / 10 / 13: 0.0057 / 0.00555 / 0.00735 / 0.00745 /
    0.00825 and 0.0347 / 0.07575 / 0.06155 / 0.06195 / 0.0603 (S0 `size_gate.json`, `size_gate_ect_chemotype.json`).
    *Reason:* brief 4.2 states that no test holds 0.025 under skewed heterogeneous nulls and that skewed nulls are
    non-gating [PC P2; P2 3]; the scripts marked emp_P1 as gating in only its benign orientation, which contradicted
    that. *Rejected alternative:* keeping both orientations gating would flag ECT_CHEM_SF and move C-ECT to fallback (i).
    *Sensitivity:* the mirrored sizes above are reported beside the test whenever it runs.
  - ECT_CHEM_SIGN, reported beside it and not sealed, reaches at most 0.0121.
  - The laboratory-level rows on fresh draws reproduce the failure: ECT_LAB_SF flags 37 of 68 rows (maximum 0.0972) and
    ECT_LAB_SIGN 14 of 68.
  - Of the file's compositions only (4, 3, 2, 1, 1, 1, 1) reaches 7 chemotypes. The declared supplementary compositions
    cover the decidable range. The minimum attainable p at 7 chemotypes is 1/128 = 0.0078.
- **Sensitivities and status.**
  - The laboratory-level C-ECT p and SIGN are reported as descriptive.
  - For C-GD, the winsorised interval stays primary and the raw interval a sensitivity, both descriptive.
  - Chemotype units can be fewer than laboratories. In S0 `supply_reach.json` (P4 hand chemotype labels; laboratories
    that span chemotypes link them into one unit):
    - the verified supply is 2 units (acidic organophosphorus; quaternary-onium nitrate IL);
    - even if every E1 + E2 laboratory converts, there are 10 distinct chemotypes but only 6 linked units;
    - unlinked distinct chemotypes would give P(≥ 7) = 0.5 (P4 rates) / 0.2757 (PC P4 rates) against 0.0 linked under
      both rate sets.
    - These are forecasts and never decide (D4).

### D4. Supply audit (brief 4.3 item 7): the kill fires for every claim, so round 5 is descriptive

- **Decision.** Decision supply = in-hand units verified from metadata under the frozen key, with the institution-link
  (P) rule applied consistently, plus SAFE's maximum (3) where SAFE units count (U2 unanswered). C-ECT counts linked
  chemotype units (D3 fallback ii). The kill reads the claims that remain tests (C-ECT); C-GD and Gd-dip transport are
  descriptive intervals under D3 (i), so their "≥ minimum" is a descriptive-run trigger. Forecasts (P4's rates; PC P4's
  rates), the all-convert scenario and the brief's realistic bracket are reported and never counted. `r5/supply.py`
  (AUDIT_RULE, `supply_audit`) applies exactly this rule since the pre-seal review (s0-statistics-3, s0-fidelity-5).
- **Reason.** A forecast is not a laboratory. Brief 4.3 item 7 and 4.6 read the audit on the frozen ledger.
- **Values** (S0 `supply_audit.json`; `SUPPLY_LEDGER.csv`):

  | claim (role) | min G | verified | decision supply (D4: verified in hand + SAFE maximum 3) | bracket high (reported, never decides) | brief realistic | forecast P4 / PC P4 | all convert | below min |
  |---|---|---|---|---|---|---|---|---|
  | C-ECT (test; linked chemotype units; 2 laboratories: Chiba-Katsuta, KULeuven-Binnemans) | 7 | 2 units | 2 | 3 | 2-3 (laboratories) | 5.5 / 4.33 laboratories | 6 units (9 laboratories) | yes |
  | C-GD (descriptive, D3 i; laboratories, ≥ 3 families) | 6 | 1 (Chiba-Katsuta) | 1 | 5 | 2-5 | 3.5 / 1.67 | 9 | yes |
  | Gd-dip transport (descriptive, D3 i; merged SAFE + literature; laboratory × chemotype units) | 10 | 2 | 5 | 6 | 2-6 | 11.5 / 7.67 | 23 | yes |
  | SAFE route (M33 gate; descriptive) | 10 | 0 | 3 | 3 | 0-3 | 3.0 / 3.0 | 3 | yes |

- **Outcome.** SUPPLY-AUDIT KILL: every claim is below its minimum G, so no confirmation runs and round 5 is reported as
  descriptive (brief 7.6). The audit reruns after each answered request (brief 4.6).
- **Sensitivity.**
  - The kill outcome is identical under the bracket reading (`bracket_high` = max(in hand, the brief's realistic high)
    = 3 / 5 / 6 / 3), because every realistic high is below its minimum G (S0 `supply_audit.json`
    `kill_under_bracket_high` = true).
  - Under P4's forecast rates, the Gd-dip transport supply (11.5) would reach 10, which is a descriptive-run trigger
    only (D3 i); under PC P4's rates (7.67) it would not. Neither decides.

### D5. T0 / T1-T3 classification at pair level; Chiba (PROVISIONAL until values)

- **Decision.** E-CT families are classified pair by pair (brief 5.5). A sibling pair is T0 when it differs only in pH,
  [HA] or phase ratio for a cation exchanger, with no pH-dependent complexant or buffer, at trace loading, below the
  acidity where solvation extraction starts. Otherwise it is T1 (aqueous medium change), T2 (organic phase change) or T3
  (mixed) [I2 confirm-4]. For Chiba-Katsuta (P4-007, HDEHP, "pH and HDEHP concentration x 2 diluents"; S0 `lab_key.csv`
  `t_classes` = T0;T2), the pH-only and [HA]-only pairs are **T0 and excluded**, and the diluent-change pairs are
  **T2**. This stays PROVISIONAL until the values (or the full text) are seen by the separate extraction agent.
- **Reason.** The brief asks for pair-level classification. The mechanical sibling parse of the abstract finds a diluent
  factor. The brief's supply table reads Chiba as "abstract-only and pH-controlled, T0" [P4 4-5; PC P4].
- **Sensitivity.** With Chiba entirely T0 (the brief 7.5 reading), Chiba does not count toward G_ECT, and the verified
  C-ECT supply falls from 2 to 1 (KULeuven-Binnemans). The kill fires either way.

### D6. Laboratory key and planning amplitudes PROVISIONAL (U6, U5 open)

- **Decision, laboratory key** (S0 `lab_key.csv`, `lab_key_links.csv`, `supply_audit.json`; `r5/supply.py`).
  - The common-surname list that never counts toward a ≥ 2-shared-surname link is **widened to P4's own list**
    (`P4_check_candidates.py` COMMON plus liu / zhang / zhou / xu).
  - Single-PI links exclude only the four named surnames.
  - The Family-Given order is repaired.
  - The institution-link (P) rule is applied consistently: CSM-Shafer, Osaka-Shinohara and Kanazawa-Imura become P;
    P4-025 is in-corpus (Y); P4-063 moves E1 → E2.
  - Second-order links (KULeuven-Binnemans via P4-026; Mendeleev/Frumkin-Stepanov via P4-001) are a sensitivity, not
    errors.
  - Conservative laboratory = connected component of L2 or LPI links; liberal laboratory = the senior person. The key is
    sealed **PROVISIONAL** pending the U6 review.
  - **The conservative laboratory key decides** every claim's unit count and the supply audit; liberal counts are
    reported beside it as a sensitivity. *Reason:* merging can only lower G and puts dependent groups into one unit, so
    the decision stays conservative with respect to both the minimum-G gate and the independence the sign-flip assumes.
- **Decision, chemotypes of candidates.** At Stage 0 the supply counts use P4's hand chemotype labels with [A336][NO3] =
  TOMANO3 merged (`chemotype_label_merged`), not the frozen gen6 map (brief 5.5).
  - *Reason:* no rebuilt SMILES exist for the candidates before the label-blind audit; P4 and PC P4 treat hand-label
    new-chemotype counts as upper bounds (the map can join candidates to corpus clusters). Whether the map can also split
    a broad hand label (for example "neutral phosphine oxide") is not established, so "upper bound" is P4's reading,
    not a proof.
  - *Sensitivity:* the verified decision units each hold a single extractant, so the D4 decision supply (2 / 1 / 5 / 3)
    and the kill cannot change under the map; forecast and all-convert chemotype counts may move in either direction
    and never decide. Fixed by the gen6 map at the U5 Phase-A audit or when values arrive (DATA_CONTRACT §9).
- **Counts** (conservative / liberal laboratories):

  | reference set | primary (widened list) | sensitivity (liu / zhang / zhou / xu only) |
  |---|---|---|
  | all registry keys (135 references) | 45 / 85 | 33 / 85 |
  | V1 cohort DOIs (58) | 24 / 44 | 20 / 44 |
  | C3 DOIs (39) | 17 / 31 | 16 / 31 |

  The E-CT annotation gives PRIMARY 12 publications = **7 / 11** laboratories and the V1 n ≥ 1 pool 19 publications = 9 /
  16. The corpus link graph has 329 links (L2 188; LPI 220). The U6 review list holds the 141 single-PI-only links and 25
  hand-vs-automatic disagreements.
- **Decision, planning amplitudes.** HDEHP-like **-0.26** (an unverified planning value [I2 confirm-1]) and P507-like
  **-0.105** (an unverified search-snippet reading [I2 confirm-2]) are sealed **PROVISIONAL / PENDING-U5**, with power
  reported for both centres. They are fixed from primary SF tables by append-only amendment before any new value is read
  (brief 4.3 item 10, 4.6).
- **Power at these centres** (CGD_W, β_frozen; S0 `power_worlds.json`):
  - HDEHP-like at G = 6: 0.8371 (sd 0.10), 0.7078 (0.12), 0.592 (0.14), 0.4936 (0.16); at G = 8: 0.9513 to 0.6463.
  - P507-like at G = 6: 0.0188-0.0203, about the size, because -0.105 sits next to β_frozen.
  - The brief's "0.90-1.00 at G = 6" for the HDEHP-like centre corresponds to β ≈ -0.094 at sd 0.10 (0.9034), not to
    β_frozen.
- **Reason.** U6 and U5 were unanswered at Stage 0. Brief 4.6 prescribes the PROVISIONAL seal with conservative and
  liberal counts and with both centres. The four-surname list merges distinct Chinese groups through li / wang / yang /
  sun, which is why the wider list is primary (`r5/supply.py` docstring).
- **Sensitivity.** The four-surname list (above); liberal vs conservative counts; second-order links.

### D7. Every annex and appendix row computed by an item's code is PLANNED (none EXCLUDED)

- **Decision.** The following are PLANNED at the seal:
  - 1.0-P01 (the M08 appendix);
  - 1.1-X01 to X06 (shift-1, recipe-1 E-REC, M02, M04, M06, laboratory annotation);
  - 1.5-X01 to X04 (M16 interval line, M23 bounds, measurements saved, E-REC own-prior row) and 1.5-P01 (E-RANK);
  - 1.6-X01 (recipe-2 row);
  - 1.7-X01 (the substitution-trap row).
- **Reason.** Brief 4.1 item 8, 6.1 and 6.10: every annex or appendix row is PLANNED or EXCLUDED at the seal, and running
  or skipping one after other results are seen would be post hoc. Each is cheap and already implemented in its item's
  code, so excluding any would be a choice without a reason.
- **Sensitivity.** None. This is a status choice; its cost is compute minutes. Arms that an ITEM_SPEC excludes from its
  code stay EXCLUDED (`REGISTRY.csv` 1.2-E01/E02, 1.3-E01 to E07, 1.8-E01 to E04): they are not annex rows computed by
  the code.

### D8. Item design choices of each ITEM_SPEC.md are adopted as frozen

- **Decision.** Every choice written in `results/{EA_CHECK, ECT, ECT_M03, ELAB, WINDOW, EDEC, ED, EW, EKX}/ITEM_SPEC.md`
  is frozen with the item code, unless the pre-seal reviewer finds it contrary to the brief. In that case it is changed by
  amendment before that item's seal. The main choices the brief left open:
  - 1.0: big fold = the fold with the most scored rows per seed; `data_location` classes by regular expression.
    **Reviewer exception (s0-fidelity-11):** the P5 F1 label is applied (DATA_CONTRACT §1: the 10 comment-located C3
    cells of pub_b9dfc468cd, pub_19783c039b and pub_f8a621ede4 are located), and the raw `cells.csv` class is shown
    beside it.
  - 1.1: spec 12 items 1-8. OFF's base is the recomputed B1_r4; placebos and OFF exist only for n_sib ≥ 1; registry
    seeds are used for SUB m and count-matched subsets; NP_LAD is the frozen E-REC no-prior policy; the M04 window is
    [-2, 2]; quantised = Q_SMALLINT or Q_COARSE; offset copies = OFFSET or NEAR_ID; exact enumeration up to 22 units.
  - 1.2: spec 13 items 1-10. Laplace kernels; grids ℓ_acid {∞, 1, 0.5, 0.25, 0.1} and ℓ_read {∞, 0.4, 0.2, 0.1, 0.05};
    two shrinkage levels; 4 reads only; the paired one-SE rule is primary; the level rule uses raw forms; LINHALF λ = 0.5.
  - 1.3: same-series = a shared `experiment_series_id`; quarantine = bit plus near-identity links; scope via the frozen
    class map; C5_in primary.
  - 1.4: OP75 is the primary operating point; nested criterion = maximum inside R² at training coverage ≥ 0.70; the CELL
    control is primary.
  - 1.5: `LTT_a10_hull_glob` is primary; hull = the bracketing rule; τ grid of 600 values; M23 limit -4.0.
  - 1.6: BISECT_Z is added to the M17 family; saturation consistency rule (< 0.48 with ≥ 3 clusters, or spread ≥ 0.05);
    one shared cluster-bootstrap matrix (seed 8675309).
  - 1.7: 33 cut pair types; Pr/Dy has ΔZ 7 (brief 6.7 writes 5-6); the giant cluster is removed from the Nd/Sm contrast.
  - 1.8: n = 12 donor publications, so w = 0.705882 and u_ext = -0.077647; E-KX-LOPO (B1 base = the BP OOF prediction,
    brief 6.8; never called BP), the BPG-base variant a sensitivity; C5_out primary; B = 2,000 permutations.
- **Reason.** The specs were written against the brief before any round-5 score. Freezing them now is what prevents a
  post hoc choice.
- **Sensitivity.** Each spec registers its own sensitivity rows (`REGISTRY.csv` kind `sensitivity`).

### D9. Chord hygiene: V2-5 censoring primary for Gd chords; the resolution rule flags only (pre-seal review)

- **Decision.** A Gd chord is not computed (censored), and the unit loses that cell, when any of its Eu / Gd / Tb
  readings is at or below a source-stated DL / LOQ **or** matches the round-3 V2-5 floor pattern (D < 1.5·10^e) (brief
  4.3 item 11; I2 confirm-3 "Apply V2-5 censoring and the resolution rule"). The chord without V2-5 censoring is a
  sensitivity. This chord-specific rule overrides the general new-data rule of DATA_CONTRACT §4 item 2 for the Gd
  claims only. The resolution rule is a frozen interpretation: it flags cells whose quantisation half-width bound is
  ≥ |β_frozen|/2 = 0.0554; the primary keeps flagged cells and a sensitivity drops them.
- **Reason.** Brief 4.3 item 11 seals V2-5 floor censoring for chords, and PROTOCOL / this log already listed it as
  sealed; CONFIRMATION_PACKAGE §13 had moved it to a sensitivity. The brief does not specify an action for the
  resolution rule, and dropping cells would cut an already sub-minimum supply.
- **Sensitivity.** The chord without V2-5 censoring; the resolution-flagged cells dropped.

### D10. The frozen predictors keep the Gd term in every medium; M29 is a reading rule (pre-seal review)

- **Decision.** Every Gd claim is defined, and the Gd term interpreted, only for nitrate and inert-salt media (IR M29,
  adopted unscored); the frozen predictors B1 (fold β in-corpus, β_frozen on new data) and SH carry the Gd term
  unchanged in every stratum, aqueous-complexant media included (C-ECT admits those changes as T1).
- **Reason.** M29 is adopted unscored, and a medium switch is a new predictor that the drop-one kill (78.5 %, one
  paper) did not support. Changing the comparator per medium would add an unregistered arm.
- **Sensitivity.** "B1@4 and SH with u = 0 for AQ_COMPLEXANT (medium map M01 / M02) targets", registered now as a
  PLANNED C-ECT sensitivity on new data only (REGISTRY `C-ECT-S01`; CONFIRMATION_PACKAGE §4.2, §14), with the
  complexant stratum reported separately. It is registered before any new value exists, so it is not post hoc.

### D11. Deferred Stage-0 inputs: the external evaluator and the HDEHP acid threshold (pre-seal review)

- **Decision.** Two Stage-0 inputs are deferred and listed in DATA_CONTRACT §9: (1) the external confirmation evaluator
  (sealed family predictions → family, laboratory and unit values; chords), specified by CONFIRMATION_PACKAGE §4.2,
  §5, §13, and written, tested on synthetic families and sealed by amendment before any new value is read; (2) the
  HDEHP solvating-regime acid threshold of C-GD eligibility (CONFIRMATION_PACKAGE §10), PENDING-U5, fixed from primary
  SF tables (U5 item 4) by amendment before any new value is read.
- **Reason.** No claim can reach its minimum G under the current supply (the kill fires, D4), and writing the
  evaluator now would freeze choices that U5 / U6 may still change. Brief 4.3 item 12 asks for the freeze before any
  new value is read, which an amendment-sealed evaluator satisfies.
- **Sensitivity.** None: nothing is scored until both are sealed.

### Notes found while writing this log (resolved at the pre-seal review; none changes a decision)

- **N1. β_frozen digest drift: resolved.** `size_gate.json` and `power_worlds.json` had recorded β_frozen =
  -0.11080937578907442 and the LF SHA-256 `7f6025fe…7ed7156` of an earlier `beta_frozen.json`; the current file holds
  -0.11080937578907438 (LF SHA-256 `2b5d0e2d…cc6264`), 4e-17 apart. At the pre-seal review `s0_sizegate.py`,
  `s0_sizegate_ect_chemotype.py`, `s0_power_worlds.py`, `s0_supply.py` and `s0_package_supplements.py` were rerun in
  that order (synthetic or metadata only): both files now record the current value and digest, every quoted size and
  power value is unchanged, and `power_cgd_beta_sens.json` again reproduces the 768 β_frozen rows of the new
  `power_worlds.csv` exactly (maximum difference 0.0).
- **N2. Resolved.** D2's power sensitivities are in S0 `power_cgd_beta_sens.json` (`scripts/s0_package_supplements.py`,
  written 07:45:59 UTC).
- **N3. Resolved.** The C-ECT chemotype-level size check is in S0 `size_gate_ect_chemotype.json`
  (`scripts/s0_sizegate_ect_chemotype.py`, written 07:42:12 UTC): it passes (D3).
- **N4. Two readings of decision supply: resolved.** The audit code now applies D4 (`decision_supply` = in hand); the
  brief's bracket is the separate, never-deciding field `bracket_high`. The kill outcome is identical under both.
- **N5. Not covered by the size gate** (S0 `size_gate.json` notes):
  - C-GD's secondary per-unit Gd-pair SSE sign-flip, and the A3 positive control (ungated by design);
  - the C-GD "≥ 3 families among ≥ 6 laboratories" constraint, which is also not simulated in the power worlds.

---

## 5. Stage-2 claims (status at the Stage-0 seal)

| claim | statistic (sealed in `CONFIRMATION_PACKAGE.md`) | status |
|---|---|---|
| C-ECT | chemotype-unit exact sign-flip (D3 (ii); passes its fresh size check); fixed sequence SH@4 then SH@3 | NOT RUN: supply-audit kill (D4) |
| Gd-dip transport (M33) with C-GD as its cation-exchanger stratum | descriptive intervals (D3 (i)) against β_frozen and β_frozen/2 | NOT RUN: supply-audit kill (D4) |
| A3 positive control (not a claim) | B1 - B0 prospective, equal-weight log SSE ratio with the pooled guard, ungated | NOT RUN: no new data |

---

## 6. Data-contract and inference choices recorded with their sources (not new decisions)

- **Duplicate rule.** The frozen 5.1 bit-identity rule (≥ 3 shared metals float64-identical and ≥ 50 % of the shared
  finite metals) flags 6 links on V1 (S0 `dupscan_summary.json`):
  - item 1 (13 of 14), item 3 (3 of 6), item 4 (8 of 8);
  - three 8-of-8 same-extractant pairs in pub_b9dfc468cd, each a V2-2 keep / drop pair;
  - none across publications.
  - Known items 1, 3 and 4 and the unflagged 4- and 5-of-14 links of af971a0b699f29ae pass their checks.
- **E-CT PRIMARY.** 99 cells / 12 publications / 14 units / 690 rows per seed, with and without the bit rule (S0
  `ect_population.json`). All seven P1 count targets match (`p1_reproduction_counts`), among them V1_GE1 (149 / 19 / 22 /
  1,111), PRIMARY_NOBIT (99 / 12 / 14 / 690), WITH_C5_NOBIT (114 / 13 / 15 / 838) and EQ1_NOBIT (16 / 8 / 9 / 130).
- **Labels.** Every in-corpus number carries EXPLORATORY (seen). DIAGNOSTIC marks E-A contrasts after M34's kill and the
  breadth tests. CONSISTENCY CHECK marks in-corpus thresholds below seen values. SENSITIVITY, ORACLE [O] and TECHNICAL
  SMOKE are used as defined in brief 9.1. No in-corpus p-value is called significant.

---

## 7. Stage-1 report rules (sealed rules; every outcome "pending")

The rules below are sealed with this file. Each outcome is recorded **below the marker** as an append-only entry with
its label and UTC date. The "outcome" column here stays "pending", so the seal still verifies.

| id | item | rule | label | outcome |
|---|---|---|---|---|
| 1.0-R01 | EA_CHECK | M35 positive control: C1 B1 - B0 on BPG, C5_in passes rel SSE ≥ 0.02, SF_POOL p ≤ 0.025 and guard ≥ -0.001 → `M35_passes_its_positive_control`; else `M35_kill_fires` (convention abandoned) | DIAGNOSTIC | pending |
| 1.0-R02 | EA_CHECK | C2 HIST_D7 - B1 gate outcome | DIAGNOSTIC | pending |
| 1.0-R03 | EA_CHECK | replay match: C1 BPG +5.85 % and p 0.00317; BP p 0.00269 (a mismatch is reported, not a stop) | CONSISTENCY CHECK | pending |
| 1.0-R04 | EA_CHECK | breadth diagnostic SMALL_POOLED (SF_LOG, IM t), never a gate | DIAGNOSTIC | pending |
| 1.0-R05 | EA_CHECK | M08 class "identified" iff ≥ 3 product units and top unit ≤ 80 % of the products | M08 appendix | pending |
| 1.1-R01 | ECT | SH - B1 on PRIMARY with LOPO interval (also PRIMARY_NOBIT, PRIMARY_I1); reported | EXPLORATORY (seen) | pending |
| 1.1-R02 | ECT | harm rule (ii): EQ1 rel SSE of SH vs B1 ≥ -0.05, else "the deployed rule depends on sibling count" | CONSISTENCY CHECK | pending |
| 1.1-R03 | ECT | artefact flag (iii): Ce-pair share of the SH - B1 gain ≥ 0.50 → "transfer of a laboratory fingerprint" | CONSISTENCY CHECK | pending |
| 1.1-R04 | ECT | mechanism wording from SH - OFF_XEXT and SH - PB_PUB per publication; "campaign fingerprint" needs ≥ 4 families plus a pre-set threshold | CONSISTENCY CHECK | pending |
| 1.1-R05 | ECT | claim wording from SH vs SIB and SH vs NEAR unit W/L/T: "own-series template + 3-4 reads", not "K = 1.78" | CONSISTENCY CHECK | pending |
| 1.1-R06 | ECT | SH with 2 reads stays excluded from claims (units and top-3 share reported) | CONSISTENCY CHECK | pending |
| 1.1-R07 | ECT | E-CT values never next to 0.452; B1 on the same rows | rule | pending |
| 1.2-R01 | ECT_M03 | deploy rule (BOTH, PRIMARY_NO_ORNL): uniform SH if dR² < +0.01, or < 7 of 11 publications improve, or uniform is chosen in more than half the folds | CONSISTENCY CHECK | pending |
| 1.2-R02 | ECT_M03 | deploy-rule sensitivities (ACID / READ; PRIMARY; WITH_C5; PRIMARY_NOBIT) | SENSITIVITY | pending |
| 1.2-R03 | ECT_M03 | level rule retired if LEVEL2_raw beats both SIB_r4 and NEAR2_raw in < 60 % of publications with an n_sib ≥ 5 target | CONSISTENCY CHECK | pending |
| 1.2-R04 | ECT_M03 | acid-trend statistic R²(LINHALF) - max(R²(MEAN), R²(PAIRMEAN)), bracketed rows, LOPO; no claim | EXPLORATORY (seen) | pending |
| 1.2-R05 | ECT_M03 | B1_r4 and SH_r4 on the same rows; never next to 0.452 | rule | pending |
| 1.3-R01 | ELAB | headline MX_LM10 - B1 on PRIMARY (LOPO, W/L, SF_POOL, top-3) | EXPLORATORY (seen); best of about 25 | pending |
| 1.3-R02 | ELAB | bridges MX_LM10_NB - B1 (SRC), MX_LM10_NC - B1 | SENSITIVITY | pending |
| 1.3-R03 | ELAB | POPX0 MX_LM10 - B1 | EXPLORATORY (seen) | pending |
| 1.3-R04 | ELAB | controls MX_LM10 - E_PBC / E_PCM / E_RM | EXPLORATORY (seen) | pending |
| 1.3-R05 | ELAB | MX_LM10 - best global K [O]; MX_LM10 - MX_GK_LOPO | ORACLE [O] / EXPLORATORY | pending |
| 1.3-R06 | ELAB | leave-two-publications-out; top-3 share | EXPLORATORY (seen) | pending |
| 1.3-R07 | ELAB | same-series vs other-series | EXPLORATORY (seen) | pending |
| 1.3-R08 | ELAB | low-D band split | DIAGNOSTIC | pending |
| 1.3-R09 | ELAB | external-kill statistic read in-corpus: "would kill / pass if applied"; not applied | DIAGNOSTIC | pending |
| 1.3-R10 | ELAB | narrowing reading: MX_LM10 - E_PCM ≤ 0 → "level-conditional weighting, not campaign" | CONSISTENCY CHECK | pending |
| 1.3-R11 | ELAB | harm-flag reading: a chem_family class with ≥ 3 conservative laboratory units losing > 5 % | CONSISTENCY CHECK | pending |
| 1.3-R12 | ELAB | scope: SCOPE_IN vs NULL_PRIOR | EXPLORATORY (seen) | pending |
| 1.3-R13 | ELAB | external leg (≥ 8 independent laboratories; kill / narrow / harm rules) | data-gated | pending |
| 1.3-R14 | ELAB | every number labelled "best of about 25 arms on seen rows; expect regression" | rule | pending |
| 1.4-R01 | WINDOW | "row flag" only if NEST_FAM inside rel SSE > 97.5 % quantile of the CELL control; else "cell or condition flag" | CONSISTENCY CHECK | pending |
| 1.4-R02 | WINDOW | tail flag passes if at OP75 the \|res\| > 0.5 ratio ≥ 3 with bootstrap lower bound > 1, all rows and without C5-BTBP | CONSISTENCY CHECK | pending |
| 1.4-R03 | WINDOW | bulk precision passes if the q80 narrowing (PTW vs PT) ≥ 15 % at OP75 (expected to fail) | CONSISTENCY CHECK | pending |
| 1.5-R01 | EDEC | descriptive item: no kill, no threshold; never an R² gain | EXPLORATORY (seen) | pending |
| 1.5-R02 | EDEC | hull rule: no calls outside the read hull; no Tm/Yb or Yb/Lu calls without bracketing reads | rule | pending |
| 1.5-R03 | EDEC | primary calibration LTT at risk 0.10, hull + global, fixed before the item seal | rule | pending |
| 1.6-R01 | ED | Pareto question: does any policy keep every block > -0.002 and improve one block with cluster lower bound > 0? | EXPLORATORY (seen) | pending |
| 1.6-R02 | ED | saturation consistency check: fires if a stratum with ≥ 3 clusters is < 0.48 or the spread is ≥ 0.05 | CONSISTENCY CHECK | pending |
| 1.7-R01 | EW | Nd/Sm B1 - LINE (clusters with ≥ 10 rows, giant removed, LOCO jackknife-t 95 %): retired if ≤ +0.05, or lower bound ≤ 0, or < 2 clusters | CONSISTENCY CHECK | pending |
| 1.7-R02 | EW | heavy cuts are explicit NON-CLAIMS, by whether Tm or Lu is read | NON-CLAIM | pending |
| 1.7-R03 | EW | seen-value table (IR M13 beside the recomputed values) | CONSISTENCY CHECK | pending |
| 1.8-R01 | EKX | narrowed to "universal Nd term" if R²(KX_CHEM) - R²(NDU) ≤ 0 with the top-3 publications removed (BP, C5_out, POOLED) | CONSISTENCY CHECK | pending |
| 1.8-R02 | EKX | rule sensitivities (C5_in; SC009_PRND; rows and donors; BPG base) | SENSITIVITY | pending |
| 1.8-R03 | EKX | external sign rider (first 4 eligible new sc009 publications) | data-gated | pending |
| 1.9-R01 | GUIDANCE | every line carries ESTABLISHED / EXPLORATORY / HYPOTHESIS and its cost scenario | rule | pending |

**Reproduction targets (TECHNICAL SMOKE; a mismatch stops the item; run after the seal):**

| id | target | outcome |
|---|---|---|
| 1.0-T01 | B1 / B0 / HIST_D7 rebuilt from round-4 fold β = REF within 8.9e-16 on 18,020 rows | pending |
| 1.1-T01 / T02 / T03 | B1 identity (≤ 8.9e-16); E6 V1 population SIB 0.5734, SH (K = 2) 0.5946, B1 0.4489; P1 PRIMARY_NOBIT B1 0.5309, SH 0.7082, SIB 0.6987, OFF 0.6963 (5e-5) | pending |
| 1.2-T01 ... T04 | B1 identity (≤ 8.9e-16); shared arms = r5.ect exactly; kernel uniform limit; P1 PRIMARY_NOBIT through the kernel code | pending |
| 1.3-T01 ... T04 | B1 identity; POPX 92 / 16 / 787 and POPX0 60 / 13 / 534; MX_LM10_NB = lab_rows2 within 1e-12; BPG B1 0.444, MX_LM10_NB 0.494 | pending |
| 1.4-T01 / T02 | B1 identity; EB Appendix 2 window (coverage 0.743-0.754, inside R² 0.540 BPG, 0.538 BP; 0.0005) | pending |
| 1.5-T01 | B1 identity; D7_r4 = REF HIST_D7 within 1e-12; y identical | pending |
| 1.6-T01 / T02 | B1 identity; P3 5-read B1 C3 0.4968 / 0.4988, C4 0.4798 / 0.4821 (5e-5) | pending |
| 1.7-T01 | B1 identity on STRICT rows | pending |
| 1.8-T01 / T02 | B1 identity; construction identities (1e-12) | pending |

<!-- APPEND-ONLY BELOW -->

## Request status (append-only; brief 4.6, 7.0)

### 2026-10-24 (UTC), initial status at Stage 0

DR's requests were shown to the user once, in rank order, on 2026-10-24 (RUN_RECORD 04:42:20 UTC; `DATA_REQUEST.md`).
None has been answered. Every request is **OPEN**. None is paraphrased as granted. The Stage-0 inputs that depend on
U5 item 4 and U6 are sealed PROVISIONAL (D6).

| id | DR rank | request | used by (brief) | status |
|---|---|---|---|---|
| U1-a | 1 | say which publishers you can read through an institution | 7.1 | OPEN |
| U1-b | 2 | allow one download: Shimada 2004 thesis (K0379.pdf), into a quarantine folder | 7.1 | OPEN |
| U1-c | 3 | allow web reading (no downloads) of 6 open-access import papers | 7.1 | OPEN |
| U1-d | 5 | allow one download: the JACS Au 2022 SI spreadsheet | 7.1 | OPEN |
| U1-e | 9 | with Taylor & Francis access: DMDPhPDA Table 1 and Figure 5 | 7.1 | OPEN |
| U1-f | 11 | one e-mail (drafted by the agent, sent by the user) to the SAFE curator | 7.1 | OPEN |
| U1-g | 12 | with access: the two C5-BTBP papers (Foreman 2005, Nilsson 2006) | 7.1 | OPEN |
| U1-h | 13 | with access: the 15 other closed import papers, in the seeded census order | 7.1 | OPEN |
| U1-i | 14 | optional contacts: Chalmers, NEA EGFRC (IDEaL), JAEA | 7.1 | OPEN |
| U2 | 6 | SAFE gap-ID export, metadata only (M33 supply gate) | 7.2 | OPEN |
| U3 | 7 | allow fetching the SI of Molecules 29:3548 (E-CT dry run; corpus-linked laboratory) | 7.3 | OPEN |
| U4 | 15 | a laboratory collaborator (M11 re-measurement, M12 reproducibility) | 7.4 | OPEN |
| U5 | 4 | web reading for the literature supply, E2 verification, targeted E-CT search, planning amplitudes | 4.3 item 10, 7.5 | OPEN (amplitudes PENDING-U5) |
| U6 | 8 | review the ambiguous links of the draft laboratory key | 5.5 | OPEN (key PROVISIONAL) |
| S1 | 10 | sign off the cohort-key remedy for the 11 V2-3 cells | 5.3 | OPEN |
| H1 | not ranked | housekeeping: optionally delete PDFs that tools auto-cached outside the repository | none (hygiene) | OPEN |

## Stage-1 report-rule outcomes (append-only; brief §10; seal b96c3bef9828, amendment A2)

### 2026-10-24 (UTC), outcomes after the A2 rerun

These are the outcomes of the sealed §7 rules, from the A2 rerun outputs in `results/<ITEM>/`. BPG decides and BP is a
SENSITIVITY, except E-KX, whose rule is read on E-KX-LOPO (BP OOF base, D8). Every value is EXPLORATORY (seen) and
carries the label given. The first-pass outcomes were identical except 1.6-R01 (corrected by A2) and the WINDOW lower
bounds (shifted by at most 0.039; no outcome change).

| id | outcome | label |
|---|---|---|
| 1.0-R01 | `M35_passes_its_positive_control`. B1 - B0 on BPG: rel SSE +0.0586, SF_POOL p 0.00317, guard +0.0050. `M35_kill_fires` = false. BP +0.0585, p 0.00269: passes | DIAGNOSTIC |
| 1.0-R02 | HIST_D7 - B1 fails the gate: BPG rel SSE -0.0019, p 0.660, guard -0.00004; BP -0.0025, p 0.696 | DIAGNOSTIC |
| 1.0-R03 | the p legs match (BPG 0.00317, BP 0.00269). The rel SSE leg reports match = false: 5.86 % seed mean vs the seen 5.85 %, which pools over seeds (0.05854); the difference is definitional | CONSISTENCY CHECK |
| 1.0-R04 | SMALL_POOLED B1 - B0: G 6, 6/0/0, SF_LOG p 0.0156, IM t 3.08 (BP 3.24). HIST_D7 - B1: 3/3/0, SF_LOG p 0.86. Never a gate | DIAGNOSTIC |
| 1.0-R05 | identified with C5 in: SE_SP (ρ 0.329), OE_SP_OC, OE_OP_OC. Not identified: SE_OP, OE_SP_SC, OE_OP_SC. Without C5-BTBP, SE_SP is not identified (top unit 87 %) | M08 appendix |
| 1.1-R01 | SH - B1 on PRIMARY: +0.1693 [LOPO +0.0553, +0.2833], 14 of 14 units up, B1 0.5309 on the same rows; BP +0.1658. PRIMARY_NOBIT +0.1774 and PRIMARY_I1 +0.1682 (SENSITIVITY) | EXPLORATORY (seen) |
| 1.1-R02 | passes: EQ1 rel SSE of SH vs B1 +0.122 ≥ -0.05 (16 cells, 8 publications; BP +0.122) | CONSISTENCY CHECK |
| 1.1-R03 | not triggered: Ce-pair share 0.334 < 0.50 (BP 0.341) | CONSISTENCY CHECK |
| 1.1-R04 | the default wording applies: "same-campaign information (extractant transfer and campaign fingerprint not separated)". SH - OFF_XEXT +0.0445 and SH - PB_PUB +0.0452, on 6 families in 3 publications; no threshold was set before data | CONSISTENCY CHECK |
| 1.1-R05 | the claim reads "own-series template + 3-4 reads", not "K = 1.78". SH vs SIB 9/5/0, SIGN p 0.212; SH vs NEAR 8/6/0, SIGN p 0.395 | CONSISTENCY CHECK |
| 1.1-R06 | SH with 2 reads stays excluded from claims: SH@2 - B1@4 +0.0746, 8 of 14 units, top-3 share 0.99 | CONSISTENCY CHECK |
| 1.1-R07 | applied: B1 on the same rows (0.5309) is the comparator; 0.452 never appears beside an E-CT value | rule |
| 1.2-R01 | nested weighting passes on PRIMARY_NO_ORNL. BOTH: dR2 +0.0203, 9 of 11 publications improve, uniform chosen in 0 of 11 folds; BP +0.0197. No claim; C-ECT stays sealed on SH@4 vs B1@4 | CONSISTENCY CHECK |
| 1.2-R02 | ACID family: deploy uniform SH (-0.0028, 0 of 11, uniform 10 of 11). READ family passes (+0.0246, 10 of 11). PRIMARY +0.0241, WITH_C5 +0.0499 and PRIMARY_NOBIT +0.0237 all pass | SENSITIVITY |
| 1.2-R03 | level rule retired: LEVEL2_raw beats both comparators in 1 of 5 publications (0.20 < 0.60); shrunk form 0.40 | CONSISTENCY CHECK |
| 1.2-R04 | +0.0048 [LOPO t(7) -0.0571, +0.0668]; positive, but the interval includes 0. No claim | EXPLORATORY (seen) |
| 1.2-R05 | applied | rule |
| 1.3-R01 | MX_LM10 - B1 +0.0763 [LOPO -0.0110, +0.1637]. Conservative laboratories (PROVISIONAL key) 9/0/0, SF_POOL p 0.0020, top-3 lab share 0.902. BP +0.0780, 8/1/0 | EXPLORATORY (seen); best of about 25 arms |
| 1.3-R02 | MX_LM10_NB - B1 (SRC) +0.0896 [+0.0031, +0.1761]. MX_LM10_NC equals MX_LM10 bitwise (+0.0763) | SENSITIVITY |
| 1.3-R03 | POPX0 MX_LM10 - B1 +0.0944 [LOPO +0.0062, +0.1827], publications 8/3/2 | EXPLORATORY (seen) |
| 1.3-R04 | vs E_PBC +0.0908; vs E_PCM +0.1003; vs E_RM +0.0179 [-0.0044, +0.0402] | EXPLORATORY (seen) |
| 1.3-R05 | vs best global K +0.0424 [O]; vs MX_GK_LOPO +0.0643 | ORACLE [O] / EXPLORATORY (seen) |
| 1.3-R06 | leave-two-publications-out minimum +0.0061; top-3 publication share 1.035 | EXPLORATORY (seen) |
| 1.3-R07 | SS_OS (35 cells, 4 publications, 1 cluster): SS - B1 +0.0750 (4/0/0); OS - B1 -0.0793 (2/1/1); SS - OS +0.1429 | EXPLORATORY (seen) |
| 1.3-R08 | rows LOWD -0.0149, HIGHD +0.1020 | DIAGNOSTIC |
| 1.3-R09 | would pass if applied: lab_cons median +0.0706 with lower bound +0.0334; lab_lib +0.0493 / +0.0058. Not applied | DIAGNOSTIC |
| 1.3-R10 | would not narrow (MX_LM10 - E_PCM +0.1003 > 0) | CONSISTENCY CHECK |
| 1.3-R11 | none | CONSISTENCY CHECK |
| 1.3-R12 | SCOPE_IN +0.0847; NULL_PRIOR +0.0712 | EXPLORATORY (seen) |
| 1.3-R13 | NOT RUN (data-gated) | data-gated |
| 1.3-R14 | applied | rule |
| 1.4-R01 | ROW FLAG on B1's inside relative MSE. NEST_FAM 0.2293 > CELL q97.5 0.2182; BP 0.2287 > 0.2155. NEST_FAM's inside R² is below its CELL control mean | CONSISTENCY CHECK |
| 1.4-R02 | PASS at OP75 on BPG: all rows 3.797 [1.929, 8.037]; without C5-BTBP 3.663 [1.197, 7.483]. FAIL on BP: without C5-BTBP the lower bound is 0.965 | CONSISTENCY CHECK |
| 1.4-R03 | FAIL (expected): q80 narrowing 0.091 < 0.15 at OP75 (BP 0.102) | CONSISTENCY CHECK |
| 1.5-R01 | descriptive: no kill, no threshold, never an R² gain | EXPLORATORY (seen) |
| 1.5-R02 | satisfied: 0 calls outside the read hull; no Tm/Yb or Yb/Lu call without bracketing reads | rule |
| 1.5-R03 | applied: LTT at per-call risk 0.10, hull + global thresholds, n_min 22, δ 0.10, fixed sequence | rule |
| 1.6-R01 | no member of the written label-free family Pareto-dominates DOPTQ, in BPG and BP and in all / noC5 / noC5cell. ORACLE_B1 [O] meets the criterion only as a bound. Corrected under A2 | EXPLORATORY (seen) |
| 1.6-R02 | fires in BPG and BP: k_max 4 stratum at 5 reads 0.4440 / 0.4466 (4 clusters); spread 0.158 / 0.156. A consistency check, not a kill | CONSISTENCY CHECK |
| 1.7-R01 | RETIRED: BPG -0.0951 [-0.7312, +0.5409], G = 3; BP -0.0971. Narrowed to "a line through 4 reads suffices at Nd/Sm outside the giant cluster" | CONSISTENCY CHECK |
| 1.7-R02 | applied: heavy cuts reported as NON-CLAIMs, by whether Tm or Lu is read | NON-CLAIM |
| 1.7-R03 | reported (`results/EW/seen_<design>.csv`) | CONSISTENCY CHECK |
| 1.8-R01 | narrowed to "universal Nd term": E-KX-LOPO, C5_out, POOLED, top-3 publications removed, R²(KX_CHEM) - R²(NDU) = -0.0012 | CONSISTENCY CHECK |
| 1.8-R02 | 6 of 9 sensitivities narrow. Not narrowed: SC009_PRND rows and donors (+0.0094); with the BPG base, SC009_PRND rows (+0.0057) and rows and donors (+0.0151). Every POOLED row narrows (BPG base -0.0017) | SENSITIVITY |
| 1.8-R03 | NOT RUN (data-gated) | data-gated |
| 1.9-R01 | applied: every line of `PRACTICAL_GUIDANCE.md` carries its evidence label and its cost scenario | rule |

**Reproduction targets (TECHNICAL SMOKE), all matched under seal b96c3bef9828:**

| id | outcome |
|---|---|
| 1.0-T01 | B1 / B0 / HIST_D7 = REF, max diff 0.0 on 18,020 rows |
| 1.1-T01 / T02 / T03 | B1 identity 8.9e-16. V1 population: SIB 0.5734, SH (K = 2) 0.5946, B1 0.4489. PRIMARY_NOBIT: B1 0.5309, SH 0.7082, SIB 0.6987, OFF 0.6963, all within 5e-5. The sealed P1 pytest passed |
| 1.2-T01 ... T04 | B1 identity; shared arms = r5.ect exactly; kernel uniform limit 0.0; P1 through the kernel code matched |
| 1.3-T01 ... T04 | matched (`results/ELAB/repro_<design>.json`) |
| 1.4-T01 / T02 | matched (`results/WINDOW/repro_<design>.json`) |
| 1.5-T01 | matched (`results/EDEC/repro_<design>.json`) |
| 1.6-T01 / T02 | B1 identity 0.0. P3 5-read targets: C3 0.4968 / 0.4988; C4 0.4798 / 0.4821 (5e-5). The A2 gate also checks the 5-read B0 = R3-B0 identity and the 4-read B0 / HIST_D7 identities: all 0.0 |
| 1.7-T01 | B1 identity on STRICT rows, 0.0 |
| 1.8-T01 / T02 | B1 identity 0.0; construction identities ≤ 1.0e-15 |
