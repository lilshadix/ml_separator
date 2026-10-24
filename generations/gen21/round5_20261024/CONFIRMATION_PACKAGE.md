# Confirmation package (CP-13), round 5

Status: **Stage-0 document, written before the Stage-0 seal and revised at the pre-seal review
(`results/stage0/PRESEAL_REVIEW.md`); not sealed.** No round-5 score had been computed when it was written. The package is the frozen frame of brief §4.3 with the claims of brief §7.5. It is scored **once**, only
on data that arrive **after** the seal, and only for claims that pass the supply audit. Inputs marked **PROVISIONAL**
(laboratory key, planning amplitudes, chemotype counts) are fixed only by an append-only amendment made **before any new
value is read** (brief §4.6).

**Current outcome (§9): the supply-audit kill fires.** Every claim is below its minimum G on the in-hand supply, so no
confirmation runs and round 5 is reported as descriptive (brief §7.6). The package stays sealed and waiting; the audit
reruns after each answered request (append-only section at the end).

Sources (paths under `generations/gen21/`): "brief" = `round5_design_20261023/CLAUDE_CODE_PROMPT.md`; IR, I2, P1–P5,
PC, DR as in the brief's source table; "S0:" = a file in `round5_20261024/results/stage0/`. Decisions D1–D11 (D9–D11 added at the pre-seal review) are recorded
with their reasons and sensitivities in `DECISION_LOG.md`; this file applies them. Every number below is a Stage-0
planning value (synthetic or seen-corpus worlds, metadata counts) or a seen value quoted from the brief's sources; none
is a round-5 score.

Planning-input disclosure (S0: `power_worlds.json`, `size_gate.json`). Corpus-based worlds and nulls use seen,
label-based values (P1 PRIMARY family log ratios; publication × chemotype unit Gd chords; the −0.217 and −0.142
centres) as planning distributions only. They are upper-side for genuinely new sources (B1's and SH's forms were chosen
on these cells) and assume exchangeability, which the supply violates: 15 of 16 probable new laboratories use chemotypes
the corpus lacks [P4].

---

## 1. Frame (brief §4.2, §4.3, §7.5)

- **One-sided 0.025 per claim.** No global claim. No Holm inside a claim at small G (with exact tests G = 6 cannot reach
  0.0125: the smallest p is 1/64 = 0.0156). Fixed-sequence gatekeeping inside a claim [I2 rules 9–10].
- **Units.** Laboratory (frozen PROVISIONAL key, DATA_CONTRACT §8.1) for every Gd-amplitude and C-ECT quantity; one value
  per laboratory (§5). Under the sealed fallback D3, C-ECT decides on **chemotype units** (§4.2).
- **G floors.** G ≤ 5 cannot reject at 0.025 with SIGN or a sign-flip (minimum p 1/32 = 0.031) [P2 §3].
- **Labels.** A result of this package is INDEPENDENT CONFIRMATION only if it comes from Stage-2 data under the sealed
  package. Every in-corpus number elsewhere in round 5 is EXPLORATORY (seen).
- **Power is stated in worlds, never as one number** (§15) [I2 rule 11].

## 2. Claims and per-claim minimum G (brief §4.3 item 1, §7.5)

| claim | statistic (as sealed after D3) | minimum G | below the minimum | status after the size gate (§3–§4) |
|---|---|---|---|---|
| **C-ECT** (confirm-4): SH@4 beats B1@4 in new laboratory × extractant families with ≥ 3 full-series conditions, for targets differing by a T1–T3 change | per family `log(Σ SSE_B1@4 / Σ SSE_SH@4)` over leave-one-condition-out targets → laboratory value (unweighted mean of its families) → **chemotype-unit value** (unweighted mean of its laboratories); exact one-sided sign-flip over chemotype units; fixed sequence SH@4 then SH@3 | **7 chemotype units** (hence ≥ 7 laboratories) | descriptive case series, no kill or promote language | **TEST (fallback ii)**; passes its fresh size check |
| **Gd-dip transport** (M33 primary) | laboratory-mean Gd-chord interval against β_frozen and β_frozen/2; non-DGA interval primary, DGA interval descriptive; frozen-interval conformal coverage and the level against a like-for-like C3 reference, never quoted as reaching 0.50; the "B1-specific failure" reading (rules in §7.2) | runs with ≥ 10 laboratory × chemotype units on the merged SAFE + literature ledger; decides with ≥ 10 laboratories | descriptive | **DESCRIPTIVE (fallback i)**: on the exclusion list |
| **C-GD** (confirm-2 + confirm-3's interval; the cation-exchanger stratum of the Gd-dip test, not a separate α claim) | laboratory-mean Gd chord, winsorised to [−0.40, +0.20] (primary) and raw (sensitivity), against 0, β_frozen/2 and β_frozen; secondary per-unit Gd-pair SSE sign-flip; chord against the signed slope outside Eu–Tb | 6 laboratories from ≥ 3 extractant families inside the stratum | descriptive case series | **DESCRIPTIVE (fallback i)**: on the exclusion list |
| A3 positive control (not a claim) | B1 − B0 prospective, equal-weight log SSE ratio with the pooled guard, ungated | — | — | reported only (not size-gated) |

Supply scenarios by claim, laboratory by laboratory from the frozen key (S0: `supply_audit.json`
"claim_supply_lab_by_lab"; probabilities S0: `supply_reach.json`, a forecast model in which each E2-only unit converts
with p_E2 and each U-only unit with p_U, independently; its mean reproduces the audit's expected-yield formula).
Forecasts are **reported, never counted** (D4).

| claim (min G) | verified in hand | E2-only / U-only candidates | forecast P4 rates (p_E2 0.5, p_U 0.25): expected; P(reach min) | forecast PC P4 rates (p_E2 2/6, p_U 0): expected; P(reach min) | all convert |
|---|---|---|---|---|---|
| C-ECT, laboratories (7) | 2: Chiba-Katsuta, KULeuven-Binnemans | 7 / 0 | 5.5; 0.2266 | 4.33; 0.0453 | 9 laboratories |
| C-ECT, chemotype units after D3 (7; the decision unit, D4) | 2 units | — | P(≥ 7 units) 0.0 linked; 0.5 if laboratories were not linked | 0.0 linked; 0.2757 unlinked | 10 hand-label chemotypes; **6 units** after linking chemotypes that share a laboratory (§4.2) |
| C-GD, laboratories (6; ≥ 3 families not modelled, so an upper bound) | 1: Chiba-Katsuta | 2 / 6 | 3.5; 0.0623 | 1.67; 0.0 | 9 |
| Gd-dip transport, laboratory × chemotype units (10) | 2 literature (Chiba-Katsuta acidic OP; KULeuven-Binnemans quaternary-onium nitrate IL) + SAFE at its maximum 3 | 8 / 10 | 11.5; 0.8459 (SAFE 0: 0.3005) | 7.67; 0.0879 (SAFE 0: 0.0002) | 23 |
| SAFE route, M33 gate (10) | 0 verified; 3 at most [IR M33] | — | 3; 0 | 3; 0 | 3 |

Reading (counts only): the chemotype count uses P4's hand labels, which are upper bounds on the frozen gen6 map
(DATA_CONTRACT §8.3); with the laboratory link rule of §4.2, the current candidate list cannot reach 7 C-ECT chemotype
units even if every candidate converts. Reaching C-ECT's minimum needs new E-CT-capable laboratories (the U5 targeted
search, DR U5 item 3).

## 3. Size gate (brief §4.3 item 2) — results

Rule: a test is dropped before sealing if its size exceeds 0.035 (the lower limit of the two-sided 95 % Clopper–Pearson
interval, i.e. a one-sided 97.5 % bound; P2's convention) at nominal one-sided 0.025, under the i.i.d. nulls or under the two-stage (chemotype → laboratory) nulls at ICC {0.23, 0.3, 0.5,
0.69}, with the 7 laboratory compositions and the sd_within grid {0.05, 0.08, 0.10} of
`ideas2/confirm1skepB_out/clustered_null_size.csv`, 20,000 replicates per cell (≥ 2,000 required), seeds
`default_rng(20261023 + crc32(key))` (S0: `size_gate.json`, `size_gate.csv`; code `scripts/s0_sizegate.py`,
`r5/confirm.py`). Technical check: CGD_R's confirm-rule size reproduces the 63 cells of `clustered_null_size.csv` (max
|Δ| 0.0069, max |z| 2.46, 1.6 % of cells with |z| > 2).

| statistic (claim, role) | gating cells | flagged (two-stage / i.i.d.) | flagged at decidable G | max size, two-stage, by ICC 0.23 / 0.3 / 0.5 / 0.69 | max size, i.i.d. | verdict |
|---|---|---|---|---|---|---|
| CGD_W (C-GD, primary, winsorised) | 496 | 328 (322 / 6) | 328 | confirm rule 0.0550 / 0.0671 / 0.1071 / 0.1471 (over gating rules 0.0567 / 0.0675 / 0.1071 / 0.1471) | confirm 0.0278; kill / harm 0.0507 | **FLAGGED** |
| CGD_R (C-GD, raw, sensitivity) | 496 | 328 (322 / 6) | 328 | confirm rule 0.0543 / 0.0671 / 0.1043 / 0.1434 (over gating rules 0.0554 / 0.0671 / 0.1043 / 0.1434) | confirm 0.0277; kill / harm 0.0507 | **FLAGGED** |
| GDT_R (Gd-dip transport, laboratory-mean t-interval) | 480 | 322 (322 / 0) | 48 (G ≥ 10) | confirm rule 0.0543 / 0.0671 / 0.1043 / 0.1434 (over gating rules 0.0554 / 0.0671 / 0.1043 / 0.1434) | confirm 0.0277; kill / harm 0.0313 | **FLAGGED** |
| ECT_SF (C-ECT, laboratory-level exact sign-flip) | 43 | 21 (21 / 0) | 11 | 0.0454 / 0.0507 / 0.0701 / 0.0979 | 0.0265 | **FLAGGED** |
| ECT_SIGN (C-ECT, laboratory-level SIGN) | 43 | 11 (11 / 0) | 1 | 0.0344 / 0.0451 / 0.0642 / 0.0979 | 0.0169 | **FLAGGED** |

Under a one-sided 95 % bound 332 / 332 / 326 rows flag for CGD_W / CGD_R / GDT_R (12 more cells: comp (2, 2, 2), ICC
0.23, size 0.0372–0.0376), with the same verdicts; the C-ECT rows are unchanged. The C-ECT gating counts exclude the
seen-pool skewed nulls (emp_P1 and its left-skewed mirror), which are descriptive rows (below; brief §4.2).

The six i.i.d. flags of the chord statistics are the kill and harm rules under the seen heavy-selective pool recentred
(emp_HS12) at G = 6, 8, 10 (size 0.0507, 0.0448, 0.0403). The existing confirm-1 file already showed the C-GD laboratory
t failing in all 63 of its cells (brief §4.2).

**Fresh size check of the C-ECT fallback (D3; S0: `size_gate_ect_chemotype.json`, `.csv`; code
`scripts/s0_sizegate_ect_chemotype.py`, design fixed in its docstring before the run; 20,000 replicates per cell).**
Statistic ECT_CHEM_SF: laboratory values averaged within chemotype, exact one-sided sign-flip over chemotype units.
Nulls: the 7 file compositions and 10 declared compositions that reach ≥ 7 chemotypes ((1,)×7, (2,1,1,1,1,1,1),
(2,2,1,1,1,1,1), (3,2,1,1,1,1,1), (2,2,2,1,1,1,1), (2,2,2,2,2,2,2), (3,3,2,2,1,1,1,1), (1,)×10,
(4,3,2,2,1,1,1,1,1,1), (1,)×13) × the four ICC values; i.i.d. chemotype values at G = 7, 8, 10, 13 (normal, t(3) and
the sign-symmetrised seen P1 family ratios, all gating; the seen ratios recentred to mean 0, a right-skewed null, and its
mirror image, a left-skewed null, both descriptive).

| statistic | cells | at decidable G (≥ 7 units) | flagged | max size (CP lower bound) | max by ICC 0.23 / 0.3 / 0.5 / 0.69 | i.i.d. max: normal / t3 / skewed emp_P1 / swap_P1 | role |
|---|---|---|---|---|---|---|---|
| **ECT_CHEM_SF** | 80 gating (88 rows) | 56 | **0** | **0.0268 (0.0246)** | 0.0246 / 0.0268 / 0.0248 / 0.0264 | 0.0249 / 0.0238 / 0.0083 (descriptive) / 0.0262; mirror 0.0723 (descriptive) | sealed fallback (ii): **passes** |
| ECT_CHEM_SIGN | 88 | 64 | 0 | 0.0121 (0.0106) | 0.0110 / 0.0118 / 0.0110 / 0.0116 | 0.0108 / 0.0106 / 0.0115 / 0.0121; mirror 0.0111 | reported beside it, not a sealed test |
| ECT_LAB_SF (same draws) | 68 | 52 | 37 | 0.0972 | 0.0443 / 0.0522 / 0.0701 / 0.0972 | — | descriptive under D3 |
| ECT_LAB_SIGN (same draws) | 68 | 52 | 14 | 0.0972 | 0.0380 / 0.0459 / 0.0656 / 0.0972 | — | descriptive under D3 |

On the one file composition that reaches 7 chemotypes, (4, 3, 2, 1, 1, 1, 1), the chemotype-level size is
0.0226–0.0245 across the ICC grid against 0.0443–0.0876 for the laboratory-level test on the same draws. Reading: under
the two-stage nulls chemotype values are independent and symmetric about 0, so the exact sign-flip over chemotypes keeps
its size by construction (exactness under a sharp, exchangeable null, not an empirical finding; brief §4.2); the
check confirms it numerically under symmetric nulls. Under the seen skewed pool the chemotype-level size is 0.0045 /
0.0083 / 0.0073 / 0.0082 right-skewed and 0.0723 / 0.0579 / 0.0614 / 0.0573 mirrored (left-skewed) at G = 7 / 8 / 10 /
13, so like every test here it does not hold 0.025 under skewed nulls, which are non-gating (brief §4.2; D3). It does
not cover a laboratory that spans chemotypes; the link rule of §4.2 removes that case.

## 4. Sealed fallbacks (brief §4.3 item 2; D3)

### 4.1 C-GD and Gd-dip transport → fallback (i): exclusion list, descriptive intervals

Every chord statistic fails the gate (§3), so the C-GD laboratory-mean chord interval (winsorised and raw) and the
Gd-dip transport interval move to the exclusion list (§6) as **descriptive intervals**. Consequences, sealed:

1. The intervals are computed and reported with their three-way **readings** (§7.2), labelled DESCRIPTIVE; none is a
   test and none is called confirmed, killed or significant.
2. **No Gd-amplitude result can change the deployed default.** The B1-harm rule of brief §4.3 item 6 (97.5 % lower
   bound above β_frozen/2 switches new chemotypes to B0) needed a valid rejection; under fallback (i) none exists, so
   B1 with β_frozen stays the default for new chemotypes under this package. A future test would need a fresh size
   check sealed by amendment before any new value is read.
3. The claims' secondary statistics (C-GD's per-unit Gd-pair SSE sign-flip and chord-versus-slope; Gd-dip's conformal
   coverage, level and "B1-specific failure" reading) were not size-gated (`size_gate.json` notes) and are descriptive
   with their primaries.
4. For C-GD the winsorised interval is primary and the raw interval a sensitivity (brief §4.3 item 2).

### 4.2 C-ECT → fallback (ii): one value per chemotype unit, exact sign-flip, minimum 7 units

- **Family value** f = log(Σ SSE_B1@4 / Σ SSE_SH@4) over the family's leave-one-condition-out targets (summed SSE, not
  per-target ratios: one corpus target has SSE_B1 = 0); targets whose scored pairs are all fixed by the reads are
  excluded [I2 confirm-4]. A family whose summed SSE is 0 in either arm has no defined value: it is reported and left
  out of its laboratory's value. Flooring each target's SSE at a stated noise variance is a sensitivity.
- **Laboratory value** = unweighted mean of its families' values (families nested; laboratory weight 1).
- **Undefined values.** A family, laboratory or unit with no defined value is excluded before the test and never enters
  it as NaN or ±inf; the sealed code raises on a non-finite unit value (`r5.confirm.signflip_p`, `sign_test_p`,
  `t_interval`, `ect_decisions`).
- **Chemotype unit** = the chemotypes of the frozen gen6 map (DATA_CONTRACT §8.3), **linked into one unit when a
  laboratory has eligible families in more than one of them** (connected components of the laboratory–chemotype
  graph), so that no laboratory contributes to two units. *Frozen interpretation*: D3 does not address a laboratory
  that spans chemotypes; linking keeps the units independent under both laboratory and chemotype clustering. With no
  spanning laboratory it is exactly one value per chemotype.
- **Unit value** = unweighted mean of the laboratory values in the unit.
- **Test.** Exact one-sided sign-flip over units, H0: mean unit value ≤ 0; full enumeration at G ≤ 13 (8,192 sign
  vectors at 13); beyond 13, B = 4,999 random flips with p = (1 + #{T* ≥ T})/(1 + B) (`r5.confirm.signflip_p`).
  **Minimum G = 7 units** (smallest attainable p 1/128).
- **Fixed sequence:** SH@4 vs B1@4 first; SH@3 vs B1@4 at 0.025 only if the first passes.
- **Reported beside it (descriptive):** the laboratory-level exact sign-flip and SIGN (both failed the gate, §3), the
  family-level p, the unit-level SIGN, and the report by mechanism class (neutral solvating, cation exchanger, ionic
  liquid) [I2 confirm-4]; the sizes of the sign-flip under the seen skewed pool in both orientations (§3).
- **Registered sensitivity (D10; REGISTRY `C-ECT-S01`):** B1@4 and SH with u = 0 (no Gd term) for AQ_COMPLEXANT
  targets (medium map M01 / M02), with the complexant stratum reported separately; the primary keeps the frozen B1 and
  SH with their Gd term in every medium.

## 5. One value per laboratory (brief §4.3 item 3)

Every Gd and E-CT quantity enters as one value per laboratory (laboratory-bootstrap power at G = 6 is 0.742 against
0.736 for units, so this costs nothing [I2 confirm-1]). Publications are aggregated within a laboratory; several papers
from one group on one ligand family are one unit (I2 rule 12: the Gd chord is shared across a laboratory's papers,
+0.37 to +0.47 on publication means, seen).

| quantity | laboratory value |
|---|---|
| Gd chord (C-GD, Gd-dip) | mean over the laboratory's eligible cells of the chord (log D(Gd) minus the Eu–Tb radius interpolation), aggregated across its chemotypes (brief §4.2 "Gd-amplitude contrasts: the laboratory") |
| C-ECT | mean of its families' log ratios (§4.2), then aggregated into chemotype units |
| A3 | the laboratory's equal-weight log SSE ratio of B1 vs B0 |

The laboratory key is PROVISIONAL (U6 open; D6): conservative and liberal laboratory counts are both reported, and the
conservative key decides.

## 6. Exclusion list, sealed in advance (brief §4.3 item 4; D3)

Not decidable with the supply; each is reported descriptively or not at all:

| excluded | reason |
|---|---|
| pooled or E-A claims | M34's kill fired: at G = 8 publication units the gate-valid power for B1 − B0 is 0.228 (BPG) / 0.236 (BP) at τ = 0.065, and SF_POOL 0.285–0.296 at τ = 0 [PC P2]; a pooled claim needs 40–80-plus sources |
| a TOST on β | power ≤ 0.08 [I2 confirm-1] |
| the level window as a row flag | power ≤ 0.15 |
| "two reads plus your own series" (SH with 2 reads) | power ≤ 0.06; an in-corpus lead carried by 3 publications |
| the sc009 Nd dip | rider only (§16) |
| any DGA-only Gd test | — |
| **C-GD laboratory-mean chord interval, winsorised and raw, and its secondary statistics** (D3) | fail the size gate in 328 of 496 cells (§3) |
| **Gd-dip transport interval and its secondary readings** (D3) | fails the size gate in 322 of 480 cells (§3) |
| **C-ECT laboratory-level sign-flip and SIGN** (D3) | fail the size gate; reported beside the chemotype-unit test |
| E-LAB external leg | descriptive unless ≥ 8 independent laboratories with ≥ 2 extractants × ≥ 6 Ln in one campaign exist (brief §6.3) |

## 7. Three-way decisions with operating characteristics (brief §4.3 item 5; I2 rule 9)

Operating characteristics come from S0: `power_worlds.csv` (10,000 replicates per world; Monte Carlo SE ≤ 0.005;
two-sided 95 % Clopper–Pearson bounds per row in `power_worlds.csv`; one family per laboratory and i.i.d.
laboratories). For C-ECT's chemotype-unit test they apply as stated when each unit holds one laboratory; when
units hold several laboratories, read the row at G = the number of units.

### 7.1 C-ECT (a test)

- **CONFIRMED** if the exact one-sided p ≤ 0.025 with ≥ 7 units (then SH@3 in sequence).
- **KILLED** if the mean unit value ≤ 0 or the unit win share < 0.5 (and not confirmed): "use B1 even when an own prior
  series exists".
- **INCONCLUSIVE** otherwise (a small real non-DGA gain, corpus +0.14 log, is "below MDE", not killed) [I2 confirm-4].
- Fewer than 7 units: **descriptive case series**, no kill or promote language.

| world (τ 0 / 0.065) | P(confirm) at G = 7 / 8 / 10 | P(kill) at G = 7 / 8 / 10 |
|---|---|---|
| corpus DGA (the corpus effect is true) | 0.914 / 0.984 / 1.000 ; 0.899 / 0.953 / 0.993 | 0.0007 / 0.0000 / 0.0001 ; 0.0005 / 0.0000 / 0.0001 |
| halved effect | 0.131 / 0.165 / 0.309 ; 0.142 / 0.186 / 0.300 | 0.222 / 0.125 / 0.106 ; 0.227 / 0.124 / 0.113 |
| sign-symmetrised null | 0.021 / 0.022 / 0.022 ; 0.024 / 0.022 / 0.025 | 0.604 / 0.544 / 0.549 ; 0.602 / 0.547 / 0.552 |
| cation-exchanger-like W1 (null-like) | 0.022 / 0.033 / 0.048 ; 0.025 / 0.031 / 0.047 | 0.661 / 0.531 / 0.559 ; 0.578 / 0.470 / 0.476 |

So P(kill | the corpus effect is true) ≤ 0.001 at G = 7–10, P(confirm | null) ≈ 0.02 (its size), and a halved effect is
falsely killed 0.11–0.23 of the time. The win-share leg makes P(kill) depend on the parity of G (it kills at ≤ (G − 1)/2
wins at odd G but ≤ G/2 − 1 at even G): halved effect 0.222 at G = 7 and 0.195 at G = 9, against 0.125 / 0.106 at G = 8
/ 10; sign-symmetrised null 0.604 / 0.591 against 0.544 / 0.549 (τ 0, `power_worlds.csv`). At the realistic supply (2
linked chemotype units) the claim is a descriptive case series: the minimum attainable p is 1/4, so P(confirm) = 0.

### 7.2 C-GD and Gd-dip transport (descriptive readings under D3)

Readings of the 95 % laboratory-mean chord interval (the rules a test would have used, kept so the descriptive interval
is read the same way): upper bound < β_frozen → "deeper than B1's amplitude"; lower bound > β_frozen → "no
under-correction"; lower bound > β_frozen/2 → "B1 harms the average new laboratory" (B1-harm reading); upper bound <
β_frozen/2 → "B1 helps" (A2); otherwise status quo; A1 (upper bound < 0) descriptive. CGD_W at β_frozen = −0.1108
(S0: `power_worlds.json`; normal worlds with between-laboratory sd 0.10–0.16):

| world | "deeper" (UB < β) at G = 6 / 8 / 13 | "no under-correction" (LB > β) at G = 6 / 8 / 13 | "B1 harms" (LB > β/2) at G = 6 / 8 / 13 |
|---|---|---|---|
| HDEHP-like −0.26 (PROVISIONAL) | 0.49–0.84 / 0.65–0.95 / 0.87–1.00 | 0.00 | 0.00 |
| P507-like −0.105 (PROVISIONAL) | 0.02 / 0.02 / 0.02 | 0.03 / 0.03–0.04 / 0.03–0.04 | ≤ 0.004 |
| at β_frozen (calibrated) | 0.023–0.030 | 0.022–0.030 | ≤ 0.005 |
| no dip (0) | ≤ 0.0002 | 0.30–0.59 / 0.41–0.78 / 0.64–0.96 | 0.12–0.20 / 0.14–0.28 / 0.22–0.46 |

The B1-harm reading, had it been a rule, fires only 0.14–0.28 at G = 8 and 0.22–0.46 at G = 13 when B1 truly harms
(the brief's sources give 0.17 / 0.25 [I2 confirm-3]); it is weak, and under D3 it switches nothing.

**Gd-dip transport's secondary readings (brief §7.5; all DESCRIPTIVE under D3):**
- *Level:* compared with a like-for-like C3 reference distribution (the frozen external read policy, 4 reads, the
  same metals and read configuration); never quoted as reaching 0.50.
- *B1-specific failure:* read only if B1 − B0 is below the 5th percentile of that C3 reference while B0 is inside its
  reference range.
- *Conformal coverage:* the empirical coverage, on the new cells, of a frozen in-corpus interval. The interval is
  **PENDING**: no frozen in-corpus interval for this reading exists at the seal (the WINDOW item's q80 split-conformal
  interval is a candidate, not yet frozen for this use); it is named, with its nominal level, by append-only amendment
  before any new value is read (DATA_CONTRACT §9).

### 7.3 A3 (positive control, not a claim)

Reported, ungated, restricted to Gd-eligible units, with the all-unit power quoted: A3 alone ≈ 0.79 at G = 13 (τ = 0);
run on all units the chain is 0.54 / 0.37 at G = 13 + 4 [I2 confirm-3]. It is not size-gated.

## 8. Non-rejection never switches the deployed model (brief §4.3 item 6)

- Failing to reject reads INCONCLUSIVE, never "withdraw B1's Gd term". A1 ("a dip exists") is descriptive only.
- Why: the "withdraw on non-rejection of A1" rule would withdraw an exactly calibrated term in 1 − P(A1) of replicates
  when the true mean equals β_frozen: 0.23–0.59 at G = 8 and 0.04–0.34 at G = 13 (S0: `power_worlds.csv`, world
  AT_BETA, CGD_W, sd 0.10–0.16; the brief's sources give 56 % at G = 8 [I2 confirm-3] and "about 17 % of a real effect"
  for A1 [I2 confirm-1]).
- Only a rejection can change a default. Under D3 no Gd interval is a test (§4.1), so the Gd default (B1 with β_frozen)
  cannot change under this package. C-ECT's KILLED outcome changes no default either: SH is not deployed unless
  confirmed.

## 9. Supply-audit kill and its current outcome (brief §4.3 item 7; D4)

- **Rule (D4).** The audit reads metadata only. The **decision supply** of each claim is the number of units verified
  in hand under the frozen key, with the institution-link rule applied consistently, **plus SAFE's maximum (3) where
  SAFE units count** (U2 unanswered). Forecasts (P4's rates, PC P4's rates) and the all-convert scenario are reported and
  **never counted**. If every claim is below its minimum G, no confirmation runs and the round is reported as
  descriptive.
- **Current outcome (S0: `supply_audit.json`, `SUPPLY_LEDGER.csv`):**

| claim | minimum G | decision supply (D4) | below minimum |
|---|---|---|---|
| C-ECT (the test claim) | 7 linked chemotype units (D3) | 2 units (2 laboratories) | yes |
| C-GD | 6 laboratories from ≥ 3 families | 1 | yes |
| Gd-dip transport (merged SAFE + literature) | 10 laboratory × chemotype units | 5 (2 + SAFE 3) | yes |
| SAFE route (M33 gate) | 10 | 3 | yes |

  **SUPPLY-AUDIT KILL FIRES: no confirmation runs; round 5 is descriptive (brief §7.6).** `supply_audit.json` records
  the D4 decision supply (2 / 1 / 5 / 3; C-ECT in linked chemotype units) and, separately, the never-deciding bracket
  high (max(in hand, the brief's realistic high) = 3 / 5 / 6 / 3); the kill fires under both, because every realistic
  high is below its minimum. The kill reads the test claim C-ECT; C-GD and Gd-dip transport are descriptive intervals
  (D3 i). Under P4's forecast rates the Gd-dip forecast (11.5) would reach its descriptive-run trigger; forecasts never
  decide.
- **Supply ledger in brief (SUPPLY_LEDGER.csv; PROVISIONAL):** verified new independent laboratories 7 as P4 / 6 strict
  / 5 frozen; E1 + E2 16 / 14 / 13 / 11 frozen; expected after verification 13.5 (P4 formula), 9.3–11.5 with
  corrections, 7.0 with every correction; 15 of 16 new laboratories on absent chemotypes; ≥ 8 Ln with Eu, Gd, Tb
  verified 3 as P4, 2 frozen; ≥ 3 full-series conditions 4 frozen, full La–Lu verified 2 (Chiba-Katsuta,
  KULeuven-Binnemans); cpg-unit upper bounds 9 / 15 / 20 with TOMANO3 merged.
- **Reruns:** after each answered request (U2, U5, U6, U3, U1) and at the end of Stage 0, recorded in the append-only
  section below (brief §4.6).

## 10. Eligibility (brief §4.3 item 8, §7.5)

**New laboratory:** `lab_in_corpus_frozen` = N under the consistent institution-link rule (P and Y laboratories are not
new; DATA_CONTRACT §8.1). The MSU group (U3, Molecules 29:3548) is corpus-linked and its run is a descriptive dry run,
never one of C-ECT's units (brief §7.3).

**Gd claims (C-GD, Gd-dip transport, A3):**
- the unit itself reports Eu, Gd and Tb;
- values below DL or LOQ are excluded (DATA_CONTRACT §4);
- aqueous-complexant media excluded (medium map M01; lactate, citrate, DTPA, HEDTA, CDTA and the others listed there),
  **acetate excluded, decided now** (M02; inclusion a sensitivity); non-inert media (M03) excluded; the medium must be
  stated (UNKNOWN is not eligible);
- the solvating regime of HDEHP at high acid excluded: by the source's own statement, and otherwise by an acid threshold
  taken from the primary SF tables of U5 item 4 and fixed by amendment before any new value is read (PENDING-U5; a
  deferred Stage-0 input, DATA_CONTRACT §9, D11);
- the frozen mechanism-class tag decides the C-GD stratum (CX_ACIDIC_OP, CX_CARBOXYLIC, CX_CHELATING; DATA_CONTRACT
  §8.2), never the slope sign; soft N-donors are a separate descriptive family; ionic-liquid diluents (flag F01) and
  synergists (F03) are sensitivities; "stratum" language needs ≥ 3 distinct extractant families; units are reported per
  extractant.

**C-ECT:**
- ≥ 3 full-series conditions of one extractant in the laboratory;
- T0 excluded **at pair level** (D5; DATA_CONTRACT §8.4): the primary SH uses only siblings that differ from the target
  in a T1–T3 field; a target is admitted only if at least one such sibling remains; the T0-inclusive version is a
  sensitivity; Chiba's pH-only pairs are T0 and its diluent-change pairs T2 (PROVISIONAL until values);
- aqueous-complexant changes are allowed as T1 (the complexant exclusion is for the Gd claims only);
- a family is flagged, not voided, if the across-condition RMS difference of in-window adjacent log SF exceeds 0.10 log;
- the mechanism contrast SH − OFF_XEXT is registered where an other-extractant pool exists; default wording
  "same-campaign information (extractant transfer and campaign fingerprint not separated)"; "campaign fingerprint"
  wording needs ≥ 4 families with a pool and a threshold set before data [I2 confirm-4].

## 11. β_frozen and its sensitivities (brief §4.3 item 9; D2)

**Sealed β_frozen = −0.110809** (−0.11080937578907438; λ = 0.003), the inner-CV value of the B1 refit on all cells
(S0: `beta_frozen.json`): one fold record with outer `train_index` = all 521 bench cells (all 229 C3 cells used; 1,374
situations, 10,812 rows), inner blocks built exactly as `g21.cache.build_fold` does (4 blocks), design BPG, declared
inner seed `gen20k.splits.inner_seed(104729, 0)` = 3246616; `r4.refs.fit_reference(bench, rec, Y, 'B1', k=3)` with the
default λ grid {0.003, 0.03, 0.3, 3, 30, ∞}; record `cache/frozen/fold_all_BPG_104729_0.pkl` (SHA-256
`7c862d87…1e835f`).

**Disclosed fragility:** λ = 0.003 beat λ = 0.03 by 7.6e-5 in inner score (0.0278030 against 0.0278790); round 4 chose
λ = 0.03 in 23 of 25 folds on both designs.

| reference | β | source |
|---|---|---|
| **β_frozen** (BPG, λ 0.003, inner CV) | **−0.1108** | S0: `beta_frozen.json` |
| design BP (λ 0.003) | −0.1095 | same file |
| λ fixed at 0.03 (BPG) | −0.0878 | same file |
| round-4 fold mean, BPG (25 folds; sd 0.0195; range −0.1398 to −0.0299) | −0.0923 | `round4_20261022/results/REF/fitted_BPG.json` via S0 |
| round-4 fold mean, BP (25 folds; sd 0.0153; range −0.1378 to −0.0748) | −0.0949 | `fitted_BP.json` via S0 |
| confirm-1 sensitivities; IR M33 rounded fold mean (reference only) | −0.118, −0.072; −0.094 | I2 confirm-1; IR M33 |

**C-GD power at every reference** (CGD_W, P(upper bound < β), between-laboratory sd 0.10–0.16; S0:
`power_cgd_beta_sens.json` for the D2 references, `power_worlds.json` for −0.118 / −0.072 / −0.094; the β_frozen rows of
the first reproduce the second exactly, 768 of 768 rows). Under D3 these are the planning strengths of a descriptive
reading, not of a test.

| reference | HDEHP-like (−0.26) G = 6 / 8 / 13 | P507-like (−0.105) G = 6 / 8 / 13 | corpus bootstrap (seen) heavy-selective pool (12 units) G = 6 / 8 / 13: context only, not the planning power (brief §7.5); kept because brief §4.3 item 9 quotes these values |
|---|---|---|---|
| β_frozen −0.1108 | 0.49–0.84 / 0.65–0.95 / 0.87–1.00 | 0.02 / 0.02 / 0.02 | 0.55 / 0.79 / 0.98 |
| BP −0.1095 | 0.50–0.84 / 0.65–0.96 / 0.88–1.00 | 0.02 / 0.02 / 0.02 | 0.57 / 0.80 / 0.99 |
| λ 0.03 −0.0878 | 0.60–0.92 / 0.76–0.99 / 0.94–1.00 | 0.04–0.06 / 0.05–0.06 / 0.05–0.08 | 0.78 / 0.96 / 1.00 |
| fold mean BPG −0.0923 | 0.58–0.91 / 0.74–0.98 / 0.93–1.00 | 0.03–0.05 / 0.04–0.05 / 0.04–0.06 | 0.74 / 0.94 / 1.00 |
| fold mean BP −0.0949 | 0.57–0.90 / 0.73–0.98 / 0.93–1.00 | 0.03–0.04 / 0.04 / 0.04–0.05 | 0.72 / 0.92 / 1.00 |
| −0.118 | 0.46–0.80 / 0.61–0.93 / 0.83–1.00 | 0.01–0.02 | 0.48 / 0.71 / 0.96 |
| −0.072 | 0.67–0.96 / 0.83–1.00 / 0.97–1.00 | 0.06–0.10 / 0.08–0.12 / 0.10–0.19 | 0.90 / 0.99 / 1.00 |
| −0.094 | 0.57–0.90 / 0.73–0.98 / 0.93–1.00 | 0.03–0.05 / 0.04–0.05 / 0.04–0.05 | 0.72 / 0.93 / 1.00 |

The −0.118 and −0.072 rows reproduce confirm-1's quoted 0.48 / 0.71 and 0.89 / 0.99 at G = 6 / 8 on the heavy-selective
pool. `size_gate.json` and `power_worlds.json` record the current β_frozen (−0.11080937578907438) and the LF SHA-256
`2b5d0e2d…cc6264` of `beta_frozen.json` since their pre-seal rerun (DECISION_LOG N1). **β_frozen's anchor depends on U1:** B1's Gd transfer to the located documents is learned mainly from the 69
unlocated cells (refit without them: β mean −0.0708, a seed-unstable tie on located rows; +0.0119 with λ fixed at 0.03)
[P5 §4b; PC P5].

## 12. Planning amplitudes — PROVISIONAL (brief §4.3 item 10; D6)

| centre | value | status | source |
|---|---|---|---|
| HDEHP-like | −0.26 (Gd/Eu ≈ 1.5, Tb/Gd ≈ 5) | **PROVISIONAL, unverified planning value; PENDING-U5** | I2 confirm-1, confirm-2 |
| P507-like | ≈ −0.105 (≈ β, so power ≈ size) | **PROVISIONAL, unverified search-snippet reading (the source page returned 404); PENDING-U5** | I2 confirm-2 |

Power is reported for both centres separately (§7.2, §11, §15), never from the corpus. Both are fixed from primary SF
tables (Peppard for HDEHP; a handbook for P507; DR U5 item 4) by an append-only amendment before any new value is read.
Corpus context only (seen): the slope-defined heavy-selective non-DGA stratum has mean chord −0.217 (12 units),
light-selective −0.073, DGA −0.051, but 11 of 12 heavy-selective units are N-donors or CMPO-type, and the corpus holds no
cation-exchanger stratum to power from (1 cell) [I2 confirm-1, confirm-2].

## 13. Chord hygiene (brief §4.3 item 11)

- **Chord** = log D(Gd) minus the Eu–Tb interpolation on the radius axis (−Shannon CN8 radius, `r3.interp` 'R', as B1).
  A laboratory value needs Eu, Gd and Tb in the same cell.
- **Floor censoring (D9).** A chord is not computed (censored), and the unit loses that cell, when any of its Eu / Gd /
  Tb readings is at or below a source-stated DL / LOQ or matches the round-3 V2-5 floor pattern (D < 1.5·10^e) (brief
  §4.3 item 11; I2 confirm-3); the chord without V2-5 censoring is a sensitivity. For Gd chords this overrides the
  general new-data rule of DATA_CONTRACT §4 item 2 (a value-derived floor stays a sensitivity elsewhere).
- **Resolution rule.** At arrival the sealed evaluator applies the frozen quantisation scans (`r5.dupes`: coarse step
  0.1 with half-median 0.01; small-integer ratios) and records each source's reporting precision. For a source that
  reports D to one decimal (or a coarser step δ), the per-metal quantisation half-width is q_m = log10(D_m + δ/2) −
  log10(D_m), and the chord's half-width bound is q_Gd + max(q_Eu, q_Tb). A cell whose bound is ≥ |β_frozen|/2 =
  0.0554 is flagged (about ±0.05 log of quantisation noise is comparable to β_frozen/2 [I2 confirm-1]); the primary
  keeps flagged cells and a sensitivity drops them. *Frozen interpretation* of item 11's "resolution rule" (D9: the
  brief specifies no action, and dropping cells would cut an already sub-minimum supply).
- **Winsorisation** to [−0.40, +0.20] (frozen; it clips exactly one corpus unit [I2 confirm-3]).
- **Gd-read rate of the frozen read policy** (part of the claims; S0: `ext_dopt.json`): among C3 cells with Gd measured,
  the external read set contains Gd in 60 of 218 at 3 reads and 95 of 218 at 4 reads.

## 14. Freeze before any new value is read; the external read policy (brief §4.3 item 12; D1)

**Frozen now (hashed by `scripts/r5_seal.py`):**
- **Predictors.** B1 = PWL_R[y_S − u_S] + u, u = β_frozen·1[Gd]; B0 = PWL_R; SH = w·SIB + (1 − w)·B1 on the same reads,
  w = n/(n + K), K = 1.78; comparators SIB, COPY, the nearest-condition sibling, OFF_XEXT / PB_PUB where a pool exists
  (`r5/ect.py`; DATA_CONTRACT §7.1). B1 and SH carry the Gd term unchanged in every medium (D10; M29 restricts the
  Gd claims and the interpretation of the Gd term, not the frozen predictor); "u = 0 for AQ_COMPLEXANT targets" is the
  registered sensitivity `C-ECT-S01` (§4.2).
- **β_frozen** (§11).
- **External read policy (D1).** A new cell has no fold, so it has neither a fold covariance nor a fold β. Reads come
  from `gen20k.design.support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2·s2_all)`; 4 reads = `seq[:3]`,
  3 reads = `seq[:2]` (`r5.frozen.load_external`, `external_sequence`, `read_set`). **C_all and s2_all are fitted once on
  all 521 bench cells** with the `g21.cache.build_fold` outer recipe (fold covariance of the G14 inner OOF residuals,
  PSD-projected; s2 = training noise × the inner-selected noise factor), from the same all-cell record as β_frozen
  (S0: `ext_dopt.json`): pickle `cache/ext/ext_dopt.pkl` (SHA-256 `fc261e21…0194f3`, 18,457 bytes, rebuilt
  bit-identical); n_train 521; s2_all 0.17853; noise_var 0.35706; noise factor 16; C_all minimum eigenvalue 5.6e-6.
  *Reason (D1):* the build_fold outer recipe trains on all outer-training bench cells. *Sensitivities:* design BP gives
  identical read sets (229 / 229 at 3 and 4 reads); **the literal C3-only fit** (229 cells; s2_all 0.0501, noise factor
  4, covariance relative Frobenius difference 0.579) changes the read set of many cells (same 4-read set in 89 / 229,
  same 3-read set in 98 / 229), so it is a recorded sensitivity, never the policy.
- **Censoring rule** (§13; DATA_CONTRACT §4) and **data inclusion** (§10).
- **Analysis code and hashes:** `r5/ect.py`, `r5/confirm.py`, `r5/frozen.py`, `r5/dupes.py`, `r5/supply.py`,
  `scripts/s0_sizegate_ect_chemotype.py`, `scripts/s0_package_supplements.py`. The external evaluator that turns sealed
  family predictions into family, laboratory and chemotype-unit values (and computes chords) **is not yet written**; it
  is written, tested on synthetic families and sealed by amendment before any new value is read (a deferred Stage-0
  input: DATA_CONTRACT §9, D11).

**Evaluation hygiene (brief §8, R4-BRIEF §9).** A separate agent extracts and seals values before modellers see them;
the evaluator exposes only the legal reads to prediction and reveals hidden labels only after the predictions are
sealed; the primary confirmation runs once. An error-triggered rerun needs a documented code defect, unchanged
scientific choices and full disclosure, and it does not create a fresh holdout.

## 15. Power in worlds (brief §4.2, §7.5)

Every rate below comes from 10,000 replicates per world (Monte Carlo SE ≤ 0.005; two-sided 95 % Clopper–Pearson bounds
per row in `power_worlds.csv`).

**C-ECT** (S0: `power_worlds.json`; one family per laboratory; G = 6 is below the floor; τ = P2's per-source noise):

| world | τ | P(confirm) G = 6 / 7 / 8 / 9 / 10 | P(SH@3 in sequence) G = 7 / 8 / 10 |
|---|---|---|---|
| corpus DGA (13 of 14 units DGA) | 0 | 0.631 / 0.914 / 0.984 / 0.997 / 1.000 | 0.720 / 0.858 / 0.962 |
| | 0.065 | 0.666 / 0.899 / 0.953 / 0.981 / 0.993 | 0.629 / 0.741 / 0.890 |
| halved effect | 0 | 0.070 / 0.131 / 0.165 / 0.227 / 0.309 | 0.087 / 0.102 / 0.176 |
| | 0.065 | 0.069 / 0.142 / 0.186 / 0.243 / 0.300 | 0.055 / 0.075 / 0.128 |
| cation-exchanger-like W1 | 0 | 0.006 / 0.022 / 0.033 / 0.036 / 0.048 | — |
| | 0.065 | 0.011 / 0.025 / 0.031 / 0.039 / 0.047 | — |
| non-DGA units W2 (5 seen units) | 0 | 0.260 / 0.578 / 0.797 / 0.918 / 0.969 | — |
| | 0.065 | 0.176 / 0.299 / 0.370 / 0.430 / 0.507 | — |
| 25 % null-like mixture | 0 | 0.319 / 0.442 / 0.503 / 0.538 / 0.574 | 0.332 / 0.405 / 0.483 |
| | 0.065 | 0.316 / 0.448 / 0.494 / 0.530 / 0.568 | 0.282 / 0.343 / 0.436 |
| 50 % null-like mixture | 0 | 0.131 / 0.194 / 0.225 / 0.241 / 0.264 | 0.136 / 0.170 / 0.203 |
| | 0.065 | 0.131 / 0.197 / 0.215 / 0.230 / 0.252 | 0.116 / 0.137 / 0.171 |
| sign-symmetrised null (size) | 0 | 0.017 / 0.021 / 0.022 / 0.023 / 0.022 | 0.015 / 0.016 / 0.015 |

These reproduce the brief's planning values (corpus DGA 0.90–0.92 at G = 7; halved 0.07–0.30 at G = 6–10;
cation-exchanger or null-like 0.01–0.58 at G = 6–7; mixtures 0.31 / 0.45 / 0.50 and 0.14 / 0.19 / 0.22 at G = 6 / 7 / 8).
At the all-convert laboratory count (9) the corpus-DGA power is 0.997 / 0.981, but the all-convert scenario yields only
6 chemotype units (§2), below the floor.

**C-GD** (CGD_W; β_frozen; S0: `power_worlds.json`): P(upper bound < β_frozen) at G = 6 / 8 / 10 / 13:

| world | sd 0.10 | sd 0.12 | sd 0.14 | sd 0.16 |
|---|---|---|---|---|
| HDEHP-like −0.26 (PROVISIONAL) | 0.84 / 0.95 / 0.99 / 1.00 | 0.71 / 0.87 / 0.94 / 0.99 | 0.59 / 0.75 / 0.86 / 0.94 | 0.49 / 0.65 / 0.76 / 0.87 |
| P507-like −0.105 (PROVISIONAL) | 0.02 at every G | 0.02 | 0.02 | 0.02 |
| corpus heavy-selective centre −0.217 (seen) | 0.56 / 0.73 / 0.86 / 0.94 | 0.43 / 0.58 / 0.71 / 0.84 | 0.34 / 0.47 / 0.58 / 0.72 | 0.29 / 0.38 / 0.48 / 0.60 |
| corpus non-DGA centre −0.142 (seen) | 0.09 / 0.12 / 0.14 / 0.18 | 0.08 / 0.09 / 0.11 / 0.13 | 0.07 / 0.08 / 0.10 / 0.11 | 0.07 / 0.08 / 0.09 / 0.10 |

Corpus bootstrap (seen; context only, not the planning power, brief §7.5; kept because brief §4.3 item 9 quotes these
values): heavy-selective 12 units 0.55 / 0.79 / 0.92 / 0.98; non-DGA 25 units 0.07 / 0.08 / 0.11 / 0.12; all 44 units
≤ 0.013; DGA 19 units ≤ 0.0002. At C-GD's all-convert supply (9 laboratories) the HDEHP-like value is 0.71–0.98.

The brief's 0.90–1.00 at G = 6 for the HDEHP-like centre corresponds to β ≈ −0.094 at sd 0.10 (0.9034); at β_frozen it
is 0.49–0.84 (sd 0.16–0.10). The brief's 0.62–0.74 (slope-defined heavy-selective) and 0.23 (metadata class) at G = 6
are not reproduced at β_frozen (EMP_HS12 0.55; non-DGA 0.07–0.09), because β_frozen is deeper than the references
those values used (DECISION_LOG D6).

**Gd-dip transport** (GDT_R, raw; decides at G ≥ 10 if it were a test; β_frozen): P(upper bound < β_frozen) at
G = 10 / 13 — HDEHP-like 0.76–0.99 / 0.87–1.00; heavy-selective centre 0.47–0.85 / 0.59–0.94; non-DGA centre 0.08–0.14
/ 0.10–0.18; P507-like 0.02 / 0.02; seen non-DGA pool 0.11 / 0.12. P(upper bound < β_frozen/2, "B1 helps") — HDEHP-like
≥ 0.94; non-DGA centre 0.34–0.68 / 0.44–0.81; P507-like 0.13–0.29 / 0.17–0.38. At the P4-rate forecast (11.5 units)
the HDEHP-like value at G = 11 is 0.80–0.99 and the non-DGA centre 0.08–0.15.

Notes: C-GD's "≥ 3 families among ≥ 6 laboratories" is not simulated; chord worlds are i.i.d. laboratories (chemotype
clustering enters only through the size gate); the Monte Carlo and analytic non-central t values agree (max |Δ| 0.0152)
(S0: `power_worlds.json` notes).

## 16. Riders (brief §7.5; never package claims)

Recorded on every new source (field list in DATA_CONTRACT §6):
- **shift-3, Ce/Pr acid sign** (DGA-class, mineral acid). Controls: within-series acid permutation, a level-matched
  control, an acidic-OP sign-reversal control, Yb/Lu as the artefact exemplar. Decision with ≥ 8 eligible laboratories
  (precision-weighted random-effects mean): support if the one-sided 95 % upper bound < 0 and ≥ 70 % of units are
  negative; kill if the pooled mean ≥ 0 or ≤ 50 % are negative; otherwise inconclusive; fewer than 8: descriptive case
  series. **Sign only; no size claim** [I2 shift-3].
- **M07 Nd sign** (sc009): the frozen shrunk value w·(−0.11) with its n recorded; killed if the Nd chord is ≥ 0 in ≥ 2 of
  the first 4 new sc009 publications, or if the Pr/Nd squared error against B1 does not fall in most of them [IR M07].
- Y present; rigid SMARTS; non-DGA Gd triad; multi-ligand campaign count; M29 complexant fields.

## 17. When data arrive

| event | what runs |
|---|---|
| any answered request | the supply audit reruns (metadata only); the result is appended below |
| U5 item 4 (primary SF tables) | the planning amplitudes (§12) and the HDEHP solvating-regime threshold (§10) are fixed by amendment before any value is read |
| U6 (key review) | the laboratory key and counts are fixed by amendment |
| U2 (SAFE export) | the SAFE audit (brief §7.2) merges into the ledger: < 3 units stops the SAFE route; 3–9 descriptive; ≥ 10 lets the Gd-dip interval run as a descriptive interval (D3) |
| U3 (Molecules 29:3548 SI) | a descriptive dry run of `r5/ect.py` with the external inputs on a corpus-linked laboratory; not one of C-ECT's units |
| U1 granted (any of U1-a … U1-i) | the brief §7.1 on-yes rules (PROTOCOL §13): V3 rules sealed before any recovered value is read; V3 numbers are sensitivities only |
| U4 granted | the brief §7.4 design is frozen before any data (M11 decision rules; M12 V_x/V_w rule) |
| U5 items 1–3 granted | the search is frozen (databases, query strings, date window, inclusion rules) and sealed by amendment before it runs; DOI-level counts only; no values read; metadata-only look at the 5 Eu- or Nd-only archive DOIs [P4 §6.1]; the supply audit reruns |
| a claim reaches its minimum | the external evaluator (§14) is sealed by amendment; values are sealed by a separate agent; the claim is scored once |

## 18. Decisions applied in this file

D1 (§14), D2 (§11), D3 (§3, §4, §6), D4 (§9), D5 (§10), D6 (§5, §12), D7 (every annex row computed by an item's code is
PLANNED; no package claim depends on one), D8 (the item specs' frozen choices, including `r5/ect.py`'s explicit
external inputs), D9 (§13), D10 (§4.2, §14), D11 (§10, §14). Reasons and sensitivities: `DECISION_LOG.md`. Two
interpretations are frozen here and marked as such: the laboratory link rule of C-ECT's chemotype units (§4.2; now
also D3's decision text) and the chord resolution rule (§13; D9).

<!-- APPEND-ONLY BELOW -->

## Supply-audit runs (append-only; UTC)

- 2026-10-24, end of Stage-0 construction (S0: `supply_audit.json`, `supply_reach.json`): decision supply (D4) C-ECT 2
  / 7, C-GD 1 / 6, Gd-dip transport 5 / 10, SAFE route 3 / 10. **Kill fires.** Requests U1–U6 and S1 open.
