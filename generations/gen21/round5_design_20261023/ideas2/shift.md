# Condition-shift physics for condition transfer (theme "shift")

Written 2026-10-23 for the round-5 design, ideas2 pass.

> **Status: EXPLORATORY.** Every number below comes from the 229 already-seen C3 cells, the 5 dev seeds and the 521-cell V1 bench. Several rules were written after looking at these rows, and those are marked **hindsight**: the λ = 0.5 shrink, the extrapolation gate, the factor-of-2 acid threshold, and the light-end-only tilt. E-CT values are a separate estimand. Never place them next to the BP/BPG 0.452.

---

## 1. Bottom line

1. **At 4 reads, physics adds nothing over the shrunk sibling template (SH).** Reads plus shrinkage already absorb what a per-metal shift and tilt would add. All rows below are identical rows, BPG, no C5-BTBP.
   - **A2** (57 cells, 8 publications): every physics arm except linear extrapolation, shrunk toward B1, lands between −0.041 and +0.025 R² of SH (0.662).
   - The best is the label-free James–Stein slope shrink AEB at +0.025. Only 3 of 8 units are positive, and leaving out pub_19783c039b turns it negative.
   - **The 23 acid-only cells** (ACLS, 5 publications): no physics arm beats SH = OFF = 0.655.
2. **Fitted freely, the physics hurts.**
   - A free per-metal log D vs log10[acid] regression over the same acid-only siblings (AREG) scores 0.145 at 0 reads against 0.424 for their plain mean (ACOPY). At 4 reads it scores 0.533 against 0.633. Both are 0 of 5 seeds.
   - Linear extrapolation in log acid (AINTX) gives R² −0.32 at 0 reads.
   - The damage is extrapolation **below** the measured acid range: AREG −0.894 against ACOPY 0.252 on 13 cells. Inside the range AREG is better: 0.533 against 0.496.
3. **What does carry signal is the acid distance itself.** It decides when no measurement is needed.
   - The zero-read template's error grows with |Δlog10[acid]|. Within-publication rank correlation is 0.26 (permutation p 0.045, 8 publications); 0.33 with C5 (p 0.005).
   - B1's error shows no such gradient (0.015), so this is a property of the copy, not of hard cells.
   - In 7 of 7 publications that have both, the near cells beat the far cells.
   - Far from the siblings (> 0.30 decade, i.e. more than a factor of 2): the copy loses to B1 at 4 reads by −0.30 R². LOPO 95 % interval [−0.58, −0.03]; 2 of 10 units positive.
   - Near the siblings (≤ 0.30 decade): the copy beats B1 at 4 reads by +0.137 pooled. But only 4 of 8 units are positive, and the equal-weight per-publication relative SSE is −0.43.
4. **Does the acid dependence of log SF transfer across publications? The sign does, at the light end only; the size does not.**
   - Sign:
     - Ce/Pr's slope d(log SF)/d(log10[acid]) is negative in 11 of 11 publications (sign-flip p 0.001; mean −0.32 per decade, sd 0.38).
     - La/Ce is negative in 8 of 10 and Pr/Nd in 8 of 11. Yb/Lu is positive in 9 of 10.
     - The middle and heavy pairs have no consistent sign.
   - Size:
     - Whole slope profiles correlate 0.51–0.66 between condition groups of the same publication, but only 0.07–0.26 across publications.
     - Within-group split-half reliability has a median of 0.52 (Spearman–Brown 0.69).
     - Used as a cross-publication prior on the full profile (TILT), it costs −0.024 (A2) and −0.111 (A1) R² at 0 reads.
5. **One physical regularity holds across chemotypes.**
   - Light Ln (La, Ce) have lower log D slopes against acid than the series mean in 12 of 16 dense groups. With floor and saturation readings removed it is still 10 of 13.
   - The same light-low pattern shows in the extractant-concentration slope (the apparent solvation number). In DMDODGA, La is 2.0 and the heavy Ln 2.8–3.2.
   - Lanthanide nitrate β1 is about flat across the series ([Luo & Byrne, USF](https://digitalcommons.usf.edu/msc_facpub/1639): no discernible La–Lu variation; log β1 ≈ 0.65 at I = 0). The pattern also appears in HCl. So aqueous nitrate complexation is an unlikely cause, and organic-phase stoichiometry is the candidate.
   - Only 19 % (pooled) to 30 % (median) of the acid-induced change in adjacent log SF is a smooth deformation that 4 reads absorb.

---

## 2. What was run

Every script is in `ideas2/scripts/`. Every output is in `ideas2/scripts/shift_out/`. Each run was one process, under 30 s.

| script | what it does |
|---|---|
| `shift_slopes.py` | Acid-series groups over the V2-clean bench (publication × extractant × every condition except acid concentration, contact time and metal concentration). Per-metal and per-pair slopes against log10[acid]; pooled per pair type; split-half reliability; cross-group correlation; stretch κ |
| `shift_build.py` | Rebuilds P1's population and arms (B1, SH, OFF and COPY reproduce exactly: PRIMARY 0.5309 / 0.7082 / 0.6963 / 0.4968). Adds the physics arms on identical rows at 0, 2, 3 and 4 reads (nested DOPT_Q) |
| `shift_analyze.py` | Read ladders and inference (unit = publication × chemotype; exact sign-flip; LOPO jackknife) |
| `shift_followup.py` | Extrapolation gate (hindsight); extrapolation direction; light-end tilt (hindsight); two-factor slope model per publication |
| `shift_eb.py` | λ sweep (hindsight) and the label-free James–Stein λ |
| `shift_absorb.py` | Share of the acid deformation absorbed by 4-read PWL_R |
| `shift_dose.py`, `shift_dose2.py` | Template error against acid distance; within-publication permutation test |
| `shift_gate0.py` | Factor-of-2 split (hindsight-adjacent) |
| `shift_lslope.py` | Per-metal extractant-concentration slopes |
| `shift_window.py` | The light-end slope pattern restricted to log D in [−2, 2.5] |

The run checks reproduce: B1 recomputed from the fold β to 8.9e−16; y exact; no read-set mismatches.

**Arms.** The 4-read predictor is read-corrected exactly like P1's SIB: X_k = T + PWL_R[y_S − T_S]. "SH" suffix means shrunk to B1 with w = n/(n + 1.78), using the arm's own sibling count.

| arm | template T |
|---|---|
| ACOPY | mean of the acid-only siblings (CLEAN siblings whose only difference is acid concentration) |
| AREG | per-metal OLS of log D on x = log10[acid] over those siblings, evaluated at the target's x |
| AREGH | AREG with each metal's slope deviation from the across-metal mean halved (λ = 0.5, hindsight) |
| AEB | AREG with λ set per target by James–Stein from the siblings' own residual variance (label-free) |
| AINT / AINTX | per-metal piecewise-linear interpolation in x; flat beyond the range / linear beyond the range |
| AGATE | AREG inside the siblings' acid range, ACOPY outside (hindsight) |
| TILT | ACOPY + g_m·Δx, where g is the cumulative per-metal profile of other publications' pooled pair slopes (same acid type when ≥ 3 publications have it) |
| CREG | per-metal additive model over siblings differing only in continuous conditions, z = (log10[acid], log10[L], 1000/T) |

Because the common slope cancels in log SF and the read correction is linear, X(λ) = ACOPY + λ(AREG − ACOPY) holds exactly at every read count.

---

## 3. Physics arms against SH, OFF, COPY and B1 on identical rows (BPG; BP within 0.005)

**A2: target has ≥ 2 acid-only siblings spanning ≥ 0.1 decade; no C5.** 57 cells, 8 publications, 441 rows per seed.

| arm | 0 reads | 2 reads | 3 reads | 4 reads | 4 reads, shrunk |
|---|---|---|---|---|---|
| B1 | – | 0.041 | 0.390 | **0.455** | – |
| COPY (all CLEAN) / SH (all CLEAN) / OFF | 0.438 | 0.552 (SH) | 0.630 (SH) | **0.662** (SH); OFF 0.665 | – |
| ACOPY | 0.424 | 0.534 | 0.605 | 0.633 | 0.642 |
| AREG (λ = 1) | **0.145** | 0.400 | 0.486 | 0.533 | 0.655 |
| AREGH (λ = 0.5, hindsight) | 0.462 | 0.575 | 0.636 | 0.667 | 0.685 |
| AEB (James–Stein λ) | 0.449 | 0.573 | 0.647 | 0.675 | 0.687 |
| AINT (flat beyond the range) | 0.431 | 0.538 | 0.595 | 0.629 | 0.662 |
| AINTX (linear beyond the range) | −0.323 | −0.053 | 0.022 | 0.118 | 0.439 |
| AGATE (hindsight) | 0.444 | 0.546 | 0.617 | 0.645 | 0.653 |
| TILT (cross-publication prior) | 0.400 | 0.465 | 0.538 | 0.570 | 0.621 |

- λ sweep at 0 reads (hindsight): 0.424 / **0.488** / 0.462 / 0.348 / 0.145 at λ = 0 / 0.25 / 0.5 / 0.75 / 1.

**Contrasts on A2, 4 reads, shrunk arm against SH.** "Units +" counts positive units out of 8.

| contrast | ΔR² | units + | LOPO 95 % interval | note |
|---|---|---|---|---|
| AREG − SH | −0.008 | 3 | – | |
| AINT − SH | 0.000 | 5 | – | |
| TILT − SH | −0.041 | 2 | – | |
| AGATE − SH | −0.009 | 4 | – | |
| AEB − SH | +0.025 | 3 (4 negative) | [−0.079, +0.128] | LOPO minimum −0.018 without pub_19783c039b |

**Contrasts on A2 at 0 reads, against ACOPY (same siblings).**

| contrast | ΔR² | units + | note |
|---|---|---|---|
| AREG | −0.279 | 2 | 0 of 5 seeds |
| AINT | +0.007 | 4 | |
| TILT | −0.024 | 2 | |
| AEB | +0.025 | 2 | |

**ACLS: P1's 23 acid-only cells, 5 publications, 219 rows per seed.**

| arm | 0 reads | 4 reads, shrunk |
|---|---|---|
| B1 | – | 0.451 (unshrunk, 4 reads) |
| SH | – | 0.655 |
| OFF | – | 0.655 |
| COPY | 0.327 | – |
| AREG | −0.250 | 0.595 |
| AREGH | 0.280 | 0.652 |
| AINT | 0.223 | 0.599 |
| TILT | 0.295 | 0.611 |

The acid-only gain that P1 reported (37 % of SH − B1) comes from the reads plus the mean template, not from an acid shift.

**Where the target sits relative to the siblings' acid range (A2).**

| position | cells | ACOPY@0 | AREG@0 | AINT@0 | B1@4 | contrast |
|---|---|---|---|---|---|---|
| inside the range | 32 | 0.496 | 0.533 | – | 0.459 | AREG@0 − B1@4 = +0.074; 5 of 8 units; LOPO [−0.31, +0.46] |
| above the range | 12 | 0.453 | 0.525 | **0.661** (nearest acid) | – | |
| below the range | 13 | 0.252 | **−0.894** | 0.095 | – | |

**Two-factor slope analysis (CREG; per-metal log D = a + b·log10[acid] + n·log10[L]).**

| publication and grid | CREG@0 | CCOPY@0 | B1@4 | SH@4 | CREG shrunk @4 |
|---|---|---|---|---|---|
| C5-BTBP, 15-cell 5 × 3 acid × [L] grid | **0.502** | 0.274 | 0.238 | 0.412 | 0.539 |
| C5-BTBP without c950dc | 0.466 | 0.304 | 0.364 | **0.540** | 0.549 |
| DMDO-pyran-DGA, 3-level [L] series (3 cells) | 0.923 | 0.595 | 0.780 | 0.905 | – |
| DHD2DGA, acid × T, 6-metal cells | −0.864 | −0.210 | – | – | – |

- C5-BTBP is provenance-pending.
- The DHD2DGA fit extrapolates in T.

---

## 4. Does the acid dependence of log SF transfer?

**Groups.** 35 acid-series groups; 27 give pair slopes, from 15 publications.
- 18 groups have ≥ 6 strict pair slopes (11 publications).
- Mean log D slope in HNO3: median 1.55 per decade. The ideal nitrate-limited value is 3; HNO3 co-extraction and activity lower it at high acid.

**Pooled slope per pair type**, equal weight per publication, all acid types; per decade of [acid]. "Positive" counts publications with a positive slope.

| pair | La/Ce | Ce/Pr | Pr/Nd | Nd/Sm | Sm/Eu | Eu/Gd | Gd/Tb | Tb/Dy | Dy/Ho | Ho/Er | Er/Tm | Tm/Yb | Yb/Lu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | −0.15 | **−0.32** | −0.05 | −0.23 | +0.01 | 0.00 | −0.03 | −0.02 | +0.07 | −0.11 | +0.01 | −0.18 | **+0.19** |
| positive | 2/10 | **0/11** | 3/11 | 4/13 | 4/12 | 4/10 | 5/11 | 5/11 | 6/12 | 3/11 | 5/9 | 4/9 | **9/10** |
| sign-flip p | 0.10 | **0.001** | 0.56 | 0.15 | 0.97 | 0.97 | 0.81 | 0.81 | 0.38 | 0.21 | 0.90 | 0.56 | 0.07 |

- HNO3 only: Ce/Pr 0 of 9, p 0.004.
- sc009 (TODGA class) only: Ce/Pr 0 of 5.
- Between-publication sd per pair: 0.21–0.56. The size is therefore not transferable.

**Reliability before correlation.**
- Within-group split-half reliability of the 12-pair slope profile (odd/even acid levels, 11 groups): median r 0.52. It ranges from −0.21 to 0.94.
- Mean between-group profile correlation:

  | relation | mean correlation |
  |---|---|
  | same publication | 0.51–0.66 |
  | other publication, same chemotype | 0.07–0.19 |
  | other publication, other chemotype | 0.16–0.26 |

- So the cross-publication agreement is at most about a third of the reliable within-publication signal.

**Stretch hypothesis.** "The acid slope of a pair is proportional to its log SF." κ through the origin is positive in 7 of 11 publications; sign-flip p 0.97. It is **closed**.

**Light-end pattern.**
- The light Ln have a negative centred log D slope in 12 of 16 groups with ≥ 10 metals.
- Using only log D in [−2, 2.5]: 10 of 13. So it is not just the detection floor. C5-BTBP cannot be checked this way, because its La and Ce sit at about −4.2.
- Exceptions:
  - two 2-level TODGA-class series at high log D (pub_af0c74798c, pub_d68ed8c8cf), where heavy-end saturation is possible;
  - malonic and lactic acid media.

**Extractant-concentration slopes (apparent solvation number n).**

| system | light Ln | middle/heavy Ln |
|---|---|---|
| DMDODGA, 15 levels | La 2.0 | 2.8–3.2 |
| DMDO-pyran-DGA | La 1.7 | rising to 2.8 at Sm |
| TBDGA | La 1.1, Nd 1.3 | Gd 3.1 |
| TODGA in malonic acid | flat, 1.3–2.2 | flat |

**How much the reads absorb.**
- The acid-induced change in adjacent log SF has an rms of 0.31 log per decade (group median).
- PWL_R through the five commonest DOPT_Q read sets absorbs 19 % (pooled) / 30 % (median). The rms residual is 0.23 log per decade.
- That residual is metal-local and partly estimation noise.

**Prediction with the transferred sign.** Applying only the light-end tilt (hindsight) at 0 reads, against ACOPY:

| tilt applied to | ΔR², A1 (no C5) | units | ΔR², with C5 | note |
|---|---|---|---|---|
| Ce/Pr only | +0.017 | 4+/3− of 12 | +0.061 | the C5 gain is 1.67 of its 1.95 SSE gain |
| La/Ce, Ce/Pr, Pr/Nd | +0.016 | – | – | |

On the no-C5 rows the Ce/Pr gain is carried mainly by pub_1605e436d7.

---

## 5. Acid distance against error (dose-response)

**A1, no C5. MSE per row by the acid distance |Δx̄| from the target to the siblings' mean acid, in decades.**

| \|Δx̄\| | ≤ 0.15 | 0.15–0.30 | 0.30–0.60 | > 0.60 |
|---|---|---|---|---|
| ACOPY, 0 reads | 0.0099 | 0.0123 | 0.0187 | 0.0346 |
| B1, 4 reads | 0.0135 | 0.0177 | 0.0165 | 0.0119 |
| SH, 4 reads | 0.0071 | 0.0094 | 0.0109 | 0.0114 |

- Within-publication rank correlation between |Δx̄| and the cell's template MSE: 0.26 (permutation p 0.045, 8 publications, 67 cells); with C5 0.33 (p 0.005).
- The same statistic for B1 at 4 reads: 0.015 (p 0.46). This is the control.

**Factor-of-2 split** (0.30 decade, hindsight-adjacent; 0.15 and 0.45 give the same ordering).

| side | cells | copy (0 reads) | B1@4 | contrast |
|---|---|---|---|---|
| near | 32 | 0.655 | 0.518 | +0.137; 4 of 8 units; equal-weight per-publication relative SSE −0.43 |
| far | 43 | 0.139 | 0.443 | −0.30; 2 of 10 units; LOPO [−0.58, −0.03] |

SH at 4 reads is about flat in distance. Shrinkage toward B1 is what protects it.

---

## 6. Ideas (full fields in the structured output)

- **shift-1: Acid-distance rule for zero-read transfer.**
  - Copy your own series' log SF only within a factor of about 2 in [acid] of the measured series.
  - Beyond that, read at least 3 metals and use SH.
  - The robust half is the far side. The near-side "copy beats 4 reads" holds only when pooled.
- **shift-2: Reliability-gated, bracketed slope transfer.**
  - Use per-metal condition slopes, in 1-D acid or as an additive acid × [L] grid, only when the siblings bracket the target and their own residuals say the deformation is real (James–Stein λ).
  - Never extrapolate below the measured acid range.
  - The expected gain at 4 reads is about 0. The value is at 0–2 reads inside a grid.
- **shift-3: Light-end acid steepening as a transferable sign.**
  - Raising acid makes La/Ce, Ce/Pr and Pr/Nd more heavy-selective in most systems.
  - Test it as a pre-declared sign on new publications, and as a Ce/Pr-only tilt.
  - The size is not claimed.

## 7. Closure lines (one line each)

- **Free per-metal log-acid regression and linear extrapolation in log acid:** retired. AREG −0.279 R² against ACOPY at 0 reads, 0 of 5 seeds. AINTX −0.32 R² at 0 reads.
- **Full-profile cross-publication acid prior (TILT):** retired. −0.024 / −0.111 R²; 1 of 12 publications positive in A1.
- **SF stretch with acid (κ):** no consistent sign (p 0.97).
- **Aqueous nitrate complexation as the source of metal-dependent acid slopes:** implausible. β1 is about flat across the series, and the pattern also appears in HCl.
- **Physics on top of SH at 4 reads:** within ±0.025 R², with the units split. Do not schedule it as a decision slot.

## 8. Caveats

- **Few publications.** A2 is 8 publications (8 units). ACLS is 5. pub_b0b3296ec5 is the only acid-type publication and holds 13 A2 cells.
- **One publication drives the favourable physics.** pub_19783c039b carries every favourable physics contrast (AREG, AEB): for AREG 0.248 → 0.621 at 0 reads. Its slope profile is the most reliable one (split-half r 0.94).
- **The two-factor result is C5-BTBP.** It is provenance-pending, and La and Ce sit on the floor there.
- **Shared extractants.** "Other publications" in §4 share extractants (TODGA appears in 6 publications). Laboratories are not annotated.
- **Hindsight.** The λ, the gate, the threshold and the light-end selection are hindsight. AEB's λ formula is label-free for the target but was written after seeing AREG fail.
- **No laboratory check of the physical mechanism.** A literature check of extracted-complex stoichiometry by Ln was not done beyond the nitrate-β1 fact.
- **Process.** Files were written only under `ideas2/` (`shift.md`, `scripts/shift_*.py`, `scripts/shift_groups.out`, `scripts/shift_out/`). Nothing outside the allowed paths was read. One web search and one web fetch were used, for the nitrate β1 fact. **Slip:** one read-only `git status --short` was run while listing my output files. It changed nothing, but it breaks the no-git rule.
