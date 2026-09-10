# Gen16 — decision report

*Six leads on the frozen gen13 cohort (fingerprint `4c3c6628ea0be949`: 521 cells, 90 extractants,
45 chemotypes, Kish n_eff 11.7 over extractants-per-chemotype).  Every number carries its regime:
{extractant-macro MAE of pairwise `log SF` | macro direction accuracy | measurements | Spearman}
× {design A | B | BR | BQ | BP} × {5 discovery seeds | 5 withheld confirmation seeds}.  Design BP
— chemotype held out **and** every training cell from a held-out cell's publication dropped —
is the design deployment is chosen under; design B never selects.  Pre-registration
`PRE_REGISTRATION.md`, sealed before any lead ran, body SHA-256
`d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e`.  Tables under `results/`,
refutations under `results/refutation/`, the single confirmation run under
`results/confirmation/`.  Written 2026-09-10.*

---

## Headline

**Gen16 ran 940 scored contrasts across six leads, sent every surviving claim to two blind
refuters, and confirmed one claim on five seeds no discovery agent had seen.  The confirmed claim
is real, it replicated, and its scope is narrower than its own lead believed: the direction call
saves about two of seven measurements when candidates are pooled across laboratories, and saves
nothing — +0.005 measurements, interval containing zero in all five designs — when the candidates
come from one laboratory.  Every other lead closed.  The xTB thermodynamic cycle (L1) is not the
clean null its lead reported: the registered estimator turns out to have zero test–retest
reliability, so that experiment had no power, and the honest verdict is "not closed, not
positive".  The curvature gate (L2) closed on an honest headroom of +0.029 that this corpus cannot
distinguish from zero.  The corpus-expansion result (L4) was killed by its own refuter: a
featureless "measure the biggest family first" order reproduces all of it.  The covariance and
calibration lead (L5) produced no claim above the margin and two defects in the deployed
measured mode.  Nothing here beats the lean block set zero-shot, and the programme's honest
zero-shot gain is still gen15's +0.088 over predicting no separation at all.**

Six findings, in decreasing order of support.

1. **Confirmed, and narrower than it looks.**  Given a target lanthanide pair and a set of
   candidate extractants, discarding the candidates whose gen14 direction call disagrees with the
   requested direction cuts the expected number of measurements to the first useful candidate from
   **6.91 to 4.92 — a saving of 1.99 measurements, 28.8 %** — under BP on the discovery seeds, and
   **+1.909** on the withheld seeds, with the interval excluding zero, 5/5 seeds and LOCO
   stability in **all five designs** in both runs.  Both refuters then showed that the saving is
   **between laboratories**: restricted to one publication's candidate set it is **+0.0046
   (0.07 %)**, interval containing zero in all five designs, where 69.6 % of tasks give every
   candidate the same call.  Both statements are true and the second governs deployment.
2. **The xTB verdict is corrected, and Stage 2 is now worth more, not less.**  Correcting the
   complex energies with explicit per-species composition terms takes the slope's correlation with
   the amplitude from +0.644 (gen15's figure, reproduced exactly on its 39 series) to **−0.084** on
   62 extractants — but the corrected slope's split-half reliability is **−0.27** and its jackknife
   reliability is **0.000**, so the registered \|ρ\| ≥ 0.40 bar was unreachable in principle, and
   an attenuation-free correction gives **+0.29 to +0.45**.  The honest sentence is *"the
   cycle-corrected slope, as constructed, does not correlate with itself"*.  The real
   reference-species energies (≈ 6.7 CPU-hours, 361 array tasks) are delivered ready to submit and
   are the only way to settle it.
3. **The corpus-expansion plan is not supported, and the reason is worth more than the plan.**
   A greedy A-optimal chemotype order beats random under BP by +0.050, but its sign flips under B
   and BQ, and a **featureless order — biggest chemotype first, zero descriptors — reproduces it
   and is positive in all five designs**, with A-optimal adding +0.005 (p = 0.36) on top.  At a
   budget of 6 chemotypes the A-optimal order already holds 72 % of the fold's training cells.
   What the retrospective simulation measures is corpus **volume**, not chemistry choice.
4. **The curvature is closed.**  The honest leave-pair-out headroom is **+0.0287** (CI
   [−0.0013, +0.0532], p = 0.061); the minimum detectable headroom on this corpus is 0.039.  About
   0.044 of gen15 §2's in-sample +0.073 was the oracle absorbing the noise of the pair it was
   scored on, confirming gen15 §1a by an independent route.
5. **Two defects in the deployed measured mode.**  The leave-chemotype-out residual covariance is
   **indefinite on 100 % of its 430 estimates**, which makes Ledoit–Wolf shrinkage catastrophic
   (BP MAE 34.4 at k = 3) because greedy D-optimal selection seeks the negative directions; and
   the intervals over-cover badly — nominal 90 % covers **98.8 %** at k = 3 — driven by the fixed
   `NOISE_VAR = 0.09`.  Gen15 §7's `MIX6meanPC` gain does survive the deployed baseline it was
   never tested against, at **+0.0070** (p < 1e-4, all five designs), three times under the margin.
6. **The cohort is not throwing chemistry away, but the laboratory is.**  No relaxation of any
   gen13 cohort rule adds a single chemotype.  The 100 excluded bundle extractants are excluded
   because **only one lanthanide was ever measured on them**; they carry **53 chemotypes the
   cohort does not contain**, and measuring one second lanthanide on one compound per absent
   chemotype would take Kish n_eff from **11.67 to 27.4**.

---

## 1. What was run

| phase | what |
|---|---|
| 0 | three anchors reproduced; L6 cohort audit; environment verified |
| 1 | `PRE_REGISTRATION.md` sealed and committed with a hash commitment to five withheld seeds |
| 2 | five lead agents in parallel: L1, L2, L3, L4, L5 |
| 4 | six refuters, two per claim, blind to each other |
| 5 | the confirmation run, executed once |
| 6 | this report, then an adversarial audit of it |

**Anchors** (`results/anchors/ANCHORS.md`, guarded by `tests/test_anchors.py`).  All three
reproduce exactly: gen13 stage-3 `G13_ET_TOPO39` under BP = `0.7683085207475452`; gen14 `G14`
under BP = `0.5000794414203691`; `FLAT` = `0.5885062528901843`; `G14 − FLAT` = +0.0884
[+0.0010, +0.1473], p = 0.047.  The fold plan hashes identically across processes and across
`PYTHONHASHSEED` values; the only `hash()` on the bench path takes an int tuple, which CPython
does not salt.  No dependency has moved since gen13.

**Comparison accounting.**  940 contrast rows: **126 registered, 814 exploratory**
(L1 90, L2 20, L3 85, L4 435, L5 230; `results/*/contrasts_*.csv`, every row carrying `family`
and `lead`).  Benjamini–Hochberg on the discovery p-values puts the confirmed claim's BP row at
**q = 0.039** within the registered family and **q = 0.025** over all 940.  Every other registered
contrast that reached nominal significance is reported with its q in the lead reports.

---

## 2. L3a — the confirmed claim, and its scope

A **task** is one (split seed, unordered metal pair); its candidates are the held-out extractants
that measured that pair, several cells collapsed by the median; tasks with fewer than 5 candidates
are dropped (none were).  A candidate **succeeds** for a requested direction if its observed
`log SF` has that sign and \|log SF\| ≥ 0.3.  Without a model the chemist measures in random order,
expected draws `(N+1)/(K+1)`; with it, candidates whose direction call disagrees are deferred.

**Table 2a — `saved`, in measurements, all five designs, both seed sets**

| design | discovery | 95 % CI | confirmation | 95 % CI | seeds | LOCO |
|---|---|---|---|---|---|---|
| B | +2.207 | [+0.521, +4.118] | **+2.086** | [+0.234, +4.170] | 5/5 | stable |
| BR | +2.149 | [+0.455, +3.985] | **+2.170** | [+0.252, +4.195] | 5/5 | stable |
| BQ | +2.179 | [+0.468, +4.070] | **+2.123** | [+0.248, +4.196] | 5/5 | stable |
| A | +2.492 | [+0.771, +4.544] | **+2.285** | [+0.620, +4.366] | 5/5 | stable |
| **BP** | **+1.988** | [+0.443, +3.781] | **+1.909** | [+0.360, +3.760] | 5/5 | stable |

`E_random` = 6.906 in both runs.  Permutation p = 0.0000 at its 1/2000 resolution in every cell.
BP is the **smallest** of the five savings, the ordering expected if no publication leak were
carrying it.

**Table 2b — the two comparators the pre-registration demands**

| comparator | `saved` under BP | note |
|---|---|---|
| no model (random order) | 0 by definition | the floor |
| **"always heavier"** (registered cheapest competitor) | **0.000000** | zero *by construction*: a constant call cannot partition a candidate set |
| 13-column gen6 donor census, used identically | **−3.14** | it *costs* measurements; G14's increment over it is +5.13 |

**Table 2c — what the refuters did to it** (`REFUTATION_LOG.md` §2)

| check | `saved` under BP | verdict |
|---|---|---|
| headline, re-derived independently from the sealed text | +1.988 (6 dp) | held |
| diglycolamides (sc009) removed | +1.459 | held, positive in all five |
| sc009-only candidate sets | +0.386 | the effect is *between* chemotypes |
| n_metals-stratified permutation | real − shuffled +1.886 | held (~5 % attributable) |
| leave-one-publication-out, all 58 | [+1.506, +2.428], 0 sign flips | held |
| **candidate sets within one publication** | **+0.0046**, CI [−0.016, +0.022] | **dents, materially** |
| candidate sets within one chemotype | +0.334, rule fails 5/5 | dents |
| rebuilt one fold at a time (fold purity) | magnitude falls, direction survives | dents the magnitude |

The within-publication test rests on 4 of 58 publications, 31 extractants and 11 chemotypes, 18 of
the 31 being diglycolamides — which is why it is recorded as material rather than fatal.  On those
same tasks a perfect sign call would save 0.414 measurements, so the regime is not vacuous: the
model captures 1.1 % of the available headroom there against 47.3 % pooled.

**What this supports.**  *Told a ligand and a metal pair, from a laboratory the model has never
seen, the model says which lanthanide enters the organic phase, and used as a filter over a
literature-wide candidate list it removes about two of every seven measurements.*  It does **not**
support "screen your own shortlist with this and save two measurements".

## 3. L3c — ranking at k = 1 is a null, and the no-model line wins on rank

Every candidate given one measured pair (its widest `dZ` excluding the target):

| arm, design BP | regret (log10) | cross-extractant Spearman |
|---|---|---|
| random | 1.762 | 0 |
| `G14` zero-shot | 1.454 | 0.352 |
| `NAIVE_LINE@k1` | **0.339** | **0.799** |
| `G14@k1` | 0.361 | 0.772 |

Registered contrast `G14@k1 − NAIVE_LINE@k1`: on regret +0.003 / +0.002 / −0.003 / −0.017 /
−0.021 (B/BR/BQ/A/BP), p 0.34–1.00 — a null; on Spearman it is **negative in all five designs**
(−0.022 to −0.027, p 0.013–0.026), i.e. the corpus makes ranking *worse*.  One measurement is
worth 1.09 log units of regret against the zero-shot model; the corpus adds nothing on top of it.
**L3b** (ranking within a chemotype) was registered as conditional on L1 and, L1 having closed,
was **dropped rather than run**.

## 4. L1 — the thermodynamic cycle: a corrected verdict

Gen15 measured a Spearman of **+0.644** between a two-way xTB energy slope and the observed
amplitude and killed it with a composition control.  L1 rebuilt the bookkeeping exactly.

**Table 4a — the dose–response, ρ(slope, `a`)**

| bookkeeping | model | S8 (n = 62) | S14 (n = 39) |
|---|---|---|---|
| none | `NAIVE` | +0.392 | **+0.644** |
| right species, wrong counts | `SPECIES_NFILLCOL` | +0.374 | +0.264 |
| element counts as covariates (gen15's B) | `ELEM` | +0.313 | +0.396 |
| exact counts, no free δ (19 series) | `SPECIES_CONST` | +0.167 | +0.155 |
| exact counts + free δ (**registered**) | `SPECIES` | **−0.084** | −0.048 |

Two supporting facts: `n_fill` counts donor **sites**, not molecules (nitrate is bidentate), and
every complex decomposes exactly as metal + n·ligand + n·NO₃ + n·H₂O with no residue; the fitted
species energies land within 0.34 eV of an independent per-element regression, so the model is
doing the cycle's arithmetic.  The naive slope and its curvature are **the same step function read
twice** (ρ = −0.981).

**Table 4b — why the registered verdict is wrong** (`REFUTATION_LOG.md` §4)

| diagnostic | value | consequence |
|---|---|---|
| split-half reliability of the `SPECIES` slope (odd vs even metals) | **−0.272** | the slope does not reproduce itself |
| jackknife reliability (between-extractant sd 0.203 eV vs within-series SE 0.329 eV) | **0.000** | max attainable \|ρ\| ≈ 0; the 0.40 bar was unreachable |
| fraction of an injected real trend retained by `SPECIES` | **0.42** | the correction removes ~58 % of a real signal |
| attenuation-free corrections (fixed reference energies, no free δ) | ρ = **+0.29 / +0.34 / +0.45** | **above** the 0.25 "closed" threshold |
| Fisher CI of the `SPECIES_CONST` row at n = 19 | [−0.337, +0.597] | could never separate +0.167 from +0.5 |

**Verdict corrected from "closed" to "not closed, not positive".**  The registered estimator is
too noisy to detect anything.  This raises rather than lowers the value of Stage 2: with real
reference energies the composition is subtracted at true species energies with **no free
parameter**, which is the one construction that is both exact and attenuation-free.
Hand-over: `results/L1/L1_STAGE2_HANDOVER.md`, 361 array tasks, 1 core and 2 GB each, ≈ 6.7
CPU-hours (≈ 20 min wall at 20 concurrent), then one local command.  `xtb` is absent from this
machine, verified by a disk walk.

## 5. L2 — the curvature gate

`O_CURV_LPO` is gen14's amplitude with the cell's own second coefficient refitted **without the
two metals of the scored pair**.

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `FLAT` | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| `G14` | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| `O_CURV` (in sample) | 0.4216 | 0.4218 | 0.4233 | 0.4168 | 0.4272 |
| **`O_CURV_LPO`** | 0.4634 | 0.4638 | 0.4642 | 0.4599 | **0.4714** |

Headroom `G14 − O_CURV_LPO`: +0.0298 / +0.0268 / +0.0264 / +0.0323 / **+0.0287**; under BP the
percentile CI is [−0.0013, +0.0532] and p = 0.061, so P1 fails and the gate closed; only design A
excludes zero and A never selects.  Bootstrap SE 0.014 puts the **minimum detectable headroom at
0.039**.  The `r0` reparametrisation was not run, per the registered stopping rule.

## 6. L4 — corpus-level experimental design

**Table 6a — learning curve, design BP, macro MAE by budget in chemotypes**

| order | 6 | 9 | 12 | 16 | 20 | 24 | full (≈ 29) |
|---|---|---|---|---|---|---|---|
| `RANDOM` | 0.626 | 0.608 | 0.578 | 0.553 | 0.524 | 0.511 | 0.500 |
| `MAXMIN` | 0.599 | 0.613 | 0.609 | 0.591 | 0.526 | 0.512 | 0.500 |
| `UNCERT` | 0.663 | 0.636 | 0.606 | 0.524 | 0.520 | 0.517 | 0.500 |
| **`AOPT`** | **0.570** | **0.523** | **0.509** | **0.492** | 0.503 | 0.503 | 0.500 |

**Table 6b — registered contrasts, area between curves**

| contrast | B | BR | BQ | A | BP | five-design rule |
|---|---|---|---|---|---|---|
| `AOPT − RANDOM` | −0.012 | +0.023 | −0.003 | +0.037 | **+0.050** | **fails** (sign flips) |
| `AOPT − MAXMIN` | −0.004 | +0.042 | +0.010 | +0.053 | +0.058 | fails |
| `UNCERT − RANDOM` | −0.031 | −0.004 | +0.003 | −0.018 | −0.011 | fails |
| `UNCERT − MAXMIN` | −0.023 | +0.014 | +0.017 | −0.002 | −0.003 | fails |

**And then the refuter.**  A featureless order — chemotypes sorted by their count of
well-determined cells — gives `SIZE − RANDOM` = +0.028 / +0.019 / +0.026 / +0.023 / **+0.045**,
**positive in all five designs**, while `AOPT − SIZE` = +0.005 (p = 0.36), below the margin and
negative under B and BQ.  Spearman(A-optimal pick position, chemotype cell count) = −0.63; at
budget 6 the A-optimal order holds 72 % of the fold's training cells against random's 18 %.

**Verdict: not supported, and the prospective ranking inherits the defect.**  The ranked lists of
95 bundle and 273 external candidates (`results/L4/ranking_pool_*.csv`) are ordered by the same
criterion that the refuter showed to be a size proxy — their top 20 is dominated by
diglycolamides, the family that already holds 375 of 521 cells, and the criterion takes only 43
distinct values over 95 candidates.  **They are delivered as an artefact of the registered
protocol, not as a recommendation.**  The defensible corpus-expansion advice in this report comes
from L6 instead (§8), and it does not need a model.

## 7. L5 — covariance and calibration

Verification first: the ported evaluator reproduces `G14@doptk3` = 0.1700393901 and
`G14@doptk1` = 0.2313450953 under BP to 0.0e+00, and the mixture port reproduces all 11 of gen15
§7's locked training-route modes to 6 dp.

**Table 7a — extractant-macro MAE under BP, D-optimal support, common scoring set**

| covariance | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| `POOLED` (deployed) | 0.4052 | 0.2238 | 0.2009 | **0.1674** |
| `LW` (Ledoit–Wolf) | 0.4052 | 0.2214 | 0.6539 | **34.41** |
| `LOWRANK2` | 0.4052 | 0.2166 | 0.1862 | 0.1667 |
| `LOWRANK3` | 0.4052 | 0.2191 | 0.1895 | 0.1683 |
| `HIER` | 0.4052 | 0.2205 | 0.1963 | 0.1711 |
| `MIX6meanPC` | 0.4052 | 0.2236 | 0.1974 | **0.1604** |
| `NAIVE_LINE` | — | 0.2310 | 0.2251 | 0.2243 |

No registered contrast reaches the 0.02 margin in any design; the largest real effect is
`MIX6meanPC − POOLED` = **+0.0070** at k = 3 (p < 1e-4, both intervals exclude zero, 5/5 seeds,
LOCO-stable, same sign in all five designs) — the comparison against the **deployed** baseline
that gen15 §7 never ran.  It is positive, significant, design-invariant and **three times under
the margin**, and it is reported as exactly that.

**Two defects.**  (i) The deployed leave-chemotype-out residual covariance is **indefinite on
100 % of its 430 estimates**; Ledoit–Wolf deepens the negative directions and greedy D-optimal
selection seeks them, which is why `LW` explodes rather than merely underperforming.  (ii)
Coverage: nominal 90 % intervals cover **98.8 %** at k = 3 and 92.6 % at k = 0, because
`NOISE_VAR = 0.09` alone exceeds the realised squared error at k = 3 (RMSE 0.248).  The
programme's measured-mode intervals are conservative by roughly 2.5× in width.

**Calibration of the zero-shot direction probability.**  The registered fix (inner-fold Platt)
makes it **worse** in every design (Brier 0.255 against the raw 0.167 and a constant base rate of
0.313).  The raw probability beats the base rate on Brier in all five designs, but its ECE under
BP is **0.139**, above the registered 0.10 bar, so the calibrated probability is **not** a
deliverable under the rule as written.

## 8. L6 — the cohort audit, and the only corpus advice this report will give

`gen16/cohort_audit.py` reproduces all three of gen13's key modes exactly, fingerprints included
(exact 521/90 `4c3c6628ea0be949`; relaxed 509/90; series 597/89), and reproduces the frozen
chemotype partition by re-clustering all 190 bundle structures.

**Every bundle extractant with ≥ 2 lanthanides measured anywhere is already in the cohort.  No
single relaxation adds a chemotype — the maximum over seven relaxations is +0.**  The gen5
`min_rows` failure does not recur.

The 100 excluded compounds are excluded because only one lanthanide was ever measured on them
(Eu 567 rows, Pr 96, Nd 35); 91 of the 100 come from one publication.  They carry **60 chemotypes,
53 of them absent from the cohort**, and 77 sit at ECFP4 Tanimoto < 0.7 from every kept extractant.

| counterfactual: a second lanthanide measured on … | extractants | Kish n_eff | ratio |
|---|---|---|---|
| — (frozen cohort) | 90 | 11.67 | 1.00× |
| one compound per absent chemotype (53) | 143 | **27.38** | **2.35×** |
| the 60 in absent chemotypes | 150 | 28.55 | 2.45× |
| all 100 | 190 | 12.27 | 1.05× |

Adding all 100 barely helps because 28 of them are themselves diglycolamides.  **This is the
programme's binding constraint stated as a laboratory action, and it needs no model.**

---

## 9. What is statistically supported, and what is not

**Supported** (registered rule met in all five designs, refuters unable to remove it, replicated
on withheld seeds):

- the direction call saves ≈ 2 of 6.9 expected measurements when candidates are pooled across
  laboratories (§2), **with the scope limitation of §2 attached**.

**Measured and reported, but not supported as claims:**

- `MIX6meanPC` over the deployed measured-mode baseline: +0.0070, real and design-invariant,
  three times under the pre-registered margin (§7);
- a featureless "biggest chemotype first" acquisition order beating random in all five designs
  (§6) — found by a refuter, and a refuter check can kill a claim, never resurrect one;
- composition-position descriptors correlating with the amplitude at ρ up to +0.83 (§10, lead 1
  for gen17) — post hoc, small n, from a geometry builder's recipe rather than from measurement.

**Not supported:**

- any new representation beating the lean block set zero-shot (L1 closed with no power, L2 closed);
- the corpus-expansion plan as a *chemistry* result (L4 killed by its featureless control);
- ranking candidates at k = 1 with the corpus (L3c null, and negative on rank correlation);
- any covariance estimator or calibrated interval at the 0.02 margin (L5);
- any cohort expansion available from relaxing a filter (L6).

**Nothing here supports an equivalence claim.**  The minimum detectable effect at 80 % power is
0.039 for the L2 curvature headroom; the L1 registered estimator has reliability 0.000 and so has
no detectable effect at all.

---

## 10. Defects found during this work

1. **Gen15 §7's `MIX6meanPC` was never compared with the deployed route.**  It is, here: the gain
   survives at +0.0070 (§7).
2. **The deployed residual covariance is indefinite on 100 % of its 430 leave-chemotype-out
   estimates** (§7).  It has not invalidated a published number — the BLUP is still a projection —
   but it makes any shrinkage estimator unsafe and it should be projected onto the PSD cone.
3. **The deployed measured-mode intervals over-cover by ≈ 2.5× in width** (§7), traced to
   `NOISE_VAR = 0.09`.
4. **Gen15 §3's two composition-controlled rows (n = 70/71) correspond to no construction in the
   frozen scripts**; the reproducible values are 79/+0.104 and 81/+0.203.  The conclusion is
   unaffected (every composition-controlled construction lands at \|ρ\| ≤ 0.22).
5. **Gen15's "39 complete series" is the energy-complete set; the geometry-complete set is 41.**
   Both reproduce exactly once the distinction is made.
6. **`n_fill` counts donor sites, not molecules** — nitrate is bidentate.  Any future use of the
   3D dataset's composition columns must use `n_NO₃ = ⌊n_fill/2⌋`-style decomposition, which L1
   verified against every one of the 1155 complexes.
7. **The frozen `key_mode="series"` uses the *relaxed* condition columns**, not exact + series; the
   docstring is silent and a re-implementation that reads it literally gets 557 cells instead of
   597.  A regression test now pins it.
8. **`gen13sep/arms_stage2.py:330` hashes bytes** with the salted built-in `hash()`, so two gen13
   arms (`S3_HIER`, `S3_EXT_LEVEL`) are not guaranteed bit-identical across processes.  Not on the
   gen14/gen15 bench path.
9. **A deviation from the sealed pre-registration in L1**, disclosed by the lead and confirmed by a
   refuter: the sealed text wrote `n_fill × γ_species`, the implementation used per-species
   molecule counts.  Both were run; the verdict is the same under either.
10. Orchestrator process failures are recorded in `REFUTATION_LOG.md` §6.

---

## 11. Guardrails — what a reader must not do with these numbers

1. **Do not quote "the model saves two of seven measurements" without the within-laboratory
   number.**  Pooled it is +1.91; inside one laboratory's candidate set it is +0.005 with an
   interval containing zero.  The second is the one a screening chemist experiences.
2. **Do not read L1 as "xTB is useless here".**  It is "this estimator has no power": reliability
   0.000, and attenuation-free variants reach +0.29 to +0.45.  The question is open and Stage 2 is
   the experiment that closes it.
3. **Do not use the L4 prospective ranking as a purchase list.**  It is ordered by a criterion its
   own refuter showed to be a proxy for family size.
4. **Do not quote the +0.83 composition-position correlations as a result.**  They are post hoc,
   n = 27–29, from a builder's recipe, and above a family-wise bar that was set for a different
   family.
5. **Do not treat `MIX6meanPC` as deployable.**  It is under the margin; the honest statement is
   that it is the only structured estimator in this programme that has ever beaten its pooled
   control, twice.
6. **Do not use the measured-mode intervals as calibrated.**  They cover 98.8 % at nominal 90 %.
7. **Do not quote any design-B number**, and do not quote `MEAN_CURVE` (0.622) as a baseline: it
   is worse than predicting no separation at all (0.589).
8. **Do not treat the L6 counterfactual Kish figures as a promise.**  They assume a second
   lanthanide is measurable on those compounds under comparable conditions, which the audit cannot
   verify.

---

## 12. Recommendation

**The one thing to do in a laboratory** is not a modelling change.  Fifty-three chemotypes are
absent from this corpus because someone measured europium alone; one second lanthanide on one
compound per absent chemotype would multiply the programme's effective sample size by **2.35×**,
which is larger than every modelling effect this generation measured put together.  That is L6,
it needs no model, and it has been the programme's standing recommendation since gen6.

**The one experiment to run on a cluster** is L1 Stage 2: 361 array tasks, ≈ 6.7 CPU-hours, one
core and 2 GB each, then one local command.  Gen16's contribution is that it now knows *why* it
matters — not because the correlation was promising, but because the only estimator gen16 could
build from existing data has zero reliability, so the question has never actually been asked.

**What to deploy is unchanged.**  `gen15_curve/scripts/g15_predict.py` still stands; gen16 adds no
zero-shot skill.  Add the direction-call filter of §2 to it as a **literature-wide screening
filter** with its scope limitation printed beside it, and project the residual covariance onto the
PSD cone before any future shrinkage work.

**Four things to pre-register for gen17**, in descending order of expected value.

1. **Composition-position as a size-match observable.**  `steppos_n_H2O` and its relatives reach
   ρ = +0.78 to +0.83 against the amplitude with no energy in them.  Either the radius at which a
   ligand's inner sphere reorganises is a real size-match observable — which would be the first
   new physics in the programme since gen2 — or it is an artefact of Architector's heuristics.
   Pre-register both hypotheses, a builder-independent replication, and a five-design MAE arm.
2. **`SIZE` acquisition.**  "Measure the biggest family first" beat random in all five designs
   from zero features.  It needs its own pre-registration, its own matched null and its own
   cell-matched control, and it is a cheap, honest corpus-growth result if it holds.
3. **`NOISE_VAR` and the PSD projection.**  Both are one-line changes to the deployed measured
   mode that will move every interval and every MAE in it; they must be pre-registered, not
   patched in.
4. **A reliability gate as a standing protocol rule.**  L1's null cost a full lead and was
   uninformative because nobody measured whether the descriptor reproduces itself.  **Any future
   descriptor must report its split-half or jackknife reliability before its correlation with a
   target is interpreted**, and an estimator with reliability below ~0.3 must not be used to close
   a lead.  That rule is this generation's most transferable output.

**And the sentence that has not changed.**  Zero-shot, this model answers *which way* and cannot
answer *which ligand*; one well-chosen measurement is still worth more than every modelling
difference in five generations; and the binding constraint is still eleven and a half effective
chemotypes.
