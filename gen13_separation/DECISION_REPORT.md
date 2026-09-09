# Gen13 — decision report

*Zero-shot lanthanide separation factors from the lanthanide-axis curve, on chemically unseen
extractants.  Cohort fingerprint `4c3c6628ea0be949`: 521 multi-metal cells, 90 extractants,
45 frozen chemotypes.  Every number carries its regime: {all pairs | adjacent (dZ = 1 + Nd–Sm) |
far (dZ ≥ 5)} × {extractant-macro | chemotype-macro | pooled} × {design B chemotype hold-out |
design A} × {zero-shot | one pair}.  Tables are under `headline_tables/`, per-extractant
tables under `metrics/`, bootstraps under `bootstrap/`.  Written 2026-09-08 after the locked
run of 2026-09-07/08; the gen13.1 section (§9) reports the exploratory follow-up ladder.*

---

## Headline

**Separation factors of chemically unseen extractants are predictable to about 0.50 log
units (macro MAE over all lanthanide pairs; the registered arm scores 0.501, the incumbent
0.495, the best exploratory bag 0.458), and to about 0.54 when the held-out chemistry's
laboratories are removed from training as well — the honest number for new chemistry from a new
group.  Neighbouring lanthanides: 0.19.  The corpus mean curve scores 0.60 and "the heavier
lanthanide always wins" 0.67.  The pre-registered curve model is exactly as good as the
incumbent row-wise design and not better (P1 fails); the apparent supremacy of experimental
conditions over molecular structure is an artefact of publication-correlated folds, and under
honest folds the compact binding-site blocks are the only representation that transfers (§5a);
what they carry is essentially one bit per extractant — which end of the lanthanide series it
prefers, called correctly three times in four (§9a); 3D "response" descriptors and an external
stability-constant prior add nothing; and one measured far pair of the new extractant is worth
more than every modelling difference in this generation.**

Six findings, in decreasing order of support.

1. **Chemistry-plus-conditions beats every no-chemistry baseline, robustly.**  `M_SELECTED`
   (the pre-registered inner-validation choice among curve models) beats the corpus mean curve
   by **+0.102** [0.047, 0.154], the per-pair-type mean by +0.104, the nearest-Tanimoto
   neighbour by +0.116 and the heavier-always rule by +0.173 — every interval excludes zero,
   5/5 seeds, no chemotype flips the sign (S1–S4 all pass the full rule).  The same holds on
   far pairs (+0.16 to +0.25) and, at the noise floor, on adjacent pairs (+0.013 to +0.040).
2. **The curve model is not better than the incumbent.**  P1, `M_SELECTED` versus
   `C_DIRECT_ROW` (row-wise log D trees, differenced in-cell): **−0.006** [−0.044, +0.034],
   p = 0.75, 2/5 seeds.  Not supported, and the minimum detectable Δ was 0.054.
3. **The apparent supremacy of experimental conditions is a laboratory fingerprint; the
   transferable signal is the compact binding-site representation.**  Under the primary folds a
   conditions-only model scores **0.471**, the best single arm, and adding every ligand block
   gives 0.495 (S5: −0.008, p = 0.77).  But the 64 condition columns identify a cell's
   publication with 94 % 1-NN accuracy, and under publication-masked folds (§5a) the same arm
   collapses to **0.625 — worse than the corpus mean curve (0.639) by only 0.014** — while
   conditions + physchem + donors + coordination reaches 0.558, a gain of **+0.096** [0.042,
   0.136] that passes the full decision rule.  Inside the ligand blocks the coordination block
   is what helps (+0.023 [0.008, 0.034], p = 0.002 under the primary folds), and ECFP with the
   206 RDKit descriptors dilute it in both designs.
4. **3D and external priors, used only as helpers, carry no signal.**  The RESP3D block (how
   the coordination sphere responds to the lanthanide swap): add −0.006 (p = 0.048, 0/5
   seeds), gated −0.003 [−0.008, −0.000] (p = 0.036; the inner gate admitted the block in 11 of
   25 folds and it did not pay), and the width-matched shuffled block is indistinguishable
   from the real one (add − shuffled −0.001, p = 0.81).  The aqueous-logK series prior: gated
   −0.000, add −0.001, shuffled −0.002.  Both residual variants hurt (−0.03 to −0.06).  The
   gate limited the harm to 0.003; it did not find anything to admit.
5. **The transferable chemistry is one bit per extractant** (§9a, second session).  Whether an
   extractant prefers the light or the heavy end of the series is called correctly 0.768 of the
   time under the publication-masked design from donor-set topology alone, against 0.559 for
   always saying heavy-selective (+0.210 [+0.067, +0.425], p = 0.002, replicated in five
   designs).  That bit plus the training fold's mean amplitude — two degrees of freedom — scores
   0.561 macro MAE where the full 209-column regression scores 0.552, and a perfect call would
   score 0.483.  The mechanism is chelate bite size: the amplitude correlates at −0.53 with the
   fraction of donor pairs within three bonds.
6. **One measurement changes more than any model.**  With one measured pair of the new
   extractant, the physics-basis curve shifted along its basis scores **0.396** against 0.481
   zero-shot (+0.084 [0.041, 0.114], p < 0.001) and beats a no-model line through the
   measured pair (+0.087 [0.033, 0.192], p = 0.0002) — but only when the measured pair is a
   *far* pair (0.367; an adjacent pair buys nothing, 0.483).  A whole measured series of the
   same extractant under any other condition brings the error to 0.25–0.28 on the same cells.

## 1. What was run

| stage | what | where |
|---|---|---|
| audit | cohort, target structure, noise floor, block inventory | `DATA_AUDIT.md`, `manifests/` |
| pre-registration | questions, arms, selection rule, decision rules, Addendum 1 | `PRE_REGISTRATION.md` |
| locked run | 12 registered arms + `M_SELECTED` + 5 exploratory arms, design B, 5 seeds | `predictions/B_primary` |
| block ablations | conditions-only, ECFP-only, without COORD, COORD + donors | `predictions/B_abl_*` |
| helpers | RESP3D (3D response) and LOGK (external logK) as add / gated / residual / shuffled | `predictions/B_3d`, `B_logk`, `B_3d_logk` |
| sensitivities | series key, relaxed key, design A | `predictions/B_series`, `B_relaxed`, `A_design` |
| one pair | five support draws per cell, three adapters, two nulls | `metrics/B_primary/fewshot_*` |
| gen13.1 | block subsets, bagging, gain calibration (exploratory) | `predictions/B_v2`, §9 |

12 invariant tests pass (`tests/`).  An adversarial code review (3 lenses, 41 findings, 2
refuters each) ran before the locked run; the defects it found and their fixes are listed in
`PRE_REGISTRATION.md` Addendum 1.  No locked prediction was read before those fixes.

## 2. The cohort and its noise floor

521 cells (extractant × publication × exact 64-condition key, NaN treated as a value) with two
or more lanthanides; 90 extractants, 45 chemotypes, Kish effective sample size **11.7
chemotypes**; 14,150 pairwise `log SF` observations per seed, 2,348 adjacent, 6,698 far.  The
diglycolamide chemotype holds 375 cells and 23 extractants: when it is held out the models
train on 103 cells.  Replicate rows under identical recorded conditions disagree by a median
0.30 log D per (cell, metal) (216 combinations), and a half-versus-half split of the nine
extractants with replicated pairs gives a pairwise MAE of 0.60 — the recorded conditions do
not fully identify an experiment.  Adjacent-lanthanide separation factors (sd 0.23) therefore
sit at the noise floor and are reported as secondary.

## 3. Zero-shot leaderboard (design B, 5 seeds, extractant-macro, `headline_tables/t1`)

| arm | all pairs | seed sd | chemotype-macro | adjacent | far | sign acc (|y| ≥ 0.3) | curve Spearman |
|---|---|---|---|---|---|---|---|
| `X_ENS_DIRECT+LOWRANK_K2` (exploratory) | **0.481** | 0.008 | 0.527 | 0.182 | 0.631 | 0.734 | 0.430 |
| `X_ENS_DIRECT+PHYSICS` (exploratory) | 0.485 | 0.010 | 0.528 | 0.182 | 0.638 | 0.734 | 0.450 |
| `C_DIRECT_ROW` (reference) | 0.495 | 0.019 | 0.537 | 0.184 | 0.654 | 0.722 | 0.411 |
| `M_LOWRANK_K2` | 0.496 | 0.010 | 0.547 | 0.189 | 0.652 | 0.730 | 0.417 |
| `M_LOWRANK_K1` | 0.497 | 0.004 | 0.543 | 0.193 | 0.652 | 0.719 | 0.450 |
| `M_PHYSICS_radius+radius_sq` | 0.498 | 0.007 | 0.543 | 0.189 | 0.657 | 0.744 | 0.462 |
| **`M_SELECTED` (primary)** | **0.501** | 0.006 | 0.545 | 0.190 | 0.662 | 0.738 | 0.459 |
| `B0_ZERO` | 0.589 | 0 | 0.609 | 0.207 | 0.807 | 0 | — |
| `B1_MEAN_CURVE` | 0.603 | 0.015 | 0.650 | 0.210 | 0.819 | 0.620 | 0.204 |
| `B2_PAIRMEAN` | 0.605 | 0.019 | 0.653 | 0.203 | 0.826 | 0.623 | — |
| `B3_NN_TANIMOTO` | 0.618 | 0.013 | 0.597 | 0.210 | 0.846 | 0.574 | 0.132 |
| `B4_HEAVIER_ALWAYS` | 0.674 | 0.005 | 0.737 | 0.230 | 0.913 | 0.630 | — |

The "heavier always preferred" rule scores 0.630 sign accuracy extractant-macro (0.855 pooled
over pairs, because the diglycolamides are both numerous and heavy-selective); every model
is above it (0.72–0.74).  `M_SELECTED` chose the tetrad physics basis 9/25, the Gd-break basis
5/25, rank-1 4/25, rank-3 4/25, radius-only 2/25, rank-2 1/25 — the candidates are
interchangeable within 0.006 and the selection adds nothing over any of them.

## 4. Pre-registered contrasts (`headline_tables/t2`)

Δ = reference − candidate per extractant, positive favours the candidate; chemotype-blocked
bootstrap, 10,000 shared draws; `passes_P1` = Δ ≥ 0.02 ∧ both intervals exclude 0 ∧ p < 0.05 ∧
≥ 4/5 seeds ∧ leave-one-chemotype-out sign-stable.

| id | contrast | Δ (all pairs) | percentile 95 % | BCa 95 % | p | MDE₈₀ | seeds | passes |
|---|---|---|---|---|---|---|---|---|
| **P1** | `M_SELECTED` − `C_DIRECT_ROW` | **−0.006** | [−0.044, +0.034] | [−0.046, +0.031] | 0.75 | 0.054 | 2/5 | **no** |
| S1 | vs `B1_MEAN_CURVE` | +0.102 | [+0.047, +0.154] | [+0.036, +0.146] | 0.001 | 0.075 | 5/5 | yes |
| S2 | vs `B3_NN_TANIMOTO` | +0.116 | [+0.020, +0.181] | [+0.039, +0.199] | 0.014 | 0.119 | 5/5 | yes |
| S3 | vs `B2_PAIRMEAN` | +0.104 | [+0.049, +0.166] | [+0.041, +0.159] | 0.001 | 0.083 | 5/5 | yes |
| S4 | vs `B4_HEAVIER_ALWAYS` | +0.173 | [+0.072, +0.309] | [+0.047, +0.285] | < 0.001 | 0.177 | 5/5 | yes |
| S5 | all blocks − conditions only (cross-run) | −0.008 | [−0.044, +0.040] | [−0.050, +0.032] | 0.77 | 0.061 | 2/5 | no |

Secondary metrics tell the same story: P1 is null on far pairs (−0.008, p = 0.82), on sign
accuracy (+0.016, p = 0.69), on the pairwise Spearman (−0.004) and on the curve Spearman
(+0.049, p = 0.49); S1–S4 are positive on every metric and significant on far pairs, and S1,
S3 and S4 (not S2, p = 0.08) on the pairwise Spearman.  On the far *similarity band* (19 extractants with no training
neighbour above Tanimoto 0.4) P1 is +0.050 [−0.045, +0.138], p = 0.30 — a hint that the
curve model degrades more gracefully far from training chemistry, unresolved at this size.

## 5. Where the signal is: block ablations (`headline_tables/t4`)

| blocks | `C_DIRECT_ROW` | `M_SELECTED` |
|---|---|---|
| **COND + MASSACT (conditions only)** | **0.471** | 0.493 |
| COND + MASSACT + ECFP | 0.512 | 0.502 |
| COND + MASSACT + PHYSCHEM + DONORS + COORD (no fingerprint) | 0.488 | **0.479** |
| all but COORD | 0.506 | 0.524 |
| all seven blocks (locked run) | 0.495 | 0.501 |

**A caveat that changes how the first row must be read.**  The 64 condition columns identify a
held-out cell's *publication* with 94 % leave-one-out 1-NN accuracy (chance 7 %) and its
extractant with 71 % (stage-2 diagnostic, `analysis/stage2/d6_condition_law/`).  A condition
vector is close to a laboratory fingerprint, and under a chemotype hold-out the same laboratory's
other chemotypes stay in training.  §5a repeats the block comparison on publication-masked folds
(design BP: chemotype hold-out, every training cell from a held-out publication removed).

Cross-run contrasts on `M_SELECTED` (same folds): all blocks versus without COORD **+0.023**
[0.008, 0.034], p = 0.002, 5/5 seeds (keeping the coordination block helps); all blocks
versus COORD + donors **−0.023** [−0.046, −0.000], p = 0.049, 0/5 seeds (the fingerprint and
the RDKit block dilute it); all blocks versus ECFP only +0.000; all blocks versus conditions
only −0.008.  For the direct model, all blocks versus conditions only is −0.024 (p = 0.57).
**The 2,254 mostly-binary structure columns are noise for this target; the 127 chemistry
columns that describe donors and coordination topology carry the small part of the signal
that is structural.**  This reproduces Gen12.2's "representation mixing" finding on a new
target.

Sensitivities: the `series` key (597 cells, rows with different reconstructed experiment
series kept apart) gives 0.501 / 0.501 for `C_DIRECT_ROW` / `M_SELECTED` (P1 −0.000); the
`relaxed` key (contact time and metal concentration dropped from the key) 0.517 / 0.519 (P1
−0.002); design A (exact-extractant hold-out, homologues allowed in training) **0.425 / 0.424**
— the chemotype hold-out costs 0.07, which is the size of the near-analogue advantage a
practitioner would see for a new member of a known family.

### 5a. Publication-masked folds (design BP) — this inverts §5

Design **BP** = the same chemotype hold-out plus a publication mask: every training cell whose
publication also appears among the held-out cells is dropped (`gen13sep/splits.py`; the integrity
check asserts zero publication overlap).  Masked training sets hold 76–492 cells against 103–507.
5 seeds, extractant-macro MAE:

| blocks | `C_DIRECT_ROW` | `M_SELECTED` | `B1_MEAN_CURVE` |
|---|---|---|---|
| COND + MASSACT (conditions only) | 0.625 | 0.654 | 0.639 |
| lean = COND + MASSACT + PHYSCHEM + DONORS + COORD | 0.569 | **0.558** | 0.639 |
| all seven blocks | 0.583 | 0.585 | 0.639 |

Cross-run contrasts on the same folds (`bootstrap/cross_run/BP_*.csv`): lean versus
conditions-only **+0.055** [+0.008, +0.120], p = 0.025, 5/5 seeds, LOCO-stable (direct model,
passes the full rule) and **+0.096** [+0.042, +0.136], p = 0.001, with far pairs +0.149
(curve model, passes on both); all blocks versus conditions-only +0.041, p = 0.10.  The
registered contrasts survive in weakened form: P1 still null (−0.002), S1 +0.054 (p = 0.004),
S2 +0.133, S3 +0.057 pass; S4 does not (+0.101, p = 0.11).

**Reading.**  The conditions-only result of §5 was a laboratory fingerprint: the 64 condition
columns identify a cell's publication with 94 % 1-NN accuracy, and with same-laboratory cells in
training a model can retrieve a held-out cell's series without any chemistry.  Once publications
leave training that arm falls to 0.625, which is **0.014 worse than the corpus mean curve**, and
the ligand blocks — physico-chemical scalars, the donor census and the coordination topology —
become the only thing that transfers, worth 0.055 to 0.096.  ECFP and the 206 RDKit descriptors
still dilute them (all blocks 0.014–0.027 worse than lean).  **The corrected statement of §5 is
therefore: the compact binding-site representation is the transferable signal, and the apparent
supremacy of conditions was an artefact of publication-correlated folds.**

**Controls: how much of that is the fingerprint and how much is lost data?**  BP removes 952
training cells and 609 condition series over the 25 folds, so two matched controls repeat it
without the publication structure.  **BR** drops the same number of training cells at random;
**BQ** drops whole random training publications, chosen closest-fit so it loses at least as many
cells *and* condition series as BP (971 / 922 — if anything harsher).

| design | conditions-only `C_DIRECT_ROW` | all blocks `C_DIRECT_ROW` |
|---|---|---|
| B (primary) | 0.471 | 0.495 |
| BR (same cells dropped at random) | 0.488 | 0.503 |
| BQ (whole random publications, cells and series matched) | 0.471 | 0.495 |
| **BP (publication-masked)** | **0.625** | **0.583** |

Read as three numbers rather than one delta: the conditions-only arm scores 0.471 with everything,
0.488 after the same quantity of training data is removed at random, and 0.625 after it is removed
as whole publications.  **Losing the data costs 0.017, losing unrelated laboratories costs 0.000, and losing the
laboratories that the held-out cells came from costs 0.154** — the conditions-only advantage was
the fingerprint.  Under the same random loss the curve
model degrades more than the row model (0.493 → 0.517 against 0.471 → 0.488), which is the
independent counterpart of the second session's finding that the amplitude is a chemotype-level
scalar fitted from about a dozen effective units: the parametrisation that spends its capacity on
that scalar is the first to break when units are removed.  BQ settles it: dropping whole *random* publications — matched on both cells and condition series,
and in fact removing more of each than BP does — costs **nothing at all** (0.471, identical to
the primary design).  Only removing the publications that the held-out cells come from costs
0.154.  The effect is specifically the co-occurrence of laboratory and test chemistry, not the
loss of data and not the loss of within-publication condition series.

## 6. Helpers: 3D response and external logK (`predictions/B_3d`, `B_logk`, `B_3d_logk`)

At the user's request 3D geometry re-entered only as a *helper*: for each ligand the slope,
intercept and residual spread of 15 first-shell descriptors (Ln–donor distances, coordination
number, donor charges, dipole, polyhedron gaps) regressed on the Shannon radius across the
ligand's xTB complexes (88 of 90 extractants have ≥ 3 metals).  Paired against the same
physics arm without the block (0.498):

| helper mode (`B_3d`, `B_logk`) | RESP3D | LOGK |
|---|---|---|
| add | −0.006 [−0.014, −0.000], p = 0.048, 0/5 seeds | −0.001 [−0.005, +0.001], p = 0.18 |
| gated (inner validation decides) | −0.003 [−0.008, −0.000], p = 0.036, 1/5 seeds; block chosen 11/25 folds | −0.000 [−0.002, +0.002], p = 0.96; chosen 15/25 |
| residual (ridge on cross-fitted residuals) | −0.056 [−0.105, −0.024], p < 0.001 | −0.034 [−0.070, −0.014], p < 0.001 |
| shuffled block (width-matched null) | −0.005 [−0.010, +0.003], p = 0.21 | −0.002 [−0.006, +0.001], p = 0.15 |
| add − shuffled | −0.001 [−0.012, +0.007], p = 0.81 | +0.001 [−0.002, +0.003], p = 0.59 |

In the joint run `B_3d_logk` (both blocks present, each helper neutralising the other) the
RESP3D gate lands at +0.000 (p = 0.98) and the LOGK gate at −0.002 (p = 0.27).  The real
blocks are indistinguishable from their permuted twins; the small RESP3D losses are what
adding 46 noise columns to a 199-column lean set costs.  This is the sixth
independent negative for 3D geometry in the programme and the first for an external
stability-constant prior (a transfer model that predicts the aqueous logK slope across the
series with cross-validated Spearman 0.67 on 273 external ligands, and still carries nothing
into extraction selectivity).  The gate protected the headline in both cases.

## 7. One measured pair (`headline_tables/t6`, `bootstrap/B_primary/fewshot_contrasts.csv`)

383 cells per seed with ≥ 3 metals, 89 extractants, five deterministic support draws per cell,
the remaining pairs scored:

| arm · adapter | macro MAE | vs zero-shot | vs no-model line |
|---|---|---|---|
| **`M_PHYSICS` · basis shift** | **0.396** | +0.084 [0.041, 0.114], p < 0.001, 5/5 | +0.087 [0.033, 0.192], p = 0.0002, 5/5 |
| `C_DIRECT_ROW` · rescale | 0.436 | +0.043 [0.019, 0.062], p < 0.001 | +0.048 [−0.021, +0.154], p = 0.21 |
| `M_SELECTED` · rescale | 0.440 | +0.045 [0.020, 0.066], p < 0.001 | +0.044 [−0.021, +0.147], p = 0.21 |
| `M_SELECTED` · basis shift (registered S6) | 0.453 | +0.032 [−0.018, +0.064], p = 0.16 | +0.031 [−0.039, +0.145], p = 0.47 |
| any arm · no-model line through the pair | 0.483 | — | — |
| `M_SELECTED` · zero-shot | 0.484 | — | — |

The registered S6 (basis shift on `M_SELECTED`) is not supported: `M_SELECTED`'s bases (tetrad,
Gd-break) shift in directions a single pair cannot identify.  The radius-basis shift on the
plain physics arm is the deployable adapter, and it works through the *far* support pairs
only: with a far measured pair 0.367, with an adjacent one 0.483 (the no-model line: 0.426
versus 0.672).  **Measure La versus a heavy lanthanide, not two neighbours.**  A one-pair
measurement alone (no model) equals the zero-shot model (0.483 vs 0.484) — the programme's
recurring lesson holds for separation too — and the model plus one far pair beats both.

**Stage-2 conditional update (a second session's diagnostic, `analysis/stage2/STAGE2_REPORT.md`,
`gen13sep/fewshot_stage2.py`).**  A cell's held-out residuals are additive in one per-cell metal
curve, so the best linear use of one measured pair is the conditional mean of the residual curve
given that measurement (a BLUP with the residual covariance between metals, estimated
leave-chemotype-out).  Over the same 383 cells and 89 extractants
(`metrics/B_primary/fewshot_stage2_leaderboard.csv`):

| support pair | best arm zero-shot | + conditional update | line through the pair, no model |
|---|---|---|---|
| widest dZ (e.g. La–Lu) | 0.455 | **0.251** | 0.272 |
| drawn at random | 0.464 | 0.362 | 0.483 |

Under the publication-masked design the lever is **larger**, because the zero-shot error it halves
is larger (`metrics/BP_lean/fewshot_stage2_leaderboard.csv`): with the widest pair measured
`C_DIRECT_ROW` goes 0.541 → **0.266**, `M_SELECTED` 0.531 → 0.274 and the corpus mean curve
0.609 → 0.278; with a random pair 0.555 → 0.412, 0.544 → 0.416, 0.626 → 0.451.  After one
well-chosen measurement the spread between the best chemistry model and no chemistry at all is
0.011 under BP and 0.020 under design B.

With the widest pair measured the chemistry model is worth about 0.02 over a straight line
through the measurement; with a random pair it is worth 0.12.  The same study on the gen13.1
curves (`metrics/B_v2/fewshot_stage2_leaderboard.csv`): `V2_BAG_MIX3` 0.435 → **0.249** with the
widest pair, 0.440 → 0.354 with a random pair; the corpus mean curve 0.574 → 0.268.  **The
0.023 zero-shot advantage of the gen13.1 bag over the locked ensemble shrinks to 0.003 once one
far pair is measured, and the spread between the best model and no chemistry at all is 0.02.**
Zero-shot gains of the size this generation can produce do not survive the deployment mode a
laboratory would actually use.  A greedy budget sweep gives
0.409 → 0.211 → 0.193 → 0.167 for 0, 1, 2, 3 measured pairs on 321 cells with ≥ 4 metals.  The
adapter's two constants move the widest-pair score only between 0.248 and 0.306.  This is the
deployment mode to use: **measure La against the heaviest available lanthanide, then update**.

**Ceiling.**  For the 457 cells whose extractant has another measured cell (26 extractants),
predicting the curve from the same extractant's other cells gives macro MAE 0.277 (mean of
the other cells) or 0.247 (nearest condition) against 0.461 for `C_DIRECT_ROW` on the same
cells.  About 0.2 log units of the zero-shot error is ligand-specific selectivity that a
single measured series recovers and structure does not.

## 8. What is statistically supported, and what is not

**Supported** (both intervals exclude zero, p < 0.05, ≥ 4/5 seeds, no chemotype flips the sign):

- every model beats the corpus mean curve, the per-pair-type mean, the nearest neighbour and
  the heavier-always rule on all pairs and on far pairs (S1–S4);
- keeping the coordination block helps the curve model by +0.023;
- adding the fingerprint and RDKit block to coordination + donors hurts by 0.023 (p = 0.049);
- one measured far pair plus the physics curve beats zero-shot and the no-model line;
- both residual helper variants are worse than no helper.

**Not supported**: P1 (curve versus direct), S5 (structure beyond conditions, curve model),
S6 as registered, any 3D or logK helper, any gain from the exploratory ensembles at the
0.02 bar (+0.014 [−0.003, +0.035], 5/5 seeds, p = 0.11 — reported, not claimed).

Under publication-masked folds (§5a) the supported set changes:

- the lean binding-site blocks beat conditions only by +0.055 (row model, passes the full rule)
  and +0.096 (curve model, and +0.149 on far pairs);
- a bag of lean arms beats the corpus mean curve by +0.102, and beats its own conditions-only
  member by +0.067 — both 5/5 seeds, both LOCO-stable;
- the conditions-only arm is *worse* than the corpus mean curve;
- dropping the same data at random costs 0.017 and dropping unrelated whole publications costs
  0.000, so the 0.154 gap is the laboratory fingerprint and not the data;
- the direction of selectivity is predictable from donor topology in all five designs
  (+0.210 to +0.263 over the constant baseline, every p ≤ 0.0022), and a two-parameter model
  built from it beats the corpus mean curve by +0.060 (p = 0.027) while a perfect direction call
  beats the full regression by +0.070 (p = 0.006).

**Nothing here supports an equivalence claim**; the MDE at 80 % power is 0.054 for P1.

## 9. gen13.1 — exploratory follow-up ladder (`predictions/B_v2`, same folds, 5 seeds)

Motivated by §5 (block dilution), §3 (candidates interchangeable, ensembles help) and the
dispersion diagnostic (predictions 0.43–0.56 as dispersed as the truth), the follow-up ladder
tried block subsets (`cond` = COND + MASSACT; `lean` = COND + MASSACT + PHYSCHEM + DONORS +
COORD), bagging of bases with the row model, and a gain calibration chosen on inner
validation.  Everything here is exploratory: the arms were designed after the locked run was
read, on the same folds.

| arm | all pairs | seed sd | far | sign acc | curve Spearman |
|---|---|---|---|---|---|
| **`V2_BAG_MIX3`** = mean(direct@cond, physics@lean, rank-2@lean) | **0.458** | 0.010 | 0.591 | 0.771 | 0.499 |
| `V2_BAG_MIX5` (+ direct@lean, rank-1@lean) | 0.459 | 0.008 | 0.592 | 0.773 | 0.510 |
| `V2_BAG4@lean` (direct, physics, rank-1, rank-2 on lean) | 0.468 | 0.006 | 0.606 | 0.765 | 0.508 |
| `V2_DIRECT@cond` | 0.471 | 0.013 | 0.608 | 0.755 | 0.487 |
| `V2_CAL_BAG4@lean` (gain by inner validation) | 0.475 | 0.017 | 0.615 | 0.765 | 0.508 |
| `V2_LOWRANK_K2@lean` | 0.477 | 0.014 | 0.617 | 0.766 | 0.515 |
| `V2_PHYSICS@lean` | 0.482 | 0.004 | 0.628 | 0.771 | 0.519 |
| `C_DIRECT_ROW` (all blocks, reference) | 0.495 | 0.019 | 0.654 | 0.722 | 0.411 |

Contrasts against the reference (chemotype-blocked, 10,000 draws): `V2_BAG_MIX5` **+0.037**
[+0.002, +0.070], BCa [+0.003, +0.071], p = 0.038, 5/5 seeds, LOCO-stable — the only
exploratory arm that clears the full P1 rule; `V2_BAG_MIX3` +0.038 [−0.006, +0.074], p = 0.089,
5/5 seeds; `V2_BAG4@lean` +0.028, p = 0.07; `V2_DIRECT@cond` +0.024, p = 0.57 (4/5 seeds, wide:
the conditions-only model is the most seed-variable arm).  Against `V2_DIRECT@cond` the mixed
bags are +0.013 (p = 0.5): **the gain over the locked reference is mostly the block pruning and
the averaging of two inductive biases; nothing in it is a new source of chemistry.**

Two branches are closed by this run: the gain calibration chose g = 1.0 in 16 of 25 folds and
scored worse than its uncalibrated bag (0.475 vs 0.468), confirming the stage-2 diagnostic that
the amplitude shrinkage is per-extractant and no scalar undoes it; and the curve models on
conditions only (`V2_PHYSICS@cond` 0.499, `V2_LOWRANK_K2@cond` 0.500) are the worst
non-baseline arms, so the row model is the one that turns conditions into a curve.

Under the exact-extractant hold-out (design A, near analogues in training) the ordering
inverts: conditions-only `C_DIRECT_ROW` scores **0.465** against 0.425 with all blocks — for a
new member of a known family the ligand blocks are worth 0.04, which is the regime a
practitioner most often faces and the one the primary design deliberately excludes.

**Tested by the second session and null (`metrics/S2_main_leaderboard.csv`,
`bootstrap/S2_main_contrasts.csv`, `analysis/stage2/STAGE2_REPORT.md` §4).**  A row learner on
the *centred* row target (`S2_CENTRED_ROW_basic` 0.489) is +0.006 over `C_DIRECT_ROW`, p = 0.71,
not LOCO-stable; sign accuracy +0.043 and pair Spearman +0.053 are 5/5 seeds positive but their
intervals include zero.  A hierarchical extractant-level ligand model with a condition head
(`S3_HIER` 0.501) is the worst arm and loses to its own condition-free ablation; extractant-
balanced and reliability weights move nothing; richer metal descriptors are slightly worse than
the two original metal columns.  These close the "target parametrisation" and "weighting"
branches at ≤ 0.006.

**Under publication-masked folds (`predictions/BP_v2`, same 11 arms, byte-identical pairs).**
The ordering changes and the conditions-only member collapses:

| arm | BP | B |
|---|---|---|
| `V2_BAG4@lean` (direct, physics, rank-1, rank-2 — all on lean blocks) | **0.536** | 0.468 |
| `V2_BAG_MIX5` | 0.545 | 0.459 |
| `S4_CENTRED_ROW@lean` (second session) | 0.557 | — |
| `V2_BAG_MIX3` | 0.557 | 0.458 |
| `S4_BAG_MIX3C` (MIX3 with a centred-row member) | 0.571 | — |
| `S2_CENTRED_ROW_basic` | 0.575 | 0.489 |
| `C_DIRECT_ROW` (all blocks) | 0.583 | 0.495 |
| `S3_EXT_LEVEL` | 0.585 | 0.496 |
| `V2_DIRECT@cond` | 0.625 | 0.471 |
| `B1_MEAN_CURVE` | 0.639 | 0.603 |

Contrasts (chemotype-blocked, same folds): `V2_BAG4@lean` beats the mean curve by **+0.102**
[+0.054, +0.143], p < 0.001, 5/5 seeds; `V2_BAG_MIX5` +0.094; `V2_BAG_MIX3` +0.081; and
`V2_BAG_MIX3` beats its own conditions-only member by **+0.067** [+0.029, +0.123], p < 0.001,
5/5 seeds, LOCO-stable — the same inversion as §5a inside a single bag.  Against
`C_DIRECT_ROW` the bags are +0.026 (p = 0.22).  The second session's centred-row swap does not
pay under BP (`S4_BAG_MIX3C` −0.013 against `V2_BAG_MIX3`, 0/5 seeds; `S2_CENTRED_ROW_basic`
+0.008 against the row reference, p = 0.66; `S3_EXT_LEVEL` −0.001), consistent with its own
5-seed null on design B.  `S4_CENTRED_ROW@lean` sits 0.026 above `C_DIRECT_ROW` in the table,
but that is the lean block subset doing the work, not the centring: the centring contribution is
the +0.008 that does not clear its own interval.  **Under the honest design the right configuration is the pure lean
bag `V2_BAG4@lean`, not the mixed bag that contains a conditions-only member.**

**Deployed configuration.**  `scripts/g13_predict.py` fits **`V2_BAG4@lean`** — direct row model,
physics-basis curve, rank-1 and rank-2 curves, all on conditions + physchem + donors +
coordination — on all 521 cells, and freezes beside it the 14 × 14 residual covariance built from
cross-validated curves, so that one measured pair updates the whole curve by the conditional mean
(§7); the greedy design over that covariance names La–Lu as the pair to measure first (then
Ce–Tm).  The mixed bags score 0.458 under the primary design but rely on a conditions-only
member that does not transfer across laboratories (0.557 versus 0.536 under BP), so the lean bag
is the configuration to deploy.
Its exploratory zero-shot score on unseen chemotypes is 0.458 (this section) and about 0.42
on near analogues (design A); read it with §5a in mind.

## 9a. Stage 3 — the transferable chemistry is one bit per extractant

*Second session, `analysis/stage3/STAGE3_REPORT.md`, with an independent from-scratch audit in
`analysis/stage3/verify_direction/VERDICT.md`.  Every number below is taken from the files
`analysis/stage3/README.md` marks as authoritative — the `*_FIVE_DESIGNS` accuracy and gain tables
and the `_value_BP` tables — and none from `s3_direction_accuracy.csv`, whose constant-baseline
column is a superseded chemotype-weighted majority (0.35, below chance) and which a later run
overwrote with two designs only.  Bench: `gen13sep/amplitude_bench.py`, which
reproduces the locked `M_PHYSICS_radius+radius_sq` arm to four decimals and scores a candidate
over 25 folds in 40 s.  Every candidate below was run under at least three hold-out designs
after the first one won under a single design and lost under four.*

Stage 2 showed that 82–87 % of the pairwise error is error in one number per cell: the
coefficient of the standardised Shannon radius, whose sign is the *direction* of selectivity
(negative = heavy-selective).  Stage 3 predicts that sign directly, out of fold, on the 289
well-determined cells of 82 extractants in 40 chemotypes, one vote per extractant:

| design | donor topology (39 columns) | full lean set (209) | always heavy-selective | gain of topology | p |
|---|---|---|---|---|---|
| B | **0.822** | 0.767 | 0.559 | +0.263 [+0.105, +0.503] | 0.0004 |
| BR | 0.816 | 0.766 | 0.559 | +0.257 | 0.0006 |
| BQ | 0.814 | 0.761 | 0.559 | +0.256 | 0.0008 |
| A | 0.798 | 0.804 | 0.559 | +0.240 | 0.0020 |
| **BP** | **0.768** | 0.721 | 0.559 | **+0.210** [+0.067, +0.425] | 0.0022 |

For an extractant whose whole chemical family is absent from training *and* whose publications
have been removed, the direction of its lanthanide selectivity is called correctly about three
times in four from the topology of its donor set — counts of bonds on the 2D graph, computable
for any candidate structure with no conformer, no DFT and no measurement.

**What that bit is worth on the report's own metric** (`s3_direction_value_BP.csv`, design BP,
extractant-macro MAE): direction from topology × the training fold's mean amplitude magnitude ×
the corpus curvature — two degrees of freedom — scores **0.561**, against 0.552 for trees on the
full 209-column lean set (difference +0.009, p = 0.76) and 0.622 for the corpus mean curve
(+0.060 [+0.008, +0.109], p = 0.027, passes the full rule).  A *perfect* direction call scores
**0.483** and beats the full regression by +0.070 [+0.024, +0.122], p = 0.006.

**The honest reading of three generations of architecture work: essentially all the transferable
ligand chemistry in this corpus is one bit per extractant, the regression captures about half of
it, and the remaining headroom is a classification problem with 82 labelled units rather than a
regression with about twelve effective ones.**

**What the audit changed, restated rather than footnoted.**  The comparison set is 209 columns,
not 137.  Topology is *no worse than* the full set, not better (+0.035 [−0.015, +0.070], p = 0.20
against the true 137-column set).  Its honest increment over the plain 13-column gen6 donor
census is +0.075 [+0.022, +0.151], p = 0.006, because the census alone already reaches 0.694 —
the +0.21 is the gain over the constant baseline, not over chemistry.  Two disclosures: the
advantage concentrates in weakly directed extractants (+0.061, p = 0.18 among strongly directed
ones), and the 39 topological columns take only 22 distinct values over 82 extractants, so 84 %
of held-out cells have a bit-identical training row — forty label permutations give a null of
−0.067 ± 0.062, so this is not leakage, but "the family was held out" is then a statement about
the chemotype label rather than about the representation.

**Nine branches closed**, each under at least two designs: kernel ridge, PLS, ridge, kNN and L1
boosting (0.03–0.13 worse than the trees already in use); in-fold univariate screening to 6, 12
or 24 features; amplitude-only prediction with fold curvature (+0.013 under BP, lost under the
other four designs); de-shrinking the ridge-attenuated training target; precision weighting;
well-determined-cells-only; and averaging predictions across an extractant's cells.  The
*magnitude* of selectivity is not predictable either (three-way class 0.482 against 0.462 for the
majority).

**Chemistry.**  Donor separations are bond counts, and a pair `d` bonds apart closes a
`(d+2)`-membered chelate ring, so this is the cavity-size argument recovered from data: a tighter
chelate bite prefers the smaller, heavier lanthanides.  Spearman of the extractant amplitude with
the fraction of donor pairs within three bonds is **−0.530** [−0.723, −0.155],
leave-one-chemotype-out stable, −0.401 with the diglycolamides removed entirely and −0.489
controlling for the acid.

## 10. Defects found during this work

1. The iterative-SVD completion of the first draft never converged and let imputed entries
   drift; replaced by alternating least squares on observed entries (Addendum 1).
2. The per-cell ridge shrank data-basis coefficients by 33 % and physics ones by 3 %; both
   families now use the same row norm.
3. The row model's chemotype weights were broadcast per row; fixed to per-cell mass.
4. The paired p-value was drawn from a different RNG stream than the intervals; unified.
5. The "curve Spearman" was a Spearman over pair contrasts; renamed and a true curve-level
   statistic added.
6. The sign-accuracy yardstick was quoted on the wrong subset (0.681 adjacent, pooled) — the
   right numbers are 0.630 macro / 0.855 pooled on strong pairs — and is now an arm.
7. The one-pair `rescale` adapter could flip and amplify a noisy support pair; gated at the
   noise floor and clipped to non-negative gains.
8. Under pandas 3 a missing-chemotype guard was inert and `stack()` kept NaN rows.
9. Under the `relaxed` key the per-cell condition values were the first row's; now NaN when
   they differ.
10. Float condition values entered the exact key as raw `str()`; rounded to 9 significant
    digits (no cell changed).

## 11. Guardrails — what a reader must not do with these numbers

1. **Do not quote any design-B number as the expected accuracy for new chemistry from a new
   laboratory.**  The condition columns identify the publication; the honest figure is the
   publication-masked one (0.536 for the deployed bag against 0.468), and a conditions-only model
   is worse than predicting the corpus mean curve there.
2. Do not quote 0.458 or 0.481 (exploratory bags) as the registered model's score: the registered
   arm is `M_SELECTED` at 0.501, and their gain is inside the 0.02 bar.
3. Do not read adjacent-pair MAEs as skill: 0.18–0.19 sits at the replicate noise floor and the
   mean curve scores 0.21.
4. Do not quote the pooled sign accuracy of the heavier-always rule (0.855) against a model's
   macro sign accuracy; the macro yardstick is 0.630.
5. Do not treat "conditions only is best" as "ligand chemistry does not matter": the
   coordination block helps and the same-extractant ceiling shows 0.2 log units of ligand
   selectivity exist — it is not recoverable from structure on 90 extractants in 45 chemotypes.
6. Do not use the model for extractant classes absent from the cohort (100 of 190 bundle
   extractants have a single metal and never entered); `max_train_tanimoto` in the predictor
   output says how far a query is.
7. Do not calibrate with an adjacent pair.

## 12. Recommendation

The two load-bearing sentences of this generation, both measured and pointing at different
parts of a deployment: **the compact binding-site blocks are the only representation that
transfers across laboratories, worth 0.055–0.096 zero-shot (§5a); and a single well-chosen
measurement is worth about 0.27 and compresses every model difference to about 0.01 (§7).**  The
first decides which ligands to screen, the second what to do once a candidate is in the laboratory.

**Deploy `scripts/g13_predict.py`** — the lean bag `V2_BAG4@lean` (direct row model plus
physics-basis, rank-1 and rank-2 curves, all on conditions + physico-chemical scalars + donor
census + coordination topology), fitted on all 521 cells with the residual covariance frozen
beside it.  Expect **0.47 macro MAE on unseen chemotypes, 0.54 when that chemistry also comes
from a laboratory the model has never seen, 0.42 for a near analogue** — then **measure the
widest pair you can** (La against Lu) under the intended conditions and apply the conditional
update, which takes the error to about **0.25–0.27**.

**Two protocol rules for the next generation, both earned by a specific failure here.**

1. **Score every candidate under all available hold-out designs and report the full set, never a
   chosen subset; a candidate is supported only if the sign is consistent across them.**  The
   weaker wording — "check it under a design that breaks a different correlation" — leaves the
   choice of the second design to whoever proposes the candidate, and that degree of freedom
   would have saved the amplitude-only arm: it scored +0.013 under BP, the design that breaks the
   laboratory correlation, and only died at −0.012, −0.007, −0.012 and −0.024 under B, BR, BQ and
   A.  Which design would kill it was not predictable from what it was meant to fix.  With
   `gen13sep/amplitude_bench.py` a five-design table costs about four minutes, less than the
   argument about which design to pick.
2. **Report the increment over the nearest trivial competitor, not only over the constant
   baseline.**  The audit's most useful correction to §9a was not that a number was wrong but
   that +0.21 is the gain over "always heavy-selective" while the gain over a plain 13-column
   donor census is +0.075.  Both are true; only the second says whether the new representation
   earned its place.  A pre-registration that demands both comparisons catches this before an
   audit has to.

**A convention for the programme, not a one-off decision about this bag.**  On this corpus any
arm that wins under design B through a conditions-only member is winning through the laboratory,
so **design B must not select the deployed model**.  Selection belongs to the publication-masked
design; design B stays as the comparison to gen2–gen12 and as the near-analogue regime, and any
number quoted for new chemistry from a new group is the BP number.

**Reframe the remaining zero-shot work as a classification problem.**  Stage 3 (§9a) measures
that essentially all the transferable ligand chemistry here is one bit per extractant — which end
of the series it prefers — that donor topology calls that bit correctly about three times in four
under the publication-masked design, and that a perfect call would be worth +0.070 macro MAE over
the full regression.  Eighty-two labelled extractants is a far better-posed problem than a
regression over about twelve effective units.

**The two headline statements sit at different scales and must not be averaged.**  Binding-site
blocks are worth 0.055–0.096 zero-shot and are the only representation that transfers; one
well-chosen measurement is worth about 0.27 and compresses every model difference to about 0.01.
A reader who takes only the second will under-invest in the representation that decides which
ligands ever reach the laboratory at all.

Do not spend further effort on 3D geometry, external stability-constant priors, gain
calibration, tetrad basis terms or target reparametrisations: each is measured at ≤ 0.006 or is
a matched null.  Do not add more fingerprint — ECFP and the extended RDKit block fail to
transfer across laboratories.  Two things would move the number materially, in this order:
**more independent chemotypes measured across several lanthanides** (the programme's standing
recommendation since gen6, and the only lever on the 0.05–0.10 that structure currently
carries), and **a better compact description of the binding site** — the block that already
carries the transferable signal is a 2D topological census, and its physical successor has not
been built.
