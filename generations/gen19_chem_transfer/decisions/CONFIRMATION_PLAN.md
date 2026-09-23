# CONFIRMATION_PLAN — the claims frozen before the confirmation run

*Pre-registration §15 ("at most **5 claims** are frozen in `decisions/CONFIRMATION_PLAN.md`"), under POST-HOC
addendum 3 item 1 (eligibility), addendum 3 item 2 (the deployed configuration is **M0 = B5**) and addendum 4
(H3 scope, the 20 h cap, the refits and the power-check wording). First written 2026-09-22; **revised
2026-09-23 after the H3 run completed and was scored**, from files only. Every number here is the **selection
half, discovery seed 104729** — a half the pre-registration itself calls optimistically biased. Nothing in this
file was computed on a withheld seed, on the confirmation half, or on V6: no such number exists anywhere in the
repository. Paths are relative to `generations/gen19_chem_transfer/`.*

**Status: NOT YET AUTHORISED TO RUN.** This file freezes *what* would be scored. The run itself needs the
orchestrator's go/no-go (§7): ~174 h of serial compute, ~89 h of wall clock at 2 workers.

---

## 1. Gate state this plan rests on

| fact | value | source |
|---|---|---|
| prereg footer digest | `135842499a86…5641` | `manifests/prereg_sha256.txt` |
| below-footer addenda | **4**, sha256 `22c61a442912…8579d7` | `manifests/digest_registry.json → stages.h3` |
| stop rule | **false** (M2 vs B3i passes items 1,2,3,5; B6 vs B3i fails) | `evaluation/discovery/decisions/stop_rule.json → stop` |
| ladder | M1 kept **false**, M2 kept **false**, retained **M0**; M3–M7 `not_run` | `evaluation/ladder/decisions/ladder.json → retained_final` |
| deployed configuration | **M0 (= B5)** | addendum 3 item 2; `evaluation/h3/h3_summary.json → deployed.arm` |
| H3 (§11) | run and scored on **10 of 12** contrast record sets (the two `ACT_PERMUTED@V1` legs are `INCOMPLETE_GUARD_FAILURE`); **B5 UNDECIDED, B6 UNDECIDED**; **F4 HOLDS** on the conservative reading (either registered interval, any registered design: BCa on V2), and does **not** hold on the percentile-only reading | `evaluation/h3/h3_verdicts.json`, `h3_f4.json → failure` = true, `failure_percentile` = false, `failure_bca` = true; `h3_summary.json → contrasts_not_run` |
| H3 compute | 5.7842 h of the 20.0 h cap used, `exhausted` false | `evaluation/h3/decisions/wall_clock.json → budget` |
| H3 records | 444 scored + 1 exploratory verify against **the entry each was written under** (444 `superseded`, 0 `current`); **0 stale** | `h3_summary.json → refits.{stale, matched_entry_counts, operative_reading}` |
| eligible for freezing (discovery §19) | **3** contrasts | `evaluation/discovery/decisions/decisions.json → freezing_candidates[*].eligible_for_freezing` |
| eligible for freezing (§19 H3 row) | **1** contrast (§2, C4; no further addendum needed) | `evaluation/h3/h3_contrasts.csv → verdict_freezing_screen` = PASS |
| withheld-seed commitment | `65e8ae8ceb8e…f82` | `manifests/confirmation_seeds_sha256.txt` (§15) |

The eligibility rule is addendum 3 item 1 verbatim: *every R19 item that R19 evaluates in discovery is PASS and no
item is FAIL*; item 4 is NOT_EVALUATED by addendum 1 item 3 and does not disqualify. **Four** registered contrasts
meet it — three from the discovery scorer and one from the H3 tables. Nothing else in §19 does: every other
registered contrast either FAILs an item or was never run.

---

## 2. The frozen claims (4 of the at most 5)

**C1–C3** are on **V5-primary**, candidate **M2** (the factorised arm, §6), metric **unit-macro MAE of log D over
the held-out metal × extractant cells**, Δ = MAE(comparator) − MAE(M2). All three are scored from the **same** set of
confirmation-half fits, so C2 and C3 cost almost nothing beyond C1.

| # | claim | family | arm (candidate) | comparator | margin | design(s) | discovery Δ (selection half) | discovery verdict |
|---|---|---|---|---|---|---|---|---|
| **C1** | M2 vs B3i @ V5 | primary (H1, S1(a)) | **M2** | **B3i** — the stronger of B3x/B3i, §5/§9 | **δ5 = 0.10568948790529951** | V5-primary; M2 on `batched_max4`, B3i exact | **+0.262976** | UNDECIDED (item 4 only); freezing screen PASS |
| **C2** | M2 vs B0 @ V5 | S1(b) | **M2** | **B0** (global mean) | **0.05** | V5-primary, same | **+1.022150** | UNDECIDED (item 4 only); freezing screen PASS |
| **C3** | M2 vs B6r0 @ V5 | S1(b) | **M2** | **B6r0** (rank-0 B6) | **0.05** | V5-primary; M2 `batched_max4`, B6r0 exact | **+0.455132** | UNDECIDED (item 4 only); freezing screen PASS |
| **C4** | B6:WITH vs B6:ACT_PERMUTED @ V2 | H3 (§19 H3 row) | **B6:WITH** | **B6:ACT_PERMUTED** (actinide log D permuted within system × publication group) | **0.05** (registered: addendum 2 §3 H3 `margins`) | V2 Ln(III) folds, `V2__element_exact` | **+0.081519** | UNDECIDED (item 4 only); freezing screen **PASS** |

Sources: C1–C3 `evaluation/discovery/contrasts_registered.csv` rows 0/4/6 (`point`, `margin`,
`verdict_freezing_screen`) and `decisions.json → S1_components.{S1a_M2_vs_B3i, S1b_M2_vs_B0, S1b_M2_vs_B6r0}.point`.
C4 `evaluation/h3/h3_contrasts.csv`, row `B6:WITH vs B6:ACT_PERMUTED` / design V2 / `cluster_unit = metal_state`:
`point` 0.0815187, `margin` 0.05, percentile 95 % [0.0355933, 0.1364510], BCa 95 % [0.0411923, 0.1475570],
`p_two_sided` 0.0000, `loco_min` +0.0578919, `mde_80` 0.0727361, `verdict_freezing_screen` PASS,
`r19_verdict_full` UNDECIDED; items in `h3_r19_items.csv` (1 PASS, 2 PASS, 3 PASS, 4 NOT_EVALUATED, 5 PASS, 6 PASS).

**C4 needs no new addendum on the margin — the earlier draft of this file was wrong on both halves** (task X finding
numbers VH-01). (i) The 0.05 V1/V2 margin **is registered**: POST-HOC addendum 2, *"3. H3" → `margins`*, reads *"δ5 on
V5 Ln cells, 0.05 on V1 / V2"* (`preregistration.md:1758`), and the same addendum's discovery reading says *"`margins`:
δ5 on V5; 0.05 for S1(b) and for V1 / V2 (§9 names none; the floor of the δ5 formula)"* (`:1741`). The addenda are
below-footer registered text, so this is registration, not a scorer's gloss; `h3_summary.json → readings.margins` now
says so. (ii) **R19 item 1 reads the point estimate**, not an interval bound: §8 defines it as *"the point estimate Δ ≥
the design's margin δ"* (`:996`). C4's point is **+0.0815187 ≥ 0.05**, and `h3_r19_items.csv` key
`B6:WITH vs B6:ACT_PERMUTED@V2` records item 1 **PASS**. The BCa lower bound 0.0411925 is what item **2** reads, and
item 2 requires the interval to exclude **0**, which it does. **C4 is eligible exactly as scored**; no addendum is
needed on this ground.

One honest caveat stays, and it is about power, not eligibility: `mde_80` = 0.0727361 exceeds the 0.05 margin, so a FAIL
of item 1 here would have been uninformative. That is not a statement that item 1 cannot pass — it did pass. D03's power
table said the opposite for this contrast and has been corrected (the label is now gated on the contrast's own item-1
status).

**None of C1–C3 is a null, and the power artefacts say so.** The §8 signal-injection check gives all three
κ_min **0.1** and verdict **POWERED_NOT_A_NULL** (`evaluation/power/power_checks.json → checks[0,2,3]`).
**Correction to addendum 4 item 4's own wording, for the report:** the addendum says these three "point estimates
favour the comparator" — that is inverted. Δ = MAE(comparator) − MAE(candidate), and
`power_checks.json → checks[0].uninjected.point` = **+0.262976**, favouring the **candidate** M2 (likewise
+1.022150 and +0.455132). The operative rule and all four outcomes are unaffected.

**Slot 5 is deliberately left empty.** §15 says *at most* five. No further registered contrast of §19 is eligible,
and nothing may be added later (§15: "No sixth claim is added and nothing is re-run").

**A frozen claim's candidate need not be the retained ladder configuration.** M2 is the H1 candidate; **M0 is what
is deployed**. §19 restricts the registered family and §15 freezes contrasts that pass the R19 screen — neither
requires the candidate to be the deployed model, and addendum 3 item 2 says so outright: *"The factorised M2 is
still the H1 candidate and is reported as such: H1 is about M2 against the lookup, and §31's architecture question
is answered by the M2 − M0 contrast, not by what is deployed."* Read otherwise, **zero** claims could be frozen and
the single V6 run, S2 and brief §34's applied Pr/Nd question could never happen — exactly what addendum 3 item 1
was written to prevent. C4's candidate is B6, a §11 transparent reference, and is not deployed either.

### 2.1 R19 items evaluated at confirmation

Applied to the confirmation half with the 5 withheld seeds (§8 R19; §15). Identical for C1–C3; C4 differs only in
its cluster unit and sensitivity list (§2.2, §2.3).

1. **Item 1** — point Δ ≥ the claim's margin above.
2. **Item 2** — percentile **and** BCa 95 % intervals exclude 0 under **every** registered cluster unit.
3. **Item 3** — two-sided bootstrap p < 0.05 under **every** registered cluster unit (the conservative reading,
   resolved 2026-09-15).
4. **Item 4 — the item discovery could not evaluate.** Δ > 0 in **5 of 5 withheld seeds** on the confirmation half,
   exactly as registered (§8 item 4; addendum 3 item 1 restates that it is required here). This is the whole point
   of the run: in discovery it is NOT_EVALUATED because addendum 1 item 3 ran learned arms on seed 104729 alone
   (`decisions.json → S1_components.S1a_M2_vs_B3i.items[3].detail`; for C4 `h3_r19_items.csv`, item 4
   NOT_EVALUATED). **C4 is deterministic in neither arm** — B6's conformal split and the ACT_PERMUTED permutation
   are both seeded — so item 4 is a real test for it too.
5. **Item 5** — leave-one-cluster-out: removing any single cluster does not make Δ ≤ 0.
6. **Item 6** — Δ > 0 in every sensitivity of the **reduced set (addendum 1 item 4)**, listed in §2.3.

Bootstrap: paired cluster bootstrap, **10,000 resamples, `numpy.random.default_rng(19)`** (§8). BH-adjusted p is
printed per family beside raw p and **decides nothing** (§8 multiplicity).

### 2.2 Cluster units and exact scoring population

**C1–C3.**
- **Scoring population:** the **confirmation half** of `V5__primary` only (`feasibility_halves.csv`; the half that
  contributed to no ladder decision, claim or preferred-model choice, §15). **105 scored cells.**
- **Primary cluster:** `extractant_system_key` — **19 systems** (§8 table: 37 scored systems, 18 selection / 19
  confirmation; `tables/preseal_contrasts.csv`, half `confirmation`, design V5, `n_clusters` = 19).
- **Secondary cluster:** publication group of each cell's majority rows — **20 groups** (same rows,
  `cluster_unit` = `publication_group`). Against the selection half's 18 / 27: **the confirmation half has fewer
  publication-group clusters, so intervals will be wider than discovery's at the same effect size.**
- **Folds:** `folds/INDEX.json → designs.V5__primary__batched_max4.by_half.C` = **135** folds across the 5
  discovery seeds, i.e. 27 batches per seed. The withheld seeds' colourings **do not exist yet** and must be built
  first by `scripts/g19_build_folds_max4.py` under the same vertex-order rule (§7 item 6); expect 27 ± 1 per seed,
  ≈ **135** folds.

**C4.**
- **Scoring population:** the Ln(III) rows of the **confirmation half** of `V2__element` — `folds/INDEX.json →
  designs.V2__element__exact.by_half.C` = **11 folds** (the design is seed-independent; B6's conformal inner folds
  are drawn with each withheld seed, so the arm is fitted 11 × 5 = **55** times per transform). Actinide rows are
  training data only and are never scored (`h3_summary.json → readings.ln_test_set`).
- **Cluster unit:** **metal state**, the only registered V2 cluster (§8 table). In discovery this was
  **7** clusters = 7 scored units (`h3_contrasts.csv`, `n_clusters` = `n_units` = 7). **Seven clusters is few**:
  item 2 under BCa at 7 clusters is the fragile part of this claim.
- **Unclustered bootstrap** is printed for reference and never decides (§8), for every claim.

**Tuning at confirmation.** Each withheld seed re-tunes under addendum 1 items 1–2 (one inner fit per
configuration, 3 inner folds, mean unit-macro MAE with the 0.005 tie rule, cross-fitted conformal with the refit of
addendum 2 item 1). M2 keeps each outer fold's retained **M1** hyperparameters (addendum 1 item 6(d)), so **M1 must
be fitted on every confirmation V5 fold as well**, and M2's record verifies against M1's record of the same fold.
C4's ACT_PERMUTED arm is **not** tuned: it is refitted at the WITH run's selected hyperparameters of the same fold
(§11; `h3.frozen_runner`), so C4 needs one tuned B6 WITH fit and one frozen refit per fold.

### 2.3 The reduced sensitivity set (R19 item 6) — as run in discovery

**C1–C3** (V5): run, Δ must be > 0 in each — `strict_setting`, `HNO3_only_cells`, `non_DGA_stratum`,
`acid_grid_rows_excluded`, `censoring_candidates_excluded_scoring`, `wildcard_copies_excluded_scoring`.
Not run, and the report must say so: `V5-P`, `V5-cell-only`, `loose_setting`, `parent_structure_hiding`,
`sr_iii_dropped_training` (addendum 1 item 4). Source: `decisions.json →
S1_components.S1a_M2_vs_B3i.items[5].detail` (identical string for C2 and C3).

The first two need **refits** on the confirmation half (`V5__strict__batched_max4.by_half.C` = **37** folds;
`V5__hno3_only__batched_max4.by_half.C` = **135** folds, `folds/INDEX.json`). The other four are **re-scorings of
the primary fits** — no extra fit. In a paired contrast a row dropped for either arm is dropped for both
(`decisions.json → readings.wildcard_filter_pairing`).

**C4** (V2): run — `acid_grid_rows_excluded`, `censoring_candidates_excluded_scoring` (both **re-scorings**, no
extra fit). Not run, and the report must say so: `sr_iii_dropped_training`, `state_level_hiding`
(`h3_r19_items.csv`, item 6 detail of `B6:WITH vs B6:ACT_PERMUTED@V2`). §11 names **no refit sensitivity** for the
ablation (`h3.REFIT_NOT_RUN`), so C4's item 6 is the two scoring filters only — a materially weaker item 6 than
C1–C3's, and the report must state that.

**Open reading, for the orchestrator to resolve before the run:** addendum 1 item 4 scopes the strict and HNO3-only
refits to "seed 104729" — a discovery statement. At confirmation the seed set is the 5 withheld seeds. This plan
costs them on **all 5 withheld seeds** (the conservative reading, §6); running them on one withheld seed instead
saves ≈ 36 h of serial compute but weakens item 6. **This needs an addendum only if the cheaper reading is chosen.**

---

## 3. Evaluated in the same single run, but **not** frozen claims

Required by §9 and §15 independently of the claim list. They carry no R19 freezing screen — S1(c)'s verdict comes
from its own registered rule, S2's from §9 — so they occupy none of the five slots.

### 3.1 S1(c) — selectivity direction and logSF magnitude (V5-PAIR)

Confirmation rule, verbatim from §9 S1(c) as redefined 2026-09-15:

- **Paired on identical pairs of the half being judged.** For each yardstick Y ∈ {HEAVIER, B3x-derived,
  B3i-derived}: Δ_Y = direction accuracy(M2) − direction accuracy(Y), on the same comparable test–test pairs where Y
  is defined and **|observed logSF| ≥ 0.3**, **cell-pair macro**, same half, **same fitted folds**.
- **Same fitted folds** = B3x and B3i are **re-fitted on exactly the batched V5-PAIR folds M2 is fitted on**
  (closed-form; `gen19ct/models/s1c_yardsticks.py`), built by the same colouring rule **with each withheld seed**.
  HEAVIER needs no fit.
- **Seed combination:** each Δ_Y is the **mean over the 5 withheld seeds** of the per-seed cell-pair-macro Δ_Y; its
  interval is the percentile interval of a **system-cluster bootstrap (10,000 resamples, seed 19)** of that seed
  mean, the **same resampled systems applied to every seed**; and Δ_Y > 0 in **5 of 5** withheld seeds.
- **Pass at confirmation:** **min_Y Δ_Y ≥ γ5 = 0.05** and every Δ_Y's system-cluster 95 % percentile interval
  excludes 0.
- **logSF part:** M2's logSF MAE below **FLAT** and below the **B3i-derived** logSF MAE by **≥ η5 = 0.02** on the
  identical pair set of the confirmation half.
- **Counterweight (already satisfied, selection half, seed 104729):** min_Y Δ_Y = **−0.016846876701018858** ≥ −0.02
  (`decisions.json → S1_components.S1c_selection.min_delta`, `…min_delta_threshold`). It clears the threshold by
  **0.0031531232989811427** (`…min_delta_margin_inside_threshold`) and the interval [−0.02991, −0.00651] is **not**
  separated from the threshold (`…min_delta_interval_separated_from_threshold` = false). S1(c) passes only if both
  halves hold.
- **Scoring population:** `folds/INDEX.json → placeholders.V5PAIR_eligible_cell_pairs.by_half.C` = **163 cell pairs
  in 13 systems**; heavy-arm batching **38 batches** per seed
  (`placeholders.V5PAIR_heavy_arm_batches.C.n_batches`). Cluster unit: **system**.
- **Honest prior from discovery** (`decisions.json → S1_components.S1c_selection.direction`): Δ_HEAVIER −0.016847,
  Δ_B3x +0.063743 (interval excludes 0), **Δ_B3i +0.030013, p = 0.0712, interval includes 0**. The binding yardstick
  at confirmation is **B3i**, not HEAVIER — the selection-half Δ_B3i is already below γ5 = 0.05. logSF MAE gains:
  FLAT +0.482946, **B3i +0.066799 (p = 0.3240, interval [−0.0715, +0.1930] includes 0)**. **S1(c) is the component
  most likely to leave S1 UNDECIDED.**

### 3.2 S1(d) calibration and S1(e) support-distance

- **S1(d)**, macro over the confirmation half's V5-primary cells: 50 % coverage ∈ [0.40, 0.60], 80 % ∈ [0.70, 0.90],
  95 % ∈ [0.88, 0.99], and 80 % ∈ [0.65, 0.92] in every domain-status category with ≥ 20 scored cells (§9 S1(d)).
  Discovery computed a band verdict for **M2 only**: **PASS** (`decisions.json → S1_components.S1d_calibration`).
- **S1(e)**: Spearman ρ(cell MAE, `support_score`) ≤ −0.10 with the system-cluster interval excluding 0, and the
  support-score components must clear the §8 reliability floor of 0.3. They do: `support_score` 0.9564, s1 0.3469,
  s2 0.9780, s3 0.9598, s4 0.8580, s5 0.8877, s6 0.9921, s8 0.9620
  (`tables/reliability_before_correlation.csv`) — but **`support_s7` is −0.1364, UNDECIDED_UNRELIABLE**, and must be
  reported as such beside S1(e).

### 3.3 The single V6 run (§3.4) — S2(a)–(c)

V6 runs **once**, in this run, **with the frozen configurations** (no re-tuning) and the withheld seeds. **13
systems, 209 comparable Pr/Nd pairs, 142 of them HNO3**; the 7-system sensitivity (168 pairs) is scored in the same
run. **580 V6 target rows** (`folds/INDEX.json → population.v6_target_rows`). No V6 fold design exists yet
(`folds/INDEX.json → designs` has none); it is built in this run by the §3.4 component-aware hiding.

- **S2(a) sign + direction.** Sign of the per-system median predicted logSF_Nd/Pr equals the observed sign in **≥ 11
  of 13 systems**; and pair-level direction accuracy (**|observed| ≥ 0.1**) beats every yardstick by **+0.05** under
  the paired rule of S1(c): Y ∈ {HEAVIER, B3x-derived, B3i-derived, **B8**} (B8 only in systems with TOPO39 columns),
  B3x/B3i/B8 **fitted on the same V6 folds and withheld seeds as M2** and the same seed combination;
  **min_Y Δ_Y ≥ 0.05** both pooled and in the HNO3 pairs.
- **S2(b) magnitude.** logSF MAE ≤ FLAT **− 0.02**, with the 13-system percentile bootstrap interval of the gain over
  FLAT excluding 0; also ≤ the lookup-derived logSF MAE; and the macro log D MAE of the hidden Pr and Nd rows ≤ the
  V5 lookup comparator.
- **S2(c) coverage.** Over the 209 V6 pairs, pooled logSF interval coverage ∈ [0.70, 0.90] at 80 % and ≥ 0.88 at 95 %.
- **S2(d)** is **not evaluated**: conditional on Phase H, and §14's process layer runs only in exploratory mode
  because S1 is UNDECIDED (addendum 3 item 3).
- Every V6 number is reported pooled **and per acid medium**; TODGA on its own (83 pairs: HCl 54, HNO3 26,
  malonic 3).
- **S2 without S1 is reported as "Pr/Nd reconstructed in the V6 systems; not established as general transfer"** (§9).

### 3.4 The §11 V6 deltas (H3 at confirmation)

§11's Ln test set includes "V6, **at confirmation only**". The arms are the **deployed configuration M0 (= B5)**
(addendum 3 item 2) and **B6**, transforms WITH / WITHOUT / `ACT_PERMUTED`, at the frozen per-fold hyperparameters.
Deltas WITH − WITHOUT and WITH − PERMUTED with R19 and TOST (ε = 0.05): Δ macro MAE log D, Δ rank accuracy,
Δ logSF MAE, Δ calibration. Four facts carry over from the discovery-side run, all now measured rather than
assumed:

1. **`ACT_METAL_SHUFFLED` is not run at confirmation either** — addendum 4 item 1 is unconditional.
2. **Δ logSF MAE is the one §11 delta V6 can supply.** It is **NOT_DEFINED** in discovery because addendum 1 item 5
   runs the V5-PAIR folds for M2 and the B3x/B3i counterweight only, so neither H3 arm has a pair record
   (`h3_summary.json → not_computed.logsf_mae_delta`). V6 has its own pair endpoint, so this is the only place the
   quantity exists.
3. **The shared-only-embedding re-run is not applicable, at confirmation as in discovery** — `h3_summary.json →
   shared_only.status` = *"not run (not applicable: B5 has no §15 metal embedding)"*. M0 is a descriptor CatBoost:
   it has no `e_series` / `e_ox` offsets to zero. This is a property of the deployed arm, not of the result, and no
   V6 run changes it.
4. **CRPS stays NOT_RUN** for every arm before M7 (addendum 1 item 6(e)); M7 is `not_run`.

---

## 4. What will **not** be claimed

Stated here so the confirmation run cannot quietly grow:

1. **Anything about actinide transfer beyond "ambiguous".** H3 is scored and **UNDECIDED for both arms**
   (`h3_verdicts.json`). On the deployed arm B5, WITH beats WITHOUT on the V5 Ln cells by Δ = **+0.066866** with
   percentile [+0.011249, +0.245690], BCa [+0.009038, +0.225950], p = **0.0064**, and beats ACT_PERMUTED by
   Δ = **+0.078152**, p = **0.0000** — both pass R19 items 2, 3, 5 and 6 and **FAIL item 1**, because δ5 = 0.105689
   is above them. `mde_80` is **0.179162** (WITHOUT) and **0.107724** (PERMUTED), i.e. **above the margin**: at δ5
   this design cannot reach item 1 whatever the truth. **This is UNDECIDED (no registered power check)** — no §8
   injection check has been run for any H3 contrast (`h3_verdicts.json → kappa_status` = "not computed"), and
   addendum 4 item 4 forbids reporting it as a null. **Actinide rows therefore do not enter the deployed Ln
   configuration** (§11: only on *helps*; `h3_verdicts.json →
   actinide_rows_enter_deployed_configuration` = false), and **nothing about H3 on V5 is frozen**.
   The V5 gain is also concentrated rather than broad — over the 59 scored cells the mean Δ is +0.066866 but the
   **median is −0.009994** and only **0.339** of cells have Δ > 0 (`h3_per_unit_deltas.csv`, design V5 / B5 /
   WITHOUT); the two cells whose eligibility depends on actinide partners carry mean Δ **+0.942409**, and dropping
   them leaves 57 cells at mean **+0.036145**, median **−0.011837** (`negative_transfer/strata.csv`, stratum
   `actinide_dependent_eligibility`).
2. **F4 (negative actinide transfer) HOLDS on the conservative reading — gen19 is unsuccessful on F4.** F4 is a
   **registered failure condition**, so the conservative reading is the one that triggers more readily: §10 F4 says
   "95 % interval excluding 0" and names no construction, §8 registers the **percentile AND the BCa** 95 % interval for
   every contrast, and addendum 2 reads F4 as holding when **ANY** registered design shows it. Under either registered
   interval on any registered design, F4 **holds**: the reversed V2 contrast (WITHOUT vs WITH, metal-state cluster) is
   **+0.026416** with BCa **[+0.003149, +0.078275]**, which excludes 0, while its percentile interval
   **[−0.001399, +0.064213]** does not. So `h3_f4.json → failure` = **true** (the conservative headline),
   `failure_bca` = **true**, `failure_percentile` = **false**, and the two readings **disagree on V2**; V1
   (−0.005429) and V5 (−0.066866) show no negative transfer under either interval. This is not softened: the headline
   is that F4 holds, and every quotation carries both readings. Which interval a failure condition reads remains an
   undisclosed reading (`h3_f4.json → interval_reading_not_registered`) and an addendum is requested below; the
   conservative value governs until one is written. The consequence for §11 is unchanged — actinide rows enter the
   deployed configuration only on *helps*, and the verdict is UNDECIDED.
3. **Completeness is per CONTRAST RECORD SET, and the two `ACT_PERMUTED@V1` legs are an `INCOMPLETE_GUARD_FAILURE`
   DEVIATION.** Every registered §11 delta is WITH − WITHOUT or WITH − PERMUTED, so the unit of completeness is the
   arm × transform × design leg (`h3_summary.json → contrast_unit_rule`; a reading of addendum 4 item 2 for which an
   addendum is requested below). **10 of 12** legs are complete and scored, including `B5:WITHOUT@V1` and
   `B6:WITHOUT@V1` (39 of 39 folds each), which now carry their own verdicts: **B5 WITH vs WITHOUT @V1 Δ +0.005429**
   (percentile [−0.064600, +0.067639], p 0.8284) and **B6 Δ −0.026489** (percentile [−0.133449, +0.059233], p 0.6048),
   both **FAIL** on the freezing screen and on the full R19 verdict, and both TOST **UNDECIDED** — so V1
   non-inferiority is **false**, not merely missing, and no *helps* verdict is available.
   `B5:ACT_PERMUTED@V1` and `B6:ACT_PERMUTED@V1` are **25 of 39** folds each and carry
   **`INCOMPLETE_GUARD_FAILURE`**, deliberately **not** addendum 4 item 2's `NOT_RUN` vocabulary, which belongs to the
   20 h cap: the cap was not reached (`budget.exhausted` = false, 14.2158 h left), so this is a **DEVIATION** from
   "That is registered and is run in full" (`contrasts_not_run.*.deviation`; `evaluation/h3/decisions/deviations.json`).
   Those two legs contribute **no contrast, no R19 item row, no delta, no per-unit row, no F4 entry and no
   non-inferiority input**, and V1's `ACT_PERMUTED` non-inferiority stays `None`.
   **The cause is measured, and the fold is not forced** (`scripts/g19_h3_guard_diagnosis.py` →
   `evaluation/h3/decisions/isolation_guard_diagnosis.json`): the §2 inner guard is
   `fold_isolation_check(level="V1", near_dup_sig=6, near_dup_value_tol=0.005)`, and ACT_PERMUTED rebuilds the corpus
   over the permuted frame, so the guard's **value** comparison reads permuted `log_D`. On fold `pub_97510df3a0` all
   three inner splits pass on the registered values and two fail on the permuted ones, each on exactly one
   near-duplicate key: on `inner_s104729_f1` a U(VI) row whose registered `log_D` 0.174458 was permuted to 0.376759,
   landing **0.00201** from a calibration row at 0.374748, where the registered values are **0.115576** apart. The key
   itself cannot move (it is built from system, metal, state, acid, solvent, concentrations and temperature, never from
   `log_D`), so the guard flags a near-duplicate pair the recorded data never had. Running the guard **value-blind**
   would be **stricter, not a fix**: it fails all three splits on the registered frame too (2, 27 and 26 shared keys)
   and would block the WITH and WITHOUT arms that scored.
4. **Everything the ladder dropped.** M1 and M2 are **not** the retained configuration — `ladder.json →
   retained_final` is **M0**. M1 vs M0 @V5 (Δ **−0.073667**) and M2 vs M0 @V5 (Δ **−0.154150**) are **FAIL**
   (`contrasts_registered.csv` rows 16/20). **M3–M7 are `not_run`** (addendum 3 item 3), so H5, the M7 deep
   ensemble and with it H6's ensemble half are not claimed. Nothing about the ladder is confirmed. **This limits
   what a confirmed claim MEANS; it is not a bar on freezing it** — see §2's paragraph on candidates.
5. **H1b (B6 vs B3i @V5).** Δ **−0.104441**, FAIL. Power check **UNDECIDED_UNDERPOWERED**, κ_min **None**
   (`tables/power_kappa.csv`; `evaluation/power/power_checks.json → checks[1]`): at κ = 1.0 items 1, 5, 6 PASS while
   the cluster bootstrap still cannot exclude 0. **Reported UNDECIDED (underpowered), never a null**; not frozen.
6. **H4.** M2 vs FLAT_CAT (Δ **+0.102763**, FAIL — below δ5 by 0.0029) and B6 vs B6r0 (Δ **+0.087714**, FAIL). No H4
   power check was run: §8 names only H1/H1b/H3 and addendum 2's `needs_power` names primary/H1b/S1(b)/H3. Both
   print **UNDECIDED (no registered power check)** (addendum 4 item 4).
7. **The process case (§14).** Exploratory only, every output labelled **transfer-unsupported**, because S1 is
   UNDECIDED in discovery by construction (addendum 3 item 3). No process recommendation is a claim; S2(d) is not
   evaluated.
8. **The secondary designs for M2.** M2 vs B3 @V1 (Δ **+0.095791**, FAIL) and M2 vs B3i @V2 focus-7
   (Δ **+0.015096**, p = 0.8834, FAIL) are not frozen. **C1–C3 are V5-only**; no confirmation statement is made
   about unseen publications, and the only unseen-metal claim is C4, which is about B6's control contrast, not
   about M2's accuracy.
9. **Anything measured on a derived quantity below the reliability floor.** `B7 n` (−1.5210), `support_s7` (−0.1364)
   and the per-system `|logSF| amplitude` (−2.0157) are UNDECIDED_UNRELIABLE. The amplitude verdict is additionally
   an **aggregator artifact** (mean of 20 Spearman–Brown values with a pole at r → −1; half 7 gives SB = −41.94
   against a mean half-correlation of **+0.2544**), and §8 registers the 20 seeded halves but **not** the
   aggregator — **no claim may be closed on that value either way.** `factor_loadings` and `embeddings` are
   NOT_IMPLEMENTED, so figures F10/F11 and their correlations stay UNDECIDED.
10. **The H3 negative-transfer correlations.** Spearman(per-cell Δ, actinide row count) = **+0.429669**, percentile
    [−0.122355, +0.765292]; Spearman(per-cell Δ, median |An/Ln logSF|) = **−0.094540**, [−0.844506, +0.822267]
    (`evaluation/h3/negative_transfer/per_cell_covariates.csv`). Both include 0 and are descriptive; neither may
    support or close a claim.

---

## 5. Order of work inside the single run

1. Reveal the withheld seeds: `scripts/g19_seal_prereg.py --verify-seeds` against
   `manifests/confirmation_seeds_sha256.txt`; write `decisions/CONFIRMATION.md` with the verdict (§15).
2. Build the withheld-seed folds: V5-primary `batched_max4`, V5-strict, V5-HNO3, V5-PAIR `batched` colourings and
   the V6 hiding — same rules, new seeds. Register the fold hashes before any fit.
3. Fit **M1 then M2** on the ≈ 135 confirmation V5-primary folds (M2 needs M1's verified record per fold).
4. Score C1, C2, C3; comparators B3i, B0 (closed-form, conformal folds drawn with each withheld seed) and B6r0
   (from B6 on `V5__primary__exact`, confirmation half, 5 seeds).
5. R19 item 6: the strict and HNO3-only refits, then the four scoring-filter re-scorings.
6. **C4** (only if the addendum of §2 registers the H3 V2 margin): B6 WITH tuned and ACT_PERMUTED frozen on the 11
   confirmation `V2__element_exact` folds × 5 withheld seeds; then the two scoring-filter re-scorings.
7. S1(c) on the V5-PAIR confirmation folds, with the B3x/B3i yardsticks refitted on those same folds.
8. S1(d), S1(e).
9. **V6, once**: S2(a)–(c) and the §11 V6 deltas on M0 (= B5) and B6.
10. Report. **Whatever it returns is the result. No sixth claim, nothing re-run** (§15).

---

## 6. Compute estimate

Unit costs are **measured** per-fold means (point + intervals) from the record `steps` of
`evaluation/discovery/<arm>/<design>/s104729/*.json` and, for the frozen H3 legs, of
`evaluation/h3/records/<arm>/<transform>/<design>/s104729/*.json`. Fold counts are the **confirmation half's own**,
from `folds/INDEX.json → designs.*.by_half.C` and `placeholders`. **Serial hours** are unit cost × folds;
**wall clock at 2 workers** divides by the measured parallel efficiency of this machine, **1.96×** (discovery:
149.91 h serial over 76.5955 h of wall clock, `evaluation/discovery/decisions/wall_clock.json → budget.used_hours`).
The frozen-configuration factor **0.2785** is measured: B5's H3 refit against its tuned discovery fit on the same
V5 folds (`evaluation/h3/decisions/cost_estimate.json → h3_over_discovery_ratio`).

| step | unit s/fold | folds (confirmation half) | serial h | wall h @ 2 workers |
|---|---|---|---|---|
| M1, V5-primary `batched_max4` | 684.8 | 135 | 25.7 | 13.1 |
| M2, V5-primary `batched_max4` | 303.6 | 135 | 11.4 | 5.8 |
| B6 → B6r0, `V5__primary__exact` × 5 seeds | 9.6 | 555 | 1.5 | 0.8 |
| B3i, B0 (closed-form + conformal) | ≈ 0 | — | < 0.5 | < 0.3 |
| **C1 + C2 + C3 core (shared)** | | | **≈ 38.6** | **≈ 19.7** |
| item 6: strict refits (M1 568.1 + M2 279.8) | 847.9 | 37 | 8.7 | 4.4 |
| item 6: HNO3-only refits (M1 680.2 + M2 283.6) | 963.8 | 135 | 36.1 | 18.4 |
| **R19 item 6 (shared by C1–C3)** | | | **≈ 44.9** | **≈ 22.9** |
| **C4**: B6 WITH tuned (9.0) + ACT_PERMUTED frozen (2.6) | 11.6 | 55 | **0.2** | **0.1** |
| S1(c): M1 686.4 + M2 286.9 on V5-PAIR | 973.3 | 38/seed × 5 = 190 | 51.4 | 26.2 |
| S1(c): B3x/B3i yardstick refits | closed-form | 190 | < 0.2 | < 0.1 |
| V6: M1 + M2 at the frozen configuration (0.2785 × 988.4) | 275.3 | 13 × 5 = 65 | 5.0 | 2.5 |
| V6: B8 yardstick at the frozen configuration (0.2785 × 699.6) | 194.8 | 65 | 3.5 | 1.8 |
| V6 §11: M0 = B5 WITH (574.3) + WITHOUT (495.8) + PERMUTED (579.1) | 1,649.2 | 65 | 29.8 | 15.2 |
| V6 §11: B6 reference, three transforms | ≈ 30 | 65 | 0.5 | 0.3 |
| **V6 run (S2 + §11 deltas)** | | | **≈ 38.8** | **≈ 19.8** |
| **TOTAL** | | | **≈ 174 h** | **≈ 89 h ≈ 3.7 days** |

**Per claim, honestly attributed.** C1–C3 share one fit set: **C1 alone is 38.6 + 44.9 = 83.5 h serial (42.6 h
wall)**, C2 and C3 add **1.5 h serial between them** (B6 for B6r0; B0/B3i are free), and **C4 adds 0.2 h serial**
(6–11 minutes) — it is by far the cheapest claim in the plan. The remaining **≈ 90 h serial (≈ 46 h wall)** is
S1(c) + V6, which §9 and §15 require whether or not any claim confirms.

**Not in the table, and not planned here: an H3 power check.** §8's injection check refits every endpoint of a
contrast at 4 values of κ. For the deployed arm's two V5 contrasts that is 3 arms (WITH, WITHOUT, ACT_PERMUTED) ×
4 κ × 28 folds = **336 refits** at the measured ≈ 535 s ≈ **50 h serial (25 h wall)** — more than C1's whole
confirmation core. **Every H3 contrast owes the check and none has one**, and the debt is now inventoried with its
cost per contrast (`h3_summary.json → power_debt`, D03's table): **20 of 20** contrasts owed, 0 run. The inventory's
totals are an upper bound because they price each contrast separately and so count a shared WITH leg twice
(`h3.POWER_COST_BASIS`); the 336-refit figure above is the de-duplicated one for the V5 pair. Addendum 4 item 4
licenses reporting those contrasts as **UNDECIDED (no registered power check)** without the check, which is what D03
does. **Running it is a separate authorisation.**

**The estimate is INFERRED** from measured unit costs on this machine (12 threads, 8 GB, ≤ 2 workers), and discovery
itself overran its 60 h budget by 28 %. **There is no registered budget for the confirmation run** — §7 item 5's
60 h is discovery's, addendum 2's 40 h is the ladder's (unconsumed, since M3–M7 are `not_run`) and addendum 4's 20 h
is H3's (5.78 h used). Treat ±30 % as the honest band: **122–226 h serial, 62–115 h of wall clock**.

---

## 7. The go/no-go question for the orchestrator

**Authorise the single confirmation run — ≈ 174 h serial, ≈ 89 h of wall clock at 2 workers (3.5–5 days), on the 5
withheld seeds and the confirmation half, with V6 once — to decide R19 item 4 for four claims, plus S1(c), S1(d),
S1(e) and S2?**

What a yes buys: the only thing discovery could not produce — **seed consistency (item 4, 5 of 5)** — on an
unbiased half, plus the one-and-only V6 / Pr–Nd answer that brief §34 asks for, plus the only V6 Δ logSF MAE
that §11 will ever have.
What a yes cannot buy: anything about the deployed model's accuracy. **The deployed configuration is M0**, and
C1–C3 are about **M2**, which the ladder dropped (M2 vs M0 Δ = −0.154150, FAIL). Confirming C1 would establish "a
factorised model beats the lookup on unseen cells", not "the model we would deploy does".

Decisions needed before launch, all flagged above:
1. **§2 / §5 step 6 — C4.** Freeze the fourth claim? It costs 0.2 h and is the cleanest form of brief §30's
   question. Its item-1 margin on V2 needs **no** addendum: POST-HOC addendum 2 registers it ("δ5 on V5 Ln cells,
   0.05 on V1 / V2", and the discovery reading "0.05 for S1(b) and for V1 / V2"), and R19 item 1 compares the **point
   estimate** with that margin, which C4 meets (+0.081519 ≥ 0.050) — `h3.READINGS['margins']`, task X finding
   numbers VH-01. C4's item 6 remains materially weaker than C1–C3's (§2.3).
2. **§2.3** — do the strict/HNO3 item-6 refits run on all 5 withheld seeds (costed: 44.9 h serial) or on one
   (≈ 9 h, needs an addendum)?
3. **§3.1** — accept that S1(c)'s binding yardstick is B3i, whose selection-half Δ (+0.030013) is already below
   γ5 = 0.05, so **S1 is more likely than not to return UNDECIDED even if C1–C3 confirm**.
4. **§3.4 / §6** — the V6 §11 deltas are the single largest V6 line (29.8 h serial). They are registered ("V6, at
   confirmation only") and are the only source of Δ logSF MAE for H3; the discovery-side V5 result they extend is
   UNDECIDED with `mde_80` above δ5. Run them, or record them NOT_RUN with the reason?
5. **§6** — authorise the ≈ 50 h serial H3 power check, or accept **UNDECIDED (no registered power check)** as the
   final H3 wording (addendum 4 item 4 permits it)?
6. **§4 item 3** — the `pub_97510df3a0` isolation-check failure is now diagnosed, not unknown
   (`evaluation/h3/decisions/isolation_guard_diagnosis.json`): the §2 near-duplicate **value** comparison
   (`near_dup_value_tol` = 0.005) reads the ACT_PERMUTED arm's permuted `log_D` and flags a pair the recorded data never
   had. Register the guard's reading for a value-permuted control arm — **proposed sentence:** *"For a control arm whose
   training `log_D` is permuted by construction (`ACT_PERMUTED`, §11 Controls), the near-duplicate VALUE comparison of
   the §2 fold-isolation guard reads the REGISTERED `log_D` of the corpus, not the permuted values: `near_dup_value_tol`
   = 0.005 asks whether two rows record the same experiment at the same value, which is a property of the recorded data
   and not of an arm's training target, and a permuted value that happens to land within 0.005 of another row's value is
   a coincidence the data never had. Every other level of the guard — the publication group, the archive duplicate group
   and the near-duplicate key at 6 significant figures — is value-independent and is unchanged, so the guard stays
   exactly as strict as registered and becomes invariant under the permutation."* — and then complete the two
   `ACT_PERMUTED@V1` legs (14 folds × 2 arms; ≈ 0.9 h serial for B5 at the measured 230.6 s/fold, minutes for B6)
   inside the remaining 14.2158 h of the cap; or leave them `INCOMPLETE_GUARD_FAILURE`, which is what this run does.
   **Do not run the guard value-blind** (`near_dup_value_tol = None`): it is stricter, not looser, and fails all three
   inner splits of that fold on the registered values too.

*Nothing in this plan has been executed. No withheld seed has been read, no confirmation-half row scored, no V6 row
touched. `preregistration.md` is untouched.*
