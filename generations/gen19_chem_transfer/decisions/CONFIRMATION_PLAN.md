# CONFIRMATION_PLAN — the claims frozen before the confirmation run

*Pre-registration §15 ("at most **5 claims** are frozen in `decisions/CONFIRMATION_PLAN.md`"), under POST-HOC
addendum 3 item 1 (eligibility) and item 2 (the deployed configuration is **M0 = B5**). Written 2026-09-22 from files
only, after the discovery run completed and was scored on the **selection half, seed 104729** — a half the
pre-registration itself calls optimistically biased. Nothing in this file was computed on a withheld seed, on the
confirmation half, or on V6: no such number exists anywhere in the repository. Paths are relative to
`generations/gen19_chem_transfer/`.*

**Status: NOT YET AUTHORISED TO RUN.** This file freezes *what* would be scored. The run itself needs the
orchestrator's go/no-go (see §7) because the compute estimate is ~180 h of wall clock at 2 workers.

---

## 1. Gate state this plan rests on

| fact | value | source |
|---|---|---|
| prereg footer digest | `135842499a86…5641` | `manifests/prereg_sha256.txt` |
| below-footer addenda | 3, sha256 `646201ec3c0b…` | `manifests/digest_registry.json` |
| stop rule | **false** (M2 vs B3i passes items 1,2,3,5; B6 vs B3i fails) | `evaluation/discovery/decisions/stop_rule.json → stop` |
| ladder | M1 kept **false**, M2 kept **false**, retained **M0** | `evaluation/ladder/decisions/ladder.json → retained_final` |
| deployed configuration | **M0 (= B5)** | addendum 3 item 2; `evaluation/h3/…/deployed.arm` |
| ladder budget | 0.0 h of 40.0 h used | `evaluation/ladder/decisions/wall_clock.json → budget` |
| eligible for freezing | exactly **3** contrasts | `evaluation/discovery/decisions/decisions.json → freezing_candidates[*].eligible_for_freezing` |
| withheld-seed commitment | `65e8ae8ceb8e…f82` | `manifests/confirmation_seeds_sha256.txt` (§15) |

The scorer's eligibility rule is addendum 3 item 1 verbatim: *every R19 item that R19 evaluates in discovery is PASS
and no item is FAIL*; item 4 is NOT_EVALUATED by addendum 1 item 3 and does not disqualify. Three contrasts meet it.
Nothing else in §19 does — every other registered contrast either FAILs an item or was never run.

---

## 2. The frozen claims (3 of the at most 5)

All three are on **V5-primary**, candidate **M2** (the factorised arm, §6), metric **unit-macro MAE of log D over the
105 held-out metal × extractant cells**, Δ = MAE(comparator) − MAE(M2). All three are scored from the **same** set of
confirmation-half fits, so claims 2 and 3 cost nothing beyond claim 1 except their comparators.

| # | claim | family | comparator | margin | design(s) | discovery Δ (selection half) | discovery verdict |
|---|---|---|---|---|---|---|---|
| **C1** | M2 vs B3i @ V5 | primary (H1, S1(a)) | **B3i** — the stronger of B3x/B3i, §5/§9 | **δ5 = 0.10568948790529951** | V5-primary, M2 on `batched_max4`, B3i exact | **+0.262976** | UNDECIDED (item 4 only); freezing screen PASS |
| **C2** | M2 vs B0 @ V5 | S1(b) | **B0** (global mean) | **0.05** | V5-primary, same | **+1.022150** | UNDECIDED (item 4 only); freezing screen PASS |
| **C3** | M2 vs B6r0 @ V5 | S1(b) | **B6r0** (rank-0 B6) | **0.05** | V5-primary, M2 `batched_max4`, B6r0 exact | **+0.455132** | UNDECIDED (item 4 only); freezing screen PASS |

**None of the three is a null, and the power artefacts now say so** (task X findings V-L3 / V-P04). The §8 signal-injection check was re-run on 2026-09-22: all three carry κ_min **0.1** and verdict **POWERED_NOT_A_NULL** — previously `INFORMATIVE_NULL` with the sentence "the null is informative", although each un-injected contrast PASSES its freezing screen. Addendum 2's `needs_power` widened the TRIGGER, not the verdict vocabulary; §8 scopes the check to a contrast *reported as a null*. Every reported Δ and every R19 item behind κ_min is now re-derivable without a refit from `tables/power_per_unit_mae.csv` (per (contrast, κ) the per-cell MAE of both arms and the bootstrap's cluster labels — metrics, not injected values, so brief §33 is unchanged; finding V-L2).

Sources: `evaluation/discovery/contrasts_registered.csv` rows 0/4/6 (`point`, `margin`, `verdict_freezing_screen`);
`decisions.json → S1_components.{S1a_M2_vs_B3i, S1b_M2_vs_B0, S1b_M2_vs_B6r0}.point`.

**Slots 4 and 5 are deliberately left empty.** §15 says *at most* five. No fourth registered contrast of §19 is
eligible, and nothing may be added later (§15: "No sixth claim is added and nothing is re-run").

### 2.1 R19 items evaluated at confirmation — identical for C1, C2, C3

Applied to the confirmation half with the 5 withheld seeds (§8 R19; §15):

1. **Item 1** — point Δ ≥ the claim's margin above.
2. **Item 2** — percentile **and** BCa 95 % intervals exclude 0 under **every** registered cluster unit.
3. **Item 3** — two-sided bootstrap p < 0.05 under **every** registered cluster unit (the conservative reading,
   resolved 2026-09-15).
4. **Item 4 — the item discovery could not evaluate.** Δ > 0 in **5 of 5 withheld seeds** on the confirmation half,
   exactly as registered (§8 item 4; addendum 3 item 1 restates that it is required here). This is the whole point of
   the run: in discovery it is NOT_EVALUATED because addendum 1 item 3 ran learned arms on seed 104729 alone
   (`decisions.json → S1_components.S1a_M2_vs_B3i.items[3].detail`).
5. **Item 5** — leave-one-cluster-out: removing any single system or publication group does not make Δ ≤ 0.
6. **Item 6** — Δ > 0 in every sensitivity of the **reduced set (addendum 1 item 4)**, listed in §2.3 below.

Bootstrap: paired cluster bootstrap, **10,000 resamples, `numpy.random.default_rng(19)`** (§8). BH-adjusted p is
printed per family beside raw p and **decides nothing** (§8 multiplicity).

### 2.2 Cluster units and exact scoring population

- **Scoring population:** the **confirmation half** of `V5__primary` only (`feasibility_halves.csv`; the half that
  contributed to no ladder decision, claim or preferred-model choice, §15). **105 scored cells.**
- **Primary cluster:** `extractant_system_key` — **19 systems** (§8 table: 37 scored systems, 18 selection / 19
  confirmation; confirmed by `tables/preseal_contrasts.csv`, half `confirmation`, design V5, `n_clusters` = 19).
- **Secondary cluster:** publication group of each cell's majority rows — **20 groups**
  (`tables/preseal_contrasts.csv`, same rows, `cluster_unit` = `publication_group`, `n_clusters` = 20). Compare
  the selection half's 18 / 27: **the confirmation half has fewer publication-group clusters, so intervals will be
  wider than discovery's at the same effect size.**
- The unclustered per-unit bootstrap is printed for reference and never decides (§8).
- **Folds:** `folds/INDEX.json → designs.V5__primary__batched_max4.by_half.C` = 135 folds / 105 units / 2,722 rows
  across the 5 discovery seeds, i.e. **27 batches per seed**. The withheld seeds' colourings **do not exist yet** and
  must be built first by `scripts/g19_build_folds_max4.py` with the same vertex-order rule (§7 item 6); expect
  27 ± 1 per seed, ≈ **135 folds** total.
- **Tuning at confirmation:** each withheld seed re-tunes under addendum 1 items 1–2 (one inner fit per configuration,
  3 inner folds, mean unit-macro MAE with the 0.005 tie rule, cross-fitted conformal with the refit of addendum 2
  item 1). M2 keeps each outer fold's retained **M1** hyperparameters (addendum 1 item 6(d)), so **M1 must be fitted
  on every confirmation fold as well**, and M2's record verifies against M1's record of the same fold
  (`decisions.json → readings.record_verification`).

### 2.3 The reduced sensitivity set (R19 item 6) — as run in discovery

Run (Δ must be > 0 in each): `strict_setting`, `HNO3_only_cells`, `non_DGA_stratum`, `acid_grid_rows_excluded`,
`censoring_candidates_excluded_scoring`, `wildcard_copies_excluded_scoring`.
Not run, and the report must say so: `V5-P`, `V5-cell-only`, `loose_setting`, `parent_structure_hiding`,
`sr_iii_dropped_training` (addendum 1 item 4).
Source: `decisions.json → S1_components.S1a_M2_vs_B3i.items[5].detail` (identical string for C2 and C3).

The first two need **refits** on the confirmation half (`V5__strict__batched_max4.by_half.C` = 37 folds / 24 cells;
`V5__hno3_only__batched_max4.by_half.C` = 135 folds / 105 cells, `folds/INDEX.json`). The other four are
**re-scorings of the primary fits** — no extra fit. In a paired contrast a row dropped for either arm is dropped for
both (`decisions.json → readings.wildcard_filter_pairing`).

**Open reading, for the orchestrator to resolve before the run:** addendum 1 item 4 scopes the strict and HNO3-only
refits to "seed 104729" — a discovery statement. At confirmation the seed set is the 5 withheld seeds. This plan
costs them on **all 5 withheld seeds** (the conservative reading, §6); running them on one withheld seed instead
saves ~36 h but weakens item 6. **This needs an addendum only if the cheaper reading is chosen.**

---

## 3. Evaluated in the same single run, but **not** frozen claims

These are required by §9 and §15 independently of the claim list. They carry no R19 freezing screen — S1(c)'s
verdict comes from its own registered rule, S2's from §9 — so they occupy none of the five slots.

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
  in 13 systems**; the heavy-arm batching is **38 batches** per seed
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
  support-score components must clear the §8 reliability floor of 0.3. They do:
  `support_score` 0.9564, s1 0.3469, s2 0.9780, s3 0.9598, s4 0.8580, s5 0.8877, s6 0.9921, s8 0.9620
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
  B3x/B3i/B8 **fitted on the same V6 folds and withheld seeds as M2**, same seed combination; **min_Y Δ_Y ≥ 0.05**
  both pooled and in the HNO3 pairs.
- **S2(b) magnitude.** logSF MAE ≤ FLAT **− 0.02**, with the 13-system percentile bootstrap interval of the gain over
  FLAT excluding 0; also ≤ the lookup-derived logSF MAE; and the macro log D MAE of the hidden Pr and Nd rows ≤ the
  V5 lookup comparator.
- **S2(c) coverage.** Over the 209 V6 pairs, pooled logSF interval coverage ∈ [0.70, 0.90] at 80 % and ≥ 0.88 at 95 %.
- **S2(d)** is **not evaluated**: it is conditional on Phase H, and §14's process layer runs only in exploratory mode
  because S1 is UNDECIDED (addendum 3 item 3).
- Every V6 number is reported pooled **and per acid medium**; TODGA is reported on its own (83 pairs: HCl 54,
  HNO3 26, malonic 3).
- **S2 without S1 is reported as "Pr/Nd reconstructed in the V6 systems; not established as general transfer"** (§9).

### 3.4 The §11 V6 deltas (H3 at confirmation)

§11's Ln test set includes "V6, **at confirmation only**". The arms are the **deployed configuration = M0 (= B5)**
(addendum 3 item 2), WITH / WITHOUT / `ACT_PERMUTED`, at M0's frozen per-fold hyperparameters. Deltas reported
WITH − WITHOUT and WITH − PERMUTED with R19 statistics and TOST (ε = 0.05): Δ macro MAE log D, Δ rank accuracy,
Δ logSF MAE, Δ calibration. **Two structural limits carry over from the discovery-side H3 run and must be stated,
not forced:** Δ logSF MAE is unreachable because `h3.logsf_delta` (`gen19ct/evaluation/h3.py:719`) is **never
called** anywhere; and the shared-only-embedding re-run is *not applicable* — `h3.shared_only_condition` returns
"B5 has no section 15 metal embedding". κ_min for any UNDECIDED H3 verdict comes from `scripts/g19_run_power.py`.

**Blocking defect to fix before this runs:** `h3.py:1665` and `:1686` look verdicts up by `dep["arm"]` ("M0") while
`verdicts` are keyed by the alias ("B5"), so D03 prints `Deployed arm reading: not computed` and forces its Decision
line to "no" **even when the verdict is *helps***. Fix to `dep.get("arm_alias") or dep.get("arm")`, in a window when
no runner holds H3 or power records, and log it in `manifests/digest_registry.json`.

---

## 4. What will **not** be claimed

Stated here so the confirmation run cannot quietly grow:

1. **Everything the ladder dropped.** M1 and M2 are **not** the retained configuration — `ladder.json →
   retained_final` is **M0**. M1 vs M0 @V5 (Δ **−0.073667**) and M2 vs M0 @V5 (Δ **−0.154150**) are **FAIL**
   (`contrasts_registered.csv` rows 16/20). **M3–M7 are `not_run`** (addendum 3 item 3), so H5, the M7 deep
   ensemble, and with it H6's ensemble half, are not claimed. Nothing about the ladder is confirmed.
   **This is a limit on what a confirmed claim MEANS, not a bar on freezing it** (task X finding V-P08): a frozen
   claim's candidate need not be the retained ladder configuration. §19 restricts the registered family, and §15
   freezes contrasts that pass the R19 screen — neither requires the candidate to be what is deployed, and addendum 3
   item 2 says so outright: "The factorised M2 is still the H1 candidate and is reported as such: H1 is about M2
   against the lookup, and section 31's architecture question is answered by the M2 − M0 contrast, not by what is
   deployed." Read otherwise, **zero** claims could be frozen — all 17 no-FAIL contrasts in `r19_items.csv` are either
   `family = exploratory` (14, excluded from freezing by §19/§15) or one of the three M2 candidates below — and the
   confirmation run, with it the single V6 run of §3.4, S2 and brief §34's applied Pr/Nd question, could never happen,
   which is exactly what addendum 3 item 1 was written to prevent. **No addendum 4 is needed for this**, and no
   condition of this plan was withdrawn: the plan already freezes the three M2 claims and §7 states what a
   confirmation of C1 would and would not establish.
2. **H1b (B6 vs B3i @V5).** Δ **−0.104441**, FAIL. Its power check is **UNDECIDED_UNDERPOWERED** — κ_min is **None**
   (`tables/power_kappa.csv`; `evaluation/power/power_checks.json → checks[].kappa_min`), and at κ = 1.0 items 1, 5, 6
   PASS while the cluster bootstrap still cannot exclude 0. **It must be reported UNDECIDED (underpowered), never as
   a null**, and it is not frozen.
3. **H4.** M2 vs FLAT_CAT (Δ **+0.102763**, FAIL — below δ5 by 0.0029) and B6 vs B6r0 (Δ **+0.087714**, FAIL). No H4
   power check was run: §8 names only H1/H1b/H3 and addendum 2's `needs_power` names primary/H1b/S1(b)/H3.
   **FIXED (task X finding V-P01):** `report.verdict_of` now prints a FAIL as a null **only** when that contrast's own
   §8 power record screened it as an `INFORMATIVE_NULL`; with an `UNDECIDED_UNDERPOWERED` record it prints
   **UNDECIDED (underpowered)** with κ_min, and with no registered power check at all it prints **UNDECIDED (no
   registered power check)**. So H1b prints UNDECIDED and these two H4 contrasts print UNDECIDED (no registered power
   check); no addendum 4 is needed, because nothing is now reported as a null that §8 does not license.
4. **The process case (§14).** Exploratory only, every output labelled **transfer-unsupported**, because S1 is
   UNDECIDED in discovery by construction (addendum 3 item 3). No process recommendation is a claim; S2(d) is not
   evaluated.
5. **The secondary designs.** M2 vs B3 @V1 (Δ **+0.095791**, FAIL) and M2 vs B3i @V2 focus-7 (Δ **+0.015096**,
   p = 0.8834, FAIL) are not frozen. **The three frozen claims are V5-only**; no confirmation statement is made about
   unseen publications or unseen metals.
6. **Anything measured on a derived quantity below the reliability floor.** `B7 n` (−1.5210), `support_s7` (−0.1364)
   and the per-system `|logSF| amplitude` (−2.0157) are UNDECIDED_UNRELIABLE. The amplitude verdict is additionally
   an **aggregator artifact** (mean of 20 Spearman–Brown values with a pole at r → −1; half 7 gives SB = −41.94
   against a mean half-correlation of **+0.2544**), and §8 registers the 20 seeded halves but **not** the aggregator
   — **no claim may be closed on that value either way.** `factor_loadings` and `embeddings` are NOT_IMPLEMENTED
   (README:457), so figures F10/F11 and their correlations stay UNDECIDED.

---

## 5. Order of work inside the single run

1. Reveal the withheld seeds: `scripts/g19_seal_prereg.py --verify-seeds` against
   `manifests/confirmation_seeds_sha256.txt`; write `decisions/CONFIRMATION.md` with the verdict (§15).
2. Build the withheld-seed folds: V5-primary `batched_max4` colourings, V5-PAIR `batched` colourings, V6 hiding —
   same rules, new seeds. Register the fold hashes before any fit.
3. Fit **M1 then M2** on the 135 confirmation V5-primary folds (M2 needs M1's verified record per fold).
4. Score C1, C2, C3; comparators B3i, B0 (closed-form, conformal folds drawn with each withheld seed) and B6r0
   (from B6 on `V5__primary__exact`, confirmation half, 5 seeds).
5. R19 item 6: the strict and HNO3-only refits, then the four scoring-filter re-scorings.
6. S1(c) on the V5-PAIR confirmation folds, with the B3x/B3i yardsticks refitted on those same folds.
7. S1(d), S1(e).
8. **V6, once**: S2(a)–(c) and the §11 V6 deltas on M0.
9. Report. **Whatever it returns is the result. No sixth claim, nothing re-run** (§15).

---

## 6. Compute estimate

Unit costs are **measured** discovery per-fold means (point + intervals, 2 workers), computed from the record
`steps` of `evaluation/discovery/<arm>/<design>/s104729/*.json`. Fold counts are the **confirmation half's own**,
from `folds/INDEX.json → designs.*.by_half.C` and `placeholders`. The frozen-config factor **0.111** is measured:
B5's H3 point step 184.435 s against its tuned discovery mean 1,658.9 s.

| step | unit cost (s/fold) | folds (confirmation half) | wall clock @ 2 workers |
|---|---|---|---|
| M1, V5-primary | 684.8 | 27/seed × 5 = **135** | **25.7 h** |
| M2, V5-primary | 303.6 | 135 | **11.4 h** |
| B6 → B6r0, V5 exact | 9.6 | 111 × 5 = 555 | 1.5 h |
| B3i, B0 (closed-form + conformal) | ≈ 0 | — | < 0.5 h |
| **C1 + C2 + C3 core (shared)** | | | **≈ 38.6 h** |
| item 6: strict refits (M1+M2) | 847.9 | **37** | 8.7 h |
| item 6: HNO3-only refits (M1+M2) | 963.8 | **135** | 36.1 h |
| **R19 item 6 (shared by all three claims)** | | | **≈ 44.9 h** |
| S1(c): M1+M2 on V5-PAIR | 973.3 | 38/seed × 5 = **190** | **51.4 h** |
| S1(c): B3x/B3i yardstick refits | closed-form | 190 | < 0.2 h |
| V6: M1+M2 at frozen config | 241.5 | 13 × 5 = **65** | 4.4 h |
| V6: B8 yardstick at frozen config | 522.0 | 65 | 9.4 h |
| V6 §11: M0 = B5, WITH/WITHOUT/PERMUTED | 587.9 × 3 | 65 | **31.9 h** |
| **V6 run (S2 + §11 deltas)** | | | **≈ 45.7 h** |
| **TOTAL** | | | **≈ 180 h ≈ 7.5 days** |

Per claim, honestly attributed: the **three claims share one fit set**, so C1 alone is ≈ 38.6 h core + 44.9 h
sensitivities = **83.5 h**, and C2 and C3 add **≈ 1.5 h** between them (B6 for B6r0; B0/B3i are free). The other
**96.9 h** is S1(c) + V6, which §9 and §15 require whether or not any claim confirms.

**The estimate is INFERRED** from discovery unit costs on this machine (12 threads, 8 GB, ≤ 2 workers per fitting
job), and discovery itself overran its 60 h budget by 28 % (76.5955 h, `evaluation/discovery/decisions/wall_clock.json
→ budget.used_hours`). **There is no registered budget for the confirmation run** — §7 item 5's 60 h is discovery's
and addendum 2's 40 h is the ladder's, still unconsumed. Treat ±30 % as the honest band: **125–235 h**.

---

## 7. The go/no-go question for the orchestrator

**Authorise the single confirmation run — ~180 h (5–10 days) at 2 workers, on the 5 withheld seeds and the
confirmation half, with V6 once — to decide R19 item 4 for three M2-vs-comparator claims on V5, plus S1(c), S1(d),
S1(e) and S2?**

What a yes buys: the only thing discovery could not produce — **seed consistency (item 4, 5 of 5)** — on an
unbiased half, plus the one-and-only V6 / Pr–Nd answer that brief §34 asks for.
What a yes cannot buy: anything about the deployed model. **The deployed configuration is M0**, and every frozen
claim is about **M2**, which the ladder dropped (M2 vs M0 Δ = −0.154150, FAIL). Confirming C1 would establish
"a factorised model beats the lookup on unseen cells", not "the model we would deploy does".

Four decisions are needed before launch, all flagged above:
1. **§2.3** — do the strict/HNO3 item-6 refits run on all 5 withheld seeds (costed: 44.9 h) or one (≈ 9 h, needs an
   addendum)?
2. ~~**§3.4 / §4.3** — fix `h3.py:1665`/`:1686` (alias lookup) and relabel `report.py:1863` (FAIL → "null").~~
   **DONE, and neither needed an addendum 4** (task X findings V-P02, V-P01): the verdict lookups take `arm_alias`
   (= B5) while every printed sentence keeps M0, and a FAIL prints as a null only where its own power record says
   `INFORMATIVE_NULL`. The `h3`, `power`, `figures` and `report` stages were re-registered after the edits
   (`manifests/digest_registry.json`), and no record was invalidated.
3. **§3.1** — accept that S1(c)'s binding yardstick is B3i, whose selection-half Δ (+0.030013) is already below
   γ5 = 0.05, so **S1 is more likely than not to return UNDECIDED even if C1–C3 confirm**.
4. Whether the V6 §11 deltas (31.9 h, the single largest line) run at all, given that H3's discovery-side run is
   itself incomplete (1 of 705 folds at last report) and Δ logSF MAE is structurally unreachable.

*Nothing in this plan has been executed. No withheld seed has been read, no confirmation-half row scored, no V6 row
touched. `preregistration.md` is untouched.*
