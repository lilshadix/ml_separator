# Adversarial figure audit

Reviewer #2's checklist, applied to every main and supplementary figure. Each figure gets a
verdict per question and an overall verdict. **Every FAIL found was fixed from existing
data before this document was finalised**; the fixes are listed in §0 so that the record of
what was wrong is not erased by the correction.

Legend — **PASS**: no objection. **WARN**: a real limitation, disclosed in the caption and
accepted. **FAIL**: would mislead; must be fixed.

---

## 0. What this review changed

| # | figure | finding | fix |
|---|---|---|---|
| 1 | Fig 5A | The caption claimed zero-shot error "is monotone in distance". It is monotone in *point estimate* only: an unpaired chemotype-block bootstrap of the far − near difference at *k* = 0 gives **+0.35 [−0.04, +0.66]**, which does not exclude zero with 17 chemotypes per tercile. | The contrast is now computed by the figure script, printed on the panel, stored in `figures/derived/fig5_values.json`, and the caption and Results text now say the observational split is underpowered and point at panels B/C for the interventional evidence. |
| 2 | Fig 2A | Drawing gen8's and gen9's *published* frontier numbers would have compared a best-of-eleven-policies envelope against gen10's single fixed rule. | All four rules are now pre-specified and recomputed from one table; the caption states the envelope values and that the ordering at *k* ≥ 2 is unchanged (`METRIC_AUDIT.md` §3.1). |
| 3 | Fig 3A–C | Plotting raw log₁₀ *D* made the level error dominate and hid the effect the figure is about; plotting centred values alone could be read as concealing the level error. | Panels are centred **and** print the per-curve level error, and Figure S7 shows the same six curves raw. |
| 4 | Fig 3 caption | The three shape statistics use two different point statistics (median for slope and span, mean for Spearman and shape MAE), following the project's own metric definitions. | The statistic is now named on every axis label and in the caption. |
| 5 | Fig S4A | Bars on an axis truncated to 0.945–1.105 exaggerated the differences between architectures. | Redrawn as points with intervals. |
| 6 | Fig 2B | The published `marginal_gains.csv` is computed on gen9's adapter chain, not the frozen one, so replotting it would have mislabelled the frozen pipeline's marginal gains. | Panel B recomputes them for the frozen rule (0.382 / 0.095 / 0.066 / 0.026) and `METRIC_AUDIT.md` §3.2 records the difference. |
| 7 | Fig 3 caption | 775 curve evaluations are 155 curves × 5 seeds; the distributions are pseudo-replicated 5×. | Disclosed in the caption; the intervals resample chemotypes and are unaffected. |
| 8 | Fig 6 caption | The panel shows 17 policies; the frozen summary lists 20. | The caption now says seventeen and names where the other three are. |
| 9 | Fig 2A | The green line was labelled simply "oracle-chosen support points", which implied an upper bound at every *k* — but it is an oracle for a **level-only** calibration, and at *k* = 5 the deployable rule overtakes it (0.441 against 0.453). A reader would have been left wondering how a method beats an oracle. | Relabelled "oracle support points for a level-only fit"; the caption explains that it bounds *k* = 1 and not beyond, and why. |
| 10 | Fig S8B | The annotation claimed "regression to the mean: over-predicts low *D*, under-predicts high *D*". The panel plots residual against *prediction*, which is the correct calibration view and shows no such pattern; the compression claim belongs to panel A. | Panel B now reports what it shows — a mean residual of −0.39, i.e. a systematic under-prediction on held-out chemistry — and the compression statement moved to panel A, where sd(predicted) 0.85 against sd(measured) 1.66 supports it. |
| 11 | Fig S10B | `gain_per_measurement` divides by the measurements a strategy actually *spent*, so a breadth strategy that cannot spend the budget scored best on it while achieving a worse macro MAE. | Panel B removed; S10 is now a single log-budget panel, and the reason the column is not plotted is recorded in the caption and in `figS10_values.json`. |
| 12 | Fig S3 | Grey inline labels for six adapters collided with each other and overflowed into panel B. | Panel A now shows six adapters, all coloured and in the legend; panel B carries the complete paired comparison and its labels moved to the right; the full 11-adapter table is in `figS3_values.json`. |

---

## 1. Figure 1 — methodology

| question | verdict | note |
|---|---|---|
| Cherry-picked? | PASS | Schematic; the only numbers are cohort counts read from the frozen outputs by the script. |
| Fair comparison? | n/a | No comparison. |
| Sample sizes clear? | PASS | 5,248 rows / 152 extractants / 79 chemotypes / 99 few-shot extractants / 5 seeds × 12 draws, all on the figure. |
| Uncertainty shown? | n/a | |
| Units independent? | PASS | The chemotype grouping is the subject of panel A. |
| Oracle shown as deployable? | PASS | No oracle appears. |
| Cohorts mixed? | PASS | One cohort; the few-shot subset is named as a subset. |
| Axis scaling? | n/a | |
| Caption overclaims? | PASS | The bit-identical level claim is a verified audit row, not an assertion. |
| Could leakage explain it? | PASS | This figure exists to answer that question; §6 of `METHOD_FIGURE_TEXT.md` states the three properties. |
| Establishes its conclusion? | PASS | |
| Negative result hidden? | **WARN** | The diagram does not show that the relative-position block makes predictions depend on the candidate list. The caption does not mention it either. **Accepted** because Figure 3's caption and Figure S2 carry it, but if only one figure is shown in a talk, this is the omission to name aloud. |

**Overall: PASS (one accepted WARN).**

## 2. Figure 2 — few-shot frontier

| question | verdict | note |
|---|---|---|
| Cherry-picked? | PASS | Four rules fixed in the script before running; no per-seed or per-extractant selection. |
| Fair comparison? | PASS | Identical extractants, identical support/query draws, identical repeats; §0 item 2 documents the one place where fairness cuts against our own method. |
| Sample sizes clear? | PASS | 99 extractants, 41 chemotypes, 5 seeds, 12 draws, in the caption; counts improved printed in B and D. |
| Uncertainty shown? | PASS | Seed range in A, paired BCa intervals in B and C. |
| Units independent? | PASS | One vote per extractant; intervals resample chemotypes. |
| Oracle shown as deployable? | PASS | Green, dashed, labelled "not deployable" in the legend. |
| Cohorts mixed? | PASS | One cohort throughout; the 0.9695 whole-corpus figure never appears here. |
| Axis scaling? | **WARN** | Panel A's *y* axis starts at 0.36, so the *k* = 0 → 1 drop occupies about 50 % of the panel height where it is 37 % of the value. Accepted for a line plot with a labelled axis; the caption gives the absolute numbers. |
| Caption overclaims? | PASS | |
| Could leakage explain it? | PASS | Support rows are excluded from scoring by construction; the model is never refitted. |
| Representative? | PASS | All 99 extractants; panel D shows the spread including the 33 that get worse. |
| Establishes its conclusion? | PASS | |
| Negative result hidden? | PASS | Panel D and the "% improved" row in B are the negative results. |

**Overall: PASS (one accepted WARN).**

## 3. Figure 3 — shape failure and repair

| question | verdict | note |
|---|---|---|
| Cherry-picked? | PASS | Rule declared in `FIGURE_PLAN.md` §9 before inspection; ranked table and runners-up shipped; Figure S7 plots six curves including the runners-up. |
| Fair comparison? | PASS | Same folds, same learner family, same held-out curves. |
| Sample sizes clear? | PASS | 155 curves, 25 extractants, 775 curve × seed, on the figure and in the caption. |
| Uncertainty shown? | PASS | Chemotype-block intervals on the point statistic in E and F; full distributions plotted. |
| Units independent? | **WARN** | Only **8** Tanimoto-0.7 chemotypes carry extractant titrations, so the chemotype-blocked interval is the *narrowest* available. Disclosed, with the publication-blocked interval (+0.196 [0.140, 0.257]) quoted in the caption as the one a sceptic should use. |
| Oracle shown as deployable? | n/a | |
| Cohorts mixed? | PASS | |
| Axis scaling? | PASS | Panels D–F have natural origins; no truncated bars. |
| Caption overclaims? | PASS | The caption states the repair is partial (1.02 vs 2.57) and exploratory. |
| Could leakage explain it? | **WARN** | The relative-position columns read the candidate condition list. This is legitimate under the deployment protocol but it is a real dependence, and it is quantified in Figure S2 (a decoy point two decades outside the window moves predictions by more than the mechanism's macro gain). Caption cites it. |
| Representative? | PASS | |
| Establishes its conclusion? | PASS | |
| Negative result hidden? | PASS | Panel C *is* a failure case, chosen by rule. |

**Overall: PASS (two accepted WARNs, both load-bearing enough to be in the abstract).**

## 4. Figure 4 — error decomposition

| question | verdict | note |
|---|---|---|
| Cherry-picked? | PASS | The seven components are the pre-registered gen10 Phase 10 list, plotted whole, including the one with a negative contribution. |
| Fair comparison? | PASS | Panel A recomputes the cascade on the same cohort and unit as the achieved values, precisely so oracle and achieved share an axis. |
| Sample sizes clear? | PASS | 99 extractants (A, C), 131 clusters / 152 extractants (B), stated per panel. |
| Uncertainty shown? | **WARN** | Panel A has no intervals — it is an exact substitution on stored residuals, not an estimate with sampling error in the usual sense, but the cohort is still finite and a bootstrap would widen every bar. Accepted; the derived JSON carries the per-seed values. |
| Units independent? | PASS | |
| Oracle shown as deployable? | PASS | Hatched, green, separated by a rule, with "not deployable" printed inside the panel. |
| Cohorts mixed? | **WARN** | Panel B is on the 152-extractant cluster-voted cohort while A and C are on the 99-extractant extractant-voted cohort. They are never on a shared axis and the caption names both, but a hurried reader could compare 0.520 (B) with 0.507 (A). |
| Axis scaling? | PASS | Bars from zero. |
| Caption overclaims? | PASS | States that the components overlap and that the sum is a floor. |
| Could leakage explain it? | PASS | Oracles read targets *by design* and are labelled. |
| Establishes its conclusion? | PASS | |
| Negative result hidden? | PASS | The 0.103 unexplained is plotted, and the "realistically removable" bar is zero for four of seven components. |

**Overall: PASS (two accepted WARNs).**

## 5. Figure 5 — generalisation and coverage

| question | verdict | note |
|---|---|---|
| Cherry-picked? | PASS | Terciles are gen10's own definition; all extractants used; both negative controls plotted. |
| Fair comparison? | PASS within panels | Panel A is one model across distance bands; panels B/C are one learner across training masks with byte-identical held-out rows. |
| Sample sizes clear? | PASS | *n* on every bar and in every legend entry. |
| Uncertainty shown? | PASS | Marginal chemotype-block intervals in A, seed range in B, paired BCa in C, plus the far − near contrast added by this review. |
| Units independent? | PASS | |
| Oracle shown as deployable? | n/a | |
| Cohorts mixed? | **WARN** | Panels A and B/C use the same corpus but different learners and different voting units. They are deliberately not on one axis and the caption says so; a reader who assumes one model across the figure will be misled. |
| Axis scaling? | PASS | Bars from zero. |
| Caption overclaims? | PASS after fix | See §0 item 1. |
| Could leakage explain it? | PASS | The held-out rows are byte-identical between arms and hashed per fold; the test extractant's own chemotype is out of every arm's training set. |
| Representative? | **WARN** | 57 of the 131 scoring units in B/C are clusters the restricted arm structurally cannot serve, so the *magnitude* is a property of this cohort's composition. Stated in the caption and in `METRIC_AUDIT.md` §5. |
| Establishes its conclusion? | **WARN** | Panel A alone does not; panels B/C do, for the interventional claim. The figure is honest about which is which only because the caption says so. |
| Negative result hidden? | PASS | The row-matched control being indistinguishable from the treatment is plotted; the +0.019 on already-covered extractants is in the caption. |

**Overall: WARN — the weakest main figure. Recommendation in `FINAL_FIGURE_REPORT.md` §4.**

## 6. Figure 6 — acquisition

| question | verdict | note |
|---|---|---|
| Cherry-picked? | PASS | All seventeen policies of the feature set shown, worst included. |
| Fair comparison? | PASS | Identical candidate pools and identical evaluation rows for every policy; this is what protocol P2 exists for. |
| Sample sizes clear? | PASS | 143 extractants, 5 seeds × 8 draws, in the caption. |
| Uncertainty shown? | **WARN** | The bars are paired-vs-RANDOM BCa intervals re-centred on each policy's own point estimate. This is the right *statistic* — the paired difference — but an unusual *placement*, and a reader could mistake them for marginal intervals. The caption states it explicitly. |
| Units independent? | PASS | |
| Oracle shown as deployable? | PASS | Green, top of the ranking, "not deployable" in the legend. |
| Cohorts mixed? | **WARN** | 143 extractants here versus 99 in Figures 2, 4A and 5A, so 0.629 is not directly comparable with Figure 2's 0.654. The caption gives the cohort; the zero-shot reference line (0.980, not 1.036) makes the difference visible. |
| Axis scaling? | PASS | Bars from zero in both panels. |
| Caption overclaims? | PASS | |
| Could leakage explain it? | PASS | Policies are target-blind by construction; a metamorphic test corrupts every target and requires identical selections. |
| Establishes its conclusion? | PASS | For *k* = 1. The caption should note — and now does not — that `D_OPTIMAL` overtakes centrality at *k* = 5. **Action: add one sentence.** |
| Negative result hidden? | PASS | The uncertainty policies, the farthest-point policy and the learned rankers are all plotted with their failures. |

**Overall: PASS (two accepted WARNs, one caption addition).**

## 7. Supplementary figures

| figure | verdict | note |
|---|---|---|
| S1 dataset composition | PASS | Descriptive; the two concentration facts (one extractant = 28 % of rows, one chemotype = 64 %) are the point. |
| S2 query robustness | PASS | A negative result about our own contribution, plotted at the same scale as the gain it undermines. |
| S3 adapters | PASS | Fixed 99-extractant cohort, so it agrees with Figure 2 exactly; note that `runs/gen10_final/adaptation/summary.csv` uses a *k*-varying cohort (143/143/140/99) and therefore prints different numbers. Panel A plots six adapters and panel B compares all eleven, so nothing is dropped from the comparison. |
| S4 architectures | PASS after fix | Points, not truncated bars (§0 item 5). |
| S5 shape by axis | **WARN** | Three of the six axes have 8–35 curves in total; the differences there are not interpretable and the caption says so. |
| S6 seed stability | PASS | |
| S7 raw example curves | PASS | Exists to make the Figure 3 centring auditable. |
| S8 diagnostics | PASS | Panel C reports data-quality strata that inflate our own headline. |
| S9 dose–response | PASS | The strongest available evidence against the circularity objection to Figure 5. |
| S10 budget | **WARN** | Two of eight strategies are omitted from the panel for legibility and two more coincide exactly below a budget of 99; both facts are in the caption and all eight are in the derived JSON. |

---

## 8. Residual objections this figure set cannot answer

These are not fixable from existing data and should be conceded in the Discussion rather
than defended.

1. **The distance effect is observational.** Figure 5A cannot separate "unfamiliar
   chemistry" from "intrinsically harder chemistry". Only the coverage experiment
   intervenes, and it intervenes on the training set, not on the held-out extractant.
2. **8 chemotypes carry the extractant-titration result.** Every shape conclusion rests on
   them. The publication-blocked interval is quoted for this reason, but no resampling
   scheme can manufacture chemotypes that were never measured.
3. **The frontier is series-local.** Nothing here shows that *k* measurements in one solvent
   system help in another; gen8's transfer matrix says they largely do not.
4. **`log D` for one lanthanide is not a separation factor.** The quantity a process chemist
   optimises is a ratio between two lanthanides under identical conditions, which is a
   different — and, on this corpus, differently structured — target.
5. **No prospective test.** Every number is retrospective cross-validation on a fixed
   corpus. The protocol is designed to make a prospective test easy; none has been run.
