# Round-5 ideas, ranked and tiered (gen21, strict-adjacent log SF at k = 3)

Written 2026-10-23. This file consolidates three inputs:
- the 39 merged ideas (`merged_ideas.json`, M01–M39);
- two skeptic verdicts per idea: **Skeptic A** (novelty, evidence and leakage) and **Skeptic B** (feasibility, power and value);
- three judge rankings.

The companion file `critiques.json` holds every idea in full, both verdicts in condensed form, the judge scores, the tier, the rank and the priors.

**Status.** This is a design document. Nothing in it is sealed or confirmatory. Every number computed on the 229 C3 cells and the 5 dev seeds is a lead, because these cells have already been seen. "Exploratory" marks a skeptic's or this note's own rerun on the round-4 OOF parquets or the V1 bench.

---

## Bottom line

1. **No idea opens a route to 0.50 on the main estimand** (E-A: BP/BPG pooled R², B1 = 0.452). Every model-side correction proposed for round 5 falls into one of three groups:
   - it sits below the dev MDE (about 0.011 R²);
   - it is carried by one or two publications;
   - it was already killed by its own criterion in the skeptics' exploratory reruns.
2. **The one regime above 0.50 is condition transfer within a laboratory or campaign** (M01, estimand E-CT).
   - On the V2-clean cells with at least 2 siblings and without C5-BTBP (99 cells, 12 publications), B1 scores 0.531 and the shrunk sibling template scores 0.707 on identical rows.
   - But leave-one-cell-out publication × pair offsets recover about 95 % of that gain (0.665–0.673 against 0.682). So the signal is source-local information, not the template's shape.
   - E-CT is a separate estimand and must never be quoted next to 0.452.
3. **Round 5 should do four cheap software things first:**
   - freeze M01 as E-CT, with its corrected design;
   - seal the inference convention (M34, as a minutes-long resampling replay);
   - seal the decision convention (M35);
   - close dead routes in one line each (M24 and the closure ledger below).
4. **Everything that could confirm a claim is data-gated:**
   - source recovery (M25);
   - the SAFE gap-ID audit (M33);
   - a re-measurement campaign (M11);
   - the external condition-transfer leg (M05);
   - inter-laboratory reproducibility (M12).
5. **One provenance-pending publication controls the outcome of at least 12 ideas.** It is C5-BTBP (pub_6a0ef8528c), and inside it the single cell c950dc2982e96b83 (11.8 % of B1's SSE). It drives or kills the effect in M02, M03, M04, M07, M08, M09, M10, M23, M24, M27, M28 and M32. Recovering that source (M25) is the highest-stakes item per hour of work in the programme.

---

## How to read this file

**Judges.** Each judge scored every idea from 0 to 10. Their lenses are inferred from the reasons they gave:
- **J1, information value:** does the answer change beliefs or programme decisions?
- **J2, practical value for a chemist.**
- **J3, feasibility and decisiveness in round 5.**

**Prior.** The prior is each skeptic's probability that the hypothesis holds and passes its own kill criterion. Where the two skeptics priced different things (for example "passes its own kill" against "the substantive claim is true"), the consolidated line says so.

**Tiers.**

| tier | meaning |
|---|---|
| Tier 1 | Run first in round 5. Software only, cheap, and it changes how round 5 decides or what it may claim. |
| Tier 2 | Run if Tier 1 leaves room. Cheap descriptive diagnostics or narrow leads, never decision slots. |
| Tier 3 | Data-gated. Each item is phrased as a request to the user. |
| Rejected | The fatal flaw is named. Where a fragment is worth keeping, it is moved to another idea or to the closure ledger. |

**Order.** Ideas are ranked first by tier (what round 5 can and should run). Within a tier, the order is the mean judge score, with ties broken by decision value and cost; any exception is stated. The table also gives each idea's merit (mean score), so a Tier-3 item can carry a higher score than a Tier-2 item.

**Hard separations kept throughout.**
- E-CT, E-W, E-DEC, E-RANK, E-KX and E-CENS are separate estimands. None of them may stand in for E-A.
- An equal-weight (per-source) estimand never replaces pooled R².

---

## Summary table

| rank | id | tier | short title | J1 | J2 | J3 | mean | survives A / B | prior A / B | fate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | M01 | 1 | Condition-transfer estimand E-CT (shrunk own-series template) | 7.5 | 8 | 6.5 | 7.33 | yes / yes | 0.50 / 0.55 | run now; absorbs M02, M03, M04 as diagnostics and M05 as its external leg |
| 2 | M34 | 1 | Few-cluster-valid paired tests, as a resampling replay | 4.5 | 4.5 | 5 | 4.67 | no / yes | 0.20 / 0.25 | run now (minutes); absorbs M36, M37 |
| 3 | M35 | 1 | One-currency decision convention (sealed amendment) | 3.5 | 3.5 | 4 | 3.67 | yes / no | 0.30 / 0.30 | seal now; absorbs M38 gate, M21 column, M08 table |
| 4 | M24 | 1 | Floor-aware predictors: closure on V1 | 3 | 3.5 | 4 | 3.50 | no / yes | 0.45 / 0.85 | one-line closure now |
| 5 | M07 | 2 | Narrowed: sc009 (TODGA-class) Nd-dip transfer, Pr/Nd endpoint | 3.5 | 5 | 2.5 | 3.67 | no / no | 0.15 / 0.15 | freeze the value now; external sign test |
| 6 | M03 | 2 | Sibling weighting factorial (inside M01) | 3.5 | 3.5 | 3 | 3.33 | yes / no | 0.40 / 0.20 | M01 secondary diagnostic |
| 7 | M09 | 2 | Level window as a gross-error-risk reporting axis | 3 | 4 | 3 | 3.33 | no / no | 0.15 / 0.15 | descriptive; absorbs M10 flags, M39 ledger |
| 8 | M22 | 2 | Re-scoped: pre-declared literature yield count | 3.5 | 3 | 3 | 3.17 | no / no | 0.08 / 0.05 | web desk work; digitisation QA later |
| 9 | M18 | 2 | Density-indexed metal ladder (H-BUDGET row) | 3 | 3 | 3 | 3.00 | no / no | 0.75 / 0.85 | descriptive; one live density test |
| 10 | M14 | 2 | Selective direction calls (E-DEC deliverable, B0 vs B1) | 2.5 | 4 | 2.5 | 3.00 | no / no | 0.20 / 0.35 | descriptive; absorbs M16 line |
| 11 | M17 | 2 | Read-policy envelope (H-READ) | 2.5 | 2.5 | 2.5 | 2.50 | no / no | 0.60 / 0.60 | descriptive row in M18 |
| 12 | M13 | 2 | Per-cut wider-pair table (E-W) | 2 | 3 | 2 | 2.33 | no / no | 0.80 / 0.65 | descriptive appendix |
| 13 | M25 | 3 | Stakes-first source recovery (accounting audit) | 5 | 5 | 3.5 | 4.50 | no / yes | 0.25 / 0.25 | **user: journal access, one download** |
| 14 | M33 | 3 | SAFE gap-ID audit (label-blind supply gate) | 4 | 4 | 3 | 3.67 | no / yes | 0.05 / 0.20 | **user: SAFE export of gap IDs** |
| 15 | M11 | 3 | Label-blind re-measurement of C3 conditions (H-CEILING redesign) | 4.5 | 4 | 2.5 | 3.67 | no / no | 0.30 / 0.30 | **user: laboratory collaborator** |
| 16 | M05 | 3 | External condition transfer (M01 prospective leg) | 4 | 3.5 | 2 | 3.17 | no / no | 0.12 / 0.15 | **user: permission for SI fetches** |
| 17 | M12 | 3 | Inter-lab reproducibility: Stage-0 lead + lab arm in M11 | 3 | 3 | 1.5 | 2.50 | no / no | 0.12 / 0.10 | lead now; lab arm via M11 |
| 18 | M29 | R | Aqueous-complexant Gd rule | 3 | 4 | 1.5 | 2.83 | no / no | 0.08 / 0.04 | adopt as data-contract rule, unscored |
| 19 | M38 | R | Pre-slot decidability gate | 2 | 3 | 2.5 | 2.50 | no / no | 0.30 / 0.20 | retire-only gate → M35 |
| 20 | M31 | R | Non-DGA Gd-triad acquisition | 3 | 3 | 1.5 | 2.50 | no / no | 0.12 / 0.10 | family-stratified arm → M11, tag → M33 |
| 21 | M06 | R | S0–S4 value-of-information ladder | 2.5 | 2.5 | 2 | 2.33 | no / no | 0.20 / 0.25 | annex → M01 |
| 22 | M27 | R | Ce reliability and registered null | 2.5 | 3 | 1.5 | 2.33 | no / no | 0.07 / 0.12 | floor route → M25; ledger |
| 23 | M39 | R | Nested re-estimation of the five post hoc leads | 2 | 3 | 2 | 2.33 | no / no | 0.30 / 0.30 | optimism table → M09 |
| 24 | M02 | R | 2 reads + sibling, shift-plus-tilt | 2 | 2.5 | 2 | 2.17 | no / no | 0.15 / 0.15 | one ladder row → M01 |
| 25 | M04 | R | Choose conditions, not metals | 2 | 2.5 | 2 | 2.17 | no / no | 0.30 / 0.55 | one line → M01 |
| 26 | M08 | R | Variance budget and futility rule | 2.5 | 2 | 2 | 2.17 | no / no | 0.20 / 0.20 | moment table → M35 appendix |
| 27 | M15 | R | Screening estimand with LINE control | 2 | 2.5 | 2 | 2.17 | no / no | 0.25 / 0.35 | one table in the reporting appendix |
| 28 | M36 | R | Publication vs cluster confirmation units | 2 | 2.5 | 2 | 2.17 | no / no | 0.25 / 0.25 | unit rule → M34 |
| 29 | M16 | R | Publication-balanced conformal intervals | 1.5 | 3 | 1.5 | 2.00 | no / no | 0.05 / 0.07 | one line → M14 |
| 30 | M23 | R | E-CENS interval scoring at limits | 2 | 2 | 1.5 | 1.83 | no / no | 0.35 / 0.80 | one sensitivity line |
| 31 | M26 | R | Nested data-version ladder | 1.5 | 2 | 2 | 1.83 | no / no | 0.12 / 0.15 | two-line closure |
| 32 | M37 | R | Sequential e-process ledger | 2 | 2 | 1.5 | 1.83 | no / no | 0.20 / 0.15 | bookkeeping → M34 behind the M33 gate |
| 33 | M10 | R | Measure in the window (sibling reproducibility) | 1.5 | 2 | 1.5 | 1.67 | no / no | 0.07 / 0.12 | flags → M09, low-D arm → M11 |
| 34 | M19 | R | Allocate the next read between cells | 1.5 | 2 | 1.5 | 1.67 | no / no | 0.20 / 0.22 | one line |
| 35 | M20 | R | Unmeasurable-end (Pm) scenario | 1.5 | 2 | 1 | 1.50 | no / no | 0.10 / 0.25 | G8 scenario note |
| 36 | M30 | R | Yttrium as a 4f-free probe | 1.5 | 2 | 1 | 1.50 | no / no | 0.05 / 0.03 | "record Y" rider |
| 37 | M32 | R | Rigid / preorganised stress set | 1.5 | 2 | 1 | 1.50 | no / no | 0.07 / 0.08 | SMARTS tag → M33/M22 |
| 38 | M21 | R | Provenance class as a reporting axis | 1 | 2 | 1 | 1.33 | no / no | 0.05 / 0.06 | one column → M35 |
| 39 | M28 | R | B1+ factorial (Ce-acid × edge) | 1.5 | 1.5 | 1 | 1.33 | no / no | 0.05 / 0.04 | edge as reference variant; Ce closed |

**Counts.** Tier 1: 4. Tier 2: 8. Tier 3: 5. Rejected: 22. Seven ideas survived at least one skeptic: M01 (both), M03 (A), M24 (B), M25 (B), M33 (B), M34 (B) and M35 (A).

---

## Cross-cutting rules the critiques imply for round 5

1. **Report C5-BTBP sensitivity everywhere.**
   - Report every round-5 number with and without pub_6a0ef8528c.
   - Add leave-one-publication-out and single-cell influence for c950dc2982e96b83.
   - A gain that lives there is a provenance sensitivity (M25), not a model result.
2. **In-corpus legs whose thresholds sit just below already-seen values are consistency checks, not tests.** This applied to most ideas, including M01, M09, M13, M14, M15, M17, M18 and M23. Only contrasts whose outcome is still open, or prospective legs, may carry decision language.
3. **Every claim must name its cheapest same-information comparator.** The skeptics found that the missing comparator changed the conclusion in these cases:
   - publication offsets for M01;
   - B0 for M14;
   - cell-matched abstention for M09;
   - a universal Nd term for M07;
   - IQR-random for M04;
   - the affected-query count for M19;
   - a width-matched row-weighted interval for M16.
4. **Seal the inference unit, the test and alpha before any round-5 contrast** (M34). Conclusions change with the unit: B1 − B0 equal-weight is p 0.007 with clusters as units, p 0.0001 with publications and p 0.26 with chemotypes.
5. **Supply is a stock, not a flow.** The "+1,618 newer SAFE records" are mostly ID gaps that pre-date the archive's last batch (this note's own check reproduces Skeptic A; see "Exploratory check"). Open-access supply is 3–5 sources in total.
6. **State the k-metal scenario.** With ICP-MS all 14 Ln come from one experiment, so the cost unit is the extractant × condition. Metal-count economics apply only to radiotracers, subset feeds and literature mining.
7. **Strata, near-copy rules and windows must be label-free.** The label-based shape-sd < 0.02 rule and windows on true log D are sensitivities only.
8. **"Same publication" is not "same laboratory".** The corpus has no laboratory key. Annotate laboratories from DOIs before any claim about "your lab".

---

## Tier 1: run first in round 5

### 1. M01: Condition-transfer estimand E-CT (same campaign, same extractant, shrunk mean-sibling template)

- **Hypothesis.** When a laboratory has already measured the same extractant at other conditions, the shrunk mean of those series predicts strict-adjacent log SF at a new condition, from 0–4 reads, clearly better than B1 with 4 reads. The source of that gain must be separated into template shape, source-local offset and shared artefact.
- **Why (evidence).**
  - E6 §6.8 reproduces exactly: 149 cells and 1,111 rows per seed. SIB scores 0.573 and SH (K = 2) 0.595, against B1 0.449.
  - Read ladder at 0 / 2 / 3 / 4 reads: 0.411 / 0.517 / 0.554 / 0.573.
  - Stratum n_sib ≥ 2 without C5-BTBP (112 cells, 12 publications): SH 0.682 against B1 0.510.
    - Family-level jackknife lower bound for the gain: +0.089.
  - V2-clean version of that stratum (99 cells): B1 0.531 against SH 0.707. 13 of 14 units are positive, and the top 3 carry 74 % of the gain (Skeptic A).
  - Within-publication subsampling, in cells with at least 5 siblings (89 cells, 7 publications):

    | siblings used (m) | 1 | 2 | 3 | 5 | all |
    |---|---|---|---|---|---|
    | raw template | 0.11 | 0.33 | 0.42 | 0.48 | 0.52 |
    | SH | 0.416 | 0.454 | 0.473 | 0.504 | 0.531 |

    B1 scores 0.352 on these cells. So averaging is the mechanism, and SH already helps with one series. In the n_sib = 1 stratum, SH scores 0.601 against B1 0.544.
  - Against:
    - Publication × pair-type offsets of B1's residual, leave-one-cell-out with K pinned, score 0.665–0.673 against SH 0.682, about 95 % of the gain (Skeptic A).
    - The decisive stratum holds only 3 source clusters, with 82–84 % of rows in g:pub_74428aeab2 (cluster Kish about 1.4).
    - Ce pairs carry 32 % of the gain, a suspected analytical artefact; Gd pairs carry 28 %.
    - Mixed-condition siblings carry 83–93 % of the gain.
    - In 6 cells of one publication the siblings differ only in contact time, and there SIB R² is 1.000.
- **Corrected test design.** This combines both skeptics' fixes.
  1. **Status.** A frozen, separately registered estimand E-CT. In-corpus values are labelled "chosen after seeing these rows". They are never placed next to 0.452.
  2. **Population.** Built from metadata only. Siblings share the publication and the extractant and cover the target's metals.
     - Drop round-3 V2 duplicates (D1 DMDPhPDA copy B, D2 double entries, D3 hidden merges), both as targets and as siblings.
     - Drop siblings that differ from the target only in contact time, metal concentration or phase ratio.
     - Keep the label-defined shape-sd rule as a sensitivity only.
     - Primary stratum: n_sib ≥ 2 without C5-BTBP. C5-BTBP included is a sensitivity.
  3. **Frozen predictor.**
     - SH = w·SIB + (1 − w)·B1, with w = n/(n + K) and K = (1 − r)/r = 1.78 from the roughness correlation r = 0.36. This fixes K from an independent statistic, not from an R² sweep; K = 1.78 gives 0.5951 against 0.5946 for K = 2.
     - At 2–3 reads, shrink toward B1 or PWL fitted on the same reads, so the predictor does not borrow 4-read B1.
  4. **Comparators on identical rows, with DOPT_Q read sets nested within seed.**
     - B1 at the same k and at 4 reads.
     - COPY at 0 reads.
     - OFF: extractant × publication pair-type offsets with K pinned.
     - Nearest-condition sibling, with the distance |Δlog acid| + |Δlog L| + diluent flag declared in advance.
     - Random single sibling, as an exact expectation.
     - Placebo blends at the same w: a same-chemotype template from another publication, and another extractant from the same publication.
  5. **Dose-response.** Within-publication sibling subsampling at m = 1, 2, 3, 5 in cells with n ≥ 5. Between-cell n_sib strata are reported only descriptively.
  6. **Decompositions.**
     - By pair type, and also excluding Ce pairs and excluding Gd pairs.
     - By condition-change type from the cond__ metadata (acid only, acid type, diluent, extractant concentration, temperature, mixed).
     - Extractant-concentration-only and diluent-only cells are reported separately, because B1 already scores 0.78–0.92 there.
  7. **Inference.** The unit is publication × chemotype (or publication × extractant).
     - Chemotype-stratified sign-flip, or IM-t, plus a leave-one-publication-out jackknife.
     - Report the top-3 share and the equal-weight per-publication relative SSE.
     - Cluster-level inference is stated as "not estimable" (3 clusters).
  8. **Laboratory annotation.** Desk work: corresponding-author groups for the 19 publications. Score same-lab, other-publication templates wherever they exist. Until then the claim reads "new condition within the same study or campaign".
  9. **Folded-in diagnostics.**
     - M03 weighting factorial.
     - M02 read-ladder row: SIB@2 against SIB@4 and COPY.
     - M04 "avoid floor or saturated references" check.
  10. **External leg.** M05, data-gated, sized as a laboratory-unit test. A +0.10 DGA-like gain needs about 11–14 independent laboratory × extractant families, each with at least 3 full-series conditions. A +0.05 gain needs about 40.
- **Kill criterion.**
  - (i) **Template-specific claim narrowed** to "publication-offset regime" if SH − OFF < +0.01 R² (or < 2 % relative SSE) on identical rows. Exploratory value now: +0.009.
  - (ii) **Harm rule.** In the n_sib = 1 stratum, SH may not be worse than B1 by more than 5 % relative SSE. If it is, the deployed rule depends on sibling count.
  - (iii) **Artefact flag.** If at least 50 % of the V2-clean gain sits on Ce pairs, reword the claim as "transfer of a laboratory fingerprint".
  - (iv) **Prospective kill (decisive).** On at least 11 new independent laboratory × extractant families, the claim is abandoned if the family-level one-sided 95 % lower bound is ≤ 0 or the mean is < +0.03. With fewer families the result is descriptive.
  - The in-corpus legs (a)–(c) of the original idea are consistency checks known to pass: SH − B1 is about +0.17.
- **Expected effect.**
  - In-corpus V2-clean stratum: about +0.17 R² over B1 (already seen); SH − OFF about +0.01; m = 1 subsample about +0.06.
  - Externally, for DGA-type laboratories with dense condition grids: +0.05 to +0.10. Elsewhere: unknown.
- **Cost.** Hours. Training-free, and the parquets exist.
- **Risks.**
  - Shared same-campaign artefacts (calibration, floor, re-reports) let the template "predict" errors.
  - Three publications carry most of the gain.
  - The strata were chosen after looking.
  - The external leg is data-gated and probably under-supplied.
- **Practical reading.**
  - With at least one of your own prior series of this extractant, use the shrunk mean of your own series plus 2–4 reads at the new condition.
  - Curves of the same extractant from other laboratories do not substitute.
  - For ICP-MS users the point is condition coverage: the zero-read copy (0.436–0.471) nearly matches B1 with 4 new reads (0.470–0.510) on the same rows.
- **Judges.** J1 7.5 · J2 8 · J3 6.5 (mean 7.33).
- **Prior.** A 0.50 · B 0.55. Consolidated about 0.5 that it holds externally for DGA-type dense-grid laboratories. The template-specific part (beating OFF) is about 0.3.

### 2. M34: Few-cluster-valid paired tests, run as a source-level resampling replay

- **Hypothesis.** Tests built for few large clusters keep their nominal size and have more power than the sealed percentile/BCa/jackknife rule. The confirmation planning number then depends on the declared new-source unit; it can be computed now.
- **Why (evidence).**
  - B1 − B0 by cluster: 11 of 12 non-tied clusters better. Pooled sign-flip p 0.0032. Mean log SSE ratio +0.148 to +0.160, p 0.007. IM-t 2.60–2.88. Publications: 31 of 38 better.
  - The answer depends on the unit. Equal-weight log ratio: +0.160 with clusters as units, +0.227 with publications (p 0.0001), +0.059 with chemotypes (p 0.26).
  - Exploratory replay power for B1 − B0 at G = 8:

    | unit | one-sided 0.05 | one-sided 0.025 |
    |---|---|---|
    | cpg cluster | 0.65–0.84 | 0.55–0.68 |
    | publication | 0.39–0.52 | 0.30–0.35 (about 0.8 at G = 20) |
    | chemotype | 0.29–0.36 | 0.21–0.23 (flat with G) |

  - Under zero-mean but skewed per-source nulls, one tail's size reaches 0.12–0.35 in one direction.
  - Much of the toolkit already exists: WCR, CR2 and Webb weights are built in gen13sep/wildcluster.py, gen16 clusterboot and gen20k.inference. What is new is the IM-t, the log-ratio statistic, the gen21 geometry and the unit study.
- **Corrected test design.**
  - Replace the days-long pointwise generator with a source-level resampling replay of per-unit contrast statistics (ΔSSE, log SSE ratio, win indicator). Sources: the stored BPG/BP OOF parquets (REF B0 / B1 / HIST_D7 / HBK@ML and S01–S11).
  - Run the replay at three unit granularities (cpg cluster, publication, chemotype), with extra per-source noise τ ∈ {0, 0.065}.
  - Primary alpha is one-sided 0.025, matching the sealed rule; 0.05 is a sensitivity. State in advance that G ≤ 5 cannot reject.
  - Tests:
    - exact binomial sign test on per-source wins (estimand π = P(a new source improves));
    - sign-flip on the mean log SSE ratio, labelled a sharp-null test;
    - WCR with CR2 and Webb weights, and a Kish-df jackknife-t, as the cheapest-alternative comparators;
    - the sealed rule as a separate row.
  - Worlds:
    - for gating: symmetric and arm-swap nulls, with at least 2,000 replicates per null cell and a Clopper–Pearson bound on size;
    - reported but not gating: skewed heterogeneous nulls and a dominant-cluster mixture.
  - Parameterise effects by consistency (π and the spread of per-source effects), not by ΔR².
  - Controls: B1 − B0 as the positive control, outside any Holm family; HIST_D7 − B1 and an arm-swap of B1 with itself as negative controls.
  - Declare the new-source unit before data: laboratory × chemotype, publications aggregated within a unit, copy groups merged at arrival from metadata (no retroactive merges).
  - Absorbs M36 (unit aggregation) and M37 (a ledger opened only behind the M33 supply gate).
- **Kill criterion.**
  - If power at G = 8 with publication-like units (one-sided 0.025) is < 0.30 for B1 − B0, keep the 40–80-source planning number and run every round-5 model contrast as a diagnostic.
  - Otherwise publish the G that gives 80 % power for each unit type and stratum (DGA / non-DGA).
  - A test is dropped if its size under symmetric or arm-swap nulls exceeds 0.035 at one-sided 0.025 (Clopper–Pearson lower bound).
- **Expected effect.**
  - Publication-like units: about 13–20 sources for 80 % power at π ≈ 0.8.
  - Chemotype-like units: unconfirmable.
  - Effects ≤ 0.01 are unconfirmable under every test.
- **Cost.** Hours. Minutes for the replay itself.
- **Risks.**
  - An equal-weight estimand answers "the average source", not "the average row", and never replaces pooled R².
  - Clusters are not fully independent (shared laboratories).
  - Dev p-values are sensitivities only.
  - Everything must be sealed before new data arrive.
- **Practical reading.** It tells the data request how many independent laboratories to ask for, and that more publications from the same laboratory add nothing.
- **Judges.** J1 4.5 · J2 4.5 · J3 5 (mean 4.67).
- **Prior.** A 0.20 · B 0.25. Consolidated about 0.2 that power reaches 0.6 at G = 6–8 for realistic sources. The replay itself is almost certain to be informative.

### 3. M35: One-currency decision convention (sealed protocol amendment)

- **Hypothesis.** A decision rule written in relative pair-SSE against B1, with a balanced-MAE non-inferiority guard and BPG deciding, passes the known positive and rejects every round-4 loser. Where pooled and equal-weight readings disagree, the result is a lead.
- **Why (evidence).**
  - Round 4's margins were in different currencies. 0.005 balanced MAE is about +0.034 to +0.053 R², and block A failed that leg even as an oracle.
  - Replay of B1 − B0:

    | statistic | value | t |
    |---|---|---|
    | pooled relative SSE | +5.85 % | 1.99 |
    | equal-weight | +12.3 % | 2.60 (IM p 0.012) |
    | log ratio | +14.75 % | 2.88 |

  - Every round-4 slot is ≤ −0.02 % pooled.
  - In the non-big folds no slot reaches 2 %; the best is S02 at +0.28 to +0.35 %.
  - Problems found:
    - Holm over the 14-contrast replay family gives B1 − B0 Holm p 0.16, so it would kill the positive control as written.
    - The equal-weight leg fires equally on HBK@ML − B0: exact sign-flip p 0.0115 against 0.0137 for B1 − B0, while HBK@ML is +0.13 % pooled.
    - A gain confined to the giant cluster gives IM t = 1 exactly.
    - 8 of the 13 equal votes go to clusters of 1–4 cells, mostly ORNL-only imports.
- **Corrected test design.** Seal before any round-5 score:
  1. E-A pooled R² for continuity.
  2. One gate: pooled relative pair-SSE reduction against B1 ≥ 2 % (about +0.011 R²). It is stated as the same margin in a consistent unit, not as new power. The test comes from M34.
  3. Balanced-MAE non-inferiority guard ≥ −0.001.
  4. BPG decides; BP is a labelled sensitivity.
  5. Mandatory tables:
     - 13-row per-cluster table with a provenance column (ORNL-only, data_location class from M21);
     - big fold against the other folds;
     - the six-design table, restricted to training-free arms where only BP/BPG parquets exist.
  6. Equal-weight per-cluster log SSE ratio as a sealed breadth diagnostic.
     - Clusters with < 3 cells or < 20 rows per seed are pooled into one "small sources" stratum.
     - ORNL-only clusters collapsed into one unit, as a sensitivity.
     - Publication-level relative SSE inside the giant cluster.
  7. M38's retire-only gate, label-free:
     - demote a slot when the deciding fold has fewer than 10 training cells sharing the test rows' donor class or chemotype;
     - demote it when its own feature set's test-label oracle ceiling is below the margin;
     - power is judged at 50 %, not 80 %.
  8. Multiplicity: B1 − B0 is the positive control outside any Holm family; Holm applies only within the round-5 decision family.
  9. M08's relation-class moment table, with an identifiability column, as a one-hour appendix.
- **Kill criterion.**
  - The equal-weight leg becomes a gate only if M34 shows size ≤ 0.05 under a heterogeneous null and power ≥ 0.5 for a homogeneous 2 % relative effect.
  - It is demoted to a diagnostic if its false-pass rate exceeds 0.05 in a "gain only in tiny clusters" world, or its power is < 0.2 for a pooled ≥ 3 % gain confined to the giant cluster.
  - The whole convention is abandoned only if the replay, with Holm fixed, fails the positive control.
- **Expected effect.** No change to 0.452. Consistent margins and fewer unwinnable slots.
- **Cost.** Hours.
- **Risks.**
  - The equal-weight leg invites cherry-picking unless sealed first.
  - The HBK@ML equal-weight "lead" is double-dipping; record it only as a hypothesis for new sources.
- **Practical reading.** None directly. It protects every round-5 claim from currency and weighting artefacts.
- **Judges.** J1 3.5 · J2 3.5 · J3 4 (mean 3.67).
- **Prior.** A 0.30 · B 0.30. Consolidated 0.3. The convention itself is sound; the 1.3× signal-to-noise claim is circular.

### 4. M24: Floor-aware predictors, closure on V1 (gate on a source-stated detection limit)

- **Hypothesis.** A reads-only floor-aware (clamp or censored) predictor cannot be validated on V1. Only a per-source detection limit can fix the floor F.
- **Why (evidence).**
  - The clamp curve reproduces:

    | F | −4.5 | −4.25 | −4.0 | −3.75 | −3.5 | −3.0 |
    |---|---|---|---|---|---|---|
    | R² (B1 0.452) | 0.464 | 0.471 | 0.458 | 0.411 | 0.317 | −0.079 |

  - At F = −4.25, 91 % of the gain comes from one cell, c950dc2982e96b83 (Ce −3.17 between La −4.49 and Pr −4.95). Without that cell the gain is +0.0012 to +0.0018, positive in 2 of 5 seeds.
  - The physically consistent all-reads clamp is negative at every F (−0.002 to −0.0075).
  - A cross-fitted F scores 0.4098, worse than no clamp.
  - No C3 cell outside C5-BTBP has a log D below −4.0.
  - The same-group anchor (SKB R-05-13, D < 1e-4) already flips sign across seeds (3 of 5 negative).
  - Under consistent interval scoring the hidden-only clamp is positive from F = −4.75 to −3.5 (5 of 5 seeds). So the "knife-edge" is a scoring artefact (Skeptic A).
- **Corrected design: a closure line now.**
  - Floor-aware predictors are retired as model claims at every F on V1.
  - Record the exploratory grid (three clamp forms × point and interval scoring × F ∈ [−5, −3]) as already seen.
  - Floors appear only in scoring (the M23 sensitivity line) and in V3 (M25).
  - Send the c950dc Ce value to M25 for a transcription check.
  - Any future floor-aware slot needs all of the following:
    - consistent interval scoring;
    - the all-reads clamp reported next to the hidden-only form;
    - a gain ≥ +0.005 with c950dc excluded, in at least 4 of 5 seeds, spread over at least 3 cells;
    - per-source limits taken from the paper, never one global F;
    - for Tobit-type forms, an F supplied from outside the data.
- **Kill / reopen.** Reopen only if M25 recovers per-source limits and the floor-consistent ΔSSE on rows with both labels ≥ F_src gives at least +0.005 R² in at least 4 of 5 seeds.
- **Expected effect.** None. It frees the H-FLOOR(a) slot.
- **Cost.** Minutes.
- **Risks.** "Retired" could be misread as "no floor exists". It means undecidable without the limit.
- **Practical reading.** Never mix clamped values with raw sub-limit reads in one pair. Report "< DL" together with the per-metal limit. Tools should give bounds for those pairs.
- **Judges.** J1 3 · J2 3.5 · J3 4 (mean 3.50).
- **Prior.** A 0.45 · B 0.85. Consolidated about 0.85 that the closure is correct. The stated "steep threshold" mechanism is probably wrong.

---

## Tier 2: run if Tier 1 leaves room (cheap diagnostics and narrow leads)

### 5. M07: Narrowed to M07-Nd, transferring the sc009 (TODGA-class) Nd dip to the Pr/Nd pair

- **Hypothesis (narrowed).** For TODGA-class DGAs, radius interpolation overstates Pr/Nd selectivity. One Nd point potential borrowed from other publications corrects it beyond a universal Nd term. The extractant-level Gd half of the original idea is dropped.
- **Why (evidence).**
  - The original estimand is empty under BP/BPG folds: 0 of 229 cells qualify in every seed.
  - On leave-one-publication-out (LOPO) rows without C5-BTBP:
    - Gd only: −0.0024 / +0.0016, positive in 2 of 5 seeds;
    - Nd: +0.011 to +0.023, positive in 5 of 5;
    - a universal Nd term: +0.0053 to +0.0063, about 56 % of the chemotype gain.
  - sc009 Pr/Nd R²: 0.265 with B1, 0.419 with the universal term, 0.522 with the chemotype Nd term. This is 46 rows per seed.
  - The sc009 Nd chord is negative in 11 of 12 publications, with an anomaly of about −0.11.
  - B1 implies a Nd/Pr SF of about 1.7 where about 1.34 is observed.
  - The top 3 publications carry 75 % of the gain. Without them: chemotype +0.0010 against universal +0.0027.
  - A random-chemotype permutation reaches at most +0.0035 to +0.0048, against the observed +0.011.
  - The anomaly depends on acid (ρ −0.28 to −0.36).
- **Corrected design.**
  - Freeze now u_Nd(sc009) = −0.11 and the shrinkage.
  - Register E-KX-LOPO as a known-chemotype deployment estimand:
    - B1 base = the BP OOF prediction;
    - donors = other publications and other copy groups;
    - never called BP; never quoted next to 0.452.
  - Primary endpoint: sc009 Pr/Nd R² and MAE.
  - Comparators:
    - B1-NdU, a universal Nd potential;
    - a Pr/Nd chemotype × pair offset (+0.0082);
    - a copy of another publication's Pr/Nd SF (−0.45: borrow the dip, not the SF).
  - Controls: random-chemotype permutation with at least 1,000 draws; publication units (sign-flip over sc009 publications); leave-top-3-out; stratification by |Δlog acid| > 0.5.
  - Drop the "4 of 5 seeds" leg and the random-extractant control.
  - Metals are selected nested: split-half reliability ≥ 0.5 and |mean| > 2 SE.
  - External leg: new TODGA-class publications reporting Pr, Nd and Sm (for example RSC Adv 12:790), predicted with the frozen value.
- **Kill criterion.**
  - In-corpus: narrowed to "universal Nd term" if chemotype-Nd − B1-NdU ≤ 0 with the top 3 publications removed. That is the current expectation.
  - External: killed if the Nd chord is ≥ 0 in at least 2 of the first 4 new sc009 publications, or if the Pr/Nd squared error against B1 does not fall in most of them.
- **Expected effect.** About +0.002 pooled in corpus, below the MDE. sc009 Pr/Nd rises from about 0.26 to about 0.5. The external sign test is the decision.
- **Cost.** Hours, plus web reading for the external papers.
- **Risks.** The lead rests on three publications. It is acid-dependent. ICP-MS measures Nd anyway, so the value is the warning.
- **Practical reading.** For TODGA-class DGAs, do not trust an interpolated Pr/Nd SF. Borrow the dip, not the SF. Do not borrow Gd or Ce values from other laboratories.
- **Judges.** J1 3.5 · J2 5 · J3 2.5 (mean 3.67).
- **Prior.** A 0.15 · B 0.15 as written. The narrowed sc009 Nd claim is about 0.5–0.6 to replicate in new publications.

### 6. M03: Sibling weighting factorial, run only inside M01

- **Hypothesis.** How siblings are weighted (acid-distance kernel, read-level matching, or the plain mean) changes E-CT accuracy.
- **Why (evidence).**
  - Rows with n_sib ≥ 2: 127 cells, 13 publications, 4 clusters.
    - Mean of the 2 level-nearest siblings 0.620, mean of all 0.577.
    - Level-farthest −0.864.
    - Random pair of siblings 0.415.
  - Kernel BLUP 0.564 / 0.626.
  - Plain empirical Bayes (the shrunk mean) 0.589–0.595.
  - The kernel's gain over plain EB is +0.038. With C5-BTBP excluded throughout it is +0.007 (7 of 11 publications improve; clusters split 3 / 3), below its own +0.01 kill.
  - The acid shuffle comes within 0.0045 of the kernel on all rows.
  - 73 % of the level-rule gain is C5-BTBP; without it the gain is +0.020.
  - Offset-removed read matching gives 0.619, as good as level matching (0.622). So "matching level" is not shown; "matching read pattern" fits equally well.
  - The level rule gains only at n_sib ≥ 5, i.e. in dense grids (7 publications). At n_sib = 3–4 the plain mean wins.
- **Corrected design.**
  - M01's primary rule (plain EB) is frozen first.
  - Factorial: {uniform, acid kernel, read-match kernel, both} × count shrinkage. Selection is nested leave-one-publication-out with a one-SE rule toward uniform, and the chosen hyperparameters are reported per fold.
  - Selection-aware nulls:
    - acid permuted within publication, at least 200 permutations, with re-selection;
    - a permuted-read control;
    - a matched-count random-2 control.
  - Add the mean of the 2 condition-nearest siblings as the symmetric competitor.
  - Exclude C5-BTBP from both training and scoring. Report sensitivities without pub_6a0ef8528c and pub_19783c039b.
  - Pre-declared strata: acid interpolation vs extrapolation; dense grids (n_sib ≥ 5); constant-acid families; diluent vs acid changes; DGA vs non-DGA.
  - Average over ties; never break them by table order.
- **Kill criterion.**
  - Deploy the uniform shrunk mean if the nested gain over SH is < +0.01 without the two named publications, if fewer than 7 of the remaining 11 publications improve, or if ℓ = ∞ is chosen in most folds.
  - The level rule is killed if it fails to beat both the plain mean and the mean of the 2 condition-nearest in at least 60 % of publications with ≥ 5 siblings.
- **Expected effect.** At most +0.02 over SH, only in dense same-lab grids. The likely outcome is "uniform shrunk mean".
- **Cost.** Hours, in the same run as M01.
- **Risks.** 4 clusters; it cannot be confirmed in round 5, because M05 tests only the template.
- **Practical reading.** Average at least two of your own series. Never use one random series in a dense grid; it is worse than B1. Weighting by nearby acidity matters only in dense same-lab acid grids.
- **Judges.** J1 3.5 · J2 3.5 · J3 3 (mean 3.33).
- **Prior.** A 0.40 · B 0.20. Consolidated 0.2–0.3 that weighting beats plain EB by ≥ 0.01 robustly.

### 7. M09: Level window as a registered gross-error-risk reporting axis

- **Hypothesis (narrowed).** B1-predicted levels of both metals in [−2, 2] mark the rows where gross errors are rare. It is a cell- or condition-level risk flag, not a measurability law.
- **Why (evidence).**
  - The window [−2, 2] scores 0.540 at 74.7 % coverage. The nested half-width choice scores 0.527–0.532 at about 75 %.
  - The coverage–R² curve is monotone, so a "maximise R² at a coverage floor" rule simply returns the narrowest window that meets the floor.
  - Matched-coverage random abstention controls:

    | control | R² |
    |---|---|
    | unstratified | 0.453 |
    | publication-stratified | 0.489–0.490 |
    | publication × pair type | 0.501 |
    | cell-matched | 0.510–0.511 [0.500, 0.522] |

  - Excess over the cell-matched control is +0.029, with cluster bootstrap [−0.069, +0.055].
  - That excess falls to +0.000–0.002 when the top 10 cells or pub_126bbe7976 are removed.
  - Outside the giant cluster and C5-BTBP, the window gives 0.505 against 0.512 on all rows at 90.4 % coverage.
  - Tail behaviour: P(|res| > 0.5) is 0.009 inside against 0.043 outside. Without C5-BTBP the median |res| is the same inside and outside (0.054 vs 0.052).
  - The window-aware Mondrian interval is only 11–12 % narrower than pair-type-only.
  - Nesting over the whole family of scores gives 0.516 at 78.7 %, with the level window chosen in 15 of 25 fold-fits.
- **Corrected design.**
  - A descriptive registered coverage–MSE and coverage–relative-SSE curve over a frozen grid: half-widths {1, 1.5, 2, 2.5, 3, ∞}, lower-only {−2, −2.5, −3, −4}, upper-only {2, 2.5, 3}. Labelled "fixed after seeing V1".
  - Operating points are label-free quantiles of s = max(|levA|, |levB|) at coverage 0.75 and 0.90, computed on training folds.
  - Two estimands:
    - (a) **Tail flag.** The outside/inside ratio of P(|res| > 0.5) and P(|res| > 0.3), with a publication bootstrap, on all rows and without C5-BTBP.
    - (b) **Bulk precision.** Median |res| and the q80 Mondrian half-width, inside the window against pair-type-only on the same rows.
  - Controls: cell-matched (primary), publication × pair type, unstratified.
  - Cheapest alternatives at matched coverage: min read ≥ −2 (the M10 reads-only flags), a whole-cell rule, "drop predicted min < −4", "drop C5-BTBP" (0.486 at 91.8 %).
  - Winsorised versions of the curves.
  - Leave-top-k-cells out (k = 5, 10) and leave-one-publication-out.
  - The M39 optimism ledger, with family nesting.
- **Kill criterion.**
  - "Row flag" may be claimed only if the family-nested inside relative SSE beats the 97.5 % quantile of the cell-matched control. Otherwise it is a cell or condition flag.
  - The tail flag passes if the outside/inside ratio for |res| > 0.5 is ≥ 3 with a CI excluding 1, both with and without C5-BTBP.
  - The bulk-precision claim needs ≥ 15 % narrowing and is expected to fail.
- **Expected effect.** (a) passes; (b) fails. On new sources the frozen rule needs at least 100 out-of-window pairs from at least 3 publications, which is data-gated.
- **Cost.** Hours.
- **Risks.** Post hoc threshold history; partly a "not C5-BTBP" flag.
- **Practical reading.** If both predicted D values lie between 0.01 and 100, about 1 pair in 110 is off by more than 0.5 log; outside that range about 1 in 23 is. Re-measure, or move D into range, before trusting light-pair SFs.
- **Judges.** J1 3 · J2 4 · J3 3 (mean 3.33).
- **Prior.** A 0.15 · B 0.15 as written. The narrowed tail-flag claim is about 0.7.

### 8. M22: Re-scoped to a pre-declared, label-blind literature yield count (digitisation calibration only as later QA)

- **Hypothesis (re-scoped).** Open-access literature holds enough new independent laboratory × chemotype units to make external confirmation feasible for the large-effect estimands (E-CT, the window).
- **Why (evidence).**
  - Figure-only papers are already counted in the 3–5-cluster supply: 4 of the 7 candidates are plot-only.
  - Under cluster_cpg, papers on known chemotypes add no new clusters, and 2 of the 7 candidates are exact C3 extractants.
  - Heterogeneous run-4 power for a B1-sized gain: 0.135 / 0.27 / 0.34 at G = 20 / 40 / 80.
  - The figure-vs-table contrast is a cohort effect: all FIG cells sit in the giant cluster, and within TODGA the roughness is the same.
  - The cited 0.119 figure-vs-table disagreement comes from 3 sc020 cells of pub_cd72e4a7bb, where the figure and the table are different series; V1 averages them (La −1.585 and −0.481 become −1.033).
  - Same-point double reports agree to about 0.003 decade.
  - LLM plot extraction has errors of about 5 % of the axis range, about 0.2 decade on a log axis.
- **Corrected design.**
  - **Phase A (web reading only, no downloads, values never read).**
    - Freeze the search first: databases (OpenAlex, Crossref, J-STAGE), query strings, date window, and inclusion rules (≥ 8 Ln including Eu, Gd, Tb; D on the axis or tabulated; an organic extractant).
    - Count at DOI level. Rebuild SMILES and assign chemotype and cluster with the frozen gen6 map.
    - Report new clusters, new laboratories, multi-condition same-extractant families, and tags: rigid SMARTS (M32), non-DGA Gd triads (M31), Y present (M30).
    - Merge into the M33 supply ledger.
    - Rerun the POWER generator at the counted G for E-CT (+0.12) and window-sized effects.
  - **Phase B (only after figures are actually acquired).**
    - Curator-fidelity check: blind re-digitisation of the figures behind the V1 FIG cells (13 publications, 107 cells). Fit the slope of digitised deviations on V1 deviations, with a CI at figure level.
    - Two blind passes with a manual digitiser. The LLM is used only for triage.
    - Record the pub_cd72e4a7bb merge as a V2 sensitivity finding.
- **Kill criterion.**
  - Close the figure route for model confirmation if the projected new independent units fall below the number that gives 50 % power for the target estimand in the run-4 simulation (likely G < 20).
  - A Phase-B slope CI inside [0.85, 1.15] closes the attenuation reading.
- **Expected effect.** Phase A yield below 20 independent clusters is likely. Phase B would pass if run.
- **Cost.** Phase A is a day or two of desk work with web reading only. Phase B is data-gated.
- **Risks.** A separate agent must seal any extracted values before modellers see them.
- **Judges.** J1 3.5 · J2 3 · J3 3 (mean 3.17).
- **Prior.** A 0.08 · B 0.05 as written. Phase A yielding at least 20 independent units: about 0.1.

### 9. M18: Density-indexed metal ladder (H-BUDGET as a descriptive E-D row)

- **Hypothesis (narrowed).** How many metals are needed to reach 0.50 depends on read density and series completeness more than on the cohort cut. At saturation (every other measured metal read) the strata should not differ.
- **Why (evidence).**
  - C3 with k = min(K, kmax), K = 4: 0.496–0.497 with β refit, 3 of 5 seeds ≥ 0.50.
  - Fixed cohorts:
    - C4 at K = 4: 0.479, 0 of 5 seeds; even the in-sample best β only reaches 0.484.
    - C5 at K = 5: 0.522–0.527.
    - C6 at K = 5: 0.524–0.531; at K = 6: 0.552–0.556.
  - The 51 kmax = 3 cells (91 % in the giant cluster) score 0.597 against 0.427.
  - At matched density (k = kmax), by kmax stratum: 0.597 (kmax 3) / 0.447 (4) / 0.485 (5) / 0.552 (6).
  - Without C5-BTBP: C4 at K = 4 0.513 (4 of 5 seeds), so k* moves from 5 to 4.
  - Series-length strata at K = 3–6:
    - 14-metal series: 0.480 / 0.537 / 0.586 / 0.620;
    - 12–13-metal series: 0.322 / 0.374 / 0.417 / 0.439.
  - Refitting β per fold moves R² by at most 0.004, so gap G3 is immaterial.
  - Cluster-bootstrap P(R² ≥ 0.50): C3min K = 4 0.40–0.46; C4 K = 4 0.17; C5 K = 5 0.71–0.78; C6 K = 6 0.92–0.96.
- **Corrected design.**
  - Rows on identical queries at every rung: the population row (C3 min(K, kmax)) and fixed cohorts C4, C5, C6. DOPT_Q-nested reads.
  - β refit per fold and per k, with λ and β logged, plus a β = 0 row.
  - Series-length strata and a read-density index (reads per measured metal, longest hidden run, share outside the hull).
  - MSE and relative SSE reported next to R².
  - P(R² ≥ 0.50) with LOCO ranges, with and without C5-BTBP and leave-one-publication-out.
  - M17's policy envelope as labelled rows.
  - One live, pre-registered hypothesis: density invariance at saturation.
- **Kill criterion (density invariance).** Killed if any stratum with at least 3 clusters is below 0.48 at saturation, or if the spread between strata is ≥ 0.05. The conclusion would then be "series completeness and cluster drive the metal count". The current kmax-4 (0.447) and kmax-5 (0.485) values suggest a genuinely open outcome.
- **Expected effect.** k* between 4 and 6 depending on C5-BTBP. The useful statement is "every other metal, including both ends".
- **Cost.** Hours.
- **Risks.** The 0.50 target is arbitrary. Under ICP-MS, K is rarely the user's choice.
- **Judges.** J1 3 · J2 3 · J3 3 (mean 3.00).
- **Prior.** A 0.75 · B 0.85 for passing as written, which carries no information. A robust "6 metals" conclusion is about 0.3.

### 10. M14: Selective direction calls (E-DEC descriptive deliverable, B0 and B1), with M16's interval line

- **Hypothesis (narrowed).** From 4 reads, read-based interpolation can sign about half of the adjacent pairs at about 10 % false sign. The Gd term adds a few points.
- **Why (evidence).**
  - Coverage at α 0.10 depends on the procedure:

    | procedure | B1 coverage |
    |---|---|
    | idea as stated | 0.606 |
    | CRC | 0.519 |
    | plug-in | 0.507 |
    | valid fixed-sequence | 0.149 |

  - B0: 0.46–0.47. B1 − B0: +0.047 to +0.058 [0.021, 0.073].
  - LINE: 0.05 with a global threshold, but 0.333 with pair-type thresholds.
  - Net benefit B1 − B0: +0.010 to +0.017 at p_t 0.2–0.5.
  - False sign by pair type: Eu/Gd 0.18, Tm/Yb 0.29, Yb/Lu 0.62.
  - 57 % of wrong calls have |y| < 0.1.
  - Only 10 % of cells have every pair called.
  - From M16: publication weighting is just width inflation. A row-weighted Mondrian interval at nominal 93–95 % matches PM.
- **Corrected design.** A descriptive deliverable, never a decision slot:
  - risk–coverage curves and AURC for B0, B1 and HBK@ML;
  - a false sign counts only when |y| ≥ 0.10 (a three-way lighter / heavier / inseparable decision);
  - valid calibration (Learn-then-Test, or CRC on a monotone loss), with the plug-in shown next to it;
  - calibration leave-one-cluster-out;
  - per-pair-type validity as the primary report: no calls outside the read hull, and no Tm/Yb or Yb/Lu calls without bracketing reads;
  - measurements saved counted in experiments;
  - net benefit against B0, HBK@ML, RIDGE2 and a Platt-scaled LINE, with calibration-in-the-large per cluster;
  - one interval line: the row-weighted Mondrian level chosen by nested LOCO (about 95 % nominal for about 90 % on new sources, roughly ×/÷ 2.3).
- **Kill criterion.** None (descriptive). A new-source evaluation needs at least 5 sources and at least 100 calls.
- **Expected effect.** About half the pairs called at about 10 % false sign (about 5 % under the |y| ≥ 0.1 rule).
- **Cost.** Hours.
- **Risks.** Endpoint switching. It must never be presented as an R² gain.
- **Judges.** J1 2.5 · J2 4 · J3 2.5 (mean 3.00).
- **Prior.** A 0.20 · B 0.35 as written.

### 11. M17: Read-policy envelope (H-READ descriptive rows)

- **Hypothesis (narrowed).** With 4 reads, read placement mostly decides which block of the series is predicted well. No label-free policy Pareto-dominates DOPT_Q. The policy alone moves pooled R² by up to about 0.02.
- **Why (evidence).**
  - BISECT with a radius midpoint reproduces: pooled 0.462; Yb/Lu 0.002 → 0.488; Ce/Pr 0.419 → 0.274; Lu read in 73 % of cells against 14 %.
  - With β refit: 0.4577 against 0.4505.
  - HULL-DOPT reads Lu just as often but recovers only 40 % of the heavy-end SSE gain.
  - In SSE units, BISECT moves −0.027 (light), +0.020 (mid) and +0.014 (heavy).
  - Five reasonable variants of the rule span pooled +0.003 to +0.025 and heavy-block −0.017 to +0.121.
  - LIGHT-GUARD with a position midpoint scores 0.476.
  - The pooled gain comes from the kmax = 3 cells (+0.052); elsewhere it is +0.001.
- **Corrected design.**
  - Freeze a written policy family first: midpoint coordinate, tie-break, guard step, hull rule.
  - Score every member on identical queries at k = 3 (and k = 2–5 inside M18).
  - Report the min–max envelope as per-block ΔSSE in pooled-R² units, plus per-pair MAE against FLAT.
  - With and without C5-BTBP; split by kmax and by cluster.
  - Pre-declared Pareto question: does any member keep every block above −0.002 and improve one block with a cluster lower bound > 0?
- **Kill criterion.** The Pareto claim fails if no member dominates, which is expected. The envelope must then be carried by every H-BUDGET claim.
- **Expected effect.** A registered negative: "at 4 reads, placement is a choice of block". Envelope about 0.02.
- **Cost.** Minutes.
- **Risks.** Post hoc rules. The heavy block has Kish 1.87.
- **Judges.** J1 2.5 · J2 2.5 · J3 2.5 (mean 2.50).
- **Prior.** A 0.60 · B 0.60 for passing as written, which carries no information.

### 12. M13: Per-cut wider-pair table (E-W descriptive appendix)

- **Hypothesis (narrowed).** Cut-point SFs (ΔZ = 2–4, including Nd/Sm) are predicted well from the same reads. The open question is whether interpolation beats a slope at Nd/Sm.
- **Why (evidence).**
  - ΔZ = 2: B1 0.676 (B0 0.659, LINE 0.447). Nd/Sm: 0.765 (LINE 0.681). ΔZ = 3–4: 0.811.
  - Macro average for ΔZ = 2: 0.521–0.529, or 0.495 without Nd/Sm or La/Pr.
  - 5,502 of 6,114 ΔZ = 2 rows are exact sums of two strict-adjacent rows.
  - var(y) is 2.2× larger while MSE is only 1.30× larger, so much of the higher R² comes from the denominator.
  - Skill against LINE is 0.28 on strict pairs and 0.26 on Nd/Sm.
  - Nd/Sm B1 − LINE: +0.084 [−0.075, +0.243], and +0.011 to +0.013 outside the giant cluster.
  - Heavy ΔZ = 2 cuts are near 0: Ho/Tm 0.16, Er/Yb 0.07, Tm/Lu 0.10.
- **Corrected design.**
  - A zero-cost table, labelled "seen", one row per cut pair:
    - skill against LINE and against B0;
    - the MSE and var(y) multipliers;
    - MAE relative to median |y|;
    - sign agreement in |y| bins, with zero predictions counted;
    - cluster coverage.
  - One pre-declared open contrast: Nd/Sm B1 − LINE, equal-weighted over clusters with ≥ 10 rows per seed and with the giant cluster removed.
  - Heavy cuts reported as explicit non-claims, linked to Tm/Lu reads.
  - A feed-defined pair set: Nd/Sm; Nd/Dy, Pr/Dy and Nd/Tb for NdFeB (ΔZ 5–6); Eu/Tb.
  - Never pooled with the strict pairs; "strict + Nd/Sm" (0.551) appears only as a labelled substitution-trap row.
- **Kill criterion.** "Interpolation beats slope at the light|middle cut" is killed if the Nd/Sm contrast is ≤ +0.05 or its lower bound is ≤ 0. That is the expected outcome: narrowed to "a line through 4 reads suffices at Nd/Sm outside DGA".
- **Expected effect.** Descriptive. Middle cuts about 0.63–0.69; heavy cuts about 0.
- **Cost.** Minutes.
- **Risks.** Readers may take it as a model improvement.
- **Judges.** J1 2 · J2 3 · J3 2 (mean 2.33).
- **Prior.** A 0.80 · B 0.65 for passing as written, which carries no information.

---

## Tier 3: data-gated (requests to the user)

### Consolidated requests to the user

| # | request | serves | effort for the user |
|---|---|---|---|
| U1 | Say whether you have institutional journal access. Allow one download: the Shimada 2004 thesis (open PDF, about 24 MB). Look up the C5-BTBP sources (Nilsson 2006, Foreman 2005: Experimental section, detection limit, attribution, equilibration time) | M25 (also M24 per-source limit, M23, the M26 V3 rung) | about an hour |
| U2 | Export from SAFE only the archive's 1,602 missing record IDs (11 runs: 7,553–8,716; 14,461–14,665; 1,343–1,482; 1,032–1,105; 1,488–1,497; 1,326–1,329; singletons 16,084, 17,113, 17,428, 17,743, 18,058) plus IDs above 18,372. Check SAFE's terms of use (safedb@lanl.gov; /terms returned 404) | M33 (gate for M05, M31, M37) | minutes |
| U3 | Allow fetching the SI of Molecules 29:3548 (Table S3; 8 diluents) as the first external condition-transfer case | M05 | minutes |
| U4 | If a laboratory collaborator exists: label-blind re-measurement of 8–12 C3 conditions as published, in triplicate, with in-window, high-D and floor arms. Optionally a reproducibility arm with ≥ 8 laboratories, an in-run TODGA reference beside any non-DGA ligand, and Y recorded | M11 (+ M12, M31, M30, M10(c), M27 replicates) | months; external |

### 13. M25: Stakes-first source recovery (an accounting audit, not a hypothesis test)

- **Hypothesis (narrowed).** A few documents decide whether about 28 % of B1's SSE and about half of its gain rest on verifiable values. The import cells are SF-faithful; the excess error sits in floor and attribution problems.
- **Why (evidence).**
  - No-location C3 cells: 79 cells in 25 publications, 38.9 % of rows, 57.7 % of SSE, 79.1 % of the gain.
  - The 54 import-only cells (22 publications, 12 clusters) are not the problem: B1 R² 0.487, B1 − B0 +0.032, positive in 11 of 12 clusters.
  - The excess sits in the 25 curated cells with no location: MSE 0.069, 45.8 % of the gain, mostly C5-BTBP.
  - The six-DOI dual-transcription idea matches 0 or 1 record under the stated rule.
  - The ORNL "decade slips" are per-condition constant offsets (4 / 3 / 2 / 1 / 0 decades) that cancel in log SF. The only gross SF-level disagreement is the known DMDPhPDA 5 M Pr/Nd row.
  - The located table cells give B1 − B0 +0.053.
  - Rows with a 1-significant-figure D have MSE 0.038 against 0.024.
  - The open SKB R-05-13 report (same group as C5-BTBP) says D < 1e-4 means no extraction, and that 5 minutes of shaking did not reach equilibrium in one system.
- **Corrected design.**
  - Ask once about access (U1). Use open routes first: the SKB reports R-04-22, R-05-13 and R-06-45, and the Shimada thesis.
  - Stakes-first order: C5-BTBP, then Shimada, then the documents ranked by the union of SSE share and |gain| share. The ranking uses a model-agnostic quantity (the maximum SSE over B0, B1 and HIST_D7). It includes pub_f8a621ede4 and the located DMDPhPDA copy A.
  - Census of the 22 import-only publications in seeded random order, publication as the unit, replacing the 15-cell sample.
  - Error classes in target currency:
    - gross = |Δ adjacent log SF| > 0.3;
    - whole-cell level offsets reported separately;
    - SOURCE-RESOLUTION (interval targets);
    - MODEL-DERIVED / GRID;
    - NON-EQUILIBRIUM;
    - MISATTRIBUTED;
    - UNLOCATABLE.
  - Blinding: auditors see the DOI, conditions and metal list only. A script does the comparison. Access status is recorded in advance.
  - Versions:
    - V3-R holds corrections from the random census only;
    - V3-HS holds value-informed corrections and is an upper bound, never a model-selection benchmark;
    - scoring is on byte-identical queries (V1 read sets frozen);
    - censored targets enter only through the M23 interval line.
  - Report the per-document Gd contrast (B0 residual on Eu/Gd minus Gd/Tb) against the located-table reference and the fold β.
- **Decision rules.**
  - "Anchored" if the V3 gain on the resolved documents is ≥ 50 % of their V1 gain.
  - Import provenance is retired as a lever if the extrapolated gross-SF-error SSE share has an upper bound < 2 % of B1's SSE.
  - If open routes plus the user's access reach < 40 % of the gain, report an "unverifiable" stratum at once.
  - Transfer doubt is explicitly out of scope.
- **Expected effect.** The C5-BTBP light-end values become floor-level interval targets, which re-levels B1 by up to +0.03. Imports are SF-faithful. The Gd term is anchored on the resolved documents.
- **Cost.** Data-gated: user access and auditor time.
- **Risks.** Value-informed selection must stay labelled. LLM digitisation of figures needs the M22 gate.
- **Judges.** J1 5 · J2 5 · J3 3.5 (mean 4.50).
- **Prior.** A 0.25 · B 0.25. About 0.5 that enough gain-carrying documents are resolved; about 0.85 "anchored" given resolution.

### 14. M33: SAFE gap-ID audit, a label-blind supply gate

- **Hypothesis (corrected).** The SAFE records missing from the archive are mostly not new experimental Ln series. A metadata-only audit of those IDs settles whether a SAFE route to independent confirmation exists.
- **Why (evidence).**
  - **This note's exploratory check** (metadata only, no D column read): master_clean's source_record_id runs from 1 to 18,372, with 1,602 IDs missing in 11 runs. This reproduces Skeptic A.
  - The SAFE homepage counts track the highest ID: the 8,716 snapshot equals the end of the largest gap; 18,367 on 2026-09-07; 18,388 on 2026-10-22.
  - So the "+1,618 newer records" are mostly records that pre-date the archive's last batch, plausibly computational or chromatography entries. At most about 16–21 records are genuinely newer.
  - Curator batches differ in yield: Clark 4 of 42 Ln DOIs with a ≥ 5-Ln condition; Summers 24 of 34; Zhang 25 of 41; Augustine (high-throughput) no Ln.
  - At G = 10, an equal-weight B1 level has a 95 % range of about [0.21, 0.74]. The "upper bound < 0.40" kill has only 0.14–0.27 power against a collapse. A unit-level Gd-dip transport test (between-unit sd about 0.1, SE about 0.03) is decisive.
- **Corrected design.**
  - The user exports only the gap IDs and the IDs above 18,372 (U2).
  - The audit script reads whitelisted metadata columns only and asserts that the D columns are absent.
  - Step 1: record type (experimental extraction, computational, chromatography). Chromatography k' stays out of the extraction set.
  - Then, per record: DOI novelty; conditions with ≥ 5 Ln and kmax ≥ 3; Eu, Gd, Tb present; chemotype via the frozen map; laboratory identity from author sets under a frozen rule; tags for Y, rigid SMARTS and non-DGA.
  - Merge with the M22 Phase-A count into one supply ledger.
  - If ≥ 10 independent laboratory × chemotype units exist, freeze everything (DOPT_Q parameters fitted on all of C3, a fixed β, a reads-only seeded condition rule, the censoring rule) and score once:
    - primary: unit-level Gd-dip transport (t-interval against 0 and against −0.094; DGA and non-DGA separately);
    - frozen-interval conformal coverage;
    - the level, against a like-for-like C3 reference distribution, never quoted as reaching 0.50.
- **Kill criterion.**
  - Fewer than 3 experimental multi-Ln units from laboratories outside the corpus: stop the SAFE route and recommend laboratory campaigns only.
  - 3–9 units: descriptive only.
  - At least 10: the Gd-dip test decides.
  - "B1-specific failure" only if B1 − B0 falls below the C3 reference's 5th percentile while B0 stays in range.
- **Expected effect.** SAFE most likely yields 0–3 units, so the acquisition branch mostly closes.
- **Cost.** Minutes for the user, hours for the audit.
- **Risks.** SAFE's terms of use. Metal presence is weakly label-informative. Unit identity must be frozen before any value is read.
- **Judges.** J1 4 · J2 4 · J3 3 (mean 3.67).
- **Prior.** A 0.05 · B 0.20. The audit itself is a useful go/no-go with probability about 0.9.

### 15. M11: Label-blind re-measurement of C3 conditions (H-CEILING redesign), with the M12 laboratory arm

- **Hypothesis (redesigned).** The corpus label error of adjacent log SF can be measured directly. Doing so decides whether 0.50 at k = 3 is limited by measurement or by modelling (gap G1).
- **Why (evidence).**
  - The replicate-versus-add-a-metal question is solved by algebra now: ΔMSE = 0.216 s² at k = 3 under DOPT_Q (0.289 on one-end-read rows, 0.067 on both-hidden rows).
  - The cost-matched break-even is 0.20–0.31 for one replicated read against one added metal. So never duplicate in-window reads; add metals.
  - The noise shape is irrelevant for a linear predictor.
  - One modern laboratory's repeatability only bounds corpus label error from below. A core in-window σ_SF ≤ 0.10 is about 0.8 likely, so the original design would not decide G1.
- **Corrected design.**
  - Primary: re-measure 8–12 C3 cells at their published conditions, 3 independent runs plus an analytical duplicate.
    - Cells are sampled label-blind, stratified by chemotype and by the read-predicted window status, never by B1 residual.
    - Estimand: per pair, var(corpus − fresh mean) − σ_rep²/n, plus B1 R² on identical cells scored against replicate means and against corpus labels.
  - Level arms: in-window, high (+2.5 to +3), and floor (about −4 by the corpus route). Direct organic-phase assay, spike recovery, per-metal LOQ, isotopes and interference corrections.
  - The common-mode analytical share (which cancels in SF) is a primary variance component.
  - Gross error pre-declared as > 4 × MAD. System-level tests with the MDE stated.
  - Extra arms:
    - a ≥ 8-laboratory own-protocol reproducibility arm (M12; ISO 5725-2; Mandel h/k);
    - a TODGA reference in the same ICP run beside any non-DGA ligand (the M31 salvage, with a curvature-adjusted Gd dip);
    - Y recorded (M30).
  - Freeze everything before data.
- **Decision rules (one-sided).**
  - If corpus label error stays below the threshold (σ_SF about 0.157–0.19 from the analytic floor algebra) and B1 R² is unchanged against replicate means, the shortfall is a modelling or curation problem.
  - If the repeatability lower bound is at or above the threshold, the problem is noise.
  - A repeatability ceiling is never quoted as the corpus ceiling.
- **Expected effect.** Repeatability σ_SF about 0.02–0.09. Corpus label error larger and concentrated at the floor and light end.
- **Cost.** Data-gated: needs a laboratory, and the user has none.
- **Risks.** Published conditions may not be reproducible. Collaborator recruitment.
- **Judges.** J1 4.5 · J2 4 · J3 2.5 (mean 3.67).
- **Prior.** A 0.30 · B 0.30 as written. The redesign has high value if it ever runs.

### 16. M05: External condition transfer (prospective leg of M01)

- **Hypothesis.** The frozen E-CT rule improves on B1 at new conditions in laboratories outside the DGA cluster.
- **Why (evidence).**
  - The original single-anchor design scores the wrong predictor: the corpus dose-response with random 1 / 2 / 3 / all siblings is −0.30 / 0.00 / +0.13 / +0.27 relative SSE.
  - The losing non-DGA publications all have n_sib = 1. The only non-DGA gainers are the provenance-pending C5-BTBP and DMDPhPDA.
  - The corpus verdict depends on the rule: raw single anchor median −0.48 (3 of 6 positive); shrunk median +0.10 (5 of 6).
  - Supply is about 3–5 papers from about 3 laboratories, one of them the Moscow phenanthroline-diamide group (which overlaps sc086 in the corpus).
  - A 6-unit sign test kills a true positive share of 0.7 with probability 0.26.
- **Corrected design.**
  - The frozen M01 SH rule with K frozen. All other conditions of the extractant in the publication serve as siblings, leave-one-condition-out.
  - n_sib strata, with n_sib ≥ 3 primary.
  - "In-window" defined label-free and co-primary with all rows.
  - Laboratory (author set) as the unit; the Moscow papers count as one unit; corpus overlap flagged.
  - Blend controls at the same w. The claim statistic is SH minus the best control.
  - Strata by perturbation type: pH scans of acidic extractants are near-copies by construction.
  - Figure sources only after the M22 gate.
  - A metadata-only supply audit before anything is extracted. First case: Molecules 29:3548 (U3).
- **Kill criterion.**
  - Fewer than 5 independent laboratories with n_sib ≥ 3: a descriptive case series, with no kill or promote language.
  - Otherwise, a laboratory-unit median (SH − best control) ≤ 0 kills transfer beyond DGA.
  - "DGA-like" only if the median is ≥ 0.30, with thresholds set from the corpus rehearsal before data.
- **Expected effect.** Most likely descriptive. If anything, about +0.1 relative SSE in window.
- **Cost.** Days, plus the user's permission for downloads.
- **Risks.** Author overlap. Digitisation. Ionic-liquid and octanol diluents are mechanism changes.
- **Judges.** J1 4 · J2 3.5 · J3 2 (mean 3.17).
- **Prior.** A 0.12 · B 0.15.

### 17. M12: Inter-laboratory reproducibility (Stage-0 lead now; laboratory arm through M11)

- **Hypothesis.** Adjacent log SF carries a laboratory signature, concentrated at the light end.
- **Why (evidence).**
  - The corpus holds exactly one genuine matched-condition cross-lab pair: ORNL 2020 (Inorg. Chem.) against Ames/Purdue 2024 (Green Chem.), 3 DGAs, 34–37 adjacent comparisons.
  - Cross-lab mean squared difference 0.027–0.029, about B1's MSE. That is about 30× Ames' own method-to-method SF variance for TODGA (0.0010).
  - It is concentrated in La/Ce, Ce/Pr and Pr/Nd (a Pr spike). Without those pairs it is 0.0043.
  - The other "matched pairs" are JAEA self-re-reports.
  - τ² cannot be identified from literature means, and a 4-laboratory ring test has too few degrees of freedom.
- **Corrected design.**
  - Report Stage 0 now as an exploratory lead: n = 1 laboratory pair, chosen after looking.
  - Laboratory defined by author overlap.
  - A frozen descriptive literature frame (TODGA/TEHDGA, 3 M HNO3, tables first, a 40-paper budget), declared "uninformative" in advance if fewer than 6 laboratories are found.
  - The ring test moves into M11 as a ≥ 8-laboratory own-protocol arm, with V_x/V_w as the estimand.
  - The light-metal fingerprint software test goes to the M01/M06 annex.
- **Decision rules.** Laboratory signature if V_x/V_w ≥ 5 with a lower bound ≥ 2. No signature if the upper bound ≤ 2. Anything else is inconclusive.
- **Expected effect.** A light-end laboratory signature.
- **Cost.** Stage 0 takes hours; the laboratory arm needs collaborators.
- **Practical reading.** How far to trust another laboratory's adjacent SF: at the light end, perhaps only to about 0.15 log.
- **Judges.** J1 3 · J2 3 · J3 1.5 (mean 2.50).
- **Prior.** A 0.12 · B 0.10.

---

## Rejected (fatal flaw, and where any useful fragment goes)

Entries are compact. The original designs are in `merged_ideas.json`, and the full verdicts are condensed in `critiques.json`.

### 18. M29: Aqueous-complexant rule (Gd term off in holdback media)
- **Hypothesis.** An aqueous complexant cancels or inverts the Gd dip, so B1's Gd term has the wrong sign in those cells.
- **Fatal flaw.**
  - The idea's own drop-one kill fires: pub_1605e436d7 (Sasaki 2004, TODGA/DTPA) carries 78.5 % of the +0.0054 gain.
  - All the Gd-chord evidence comes from that one JAEA paper.
  - 62 % of its gain sits in V2-flagged or floor-like cells.
  - The observed shift (+0.18) is about twice the tabulated DTPA chord (+0.09).
- **Salvage.** Adopt the data-contract rule now, unscored:
  - record complexant identity, concentration and pH;
  - define the Gd term only for nitrate and inert-salt media;
  - treat complexant cells as a separate stratum.

  A quantitative external paired test (complexed fraction × the ligand's Gd chord; ≥ 8 pairs from ≥ 3 laboratories outside JAEA) is logged in the supply ledger.
- **Cost (original).** Hours. **Judges** 3 · 4 · 1.5 (2.83). **Prior** A 0.08 · B 0.04. The chemistry is about 0.6–0.7 likely to be right for strong aminopolycarboxylate holdbacks.

### 19. M38: Pre-slot decidability gate
- **Hypothesis.** Four checks blind to the outcome would have retired every round-4 loser and passed B1 − B0.
- **Fatal flaw.**
  - The checks use labels from the same 229 cells.
  - Under a consistent pre-declared test (IM-t, df 12, 80 % power), B1 − B0 has MDE/effect 1.02 (one-sided) or 1.18 (two-sided), so the gate retires the only positive. Round 4's mde_80 is 0.044 against an effect of 0.034.
  - Check (a) depends on how the class is drawn: the Gd signal as a one-parameter class gives +3.5 % SSE; as pair-type constants it gives −1.45 %.
  - Check (d) confuses sample size with composition: the giant cluster is 95 % DGA, the rest 4.5 %.
- **Salvage.** A retire-only, label-free support gate plus 50 % power inside M35. Levels reported with intervals.
- **Judges** 2 · 3 · 2.5 (2.50). **Prior** A 0.30 · B 0.20.

### 20. M31: Non-DGA Gd-triad acquisition sized from the between-unit spread
- **Hypothesis.** In-window non-DGA units show a Gd dip of −0.12 or deeper, against −0.04 for DGA.
- **Fatal flaw.**
  - The kill rule fires 55–60 % of the time even when the hypothesis is true; the sign-leg win rate is only 0.56–0.57.
  - "Non-DGA" is a mixture: phenanthroline / pyridine-dicarboxamide −0.077 (12 units); malonamide −0.175; triazinyl about −0.26.
  - The DGA comparator rests on one cell (+0.663). Without it the DGA dip is −0.068, about HIST_D7.
  - Only 4 of the 29 corpus units meet the ≥ 2-condition rule.
  - Supply of 16–20 laboratories is unreachable.
  - The null (−0.072) is a strawman against B1's −0.094.
- **Salvage.** A family-stratified, curvature-adjusted Gd dip with an in-run TODGA reference, as an arm of M11. Family predictions frozen now. Tags in the M33/M22 audit.
- **Judges** 3 · 3 · 1.5 (2.50). **Prior** A 0.12 · B 0.10.

### 21. M06: S0–S4 value-of-information ladder
- **Hypothesis.** Information beyond B1 is laboratory-local and extractant-specific. The same-publication other-extractant channel is homologue chemistry, not an analytical fingerprint.
- **Fatal flaw.**
  - The S1/S2 kill cannot fire: S2 without C5-BTBP is one cluster (675 of 685 rows in the giant cluster).
  - The S3 thresholds sit at seen values (10 of 17 = 59 % against a 60 % bar).
  - Level offsets and pair offsets are reparameterisations of each other (12 free parameters each).
  - The homologue advantage is one cluster, and its sign flips outside that cluster.
  - The isotope refit has zero eligible rows.
  - S4 duplicates M01.
- **Salvage.** A descriptive annex to M01 (same campaign against other publication). The homologue channel recorded as a frozen lead with an external test only.
- **Judges** 2.5 · 2.5 · 2 (2.33). **Prior** A 0.20 · B 0.25.

### 22. M27: Ce reliability first, registered null, discriminating experiment
- **Hypothesis.** The Ce deviation is a sporadic, unreliable excess, so no Ce term helps.
- **Fatal flaw.**
  - The ICC gate cannot be powered. With k = 2 the CI is [−0.71, 0.75] at 6 cells; about 35–60 true replicate cells are needed and 9–15 exist (mostly transcription doubles or merges).
  - The Ce error is mainly a floor phenomenon:
    - 59 % of Ce-pair SSE is out of window;
    - 43 % is C5-BTBP;
    - 61 % of the squared Ce deviation sits in the 23 cells below −2.
  - Without the floor cells, the by-publication ICC is 0.26.
  - Ce(IV) is implausible in HNO3, and the HCl excess points the wrong way.
- **Salvage.**
  - The floor stratum goes to M25 and H-FLOOR.
  - The Ce null goes into the closure ledger.
  - Ce-pair interval widening goes into M14.
  - The replicate ICC becomes a data request in M11.
  - Any Ce-vs-level correlation must use a level that excludes La, Ce and Pr; otherwise the correlation is mechanical.
- **Judges** 2.5 · 3 · 1.5 (2.33). **Prior** A 0.07 · B 0.12.

### 23. M39: Nested re-estimation of the five post hoc leads
- **Hypothesis.** Nesting each analyst's selection step prices its optimism.
- **Fatal flaw.**
  - Publication offsets are a laboratory-known estimand: BPG removes the test publication.
  - Window nesting just follows the coverage floor (the narrowest window in 23–25 of 25 fold-fits) and understates the optimism. Nesting over the family of scores gives 0.516.
  - The "chosen in ≥ 50 % of folds" rule measures how flat the optimum is:
    - K = 10 is chosen in 1 of 25 folds although its nested value (0.526) beats the fixed value (0.524);
    - the edge rule nests to −0.0009 against a stable fixed +0.0058.
- **Salvage.** An optimism table inside M09 (family nesting) and M35.
- **Judges** 2 · 3 · 2 (2.33). **Prior** A 0.30 · B 0.30.

### 24. M02: Two reads plus a sibling; shift-plus-tilt deformation
- **Hypothesis.** A sibling plus 2 reads matches B1 with 4 reads; condition change is shift + tilt.
- **Fatal flaw.**
  - C5-BTBP carries 73 % of the gain. Without it, the non-dominant stratum scores −0.127 to −0.18.
  - At n_sib = 1: SIB@2 0.42 against B1@4 0.54–0.55.
  - The global shift + tilt (LIN2, 0.419) about equals the copy (0.411), and the oracle shift + tilt (0.482) is below SIB@2 (0.490), so the mechanism is refuted.
  - The LFER control is a strawman (slope IQR 0.71–1.35).
  - Unit-balanced, the method is 11.9 % worse than B1@4.
- **Salvage.** One read-ladder row in M01 (SIB@2 against SIB@4 and COPY).
- **Judges** 2 · 2.5 · 2 (2.17). **Prior** A 0.15 · B 0.15.

### 25. M04: Choose conditions, not metals
- **Hypothesis.** Taking full series at screen-level spread conditions beats random choice.
- **Fatal flaw.**
  - 16 families in 3 clusters (14 in the giant cluster).
  - C5-BTBP floor references give 75–84 % of the advantage.
  - Against random-in-window or IQR-random the gain is +0.02 to +0.045, with CIs that include 0.
  - Rows differ by policy (B1 fallback 7 % against 11 %).
  - The cost-matched spread of reads nearly ties (0.676 against 0.655).
  - Under ICP-MS a screen costs as much as a full series.
- **Salvage.** One line in M01: "avoid floor or saturated reference conditions; average at least 2".
- **Judges** 2 · 2.5 · 2 (2.17). **Prior** A 0.30 · B 0.55 for passing its own kill; the decisive version is about 0.15–0.2.

### 26. M08: Variance budget and futility rule
- **Hypothesis.** The transferable share of B1's residual is below 8.8 % of SSE, so 0.50 is futile for cross-source models.
- **Fatal flaw.**
  - T is a BPG oracle and is not identified: one cluster holds 97.8 % of the identifying pairs.
  - The tail comes from the C5-BTBP 15-versus-1 pair, a possible duplicate attribution.
  - Cluster bootstrap for T: [0.003, 0.346].
  - The sibling dose-response bounds the wrong side.
  - The cross-cluster share (−0.13 %) is already known to be about 0.
- **Salvage.** A one-hour relation-class moment table with an identifiability column, in the M35 appendix.
- **Judges** 2.5 · 2 · 2 (2.17). **Prior** A 0.20 · B 0.20.

### 27. M15: Screening estimand with a slope-only control
- **Hypothesis.** Ranking skill from partial series is mostly amplitude; LINE recovers ≥ 80 % of it.
- **Fatal flaw.**
  - The thresholds are post hoc (80 % against an observed 81.7 %).
  - The kill structure contradicts itself: +0.032 falls between the 0.02 and 0.035 bars.
  - "Screening = amplitude" is already contradicted: LINE's share is 61 % on Spearman, 67 % on top-3, 66 % within publication.
  - The cluster MDE is 0.066 against an expected 0.03.
  - gen16 L3c already settled model against line.
  - B1 already sits at the amplitude-oracle ceiling.
- **Salvage.** One descriptive E-RANK table (with endpoint-wise LINE shares) in the reporting appendix.
- **Judges** 2 · 2.5 · 2 (2.17). **Prior** A 0.25 · B 0.35.

### 28. M36: Publication versus cluster confirmation units
- **Hypothesis.** Publication-level tests stay valid with chemotype stratification at the measured ICC.
- **Fatal flaw.**
  - Stratified publication flips are identical to unstratified flips, and they are anti-conservative under shared layouts (0.07–0.18).
  - Only aggregation to chemotype or laboratory units holds size (0.041–0.053).
  - The independent-laboratory kill is tautological.
  - The ICC is unidentified (B1 − B0 0.11, CI [−0.11, 0.81]).
  - The "10–20 versus 40" dichotomy confuses the unit with the test.
- **Salvage.** The aggregation rule (one value per laboratory × chemotype) goes into M34. One sentence goes into the data request.
- **Judges** 2 · 2.5 · 2 (2.17). **Prior** A 0.25 · B 0.25.

### 29. M16: Publication-balanced conformal intervals
- **Hypothesis.** Equal-publication calibration gives near-nominal coverage per cluster at ≤ 35 % extra width.
- **Fatal flaw.**
  - Its own Winkler leg (+12 %) and certified-direction leg (0.913–0.923 < 0.93) fire.
  - A width-matched row-weighted interval (nominal 0.932) matches PM.
  - The minimum-cluster statistic is decided by a 2-cell cluster, where one row moves coverage by 0.08.
  - "Certified inseparable" is impossible, because every half-width is ≥ 0.15.
- **Salvage.** One interval line in M14.
- **Judges** 1.5 · 3 · 1.5 (2.00). **Prior** A 0.05 · B 0.07.

### 30. M23: E-CENS, two-sided interval scoring at limits
- **Hypothesis.** Scoring pairs at measurement limits as intervals re-levels R² without re-ranking arms.
- **Fatal flaw.**
  - At F_lo = −4, 42 of the 46 censored rows per seed are C5-BTBP, and R²_int 0.487 equals the C5-BTBP exclusion (0.486).
  - F_hi = 3 has no anchor (80 % of its effect comes from two cells).
  - The score can be gamed: a trivial B1 ± 20 transform gains +0.015.
  - The kill legs were already passed when written.
  - It reintroduces censoring that G-V3 refused.
- **Salvage.** One sensitivity line, labelled "C5-BTBP floor sensitivity". Deployable one-sided bounds for pairs whose read is at or below a user-stated limit, in M14.
- **Judges** 2 · 2 · 1.5 (1.83). **Prior** A 0.35 · B 0.80; the mechanical claims were already computed.

### 31. M26: Nested data-version ladder and fragility flag
- **Hypothesis.** A ladder of data versions with a binding fragility flag makes conclusions robust to contested data.
- **Fatal flaw.**
  - The premise number is wrong: V2 moved PWL_R by +0.006 and R23 by +0.011, not 0.06.
  - V2-S is null (B1 0.452 → 0.453); every arm's shift lies inside the random-removal null.
  - V2-5's 16 pairs per seed (about 1 % of rows, 4.1 % of SSE, selected by label) carry all of V2's move.
  - S01's sign flips in 29–30 % of random 20-cell removals, so the flag fires on noise.
- **Salvage.** A two-line closure:
  - D-7 is closed: V2-5 explains V2, and V2-S is null;
  - robust losses add nothing (Huber β −0.1167 against least squares −0.1147; R19 +0.002).
- **Judges** 1.5 · 2 · 2 (1.83). **Prior** A 0.12 · B 0.15.

### 32. M37: Sequential e-process confirmation ledger
- **Hypothesis.** An e-process on per-source wins reaches 70–90 % power within 20–30 new sources.
- **Fatal flaw.**
  - Supply is a stock of 3–7, not a flow.
  - With λ = 0.6 the process needs 7 straight wins, so power is 0 at G ≤ 6.
  - Futility takes a median of 68 sources (π 0.5) to 168 (π 0.55), not 15.
  - No finalist beats B1: the best publication win rate is 0.58.
  - The magnitude e-guard has about 0 power.
  - Retroactive merging breaks the filtration.
- **Salvage.** A zero-cost B1-versus-B0 ledger (π0 0.6, mixture bet, separate futility e-process) inside M34, opened only behind the M33 gate.
- **Judges** 2 · 2 · 1.5 (1.83). **Prior** A 0.20 · B 0.15.

### 33. M10: Measure in the window (sibling reproducibility and a condition-choice rule)
- **Hypothesis.** Adjacent-SF irreproducibility concentrates below log D −3; reads-only flags form a domain.
- **Fatal flaw.**
  - The tail below −3 is one publication: C5-BTBP supplies 783 of 891 rows. Without it the ratio is 1.08.
  - The idea's own publication-share leg fails (1 of 2 qualifying publications).
  - Siblings are not replicates: rms rises from 0.140 to 0.242 with the level offset between them.
  - The reads-only flags gain only +0.022 over the cell control and are dominated by M09 and by "drop C5-BTBP" (0.486).
- **Salvage.** The flags become rows in M09. The low-D arm goes to M11. The floor claim goes to M25.
- **Judges** 1.5 · 2 · 1.5 (1.67). **Prior** A 0.07 · B 0.12.

### 34. M19: Allocate extra reads between cells
- **Hypothesis.** Giving the 5th read to cells with the largest predicted SF mass beats random and size-only allocation.
- **Fatal flaw.**
  - On the registered primary (DOPT_Q-nested), score − size is +0.0019 against a +0.002 kill, positive in only 2 of 5 seeds, and the cluster interval includes 0.
  - A geometric affected-query count beats the score (0.4851 against 0.4833).
  - Cluster-balanced, the score (0.4664) ties random (0.4650) and loses to size (0.4801).
  - Its regret is worse than size-only.
- **Salvage.** One line: "give extra reads to the extractants with the most unmeasured metals".
- **Judges** 1.5 · 2 · 1.5 (1.67). **Prior** A 0.20 · B 0.22.

### 35. M20: Unmeasurable-end (Pm) scenario
- **Hypothesis.** Radius interpolation predicts an unmeasurable metal's two adjacent SFs at R² about 0.55.
- **Fatal flaw.**
  - Its own kill (a) fires: Eu's cluster lower bound is 0.22–0.35.
  - R² scales with the neighbour SF's variance (terciles 0.17 / 0.46 / 0.57), so it does not transfer.
  - Radius and Z interpolation differ by about 0.01 log at Pm (t = 0.533 against 0.5).
  - The tetrad test needs about 150 series.
  - External Pm series are HDEHP/LN-resin only, with no DGA.
- **Salvage.** A G8 scenario note: log SF(Nd/Pm) ≈ 0.53 × log SF(Nd/Sm). Bench |d| quantiles at 50 / 80 / 90 %: 0.037 / 0.104 / 0.188, using Nd and Sm as the gap-crossing analogues.
- **Judges** 1.5 · 2 · 1 (1.50). **Prior** A 0.10 · B 0.25.

### 36. M30: Yttrium as a 4f-free internal probe
- **Hypothesis.** Y's offset from the Ln radius curve indexes non-radius chemistry.
- **Fatal flaw.**
  - The archive has one TODGA dataset (with duplicated JAEA sets). Y_dev is a constant −0.28 to −0.35, so there is no meaningful zero; the idea's −0.01 does not reproduce.
  - A raw Y read hurts Ho-region pairs (Dy/Ho MSE 0.0099 → 0.0989).
  - The read-only endpoint is mechanical: a Ho analog shows partial ρ 0.64–0.70 with B1 MSE.
  - Power at 8 units is 0.01–0.06.
- **Salvage.** A "record Y" rider in M33 and M11. A separate Y/Ho estimand registered for the future.
- **Judges** 1.5 · 2 · 1 (1.50). **Prior** A 0.05 · B 0.03.

### 37. M32: Rigid / preorganised chemotype stress set
- **Hypothesis.** Rigid extractants with an interior extremum need at least 2 more metals than monotone chemistry.
- **Fatal flaw.**
  - The extremum penalty is one floor cell (c950dc, about 50 % of the subset's SSE). Without it: 0.468–0.513 against 0.479.
  - 6 of 17 extremum cells sit on round floor values.
  - Rigid SMARTS cells show no more extrema (8.6–11 % against 7–9 %), and without C5-BTBP they are easier (0.536).
  - Phenanthroline diamides are monotone.
  - The control is selected on the outcome.
  - With at most 6 legal reads, the "supported" branch can never be observed.
- **Salvage.** A rigid SMARTS tag in the M33/M22 audit.
- **Judges** 1.5 · 2 · 1 (1.50). **Prior** A 0.07 · B 0.08.

### 38. M21: Provenance class as a frozen reporting axis
- **Hypothesis.** The data_location class strongly modifies B1's difficulty and its Gd gain.
- **Fatal flaw.**
  - The idea's own kill is already met: within sc009, FIG − TAB is +0.0020, permutation p 0.12, CI includes 0. There are 3 TAB publications, and one of them carries 45 % of TAB.
  - The FIG sign flip is one aqueous-complexant paper.
  - The class is almost the same as cluster, fold and chemotype.
  - One ambiguous cell moves FIG R² from 0.490 to 0.511.
- **Salvage.** data_location as a descriptive column in the M35 tables, and as mandatory metadata for new sources.
- **Judges** 1 · 2 · 1 (1.33). **Prior** A 0.05 · B 0.06.

### 39. M28: B1+ factorial (Ce log-acid offset × damped heavy-end continuation)
- **Hypothesis.** A Ce acid term and an edge term each improve B1, additively.
- **Fatal flaw.**
  - The Ce-acid gain is C5-BTBP plus pub_1605e436d7. On other rows it is −0.0034 under BPG (0 of 5 seeds).
  - The within-publication acid permutation keeps 61–76 % of the gain, which triggers the idea's own ≥ 50 % kill.
  - The within-publication slope is −0.024 per decade (ρ −0.085).
  - Fold starvation is contradicted: the big fold gains, and the learning curve is flat.
  - The factorial interaction is zero by construction.
  - The edge term depends on the estimator (−0.0001 to +0.0055).
- **Salvage.** The edge continuation is adopted by argument as a labelled reference variant (B1 + edge), and later contrasts are scored against both. The Ce-acid term is a closed lead in the ledger.
- **Judges** 1.5 · 1.5 · 1 (1.33). **Prior** A 0.05 · B 0.04.

---

## Closure ledger (one line each, no slot; record in the round-5 decision log)

| line | closes | evidence |
|---|---|---|
| Floor-aware read clamps are retired as model claims on V1 at every F | H-FLOOR(a), M24 | 91 % of the gain is one cell; cross-fitted F 0.4098; the consistent clamp is negative |
| E-CENS at F_lo = −4 is the C5-BTBP exclusion | M23 | 0.487 against 0.486; F_hi has no anchor |
| V2-S is null; V2-5 carried all of round 3's V2 move | M26 (D-7) | B1 0.452 → 0.453; 16 pairs per seed, 4.1 % of SSE |
| Robust losses add nothing | M26 | Huber β −0.1167 against −0.1147; R19 +0.002 |
| The Ce-acid association is a two-publication effect | M28, M27 | within-publication slope −0.024 per decade; permutation keeps 61–76 % |
| The edge continuation is a reference variant, not a claim | M28, H-EDGE | +0.004 to +0.0055 cross-fitted, depending on the estimator |
| Low-D sibling irreproducibility is C5-BTBP | M10 | 88 % of rows below −3; ratio 1.08 without it |
| The provenance class is a column, not a lever | M21 | permutation p 0.12, 3 TAB publications |
| No data-contract Gd term in aqueous-complexant media | M29 | adopted as a rule; scoring killed by drop-one (78.5 %) |
| Cross-cluster residual share is about 0 | M08 | −0.13 % [−0.41, +1.37] |
| Publication-weighted conformal is width inflation | M16 | width-matched row-weighted interval at 0.932 is equivalent |
| Replicate a read only if per-metal σ > about 0.2; otherwise add a metal | M11 | ΔMSE = 0.216 s²; cost-matched break-even 0.20–0.31 |
| SAFE "+1,618 newer records" are mostly pre-existing gap IDs | M33 | 1,602 missing IDs in 11 runs; highest ID 18,372 (this note's check) |

## Merge map

| host | absorbs |
|---|---|
| M01 (E-CT) | M02 (read-ladder row), M03 (weighting factorial), M04 (reference-condition line), M06 (same-campaign vs other-publication annex), M05 (external leg) |
| M34 (inference replay) | M36 (unit aggregation), M37 (ledger behind the M33 gate) |
| M35 (decision convention) | M38 (retire-only gate), M21 (provenance column), M08 (moment table) |
| M09 (window axis) | M10 (reads-only flags), M39 (optimism ledger) |
| M18 (metal ladder) | M17 (policy envelope) |
| M14 (E-DEC) | M16 (interval line), M23 (deployable one-sided bounds) |
| M25 (source recovery) | M24 (per-source limit), M23 (censoring rule), M26 (V3 rung), M27 (floor stratum) |
| M33 + M22 (supply ledger) | M32 (rigid tag), M31 (non-DGA tag), M30 (Y tag) |
| M11 (re-measurement) | M12 (laboratory arm), M10(c) (low-D arm), M27 (replicate ICC), M31 (TODGA-paired triads), M30 (Y rider) |

---

## Exploratory check done for this note

To check the factual claim behind M33, one Python process read only the source_record_id column of `dataset_all_metals/clean/master_clean.parquet`; no D values were read.
- Results: 16,770 unique IDs, minimum 1, maximum 18,372, and 1,602 missing IDs in 11 runs.
- The runs are 7,553–8,716 (1,164), 14,461–14,665 (205), 1,343–1,482 (140), 1,032–1,105 (74), 1,488–1,497 (10) and 1,326–1,329 (4), plus singletons at 16,084, 17,113, 17,428, 17,743 and 18,058.
- This reproduces Skeptic A's finding. It is exploratory; nothing is sealed.

## Process note

- Only the two assigned files were written: this file and `critiques.json`. No existing file was modified, and no git command was run.
- All other numbers are quoted from the skeptics' and judges' verdicts, the idea texts and EVIDENCE_BASE.md. Where the two skeptics' reproductions differ, both values or a range are given.
- Several skeptics recorded their own process slips (temporary pickles and scripts in the session scratchpad, one read-only `git status`, reads of other agents' scratch files). None of them touched the repository.
