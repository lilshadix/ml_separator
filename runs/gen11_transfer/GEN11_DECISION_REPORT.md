# gen11 — decision report

*Rewritten 2026-09-04 after the pre-registered `matched` stage ran. Supersedes
`GEN11_DECISION_REPORT_SUPERSEDED_20260823.md`, which was rendered 2026-08-23T00:24 when 5 of the
34 pre-registered arms existed and which concluded "No auxiliary arm passes any §18 criterion".
That verdict was written before three of the arms it judged had been fitted (`arms/leaderboard.csv`
was rewritten at 13:35 the same day) and must not be quoted.*

**Headline: the stage is INCONCLUSIVE.** It neither established nor refuted the hypothesis it was
built to test, and the reason is a power failure that no amount of re-analysis fixes.

---

## 1. What has been run

| stage | arms | status |
|---|---|---|
| `primary` | 8 (4 controls + B, C, D, E) | complete, 5 split seeds |
| `matched` | 10 (F and G, 5 fixed draws each) | complete, 5 split seeds |
| `mechanisms`, `weighting`, `controls`, `policy` | 16 | **not run** |

26 of 34 pre-registered arms remain unrun, including both negative controls
(`PERMUTED_AUX_TARGET`, `SHUFFLED_METAL_LABELS`), which §19 requires to show no benefit before any
gain is believed.

Evaluation is gen10's, unchanged: cohort fingerprint `bed178ec1a7a82b0`, 5,248 rows, 152
extractants, 131 ECFP clusters (the macro unit), 79 Tanimoto chemotypes (the resampling block),
seeds 104729/130363/155921/196613/262147, model seed `42 + fold*1009 + 9_999_991`. Auxiliary rows
enter training only and are never scored (`assert_evaluation_unchanged`: 0 extra rows, max target
delta 0.0 in every arm).

## 2. The comparison is forced, and the reference is not the frozen anchor

The frozen 3-column metal block is NaN on 98.44 % of auxiliary rows (median-imputed to europium),
and two MASSACTION columns are null on 98.44 % / 40.66 % of them against 0.02 % of cohort rows —
they would act as an "is-auxiliary" flag. So `GENERAL` + `ANNOTATION_SAFE` are **forced** on every
auxiliary arm, and the honest reference is the design-matched control, not the frozen anchor.

| arm | macro MAE | eff = control − arm |
|---|---|---|
| frozen gen10 anchor | 0.96948 | — |
| `GENERAL` + `FROZEN_8` | 0.96919 | — |
| **design-matched control** | **0.97324** | — |
| `FROZEN_3` + `ANNOTATION_SAFE` | 0.97809 | −0.0086, BCa [−0.0156, −0.0041] |
| B (auxiliary lanthanides, n≈38) | 0.97695 | −0.0037 |
| **C (all actinides, n≈3,852)** | **0.92410** | **+0.0491** |
| **D (non-actinides, n≈357)** | **1.02662** | **−0.0534** |
| E (everything, n≈4,247) | 0.92598 | +0.0473 |
| **F (actinides matched to D, n≈357)** | mean 0.01244 above control | **−0.0392**, BCa [−0.063, −0.018] |

> **A caveat that applies to gen11's own headline:** at the pre-registered 5,000 replicates,
> `eff(C)` has BCa **[−0.00049, +0.11865] — which includes zero.** It excludes zero only at 20,000
> replicates. C does not clear the bar that F is being judged against.

## 3. Primary endpoint — and why it settles nothing

`eff(F) − eff(D)`: F and D have **bit-identical `n_aux` in all 25 folds** (max difference 0), the
same `aux_lambda`, and therefore the same per-row up-weighting. Per draw:

−0.0017, +0.0243, +0.0119, +0.0159, +0.0205 → **mean +0.0142**

| statistic | value |
|---|---|
| chemotype-blocked BCa on the mean | **[−0.034, +0.085]** |
| block-bootstrap SE | 0.0309 |
| draws with BCa excluding zero | **0/5** |
| draws with BCa excluding +0.04 | **0/5** |
| split seeds positive | 4/5 (+0.0213, +0.0221, −0.0027, +0.0159, +0.0143) |
| draw-level t-test on the five means | t = 3.16, **p = 0.034** |

**The interval contains zero, the pre-registered H_CHEM value of +0.04, and the whole H_MIX band.**
z(+0.0142 vs +0.04) = −0.84, p = 0.40. Likelihood ratio H_CHEM : H_AMP = **0.78** — the experiment
carries essentially no information separating the two hypotheses it exists to separate.

### 3.1 The power failure

* Minimum detectable effect at 80 % power: **+0.087** — 2.2× the pre-registered bar, and 1.8× C's
  entire measured gain.
* Power to detect a true +0.04 under the three-part rule: **≈22–25 %**.
* TOST against the H_AMP band |·| < 0.02: **fails, p = 0.43.** The interval is three times wider
  than the equivalence band.
* The design's one precision lever was inert: **87–94 % of the endpoint's variance is chemotype
  sampling shared by all five draws**, so averaging draws bought ~6 % variance reduction where
  independence would have bought 80 %. Resolving +0.04 at 80 % power needs roughly **370
  independent chemotype blocks against the 79 that exist.** No number of extra draws or seeds
  reaches it.
* **No power calculation appears anywhere in the pre-registration.** That is the defect.

**Therefore: H_CHEM is not rejected (a non-significant result is not a rejection) and H_AMP is not
established (a point estimate inside a band whose interval is three times wider is a
non-detection).** Both remain unsupported.

## 4. The real finding: the aggregate null is a mixture of two opposite effects

Stratifying by the pre-declared chemotype bands (§7.3, declared before the run):

| band | rows | ECFP units | **F − D** | draws > 0 | F − control | D − control |
|---|---|---|---|---|---|---|
| **far** (nn ≤ 0.4) | 2,138 | 30 | **+0.2018** | **5/5** | −0.0700 | **−0.2717** |
| mid (0.4–0.6) | 5,367 | 71 | −0.0524 | 0/5 | −0.0674 | −0.0150 |
| near (> 0.6) | 18,735 | 71 | +0.0015 | 3/5 | −0.0053 | −0.0068 |

Far-band `F − D` by split seed: +0.1998, +0.1998, +0.1865, +0.2023, +0.2204 — **5/5**. The five G
draws reproduce it independently at +0.1810, also 5/5, so the far-band separation from D holds in
**10 of 10** actinide-dominated draws.

**On genuinely distant chemistry — the regime the whole archive-transfer programme exists to serve
— non-actinide auxiliary rows do enormous damage (−0.272) and actinide rows at the identical block
size do far less (−0.070).** The aggregate +0.0142 is a mixture dominated by the near band, which
holds 71 % of evaluation rows and where there is almost nothing to rescue.

Stated honestly and in both frames: **actinide chemistry recovers roughly 70 % of D's far-band
damage at equal n, but it does not help — F on the far band is still −0.070 against the control.**
On 30 units and 28 blocks this is suggestive, not established, and a separate verification pass
using a different aggregation put the same contrast at +0.132 with BCa [+0.0015, +0.3874]. The
direction and the 5/5 draw and seed agreement are robust; the magnitude is not pinned down.

## 5. Arithmetic check on the sampler — G passes, and G is therefore uninformative

`G_RANDOM_AUX_MATCHED` was designed as the "does row count alone explain it" control. It draws from
`(ACTINIDE,) + NON_ACTINIDE_OTHER`, and **that pool is 5,050 cells of which 4,607 — 91.2 % — are
actinide.** A size-matched draw of ~357 is therefore ~91 % actinide against F's 100 %: the two arms
differ by roughly **31 cells out of 357**. The pre-registration predicted `|eff(G) − eff(F)| < 0.02`
and said that if G instead tracked D, the reading of the sampler was wrong and §2 had to be
withdrawn.

| | per draw | mean |
|---|---|---|
| eff(F) | −0.0551, −0.0291, −0.0415, −0.0375, −0.0328 | **−0.0392** |
| eff(G) | −0.0282, −0.0438, −0.0301, −0.0361, −0.0405 | **−0.0357** |

**|mean eff(G) − mean eff(F)| = 0.0035.** The check passes; the sampler reading stands. The stage
leaderboard shows the two arms fully interleaved (G 1.0014, F 1.0024, G 1.0034, F 1.0061, G 1.0093,
F 1.0107, G 1.0137, F 1.0147, G 1.0170).

**The consequence is that G carries almost no independent information.** As coded it is not a
"random auxiliary" control at all but a second actinide arm, so it cannot test whether row count
alone explains D's harm. Half the compute of this stage went to an arm that could not answer its
own question. A control that would answer it must sample from the **non-actinide** pool, or the
contrast must be F against D directly — which is what §3 does.

What G does add is replication: `G − D` is **+0.0176** against F's +0.0142, and on the far band
`G − D` is **+0.1810, 5/5 draws** against F's +0.2018, 5/5. Treating the ten draws as ten samples
of "an actinide-dominated block at D's exact size", the far-band separation from D is positive in
**10 of 10** draws.

## 6. What the C-vs-D contrast can and cannot support

It cannot support chemical relevance, but **not for the reason the pre-registration gave.** C and
D differ simultaneously in four ways:

| | C | D |
|---|---|---|
| auxiliary rows per fold | 3,852 | 357 |
| per-row weight vs core | 1.09× | 11.76× (mean of ratios: **17.88×**) |
| **per-chemotype weight** — the unit `HIERARCHICAL` actually equalises | ~1× | **~26×** |
| auxiliary structures / chemotypes | 153 | **7 / ~3** (71 % of rows on one ligand) |

Three corrections to the pre-registration's own §3 follow, and they are corrections to this
session's work, not to the original study:

1. The amplification table is captioned "mean over 25 folds" but reports **ratios of fold-means**.
2. Under `HIERARCHICAL`, `group_balanced_weights` assigns 1/count per row *within* the auxiliary
   block, so the objective's unit is the ECFP cluster, not the row. **Per-row amplification is the
   wrong statistic.** On the right one, F sits at ~1.7× per chemotype against D's ~26×.
3. Consequently **Addendum 1(b)'s "F and D differ only in which metals the block contains" is
   false.** They match on `n_aux` exactly and differ ~15× in per-chemotype weight and ~12× in
   auxiliary ligand diversity — both flattering F.

**And no amplification statistic orders the arms.** Per row, B is the most amplified (109.9×, or
115.8× correctly averaged) and is harmless (−0.0037). Per chemotype, F is barely amplified (~1.7×)
and still harms (−0.0392). H_AMP is no better supported by the arms on disk than H_CHEM is.

The strongest supported contrast in the whole stage is one the primary endpoint does not contain:
**C − F = −0.0884, BCa [−0.174, −0.034], excluding zero.** At fixed chemistry — both blocks
entirely actinide — a 10.8× change in block size is worth six times the F−D contrast and is
statistically supported where F−D is not. It bundles "more rows" with "more coverage" and "less
up-weighting", so it isolates nothing, but it is the real signal in these data.

## 7. A defect in the shipped level/shape numbers

`gen6.metrics.per_unit_statistics` runs `decompose_level_shape` on the **seed-pooled** frame, so
each ligand's residuals are centred on a mean taken across all five split seeds, and between-seed
level wobble is scored as *shape*. Measured per-ligand cross-seed level SD: F 0.288–0.339,
D 0.258, control 0.248, C 0.196 log units.

Holding one axis at a time on `d_shape(F − D)`:

| change | value |
|---|---|
| pre-registered statistic (131 clusters, seed-pooled) | −0.0263 |
| averaging unit only → 152 ligands | −0.0285 *(moves away from zero)* |
| **centring only → per (split_seed, ligand)** | **−0.0061** |
| both | −0.0063 |

**77 % of the collapse is the centring, not the unit.** `gen11.analysis.decompose` centres per
(split_seed, extractant) and is unaffected.

**This has reached a published headline number.** `headline_tables/t3_paired_bootstrap.csv` ships
`B_LN_EXPANDED` `shape_mae` = **+0.011904, BCa [0.00362, 0.02172], `significant_bca = True`** — the
only significant transfer effect in the published tables. Recomputed with per-seed centring it is
**+0.005** (independently reproduced here as +0.00516 against a pooled +0.01212), below the 0.01
interpretability floor. The same artefact inflates C's shape gain from **+0.0026** (per ligand, per
seed) to **+0.0179** (pooled, per cluster).

**Action: every offset/shape number in gen11's published tables needs recomputation with per-seed
centring before it is cited.** The macro-MAE numbers are unaffected.

## 8. Corrections to this session's own analysis

Recorded because the audit trail is part of the result.

1. **A shape effect was claimed and then retracted, and the stated reason was wrong.** The
   retraction attributed the collapse to the averaging unit; §7 shows it is the seed-pooled
   centring. The recorded lesson ("use ligands, not clusters") would not prevent a recurrence.
2. **The retraction substituted an unregistered statistic for the pre-registered one after the
   result was seen.** §6 pre-registers the ECFP-cluster unit and `paired_chemotype_bootstrap`;
   under that convention `d_shape(F−D)` is −0.0264 with 4/5 draws BCa-excluding-zero. The
   substitution is correct on the merits — seed-pooled centring is a genuine artefact — but it was
   chosen post hoc and is reported as a correction with its mechanism, not as a bare retraction.
3. **The addenda are mis-dated relative to the fits.** Both say "before any result". The file was
   last written 20:33:46; F draw11's OOF landed 20:27:44, and Addendum 1(b) quotes a diagnostics
   file that exists only because that fit finished. The §1 header "before the `matched` stage was
   run" is false by 4 minutes 49 seconds. No matched-stage *outcome* informed the predictions —
   they were fixed in §5 before the stage started — but the header is wrong and is corrected here.
4. **`scripts/gen11_matched_analysis.py` was edited after the F draws landed** (mtime 21:33:54 vs
   last F draw 21:29:52). The edit fixed a column name and added a per-statistic filter and a
   "BCa excluding zero: k/5" counter. It changed how results were summarised, not which arms ran.
5. **Leave-one-chemotype-out on the primary endpoint spans −0.0008 to +0.0356**, driven by
   `tan055`, which holds 28 of the 131 scoring units.

## 9. Status against the pre-registered stopping rule

| criterion | bar | outcome |
|---|---|---|
| A zero-shot | ≥ 0.02 macro vs design-matched control, seed-consistent | **not met** — F is −0.039; C is +0.049 but its BCa includes zero at the registered replicate count |
| B few-shot | ≥ 0.01 at two or more k | **not evaluated** — the adapter chain was never run on these arms |
| C shape | ≥ 0.10 shape MAE, macro non-worse | **not met**, and the shape statistic itself is contaminated (§7) |
| D far chemotype | statistically supported improvement on distant chemotypes | **suggestive, not established** — F−D is +0.20 on the far band, 5/5 draws and seeds, but F is still −0.070 against the control there |
| E robustness | lower query-design sensitivity at matched accuracy | **fails** — auxiliary arms are markedly *more* decoy-sensitive (DECOY median 0.138 C / 0.163 E vs 0.078 anchor) |

**No criterion is met. gen11 does not ship.**

## 10. What would settle it

1. **The λ ladder the pre-registration already names (§9):** D at `aux_lambda ≈ 0.085`, i.e.
   per-row parity with the core, against D at λ = 1. If D's harm disappears at parity the pathology
   is weighting; if it persists it is chemistry. One arm, five seeds.
2. **C sub-sampled to D's *chemotype* count rather than its row count.** F matched the wrong
   quantity: it equalised `n_aux` and left a 15× gap in per-chemotype weight.
3. **A far-band-powered design.** The far band carries the only directional signal in this stage
   and has 30 units and 28 blocks. Nothing at this cohort size resolves +0.04 aggregate; a design
   targeting the far band specifically is the only one with a chance.
4. Recompute every published offset/shape number with per-seed centring (§7).
5. Run the two negative controls §19 requires before any of this is believed.
