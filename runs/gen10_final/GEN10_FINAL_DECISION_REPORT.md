# GEN10 final decision report

**Frozen: gen9's `SHAPE_RECOMPOSED` global model, unchanged, with a new k ≥ 2
adapter (`SERIES_ML`), gen8's slope repair at k = 1, and central-then-spread as
the first point. Common-cohort frontier 1.036 / 0.654 / 0.559 / 0.493 / 0.441 at
k = 0 / 1 / 2 / 3 / 5, against gen9's 1.036 / 0.654 / 0.575 / 0.511 / 0.468.
No further model generation is justified on this cohort.**

Every number below carries its seed count and cohort. Nothing from a single
seed is quoted as a result. The files a number comes from are named in each
section; `headline_tables/` holds every table as CSV/JSON, regenerated from the
raw predictions.

---

## 0. What broke, what was fixed, what was rerun, whether any headline changed, what is frozen

### 0.1 What broke during gen10

Seven defects in gen10's own new code, found by the checks the brief asks for
and fixed before any number depending on them was read. None is in gen9's code.

| # | defect | how it was caught | fix | rerun |
|---|---|---|---|---|
| 1 | The additive residual arm (`ResidualShapeModel`) was **irreproducible at 3.4e-2 log units on 372 of 382 held-out rows**: the base forest's unrounded in-sample prediction fed the correction target and the ~1e-15 thread-order noise grew a different second forest — gen9's issue 6 in new clothes | refit twice, compare | `stabilise()` rounds every model output that becomes another model's target to 1e-9; final predictions are never rounded so the gen9 nesting stays exact | the arm, before any 5-seed run |
| 2 | The same arm's correction target was the base forest's **in-sample** residual, which an ExtraTrees with leaf size 2 nearly interpolates (target sd 0.20 vs 0.74 held-out); the arm collapsed toward the frozen model | target spread inspected | inner chemotype cross-fitting; the in-sample version kept as a labelled trap control | the arm |
| 3 | Phase 7's no-chemistry baseline pooled **same-fold** ligands into the condition-cell lookup — same-chemotype siblings the model never sees — and "beat" the model on consistent rows 0/5 seeds | sign of the comparison was implausible | other-fold rows only | Phase 7; the comparison reversed to model-better 5/5 |
| 4 | The consistency benchmark predicted the whole ligand frame per variant (TODGA: 1,488 rows, hundreds of curves) and would not finish | runtime | same-series restriction (exact: a `CONTEXT` variant checks whole-ligand vs series predictions agree at 1e-15) and a 12-curve cap per ligand, recorded in every row | Phase 1 |
| 5 | Error-decomposition components 4–6 substituted a pooled mean into small clusters and came out negative | negative "contributions" | ligand-level macro excess, raw and clipped both reported | Phase 10 |
| 6 | A driver seeded an RNG through Python's salted `hash` | code review | removed; the AST scan now covers gen10 and the shared gen6/gen8 modules | — |
| 7 | gen9's Phase-0 script was launched with its default `--out` and would have overwritten `runs/gen9_shape/reproduction` | noticed on launch | killed before it wrote; gen9 manifest re-verified 154/154 zero drift; rerun into `reproduction/gen9_reproduce_rerun` | — |

Two further audit-script bugs (a by-construction NaN in guarded span columns
failing the NaN scan; tuple keys breaking the JSON record) were fixed in the
self-audit itself; they never touched a result.

### 0.2 What was reproduced (Phase 0, `reproduction/phase0.json`, 12/12)

* gen9's manifest verifies: 154 artefacts, zero drift.
* gen9's own Phase-0 script reproduces all **15** gen8 references in this
  environment (`reproduction/gen9_reproduce_rerun/`).
* Cohort fingerprint `bed178ec1a7a82b0`; 1,176 curves / 7,207 memberships; no
  curve straddles any of 25 fold boundaries; every held-out partition is closed.
* The three gen9 arms are reproduced inside gen10's classes at the float floor:
  max |Δ| ≤ 1.3e-15 for `REC_ecfp_plus_recovered`, `GEN9_REL_MONOLITH`,
  `GEN9_SHAPE_RECOMPOSED`. Every gen9 five-seed number quoted here is therefore
  the same arithmetic, not a re-implementation.
* Same configuration twice in one process and in fresh subprocesses under
  `PYTHONHASHSEED` 0 / 4242: identical at the floor.
* 214 tests pass, including the metamorphic tests the brief lists (MEDOID vs
  FARTHEST disagree on an asymmetric pool; non-constant axes stay non-constant;
  target removal/corruption leaves predictions and selections unchanged; row
  permutation permutes predictions) and one regression test per gen9 issue.

### 0.3 What was rerun, and did any headline change?

Every gen10 number is post-fix. No gen9 headline changed: k = 0 and k = 1 of the
frozen pipeline are gen9's numbers to four decimals (1.0358, 0.6539), the
FROZEN / QUARANTINED / CORRECTED cohorts move macro MAE by < 0.001, and gen9's
`SERIES_MAP` and `OFFSET_K3` reproduce to three decimals inside the new
adaptation study.

### 0.4 What is frozen

`final_locked/pipeline_frontier.json`: global model `GEN9_SHAPE_RECOMPOSED`;
k = 1 `SLOPE_L_s1_K1`; k ≥ 2 `SERIES_ML`; first point `CENTRAL_THEN_SPREAD`.
Model card: `GEN10_FINAL_MODEL_CARD.md`. Self-audit: `self_audit/self_audit.json`.

---

## 1. The result in one table

Five seeds, FROZEN cohort, extractant axis (775 curves / 25 ligands); common
cohort for the k-shot columns (99 ligands):

| | frozen (gen8) | gen9 `SHAPE_RECOMPOSED` | **gen10 frozen pipeline** | `GEN10_REC_HYBRID` (reported) |
|---|---|---|---|---|
| macro MAE, zero-shot | 0.9807 | 0.9695 | **0.9695** | 0.9696 |
| extractant slope (measured 2.574) | 0.116 | 1.024 | **1.024** | 1.647 |
| extractant shape MAE | 0.665 | 0.469 | **0.469** | 0.394 |
| span recovery | 0.051 | 0.423 | **0.423** | 0.615 |
| k = 1 / 2 / 3 / 5 | 0.667 / 0.588 / 0.523 / 0.474 | 0.654 / 0.575 / 0.511 / 0.468 | **0.654 / 0.559 / 0.493 / 0.441** | 0.652 / 0.558 / 0.493 / 0.441 |
| median shift under two far decoy points | **0** | 0.117 | **0.117** | 0.108 |
| median shift under interior density change | **0** | 0.019 | **0.019** | 0.041 |

---

## 2. How the selection was made

The brief fixed the rule before the runs: a change enters the frozen model only
if it achieves (A) ≥ 0.02 macro MAE on the locked common cohort, or (B) ≥ 0.10
shape MAE with macro non-worse, or (C) meaningful k = 1/2 improvement without
later degradation, or (D) materially better query-set robustness at matched
accuracy.

| candidate change | A | B | C | D | decision |
|---|---|---|---|---|---|
| `SERIES_ML` adapter (Phase 5) | **yes** at k = 5 (−0.027); −0.015 / −0.018 at k = 2 / 3, CI excluding 0, 5/5 seeds | — | **yes** at k = 2 | n/a | **frozen** |
| rank/spacing representation `REC_HYBRID` (Phase 4) | no (−0.0045 at k = 0) | no (0.075, CI [0.042, 0.085]) | no (−0.002 at k = 1) | **no** — more robust to decoys/extension, less to interior changes (§3.1) | reported |
| `REC_GEN9_PLUS` (gen9 columns + on-curve flag + absolute coordinate) | no (+0.001–0.005 at every k, CIs excluding 0) | no | no | same class as gen9 | reported |
| explicit level + shape, residual, two-branch, set encoder (Phases 2–3) | all worse on macro | — | — | — | rejected (`GEN10_FAILURE_ANALYSIS.md`) |
| learned first point on realised regret (Phase 6) | α = 1.0 in 24/25 folds | | | | rejected; MEDOID / central shipped |
| feature access (Phase 2B) | none beats `REL_MONOLITH` | | | | closed |

No arm was chosen per seed or per ligand; the pipeline's rule is the same
everywhere.

---

## 3. The thirteen questions

### 3.1 Is gen9's relative-context mechanism robust to query-window changes?

**No.** Five seeds, same fitted model, same absolute points, different candidate
designs (`query_consistency/summary_by_arm.csv`; 66 ligands with synthesisable
concentration axes, 130 for subset variants). The frozen model moves by exactly
zero under every variant; the gen9 arms move:

| perturbation | `REL_MONOLITH` median / p95-of-p95 | `SHAPE_RECOMPOSED` median / p95-of-p95 |
|---|---|---|
| two decoy points 2–3 decades outside the window | 0.090 / 0.54 | **0.117 / 0.68** |
| one boundary extended 0.5–2 decades | 0.060 / 0.48 | 0.073 / 0.59 |
| one or two end points dropped (nested) | 0.029 / 0.36 | 0.046 / 0.52 |
| thinned to 2–8 points, endpoints kept | 0.019 / 0.56 | 0.042 / 0.64 |
| interior points added, window fixed | 0.008 / 0.37 | 0.019 / 0.62 |

The shift is almost entirely *shape* (decoy: 0.12 shape vs 0.02 level), it is
present on both concentration axes (acid decoy 0.113, extractant decoy 0.128),
and it is largest for the narrowest windows (Q1 0.043 vs Q4 0.022). A 1-decade
translation of the whole design changes the centred shape by 0.03 (extractant)
to 0.11 (acid) — less than the frozen model's own 0.05–0.10, so the model does
read relative position rather than absolute concentration. **A decoy point two
decades away moves a prediction by more than gen9's whole 0.025 macro gain.**
The mechanism is real and it is fragile; the fragility lives in the four
endpoint-based columns, as predicted in `features.SENSITIVITY` before the run.

*On the finalists (2 seeds, `query_consistency_finalists/`)*: the rank/spacing
representations halve the decoy tail (p95-of-p95 0.54 `HYBRID`, 0.51 `RANK` vs
0.68) and cut the extension shift (0.063 / 0.057 vs 0.066), but **double the
interior sensitivity** (density median 0.041 / 0.048 vs 0.018; sparse 0.069 /
0.067 vs 0.040), because a rank percentile moves by 1/n whenever an interior
point is added. Neither family dominates; criterion D is not met.

### 3.2 Is recomposition still necessary?

**Yes.** Phase 2B: no `max_features` (sqrt, 0.25, 0.5, 1.0) or context-column
replication (×8, ×32) moves the monolith's macro (0.985–1.071 vs 0.986) or
extractant shape (slope 0.29–0.52 vs 0.52) toward the recomposition (0.9695,
slope 1.02). Five seeds, `feature_access/`. The recomposition's gain is not a
split-proposal artefact.

### 3.3 Does explicit level+shape modelling beat it?

**No — it reproduces the shape and loses on the level.** `GEN10_LEVEL_SHAPE`:
macro 0.9818 vs 0.9695; extractant slope 1.036 vs 1.024, shape MAE 0.477 vs
0.469, span 0.430 vs 0.423; offset MAE 0.830 vs 0.821. Every architectural
alternative (residual 0.982, two-branch 0.996–1.003, set encoder 1.006–1.009)
fails the same way: the monolith's row-wise prediction averaged over a curve is
the best level this corpus supports, and the recomposition is the only design
that touches the shape while leaving it bit-identical. Five seeds,
`level_shape/`, `set_context/`.

### 3.4 Does a learned set representation add anything beyond handcrafted coordinates?

**No.** DeepSets (mean) 1.0059, attention 1.0087 macro vs 0.9695, with the
identical level; extractant shape MAE 0.452–0.459 (matches 0.469), acid and
lanthanide shape worse. Overfit without early stopping; an inner chemotype
hold-out for the epoch count was the one fitted knob. Fails the brief's bar.

### 3.5 Which axis representation is best?

**For shape, rank/spacing; for stability, it depends on the perturbation; for
macro, none.** Five seeds, recomposed arms (`axis_representation/`): macro
0.9690–0.9700 for all but LOCAL (0.975); extractant shape MAE ALL 0.383 <
HYBRID 0.394 < RANK 0.406 < GEN9_PLUS 0.456 < GEN9 0.469; acid shape 0.50–0.51
vs 0.545; lanthanide shape 0.305 vs 0.291 (a small cost). `HYBRID` vs gen9's
columns: extractant shape +0.075 [0.042, 0.085], 115/155 curves, 5/5 seeds;
slope +0.28; span-distance +0.068. The whole-curve window instead of gen9's
primary-curve window changes nothing (0.9700, slope 1.004). Under the stopping
rule the representation is reported, not frozen (§2, §3.1).

### 3.6 Can corrected series-local adaptation beat `OFFSET_K3`?

**Yes, and it is the one change gen10 freezes.** Five seeds × 12 repeats,
`adaptation/`, on `SHAPE_RECOMPOSED`, central-then-spread:

| adapter | k = 2 | k = 3 | k = 5 | vs `OFFSET_K3` (k = 3; CI; ligands; seeds) |
|---|---|---|---|---|
| `OFFSET_K3` (gen8) | 0.5447 | 0.4915 | 0.4789 | — |
| `SLOPE_L_s1_K3` (gen9 frontier) | 0.5366 | 0.4846 | 0.4714 | +0.007 [0.001, 0.013] |
| `SERIES_MAP` (gen9) | 0.5915 | 0.5660 | 0.5572 | −0.075 [−0.103, −0.052] |
| `SERIES_FIXED` | 0.5425 | 0.4868 | 0.4643 | +0.005 [0.000, 0.009] |
| `SERIES_INNER` | 0.5350 | 0.5231 | 0.4651 | −0.032 [−0.052, −0.018] |
| **`SERIES_ML`** | **0.5193** | **0.4609** | **0.4405** | **+0.031 [0.020, 0.044], 91/140, 5/5** |

`SERIES_ML` clears the pre-registered bar (≥ 0.01 at two of k = 2, 3, 5 with
consistent seeds) at all three k, on both global models and all three policies,
and beats its own no-series ablation (0.5295 / 0.4745 / 0.4670), so the series
term earns its place. The k = 1 identity holds at 3e-14. gen9's estimator is
rediagnosed in `GEN10_FAILURE_ANALYSIS.md` §7: its series family sat at the
penalty *floor*, not the ceiling.

### 3.7 Can realised-regret acquisition beat MEDOID?

**No.** Five seeds × 8 repeats, 143 ligands (`acquisition/realised_*`):
`LEARNED_BLEND[geometry]` 0.634 vs MEDOID 0.629 (−0.006 [−0.016, +0.007], 0/5
seeds); scalar 0.640, rank 0.648; oracle 0.463. The blend's α, chosen on inner
ligands the ranker never saw, is **1.0 — pure centrality — in 24 of 25 folds**.
The closed fraction of the MEDOID→oracle gap is negative. The direction is
closed; MEDOID / central-then-spread is shipped.

### 3.8 How much error is associated with incomplete system representation?

**Real on the rows, small in the headline.** Phase 7 (`data_ceiling/`): mismatch
rows carry an irreducible residual (after a per-ligand offset) of 0.934 vs 0.582
on consistent rows, ligand-level p = 0.0018 — the missing second species changes
shape, not only level. But the excess pooled MAE attributable to all
non-consistent cohorts together is 0.015 of 1.122, and at the ligand-macro
level mismatch contributes ≈ 0 while duplicates, TWE-24 and flagged outliers
contribute 0.037. On those rows a same-cell peer from another publication
predicts three times better than the model (1.17 vs 0.33). The mismatch audit
table lists the 15 structures, their recorded names, complexants and DOIs;
nothing was corrected or deleted.

### 3.9 How publication-dependent is the shape result?

**It survives every clustering, and the chemotype-blocked interval gen9 quoted
was the optimistic one.** Phase 8 (`publication_sensitivity/`), extractant axis,
`SHAPE_RECOMPOSED` vs frozen, one unit per curve: shape MAE +0.196 with CI
[0.118, 0.203] chemotype-blocked, [0.140, 0.257] publication-blocked,
[0.140, 0.258] two-factor; slope MAE +0.92, span-distance +0.38, Spearman +0.30,
all positive under all four schemes. Leave-one-publication-out: the largest of
32 publications carries 16.9 % of the summed gain; removing any one leaves
≥ 0.176. Only **8 chemotypes** carry extractant titrations, which is why the
chemotype-blocked interval is the narrowest; quote the publication or two-factor
interval.

### 3.10 What is the final k = 0 / 1 / 2 / 3 / 5 frontier?

Common cohort, 99 ligands, frozen rule: **1.0358 / 0.6539 / 0.5593 / 0.4928 /
0.4405** (gen9: 1.0358 / 0.6539 / 0.5746 / 0.5109 / 0.4675; gen8: 1.0605 /
0.6674 / 0.5879 / 0.5230 / 0.4743). `final_locked/frontier_best.csv`. The
improvement is entirely at k ≥ 2 and entirely the adapter.

### 3.11 What portion of remaining error appears realistically removable?

From `GEN10_ERROR_CEILING.md`: of 0.970 macro, **0.52 is a per-ligand level**
(0.35 realistically removed by one central measurement), 0.19 is within-curve
shape (0.008 removed at k = 1 by the repair; the rest needs in-series points),
0.06 series-local level (removed at k ≥ 3), 0.06 distant-chemotype support and
0.04 data quality (removable only by measuring and by source checks), and
**0.10 unexplained**. With a perfect per-curve level and slope the floor is
0.19. A zero-shot architecture can touch at most the 0.19 shape component, of
which gen9/gen10 already recovered half.

### 3.12 What exact pipeline should be frozen?

`GEN9_SHAPE_RECOMPOSED` (ExtraTrees 400 / 0.30 / leaf 2 monolith for the level;
a second forest on curve-centred targets with gen9's five relative columns for
the shape; mean-preserving recomposition per requested curve) → first point
`CENTRAL_THEN_SPREAD` → k = 1 `SLOPE_L_s1_K1` → k ≥ 2 `SERIES_ML`. Exact
configuration in `final_locked/pipeline_frontier.json`; model card in
`GEN10_FINAL_MODEL_CARD.md`.

### 3.13 Is there enough evidence to justify any further model generation?

**No.** Every architectural alternative lost on the level; the best
representation change buys 0.075 shape-MAE at neutral macro and trades one kind
of query fragility for another; the zero-shot headroom is bounded by a 0.19
shape component; and one measurement on a far-chemotype ligand is worth 0.49
macro — ten times any modelling gain since gen6. What would change the picture
is data: extractant titrations on new chemotypes (8 today), the primary document
for TWE-24, and the second species on 271 rows. None of those is a model.

---

## 4. What was wrong, what was fixed, what survived — in plain language

**Did gen10's first versions have bugs?** Yes, seven, all in new code, all
caught by refitting twice or by a comparison that came out the wrong way round
(§0.1). The worst would have made a new arm look reproducible when it was not,
and another would have made a second model learn nothing while appearing to
train. Neither reached a reported number.

**What did gen10 find?** That gen9's new feature — telling the model where a
point sits inside the planned titration — has a cost gen9 did not measure: the
model's answer for a point changes when you add points you never intended to
run, by about as much as the feature gained. No architecture gen10 tried removes
that; the more robust coordinate systems are robust to a different set of
changes. The honest thing is to freeze gen9's model as it is, report the
dependence, and tell the user to hand over the plan they will actually run.

**What did gen10 fix?** The series-local calibration. gen9 built the right
structure with a mis-estimated prior; with the prior estimated by marginal
likelihood it beats gen8's hand-set penalty by 0.03 at three and five
measurements, on every seed.

**What did gen10 close?** Learned first-point acquisition (the honest selector
chooses plain centrality 24 times out of 25), feature access as an explanation
of the recomposition, explicit level heads, residual corrections, two-branch
models, learned set encoders, and the idea that another zero-shot generation
would pay. The repository ends with one reproducible, leakage-safe pipeline whose
limits are measured: half its error is a number one experiment supplies, a tenth
is unexplained, and the rest is data.
