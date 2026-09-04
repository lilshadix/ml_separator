# Final figure report

---

## 1. Recommended paper figure set

| # | file | what it establishes |
|---|---|---|
| 1 | `main/Fig1_methodology.{pdf,png}` | The target, the chemotype-blocked hold-out, the two-forest mean-preserving model, and — the load-bearing part — the separation of training information, target-free candidate conditions, *k* measured support targets and held-out query targets, so that the *k*-shot numbers cannot be read as leakage. |
| 2 | `main/Fig2_fewshot_frontier.{pdf,png}` | On 99 extractants with no analogue in training, macro MAE falls 1.036 → 0.654 → 0.559 → 0.493 → 0.441 at *k* = 0/1/2/3/5; the first measurement removes 0.382 [0.264, 0.565], an order of magnitude more than any model change here; and one third of extractants are still made worse by it. |
| 3 | `main/Fig3_shape_recovery.{pdf,png}` | The baseline model draws held-out extractant titrations at 1/20th of their measured slope (0.12 against 2.57) and recovers 5 % of their range; telling it where a row sits inside its own measurement window and recombining mean-preservingly moves those to 1.02 and 42 %, with within-curve Spearman 0.58 → 0.89. |
| 4 | `main/Fig4_error_decomposition.{pdf,png}` | Half the remaining error (0.520 of 0.970) is a per-extractant constant; one *optimally chosen* measurement (0.504) is worth as much as knowing that constant exactly (0.507); a better zero-shot model moves shape error and leaves level error bit-identical, while measurements do the opposite; 0.103 is unexplained. |
| 5 | `main/Fig5_generalization.{pdf,png}` | A controlled coverage experiment on byte-identical held-out rows: relaxing the training-eligibility rule from ten condition cells to three improves macro MAE by 0.163 [0.012, 0.295] overall and 0.463 [0.251, 0.685] on extractants below Tanimoto 0.4, with a row-count-matched arm reproducing it and a shuffled-target arm recovering none of it. |
| 6 | `main/Fig6_acquisition.{pdf,png}` | *(publish if a sixth figure is available, otherwise supplementary)* Choosing the medoid candidate rather than a random one is worth 0.063 [0.028, 0.105] at *k* = 1, more than twice the best model change; no uncertainty rule beats random; and 28 % of extractants are still made worse. |

Supplementary set: `supplementary/FigS1`–`FigS10`, described in `FIGURE_CAPTIONS.md`.
`FigS2` (query-set fragility) is not optional — it is the negative result attached to
Figure 3.

## 2. The central visual narrative

Figure 1 says **what was done** and, more importantly, what was withheld: whole chemotypes
out of training, a candidate pool disjoint from the scored rows, a frozen model that is
never refitted.

Figure 2 says **why the task is hard and what actually helps**. Zero-shot error on genuinely
new chemistry is about one log unit — larger than most published lanthanide-extraction
models would suggest, because most are not evaluated on unseen chemotypes. One measurement
removes 37 % of it. Every model change in a seven-generation study removes 2–3 % each.

Figure 3 says **what the model gets wrong that the average hides**. Even where the mean
error is acceptable, the internal structure of a held-out titration is drawn flat, and the
reason is a coordinate-system omission rather than a capacity limit. This is the figure a
methods reader will remember, and it is the one that generalises beyond this dataset.

Figure 4 says **why Figure 2 looks the way it does, and where the ceiling is**. The error is
a level problem; measurements fix levels; models fix shapes; the two move almost
orthogonally; and even with perfect per-curve levels and slopes the floor is 0.181.

Figure 5 asks **does any of this generalise, and can more data fix it**. The observational
distance split is suggestive but underpowered; the interventional coverage experiment is
clean and says that the binding constraint is chemical breadth of the training set, not
model capacity — with the qualification that most of the measured gain lands on chemistry
the restricted arm could never have served.

Figure 6, if space allows, closes the loop back to Figure 2 by decomposing its *k* = 1
point: the gap between the deployable 0.654 and the level-oracle 0.504 is the cost of not
knowing which experiment to run, and geometric centrality recovers a measurable part of it
while model uncertainty recovers none.

**what was done → why it is hard → what failed → what fixed it → how much → does it
generalise.**

## 3. Strongest figure

**Figure 2.** It is paired at the level of the individual support/query draw, it carries a
model-free null and an explicitly labelled oracle, its effect size is large relative to
everything else in the paper, its uncertainty is computed on the right resampling block,
and it shows its own failure mode (33 of 99 extractants worse) in the same figure. Every
one of its numbers reproduces from `kshot_detail.parquet` to five decimal places
(`METRIC_AUDIT.md` rows 1–11).

## 4. Weakest figure

**Figure 5**, and specifically panel A. The distance-tercile comparison is observational,
rests on 17 chemotypes per tercile, and its far − near contrast at *k* = 0 is +0.35
[−0.04, +0.66] — it does not exclude zero. Panels B and C are strong, but they use a
different learner and carry the qualification that 57 of 131 scoring units are clusters the
restricted arm structurally cannot serve.

**Recommendation.** Keep Figure 5 in the main text, but restructure it so that the
*interventional* result leads: promote panels B and C to A and B, demote the tercile panel
to a third panel or to the supplement, and retitle the figure "Training coverage, not model
capacity, is the binding constraint". The claim then rests on the controlled experiment,
with the distance split as supporting context rather than as evidence. If the journal
allows only five figures, Figure 6 goes to the supplement before Figure 5 does, because
Figure 5 answers a question a reviewer will certainly ask ("does it generalise?") and
Figure 6 answers a question that only arises after Figure 4.

## 5. Results that should NOT be plotted

* **The gen1–gen4 pair benchmarks (`log SF`).** A different target (a difference of two
  `log D` values), a different cohort, a different leakage protocol. Placing them on a
  macro-MAE axis with the `log D` results would be meaningless.
* **gen5's `unseen_series` regime.** 27 % of its cross-validation groups straddle a
  publication boundary, so the regime is not the held-out condition it claims to be.
* **Any cross-generation macro-MAE line spanning the 91-extractant and 152-extractant
  cohorts.** The eligibility rule changed between gen5 and gen6; the numbers are not on the
  same population.
* **gen11 transfer.** 5 of 34 pre-registered arms have run, and the one completed auxiliary
  arm is a wash whose sign is dominated by a design change it had to make
  (−0.0037 transfer, −0.0038 design). Reporting it now would be reporting an unfinished run.
* **The learned acquisition rankers as a positive result.** They do not beat plain
  centrality; they appear in Figure 6 as the negative result they are.
* **The published "gen8 frontier" and "gen9 frontier" columns.** Best-of-eleven-policies
  envelopes; see `METRIC_AUDIT.md` §3.1.
* **`runs/gen9_shape/figures/` and `runs/gen7_architecture/figures/`.** These are working
  diagnostics from the individual runs, generated before the cohort and metric conventions
  were unified. They are superseded by this set and should not be reused.
* **Model uncertainty as a calibration story.** Presenting the ensemble spread as
  "well-calibrated but unhelpful" would over-interpret it; the measured statement is
  narrower — under an offset calibration the useful quantity is `|r − median(r)|`, and the
  spread does not track it.

## 6. Ablation: what can and cannot be reconstructed (Phase 9)

**A clean, controlled ablation *can* be reconstructed for the model itself, and it is
already in the figure set.** Three chains are valid because every stage was run on the same
folds, the same cohort, the same seeds and — for the *k*-shot chain — the same
support/query draws:

1. **Deployment chain (Figure 2C).** baseline global model → + shape recomposition
   (+0.0247 [0.0157, 0.0375] at *k* = 0) → + series-local adaptation (+0.0309
   [0.0093, 0.0476] at *k* = 5). Exactly one component changes per step; each step's effect
   is measured paired at every *k*, and the two act at opposite ends of the *k* axis.
2. **Representation chain (Figure 3D–F, Figure S5).** baseline → + relative-position columns
   → + mean-preserving recomposition, on the shape metrics. Note that this chain is
   **not monotone in macro MAE**: the intermediate arm is slightly *worse* on macro (0.9858
   against 0.9807 on the full cohort) and it is the recomposition that converts the shape
   gain into an accuracy gain. That non-monotonicity is a result and should be stated, not
   smoothed.
3. **Zero-shot architecture ablation (Figure S4).** 24 arms — feature access and capacity,
   explicit level + shape heads, residual corrections, two-branch models, learned set
   encoders, six axis representations — all on identical folds and rows, five seeds.

**A clean ablation cannot be reconstructed across generations.** Specifically, the gen6
coverage expansion (a different learner and feature set), the gen7 descriptor-block
ablations (three seeds, separate suites) and the gen8 calibration study (a different
evaluation protocol) were not run under a common configuration. Any figure showing a single
descending staircase from gen2 to gen10 would be assembling incompatible experiments, and
none is included here.

*Minimum reruns that would close the gap*, if a cross-generation ablation is wanted: rerun
the gen6 `BASE`/`EXPANDED` contrast with the frozen `GEN9_SHAPE_RECOMPOSED` architecture on
the gen10 fold plan (roughly two arms × five seeds × five folds), which would put Figure 5B
on the same learner as Figure 5A and allow coverage to enter the Figure 2C chain as a
fourth stage.

## 7. The one experiment that would most improve the visual evidence

**Measure extractant-concentration titrations on chemotypes that currently have none.**

Every shape conclusion in this paper — Figure 3, Figure S5, the +0.196 publication-blocked
shape gain — rests on **8** Tanimoto-0.7 chemotypes, 25 extractants and 155 curves. That is
why Figure 3's intervals are wide, why the chemotype-blocked interval is the *narrowest*
available rather than the most conservative, and why the paper has to quote a
publication-blocked interval instead. Four to six new extractant titrations on chemotypes
outside the diglycolamide family — five points each, roughly 30 measurements — would
roughly double the number of independent blocks carrying the result and would convert
Figure 3 from "reproducible across seeds within a narrow chemistry" into "reproducible
across chemistry".

It is also the experiment the rest of the paper argues for: Figure 5C says coverage is the
binding constraint, Figure S10 says breadth beats depth at a fixed budget, and Figure 4B
says distant chemistry is one of only two components with no deployable remedy. A
prospective version — freeze the model, predict the new titrations, then measure — would
additionally supply the prospective test this repository does not have.

*Runner-up:* obtain the primary document for the extractant TWE-24 and re-check the
decade-shifted duplicate cells. Thirty rows currently contribute 0.021 of a 0.970 macro
MAE, and one ligand's suspected exponent error costs its nearest neighbour several log
units. That is a library task, not an experiment, and it is cheaper.

## 8. Verification status

| figure | reproduces directly from frozen outputs? | new calculation? |
|---|---|---|
| Fig 1 | yes — cohort counts read from the frozen OOF and curve tables | none |
| Fig 2 | yes — all values recomputed from `kshot_detail.parquet` and matching `frontier_best.csv` / gen9's `table_b_common_cohort.csv` to ≤ 5 × 10⁻⁵ | the paired chemotype-block intervals in B and C are computed here with the repository's own `paired_chemotype_bootstrap`; the marginal gains are recomputed for the frozen adapter chain rather than gen9's |
| Fig 3 | yes — every value in D, E, F matches `shape_by_axis.csv` to ≤ 5 × 10⁻⁶ | the block-bootstrap intervals of the median/mean in E and F; the example-selection rule |
| Fig 4 | panel B is read directly from `error_decomposition/decomposition.csv`; panel A's cascade reproduces `summary.json` to 5 × 10⁻⁹ before being rerun | the cascade on the 99-extractant cohort with per-extractant voting (so oracle and achieved share one axis); the level/shape split of the four model stages, using the repository's `decompose_level_shape` |
| Fig 5 | panel A's tercile machinery reproduces all 15 values of `marginal_gains.csv` to 10⁻⁹; panels B and C are read from `hard_chemistry_metrics.csv` and `contrast_summary.csv` | the far − near tercile contrast (unpaired chemotype-block bootstrap), added during the adversarial review |
| Fig 6 | yes — all 17 policies reproduce `realised_summary.csv` exactly | the paired-vs-RANDOM intervals, computed here with the repository's bootstrap |
| S1, S5, S7, S8, S10 | yes | none beyond aggregation |
| S2, S4 | yes — rendered without transformation | none |
| S3 | recomputed on a fixed 99-extractant cohort so it agrees with Figure 2; note that `adaptation/summary.csv` uses a *k*-varying cohort and prints different numbers | the fixed-cohort restriction |
| S6 | yes | none |
| S9 | recomputed per extractant from the stored gen6 out-of-fold predictions; reproduces the published dose–response table exactly (+0.038 / +0.126 / +0.321 / +0.491, ρ = +0.290, p = 2.9 × 10⁻⁴) | the per-extractant recomputation |

**67 of 67 audit checks PASS** (`figures/derived/metric_audit.csv`). Three figure scripts
refuse to draw if their own reproduction check fails. No figure in this set required an
experiment to be rerun.

---

### How to regenerate everything

```bash
for s in figures/scripts/prepare_*.py figures/scripts/verify_metrics.py \
         figures/scripts/plot_*.py; do .venv/bin/python "$s"; done
```

Each script runs standalone from the repository root, reads only frozen run outputs, and
writes its plotted values to `figures/derived/`.
