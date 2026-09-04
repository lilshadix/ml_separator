# Figure 6 — Which single measurement to make

## Source

Repository: `ml_separator` (branch `descriptor-arm-metal-site`)
Original script: `figures/scripts/plot_fig6_acquisition.py`
Original render: `figures/main/Fig6_acquisition.png`
Caption context: "## Figure 6" of `figures/FIGURE_CAPTIONS.md`

Input data/results:
* `runs/gen10_final/acquisition/realised_detail.parquet` — one row per
  (policy, feature set, extractant, split seed, pool draw); the `geometry` feature set only,
  exactly as in the published figure
* `runs/gen10_final/acquisition/realised_summary.csv` — the frozen per-policy summary; the
  script reproduces it for all 17 policies to < 1e-9 before it draws anything and exits
  with `acquisition summary did NOT reproduce` otherwise
* paired intervals from `lanthanide_separation.gen8.inference.paired_chemotype_bootstrap`
  via `figures/scripts/_stats.py`, re-exported by `figure_refinement/common/mlsep_data.py`
* every plotted number is written to `values.json` in this folder

Cohort **C-KSHOT** of `figures/METRIC_AUDIT.md`: 143 held-out extractants, 5 split seeds
× 8 pool draws, frozen global model, *k* = 1, one vote per extractant.

## Problem

Defects visible in `figures/main/Fig6_acquisition.png`:

* **Repository identifiers as the y axis.** All seventeen rows were labelled with their
  registry keys in capitals and underscores — `MEDOID`, `MAX_ENSEMBLE_SD`,
  `FARTHEST_FROM_EXISTING`, `MIN_GP_DESIGN_VAR`, `LEARNED_BLEND[geometry]`,
  `MAX_CONDITION_COVERAGE`. A reader outside the codebase cannot tell what any of them
  measure, and `LEARNED_BLEND[geometry]` exposes an internal feature-set switch.
* **The right statistic in the wrong place.** The horizontal whisker on each bar in panel
  A was the BCa 95 % interval of the *paired difference against `RANDOM`*, translated so
  that it straddled that policy's own point estimate. Nothing in the artwork said so. Read
  naively — and it will be — it is a marginal interval on the level, which would make
  `MEDOID` at 0.629 ± 0.03 look statistically indistinguishable from `RANDOM` at 0.692,
  the opposite of what the statistic actually says.
* **Legend inside the bars.** The six-entry family legend sat in the upper-right of panel
  A's data area, occupying the region between the bars and the zero-shot line.
* **Colliding reference label.** "  zero-shot" was drawn at `y = n - 0.4`, i.e. on the top
  bar row, so it overlapped the `ORACLE` bar's row and sat on the dashed line.
* **Annotation over data.** The `MEDOID vs RANDOM +0.063 …` text block in the bottom-right
  of panel A was drawn across the `MIN_PREDICTION` bar and its whisker.
* **Cohort invisible.** Nothing in the artwork distinguished this figure's 143-extractant
  cohort from the 99-extractant cohort of the other main-text k-shot panels; the only
  clue was a 6.8 pt grey title.
* **Type sizes 6.0–6.8 pt** throughout (tick labels, legend, annotations, both titles) —
  marginal on screen, unreadable at 180 mm print width.
* Panel letters placed with hand-tuned offsets (`dx=-0.42`, `dx=-0.08`).
* Panels A and B shared a y order but nothing tied them visually; the eye had to count
  rows to carry a policy from one panel to the other.

## Changes

**Naming.** Every registry key is replaced by what the rule does. The mapping is
one-to-one and is reproduced here so any bar can be traced back to the registry; it is
also written into `values.json` as `policy_label_map`. Each phrase describes the
policy's *k* = 1 behaviour, because this figure is entirely at *k* = 1:

| registry key | figure label | source |
|---|---|---|
| `ORACLE` | best in hindsight (not deployable) | `gen8.kshot.make_oracle_policy` |
| `SURROGATE_ORACLE` | best on ranker's target (not deployable) | `scripts/gen9_acquisition.py`, argmin of `oracle_deviation` |
| `MEDOID` | medoid candidate | `gen8.kshot.policy_medoid` (L1 medoid of the pool) |
| `CENTRAL` | nearest the mean condition | `gen8.kshot.policy_central` |
| `LEARNED_BLEND[geometry]` | learned ranker (blended with centrality) | `gen9.acquisition.BlendedAcquisition` — the anchor is selected per fold between `dist_medoid_l1` and `dist_centre_l2`, so the label names the family rather than one member |
| `LEARNED_SCALAR[geometry]` | learned ranker (regression) | `gen9.acquisition.ScalarAcquisition` |
| `MEDIAN_PREDICTION` | median predicted extraction | `gen8.kshot.policy_median_prediction` |
| `LEARNED_RANK[geometry]` | learned ranker (pairwise) | `gen9.acquisition.PairwiseAcquisition` |
| `MAX_CONDITION_COVERAGE` | smallest covering radius | `gen8.kshot.policy_coverage` (greedy k-centre) |
| `MIN_GP_DESIGN_VAR` | lowest design variance | `gen9.acquisition.policy_min_gp_design_variance` |
| `MID_ACID` | median acidity | `gen8.kshot.policy_mid_acid` |
| `MAX_PREDICTION` | strongest predicted extraction | `gen9.acquisition.policy_max_prediction` |
| `MAX_ENSEMBLE_SD` | most uncertain candidate | `gen8.kshot.policy_max_sd` |
| `RANDOM` | random choice | `gen8.kshot.policy_random` |
| `MIN_ENSEMBLE_SD` | least uncertain candidate | `gen8.kshot.policy_min_sd` |
| `FARTHEST_FROM_EXISTING` | farthest from the pool centre | `gen8.kshot.policy_farthest` |
| `MIN_PREDICTION` | weakest predicted extraction | `gen9.acquisition.policy_min_prediction` |

Two of these are worth stating explicitly. `FARTHEST_FROM_EXISTING` is named for its
*k* ≥ 2 behaviour (farthest from the already-selected points); with nothing yet selected
it takes the branch that maximises distance from the pool mean, which is what this figure
measures, so the label says "farthest from the pool centre". `MAX_CONDITION_COVERAGE` is a
greedy k-centre rule and at *k* = 1 it *minimises* the pool's covering radius, so calling
it "widest coverage" would have inverted its meaning; the label says "smallest covering
radius", which is why it sits in the centrality family. The script asserts that no
registry key survives into any drawn text artist, so this cannot silently regress.

**The paired interval gets its own panel.** I split rather than re-encoded. The two
quantities answer different questions — "where does one measurement land you" (a level,
with a natural zero and a natural reference at the zero-shot error) and "is this rule
better than choosing at random" (a paired contrast on identical pools, with a natural zero
at no difference). They have different units of comparison and different reference lines,
and an annotation saying "these whiskers are paired differences drawn about each level"
would be a caption asking the artwork to be forgiven. Panel B plots the same numbers as
the old whiskers, unchanged: bar = paired point estimate, whisker = BCa 95 %, sign
convention "macro MAE removed vs random choice" so that positive is better; a solid rule
at zero; `RANDOM` itself drawn as a blank row marked "reference". The oracle bars stay in
panel B — they compress the deployable rules into the left half, and that compression is
the honest picture of how much headroom is left.

**Layout.** Three panels sharing one vertical frame, `y`, `ylim`, row bands and order; the
sharing is asserted at the end of `main`. Alternating row stripes run across all three so
a policy named once in panel A can be carried into B and C without counting rows. Panel A
carries a value label at every bar end, in a region that is empty by construction (longest
bar 0.781, axis to 1.15, rule at 0.98).

**Oracle arms.** Three redundant markers, all in the artwork: a green tint band behind the
two rows in every panel, a white hatch on the bars, and the words "not deployable" inside
both y tick labels. The legend entry reads "oracle — reads the answer, not deployable".

**Legend** moved out of the data entirely, to `loc="outside lower center"` beneath the
panels, six entries in three columns, space reserved by constrained layout.

**Reference lines** are now `vlines` limited to the bar rows rather than `axvline` across
the whole panel, which leaves the reserved headroom at the top and bottom genuinely empty:
the top strip holds "zero-shot error 0.98" (panel A), "worse | better than random"
(panel B) and "random choice" (panel C); the bottom strip of panel A holds the cohort
block. No annotation is placed at a hand-chosen figure coordinate.

**Cohort in the artwork.** "cohort: 143 held-out extractants" in ink, with "5 split seeds
× 8 pool draws" and "the multi-*k* figures use 99 of them" under it in grey, plus "% of
143" in panel C's axis label. I deliberately did not name the other figures by number:
`METRIC_AUDIT.md` lists C-COMMON as feeding Fig 2, 4A, 4C, 5A, S3 and S6, so a hard-coded
"Figs 2, 4A, 5C" in the artwork would be a numbering claim this script cannot verify.

**Typography.** Base 8.0 pt from `pubstyle`: axis labels 8.5, tick labels 7.5,
annotations and value labels 7.0, panel letters 9.5. Nothing below 7.0 pt (was 6.0–6.8).
Panel letters placed by `pubstyle.add_panel_letters` from the measured tight bounding box.
Export: 600 dpi PNG and a fully vector PDF (no raster XObjects) from one figure object.

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 67 text artists checked**
(`figure.lint.json`), and the render was inspected by eye at full size, at 90 mm
reproduction width, and in two full-resolution crops.

## Scientific integrity

**No change.** Same file, same `geometry` feature set, same 17 policies, same 143-extractant
cohort, same 5 seeds × 8 pool draws, same per-extractant averaging, same macro mean, same
`paired_chemotype_bootstrap` with the same BCa intervals, same oracle arms. The script
still asserts a bit-level reproduction of `realised_summary.csv` for every policy
(`reproduces realised_summary.csv for all 17 policies: True`) and refuses to draw
otherwise. Spot values against the published figure: `MEDOID` 0.6286 / 28.5 % harmed,
`RANDOM` 0.6917 / 34.2 %, `ORACLE` 0.4628 / 7.7 %, zero-shot 0.9796; `MEDOID` vs `RANDOM`
+0.0631 [+0.0280, +0.1050], 98/143 — identical to `figures/derived/fig6_values.json`.

Nothing was removed from the figure: the whiskers that used to sit in panel A are panel B,
drawn from the same `paired` table.

Two observations, neither acted on:

1. *Not a bug, but worth recording.* The published panel A drew the paired interval as
   `[mae − bca_high + point, mae − bca_low + point]`. That is arithmetically the paired
   interval reflected through the policy's own level, and it is correct as such — but the
   reflection means a *wider* paired interval renders as a whisker whose left end moves
   right, which is the opposite of the usual reading. This is a presentation hazard, not a
   numerical error, and it is the reason for the split into panels A and B.
2. `FARTHEST_FROM_EXISTING` and `MAX_CONDITION_COVERAGE` are registry names for *k* ≥ 2
   behaviour; at the *k* = 1 of this figure both take a different code branch (distance
   from the pool mean, and the 1-centre respectively). The published figure's family
   assignment — `FARTHEST_FROM_EXISTING` as "extreme", `MAX_CONDITION_COVERAGE` as
   "centrality" — is consistent with that branch and is left exactly as it was.

## Paper role

**Results** — the acquisition result: which experiment to run, and which signals do not
tell you.

## Main message

Choosing *where* to measure moves macro MAE by 0.15 log units across seventeen rules on
identical candidate pools, and the winner is plain geometric centrality: the medoid
candidate removes 0.063 [0.028, 0.105] more error than choosing at random, while the
model's own uncertainty is worth nothing (+0.004 [−0.023, +0.036]) and deliberately
measuring far from the rest of the design is significantly worse than random. Even the best
deployable rule still makes 28 % of extractants worse than not measuring at all.
