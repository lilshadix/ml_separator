# Figure 5 — Training coverage, not model capacity, is the binding constraint

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `figures/scripts/plot_fig5_generalization.py`
Original render: `figures/main/Fig5_generalization.png`
Caption: `figures/FIGURE_CAPTIONS.md`, "## Figure 5"

Input data/results:

* `runs/gen6_expA_5seed/hard_chemistry_metrics.csv` — panel A. gen6 Experiment A, one
  shared cohort built at `min_cells = 3`, folds holding out whole Tanimoto-0.7
  super-clusters; an arm is a row mask over the *training* side only, so the held-out rows
  are byte-identical between arms. Feature set `MC_lig2d_ext_massaction`, 5 split seeds.
* `runs/gen6_expA_5seed/contrast_summary.csv` — panel B. Paired chemotype-block BCa 95 %
  intervals over the 131 ECFP clusters (105 / 42 at the two distance endpoints).
* `runs/gen10_final/final_locked/kshot_detail.parquet`, derived by
  `figures/scripts/prepare_kshot_tables.py` into `figures/derived/kshot_per_ligand.csv` —
  panel C, 99 held-out extractants of the k-shot common cohort, final pipeline.
* `runs/gen10_final/budget_simulation/marginal_gains.csv` — used as a *check*: the script
  recomputes all 15 published marginal-gain numbers from the per-ligand table under
  gen10's own adapter rule and refuses to draw the figure unless every one matches to
  1e-9. Current status: **15/15 pass**.

## Problem

The restructure itself was the main defect, and it was already on record:
`FINAL_FIGURE_REPORT.md` §4 names Figure 5 the weakest main figure and its distance-tercile
panel the weakest panel, and recommends promoting the controlled experiment to the front.
In the published render the reader met the observational panel first — a panel whose own
headline contrast, far − near at *k* = 0, is +0.35 [−0.04, +0.66] and does not exclude
zero — and only reached the interventional evidence after it.

Defects I saw in the render itself:

* **Reading order.** Panel A (terciles, observational, 17 chemotypes per band) led;
  panels B and C (controlled, byte-identical held-out rows) followed. The strongest
  evidence was read last.
* **No grouping.** Three equal panels in a row, with nothing to say that two of them are
  one controlled experiment on one cohort and the third is a different model on a
  different cohort. `FIGURE_AUDIT.md` §5 flags exactly this ("a reader who assumes one
  model across the figure will be misled").
* **Legend over data.** The three-entry key in the bars panel sat inside the bars region
  at the upper left, above the first group only because that group happened to be short.
* **Cramped two-line tick labels.** The paired-interval panel's row labels
  ("full coverage / vs base, all", "row-matched / control, all") were set at 6.2 pt in a
  gutter too narrow for them.
* **Type sizes 5.8–7.0 pt** throughout — the panel straplines at 6.8 pt, the far − near
  caveat at 5.9 pt, legends at 5.9–6.0 pt.
* **Straplines colliding with panel letters.** Letters were placed by hand-tuned axes
  offsets (`dx = −0.145`, `−0.19`, `−0.46`), which put the letter on the same line as the
  grey strapline and, in the tercile panel, hard against the top y tick label.
* **Repository wording leaking into the artwork.** "full coverage vs base", "row-matched",
  "vs shuffled target" are arm-registry shorthand, not reader-facing names.
* **Overplotted intervals.** In the tercile panel all three terciles' marginal intervals
  were drawn at the same *k*, on top of each other, at alpha 0.55 — at *k* = 0 three
  intervals occupy one vertical line and cannot be attributed.
* **Clipped intervals.** The tercile panel's y axis started at 0.28 while the mid
  tercile's interval reaches 0.193 at *k* = 5 (and 0.236, 0.210 at *k* = 2, 3), so three
  intervals ran off the bottom of the axes and looked open-ended.
* **Invisible error-bar caps.** `capsize` was set but the shared style sets
  `lines.markeredgewidth = 0`, so caps were never drawn.

## Changes

**Information design (the point of the pass).**

* Panels reordered as instructed: **A** = coverage-experiment levels (was B), **B** =
  paired improvements with BCa intervals (was C), **C** = distance terciles versus *k*
  (was A, demoted). Same three panels, same numbers.
* A and B now sit on a pale grey band and C does not, so the controlled experiment reads
  as one block. The band is measured from the two panels' tight bounding boxes and their
  panel letters after a draw, not hand-placed; the top 5.5 % of the canvas is reserved
  through the constrained-layout `rect` so the letters and the band stay inside the
  declared figure size (drawn extent is 1.00× the declared 7.09 × 3.45 in).
* Straplines carry the epistemic status rather than a topic: "controlled: training set
  varied", "controlled: paired gain", "observational, not an intervention". The repeated
  word plus the band is the grouping cue that survives being read at 180 mm.
* Panel letters are placed by `pubstyle.add_panel_letters`, which measures each axes'
  drawn tight bounding box (tick labels, axis label and strapline included) after
  rendering, so a letter cannot land on a strapline or a tick label.

**Panel A (coverage levels).**

* Reader-facing arm names: "Restricted coverage", "Full coverage", "Shuffled-target
  control" (registry: `BASE` ≥ 10 condition cells, `EXPANDED` ≥ 3, `EXPANDED_SHUFFLED`).
* Key moved clear of the data into an explicitly reserved band: the tallest bar plus its
  seed range reaches 1.67, the y limit is 2.45, and the left spine stops at the last tick
  (2.0) so the reserved strip is not read as part of the scale.
* A fourth key entry states what the whiskers are ("range over 5 split seeds"), so the
  panel no longer depends on the caption for that.
* *n* per held-out subset moved onto a second line of each x tick label, read from the
  data rather than written into the script (see "Scientific integrity" below).

**Panel B (paired gain).**

* Row labels shortened to one idea each ("all extractants", "$T<0.6$", "$T<0.4$",
  "row-count control", "shuffled control") at 7.5 pt with room for two lines, and the
  contrast itself named once in the strapline instead of three times down the axis.
* The two negative controls are separated from the three subset rows by a dashed rule and
  a grey group tag "controls, all extractants", and are given their own colours
  (row-count control in sky, shuffled control in grey) so they cannot be mistaken for the
  treatment rows.
* A "clusters improved" column (79/131, 62/105, 36/42, 71/131, 85/131) fills the empty
  right third of the panel. Those counts were already in `contrast_summary.csv` and in
  the caption; putting them in the panel answers "is one cluster carrying this?" without
  a reader having to leave the figure.
* The left spine is removed (the rows are categories, and a spine 0.075 units from the
  zero reference line reads as a second, meaningless vertical rule) and the bottom spine
  stops where the plotting region stops.

**Panel C (distance terciles, demoted).**

* The honest framing is kept and made *more* prominent, not less: the far − near contrast
  at *k* = 0 is drawn in a boxed note at 7.0 pt in ink (was 5.9 pt grey) and now says in
  the artwork that the interval **includes 0**. `tercile_contrast()` is unchanged —
  same unpaired chemotype-block bootstrap, same 5 000 replicates, same seed 8675309.
* Interval bars are offset by ±0.09 in *k* (1.5 % of the axis) so the three terciles'
  intervals can be told apart; the markers stay on the integer *k*, so no *k* is
  misreported. Alpha raised 0.55 → 0.75.
* y limits widened to 0.16–2.02 so that every interval is shown whole; the published
  render cut the axis at 0.28 and clipped three of the mid tercile's intervals.
* Terciles also carry distinct markers (near ●, mid ■, far ▲), so the panel does not rely
  on colour alone.

**Typography and export.** Base type 8.0 pt; nothing in the artwork is below 6.8 pt (the
"clusters improved" column header); tick labels 7.5 pt, axis labels 8.5 pt, legends
7.3 pt. Every axis label carries its unit, "macro MAE (log$_{10}$ $D$)" and "macro MAE
removed (log$_{10}$ $D$)". Error-bar caps are drawn explicitly with `capthick`. Export is
600 dpi PNG plus vector PDF from one figure object with one bounding box. The layout
linter that `pubstyle.save` runs over every text artist reports **layout OK, 57 text
artists checked** (`figure.lint.json`).

## Scientific integrity

**No number, cohort, arm, seed, metric or uncertainty method changed.** Panels A and B
read the same two gen6 Experiment A CSVs with the same feature-set filter
(`MC_lig2d_ext_massaction`), the same per-seed aggregation, the same seed-range whiskers
and the same pre-registered BCa contrasts. Panel C reads the same derived per-ligand
table, builds the terciles with the same `pd.qcut` on 1 − nearest-training Tanimoto, and
uses the same block bootstrap. The script re-derives gen10's published marginal-gain table
from the per-ligand data as a guard and **refuses to draw** unless all 15 numbers match to
1e-9; they do. `values.json` records every plotted number.

Changed in presentation only, and listed here so no one has to diff the scripts: panel
order; two colour assignments (the row-count control is now sky rather than grey, so the
two controls are distinguishable); marker shapes in panel C; the ±0.09 horizontal offset
of panel C's interval bars; panel C's y limits (widened, so intervals that were clipped
are now shown whole); the addition of the "clusters improved" column and of the
"range over 5 split seeds" key entry, both from values already in the source tables.

Qualifications this figure inherits and that the caption must keep carrying:

* Panels A/B and panel C are **different learners on different cohorts** — a controlled
  gen6 experiment (macro = one vote per ECFP cluster, 131 clusters) versus the deployed
  gen10 pipeline (99 held-out extractants). They are deliberately not on one axis; the
  band and the straplines are the visual warning, the caption is the explicit one.
* 57 of the 131 scoring units in panels A/B are ECFP clusters composed entirely of
  extractants the expansion added, which the restricted arm structurally cannot serve; on
  the 91 extractants the restricted arm could already cover, the effect is +0.019. The
  magnitude is therefore a property of this cohort's composition. Not drawable in the
  panel without inventing a sixth row; it stays in the caption and in
  `figures/METRIC_AUDIT.md` §5.
* Panel B's $T<0.6$ row has a BCa interval of [−0.010, +0.372] that crosses zero. It is
  plotted as it is, against a zero reference line, with no annotation suppressing it.

**Suspicion, reported rather than silently fixed.** In the original script the *n* printed
under the bar panel's x tick labels comes from the `ns` list left over from the last
iteration of the arm loop (`EXPANDED_SHUFFLED`) rather than from a per-endpoint quantity
computed independently of the arm. The three values are identical for every arm
(152 / 87 / 36 held-out extractants, because the held-out rows are byte-identical across
arms), so the published numbers are correct — but the code makes that a coincidence of
loop order rather than a guarantee. The refined script prints the same three numbers, now
collected per (arm, endpoint) and asserted to be identical across arms before they are
drawn; if an arm ever disagreed the script would stop rather than label the axis with
whichever arm happened to be last. No plotted value changes.

## Paper role

**Results.** With the reorder it is the figure that answers "does it generalise, and what
would fix it?" — and it now answers with the controlled experiment first.

## Main message

Holding the learner, the folds and the held-out rows fixed and changing only which
extractants are allowed into training removes 0.163 [0.012, 0.295] of macro MAE overall
and 0.463 [0.251, 0.685] on the chemistry furthest from the training set, while a
row-count-matched arm reproduces the gain and a shuffled-target arm does not — so the
binding constraint is chemical coverage of the training set, not model capacity. The
observational distance split points the same way but cannot establish it on its own: its
far − near contrast at *k* = 0 is +0.35 [−0.04, +0.66].
