# Figure 3 — Amplitude compression on held-out titrations, and its repair

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `figures/scripts/plot_fig3_shape_recovery.py`
Original render: `figures/main/Fig3_shape_recovery.png`
Caption: `figures/FIGURE_CAPTIONS.md`, section "## Figure 3"
Example-selection rule: `figures/FIGURE_PLAN.md` §9 (declared before any curve was looked at)

Input data/results — all frozen run outputs, nothing transcribed:

* `runs/gen9_shape/shape/curve_shape.parquet` — per-curve slope, span recovery, shape MAE
  and within-curve Spearman; 775 curve × seed rows per arm (155 curves, 25 extractants,
  8 Tanimoto-0.7 chemotypes, 5 split seeds)
* `runs/gen9_shape/curves/curve_membership.parquet` — row → curve and its position on the
  extractant-concentration axis
* `runs/gen9_shape/phase1/oof_REC_ecfp_plus_recovered.parquet`,
  `runs/gen9_shape/relmono/oof_GEN9_REL_MONOLITH.parquet`,
  `runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet` — out-of-fold predictions
  for the three stages, split seed 104729 for panels A–C
* intervals from the repository's own chemotype-block bootstrap
  (`lanthanide_separation.gen8.inference`, via `figures/scripts/_stats.py`)

## Problem

Defects I saw in `figures/main/Fig3_shape_recovery.png`:

* **Three unrelated y-ranges in A–C.** A ran roughly −1.4…0.75, B −1.6…2.1, C −1.9…1.2.
  Each panel was internally correct, but the figure's subject is how much of the measured
  *amplitude* survives, and with three different scales a decade of log₁₀ *D* was a
  different number of millimetres in each panel, so nothing could be compared across them.
  Panel A's 2.02-decade curve and panel B's 3.50-decade curve looked the same height.
* **D ordered against E and F.** In D the rows ran Measured (bottom) → baseline → relative
  position → recomposition (top); in E and F they ran baseline (bottom) → recomposition
  (top). The same three stages appeared in opposite vertical order in adjacent panels.
* **Two marks that could not be told apart.** In E and F the point statistic was a bare
  `|` tick drawn inside the violin and its 95 % interval was a detached horizontal bar
  0.30 of a row *below* it. Nothing in the artwork said which mark was which, and the
  detached bar read as a second, separate quantity.
* **Type sizes.** Panel role titles 7.0 pt, value labels 6.0 pt, legend 6.0 pt, and the
  per-curve level-error note 5.9 pt — marginal at 180 mm print width.
* **A printed zero that is not zero.** Panel B showed "curve level error +0.00" for a
  level error of +0.0047.
* **Three repeated label columns.** D, E and F each carried the same long stage names as
  y-tick labels, spending roughly 40 % of the bottom row's width on the same four strings.
* **A silently truncated tail.** E's x-limit was 1.15; 6.6 % of the recomposed
  distribution lies beyond it, and the violin simply ran off the panel edge.
* Annotation-on-curve check: the original's "curve level error" notes sat in the empty
  lower-right corner of each of A–C and did **not** touch a curve. That was already
  correct and is preserved.

## Changes

**A–C — one shared y-range (`sharey`, −2.15 … 2.25), y-tick labels on A only.**
I chose a shared range over per-panel ranges with the range stated in words. Both fix the
stated defect, but a printed range asks the reader to do arithmetic in their head to
compare three panels, whereas a shared axis makes the comparison pre-attentive: the
measured black curve is visibly taller in B than in A, and the three coloured stage curves
are visibly the same small height in all three panels — which *is* the finding. The cost
is that panel C's stage curves are squashed into a narrow band around zero; that is the
honest depiction of "the repair fails here", and each panel still prints its measured span
so the amplitude can also be read as a number. The x-ranges stay per-panel (the three
titrations sit at different absolute concentrations) with symmetric 10 % padding.

* Legend moved into panel A's upper-left, which the shared y-limit leaves empty by
  construction; it lists the four series in the same top-to-bottom order as the rows of
  D, E and F, so one reading order serves the whole figure.
* Corner note now carries both the measured span and the per-curve level error, on two
  lines, at 7.0 pt, in the lower-right region that is empty in all three panels because
  every example curve rises to the right. The level error is printed to three decimals
  when two would round it to zero, so panel B now reads "+0.005", not "+0.00".

**D, E, F — one row order, one label column.** All three panels share a four-slot y axis
read top to bottom: **Measured, Baseline model, + relative position, + recomposition** —
the measurement first, then the stages in the order they are built. The stage names are
written once, on D. The measured row is tinted in all three panels so the slots line up by
eye, and in E and F that row holds the label for the dashed reference line
("measured = 1"), so the fourth slot is used rather than left blank. The three panels now
sit on a common set of rows and the freed width goes to the plots.

* The dashed reference now *starts* at the measured row and drops through the model rows,
  in all three panels: in D it drops from the measured median (2.57, named in the panel),
  in E and F from 1. That also keeps the reference clear of the value label printed above
  each row.
* **E and F glyph.** The point statistic and its interval became one composite glyph: a
  capped horizontal bar with a white-filled, colour-ringed dot on it — the conventional
  "estimate ± interval" mark. Each panel also carries a strapline naming the two marks
  ("dot: median, bar: 95 % CI"; "dot: mean, bar: 95 % CI"), so the reader never needs the
  caption, and the statistic is still named on the axis label as the project's metric
  tables define it (median for slope and span recovery, mean for Spearman).
* **D strapline** names the box parts the same way ("box: quartiles, number: median").
* **Axis labels** now all carry the statistic on line 1 and the definition/unit on line 2:
  "fitted slope, median (Δlog₁₀ D / Δlog₁₀ [extractant])", "span recovery, median
  (predicted range ÷ measured range)", "within-curve Spearman ρ, mean (rank correlation,
  −1 to 1)".
* **E's truncated tail is now stated in the artwork** ("2 % of curves lie beyond 1.6") and
  the limit was moved from 1.15 to 1.60 so the reference at 1 is a line inside the panel
  rather than the panel edge, and only a thin 2 % tail is out of view instead of 6.6 %.

**Throughout.** Base type 8.0 pt, tick labels 7.5 pt, every annotation 7.0 pt, panel
letters 9.5 pt placed by `pubstyle.add_panel_letters` from each axes' measured tight
bounding box. Reader-facing names only — no `GEN9_SHAPE_RECOMPOSED` or
`REC_ecfp_plus_recovered` anywhere in the artwork; those ids live in `values.json`. Value
labels are given to two decimals. Okabe-Ito colours from `pubstyle`, each series also
carrying a line style and marker. Export is 600 dpi PNG plus vector PDF from one figure
object; `pubstyle.save` runs the bounding-box linter first. Current status:
**layout OK, 81 text artists checked** (`figure.lint.json`).

Layout mechanics worth recording: constrained layout ignores `hspace` given to an outer
gridspec whose cells are themselves subgridspecs, so the gap between the two panel rows —
without which the bottom row's panel letters land on the top row's x-axis labels — is made
by an empty spacer row in the outer gridspec.

## Scientific integrity

**No number, cohort, seed, statistic, aggregation or arm changed.** Same three arms, same
`axis_label == "extractant"` cohort (775 curve × seed evaluations, 155 curves, 25
extractants, 8 chemotypes), same example seed 104729, same pre-declared §9 selection rule
producing the same three curves (`45d32f69aed37aee`, `e10df887ab66e002`,
`0bf12a4391a4e25b`), same chemotype-block bootstrap with the repository's default
replicates and seed. The plotted values reproduce the frozen caption exactly:

| panel | baseline | + relative position | + recomposition |
|---|---|---|---|
| D, median fitted slope (measured 2.57) | 0.12 | 0.52 | 1.02 |
| E, median span recovery | 0.051 [0.045, 0.086] | 0.210 [0.116, 0.289] | 0.423 [0.327, 0.544] |
| F, mean Spearman ρ | 0.582 [0.264, 0.708] | 0.840 [0.444, 0.880] | 0.886 [0.674, 0.914] |

Every plotted number is written to `values.json`.

Three display facts a reader should have, none of which is a change of method:

1. **E's x-limit is presentation, and I moved it** from 1.15 to 1.60. No value changed;
   the visible window did. 2.1 % of the recomposed distribution is still outside it and
   the panel now says so. The per-arm fractions beyond the limit are in `values.json`.
2. **F's violins extend slightly past ρ = ±1**, which is a kernel-density artefact at a
   hard bound, present in the original and left untouched — correcting it would change the
   density estimate, not the layout.
3. **D pools five correlated evaluations per curve** (155 curves × 5 seeds), as the caption
   already states; the boxes therefore describe the curve × seed population, while E's and
   F's intervals resample chemotypes.

**One suspicion, not acted on.** The caption says each A–C panel prints the per-curve level
error "which the recomposition leaves unchanged by construction". In the frozen outputs the
recomposed level error equals the baseline one exactly for the "large repair" curve
(+0.004669 both) but not for the other two: median case −1.9780 → −1.9535, repair-fails
−0.9004 → −0.8981. The differences are small and may simply mean the recomposition is
mean-preserving over the *series* rather than over each curve, but "unchanged by
construction" is not literally true for two of the three examples. I plotted exactly what
the original plotted (the recomposed stage's level error) and recorded the baseline level
error alongside it in `values.json` so the discrepancy is checkable; I did not change the
figure or the caption.

## Paper role

**Results** — the mechanism figure that sits behind the shape-recomposition step of the
pipeline quantified in Figure 2.

## Main message

A model that has never seen an extractant draws its titration almost flat — 5 % of the
measured amplitude and a median slope of 0.12 against a measured 2.57 — and telling the
model where each point sits inside its own titration window, then recomposing the curve,
recovers about 42 % of the amplitude and most of the within-curve ordering (ρ 0.58 → 0.89)
without touching the curve's level, which stays wrong by whole log units.
