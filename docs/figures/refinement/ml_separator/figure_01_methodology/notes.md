# Figure 1 — Task, held-out protocol and model

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `figures/scripts/plot_fig1_methodology.py`
Caption: the "## Figure 1" section of `figures/FIGURE_CAPTIONS.md`
Methods text it must support: `figures/METHOD_FIGURE_TEXT.md`

The figure is a schematic and contains no fitted numbers. The only quantities drawn are
cohort counts, and they are read from the frozen run outputs rather than typed in:

* `runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet` — rows, extractants,
  lanthanides, condition cells, Tanimoto-0.7 chemotypes (one split seed, the minimum one)
* `runs/gen9_shape/curves/curve_table.parquet` — the 1,176 curves
* `figures/derived/kshot_cohort.json` — the 99-extractant *k*-shot cohort, 5 split seeds,
  12 pool draws per seed

`values.json` records every count drawn; it is identical, key for key, to the frozen
`figures/derived/fig1_counts.json`.

## Problem

Defects I saw in the original render (`figures/main/Fig1_methodology.png`):

* **Type below the floor.** Box text 6.0 pt, chip captions 6.1–6.2 pt, the "target-free"
  note 5.8 pt, paragraphs 6.1 pt. Readable on a 4000 px screen render, marginal on paper
  at 180 mm and below the 6.5 pt floor everywhere.
* **Geometry pasted in by hand.** Every box, arrow endpoint, chip and caption was a bare
  axes-fraction literal — `box(axC, 0.417, flow_y, 0.158, flow_h, ...)`,
  `arrow(axC, (0.575, ...), (0.622, ...))`, `axC.text(0.045, 0.345, ...)`. Roughly 70 such
  numbers across three panels, mutually dependent: widening one flow box meant re-deriving
  the four that follow it and both of its arrows.
* **Collisions patched by nudging.** Panel A's chemotype row and panel B's "target-free"
  note had both been moved by hand after collisions; nothing checked that they still
  cleared their neighbours, and panel A's paragraph in fact overflowed the bottom of its
  own axes into the gap above panel C.
* **Text wider than its box.** Nothing measured text against the box holding it, so
  "curve mean of forest 1" ran into the padding of the recomposition box.
* **The colour code was not stated in the artwork.** The four information classes are the
  entire argument of the figure, but the legend read as four ordinary series entries at
  6.0 pt, and there was no statement of what a *filled* versus an *outlined* box meant —
  the distinction the figure relies on to show that an operation may read one class and
  not another.
* **Two near-identical fills.** Training blue and target-free sky were tinted 0.16 and 0.22
  out of white and were almost the same pale blue at print size.
* **Arrows that assert nothing.** The arrow into the held-out chemotype implied the
  held-out group came *from* the training groups; the acquisition step was drawn in the
  support colour although it reads no target, which is precisely the point being made.

## Changes

**Geometry.** Three primitives replace the pasted coordinates, and the module docstring
explains them:

* `Panel` wraps one axes and converts *points* to axes fractions in x and y, so gaps are
  written as "7 pt" and stay 7 pt at any figure size. Panel sizes are measured after one
  `fig.canvas.draw()`, once constrained_layout has settled.
* `Box` is a rectangle that knows its own edges, so every arrow runs between two named
  anchors (`design.south(0.86)` → `f2.west(0.74)`) and never between two typed points.
* `split(x0, x1, weights, gap)` divides a span into weighted columns that fill it exactly;
  the five-step flow, the eight pool chips, the six query chips, panel B's three columns,
  panel A's chemotype row and the key band are all one call each.

Box heights are derived from the text they hold, box text is wrapped to the box width
using real font metrics, and vertical stacks run off a top-down cursor, so moving one
block moves everything below it. Two QC reports print on every run: a **box-fit** report
(any line wider than its box) and a **content-bottom** report (any panel whose content has
spilled out of its axes). Both are clean, and the second is what caught panel A's
overflow, which the text-overlap linter alone had not.

**Typography.** Paragraph body is `PS.BASE` = 8.0 pt; panel headings 8.5 pt bold; box
text, chip captions and the key sub-line 7.5 pt. Nothing on the figure is below 7.5 pt,
against a 6.5 pt requirement. The figure is drawn at exactly 180 mm wide, so those are the
printed sizes.

**Information classes.** Four classes, in pubstyle's Okabe-Ito palette, used identically in
all three panels: training data (blue), target-free conditions of the new extractant (sky),
the *k* measured support targets (vermillion), held-out query targets (ink). The two "no
target" fills were re-tinted (0.24 blue / 0.15 sky) so they separate at print size while
staying visibly a family. The key is now its own gridspec row — it cannot overlap a panel —
with 8 pt two-line entries and swatches drawn by the same primitive as the boxes, plus one
grey line stating the grammar: *filled = information · outline and arrow = coloured by the
information they may read · dashes = a boundary no target crosses*.

**What the artwork now asserts.**

* Filled = the information itself; outlined = an operation, in the colour of what it may
  read. The acquisition step therefore carries a *sky* outline, not a vermillion one: it
  reads conditions and predictions and never a target, and the vermillion class starts
  exactly where a target is measured.
* Arrows are coloured by what travels along them, so the reader can follow where a measured
  target may and may not go.
* A dashed boundary marks each place nothing crosses: between training and held-out
  chemotypes in panel A, and between the candidate pool and the query rows in panel C,
  where it now runs the full height of the row including both captions.
* Panel C's two lanes end where they should: the query chips lead into the scoring box, the
  calibrated model comes down into it from above, and the pool leads up into the
  acquisition step.

**Copy.** Detail that made boxes overflow moved into the paragraph beneath them (the design
matrix's blocks); the "target-free" caveat moved *inside* its box, because as loose text it
had to be sky blue to mark its class and sky blue on white is too weak for 7.5 pt type. A
task line was added under panel A's heading — the figure is titled "task, ... and model"
and the task was not stated in the artwork. No claim was added, removed or weakened; the
leakage statements are the same three the Methods make.

**Export.** 600 dpi PNG and vector PDF from one figure object, panel letters placed by
`pubstyle.add_panel_letters`. Lint status: **layout OK, 33 text artists checked**
(`figure.lint.json`), box fit clean, all three panels inside their axes.

## Scientific integrity

**No change.** This figure contains no fitted number, no metric and no uncertainty; it is a
schematic. The twelve cohort counts it draws are read from the same three frozen files the
original read, by the same code path, and `values.json` matches
`figures/derived/fig1_counts.json` key for key: 5,248 rows, 152 extractants, 14 lanthanides,
2,055 condition cells, 79 Tanimoto-0.7 chemotypes, 1,176 curves, and a *k*-shot cohort of
99 extractants over 5 split seeds × 12 pool draws. No cohort, protocol, colour meaning or
statement of method was altered — only wording, type size, layout and the explicitness of
the colour code.

One judgement call worth recording: the corpus box in panel A is drawn in the training
class colour although, strictly, the corpus at that point still contains the chemotype that
will be held out — the class colours only become meaningful one row lower, after the split.
Drawing it neutral grey was rejected because grey is already the held-out query class and
the two would have clashed. This matches the original figure's choice and is not a change.

## Paper role

**Methodology** — Figure 1, the protocol figure. It is the figure a referee reads to decide
whether the *k*-shot numbers in Figure 2 are calibration or leakage.

## Main message

A held-out extractant has no analogue closer than Tanimoto 0.7 anywhere in training, and at
test time only two things about it are ever read: its candidate *conditions*, which carry no
target, and the *k* targets the acquisition policy bought — never the query rows it is scored
on, and never by the model, which is frozen.
