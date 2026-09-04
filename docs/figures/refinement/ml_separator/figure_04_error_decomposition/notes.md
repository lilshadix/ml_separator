# Figure 4 — Where the remaining error is, and how much of it is reachable

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `figures/scripts/plot_fig4_error_decomposition.py`
Caption: `figures/FIGURE_CAPTIONS.md`, "## Figure 4"
Input data/results:
* `runs/gen9_shape/recomposed/oof_all.parquet` — raw out-of-fold predictions of the final
  model, from which `figures/scripts/prepare_error_budget.py` rebuilds the oracle cascade
* `runs/gen10_final/error_decomposition/summary.json` — the frozen published cascade, used
  as the verification target (10 stage/metric values, tolerance 5 × 10⁻⁹)
* `runs/gen10_final/error_decomposition/decomposition.csv` — the seven-component budget
  drawn in panel B
* `runs/gen10_final/final_locked/kshot_detail.parquet` → `figures/derived/kshot_per_ligand.csv`
  — the realised k-shot values in panels A and C
* `runs/gen7_architecture/finalists/oof_predictions.parquet`,
  `runs/gen9_shape/relmono/…`, `runs/gen9_shape/recomposed/…` — the four zero-shot model
  stages of panel C, decomposed with `lanthanide_separation.gen6.metrics.decompose_level_shape`

The verification guard is kept and tightened. `prepare_error_budget.py` is re-run on every
render (1.6 s) rather than only when its output is missing, and this script re-asserts all
ten reproduction deltas before a single artist is created. Current run: **10/10 pass, max
absolute delta 0.0**, recorded in `values.json`.

## Problem

Defects I saw in `figures/main/Fig4_error_decomposition.png`:

* **Panel C was unreadable at the right-hand edge.** The four model stages occupy a box
  0.06 wide in level error and 0.08 tall in shape error, so the four written-out labels
  ("no ligand information", "baseline model", "+ relative position", "+ recomposition")
  were stacked on top of one another against the axis; "no ligand information" ran into
  the panel edge, and "baseline model" / "+ relative position" / "+ recomposition" formed
  a solid block of text with no way to tell which name belonged to which marker. This was
  the worst defect in the figure.
* **Two cohorts, one visual footing.** Panel B is measured on 152 extractants with one
  vote per ECFP cluster; panels A and C are 99 extractants with one vote per extractant.
  Nothing in the artwork said so — the fact lived only in the caption — so B's 0.520
  invited comparison with A's 0.507, which is a different quantity on a different cohort.
* **Panel A's group headings floated.** "deployable: measure k points" and
  "oracle bounds — not deployable" were free-floating 6.5 pt strings at y = 1.14, close
  enough to the 1.036 bar's value label (drawn at 1.054) to read as one cluster of text,
  and the only thing separating the two halves was a dotted vertical line.
* **The oracle hatch read as decoration.** Solid green fill with a white hatch reads as
  "another series", not as "this is not a thing you can do".
* **Panel B's second bar family was ambiguous.** The single value label per row was placed
  at the end of whichever bar was longer; for "series-local level" the deployable action
  (0.140) is longer than the component's own share (0.057), so the number 0.057 appeared
  at the end of the 0.140 bar.
* **Type sizes 5.6–7.0 pt** throughout — legible on a 2000 px screen render, marginal at
  180 mm print width.
* Panel letters placed by hand-tuned axes-fraction offsets (`dx=-0.062`, `-0.44`, `-0.22`).

## Changes

Panel A
* Reader-facing names only; the `k`-ladder and the oracle cascade are labelled in one or
  two short lines each, so nine categories fit without rotation or truncation.
* The two halves are separated by a physical gap plus a faint rule, not a dotted line
  alone; each half carries a coloured heading on its own rule at y = 1.155, above every
  bar top (max 1.036 + label ≈ 1.10). Headings and value labels can no longer collide.
* Oracle bars are now **hollow** — white fill, green edge, coarse green hatch — so they
  read as hypothetical in colour, in greyscale and at print size; the heading over them
  carries the words "not deployable" in the artwork.
* A faint y grid was added: the two values the reader must rank (0.493 deployable at
  k = 3 and 0.507 for the true per-extractant level) are 60 mm apart at print width and
  on opposite sides of the group gap.
* Y ticks stop at 1.0 and the left spine is bounded to its last tick, so no gridline runs
  between a heading and its rule.

Panel B — the cohort problem
* Solved in three independent channels rather than in the caption: a **tinted panel
  background** (the only tinted panel in the figure), a **title in ink rather than grey**
  reading "different cohort from A and C / 152 extractants · one vote per structural
  cluster", and an axis label that names the total it is a share of ("share of *this
  cohort's* 0.970 macro MAE"). A and C carry the matching statement ("99 extractants ·
  one vote per extractant"), so the reader is told the cohort on every panel, at the
  place they are looking.
* The chart form also differs deliberately: A is vertical bars, B is horizontal bars, so
  the two are not read as one series.
* Current-contribution values now sit in a right-aligned column in the reserved right
  margin, and the removable values sit at the end of their own vermillion bar in
  vermillion. Neither number can be attributed to the wrong series any more, including on
  the "series-local level" row where the removable bar overshoots.
* Two empty rows are reserved below the shortest bars by an explicit y limit so the
  two-entry legend sits in guaranteed-empty space instead of on the bottom bars.
* The duplicated left spine (a line at the axes edge, 2 px from the zero line) was
  removed; the zero line is the baseline.

Panel C — the text-crowding fix
* The four model stages became **numbered nodes 1–4 with a key**, the key placed in the
  upper-left corner, which is empty by construction: nothing in this study has a small
  level error together with a large shape error. Node numbers alternate left/right of the
  marker so that stages 3 and 4, 0.013 apart in shape error, cannot touch.
* The legend was deleted. Both series are **direct-labelled** in their own colour — the
  key doubles as the label for the model path, and "more measurements on the final model"
  sits in the lower-right corner, empty for the mirror-image reason.
* Both axis labels now carry their unit; the absolute-value bars are set in mathtext so
  they do not read as the letter "l".
* Y limits were raised to 0.685 to *reserve* the key's space rather than hope for it.

Whole figure
* Base type 8.0 pt (was ~7.0); no text below 7.0 pt (was 5.8–6.3 pt).
* Panel letters placed by `pubstyle.add_panel_letters`, which measures each axes' drawn
  tight bounding box, so a letter cannot land on a tick label.
* 600 dpi PNG and vector PDF from one figure object; `pubstyle.save` runs a bounding-box
  linter over every text artist before writing. Current status: **layout OK, 83 text
  artists checked**, `figure.lint.json` clean.
* Verified by eye at 180 mm print width (downsampled render) as well as at full size, in
  four render–look–fix cycles.

## Scientific integrity

**No number, cohort, arm, seed, metric or aggregation changed.** Every plotted value is
identical to the original script's output — checked field by field against
`figures/derived/fig4_values.json`:

| | panel A ladder | panel B components | panel C model stages | panel C k-shot |
|---|---|---|---|---|
| identical | yes (9/9) | yes (7/7, both columns) | yes (4/4) | yes (5/5) |

The oracle-cascade reproduction guard still runs before the figure is drawn and now runs
on *every* render: 10/10 stage/metric values reproduce `summary.json` with a maximum
absolute delta of 0.0.

Three things I did **not** change, and record here instead:

1. **Panel B, "series-local level": the removable bar (0.140) is longer than the
   component's own contribution (0.057).** This is what `decomposition.csv` says — the
   two quantities come from different contrasts (`per-series oracle offset vs per-ligand`
   for the contribution; `OFFSET_K1@k1 vs OFFSET_K3@k3` for the removal) and the
   components overlap, as the table's own note for component 7 states ("components
   overlap; this is a floor, not an exact remainder"). It is not a plotting bug, but a
   reader will notice it, and the figure does not currently explain it. I left the
   artwork faithful; the caption may want a clause.
2. **Panel C's two trajectories do not meet, although they end and start at the same
   model.** The blue path's stage 4 and the vermillion path's k = 0 are both the final
   recomposed model, yet they plot at (0.847, 0.527) and (0.861, 0.514). They are
   computed on different row sets and aggregations — the model stages from all
   out-of-fold rows via `decompose_level_shape`, the k-shot points from the k-shot
   harness's query rows (support points excluded), averaged per extractant over seeds and
   repeats. The original figure has the same property. Because of it, the vermillion path
   is labelled "more measurements on the final model" and *not* "added to stage 4", which
   would invite a question the figure cannot answer.
3. **Three-decimal value labels** were kept rather than reduced to 2–3 significant
   figures, because these are the numbers the paper caption quotes verbatim (1.036,
   0.654, 0.507, 0.504, 0.181, 0.520, 0.349, 0.192, 0.103) and a mismatch between figure
   and caption is worse than one extra digit.

## Paper role

**Results** — the error-budget figure: it states what is left after the frontier of
Figure 2 and how much of the remainder any action could reach.

## Main message

Half of the remaining macro MAE on an unseen extractant is a single per-extractant
constant, and one deployable measurement removes most of it — an optimally chosen
measurement (0.504) is worth as much as knowing the true per-extractant level exactly
(0.507), while a better zero-shot model moves only the curve shape and leaves that
constant untouched.
