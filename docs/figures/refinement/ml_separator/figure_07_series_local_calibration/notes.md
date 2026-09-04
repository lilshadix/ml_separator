# Figure S7 — Calibration is series-local

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `scripts/gen8_figures.py::transfer_heatmap`
(rendered to `runs/gen8_architecture/figures/cross_series_transfer.png`, 7.4 in, 170 dpi)
Estimator: `scripts/gen8_calibration_geography.py::transfer_matrix` — **not re-run, not modified**
Input data/results:

* `runs/gen8_architecture/cross_series/transfer_matrix.csv` — 49 rows (7 × 7 curve types),
  columns `calibration_axis, target_axis, mae, zero_shot, n_ligands, gain`
* `runs/gen8_architecture/cross_series/transfer_detail.parquet` — the same estimator's
  per (split seed, extractant) output, used **only** to attach an interval; averaging it
  over seeds and then over extractants reproduces the CSV to 2.2 × 10⁻¹⁶ (asserted in the
  script)
* `runs/gen8_architecture/cross_series/one_shot_candidate_scores.parquet` — the
  extractant → Tanimoto-0.7 chemotype map used to block the bootstrap (143 extractants,
  72 chemotypes, 1:1 join)

Model `REC_ecfp_plus_recovered`, 5 split seeds, ligands with at least 4 held-out rows.

## Problem

The predecessor plots the **derived `gain` matrix** under the title *"Gain from one
measurement: measured series (rows) → predicted series (columns)"*, with an x label
"rows being predicted" and a y label "row that was measured". That is a directional claim
and the quantity cannot support it — see **Scientific integrity** below. On top of that:

* **Clipped axis title.** The title ran off the canvas; the words "(columns)" were never
  rendered at all.
* **Blank cells that look like zero.** The script filters `n_ligands >= 3`, so 8 of the 49
  cells were drawn white — the same white the diverging colour map uses for *zero gain* —
  and nothing on the figure said a filter had been applied.
* **No sample size, no uncertainty.** Median `n_ligands` over the plotted cells is 5 and
  30 of 49 cells have `n ≤ 5`. The three most saturated, most persuasive cells rested on
  n = 3, 8 and 3; the palest diagonal cell rested on n = 85. Nothing in the artwork said so.
* **A colour limit set by one thin cell.** `vmin/vmax = ±max|gain| = ±1.823`, which is the
  single `none → contact_time` cell at n = 3, so roughly 60 % of the red half of the bar
  was never used.
* **Registry identifiers as tick labels.** `metal_series`, `metal_concentration`,
  `contact_time`, plus a sentinel `none` that the figure never defined.
* **No unit on the colour bar.** The quantity is log₁₀ *D* units.
* **The diagonal — the whole point — was unmarked**, and three of the seven diagonal cells
  are at or below zero gain — `none` −0.118 and `metal_concentration` −0.045 are negative
  and `temperature` +0.025 is indistinguishable from zero — which the presentation hid.
* 7.5–10 pt type, 170 dpi, no vector companion.

## Changes

* **Panel A** now shows the *symmetric* quantity — the post-calibration one-shot MAE — as
  a lower triangle, with a note in the empty upper triangle saying it is symmetric by
  construction and quoting max |M − Mᵀ| = 4.4 × 10⁻¹⁶. Neither axis is directional; both
  are labelled "curve type", and the x label defines `none` ("rows belonging to no
  titration curve"). Tick labels are the spaced English forms used by Figure S5.
* **Every one of the 28 lower-triangle cells is drawn**, each carrying its value and its
  `n`. Nothing is blank, so nothing can be confused with a mid-scale value. Cells below
  the interpretation threshold carry a distinct hatch and parenthesised value, with a key.
* **Coverage filter raised from `n ≥ 3` to `n ≥ 5`** for *interpretation* only — and the
  change is stated in the artwork (the key reads "fewer than 5 extractants") rather than
  applied silently. 16 of the 28 cells clear it.
* **Sequential colour scale, 0 → 1.5 log₁₀ *D* units, with an `extend` arrow.** The limit
  is robust: 25 of the 28 cells fall below it and the three that do not (2.18, 2.23, 1.61)
  rest on one extractant each. The colour bar carries the unit and the direction
  ("lower is better"); it is placed inside the empty upper triangle, which both removes
  the wasted space and keeps the cells close to square.
* **Diagonal cells carry a heavy outline** and a key entry, because "measure inside the
  series you intend to predict" is what the figure is about.
* **Panels B and C are new** and answer the question the MAE matrix alone cannot: did the
  measurement beat *no* measurement? Each row is one cell of panel A, drawn as the
  calibrated error (filled circle, 95 % CI) against the zero-shot baseline(s) (open
  squares), with the connector coloured by whether the error fell or rose. Panel B is the
  seven within-series cells; panel C is the ten cross-series pairs that clear `n ≥ 5`.
* **Confidence intervals added** (95 %, chemotype-blocked, resampled over extractants).
  Intervals that run past the axis are drawn to the edge and marked with a triangle,
  with a key entry — never silently truncated. True bounds are in `values.json`.
* Base type 8.0 pt; nothing on the figure is below 6.0 pt. 600 dpi PNG plus a vector PDF
  from the same figure object. Layout is machine-checked: `pubstyle.save` runs a
  bounding-box linter over every text artist. Current status: **layout OK, 133 text
  artists checked** (`figure.lint.json`).

## Scientific integrity

**The presentation changed; no number did.** The estimator was neither re-run nor edited,
and every value drawn is read straight from `transfer_matrix.csv`. The reframing is
required, and here is why.

### The transfer matrix is symmetric, so the predecessor's arrow is unsupported

`transfer_matrix` forms, per ligand,

```python
residual = block["log_D"].to_numpy(float) - block["prediction"].to_numpy(float)
diff     = np.abs(residual[:, None] - residual[None, :])   # <- symmetric by construction
np.fill_diagonal(diff, np.nan)
sub      = diff[np.ix_(source_mask, target_mask)]
"mae":     float(np.nanmean(sub))
```

`diff` is symmetric, so the mean of the `(source × target)` block is **identically** the
mean of the `(target × source)` block. The line responsible is

```python
diff = np.abs(residual[:, None] - residual[None, :])
```

together with the block mean `float(np.nanmean(sub))` two lines later. Verified on the
stored matrix (reproduced in `values.json` as `symmetry_check_max_abs_M_minus_MT`):

| quantity | max &#124;M − Mᵀ&#124; | |
|---|---|---|
| `mae` (post-calibration) | **4.441 × 10⁻¹⁶** | symmetric |
| `n_ligands` | **0** | symmetric |
| `zero_shot` | 1.895 | directional |
| `gain = zero_shot − mae` | 1.895 | directional |

```bash
.venv/bin/python -c "
import pandas as pd, numpy as np
d = pd.read_csv('runs/gen8_architecture/cross_series/transfer_matrix.csv')
M = d.pivot(index='calibration_axis', columns='target_axis', values='mae')
c = sorted(set(M.index) & set(M.columns)); A = M.loc[c, c].to_numpy(float)
print(np.nanmax(np.abs(A - A.T)))"
```

**Consequence.** Every off-diagonal asymmetry a reader saw in the predecessor — for
example `acid → contact_time` +0.69 against `contact_time → acid` +0.41 — comes from a
*different zero-shot baseline being subtracted from an identical post-calibration MAE*,
not from calibration transferring better in one direction. `zero_shot` for cell (A, B) is
the mean |residual| over the **B** rows of the shared ligand cohort; for cell (B, A) it is
the mean over the **A** rows. Same ligands, same calibrated error, different denominator
rows. The diagonal-versus-off-diagonal contrast — the claim the paper actually makes —
survives, because it does not depend on direction.

**How the refined figure handles it.** Panel A draws the symmetric quantity as a lower
triangle and says in the artwork that it is symmetric by construction, with the
4.4 × 10⁻¹⁶ number. Panel C draws both baselines of a pair on one row, joined to the one
calibrated error, and prints how far apart the two published "gains" therefore are; its
note states that the difference *is* the gap between the baselines and is the whole of the
apparent direction in the published matrix. The numbers printed at the right of panel B
are exactly the `gain` column of `transfer_matrix.csv`, so a reader can map the refined
figure back onto the old one.

**What was not done.** Fixing the estimator — fit the offset on the source rows, then
score the target rows, which *would* be directional — is a re-run, not a re-plot, and is
out of scope here. It is recommended, and is recorded in
`figure_refinement/ml_separator/SCIENTIFIC_DEFECTS.md` §1.

### The one thing added

`transfer_matrix.csv` carries no uncertainty, so 95 % intervals were computed from
`transfer_detail.parquet` — the same estimator's own per (seed, extractant) output —
by the repository's chemotype-block bootstrap (`_stats.block_bootstrap_mean`, blocks =
Tanimoto-0.7 chemotype). The point estimates are unchanged: the script asserts that the
re-aggregated table matches the frozen CSV to < 10⁻¹², and the observed drift is
2.2 × 10⁻¹⁶. Only the resampling is new.

### Limitations that remain, and could not be fixed from these data

* **The cells do not share a cohort.** A diagonal cell (A, A) averages every ligand with
  rows of type A; an off-diagonal cell (A, B) averages only the ligands that have rows of
  *both*. So `metal series` on the diagonal rests on 85 extractants while
  `metal series – contact time` rests on 1. The diagonal-versus-off-diagonal comparison is
  therefore *not* cohort-matched, and matching it would mean re-deriving the estimator.
  The per-cell `n` is printed in every cell so a reader can see this.
* **One interval collapses.** The four extractants behind `metal concentration` all belong
  to a single chemotype, so the chemotype-blocked bootstrap has nothing to resample and
  its interval is a point (`n_chemotypes` is recorded per cell in `values.json`). Its row
  in panel B is greyed as sub-threshold, but a reader could still misread a zero-length
  bar as precision. It is not.
* **Three intervals run past the axis** (up to 2.94) and are marked with a triangle rather
  than drawn in full; extending the axis to hold them would have compressed everything
  else into the left third. The bounds are in `values.json`.
* **The panel A colour and the panel B/C intervals answer different questions.** Panel A's
  colour is the post-calibration error only; whether that error is better or worse than
  doing nothing is *only* readable from panels B and C, and only for cells with n ≥ 5.
  For the twelve sub-threshold cells the honest answer from these data is "not known".

## Paper role

**Supplementary** — the evidence behind the main text's assertion that calibration is
series-local. The predecessor figure was never included; the claim was asserted with no
figure behind it.

## Main message

One measurement calibrates the series it was taken in and largely does not calibrate a
different one. Of the 16 cells with at least five extractants, the six on the diagonal are
each the smallest value in their own row and column (0.095–1.022 log₁₀ *D*) while the ten
off-diagonal ones run 0.935–1.434. Panel B sharpens it: measuring inside the series lowers
the error for five of seven series (metal series +0.33, extractant +0.24, acid +0.19,
contact time +1.77, temperature +0.02) but *raises* it for `none` (−0.12) and
`metal concentration` (−0.05), so "measure in the series you intend to predict" is
necessary and not sufficient. Panel C shows that in 8 of the 10 cross-series pairs a
measurement in the other series leaves at least one of the two scored row-sets worse than
no measurement at all; the only pairs that help in both directions are the two involving
contact time, which rest on 7 and 5 extractants.
