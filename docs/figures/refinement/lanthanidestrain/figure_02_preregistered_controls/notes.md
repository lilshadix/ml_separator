# Figure 2 — The three pre-registered contrasts, and the control that must come out flat

## Source

Repository: `github.com/mironovb/lanthanidestrain` (vendored at commit
`1cdeac39d0b018529a735f967d3c83120d04c714`, see `data/PROVENANCE.json`)
Original script: `automl/figures_pi_email.py::fig_control`
Original render: `original.png` in this folder, copied from
`automl/reports/figures_ORIGINAL_BACKUP/email_figC_control.png`

Input tables — the brief lists six candidates; the upstream function reads exactly one:

* `automl/reports/stack_test.csv` — **the only table `fig_control` touches**
  (`_need("stack_test.csv")`). Rows `1_primary`, `2_control`, `3_decisive`; the fourth
  row `desc_S2` is descriptive and is drawn by neither the original nor this version.
* `best_stack.csv` is read by the *sibling* function `fig_result`, not by this one.
  `control_factorial.csv`, `control_blend.csv`, `control_attribution.csv` and
  `control_cells.csv` belong to the earlier factorial (`automl/figures_topo.py`) and are
  not inputs here. All were checked before being excluded.

Method and prose context: `automl/reports/STACK_PREREGISTRATION.md` (the design),
`automl/reports/STACK_RESULTS.md` (the verdict), `automl/reports/CONTROL_RESULTS.md`
(where the repaired baseline and the matched control come from),
`automl/topo/stack_test.py` (the analysis that wrote the table).

Cohort: out-of-fold predictions under leave-extractants-out CV (5 folds × 3 repeats);
**149 extractants**; paired resampling over those extractants, 400 draws, seed 0.

## Problem

Defects I saw in `original.png`:

* **The title carried the whole argument, at the largest type on the page.** Two lines of
  ~15 pt above the axes — "A second model does not help on its own; the 3D one does" plus
  both point estimates and both verdicts. The artwork underneath could not be read without
  it: the three rows were labelled but the *structure* of the comparison was invisible.
* **Nothing showed what each contrast holds fixed.** The y labels read "add the 3D encoder
  to the fingerprint network", "add a matched model with no 3D input instead", "3D encoder
  against that control, in the same slot". That third phrase — "in the same slot" — is the
  entire logic of the design, and the reader had to reconstruct which two systems were
  being differenced, and what they shared, from prose. A referee cannot audit a control
  they have to imagine.
* **The multiplicity annotation was orphaned and misplaced.** "corrected for multiple
  looks, this one spans zero" was drawn at `x = lo − 0.002`, `ha="right"`, i.e. it ran
  *leftwards* from the bottom row's lower bound, ending up under the y-axis region and
  above the x-axis label, attached to no row in particular.
* **"corrected for 3 tests" in the x-axis label was ambiguous in the worst possible way.**
  The figure has exactly three rows, so "3 tests" reads as "corrected for the three
  contrasts shown". It is not: `stack_test.py` corrects for **three looks at one question**
  — the S0 attempt, the S2 attempt, and this stack (`STACK_PREREGISTRATION.md` §3).
* **The thin corrected interval was unreadable at its ends.** Drawn at `lw=1.1` with no end
  caps and no distinct colour, while the thick 90 % bar used `solid_capstyle="round"`,
  which draws each interval half a line width past its own endpoint. Whether the corrected
  bound sat left or right of zero — the only thing rows 1 and 3 differ on — was decided by
  ~1 px of a rounded cap.
* **No sample size anywhere.** Neither the number of extractants resampled nor the number
  of bootstrap draws appeared in the artwork, though both set the interval widths.
* **The metric was under-specified.** "Δ adjacent-pair R²" does not say R² *of what*; this
  project also reports overall log D R², and the two move in opposite directions under the
  scaler change that produced this baseline (`CONTROL_RESULTS.md` §4.2).
* **No sign convention.** Nothing said which direction of Δ favoured which system.
* Non-`pubstyle` house colours (`C["blue"]`, `C["red"]`), a wasteful 8.2 × 3.2 in canvas
  with a third of the width empty, and 7.5–9.5 pt annotation type.

## Changes

**Panel A is new: the design, drawn.** Each contrast is two systems side by side, each
system a stack of two boxes — the part every contrast holds fixed on top (grey), the slot
that changes underneath (colour). Reading down the second column gives the whole
pre-registration: *nothing added* → *3D encoder*, *nothing added* → *matched no-3D model*,
*matched no-3D model* → *3D encoder*. The third row is visibly the only one in which both
sides have the slot filled, which is what "in the same slot" meant. Row 2 is tagged
*must come out flat* in the artwork, so the null is named as a null before its interval is
seen, not after.

**Naming.** No repository identifier reaches the artwork; the script asserts this over
every drawn `Text` artist and exits if one does. The mapping, with the upstream source for
each:

| identifier | figure label | what it is |
|---|---|---|
| `repaired` | fingerprint network | the published FCNN baseline — sklearn MLP on ECFP + RDKit descriptors — with `QuantileTransformer` swapped for `StandardScaler`, 16-seed ensemble (`CONTROL_RESULTS.md` §4.2, `oof_fcnn_std_scaler_ens16.parquet`) |
| `S0` | 3D encoder | simplicial message-passing network over the GFN2-xTB complex, trained with the pairwise-contrast objective, 16 seeds (`CONTROL_RESULTS.md` §1) |
| `T0w` | matched no-3D model | the pre-registered control: same harness, folds, seeds and objective as the 3D encoder, encoder removed, wide head (`CONTROL_PREREGISTRATION.md` §5) |
| absent second model | nothing added | the reference in contrasts 1 and 2 is the fingerprint network alone |
| `blend(a, b)` | "blends its two models at a weight fitted per held-out extractant" | nested leave-one-extractant-out weight, grid 0–1 step 0.05 (`stack_test.py::nested_blend`) |
| `1_primary` / `2_control` / `3_decisive` | PRIMARY / CONTROL / DECISIVE, each with its question in words | the pre-registration's own names for the three endpoints |
| `sel_adj_logSF_r2` | adjacent-pair separation R² | R² of the log₁₀ separation factor between neighbouring lanthanides, within one extractant and condition block (`evaluation.adjacent_pair_arrays`) |

**The multiplicity correction is labelled for what it is.** Legend: "corrected for three
looks"; footnote: "corrected for three looks **at this question** (Bonferroni-style)". The
number 3 is no longer next to three rows without qualification. The one contrast whose
verdict the correction removes is annotated on its own row, directly beneath it:
*corrected interval spans zero*.

**Intervals are drawn to their real extent.** Both intervals now use butt caps — a rounded
cap on a 3.6 pt line overstated each bound by ≈ 0.001 R² — and the corrected interval
carries end ticks, so a reader can see that +0.0042 (row 1) is right of zero and −0.0095
(row 3) is left of it without measuring pixels.

**Axis and footnote carry metric, sign and n.** Axis: "Δ adjacent-pair separation R²
(tested − reference)". Footnote, four short grey lines: the pre-registration fact, the
metric definition, the sign convention and the blend rule, and the two interval
definitions with 400 draws and 149 extractants. The x-axis label and the bottom spine are
trimmed to the data range so the value column to their right reads as a table column, not
as part of the scale.

**Levels are deliberately absent.** The paired bootstrap Δ is the mean of the resampled
difference on shared pairs and is *not* the difference of two whole-set scores: for
contrast 1 it is +0.0351 against a level difference of +0.0305 (both written to
`values.json` so the discrepancy is on the record). No bar height on this figure can be
subtracted from another to obtain a contrast, because there are no levels on it. The levels
live in the sibling figure (`fig_result` / `email_figB_result`), which is where the
upstream code puts them and says so.

**Layout and type.** `pubstyle` throughout: Okabe-Ito blue for the 3D-encoder rows,
vermillion for the control row (CVD-safe pair, and each row is also named in words), grey
for everything held fixed; 7.09 in (180 mm) × 3.78 in with `constrained_layout`; base 8.0
pt, nothing below 6.8 pt; panel letters from `PS.add_panel_letters` after every axes is
populated; the question column in panel A is placed from the *measured* width of the widest
role word rather than a hand-tuned offset. 600 dpi PNG and a fully vector PDF (no image
XObject) from one figure object.

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 47 text artists checked**
(`figure.lint.json`). Inspected by eye at full size and in two full-resolution crops
(footnote block, legend + row 1).

## Scientific integrity

**No number, cohort, statistic or method changed.** The figure reads the same single table
the upstream function reads, plots the same three rows in the same pre-registered order,
and reproduces every value to the four decimals the original printed: +0.0351
[+0.0167, +0.0646] / corrected [+0.0042, +0.0660]; +0.0055 [−0.0174, +0.0184] / corrected
[−0.0176, +0.0287]; +0.0296 [+0.0050, +0.0654] / corrected [−0.0095, +0.0687]. Nothing was
added to or removed from the plotted set; `desc_S2` is descriptive and was not drawn
upstream either.

Four things found while reading the upstream code and reports. None is fixed here; all are
recorded.

1. **`STACK_RESULTS.md` says "four-test corrected" where the code corrects for three.**
   The prose in the verdict section reads "Contrast 3's four-test corrected interval is
   [−0.0095, +0.0687]" and "corrected for four attempts", but that interval is exactly what
   `stack_test.py` produces with `_corrected(..., n_tests=3)`, and
   `STACK_PREREGISTRATION.md` §3 fixes the correction at three looks (S0, S2, this stack).
   The *number* is the code's; only the report's wording disagrees. The figure says three,
   which is what was computed. **A documentation defect, not a numerical one — flagged, not
   fixed.**
2. **The "corrected" interval is a normal approximation, not a bootstrap quantile.**
   `_corrected` recovers a standard error from the 90 % width (`se = (hi − lo)/(2·1.645)`)
   and applies a z for `1 − 0.05/3`. The upstream docstring says so plainly ("An
   approximation, and said to be one"). The artwork therefore says "Bonferroni-style"
   rather than implying a resampled corrected interval.
3. **The resampling is a subsampling bootstrap, not the cluster bootstrap named elsewhere
   in the project.** `paired_adjacent_fast` draws `np.unique(rng.integers(0, n, n))`, so
   each draw is the *set* of extractants drawn at least once (≈ 63.5 % of them), not a
   multiset; `CONTROL_RESULTS.md` §5 identifies this and measures the published intervals
   as 12–29 % too narrow. The artwork says "400 resamples of the 149 extractants" and does
   not call it a cluster bootstrap. The numbers are left exactly as computed.
4. **`stack_test.csv` carries no per-contrast sample size.** The 149 extractants stated in
   the footnote come from two upstream statements about this same analysis —
   `STACK_PREREGISTRATION.md` §3 ("the weight is picked on the other 148 only") and the
   `149 x 21 x 4 blends` comment in `stack_test.py::nested_blend`. Contrasts 1 and 2 are
   computed on the fingerprint-network ∩ 3D-encoder and ∩ control index intersections
   respectively; I could not verify from the vendored tables alone that all three contrasts
   resample identically 149 clusters, only that the analysis as a whole runs on 149. Stated
   here rather than presented as a per-row n.

One further limitation the figure cannot carry, and which the caption should: the control
is *matched*, not *strongest*. `CONTROL_RESULTS.md` is explicit that the matched control
alone scores +0.2006 against the 3D encoder's +0.2382, and that against the repaired
baseline the 3D encoder alone is not distinguishable from zero (+0.0261 [−0.005, +0.076]).
That is a statement about levels, and levels are not on this figure by design.

## Paper role

**Results** — the attribution result: the adjacent-pair gain is specific to the 3D encoder
rather than to adding a second model, with the multiplicity caveat attached. Panel A
doubles as the methods diagram for the pre-registered design.

## Main message

Adding the 3D encoder to the fingerprint network raises adjacent-pair separation R² by
+0.0351 [+0.0167, +0.0646] while adding a matched model with no 3D input in the same slot
adds +0.0055 [−0.0174, +0.0184] — the control comes out flat, so the gain is not generic
ensembling. Head to head in that slot the encoder is worth +0.0296 [+0.0050, +0.0654],
which is suggestive rather than established: corrected for three looks at this question,
that interval spans zero.
