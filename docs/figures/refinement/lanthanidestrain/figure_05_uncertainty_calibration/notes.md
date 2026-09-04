# Figure 5 — What "calibrated" means here, and which way the model is wrong

## Source

Repository: `github.com/mironovb/lanthanidestrain` (`main`, commit `1cdeac3`, 2026-08-24)

Two upstream renders, both called *calibration*, folded into one figure:

| upstream script + function | render | fate |
|---|---|---|
| `automl/figures.py::fig_uncertainty` | `automl/reports/figures/fig6_uncertainty_calibration.png` → `original.png` | rebuilt as panel A |
| `automl/figures_reanalysis.py::fig_calibration` | `automl/reports/figures/re_fig4_calibration.png` → `original_superseded.png` | **superseded** by panels B and C |

Input tables, all read from the vendored `figure_refinement/lanthanidestrain/data/`:

* `uncertainty_calibration.csv` — 5 rows, `quintile / mae / spread`; written by
  `automl/ensemble.py::main`, where `spread = P.std(axis=1)` over the base-model
  out-of-fold predictions and `mae = |nnls_oof − y|`, both averaged inside quintiles of
  `spread`. **Panel A.**
* `calibration_test_binned.csv`, `calibration_test_strict.csv` — written by
  `automl/topo/calibration_test.py`; `span_ratio = std(predicted difference) /
  std(measured difference)`, `r2` = adjacent-pair log SF R², and one `scale_gain` row per
  key carrying the cluster-bootstrap interval. **Panels B and C.**
* `calibration_test.csv` — named in the brief; **not used**, and it should not be. See
  "Scientific integrity".

Report: `automl/reports/CALIBRATION_RESULTS.md` (29 July 2026, erratum 30 July).
Narrative for panel A: `automl/reports/FINDINGS.md` §5.

Cohorts, which are **not** the same across panels and are named in the artwork:

* Panel A — 5,946 out-of-fold predictions of the 37-model stack, five equal quintiles of
  about 1,189. Absolute log D.
* Panels B and C — the deployed three-model stack, 905 adjacent pairs under the binned
  condition key and 1,417 under the strict one. Adjacent-pair log separation factor.

## Problem

Defects I saw in the two original renders.

**`fig6_uncertainty_calibration.png` (panel A's source):**

* **Half the table was not drawn.** `uncertainty_calibration.csv` has two columns —
  `mae` (the errors made) and `spread` (the error bar the model quotes). The figure
  plotted only `mae`. The whole point of a *calibration* plot is the comparison between
  the two, and the comparison is dramatic: the quoted spread runs 0.15–0.40 log D while
  the realised error runs 0.78–1.06. Dropping the second column turned a calibration
  figure into a five-bar monotonicity check.
* **A title that overstated what was drawn.** "Ensemble disagreement is a usable error
  bar" — but with only `mae` on the page there is no error bar to judge, and the
  repository's own `FINDINGS.md` immediately qualifies it: "Treat it as a three-bucket
  confidence flag (low / medium / high) rather than a calibrated variance." The
  qualification never reached the artwork.
* **The non-monotonicity was drawn but not acknowledged.** Q4 (0.87) sits below Q3
  (0.90). The bars show it; the title says the opposite.
* **No sample size anywhere.** Nothing said how many rows a quintile holds.
* **Y axis said "mean |error| in log D"** without stating that this is absolute log D
  rather than the adjacent-pair difference the rest of the study reports.
* **Type set for a screenshot, not a page.** 12 pt title, 10 pt axis labels, 9 pt ticks
  on a 5.6 × 3.8 in canvas — the title alone outweighs every number on the figure, and
  the whole panel carries five numbers.

**`re_fig4_calibration.png` (superseded by panels B and C):**

* **The legend sat on the data.** The two-entry legend was drawn at `loc="upper right"`,
  which put it across the `y = 1.0` reference line: the black rule runs straight through
  the words "after nested recalibration", and the "true spread" label sits on top of the
  legend's first row.
* **A hidden argmax.** The orange bar was labelled "after nested recalibration", but the
  code computes `best_span = span_ratio of the transform with the highest R²`
  (`cal.loc[cal["r2"].idxmax(), "span_ratio"]`). It is therefore *not* the best
  achievable spread — under the binned key the linear rescale reaches 0.55 against the
  plotted 0.53 — and it is not the largest R² gain either, since the selection criterion
  and the plotted quantity are different variables. Nothing in the artwork said which of
  three transforms produced the bar, or that a selection had happened at all.
* **The claim the figure was drawn to support was not on it.** The report's finding is
  that recalibration *cannot* repair the compression, and the evidence for "cannot" is
  that R² does not improve — the isotonic map costs the deployed stack −0.045, and the
  one gain that clears zero (+0.0087) does not reproduce under the strict key
  (−0.0004, P = 0.54). None of that appeared. A reader saw 0.42 → 0.53 and would
  reasonably conclude that recalibration helps, which is the opposite of the result.
* **Repository identifiers as tick labels:** `binned` and `strict`, with no statement
  that they are two definitions of "measured under the same conditions" or that they
  cover 905 and 1,417 pairs.
* **No interval**, though the table carries a 400-draw cluster bootstrap, and no mention
  of the multiplicity caveat the report calls the reason the result is "not established".
* Two-line 12 pt title carrying the entire argument; axis label
  "predicted spread / true spread" without saying the spread is a standard deviation of
  what quantity.

## Changes

**Folded, not kept separate — and why.** The two originals do not make the *same* claim:
one is about the model's quoted uncertainty on absolute log D, the other about the width
of its predicted adjacent-pair separations. But they are two readings of one word, on one
system, pointing the same way, and split across two figures a reader has no way to tell
which "calibration" is meant. Put together they also become a complete argument, which
neither is alone: the predictions are compressed (B) *and* rescaling them buys no
accuracy (C) *because* the model is genuinely more uncertain than it admits (A). Panels B
and C therefore **supersede `re_fig4_calibration`**, which should be dropped rather than
published alongside this figure. Panel A's cohort and units differ from B and C's, so each
panel names its own cohort and no axis is shared between them.

**Panel A — the missing column restored.** Both series of
`uncertainty_calibration.csv` are drawn on one log D axis, joined by a pale bar whose
length *is* the miscalibration, with the ratio of the two printed at each quintile:
5.3×, 4.2×, 3.8×, 3.1×, 2.6×. The direction is now unambiguous and stated in words on
the panel — "calibrated would mean the two series coincide" — and the ratio's decline
from Q1 to Q5 says where it is worst: the model is most over-confident exactly where it
claims to be most confident. The non-monotonicity is stated in grey next to the series
label instead of being contradicted by a title.

**Panels B and C — the selection removed and the accuracy question added.** All four
transforms are drawn, named for what they do, under both keys; there is no argmax
anywhere in the artwork. Panel B is the level (how wide are the predictions), panel C is
the contrast (what recalibration changes), on separate axes with separate reference
lines — a reader cannot subtract a B bar from a C bar and get anything. Panel C carries
the 90 % cluster-bootstrap whisker the table already held, and states in grey that the
one-factor rescale was the best of three and that the interval is not corrected for that
choice. The reference row shows the level the changes are measured from
(R² = +0.267 and +0.192) as text, not as a bar.

**Naming.** Every repository identifier is replaced, and the script asserts at the end of
`main()` that none of them survives into a drawn text artist:

| identifier | figure label | source |
|---|---|---|
| `full (CatBoost+repaired+S0)` | "deployed three-model stack" | `automl/topo/best_stack.py`, `combos`; the components are CatBoost on 2D descriptors, the repaired fully-connected network (`oof_fcnn_std_scaler_ens16.parquet`) and the S0 simplicial 3D encoder |
| `composition_key` / `binned` | "conditions matched in bins" (905 pairs) | `automl/topo/dualkey_test.py`, `BINNED`; built from binned condition columns |
| `strict_composition_key` / `strict` | "conditions matched exactly" (1,417 pairs) | ibid., `STRICT`; "every numeric condition included, so the only thing that varies inside a block is the lanthanide" |
| `raw` | "as measured (reference)" | `calibration_test.py::_fit_apply` |
| `scale` | "one-factor rescale" | `d → a·d` |
| `affine` | "linear rescale (factor + offset)" | `d → a·d + b` |
| `isotonic` | "any monotone map" | isotonic regression, the upper bound on rank-preserving calibration |
| `span_ratio` | "SD of predicted ÷ SD of measured separation" | `std(dp)/std(dy)` on adjacent-pair differences |
| `mae` | "errors actually made — mean absolute error" | `automl/ensemble.py`, `|nnls_oof − y|` |
| `spread` | "error bar the model quotes — spread across the base models" | ibid., `P.std(axis=1)` |
| `r2` (in `calibration_test_*`) | "adjacent-pair log SF R²" | `adjacent_test.adj_r2` |

**Axes and sign conventions.** Panel A: "log D units (absolute prediction, not a
difference)" — this is the one panel in the set that is *not* about the adjacent-pair
difference, and it says so. Panel B: "SD of predicted ÷ SD of measured separation, ratio:
1 = correct spread, below 1 = too flat". Panel C: "change in adjacent-pair log SF R²,
+ = recalibration buys accuracy". Both keys' n appear in the panel B legend.

**Layout and typography.** `pubstyle` base 8.0 pt: axis labels 8.5, tick labels 7.5,
annotations and value labels 7.0, panel letters 9.5 placed by
`pubstyle.add_panel_letters` from the measured tight bounding box. Okabe-Ito blue and
orange for the two pair cohorts, blue and vermillion for the two series of panel A; both
pairs are CVD-safe and both are also separated by marker shape or by row position.
Reference lines are drawn only across the data rows, so the reserved band at the top of
panels B and C is genuinely empty and the panel B legend cannot cover anything.
`figure.png` at 600 dpi and a fully vector `figure.pdf` (0 image XObjects) from one
figure object.

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 69 text artists
checked** (`figure.lint.json`). The render was read at full size, at 180 mm print width,
and in three full-resolution panel crops.

## Scientific integrity

**No number, cohort or method changed.** Same three tables, same rows, same model
(`full (CatBoost+repaired+S0)`), same 905/1,417 pair cohorts, same 5,946-row out-of-fold
set, same nested-by-extractant transforms, same 400-draw cluster bootstrap at 90 %. Every
plotted value is written to `values.json`.

Two derived quantities appear, both arithmetic on numbers already in the plotted tables:

1. Panel A's ratio is `mae / spread`, the ratio of the two series drawn beside it.
2. Panel C's change is `r2(transform) − r2(raw)` within one key. For the one-factor
   rescale this must equal the `scale_gain` point estimate the table already carries;
   the script asserts it to 1e-9 and refuses to draw otherwise. It holds exactly
   (+0.008662295194 binned, −0.000420160328 strict), which is what licenses computing
   the other two the same way.

Three things found and **not** silently fixed:

* **`calibration_test.csv` is a stale duplicate and should be deleted upstream.** Its 24
  rows are byte-identical to the strict run's rows in `calibration_test_strict.csv`
  (verified: all six shared columns equal). `calibration_test.py` says why — "the two
  keys are two different analyses and the first run was silently overwritten by the
  second" — and the fix was to write key-tagged files, but the overwritten file was left
  on disk. Anyone reading `calibration_test.csv` today gets the strict analysis under a
  name that implies it is the only one. The brief names it as an input; this figure does
  not use it, and no figure should.
* **`re_fig4_calibration`'s "after nested recalibration" bar is an argmax over a
  different variable from the one plotted** (selected by R², drawn as span ratio). It is
  not a numerical error — the value is correctly read from the table — but the artwork
  cannot be read correctly without knowing the selection rule, and under the binned key
  the plotted 0.53 is not the largest span the three transforms reach (0.55 is). Left
  untouched upstream; removed here by drawing all three transforms.
* **An upstream inconsistency in the base-model count for panel A.** `FINDINGS.md` §5
  says "Stacking 31 protocol-B base models" above a table whose numbers match
  `ensemble.txt`, which says "stacking 37 base models over 5946 rows". The calibration
  table printed in `ensemble.txt` is identical to `uncertainty_calibration.csv` to every
  digit it prints, so the run is the 37-model one; because the two documents disagree,
  the figure gives no model count and says only "spread across the base models".

**A defect I could not fix from the vendored data.** Panel A has no interval on the
quintile means. `uncertainty_calibration.csv` stores only the group means — the per-row
errors are in the out-of-fold parquets under `automl/artifacts/`, which are not committed
upstream (`REPRODUCIBILITY.md`). The comparison this matters for is Q3 (0.895) against
Q4 (0.868), a 0.027 log-unit difference across roughly 1,189 rows each; without the row
data I cannot say whether that dip is real, and I have not invented a whisker for it. The
figure therefore states the dip as an observed feature of the table and makes no claim
about its significance. The panel's actual message — a 2.6–5.3× gap between quoted and
realised error — is far larger than any plausible standard error on these means, so it
does not rest on the missing interval.

**The cohort of panel A is not the cohort of panels B and C**, and they are not the same
model either (panel A's is the 37-model out-of-fold stack from `automl/ensemble.py`;
panels B and C use the three-model nested stack from `automl/topo/best_stack.py`). Each
panel's strapline names its own cohort and model, and nothing on one panel is comparable
by subtraction with anything on another. Neither figure is the headline adjacent-pair
result: the +0.267 shown as panel C's reference is this stack under this analysis, not
the study's best system (+0.326, a different arm on a different fit).

## Paper role

**Results**, immediately after the headline accuracy figure — it is the honest limit
statement that must accompany any reported R². It could equally serve as
**Supplementary** if the paper reports only the raw numbers, since its conclusion is
precisely that the calibrated numbers should not be reported.

## Main message

Read as an error bar, the model's own uncertainty is 2.6–5.3× too small, and it is
furthest off where the model claims to be most confident; read as a magnitude, its
predicted adjacent-lanthanide separations span only 0.38–0.42 of the measured spread.
Rescaling closes the second gap on paper but buys no accuracy — even a free monotone map
loses 0.045 R² — so the compression is genuine uncertainty about the chemistry, not a
scale error waiting to be corrected.
