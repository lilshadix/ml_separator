# ml_separator — figure refinement summary

Repository: `github.com/lilshadix/ml_separator`
Scope: every rendered figure in the repository was inspected — the 15 diagnostics under
`runs/*/figures/` (one independent audit each, image first, code second) and the 16-figure
publication set under `figures/`. Eight figures were rebuilt into
`figure_refinement/ml_separator/`; the rest are accounted for below.

**Status: 8 of 8 rebuilt figures pass the layout linter and the export check** (600 dpi PNG
+ vector PDF, matched geometry). **No scientific number was recomputed and no estimator was
re-run.** One figure (`figure_07`) changes which *quantity* it plots, for a reason given in
full below and in its `notes.md`; that is declared rather than slipped in. Two genuine implementation
defects were found and are documented, not silently corrected, in
[`SCIENTIFIC_DEFECTS.md`](SCIENTIFIC_DEFECTS.md).

---

## 1. Rebuilt figures

| Figure | Purpose | Original problem | Main fix | Paper section | Status |
|---|---|---|---|---|---|
| `figure_01_methodology` | What the model sees, what is held out, why *k*-shot is not leakage | ~70 hand-placed coordinates; 5.8–7.2 pt type; panel A's paragraph overflowed its axes into panel C's letter; the acquisition step was coloured as if it read a target | Laid out by primitives (points→axes fractions, named box anchors, weighted column tracks) so arrows are written between boxes, not between numbers; body type 8.0 pt on a 180 mm canvas; four information classes made the explicit grammar of the figure with a key row of its own | Methodology | rebuilt |
| `figure_02_few_shot_frontier` | How fast error falls as *k* measured points arrive | Value labels drawn on top of three converging curves; a legend entry that spanned the panel; "% improved" written in white inside a bar 0.026 tall, so it vanished; a leader line through an error bar | Labels moved into declared-empty regions; legend shortened and its space reserved by the axis limit; percentages moved into the tick labels; leader line removed | Results | rebuilt |
| `figure_03_shape_recovery` | The flattening of unseen titrations, and its repair | Three unrelated y-ranges in A–C so the compression could not be compared; D ordered the stages opposite to E and F; the point statistic and its interval were two indistinguishable marks | One shared y-range across A–C with the measured span printed per panel; one four-slot stage axis shared by D, E and F; point statistic and interval merged into one composite glyph, with the statistic named on every axis | Results | rebuilt |
| `figure_04_error_decomposition` | Where the remaining error is, and how much is reachable | Four stage labels stacked at the right edge of panel C, one running to the axis; panel B is a *different cohort* from A and C with nothing in the artwork to say so | Panel C stages numbered with a key in the empty corner; panel B given its own shaded field and an explicit "different cohort from A and C" line; oracle half of A separated by a rule and hatched | Results | rebuilt |
| `figure_05_training_coverage` | Whether coverage of training chemistry is the binding constraint | The underpowered observational panel led the figure; legend inside the bars; two-line y labels cramped | Restructured so the **controlled experiment leads** (A, B) and the observational terciles are demoted to C on a separate field; the far−near contrast and its interval, which includes zero, printed in the panel | Results | rebuilt |
| `figure_06_acquisition` | Which single measurement to make | 17 registry identifiers as labels; a paired-difference interval drawn re-centred on each policy's own bar, readable as a marginal interval; family legend over the bars; cohort differs from Figures 2/4/5 with nothing saying so | Reader-facing policy names (verified against the registry, mapping in `notes.md`); split into level / paired difference / deployment risk; legend below; cohort stated in the panel | Results | rebuilt |
| `figure_07_series_local_calibration` **†** | Whether a measurement in one series calibrates another | Title clipped at the canvas edge; missing cells the same white as zero gain; no *n* anywhere though 30 of 49 cells have *n* ≤ 5; colour limit set by a single *n* = 3 cell; **and the directional arrow in the title is not supported by the estimator** | Presented as the symmetric quantity it is, lower triangle, with `max|M − Mᵀ| = 4.4e-16` printed on the figure; *n* in every cell and low-*n* cells hatched with a key; diagonal outlined; panel C demonstrates that the apparent direction is exactly the gap between two baselines | Supplementary | rebuilt |
| `figure_08_level_representations` | Whether any representation predicts a ligand's level better than a lookup | Three panels with independent unannounced x-scales, so a 6.4 % bar was drawn longer than a 25.4 % bar; legend on the bottom bar and struck through by the line it described; no uncertainty on a ranking whose fold sd exceeds the gaps | Raw and condition-adjusted targets on **one** shared axis as paired markers, so the collapse is the visual event; ±1 sd whiskers; a second panel showing **all 120 fits** as a dot histogram, annotated that panel A's whiskers are 2.4× the width of that entire axis | Supplementary | rebuilt |

**†** `figure_07` is the one figure in this tree whose *plotted quantity* changed. The
predecessor drew the derived, directional `gain`; the refined figure draws the symmetric
post-calibration MAE that `gain` is built from, because `gain`'s apparent direction is an
artefact of the baseline subtracted (verified: `max|M − Mᵀ|` is 4.441 × 10⁻¹⁶ for the
calibrated MAE and 0 for `n_ligands`, while only `zero_shot` is directional). The `gain`
values are not discarded — they are printed verbatim in panel B. Two further declared
changes: the interpretation threshold was raised from `n ≥ 3` to `n ≥ 5`, with every
sub-threshold cell still drawn and hatched rather than dropped; and 95 % chemotype-blocked
intervals were added by resampling the estimator's own per-(seed, extractant) output, with
the script asserting that re-aggregating it reproduces the frozen CSV (observed drift
2.2 × 10⁻¹⁶). All three are written up in that figure's `notes.md`.

## 2. Diagnostics not rebuilt, and why

All 13 were audited from the rendered image. Each carries a claim the publication set
already makes, usually on a larger cohort and with uncertainty the diagnostic lacks.

| Figure | Superseded by | Note |
|---|---|---|
| `gen7/error_budget` | `figure_04` | also carries a data-integrity defect — see `SCIENTIFIC_DEFECTS.md` §2 |
| `gen7/gain_vs_similarity` | `figure_05` + `figure_04` panel C | |
| `gen8/adaptation_curve` | `figure_02` | |
| `gen8/oracle_gap` | `figure_04` panel A + `figure_02` | older, smaller cohort; no seed spread |
| `gen8/policy_comparison` | `figure_06` | 99 extractants and no intervals, against 143 with paired intervals |
| `gen8/calibration_geography` | `figure_06` | scores strata descriptively rather than as deployable policies |
| `gen8/slope_flattening` | `figure_03` panel D + `FigS5` | |
| `gen9/fig1_slope_recovery` | `figure_03` panel D | two of its four columns are near-identical baselines |
| `gen9/fig2_span_recovery` | `figure_03` panel E + `FigS5` | a median where the refined figure shows the distribution |
| `gen9/fig3_response_surfaces` | `FigS7_raw_example_curves` | examples chosen as extremes rather than by a declared rule |
| `gen9/fig4_acquisition_regret` | `figure_06` | |
| `gen9/fig5_adaptation` | `figure_02` | |
| `gen9/fig6_decomposition` | `figure_04` panel C + `FigS3`/`FigS4` | |

## 3. Carried over unchanged

The ten supplementary figures in `figures/supplementary/` were built in an earlier pass and
were **re-checked with the new layout linter during this one**. One real collision was found
and fixed (`FigS8_diagnostics`: a note in panel B ran into panel C's tick labels). All ten
now report a clean layout. They are not duplicated into this tree; they remain at
`figures/supplementary/` with their scripts in `figures/scripts/`.

## 4. Strongest figures for the main paper

In reading order, and this is also the priority order if the journal allows fewer:

1. **`figure_02`** — the result. Paired at the level of the individual support/query draw,
   with a model-free null, an explicitly labelled oracle, and its own failure mode (33 of 99
   extractants made worse) in the same figure.
2. **`figure_03`** — the most transferable finding: an error that pooled MAE is blind to, a
   mechanistic diagnosis, and a cheap fix.
3. **`figure_04`** — turns "the model is not very good" into an accounting with an
   intervention attached to each line, and bounds what is left.
4. **`figure_01`** — carries no result, but without it a referee cannot rule out leakage,
   which is the most likely reason to reject the *k*-shot numbers.
5. **`figure_05`** — the interventional half is strong; see §7 for the caveat.
6. **`figure_06`** — the deployment recommendation plus a clean negative result. Publish if
   a sixth figure is available; it is a corollary of `figure_02`.

## 5. Better suited to supplementary

* `figure_07_series_local_calibration` — supports the "measure in the system you intend to
  predict" claim that Figures 2 and 6 currently assert without evidence, but it rests on a
  matrix whose median cell has 5 extractants.
* `figure_08_level_representations` — a strong negative result, but about an isolated
  sub-problem rather than the deployed task.
* The existing `FigS1`–`FigS10`. Of these, `FigS2` (query-set fragility) is not optional: it
  is the negative result attached to Figure 3 and its absence would be a reporting failure.

## 6. Redundant

The 13 diagnostics in §2, and within them two specific redundancies worth naming:
`gen9/fig1_slope_recovery` spends two of four columns on baselines that differ by 0.00–0.02
in median ratio, and `gen8/adaptation_curve`, `gen8/oracle_gap` and `gen9/fig5_adaptation`
are three renderings of one adaptation curve on three cohorts.

## 7. Results whose interpretation still needs checking

* **The observational distance effect** (`figure_05` panel C). The far−near contrast at
  *k* = 0 is +0.35 with a 95 % interval of [−0.04, +0.66] over 17 chemotypes per tercile. The
  point estimates are monotone; the comparison is not separated from zero. The figure says
  so; the text must not overstate it.
* **`figure_07`'s estimator.** Fixing the direction — fit the offset on source rows, score
  target rows — is a re-run, not a re-plot, and was deliberately not attempted here.
* **`figure_08`'s winner's curse.** Each representation's score is the best of six regressor
  recipes with no selection penalty. Panel B shows the full distribution so the reader can
  see how little separates them, but the point estimates remain optimistic.
* **Panel B of `figure_04`.** Its components overlap by construction, so for one component
  ("series-local level") the deployable-removal bar exceeds the current-contribution bar.
  That is a property of the accounting, not an error, and the caption must say it.
* **`figure_07`'s cells are not cohort-matched.** A diagonal cell averages every extractant
  with rows of that curve type (85 for the metal series); an off-diagonal cell averages only
  extractants with rows of *both* (1 for metal series × contact time). The
  diagonal-versus-off-diagonal comparison the figure rests on is therefore not a matched
  comparison. Every cell prints its *n* so a reader can see it, but the artwork cannot fix
  it — matching the cohorts means re-deriving the estimator.
* **One interval in `figure_07` collapses to a point.** The four extractants behind
  "metal concentration" all belong to one Tanimoto-0.7 chemotype, so the chemotype-blocked
  bootstrap has nothing to resample. The row is greyed as sub-threshold, but a zero-length
  bar could still be misread as precision. Three further intervals run past the axis and are
  marked with a triangle rather than drawn in full; their bounds are in `values.json`.

## 8. Results with no good visualisation

* **The 0.10 log units of macro MAE that no identified component explains** appears as one
  grey bar in `figure_04` panel B and nowhere else. There is no diagnostic that
  characterises those rows.
* **Series-local transfer** now has `figure_07`, but only as a symmetric matrix; the
  directional question the deployment advice actually rests on is unmeasured.
* **A prospective test.** Every figure in this repository is retrospective cross-validation
  on a frozen corpus.

---

### Regenerating

```bash
for f in figure_refinement/ml_separator/figure_*/figure_script.py; do .venv/bin/python "$f"; done
.venv/bin/python figure_refinement/common/verify_exports.py
```
