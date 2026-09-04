# Figure 1 — Where each model lands, and what the 3D arm itself contributes

## Source

Repository: `github.com/mironovb/lanthanidestrain` at commit
`1cdeac39d0b018529a735f967d3c83120d04c714` (vendored: `data/PROVENANCE.json`)
Original script: `automl/figures_pi_email.py::fig_result`
Original render: `original.png` (upstream `automl/reports/figures_ORIGINAL_BACKUP/email_figB_result.png`)
Reports: `automl/reports/PI_REPORT.md`, `COLLAB_UPDATE_REPORT.md`, and — for what each
arm actually is — `STACK_RESULTS.md`, `DUALKEY_RESULTS.md`, `CONTROL_PREREGISTRATION.md`,
`CAMPAIGN_AUG2026.md`, `AUDIT_2026-07-30.md`.

Input tables, all from the vendored `figure_refinement/lanthanidestrain/data/`:

| table | what this figure takes from it |
|---|---|
| `dualkey_arms.csv` | the three single-model levels (`adj_r2_binned`) |
| `best_stack.csv` | the three-model levels, the level interval of the best stack, and the two paired contrasts with their 90 % and 5-test-corrected intervals |
| `stack_test.csv` | the two-model blend level and its interval; the fingerprint baseline level |
| `adjacent_ensemble.csv` | the 3D encoder's level interval, its seed count (16) and single-seed SD (0.047) |
| `anchored_3d.json` | the nested blend weights (mean 0.349, range 0.30–0.40) that the fixed 0.35 is the median of |
| `anchored_3d_confirm.json` | the pre-declared confirmation: blend and 2D-only level on each of the three populations, the contrast, `w_fixed`, `primary_pass` |

Every plotted number is written to `values.json`. The script asserts the joins
(the encoder arm in `adjacent_ensemble.csv` is the same arm as in `dualkey_arms.csv`; the
baseline of the primary contrast is the fingerprint network; the confirmation's contrast
equals blend − 2D-only on every population; 0.35 lies inside the fitted weight range) and
refuses to draw if any of them fails.

## Problem

Defects I saw in `original.png`:

* **The figure's own docstring forbids what the figure invites.** `fig_result` warns that
  the paired contrasts "do not equal the difference of two whole-set scores" — and then
  the title reads "…reaches 0.267, **against** 0.226 without the 3D arm and 0.221 when
  that slot holds a model with no 3D input". Three levels, in one sentence, joined by
  "against". The only way to extract a contribution from that sentence is to subtract, and
  subtracting is wrong: the paired contrast is +0.0381, not 0.267 − 0.226 = 0.0408.
* **No uncertainty anywhere.** Not one interval, although the tables the script already
  reads carry them: the best stack's own level interval is [0.191, 0.296], which
  *contains* the no-3D bar at 0.226. A reader looking at seven bare bars cannot tell that
  the level of the top bar is far less certain than the gap the title asks them to read.
* **No sample size, no cohort, no design.** 905 adjacent pairs, 552 condition blocks,
  leave-extractants-out CV with 5 folds × 3 repeats, 16-seed ensembles whose single seeds
  scatter by SD 0.047 — none of it appears. That single-seed spread is larger than the
  0.041 and 0.046 gaps the title asks the reader to read.
* **The pre-declared confirmation is missing.** The one estimate in the project that was
  fixed in advance and scored on pairs never used for selection (+0.0156 on 444 held-out
  pairs) is not on the figure that carries the headline.
* **The metric is under-specified.** "adjacent-lanthanide separation R²" never says the
  predicted quantity is a log separation factor, never gives its sign convention, and
  never says these are the *binned*-condition-key values — the qualification that
  `DUALKEY_RESULTS.md` requires of "every table from here on".
* **Alpha as a semantic channel.** Bars carry alpha 0.5 / 0.9 / 0.95; the same blue at two
  opacities means the same thing (contains the 3D encoder) but reads as two series.
  Red is used for the matched control, which reads as "error" rather than "the null arm".
* **A four-hue palette that the repository's own colour test never checked, and that
  fails it.** `automl/tests/test_palette.py` computes CVD safety (Machado simulation,
  OKLab ΔE) for a registry of named subsets, none of which is this figure's
  violet + orange + blue + red. Running that test's own functions on those four returns a
  worst-pair ΔE of **5.62 under deuteranopia**, below its 6.0 hard floor, and a
  normal-vision ΔE of 7.13 against a floor of 15.0 — and the colliding pair is
  **orange against red**, i.e. the fingerprint network against the matched no-3D control,
  the two arms whose comparison is the entire point of the control. The same file already
  records that no four-colour subset of the house palette passes all three checks.
* **Grouping is implied and never named.** Two blank rows separate three groups; nothing
  says what the groups are.
* **Indentation that does not indent.** Row labels are right-aligned tick labels, so the
  leading spaces in `"      + a matched model with no 3D input"` do nothing.
* **Layout.** The canvas is 8.4 × 4.6 in with a three-line 11 pt title that is longer than
  the plotted content; the two spacer rows and the strip right of the bars leave a large
  empty region that carries nothing, while the x axis runs to 1.18 × the longest bar.

Two things the original did *well* and I kept: it never printed a repository identifier in
the artwork, and it put value labels at every bar end.

## Changes

**Levels and contrasts are on different panels, with different axes.** Panel A: the seven
levels of the combination study. Panel B: the levels of the pre-declared confirmation, on
its three populations. Panel C: the contrasts, on a Δ axis with a zero rule. Nothing on a
level axis is ever a difference, and nothing on the Δ axis is ever a level.

**The contrast the bars invite is drawn explicitly.** Panel C's first two rows are the two
paired contrasts that belong to panel A's bars, read from `best_stack.csv`: adding the 3D
encoder to the two 2D models (+0.0381 [+0.019, +0.050], 5-test corrected [+0.017, +0.060])
and putting it in the slot the matched control occupies (+0.0446 [+0.030, +0.054],
corrected [+0.027, +0.062]). Thick bar = 90 %, thin = corrected, exactly the convention
`fig_control` uses upstream. Panel A carries the sentence "A difference between two bars is
not the 3D arm's contribution — that is panel C" under the bars.

**The pre-declared confirmation is added**, as the brief requires, and it is given both a
level panel and a contrast row so it cannot be read as a bar difference either. Panel B
shows 2D-only → with-3D-shape on each population (0.105→0.121, 0.296→0.311,
0.230→0.245) and says in the artwork that a 0.015 increment cannot be read at that scale;
panel C shows the same three quantities as contrasts (+0.0156, +0.0146, +0.0149) with
their n. A grey band ties the three rows across the two panels. The rule is stated where
the estimates are: the weight was fixed at 0.35 in advance, the median of the 0.30–0.40
fitted earlier (`anchored_3d.json`), and the source reports no interval for these three,
because the pre-declared endpoint is the *sign* of the contrast.

**Uncertainty, wherever the tables carry it.** Level whiskers on the three rows whose
interval is in a vendored table, with "where reported" said in the artwork; both interval
widths on the two contrasts; and the single-seed spread (SD 0.047, 16 seeds) written out,
because it is larger than the gaps a reader is tempted to measure between bars.

**n on every estimate.** 905 in panel A's strapline (one cohort for all seven rows) and in
both contrast row labels; 444 / 905 / 1,349 in panel B's row labels *and* beside each
diamond in panel C.

**Units, definition and sign convention on the axis.** "adjacent-pair log SF R² · log SF =
log₁₀ D(lighter lanthanide) − log₁₀ D(heavier neighbour) / 0 = every pair separates by the
series average, −0.072". The null is the constant −0.0724 mean separation
(`AUDIT_2026-07-30.md`), so R² > 0 means beating the series-average trend — predicting
ligand-specific selectivity, not the lanthanide contraction. The binned condition key is
named in the footnote, with the fact that every level falls under a stricter one.

**Naming.** Identifiers were already absent upstream; these labels are aligned with the
sibling refined figures so one arm has one name across the set, and the script asserts that
no repository identifier reaches a drawn text artist.

| repository identifier | figure label | source |
|---|---|---|
| `CatBoost` | gradient boosting (2D) | `best_stack.py` header; matches figure 3 of this refinement |
| `repaired` | fingerprint network (2D) | `C6_PREREGISTRATION.md` "repaired fingerprint net"; `CONTROL_RESULTS.md` §4.2 — sklearn MLP on ECFP + RDKit, StandardScaler, 16 seeds |
| `S0` | 3D encoder | `CONTROL_PREREGISTRATION.md` §2 — simplicial message passing over the GFN2-xTB complex, contrast objective, 16 seeds |
| `T0w` | matched 2D control (no 3D input) | `CONTROL_PREREGISTRATION.md` §2/§5 — same harness, folds, seeds and objective, 3D input removed, wide head |
| `no topology (CatBoost+repaired)` | gradient boosting + fingerprint network | `best_stack.py` `combos` |
| `topology swapped for control` | + matched 2D control (no 3D input) | `best_stack.py` `combos` |
| `full (CatBoost+repaired+S0)` | + 3D encoder | `best_stack.py` `combos` |
| `adj_r2_binned` | adjacent-pair log SF R² (conditions binned into blocks) | `evaluation.py::adjacent_pair_metrics`, `DUALKEY_RESULTS.md` |
| drop-in contrast | 3D encoder added to the 2D pair | `best_stack.csv` question column |
| swap contrast | 3D encoder in place of the control | `best_stack.csv` question column |
| `fresh` / `legacy` / `all` | held-out confirmation pairs / pairs used throughout / both sets together | `fresh_eval.py`, `CAMPAIGN_AUG2026.md` |

The map is also written to `values.json` as `name_map`.

**Colour is one channel with one meaning**, from the Okabe-Ito set in `pubstyle`: blue =
contains the 3D encoder (in every panel), dark grey = the matched combination with no 3D
input, vermillion = the control that occupies the 3D slot but sees no 3D input, purple and
orange = the two single 2D models. No alpha, no opacity as a second encoding.

The five were put through the source repository's own palette test, unchanged, and the
numbers are in `values.json` under `palette_check`. At `pubstyle`'s grey (`#7F7F7F`) the
worst pair is purple against grey at ΔE 6.2 under protanopia — inside the 6–8 band that
test admits only with a second encoding — so the one data grey was darkened to `#4D4D4D`,
which lifts the worst pair to **ΔE 9.6** (above the 8.0 target), normal-vision ΔE to 15.6
and its contrast against white to 8.4. Grey *text* stays at `pubstyle`'s grey: that is
de-emphasis, not data. One threshold is still not met — the shared style's orange has
contrast 2.25 against white, below the test's 3.0 — and it is kept rather than invented
around, because the orange is the fingerprint-network identity across the whole refined
set and it appears here only as a large filled bar with its own row label.

**Layout.** 7.09 × 5.60 in (180 × 142 mm), constrained layout, nested gridspec so panel A's
deep row-label gutter is not imposed on panel B. Type: 8.5 pt axis labels, 7.5 pt row
labels and ticks, 7.0 pt value labels, 6.8 pt notes, 9.5 pt panel letters placed by
`pubstyle.add_panel_letters` from the measured bounding boxes. In-axes notes are clipped to
their axes on purpose: an unclipped text artist that is wider than its axes makes
constrained layout shrink the axes, which makes the text stick out further — a runaway that
collapsed the bottom row to 0.4 in before it was fixed.

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 59 text artists checked**
(`figure.lint.json`). The render was read at full size, and in two full-resolution crops
(panel A bars, bottom row).

## Scientific integrity

**No number, cohort or method changed.** Same tables, same rows, same 905-pair cohort for
panel A, same three confirmation populations, same intervals. Spot values against the
original render: 0.142 / 0.221 / 0.238 / 0.251 / 0.226 / 0.221 / 0.267 — identical, now
printed to four decimals because +0.2206 (fingerprint network) and +0.2208 (the control
stack) are two different arms that both round to +0.221.

Five things found and **not** silently fixed:

1. **`best_stack.csv`'s `delta` is the bootstrap mean of the resampled difference**
   (0.03807), not the observed difference of the two whole-set scores
   (0.26717 − 0.22634 = 0.04083). Upstream plots and quotes `delta`; so do I. The two
   differ by 0.003 for a second reason as well — the contrast is computed on the pairs the
   two systems share. This is exactly the gap the upstream docstring warns about, and it
   is the reason panel C exists rather than a difference-of-bars annotation.
2. **Level intervals exist for only three of panel A's seven rows.** In the paired
   comparison tables the *arm* carries `arm_lo`/`arm_hi` and the *baseline* carries only
   `baseline_obs`. The three rows that happen to have intervals are the three that contain
   the 3D encoder, which could be misread as "only the 3D arms are uncertain", so the
   artwork says "where reported". Computing the missing four would require the
   out-of-fold prediction vectors, which are under `automl/artifacts/` and are not
   committed upstream (see `../REPRODUCIBILITY.md`). **This is a limitation I could not
   fix from the vendored data.**
3. **The confirmation contrasts have no interval at all.** `anchored_3d_confirm.json`
   reports point estimates only; the pre-declared endpoint in
   `automl/topo/anchored_3d_confirm.py` is the *sign* of the contrast on the held-out 444,
   explicitly "not a magnitude threshold". Drawn as points, with "no interval reported for
   these" in the panel. **Also unfixable from the vendored data.**
4. **`anchored_3d.json`'s own legacy-905 result is deliberately not drawn.** The same
   architecture with the weight re-fitted per held-out extractant scores 0.3258 against
   0.3182 for the 2D-only reference, while the confirmation (retrained encoder, weight
   fixed at 0.35) scores 0.3109 against 0.2964 on the same 905 pairs. Two different runs;
   putting both level pairs side by side would invite precisely the subtraction this
   figure exists to prevent. Recorded in `values.json` under `not_plotted`. One
   consequence worth stating: the project's headline "+0.326" is this un-drawn number, so
   the highest level on this figure is 0.311, not 0.326.
5. **Two extractant counts exist in the repository for panel A's cohort** — 162
   (`AUDIT_2026-07-30.md` §1, for the 905-pair set) and 149 (`best_stack.py` docstring and
   `STACK_PREREGISTRATION.md` §3, for the nested weight fitting). Which one the cluster
   bootstrap resamples is not decidable from the vendored tables, so no extractant count is
   drawn; both are recorded in `values.json` with their sources.

No implementation defect was found in `fig_result` itself: it reads the right rows and its
numbers reproduce exactly. Its defect is presentational and it is the one its own docstring
predicts — a title that quotes three levels joined by "against" and thereby invites the
subtraction the docstring forbids.

One qualification carried into the artwork rather than left to the caption: these are the
binned-condition-key values, and under `strict_composition_key` every level falls
(`DUALKEY_RESULTS.md`) and the drop-in contrast stops clearing zero. Panel A's footnote
says the first half; the strict-key analysis is a separate figure in this set.

## Paper role

**Results** — the primary performance figure: what the system reaches, and whether the 3D
arm earned its place in it.

## Main message

The deployable combination reaches +0.267 adjacent-pair log SF R² on 905 pairs, and every
combination containing the 3D encoder outscores every combination without it — but the 3D
arm's own contribution is not the gap between those bars: it is a paired contrast of
+0.038 to +0.045 inside that combination, and +0.0156 in the later anchored system on 444
pairs that were frozen before the model ran and never used for any selection, where the
weight was fixed in advance and only the sign of the contrast was at stake.
