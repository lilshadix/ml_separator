# Figure 3 — The two qualifications the 3D result carries

## Source

Repository: `github.com/mironovb/lanthanidestrain`, `main` @ `1cdeac3`
(`data/PROVENANCE.json`)
Original script: `automl/figures_pi_email.py::fig_limits`
Original render: `original.png` (= `automl/reports/figures_ORIGINAL_BACKUP/email_figD_limits.png`)
Reports that define the two tests: `automl/reports/ENCODER_RESULTS.md`,
`automl/reports/DUALKEY_RESULTS.md` (pre-registrations `ENCODER_PREREGISTRATION.md`,
`DUALKEY_PREREGISTRATION.md`); `TOPOLOGY_TESTS.md` gives the later, independent
topology nulls and is not plotted here.

Input tables — all read through `figure_refinement/common/lstrain_data.py` from the
vendored copy in `figure_refinement/lanthanidestrain/data/`:

* `encoder_test.csv` — the three pre-registered encoder contrasts, both block keys.
  Panel A uses the `composition_key` rows only, exactly as the original did.
* `encoder_arms.csv` — single-arm adjacent-pair R² for each candidate (panel B).
* `dualkey_test.csv` — the two pre-registered stack contrasts under both block keys
  (panel C) and the stack levels they were computed from, `baseline_obs` / `arm_obs`
  (panel D).

Every plotted number is written to `values.json`. Nothing is recomputed: the point
estimates, both interval widths, `p_better` and the verdicts are read from the columns.

## Problem

Defects I saw in `original.png`:

* **A level and a contrast on the same row, with opposite signs.** The left panel's
  third y label read "simplicial against distance, same slot (+0.238 vs +0.247)" and the
  value drawn on that row was **+0.0091**. The two single-arm levels differ by
  **−0.0092** (0.2382 − 0.2474); the plotted contrast is the *in-stack* paired
  difference, **+0.0091**. Same magnitude, opposite sign, one row. A reader who
  subtracts the parenthetical levels gets the opposite answer to the marker beside them.
  This is precisely the hazard the sibling function's own docstring warns about
  ("the paired bootstrap contrasts are computed on shared pairs and do not equal the
  difference of two whole-set scores") — and this figure did it anyway.
* **The same number drawn twice with no explanation.** The right panel showed
  **+0.0177** with an identical interval on both strict-key rows. It is not an error —
  under the strict key the matched control receives a fitted stack weight of 0.00, so
  "swap" and "drop-in" become the same comparison (`DUALKEY_RESULTS.md` §3) — but the
  artwork gave no clue, so the duplicate reads as a plotting bug.
* **One colour, two meanings.** The left panel's blue marked "encoder arm"; the right
  panel's blue marked "binned conditions". The legend sat at the bottom **right**,
  below the right panel but visually figure-level, so its "binned conditions
  (published)" key appeared to describe the left panel's blue rows as well.
* **Two line weights, no key.** Each row was drawn as a thick 90 % interval *and* a thin
  multiplicity-corrected one. Nothing anywhere said so; the thin line is the one that
  decides every verdict.
* **Titles doing the caption's job.** Three-line 11 pt claims ("It is 3D, not topology…",
  "Matching conditions strictly halves it…") carrying numbers that were also on the plot.
  The second is also loose: it describes the drop-in contrast (+0.0375 → +0.0177, a
  halving) while the panel also shows the swap contrast (+0.0438 → +0.0177, −60 %).
* **Different x limits for the same quantity.** Left −0.04…+0.06, right −0.02…+0.08, both
  labelled "Δ adjacent-pair R²", so interval widths were not comparable across panels.
* **No cohort, no method, no sample size** anywhere in the artwork: not the CV scheme, not
  the 400-draw cluster bootstrap, not the block counts that *define* the two keys, not
  what R² = 0 means.
* **Which R² was left implicit.** "Δ adjacent-pair R²" without units, without the sign
  convention, and without saying that it is the R² of the adjacent-pair log separation
  factor rather than overall log *D* R².
* **Layout.** The right panel's two y labels floated at the group centres with ~40 % of
  the panel empty between rows; the legend hung off the bottom-right corner; there were
  no panel letters.

## Changes

**The two qualifications are now two rows, and each has its supporting evidence beside
it.** Row 1 (A, B) answers "is the effect topological?"; row 2 (C, D) answers "how
strictly must conditions match?". The titles are the two questions, prefixed
*Qualification 1* / *Qualification 2*, at 8.5 pt — the panels answer them, the paper
caption explains them.

**Levels and differences are on different axes, in different columns, with different
marker shapes.** Left column = differences (circles, Δ axis with a zero rule); right
column = levels (squares, R² axis whose label ends "a level, not a difference"). The
single-arm numbers the original hid in the left panel's y labels are panel B; the stack
levels the contrasts were computed from are panel D. A figure-level line under the panels
states the rule explicitly: *"Differences (A, C) are a paired bootstrap over the pairs two
systems share — they are not the gaps between the levels in B and D."*

**The sensitivity reads as a sensitivity.** In panel C each question appears twice, once
per definition of "identical conditions", as two rows of one group sharing a y label; the
colour is the only thing that changes. Panel D shows where the shrinkage comes from —
every level falls under strict matching — and carries the degeneracy in words: the two
identical strict points are joined by a dotted rule and annotated *"under strict matching
the control is given zero weight in the fit, so the two upper rows are the same model"*,
with *"one number for both questions"* on the matching pair in C.

**Colour semantics, one meaning per channel.**

| channel | meaning |
|---|---|
| blue | conditions binned — the published definition (552 blocks) |
| vermillion | every condition matched (2,109 blocks) |
| filled marker | multiplicity-corrected interval clears zero (`verdict_corrected = adds`) |
| open marker | corrected interval spans zero (`not distinguishable`) |
| thick line | 90 % cluster-bootstrap interval, 400 draws over whole extractants |
| thin line | the same interval after Bonferroni correction for the look count |
| circle / square | a difference / a level |

Panel A is binned-key data, so it is all blue — the conflation in the original is gone
because blue now means the same thing in every panel. Palette is Okabe-Ito from
`pubstyle` (the repository's own palette is not used; its `test_palette.py` found no
4-colour subset that passes CVD and contrast at once).

**Verdicts come from the table, not from the artwork.** `forest()` reads
`verdict_corrected` to decide the marker fill, so the drawing cannot disagree with the
pre-registered decision rule. `P(Δ > 0)` is printed on exactly the rows whose corrected
interval spans zero — the three cases where "still positive but not clear of zero" is the
whole point (`P = 0.65` in A; `P = 0.93` twice in C) — and a grey line in A says so.

**Axes.** A and C share one x scale (−0.036…+0.090) so interval widths are comparable;
B and D share another (0.126…0.306), which incidentally shows that the combination
(+0.267) clears every single arm (max +0.247). Every axis label names the quantity, the
sign convention and whether it is a level or a difference. The look counts are parsed from
the `lo_NNlook` column names, so the legend says "13 looks in A, 10 in C" without a
hard-coded number.

**Sample size and method in the artwork.** Under the panels: the CV scheme
(leave-extractants-out, 5 folds × 3 repeats), the dataset size (4,746 measurements, 162
extractants, 14 lanthanides), what R² = 0 means, the bootstrap (400 draws, clustered on
whole extractants, 90 %), and the block counts that define the two keys.

**Naming.** No repository identifier reaches the artwork; the script asserts this
(`BANNED` regex over every visible `Text`) and exits rather than rendering if one does.

| identifier | figure label | source for the wording |
|---|---|---|
| `S0` | 3D: edges + triangles | `ENCODER_RESULTS.md` "the arms": published simplicial net (nodes, edges, **triangles**) |
| `G0` | 3D: triangles deleted | same table: "the same network with the 2-simplex level removed" |
| `D0` | 3D: distances only | same table: "SchNet-style continuous filters over interatomic distance — no simplices at all" |
| `T0w` | matched 2D control (same net, no encoder) | `CONTROL_RESULTS.md` §"tabular + contrast, wide head — *the control*"; "same harness, folds, seeds and objective" |
| `CatBoost` | boosting | `PI_REPLY_2026-08-03.md`: "gradient boosting on 2D descriptors" |
| `repaired` | fingerprint net | `C6_PREREGISTRATION.md` "repaired fingerprint net"; `PI_REPLY`: "the fingerprint network" |
| `no topology (CatBoost+repaired)` | the 2D-only pair / "boosting + fingerprint net" | `ENCODER_PREREGISTRATION.md` contrast table |
| `topology swapped for control` | + matched 2D control | `DUALKEY_RESULTS.md` stack table (`CatBoost + repaired + T0w`) |
| `full (CatBoost+repaired+S0)` | + the 3D encoder (edges + triangles) | same |
| `composition_key` | conditions binned — 552 blocks (published) | `DUALKEY_RESULTS.md` "Which key is right?"; `dataset.py:387` |
| `strict_composition_key` | every condition matched — 2,109 blocks | same |
| `adj_r2_binned` | R², adjacent-pair log SF, blue (binned) | `README.md` "adjacent-pair log SF R²" |
| `p_better` | P(Δ > 0) | `DUALKEY_RESULTS.md` "`P(Δ>0) = 0.93`" |
| `lo_13look` / `lo_10look` | after Bonferroni correction (13 looks in A, 10 in C) | `ENCODER_RESULTS.md`, `DUALKEY_RESULTS.md` |

**Typography.** `pubstyle` base 8.0 pt: titles 8.5, y labels 7.5, tick labels 7.5,
values and notes 7.0, panel letters 9.5 placed by `PS.add_panel_letters` from the measured
tight bounding boxes. Figure is 180 × 136 mm (`PS.W2`), 600 dpi PNG plus a fully vector
PDF (no raster XObjects, three embedded Helvetica subsets) from one figure object.

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 71 text artists checked**
(`figure.lint.json`). The render was read back and inspected at full size and in two
full-resolution crops across three fix cycles.

## Scientific integrity

**No number, cohort, interval or method changed.** Same three tables, same row selection
(panel A is still `key == "composition_key"` only), same point estimates, same 90 % and
same multiplicity-corrected intervals, same `p_better`, same pre-registered verdicts. Spot
values against the original render: +0.0343 / +0.0284 / +0.0091 (A), +0.0375, +0.0177,
+0.0438, +0.0177 (C), and the arm levels +0.246 / +0.247 / +0.238 that the original
carried inside its y labels (B). Panel D's six stack levels come from the `baseline_obs`
and `arm_obs` columns of the same `dualkey_test.csv` rows.

**Content added, all from the same tables**, because the original stated these numbers in
prose (its titles and y labels) rather than plotting them: the matched control's own level
`T0w = +0.201` (`encoder_arms.csv`), and the six stack levels of panel D. Two facts are
transcribed from `DUALKEY_RESULTS.md` because they are not in any vendored CSV and they
are what makes "strict vs binned" mean anything: the **552 → 2,109 block counts** (legend)
and the **zero fitted weight on the control under the strict key** (panel D note). The
cohort line comes from the source `README.md`. These are marked in the script with the
reports they came from.

**Defects found and not silently fixed:**

1. *Presentation hazard, upstream, now designed out rather than patched.* `fig_limits`
   printed single-arm levels (`+0.238 vs +0.247`) inside the y label of the row whose
   drawn value is an in-stack paired contrast (`+0.0091`). Those two quantities disagree
   in sign. The upstream code is untouched; the refined figure puts levels on their own
   axis and labels it "a level, not a difference".
2. *Not a bug, worth recording.* The two strict-key rows of `dualkey_test.csv` agree only
   to floating point (Δ = 0.0176730746183684**90** vs …**506**; baseline 0.17372987271712
   **16** vs …**148**). They are the same quantity reached by two code paths, which
   confirms the degeneracy is exact and not a coincidence of rounding.
3. *Loose title in the original*, not carried over: "Matching conditions strictly halves
   it" describes the drop-in contrast (+0.0375 → +0.0177). The swap contrast on the same
   panel falls +0.0438 → +0.0177, i.e. by 60 %.

**What this figure deliberately does not show**, and could not from these tables: the
strict-key values of the *encoder* contrasts (they exist in `encoder_test.csv` and weaken
in the same direction — `ENCODER_RESULTS.md`), the level intervals for panel B (
`encoder_arms.csv` carries none), and the number of adjacent pairs scored (not in these
tables — the 905/1,220-pair populations quoted in `README.md` belong to the later anchored
champion, not to this dual-key re-scoring, so quoting them here would be wrong).

## Paper role

**Supplementary** — the limitations panel for the 3D-encoder result: what is *not* claimed
(the mechanism is not topological) and how much of the claim depends on an analysis choice
(the block key). Referenced from the Results section where the +0.0375 binned-key contrast
is reported.

## Main message

The 3D arm's contribution is not topological — deleting the 2-simplices costs nothing and a
distance-only network with no simplices at all earns the same slot (head to head
+0.0091 [−0.029, +0.047], not distinguishable) — and its size depends on how strictly
"identical conditions" is defined: under the published binned key the encoder adds
+0.0375 [+0.011, +0.064] and clears zero, under strictly matched conditions it adds
+0.0177 [−0.013, +0.048], still positive at P(Δ > 0) = 0.93 but no longer clear of zero.
