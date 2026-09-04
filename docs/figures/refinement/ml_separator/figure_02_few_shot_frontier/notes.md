# Figure 2 — What a handful of measurements buys

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `figures/scripts/plot_fig2_fewshot_frontier.py`
(itself a refinement of the diagnostics `runs/gen8_architecture/figures/adaptation_curve.png`
and `runs/gen9_shape/figures/fig5_adaptation.png`)
Input data/results:
* `runs/gen10_final/final_locked/kshot_detail.parquet` — 12,834,720 rows, one per
  (global model, adapter, acquisition policy, k, extractant, split seed, repeat)
* derived via `figures/scripts/prepare_kshot_tables.py` into
  `figures/derived/kshot_per_ligand.csv` and `kshot_per_seed.csv`
* paired intervals from `lanthanide_separation.gen8.inference.paired_chemotype_bootstrap`

## Problem

Defects in the predecessor renders:

* **Text over data.** The end-point value labels (`1.036`, `0.654`, `0.441`) were drawn on
  top of the frontier lines; at `k = 5` the label sat across three converging curves.
* **Legend length.** The oracle entry read
  "Oracle support points for a level-only fit (not deployable)" and spanned most of the
  panel width, forcing the legend into the data region.
* **Invisible annotation.** The "% of extractants improved" figures were drawn in white
  inside the bars of panel B; the fourth bar is 0.026 tall, so its label was clipped by
  its own bar and became unreadable.
* **Leader line through data.** The "series adaptation is the identity at k ≤ 1" note in
  panel C had an arrow that crossed the k = 1 error bar and pointed at empty space
  between two bars.
* **Type size.** Base 7.0 pt with 6.0–6.5 pt annotations — legible on screen at 4000 px,
  marginal at 180 mm print width.
* **Panel letters.** Placed in axes coordinates at a hand-tuned offset; the letter for
  panel A sat against the topmost y-tick label.
* The older `runs/*/figures` versions additionally used repository arm identifiers
  (`GEN9_SHAPE_RECOMPOSED`, `REC_ecfp_plus_recovered`) as legend entries and saved at
  140–170 dpi.

## Changes

* Base type raised to 8.0 pt; every annotation is now ≥ 6.5 pt.
* Value labels moved into empty regions — above the `k = 0` point, below `k = 1`, and to
  the right of the `k = 5` point, for which the x-limit was extended to 5.95 to create a
  margin. No label now sits on a line.
* Legend entries shortened ("Oracle points (not deployable)") so five entries fit in the
  empty upper-right corner without touching a curve; the y-limit was raised to 1.30 to
  reserve that space explicitly rather than relying on luck.
* Panel B: percentages moved from inside the bars into a second line of the x tick
  labels, so a short bar no longer hides its own annotation.
* Panel C: the leader line was removed and the note placed in the empty upper-right
  corner of the panel.
* Panel letters now placed by `pubstyle.add_panel_letters`, which measures each axes'
  drawn tight bounding box after rendering, so a letter cannot land on a tick label.
* Colours unchanged (Okabe-Ito); reader-facing names throughout; export at 600 dpi PNG
  plus vector PDF from one figure object.
* The script asserts a clean layout: `pubstyle.save` runs a bounding-box linter over
  every text artist and reports overlaps, clipped annotations and layout overflow.
  Current status: **layout OK, 72 text artists checked** (`figure.lint.json`).

## Scientific integrity

**No change.** Same cohort (99 extractants, 41 chemotypes), same five split seeds × twelve
pool/query draws, same four pre-specified deployment rules, same oracle arm, same
chemotype-block bootstrap, same paired comparisons. The plotted values reproduce the
frozen frontier exactly:

| k | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| final pipeline | 1.0358 | 0.6539 | 0.5593 | 0.4928 | 0.4405 |

which are rows 1–5 of `figures/METRIC_AUDIT.md` (PASS, ≤ 5 × 10⁻⁵ against
`runs/gen10_final/final_locked/frontier_best.csv`). `values.json` in this folder records
every plotted number.

## Paper role

**Results** — the primary quantitative figure.

## Main message

On extractants with no analogue in the training set, a single chosen measurement removes
0.382 log units of macro MAE — an order of magnitude more than any modelling change in
this study — but a third of extractants are still made worse by it, so the deployment unit
is a prediction plus a *chosen* experiment rather than a prediction alone.
