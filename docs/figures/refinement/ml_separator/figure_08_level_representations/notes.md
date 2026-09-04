# Figure 8 (supplementary) — no learned representation predicts a ligand's level

## Source

Repository: `ml_separator` (`github.com/lilshadix/ml_separator`)
Original script: `scripts/gen7_figures.py::figure_level_capture`
Original render: `runs/gen7_architecture/figures/level_capture.png` (200 dpi, 3 panels, 8 bars each)
Input data:

* `runs/gen7_architecture/level_benchmark/level_leaderboard_raw.csv` — raw ligand mean,
  unseen-chemotype hold-out, 20 feature blocks × 6 regressor recipes + 3 nulls
* `runs/gen7_architecture/level_benchmark/level_leaderboard_adjusted.csv` — the same, with
  the target replaced by the ligand's mean residual after a global metal + conditions model
* `runs/gen7_architecture/level_benchmark/level_leaderboard_raw_ligandsplit.csv` — the
  weak (random-ligand) hold-out; **read but not plotted**, see below
* producer: `scripts/gen7_level_benchmark.py` (3 seeds × 5 chemotype folds = 15 folds;
  152 extractants, each held out once per seed → 456 held-out ligand evaluations)

Block sizes and reader-facing names were taken from the repository, not guessed:
`docs/gen5_levels_protocol_20260817.md` (PHYSCHEM 10, ECFP 2,048 Morgan-radius-2 bits →
ECFP4, LIG2D\_EXT 206, DONORS 13, POLYHEDRON 58, COMPLEX\_PHYS 26),
`scripts/gen8_recommend_experiment.py:208` (Morgan radius 2, 2,048 bits, verified against
the cohort), and the column counts in
`dataset with 3D structures/ligand_pretrained_embeddings.parquet`
(ChemBERTa mean-pooled 384-d, MoLFormer mean-pooled 768-d). `_pca16` is a 16-component PCA
fitted inside the fold on the training ligands. The full name map for all 20 blocks and all
6 recipes is in `figure_script.py` (`REPRESENTATIONS`, `RECIPES`) and every winner is named
in prose in `values.json`.

## Problem

The predecessor is a three-panel bar leaderboard, and the grammar is the defect:

* **Three unannounced x-scales.** Each panel autoscales to its own longest bar (ticks reach
  30, 30 and 8), so panel 3 is magnified **3.3×** relative to panel 1 and **4.0×** relative
  to panel 2 — 55.5 px per percentage point against 16.9 and 14.0 on the 200 dpi
  render. Measured on the bar pixels, the 6.4 % bar in panel 3
  (`chemberta_pca16`) is 356 px long and the 25.4 % bar in panel 1
  (`donors_physchem_lig2d`) is 430 px: a value four times smaller is drawn 83 % as long. The
  figure exists to show a collapse from ~29 % to ~6 % and the layout all but erases it.

  *Correction to the refinement brief for this figure:* the brief states that the 6.4 % bar
  is drawn **longer** than the 25.4 % bar. Measured on `level_capture.png` that is not the
  case — it is 83 % as long, and no panel-3 bar exceeds any panel-1 bar (panel 3's longest
  is 356 px, panel 1's shortest 401 px). The defect the brief names is real; only its
  extreme form is not. Verify:

  ```bash
  .venv/bin/python -c "
  import numpy as np; from PIL import Image
  m = np.array(Image.open('runs/gen7_architecture/figures/level_capture.png').convert('RGB'), int)
  g = np.abs(m - np.array([0x0b,0x6e,0x4f])).sum(2) < 40          # the bar colour
  rows = np.where(g.any(1))[0]
  bands, s = [], rows[0]
  for a,b in zip(rows, rows[1:]):
      if b-a > 3: bands.append((s,a)); s = b
  bands.append((s, rows[-1]))
  for r0,r1 in bands:
      xs = np.where(g[r0:r1+1].any(0))[0]
      runs, s = [], xs[0]
      for a,b in zip(xs, xs[1:]):
          if b-a > 60: runs.append((s,a)); s = b       # 60 px merges across the x grid
      runs.append((s, xs[-1]))
      print([b-a for a,b in runs])"          # bar 1 -> [498, 497, 356];  bar 5 -> [430, 434, 273]
  ```
* **Mislabelled quantity.** The x axis read "% of ligand-level spread captured". The plotted
  quantity is `100 × (MAE_null − MAE_model) / MAE_null` — a percent reduction in mean
  absolute error against predicting the training-set mean level. A reader takes "spread
  captured" for an R², which it is not, and the null and its absolute MAE were never named,
  so log-unit errors could not be recovered.
* **No uncertainty at all.** `level_mae_sd_over_folds` sits in the same CSV and runs
  0.141–0.289 on the raw target and 0.192–0.274 on the condition-adjusted one — 10.7–22.0
  and 18.1–25.9 percentage points once normalised by the null. The bars rank differences of
  1–5 points.
* **Winner's curse, unflagged.** Each bar is `idxmin` over the six regressor recipes for that
  block, with no selection penalty.
* **Legend on the data, rule through the legend.** The legend sat on the bottom bar, and the
  vertical rule it described struck through its own legend text.
* **Duplicated grids.** Both an x grid and a horizontal rule were drawn; the rule ran through
  the 7 pt grey in-plot regressor annotations (`svr_rbf`, `gbdt_depth2`, …).
* **Registry identifiers as labels.** `donors`, `ecfp`, `chemberta_pca16`,
  `donors_physchem_lig2d_pca16`, `polyhedron3d` were printed verbatim as tick labels.
* 24 bars for a claim about five families; 200 dpi; no vector export.

## Changes

* **One shared x axis (panel A).** Five representations, each a paired marker — filled
  square for the raw ligand mean, filled circle for the condition-adjusted level — joined by
  a pale arrow. The collapse from ~29 % to ~6 % is now a distance on the page.
* **Points with intervals, not bars.** Whiskers are ±1 sd of the fold-level MAE over the 15
  folds, normalised by the same null as the point. They are wide on purpose (see below).
* **Axis relabelled to what it is:** "Reduction in level MAE against the training-mean null
  (%)", with the sign convention ("positive = better than the null") and the null's absolute
  MAE in both target definitions (1.313 and 1.058 log₁₀ *D*) on the second line, so a reader
  can convert any point back to log units.
* **Reference lines labelled outside the plotting area.** The two nearest-neighbour lookup
  rules carry their labels above the axes — one running left from the adjusted rule, one
  running right from the raw rule, so they cannot collide — and every rule stops short of
  both label bands, so no line crosses text. The training-mean null is the zero line and is
  defined in the axis label rather than duplicated as a third label.
* **Series key instead of a legend box:** two marker-plus-text entries in the empty band
  above the top row.
* **Grids.** No x grid; a single very pale horizontal guide per category, which is the
  ordinary dot-plot leader and now has no in-plot text to strike through.
* **Regressor names moved out of 7 pt grey in-plot text.** The winner for every block, in
  both targets, is recorded in prose in `values.json`; the *scientifically* load-bearing part
  of that information — that each point is the maximum of a set of recipes — is now shown
  directly in panel B.
* **New panel B: every fit, not only the winners.** All 120 condition-adjusted fits (20
  representations × 6 recipes) as a one-dot-per-fit stack on an expanded scale, with the five
  panel-A selections filled. The nearest-neighbour lookup rule sits to the right of all 120.
* **Prose everywhere.** "Donor-atom census, 13 features", "ECFP4 fingerprint, 2,048 bits",
  "ChemBERTa embedding, 16 principal components", and so on.
* 600 dpi PNG plus vector PDF from one figure object; `pubstyle.save` bounding-box linter
  over every text artist. Current status: **layout OK, 42 text artists checked**
  (`figure.lint.json`).

### Which five representations, and why

One compact descriptor set (donor-atom census, 13), one large descriptor set (extended 2D,
206), one fingerprint (ECFP4, 2,048 bits), one pretrained embedding (ChemBERTa, 16 PCs) and
one fusion (donors + physicochemical + 2D, 229) — the coverage the brief asked for. Between
them they hold **both champions**: the donor census wins the raw target (29.4 %) and the
ChemBERTa embedding wins the condition-adjusted one (6.4 %), so no better number is hidden
by the selection. Panel B then shows all 20 blocks anyway, so "you only plotted five" cannot
change the conclusion. The other 15 blocks are in `values.json`.

### The ligand hold-out panel was dropped

Panel 2 of the predecessor (`level_leaderboard_raw_ligandsplit.csv`) is not reproduced. It
**does** show something the chemotype hold-out does not — with near-twins left in training,
the best block reaches 35.6 % against 21.6 % for the lookup, i.e. models beat the lookup —
but that comparison cannot be made honestly here, because **the ligand split was only ever
run on the raw target**. There is no `adjusted_ligandsplit` leaderboard. Its 35.6 % is
therefore the same confounded quantity that inflates the raw chemotype number to 29 %, and
with a ligand's near-twins (and its own laboratory's condition regime) left in training the
confound is *more* available, not less. So the panel cannot separate "the level is chemical"
from "the confound is copyable from a near twin", which is the one question the figure is
about. Its numbers are kept in `values.json` under `not_plotted_ligand_holdout` and the
diagnostic remains available in the run directory.

## Scientific integrity

**The estimator was not re-run, re-fitted or re-pooled.** Every plotted number is
`level_mae` (or `level_mae_sd_over_folds`) straight out of the three leaderboard CSVs, and
the per-block selection is the same `idxmin` over the six recipes that the predecessor used.
Four presentation changes are nevertheless worth stating loudly.

**1. The x quantity was renamed, not recomputed.** The predecessor's "% of ligand-level
spread captured" and this figure's "reduction in level MAE against the training-mean null"
are the *same arithmetic*: `100 × (MAE_null − MAE_model) / MAE_null`. Only the label changed,
because the old one invites an R²-like reading of a quantity that is a relative MAE
reduction. Verify:

```bash
.venv/bin/python -c "
import pandas as pd
b = pd.read_csv('runs/gen7_architecture/level_benchmark/level_leaderboard_adjusted.csv')
n = float(b.loc[b.model=='NULL_mean','level_mae'].iloc[0])
m = b[~b.model.str.startswith('NULL')]
best = m.loc[m.groupby('features').level_mae.idxmin()]
print((100*(n-best.level_mae)/n).max())"      # 6.4200 — the '6.4 %' in panel B
```

**2. The whiskers are fold dispersion, and they do not bound the difference.** The plotted
interval is ±1 sd of the 15 per-fold MAEs, which is 10.7–22.0 points on the raw target and
18.1–25.9 on the condition-adjusted one once normalised — several times any difference being
ranked. That is the honest scale of the *fold-level*
variation, and it is inflated by two things a reader should know: the chemotype folds are
very uneven (18, 18, 19, 20, 21, 22, 22, 22, 24, 27, 28, 35, 54, 56, 70 held-out ligands),
and the fold difficulty is *shared* by every representation, so most of the whisker is common
mode. A paired, per-fold analysis would give a much tighter interval on a difference, and the
per-fold table (`level_detail_adjusted.csv`) would support one — **it was deliberately not
performed here**, because that would be a new estimator rather than a re-presentation of the
existing one. Consequently: the whiskers say "do not rank these representations"; they do
**not** say the ~29 % → ~6 % collapse is unresolved, and they are not evidence for or against
any single pairwise contrast.

**3. Winner's curse.** Each panel-A marker is the best of six regressor recipes for that
block, chosen on the same test metric it is then reported at, with no penalty. Panel B is the
figure's answer to this: on the condition-adjusted target the six recipes for one block
spread over 1.7–6.4 percentage points (median 4.6) — as large as the whole effect being
reported — so a per-block "winner" is an upward-biased estimate. The claim the figure makes is deliberately the one
that survives that bias: even the **maximum over all 120 fits** (6.4 %) — a comparison
biased in the models' favour — does not reach the lookup (8.9 %).

**4. The pooled claim has a real qualifier, and it is on the figure.** "No representation
beats the nearest-neighbour lookup" is true of the pooled metric over all 456 evaluations.
It is **not** true on the hard subset: restricted to the 54 evaluations whose nearest
training ligand is below Tanimoto 0.4, the lookup falls to **−2.3 %** (worse than the
training mean) and 64 of the 120 fits are ahead of it, the best by +7.6 %
(`donors_physchem`/SVR-RBF, absolute MAE 1.090 against a null of 1.179). In other words the
lookup's pooled advantage comes from held-out ligands that *do* have a close training
relative. That note is printed in panel B and every number is in `values.json`
(`hard_subset_nn_tanimoto_below_0_4`); it is not plotted as a third series because the
benchmark stores no per-fold dispersion for a subset, so it would have to be drawn without
uncertainty on n = 54. Verify:

```bash
.venv/bin/python -c "
import pandas as pd
b = pd.read_csv('runs/gen7_architecture/level_benchmark/level_leaderboard_adjusted.csv')
h = float(b.loc[b.model=='NULL_mean','level_mae_nn_lt_0_4'].iloc[0])
nn = float(b.loc[b.model=='NULL_nn_tanimoto','level_mae_nn_lt_0_4'].iloc[0])
m = b[~b.model.str.startswith('NULL')]
r = 100*(h-m.level_mae_nn_lt_0_4)/h
print(round(100*(h-nn)/h,2), int((r>0).sum()), len(m), round(r.max(),2))"   # -2.33 64 120 7.60
```

Nothing else about the science changed. The two markers in a panel-A row are two *separate*
selections (the recipe that wins on the raw ligand mean is often not the one that wins on the
condition-adjusted level — for the donor census, ridge and then PLS-3), which is why the
strapline says "per representation **and target**"; the arrow connects a representation
across two target definitions, not one fitted model evaluated twice.

### Known limitation not fixed here

The two targets are normalised by different nulls (1.313 and 1.058 log₁₀ *D*), because each
percent is measured against the null *for its own target*. That is the only way to put them
on one axis, and it means the axis is a ratio scale within a target and only a qualitative
comparison across the two. Both nulls are printed under the axis so the conversion is
available; a reader who wants absolute MAEs will find all of them in `values.json`.

## Paper role

**Supplementary** — the representation-level evidence behind the main text's statement that
the ligand "level" is not recoverable from structure. Reads with the k-shot frontier
(Figure 2): if the level cannot be predicted, it has to be measured.

## Main message

The apparent ~29 % that a molecular representation "captures" of a ligand's extraction level
is mostly a confound: it is measured against the ligand's raw mean log *D*, which moves with
the conditions that ligand happened to be measured under. Remove a global metal + conditions
model first and the best of 120 fits — 20 representations, from a 13-feature donor census to
a 2,048-bit fingerprint to two pretrained transformer embeddings, each with 6 regressors —
reduces error by 6.4 %, behind an 8.9 % one-nearest-neighbour Tanimoto lookup, and on
genuinely distant chemistry neither reaches the training mean by more than a few points. On a
new chemotype the level is not something structure tells you; it is something you measure.
