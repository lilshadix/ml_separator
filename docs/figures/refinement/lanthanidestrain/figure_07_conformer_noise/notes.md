# Figure 7 — What a 3-D descriptor tracks along the lanthanide series

## Source

Repository: `lanthanidestrain` (`github.com/mironovb/lanthanidestrain`), vendored at commit
`1cdeac39d0b018529a735f967d3c83120d04c714` (`data/PROVENANCE.json`).
Original script: `automl/figures.py::fig_noise_diagnostic`
Original render: `automl/reports/figures/fig3_conformer_noise.png` → `original.png`
Report context: `automl/reports/FINDINGS.md` §4.2 and `automl/reports/CONFORMER_RESULTS.md`

Input tables, both vendored under `figure_refinement/lanthanidestrain/data/`:

* `noise_by_block.csv` — 9 rows, `block, mean, median, count`. The `median` column is the
  nine numbers the original figure drew.
* `noise_by_feature.csv` — 293 rows, `feature, fit_r2, block`. This is the per-descriptor
  table that `noise_by_block` is the block aggregate of. The script re-derives
  `noise_by_block` from it before drawing anything and exits if the medians, means or
  counts disagree; at run time the worst difference is **5.6e-17** and all nine counts
  match, so panel A's bars and panel B's dots and histogram are provably the same numbers.

The quantity itself is defined in `automl/dataset.py::add_series_smoothed` (block
`g13__fitr2`) and re-implemented, with a fuller explanation, in
`automl/qc/scatter_diagnostic.py::family_fit_r2`: inside one ligand + anion family, every
3-D descriptor is fitted with a straight line against the Shannon ionic radius, and the fit
R² is recorded. Nothing is recomputed here.

Every plotted number is written to `values.json`.

## Problem

Defects visible in `original.png`:

* **The axis stops at 0.46, so the thing being measured is off the chart.** The quantity is
  a *share* of a descriptor's within-family variation; its denominator is 1. Drawing it on
  a 0–0.46 axis makes the top bar fill 80 % of the panel, and a reader takes away
  "electronic descriptors track the ionic radius strongly, and roughly twice as well as the
  contraction block". The measurement actually says the opposite: **the best block is
  0.37 — between 63 % and 81 % of the variation in every block is scatter about the fitted
  line.** That share is the entire point of the figure and none of it is drawn. The
  truncation is a side effect of `set_xlim(0, max * 1.25)`, which existed only to make room
  for value labels placed *outside* the bar ends.
* **The claim lives in the title, not in the artwork.** "Most of the 3D variation along the
  series is conformer noise" is a 12 pt headline — half again the size of every other text
  element on the page — asserting a conclusion the bars do not show. Remove the title and
  the figure argues the reverse.
* **Repository identifiers as row labels.** `electronic`, `global_shape`, `first_shell`,
  `rdf`, `polyhedron`, `contraction`, `chelate`, `steric`, `topology` are the snake_case
  block keys from `geom3d_features.BLOCK_PREFIX`. `contraction` is the worst of them: it does
  not mean the lanthanide contraction, it means "the first shell after the ionic radius has
  been subtracted", i.e. the opposite of what a reader assumes.
* **No sample size, although the table carries it.** The `count` column runs from 11
  (contraction) to 128 (RDF). The nine bars are drawn identically weighted, so a median
  over 11 descriptors and a median over 128 look equally solid.
* **No spread at all.** Nine point estimates, one per block, with nothing to say that the
  per-descriptor values behind them run from 0.00 to 0.68 and overlap almost completely.
  The visible rank order — 0.19, 0.20, 0.21, 0.21, 0.22, 0.25, 0.26, 0.26 — invites the
  reader to rank eight blocks on differences of 0.01, which the underlying distributions do
  not support.
* **The `mean` column of the same table is unused**, and the one place it disagrees sharply
  with the median (contraction: mean 0.277 vs median 0.194) is caused by a single
  descriptor at R² = 1.000 that is the fit's own predictor. Nothing in the original render
  hints at it.
* **The axis label under-specifies the statistic.** "median R² of descriptor vs Shannon
  ionic radius, within a ligand family" does not say the fit is a straight line, does not
  say the R² is a share *of the descriptor's within-family variation*, and does not say the
  median is taken over the descriptors in the block.
* **Style.** Off-white surface (`#fcfcfb`) with the vertical gridlines left switched on and
  showing through the bars — `fig_noise_diagnostic` turns off only the y grid — and bars in
  the repository's violet `#4a3aa7`, which is not one of the four colours
  `automl/figures.py`'s own docstring names as the "validated categorical order". The
  repository's palette is not reused here: `automl/tests/test_palette.py` reports that no
  four-colour subset of it passes CVD and contrast checks together.

## Changes

**The axis is the full share, 0 to 1, and the remainder is drawn.** Each row is a bar of
constant total length: blue from 0 to the block median (variation the straight line in
ionic radius explains), pale grey from there to 1 (scatter about that line). This is the
single change that makes the figure say what the report says. The value labels move
*inside* the blue segment in white, which is what freed the axis to end at 1.0.

**Every descriptor is plotted.** Under each bar sits a jittered dot strip carrying the
individual `fit_r2` of every descriptor in that block, from `noise_by_feature`. The bar
band and the dot band never overlap, so no value label can land on a marker. This is
strictly more information from the same table, and it is what shows that the block ordering
is not a ranking: the RDF block's 128 descriptors span 0.00–0.56 and the electronic block's
23 span 0.09–0.64.

**Sample size on every row.** `n = 11` … `n = 128`, right-aligned inside the pale segment,
where the bar geometry guarantees empty space.

**A pooled panel B** answers the question the block medians leave open — how many
individual descriptors are actually usable. It is the same 290 values, binned, with the
"equal parts signal and scatter" line at 0.5 and the count beyond it: **25 of 290 (9 %)**.
Panels A and B share the x axis and the script asserts their data areas are aligned to
within 1e-6 of figure width, so the dot at 1.0 in panel A sits directly above the histogram
bin it belongs to.

**Naming.** Each block key is replaced by the phrase from its own docstring in
`automl/geom3d_features.py` (module header, blocks G1–G9). No `gN__` prefix or snake_case
key reaches the artwork. The mapping, also written to `values.json` as `block_label_map`:

| table key | figure label | upstream definition |
|---|---|---|
| `first_shell` (G1) | First coordination shell | "inner coordination sphere distances / donor composition" |
| `contraction` (G2) | Shell minus the ionic radius | "the same shell after subtracting the Shannon ionic radius" |
| `polyhedron` (G3) | Donor polyhedron shape | "shape of the donor polyhedron (CShM vs ideal polyhedra, inertia/asphericity, convex-hull volume, solid angles)" |
| `steric` (G4) | Steric bulk at the metal | "%V_bur, exposure of the metal, radial atom counts" |
| `electronic` (G5) | xTB charges, dipole and forces | "xTB charges, charge transfer, dipole geometry, force residuals (a strain proxy)" |
| `rdf` (G6) | Metal-centred radial distribution | "element-resolved metal-centred radial distribution functions" |
| `global_shape` (G7) | Whole-complex shape and surface | "whole-complex size/shape/lipophilic-surface descriptors" |
| `chelate` (G8) | Chelate rings and bite angles | "donor-donor connectivity, bite angles, chelate ring sizes" |
| `topology` (G9) | Persistent topology | "persistence statistics (H0/H1) of the complex point cloud" |

**The statistic is named in full.** The x axis reads "share of a descriptor's within-family
variation explained by a straight line in Shannon ionic radius (fit R²; 0 = none, 1 = all)"
— which R², what the denominator is, what is fitted, and the sign convention. Panel A's
strapline states the cohort unit that the word "family" hides: *a family is one extractant
and anion, with 4 to 14 lanthanides* (`add_series_smoothed(min_members=4)`, 14 lanthanides
in the dataset).

**The title is gone.** The encoding now carries the conclusion; the paper caption carries
the explanation.

**Legend in reserved space.** Three entries in a band created by extending panel A's y
limit past the last row, so it cannot cover a bar or a dot. Okabe-Ito throughout: blue for
the explained share, pale grey for the remainder, sky for individual descriptors, vermillion
for the flagged artefact (see below). Type 6.5–8.5 pt on an 8.0 pt base, nothing smaller
than 6.5 pt; panel letters placed by `pubstyle.add_panel_letters` from the measured tight
bounding boxes and then aligned to a common left edge.

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 47 text artists checked**
(`figure.lint.json`). The render was read at full size, in two crops, and downsampled to
about half the print width. Export: 600 dpi PNG and a fully vector PDF (0 raster XObjects)
from one figure object.

## Scientific integrity

**No number, cohort or method changed.** The nine bars are exactly the `median` column of
`noise_by_block.csv`; the dots and the histogram are exactly the `fit_r2` column of
`noise_by_feature.csv`, and the script refuses to draw unless the second aggregates to the
first (worst difference 5.6e-17, all nine counts identical). Nothing is recomputed, no row
is dropped that upstream kept, and the block ordering is upstream's own
(`sort_values("median")`).

Three defects found in the upstream pipeline. **None is fixed** — all three are drawn or
counted exactly as published:

1. **The fit's own predictor is in the table.** `g13__fitr2__g2__contraction__ionic_radius`
   is the Shannon ionic radius fitted against the Shannon ionic radius, so its R² is 1.000
   by construction. `automl/qc/scatter_diagnostic.py::family_fit_r2` excludes this column
   explicitly (`c != RADIUS_COL`); `automl/dataset.py::add_series_smoothed`, which produced
   the published `g13__fitr2` block, does not. It is one of the contraction block's 11
   descriptors: it is the reason that block's mean (0.277) sits far above its median
   (0.194), and it moves the published median from 0.179 (the other ten) to 0.194. In this
   figure it is plotted, drawn as an open vermillion circle, and labelled *"the ionic
   radius, fitted against itself"* — visible rather than removed.
2. **A zero can mean "constant", not "pure noise".** `add_series_smoothed` assigns
   `fitr2 = 0.0` when a descriptor does not vary inside a family (`np.ptp(yv) < 1e-12`).
   Ten of the 290 descriptors carry exactly 0: `n_donor_S/F/Cl/Br/I`, four far-field
   nitrogen RDF bins, and `ring_size_max` — columns that are constant across the whole
   series, not columns swamped by scatter. They are inside the published block medians and
   are plotted as published; they are part of the leftmost histogram bin in panel B.
3. **Three descriptors are undefined everywhere** (`g2__contraction__excess_S_mean`,
   `g3__polyhedron__cshm_OC`, `g3__polyhedron__cshm_TPR`) and the published aggregate drops
   them. The figure therefore says 290, and `values.json` records
   `n_descriptors_undefined: 3` against the 293 rows in the table.

**One thing the vendored data cannot pin.** The script that wrote `noise_by_block.csv` and
`noise_by_feature.csv` is not in the upstream tree — `automl/figures.py` is the only file
that references either. The *quantity* is pinned (it is the `g13__fitr2` block, defined in
`automl/dataset.py` and re-implemented in `automl/qc/scatter_diagnostic.py`), but how each
descriptor's per-family fit R² was collapsed to the one number per descriptor in
`noise_by_feature` is not. Against `scatter_diagnostic.csv`, which recomputes the fit per
(family, descriptor) on the 91-family re-optimisation subset, the vendored value correlates
r = 0.87 with a family mean and r = 0.82 with a family median, so it is mean-like — but
that is inference, not a citation. The figure therefore says only "one descriptor" per dot
and never names which average over families it is. **This is a defect I could not resolve
from the vendored data, and it is why panel A shows the descriptor spread rather than a
confidence interval:** no per-family table exists here for the full cohort, so a sampling
interval on each block median cannot honestly be drawn. The dots are a spread of the
aggregated quantity, not an interval on the median, and the legend says exactly that
("one descriptor").

**The word "conformer" is deliberately absent from the artwork.** Upstream reads the
remainder as single-conformer scatter, and `CONFORMER_RESULTS.md` supports that reading
independently (82 % of shipped geometries are not the global minimum; the within-family SD
of the energy gap is 0.503 eV, 69 % of the 0.731 eV scatter). But `FINDINGS.md` §4.2 argues
the case with a "~0.05 Å conformer scatter in an M–O distance against a 0.013 Å adjacent
radius step", and `README.md` §4 records that the closely related "~0.04 Å
optimisation-noise floor" **was never measured**, traces to an asserted conformer scatter,
and is 0.0002 Å when measured directly. Putting either number into the artwork would be
drawing an assertion. The figure states what this table measures — the share explained by a
straight line in ionic radius, and the scatter about that line — and leaves the conformer
attribution to the caption, where it can be cited.

## Paper role

**Results** — the limits section. It is the measurement behind headline result 3 (fixing
the chemistry does not reach the adjacent-pair problem) and the justification for the design
choice that follows in `FINDINGS.md` §4.2: integrating the metal dependence out of the 3-D
features (`g14c`) raises within-extractant R² and preserves the series ordering, while the
raw descriptors destroy it.

## Main message

Fitted against the Shannon ionic radius inside one extractant's own series, no family of
3-D descriptors is mostly a size response: the best block, the xTB electronic descriptors,
reaches a median of 0.37 and every other block sits between 0.19 and 0.26, so 63–81 % of
what these features do along the lanthanide series is scatter about that line. Only 25 of
290 individual descriptors carry more size response than scatter, which is why no block
here can be treated as trustworthy at the ~0.013 Å scale that separates adjacent
lanthanides.
