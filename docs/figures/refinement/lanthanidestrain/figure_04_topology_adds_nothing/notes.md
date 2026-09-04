# Figure 4 — Beyond the distance encoder, topology adds nothing

## Source

Repository: `github.com/mironovb/lanthanidestrain` at `main`, commit `1cdeac3`
(2026-08-24), vendored into `figure_refinement/lanthanidestrain/data/`.

Result this figure is for: **`automl/reports/TOPOLOGY_TESTS.md`** — "Do topological
representations add anything? Four tests, all negative" (finding I17, listed in the
repository README's headline table as *"Topology beyond the distance encoder: nothing to
add"*).

**The negative result had no figure at all.** The four candidate renders named in the brief
all argue the *earlier, positive* claim, from `automl/figures_topo.py`:

| candidate | function | used? |
|---|---|---|
| `topo_stack.png` | `fig_stack` | **yes**, right-hand panel only → panel E |
| `topo_ladder.png` | `fig_ladder` | no |
| `topo_forest.png` | `fig_forest` | no |
| `topo_blend_curve.png` | `fig_blend_curve` | no |

`topo_stack.png` is copied here as `original.png`: it is the only one of the four whose data
survives into the new figure, the only one that carries intervals, and the direct ancestor
of panel E.

Input tables (all vendored, all read by the script — nothing is recomputed):

* `topo_shape.json` → panels A and C
  (`part1_encoder_comparison`, `part2_triangle_ablation`; written by `automl/topo/topo_shape.py`)
* `anchored_champion.csv` → panel D (written by `automl/topo/anchored_champion.py`)
* `stack_test.csv` → panel E (written by `automl/topo/stack_test.py`)
* five pair-level correlations in panel B — **transcribed**, see *Scientific integrity*

Every plotted number is written to `values.json`.

## Problem

### What I saw in `original.png` (`topo_stack.png`)

* **A contrast panel next to a level panel, on the same row, in the same colours.** The left
  panel is four blend *scores* (0.2206, 0.2289, 0.2382, 0.2511); the right panel is three
  paired-bootstrap *contrasts*. Subtracting two left-hand bars gives 0.2511 − 0.2206 =
  **0.0305**, which is not the +0.0351 the right panel reports for the same comparison.
  This is precisely the failure `fig_result`'s docstring warns about elsewhere in the
  project, committed inside one figure.
* **A truncated bar axis with no break.** The left panel runs 0.19–0.275, so a 14 %
  difference in score (0.2206 → 0.2511) is drawn as a 2.0× difference in bar length.
* **The multiplicity correction is mislabelled.** The x axis reads
  `Δ adjacent-pair R² (dark = 90 %, pale = 4-test corrected)` and the footnote repeats
  "4-test". `automl/topo/stack_test.py` calls `_corrected(r["delta"], r["lo"], r["hi"], 3)`
  with `def _corrected(..., n_tests=3)`, and `STACK_PREREGISTRATION.md` says "the three-test
  corrected interval is reported beside it every time". The words are wrong; the numbers are
  right. (See *Scientific integrity*.)
* **Repository identifiers as the artwork**: `S0`, `T0w`, `blend(S0) − repaired`,
  `blend(T0w, repaired)`, `repaired baseline`, and a filename — `STACK_PREREGISTRATION.md` —
  printed in the footnote.
* **A title that is a claim and a tally**: "2/3 clear zero nominally, 1/3 after correction".
* **Legend inside the data.** The two-entry legend sits in the lower-right of the left
  panel's bar field.
* **An unlabelled reference line.** The dashed rule at 0.2206 is explained only in a
  three-line grey footnote at 7.5 pt.
* **Hatching used for "no topology"**, which reads as "estimated / provisional" instead.
* Every left-panel bar value is hardcoded in the function (`bars = [... 0.2206 ... 0.2511]`)
  rather than read from `stack_test.csv`, which sits two lines above.

### Why the other three were dropped

* **`topo_forest.png`** — "Every test clears zero". All five effect sizes and intervals are
  **hardcoded literals in the script** (`tests = [("SNN ensemble (16 seeds)", "vs FCNN",
  0.2426, 0.181, 0.333, …), …]`) with no vendored table behind them, so nothing in it can be
  verified from a fresh checkout. Every comparison is against a *convenience* baseline
  (the FCNN as published, plain CatBoost) — which the repository's own portable-findings
  table names as a trap: *"Test against the champion, not the convenience baseline — four
  signals vanished or reversed when the baseline was strengthened."* Five tests are shown
  with no multiplicity correction.
* **`topo_blend_curve.png`** — "The blend beats both endpoints", with the footnote *"An
  interior maximum can only arise from complementary information"*. That inference is
  superseded inside the same repository: `control_blend.csv` holds the matched
  non-topological control, whose curve has an interior maximum too (+0.2469 at *w* = 0.6).
  Reproducing the panel without its control would restate an argument the project has since
  qualified. It also puts two different R² (adjacent-pair and overall log *D*) on one axis
  labelled just "R²".
* **`topo_ladder.png`** — a scatter of R² *levels* on two axes with **no interval anywhere**,
  a bolded 3×-sized star reading "BEST: stack + SNN", nine hand-tuned per-point label
  offsets, an unlabelled dashed reference, a legend inside the axes, and seven repository
  identifiers in the artwork (`S0`, `P0`, `T0w`, `PI-CNN`, `SNN`, `repaired FCNN`,
  `stack, no topology`). Its headline arm is the one the later test retracts.

### The larger problem

None of the four shows the *evidence for the null* the report actually establishes: the
blend weight the fit assigns, the correlation between the two encoders, the triangle
ablation, or the persistence collapse. Panels A–D of this figure are drawn from tables that
had never been plotted.

## Changes

**One figure, five beats, in the order the argument runs.**

* **A — the fit gives it no weight.** Mean nested blend weight per row, with the resulting
  adjacent-pair R² as a right-hand text column (not a second bar). Offered the slot the
  distance encoder earns 0.35 in, the simplicial encoder is given **0.01**, and its blend
  scores +0.3172 — *below* the +0.3182 of using no encoder at all. Offered both at once, the
  fit puts **0.00** on it. A weight of exactly zero has no bar, so it is marked with a tick
  and a printed `+0.00`; the reference row is marked "no encoder offered" for the same reason.
* **B — because the two encoders predict the same thing.** Pearson *r* on a 0→1 axis that
  stops at 1: the encoders agree with each other at **0.963**, they agree with the tabular
  shape equally (0.743 / 0.749), and they correlate with the tabular model's error — the only
  part a blend can exploit — equally too (0.196 / 0.176).
* **C — triangles are not better than edges.** Four matched-seed comparisons as offset
  dumbbells; all four differences favour the edge-only model. The two ends differ by as
  little as 0.0006, so they are separated by shape, colour *and* a vertical offset — drawn on
  one line they read as a single measurement. `n = 8 shared seeds` is on the axis, because
  eight seeds is a small *n* and the report itself says the blend differences are within seed
  noise.
* **D — persistence descriptors collapse the shape model.** Bars from a true zero, with all
  four individual seeds drawn as open circles and their range as a hairline, so the reader
  can see both the collapse (+0.3188 → −0.0387) and that ensembling partly rescues it. The
  block-mean control is the same colour with a hatch and is named in the legend as "same
  columns, within-block variation removed" — it recovers 78 % of the damage, which is the
  mechanism.
* **E — and the pre-registered topology-specific test does not survive correction.** All
  three pre-registered contrasts, never mixed with a level, with a zero rule, the 90 %
  paired cluster bootstrap as the thick line and the three-test corrected interval as the
  pale band, plus a plain-word verdict column ("clears zero" / "spans zero") derived exactly as
  the table's own `verdict_3test` field is, from `lo_3test > 0`. All three are shown, including the one that *is*
  positive: showing only the two that span zero would be the mirror image of the original's
  sin.

**Levels and contrasts never share a panel.** Panel E is the only paired-bootstrap panel and
carries no level bars. Panels A, C and D carry levels only; where a difference of levels is
shown (panel C's `triangles − edges` column) it is the same quantity the upstream code
computes as `blend_r2_tri_minus_notri`, on identical rows and identical seeds, and it is
labelled as a difference of the two points beside it, not as a bootstrap contrast.

**Naming.** Every repository identifier is replaced. The script asserts that none of
`anch_`, `q60`, `c15`, `c17`, `plw`, `snn`, `picnn`, `T0w`, `g9__`, `adj_r2`, `sel_`,
`_ens`, `oof`, `resid_blocks` survives into any drawn text artist, so this cannot silently
regress. The mapping is also in `values.json` as `name_map`:

| identifier | figure label | where the meaning comes from |
|---|---|---|
| `c15_plw4` | distance encoder (edges only) | `topo/topo_shape.py` docstring: "dist encoder, c15_plw4 32-seed ensemble (current system)"; TOPOLOGY_TESTS.md §1 "All the 3D contribution so far comes from the *distance* encoder — edges only" |
| `c17_plw4` | simplicial encoder (edges + triangles) | same docstring: "snn encoder … message passing over edges AND triangles" |
| `c17_plw2` | the same encoder, as originally published | `C17_PREREGISTRATION.md`: "snn sits at `plw = 2.0` in **371 of 380** runs … The published 3D model is not worse; it is **untuned**". `plw4` is the tuning it never received, so the 0.01 headline is the *tuned* simplicial encoder against the *tuned* distance encoder |
| `tabular_only` | tabular shape only (reference) | `topo_shape.py`: the anchored CatBoost's own shape channel, no encoder |
| `w_mean` | mean blend weight, chosen nested per held-out extractant | `topo_shape.py::nested_1` — the weight for an extractant is fitted on the others only |
| `f4.0`, `f3.5` | 4.0 Å, 3.5 Å cut-off | `--filtration-max`, the Vietoris–Rips radius over heavy atoms |
| `--no-triangles` | edges only | `topo_shape.py` part 2: the same network with 2-simplices removed |
| `anch_q60_q60` | no persistence features (reference) | `anchored_champion.py::CELLS` — the anchored champion, untouched |
| `g9` | 22 persistence summary statistics | `dataset.py` `("g9", "topology")`; TOPOLOGY_TESTS.md §3 enumerates them (H0/H1 totals, maxima, entropies, counts, birth/death means) |
| `g11` | 279 persistence-image pixels | `dataset.py`: "flattened GFN2-xTB persistence images (20×20) + PCA scores" |
| `anch_g9_bm`, `block_mean` | block-mean control | `anchored_champion.py`: "identical columns, but each replaced by its within-block mean" |
| composition block | one extractant under one set of conditions, across the lanthanide series | `dataset.py`: `composition_key = extractant_group ‖ condition key` |
| `S0` | topological network | `STACK_PREREGISTRATION.md` §2: "simplicial message passing over 3D complexes" |
| `T0w` | matched control network (same recipe, no 3-D input) | `STACK_PREREGISTRATION.md` §3: "same harness, folds, seeds and objective as S0, encoder removed" |
| `repaired` | the 2-D baseline | `oof_fcnn_std_scaler_ens16`: the fingerprint + descriptor network with its feature-scaling repaired (README portable findings: "+0.005 → +0.221"), 16 seeds |
| `sel_adj_logSF_r2`, `adj_r2` | adjacent-pair separation-factor R² (log₁₀ scale) | `evaluation.adjacent_pair_arrays` |
| `lo_3test` / `hi_3test` | corrected interval (three pre-registered tests) | `stack_test.py::_corrected(..., n_tests=3)` |

**Axes carry their quantity, scale and sign.** Panels C, D and E say
*adjacent-pair separation-factor R², log₁₀ scale, higher is better* — never a bare "R²", and
never confusable with the overall log *D* R² that the dropped `topo_blend_curve` mixed onto
the same axis. Panel E's axis says *positive = the added arm is better*. Panel A's axis says
*0 = ignored*. Panel B's axis stops at *r* = 1 and the values are printed to the left of
their markers rather than running the axis past the largest value a correlation can take.

**Sample sizes in the artwork**: 32-seed ensembles (A), 905 adjacent pairs (B), 8 shared
training seeds (C), 4 seeds drawn individually (D), and the full cohort line under panel E.

**Colour and shape.** Okabe-Ito from `pubstyle`, three semantic roles held across all five
panels: blue = *edges only*, vermillion = *higher-order topology (2-simplices or persistent
homology)*, grey = *no 3-D topology (reference or matched control)*. Identity never rests on
colour alone — circle vs triangle-up wherever the two arms sit side by side, hatch for the
block-mean control. The repository's own palette is deliberately not reused; its
`automl/tests/test_palette.py` found no four-colour subset of it that passes CVD and contrast
checks at once.

**Layout.** Declared size 7.09 × 6.75 in (180 mm double column); drawn extent 7.15 × 6.81 in
(**1.009×**), so the panels are the size they say they are when the figure is placed. One
figure-level legend below the panels, outside every data area, space reserved by constrained
layout. Reference rules are `vlines` bounded to the rows they refer to rather than `axvline`
across the whole panel — an `axvline` at zero in panel E drew straight through the cohort
footnote, and one in panel A through the ensemble note. Value columns sit outside the spine
bounds, and each bottom spine is bounded to the range where the scale is meaningful. Panel
letters placed by `pubstyle.add_panel_letters` from the measured tight bounding box.

**Typography.** `pubstyle` base 8.0 pt: axis labels 8.5, tick and row labels 7.5, every
annotation and column header 7.0. Nothing below 7.0 pt. Real minus signs (U+2212) in the
printed values, matching the tick labels.

**Automatic layout check**: `pubstyle.lint` reports **layout OK, 98 text artists checked**
(`figure.lint.json`). The render was read and inspected by eye four times — full size, and
in three full-resolution crops (panel A/B, panel D, panel E + legend) — with fixes between
each pass.

**Export**: 600 dpi PNG and a fully vector PDF from one figure object (0 raster XObjects,
5 embedded font subsets).

## Scientific integrity

**No number, cohort, metric or method changed.** Same tables, same 905-pair legacy
evaluation set, same leave-extractants-out CV (5 folds × 3 repeats), same nested
per-extractant blend weights, same 400-draw paired cluster bootstrap, same intervals. Spot
checks against `TOPOLOGY_TESTS.md`: weights 0.35 / 0.01 / 0.06 / 0.00; blend R² +0.3258 /
+0.3172 / +0.3184 against +0.3182 for tabular only; triangle differences −0.0199, −0.0066,
−0.0157, −0.0006; persistence +0.3188 → +0.0897 → +0.0022 → −0.0387, block-mean control
+0.2687. The script re-derives the two triangle differences and asserts them against the
`blend_r2_tri_minus_notri` field, and asserts the panel-D reference against the +0.3188 the
report quotes.

Four things found and **documented rather than silently fixed**:

1. **The original's "4-test corrected" label is wrong.** `automl/topo/stack_test.py` computes
   the widened interval as `_corrected(delta, lo, hi, 3)` with
   `z = norm.ppf(1 − 0.05 / n_tests)`, `n_tests = 3`, writes it to columns named `lo_3test` /
   `hi_3test`, and `STACK_PREREGISTRATION.md` declares "the three-test corrected interval".
   `fig_stack` labels the same band "4-test corrected" in both the axis label and the
   footnote — presumably because `stack_test.csv` has four rows, the fourth being the
   descriptive `desc_S2` secondary that was never part of the correction. **The interval
   values are unchanged**; only the words describing them are corrected to "three". This is
   a labelling defect in the upstream figure, not a numerical one.

2. **`anchored_champion.csv` contains two different cells with the same name.**
   `anch_q60_q60_ens4` appears twice: once for seeds 42/51/67/83 (+0.3188) and once for seeds
   91/103/107/109 (+0.3113). The persistence cells were run at 42/51/67/83, so the matched
   reference is the first occurrence, which is what the script takes — and it asserts the
   value against +0.3188 so that a change in row order fails loudly instead of silently
   swapping the reference. A duplicate primary key in a results table is a latent hazard;
   recorded, not repaired.

3. **Panel B's five correlations are transcribed, not computed.** Upstream derives them from
   out-of-fold parquets under `automl/artifacts/`, which is not committed upstream (see
   `REPRODUCIBILITY.md`) and therefore not vendored here. They are hardcoded in the script as
   a documented `CORRELATIONS` constant, cited to `TOPOLOGY_TESTS.md` §1 at commit `1cdeac3`,
   and flagged in `values.json` under `panelB_correlations.provenance`. Every other number in
   the figure is read from a vendored table.

4. **Panel E's baseline is not panels A–D's baseline, and the figure says so.** Panels A–D
   test topology inside the *anchored champion*'s shape channel (tabular shape +0.3182).
   Panel E's pre-registered stack tests were run against the *repaired 2-D network*
   (+0.2206), a much weaker comparison, on the same 905 pairs. The panel's grey strapline
   reads "pre-registered stack tests against a weaker, 2-D baseline" and each row label names
   its comparison. Panel E's first contrast is genuinely positive and clears zero even after
   correction; it is shown as such. What the panel contributes to the null is the third
   contrast — the only test of whether the gain is *specific to topology* — whose corrected
   interval spans zero, together with the pre-registered rule in `STACK_PREREGISTRATION.md`:
   *"If contrast 1 is positive but contrast 3 is not, the gain is generic ensembling and
   must be reported as such."*

**A gap I could not close from the vendored data.** There is no interval anywhere in the
vendored tables for the panel A–D comparisons: neither `topo_shape.json` nor
`anchored_champion.csv` carries a bootstrap, and the upstream code that writes them computes
none. Panel C therefore shows the matched seed count (8) and panel D shows all four seeds
individually, so the reader can see the spread behind each point — but **neither is a
confidence interval and neither is drawn as one**. The only intervals in the figure are
panel E's. Closing that gap needs the out-of-fold parquets, which are not committed upstream.

## Paper role

**Results** — the third headline result of the project ("negative results with teeth"), and
the figure the README's "Topology beyond the distance encoder: nothing to add" row currently
has no artwork for. It also supersedes the four earlier topology figures as the main-text
statement of what the topological arms are worth.

## Main message

Given the same slot the distance encoder earns a weight of 0.35 in, the simplicial encoder is
assigned 0.01 — and 0.00 when both are offered — because the two encoders' pair-level
predictions correlate at 0.963 and carry the same 0.18–0.20 correlation with the tabular
model's error; 2-simplices never beat edges alone at matched seeds, persistence descriptors
drive the shape model from +0.32 to −0.04, and the one pre-registered test of a
topology-*specific* gain has a corrected interval that spans zero.
