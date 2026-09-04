# Figure 6 — Which comparisons the adjacent-pair score averages over

## Source

Repository: `lanthanidestrain` (`github.com/mironovb/lanthanidestrain`, `main`,
commit `1cdeac39d0b018529a735f967d3c83120d04c714`, 2026-08-24 — `data/PROVENANCE.json`)
Original script: `automl/figures_pi_email.py::fig_coverage`
Original render: `automl/reports/figures/email_figA_coverage.png`, kept here as
`original.png`
Caption context: `automl/reports/PI_REPLY_2026-08-03.md`, the attachment list
("per-metal row share against pair share")

Input tables, both vendored under `figure_refinement/lanthanidestrain/data/`:

* `pair_coverage.csv` — the only table the original used. Written by
  `automl/pair_coverage.py` over the published `ok_only` subset
  (`geometry_ok & has_3d`): per lanthanide, `rows`, `pairs_binned`, `row_share`,
  `pair_share_binned`. Panel A is these four columns and nothing else.
* `adjacent_decomposition_position.csv` — written by
  `automl/topo/adjacent_decomposition.py`; `n_pairs` per neighbour position. Panel B is
  that one column. It is a *second* table over the *same* enumeration, not a new
  computation (see **Scientific integrity**).

Cohort: 4,746 measurements over 14 lanthanides, blocked by extractant × binned
conditions, giving 905 adjacent pairs — the same 905 that
`anchored_3d.json` and `series_shape_summary.json` report for the headline
adjacent-pair result. Pm is absent from the dataset, so `n_positions` is 12, not 13.

## Problem

Defects visible in `original.png`:

* **The y axis label is wrong for one of the two series it labels.** "% of the modelled
  dataset" sits on an axis carrying two bars with two different denominators: the orange
  bars are a percentage of 4,746 measurements, the blue bars a percentage of 1,810
  *pair endpoints*. Neither denominator is stated, and "the modelled dataset" names only
  the first.
* **The title mis-states its own number.** "Eu is 28 % of the rows but 11 % of the pairs"
  — 11 % is 196/1,810, the share of pair *endpoints*. Eu is one of the two lanthanides in
  196 of the 905 scored pairs, i.e. 22 % of the pairs. The plotted number is right; the
  sentence describing it is off by the factor of two that the doubling introduces.
* **No counts anywhere.** Not 4,746, not 905, not a single per-element n. A reader cannot
  see that Lu enters 47 pairs and Eu 196, which is exactly the "small n could mislead"
  case a coverage figure exists to expose.
* **The Pm gap is invisible.** The x axis runs La Ce Pr Nd Sm Eu … at even spacing, so
  nothing signals that Nd and Sm are not neighbours. That gap is the reason Nd's 562
  measurements — the second-largest row count in the series — yield only 97 pair slots,
  fewer than nine of the thirteen other lanthanides. Upstream states it explicitly
  (`evaluation.adjacent_pair_metrics`: "Pm is absent from the dataset … the Nd/Sm pair
  (4 → 6) is correctly *not* counted as adjacent"); the figure hides it.
* **The mechanism is absent.** `pair_coverage.py`'s own docstring gives the two reasons
  the imbalance shrinks — replicates are averaged within a (block, metal) cell, and a
  pair exists only when both neighbours are in the same block. Neither is on the figure,
  so the shrinkage looks like luck.
* **The figure is about pairs and never shows a pair.** Nothing in it says which
  comparisons the headline adjacent-pair R² is an average over, or how unevenly the 905
  are spread across the 12 neighbour positions (108 for Eu–Gd against 44 for Tm–Yb).
* **Paired vertical bars make the per-element change hard to read.** The figure's claim is
  a *movement* (28 % → 11 %); grouped bars force the eye to jump between two bar tops per
  element, and the movement itself ends up as prose in the title.
* **A two-line, 11 pt title carrying the entire claim** occupies the top fifth of the
  canvas, with both summary statistics (11.4× → 4.2×) available only as prose.
* **Legend inside the axes**, upper right, in the band immediately beside the tallest bar.
* **Palette** is the repository's own `C` dict (orange/blue), not a colour-vision-checked
  one; this repository's `test_palette.py` found no 4-colour subset of it passing CVD and
  contrast checks together.

## Changes

**Two panels, one shared series axis.** Panel A is per lanthanide, panel B per neighbour
pair, sharing the same vertical axis; a pair bar sits at the half-integer position between
the two elements it compares, so the reader can see the pair and its two members at once.
The absent element keeps an empty, unbanded slot at its position in both panels, and the
two pair slots that would have touched it are simply blank — the gap is drawn rather than
described. The 12 pair names are listed in a reserved right-hand column of panel B so no
name can be mistaken for part of a bar.

**Direct comparison instead of grouped bars.** Panel A draws one row per lanthanide with
the two shares as two markers joined by a connector. Whether a lanthanide gains or loses
share, and by how much, is one line segment; Eu's collapse is the longest segment on the
panel. Both markers are on one axis, so the two distributions are directly comparable, and
a dashed reference at 1/14 (and, in panel B, at 1/12) shows what perfectly even coverage
would look like for each.

**Counts in the artwork.** Panel A carries a two-column count table — measurements and
pair slots per lanthanide, with a totals row (4,746 / 1,810) and a most ÷ least row
(11.4× / 4.2×, the two ratios the old title carried as prose). The table lives right of a
divider rule, outside the plotted x range, so no number can be read as a bar length. Panel
B labels every bar with its count and carries "905 pairs in total" and "most ÷ least 2.5×".

**Denominators stated where the number is.** The legend reads "measurements (100 % =
4,746)" and "scored-pair slots (100 % = 1,810 = 2 × 905 pairs)" — the doubling that the
old title got wrong is now on the figure. The x axis label is "share of the total (%)",
with the totals themselves in the legend and in the table.

**The mechanism, in the space the table leaves free**: "a pair is scored only when both
neighbours were measured in the same extractant × conditions block; replicates averaged
first", and, at the empty element slot, "Pm absent — Nd and Sm each have one neighbour".
Both are transcriptions of upstream docstrings, not new claims.

**No title.** The paper caption carries the explanation. Each panel has a grey strapline
that orients rather than concludes ("per lanthanide — La to Lu, ionic radius decreasing
downward"; "per neighbour pair — each bar spans the two it compares").

**Legend outside the data**, one row beneath both panels (`loc="outside lower center"`),
space reserved by constrained layout.

**Palette** from `pubstyle` (Okabe-Ito): blue `#0072B2` for a measurement, vermillion
`#D55E00` for a scored pair, in both panels and in the count-table headers, so colour
carries the quantity rather than the panel. Shape is redundant with colour (circle /
diamond) because Sm's two shares differ by 0.13 points and the markers coincide; the
diamond is drawn larger and behind so both remain visible. Alternating row bands run
across both panels so an element named once can be carried into the other panel.

**Typography.** `pubstyle` base 8.0 pt: axis labels 8.5, tick labels 7.5, bar counts and
pair names 7.5, count table and notes 7.0, panel letters 9.5. Nothing below 7.0 pt. The
figure is 180 mm wide, i.e. reproduced 1:1, so these are the printed sizes. Panel letters
placed by `pubstyle.add_panel_letters` from the measured tight bounding box. Export: 600
dpi PNG and a fully vector PDF (0 image XObjects) from one figure object.

**Naming.** No repository identifier reaches the artwork; the script asserts this over
every drawn text artist before saving, so a later edit cannot reintroduce one.

| table column / key | figure label | source of the reading |
|---|---|---|
| `rows`, `row_share` | measurements | `pair_coverage.modelled_rows` — one measured row |
| `pairs_binned`, `pair_share_binned` | (scored-)pair slots | `pair_coverage.coverage` — one appearance in an adjacent pair; each pair counted at both ends |
| `metal` | element symbol (La … Lu) | as-is |
| `lanthanide_index` | position down the series | never printed; used only for row order and adjacency |
| `composition_key` ("binned") | extractant × conditions block | `dataset.build_matrix`: extractant group + the ten binned condition columns (acid class and strength, nitrate activity, pH, diluent family, modifier class, temperature, contact time, shaking time, phase ratio) |
| `pair` (`Ce-Pr`) | Ce–Pr | as-is, hyphen set as an en dash |
| `n_pairs` | adjacent pairs scored (count) | `adjacent_decomposition.group_decomposition` |
| `pairs_strict` | *not drawn* | the strict-blocking variant; the original did not draw it either |

**Automatic layout check:** `pubstyle.lint` reports **layout OK, 101 text artists
checked** (`figure.lint.json`); the render was then inspected by eye at full size and in
three full-resolution crops (the near-coincident Sm markers, the bar-count labels where
one falls on the reference line, and the legend row).

## Scientific integrity

**No number, cohort or method changed.** Panel A plots `row_share` and
`pair_share_binned` from `pair_coverage.csv` exactly as `fig_coverage` did, in the same
series order, over the same `geometry_ok & has_3d` subset with the same binned blocking
key. Nothing is recomputed: `values.json` carries every plotted number.

**Panel B is a second vendored table, not a new derivation.**
`adjacent_decomposition_position.csv` counts the same adjacent pairs, enumerated by the
same rule (`|Δ index| = 1` inside an extractant × binned-conditions block, replicates
averaged first) in a different module. The script asserts the three identities that tie
the two tables before it draws anything, and refuses to draw if any fails:

* every position joins true neighbours (`i_hi − i_lo = 1`) — 12 positions, none spanning
  the absent element;
* each lanthanide's `pairs_binned` equals the sum of `n_pairs` over the positions it
  touches — holds for all 14;
* total slots = 2 × total pairs — 1,810 = 2 × 905, and 905 is the pair count
  `anchored_3d.json` reports for the headline result.

The shares in the file also reproduce from the raw counts (`np.allclose`).

**One defect found in the original, documented and not silently fixed.** The original
title reads "Eu is 28 % of the rows but 11 % of the pairs". The 11 % is
`pair_share_binned` = 196/1,810, which is a share of pair *endpoints*: each of the 905
pairs is counted once at each of its two lanthanides. Eu is a member of 196/905 = 22 % of
the scored pairs. The plotted bar was and is correct; the sentence describing it was not.
I have not changed the quantity — panel A still plots 10.8 % for Eu, from the same column
— but the legend now states the denominator ("100 % = 1,810 = 2 × 905 pairs"), the column
is called "pair slots" rather than "pairs", and panel B shows the 905 pairs themselves so
the two counts cannot be conflated. The same ambiguity is what made the original y axis
label ("% of the modelled dataset") wrong for one of its two series.

**Two things this figure deliberately does not do.** It carries no model score, so
nothing here can be mistaken for a level or a contrast; and it draws only the binned
blocking key, as the original did, leaving the strict variant (`pairs_strict`, 2,834 slots
= 1,417 pair instances, because a finer key splits one averaged cell into several
separately-counted ones) to the figure that is about blocking keys.

**Nothing is missing that the original had.** The two ratio statistics from the old title
are in the count table; the claim itself belongs to the caption.

## Paper role

**Methodology** — the coverage figure that establishes which comparisons the
adjacent-pair result is an average over, and how unevenly they are distributed across the
series. It is a prerequisite for reading any per-metal or per-pair result, and it is the
answer to "does the adjacent-pair metric inherit the 28 % Eu row imbalance?".

## Main message

Averaging replicates within a block and requiring both neighbours in the same block cut
the series imbalance from 11.4× to 4.2×: Eu supplies 27.6 % of the 4,746 measurements but
only 10.8 % of the 1,810 pair slots. The 905 scored comparisons are still not evenly
spread — 108 of them are Eu–Gd and only 44 are Tm–Yb, and none at all spans the absent
Pm, which leaves Nd and Sm with a single neighbour each.
