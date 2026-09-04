# lanthanidestrain — figure refinement summary

Repository: `github.com/mironovb/lanthanidestrain`, `main` @ `1cdeac39`
Scope: all **37** rendered figures in `automl/reports/figures/` were audited, one independent
audit per figure, image first and code second. Seven were rebuilt into
`figure_refinement/lanthanidestrain/`.

Two documents carry findings that are not about layout and must be read with this one:

* [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — **19 of the 37 figures regenerate from what is
  committed; 18 do not.** This is the binding constraint on what could be refined.
* [`SCIENTIFIC_DEFECTS.md`](SCIENTIFIC_DEFECTS.md) — six defects that are not cosmetic,
  each re-verified by hand against the repository's own tables and reports.

## 0. What the audit found, in one line

Across 37 figures, **34 were rated high severity overall**, and the leading high-severity
defect was not crowding — it was **misleading encoding (36 occurrences)**, ahead of
**missing uncertainty (26)** and **unreadable-when-scaled (24)**. The house style is
careful (a CVD-tested palette, matched PNG + PDF, titles built as f-strings over the data
in three of the four modules); the failures are of a different kind: bare point estimates
where the source CSV already carries the interval, arms selected on the statistic being
drawn, and titles asserting more than the panel shows.

---

## 1. Rebuilt figures

| Figure | Purpose | Original problem | Main fix | Paper section | Status |
|---|---|---|---|---|---|
| `figure_01_headline_result` | Adjacent-pair log SF R² of each arm, and what the 3D arm actually contributes | Reference bar is untuned CatBoost (+0.142) while the repo's own C15 report has tuned CatBoost alone at +0.278; seven bars, no intervals, though `best_stack.csv` and `stack_test.csv` carry them; the current headline (+0.326 / +0.288 / fresh-444 +0.0156) appears in **no** figure in the repository | Levels and the paired *contrast* separated so no reader can subtract two bar heights to get the 3D effect; intervals drawn from the tables that already hold them; the pre-declared confirmation on 444 never-touched pairs shown beside the legacy population | Results | rebuilt |
| `figure_02_preregistered_controls` | The three pre-registered contrasts, including the matched-control null | The retraction note ("corrected, this one spans zero") thrown left across the zero rule and out of the axes; value labels anchored at the interval end rather than the estimate; the interval key existed only in the axis label | Contrasts named in words; labels at their estimates; thick 90 % and thin corrected intervals with a key in the artwork | Results | rebuilt |
| `figure_03_limits` | The two qualifications: it is 3D and not topology; the gain depends on how "identical conditions" is defined | The left-panel null was sold as equivalence although its interval reaches +0.047 — larger than the celebrated effect; the two strict rows are the same number to 1 × 10⁻¹⁷ and were drawn twice with no note; "P = 0.93" reads as a p-value | The two qualifications visually separated as different kinds of caveat; the null's interval shown rather than asserted; the duplicated row collapsed or annotated | Results | rebuilt |
| `figure_04_topology_adds_nothing` | Topology beyond the distance encoder adds nothing | Spread across four figures, several with hard-coded numbers; `topo_forest` titled "every test clears zero" when its own matched-control row does not | One five-panel figure carrying four independent kinds of evidence for the null — blend weight (0.35 vs 0.01), encoder correlation (0.963), triangles-minus-edges (all negative), persistence features collapsing the model — and the pre-registered stack tests, where the topological network beats the 2-D baseline (+0.035, clears zero) but **not** its matched control (+0.030, spans zero) | Results | rebuilt |
| `figure_05_uncertainty_calibration` | Whether ensemble spread is a usable error bar, and whether recalibration repairs the compression | The quintile-rank bars cannot show calibration at all: the quoted spread (0.15–0.40 log D) under-disperses the realised error (0.78–1.06) about fourfold, and the chart hid exactly that. Its companion `re_fig4` labelled a one-parameter rescale as "a free monotone map" and drew its own reference line through its legend | Both series plotted on one log D axis with the ratio annotated per quintile (5.3× … 2.6×), so the under-dispersion is the visual event; all four transforms named and shown for both key definitions, with 90 % cluster-bootstrap intervals and the "best of three transforms, uncorrected" caveat stated | Supplementary | rebuilt |
| `figure_06_pair_coverage` | Which comparisons the headline average is actually over | The second series counts pair **endpoint slots**, not pairs, so europium's share read as ~11 % where its share of the 905 pairs is 21.7 %; the Pm gap — the reason the Nd and Sm bars are short — was unmarked | Series explicitly labelled "scored-pair slots (100 % = 1,810 = 2 × 905 pairs)"; the Pm gap marked with "no pair spans the gap"; counts and most-÷-least ratios (11.4× collapsing to 4.2×) printed | Methodology | rebuilt |
| `figure_07_conformer_noise` | How much within-family 3-D variation tracks the lanthanide contraction rather than conformer noise | Medians only with n unshown across a 12× range; `contraction` ranks 9th by median but 4th by mean; raw CSV identifiers as labels | Absorbs `re_fig3_energy_snr`; n and spread shown; reader-facing block names | Supplementary | rebuilt |

## 2. Not rebuilt

**Blocked by missing inputs** (see `REPRODUCIBILITY.md`): `fig1_baseline_decomposition`,
`fig2_block_ablation`, `fig4_architectures`, `fig5a/5b_parity`, `fig7_per_metal`,
`fig8_metal_free_3d`, `fig9_split_variability`, `fig10_split_series` need
`automl/reports/all_results.csv` or `automl/artifacts/`, neither of which is committed;
`pi_fig1`–`pi_fig5` have no generator among the 163 committed `.py` files at all.

**Superseded or dropped on the evidence:**

| Figure | Verdict | Reason |
|---|---|---|
| `re_fig1_ceiling` | **drop and delete** | every row of `ceiling_test.csv` is `valid = False` with a `withdrawn_reason`; the README supersedes the number; the code already refuses to redraw it, so a stale PNG stands beside fresh siblings |
| `fig8_metal_free_3d` | drop | the three presets it celebrates fail the paired test; best-of-N row selection is the cause |
| `pi_fig5_per_extractant` | drop | no generator, no source CSV, and the render refutes its own caption — one extractant supplies about half the total error reduction |
| `re_fig4_calibration` | folded into `figure_05` | four numbers already tabulated more fully in `CALIBRATION_RESULTS.md` |
| `fig6_uncertainty_calibration` | replaced by `figure_05` | the plotted quantity cannot support the claim |
| `fig5a_parity_baseline` | superseded | pixel-indistinguishable from `fig5b`; cited nowhere |
| `re_fig3_energy_snr` | folded into `figure_07` | same claim, one comparable axis |
| `pi_fig1`–`pi_fig4` | superseded | by `email_figB` / `email_figC` / `re_fig5_encoder` / `topo_adjacent_parity`; all four have no generator and carry superseded numbers |
| `fig2`, `fig4`, `fig7`, `topo_ladder`, `topo_tradeoff`, `topo_forest`, `topo_adjacent_parity` | superseded | by the rebuilt figures above, or blocked and superseded both |

## 3. Recommended main figure set

The audit's synthesis proposes six main figures. Four are covered by the rebuilds; two
cannot be built from what is committed and are recommendations, not deliverables.

| | establishes | status here |
|---|---|---|
| **F1** target, cohort and protocol | absolute log D is between-extractant ranking; condition effects cancel on differencing; no single split decides anything | **partly built** — `figure_06_pair_coverage` is its middle panel; the other two need `all_results.csv` and the champion OOF |
| **F2** the chemistry benchmark | GFN2 underestimates the contraction 2.47×, g-xTB reproduces it on 71/71 ligands | **not buildable** — see §5 |
| **F3** primary predictive performance | the arm ladder and the pre-declared paired contrasts | **built** — `figure_01` + `figure_02` |
| **F4** generalisation and confirmation | the 3D contrast survives a pre-declared look on 444 never-touched pairs | **partly built** — the confirmation is in `figure_01`; the parity panel needs the champion OOF |
| **F5** where the 3D signal enters and stops | a learned 3D representation pays; topology beyond the distance encoder does not | **built** — `figure_04` + `figure_03` |
| **F6** negative results in one forest | fixing the chemistry does not improve the score | **not built** — the rows are table-only upstream and several need absent artefacts |

## 4. Defect patterns — fixable once, in `automl/figures.py`

From the audit synthesis, in descending value:

1. **No uncertainty anywhere.** About 30 of 37 figures plot bare point estimates while the
   CSV the figure opens already carries `arm_lo`/`arm_hi`, `lo`/`hi` or a per-seed spread.
2. **Titles written ahead of the data.** Several assert conclusions their own panels refute.
   Three of the four modules already state the right rule in their docstrings — titles as
   f-strings over the plotted values — and it is simply not applied everywhere.
3. **Selection on the plotted metric.** `sort_values(metric).iloc[0]`,
   `max(others, key=_r2_of)`, best-of-six against best-of-one.
4. **Authored at screen size.** 160 dpi, 7.5–9 pt type on 160–330 mm canvases, and
   `bbox_inches="tight"` lets an unwrapped title rather than the data set the width.
5. **`pdf.fonttype` left at 3.** Type-3 fonts are rejected by several publishers' preflight;
   the fix is one rcParam.
6. **Value labels anchored to the interval end** rather than the estimate.
7. **Machine identifiers as labels** — `BEST +G5`, `S0`, `T0w`, `plus_g14c` — and titles
   derived from filename stems.

Items 4 and 5 are a single global change; the shared `pubstyle.py` in this tree implements
both, plus the layout linter, and can be lifted across.

## 5. Results with no good visualisation

1. **The chemistry benchmark — headline 1, and the most publishable single result in the
   repository — has no figure at all.** Nothing plots the per-ligand compliance `c_L`, the
   71/71 paired improvement, GFN2's linear-in-Z parameter interpolation, or the +1.15 eV
   Gd half-shell break. Verified independently: no figure module references compliance,
   contraction or Shannon radii, and `automl/qc/compliance_test.py` **prints** its results
   rather than writing a table, so the per-ligand data is not committed either. Building it
   needs one script change upstream (write the table) before any figure work.
2. **The current headline number is unplotted upstream.** +0.326 / +0.288 / fresh-444
   +0.0156 appear in no committed figure; every performance panel shows +0.238 to +0.267
   from the superseded July stack, and two of them label it "BEST". `figure_01` in this tree
   fixes that.
3. **The anchored decomposition has no schematic** — a reader cannot see what the level
   head, the tabular shape head and the encoder shape each contribute.
4. **"Fixing the chemistry does not improve the score"** is table-only, including the
   scale-free reversal that killed the one positive chemistry arm.
5. **The collaborator cross-validation** — the strongest external-validity evidence in the
   project — is unplotted.

## 6. Figures whose interpretation still needs checking

* Everything in `SCIENTIFIC_DEFECTS.md`, especially the untuned reference bar (§1) and the
  withdrawn ceiling (§5).
* Any conclusion drawn from the eight blocked figures: they cannot be regenerated, so
  nobody can check what they show against the current tables.

---

### Regenerating

```bash
for f in figure_refinement/lanthanidestrain/figure_*/figure_script.py; do .venv/bin/python "$f"; done
.venv/bin/python figure_refinement/common/verify_exports.py
```

Each script reads the vendored result tables in `data/` (`PROVENANCE.json` records the
upstream commit), so no re-clone is needed.
