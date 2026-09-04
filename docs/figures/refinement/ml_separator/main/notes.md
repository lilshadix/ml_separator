# Main results figures — `ml_separator`

Two figures. One claim each. Everything that used to be printed inside the panels now lives
in the captions below, which is where a reader expects it.

The detailed, fully annotated versions remain in
`figure_refinement/ml_separator/figure_0*/` as the audit trail — they carry the cohort
accounting, the controls and the uncertainty machinery, and they are what a referee should
be shown if they ask.

---

## R1 — What one measurement on an unseen extractant is worth

**Claim.** A model that has never seen an extractant predicts its distribution ratios with
about 1.0 log units of error, and one measured point removes 37 % of that — more than every
modelling improvement in the study put together.

**Caption.**
> **One measurement on a new extractant is worth more than any model change.** Mean absolute
> error in log₁₀ *D* on 99 held-out extractants, against the number of measurements taken on
> the extractant itself. Extractants are held out by Tanimoto-0.7 chemotype, so neither the
> extractant nor any analogue closer than *T* = 0.7 appears in training. Orange: the frozen
> pipeline — a zero-shot prediction, then a shrunk calibration fitted to the *k* measured
> points. Grey: the same *k* measurements fitted with no model at all. Points are means over
> five split seeds × twelve draws of which rows are measured; the measured rows are always
> disjoint from the rows scored. The first measurement removes 0.38 log units
> (chemotype-block bootstrap 95 % CI [0.26, 0.56]); the second removes 0.09 and the fifth
> 0.03. The average hides real spread: 33 of the 99 extractants are *worse* after one
> measurement than before it, so the gain is an expectation, not a guarantee.

**Source.** `runs/gen10_final/final_locked/kshot_detail.parquet` via
`figures/derived/kshot_per_ligand.csv`. Values in `R1_values.json`. Distilled from
`figure_refinement/ml_separator/figure_02_few_shot_frontier/`.

**What was cut, and why it is safe.**

| cut | where it went |
|---|---|
| two extra deployment rules (baseline model, + shape recomposition) | they differ from the plotted rule by 0.02–0.04 and are a *methods* comparison, not a result — detailed figure, panels A and C |
| the oracle acquisition bound | it bounds *which point to measure*, a separate question — detailed figure and R-series successor |
| the marginal-gain bar panel | its three numbers are now one sentence in the caption |
| the per-extractant scatter | the "33 of 99 are worse" caveat it carried is stated in the caption |
| seed-range whiskers, cohort strip, "% improved" row | caption |

Text artists: **17** (budget 26). Layout linter clean.

---

## R2 — Unseen titrations come out flat

*(caption written by the R2 build; see below)*
