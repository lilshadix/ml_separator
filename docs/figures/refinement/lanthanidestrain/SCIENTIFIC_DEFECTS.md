# Scientific defects found during the lanthanidestrain figure audit

Not layout problems. Per the refinement brief these are documented rather than silently
corrected. Each was found by an independent audit of a rendered figure and then
**re-verified by hand against the repository's own data and reports**; the verification is
shown so a reader can repeat it.

---

## 1. `email_figB_result` — the headline comparison uses a baseline the project itself has superseded

**Figure:** `automl/reports/figures/email_figB_result.png`
**Code:** `automl/figures_pi_email.py::fig_result`
**Data:** `automl/reports/dualkey_arms.csv`, `best_stack.csv`, `stack_test.csv`

The figure exists to show that adding the 3D simplicial encoder raises adjacent-pair
separation R². Its reference bar is **"CatBoost alone, +0.142"**, taken from
`dualkey_arms.csv`:

| arm | `adj_r2_binned` |
|---|---|
| `CatBoost` | **0.1422** |
| `repaired` (fingerprint network) | 0.2206 |
| `S0` (3D encoder) | 0.2382 |
| `T0w` | 0.2006 |

That +0.1422 is CatBoost **as published** — the untuned RMSE configuration. The
repository's own `automl/reports/C15_RESULTS.md` §2, *"The tabular arm alone now exceeds the
published 3-model stack"*, records on the same 4,746-row / 162-extractant population:

| configuration | adjacent-pair log SF R² |
|---|---|
| `q60_rsm03_deep` (Quantile α = 0.6, depth 9, rsm 0.3) | **+0.2784** |
| `q60` | +0.2579 |
| `mae` | +0.2487 |
| CatBoost as published (the figure's reference bar) | +0.1422 |
| published 3-model stack (the figure's headline) | +0.2672 |

and states in the text: *"The tuned single tabular model reaches +0.2784, above the +0.2672
stack that required CatBoost plus a fingerprint network plus a simplicial network over 3D
structures."*

**Consequence.** The figure shows a 3-model stack at +0.267 beating "CatBoost alone" at
+0.142, and reads as evidence that the 3D arm is carrying that difference. On the project's
own numbers a *tuned* CatBoost alone — no fingerprint network, no 3D structures — reaches
+0.2784, above the whole stack. The comparison as drawn is against a configuration the same
repository has already replaced, and the gap it displays is mostly loss function and
hyperparameters, not 3D information.

**The caveat that cuts the other way, and belongs with it.** `C15_RESULTS.md` is explicit
that `q60_rsm03_deep` was selected on `screen_select` and that the full population contains
those rows, so +0.2784 "carries mild optimism"; its independent evidence is +0.0159 on the
held-out `report` third, which passed its pre-registered bar but whose 90 % CI spanned zero.
So the honest reading is not "the tuned model definitely beats the stack" — it is that
**+0.142 cannot stand unqualified as "CatBoost alone"**, and any figure using it as the
reference must say which configuration it is and what a tuned one reaches.

**What was done.** Nothing to the data. The refined `figure_01_headline_result` does not
present +0.142 as an unqualified baseline; see its `notes.md`. The separate, better-founded
quantity — the 3D arm's own *contrast* — is unaffected by this and is reported directly from
`anchored_3d_confirm.json`: **+0.0156** on 444 never-touched pairs, +0.0146 on the 905
legacy pairs, +0.0149 on all 1,349, at a fixed blend weight `w = 0.35` with
`primary_pass = true`.

---

## 2. `email_figB_result` and most of the set — intervals exist in the source and are not drawn

The same figure draws seven bars with no error bars, although the two CSVs the function
already opens carry the intervals: `best_stack.csv` gives the +0.267 bar as
**[0.1905, 0.2955]** and `stack_test.csv` gives the +0.251 blend as **[0.1819, 0.2782]**.
Those whole-set intervals span essentially every bar in the panel (0.142 to 0.267), so the
ranking the title asserts has no visual basis in the render.

This is not isolated. Across the 35 audited figures, **missing uncertainty was the second
most common high-severity defect (26 occurrences)**, behind misleading encoding (36) and
ahead of unreadable-when-scaled (24) — in a repository whose result tables usually *do*
carry bootstrap intervals.

---

## 3. `re_fig4_calibration` — the annotation does not describe the bar it labels

**Figure:** `automl/reports/figures/re_fig4_calibration.png`
**Code:** `automl/figures_reanalysis.py::fig_calibration`

The subtitle claims that "even a free monotone map leaves predictions at about half the true
spread". The orange bar is not the free monotone map: the code selects
`cal.loc[cal['r2'].idxmax(), 'span_ratio']` over `{scale, affine, isotonic}`, and for the
plotted model that argmax is **`scale`** — a one-parameter linear rescale — in both panels.
The isotonic (genuinely free monotone) span ratios are different numbers. The figure also
draws a `y = 1.0` reference line straight through its own legend, striking out the entry
"after nested recalibration".

The audit's verdict on this figure is **drop**: it carries four numbers, nothing in the
repository cites it, and its companion `CALIBRATION_RESULTS.md` already presents the same
numbers across seven models and four transforms with the bootstrap intervals the figure
omits.

---

## 4. `fig6_uncertainty_calibration` — the plotted quantity cannot support the claim

**Figure:** `automl/reports/figures/fig6_uncertainty_calibration.png`

The title asserts a "usable error bar". The chart plots realised error by quintile of
ensemble spread, which can show *rank order* only. The predictor itself is discarded: the
source CSV's `spread` column runs 0.146 → 0.404 log D across the quintiles while the
realised errors it is meant to bracket run 0.78 → 1.06 log D — the "error bar" is
**under-dispersed by roughly a factor of four**, and the chart as drawn hides exactly that.
Five bars carry no error bars and no n, in the one figure of the set whose subject is
uncertainty; the Q3/Q4 inversion (0.90 vs 0.87) and the Q1/Q2 step (0.78 vs 0.82) cannot be
judged.

---

## 5. `re_fig1_ceiling` — a committed figure asserting a result the study has withdrawn

**Figure:** `automl/reports/figures/re_fig1_ceiling.png` (PNG and PDF both committed)
**Data:** `automl/reports/ceiling_test.csv`

The figure states that the model reaches about 39 % of a **+0.679** attainable ceiling.
Every row of the table it is built from is marked invalid, and the table carries a
`withdrawn_reason` column saying why:

| key | estimator | `ceiling_r2` | `valid` | `withdrawn_reason` |
|---|---|---|---|---|
| composition_key | E1 row-level split-half | −1.066 | **False** | non-representative subset: cells acquire duplicates non-randomly |
| composition_key | **E2 condition-level** | **+0.679** | **False** | **measures condition variation, which the model predicts and the metric averages out** |
| composition_key | E3 propagated | −4.366 | **False** | non-representative subset: cells acquire duplicates non-randomly |
| strict_composition_key | E1 row-level split-half | −2.028 | **False** | non-representative subset … |
| strict_composition_key | E2 condition-level | +0.530 | **False** | measures condition variation … |
| strict_composition_key | E3 propagated | −8.642 | **False** | non-representative subset … |

```bash
python -c "
import pandas as pd
c = pd.read_csv('automl/reports/ceiling_test.csv')
print(c['valid'].value_counts().to_dict())"     # -> {False: 6}
```

The +0.679 the figure displays is the E2 row, whose recorded reason for withdrawal is that
it *measures the wrong quantity*. The repository's own `README.md` §0 agrees — "Ceiling:
**not identifiable** from this dataset (supersedes the old 0.53)" — and the current code
already refuses to draw it, printing `skip ceiling: no valid estimator`.

**Consequence.** A stale PNG and PDF asserting a withdrawn number sit committed beside
siblings that regenerate cleanly, with nothing in the file or its name marking it as
superseded. Anyone reading `automl/reports/figures/` — or a reviewer handed the directory —
cannot tell the retracted figure from the current ones.

**What was done.** Nothing to the data. The figure is not carried into the refined set. The
files should be deleted from the repository, or moved to a clearly named `withdrawn/`
directory; leaving them where they are is the defect.

---

## 6. `email_figA_coverage` — the pair series is endpoint *slots* read as *pairs*

**Figure:** `automl/reports/figures/email_figA_coverage.png`

The figure compares each lanthanide's share of measurement rows with its share of the pairs
the metric scores. The second series counts **pair endpoint slots**, not pairs: each of the
905 adjacent pairs contributes two slots, so the denominator is 1,810. Europium occupies 196
slots, which the original renders as about **11 %** — but as a share of the 905 *pairs* it
is 196/905 = **21.7 %**, twice the figure's number. The Pm gap, which is what shortens the Nd
and Sm bars, is also unmarked.

**What was done.** The refined `figure_06_pair_coverage` labels the series explicitly as
"scored-pair slots (100 % = 1,810 = 2 × 905 pairs)", marks the Pm gap with "no pair spans
the gap", and prints both the counts and the most-÷-least ratios, so the quantity cannot be
misread. No number changed.

---

## 7. Recurring, and fixable once

Across 35 audited figures, 34 were rated high severity overall. The high-severity defect
kinds, by frequency:

| count | defect |
|---|---|
| 36 | misleading encoding |
| 26 | missing uncertainty |
| 24 | unreadable when scaled to a journal column |
| 12 | other |
| 5 | text over data |
| 4 | text-text overlap |
| 3 | empty or broken panel |
| 3 | legend obscures data |

The three leaders are not decoration problems. "Unreadable when scaled" is a single global
fix — the house style saves at 160 dpi with 9–12 pt type on 7–9 inch canvases, so a
reduction to an 86 mm column puts almost every label between 3 and 5 pt. The other two need
per-figure work.

---

*Every claim on this page was re-derived from the repository's own tables and reports during
the refinement pass; none rests on an agent's report alone.*
