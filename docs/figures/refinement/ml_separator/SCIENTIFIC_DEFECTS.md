# Scientific defects found during the figure audit

These are **not** layout problems, and per the refinement brief they are documented here
rather than silently corrected. Each was found by an independent audit of a rendered figure
and then **re-verified directly from the data by hand**; the verification command is given
so a reader can repeat it. Neither defect changes any number in the refined publication set
— both live in older diagnostic figures that the audit marks superseded or supplementary.

---

## 1. `cross_series_transfer` — the transfer matrix is symmetric, so its arrow is unsupported

**Figure:** `runs/gen8_architecture/figures/cross_series_transfer.png`
**Estimator:** `scripts/gen8_calibration_geography.py::transfer_matrix`
**Renderer:** `scripts/gen8_figures.py::transfer_heatmap`
**Data:** `runs/gen8_architecture/cross_series/transfer_matrix.csv`

The figure is titled *"Gain from one measurement: measured series (rows) → predicted series
(columns)"* and is read as a directional statement: a measurement taken in curve type A
repairs curve type A and harms curve type B. The post-calibration quantity it is built from
cannot support a direction.

`transfer_matrix` forms, per ligand,

```python
residual = block["log_D"] - block["prediction"]
diff     = np.abs(residual[:, None] - residual[None, :])     # symmetric by construction
sub      = diff[np.ix_(source_mask, target_mask)]
mae      = float(np.nanmean(sub))
```

`diff` is symmetric, so the mean of the `(source × target)` block equals the mean of the
`(target × source)` block exactly. Verified on the stored matrix:

| quantity | max &#124;M − Mᵀ&#124; | |
|---|---|---|
| `mae` (post-calibration) | **4.441 × 10⁻¹⁶** | symmetric |
| `n_ligands` | **0** | symmetric |
| `zero_shot` | 1.895 | directional |
| `gain = zero_shot − mae` | 1.895 | directional |

```bash
.venv/bin/python -c "
import pandas as pd, numpy as np
d = pd.read_csv('runs/gen8_architecture/cross_series/transfer_matrix.csv')
M = d.pivot(index='calibration_axis', columns='target_axis', values='mae')
c = sorted(set(M.index) & set(M.columns)); A = M.loc[c, c].to_numpy(float)
print(np.nanmax(np.abs(A - A.T)))"
```

**Consequence.** Every off-diagonal asymmetry a reader sees in that heat map — for example
acid→contact_time +0.69 against contact_time→acid +0.41 — comes from a *different zero-shot
baseline being subtracted from an identical post-calibration MAE*, not from calibration
transferring better in one direction than the other. The diagonal-versus-off-diagonal
contrast (calibration is series-local) survives, because that comparison does not depend on
direction; the directional reading does not.

**What was done.** The refined figure
(`figure_07_series_local_calibration/`) presents the quantity as symmetric and says so, and
the underlying estimator was **not** modified or re-run. Fixing the estimator — fit the
offset on source rows, score target rows — is a re-run, not a re-plot, and is left as a
recommendation.

**Secondary issues in the same figure**, all verified: the coverage filter is
`n_ligands >= 3`, and 30 of the 49 cells have `n ≤ 5` (median 5); the three most saturated
cells rest on n = 3, 8 and 3 while the palest diagonal cell rests on n = 85; the diverging
colour limit is `max|gain| = 1.823`, set by a single n = 3 cell, so most of the negative half
of the scale is unused; missing cells render in the same white as zero gain.

---

## 2. `error_budget` — two runs are pooled, double-counting three split seeds

**Figure:** `runs/gen7_architecture/figures/error_budget.png`
**Renderer:** `scripts/gen7_figures.py::figure_error_budget`
**Data:** `runs/gen7_architecture/oracles/scores_by_seed.csv` **and**
`runs/gen7_architecture/finalists/scores_by_seed.csv`

The script pools the two run tables with `pd.concat` and takes a plain per-model mean.
`NULL_metal_cond` — the no-ligand-information baseline that carries the figure's headline —
is the **only** model present in both files, so it alone is averaged over a union that
repeats three seeds:

```
NULL_metal_cond after concat: 8 rows
  seeds  104729, 104729, 130363, 130363, 155921, 155921, 196613, 262147
  duplicated (model, seed) pairs: 3
  pooled mean          1.0953      <- what the figure plots
  oracles-only mean    1.0882      (3 seeds)
  finalists-only mean  1.0995      (5 seeds)
```

```bash
.venv/bin/python -c "
import pandas as pd
a = pd.read_csv('runs/gen7_architecture/oracles/scores_by_seed.csv')
b = pd.read_csv('runs/gen7_architecture/finalists/scores_by_seed.csv')
c = pd.concat([a, b]); s = c[c.model == 'NULL_metal_cond']
print(len(s), sorted(s.split_seed), s.duplicated(['model','split_seed']).sum(), s.macro_mae.mean())"
```

**Consequence.** The plotted 1.0953 matches neither documented cohort. `figures/METRIC_AUDIT.md`
§3.5 records `NULL_metal_cond` as **1.0882 at three seeds** and **1.0995 at five**, and states
that the cohort must be quoted with the number. The figure's headline arrow — "all of
chemistry is worth −0.098" — therefore compares a 3-seed mean (`REAL_best_tree`, 0.998, from
`oracles/` only) against an 8-seed mean with three duplicated seeds. It is not a paired
comparison and it is not on one cohort.

**What was done.** Nothing to the data. The figure is superseded by the refined
`figure_04_error_decomposition`, which makes the same argument on one named cohort with the
oracle cascade reproduced to 5 × 10⁻⁹ against the frozen study output. The one fact worth
carrying across — no-ligand-information against the best ligand-aware model — is stated in
the refined Figure 4 with its cohort attached (1.1149 vs 1.0324 on the 99-extractant common
cohort; 1.0995 vs 0.9695 on the 152-extractant cohort).

---

## 3. Lesser issues recorded but not acted on

* **`level_capture`** ranks feature blocks whose per-fold standard deviations
  (0.13–0.23) are several times the between-block differences being ranked, and the winning
  regressor per block is chosen by `idxmin` over ~15 recipes with no selection penalty, so
  the bar heights are winner's-curse estimates. The refined
  `figure_08_level_representations` shows the per-fold spread and states the caveat; it does
  not re-fit anything.
* **`policy_comparison`, `oracle_gap`, `fig5_adaptation`** carry no uncertainty at all on
  comparisons of the same order as their seed-to-seed spread. All three are superseded by
  figures that do.

---

*Every claim on this page was re-derived from the stored artefacts during the refinement
pass; none rests on an agent's report alone.*
