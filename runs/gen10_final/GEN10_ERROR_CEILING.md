# GEN10 error ceiling — where the remaining error lives, and how much is reachable

*Every number here is computed on `GEN9_SHAPE_RECOMPOSED`'s stored out-of-fold
predictions (five seeds, FROZEN cohort, 5,248 rows / 152 extractants) and on the
stored k-shot detail of the same model. Nothing was refitted for this document.
Sources: `error_decomposition/`, `data_ceiling/`, `budget_simulation/`. The
finalist chosen in Phase 11 shares this model's level exactly (its offset MAE is
bit-identical, 0.8212), so the level components below carry over unchanged; its
shape component is smaller by the amount Phase 4 recovered.*

## 1. The seven components

Macro MAE is gen6's one-ECFP-cluster-one-vote metric. "Oracle-removable" is what a
perfect version of the named intervention would remove; "realistically removable"
is what a *deployable* arm in this study actually removed, read off the stored
k-shot detail (143-ligand protocol, central-then-spread). Components overlap, so
the column does not sum exactly to the total; the residual is a floor.

| component | current contribution | oracle-removable | realistically removable | required intervention |
|---|---|---|---|---|
| 1 predictable level (per-ligand constant) | **0.520** | 0.520 | **0.349** (zero-shot 0.980 → `OFFSET_K1` at k = 1: 0.631) | one measurement per ligand, central point |
| 3 series-local level (beyond the ligand constant) | 0.057 | 0.057 | 0.140¹ (`OFFSET_K1` k = 1 → `OFFSET_K3` k = 3) | measure *inside* the series you will predict; `SERIES_ML` at k ≥ 2 |
| 2 curve shape (within-curve, after the curve's level) | **0.192** | 0.192 | 0.008 (slope repair at k = 1) | the recomposition is already in the model; the rest needs measurements |
| 4 system identity (name/structure mismatch) | −0.002 | 0 | 0 | data work (record the second species) |
| 5 sparse support (far tercile of Tanimoto distance) | 0.060 | 0.060 | 0 | measure new chemotypes first (Phase 9) |
| 6 suspected data quality (duplicates, TWE-24, flagged) | 0.037 | 0.037 | 0 | primary-source check |
| 7 residual unexplained | **0.103** | — | — | none identified |
| **total (macro MAE)** | **0.970** | | | |

¹ `OFFSET_K3` also fits response slopes, so its k = 3 gain exceeds the pure
series-level oracle; the component and the realised number measure overlapping
things and are both reported rather than reconciled.

Stages of the oracle chain (`error_decomposition/summary.json`):

| after removing… | pooled MAE | macro MAE |
|---|---|---|
| nothing (current) | 1.122 | 0.970 |
| a per-ligand oracle constant | 0.462 | 0.450 |
| a per-series oracle constant | 0.401 | 0.393 |
| a per-curve oracle level | — | 0.385 |
| a per-curve oracle level **and** slope | 0.189 | 0.194 |

**Reading.** Just over half of the zero-shot error is the level of an unseen
ligand, and one measurement removes two thirds of it. Another quarter is
within-curve shape plus series-local level, and three measurements inside the
series remove most of that (Phase 5: `SERIES_ML` at k = 3 reaches 0.461 on the
143-ligand protocol). What no number of measurements on *other* ligands removes
is about 0.10 of unexplained scatter plus ~0.10 of data and support problems.

## 2. Is another zero-shot model generation justified?

A zero-shot architecture can only touch components 2 and 7 — the level is not
predictable from structure (gen7) and the data components are not modelling. The
shape component is 0.19 macro. gen9 recovered shape MAE 0.665 → 0.469 on the
extractant axis; Phase 4 of gen10 takes it to 0.383–0.406 with macro neutral; the
oracle floor (per-curve level and slope known) is 0.194 macro overall. The
headroom a new zero-shot generation could target is therefore on the order of
0.05–0.10 macro, and the evidence of gen7–gen10 is that each step on it costs
a full generation for 0.01–0.02. Against that, **one measurement on a ligand is
worth 0.35–0.38, and one measurement on a far-chemotype ligand 0.49** (Phase 9).
The realistic lever is measurement allocation, not architecture. **No further
model generation is justified on this cohort.**

## 3. Data-schema ceiling (Phase 7)

Mutually exclusive row cohorts, assigned in the priority TWE-24 > duplicate >
mismatch > uncertain > consistent (`data_ceiling/row_cohorts.csv`):

| cohort | rows | ligands | ligand macro MAE (RECOMPOSED) | frozen |
|---|---|---|---|---|
| A consistent | 4,691 | 145 | **0.947** | 0.958 |
| B name/structure mismatch | 271 | 15 | 1.000 | 1.030 |
| C exact-decade duplicates (incl. DMDPhPDA) | 218 | 3 | 1.302 | 1.304 |
| D TWE-24 | 6 | 1 | 4.098 | 4.098 |
| E flagged level outliers | 62 | 3 | 1.405 | 1.503 |

(The 353 `rec__name_mismatch` rows split as 271 in B and 82 absorbed by the
higher-priority C and D cohorts; ligand counts are ligands with at least one row
in the cohort, so they overlap across rows of one ligand.)

* **Mismatch rows carry a systematically larger irreducible residual** — the
  error left after a per-ligand, per-seed offset: median 0.934 vs 0.582 on
  consistent rows (ligand-level Mann–Whitney p = 0.0018; frozen model 0.986 vs
  0.632, p = 0.0009). The missing second species changes the *shape*, not only
  the level. `data_ceiling/irreducible_residual_test.csv`.
* Against an honest no-chemistry baseline — the mean `log D` of the same
  condition cell among ligands of *other* folds — the model wins on consistent
  rows in 5/5 seeds (median residual 0.536 vs 0.639, p < 1e-7), is a statistical
  tie on mismatch rows (0.918 vs 0.748 by median, model better by mean, p ≈ 0.3),
  and **loses badly on duplicate and outlier rows** (1.17 vs 0.33; 1.60 vs 0.60,
  0/5 seeds): a same-cell peer from another publication predicts those rows three
  times better than the model. That is the signature of a data-identity artefact,
  not of chemistry. `data_ceiling/neighbour_vs_model_residuals.csv`.
* **Headline attribution.** If every non-consistent row errored like a consistent
  one, pooled MAE would fall by 0.015 of 1.122 (1.3 %). At the ligand-macro
  level, the mismatch cohort contributes −0.002 (its ligands are not harder on
  average) and the duplicate/TWE-24/outlier cohorts 0.037. The mismatch problem
  is real and shape-relevant on the 353 rows it touches, and small in the
  headline; the data-quality cohorts are the larger data lever.
* The audit table `data_ceiling/mismatch_audit.csv` lists, per represented
  structure, the recorded names on its mismatch rows, every name the corpus
  attaches to that structure, the recorded aqueous complexant where one exists,
  the publications, the row counts and the current error. Nothing was corrected
  and nothing was deleted.

## 4. Breadth versus depth (Phase 9)

Using each of the 99 common-cohort ligands' held-out adaptation curve
(`budget_simulation/adaptation_curves.csv`), the expected macro-MAE reduction
from an additional measurement on an unseen ligand is:

| measurement | mean gain | near tercile | mid | far |
|---|---|---|---|---|
| first point | **0.382** | 0.235 | 0.448 | **0.495** |
| second point | 0.078 | | | |
| third point | 0.063 | | | |
| fourth / fifth, each | 0.021 | | | |

Under a fixed budget *B* over 99 unseen ligands (macro MAE, zero-shot 1.036):

| B | new-chemotype-first | Tanimoto max-min diversity | one point, random ligands | two points each | three each | five each | oracle ordering |
|---|---|---|---|---|---|---|---|
| 9 | **0.979** | 1.008 | 0.999 | 1.017 | 1.020 | 1.030 | 0.872 |
| 49 | **0.782** | 0.836 | 0.847 | 0.924 | 0.953 | 0.986 | 0.680 |
| 99 | 0.654 | 0.654 | 0.654 | 0.808 | 0.863 | 0.930 | 0.654 |
| 198 | 0.576 (central-then-spread) | | | | 0.691 | 0.816 | |
| 495 | 0.471 (central-then-spread) | | | | 0.513 | 0.471 | |

Breadth dominates depth until every ligand has one point; ordering ligands by
distance to the training chemistry (new chemotype first) beats both random
breadth (−0.065 at B = 49) and structural diversity. Gain per newly covered
chemotype is in `budget_simulation/budget_strategies.csv`. This operationalises
gen6's breadth-over-depth finding with a deployable rule.
