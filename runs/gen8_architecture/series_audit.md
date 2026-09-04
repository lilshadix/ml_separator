# gen8 series audit — what the model is actually supposed to represent

*Frozen gen6/gen7 cohort: **5,248 rows / 152 extractants / 79 Tanimoto chemotypes / 303 series**. Built by `scripts/gen8_series_audit.py`; tables in `runs/gen8_architecture/series/`.*

## The headline

**5,123 of 5,248 rows (97.6 %) lie on at least one reconstructable curve** — a maximal run in which exactly one experimental axis varies and every other axis is held fixed, with at least 3 points on the axis. There are **1,176 such curves** across 152 ligands. The response surface gen8 proposes to model is therefore actually present in the data; it is not an idealisation.

Two facts decide how it must be modelled:

* **56 series are grids, not curves.** A paper reporting a metal series at four acidities produces one `series_id` covering a 2-D surface. Fitting one smooth curve through it — the brief's explicit warning — would be fitting a curve through a surface. Every grid is therefore *decomposed* into its constituent one-axis curves rather than discarded: a row at the intersection of an acid titration and a metal series belongs to both curves, which is what makes cross-series transfer (§18) a measurable question at all.
* **The mass-action law is visible in the raw slopes.** Extractant-concentration curves have median slope 2.31 (IQR [1.96, 2.99]) and median linear R² 0.996 on the log–log axis; acid curves have median slope 1.57 and median R² 0.918. Those are solvation numbers and nitrate stoichiometries, recovered without being told to look for them.

## Usable rows and ligands per curve type

| curve_type | n_curves | n_rows | n_ligands | n_chemotypes | median_points | median_linear_r2 | median_slope | iqr_slope | median_y_span | pct_monotone |
|---|---|---|---|---|---|---|---|---|---|---|
| metal_series | 386 | 3,094 | 89 | 44 | 7.000 | 0.885 | 0.087 | [0.02, 0.18] | 0.923 | 43.523 |
| acid | 496 | 2,751 | 74 | 41 | 5.000 | 0.918 | 1.573 | [0.76, 2.28] | 1.732 | 63.105 |
| extractant | 241 | 1,078 | 27 | 10 | 4.000 | 0.996 | 2.305 | [1.96, 2.99] | 1.541 | 80.083 |
| temperature | 35 | 149 | 6 | 3 | 4.000 | 0.989 | -0.040 | [-0.05, -0.03] | 1.130 | 85.714 |
| metal_concentration | 10 | 85 | 4 | 1 | 7.000 | 0.766 | -1.140 | [-1.51, -0.07] | 0.887 | 30.000 |
| contact_time | 8 | 50 | 8 | 3 | 5.500 | 0.165 | 0.000 | [-0.00, 0.00] | 0.063 | 12.500 |

`median_slope` is d(log D)/d(axis), the axis being **log10 concentration** for the three concentration sweeps (the scale the mass-action law is linear in), degrees Celsius for temperature, minutes for contact time, and the lanthanide index for the metal series. `pct_monotone` is the share of curves that never reverse direction.

## Series classification

| series_type | n_series | n_rows | n_ligands | n_chemotypes |
|---|---|---|---|---|
| grid | 56 | 3,183 | 30 | 12 |
| metal_series | 104 | 1,263 | 80 | 41 |
| acid_sweep | 90 | 650 | 57 | 36 |
| extractant_sweep | 8 | 44 | 5 | 4 |
| metal_concentration_sweep | 2 | 42 | 2 | 1 |
| unusable | 14 | 30 | 8 | 7 |
| single_point | 28 | 28 | 18 | 6 |
| contact_time_sweep | 1 | 8 | 1 | 1 |

`single_point` and `unusable` together are 58 rows — the residue that carries no curve at all and against which no series-aware method can be scored.

## Provenance

* **303 of 303 series (100 %) carry a DOI** recovered from the upstream SAFE exports.
* **28 series mix more than one publication** (9.2 %). A series is defined by its categorical conditions, so two papers that ran the same setup collapse into one series id; this is recorded rather than corrected, because splitting on DOI would change the gen5/gen6/gen7 series definition and break comparability.

## Per-ligand inventory

Per ligand: median 1 curves, IQR [1, 2], max 339. **7 ligands carry no curve at all** and 37 carry curves along two or more different axes — the latter is the population on which cross-series transfer can be measured.

### The twenty largest series

| series_id | n_rows | series_type | primary_axis | varying_axes | n_metals | n_curves | log_d_span | n_doi |
|---|---|---|---|---|---|---|---|---|
| 623e74392c113263 | 376 | grid | metal_series | acid,contact_time,extractant,metal_concentration,temperature,metal_series | 14 | 67 | 6.07 | 23 |
| d60bb34e127430d4 | 195 | grid | acid | acid,contact_time,extractant,metal_concentration,temperature,metal_series | 10 | 49 | 3.78 | 5 |
| b184fe5ac72d4d44 | 194 | grid | acid | acid,extractant,metal_series | 13 | 118 | 4.32 | 1 |
| 2684502923d3a930 | 168 | grid | metal_series | acid,extractant,metal_series | 14 | 53 | 5.84 | 1 |
| b99de803167c3e1b | 131 | grid | extractant | acid,extractant,temperature,metal_series | 14 | 38 | 3.07 | 1 |
| c66b1102ddfd9909 | 113 | grid | acid | acid,metal_concentration,metal_series | 14 | 28 | 6.40 | 1 |
| e694759d2e8ab2d6 | 110 | grid | metal_series | acid,extractant,metal_series | 14 | 38 | 4.48 | 1 |
| abf18e2a44731d7c | 102 | grid | acid | acid,contact_time,metal_concentration,temperature,metal_series | 9 | 32 | 5.59 | 1 |
| bab2d2e736b8af1b | 91 | grid | acid | acid,extractant,temperature,metal_series | 8 | 34 | 3.08 | 1 |
| d58ebecc3229e99a | 90 | grid | acid | acid,extractant,metal_series | 6 | 29 | 5.55 | 2 |
| 236171f150e9fbde | 88 | grid | acid | acid,metal_concentration,metal_series | 10 | 21 | 5.60 | 1 |
| d119db4aff89e490 | 88 | grid | acid | acid,metal_concentration,metal_series | 10 | 16 | 5.89 | 1 |
| 8f48afa3d98adae8 | 87 | grid | metal_series | acid,extractant,temperature,metal_series | 6 | 33 | 3.59 | 1 |
| fc9d81934a7468f5 | 74 | grid | metal_series | acid,extractant,metal_concentration,metal_series | 14 | 24 | 2.88 | 1 |
| 6ea59c50e60297e4 | 70 | grid | acid | acid,metal_series | 14 | 19 | 1.56 | 1 |
| 5eed069691fad55f | 70 | grid | acid | acid,metal_series | 14 | 19 | 3.70 | 1 |
| 54245e3239ad5d14 | 70 | grid | acid | acid,metal_series | 14 | 19 | 5.56 | 1 |
| 8ac9d6b3b370d969 | 69 | grid | acid | acid,contact_time,extractant,metal_concentration,metal_series | 2 | 10 | 5.95 | 7 |
| fcdde22e4670f96c | 61 | grid | acid | acid,extractant,metal_series | 8 | 18 | 2.52 | 1 |
| f3e4b4269d469ea0 | 60 | grid | metal_series | acid,contact_time,extractant,metal_concentration,metal_series | 14 | 22 | 4.10 | 3 |

## What this licenses, and what it does not

* A **functional** model (one that predicts a curve rather than a row) has 1,176 training curves — enough to learn a response surface, not enough to learn one per chemotype.
* A **smoothness or monotonicity** penalty is defensible on the extractant and acid axes (median linear R² 0.996 and 0.918) and is **not** defensible globally: contact-time curves have median linear R² 0.165, i.e. they are flat noise once equilibrium is reached, and metal series are non-monotone by construction wherever the tetrad effect bites.
* **k-shot calibration has a well-defined candidate pool**: for a held-out ligand the candidate experiments are its curve points, and the stratification the brief asks for (§17 — low/middle/high end of a sweep) is computable from `axis_value` alone, without looking at any target.
