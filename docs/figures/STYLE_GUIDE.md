# Style guide

Everything below is implemented in `figures/scripts/_style.py`; a figure script never sets
a colour or a font itself, it calls `_style.m("<method key>")`. Changing the palette in one
place changes every figure.

## 1. Method identity — one method, one look, in every figure

| key | label used in artwork | colour | line | marker | what it is |
|---|---|---|---|---|---|
| `measured` | Measured | `#000000` black | solid | ● | ground truth |
| `no_model` | Measurements only (no model) | `#8C8C8C` grey | dotted | ▼ | the k support points fitted with no model at all |
| `no_ligand` | No ligand information | `#8C8C8C` grey | dashed | ✕ | metal + conditions only (`NULL_metal_cond`) |
| `frozen` | Baseline model | `#E69F00` orange | dashed | ■ | `REC_ecfp_plus_recovered`, the pre-existing champion |
| `relpos` | + relative position | `#56B4E9` sky | dash-dot | ▲ | `GEN9_REL_MONOLITH` |
| `recomposed` | + recomposition (global model) | `#0072B2` blue | solid | ◆ | `GEN9_SHAPE_RECOMPOSED`, the frozen global model |
| `pipeline` | Final pipeline (+ series adaptation) | `#D55E00` vermillion | solid | ● | frozen model + `SLOPE_L_s1_K1` / `SERIES_ML` + central-then-spread |
| `oracle` | Oracle (not deployable) | `#009E73` green | dashed | ★ | any arm that reads a held-out target |
| `base91` | Restricted training coverage | `#CC79A7` purple | dashed | ■ | gen6 `BASE` (extractants with ≥ 10 condition cells) |
| `expanded152` | Full training coverage | `#0072B2` blue | solid | ◆ | gen6 `EXPANDED` (≥ 3 cells) — the corpus every other figure uses |
| `rowmatched` | Row-count-matched control | `#56B4E9` sky | dash-dot | ▲ | gen6 `EXPANDED_ROWMATCHED` |
| `shuffled` | Shuffled-target control | `#8C8C8C` grey | dotted | ✕ | gen6 `EXPANDED_SHUFFLED` |

The palette is Okabe–Ito, which is distinguishable under deuteranopia, protanopia and
tritanopia. No rainbow scales, no sequential colour used as a categorical key, no colour
carrying information that is not also carried by line style or position.

**Oracle rule.** Any arm that reads a held-out target is drawn in `#009E73`, hatched where
it is a bar (`//`, white hatch lines), and the words "not deployable" appear in the artwork,
not only in the caption. This applies to Figures 2A, 4A and 6A.

**Distance terciles** (Figure 5A only) use sky / blue / vermillion for near / mid / far,
which is an ordered use of three palette entries and is stated in that figure's legend.

## 2. Typography

* Family: Helvetica → Arial → DejaVu Sans (first available).
* Base 7.0 pt; axis labels 7.5 pt; tick labels 6.8 pt; legends 6.0–6.5 pt;
  in-panel annotations 5.6–6.3 pt; panel letters 8.5 pt bold.
* No figure titles. Where a panel needs a strapline it is set 6.6–7.0 pt in `#8C8C8C`,
  left-aligned, and is descriptive ("distance to training chemistry"), never a claim.
* Panel letters are bold capitals in axes coordinates via `_style.panel(ax, "A")`, so they
  sit in the same place relative to each panel in every figure.
* Units are always in the axis label: `macro MAE (log₁₀ D units)`, never a bare "MAE".

## 3. Geometry

* Single column `W1 = 3.39 in` (86 mm); 1.5 column `W15 = 5.00 in`; double column
  `W2 = 7.01 in` (178 mm). Every main figure is `W2`; Figure 3 and Figure 1 are the two
  tallest at 4.55 and 4.75 in.
* Top and right spines removed; ticks outward; no gridlines anywhere.
* Bars start at zero **or** the plot uses points with intervals instead of bars. Figure S4
  is a dot-and-interval plot for exactly this reason: its axis is truncated to 0.945–1.105
  and truncated bars would exaggerate the differences.
* Aspect ratio is fixed to 1 on any panel with a `y = x` reference line (Figures 2D, S8A).

## 4. Uncertainty

* **Bands and whiskers around a mean**: chemotype-block bootstrap, 5,000 replicates,
  resampling the 41 (or 79) Tanimoto-0.7 chemotypes with replacement and taking every
  member of a drawn block. Implemented in `_stats.block_bootstrap_mean` /
  `block_bootstrap_stat`, which share their block construction, replicate count and RNG
  seed with the repository's `gen8.inference.paired_chemotype_bootstrap`.
* **Paired differences**: `gen8.inference.paired_chemotype_bootstrap`, imported, not
  reimplemented. BCa intervals are drawn; the percentile interval is stored alongside in
  the derived JSON.
* **Seed spread**: thin vertical ticks showing min–max over the five split seeds
  (Figure 2A). This is labelled as reproducibility, never as a confidence interval.
* **Violins** (Figure 3E, 3F) show the full per-curve distribution; the tick is the point
  statistic named in the axis label and the bar beneath it is that statistic's block
  bootstrap interval. One statistic per panel, named on the axis.
* No significance stars anywhere. Effect size, interval, number of units improved and
  number of seeds positive are printed instead.

## 5. Export

* Vector `.pdf` (fonts embedded as TrueType, `pdf.fonttype = 42`) **and** `.png` at
  **600 dpi**, from the same figure object, so the two never diverge.
* `bbox_inches="tight"`, `pad_inches=0.02`.
* No transparency in the saved background; no drop shadows, bevels, 3-D axes or textures
  except the oracle hatch.

## 6. Naming

* Main text: `figures/main/Fig<N>_<slug>.{pdf,png}`.
* Supplement: `figures/supplementary/FigS<N>_<slug>.{pdf,png}`.
* Script: `figures/scripts/plot_fig<N>_<slug>.py` or `plot_supp_s<N>_<slug>.py`; each runs
  standalone from the repository root and regenerates exactly one figure.
* Shared preparation: `prepare_*.py` writes to `figures/derived/`; `verify_metrics.py`
  writes `figures/derived/metric_audit.csv`.
* Every plotting script writes a `figures/derived/fig<N>_values.json` containing the
  plotted numbers, the cohort, the sample sizes and the source paths, so a reviewer can
  read the figure's contents without rerunning it.
* Model names in artwork are descriptive ("+ shape recomposition"), never repository
  identifiers — except in Figure S3 and S4, whose subject *is* the arm registry.
