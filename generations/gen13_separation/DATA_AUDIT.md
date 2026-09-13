# Gen13 — data audit for the separation target

*Written 2026-09-07, before the pre-registration and before any locked Gen13 model was
fitted.  Numbers come from `scripts/audit/gen13_sf_data_audit_part{1,2}.py` (untracked
audit scripts run on 2026-09-07), from `gen13sep.cohort.build_cohort` (the cohort manifest
under `manifests/`), and from the repository's own reports.  Exploratory prototypes run
before this document are listed in §7 and disclosed in the pre-registration.*

## 1. Why the separation target, and why now

Six generations (gen5–gen12.2) established that the per-extractant **level** of `log D`
is the bottleneck of zero-shot prediction (level 0.738 of the 1.048 macro MAE in Gen12.2;
molecular structure worth ~0.09–0.13; no model beats a 1-NN lookup convincingly) and that
the **within-extractant shape** is the part chemistry can predict.  The user's question is
*separation*: `log SF(A/B) = log D(A) − log D(B)` for two lanthanides under one extractant
and one set of conditions.  In that quantity the level cancels **exactly**.  Gen2–gen4
studied pairs but on a cohort that required all 64 condition columns to be *present*,
which collapsed 190 extractants to 34 (TODGA alone 43.6 % of pairs), and the programme
then pivoted to `log D`.  Gen13 returns to the pair quantity with three changes: a
condition key that treats "not reported" as a value, publication provenance in the key,
and a target that is the whole lanthanide-axis curve of a cell rather than one pair at a
time.

External confirmation that the contrast is the right target: Sumiyoshi, Kuroki & Mori
(Inorg. Chem. 2026, 65, 7676) reach MAE 2.6 kJ/mol on An/Ln selectivity *differences*
against 17.5 kJ/mol on absolute binding free energies with the same descriptors.

## 2. The cohort (`manifests/cohort_exact.json`, fingerprint `4c3c6628ea0be949`)

*Fingerprint updated 2026-09-07 after the adversarial code review: float condition values are
now rounded to 9 significant digits inside the key (no cell changed; the first build, fingerprint
`179c8de1fd4715af`, used raw `str()`), and the manifest carries the per-(cell, metal) replicate
statistic.  No locked result had been read.*

| quantity | value |
|---|---|
| bundle rows | 5,992 → 5,860 after TODGA-name quarantine (129) and `log D ≤ −6` sentinels (3) |
| condition key | canonical SMILES + `publication_id` (gen6 provenance) + all 64 `cond__` columns, NaN == NaN |
| cells with ≥ 2 metals | **521** |
| extractants | **90** (of 190 in the bundle; 100 have a single metal and cannot form a pair) |
| frozen chemotypes (Tanimoto-0.7 single linkage) | **45**; ECFP clusters 77; publications 58 |
| rows inside cells | 3,871 |
| metals per cell | 2: 138 · 3: 62 · 4: 32 · 5: 21 · 6: 72 · 7: 8 · 8: 57 · 9: 11 · 10: 4 · 11: 2 · 12: 1 · 13: 35 · **14: 78** |
| largest chemotype | 375 of 521 cells, 23 of 90 extractants (the diglycolamides) |
| pairwise `log SF` observations per split seed | 14,150 (all pairs) · 2,348 adjacent (`dZ = 1` plus Nd–Sm) · 6,698 far (`dZ ≥ 5`) — corrected 2026-09-07 from the frozen manifest (the first draft quoted the pre-quarantine prototype counts 14,173 / 4,137 / 6,711) |

**Effective sample size.** Kish `n_eff` over chemotypes by extractant count is small (the
diglycolamide chemotype holds 23 of 90 extractants); the analysis script prints the exact
value beside every leaderboard.  Every interval in Gen13 resamples chemotypes.

**Publication mixing.** With the exact 64-column key 7 of 521 multi-metal cells (59 rows)
spanned two publications; adding `publication_id` to the key splits them.  Only 8 exact
cells in the whole bundle cross a publication, so exact cells are almost always one table
of one paper.  `docs/audits/LEAKAGE_AUDIT.md:307` ("no publication_id exists") is stale:
`runs/gen6_provenance/provenance_table.parquet` joins 5,992 / 5,992 rows.

**Sensitivity key.** Dropping `cond__contact_time_min` and `cond__metal_concentration_mM`
from the key (`key_mode=relaxed`) gives 509 cells / 90 extractants with 113 all-14 cells
(against 78) and 4,026 rows; it is run as a sensitivity, never as the headline.

## 3. What the target looks like

* `log D` variance is 89 % between (extractant × condition) and **11 % along the Ln axis**
  (mean within-cell variance 0.313 against total 2.776).  Gen13 models only the 11 %.
* Adjacent pairs (`dZ = 1`, n = 2,115): mean +0.089 (heavier preferred), **sd 0.231**,
  mean |·| 0.167; 68 % heavy-preferred.  Per pair type the sign consistency ranges from
  50 % (Er–Tm, Yb–Lu, Eu–Gd) to 88 % (Gd–Tb).
* Across the series the contrast is large: La–Lu and similar far pairs have |log SF|
  approaching 1 (sd of all-pair `log SF` 0.32 for adjacent-in-cell, far larger for `dZ ≥ 5`).
* **Low rank.**  On the 290 cells with ≥ 5 metals a quadratic in Z fits with median R²
  0.960 (81.7 % of cells ≥ 0.8); 81 % of cells have a positive slope and 69 % are concave.
  An in-sample iterative-SVD completion of the 521 × 14 centred matrix explains 84 % of the
  centred variance at rank 1 and 94 % at rank 2.  The corpus mean centred curve rises
  monotonically from La (−0.61) to Lu (+0.39) with a dip at Gd (the Gd break).
* **Replicate noise floor.**  313 (cell, metal) combinations carry replicate rows (1,003
  rows, TODGA 261): within-replicate sd of `log D` has median **0.237** and mean 0.419, so a
  single adjacent `log SF` carries noise of order √2·0.237 ≈ 0.34 — larger than its signal
  sd.  On the frozen cohort itself (after quarantine and the publication split) the same
  statistic is median **0.302** over 216 replicated (cell, metal) combinations
  (`manifests/cohort_exact.json: replicate_sd_median_per_cell_metal`); a replicate-split
  probe on the 9 extractants with replicated pairs gives a half-versus-half pairwise MAE of
  0.60, i.e. rows recorded under identical conditions often disagree by more than any model
  error — the `key_mode=series` sensitivity keeps such rows apart.  Gen13 therefore (a) averages replicates before forming any target, (b) reports the
  floor beside every metric, (c) treats the adjacent-only metric as secondary and the
  all-pair and far-pair metrics as primary and secondary, and (d) quotes sign accuracy only
  on pairs with |log SF| ≥ 0.3.

## 4. Trivial baselines every arm must beat (in-sample, adjacent `dZ = 1`)

| baseline | MAE | note |
|---|---|---|
| predict 0 | 0.167 | |
| per pair-type mean | 0.141 | R² 0.113 |
| per-extractant mean | 0.138 | R² 0.163 |
| additive extractant + pair-type | — | R² 0.275 |
| (extractant × pair-type) mean | — | R² 0.680 with 764 groups: ≥ 32 % of adjacent-SF variance is condition/replicate residual even knowing extractant and pair |
| "heavier always preferred" sign rule | 0.681 accuracy | the yardstick for any sign-accuracy claim |

## 5. Feature blocks (all target-free; `gen13sep.features`)

| block | columns | missing | source |
|---|---|---|---|
| COND | 64 | 0.9 % | bundle `cond__*` |
| MASSACT | 8 | 7.3 % | gen5/gen12 mass-action law: log10 of positive continuous conditions, `log[L]·DENTATE`, `log[L]·coreCN`, `log[L]·log[H+]` (recipe columns as per-cell mode) |
| PHYSCHEM | 10 | 0 | bundle RDKit scalars |
| DONORS | 13 | 0 | frozen gen6 donor census (`chem__donor__*`, dentate, core CN, n ligands, n fill) |
| ECFP | 2,048 | 0 (1,623 constant bits) | bundle Morgan fingerprint |
| LIG2D | 206 | 0.7 % | bundle extended RDKit descriptors; `Ipc` log10-transformed (spans 1e8–1e29) |
| COORD | 114 | 3.5 % (2 extractants uncovered) | Gen12.2 frozen coordination-topology block v1.1.0 |

Excluded on purpose: every `feat3d__` geometry/xTB column (five generations found no
transferable 3D signal; the xTB energies were probed in §7 and carry none), the pretrained
ChemBERTa/MoLFormer embeddings (file absent; gen7 verdict negative), and every identity or
provenance column (asserted by `FORBIDDEN_FEATURE_TOKENS`).

## 6. Lanthanide-side constants (`gen13sep.metals`)

Shannon crystal radii (CN 8 and CN 9), Kepp-2019 hydration free energies, the 4f count, a
Gd-break step and Jørgensen-type tetrad functions, each standardised and centred over the 14
lanthanides.  They enter only as candidate *basis curves*; their coefficients are what the
ligand model predicts.

## 7. Exploratory work done before this audit (disclosed, not selected on)

Three scratch probes were run on 2026-09-07 on the same data before Gen13 code existed:

1. **xTB metal-exchange energy probe.**  Per pair-type-centred differences of the bundle's
   GFN2-xTB complex energies between two metals of the same cell, restricted to identical
   ligand composition, correlate with `log SF` at Spearman **0.035** (7,413 pairs, 87
   extractants; extractant-level 0.027).  The ionic-radius difference alone correlates at
   0.457.  Verdict: the single-conformer f-in-core energies carry no selectivity signal
   across ligands; they are excluded.
2. **Low-rank probe** (rank 1/2/3/4 in-sample explained variance 0.842/0.944/0.967/0.979)
   and the corpus mean curve, quoted in §3.
3. **Prototype ladder** on the same 521 cells with leave-chemotype-out folds built by the
   same round-robin dealer (3 then 5 seeds): a rank-2 data basis whose coefficients are
   predicted by extremely randomised trees from ECFP + RDKit + conditions scored all-pair
   macro MAE ≈ 0.50 against ≈ 0.64 for the corpus mean curve, with chemotype-bootstrap
   intervals excluding zero in every seed.  The pre-registration fixes the candidate set
   and the selection rule so that this observation does not become the headline by
   construction; the locked run uses the frozen fold plan written by `gen13sep.splits`.
