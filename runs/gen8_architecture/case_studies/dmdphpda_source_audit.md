# DMDPhPDA duplicate-exponent forensic audit

**gen8 brief, section 14.**  Audit date 2026-08-19; independently re-verified 2026-08-20 (see §8).  Sources: `dataset with 3D structures/dataset.parquet`,
`runs/gen7_architecture/cache/cohort.parquet`, `runs/gen7_architecture/cache/recovered_cells.parquet`,
and the upstream builder `~/PycharmProjects/lanthanide_dataset_builder/raw_data/*_SAFE.csv`.
No dataset file was modified.

Machine-readable companions in this directory:
`dmdphpda_rows.csv` (140 rows, one per cohort row) and `dmdphpda_cohorts.json`.

---

## 1. Summary

One publication — Shimada, Yaita, Narita, Tachimori & Okuno, *Solvent Extraction and Ion Exchange*
**22**(2), 147–161 (2004), DOI [10.1081/SEI-120030392](https://doi.org/10.1081/SEI-120030392) — was
entered into the corpus **twice**, under two different extractant names and two different spellings of
the same solvent.  Both copies survive into the frozen gen7 evaluation cohort as **140 rows, 2.67 % of
the 5 248-row cohort**, under a **single** `extractant` SMILES and a single Tanimoto chemotype.

The two copies disagree by **exactly 0, 1, 2, 3 or 4 whole decades**, and the shift is a **constant per
HNO₃ column**, not per row.  The ORNL-import copy is the corrupted one: it is the printed Table 1
mantissa with the column's power-of-ten multiplier discarded.

| | copy A | copy B |
|---|---|---|
| `extractant_name` | `DMDPhPDA` | `2-N-6-N-dimethyl-2-N-6-N-diphenylpyridine-2-6-dicarboxamide` |
| solvent string | `Chloroform` → `cond__diluent__chloroform` | `CH3Cl` → `cond__diluent__ch3cl` |
| rows in cohort | 70 | 70 |
| `condition_id`s (one per HNO₃ level, 14 rows each) | `f5a6b6e93ce4ac42` (1 M), `d5166599e4db7c09` (2 M), `c56a1d2545057312` (3 M), `8ef8ca0482833c77` (4 M), `12961d22b7540dd5` (5 M) | `3b5aeefb445a5152` (1 M), `54e8e5554dfeca2e` (2 M), `a47d8219c2351b58` (3 M), `f3eaafebe0ce97f5` (4 M), `1cb6dfc8e5dccd56` (5 M) |
| upstream file(s) | `Ce/Dy/Er/Eu/Gd/Ho/La/Lu/Nd/Pd_SAFE.csv` | `Am_SAFE.csv` (all 70) |
| `comments_description` | full record: `Data Location: Table 1`, title, authors, year 2003 | `updated ORNL` |
| `entry_author` / date | Thomas Summers, 2026-01-20 | Baosen Zhang, 2026-02-10 |
| temperature / contact time / phase ratio | 25 °C / 30 min / 1.0 | absent / `- -` / absent |
| log D range | −4.00 … +1.56 | 0.00 … +1.56 |
| **verdict** | **FROZEN (keep)** | **QUARANTINED (exclude)** |

Everything else about the two copies is identical: same canonical SMILES
`CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1`, same 14 lanthanides (La…Lu, no Pm), same
HNO₃ = 1/2/3/4/5 M, same 0.5 M extractant, same 0.01 mM metal, same DOI.

---

## 2. Exact quantification

* **140 cohort rows** carry this SMILES; **70 / 70** split between the two names.
* **70 duplicated (condition, metal) cells** — a perfect 14 metals × 5 acidities bijection.
  Index overlap between the copies is 70/70; nothing is unmatched.
* Every cohort row has `n_replicates = 1`, so no averaging hid the duplication.  The copies stay
  separate because **four** `cond__` columns differ between paired rows — `cond__diluent__ch3cl`,
  `cond__diluent__chloroform`, `cond__temperature_C` (25 vs NaN) and `cond__contact_time_min`
  (30 vs NaN) — and `condition_id` is a hash over the whole `cond__*` vector
  (`levels.condition_labels`), so any one of the four is enough to keep the `row_id`s apart.
  Ten numeric feature columns differ in total (those four, plus `massact__log10_*` of the two
  numeric ones and four `geom_cond__*_bin` indicators).
* **DOI**: one, `https://doi.org/10.1081/SEI-120030392`, on both copies.
  (Both DOI strings additionally carry a spurious appended token
  `, https://doi.org/10.1021/jacs.5c19738`, which is present on 31 132 of the 48 138 upstream rows and
  is unrelated to this ligand — corpus-wide provenance noise, flagged for a separate ticket.)
* **Data locations**: copy A = `Table 1`; copy B = none (comment is the literal string `updated ORNL`).
* **Per-cell difference** `log_D(B) − log_D(A)`:

  | HNO₃ | cells | decade shift |
  |---|---|---|
  | 1 M | 14 | **+4** |
  | 2 M | 14 | **+3** |
  | 3 M | 14 | **+2** |
  | 4 M | 14 | **+1** |
  | 5 M | 11 | **0** |
  | 5 M | 3 (La, Ce, Pr) | **+1** |

* **The differences really are integer decades.**  Max |difference − nearest integer| over all
  70 cells = **4.9 × 10⁻¹⁰** (IEEE-754 noise from `log10`).  The ratio `D_B / D_A` reproduces
  10⁴, 10³, 10², 10¹, 10⁰ to ten significant figures.  Because the ratio is exact and
  column-constant, **copy B is not an independent measurement of the same system** — it is a second
  transcription of the same printed table.

* Upstream, both copies are mirrored across the per-metal raw CSVs.  `DMDPhPDA` has 160 raw rows
  (70 in `Pd_SAFE.csv`, 5 in each of the 14 per-metal files = 70, 10 in `U_SAFE.csv`, 10 in
  `Y_SAFE.csv`); the ORNL name has 430 (70-row blocks in `Am/Ca/In/Th/Y_SAFE.csv`, 5 in each
  per-metal file, 10 in `U_SAFE.csv`).  The `_SAFE.csv` file name is therefore **not** a metal
  attribution for either copy.  The bundle de-duplicated each to 70 unique cells, so the mirroring
  itself needs no action here.

---

## 3. Which copy is corrupted — internal evidence

### E1. Acid dependence (decisive)

Fitting `log D` against `log₁₀[HNO₃]` for each of the 14 metals:

| | copy A | copy B |
|---|---|---|
| metals with strictly monotone log D vs [HNO₃] | **14 / 14** | **0 / 14** |
| mean Pearson r | **0.970** | 0.335 (negative for La, Nd, Sm, Eu) |
| mean slope d log D / d log[HNO₃] | **5.88** | 0.50 |
| mean log D span over 1→5 M | **4.17** | 0.67 |
| fraction of 1-M steps that go *down* | **0 %** | **43 %** |

DMDPhPDA is a neutral, solvating N,O,O-tridentate extractant; it extracts Ln(NO₃)₃·nL, so D must rise
steeply and monotonically with nitrate.  Copy A does exactly that.  Copy B is flat (0.67 log units over
a five-fold acid range), non-monotone for every single metal, and *negatively* acid-dependent for four
lanthanides while positive for the other ten — which cannot happen when all fourteen share one
mechanism.  The published abstract's own conclusion (see §4) is that nitric acid concentration is the
*dominant* influence on D; copy B contradicts the paper it came from.

### E2. The curator's own reconciliation note (decisive)

Six upstream rows of copy A carry the comment:

> `Value in Table 1 off from value shown in Figure 5 by factor of 10^-1. Value adjusted to match Figure 5`

They de-duplicate to exactly **three cells**: La, Ce and Pr at 5 M HNO₃
(`La_SAFE:12888`, `Ce_SAFE:12893`, `Pd_SAFE:12898`).  Those are precisely the three cells where copy B
sits one decade above copy A while the other eleven 5-M cells agree exactly.

So the record states, in the curator's own words, that copy B's 5-M values (3.6, 6.3, 9.7) are the
values **as printed in Table 1**, and that copy A divided them by ten to match Figure 5.  Copy B is
therefore a raw table transcription; copy A is a reconciled one.

### E3. The corruption has a column-multiplier signature

Copy A's D values, laid out as the paper's table, show what happened:

| metal | 1 M | 2 M | 3 M | 4 M | 5 M |
|---|---|---|---|---|---|
| La | 0.0003 | 0.0036 | 0.020 | 0.10 | 0.36 |
| Ce | 0.0002 | 0.0034 | 0.024 | 0.16 | 0.63 |
| … | … | … | … | … | … |
| Yb | 0.0003 | 0.0020 | 0.040 | 1.70 | 31.0 |
| Lu | 0.0004 | 0.0022 | 0.044 | 2.00 | 36.0 |

Every 1-M value is (1.0–4.0)×10⁻⁴, every 2-M value is (1.8–4.2)×10⁻³, every 3-M value is
(1.7–4.4)×10⁻², every 4-M value is (1.0–20.0)×10⁻¹.  That is a table printed with a
**per-column multiplier** (`D × 10⁴`, `× 10³`, `× 10²`, `× 10`), and copy B contains the bare printed
mantissas — `3.0, 3.6, 2.0, 1.0` for La and so on.  The 5-M column spans two decades and evidently
carries per-cell exponents instead of a column header, which is why only three of its cells shift.

A corruption whose magnitude is a function of the *table column* and not of the measurement is a
transcription artefact by construction.  Nothing physical shifts fourteen different metals by the
identical factor of exactly 10⁴ at one acidity and exactly 10³ at the next.

### E4. Lanthanide-series smoothness confirms the Figure-5 correction too

The 4 M → 5 M increment of log D, plotted along the series, is the sharpest available test of whether
La/Ce/Pr at 5 M should be 0.36/0.63/0.97 (copy A) or 3.6/6.3/9.7 (copy B / Table 1 as printed):

| | Spearman(Z, Δlog D₄→₅) | max jump between adjacent Ln | sign reversals |
|---|---|---|---|
| copy A (Figure-5 correction applied) | **+0.992** | **0.151** | 1 (Yb→Lu, −0.006) |
| counterfactual (Table 1 as printed) | −0.024 | **0.964** | 3 |

(`scipy.stats.spearmanr`, n = 14, one tie at Δ = 1.255273 for Tm and Lu: ρ = +0.99230 for copy A,
−0.02420 for the counterfactual.  Copy B's own Δ₄→₅ gives the same two statistics, because it differs
from the counterfactual by a constant −1.)

Copy A gives a smooth monotone rise from 0.556 (La) to 1.261 (Yb) / 1.255 (Lu).  The counterfactual puts
La/Ce/Pr at 1.556/1.595/1.625 and then drops to 0.661 at Nd — a 0.96-log-unit cliff between two
adjacent lanthanides, six times larger than any genuine step in the series.  The curator's Figure-5
adjustment is independently corroborated.

### E5. Corpus context — chemical plausibility of the magnitudes

Comparators inside the same frozen cohort:

| extractant | diluent | HNO₃ | [L] | log D range |
|---|---|---|---|---|
| N,N′-dimethyl-N,N′-diphenyl**propanediamide** (the malonamide analogue of this exact ligand) | chloroform | 4 M | 0.2 M | −2.70 … −1.19 |
| **DMDPhPDA, copy A** | chloroform | 4 M | 0.5 M | −1.00 … +0.30 |
| **DMDPhPDA, copy B** | CH₃Cl | 4 M | 0.5 M | 0.00 … +1.30 |
| three diethyl-bis(methylphenyl)-PDA congeners | phenyl trifluoromethyl sulfone | 2 M | 0.2 M | −0.52 … +0.95 |
| **DMDPhPDA, copy A** | chloroform | 2 M | 0.5 M | −2.74 … −2.38 |
| **DMDPhPDA, copy B** | CH₃Cl | 2 M | 0.5 M | +0.26 … +0.62 |

Copy A sits 1–2 decades above its own malonamide analogue at the same diluent and acidity — the
expected gain from swapping a bidentate O,O malonamide for a tridentate N,O,O pyridine dicarboxamide
at 2.5× the ligand concentration.  Copy B sits 2.5–3.5 decades above it, which would make this small,
unfluorinated, plain-phenyl diamide one of the strongest lanthanide extractants in the entire corpus.
Copy A also sits ~3 decades *below* the PDA congeners measured in phenyl trifluoromethyl sulfone,
consistent with that solvent being far more strongly solvating than chloroform; copy B would make
chloroform equal to or better than the sulfone, which inverts a well-established diluent effect.

### E6. Metadata quality

Copy B is a thin bulk import: machine-generated IUPAC name, `updated ORNL` in place of a data location,
no temperature, no contact time (`- -`), no phase-volume ratio, and a later entry date under a
different author.  Copy A carries the publication title, authors, year, table number, 25 °C, 30 min
contact, volume ratio 1.0.  (The retained `safe_exp_id`s put all 70 copy-B rows in `Am_SAFE.csv`
although none is americium — but this is *not* differential evidence: 25 of copy A's 70 retained ids
come from `Pd_SAFE.csv`, and upstream both copies are mirrored across metal files unrelated to their
own metals.  Discount this point.)

### E7. This is the only such case in the bundle

Nine structures appear under both an `updated ORNL` import and a curated entry.  For **six** of them
the two channels carry entirely different DOIs — genuinely different experiments on the same molecule.
**Three** share a DOI: DMDODGA, DODDbDGA and DMDPhPDA.  Of those three, only DMDPhPDA is actually a
double entry of the same measurements:

| structure | ORNL rows | curated rows | rows under the shared DOI | overlapping (metal, acid, [L]) cells | mean \|Δ log D\| | fraction of Δ that are exact integer decades |
|---|---|---|---|---|---|---|
| DMDODGA | 70 | 379 | 70 / 168 | 7 | 0.054 | **0 %** |
| DODDbDGA | 1 | 13 | 1 / 13 | 0 | — | — |
| **DMDPhPDA** | **70** | **70** | **70 / 70** | **70** | **2.043** | **100 %** |

So the discriminating fact is not the shared DOI on its own — it is the *complete* cell overlap with
*every* difference an exact power of ten, and the exact 50/50 row split.  On that test DMDPhPDA is
unique in the bundle.  The ORNL import channel is not globally corrupt (e.g. the malonamide rows
above are also `updated ORNL` and are perfectly sane); the failure is specific to this one re-entry.

### Evidence table

| # | Evidence | Points to | Strength |
|---|---|---|---|
| E1 | 14/14 monotone acid dependence (r = 0.97) vs 0/14 (r = 0.34, four negative) | B corrupt | decisive |
| E2 | Curator's note: Table 1 values = copy B's 5-M values, adjusted ÷10 to match Figure 5 | B = raw table | decisive |
| E3 | Shift is exactly 10⁴/10³/10²/10¹ **per HNO₃ column**, not per row | B = mantissa only | decisive |
| E4 | Ln-series smoothness of Δlog D₄→₅: 0.151 max jump vs 0.964 counterfactual | A correct at 5 M | strong |
| E5 | Copy A is 1–2 decades above its malonamide analogue; copy B is 2.5–3.5 | B implausible | strong |
| E6 | Copy B has no data location, temperature, contact time or phase ratio | B low provenance | weak |
| E7 | Three structures have same-DOI ORNL+curated copies, but only DMDPhPDA has a 70/70 cell overlap with 100 % integer-decade differences | double entry | supporting |
| — | `D_B/D_A` exact to 10 significant figures | not an independent measurement | decisive |

---

## 4. Source publication — what was and was not verified

**Verified.**  The DOI resolves.  Crossref (`api.crossref.org/works/10.1081/SEI-120030392`) returns:
*"Extraction Studies of Lanthanide(III) Ions with N,N′-Dimethyl-N,N′-diphenylpyridine-2,6-dicarboxyamide
(DMDPhPDA) from Nitric Acid Solutions"*, A. Shimada, T. Yaita, H. Narita, S. Tachimori, K. Okuno,
*Solvent Extraction and Ion Exchange* **22**(2), 147–161, 2004.  This matches the title/author/journal
metadata embedded in copy A's `comments_description` exactly (the record's `Publication_Year: 2003`
is the online/accepted year; Crossref says 2004).

From publicly indexed abstract text for this article (Semantic Scholar record
`95f4eb70628b289cf06955e65a95ea303618170a`; the abstract is *not* in the Crossref record and
Semantic Scholar's API withholds it, so this is the search-index copy, re-checked on 2026-08-20), the
study determined lanthanide distribution ratios from **1 to 5 M HNO₃ into a DMDPhPDA/CHCl₃
solution**; the number of DMDPhPDA molecules in the
extracted complex rises from 3 for light to 4 for heavy lanthanides; and **nitric acid concentration
has more influence on the distribution ratio, and on the spread of D across the series, than the
ligand concentration does**.  All three statements are properties of copy A and are contradicted by
copy B, whose D is essentially independent of acidity.

**NOT verified — stated explicitly.**  The full text is paywalled at Taylor & Francis
(`tandfonline.com/doi/abs/10.1081/SEI-120030392` returns HTTP 403) and no open copy of the article,
its Table 1, or its Figure 5 could be reached.  **I did not read the paper's tables or figures, and no
number in this audit is claimed to have been read from the publication.**  The "printed Table 1"
values discussed in §3 are inferred from the dataset's own two transcriptions plus the curator's
comment field — they are not a reading of the source.  A physical copy of SX&IE 22(2) 147–161 would
settle the residual questions in §7 in about five minutes.

---

## 5. Verdict

| Cohort | Rows | Contents | Action |
|---|---|---|---|
| **FROZEN** | 70 | copy A — `DMDPhPDA`, `cond__diluent__chloroform == 1` (the five `condition_id`s in §1; select by the `frozen` list in the JSON, **not** by a single `condition_id`, which only ever names 14 rows) | keep unchanged; this is the corpus record for DOI 10.1081/SEI-120030392 |
| **QUARANTINED** | 70 | copy B — `2-N-6-N-dimethyl-…-dicarboxamide`, `cond__diluent__ch3cl == 1` (five `condition_id`s, see §1) | **exclude from training and evaluation** |
| **CORRECTED** | 70 | the same copy-B rows, rescaled | supplied for sensitivity analysis only — see the caveat below |

**Exact transformation for the CORRECTED variant**

```
log_D_corrected(row) = log_D(row) − k(HNO3)

k = 4   for [HNO3] = 1 M      (all 14 metals)
k = 3   for [HNO3] = 2 M      (all 14 metals)
k = 2   for [HNO3] = 3 M      (all 14 metals)
k = 1   for [HNO3] = 4 M      (all 14 metals)
k = 1   for [HNO3] = 5 M      for La, Ce, Pr only
k = 0   for [HNO3] = 5 M      for the other 11 metals
```

equivalently `D_corrected = D × 10^-k`.  The per-row values are in `dmdphpda_rows.csv`
(`log_D_corrected`, `decade_shift_vs_pair`) and in `dmdphpda_cohorts.json`.

**Why quarantine and not correct is the primary recommendation.**  Applying the transformation
reproduces copy A's `log_D` on all 70 cells to within 1e-9 (max |Δ| = 4.86 × 10⁻¹⁰) — verified
independently.  A corrected copy B therefore adds **no information**: it adds 70 exact duplicate
measurements, which double this ligand's weight in every pooled metric.  And because the corrected
labels are *computed from copy A's labels*, any split that separates the two copies — they sit in two
different `series_id`s, so `unseen_series` and `unseen_conditions` both can — turns the corrected
cohort into a target-leakage vector: the test target is then a deterministic function of a training
target.  Keep `frozen` for the primary analysis; run `corrected` only as the control that shows any
observed change comes from fixing the decades rather than from deleting 70 rows, and only with a
split that keeps the whole extractant on one side.

The 5-M correction for La/Ce/Pr (k = 1) is inherited from copy A's Figure-5 reconciliation, not
derived from a numeric rule — the printed 5-M column has no uniform multiplier, so those three cells
rest on E2 + E4 rather than on E3.

**Do not** clean up the diluent strings or impute the missing metadata before quarantining.
Merging `cond__diluent__ch3cl` into `cond__diluent__chloroform` *alone* does **not** collapse the
copies — measured: all 140 `row_id`s survive, because `cond__temperature_C` and
`cond__contact_time_min` still differ.  The collapse happens as soon as a clean-up also fills copy
B's missing temperature/contact time (or drops those columns from the condition key): the two copies
then become one cell and get averaged, silently fabricating a log D that matches neither — for the
1 M column, the mean of a value and its own 10⁴-times-larger twin.  Both clean-ups are legitimate on
their own and must happen **after** the quarantine, never before.  Note also that the cohort carries a
**third** spelling of the same solvent, `cond__diluent__chcl3` (61 rows, other ligands); a complete
merge has to fold in all three.

---

## 6. Why this matters numerically

* The two copies are mutually inconsistent by a mean of **2.043 log units** over 70 identical cells
  (Σ|Δ| = 143.0).
* **The error floor, stated precisely.**  A predictor that gives the two copies the same prediction
  (i.e. that does not treat the solvent *spelling* or the missing temperature/contact time as
  physics) has minimum total |error| = Σ|Δ| = 143.0 over these 140 rows: a floor of **0.0272 on
  pooled MAE** across the 5 248-row cohort (143.0 / 5 248) and **1.021 on this extractant's own MAE**
  (143.0 / 140) — about what the best whole-corpus model (≈1.0) manages anywhere.
  The earlier phrasing "no predictor can fit both" is **too strong**: ten numeric feature columns do
  differ between paired rows (§2), so a sufficiently flexible model *can* separate them — and that is
  the worse outcome, because fitting both means memorising a solvent-name spelling and two
  missing-metadata flags as a 1–4 decade "diluent effect".  Either way the duplication costs; the
  0.0272 figure is the floor for the honest model, not a bound on the damage.
* **The `ch3cl` flag is mostly this one publication.**  70 of the 84 `cond__diluent__ch3cl` rows in
  the cohort (83 %) are copy B; the remaining 14 are a single phenanthroline-carboxamide.  Any
  learned "CH₃Cl vs chloroform" contrast is therefore very nearly a learned copy-A-vs-copy-B contrast.
* The ligand is a **singleton chemotype** (`tanimoto_cluster = tan084`, `ecfp_cluster =
  d2686f132d536208`, no other extractant in either cluster), so under a **chemotype/extractant**
  hold-out all 140 rows land in the same fold together — there the damage is a doubled test weight
  and 70 unfittable rows, not leakage.
* **Under a series- or condition-level split it *is* leakage.**  The two copies carry two different
  `series_id`s (copy A `54245e3239ad5d14`, copy B `6ea59c50e60297e4`) and ten different
  `condition_id`s, so an `unseen_series` / `unseen_conditions` fold can put copy A in train and copy B
  in test.  That fold guarantees a 1–4 decade error on 70 test rows.  Worse, the **CORRECTED** variant
  builds copy B's labels out of copy A's labels, so under such a split a test-fold target is a
  deterministic function of a train-fold target — straightforward target leakage.  Use `corrected`
  only with a split that keeps the whole extractant together.
* These 140 rows are **43 % of all chloroform-family rows in the cohort** (140 of 324: 179
  `chloroform` + 84 `ch3cl` + 61 `chcl3` — the cohort spells the same solvent three ways).  Counting
  only the two spellings this ligand uses gives 140/263 = 53 %, which is the figure to quote only if
  `chcl3` is deliberately excluded.  Either way, any diluent-level or solvent-physics effect estimated
  on this corpus rests substantially on this double-entered publication.
* For scale: 0.027 pooled MAE on a ≈1.0 baseline is ≈2.7 %, i.e. at the bottom edge of the MASSACTION
  feature-block gain (−3…−8 % macro MAE) recorded in gen5.  Removing one transcription error buys
  about as much as the smallest confirmed feature-engineering win of the whole programme.

---

## 7. Residual uncertainty

1. **The source was not read.**  The verdict rests on internal evidence and on the curator's comment
   field, not on Table 1 or Figure 5.  If — contrary to E1, E4 and E5 — the paper's real distribution
   ratios were the flat 1–36 range, then copy A would be the corrupted copy and the entire verdict
   inverts.  I judge this very unlikely (a neutral solvating extractant with no nitrate dependence is
   not a thing), but it is not excluded by anything I could read.
2. **The three 5-M cells for La/Ce/Pr** depend on the curator's reading of Figure 5.  E4 corroborates
   it strongly but does not prove it.  If those three are wrong, the affected rows move by exactly one
   decade in **both** cohorts:

   | cell | frozen `row_id` (copy A) | quarantined `row_id` (copy B) |
   |---|---|---|
   | La, 5 M | `c470272390d84baa` | `d146b2b62a50c06b` |
   | Ce, 5 M | `10d67f043c384919` | `c5b795856ba8ff35` |
   | Pr, 5 M | `b59cabe01e035868` | `97bbd0581b0c0a35` |
3. **The Publication_Year field says 2003, Crossref says 2004.**  Same article; the record is not
   pointing at a different paper.
4. **Corpus-wide, unrelated:** every DOI string in the upstream builder that ends in
   `, https://doi.org/10.1021/jacs.5c19738` (31 132 of 48 138 rows) carries a token that does not
   belong to the cited work, and one title string in the upstream builder CSVs contains a literal
   spreadsheet range (`Benze+A1514:V1519ne-centered tripodal diglycolamides: …`, present in 10 of the
   `*_SAFE.csv` files and reaching **47 rows** of `runs/gen7_architecture/cache/recovered_cells.parquet`
   through `nuisance__data_location`; the bundle parquet has no title field, so it is untouched).
   Neither affects this audit; both indicate the provenance columns have been edited in a spreadsheet
   and deserve their own pass.

---

## 8. Reproduction

Both copies are found by canonical SMILES, not by name:

```python
smi = "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1"
```

`row_id` is `sha1(f"{extractant}|{condition_id}|{metal_symbol}").hexdigest()[:16]` over rows with
`log_D > -6.0`, per `gen7/recovered.py::recovered_cell_table`, and `condition_id` is a sha1 over the
whole `cond__*` vector (`levels.condition_labels`).  All 140 ids were verified present in
`runs/gen7_architecture/cache/cohort.parquet` with identical `log_D`.

Select the cohorts by the `row_id` lists in `dmdphpda_cohorts.json`, or equivalently by
`extractant == smi` plus `cond__diluent__chloroform == 1` (frozen) / `cond__diluent__ch3cl == 1`
(quarantined).  A single `condition_id` selects only 14 rows.

Note on `dmdphpda_rows.csv`: the `D` column is `10 ** log_D` and `log_D` is stored to 9 dp upstream,
so `D` round-trips to ~1e-8 relative (e.g. copy B's La at 5 M is `3.600000001929`, not `3.6`).
Compare `D` with a tolerance; `log_D` is the authoritative column.

### Adversarial re-verification, 2026-08-20

Every number above was recomputed from `cohort.parquet`, `dataset.parquet`, `recovered_cells.parquet`
and the upstream `*_SAFE.csv` by a second agent; 63 checks pass.  Six statements in the first draft
did not survive and have been corrected in place: the E4 Spearman values (+0.987 → **+0.992**,
−0.029 → **−0.024**), the E7 DOI tally (eight of nine → **six of nine**; three structures share a
DOI), the `condition_id` "prefixes" (each names **14** rows, not 70), the diluent-merge warning
(a diluent-only merge does **not** collapse the copies), "no predictor can fit both" (**ten** feature
columns differ), and the chloroform denominator (**324** rows across three spellings, not 263).
The verdict itself — copy B corrupted, quarantine it — is unchanged and, on the new E7 evidence,
better supported than in the first draft.
