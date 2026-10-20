# Data-quality inventory of the 521-cell gen13 cohort used by gen20/gen21 (direction E)

Nothing in the repository was changed. All scripts and intermediate files are in the scratch folder `C:/Users/Bandai/AppData/Local/Temp/claude/D--ml-separator-gh/05bd353e-2298-45ac-ba07-8e09a6694436/scratchpad/r3inv/` (`s0_load.py` … `s19_noise.py`). Python ran with `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` was written.

**Labels used below:**
- **VERIFIED**: shown by source fields in the bundle or archive (strings, provenance, archive-parsed conditions).
- **DATA-INTERNAL**: found by a fixed rule over the values themselves, with no model and no smoothness test.
- **SUSPECTED**: rests only on curve shape. These are reported but not used for V2.

## 0. Reproduction and base facts (verified)

- **Bench check passes.** `gen20k.data.load_bench()` passes its manifest check. The bench has 521 cells, 3,871 raw rows, 3,359 (cell, metal) entries and 216 replicated entries (728 raw rows).
- **C3.** C3 means a non-empty strict-adjacent query and `make_design(cell_view,'adjacent','metal').k_max >= 3`. This is the same definition as `generations/gen20_1/gen201/data.py:159-161,170-173`. C3 has **229 cells and 1,802 query pairs**. All cells with ≥7 metals are C3; of the 5- and 6-metal cells, 5 and 28 are C3.
- **Rebuild matches exactly.** I rebuilt the cohort from `dataset with 3D structures/dataset.parquet` plus `runs/gen6_provenance/provenance_table.parquet`, using the gen13 key (`gen13sep/cohort.py:96-110,148-154`). It reproduces 521 / 3,871 / 3,359 / 216 / C3 229 / 1,802 exactly.
- **Archive join is complete.** All 3,871 cohort rows join the gen19 archive `dataset_all_metals/clean/master_clean.parquet` on `source_record_id = int(safe_exp_id.split(':')[-1])`, and log D agrees to 1e-6 on every row. This join gives access to `D_raw`, `comments_raw`, `data_location`, `entry_author`, and parsed `complexant_*` / `solvent_key` fields that the bundle does not carry.
- **Missing audit files.** `generations/gen21/AUDIT_2026-10-19.md` and `audit.pdf`, both named in the round-3 brief (§2), are not in the repository. The DMDPhPDA "mantissa" text comes from `generations/gen20_1/IMPLEMENTATION_PLAN.md:82-90` and `runs/gen8_architecture/case_studies/dmdphpda_source_audit.md`.

## 1. Defect summary

| ID | Defect | Status | Cells (C3) | Entries | C3 query pairs touched | Main source |
|---|---|---|---|---|---|---|
| D1 | DMDPhPDA copy B: printed Table-1 mantissa without its column exponent | VERIFIED (internal, exact); paper not read | 5 (5) | 70 | 60 (59 duplicate copy A's SF; the 5 M Pr/Nd pair is wrong by +1.000) | gen8 audit; this scan |
| D2 | Same Table 1 entered by two curators; cells split only by contact time 15 vs NaN | VERIFIED | 3 dropped members (3) | 24 | 6 | archive authors/locations |
| D3 | Hidden-condition merge: rows with different aqueous complexant (name or concentration) or different diluent mixture averaged into one cell | VERIFIED (archive fields) | 17 (11) | 137 | 88 (77 pairs at pair level) | archive `complexant_*`, `solvent_key` |
| D4 | ORNL re-import of the cis- and trans-Me2-TODGA curves under a stereo-free SMILES; two curves 1–2 decades apart in one cell (`0ff595d576155f6c`) | VERIFIED provenance; strongly supported by values | 1 (1) | 14 | 12 | SMILES, publication, author, values |
| D5 | Detection / reporting-resolution floor (ties at one unit of the last reported decimal) | DATA-INTERNAL | 6 (4) | 24 | 20 | `D_raw` strings |
| D6 | Replicate disagreement (repsd > 0.5); this is COH-REPCAP | DATA-INTERNAL | 15 (8) | 78 | 47 | `repsd__*` |
| D7 | "Integer-decade replicate errors" (gen20 review) | Not supported: none exact; all explained by D3, D4 or a different figure | — | 12 | — | see §4 |
| D8 | Isolated integer-decade jumps along the series | SUSPECTED only | ≈6 C3 | — | ≈10 | curve shape |
| D9 | Condition-key precision (too fine / too coarse) | VERIFIED | see §6 | | | key code |
| D10 | Low reporting precision (1 significant figure) | DATA-INTERNAL flag | 24 C3 cells | 90 readings with m = 1 | 58 pairs with summed rounding half-width ≥ 0.2 dex | `D_raw` |
| D11 | Cross-publication copy groups | Handled by design BPG | 44 (32) | | 299 | gen19 copy groups |

Quarantines applied upstream, before the 521 cells were built: 129 rows carrying TODGA's SMILES under another name, and 3 sentinel rows at log D ≤ −6 (`gen13sep/cohort.py:118-130`; `generations/gen13_separation/DATA_AUDIT.md:37`). None of these 132 rows sits in a cell with ≥2 metals (`gen20_1/IMPLEMENTATION_PLAN.md:72-73`).

## 2. D1: the DMDPhPDA mantissa-only copy (question 1)

**What it is.** Shimada et al., *SX&IE* 22(2) 147–161, DOI 10.1081/SEI-120030392, was entered twice. Copy A is curated: author Thomas Summers, 2026-01-20, "Data Location: Table 1", 25 °C, 30 min, `cond__diluent__chloroform`. Copy B is an ORNL import: author Baosen Zhang, 2026-02-10, comment `updated ORNL`, no data location, no temperature or contact time, `cond__diluent__ch3cl`, all `safe_exp_id`s from `Am_SAFE`. Both sit in publication `pub_8415538291` under the same SMILES `CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1` (singleton chemotype). Sources: `runs/gen8_architecture/case_studies/dmdphpda_source_audit.md:15-36,61-76,108-142`; `docs/results/gen7_architecture_results_20260819.md:654-663`; `docs/results/gen8_architecture_results_20260820.md:685-702`.

**Cells in the 521.** All 10 cells have 14 metals, k_max 6, are in C3 and have 12 query pairs each.

| HNO₃ | Copy B (quarantine) | Copy A (keep) |
|---|---|---|
| 1 M | `9754896d384a387f` | `f457e9a3d6532cc8` |
| 2 M | `cd097fce666bd34a` | `e48388064c30fbb4` |
| 3 M | `196797dc24f71fda` | `fbd013078d5396c3` |
| 4 M | `8a78c3b1777b0625` | `575484a7b0e30ce4` |
| 5 M | `7054e94bc4c6833f` | `ceb21275e22d7bc5` |

**Evidence in the raw data (recomputed here).**
- **Exact decade shifts.** logD(B) − logD(A) is +4 / +3 / +2 / +1 for all 14 metals at 1 / 2 / 3 / 4 M. At 5 M it is +1 for La, Ce and Pr and 0 for the other 11. The largest |Δ − round(Δ)| is 4.9e-10.
- **Copy B carries the bare mantissas.** Archive `D_raw` for copy B are the mantissas: at 1 M, La is `3.000000001936503` against copy A's `0.0003`; at 4 M, Lu is `20.0000…` against `2`. Copy B's values are 10**logD round-trips.
- **59 of 70 entries differ.** Copy B's strict-adjacent log SF equals copy A's on 59 of 60 pairs. The exception is the **5 M Pr/Nd pair, +1.000**.
- **Curator's note.** Three copy-A rows (`La_SAFE:12888`, `Ce_SAFE:12893`, `Pd_SAFE:12898`; experiment series `9584bcdf303cbf50`) carry the comment "Value in Table 1 off from value shown in Figure 5 by factor of 10^-1. Value adjusted to match Figure 5".
- **No other exact copy in the cohort.** No other pair of cells with the same extractant and publication shows exact integer-decade differences on all shared metals. The other 13 such pairs have one coincidental exact-decade difference each, between low-precision values.

**How it was detected.** gen7 matched on metal, acid, acid concentration and extractant concentration rather than on solvent name. A scan keyed on the full condition set had found 0 exact-decade pairs ("61 after the fix", `docs/HANDOFF_FOR_CHATGPT.md:1374-1378`). gen8 then ran a forensic audit (E1–E7). gen20.1 defined the rule "SMILES AND `cond__diluent__ch3cl == 1`" (`gen20_1/gen201/data.py:38-39,119-128`) and the sensitivity COH-NODMD, which changed no verdict (`gen20_1/README.md:34`).

**Can the correction be verified?**
- Quarantining copy B is **VERIFIED** on internal evidence: the ratios are exact and constant per acid column, and the curator's note confirms it. The paper itself was not read; it is paywalled (gen8 audit:246-252).
- A "CORRECTED" copy B would equal copy A exactly, so it adds no information and would act as leakage under series splits (audit:280-289). **Quarantine is the right action; correction is not.**
- **Residual uncertainty (1 pair):** the Pr/Nd pair of copy A at 5 M (`ceb21275e22d7bc5`) rests on the curator's reading of Figure 5, supported by series smoothness (E4).
- Copy B has nrep = 1 everywhere, so **neither COH-REPCAP nor `REPLICATE_SD_CAP` touches it**.
- Copy A's 1 M cell (`f457e9a3d6532cc8`) has only one-significant-figure D values (1–4 × 10⁻⁴), so its 12 strict-adjacent SF values are quantized. This is a D10 precision flag, not an error.

## 3. D5: detection floors and limits (question 2)

**No censoring is written in the data.**
- All 3,871 `D_raw` values parse as plain numbers; none contains "<" or ">".
- No `comments_raw` string in the 521 cells contains '<', '>', 'detection', 'limit', 'below', 'n.d', 'not detect', 'LOD' or 'DL' (0 matches).
- Floors can therefore only be inferred from patterns.

**Exact-decade log D values:** 90 rows (−4: 3, −3: 7, −2: 16, −1: 21, 0: 27, +1: 8, +2: 5, +3: 3).

**gen19 rule.** Definition at `generations/gen19_chem_transfer/scripts/g19_feasibility.py:1518-1528`; see also `gen19 DATA_AUDIT.md:366-373,468-471`. The rule: an exact-decade value ≤ −2 that equals the minimum of its (publication, extractant) group and is shared by ≥2 rows. Applied to the 521 cells it finds **18 rows in 4 cells, all C3**, and 0 ceiling rows:
- `pub_1605e436d7` TODGA at −2: `029b985df55ed873` (Sm, Eu) and `d294b06ac73b17a1` (Sm, Eu, Gd, Tb).
- `pub_a49146c8fd` tetraoctyl-1,10-phen-2,9-dicarboxamide at −3: `4ab68efc604014ca` (Ce, Tb, Ho, Yb, Lu at D = 0.001; the other metals are 0.002–0.009).
- `pub_a49146c8fd` tetrazapentacycle at −2: `b33dc03c40457417` (Eu, Dy, Ho, Er, Tm, Yb, Lu at D = 0.01).

**Resolution-floor rule (proposed).** Flag a reading whose D equals one unit of the finest decimal place used anywhere in its cell, that is also the cell minimum, and that is tied by ≥2 readings. It finds **24 entries in 6 cells, 4 of them C3, touching 20 C3 pairs**: the 4 cells above plus TBDGA `pub_0a8da089ad` `3bb58e5ef9b428b7` (La, Nd, Eu, Tb, all D = 0.1) and `786daf3e0a82df88` (Ce, Gd at D = 0.1), neither in C3. The gen19 rule is a subset of this one. Such ties produce spurious log SF = 0. The −0.954 Ce/Pr value in `4ab68efc604014ca` equals log₁₀(1/9), an artefact of the floor.

**Other tie patterns.** 23 cells have ≥2 metals tied at the cell minimum (61 entries); most are non-decade coincidences at 2–3 significant figures.

**Suspected low-D floor (pattern only, not confirmed).** C5BTBP, `pub_6a0ef8528c`, 15 cells, all C3, 148 C3 pairs. La's log D moves only 0.6 across 15 acid × [L] conditions (median −4.24), while Lu moves 2.9. Cohort-wide, 89 entries have log D ≤ −3.5 (18 C3 cells); 79 C3 pairs touch one and 58 have both metals ≤ −3.5. At ≤ −4 there are 47 entries, 46 pairs and 22 "both". These deserve an interval / censoring sensitivity but are not a source-verified floor.

## 4. D6/D7: decade-level anomalies and the replicate analysis (question 3)

**Replicate spread.** `repsd__` quantiles over the 216 replicated entries (10/25/50/75/90 %) are 0.007 / 0.061 / 0.302 / 0.679 / 1.066 (max 1.613). 78 entries exceed `REPLICATE_SD_CAP = 0.5` (`gen20k/cov.py:101-114`), and 70 have a range ≥ 0.95. The C3 counts reproduce gen20.1 COH-REPCAP exactly: 52 entries, and 1,802 → 1,755 pairs.

**The "12 % integer-decade" claim** (`gen20_kshot/manifests/review_reports.json:250`; `GEN20_REPORT.md:402`; cov.py docstring "decade-level double entries"). I reproduce the count: 12 of the 24 nrep = 2 entries with range ≥ 0.95 lie within 0.05 of a whole decade. But none of them is an exact power of ten (|Δ − round| is 0.016–0.046, against ≤ 5e-10 for DMDPhPDA), and every one is explained by source fields:
- **6 are hidden-condition merges (D3):**
  - `bd699b6a4b2c634b`, DOODA(C12), Dy / Er / Gd / Ho / Tb: DTPA 0.05 M with NaNO₃ 3 M vs 5 M (comments of `Ca_SAFE:13766` vs `13780`).
  - `48baeebc441c0964`, TODGA, Sm: DTPA 0.02 M with NaNO₃ 0.54 vs 1.73 M.
- **1 comes from two different figures:** `b35f13943d04ca50` Er, Figure 2 vs Figure 4.
- **5 are cell `0ff595d576155f6c` (D4).**

**Conclusion:** no replicate in the 521 cells shows the exponent-corruption signature. The only verified decade corruption is DMDPhPDA copy B, which is not a replicate.

**One exact power-of-ten pair.** `a3bc2e7b55a95970` (DOODA C8), Tm: 1.25454096910458 with no complexant vs 0.125454096910458 with TEDGA 0.001 M. These are different hidden conditions in the same Figure 3. The identical 15-digit significand is most likely a coincidence of log-axis digitizer pixels (digitized values repeat, e.g. 156.449219 for three metals in `b4fd2a49bcbf3d25`). This is a conjecture, not counted as corruption.

**D8, isolated jumps (smoothness only).** 11 strict-adjacent pairs have |log SF| within 0.1 of a non-zero integer (9 in C3).
- **Explained by verified rules:** `7054e94bc4c6833f` Pr/Nd (D1) and `4ab68efc604014ca` Ce/Pr (D5).
- **Verified explained jump that is not near an integer:** `71f1d37fa44d6ef0` Eu/Gd +1.18. Gd comes from Table 4 with TEDGA 0.01 M; La–Eu come from Figure 3 with Amicacid 0.5 M (D3).
- **Unexplained, suspected only:**
  - `543ce167788b1f3e` DODDdDGA (Table SI1): Ho = 1228 against Dy 127 and Er 378; La/Ce −0.954.
  - `6ff6056da7469332` (ORNL): Tm = 0.065 against Er 0.008 and Yb 0.005.
  - `8cb0deef770e3ca0` (ORNL, `ch3cl`): the Tm–Lu block is about 10× above the trend.
  - `3ebd75679757f580` La/Ce, `da6bc9ba440520e9` Ce/Pr (ORNL), `bdb8b0e62f40bcf2` Eu/Gd.
  - Not in C3: DMDODGA `9dc40db146f97226` and `30c8114cc055d4b0` (Pr/Nd).
- **Non-integer spikes in C5BTBP ORNL cells:** `c950dc2982e96b83` Ce, `326289072f176f80` Sm, `9586db05ac5e234b` Ce.
- None of these can be checked against a source; ORNL rows have no data location. **Exclude them from V2.**

## 5. Replicate definition and averaging (question 4)

**Definition.** A "replicate" is two or more bundle rows, after quarantine, that share canonical SMILES, gen6 `publication_id`, the 64 `cond__*` columns (floats formatted to 9 significant figures, NaN treated as a value) and the metal. They are averaged as the arithmetic mean of log₁₀ D, i.e. the geometric mean of D, with no weights. `nrep__` is the count and `repsd__` is the pandas std (ddof 1). `replicate_sd_median` is the median over the cell (`gen13sep/cohort.py:159-168,184-185`). Pairs are always formed from the means (`gen20k/data.py:88-92`).

**Upstream deduplication.** Rows with identical conditions and identical D were already collapsed upstream, and `replicate_id` is a surrogate (all 216 entries are 'surrogate'; `runs/gen6_provenance/provenance_report.md:11,32-53`). Every remaining "replicate" therefore differs in something.

**Classification using the archive.**

| Class | Entries | Cells | Median repsd | repsd > 0.5 | What differs |
|---|---|---|---|---|---|
| A | 117 | 14 | 0.494 | 57 | complexant name or concentration, or solvent mixture (archive fields); after splitting on them, the residual within-group sd is **≤ 0.025** |
| B | 68 | 21 | 0.048 | 7 | only `data_location` (figure vs table) |
| C | 31 | 6 | 0.133 | 14 | nothing recorded; 13 of the 14 are in `0ff595d576155f6c` |

- Excluding class A and that cell, the median repsd is **0.049** (n = 86; 75th percentile 0.111), against the 0.302 used in gen13/gen20.
- This revises how gen6 read the free-text differences ("free text only: 208 cells … NOT evidence of a different experiment"). The archive parses physical complexant concentrations out of that same comment text.
- **Noise of the log SF pair.** I found 53 adjacent pairs, in 10 cells, read twice under the same hidden condition from different data locations. For these, sd(Δlog D_A) = 0.244, sd(Δlog D_B) = 0.167 and corr = 0.983, so sd(Δlog SF) = 0.086 (median |Δ| 0.007). The error is common-mode. Conjecture: these are mostly the same measurement reported twice, so they bound transcription noise, not laboratory repeatability.

## 6. Precision of condition merging (question 5)

**Key exactness.** The key is the 64 bundle columns at 9 significant figures with NaN as a value. The bundle has no complexant, holdback, nitrate or phase-ratio columns, so these merge too coarsely (D3).

**D3 cells (17).**
- C3: `b4fd2a49bcbf3d25`, `a1d1bb9061317e70`, `bd699b6a4b2c634b`, `d5e6b9dc02e2364e`, `372778033811af53`, `71f1d37fa44d6ef0`, `a3bc2e7b55a95970` (`pub_d68ed8c8cf`); `d294b06ac73b17a1`, `48baeebc441c0964` (`pub_1605e436d7`, DTPA / NaNO₃); `c41d53a5c2625ad2` (BTP-4Me 0.05 vs none); `a9cb3ebbf14bf055` (Isopar L with 10 % vs 50 % Exxal 13, both coded `cond__diluent__other`).
- Not C3: `754e9dee83f16dd1`, `818f1d78902d2a6d`, `b3438d0b792d8a58`, `180c0a7827d406e0`, `2cf8480c768c5553`, `0ecd4ac4bf29bfb9` (La without complexant, Eu with CDTA).
- `372778033811af53` and `71f1d37fa44d6ef0` are assembled from two different complexant systems even without replicates.

**Too-fine splits.**
- **NaN vs value:** exactly 3 cell pairs differ only by NaN, all with contact time 15 min vs NaN. They are bit-identical on 8 metals: `5a6a2f3bf13a7d55` / `c61ecb085d8df371` (T2EHDGA / TEHDGA), `5df28757679cae26` / `f8ed87d3cf7f4281` (TODGA), `247f57741141188a` / `62e7cdd6fa003391` (THDGA), all in `pub_b9dfc468cd`. In each pair, Thomas Summers 2026-01-20 recorded "Table 1" and Aurora Clark 2024-12-16 recorded "table1": the same table entered twice.
- **Correction to `gen20_kshot/AUDIT.md:177-178`.** The other two "identical" pairs it lists differ in recorded values, not by NaN, and are not C3: DMDODGA `2e7197a7efd242c9` / `1052c7804a84fd4c` (acid 4.97 vs 5.96 M) and TBDGA `170f4449e685df34` / `80831f6aeb20c04c` (toluene vs n-octane).
- **Suspected ORNL copy across acid conditions:** `44012938119033a5` (4 M) vs `5169c377777d282b` (5 M), 13 of 14 metals bit-identical; both C3, 24 pairs.
- **Diluent spellings.** In the cohort, `chloroform` appears in 16 cells, `ch3cl` in 6 (5 are DMDPhPDA copy B; the 6th is the ORNL cell `8cb0deef770e3ca0`) and `chcl3` in 18. "Hydrogenated tetrapropene" and "tetrapropylene" are also both used.
- **Digitization noise.** 123 same-extractant, same-publication cell pairs differ only by < 1 % relative in acid or [L], most with disjoint metal sets (e.g. `3bb58e5ef9b428b7` vs `786daf3e0a82df88`, acid 0.482051 vs 0.482055 M).

**Effect of key precision.**

| Key precision | Cells | C3 cells | C3 pairs |
|---|---|---|---|
| 9 s.f. (current) | 521 | 229 | 1,802 |
| 4 s.f. | 539 | 231 | 1,808 |
| 3 s.f. | 547 | 244 | 1,870 |
| 2 s.f. | 520 | 265 | 2,021 |

The SAE archive itself rejects 3 s.f. as merging real titration points (`runs/sae_dataset_audit/quality_report.md:89`).

**Publication mixing and copy groups.** Seven cells that would have mixed two publications are already split by adding `publication_id` to the key (`DATA_AUDIT.md:51-55`). Two gen19 copy groups with ≥2 cohort publications hold 44 cells (32 C3, 299 pairs); gen20's audit counts 45 cells, 5 publications, 3 groups. These are handled by design BPG (`gen20k/splits.py:1-9`).

## 7. Proposed V2 rules

None of these rules uses model residuals or curve smoothness; nothing has been applied. Every V2 changes the benchmark, so all baselines must be rerun on it.

| Rule | Exact definition | Status | Raw rows | Entries | Cells (C3) | C3 pairs touched | C3 after removal |
|---|---|---|---|---|---|---|---|
| V2-1 ORNL re-import | Cell where every row is `updated ORNL`; a curated cell exists with the same publication, the same stereo-free SMILES (RDKit `isomericSmiles=False`), acid and [L] within 1 %, and the same diluent under a fixed synonym table {ch3cl, chcl3 → chloroform; hydrogenated tetrapropylene → tetrapropene}. Quarantine the ORNL cell. Matches exactly the 5 DMDPhPDA copy-B cells plus `0ff595d576155f6c` | VERIFIED | 97 | 84 | 6 (6) | 72 | — |
| (V2-1 = D1 only) | | | 70 | 70 | 5 (5) | 60 | 224 / 1,742 |
| (V2-1 = D4 only) | | | 27 | 14 | 1 (1) | 12 | 228 / 1,790 |
| V2-2 duplicate | Same extractant and publication, conditions equal wherever both present and differing only by NaN, log D bit-identical on ≥3 shared metals: drop the member with NaN (`c61ecb085d8df371`, `f8ed87d3cf7f4281`, `62e7cdd6fa003391`) | VERIFIED | 24 | 24 | 3 (3) | 6 | 226 / 1,796 |
| V2-3 hidden merge | Rows of a cell differ in archive `complexant_name`, `complexant_concentration_raw` or `solvent_key`. (a) Primary: re-key the cell including these fields. (b) Alternative: quarantine | VERIFIED | 506 | 137 | 17 (11) | 88 | quarantine: 218 / 1,714; split: 62 sub-cells, 33 C3 sub-cells, 254 C3 pairs |
| V2-4 REPCAP residual | nrep ≥ 2 and repsd > 0.5, recomputed after V2-1 to V2-3: censor the entry | DATA-INTERNAL, label-derived | | 21 before split (16 C3 entries, 15 pairs), of which 13 in `0ff595d576155f6c`; **8 after split**: `b35f13943d04ca50` Dy/Er/Yb, `815c365f83ac70cf` Dy, `5fe61fb5cb043ac8` La/Ce, `75373a48cee6831b` La, `caae8f09ae6fe8a6` Gd | | | |
| V2-5 floor | Resolution floor defined in §3 (contains the gen19 rule): left-censored at D < 1.5·10^e; excluded from point scoring, or treated as an interval in a sensitivity | DATA-INTERNAL | 28 | 24 | 6 (4) | 20 | 228 / 1,782 |

**Union of V2-1 to V2-5 (quarantine variant):** 665 raw rows, 273 entries, 36 cells (25 C3), 230 C3 entries, **185 of 1,802 C3 pairs touched**. C3 after removal: **207 cells / 1,614 pairs**.

**Split variant:** quarantine V2-1 and V2-2, split V2-3, drop the V2-5 floor readings, then drop the 8 residual V2-4 entries. Result: **555 cells, 240 C3 cells, 1,866 C3 pairs**. 204 of the original C3 cells (1,596 pairs) are untouched by any rule.

**Flag-only sensitivities, not in the V2 core:**
- **Low precision:** 90 readings with a one-digit significand m = 1; 58 C3 pairs in 24 cells with summed rounding half-width ≥ 0.2 dex.
- **Suspected low-D floor:** log D ≤ −3.5, 89 entries, 79 C3 pairs.
- **Suspected cross-acid copy:** `44012938119033a5` / `5169c377777d282b`, 24 C3 pairs.
- **D8 smoothness list:** never used for V2.
- **Provenance stratum:** all-ORNL cells, 54 C3 cells and 529 C3 pairs; 8 of the 25 flagged C3 cells are ORNL.
- **BPG copy groups.**

**Notes.**
- Once V2-3 splits are applied, the "replicate floor" of 0.302 no longer represents measurement noise; the gen20 noise constant (cov.py) would change from about 0.30 to about 0.05.
- gen21 never ran COH-NODMD or REPCAP (`GEN21_REPORT.md:213` names decade-level replicate errors as a target for robust modelling). The evidence here says those errors are mostly merged conditions, not decade errors.

## key facts
- Bench check passes; a rebuild from the bundle plus gen6 provenance at 9 s.f. reproduces exactly 521 cells / 3,871 rows / 3,359 entries / 216 replicated entries / C3 229 cells / 1,802 strict-adjacent query pairs (C3 = n_q>0 and metal-model k_max>=3).
- All 3,871 cohort rows join dataset_all_metals/clean/master_clean.parquet on source_record_id = exp_id, with log D agreeing on every row; this exposes D_raw, comments, data_location, author and parsed complexant/solvent fields that the bundle lacks.
- DMDPhPDA copy B cells are 9754896d384a387f, cd097fce666bd34a, 196797dc24f71fda, 8a78c3b1777b0625, 7054e94bc4c6833f (all C3, k_max 6, 12 pairs each; 70 entries, 60 C3 pairs). Shifts vs copy A are exactly +4/+3/+2/+1 at 1/2/3/4 M and +1 for La/Ce/Pr at 5 M (max |d-round| 4.9e-10). Copy B D_raw values are the bare mantissas. Only the 5 M Pr/Nd log SF differs (+1.000); copy B has nrep=1, so REPCAP and REPLICATE_SD_CAP never touch it.
- Copy A rows La_SAFE:12888, Ce_SAFE:12893 and Pd_SAFE:12898 carry the curator note 'Value in Table 1 off from value shown in Figure 5 by factor of 10^-1. Value adjusted'. Copy A's 5 M Pr/Nd pair (ceb21275e22d7bc5) therefore rests on that note plus smoothness; the paper was never read (gen8 audit lines 246-252).
- No 'D<' entries exist: every D_raw is numeric and 0 comments in the cohort mention '<', detection, limit or LOD. Floors are only pattern-identifiable: the gen19 rule finds 18 rows in 4 C3 cells; the resolution-floor rule finds 24 entries in 6 cells (4 C3, 20 C3 pairs): 029b985df55ed873, d294b06ac73b17a1, 4ab68efc604014ca, b33dc03c40457417, 3bb58e5ef9b428b7, 786daf3e0a82df88.
- Of the 78 entries with repsd>0.5: 57 are hidden-condition merges (different archive complexant concentration/name or diluent mixture), 7 differ only by figure/table, 13 are cell 0ff595d576155f6c, and 1 (caae8f09ae6fe8a6 Gd) is unexplained. None is an exact power-of-ten copy, so the gen20 claim of '12% integer-decade double entries' is not supported (the 12 near-integer nrep=2 cases are 6 hidden-condition merges + 5 from 0ff595d576155f6c + 1 different-figure).
- Splitting the 17 hidden-condition cells by archive complexant_name / complexant_concentration_raw / solvent_key gives 62 sub-cells (33 C3, 254 C3 pairs) with residual within-group sd <= 0.025. Excluding these merges and 0ff595d576155f6c, the replicate sd median is 0.049 (n=86), not 0.302.
- Cell 0ff595d576155f6c (ORNL import, stereo-free Me2-TODGA SMILES, pub_0af6736cb1, 4.3 M HNO3 / 0.1 M) averages two curves that track the curated cMe2-TODGA (ac87efdb82643345) and tMe2-TODGA (cf940348affe2b22) cells of the same publication and conditions.
- Rule 'ORNL re-import duplicating a curated cell' (same publication, same stereo-free structure, acid and [L] within 1%, diluent equal under a fixed synonym table) matches exactly 6 cells: the 5 DMDPhPDA copy-B cells plus 0ff595d576155f6c.
- Only 3 cell pairs are split purely by NaN (contact time 15 vs NaN): T2EHDGA/TEHDGA, TODGA and THDGA in pub_b9dfc468cd. Each is bit-identical on 8 metals, entered as Table 1 by two curators (2024-12-16 and 2026-01-20); 6 C3 pairs. gen20 AUDIT.md:177-178's 'five pairs split only by a NaN condition' is wrong for the DMDODGA and TBDGA pairs (they differ in acid or diluent values).
- V2 quarantine union (ORNL re-import + duplicates + hidden-merge quarantine + REPCAP + floor) touches 273 entries / 36 cells / 25 C3 cells / 185 of 1,802 C3 pairs and leaves C3 = 207 cells / 1,614 pairs. The split variant gives 555 cells / 240 C3 / 1,866 C3 pairs; 204 original C3 cells (1,596 pairs) are untouched by any rule.
- Condition-key precision: 9 s.f. gives 521/229/1,802 (cells/C3/C3 pairs); 4 s.f. 539/231/1,808; 3 s.f. 547/244/1,870. There are 123 same-extractant-and-publication cell pairs whose conditions differ by <1% relative (digitizer noise), mostly with disjoint metal sets.
- Smoothness-only suspected decade jumps (excluded from V2): 6ff6056da7469332 Tm, 8cb0deef770e3ca0 Tm-Lu block, 543ce167788b1f3e Ho, plus spikes in the C5BTBP ORNL cells c950dc2982e96b83, 326289072f176f80 and 9586db05ac5e234b. Verified-explained jumps: 7054e94bc4c6833f Pr/Nd (DMDPhPDA), 4ab68efc604014ca Ce/Pr (floor), 71f1d37fa44d6ef0 Eu/Gd (Gd measured under TEDGA, La-Eu under Amicacid).
- All analysis scripts are in C:/Users/Bandai/AppData/Local/Temp/claude/D--ml-separator-gh/05bd353e-2298-45ac-ba07-8e09a6694436/scratchpad/r3inv/ (s0_load.py to s19_noise.py); no repository file was created or changed.

## open
- generations/gen21/AUDIT_2026-10-19.md and audit.pdf, both cited by the round-3 brief, are not in the repository; should the V2 log cite the gen8 forensic audit and gen20_1/IMPLEMENTATION_PLAN.md:82-90 instead?
- Should V2-3 (hidden-condition merges) split cells by the archive complexant/solvent fields, or quarantine them? Splitting creates sub-cells whose bundle features are identical (complexant is not a feature), and it changes C3 from 229 to 240 cells.
- Is REPCAP (repsd>0.5) acceptable as a V2 rule even though it is label-derived (it uses replicate spread, not model error)? After the V2-3 split it removes only 8 entries.
- The curator's Figure-5 correction for copy A at 5 M (La/Ce/Pr) cannot be checked without the paper (SX&IE 22(2) 147-161 is paywalled). Is anyone able to obtain Table 1 and Figure 5?
- Are the C5BTBP light-lanthanide values (log D around -4.2, nearly insensitive to conditions) a real detection floor? An interval / censoring sensitivity at log D <= -3.5 would affect 79 C3 pairs; there is no source statement of the detection limit.
- Are the ORNL DMDODGA cells (labelled diluent '1-octanol 0.5 / kerosene 0.5') re-transcriptions of the curated kerosene / kerosene+40% octanol cells? Median |Δ| is 0.01-0.1 against curated cells under different recorded diluents, and 44012938119033a5 vs 5169c377777d282b are 13 of 14 bit-identical at 4 vs 5 M acid.
- The common-mode estimate (corr 0.983, sd(ΔSF) 0.086) comes from same-condition readings at different data locations, which are probably double reports of one measurement; true laboratory repeatability of log SF remains unmeasured in this cohort.