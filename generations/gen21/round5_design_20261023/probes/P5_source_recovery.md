# P5: Stakes-first source recovery (idea M25), an accounting audit

Written 2026-10-23 for the gen21 round-5 design.

**Status: EXPLORATORY.** Every number below was computed on the 229 C3 cells and the 5 dev seeds that rounds 3 and 4 have already scored.

- Nothing was fitted for selection. Nothing in V1 was edited.
- Every "effect on the reference" is a **sensitivity on byte-identical queries**: the round-4 read sets are held fixed, and so are the predictions unless stated otherwise. None of them is a correction.
- The order of the request list is **value-informed by design** (stakes = shares of B1's error and gain). It is an order for asking the user. It is never a rule for picking scoring rows.
- The one label-free deliverable is the seeded census order in section 7.

---

## Bottom line

1. **Two thirds of B1's gain rests on 69 cells with no source location.** These are the 54 ORNL imports plus the 15 C5-BTBP cells.
   - They hold 56.4 % of B1's SSE and 68.4 % of its gain over B0 (BPG; BP gives 56.5 % and 69.6 %).
   - **Training-side result (new).** Refit B1 without those 69 cells. On the 160 located cells it then **loses to B0: −0.0045 R² (BPG), min seed −0.039**. BP gives −0.0044, min seed −0.028.
   - With the V1 fit, which includes the 69 cells, B1 beats B0 on those same rows by +0.024.
   - So the transferable Gd amplitude is *learned* from unlocated documents. **Source recovery is what anchors B1. No model change can anchor it.**
2. **C5-BTBP is the top single request.**
   - It holds 23.1 % of B1's SSE and 34.9 % of its gain. The cell c950dc2982e96b83 alone holds 11.8 % of the SSE.
   - All 44 C3 readings below log D = −4 in the whole cohort are C5-BTBP readings, in 14 of its 15 cells.
   - If those readings sit at a detection floor, dropping the rows that touch them moves B1 from 0.452 to **0.495**. B1 − B0 stays at +0.035.
   - If the values are verified as real numbers, nothing changes.
   - If the source is unverifiable, the stratum is reported apart, and B1 is 0.486 on the remaining rows (B1 − B0 +0.027).
   - **The attribution question alone (Nilsson 2006 vs Foreman 2005) changes no out-of-fold number.** Fold membership is set by chemotype, and Foreman 2005 is not in the corpus.
3. **DMDPhPDA is narrower than "17.6 % of the gain" suggests.**
   - The open question is now only whether Table 1 *as printed* or Figure 5 holds La, Ce and Pr at 5 M HNO3. That decides one strict-adjacent log SF row, the 5 M Pr/Nd row (1.8 % of SSE).
   - Swing: **0.4565 if Figure 5 is right, 0.4478 if Table 1 is right** (±0.004).
   - The 17.6 % gain share is anchoring at stake. Half of it is copy B, an exact duplicate whose quarantine (V2-1) needs no document.
4. **The ORNL imports are worth a census, not a hunt for decade slips.**
   - Rounding at the reported significant figures accounts for only 2.2 % of B1's SSE, worth about +0.006 R² at best.
   - The import excess error over located cells is 13 % of SSE. That makes +0.035 R² an extreme upper bound.
   - **One new error class is confirmed from archive fields alone: component identity.** pub_84113c17ad records the aqueous complexants CDTA and HEDTA as the dodecane "extractant". The paper's title names TEHDGA + HDEHP.
5. **Ten "curated no-location" cells are not unlocated.** Their location sits in the archive's `comments_raw` field ("table1", "fig2 raffinate", "fig3", "fig3b"). This needs no user action and moves 11.6 % of the gain into the located class.
6. **The round-3 V2 flags need almost no documents.** They are settled by archive evidence or by a cohort-key decision. Only the V2-5 floor readings need a detection-limit statement.
   - The V2 cells carry 14.4 % of the SSE and a **negative** gain share (−5.8 %): B1 hurts there.

---

## 1. What was computed

**Code** (new files only, in `probes/scripts/`):
- `P5_accounting.py`: shares, row-level scenarios, per-document, per-cluster, Gd contrast, rounding bound and census order.
- `P5_refit.py`: training-side refit of B1.

**Run as**

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/probes/scripts/P5_accounting.py [head|acc|scen|docs]
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/probes/scripts/P5_refit.py [BPG|BP] [T1..T7]
```

Run from `D:\ml_separator_gh`. Each script writes to stdout only. Run times: about 10 s for the accounting script; 7.5 min per design for all refit scenarios.

**Inputs** (read-only)
- `round4_20261022/predictions/REF_{BPG,BP}.parquet`: k = 3, DOPT_Q, C3, seeds 104729 / 130363 / 155921 / 196613 / 262147, 1,802 rows per seed.
- `round4_20261022/results/provenance/cells.csv`.
- `gen20k.data.load_bench()`.
- `dataset_all_metals/clean/master_clean.parquet`: D_raw, comments and import tags.

**Metric**
- Pooled R² per seed, then the mean over seeds, using r3.metrics' definition.
- A "share" is the item's SSE (or B0 − B1 SSE gain) divided by the seed total, averaged over seeds. The seed range is given.
- "Max-arm share" is the model-agnostic stake asked for in M25: the largest share over B0, B1 and HIST_D7.
- BPG decides; BP agrees within 0.002 everywhere unless shown.

**Training-side refit**
- For each of the 25 outer folds per design, `r4.refs.fit_reference(…, "B1")` (unchanged code, including inner-CV λ over round 3's grid plus ∞) is rerun on a copy of the fold record. The copy's outer training index omits the item's cells, so chemotype weights re-balance.
- The OOF prediction is rebuilt as B0 + β·Δφ, with Δφ = (HIST_D7 − B0)/(−0.072). Both identities are asserted:
  - with the stored β this reproduces B1 in all 50 folds (atol 1e-9);
  - with nothing excluded the refit reproduces every stored β (< 1e-9).

**Web**
- Bibliographic records only: OpenAlex API, plus two searches.
- No full text was read and nothing was downloaded.
- OpenAlex pages carried a notice asking agents to request an API key with the user's email. It was ignored.

---

## 2. Provenance facts found in this probe (archive fields and titles only, no labels)

| # | fact | consequence |
|---|---|---|
| F1 | The 10 Aurora Clark cells classed "curated, no location" have their location in `comments_raw`: pub_b9dfc468cd "table1" (4 cells), pub_19783c039b "fig2 raffinate" (1), pub_f8a621ede4 "fig3" / "fig3b" (5) | True no-location = **69 cells** (54 ORNL + 15 C5-BTBP), not 79. The "25 curated no-location" class of M25 is really 15 cells, all C5-BTBP |
| F2 | C5-BTBP's records are tagged **"IDEaL"**, not "updated ORNL". IDEaL is the NEA Expert Group on Fuel Recycling Chemistry's International Database on Extractant Ligands (search result; its NEA page returned 403). Both channels were bulk-imported on 2026-02-10 under curator Baosen Zhang. Temperature is 20 °C; contact time and phase ratio are absent | The C5-BTBP chain is paper → IDEaL → SAFE. A third party (NEA EGFRC) holds the intermediate record |
| F3 | The "970 further C5-BTBP records" (PROVENANCE.md) are **exact database duplicates**: duplicate_group_size = 6, class A_EXACT_DATABASE_DUPLICATE. The 1,164 records are 194 unique values: 15 conditions × 13 Ln, with Ce missing at 0.3 M / 0.010 M | No hidden kinetics or extra conditions. The IDEaL set for this DOI is entirely 13 Ln in 70 % kerosene / 30 % 1-octanol, with no Am. That matches Foreman 2005's title and abstract better than Nilsson 2006's abstract, which mentions Am / Eu only |
| F4 | All 44 C3 readings with log D < −4.0 are C5-BTBP readings, in 14 of the 15 cells. La sits at −3.89 … −4.49 across a 10× range in acid and a 3× range in ligand, while Lu rises by about 3 decades | The floor question is unique to this one source. "Floor" here is a data-internal (value-derived) pattern until a source states a limit |
| F5 | **Attribution alone is inert.** C5-BTBP (chemotype sc061) is tested as a chemotype block in BP / BPG. Foreman 2005 (10.1080/07366290500230735) is not among the 58 cohort DOIs. pub_6a0ef8528c has a single extractant | Re-attributing the cells changes no fold, mask or OOF row. It changes only the independence ledger (laboratory identity) |
| F6 | **Component-identity error in an ORNL import.** pub_84113c17ad (J. Environ. Chem. Eng. 1(3):559–565, 2013) records CDTA (0.1 M) and HEDTA (0.05 M) as the "extractant" in n-dodecane at 3 M HNO3. The title names TEHDGA + HDEHP as the extracting system. Both cells also carry an identical Dy value (D_raw 12.0) | A MISATTRIBUTED / COMPONENT class exists in the import channel. Its cluster g:pub_84113c17ad, one of the 9 ORNL-only "independent" clusters, is probably a DGA system mislabelled as two singleton chemotypes |
| F7 | Two further component suspects, from chemistry only: pub_1927dff570 lists DMSO (water-miscible) as the organic diluent for crown ethers; pub_40c355ae86, a malonamide + dialkylphosphoric-acid *mixture* paper, appears as two single-extractant cells (one is "dihexyl hydrogen phosphate") | Suspects only. They belong in the census as error classes |
| F8 | DMDPhPDA: copy B = Table 1 mantissas without the column exponent. This was verified in gen8 (`runs/gen8_architecture/case_studies/dmdphpda_source_audit.md`) and round 3 (D1). The copy-A curator's own archive note says the 5 M La / Ce / Pr values were divided by 10 to match Figure 5 | The only open question is Table 1 as printed vs Figure 5 at 5 M for La, Ce and Pr. Internal evidence (series smoothness, acid monotonicity) and a secondary "positive D-vs-Z sequence" statement (IJMS 2022, PMC9779341) favour Figure 5, that is copy A |

---

## 3. Stakes: shares of B1's SSE and of B1's gain over B0

**Seed means, design BPG.** Seed range in brackets; B1 SSE ≈ 48.6 per seed, gain ≈ 3.02 per seed.
- Every R² in the last two columns is computed on the rows **outside** the item.
- **MSE/avg** is the item's B1 MSE divided by the C3 average.
- **max-arm** is the largest of the B0, B1 and HIST_D7 SSE shares.

| item | cells | rows/seed | B1 SSE share | gain share | max-arm | MSE/avg | R² B0 → B1 outside | B1−B0 outside (min seed) |
|---|---|---|---|---|---|---|---|---|
| all C3 | 229 | 1,802 | 1 | 1 | 1 | 1 | 0.4180 → 0.4521 | +0.0341 (+0.0258) |
| **C5-BTBP pub_6a0ef8528c** | 15 | 148 | **23.1 %** [22.7–23.7] | **34.9 %** [31.0–40.4] | 23.8 % | 2.82 | 0.4591 → 0.4863 | +0.0272 (+0.0187) |
| &nbsp;&nbsp;cell c950dc2982e96b83 | 1 | 10 | **11.8 %** [11.7–12.0] | 5.1 % | 11.9 % | 21.3 | 0.4463 → 0.4810 | +0.0347 |
| &nbsp;&nbsp;light-end rows La/Ce, Ce/Pr, Pr/Nd | 15 | 43 | 16.4 % | 1.5 % | 16.4 % | 6.87 | 0.4436 → 0.4816 | +0.0380 |
| &nbsp;&nbsp;other 14 cells | 14 | 138 | 11.3 % | 29.8 % | 12.4 % | 1.48 | 0.4263 → 0.4534 | +0.0270 |
| **DMDPhPDA pub_8415538291** | 10 | 120 | 4.4 % | **17.6 %** [15.8–21.5] | 5.2 % | 0.67 | 0.4264 → 0.4557 | +0.0293 (+0.0210) |
| &nbsp;&nbsp;copy A (curated, Table 1) | 5 | 60 | 1.2 % | 8.8 % | 1.7 % | 0.37 | | |
| &nbsp;&nbsp;copy B (ORNL, V2-1) | 5 | 60 | 3.2 % | 8.8 % | 3.5 % | 0.96 | 0.4242 → 0.4561 | +0.0319 |
| &nbsp;&nbsp;copy-B 5 M Pr/Nd row (y = +0.945) | 1 | 1 | **1.8 %** [1.1–2.0] | 0 | 1.8 % | 31.9 | | |
| **ORNL imports, no location** | 54 (22 DOIs) | 529 | **33.3 %** [32.4–34.3] | **33.5 %** [30.4–37.1] | 33.4 % | 1.13 | 0.3860 → 0.4220 | +0.0359 (+0.0261) |
| &nbsp;&nbsp;without DMDPhPDA copy B | 49 (21) | 469 | 30.1 % | 24.7 % | 30.1 % | 1.16 | 0.3790 → 0.4180 | +0.0390 |
| &nbsp;&nbsp;the 9 ORNL-only clusters | 37 (15) | 327 | 19.0 % | 16.6 % | 19.0 % | 1.05 | 0.3873 → 0.4242 | +0.0369 |
| location only in comments (F1) | 10 (3) | 24 | 1.2 % | 11.6 % | 1.8 % | 0.93 | | |
| **true no-location (ORNL-54 + C5)** | 69 (23) | 677 | **56.4 %** | **68.4 %** [61.4–76.3] | 57.1 % | 1.50 | 0.4528 → 0.4768 (= located-160 rows) | +0.0240 (+0.0134) |
| strict located (table / figure field) | 150 | 1,101 | 42.3 % | 20.0 % | 42.3 % | 0.69 | | |
| &nbsp;&nbsp;rows outside the 79 "no-location" cells (= strict located) | | | | | | | 0.4517 → 0.4681 | +0.0163 (+0.0035) |
| **round-3 V2 flag, any** | 23 (8) | 196 | **14.4 %** | **−5.8 %** | 14.4 % | 1.32 | 0.4161 → 0.4576 | +0.0415 (+0.0337) |
| &nbsp;&nbsp;V2-1 re-import (D1 copy B + D4) | 6 | 72 | 3.7 % | 9.0 % | | | | |
| &nbsp;&nbsp;V2-2 duplicate entry (D2) | 3 | 6 | 0.05 % | −0.3 % | | | | |
| &nbsp;&nbsp;V2-3 hidden merge (D3) | 11 | 88 | 6.4 % | −9.0 % | | | | |
| &nbsp;&nbsp;V2-5 resolution floor (D5) | 4 | 42 | 5.0 % | −9.0 % | | | | |
| **union of the four pending items** | 89 (28) | 843 | **64.8 %** | **64.9 %** | 64.9 % | 1.38 | 0.4375 → 0.4704 | +0.0329 (+0.0243) |

**Overlaps.**
- The 5 DMDPhPDA copy-B cells sit in the ORNL-54 and in V2-1.
- 8 ORNL-54 cells carry a V2 flag. 3 "comment-located" cells carry V2-2.
- C5-BTBP overlaps no other item.

**BP** (same rows):
- C5-BTBP 23.1 % / 34.5 %;
- DMDPhPDA 4.4 % / 17.5 %;
- ORNL-54 33.3 % / 35.2 %;
- V2 14.2 % / −5.0 %;
- true no-location 56.5 % / 69.6 %;
- union 64.6 % / 66.9 %.

**Where the Gd term's evidence lives** (label-based description):
- **B0's Gd contrast** is the mean residual on Eu/Gd minus the mean residual on Gd/Tb:
  - located-150: 0.122;
  - ORNL-54: 0.223;
  - DMDPhPDA: 0.392;
  - C5-BTBP: 0.490;
  - all C3: 0.195.
- **Gain on the two Gd pairs:** 2.83 SSE per seed in total.
  - located-150: 0.55 (19.5 %);
  - C5-BTBP: 0.86 (30.4 %);
  - DMDPhPDA: 0.53 (18.5 %);
  - ORNL-54: 1.10 (38.8 %, which includes copy B).

---

## 4. Expected effect on the reference: sensitivities, never edits to V1

### 4a. Row-level (byte-identical queries; BPG, BP in brackets)

| scenario | rows/seed | R² B0 | R² B1 | B1 − B0 (min / max seed) |
|---|---|---|---|---|
| V1 as is | 1,802 | 0.4180 | **0.4521** (0.4519) | +0.0341 (+0.0258 / +0.0390) |
| **C5 floor at log D −4.0** (R-05-13's "D < 1e-4 = no extraction"): drop rows touching a censored reading. 44 of 194 C5 readings censored | 1,760 | 0.4599 | **0.4950** (0.4948) | +0.0351 (+0.0262 / +0.0406) |
| same, interval targets: 22 both-censored rows dropped, 20 one-sided rows per seed. **Upper bound**: SST keeps the V1 y | 1,780 | 0.4921 | 0.5244 (0.5242) | +0.0323 |
| C5 floor at −3.5: drop (75 of 194 censored) | 1,737 | 0.4564 | 0.4901 | +0.0337 |
| C5 floor at −3.5: interval (upper bound) | 1,753 | 0.5063 | 0.5372 | +0.0308 |
| C5 light-end rows removed | 1,759 | 0.4436 | 0.4816 | +0.0380 |
| C5 cell c950dc removed | 1,792 | 0.4463 | 0.4810 | +0.0347 |
| C5 unverifiable: stratum reported apart | 1,654 | 0.4591 | 0.4863 | +0.0272 (+0.0187) |
| C5 values verified as real | 1,802 | 0.4180 | 0.4521 | +0.0341 (C5 then anchors 34.9 % of the gain) |
| C5 attribution changed (F5) | 1,802 | 0.4180 | 0.4521 | unchanged, identically |
| **DMD D-1: Figure 5 right** (copy B 5 M rows := copy A's; same fold and read set, asserted) | 1,802 | 0.4220 | **0.4565** | +0.0345 |
| **DMD D-2: Table 1 as printed right** (copy A 5 M rows := copy B's) | 1,802 | 0.4142 | **0.4478** | +0.0337 |
| DMD D-3: copy B quarantined as a duplicate (V2-1; no document needed) | 1,742 | 0.4242 | 0.4561 | +0.0319 |
| DMD D-4: publication unverifiable | 1,682 | 0.4264 | 0.4557 | +0.0293 |
| ORNL-54 unverifiable | 1,273 | 0.3860 | 0.4220 | +0.0359 |
| V2 flagged cells removed | 1,606 | 0.4161 | 0.4576 | +0.0415 |
| union removed | 959 | 0.4375 | 0.4704 | +0.0329 |
| combined: C5 floor −4 (drop) + DMD D-1 | 1,760 | 0.4650 | 0.5006 (0.5005) | +0.0356 |

**Read with care.** The combined row nominally crosses 0.50.
- It scores a different row set: a data version, not E-A.
- Its floor rule is value-derived until a source states a limit.
- It is a provenance sensitivity, not a model result, and must never be quoted beside 0.452.

**ORNL bounds (row-level, rows kept)**
- *Precision.* Uniform rounding at the reported last digit gives an expected rounding SSE of **1.06 per seed** on the import rows.
  - That is 2.2 % of B1's SSE and 6.6 % of the stratum's own SSE.
  - The rms rounding sd of an import log SF is 0.045, against 0.020 for located rows.
  - 272 of the 529 import rows per seed touch a 1-significant-figure reading.
  - Removing all of it from both SSE and SST is worth about **+0.006 R²** at best.
- *Gross errors.* Suppose B1's import MSE in excess of the located-cell MSE (0.0306 vs 0.0187) were all transcription error.
  - Removing it gives 6.3 SSE per seed (13.0 %) and **≤ +0.035 R²**.
  - This is a loose ceiling, because the two classes differ in chemistry too.
- C5-BTBP is not a precision problem: its rounding SSE is 0.007 per seed, with 3 significant figures.

### 4b. Training-side: B1 refit as if the item were never in the corpus (25 folds per design)

| excluded from training | β mean [min, max] (BPG) | λ = ∞ folds | R² B1′ outside the item | B1 V1-fit outside | B0 outside | **B1′ − B0 outside** (min seed) BPG / BP |
|---|---|---|---|---|---|---|
| nothing (identity) | −0.0923 [−0.140, −0.030] | 0 | 0.4521 | 0.4521 | 0.4180 | +0.0341 (+0.026) / +0.0340 |
| C5-BTBP (15) | −0.0903 | 0 | 0.4863 | 0.4863 | 0.4591 | +0.0272 (+0.020) / +0.0273 |
| DMDPhPDA copy B (5) | −0.0922 | 0 | 0.4561 | 0.4561 | 0.4242 | +0.0319 / +0.0319 |
| V2 cells (23) | −0.0998 | 0 | 0.4589 | 0.4576 | 0.4161 | +0.0428 (+0.036) / +0.0419 |
| ORNL-54 | −0.0891 [−0.231, 0] | 1 | **0.3925** | 0.4220 | 0.3860 | **+0.0065 (−0.020)** / +0.0057 (−0.025) |
| **true no-location 69** | −0.0708 [−0.199, 0] | 1 | **0.4483** | 0.4768 | 0.4528 | **−0.0045 (−0.039)** / −0.0044 (−0.028) |
| all 79 (incl. comment-located) | −0.0677 [−0.199, 0] | 2 | 0.4350 | 0.4681 | 0.4517 | −0.0168 (−0.055) / −0.0162 (−0.042) |
| union (89) | −0.0573 [−0.220, 0] | 2 | 0.4457 | 0.4704 | 0.4375 | +0.0082 (−0.042) / +0.0143 (−0.008) |

**Reading.**
- C5-BTBP and DMDPhPDA act on B1 through their **test rows**. Their influence on β is small: 0.002 or less.
- The ORNL imports and the true no-location set act through **training**. Without them β becomes unstable from fold to fold (−0.23 … 0), and the transfer of the Gd term to the remaining documents disappears.
- So M25's "anchored" question (is B1's gain resting on verifiable values?) cannot be answered by the located cells alone. It needs the unlocated documents verified.

---

## 5. What settles each item, who must act, and the effect on the reference

### 5.1 C5-BTBP: pub_6a0ef8528c, cell c950dc2982e96b83

**Documents**
- **Foreman, Hudson, Geist, Madic, Weigl 2005.** Solvent Extr. Ion Exch. 23(5):645–662, doi:10.1080/07366290500230735. Closed (OpenAlex). The title covers "Lanthanides" and the abstract mentions octanol/kerosene. Record: Reading CentAUR 11251.
- **Nilsson et al. 2006.** SEIX 24(3):299–318, doi:10.1080/07366290600646947. Closed. Its abstract covers Am / Eu only.

**What to find**
- The table or figure giving Ln D (13 Ln, no Pm or Tm) in 70 % kerosene / 30 % 1-octanol, at 0.005 / 0.010 / 0.015 M C5-BTBP and 0.1 / 0.2 / 0.3 / 0.5 / 1.0 M HNO3, 20 °C.
- Then the Experimental section:
  - analytical method: ICP-MS or radiotracer;
  - detection limit, or the "no extraction" convention;
  - contact / shaking time (R-05-13 says 5 min was not enough in one system);
  - whether the light-Ln values are printed as numbers or as "<" values.
- In the same table: the Ce value of c950dc at 0.1 M HNO3 / 0.005 M ligand, D_raw 6.72e-4 between La 3.25e-5 and Pr 1.11e-5. That is a transcription check M27 asked for.

**Who acts**
- The user with Taylor & Francis access.
- Without access: a library or ILL request (Chalmers record research.chalmers.se/publication/25110; research.lib@chalmers.se).
- Optionally an NEA EGFRC (IDEaL) contact, for the record from which the 194 values were imported.

**Error classes it can resolve:** SOURCE-RESOLUTION (floor), NON-EQUILIBRIUM, MISATTRIBUTED, and the c950dc Ce transcription.

**Expected effect (section 4)**
- Floor confirmed: 0.452 → 0.495 (drop); interval upper bound 0.524.
- Verified real: 0.452 unchanged, and 34.9 % of the gain anchored.
- Unverifiable: reported apart, 0.486 on the rest (B1 − B0 +0.027).
- Misattribution only: no change.

### 5.2 DMDPhPDA: pub_8415538291, Shimada et al. 2004

**Documents**
- **Shimada, Yaita, Narita, Tachimori, Okuno 2004.** SEIX 22(2):147–161, doi:10.1081/SEI-120030392. Closed. Read **Table 1** (the 5 M column) and **Figure 5** (La, Ce, Pr at 5 M HNO3).
- **Open route:** A. Shimada's 2004 doctoral thesis, Shizuoka University, doi:10.14945/00003509. Record: shizuoka.repo.nii.ac.jp/records/2173; file K0379.pdf, 23.8 MB, Japanese, possibly a scan.
- **Companion paper:** SERDJ 11 (2004) 1–10, not online. Check it only for a duplicate-source risk.

**Who acts**
- With T&F access: the user, about 5–10 min (gen8 audit's estimate).
- Otherwise: one download permission for the thesis, into a quarantine folder. The agent then needs about 1–2 h, with OCR if needed.

**Expected effect**
- Figure 5 right: 0.4565 (+0.004).
- Table 1 right: 0.4478 (−0.004).
- The copy-B duplicate quarantine (V2-1) needs no document: 0.4561.
- The thesis table also lets copy A's other 69 values be checked. That anchors the 8.8 % of the gain that copy A carries.

### 5.3 ORNL imports without a data location: 54 cells, 22 DOIs, 9 of 13 clusters entirely

**Settles:** for each source_record_id, the table or figure it came from, then a blind comparison of every value.

**Who acts**
- **(a)** The SAFE / LANL curator (safedb@lanl.gov; imports by Baosen Zhang, 2026-02-10). One request covers all 69 true no-location cells, because the IDEaL tag is the same channel and the same date. It must be **sent by the user**.
- **(b)** The 6 open-access DOIs, which the agent can audit if the user permits reading OA full-text pages (no downloads):
  - d0ra05182a (gold);
  - jacsau.2c00122 (gold);
  - chem.201806443 (green, OSTI 1526392);
  - chem.201806161 (green, U Twente);
  - 01496395.2012.697523 (hybrid);
  - jas.2010.1781.1786 (bronze).
  - Their stake (publication level): 9.4 % of the SSE and 10.4 % of the gain.
- **(c)** The 16 closed DOIs, through the user's journal access: T&F, Springer, ACS, RSC, Elsevier and Pleiades.

**Expected effect**
- Verified SF-faithful: no change. 33.5 % of the gain and 9 clusters are anchored.
- Unverifiable: 0.422 on the rest at row level, and B1 − B0 shrinks to +0.006 when the imports also leave training.
- Precision fixes: about +0.006 at best.
- Component errors (F6, F7): affect cluster counts, and 1–4 % of the gain per document.

**The 22 DOIs** (publication-level shares, BPG; "mixed" = the publication also has curated cells):

| DOI | pub | import cells | B1 SSE share | gain share | access (OpenAlex) | note |
|---|---|---|---|---|---|---|
| 10.1002/chem.201806443 | a49146c8fd | 4 | 5.60 % | 0.47 % | green OA | V2-5 floor on 2 cells |
| 10.1007/s10967-020-07242-1 | 19783c039b (mixed) | 5 | 4.74 % | 3.00 % | closed | open cross-check candidate: RSC Adv 12:790 (doi 10.1039/d1ra07020g, gold) Table 1; conditions unverified |
| 10.1007/bf02349516 | 74428aeab2 (mixed) | 1 | 4.55 % | 0.10 % | closed | |
| 10.1080/07366290600762322 | 1bd76c48d9 | 6 | 4.01 % | 0.89 % | closed | |
| 10.1081/sei-120030392 | 8415538291 (mixed) | 5 | see 5.2 | | closed | DMDPhPDA copy B |
| 10.1021/acs.inorgchem.1c01729 | b8175dc3a1 | 1 | 1.86 % | 0.26 % | closed | |
| 10.1021/jacsau.2c00122 | dfa96a5118 | 3 | 1.78 % | 4.18 % | gold OA | |
| 10.1039/c6ra08946a | 3e0f6bb1b5 | 2 | 1.38 % | −2.23 % | closed | |
| 10.1080/07366299.2011.539129 | 1845fd6492 | 1 | 1.22 % | 4.72 % | closed | |
| 10.1016/j.jece.2013.06.023 | 84113c17ad | 2 | 1.06 % | −4.15 % | closed | **F6 component error** |
| 10.1039/b9nj00371a | af7a1cd919 | 1 | 1.05 % | 1.56 % | closed | ESI-MS paper; D source unclear |
| 10.1002/chem.201806161 | 0af6736cb1 (mixed) | 1 | 1.00 % | 1.53 % | green OA | D4 stereo-merge cell (V2-1) |
| 10.1080/07366290701285512 | 40c355ae86 | 2 | 0.91 % | 3.42 % | closed | F7 mixture split |
| 10.1134/s1066362214060010 | e9799b75f3 | 4 | 0.85 % | 3.72 % | closed | |
| 10.1080/07366290600761936 | 401a06b4b7 | 1 | 0.56 % | 2.39 % | closed | |
| 10.1039/d0ra05182a | 0fd6ab4e40 | 2 | 0.54 % | 2.13 % | gold OA | |
| 10.1039/c8dt03734e | 3f3b6220e0 | 4 | 0.45 % | 2.08 % | closed | |
| 10.1080/01496395.2012.697523 | ca616a18dd | 2 | 0.38 % | 2.09 % | hybrid OA | |
| 10.1080/07366290600952659 | 723ceb6c2c | 1 | 0.34 % | 1.33 % | closed | C5-BTBP in cyclohexanone |
| 10.1080/01496395.2021.1970767 | 4fb6e0c041 | 1 | 0.10 % | 0.67 % | closed | |
| 10.3923/jas.2010.1781.1786 | 1927dff570 | 4 | 0.10 % | 0.00 % | bronze OA | F7 DMSO diluent |
| 10.1134/s1066362215060077 | f6d85bdb79 | 1 | 0.00 % | 0.15 % | closed | |

### 5.4 Round-3 V2 flags: 23 cells, 8 publications

| rule | what settles it | who | document needed? |
|---|---|---|---|
| V2-1, D1 (DMDPhPDA copy B, 5 cells) | verified mantissa pattern (gen8, round 3); only the 5 M La / Ce / Pr question is open (5.2) | — / 5.2 | no, except 5.2 |
| V2-1, D4 (0ff595d576155f6c, Me2-TODGA stereo merge) | the curated stereo-specific cells of the same paper; source Chem. Eur. J. 25:5507 (doi:10.1002/chem.201806161, green OA) | agent, if OA reading is allowed | optional |
| V2-2, D2 (3 cells, pub_b9dfc468cd) | bit-identical duplicates, Table 1 (located in comments, F1) | — | no |
| V2-3, D3 (11 cells: SERDJ 18:93 (doi:10.15261/serdj.18.93, diamond OA); SEIX 22:997; SEIX 31:253; Green Chem 26:7188) | the archive's complexant / solvent fields (verified); the remedy is a cohort-key decision (split by complexant), which round 5 must decide | user / design decision | no |
| V2-5, D5 (4 cells) | a detection-limit statement: Apichaibukol, Sasaki, Morita 2004, SEIX 22(6):997–1011 (doi:10.1081/sei-200037727, Figures 2 and 5, closed); the BLPhen paper Chem. Eur. J. 25:6326 (doi:10.1002/chem.201806443, green OA, OSTI 1526392) | user (T&F) / agent (OA) | yes |

**Effect.**
- Removing every V2 cell: B1 0.4576 on the rest, B1 − B0 +0.0415.
- Training without them: β −0.0998, all-row R² 0.4528 (+0.0007).
- B1 *loses* on V2-3 and V2-5 rows (gain share −9 % each), so these flags lower the Gd term's apparent value. They do not raise it.

### 5.5 Located but high-stake documents (for completeness; lower priority)

| document | location | B1 SSE share | gain share |
|---|---|---|---|
| pub_126bbe7976 (Inorg. Chem. 2020, Table SI1) | located | 14.1 % | 11.1 % |
| pub_1605e436d7 (SEIX 22:997, V2-3/5) | located | 5.2 % | −11.3 % |
| pub_f8a621ede4 (DMDODGA, SEIX 28(4):482–494, closed) | figure 3 / 3b in comments | 0.95 % | 10.1 % |

Each is a figure-digitisation QA item (M22 gate), not a provenance gap.

---

## 6. Ranked request list for the user

Ranked by stake per minute of the user's own effort. The ranking is value-informed and is labelled as such. User minutes are estimates.

| rank | request | user effort | agent effort after | stake (BPG SSE / gain) | P(resolves) | best / worst effect on B1 (sensitivity) |
|---|---|---|---|---|---|---|
| 1 | **Say whether you have institutional access** to Taylor & Francis (SEIX), Springer (JRNC), ACS, RSC, Elsevier and Pleiades | 1 min | — | gates 2, 3b, 6 | — | — |
| 2 | **Allow one download**: Shimada 2004 thesis K0379.pdf (23.8 MB, shizuoka.repo.nii.ac.jp/records/2173, doi:10.14945/00003509), into a quarantine folder, read with OCR if needed | 1 min | 1–2 h | 4.4 % / 17.6 % (contested row 1.8 % SSE) | ≈ 0.6 (thesis may not hold the table; scan quality) | 0.4565 / 0.4478; copy A anchored or not |
| 3 | **C5-BTBP, with access**: open Foreman 2005 (SEIX 23:645) and Nilsson 2006 (SEIX 24:299); report which one holds the 70/30 kerosene/octanol Ln table or figure (number and page), the method, the detection limit, the contact time, and the c950dc Ce value | 30–45 min | 15 min | **23.1 % / 34.9 %** | ≈ 0.8 with access | 0.495 (floor) / 0.452 (real); unverifiable 0.486 on the rest |
| 3b | DMDPhPDA with access instead of rank 2: SEIX 22:147, Table 1 (5 M column) and Figure 5 | 5–10 min | 5 min | as rank 2 | ≈ 0.95 | as rank 2 |
| 4 | **Allow reading OA full-text web pages** (no downloads) for the 6 open import DOIs plus 2 open V2 sources | 1 min | 2–3 h | ≈ 9.4 % / 10.4 % (+ D4, D5 on the BLPhen cells) | ≈ 0.8 per document | per document; census rules in section 7 |
| 5 | **Send one e-mail** (agent drafts, user sends) to the SAFE / LANL curator (safedb@lanl.gov), asking for the source table or figure of each imported source_record_id: 54 "updated ORNL" + 15 "IDEaL" cells, bulk-imported 2026-02-10 | 15 min | 1 h per reply | **56.4 % / 68.4 %** | ≈ 0.3 (unknown response) | anchors or strands the Gd term (section 4b: −0.0045 vs +0.024 on located rows) |
| 6 | Closed import DOIs with access (16, DMDPhPDA covered by ranks 2 / 3b), in the **seeded census order** of section 7 | ≈ 15 min each, ≈ 4 h | 15 min each | 23.1 % / 17.9 % (publication level, the 15 besides DMDPhPDA; mixed publications include their curated cells) | ≈ 0.7 | +0.006 (precision) … ≤ +0.035 (gross-error ceiling) |
| 7 | Laboratory / database contacts (optional, slow): Chalmers nuclear chemistry (C5-BTBP raw data, detection limit); NEA EGFRC IDEaL (C5-BTBP source record); JAEA (DMDPhPDA) | 20 min each | — | as ranks 2–3 | low, weeks | as ranks 2–3 |
| — | **No user action**: F1 reclassification of the 10 comment-located cells; V2-1 (D1), V2-2, V2-3 are settled by archive evidence or a design decision | 0 | minutes | 1.2 % / 11.6 % (F1) | 1 | class labels only |

**Minimum useful package: ranks 1 + 2 + 3, about 35–50 minutes of the user's time.**
- It covers about 28 % of B1's SSE and about 52 % of its gain (C5-BTBP + DMDPhPDA, which do not overlap).
- That matches E3's "three documents" estimate.

---

## 7. Label-free census order for the 22 import DOIs (fixed before any value is read)

**Order.** numpy `default_rng(20261023).permutation` over the sorted DOI list. SHA-256 of the newline-joined order: `0060ef804f78c13cbf742b1c2ab9f6a461073401d4fa84dd8dcddb67dc50771d`.

1. 10.1080/01496395.2012.697523
2. 10.1039/d0ra05182a
3. 10.1134/s1066362215060077
4. 10.1081/sei-120030392
5. 10.1002/chem.201806443
6. 10.1007/s10967-020-07242-1
7. 10.1080/07366290600761936
8. 10.1016/j.jece.2013.06.023
9. 10.3923/jas.2010.1781.1786
10. 10.1039/c8dt03734e
11. 10.1080/07366299.2011.539129
12. 10.1021/acs.inorgchem.1c01729
13. 10.1080/07366290600952659
14. 10.1080/07366290701285512
15. 10.1039/b9nj00371a
16. 10.1134/s1066362214060010
17. 10.1007/bf02349516
18. 10.1080/01496395.2021.1970767
19. 10.1021/jacsau.2c00122
20. 10.1080/07366290600762322
21. 10.1002/chem.201806161
22. 10.1039/c6ra08946a

**Census rules** (from M25, made concrete here)
- Auditors see the DOI, the conditions and the metal list only, never V1 values or residuals. A script does the comparison.
- The publication is the unit.
- Error classes, in target currency:
  - **gross:** |Δ adjacent log SF| > 0.3;
  - **whole-cell level offset** (cancels in log SF; reported apart);
  - **SOURCE-RESOLUTION;**
  - **MODEL-DERIVED / GRID;**
  - **NON-EQUILIBRIUM;**
  - **MISATTRIBUTED;**
  - **COMPONENT** (new, F6 / F7);
  - **UNLOCATABLE.**
- Corrections from this random census go to V3-R. Value-informed corrections (ranks 2–3) go to V3-HS, an upper bound only.
- Scoring uses the frozen V1 read sets.

---

## 8. What would change these conclusions

- **A source states a light-Ln limit well below 10⁻⁴**, or prints the La–Gd values as measured numbers with error bars. Then the floor scenarios are void, C5-BTBP's light-end error (16.4 % of SSE) is genuine model error, and B1 stays at 0.452.
- **A source shows the C5-BTBP cells were not at equilibrium** (short contact time). Then the whole publication, not only the light end, becomes an interval or unverifiable stratum (0.486 on the rest).
- **Table 1 as printed is confirmed at 5 M.** Then copy A, not copy B, carries the error, and the reference moves to 0.4478.
- **The SAFE curator returns locations and a blind check finds the imports SF-faithful** (gross-error rate upper bound < 2 % of SSE). Then import provenance is retired as a lever, and B1's Gd transfer is anchored on 9 more clusters.
  - If COMPONENT errors like F6 recur, the independent-cluster count drops below 13, and every few-cluster inference (M34) must be redone.
- **The training-side result (section 4b)** comes from 25 fold-fits per design, with inner-λ selection on small chemotype blocks. β is unstable (0 … −0.23) once the 69 cells leave.
  - A different λ grid, or a fixed β, could soften the "B1 loses on located rows" finding.
  - It is a statement about what the corpus can teach, not about chemistry.
- **The floor cut (−4.0)** comes from a statement for a different diluent and method (SKB R-05-13). At −3.5 the drop scenario gives 0.490 instead of 0.495, so the conclusion does not hinge on the exact cut.

---

## 9. Caveats

- **Exploratory.** Every share, scenario and refit is on already-seen cells and seeds. No interval is attached to the shares. The seed ranges are dispersion over 5 fold plans, not inference.
- **Row-level scenarios hold reads and predictions fixed.**
  - In a real V3, censored *read* metals would also change the predictions.
  - The interval-scored R² keeps the V1 y in SST and is an upper bound.
- **The DMDPhPDA swap scenarios** replace whole duplicate cells. That is valid only because the two copies share fold, read set and metals in all 5 seeds (asserted).
- **The training-side refit** re-balances chemotype weights. It keeps the inner DOPT_Q covariances and the OOF read sets.
- **The rounding bound** assumes uniform rounding at the last reported digit. ORNL D_raw strings carry float noise from a log → D conversion and were read at 8 significant figures. Replicate-averaged cells use var / n.
- **Bibliographic records.**
  - Access status is from OpenAlex on 2026-10-23, not checked at the publishers.
  - IDEaL's identity comes from a search result; the NEA page returned 403.
  - The C5-BTBP attribution and the SKB statements are from round 4's verified desk research (`round4_20261022/results/source_recovery.json`). No full text was read in this probe.
- **Facts F6 and F7** come from archive fields and titles. They do not come from the papers.
