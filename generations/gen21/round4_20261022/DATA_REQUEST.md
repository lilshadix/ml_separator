# Data request (round 4; brief §10)

Software cannot produce any of the following. Each item says what it would change.

## 1. Source recovery (desk work; needs access the session does not have)

1. **Shimada 2004 PhD thesis** (Shizuoka University, DOI 10.14945/00003509; open PDF of about 24 MB, in Japanese).
   - **Settles:** the DMDPhPDA 5 M HNO3 Pr/Nd value. Copy A and copy B differ by exactly one decade.
   - **Needs:** the user's permission to download it, then reading or OCR of the lanthanide D tables.
   - **Changes:** whether a V3 corrects or keeps copy B (V1 is never edited).
2. **C5-BTBP cells (pub_6a0ef8528c, 15 cells, 148 C3 pairs).**
   - **Settles:** whether these cells come from Nilsson et al. 2006 (10.1080/07366290600646947) or Foreman et al. 2005 (10.1080/07366290500230735). Both are closed access, and their abstracts fit Foreman better.
   - **Also needed:** the stated detection limit, or lowest reported D, for the lanthanide measurements.
   - **Needs:** journal access (Experimental section and the tables).
   - **Changes:** possibly the publication / copy-group assignment, and a censoring treatment if a limit is stated (G-V3).
3. **ORNL-imported cells** (54 C3 cells, 529 pairs, no data location): the table or figure each came from.
4. **SAFE inventory** (safe.lanl.gov needs the user's sign-up / login).
   - **Gives:** the lanthanide extraction records with DOI, table / figure and conditions, which are needed to count records that are new to this corpus.
   - **Note:** the homepage total (18,388 records) is not a count of eligible rows.

## 2. Independent confirmation data (the binding limit; see POWER.md)

**Why.**
- The corpus has about 13 source clusters, with Kish-effective count 2.2.
- Even B1's replicated +0.034 R² gain over interpolation fails the small-cluster interval on it.
- Under the realistic gain structure, 80% power for the sealed confirmation rule needs **about 40 new independent sources** (chemotype-like units with full or near-full La–Lu series). With 20 sources, power is 33–50%.

**What, in order of value.**
1. **Literature not in the corpus, curated only after a candidate is frozen.** Verified open-access candidates with ≥ 10 lanthanides (`results/source_recovery.json`):
   - phenanthroline-diamide and BLPhen N-donor series (Molecules 29:3548, 2024; IJMS 24:5569, 2023; IJMS 23:15538, 2022; RSC Adv 9:26537, 2019);
   - an isomeric-DGA paper with a numeric 14-Ln table;
   - a phosphonium-nitrate paper;
   - an HDEHP / ionic-liquid paper.

   That is fewer than 10 sources, so not sufficient alone.
   - **Protocol:** freeze the candidate and its read policy first. Then an analyst extracts the values, gives the model only the four DOPT_Q-selected metals per cell, and scores once.
2. **New laboratory series** (as in round 3's EXPERIMENTAL_DATA_REQUEST.md, unchanged):
   - **Scope:** about 6 extractant systems across donor classes, at ≥ 2–3 conditions each.
   - **Measurement:** the full La–Lu series in independent runs (fresh phases, different days, ≥ 3 repeats). Per-metal detection limits, analytical duplicates and the complete condition record.
   - **Who:** a laboratory not among the corpus's main publications.
   - **What it gives:** the noise ceiling (whole-curve repeatability) and a few independent sources. It is not enough for 80% power by itself.

## 3. What is not needed

- More rows from the same publications. The archive's "extra" lanthanide series all come from cohort publications or extractants (PROVENANCE.md).
- Duplicated transcriptions, or two readings of one sample: they do not estimate laboratory repeatability.
