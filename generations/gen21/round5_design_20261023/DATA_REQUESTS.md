# Round-5 data requests to the user (gen21, strict-adjacent log SF)

Written 2026-10-24 for the round-5 design. This is the user-facing list of everything round 5 cannot do in software.

> **Status: exploratory design. Nothing here is sealed, nothing is committed.**
> - Every in-corpus number below was computed on the 229 C3 cells and the 5 dev seeds that rounds 1–4 and the round-5
>   probes have already seen. All of them are EXPLORATORY.
> - "Effect on B1" means a **sensitivity of a data version (V3)**. V1 is never edited. Where a scenario scores a different
>   row set, the row count is given, and the number is not comparable with the E-A reference (B1 = 0.452).
> - Where a skeptic check in `probes/PROBE_CHECKS.md` corrects a probe, the corrected number is used here.

---

## Why these requests exist

- **Stage 1 of round 5 needs nothing from you; Stage 0 can use two quick answers.** Stages 0 and 1 are software:
  sealing conventions, the E-CT estimand and descriptive deliverables (`README.md`, plan summary). Two Stage-0 inputs
  depend on you: the planning amplitudes from primary SF tables (U5 item 4, web reading) and the review of the
  laboratory key (U6). Without those answers Stage 0 seals them as PROVISIONAL (the key applied mechanically with
  conservative and liberal counts; power for both an HDEHP-like and a P507-like centre) and fixes them by an append-only
  amendment before any new value is read (`CLAUDE_CODE_PROMPT.md` §4.6). Every other item below is **Stage 2**.
- **E-A is closed to new model slots on this corpus.** No round-1 or round-2 idea opened a model route to 0.50
  (IDEAS_RANKED "Bottom line" 1; IDEAS2_RANKED "Bottom line" 1). 0.50 at four read metals (E-A) is not reached by any
  tested or proposed model, and gains below about 0.01 R² are undetectable on this corpus (EVIDENCE_BASE §3). On E-A
  itself only new or cleaner data can move the level, and those results are reported as V3 or U4 sensitivities. More
  reads are a different read budget (E-D), never E-A. These requests are the data side.
- **What is at stake, provenance.** The 69 C3 cells with no source location (54 ORNL imports and 15 C5-BTBP cells) hold
  56.4 % of B1's SSE and 68.4 % of its gain over B0 (BPG; P5 §3, reproduced in PROBE_CHECKS "P5").
  - B1's Gd transfer to the located documents is learned mainly from those 69 cells under the current inner-CV ridge.
    Refit without them, B1 − B0 on the 160 located cells shrinks to about 0 (−0.0045, 3 of 5 seeds negative), or to
    +0.012 with a fixed λ = 0.03. The training loss (−0.0285) lies outside 40 random-removal draws (PROBE_CHECKS "P5").
  - So source recovery decides whether B1's gain rests on verifiable values. No model change answers that.
- **What is at stake, supply.** Any confirmation needs independent laboratories. The literature desk count gives 5–6
  verified new laboratories and about 7.0–11.5 expected after the check's corrections (13.5 under P4's own tier rule; P4
  §0, PROBE_CHECKS "P4"). Each confirmation claim has
  a minimum number of laboratories (confirm-1, confirm-2, confirm-4 in IDEAS2_RANKED). The supply requests decide which
  claims can reach theirs.

## How the ranking works

- **Order: value per minute of your effort.**
- **Provenance requests** are scored as in PROBE_CHECKS "P5": share of B1's gain the item could settle × P(it resolves)
  ÷ your minutes, in gain-% per minute. The shares are label-based, so this order is **for asking only**. It is never a
  rule for choosing scoring rows (P5 header).
- **Supply requests** have no SSE stake. They are placed by the decision they unlock. That placement is a judgement and
  is labelled as one.
- Your minutes are the probes' estimates. Where the sources give none, the entry says "not estimated".

---

## Summary table

| rank | id | what you do | your time | agent time after | what it settles (stake) | P(resolves) | basis |
|---|---|---|---|---|---|---|---|
| 1 | U1-a | say which publishers you can read through an institution | 1 min | — | gates U1-e, U1-g, U1-h and closed literature candidates | — | gate (P5 §6 rank 1) |
| 2 | U1-b | allow one download: the Shimada 2004 thesis | 1 min | 1–2 h | DMDPhPDA: 4.4 % SSE / 17.6 % gain; contested row 1.8 % SSE | ≈ 0.6 | ≈ 10.6 gain-%/min |
| 3 | U1-c | allow web reading (no downloads) of 6 open-access import papers | 1 min | 2–3 h | 9.4 % SSE / 10.4 % gain | ≈ 0.8 per document | ≈ 8.3 gain-%/min |
| 4 | U5 | extend the same web-reading permission to the literature-supply work; optionally review the candidate list | 1 min (+ optional review, not estimated) | 1–2 days (M22 Phase A); 2–3 days (targeted search) | supply-audit kill: which claims reach their minimum number of laboratories | — | decision unlocked (judgement) |
| 5 | U1-d | allow one download: the JACS Au 2022 SI spreadsheet | 1 min | not estimated | possible route to the cited source of all 54 ORNL cells (33.3 % SSE / 33.5 % gain); match unverified | not estimated | cheaper than U1-f (PROBE_CHECKS "P5"); unscored |
| 6 | U2 | export SAFE metadata for the archive's missing record IDs | minutes (E3 budgeted about 1 h for a broader export) | hours | SAFE supply gate (M33) | ≈ 0.9 a useful go/no-go | decision unlocked (judgement) |
| 7 | U3 | allow fetching the SI of Molecules 29:3548 | minutes | extraction + sealing | first out-of-corpus-data E-CT case: a corpus-linked laboratory, so a dry run of the sealed pipeline (descriptive) | — | decision unlocked (judgement) |
| 8 | U6 | review the ambiguous links in a draft laboratory key | not estimated (short; table prepared for you) | hours | laboratory counts G for every laboratory-unit claim | — | decision unlocked (judgement) |
| 9 | U1-e | with Taylor & Francis access: read DMDPhPDA Table 1 and Figure 5 (replaces U1-b) | 5–10 min | 5 min | as U1-b | ≈ 0.95 | as U1-b, at higher P |
| 10 | S1 | sign off the cohort-key remedy for the 11 V2-3 cells | not estimated (one decision) | minutes | 6.4 % SSE, gain share −9.0 % | — | design sign-off |
| 11 | U1-f | send one e-mail (drafted for you) to the SAFE curator | 15 min | ≈ 1 h per reply | all 69 unlocated cells: 56.4 % SSE / 68.4 % gain | ≈ 0.3 | ≈ 1.4 gain-%/min |
| 12 | U1-g | with access: read the two C5-BTBP papers | 30–45 min | 15 min | 23.1 % SSE / 34.9 % gain (largest single document) | ≈ 0.8 with access | ≈ 0.7 gain-%/min |
| 13 | U1-h | with access: the 15 other closed import papers, in a fixed seeded order | ≈ 15 min each, ≈ 4 h | 15 min each | 23.1 % SSE / 17.9 % gain (publication level) | ≈ 0.7 | per document |
| 14 | U1-i | optional contacts: Chalmers, NEA EGFRC (IDEaL), JAEA | 20 min each | — | as U1-b and U1-g | low; weeks | optional |
| 15 | U4 | a laboratory collaborator for a label-blind re-measurement | months; external | — | laboratory repeatability (gap G1); M11 / M12 | — | highest value if it ever runs; not per-minute |
| — | H1 | housekeeping: optionally delete PDFs that tools auto-cached outside the repository | not estimated (very short) | — | none (hygiene) | — | — |

Sources for the table: P5 §6 with the corrections in PROBE_CHECKS "P5"; IDEAS_RANKED "Consolidated requests to the user"
and §§ M22, M25, M33, M11; IDEAS2_RANKED confirm-1, confirm-4; P4 §0, §7.

## Packages

| package | requests | your time | coverage (EXPLORATORY) | source |
|---|---|---|---|---|
| **Three-minute package** (cheapest by the probes' own rule) | U1-a + U1-b + U1-c | about 3 min | about 14 % of B1's SSE and 28 % of its gain. This counts DMDPhPDA in full, including copy B's 8.8 % of the gain, which the V2-1 quarantine settles without a document. The document-dependent part is about 12 % SSE (copy A 1.2 + contested row 1.8 + open-access papers 9.4) and about 19 % gain (copy A 8.8 + open-access papers 10.4) (P5 §3) | PROBE_CHECKS "P5"; P5 §3 |
| **Highest-stake document package** | U1-a + U1-b (or U1-e) + U1-g | about 35–50 min | document-dependent stake about 26 % SSE / 43.7 % gain; about 20 % / 33 % weighted by P(resolves) | PROBE_CHECKS "P5" |
| same, without Taylor & Francis access | U1-a + U1-b | about 2 min | the thesis only: 4.4 % / 17.6 % | PROBE_CHECKS "P5" |
| **Supply package** | U5 + U2 + U6 (+ U3) | minutes | decides whether any confirmation claim can be decided at all | confirm-1 |

**Correction to the earlier headline.** P5 and EVIDENCE_BASE §6.3 quoted "about 28 % of SSE and about 52 % of the gain"
for the document package. That figure double-counts DMDPhPDA copy B (8.8 % of the gain), which the V2-1 duplicate
quarantine settles without any document. The corrected figures above replace it (PROBE_CHECKS "P5").

---

## Requests in detail

Entries follow the request IDs (U1–U6, S1, H1); the rank of each is in its heading and in the summary table. "D1"
elsewhere in the project (for example "V2-1 (D1)") is a round-3 data-quality flag, not a request.

### U1: source recovery (M25; P5 and its check)

U1 is a package of nine asks. They are listed in rank order below. All of them feed V3, which is reported only as
sensitivities next to V1.

#### U1-a (rank 1): which publishers can you read?

- **What to do.** Tell us whether you have institutional access to:
  - Taylor & Francis (Solvent Extr. Ion Exch., Sep. Sci. Technol.);
  - Springer (J. Radioanal. Nucl. Chem.);
  - ACS, RSC, Elsevier;
  - Pleiades (Radiochemistry).
- **Effort.** 1 minute.
- **What it decides.** Whether U1-e, U1-g and U1-h can run, and whether closed literature candidates can be verified
  (20 of the 32 "probable" candidates are closed; P4 §7).
- **If you decline.** We assume no access. Closed items go to library or interlibrary-loan routes, or into an
  "unverifiable" stratum. Without Taylor & Francis access the document package is the thesis only (4.4 % SSE / 17.6 %
  gain; PROBE_CHECKS "P5").
- **Terms.** The agent never uses your credentials. You read the papers. Report the table or figure location and the
  specific values asked for, not copies of the article.

#### U1-b (rank 2): allow one download, the Shimada 2004 thesis

- **What to do.** Permit one download into a new, empty quarantine folder:
  - A. Shimada, 2004 doctoral thesis, Shizuoka University, doi:10.14945/00003509;
  - record shizuoka.repo.nii.ac.jp/records/2173, file K0379.pdf, 23.8 MB, in Japanese, possibly a scan (P5 §5.2,
    confirmed in PROBE_CHECKS "P5").
- **Effort.** 1 minute. Agent: 1–2 hours, with OCR if needed.
- **What the agent looks for.** The DMDPhPDA D values for La, Ce and Pr at 5 M HNO3: does the thesis agree with Table 1
  as printed, or with Figure 5? It also checks copy A's other 69 values.
- **Stake.** DMDPhPDA (pub_8415538291) holds 4.4 % of B1's SSE and 17.6 % of its gain.
  - Only one strict-adjacent row is contested: the 5 M Pr/Nd row, 1.8 % of the SSE.
  - Half of the gain share sits on copy B (8.8 %). The V2-1 duplicate quarantine settles copy B without any document.
- **P(resolves) ≈ 0.6.** The thesis may not hold the table, and scan quality is unknown.
- **What it decides** (sensitivities on the identical 1,802 queries; P5 §4a):
  - Figure 5 right: B1 reads 0.4565 in V3;
  - Table 1 as printed right: 0.4478;
  - either way, copy A's 8.8 % of the gain is anchored or not.
- **If you decline.** Both readings are reported as a ±0.004 sensitivity. The copy-B quarantine (0.4561) still runs. If
  you have Taylor & Francis access, U1-e does the same job.
- **Terms.** The repository's terms of use were not checked in these sources; please glance at the record page. The file
  is treated as untrusted input and read by scripts from the quarantine folder.

#### U1-c (rank 3): allow web reading of six open-access import papers

- **What to do.** Permit the agent to read open-access full-text **web pages** (no downloads, no PDF links) for the six
  open DOIs among the ORNL imports (P5 §5.3):

  | DOI | access | also settles |
  |---|---|---|
  | 10.1039/d0ra05182a | gold | |
  | 10.1021/jacsau.2c00122 | gold | |
  | 10.1002/chem.201806443 | green (OSTI 1526392) | round-3 flag V2-5: the BLPhen floor statement |
  | 10.1002/chem.201806161 | green (U Twente) | round-3 flag V2-1, D4: the Me2-TODGA stereo merge |
  | 10.1080/01496395.2012.697523 | hybrid | |
  | 10.3923/jas.2010.1781.1786 | bronze | the DMSO-diluent suspect (P5 F7) |

  - The "2 open V2 sources" of P5 §6 are two of these six (P5 §5.4).
  - Optional, same permission: RSC Adv 12:790 (doi:10.1039/d1ra07020g, gold) as an open cross-check for
    pub_19783c039b, conditions unverified (P5 §5.3). Also the open SKB reports R-04-22, R-05-13 and R-06-45 from the
    C5-BTBP group, if readable without a download (IDEAS_RANKED §M25).
- **Effort.** 1 minute. Agent: 2–3 hours.
- **Stake.** 9.4 % of B1's SSE and 10.4 % of its gain (publication level). P(resolves) ≈ 0.8 per document. About 8.3
  gain-% per minute (PROBE_CHECKS "P5").
- **What it decides.** For each paper: the table or figure behind each cell, then a blind value check in the census (see
  "How values will be handled"). It also settles the two V2 items above.
- **If you decline.** These six stay in the "unverifiable" class.
- **Terms.** The agent respects publisher blocks (403s) and does not work around them. It requests HTML pages, never PDF
  links: earlier passes had PDFs auto-cached by the fetch tool (P4 §8). OpenAlex pages ask agents to register an API key
  with your e-mail address. That notice is ignored; nothing of yours is sent (P4 §8, P5 §1).

#### U1-d (rank 5): allow one download, the JACS Au 2022 SI spreadsheet

- **What to do.** Permit one download into a quarantine folder:
  - the SI of Liu, Johnson, Jansone-Popova & Jiang, JACS Au 2022 (doi:10.1021/jacsau.2c00122, gold OA);
  - file au2c00122_si_002.xlsx, about 9.4 MB, per the PMC9241157 page (PROBE_CHECKS "P5").
- **Why.** The archive's "updated ORNL" channel looks like the ORNL / UC Riverside compilation behind this paper:
  - the channel: 714 rows, 33 DOIs, all lanthanides, all single-extractant, one curator, one date (2026-02-10), log D in
    exactly [−4.0, +3.18];
  - the paper: 1,202 single neutral-extractant Ln log D values in −4 … +4, and its SI lists the literature source of each
    point.
  - **The match is not verified.** It needs this download.
- **Effort.** 1 minute. Agent time not estimated (script work).
- **Stake and limits.** The 54 ORNL cells hold 33.3 % of B1's SSE and 33.5 % of its gain (P5 §3). But the SI gives the
  cited source and the original log D, not the table or figure location, and it is not independent of the import.
  PROBE_CHECKS "P5" judges it cheaper than the curator e-mail (U1-f) and able to reorder P5's ranks 5 and 6. It was not
  scored in gain-% per minute.
- **What it decides.** First, whether the match holds; if not, the route is void. If it holds: a per-cell cited source to
  steer the census (U1-h), a transcription check between SI and archive, and the −4.0 clip (the ORNL channel's minimum is
  exactly log D −4.0, so floor questions belong in the ORNL census too).
- **If you decline.** ORNL locations come only through U1-f and U1-h.
- **Terms.** The SI's licence was not checked in these sources. The file contains log D values. It is read by script
  only and kept from modellers until the V3 rules are frozen.

#### U1-e (rank 9): with Taylor & Francis access, the DMDPhPDA paper

- **What to do.** Open Shimada, Yaita, Narita, Tachimori, Okuno 2004, Solvent Extr. Ion Exch. 22(2):147–161
  (doi:10.1081/SEI-120030392). Report Table 1 (the 5 M HNO3 column) and Figure 5 for La, Ce and Pr at 5 M HNO3, and
  whether the two agree.
- **Effort.** 5–10 minutes. Agent: 5 minutes. P(resolves) ≈ 0.95 (P5 §6, rank 3b).
- **What it decides.** The same as U1-b. With access, it replaces U1-b. The companion paper SERDJ 11 (2004) 1–10 is not
  online and matters only for a duplicate-source check (P5 §5.2).
- **If you decline.** U1-b.

#### U1-f (rank 11): one e-mail to the SAFE curator

- **What to do.** The agent drafts one e-mail; you edit it and send it to the SAFE / LANL curator (safedb@lanl.gov). It
  asks for the source table or figure of each imported source_record_id:
  - the 54 "updated ORNL" cells;
  - the 15 "IDEaL" cells (C5-BTBP; IDEaL is the NEA EGFRC's International Database on Extractant Ligands);
  - both were bulk-imported on 2026-02-10 by curator Baosen Zhang (P5 F2, §5.3).

  The same e-mail can ask about SAFE's terms of use for U2.
- **Effort.** 15 minutes. Agent: about 1 hour per reply.
- **Stake.** All 69 unlocated cells: 56.4 % of B1's SSE and 68.4 % of its gain. P(resolves) ≈ 0.3 (the response is
  unknown). About 1.4 gain-% per minute (PROBE_CHECKS "P5").
- **What it decides.** Whether B1's Gd term is anchored. M25's rule: "anchored" if the V3 gain on the resolved documents
  is at least 50 % of their V1 gain (IDEAS_RANKED §M25).
- **If you decline.** The 69 cells stay unlocated unless U1-d or the census locates them. The round-5 report then states
  that B1's Gd transfer to the located documents is learned mainly from unlocated cells under the current inner-CV
  ridge (PROBE_CHECKS "P5").
- **Terms.** You send it; the agent never sends mail. The draft contains record IDs and the request only, no values.

#### U1-g (rank 12): with access, the two C5-BTBP papers

- **What to do.** Open both closed papers:
  - Foreman, Hudson, Geist, Madic, Weigl 2005, Solvent Extr. Ion Exch. 23(5):645–662 (doi:10.1080/07366290500230735);
  - Nilsson et al. 2006, Solvent Extr. Ion Exch. 24(3):299–318 (doi:10.1080/07366290600646947).

  Report:
  - which paper holds the table or figure of Ln D (13 Ln, no Pm or Tm) in 70 % kerosene / 30 % 1-octanol, at 0.005,
    0.010 and 0.015 M C5-BTBP and 0.1, 0.2, 0.3, 0.5 and 1.0 M HNO3, 20 °C (number and page);
  - the analytical method (ICP-MS or radiotracer);
  - the detection limit, or the "no extraction" convention;
  - the contact or shaking time;
  - whether the light-lanthanide values are printed as numbers or as "<" values;
  - the printed Ce value at 0.1 M HNO3 / 0.005 M ligand. This is a transcription check of cell c950dc2982e96b83. You will
    not be shown the archive value, so the check stays blind.

  Without access: a library or interlibrary-loan request (Chalmers record research.chalmers.se/publication/25110;
  research.lib@chalmers.se) (P5 §5.1).
- **Effort.** 30–45 minutes. Agent: 15 minutes. P(resolves) ≈ 0.8 with access. About 0.7 gain-% per minute (PROBE_CHECKS
  "P5").
- **Stake.** The largest single document: 23.1 % of B1's SSE and 34.9 % of its gain. Cell c950dc2982e96b83 alone holds
  11.8 % of the SSE. All 44 C3 readings below log D −4 are C5-BTBP readings, in 14 of its 15 cells (P5 §3, F4).
- **What it decides** (sensitivities; P5 §4a, §5.1):

  | outcome | consequence (EXPLORATORY) |
  |---|---|
  | a detection limit is stated (light lanthanides at a floor) | rows touching the censored readings leave scoring: B1 0.495 on 1,760 rows per seed, B1 − B0 +0.035; interval-scored upper bound 0.524. A different row set, a data version, not an E-A result |
  | the values are real measured numbers | nothing changes; 34.9 % of the gain is anchored |
  | not at equilibrium (short contact time) | the whole publication becomes an interval or "unverifiable" stratum: B1 0.486 on the remaining 1,654 rows (different row set) |
  | attribution only (Nilsson vs Foreman) | no out-of-fold number changes (folds are set by chemotype); only the independence ledger |
  | a per-source limit is recovered | M24 may reopen: a floor-aware predictor must gain ≥ +0.005 R² in ≥ 4 of 5 seeds on rows with both labels above the limit (IDEAS_RANKED §M24) |

  - C5-BTBP also decides the like-for-like k = 4 row: it is the only one of 32 publications whose removal moves C4
    across 0.50 (0.480 to 0.517). For the C3 population row, 5 of 39 single-publication removals cross 0.50 (BPG; 9 of
    39 under BP), so C3 simply sits on 0.50 (PROBE_CHECKS "P3"). These are sensitivities of a different read budget,
    not of E-A.
- **If you decline.** Every round-5 number is reported with and without pub_6a0ef8528c (IDEAS_RANKED rule 1). The floor
  scenarios stay value-derived. M24 stays closed.
- **Terms.** As U1-a.

#### U1-h (rank 13): with access, the 15 other closed import papers

- **What to do.** Work through the closed import DOIs in the **label-free seeded census order** below (P5 §7: numpy
  `default_rng(20261023).permutation` over the sorted DOI list; SHA-256 of the order
  `0060ef804f78c13cbf742b1c2ab9f6a461073401d4fa84dd8dcddb67dc50771d`). For each paper, report the table or figure that
  holds the values and transcribe them at the conditions you are given. You see the DOI, the conditions and the metal
  list only; a script does the comparison.

  | order | DOI | who reads it |
  |---|---|---|
  | 1 | 10.1080/01496395.2012.697523 | agent (U1-c) |
  | 2 | 10.1039/d0ra05182a | agent (U1-c) |
  | 3 | 10.1134/s1066362215060077 | you (Pleiades) |
  | 4 | 10.1081/sei-120030392 | DMDPhPDA (U1-b / U1-e) |
  | 5 | 10.1002/chem.201806443 | agent (U1-c) |
  | 6 | 10.1007/s10967-020-07242-1 | you (Springer); this is pub_19783c039b, see below |
  | 7 | 10.1080/07366290600761936 | you (T&F) |
  | 8 | 10.1016/j.jece.2013.06.023 | you (Elsevier); component error F6, see below |
  | 9 | 10.3923/jas.2010.1781.1786 | agent (U1-c) |
  | 10 | 10.1039/c8dt03734e | you (RSC) |
  | 11 | 10.1080/07366299.2011.539129 | you (T&F) |
  | 12 | 10.1021/acs.inorgchem.1c01729 | you (ACS) |
  | 13 | 10.1080/07366290600952659 | you (T&F) |
  | 14 | 10.1080/07366290701285512 | you (T&F) |
  | 15 | 10.1039/b9nj00371a | you (RSC) |
  | 16 | 10.1134/s1066362214060010 | you (Pleiades) |
  | 17 | 10.1007/bf02349516 | you (Springer) |
  | 18 | 10.1080/01496395.2021.1970767 | you (T&F) |
  | 19 | 10.1021/jacsau.2c00122 | agent (U1-c) |
  | 20 | 10.1080/07366290600762322 | you (T&F) |
  | 21 | 10.1002/chem.201806161 | agent (U1-c) |
  | 22 | 10.1039/c6ra08946a | you (RSC) |

- **Effort.** About 15 minutes each, about 4 hours in all. Agent: 15 minutes each. P(resolves) ≈ 0.7.
- **Stake.** 23.1 % of B1's SSE and 17.9 % of its gain (publication level; mixed publications include their curated
  cells).
- **What it decides.**
  - V3-R corrections (random census only).
  - Precision fixes are worth about +0.006 R² at best. Treating all excess import error as transcription error gives a
    loose ceiling of ≤ +0.035 (P5 §4a).
  - Import provenance is retired as a lever if the gross-error SSE share has an upper bound below 2 % of B1's SSE
    (IDEAS_RANKED §M25).
  - If COMPONENT errors recur, the count of independent clusters falls below 13, and every few-cluster inference must be
    redone (P5 §8).
- **Two items to know about.**
  - **pub_84113c17ad (order 8) records the aqueous complexants CDTA and HEDTA as the dodecane "extractant".** The title
    names TEHDGA + HDEHP (P5 F6). It also drives the deployed-row loss of the heavy-selective Gd amplitude outside
    C5-BTBP (IDEAS2_RANKED "Reconciliation check").
  - **pub_19783c039b (order 6) carries all of P1's SH − OFF.** It also holds a suspected cross-acid copy pair (13 of 14
    log D bit-identical; PROBE_CHECKS "P1"). A second E-CT-relevant publication, pub_b0b3296ec5 (41 % of P1's SH − B1
    gain), needs a provenance check too; these sources do not record its access status (IDEAS2_RANKED "Duplicate and
    provenance candidates").
  - The F7 mixture suspect (10.1080/07366290701285512) is probably not an error: its abstract reports single-extractant
    studies as well (PROBE_CHECKS "P5").
- **If you decline.** The ORNL imports form an "unverifiable" stratum: B1 0.422 on the remaining 1,273 rows (different
  row set; P5 §4a). M25's rule applies: if open routes plus your access reach less than 40 % of the gain, the
  "unverifiable" stratum is reported at once.
- **Terms.** As U1-a.

#### U1-i (rank 14): optional contacts

- **What to do.** Optionally write to:
  - Chalmers nuclear chemistry: C5-BTBP raw data and detection limit;
  - NEA EGFRC (IDEaL): the record from which the 194 C5-BTBP values were imported (the 1,164 archive records are 194
    unique values; P5 F3);
  - JAEA: DMDPhPDA.
- **Effort.** About 20 minutes each. Low probability of an answer; weeks (P5 §6, rank 7).
- **If you decline.** Nothing changes.

### U2 (rank 6): SAFE gap-ID export, metadata only (M33)

- **What to do.** Log in to SAFE (safe.lanl.gov; the agent never logs in). Export **metadata only** for:
  - the 1,602 record IDs missing from the archive, in 11 runs: 7,553–8,716; 14,461–14,665; 1,343–1,482; 1,032–1,105;
    1,488–1,497; 1,326–1,329; singletons 16,084, 17,113, 17,428, 17,743, 18,058;
  - every ID above 18,372.

  Useful fields: record ID, record type (experimental extraction, computational, chromatography), DOI, conditions, metals
  present, authors, curator and date. **No D or log D columns.** Put the file in a new folder.
- **Effort.** Minutes (IDEAS_RANKED §M33). E3 §8 budgeted about 1 hour of login time for a broader export. Agent: hours.
- **Why.** SAFE showed 18,388 records on 2026-10-22 against 16,770 in the archive (E3 §1). But the archive's IDs run from
  1 to 18,372 with 1,602 missing, and the homepage count tracks the highest ID. So most of the "+1,618 newer records" are
  pre-existing gap IDs; at most about 16–21 are genuinely newer (IDEAS_RANKED §M33 and closure ledger).
- **What it decides** (M33's supply gate). The SAFE units are merged with the literature count (U5) into one supply
  ledger before the gate is read (IDEAS_RANKED §M33):

  | experimental multi-Ln laboratory × chemotype units from outside the corpus, SAFE + literature ledger | consequence |
  |---|---|
  | fewer than 3 from SAFE | stop the SAFE route; recommend laboratory campaigns only |
  | 3–9 on the merged ledger | descriptive only; C-GD is then a descriptive case series |
  | at least 10 on the merged ledger | freeze everything and run the one-shot unit-level Gd-dip transport test. C-GD (confirm-2) is its pre-specified stratum and is decided only when that stratum has at least 6 laboratories from at least 3 families (IDEAS2_RANKED confirm-2) |

  Expected yield: 0–3 units, so the branch most likely closes. The audit is a useful go/no-go with probability about 0.9
  (IDEAS_RANKED §M33). It also tells whether some "new" literature DOIs already sit in SAFE (P4 §7).
- **If you decline.** The SAFE route is recorded as "not audited". The ledger line "SAFE's +1,618 are mostly gap IDs"
  stays a working assumption. Supply rests on the literature alone (U5).
- **Terms.** SAFE's /terms page returned 404. Please read the terms after login, or ask safedb@lanl.gov (U1-f can carry
  the question). Export metadata only. The audit script asserts that no D column is present. Metal presence is weakly
  label-informative, so unit identity is frozen before any value is read (IDEAS_RANKED §M33).

### U3 (rank 7): allow fetching the SI of Molecules 29:3548 (M05, confirm-4)

- **What to do.** Permit the agent to fetch the supporting information of Molecules 29:3548 (2024;
  doi:10.3390/molecules29153548; gold OA): Table S3, a 4,7-Cl2-phen-2,9-diamide (Et / p-hexylphenyl) in 3 M HNO3 with 8
  diluents, 14 lanthanides (La–Lu, no Pm) (`probes/P4_candidates.csv` row P4-001; IDEAS_RANKED U3).
- **Effort.** Minutes. Agent: extraction plus a separate sealing step. M05's whole external leg is budgeted in days
  (IDEAS_RANKED §M05).
- **What it decides.** The first out-of-corpus-data E-CT case: a diluent-change family (class T2, not a T0 near-copy) scored with
  the frozen SH rule (K = 1.78) against B1 with 4 reads and the other comparators of confirm-4.
- **Limits.**
  - The laboratory (MSU, Nenajdenko group) is already in the corpus: four of its authors appear on corpus paper
    10.1039/d0ra05182a (P4-001). Its nearest C3 chemotype is sc059 at Tanimoto 0.623 (E3 §2.4).
  - So it is one corpus-linked laboratory unit: a dry run of the sealed pipeline and a descriptive case. It is never a
    confirmation; confirm-4 needs at least 7 laboratories.
  - A diluent-only family is also a weak test: in the corpus, B1 already scores 0.78–0.92 on extractant-concentration-only
    and diluent-only cells (IDEAS_RANKED §M01, step 6).
- **If you decline.** The external E-CT leg waits for the literature supply (U5). E-CT stays an in-corpus, exploratory
  estimand.
- **Terms.** P4 met 403s on MDPI publisher PDFs, so the fetch may fail. You could then download it yourself into a
  quarantine folder. The SI's licence was not checked here. The confirm-4 annex must be sealed, and the values sealed by
  a separate agent, before any modeller reads them.

### U4 (rank 15): a laboratory collaborator (M11, with the M12 arm)

- **What to do.** If you have, or can find, a laboratory collaborator: a label-blind re-measurement of C3 conditions as
  published (IDEAS_RANKED §M11):
  - 8–12 C3 cells, sampled label-blind, stratified by chemotype and by read-predicted window status, never by B1
    residual;
  - 3 independent runs plus an analytical duplicate;
  - level arms: in-window, high (+2.5 to +3) and floor (about −4), with direct organic-phase assay, spike recovery and
    per-metal LOQ;
  - optional: a reproducibility arm of at least 8 laboratories (M12; ISO 5725-2); record Y (M30).
  - optional: one fixed-condition TODGA ring-standard tube per laboratory and an in-run TODGA reference beside any
    non-DGA ligand, kept for the M12 ring test (V_x/V_w) and the M31 family-dip salvage. It is not needed for
    prediction: any second own-campaign ligand carries the same signal (IDEAS2_RANKED lab-2 closure and salvage;
    IDEAS_RANKED §M11).
- **Effort.** Months; external. IDEAS_RANKED notes that no laboratory is available now.
- **What it decides.** Gap G1, laboratory repeatability, which no estimate exists for:
  - 0.50 at k = 3 is impossible on single literature targets if the repeatability sd of an adjacent log SF is at least
    about 0.157–0.19 (EVIDENCE_BASE §1.4);
  - the noise share of B1's MSE is now only known to lie in [0.08, 0.87];
  - the expected result is repeatability σ_SF about 0.02–0.09, with corpus label error larger at the floor and light end.
  - A repeatability ceiling is never quoted as the corpus ceiling.
- **If you decline.** G1 stays open, and every "ceiling" statement stays a bound.
- **Terms.** Data ownership and publication rights are between you and the collaborator. Everything (cells, arms,
  thresholds, analysis) is frozen before data arrive.

### U5 (rank 4): web reading for the literature supply, and an optional list review

- **What to do.** Extend the U1-c permission (web reading, no downloads) to four pieces of desk work. One answer can cover
  both U1-c and U5.
  1. **M22 Phase A.** A literature count frozen before it starts (OpenAlex, Crossref, J-STAGE; inclusion ≥ 8 Ln including
     Eu, Gd and Tb; values never read), counted at DOI level into the M33 supply ledger (IDEAS_RANKED §M22).
  2. **Verification** of the open-access "probable" (E2) and "undeterminable" (U) candidates in
     `probes/P4_candidates.csv` (P4 §7).
  3. **A targeted search for E-CT-capable laboratories**: at least 3 non-T0 full-series conditions of one extractant,
     full La–Lu (confirm-1, corrected design 4).
  4. **Planning amplitudes from primary SF tables** for C-GD (Peppard for HDEHP; a handbook for P507). confirm-1 asks for
     them before the Stage-0 seal (corrected design 5). If this permission is not given by then, Stage 0 seals them as
     PENDING-U5, with power reported for both an HDEHP-like and a P507-like centre, and they are fixed by amendment
     before any new value is read (confirm-2 corrected design 3; `CLAUDE_CODE_PROMPT.md` §4.3 item 10).

  Optional: look over `probes/P4_candidates.csv` (92 candidates) and flag papers or groups you know, or closed papers you
  can open.
- **Effort.** 1 minute for the permission. The list review is optional and not estimated. Agent: Phase A a day or two;
  the targeted search 2–3 days.
- **What it decides.** confirm-1's supply-audit kill: a claim below its minimum number of laboratories is descriptive,
  and if every claim is below its minimum, no confirmation runs.

  | claim | minimum | source |
  |---|---|---|
  | C-ECT (own-series template beats B1 at a new condition) | 7 laboratories; fixed sequence SH@4 then SH@3 | confirm-4 |
  | Gd-dip transport (M33's primary test) | 10 laboratory × chemotype units on the merged SAFE + literature ledger | IDEAS_RANKED §M33 |
  | C-GD (cation-exchanger Gd chord deeper than β_frozen), the pre-specified stratum of the Gd-dip test | 6 laboratories from at least 3 extractant families inside that stratum; below the Gd-dip trigger it is a descriptive case series | confirm-2 |

  The current desk estimate (EXPLORATORY; P4 with PROBE_CHECKS "P4"):
  - verified new independent laboratories: 6 by P4's strict rule, 5 with the institution-link rule applied consistently;
  - expected after verification: 13.5 by P4's tier rule; 9.3–11.5 with the corrections; 7.0 with every correction;
  - 15 of the 16 probable new laboratories use chemotypes absent from the Ln corpus;
  - laboratories with at least 3 full-series conditions: about 4 reported by P4 (KU Leuven, Chiba, Turanov,
    Atanassova; P4 §0, IDEAS2_RANKED confirm-4); 3 after PROBE_CHECKS "P4" moves Turanov (P4-063) to E2; full La–Lu is
    verified only for KU Leuven and Chiba (Atanassova U in P4 §4; Chiba rests on an abstract and is pH-controlled, so
    T0) (P4 §4, §5);
  - heavy-selective cation-exchanger units with Eu, Gd and Tb: Chiba (P4-007) and Osaka (P4-083), both HDEHP, verified
    from abstracts only. Under the consistently applied institution-link rule Osaka is P (JAEA co-authors, not confirmed
    from metadata), and P4-083 reports extraction constants at one condition, not D (PROBE_CHECKS "P4"). Verified
    independent heavy-selective units: 1–2. The realistic G of 2–5 (IDEAS2_RANKED confirm-2) is an upper range that
    counts Osaka; Sofia 8-HQ is probable only (confirm-1).
- **If you decline.** The supply ledger stays at P4's desk estimate, which was not frozen in advance and is not a census.
  Every external leg is then a descriptive case series.
- **Terms.** As U1-c. The summariser returns metadata only (label blindness, P4 §8). Scopus and Web of Science are not
  accessible and are not needed.

### U6 (rank 8): review the ambiguous links in a draft laboratory key

- **What to do.** The agent builds a frozen laboratory key (lab_id) from public author metadata, using P4's rule repaired
  as the lab-4 skeptics asked: drop common-surname links, handle Family–Given name order, and link through a single
  principal investigator (IDEAS2_RANKED, rejected idea lab-4). It covers the corpus publications, including the E-CT
  publications (IDEAS_RANKED §M01, step 8), and the literature candidates. You review only the ambiguous links the P4
  check named:
  - CSM-Shafer: Argonne co-authors;
  - Osaka-Shinohara: co-authors who may be JAEA (not confirmed from metadata);
  - Kanazawa-Imura: co-author Okamura at JAEA;
  - the CIAC merge of three groups; UFV / Aveiro through Schaeffer; ISSP-Turanov with INEOS-Odinets;
  - Binnemans' co-authorship with the corpus Jülich laboratory (P4-026); Kostikova's with the corpus MSU group (P4-001).
- **Effort.** Not estimated in the sources; the agent prepares a one-page table. Agent: hours.
- **What it decides.**
  - The number of laboratories G for every laboratory-unit claim. The laboratory is the unit for any Gd-amplitude claim,
    because a laboratory's Gd chord is shared across its papers (IDEAS2_RANKED rule 12).
  - Applying P4's tier rule as written moves the count of probable new laboratories from 16 to 14; these links then
    move it to 13 (CSM-Shafer as P) and 11 (Osaka and Kanazawa as P) (PROBE_CHECKS "P4").
  - Whether E-CT may ever say "same laboratory" rather than "same study or campaign" (P1 §14; IDEAS_RANKED rule 8).
  - The key is a Stage-0 input. Ideally you review it before the Stage-0 seal; otherwise it is sealed as PROVISIONAL
    and the review is applied by amendment. Either way it is fixed before any new value is read.
- **If you decline.** The rule is applied mechanically: institution-linked ("P") laboratories are not counted as new
  (P4 §1), and both conservative and liberal counts are reported.
- **Terms.** Public bibliographic metadata only.

### S1 (rank 10): sign off the cohort-key remedy for the V2-3 cells

Earlier drafts called this request "D1"; it was renamed so that it cannot be confused with the round-3 flag D1
(DMDPhPDA copy B, V2-1).


- **What to decide.** Round-3 flag V2-3 (D3) marks 11 hidden-merge cells (SERDJ 18:93, SEIX 22:997, SEIX 31:253, Green
  Chem 26:7188). The archive's complexant and solvent fields are verified. The remedy is a cohort-key decision: split
  them by aqueous complexant (P5 §5.4).
- **Stake.** 6.4 % of B1's SSE, with a gain share of −9.0 %: B1 hurts there (P5 §3).
- **Effort.** One decision; not estimated. Agent: minutes.
- **If you do not decide.** The cells are reported as a separate condition-aware stratum, as a sensitivity, with V1
  unchanged (E3 §9, D-COND).

### H1: housekeeping (not ranked)

Earlier passes recorded files that a web tool cached outside the repository. None was opened. P4 did not delete its two,
because that would modify files outside its allowed area. You may delete them:
- `C:\Users\Bandai\.claude\projects\D--ml-separator-gh\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\tool-results\webfetch-1792772704768-6pndqr.pdf`
  and `C:\Users\Bandai\.claude\projects\D--ml-separator-gh\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\tool-results\webfetch-1792772703606-ljpixn.pdf`
  (two J-STAGE PDFs; P4 §8);
- a scanned PDF cached by the confirm-2 skeptic B fetch under the `.claude` project folder (IDEAS2_RANKED "Process
  note"), and one WebFetch result from E4 cached outside the repository (EVIDENCE_BASE §5.2, G10). Their file names are
  not recorded.

---

## What needs nothing from you

Settled from archive fields or by design, with no document (P5 §2, §5.4, §6):
- **F1.** Ten "curated, no location" cells have their location in the archive's `comments_raw` field. They move 11.6 % of
  the gain into the located class.
- **V2-1 (D1).** DMDPhPDA copy B is a verified mantissa copy; its quarantine needs no document.
- **V2-2 (D2).** Bit-identical duplicates, located in Table 1.
- **Duplicate scan before any sibling or donor pool** (IDEAS2_RANKED rule 15). This is software, in Stage 1.

## What useful new data look like (for U4 and any literature source)

From confirm-1 ("Practical value"), confirm-4, confirm-2 and E3 §8:
- full La–Lu series, at least Eu, Gd and Tb, measured in one ICP-MS run where possible;
- at least 3 conditions per extractant that change the SF pattern (aqueous medium, organic phase, or both). pH, [HA] or
  phase-ratio scans of a cation exchanger ("T0") are near-copies by construction and serve only as a data check;
- independent laboratories; several papers from one group on one ligand family count as one unit;
- per-metal detection limits or LOQ, nitrate and any complexant recorded; Y recorded where measured;
- for Gd claims, no aqueous-complexant media (lactate, citrate, DTPA, HEDTA, CDTA), no solvating regime of HDEHP at high
  acid, no values below the detection limit.

## How values will be handled when they arrive

1. **Freeze first.** The V3 rules (C5-BTBP floor and attribution, DMDPhPDA copy B, ORNL locations) are written before any
   recovered value is read (E3 §9, D-V3). The confirmation package and the confirm-4 annex are sealed before any new value
   is read. A separate agent seals extracted values before modellers see them (confirm-1, M22).
2. **Blind census.** Auditors see the DOI, the conditions and the metal list only. A script compares. The publication is
   the unit (P5 §7).
3. **Two versions.** V3-R holds corrections from the random census only. V3-HS holds value-informed corrections (U1-b,
   U1-e, U1-g); it is an upper bound and never a model-selection benchmark (IDEAS_RANKED §M25).
4. **Same queries.** Scoring uses the frozen V1 read sets on byte-identical queries. Censored targets enter only through
   the interval line (M23).
5. **V1 is never edited.** Every effect above is a sensitivity.
6. **Downloads** go into a new, empty quarantine folder and are read by scripts as untrusted input.
7. **Decisions are three-way.** Each has an inconclusive zone, and non-rejection never switches the deployed model
   (IDEAS2_RANKED rule 9).

## What none of these requests can do

- **They cannot reopen an E-A model route.** A new E-A model slot must first pass the retire-only pre-slot decidability
  gate: its own test-label oracle ceiling at least 2 % pooled relative SSE against B1; power of at least 50 % at its
  plausible effect; and demotion if the deciding fold has fewer than 10 training cells sharing the test rows' donor class
  or chemotype (IDEAS_RANKED §M35 item 7; `CLAUDE_CODE_PROMPT.md` §4.5). Otherwise E-A stays closed on this corpus.
- **0.50 at four read metals (E-A) is not reached by any tested or proposed model,** and gains below about 0.01 R² are
  undetectable on this corpus (EVIDENCE_BASE §3). On E-A itself only new or cleaner data can move the level, reported as
  V3 or U4 sensitivities. A fifth read is a different read budget (E-D, EXPLORATORY), reported as a paired contrast on
  E-A's queries and never as an E-A value: like-for-like C4 0.480; the C3 min(4, k_max) population row, in which 15.9 %
  of rows keep 4 reads, 0.497; without C5-BTBP 0.517 and 0.531, an outcome-dependent removal of the publication chosen
  for its SSE share that also removes about 18 % of SST, so a provenance sensitivity, not a budget effect
  (PROBE_CHECKS "P3").
- **Source recovery gives V3 sensitivities, not a new E-A number.** Even the combined scenario that nominally crosses 0.50
  scores a different row set with a value-derived floor rule, and is never quoted beside 0.452 (P5 §4a).
- **The literature cannot carry a pooled confirmation of a B1-sized gain.** Pooled tests need about 40 to more than 80
  publication-like sources (PROBE_CHECKS "P2"). The sealed R² leg needs about 38–41 publication-like units (P2 grid 39
  BPG / 38.3 BP; the P2 check's direct evaluation about 41) or about 20 cpg units (PROBE_CHECKS "P2", "P4"). The
  expected supply is about 7.0–11.5 new laboratories after the check's corrections (13.5 under P4's own tier rule).
- **A laboratory campaign measures repeatability.** Its ceiling is a bound on the corpus, not the corpus ceiling.

## Sources

- `probes/P5_source_recovery.md` §§2–7 and `probes/PROBE_CHECKS.md` "P5" (provenance stakes, routes, corrected package
  coverage and ranking).
- `probes/P4_literature_supply.md` §§0, 4, 6–8, `probes/P4_candidates.csv` and PROBE_CHECKS "P4" (literature supply).
- `probes/PROBE_CHECKS.md` "P1", "P2", "P3".
- `ideas/IDEAS_RANKED.md`: "Consolidated requests to the user"; §§ M01, M11, M22, M24, M25, M33; cross-cutting rules
  1 and 8; closure ledger.
- `ideas2/IDEAS2_RANKED.md`: confirm-1, confirm-2, confirm-4, rules 9–16, closure ledger additions, "Duplicate and
  provenance candidates", "Reconciliation check", "Process note".
- `evidence/EVIDENCE_BASE.md` §§1.4, 5.2, 6.3; `evidence/E3_data_inventory.md` §§1, 2.4, 8, 9.
