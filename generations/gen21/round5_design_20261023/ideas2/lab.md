# E-LAB: laboratory fingerprint calibration (theme report, round-5 ideas pass 2)

Written 2026-10-23.

> **Status: EXPLORATORY.** Every number here comes from the 229 C3 cells and the 5 dev seeds, which rounds 1–4 and the
> round-5 probes have already seen. K = 6.1 and K = 1.78 come from roughness correlations measured earlier on the same
> data (E6 §5). The level-match tolerance of 1.0 log was fixed in the script before any score was seen. About 25 arms
> were scored and the best one is discussed most. **No p-value below is adjusted for that, and none is a test.**
>
> E-LAB is a separate estimand: a new extractant, measured in a laboratory or campaign whose other series are known.
> **Its R² values must never be placed next to the BP/BPG 0.452.**

---

## 1. Bottom line

1. **A per-publication per-METAL offset learned from the publication's other cells explains 12.8 % of B1's SSE on all
   229 cells** (R² 0.452 → 0.522, BPG; BP 0.452 → 0.521).
   - The per-PAIR version at the same K is 13.9 % (0.528), so the two parameterisations carry the same information.
   - **About 90 % of it comes from same-extractant cells**: the same-extractant-only shape gives 0.515, +11.5 %.
   - Other extractants alone give only +1.6 % (0.461), or +3.8 % (0.473) with level matching.
2. **The other-extractant channel (E-LAB proper):** 92 CLEAN targets in 16 publications (POPX). Of these, 60 cells in 13
   publications have no same-extractant cell at all (POPX0: the "new ligand in a known lab" case).
   - Unmatched per-metal shape: +4.2 % relative SSE on POPX, with 8 publications up and 8 down. It is carried by 2 publications; without them it is **−4.2 %**.
   - **Level-matched shape** (MX_LM10: per metal, use only donor series whose log D there is within 1 log of the target's B1 level):
     - POPX: **0.444 → 0.494, +9.0 % [+0.3, +17.6]**, 10 publications up, 4 down, sign-flip p 0.037;
     - POPX0: **0.401 → 0.467, +11.1 % [+2.3, +19.8]**, 8 up, 3 down;
     - no harm where same-extractant cells exist (−0.9 %; the unmatched shape loses 15 % there);
     - BP is the same to ±0.003.
   - **It is laboratory- or campaign-specific.** The identical rule fed by other publications' cells loses −0.9 %. The BPG-legal version, with training-fold donors only, loses −4.6 % on POPX and −5.8 % on all rows (26 of 39 publications worse). Same chemotype in other publications: −0.7 %.
   - It is worth about one read: at 3 reads it scores 0.425, against B1's 0.401 at 4 reads.
   - It scales with the number of the lab's other series: 1 / 2 / 3 / all series give +3.4 / +5.9 / +7.8 / +13.1 %.
3. **It is not a Ce, light-end or floor artefact.**
   - In-window rows carry all of the unmatched gain: inside +10.9 %, outside −3.6 %.
   - Light pairs carry 0 % of the unmatched gain (30 % of the level-matched gain). Ce-only and light-only versions give +0.15 % and +0.2 %. The Sm–Lu-only version gives +4.0 %.
   - The Ce deviation is not shared across chemotypes inside a publication (pub-weighted ρ −0.07).
   - There is no Ce(IV) signature: 7 of 162 cells have a Ce excess above +0.3, and HCl media point the wrong way.
   - ICP-MS oxide-victim metals (Gd–Lu) share no more than La–Eu (0.25 vs 0.24).
   - Sharing does not grow toward low D, as a mass-balance or blank bias would.
4. **Descriptively, per-metal deviation patterns belong to the campaign, not to the ligand.** Deviations are from each cell's own quadratic, with the corpus mean removed, pooled over metals.

   | cell pair | shared fraction r |
   |---|---|
   | same publication, same extractant | 0.39 (pub-weighted 0.53, 19 of 19 > 0) |
   | same publication, homologue | 0.24 (pub-weighted 0.52, 10 of 10 > 0) |
   | same publication, other chemotype | 0.30 (pub-weighted 0.29, 9 of 15 > 0) |
   | other publication, same extractant | **0.01** (0.13 when the aqueous medium is matched; 9 pairs) |
   | other publication, other extractant | −0.03 to 0.01 |

   This survives restricting to pairs with identical metal coverage (SO 0.34–0.58; pub-weighted 0.36, 10 of 13 > 0).
5. **A reference-series check with TODGA** uses publications that report TODGA plus other extractants: 5 publications, 46 non-TODGA targets, all DGA, one source cluster.

   | arm | relative SSE vs B1 |
   |---|---|
   | own-campaign TODGA | **+5.3 %** (5 of 5 publications up) |
   | its lab-specific part only (own minus consensus TODGA) | +4.4 % [+3.2, +5.6] |
   | consensus TODGA shape | +0.6 % |
   | another laboratory's TODGA | −4.1 % |
   | a literature TODGA at the *same aqueous medium* | **0.0 %** (21 cells; own TODGA +5.7 % on the same rows) |

   A user would have to measure the reference in-house, in the same campaign (see point 6).
6. **The fingerprint belongs to the campaign (publication), not the laboratory.** Laboratories were defined as author
   lists sharing ≥ 2 surnames (§6.1).

   | relation | descriptive r |
   |---|---|
   | same publication, same / other extractant | 0.39 / 0.27 |
   | same laboratory, other publication | 0.06 / 0.05 |
   | other laboratory | 0.00 / −0.03 |

   - Predictively, a level-matched shape from the same laboratory's *other* papers gives +2.2 % [−0.9, +5.3] (129 cells, 22 publications).
   - On rows that also have own-publication donors, that is +0.1 %, against +4.4 % for the own campaign.
   - **So a reference series has to be run in the same campaign; a laboratory's historical reference is worth little.**
7. **What cannot be separated here.** Within a campaign, "analytics", "same feed and aqueous preparation" and "same run" are
   confounded. The corpus records no analytical method. That split needs the prospective design in lab-2.

---

## 2. What was run

**Scripts.** All are in `ideas2/scripts/`. Run them from `D:\ml_separator_gh` with `.venv\Scripts\python.exe -W ignore <script>`.
Each one is a single process that runs in under 30 s.

| script | what it does |
|---|---|
| `lab_inventory.py` | which publications hold several extractants |
| `lab_calib.py` | builds every arm row by row → `lab_out/lab_rows.parquet` |
| `lab_analyze.py` | tables and inference |
| `lab_calib2.py` | level-matched arms and their controls, TODGA consensus and fingerprint arms → `lab_out/lab_rows2.parquet` |
| `lab_varcomp.py` / `lab_varcomp2.py` / `lab_varcomp3.py` | descriptive sharing: by class, by medium match, with identical coverage |
| `lab_extra.py` | level bands, and the borrowed medium-matched TODGA |
| `lab_popx0.py` | the new-extractant stratum |
| `lab_ladder.py` | read ladder and donor-count subsampling |
| `lab_samelab.py` | same laboratory, other publication: author-overlap rule |

Logs are in `ideas2/lab_out/*.log`.

**Checks that passed.** B1 was recomputed from each fold's fitted Gd amplitude and the fold-cache DOPT_Q read sets. It
matches the REF parquets to 8.9e−16 on all 18,020 rows.

**Model.** Every arm is a shape-vector interpolation:

  levels = PWL_R[y_S − v_S] + v

This is exact at the 4 reads. B1 is v = u, with β at Gd and 0 elsewhere. A laboratory fingerprint uses

  v_m = (1 − w_m) u_m + w_m c_m

where:
- c_m is the mean, over donor series, of each donor's deviation from its own least-squares quadratic in −r;
- w_m = n_m / (n_m + K), with n_m the number of donors measured at metal m.

Because the shape passes through the target's reads, the part of the fingerprint that the reads already capture is not
added twice. Donors are bench cells (all 521) outside the V2-1/2/3 lists, with at least 6 metals. The target is never its
own donor.

**Inference.**
- R² is pooled within each seed, then averaged over seeds.
- "rel" is 1 − SSE_arm / SSE_B1 on identical rows.
- The unit is the publication: up/down counts, exact one-sided sign-flip, leave-one-publication-out (LOPO) jackknife t interval, and the top-3 share.
- Source clusters cannot be estimated as units: POPX has 4 clusters, and one holds 68 of the 92 cells.

---

## 3. Inventory: where the other-extractant channel exists

- **17 of the 39 C3 publications report more than one extractant.** They hold 108 C3 cells (872 of 1,802 rows per seed).
  - 64 of those cells have no same-extractant cell in their publication.
  - 69 have a same-chemotype homologue donor.
- **CLEAN targets with a usable donor series:** 92 cells in 16 publications (POPX).
  - By cluster: 68 cells in g:pub_74428aeab2, 14 in g:pub_3e0f6bb1b5, 8 in g:pub_f6d85bdb79, 2 in g:pub_84113c17ad.
- **The main multi-extractant publications:**

  | publication | content |
  |---|---|
  | pub_126bbe7976 | 15 DGAs, one condition each (a ligand screen) |
  | pub_b9dfc468cd | 11 DGAs |
  | pub_1bd76c48d9 | 6 malonamide / propanediamide variants |
  | pub_4cba27f5f3 | TODGA + 4 DGAs |
  | pub_5a68dc5665 | TODGA / TEDGA / TPrDGA / DMDPDGA at 4 diluents |
  | pub_d68ed8c8cf | TODGA / TDdDGA / DOODA |
  | pub_3f3b6220e0, pub_e9799b75f3, pub_a49146c8fd, pub_3e0f6bb1b5 | phenanthroline / pyridine diamides |

- **TODGA as an in-paper reference:** 5 C3 publications (126bbe7976, 4cba27f5f3, 5a68dc5665, b9dfc468cd, d68ed8c8cf) with 46 non-TODGA targets. Two more bench publications (51c0693f9c, bef16e560c) have no C3 targets.

---

## 4. Per-metal vs per-pair publication offsets (the explicit question), all 229 cells

All arms are leave-the-target-out. Cells without donors keep B1.

| arm (BPG; BP within 0.002) | R² | rel SSE vs B1 |
|---|---|---|
| B1 | 0.452 | 0 |
| **MP: per-publication per-metal shape, all other cells of the publication, K 6.1** | **0.522** | **+12.8 %** |
| OFFP: per-publication pair-type offsets of B1's OOF residual, K 6.1 / K 1.78 | 0.528 / 0.527 | +13.9 % / +13.7 % |
| MS: per-metal shape, same extractant only, K 1.78 | 0.515 | +11.5 % |
| MX / OFFX: other extractants only, per-metal / per-pair (K 6.1) | 0.461 / 0.461 | +1.6 % / +1.6 % |
| MX_LM10: other extractants, level-matched | 0.473 | +3.8 % |
| MG: universal per-metal shape from all other publications | 0.448 | −0.8 % |
| MCO: same chemotype, other publications (not fold-legal) | 0.462 | +1.8 % [−3.2, +6.7]; **−0.2 % without C5-BTBP** (any BTBP template repairs part of C5's residual, as in P1 §11) |
| MG_LM10_tr: universal, level-matched, training-fold donors only (BPG-legal) | 0.420 | **−5.8 %** (26 of 39 publications worse) |
| E_MRP: one random other publication's shape (expectation) | 0.426 | −4.8 % |

[I] Per-metal and per-pair offsets are reparameterisations of the same roughly 12 degrees of freedom. The two chains are
La–Nd and Sm–Lu. They differ only in how the target's reads absorb them: MX − OFFX is +0.2 % [−0.7, +1.1]. Use whichever
is simpler. The information is publication-local. The universal or level-conditional per-metal priors are a new closure line
for E-A (see lab-3).

---

## 5. The other-extractant channel (POPX, BPG)

### 5.1 Arms on identical rows

| arm | POPX (92 cells, 16 pubs, 787 rows/seed) | POPX0, no same-extractant cell (60 / 13 / 534) | POPX1, same-extractant cell exists (32 / 5 / 253) |
|---|---|---|---|
| B1 | 0.444 | 0.401 | 0.574 |
| MX (unmatched, K 6.1) | 0.467 (+4.2 %) | 0.450 (+8.2 %) | 0.510 (−15.0 %) |
| MX, K 1.78 | 0.435 (−1.6 %) | 0.447 | 0.383 |
| OFFX (pair offsets) | 0.466 (+4.0 %) | 0.450 | 0.505 |
| **MX_LM10** (tolerance 0.5 / 1 / 2: +8.5 / +9.0 / +7.4 %) | **0.494 (+9.0 %)** | **0.467 (+11.1 %)** | 0.570 (−0.9 %) |
| MG_LM10_all (same rule, every other publication) | 0.439 (−0.9 %) | 0.392 | 0.581 |
| MG_LM10_tr (BPG-legal donors) | 0.418 (−4.6 %) | | |
| MCO_LM10 (same chemotype, other publications) | 0.440 (−0.7 %) | | |
| E_MRP (random other publication) | 0.418 (−4.7 %) | 0.374 | 0.549 |

### 5.2 Inference with the publication as unit

| contrast | rel SSE | LOPO 95 % | up / down of G | sign-flip p | top-3 share | LOPO minimum (without) |
|---|---|---|---|---|---|---|
| MX − B1 (POPX) | +4.2 % | [−8.2, +16.5] | 8 / 8 of 16 | 0.23 | 1.78 | +0.7 % (1bd76c48d9) |
| MX_LM10 − B1 (POPX) | +9.0 % | [+0.3, +17.6] | 10 / 4 | 0.037 | 0.93 | +6.0 % (1bd76c48d9) |
| MX_LM10 − MG_LM10_all | +9.8 % | [−0.8, +20.4] | 9 / 7 | 0.069 | | |
| MX_LM10 − MG_LM10_tr | +13.0 % | [+3.6, +22.3] | 12 / 4 | 0.005 | | |
| MX_LM10 − MX | +5.0 % | [+0.6, +9.5] | 11 / 2 | 0.015 | | |
| MX_LM10 − B1 (POPX0) | +11.1 % | [+2.3, +19.8] | 8 / 3 of 13 | 0.025 | | |

**Robustness splits (BPG):**
- Without pub_126bbe7976 and pub_1bd76c48d9 (the leave-outs Skeptic B made mandatory): MX −4.2 %, MX_LM10 **+2.6 %**.
- Outside the giant cluster (24 cells, 7 publications): MX +3.5 %, MX_LM10 **+11.2 %**.
- Inside the label-free level window (82 cells): B1 0.562 → MX 0.610 (+10.9 %) → MX_LM10 0.622 (+13.7 %).
- Outside the window (171 rows per seed): MX −3.6 %, MX_LM10 +3.5 %.

### 5.3 Which publications and pairs carry it

- **Unmatched MX by publication:**
  - pub_1bd76c48d9: +36.5 % (0.71 SSE per seed, mostly Sm/Eu);
  - pub_126bbe7976: +8.7 % (0.60; La/Ce, Ce/Pr, Tb/Dy, Yb/Lu);
  - dfa96a5118, 84113c17ad, 4cba27f5f3 and three others: up;
  - a49146c8fd: −19.5 % (La/Ce, Ce/Pr);
  - 56338b6e47: −15.5 %;
  - 3e0f6bb1b5, b9dfc468cd, d68ed8c8cf: down.
- **Pair-type share of the unmatched MX gain:** Sm/Eu 0.55, Gd/Tb 0.29, Yb/Lu 0.21, Tb/Dy 0.21, Pr/Nd 0.16, Eu/Gd 0.10, La/Ce −0.12, Ce/Pr −0.04, Dy/Ho −0.12, Ho/Er −0.11.
  - Light pairs 0.00; Ce pairs −0.16; Gd pairs 0.39.
  - The Gd-only version is −1.0 %, so the Gd-pair gain comes from the Eu and Tb deviations, not from a publication-specific Gd dip.
- **MX_LM10:** light pairs 0.30, Ce pairs 0.28 (below the 0.50 artefact flag of M01), Sm/Eu 0.31, Gd pairs 0.18.

### 5.4 Homologue vs other chemotype

| rows | arm | rel SSE | detail |
|---|---|---|---|
| both kinds of donor exist (50 cells, 8 pubs) | homologue donors (MXH) | +9.5 % | [+3.3, +15.8] |
| | other-chemotype donors (MXO) | +3.0 % | |
| | MXH − MXO | +6.7 % | [−2.9, +16.3]; 4 up, 4 down |
| other-chemotype donor exists (85 cells, 14 pubs) | MXO | +0.1 % | 7 up, 7 down |
| | count-matched random-lab placebo | −2.1 % | |
| | MXO_LM10 (level-matched) | **+4.7 %** | [−3.5, +12.9]; 8 up, 3 down; p 0.052 |

---

## 6. Analytical fingerprint or chemistry? (descriptive; all 243 bench cells with ≥ 6 metals, 38 publications)

| pair class | pooled r | in window | out of window | levels within 1 log | levels > 1 log apart |
|---|---|---|---|---|---|
| SS: same publication, same extractant | 0.393 | 0.464 | 0.324 | 0.526 | 0.150 |
| SH: same publication, homologue | 0.243 | 0.299 | 0.188 | 0.359 | 0.095 |
| SO: same publication, other chemotype | 0.304 | 0.394 | 0.168 | 0.410 | 0.191 |
| XE: other publication, same extractant | 0.010 | 0.071 | −0.132 | 0.024 | 0.000 |
| XC: other publication, same chemotype | 0.011 | 0.027 | −0.010 | 0.006 | 0.015 |
| XX: other publication, other chemotype | −0.033 | −0.046 | −0.021 | −0.041 | −0.030 |

- **Aqueous-medium match** (same acid, |Δlog acid| ≤ 0.15, same diluent):
  - cross-publication XE 0.13 (9 pairs, 5 publications), XC 0.08, XX 0.11, against ≈ 0 unmatched;
  - inside publications, unmatched SO pairs share *more* (0.52; 5 publications) than matched ones (0.24);
  - so a matched medium explains a small part (≈ 0.1); the campaign explains the rest.
- **Coverage control** (identical metal sets): SS 0.43–0.48, SH 0.32–0.54, SO 0.34–0.58, XC 0.05, XX −0.03. The pattern is not a quadratic-fit artefact of shared coverage.
- **Level bands** (SO + SH metal pairs within 1 log):

  | band of mean log D | < −2 | −2…−1 | −1…0 | 0…1 | 1…2 | > 2 |
  |---|---|---|---|---|---|---|
  | r | 0.53 (n = 72) | 0.27 | 0.32 | 0.41 | 0.37 | 0.40 |

  The curve is flat. A mass-balance, standard or blank bias, whose log error grows like ε(1 + 1/D), would rise steeply toward low D.
- **Per metal** (pub-weighted SO): La 0.09, Ce −0.07, Pr 0.33, Nd 0.43, Sm 0.33, Eu 0.24, Gd 0.32, Tb 0.33, Dy 0.25, Ho 0.11, Er −0.18, Tm −0.28, Yb 0.04, Lu 0.12. The campaign-wide component sits at **Pr–Dy**, not at La/Ce and not on the heavy oxide-victim metals.
- **ICP-MS oxide victims** (Gd–Lu: LaO/CeO/PrO → Gd, NdO → Tb/Dy, SmO → Ho/Er, EuO → Tm, GdO → Yb, TbO → Lu) vs La–Eu: SH 0.248 vs 0.242, SO 0.145 vs 0.134. There is no interference signature. This is consistent with M06's earlier null, where an oxide model reached r −0.05.
- **Ce** (n = 162):
  - deviation mean +0.035, median +0.033;
  - 7 cells above +0.3, 1 above +0.5;
  - below the floor (−2): mean +0.05, sd 0.27, against an in-window sd of 0.11;
  - HCl +0.095 (n = 27) vs other media +0.023.

  Ce(IV) extraction would give large positive excesses in oxidising nitrate media. They are absent. The floor cells carry the Ce scatter (as in M27).

[I] **Reading.**
- The shared within-campaign pattern is mid-series, in-window, level-flat and chemotype-independent at about 0.3 (about 0.5 for homologues). It is absent across campaigns even for the same extractant.
- No classic analytical artefact fits it: not the floor, not oxide interference, not Ce(IV), not mass balance.
- The candidates that remain are campaign-level factors:
  - one aqueous feed and its preparation (actual ionic strength and nitrate activity, metal loading);
  - one ICP calibration and run (matrix-matched standards, drift);
  - one person's procedure (phase-ratio and contact-time habits).
- The corpus cannot rank these.

### 6.1 Laboratory or campaign? (`lab_samelab.py`)

**Laboratory rule.** It was frozen before scoring and uses metadata only. Two publications belong to the same laboratory
when their author lists share ≥ 2 surnames. Authors come from `probes/P4_corpus_registry.csv`, and DOIs from round-4
`sources_by_doi.csv`. The rule gives 51 links among 58 publications: the JAEA Sasaki group (7 publications), the Mowafy
group, Jülich/KIT, Chalmers/Reading, the Moscow Babain group, ORNL, and others.

**Descriptive pooled r:**

| relation | same extractant | other extractant |
|---|---|---|
| same publication | 0.393 (741 pairs) | 0.269 (521) |
| same laboratory, other publication | 0.059 (143 pairs, 6 publications) | 0.054 (874, 22) |
| other laboratory | 0.004 | −0.026 |

**Predictive** (BPG; C3 targets whose laboratory has another publication with donor series: 129 cells, 22 publications,
985 rows per seed; B1 0.409):
- same laboratory's other publications, level-matched (SLP_LM10): 0.422, +2.2 % [−0.9, +5.3], 13 publications up, 8 down;
- other laboratories, level-matched: −1.9 %;
- on the 68 cells that also have own-publication other-extractant donors: SLP +0.1 % vs own campaign (MX_LM10) +4.4 %.

[I] The fingerprint is a property of the campaign. It is not a stable laboratory constant. The rule can create false links
through common surnames (liu, zhang, zhou, xu), which would dilute the same-laboratory class.

---

## 7. Reference-series (TODGA) check (46 targets, 5 publications, 349 rows per seed, BPG)

| arm | R² | rel SSE vs B1 | publications up |
|---|---|---|---|
| B1 | 0.386 | 0 | |
| own-campaign TODGA shape | 0.419 | **+5.3 %** [−0.3, +11.0] | 5 / 5 (p 0.031, the floor for G = 5) |
| lab-specific part only (own − consensus) | 0.413 | +4.4 % [+3.2, +5.6] | 3 / 5 |
| consensus TODGA shape (equal-weight over other publications) | 0.389 | +0.6 % | |
| another laboratory's TODGA (expectation over publications) | 0.361 | −4.1 % | |
| all other publications' TODGA pooled | 0.337 | −8.0 % | |
| medium-matched literature TODGA (21 targets have one) | | **0.0 %** (own TODGA +5.7 % on the same rows) | 2 / 4 |
| pair offsets from own TODGA C3 cells (OFFREF) | 0.421 | +5.8 % [+2.6, +9.0] | 5 / 5 |
| all own other extractants (MX) / level-matched (MX_LM10) | 0.422 / 0.441 | +5.9 % / +9.0 % | |

**Per publication, own TODGA:** 126bbe7976 +4.9 %, 4cba27f5f3 +22.8 %, 5a68dc5665 +19.2 %, b9dfc468cd +0.1 %,
d68ed8c8cf +3.9 %.

**Caveat.** All five are DGA campaigns in one source cluster, so the reference is a homologue of every target here. The
chemistry and laboratory parts cannot be separated in this set.

---

## 8. Protocol sizing (POPX0, BPG)

**Read ladder** (all on the 4-read scored rows):

| | 2 reads | 3 reads | 4 reads |
|---|---|---|---|
| B1 | 0.235 | 0.348 | 0.401 |
| MX_LM10 | 0.298 | **0.425** | 0.467 |

MX_LM10 at 3 reads against B1 at 4 reads: +4.0 % [−5.3, +13.2], 6 publications up, 7 down. **The lab fingerprint is
worth about one read.** The same-extractant sibling of P1 is worth about two.

**Donor-count subsampling** (48 cells, 7 publications with ≥ 3 donor series), exact expectation over subsets:

| donor series | 0 (B1) | 1 | 2 | 3 | all |
|---|---|---|---|---|---|
| R² | 0.381 | 0.402 (+3.4 % [−3.8, +10.6]; 5 up, 2 down) | 0.417 (+5.9 %) | 0.429 (+7.8 %) | 0.462 (+13.1 %) |

One reference series alone is about +3 %. That matches the TODGA check (+4–5 %).

---

## 9. Ideas (details in the StructuredOutput)

- **lab-1.** E-LAB estimand with a level-matched campaign calibration (MX_LM10). Freeze it now. Score it once on unseen multi-ligand papers, as a within-paper test.
  - In P4_candidates.csv, about 16 E1/E2 candidates describe ≥ 2 extractants in one campaign. They come from about 11 laboratories; 8 papers from 7 laboratories are E1.
  - Comparators: B1, the same rule with other-publication donors, and a count-matched random-lab placebo.
- **lab-2.** An in-run reference-series protocol for a user laboratory: TODGA plus the new ligand in the same campaign and ICP run, in **every** campaign (a laboratory's reference from another paper is worth +0.1 %, §6.1), with analytical arms that split "analytics" from "campaign chemistry":
  - direct assay of both phases vs mass balance;
  - two isotopes or two lines per metal;
  - two dilutions;
  - a duplicate day.

  Data-gated.
- **lab-3.** Closure line for E-A: universal and level-conditional per-metal priors from other publications lose under BPG (−0.8 % / −5.8 %). Register it as closed with a one-minute replay at three tolerances.

---

## 10. Caveats

- **Concentration.** POPX has 4 source clusters and one holds 68 of 92 cells. The top-3 publications carry 93 % of the MX_LM10 gain. pub_1bd76c48d9 (6 cells) alone carries the largest share.
- **Post hoc choices.** The unmatched rule fails without its two lead publications (−4.2 %). The level-matched rule survives (+2.6 %), but that rule was devised in this probe on these rows.
- **Units.** Publications are not laboratories. Same-laboratory publications (JAEA, the Moscow groups) are separate units here, so the intervals are anti-conservative.
- **Level matching.** It uses the target's B1 levels, which are label-free except at the 4 reads (which are known), and the donors' measured levels (known in the E-LAB scenario).
- **Not E-A-legal in this form.** The donors are test-fold cells of the same publication. Only the MG_*_tr arms respect BPG.
- **Descriptive sharing** uses all labels and only describes. Products are pair-weighted except where marked pub-weighted.
- **Power.** For an effect of about +5–10 % relative SSE, P2's replay suggests that roughly 19–28 equal-weight units are needed for 80 % power on a B1-sized effect. With fewer than about 10 units, any prospective leg is descriptive.
