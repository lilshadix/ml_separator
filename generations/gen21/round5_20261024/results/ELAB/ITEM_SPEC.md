# Item 1.3 specification: E-LAB, campaign calibration of a new extractant (lab-1; the E-CT annex)

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10).  Every in-corpus number is EXPLORATORY (seen) and carries **"best of about 25 arms on seen rows; expect regression"**; nothing here is a test and no in-corpus p-value is called significant.  E-LAB is a separate estimand (a new extractant read at 4 metals in a campaign that also reports other extractants' full series): **never placed next to E-A's 0.452**; B1 on the same rows is the comparator in every table.  Sources: brief 6.3, 6.0, 6.10, 8, 9; I2 lab-1 (with both skeptics); `ideas2/lab.md`.

Code: `r5/elab.py`, `scripts/s1_elab.py`, `tests/test_elab.py`; shared statistics `r5/common.py`; frozen duplicate rules `r5/dupes.py`; metadata `r5/population.py`; frozen mechanism map `r5/supply.py`.  Copied and adapted (never run in place) from `round5_design_20261023/ideas2/scripts/` (LF SHA-256):

| source | LF SHA-256 | what was taken |
|---|---|---|
| `lab1_recompute_skA.py` | 8a82cd2a5d38a7b60f6daa0c4ffc4b20f3c6f47bbdd384ba6d04b7700faf253e | qdev, interp_levels, lm_cw (-> lm_shape), E_RM, E_PBC, E_PBA, NC05 / NC03, leave-two-out |
| `lab1skepB_placebo.py` | df14531850d0521e679260823990c19404993fe7c4b7ebf1cb056d9ce8f60f3f | E_PCM, E_PCMC (per-metal count-matched, level-matched placebos) |
| `lab1skepB_nearcopy.py` | 1f5434f6c23e7933416c9a91f0a28fd9aef39ac4c7941f4bdb1bedb2c84cb902 | label-free near-copy rule (through `r5.dupes.near_copy_mask`, copied from the same file) |
| `lab1_controls_skA.py` | b0d7f2a8cc09ea114d2d49fa85074a3c7cbde54bf610e989063bbca3b815dd7b | global-K grid |
| `lab_calib.py` | 9b80f5ca68d29c38d0ec5f577d8595bb09d03c5c7842ae5facedc5477ee71de8 | POPX0 (no same-extractant cell), unmatched shape MX |
| `ideas2/lab.md` | aa66feee53e133000985272f901df4a740fb4a73723febb60390588fb7156bf8 | theme note (definitions, seen values) |

## Rows, base, model

- Rows: round-4 REF rows of the POPX targets (`r5.common.load_rows`: C3, k = 3, i.e. **4 reads** with E-A's DOPT_Q read sets, the union of the cached fold sequence `seqs['DOPT_Q'][:3]`); 5 dev seeds; **BPG primary, BP SENSITIVITY**.  Pooled R^2 within each seed, then the seed mean; MSE; relative pair-SSE vs B1 (seed mean of 1 - SSE_cand / SSE_ref; the source's all-seed pooled form is also written as `rel_pooled`); balanced MAE for point arms.
- Model (every arm; exact at the 4 reads): levels = PWL_R[y_S - v_S] + v (-Shannon CN8 radius axis, `r3.interp` 'R'); B1 is v = u = beta 1[Gd], beta the fold's fitted round-4 amplitude (`round4_20261022/results/REF/fitted_<design>.json`).  B1 is rebuilt the same way as every arm and checked against the parquet (8.9e-16) and against `r4.refs.predict_levels`.
- **MX_LM10** (frozen predictor): v_m = (1 - w_m) u_m + w_m c_m, w_m = n_m / (n_m + **6.1**); c_m = mean over donor series of each donor's deviation from its own least-squares quadratic in the radius axis (donors with >= 6 finite metals), using only donor metals whose V1 log D lies within **1.0** log of the target's B1 level L_t,m (label-free: the true value at the 4 reads, B1's interpolation elsewhere); n_m = the donors that pass at m.  K = 6.1 and the 1.0 tolerance were pinned before any lab-1 score but on these data (disclosed as seen).

## Pools (per C3 target t; donors: bench cells, all 521, outside V2-1/2/3, >= 6 finite metals, never t)

| pool | definition |
|---|---|
| SRC | same publication, other extractant (lab-1 as published; the reproduction pool) |
| CLEAN | SRC minus the links t-j flagged by the frozen section-5.1 bit-identity rule or by `r5.dupes.offset_scan`'s near-identity flag (max abs d <= 0.05 over >= 6 shared metals): the V2-missed duplicates of I2 items 4-7 are **quarantined**; metal-concentration duplicate groups (`r5.dupes.metal_conc_duplicates`; 0 groups on V1) reduced to their first member present |
| PRIM (primary) | CLEAN minus the **label-free lab-1 near-copies** of the design / seed read set: same medium (acid type, diluent, abs dlog10 acid <= 0.15) and log D within 0.15 of t's at every one of t's 4 read metals |
| OFS | CLEAN minus constant-offset links (`flag_offset`), then the near-copy rule (SENSITIVITY) |
| XPUB | every other publication's donor cell minus the quarantined links with t (placebo pool) |
| XPUB_FAM | XPUB restricted to t's `chem_family` (broad class: diglycolamide, malonamide, amide_other, n_heterocyclic_polydentate, phosphoryl, podand_ether, hydroxyl_acid) |

The duplicate links are a frozen Stage-0 data rule applied once to V1 (label-based) and held fixed in the intervention.  Counts on V1 (TECHNICAL SMOKE, no score): links bit 6, near-identity 9, offset 418, metal-concentration groups 0; targets with a near-copy donor in some read set 23 (SRC pool; Skeptic B's 23) and 17 (CLEAN pool: 6 targets' near-copies are already quarantined); no target-seed has an empty PRIM pool.

## Populations and strata (row flags `in_<stratum>`)

| stratum | definition | cells / publications / rows per seed (counts, smoke) |
|---|---|---|
| SRC | POPX as published: non-V2 target with >= 1 SRC donor (SENSITIVITY bridge) | 92 / 16 / 787 |
| **PRIMARY** | POPX with near-copies excluded: >= 1 CLEAN donor; the near-copy rule removes donors per read set (PRIM may fall back to B1; it never does on V1) | 92 / 16 / 787 |
| POPX0 | PRIMARY without a same-publication same-extractant donor cell (bit links removed): new extractants | 60 / 13 / 534 |
| POPX1 | PRIMARY with one | 32 / 5 / 253 |
| SCOPE_IN | amide O-donor solvating extractants (frozen map `r5.supply.classify_extractant(chem_family, extractant_name)` = NEUTRAL_AMIDE) in nitrate media (acid one-hot `hno3`) | 59 / 11 / 428 |
| NULL_PRIOR | PRIMARY outside SCOPE_IN (the declared null-prior strata; by class in `classes_<design>.csv`) | 33 / 7 / 359 |
| LOWD / HIGHD | rows with min(L_A, L_B) < -2 / >= -2 (label-free B1 levels): the low-D band split | 10 cells, 55.4 rows / 88 cells, 731.6 rows |
| SS_OS | targets with both same-series (sharing an `experiment_series_id` with t) and other-series CLEAN donors | 35 / 4 / 225 (I2: 36 / 4 on SRC) |
| SM_DM | the same split by medium (Skeptic B's reading; DIAGNOSTIC) | 36 / 6 / 311 |
| NO_LEAD2 | PRIMARY without pub_126bbe7976, pub_1bd76c48d9 (Skeptic B's mandatory leave-outs; SENSITIVITY) | 71 / 14 / 590 |
| OUT_GIANT | PRIMARY outside source cluster g:pub_74428aeab2 (SENSITIVITY) | 24 / 7 / 222 |

Laboratories (frozen PROVISIONAL key `results/stage0/lab_key.csv`, all 16 POPX publications covered): PRIMARY holds 9 conservative / 14 liberal laboratory units and 4 source clusters.

**C5-BTBP** (pub_6a0ef8528c, one extractant, so never a POPX target): **C5_in primary** (as the seen lab-1 numbers); C5_out (rows and every pool without pub_6a0ef8528c) and C5_cell_out (without c950dc2982e96b83) SENSITIVITY.  Only the placebo pools change.

## Arms (frozen; identical rows; point arms are pair-SF predictions, E_* expected squared errors over NDRAW = 30 seeded draws)

| arm | definition | label |
|---|---|---|
| B1 | v = u (comparator) | EXPLORATORY (seen) |
| **MX_LM10** | the rule above on PRIM | EXPLORATORY (seen); best of about 25 |
| MX_LM05, MX_LM20 | tolerance 0.5 / 2.0 | SENSITIVITY |
| MX_K6p1, MX_K12, MX_K20, MX_K35, MX_K60, MX_K100 | unmatched shape over PRIM, w = n / (n + K) (the **global-K shrinkage control**) | EXPLORATORY (seen) |
| MX_GK_O | the best K of the grid on the scored rows of each stratum (ties averaged) | ORACLE [O] (optimistic for the control) |
| MX_GK_LOPO | K chosen on the other publications' PRIMARY rows (ties averaged); publications outside PRIMARY use the all-PRIMARY choice | EXPLORATORY (seen) |
| MX_LM10_SS / MX_LM10_OS | PRIM split into same-series / other-series donors | EXPLORATORY (seen) |
| MX_LM10_SM / MX_LM10_DM | PRIM split into same-medium / other-medium donors | DIAGNOSTIC |
| MX_LM10_LO / MX_LM10_HI | MX_LM10's blend applied only at metals with L < -2 / >= -2 (low-D band split, metal side) | DIAGNOSTIC |
| MX_LM10_NB | SRC pool, no cleaning (lab-1 as published; +9.0 % seen) | SENSITIVITY |
| MX_LM10_NC | SRC minus the near-copy rule only (Skeptic B's +7.6 %) | SENSITIVITY |
| MX_LM10_OFS | OFS pool | SENSITIVITY |
| MX_LM10_NC05 / MX_LM10_NC03 | SRC minus donors with Skeptic A's offset-removed residual sd < 0.05 / 0.03 (>= 6 shared); label-based, reads the target's unread labels | SENSITIVITY; ORACLE [O] |
| E_RM | random own-campaign donors: per metal the number MX_LM10 keeps, drawn from PRIM donors measured there (same w_m) | EXPLORATORY (seen) |
| E_PBC | each PRIM donor replaced by a random XPUB cell of that donor's chemotype (any XPUB cell if none left), then level-matched (count- and chemotype-matched) | EXPLORATORY (seen) |
| E_PBA | as E_PBC from any XPUB cell | EXPLORATORY (seen) |
| E_PCM | per metal min(n_m, available) level-matched donors from XPUB, w = k / (k + 6.1) (same n_m and w_m) | EXPLORATORY (seen) |
| E_PCMC | E_PCM from XPUB_FAM (same broad class) | EXPLORATORY (seen) |

Every arm is computed for each C5 variant (suffix `@C5_out`, `@C5_cell_out` in the rows parquet).  Arm list frozen; an arm added after a result is seen goes into AMENDMENTS.md labelled post hoc.

**EXCLUDED at this freeze** (named so their absence is not post hoc): MG_LM10_all (pooled placebo, w ~ 1: brief 6.3 "not the pooled MG_LM10_all"); MG_LM10_tr and the universal / level-conditional per-metal priors (the lab-3 closure line for E-A, a closure-ledger item); the read ladder "worth about one read" (I2 robustness row, not in the 6.3 list); the TODGA reference-series arms (lab-2, data-gated); the homologue / other-chemotype split MXH / MXO (lab.md 5.4); Skeptic B's medium-matched placebo E_PCMM; the V2-5 floor sensitivity (an E-CT row).

## Contrasts, units, spread

- MAIN (every stratum and C5 variant): MX_LM10 - B1, MX_LM10 - E_PBC, MX_LM10 - E_PCM, MX_LM10 - E_RM, E_PBC - B1, E_PCM - B1.
- FULL (C5_in; PRIMARY, POPX0, SCOPE_IN): MAIN plus MX_LM10 - E_PCMC, MX_LM10 - E_PBA, E_PCMC - B1, E_PBA - B1, E_RM - B1, MX_LM10 - MX_GK_O, MX_LM10 - MX_GK_LOPO, MX_GK_LOPO - B1, MX_LM10 - MX_K6p1, MX_K6p1 - B1, MX_LM10_LO - B1, MX_LM10_HI - B1, and every sensitivity arm - B1.  SS_OS adds MX_LM10_SS - B1, MX_LM10_OS - B1, MX_LM10_SS - MX_LM10_OS; SM_DM the medium analogues; SRC adds MX_LM10_NB - B1, MX_LM10_NC - B1.  C5_out / C5_cell_out: PRIMARY and POPX0 with MAIN.
- Units: **laboratory, conservative key (`lab_cons`; PROVISIONAL, U6 open) = the primary unit (brief 6.3 "Units.
  Laboratory (frozen key), with the cluster level and leave-two-publications-out reported"; s0-items-5)**; the liberal
  laboratory (`lab_lib`), **publication** (the bridge to lab-1's seen +7.6 % [-1.1, +16.4], p 0.093) and **source
  cluster** (`cluster_cpg`) are reported beside it.  Per unit (brief 9.2): seed means of n, SSE_ref, SSE_cand; dSSE; L; rel; W/L/T with a tie an exact 0.  Tests (`r5.common.unit_tests`, descriptive): SF_POOL (exact for G <= 13, else 4,999 random flips), SIGN, SF_LOG, IM_REL with its one-sided 95 % lower bound; unit-median rel; top-3 share of dSSE.
- Spread (publication): LOPO jackknife with t(P - 1) (`r5.common.lopo`); leave-two-publications-out (every pair; minimum and the pair; the named pair pub_126bbe7976 + pub_1bd76c48d9); leave-top-3-out (the three publications with the largest dSSE).  Cluster-level inference: 4 clusters, one holding most rows; reported, never called a test.

## Report rules (in-corpus readings; no kill fires in-corpus; `rules_<design>.csv`)

| rule | seen | outcome |
|---|---|---|
| headline MX_LM10 - B1 on PRIMARY with LOPO interval, W/L, SF_POOL, top-3, on the conservative-laboratory unit (the publication and liberal-laboratory readings beside it; the controls likewise) | +7.6 % [-1.1, +16.4], p 0.093 on publication units (cleaned; Skeptic B) | reported, not tested |
| bridges MX_LM10_NB - B1 (SRC), MX_LM10_NC - B1 | +9.0 % [+0.3, +17.6], 10 / 4, p 0.037 | reported (SENSITIVITY) |
| POPX0 MX_LM10 - B1 | +11.1 % | reported |
| controls MX_LM10 - E_PBC / E_PCM / E_RM | +10.2 % [+1.6, +18.9] / +11.4 % [+2.3, +20.4] / +1.8 % [-0.5, +4.2] | reported |
| MX_LM10 - best global K (oracle), MX_LM10 - MX_GK_LOPO | +4.7 % [+2.2, +7.2] (K = 12) / not run by lab-1 | reported |
| leave-two-publications-out (named pair, minimum), top-3 share | +2.6 %; 0.93 | reported |
| same-series vs other-series | +3.8 % vs -3.6 % (36 cells, 4 publications) | reported |
| low-D band split (rows LOWD / HIGHD; metals LO / HI) | sharing r 0.53 at log D < -2 (descriptive) | reported (DIAGNOSTIC) |
| external-kill statistic read in-corpus (laboratory-median rel SSE; IM one-sided 95 % lower bound; conservative and liberal labs) | n/a | "would kill / pass if applied"; it is not applied (DIAGNOSTIC) |
| narrowing reading MX_LM10 - E_PCM <= 0 -> "level-conditional weighting, not campaign" | +11.4 % | reading (CONSISTENCY CHECK) |
| harm-flag reading: a chem_family class with >= 3 conservative laboratory units losing > 5 % | DGA +11.2 %, malonamide +16.5 %, pyridine amides 0.0 %, other -2.2 % | reading (CONSISTENCY CHECK) |
| scope: SCOPE_IN vs NULL_PRIOR | n/a | reported |
| **external leg** | supply about 3-6 laboratory units | descriptive unless >= 8 independent laboratories with >= 2 extractants x >= 6 Ln in one campaign exist; kill if the laboratory-median relative SSE (MX_LM10 - B1) <= 0 or its one-sided 95 % lower bound <= 0; narrow if MX_LM10 - E_PCM <= 0; harm flag if any class with >= 3 units loses > 5 %; false kill of a true effect 0.41-0.52 at 8 units, 0.25 at 11, 0.06-0.10 at 16.  NOT RUN in Stage 1 (data-gated, brief 7) |
| label rule | | every number carries "best of about 25 arms on seen rows; expect regression" |

The reproduction targets are restated on the cleaned population (`summary_<design>.json` -> `restated_targets`: PRIMARY and POPX0 counts, B1 and MX_LM10 R^2).

## Reproduction targets (brief 6.0; `--repro`, TECHNICAL SMOKE; a mismatch stops the item)

- B1 rebuilt from the fold betas equals `REF_<design>.parquet` within 8.9e-16 on every REF row (9,010 per design; 18,020 over both) and equals `r4.refs.predict_levels`.
- POPX 92 cells / 16 publications / 787 rows per seed; POPX0 60 / 13 / 534.
- MX_LM10_NB (SRC pool) equals `ideas2/lab_out/lab_rows2.parquet` MX_LM10 within 1e-12 on every POPX row (target 0.0 over 7,870 rows, both designs); also MX_LM05 / MX_LM20 / MX (unmatched, K = 6.1) against lab_rows2, and n_MS against `lab_rows.parquet` (read-only references).
- BPG: R^2 of B1 = 0.444 and of MX_LM10_NB = 0.494 on POPX (3 decimals).  Consistency lines (do not stop the item): 4-decimal values of `lab_calib2.log` (BPG 0.4437 / 0.4936; BP 0.4423 / 0.4936) and `lab_popx0.log` (POPX0 BPG 0.4008 / 0.4670; BP 0.3987 / 0.4669).

## Leakage and controls (brief 8)

- Information contract: allowed the target's 4 legal reads, the campaign's other series in full, the fold's beta.  Forbidden: the target's unread metals; the target or a bit-identical copy in a strict pool.  `Campaign.guard` refuses a pool holding the target, or (except SRC) a bit-identical copy.
- Prediction intervention (`Campaign.intervention`): every unread label of each POPX target replaced by seeded noise (and its detrended row recomputed); every point arm and every draw must be bit-identical, except the declared label-based sensitivities NC05 / NC03 (their changes are counted).  Positive controls: target in its own CLEAN pool (guard must raise), a bit-identical copy in a strict pool (the frozen item-4 link; guard must raise), target leaked into PRIM with the guard off (MX_LM10 must change).  All 5 dev seeds in the main run (`--iv-seeds`).
- Fit poisoning: the LOPO K choice of each publication must not change when that publication's labels are replaced by noise (`gk_poison_check`).  The check runs the production `gk_lopo` on rows whose y and every arm / E_* / ndraw column of the held-out publication are seeded noise, and carries a positive control (the K chosen with the publication included must change for at least one publication; s0-mechanics-15).  Nothing else is fitted (K, tolerance and NDRAW pinned; beta = the stored round-4 fold fit).
- Order: every scoring run executes the reproduction target first and the leakage checks before any scored write; a mismatch or failure stops the item; `--iv-seeds` is smoke-only, and a pass needs every dev seed (s0-mechanics-3, -5, -17).
- Cache poisoning: no new cache.  Isolation: not applicable by design (E-LAB's donors are test-fold cells of the target's own campaign, which its information contract allows; the BPG-legal universal version is the lab-3 closure line).  Support: no scored pair with both ends read; strict adjacency, lighter first; read sets equal to the parquet's; every arm finite on every PRIMARY row (fixed denominators).
- Output: `results/leakage/ELAB_<design>.json`.

## RNG seeds (REGISTRY.csv scheme `default_rng(20261023 + crc32(key))`, `r5.paths.seed_for`)

| procedure | key |
|---|---|
| draws of E_RM, E_PBC, E_PBA, E_PCM, E_PCMC (NDRAW = 30 per target and read set) | `ELAB|<arm>|<c5 variant>|<design>|<split seed>|<cell id>` |
| random sign flips when G > 13 (B = 4,999) | `ELAB|<design>|<c5>|<stratum>|<cand>-<ref>|<unit>|SF_POOL` (and `|SF_LOG`) |
| intervention noise | `ELAB|INTERVENTION|<design>` |
| LOPO-K fit-poisoning noise | `ELAB|GK_POISON` |
| smoke surrogate of V1 | `ELAB|SMOKE_SURROGATE|<design>` |

## Interpretations fixed at this freeze

- "Same-series" = sharing an `experiment_series_id` with the target (on V1 this reproduces I2's 36 cells / 4 publications on the SRC pool; 35 / 4 on CLEAN); the medium split is kept as a DIAGNOSTIC alternative reading.
- "Low-D band split" = the row split min(L_A, L_B) < -2 plus the metal-side LO / HI arms, both label-free.
- "Global-K shrinkage control" = the frozen K grid of `lab1_controls_skA.py`, reported per K, at the oracle best K ([O]) and at a LOPO-chosen K.
- "V2-missed duplicates quarantined" = bit-identity plus near-identity links removed from every pool; constant-offset links are a sensitivity (brief 5.1: offset copies are sensitivities).
- Scope via the frozen mechanism map and the acid one-hot; E_PCMC's "broad class" = `chem_family`.
- Primary C5 variant = C5_in (no POPX target is C5-BTBP; the seen lab-1 numbers include C5 cells in the placebo pools).

## Outputs (`results/ELAB/`)

`rows_<design>.parquet` (every arm and variant, strata flags, units, B1 levels, label, seal), `pools_<design>.csv` (per target and seed pool counts, near-copy counts, fallback anatomy; counts only), `populations_`, `arms_`, `contrasts_`, `units_`, `lopo_`, `ltpo_`, `globalK_`, `classes_`, `rules_<design>.csv`, `PER_ITEM_TABLE_<design>.md` (brief 9.3), `summary_<design>.json`, `repro_<design>.json` (`--repro`), `SMOKE_COUNTS_<design>.json` (`--smoke`); `results/leakage/ELAB_<design>.json`.

## Command lines (from `D:\ml_separator_gh`, one process at a time)

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_elab.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_elab.py --smoke --design BPG
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_elab.py --smoke --design BP
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_elab.py --repro --design both
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_elab.py --design BPG
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_elab.py --design BP
```

Compute (smoke, per design): build about 85 s, tables about 47 s, intervention about 23 s per seed; the main run is about 4-5 minutes per design.
