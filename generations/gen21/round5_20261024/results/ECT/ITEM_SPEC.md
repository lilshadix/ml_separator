# Item 1.1, E-CT core: frozen specification

Stage-1 item 1.1 of `round5_design_20261023/CLAUDE_CODE_PROMPT.md` (section 6.1; conventions 6.0, 6.10, 8, 9).
Estimand **E-CT** (condition transfer): a publication has measured this extractant's series at other conditions; how
well do 0-4 reads at a new condition predict the strict-adjacent log SF? Laboratory wording waits for the reviewed
laboratory key; until then "same publication" reads "same study or campaign".

Status: written before the item seal. **No round-5 score had been computed when this file was written**: only the
technical smoke (`--smoke`: every metric suppressed, expectation arms and tables run on seeded noise) has run on real
data. Every in-corpus number this item produces will be EXPLORATORY (seen). None is placed next to E-A's 0.452; B1 on
the same rows is the comparator (P1 saw 0.531 on PRIMARY).

Code: `r5/ect.py` (frozen predictor, comparators, in-corpus builder, leakage checks), `r5/ect_report.py` (tables,
inference, report rules), `scripts/s1_ect.py` (driver), `tests/test_ect.py`. Adapted from
`probes/scripts/P1_ct_build.py` (LF SHA-256 `afd7ae54…7138d0`), `P1_ct_analyze.py` (`d9283ec0…fc346f`),
`P1_ct_diag.py` (`4234aa3d…f66b2d`), `P1SK_recompute.py` (`1bbca9d6…7663de`), `ideas2/scripts/shift_build.py`
(`3ed0db9e…c1ccb`), `shift1_skA.py` (`0396f409…4456e4`), `recipe_fixedladder.py` (`dba2e664…47676ea`),
`recipe_build.py` (`0b572f7b…183070e`); full digests in the module docstrings.

## 1. Order of operations (section 1.2)

1. Stage-0 seal (`scripts/r5_seal.py --seal`), including this file, the code above and `REGISTRY.csv`.
2. Item code seal (this file + code digests in the manifest).
3. Reproduction run, labelled TECHNICAL SMOKE: `--repro` (section 4 below). A mismatch stops the item; a fix by code
   change is resealed by an append-only amendment listing what was seen.
   Every scoring run (not only `--repro`) executes the reproduction target first and stops on a mismatch
   (SystemExit, a STOP line in RUN_RECORD.md, no scored file); its leakage checks run before any scored
   write and stop the item the same way; the run ends with the post-run seal check (`r5.sealcheck`). Options
   that shrink a leakage check are smoke-only (s0-mechanics-3, -4, -5, -17).
4. Arms: the full run, BPG (decides labels) and BP (sensitivity).

## 2. Exact command lines (from `D:\ml_separator_gh`)

```
# technical smoke (no seal; every metric suppressed)
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py --smoke --design BPG
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py --smoke --design BP

# after the seal: reproduction targets (TECHNICAL SMOKE), then the arms
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py --repro --design BPG
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py --repro --design BP
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py --design BPG
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py --design BP

# tests (synthetic / construction-only; the sealed reproduction test is skipped)
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ect.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest

# the sealed reproduction test (REQUIRES THE SEAL; only after steps 1-2)
R5_RUN_SEALED=1 PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ect.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest -k test_p1_reproduction_requires_seal
```

After every run: `round4_20261022/scripts/r4_seal.py --verify` must still report 5bacfe4c, and the round-3 LF digests
must match (section 8). Runtime measured in the smoke: build 48-52 s per design, intervention about 45 s per dev seed
(the full run checks all 5), tables about 50 s.

## 3. Population, strata, information contract

- Sibling of target t: any bench cell (521) with the same `publication_id`, the same extractant SMILES and a finite V1
  log D at every metal of t (`r5.population.build_pools`, metadata only).
- CLEAN: t and sibling outside V2-1 / V2-2 / V2-3; sibling `cond__` differs in a family other than contact time / metal
  concentration. Then the frozen section-5.1 bit-identity rule (`r5.dupes.bit_identity`: >= 3 shared metals float64
  identical and >= 50 % of the shared finite metals) removes flagged links (6 on V1; inside PRIMARY items 1 and 3).
  n_sib is counted after removal.
- Strata (counts from the Stage-0 population, identical in BP and BPG): **PRIMARY** = BIT, n_sib >= 2, without C5-BTBP
  (99 cells, 12 publications, 14 publication x chemotype units, 690 rows per seed; 3 cpg clusters, Kish 1.25);
  WITH_C5 (114 / 838); GE1 = n_sib >= 1 without C5 (115 / 820); GE1_WITH_C5 (130 / 968); EQ1 = n_sib = 1 (16 / 130,
  the harm rule); PRIMARY_NOBIT etc. = P1's CLEAN rule without the bit rule (P1's reproduction target; sensitivity);
  V1_GE1 (149 / 1,111, E6 reproduction) and V1_GE2_NOC5 (112 / 821, the bridge to IR's numbers).
- Allowed inputs: the target's legal reads, the siblings' full measured series, the fold's B1 beta (fitted without the
  target's publication), siblings' OOF residuals for OFF (leave-target-out). Forbidden: the target's unread metals, the
  target or a bit-identical copy in any pool, any quantity tuned on E-CT test rows. K = 1.78 = (1 - r) / r with
  r = 0.36 measured on these data: disclosed as seen.

## 4. Reproduction targets (section 6.0; TECHNICAL SMOKE; a mismatch stops the item)

| target | rows | expected | tolerance |
|---|---|---|---|
| B1 recomputed from the fold betas vs the REF parquet | all 18,020 rows (9,010 per design) | max abs diff 8.88e-16 seen | <= 8.9e-16 (brief 6.0); y exact; read sets identical |
| E6 section 6.8 V1 population (BPG) | V1_GE1: 149 cells, 1,111 rows per seed | V1_SIB_r4 0.5734, V1_SHK2_r4 0.5946, B1_r4 0.4489 | 5e-5 |
| P1 PRIMARY without the bit rule (BPG) | PRIMARY_NOBIT: 99 / 12 / 14 / 690 | B1_r4 0.5309, NB_SH_r4 0.7082, NB_SIB_r4 0.6987, NB_OFF 0.6963 | 5e-5 |

BP: identity and counts only (no BP value target is registered; BPG decides). The smoke already confirmed the
B1 identity (8.88e-16), y exact, 0 read-set mismatches and every stratum count above.

## 5. Frozen arm list (all on the identical 4-read scored rows; reads r; code budget k = r - 1)

n = |pool|, w = n / (n + K), K = 1.78; read sets = union of `cell['seqs']['DOPT_Q'][:r - 1]` (nested within seed).

| column | computation |
|---|---|
| `B1_r2`, `B1_r3`, `B1_r4` | PWL_R[y_S - u_S] + u, u = beta 1[Gd], beta the fold's fitted round-4 amplitude |
| `B0_r2..4` | PWL_R[y_S] |
| `SIB_r2..4` | mean_j (Y_j + PWL_R[y_S - Y_j,S]) over the BIT pool |
| **`SH_r2..4`** | w SIB_r + (1 - w) B1_r (frozen predictor; shrinks toward B1 on the same reads) |
| `SHK2_r4` | SH with K = 2 (E6) |
| `COPY`, `SH0` | T (0 reads); w COPY + (1 - w) G14@ZERO |
| `OFF`, `OFF_ALL`, `OFF_PUB`, `OFF_XEXT` (+ `_np`) | B1_r4 + sum_j r_j,p / (n_p + K), r = y - B1 OOF of: the BIT siblings that are C3 / every other non-V2-dropped C3 cell of the publication x extractant (no cond__-difference or coverage rule; P1_ct_build.py parity; bit links removed; same-condition and non-covering cells enter; P1's word CLEAN here means non-dropped; s0-items-6) / every other non-dropped C3 cell of the publication / the publication's other extractants only; same seed and design; no donor at the pair: OFF, OFF_ALL = B1, OFF_PUB, OFF_XEXT = NaN |
| `NEAR_r4`, `NEARSH_r4` | nearest-condition sibling(s) (\|dlog10 acid\| + \|dlog10 L\| + 1[diluent differs]; concentrations floored at 1e-4 M; ties averaged), shrunk with n_tied / (n_tied + K) |
| `E_RAND1_raw_r{0,2,3,4}`, `E_RAND1_sh_r{2,3,4}` | exact expectation over one uniformly drawn sibling, raw and shrunk (w = 1 / (1 + K)) |
| `E_RAND2_raw_r{0,2,4}`, `E_RAND2_sh_r{2,4}` | the same over every sibling pair (w = 2 / (2 + K)); n_sib >= 2 |
| `E_SUB{1,2,3,5}_raw_r4`, `E_SUB{m}_sh_r4` | n_sib >= 5: all m-subsets or 5,000 seeded random subsets; raw SIB_m and SH_m (w = m / (m + K)); m = all is SIB_r4 / SH_r4 |
| `PB_CHEM`, `PB_XEXT`, `PB_PUB` (+ `_raw`) | placebo template at the target's own w: same chemotype other publications / same extractant other publications / other extractants same publication; scored where the pool exists (target n_sib >= 1) |
| `E_PB_{CHEM,XEXT,PUB}_CM` (+ `_raw`) | count-matched placebo: n_sib-subsets of the placebo pool (all or 5,000 seeded), blended at w = n / (n + K) (the arm's weights); NaN when the pool has fewer than n series |
| `PB_LABX` (+ `_raw`) | same extractant, same laboratory (frozen PROVISIONAL key, conservative id), other publication |
| `PB_PUBHOM`, `PB_PUBHET` (+ `_raw`) | PB_PUB split: same chemotype (homologue) / other chemotype (M06 annex) |
| `ACOPY_r0`, `AINT_r0` | shift-1: mean / per-metal interpolation in log10[acid] (flat beyond) of the acid-only BIT siblings; `dx_mean_abs`, `dx_near`, `acid_position` (interior / above / below) are metadata |
| `NP_LAD_r{2,3,4,5}` (+ `np_lad_src_r*`) | E-REC frozen no-prior policy: recipe-2 ladder La,Ho / La,Nd,Ho / La,Nd,Er,Lu / La,Nd,Gd,Er,Lu when measured and legal (gen20k legal_metals in ladder order), else legal HULL (lightest, heaviest, then max-min -CN8 radius, ties to the lower Z); B1 form with the fold beta |
| `NP_DOPTQ_r5` | B1 at the 5-read DOPT_Q set seq[:4] (k_max >= 4); with B1_r2..4 the DOPT_Q no-prior column |
| `NB_*`, `V1_*` | COPY, SH0, SIB_r2..4, SH_r2..4, SHK2_r4, OFF (+ NB_OFF_ALL) on P1's CLEAN pool without the bit rule / the V1 pool |
| `I1_*`, `SD_*`, `OFS_*`, `QNT_*`, `V25_*` | COPY, SIB_r4, SH_r4 on: CLEAN minus the item-1 link / BIT minus shape-sd < 0.02 links (label-based, E6) / BIT minus `r5.dupes.offset_scan` OFFSET or NEAR_ID links / BIT minus Q_SMALLINT or Q_COARSE cells (`quant_scan`; such targets dropped) / V2-5 floor readings removed (coverage recomputed; targets whose 4-read set holds a floor reading and pairs touching a floor metal dropped via `v25_row_ok`; beta unchanged) |
| `OFS_PB_{CHEM,XEXT,PUB,PUBHOM,PUBHET,LABX}` (+ `_raw`), `E_OFS_PB_{CHEM,XEXT,PUB}_CM` (+ `_raw`), `OFS_OFF_{ALL,PUB,XEXT}` (+ `_np`) | duplicate sensitivity of the control pools (REGISTRY 1.1-S20; s0-items-1): every placebo and OFF pool minus the OFFSET / NEAR_ID links of `offset_scan`, at the OFS pool's own w (count-matched to the OFS pool; RNG `ECT|CM_OFS_<pool>|<cell_id>`); OFS_OFF_ALL exists only with an OFS sibling. **The primary pools follow the frozen bit rule; the OFS pools are the registered duplicate sensitivity.** |
| `OFS_NEAR_r4`, `OFS_NEARSH_r4`, `OFS_ACOPY_r0`, `OFS_AINT_r0` | near siblings re-selected on the OFS pool (the population.py nearest-condition rule) and the acid-only pool minus OFFSET / NEAR_ID links (REGISTRY 1.1-S21; s0-items-2) |
| `INWIN_*`, `OUTWIN_*` | M04: SH on BIT siblings whose V1 log D lies inside / outside [-2, 2] at every target metal |
| `geom_same` | read-alignment flag: every BIT sibling is a C3 OFF donor read at the target's read set in the same fold with a row for the pair (there SH = OFF exactly) |

## 6. Tables (results/ECT/, one CSV per design; every row labelled)

`populations`, `comparators` (every stratum; R2 seed mean / min / max / per seed, MSE, MAE, balanced MAE, relative SSE
vs B1_r4), `placebos` (+ count-matched), `ladder` (0 / 2 / 3 / 4 reads, identical rows), `subm` (dose-response),
`nsib` (between-cell strata 1, 2, 3-4, 5-7, >= 8; descriptive), `pairtype`, `pairexcl` (all / without Ce pairs /
without Gd pairs / without both; Ce and Gd shares), `condclass` (acid concentration only, acid type +- concentration,
diluent only, extractant concentration only, temperature only, mixed; diluent-only and extractant-concentration-only
flagged), `alignment` (DIAGNOSTIC), `mechanism` (SH - OFF_XEXT, SH - PB_PUB, SH - E_PB_PUB_CM pooled and per
publication, on each comparator's rows and on the rows where both exist; recovered share; and the same contrasts on
the OFS pools, OFS_SH_r4 - OFS_OFF_XEXT / OFS_PB_PUB / E_OFS_PB_PUB_CM, column `pool_variant`, SENSITIVITY; the
placebos, the M06 annex and the PB_LABX row likewise; `dupsens` adds the OFS control-pool contrasts and OFS_NEAR rows;
`annex_shift1` the OFS ACOPY / AINT rows), `inference`, `units`,
`influence`, `dupsens`, `v25`, `annex_shift1`, `annex_erec`, `annex_m02`, `annex_m04`, `annex_m06`, `lab`, `rules`;
plus `rows_<design>.parquet`, `populations.csv`, `links.csv`, `summary_<design>.json`, `PER_ITEM_TABLE_<design>.md`
(section 9.3), `repro_<design>.json`, and `results/leakage/ECT_<design>.json`.

Labels: PRIMARY rows EXPLORATORY (seen); WITH_C5, GE1, NOBIT, V1, duplicate and V2-5 variants SENSITIVITY; EQ1 and the
report rules CONSISTENCY CHECK; the read-alignment identity DIAGNOSTIC; reproduction TECHNICAL SMOKE. No ORACLE arm.

## 7. Inference (descriptive; section 6.1)

Unit = publication x chemotype; c_u = mean_s (SSE_ref,u,s - SSE_cand,u,s) / SST_s (sum = pooled dR2). Exact one-sided
sign-flip over units (full enumeration; up to 22 units, which covers every registered stratum); the chemotype-block
sign-flip (floor 1 / 2^G_chem, not a strength measure); SIGN (ties = exact 0 after |c| <= 1e-12); IM t on the per-unit
relative SSE and on the log SSE ratio; leave-one-publication-out jackknife with t(P - 1) (r5.common.lopo); top-3 unit
share; equal-weight per-publication relative SSE. Cluster-level inference: "not estimable" (3 cpg clusters).
Contrasts (every stratum of `t_inference`): SH-B1, SH-OFF (read-alignment effect only), SH-OFF_ALL, OFF-B1, SIB-B1,
COPY-B1, SH0-B1, SH@3-B1@4, SH@2-B1@4, SH@2-B1@2, SH@3-B1@3, SH-NEAR, SH-NEARSH, SH-SIB, SH-RAND1sh, SIB@2-SIB@4,
SIB@2-COPY, SIB@2-B1@4; on placebo rows SH-PB_* and SH-E_PB_*_CM; the registered mechanism contrast SH-OFF_XEXT and
SH-PB_PUB (and SH-E_PB_PUB_CM); SH-PB_LABX. Influence rows: pub_b0b3296ec5 (share of SH-B1 and SH-B1 without it),
pub_19783c039b (share of SH-OFF and SH-OFF without it), c950dc2982e96b83 in WITH_C5.

## 8. Report rules (in-corpus consistency checks; no kill fires in-corpus; the decisive leg is C-ECT, section 7.5)

| rule | statistic | threshold / outcome |
|---|---|---|
| SH - B1 on PRIMARY | dR2 with LOPO interval; also PRIMARY_NOBIT and PRIMARY_I1 | reported (seen: +0.177 [+0.076, +0.279] without the bit rule; +0.168 with the item-1 pair dropped) |
| harm rule (ii) | EQ1 relative SSE of SH vs B1 | >= -0.05 passes; else "the deployed rule depends on sibling count" (seen +12.2 %) |
| artefact flag (iii) | Ce-pair share of the SH - B1 SSE gain (PRIMARY) | >= 0.50: reword as "transfer of a laboratory fingerprint" (seen 32.8 %) |
| mechanism wording | SH - OFF_XEXT, SH - PB_PUB per publication; families with an other-extractant pool | default wording "same-campaign information (extractant transfer and campaign fingerprint not separated)"; "campaign fingerprint" needs >= 4 families AND a threshold set before data, which in-corpus never holds |
| claim wording | SH vs SIB and SH vs NEAR unit W/L/T | claim reads "own-series template + 3-4 reads", not "K = 1.78" |
| SH with 2 reads | SH@2 - B1@4 units, top-3 share | stays excluded from claims (section 4.3 item 4) |
| comparator | B1 on the same rows | E-CT values never next to 0.452 |

## 9. Descriptive annex rows (each PLANNED or EXCLUDED in REGISTRY.csv at the Stage-0 seal)

- shift-1: ACOPY@0, COPY@0, AINT@0, SH@2, SH@4, B1@4, OFF by |dlog10 acid| bin (<= 0.15, 0.15-0.30, 0.30-0.60,
  > 0.60) x interior / above / below, for both distance metrics (dx_mean, dx_near), on all / without C5 / without
  pub_19783c039b / without both; targets with >= 1 acid-only sibling (GE1_WITH_C5).
- recipe-1 E-REC: SH_own@0 (COPY), @2, @3, @4, SH@2 with m = 1 / 2 random own siblings, against NP_LAD (the frozen best
  label-free no-prior policy) and NP_DOPTQ at 2-5 reads; strata n_own = 1, n_own >= 2, the giant cpg cluster, the
  rest; with and without pub_b0b3296ec5; reads-equivalent = the smallest r with NP@r >= SH@2. The 5-read rung is a
  different read budget and uses the 4-read fold beta (the per-k refit belongs to E-D, item 1.6).
- M02: SIB@2 against SIB@4, COPY and B1@4.
- M04: "avoid floor or saturated reference conditions; average at least 2": INWIN vs OUTWIN references on rows with
  both; one vs two random own siblings (RAND1 vs RAND2, shrunk and raw).
- M06: same campaign (PB_PUB) against other publications (PB_XEXT, PB_CHEM); homologue (PB_PUBHOM) vs other-chemotype
  (PB_PUBHET) extractants of the same publication; the homologue channel is a frozen lead with an external test only.
- Also run (not run by P1): V2-5 floor readings dropped (`v25`); laboratory annotation (`lab`, PROVISIONAL key).

## 10. RNG seeds (registry scheme `paths.seed_for(key)` = 20261023 + crc32(key))

| procedure | key | note |
|---|---|---|
| SUB m random subsets | `ECT\|SUB\|<cell_id>\|m<m>` | only when C(n, m) > 5,000; fixed per cell across seeds and designs |
| count-matched placebos | `ECT\|CM_<CHEM\|XEXT\|PUB>\|<cell_id>` | only when C(N, n) > 5,000 |
| random sign-flips | `ECT\|<stratum>\|<cand>-<ref>\|unit` / `\|chem`, `ECT\|LAB\|<stratum>` | only when G > 22 (no registered stratum) |
| intervention noise | `ECT\|INTERVENTION\|<design>` | |
| smoke surrogates | `ECT\|SMOKE_SURROGATE\|<design>`, `ECT\|SMOKE_TABLES\|<design>` | technical smoke only |

## 11. Leakage and controls (section 8)

- Isolation and fold betas: round 4's (REF runs pass r4_decide's intervention / identity rule; `r5.common.load_rows`).
- Prediction intervention (`Corpus.intervention`): (1) every metal outside the union of every read set the arms use
  replaced by noise in the target's row: every prediction column bit-identical; (2) every metal outside the 4-read
  DOPT_Q set replaced: every column except the larger-budget NP_* columns bit-identical.
- Positive controls (must be caught on every target with a sibling): the target itself in its own BIT pool (the guard
  raises), a bit-identical copy of the target in its pool (the guard raises), the target in its own pool with the
  guard off (the intervention detects it). Smoke, BPG seed 104729: 229 targets, 0 differences, 130 / 130 / 130 caught.
  A pass needs every dev seed and at least one target and positive control (`--intervention-seeds` is smoke-only).
- Support: nested read sets, every DOPT_Q step inside `legal_metals`, no scored pair with both ends read, B1 exact.
- Fit poisoning: not applicable (no fit; beta is the stored round-4 fold fit, K fixed). Cache poisoning: not
  applicable (no new cache).

## 12. Decisions the brief leaves open, frozen here

1. OFF's base is the recomputed B1_r4 (equal to the parquet B1 within 8.9e-16), so the same code runs on new data.
2. Placebos and OFF / OFF_ALL exist only for targets with n_sib >= 1 (as P1).
3. SUB m and count-matched subsets use the registry seed scheme (P1 used SHA-256 seeds; SUB m is not a reproduction
   target).
4. The E-REC no-prior policy is frozen as NP_LAD; NP_DOPTQ is reported beside it.
5. M04's window is [-2, 2] at every target metal (the seen M09 window).
6. Quantised cells = Q_SMALLINT or Q_COARSE; offset copies = OFFSET or NEAR_ID links (r5.dupes, the Stage-0 scans).
7. Exact sign-flip enumeration up to 22 units; r5.common's random 4,999 flips beyond.
8. If REGISTRY.csv marks an annex row EXCLUDED, its table is dropped before the item seal (an edit to
   `ect_report.all_tables`, recorded in the seal); no annex row is optional after the seal.
