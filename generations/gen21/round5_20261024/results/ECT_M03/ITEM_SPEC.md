# Item 1.2, M03 sibling-weighting factorial (E-CT annex): frozen specification

Stage-1 item 1.2 of `round5_design_20261023/CLAUDE_CODE_PROMPT.md` (section 6.2; conventions 6.0, 6.10, 8, 9).
Estimand: **E-CT annex**. Question: does how siblings are weighted change E-CT accuracy beyond the plain shrunk mean SH
(r5/ect.py, K = 1.78)? IR M03 saw a kernel gain over plain empirical Bayes of +0.038, but +0.007 with C5-BTBP excluded
(7 of 11 publications); the level rule gained only at n_sib >= 5. **Expected outcome: "uniform shrunk mean".**

Status: written before the item seal. **No round-5 score had been computed when this file was written**: only the
technical smoke (`--smoke`: rows built on real data, then y, the templates and every prediction replaced by seeded
noise; every metric suppressed) has run on real data. Every in-corpus number this item produces will be EXPLORATORY
(seen). None is placed next to E-A's 0.452; B1 and SH on the same rows are the comparators.

Code: `r5/m03.py` (arms, nested selection, nulls, tables, rules, leakage checks), `scripts/s1_m03.py` (driver),
`tests/test_m03.py`. Built on `r5/ect.py` (Target, read-corrected templates T_j, B1, SH, SIB, COPY, SH0, NEAR, RAND2,
ACOPY, AINT, pool guard) and `r5/population.py` (BIT pools, cond__ distance, change classes, strata). Adapted from
`ideas2/scripts/recipe_acidfit.py` (LF SHA-256 `417280dd…7e832f`), `recipe4_acid_skA.py` (`59c48737…279778`),
`recipe4_skepB_check.py` (`45771a80…9266ab`), and the acid-kernel grid of IR M03's test design
(`ideas/critiques.json`, M03); full digests in the module docstring.

## 1. Order of operations (section 1.2)

1. Stage-0 seal (`scripts/r5_seal.py --seal`), including this file, the code above and `REGISTRY.csv`.
2. Item code seal (this file + code digests in the manifest).
3. Reproduction run, labelled TECHNICAL SMOKE: `--repro` (section 9 below). A mismatch stops the item; a fix by code
   change is resealed by an append-only amendment listing what was seen.
4. Arms: the full run, BPG (decides labels) and BP (sensitivity).

## 2. Exact command lines (from `D:\ml_separator_gh`)

```
# technical smoke (no seal; every metric suppressed; 3 permutations per null; output: SMOKE_COUNTS_<design>.json)
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py --smoke --design BPG
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py --smoke --design BP

# after the seal: reproduction targets (TECHNICAL SMOKE), then the arms (B = 500 permutations per null)
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py --repro --design BPG
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py --repro --design BP
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py --design BPG
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py --design BP

# tests (synthetic / construction-only)
PYTHONIOENCODING=utf-8 D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_m03.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
```

Defaults of the full run: `--perms 500` (the script refuses fewer than 200), r5.ect identity checked on all 5 dev
seeds, intervention on all 5 dev seeds. After every run: `round4_20261022/scripts/r4_seal.py --verify` must still
report 5bacfe4c and the round-3 LF digests must match (section 8). Smoke timings per design: build 21 s (both pools,
r5.ect identity on 1 seed), intervention 3.4 s per seed, tables 47 s, nulls 0.017 s per permutation (about 35 s for
500 x 4 nulls), fit poisoning 9 s; the full run is expected at about 4 minutes per design.

## 3. Populations (nested selection is re-run inside each; counts from the smoke, identical in BP and BPG)

| population | pool | definition | cells / publications / pub x chemotype units / rows per seed | label |
|---|---|---|---|---|
| **PRIMARY** | BIT | CLEAN + section-5.1 bit rule, n_sib >= 2, without C5-BTBP (training and scoring) | 99 / 12 / 14 / 690 (3 cpg clusters) | EXPLORATORY (seen) |
| PRIMARY_NO_ORNL | BIT | PRIMARY without pub_19783c039b: **the deploy-rule population** ("without the two named publications") | 85 / 11 / 13 / 581 | SENSITIVITY (rule rows CONSISTENCY CHECK) |
| WITH_C5 | BIT | n_sib >= 2 with C5-BTBP in training and scoring | 114 / 13 / 15 / 838 | SENSITIVITY |
| WITH_C5_NO_C5CELL | BIT | WITH_C5 without the single cell c950dc2982e96b83 (single-cell influence) | 113 / 13 / 15 / 828 | SENSITIVITY |
| PRIMARY_NOBIT | CLEAN without the bit rule | P1's PRIMARY pool (duplicate sensitivity: the item-1 and item-3 copies back in) | 99 / 12 / 14 / 690 | SENSITIVITY |
| PRIMARY_OFS | OFS (BIT minus OFFSET / NEAR_ID links) | duplicate sensitivity (s0-items-2): n_sib >= 2 on the OFS pool, without C5; acid pool minus those links; near siblings re-selected on it; nested selection re-run; the level rule and the acid-trend claim statistic on it | counted in the run (`populations` table) | SENSITIVITY |
| PRIMARY_NO_ORNL_OFS | OFS | PRIMARY_OFS without pub_19783c039b: the deploy rule (BOTH, ACID, READ) on the OFS pool | counted in the run | SENSITIVITY |

Acid-trend rows (>= 2 acid-only BIT siblings at >= 2 acid levels): PRIMARY 57 cells, PRIMARY_NO_ORNL 52, WITH_C5 72.
Dense grids (n_sib >= 5): PRIMARY 60 cells. Information contract: the target's legal 4 DOPT_Q reads, the siblings' full
series, acid metadata and the fold's beta; a publication's hyperparameters are chosen on the other publications only.

## 4. Frozen arm list (all on the identical 4-read E-CT rows; code k = 3; n = |pool|; K = 1.78)

**Kernel factorial** {uniform, acid kernel, read-match kernel (offset removed), both} x count shrinkage, frozen as one
product grid (column names `KW_a<ell_acid>_r<ell_read>_<sh|raw>`, 25 x 2 = 50 fixed arms):

- weight of sibling j: `w_j = exp(-|dlog10 acid_j| / ell_acid) * exp(-d_read_j / ell_read)`; ell = inf gives factor 1.
- `|dlog10 acid|`: concentrations floored at 1e-4 M; one side missing 1, both missing 0 (r5.population's cond_dist rule).
- `d_read_j` = RMS over the target's 4 DOPT_Q read metals of `(y_S - Y_j,S) - mean_S(y_S - Y_j,S)` (offset removed).
- raw: `sum_j w_j T_j / sum_j w_j`, T_j the read-corrected template of r5.ect (`T_j + PWL_R[y_S - T_j,S]`).
- sh (count shrinkage): `w B_kernel + (1 - w) B1_r4` with `w = n_eff / (n_eff + K)`, `n_eff = (sum w_j)^2 / sum w_j^2`.
- **Grids** (frozen; each includes the uniform limit ell = inf):
  `ell_acid` in {inf, 1, 0.5, 0.25, 0.1} decades of [acid] (IR M03's grid);
  `ell_read` in {inf, 0.4, 0.2, 0.1, 0.05} log10 D units.
- Families: **ACID** = the 5 points with ell_read = inf; **READ** = the 5 points with ell_acid = inf; **BOTH** = all 25.
  **Uniform limit** (inf, inf) = `SIB_r4` (raw) and `SH_r4` (sh) of r5/ect.py exactly (the computed value is replaced
  by r5.ect's; the smoke deviation before replacement is 0.0).
- IR M03's other-extractant level (rho) is EXCLUDED here (it is OFF_XEXT / PB_PUB, item 1.1); its kappa grid is
  replaced by the two frozen shrinkage levels (none, K = 1.78 on the Kish count).

**Nested arms** `NEST_<ACID|READ|BOTH>_<sh|raw>[_classic|_argmin]` (18 columns): nested leave-one-publication-out
choice inside the family, defined in section 5. The primary nested rule is `NEST_BOTH_sh` (vs `SH_r4`).

**2-nearest arms** (raw, and `_sh` = blend with B1_r4 at w = 2 / (2 + K)):
`NEAR2` (the 2 condition-nearest siblings, distance = r5.population cond_dist: |dlog10 acid| + |dlog10 extractant
concentration| + 1[diluent differs]), `LEVEL2` (2 nearest in level at the 4 reads, RMS of y_S - Y_j,S: the "level
rule"), `READ2` (2 nearest in read pattern, d_read). Ties at the 2nd distance (np.isclose, atol 1e-9, the population's
near-tie rule): the exact expectation over random tie-breaking, `(sum of strictly nearer + k mean of tied) / 2`.

**Context arms** (bit-identical to r5.ect's Corpus.target_block on every group; smoke max difference 0.0):
`B1_r4`, `SH_r4`, `SIB_r4`, `COPY`, `SH0`, `NEAR_r4`, `NEARSH_r4`, and the **matched-count random-2 control**
`E_RAND2_raw_r4`, `E_RAND2_sh_r4` (exact expected squared error over every sibling pair).

**0-read acid-trend arms** (acid-only BIT siblings: cond__ differs in acid concentration only; rows with >= 2 such
siblings at >= 2 distinct log10 acid values; I2 shift-2, recipe-4):

| arm | computation |
|---|---|
| `ACOPY_r0` (MEAN) | mean of the acid-only siblings' log SF (r5.ect) |
| `AINT_r0` | piecewise-linear interpolation in log10 acid, flat beyond, cells at one level averaged (r5.ect) |
| `PAIRMEAN_r0` | mean of the two bracketing acid levels (levels = log10 acid rounded to 4 decimals, level means); NaN when the target is not bracketed |
| `LIN_r0` | OLS line of log SF on log10 acid over the siblings, at the target acid |
| `LINCLIP_r0` | LIN with the target acid clamped to the siblings' range |
| `LINHALF_r0` | MEAN + 0.5 (LIN - MEAN) (half-shrunk slope; recipe-4's lambda = 0.5, fixed) |
| `LINHALFCLIP_r0` | MEAN + 0.5 (LINCLIP - MEAN) |
| `POOLPAIR_r0` | MEAN + b_pair,-pub (clamped target acid - mean sibling acid); b = sum sxy / sum sxx over the unique (target, pair) sibling sets of the population's **other** publications (0 when sum sxx = 0); computed per population |

## 5. Selection (section 6.2) and the one-SE rule toward uniform

For each population and outer publication p: the loss of every family grid point on the training publications Q (all
others) is the sum over q in Q of the seed-mean SSE of q's rows (the predictions of a grid point do not depend on
training data; the β is the stored round-4 fold fit). Best = the minimal loss (exact ties within 1e-12 relative form
the best set). One-SE set:

- `paired` (**primary**): every config c with `L_c - L_min <= sqrt(|Q|) sd_q(SSE_c,q - SSE_best,q)` (paired SE of
  the summed excess; the best set's mean per-publication SSE as reference).
- `classic` (SENSITIVITY): `L_c - L_min <= sqrt(|Q|) sd_q(SSE_best,q)` (the SE of the best config's summed loss).
- `argmin` (SENSITIVITY): no one-SE rule; the best set itself.

Inside the one-SE set: the least complex (number of finite ells, then the summed grid steps away from inf), then the
lowest loss; **exact ties are averaged** (prediction mean of the tied configs), never broken by table order. The chosen
configs, `uniform_chosen`, L_min, L_uniform, SE_uniform and the one-SE set size are reported per outer publication
(`choices_<design>.csv`). Leave-one-publication-out intervals of nested contrasts **re-run the whole nested selection**
without the left-out publication (`lopo_renested_*`); the row-drop LOPO of r5.ect_report.infer is reported beside it.

## 6. Selection-aware nulls (section 6.2; DIAGNOSTIC; no in-corpus p-value is called significant)

| null | procedure | replicates | populations | statistic |
|---|---|---|---|---|
| acid permuted within publication | the finite acid concentrations of each publication's bench cells permuted among those cells (missing stay missing; the same permutation for targets and siblings, all seeds); acid distances recomputed; the full nested selection re-run (primary rule) | B = 500 | PRIMARY, PRIMARY_NO_ORNL | dR2 of NEST_ACID_sh and NEST_BOTH_sh vs SH_r4; uniform-chosen folds; p = (1 + #{null >= obs}) / (1 + B) |
| permuted read | each target's read distances shuffled among its siblings (one permutation per target, fixed across seeds); re-selected | B = 500 | PRIMARY, PRIMARY_NO_ORNL | dR2 of NEST_READ_sh and NEST_BOTH_sh vs SH_r4 |
| matched-count random-2 | `E_RAND2_{raw,sh}_r4` (exact expectation over every sibling pair) against NEAR2, LEVEL2, READ2 at the same count | exact | every population | contrasts in `inference` |

## 7. Strata (section 6.2; descriptive slices of each population's nested predictions; selection is not re-run inside a stratum)

ACID_INTERP (target acid inside its BIT siblings' acid range), ACID_EXTRAP (outside), CONST_ACID (every BIT sibling at
the target's acid: constant-acid families), ACID_UNKNOWN (acid missing; 0 rows on V1); DENSE (n_sib >= 5) and SPARSE
(2-4); DILUENT (change class diluent only) vs ACIDCHG (acid concentration only or acid type +- concentration) vs
OTHERCHG; DGA (chemotype sc009) vs NONDGA. Reported for PRIMARY (EXPLORATORY) and WITH_C5 (SENSITIVITY). The acid-trend
arms are reported on all acid-trend rows, the bracketed rows (interior), the exterior rows and the edge rows (inside the
range, not bracketed).

## 8. Report rules (in-corpus consistency checks; IR M03, I2 recipe-4; REGISTRY threshold +0.01 pooled R², seed mean)

| rule | population | statistic | outcome |
|---|---|---|---|
| **deploy rule** (BOTH family) | PRIMARY_NO_ORNL (C5-BTBP and pub_19783c039b out) | dR2 NEST_BOTH_sh - SH_r4; publications with dSSE > 0 (seed means); folds choosing ell = inf (uniform) | **deploy the uniform shrunk mean SH** if dR2 < +0.01, or fewer than 7 of the remaining 11 publications improve, or uniform is chosen in more than half the folds; else "nested weighting passes the in-corpus consistency check (EXPLORATORY; no claim)". CONSISTENCY CHECK |
| deploy rule, ACID / READ families; on PRIMARY; WITH_C5; PRIMARY_NOBIT | as named | same | SENSITIVITY rows |
| **level rule** | PRIMARY (raw forms) | share of publications with a target at n_sib >= 5 (rows of those targets) where LEVEL2_raw beats both SIB_r4 (plain mean) and NEAR2_raw (seed-mean SSE per publication) | retired if the share < 0.60; CONSISTENCY CHECK. Shrunk forms (LEVEL2_sh vs SH_r4 and NEAR2_sh), PRIMARY_NO_ORNL and WITH_C5: SENSITIVITY |
| **acid-trend claim statistic** | PRIMARY bracketed acid-trend rows | R2(LINHALF) - max(R2(MEAN), R2(PAIRMEAN)), LOPO interval | reported (sign, LOPO); in-corpus no claim. PRIMARY_NO_ORNL, WITH_C5: SENSITIVITY |
| substitution | every table | B1_r4 and SH_r4 on the same rows | E-CT annex values never next to 0.452 |

## 9. Reproduction targets (section 6.0; TECHNICAL SMOKE; a mismatch stops the item)

| target | rows | expected | tolerance |
|---|---|---|---|
| B1 recomputed from the fold betas vs the REF parquet | every M03 row (4,190 BIT + 3,450 NB per design, plus the OFS pool) | max abs diff 8.9e-16 (smoke: 8.9e-16 BPG, 8.9e-16 / 4.4e-16 BP) | <= 8.9e-16 (brief 6.0); y exact; read sets identical; on the OFS pool every shared arm equals r5.ect's OFS_* column exactly |
| shared arms vs r5.ect Corpus.target_block | every group, all 5 seeds | B1, SH, SIB, COPY, SH0, NEAR, NEARSH, E_RAND2 (raw, sh), ACOPY, AINT; NB: SH, SIB, COPY, SH0 | exactly 0 (smoke: 0.0 on seed 104729) |
| the kernel's uniform limit | every group | = SIB_r4 / SH_r4 | <= 1e-12 before replacement (smoke: 0.0) |
| P1's PRIMARY without the bit rule through this module's kernel code (NB pool, uniform limit computed, not replaced) | PRIMARY_NOBIT: 99 / 12 / 14 / 690 | B1_r4 0.5309, uniform sh = SH 0.7082, uniform raw = SIB 0.6987 (BPG) | 5e-5; BP: counts and identities only |

## 10. Outputs (results/ECT_M03/; every row labelled per section 9.1)

`rows_bit_<design>.parquet`, `rows_NB_<design>.parquet` (every arm incl. the 50 grid points, strata flags),
`nested_rows_<design>.parquet` (every nested arm per population, long format), `populations`, `comparators` (R2 seed
mean / min / max / per seed, MSE, MAE, balanced MAE, relSSE vs B1_r4), `grid` (every fixed grid point: R2, dR2 vs its
uniform limit, per seed, publications improving; plus the **best fixed grid point in hindsight, labelled ORACLE [O]**,
headroom only), `choices` (per outer fold), `inference` (r5.ect_report.infer: per-seed dR2, relSSE, unit W/L/T, exact
sign-flip over pub x chemotype units and chemotype blocks, SIGN, IM t, row-drop LOPO with t(P - 1), top-3 share,
equal-weight per-publication relSSE; + `lopo_renested_*` for nested arms), `units` (unit table, c_u per unit),
`strata`, `acid`, `acid_claim`, `level_rule`, `nulls`, `null_draws`, `rules` (`<table>_<design>.csv`),
`grid_definition.csv`, `PER_ITEM_TABLE_<design>.md` (section 9.3), `summary_<design>.json`, `repro_<design>.json`,
`SMOKE_COUNTS_<design>.json`; leakage report `results/leakage/ECT_M03_<design>.json`.

Labels: PRIMARY EXPLORATORY (seen); other populations, the classic / argmin one-SE rules and the shrunk level rule
SENSITIVITY; rules CONSISTENCY CHECK; nulls and choices DIAGNOSTIC; the hindsight grid optimum ORACLE [O];
reproduction TECHNICAL SMOKE. Contrasts reported in every population: NEST_{BOTH,ACID,READ}_sh - SH_r4,
NEST_{BOTH,ACID,READ}_raw - SIB_r4, NEST_BOTH_sh_{classic,argmin} - SH_r4, NEST_{ACID,READ}_sh_classic - SH_r4,
SH_r4 - B1_r4, NEST_BOTH_sh - B1_r4, {NEAR2,LEVEL2,READ2}_raw - SIB_r4, LEVEL2_raw - NEAR2_raw,
{NEAR2,LEVEL2,READ2}_raw - E_RAND2_raw_r4, {NEAR2,LEVEL2,READ2}_sh - SH_r4, {NEAR2,LEVEL2}_sh - E_RAND2_sh_r4;
acid-trend rows: LINHALF, PAIRMEAN, AINT, LINCLIP, LINHALFCLIP, POOLPAIR, LIN and SH_r4 against MEAN, LINHALF and AINT
against PAIRMEAN (bracketed). C5-BTBP in / out = PRIMARY vs WITH_C5; the c950dc2982e96b83 influence =
WITH_C5_NO_C5CELL; pub_19783c039b = PRIMARY_NO_ORNL. Cluster-level inference: not estimable (3-4 cpg clusters).

## 11. RNG seeds (registry scheme `paths.seed_for(key)` = 20261023 + crc32(key))

| procedure | key |
|---|---|
| acid permutations within publication | `M03\|ACIDPERM\|<population>` (the same draws for BPG and BP) |
| read-distance permutations | `M03\|READPERM\|<population>` |
| random sign-flips (only when > 22 units; none registered) | `M03\|<population>\|<cand>-<ref>\|unit` / `\|chem` |
| intervention noise | `M03\|INTERVENTION\|<design>\|<pool>` |
| fit-poisoning noise | `M03\|FITPOISON\|<design>` |
| smoke surrogates | `M03\|SMOKE_YS\|<design>\|<pool>`, `M03\|SMOKE_SURROGATE\|<design>\|<pool>` (technical smoke only) |

## 12. Leakage and controls (section 8)

- Isolation, fold betas and the read sequences: round 4's (REF runs pass r4_decide's rule; `r5.common.load_rows`).
- Prediction intervention (`m03.intervention`, all 5 seeds in the full run): every metal outside the target's 4-read
  DOPT_Q set replaced by noise; every M03 input and arm (templates, B1, distances, every grid point, 2-nearest arms,
  acid-trend arms) must be bit-identical. Positive controls, all caught on every group in the smoke (BPG and BP, seed
  104729: BIT 114 / 114 / 114 / 114, NB 99 / n.a. / 99 / 99): the target in its own pool (guard raises), a
  bit-identical copy (BIT pool; guard raises), the target in its own pool with the guard off (the kernel arms change),
  read distances from the target's full series (they change).
- Fit poisoning (`m03.fit_poisoning`, section 8 item 2 "M03's nested selection"): for every population, family,
  shrinkage, one-SE rule and outer publication, the held-out publication's labels and predictions are randomised; the
  selection (chosen configs, L_min, SE) must be bit-identical (smoke: 1,098 / 1,098), and the POOLPAIR slope of each
  publication's rows after randomising its own sxy (smoke: 33 / 33). Positive control (s0-mechanics-16): the same
  selection with the outer publication's own losses included (a leak) must change for at least one fit of every kernel
  family, and a leaky POOLPAIR slope that includes the publication's own sxy must change; the check passes only then.
- Order: the reproduction target runs first in every scoring run and the intervention before any scored computation;
  a mismatch or a failed check stops the item before any scored write (s0-mechanics-3, -17); `--ect-check-seeds` and
  `--intervention-seeds` are smoke-only (s0-mechanics-5).
- Support: B1 exact, y exact, nested read sets of the round-4 cache, no scored pair with both ends read (0 rows).
- Cache poisoning: not applicable (no new cache).

## 13. Decisions the brief leaves open, frozen here

1. Kernel forms: exponential (Laplace) in |dlog10 acid| and in the offset-removed read RMS; product kernel for "both".
   The read grid {0.4, 0.2, 0.1, 0.05} is fixed a priori (no look at the distance distribution).
2. Count shrinkage has two levels: none, and K = 1.78 on the Kish effective count (uniform limit = SH exactly).
3. The factorial runs at 4 reads only (the frozen E-CT budget); 2- and 3-read factorials are EXCLUDED.
4. Selection uses seed-mean per-publication SSE, one choice per outer publication shared by the 5 seeds.
5. The primary one-SE rule is the paired SE of the summed excess; classic and argmin are sensitivities.
6. "Publications improve" = publication-level dSSE > 0 (seed means); a fold choosing uniform gives dSSE = 0 exactly.
7. The level rule uses raw forms (which siblings, not how much shrinkage); the shrunk forms are a sensitivity.
8. The acid-trend claim statistic is evaluated on the bracketed rows (where PAIRMEAN exists); lambda = 0.5 is fixed.
9. POOLPAIR pools the sibling-set slopes of the population's other publications (unique target x pair records).
10. The duplicate sensitivities are P1's CLEAN pool without the bit rule (PRIMARY_NOBIT; the duplicates added back)
    and the OFS pool (PRIMARY_OFS, PRIMARY_NO_ORNL_OFS; offset / near-identity links removed, s0-items-2); acid-trend
    arms run on the BIT and OFS pools.
