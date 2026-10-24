# Metrics (round-5 addendum; code `r5/common.py` and `r5/confirm.py`, on top of R4-METRICS)

**What stays in force.** Round 4's `METRICS.md` (`../round4_20261022/METRICS.md`) still defines E-A's level statistics: R², balanced MAE, row MAE, RMSE and the subset tables. Its paired-contrast columns (percentile, BCa, cluster-jackknife t(G − 1)) survive in round 5 only as comparators (PROTOCOL §6).

**What this addendum adds:**
- the one-currency relative pair-SSE (brief §4.1);
- the per-unit statistics (brief §9.2);
- the M34 tests (brief §4.2), with the SF_POOL enumeration rule;
- LOPO and the top-3 share;
- what was dropped (JK_G1);
- contrast-specific MDEs;
- the confirmation-package statistics (brief §4.3, §7.5).

Item-level details are in each `results/<ITEM>/ITEM_SPEC.md`. Labels follow PROTOCOL §4.

## 1. Conventions

- **Signs.** Every contrast is "candidate − reference", signed so that a positive value favours the candidate:
  - ΔSSE = SSE_ref − SSE_cand;
  - Δbalanced-MAE = MAE_bal(ref) − MAE_bal(cand);
  - ΔR² = R²(cand) − R²(ref).
- **Seeds.** Every statistic is computed inside each split seed (104729, 130363, 155921, 196613, 262147), then averaged over the seeds.
  - The seeds share cells, so per-seed standard errors are never read as precision, and "k of 5 seeds" is not a replication count [P3 §1].
  - BP and BPG are not independent; BPG decides labels (PROTOCOL §2).
- **Rows.** Every arm of a contrast is scored on identical rows and identical supports. Denominators are fixed and asserted.
- **Rounding.** Every decision and report rule is judged on unrounded values. Tables print 4 decimals unless a reproduction target states another precision.
- **R²** is never clipped. It is NaN when undefined (`r3.metrics._r2`).
- **Reproduction targets (brief §6.0).** B1 recomputed from the fold betas must match the round-4 REF parquet within
  **8.9e-16** (max |B1 − REF| ≤ 8.9e-16; seen 8.88e-16) on all 18,020 rows, for every item that registers the B1
  identity (every `*-T01` row of `REGISTRY.csv`; the same constant in every item's code). A mismatch stops the item.

## 2. Relative pair-SSE: the decision currency (M35)

For a candidate `c` and a reference `r` (B1 for every E-A gate) on the scored rows of seed s:

```
rel_s = 1 − SSE_c,s / SSE_r,s ,      SSE_x,s = Σ_rows (y − p_x)²
rel   = mean_s rel_s                                   (r5.common.rel_sse)
```

- **Link to R².** On identical rows, `ΔR²_s = rel_s · SSE_r,s / SST_s` exactly. At B1's SSE/SST of 0.548 [EB §1.1], the 2 % gate is about +0.011 R². It is the same margin as round 4's +0.01, restated in one unit; it adds no power [IR M35; EB §6.5].
- **The E-A gate** (PROTOCOL §5) passes only if all three hold:
  - rel ≥ 0.02 against B1 on BPG;
  - SF_POOL p ≤ 0.025 (§5.1);
  - Δbalanced-MAE (B1 − candidate) ≥ −0.001 on the seed-mean point estimate.

  A pass is DIAGNOSTIC.
- **The all-seed pooled form** `rel_pooled = 1 − Σ_s SSE_c,s / Σ_s SSE_r,s` is written only by E-LAB, as a bridge to lab-1's published form. It is never a gate.
- **Reported with it:** MSE (`SSE / n` per seed, then the seed mean) next to R². Every E-D rung and every separate-estimand table shows relative SSE against B1 **on the same rows** (PROTOCOL §3).

## 3. Units and effective counts

| unit | where it is used | count (V1, C3 unless stated) | source |
|---|---|---|---|
| `cluster_cpg`: connected components of chemotype ∪ publication ∪ gen19 copy group | E-A inference (M35 gate, every unit table on E-A rows) | 13; Kish 2.24; the largest holds 65 % of rows | `r4.metrics.add_clusters` [EB §2.19] |
| publication | LOPO spread; E-LAB; E-KX's sc009 sign-flip (G = 10) | 39 on E-A rows; Kish 18.8 | [EB §2.19] |
| chemotype | breadth context; E-CT chemotype blocks | 38; Kish 3.85 | [EB §2.19] |
| publication × chemotype (`pubchem`) | E-CT and its annexes in corpus (P1's unit); E-REC | 64 on E-A rows; E-CT PRIMARY: 14 | P2 §1; `results/stage0/ect_population.json` |
| laboratory (frozen PROVISIONAL key: conservative `LABnnn`, liberal senior-PI id) | E-LAB and E-CT annotation; new-source claims | E-CT PRIMARY publications: 7 conservative / 11 liberal laboratories | `results/stage0/lab_key.csv`; `supply_audit.json` |
| laboratory × chemotype | new-source unit (M34) | — | brief §4.2 |
| C-ECT unit: a connected component of the laboratory–chemotype graph of eligible families (one value per chemotype when no laboratory spans chemotypes) | C-ECT on new data under D3 | minimum G = 7 units | PROTOCOL §7, item 2 |

**Kish effective count** over row shares: `Kish = (Σ_u n_u)² / Σ_u n_u²`, where n_u is the unit's rows. It is reported beside every nominal count. In-corpus E-CT has 3 cpg clusters with Kish 1.25, so cluster-level inference there is stated as "not estimable" [P1 §3, §12].

## 4. Per-unit statistics (brief §9.2; `r5.common.unit_stats`)

For each unit u, the per-seed sums are computed and then averaged over the seeds that hold the unit (on E-A rows, every unit is in every seed):

| statistic | definition |
|---|---|
| n_u, SSE_ref,u, SSE_cand,u | seed means of the unit's rows and of the two arms' summed squared errors |
| ΔSSE_u | SSE_ref,u − SSE_cand,u |
| L_u | log(SSE_ref,u / SSE_cand,u). Undefined when either SSE is 0; then SF_LOG is not computed for that frame |
| rel_u | 1 − SSE_cand,u / SSE_ref,u |
| win / loss / tie | ΔSSE_u > 0 / < 0 / an exact 0. E-CT and M03 treat \|c_u\| ≤ 1e-12 as a tie |
| share_u | ΔSSE_u / Σ_u ΔSSE_u: the unit's share of the pooled gain |
| c_u (E-CT) | mean_s (ΔSSE_u,s / SST_s), so that Σ_u c_u = the seed-mean pooled ΔR² [P1 §2, §12] |

Units whose SSE is computed against an expectation report the same columns with that expectation as the reference: WINDOW's CELL control, `E[SSE_CELL] − SSE_window`, and E-DEC's per-cluster coverage contributions. Each item specification states its reference.

## 5. Tests (brief §4.2; one-sided, H1: the candidate is better; nominal 0.025, sensitivity 0.05)

### 5.1 SF_POOL: sign-flip on pooled ΔSSE (primary for pooled claims)

- **Statistic and null.** `T = Σ_u ΔSSE_u` over the G units; H0: mean ΔSSE ≤ 0. Each flip vector f ∈ {±1}^G gives `T*(f) = Σ_u f_u ΔSSE_u`.
- **Exact enumeration (G ≤ 13).** Every one of the 2^G sign vectors is enumerated, the identity included: `p = #{f : T*(f) ≥ T} / 2^G`.
  - At G = 13 (the `cluster_cpg` units) that is 8,192 vectors, which is how the seen replay p values were obtained [PC P2].
  - The comparison uses a relative tolerance of 1e-10 · Σ|ΔSSE_u|, so exact ties count against the candidate.
  - No RNG is used.
- **Random flips (G > 13).** B = 4,999 random sign vectors from `default_rng(20261023 + crc32(key))` (`r5.paths.seed_for`, keys in `REGISTRY.csv` and the item specifications): `p = (1 + #{T* ≥ T}) / (1 + B)`.
- **Item extension (D8).** E-CT and M03 enumerate exactly up to G = 22 units, which covers every registered stratum, and use random flips only beyond that. Every other item follows `r5.common` (exact at G ≤ 13). Every p value records whether it is exact, and the seed when it is not.
- **Floors.**
  - The smallest attainable exact p is 1/2^G. G ≤ 5 cannot reject at 0.025 (1/32 = 0.031). G = 6 cannot reach a Holm-adjusted 0.0125 (1/64 = 0.0156) [P2 §3; brief §4.2].
  - E-CT's chemotype-block flip (whole chemotypes flipped together) has floor 1/2^G_chem: 0.0156 at 6 blocks. That floor is not a measure of strength [P1 §12].
- **What exactness means.** Size holds by construction under a sharp, exchangeable null (arm swap, symmetric). It does not hold under skewed heterogeneous nulls: 0.103 under C-POOL at cpg [P2 §3; PC P2]. That is why every in-corpus SF_POOL p is DIAGNOSTIC and never called significant.

### 5.2 SIGN: exact binomial sign test (primary for win-share claims)

- `W` units win, `L` lose, and ties (exact 0) are dropped. `n = W + L`, and `p = P(Bin(n, ½) ≥ W)` (`r5.common.sign_test`). A boolean win vector is refused, because it would hide the ties.
- The estimand is π, the share of units that improve, not the pooled gain. SIGN is conservative (mean size 0.012), and was never flagged under the gating nulls [PC P2].
- **Exception.** Under the publication-level skewed null C-MED it reached 0.106 (τ 0.065, G 80) [P2 §3].

### 5.3 Equal-weight breadth tests (E-S; diagnostics only)

- **SF_LOG:** SF_POOL's construction applied to L_u (the mean log SSE ratio).
- **T_LOG:** a one-sample t on L_u, df G − 1. This is the IM t, which equals the jackknife t with Kish = G.
- **IM_REL:** the Ibragimov–Müller t, a one-sample one-sided t over the G finite rel_u, df G − 1 (`r5.common.im_t`). It reports the mean, SE and t. E-LAB also uses its one-sided 95 % lower bound.

These answer "does the average source improve?". They never replace pooled R² or relative SSE, and never gate. The equal-weight leg fires on HBK@ML − B0 (p 0.0115) although HBK@ML is +0.13 % pooled [IR M35].

### 5.4 Comparators, non-gates and dropped tests

| test | status | why |
|---|---|---|
| WCR_POOL (wild cluster bootstrap, CR2, Webb weights, on per-row ΔSSE clustered by unit) | comparator only | 1 gate flag [P2 §3] |
| SEALED (round 4's R² leg: percentile, BCa and jk-t(G − 1) lower bounds all > 0, plus Kish ≥ 3) | comparator only | 2 gate flags |
| JK_KISH (cluster jackknife t on the pooled relative SSE, df = max(Kish_rows − 1, 1)) | not a gate | real-pool maximum size 0.0515; flags start at G = 8 [PC P2] |
| WCR_LOG | dropped | 6 flags; size up to 0.113 |
| **JK_G1** | **dropped** | see below |

**JK_G1** is round 4's sealed cluster-jackknife t with df = G − 1 [R4-METRICS: `SE = sqrt((G − 1)/G · Σ(θ_i − θ̄)²)`]. It is anti-conservative: on real contrast pools, mean size 0.047 and maximum 0.096 at nominal 0.025; on synthetic scaled pools, maximum 0.1245 [PC P2; P2 §3]. Round 5 uses it nowhere as a test, gate or confirmation leg.

**Two jackknife-t forms remain, both descriptive and labelled as such:**
- the leave-one-publication-out interval of §6;
- the M13 bound fixed by brief §6.10: a leave-one-cluster-out jackknife-t 95 % interval, t(G − 1), stated once, for E-W's retire-only Nd/Sm rule. It shares JK_G1's construction, so it is too narrow and its lower bound errs high.

## 6. Spread (brief §6.0, §9.2): LOPO, LOCO and the top-3 share

**Leave-one-publication-out (LOPO; `r5.common.lopo`).**
- For a statistic θ (relative SSE, ΔR², a rule statistic), θ_(−p) is recomputed with publication p's rows removed, for each of the P publications.
- The report gives:
  - the full-data point;
  - the jackknife SE, `sqrt((P − 1)/P · Σ_p (θ_(−p) − θ̄)²)`;
  - the interval `point ± t_0.975(P − 1) · SE`;
  - the minimum and maximum of θ_(−p), each with the publication left out;
  - the count of undefined leave-outs, which are excluded.
- LOPO and LOCO ranges are the honest spread on this corpus. The interval is descriptive: it is never a test and never called significant [EB §3].
- The t(P − 1) interval ignores the clustering of publications (39 in 13 cpg clusters on E-A, Kish 2.24; 12 in 3 on E-CT PRIMARY, Kish 1.25), so like JK_G1 it is too narrow; the min / max leave-out range is the honest spread.

**Leave-one-cluster-out (LOCO)** is the same construction over `cluster_cpg` clusters, where an item registers it.

**Leave-k-out variants**, each where its item registers it, and each labelled outcome-selected where it is:
- E-LAB's leave-two-publications-out (every pair; the minimum and the named pair);
- E-LAB's and E-KX's leave-top-3-out;
- WINDOW's leave-top-k-cells-out, k ∈ {5, 10}.

**Nested selections** (M03) re-run the whole nested selection inside each LOPO leave-out (`lopo_renested_*`). The row-drop LOPO is reported beside it.

**Top-3 unit share** (`r5.common.top3_share`):
- The share of the total contribution held by the three largest unit contributions: `Σ(top 3 of ΔSSE_u, or of c_u) / Σ_u (all)`.
- It is NaN when the total is ≤ 0, and it can exceed 1 when other units lose.
- Every item reports it for its headline contrast. Seen values: E-CT SH@2 0.99 [P1 §6]; E-LAB 0.93 [I2 lab-1]; M07 0.75 [IR M07].

**C5-BTBP in and out, everywhere** (brief §5.2):
- every number with and without pub_6a0ef8528c, plus the single-cell influence of c950dc2982e96b83;
- a gain that lives in C5-BTBP is a provenance sensitivity, not a model result. Dropping it raises R² partly by removing SST, since its rows carry about 18 % of SST [PC P3].

## 7. Cluster bootstrap and contrast-specific MDEs

**Cluster bootstrap (descriptive).**
- Items that compute one (E-D; the round-4 comparators) use round 4's shared draw matrix `gen20k.inference._draw_counts(13, 2000, 8675309)` over the 13 C3 `cluster_cpg` clusters.
- Percentile intervals are descriptive only. With Kish 2.24, the percentile cluster bootstrap rejects at 0.07–0.08 for a nominal 0.05 [EB §3].
- A bootstrap tail fraction such as P(R² ≥ 0.50) is a descriptive frequency, never a posterior probability [P3 §1].

**Contrast-specific MDE.**
- The implied 80 % MDE of a contrast is `MDE_80 = 2.8 × SD` of **that contrast's** cluster-bootstrap distribution, where 2.8 ≈ z_0.975 + z_0.80.
  - This is round 4's `mde_80` column of `results/<slot>/contrasts_*.csv` (2.8 × the bootstrap SE), and PC P3's yardstick (`r5/ed.py`: `MDE80 = 2.8 × boot_sd`).
- Because the bootstrap is anti-conservative at Kish 2.24, the implied MDE is optimistic: too small.
- **An MDE is never borrowed from another contrast.** Each contrast with a bootstrap reports its own. A contrast without one states "MDE not computed".
- Seen values:

  | contrast | MDE_80 | source |
  |---|---|---|
  | B1 − B0 (k = 3, BPG, cpg) | 0.044 | EB §3; E5 §5.3 |
  | S05 − B1 | 0.011 | EB §3; E5 §5.3 |
  | S11 − B1 | 0.003 | EB §3; E5 §5.3 |
  | BISECT_R − DOPTQ (5 reads) | ≈ 0.020 (bootstrap sd 0.0070 BPG / 0.0071 BP) | PC P3 |
  | RANDOM − DOPTQ | ≈ 0.012 | PC P3 |
  | BISECT_R − RANDOM | ≈ 0.009 (sd 0.0031 / 0.0032) | PC P3 |

  The MDE scales with how far a candidate departs from B1 [E5 §5.3]. P3's "envelope 0.009 < MDE 0.011" borrowed S05's yardstick, and PC P3 corrected it.
- **Effects below about 0.01 R² are invisible on this corpus** [EB §3; E5 §5.3].

## 8. Breadth diagnostic (brief §4.1 item 6; item 1.0)

- **Statistic:** the equal-weight mean over units of L_u, from seed-mean unit SSEs, tested with SF_LOG (§5.3), with the IM t and the equal-weight mean of rel_u beside it.
- **Variants:**
  - **SMALL_POOLED** (the sealed diagnostic): clusters with < 3 cells or < 20 rows per seed in the stratum are pooled into one "small sources" unit;
  - **ORNL_COLLAPSED** (sensitivity): the ORNL-only clusters are collapsed into one unit;
  - **RAW** (reference): the 13 clusters.
- **Inside the giant cluster:** publication-level n, SSE, ΔSSE, L, rel and share.
- This diagnostic is never a gate (PROTOCOL §5).

## 9. E-CT in-corpus statistics (brief §6.1; descriptive)

The contrasts are SH − B1, the mechanism contrasts SH − OFF_XEXT and SH − PB_PUB, and the other contrasts of `results/ECT/ITEM_SPEC.md` §7. Each is reported with:
- the per-seed ΔR² and relative SSE;
- unit W/L/T over publication × chemotype units, with c_u (§4);
- the exact sign-flip over units, and the chemotype-block sign-flip with its floor;
- SIGN;
- IM t on rel_u and on L_u;
- LOPO with t(P − 1);
- the top-3 share;
- the equal-weight per-publication relative SSE.

Cluster-level inference is "not estimable". SH − OFF is reported only as the read-alignment effect: SH ≡ OFF exactly where every sibling was read at the target's metals in the same fold [P1 §5; PC P1].

## 10. Confirmation-package statistics (brief §4.3, §7.5; `r5/confirm.py`; CONFIRMATION_PACKAGE.md)

These are scored once, on Stage-2 data that arrive after the seal. The supply-audit kill fires at this seal (PROTOCOL §7, item 7), so none runs in round 5.

**C-ECT.**
- **Family statistic:** `F = log(Σ SSE_B1@4 / Σ SSE_SH)` over a laboratory × extractant family's leave-one-condition-out targets.
  - Targets whose pairs are all fixed by reads are excluded.
  - The ratio of sums avoids undefined per-target ratios: one corpus target has SSE_B1 = 0.
- **Laboratory value:** the unweighted mean of its families' F. The family-level p is reported beside it.
- **Decision unit (D3, fallback (ii)):** C-ECT unit = a connected component of the laboratory–chemotype graph of eligible families (chemotypes from the frozen gen6 map, DATA_CONTRACT §8.3), so that no laboratory contributes to two units; unit value = unweighted mean of its laboratories' values; exact one-sided sign-flip over units (§5.1); minimum G = 7 units. With no spanning laboratory this is exactly one value per chemotype. A unit with no defined value is excluded before the test; a NaN or ±inf unit value raises.
- **Three-way rule, read on the same chemotype values:**
  - confirm: exact p ≤ 0.025;
  - kill: mean ≤ 0, or win share < 0.5;
  - otherwise INCONCLUSIVE.
- **Fixed sequence:** SH@4, then SH@3.
- **Descriptive beside it:** the laboratory-level exact sign-flip and SIGN, which fail the size gate (ECT_SF maximum 0.0979, ECT_SIGN maximum 0.0979; `results/stage0/size_gate.json`).
- **Fresh size check:** `results/stage0/size_gate_ect_chemotype.json`. The chemotype-level sign-flip passes on every gating cell: 0 of 80 gating rows flagged, maximum size 0.0268, 20,000 replicates per cell. At the floor of 7 chemotypes the smallest attainable p is 1/128 = 0.0078. The seen-pool skewed nulls are descriptive: right-skewed at most 0.0083, mirrored (left-skewed) 0.057–0.072.

**Gd chord** (C-GD, Gd-dip transport).
- **Cell chord:** `a_Gd = y_Gd − [(1 − w) y_Eu + w y_Tb]`, with `w = (x_Gd − x_Eu) / (x_Tb − x_Eu)` along the −Shannon CN8 radius. It is log D(Gd) minus the Eu–Tb radius interpolation.
- **Laboratory value:** the laboratory's mean Gd chord over its eligible units (eligibility in PROTOCOL §7, item 8; the aggregation is in CONFIRMATION_PACKAGE). One value per laboratory.
- **Interval:** the 95 % laboratory-mean t-interval, df G − 1. Its 97.5 % one-sided bounds are read against β_frozen = −0.110809, β_frozen/2 and 0.
  - For the winsorised form (primary), every laboratory value is first clipped to [−0.40, +0.20]. The raw form is a sensitivity.
- **Readings:**
  - confirm: upper bound < β_frozen;
  - kill: lower bound > β_frozen;
  - helps (A2): upper bound < β_frozen/2;
  - B1-harm: lower bound > β_frozen/2;
  - A1 (against 0): descriptive.
- **Under D3 fallback (i)** every one of these intervals is a descriptive interval on the exclusion list, and its readings are descriptive (PROTOCOL §7, item 2).

**Size gate.**
- A statistic is flagged when the two-sided Clopper–Pearson 95 % lower bound of its rejection rate exceeds 0.035 at nominal one-sided 0.025 (the lower limit of the two-sided interval, i.e. a one-sided 97.5 % bound; P2's convention). Under a one-sided 95 % bound 332 / 332 / 326 rows flag for CGD_W / CGD_R / GDT_R (12 more cells: comp (2, 2, 2), ICC 0.23), with the same verdicts; the ECT rows are unchanged.
- Skewed seen-pool nulls are reported in both orientations and never gate (brief §4.2).
- Nulls: i.i.d., and two-stage (chemotype → laboratory) at ICC {0.23, 0.3, 0.5, 0.69}.
- ≥ 2,000 replicates per cell; 20,000 were used in `results/stage0/size_gate.json`.

**Power in worlds.**
- The Monte Carlo rejection rate with its Clopper–Pearson 95 % interval for every reported rate (Monte Carlo SE ≤ 0.005), 10,000 replicates per world (`results/stage0/power_worlds.json`, `.csv`).
- For the chord rules in normal worlds, it is checked against the analytic non-central t: maximum |difference| 0.0152.
- Each world is labelled synthetic, or seen-corpus (label-based, upper-side for new sources).

## 11. Item-specific statistics (definitions in each ITEM_SPEC.md)

| item | statistics beyond §2–§6 |
|---|---|
| 1.0 EA_CHECK | the M35 gate quantities; six-design levels; the M08 moment ρ = Σ e_ip e_jp / √(Σ e_ip² · Σ e_jp²) per relation class, with its identifiability column |
| 1.1 ECT, 1.2 ECT_M03 | c_u; the read ladder and dose-response; M03's nested one-SE selection (paired SE primary) and its selection-aware permutation p = (1 + #{null ≥ obs}) / (1 + B), with B = 500 |
| 1.3 ELAB | MX_LM10 relative SSE vs B1 and vs placebos at matched counts (E_* expected squared errors over 30 seeded draws); laboratory-median relative SSE (DIAGNOSTIC) |
| 1.4 WINDOW | coverage; inside / outside R² and MSE; inside relative SSE `1 − MSE_in / MSE_all`; tail ratio P_out / P_in of \|res\| > 0.5 and > 0.3 with a 95 % publication bootstrap (B = 2,000); q80 split-conformal half-widths and their narrowing |
| 1.5 EDEC | false sign (a call whose sign differs from y's with \|y\| ≥ 0.10); per-call and per-pair rates; coverage; AURC and E-AURC [O]; LTT (primary), CRC and plug-in thresholds; net benefit; calibration-in-the-large; E-RANK endpoints and LINE shares |
| 1.6 ED | per-rung pooled R², MSE and relative SSE; per-block ΔSSE in pooled-R² units; envelope width; MDE_80 per contrast (§7); P(R² ≥ 0.50) as a bootstrap fraction (descriptive) |
| 1.7 EW | per-cut skill `1 − SSE_B1 / SSE_ref` vs LINE and B0; MSE and var(y) multipliers; sign agreement in \|y\| bins; the Nd/Sm cluster-equal-weighted B1 − LINE with its LOCO jackknife-t bound (§5.4) |
| 1.8 EKX | Nd chord `a_Nd = y_Nd − [(1 − w) y_Pr + w y_Sm]`; shrinkage `n/(n + 5)`; sc009 Pr/Nd R² and MAE; random-chemotype permutation p (B = 2,000) |

## 12. The per-item table (brief §9.3; `r5.common.per_item_table`)

Every item writes one table per design with these fields:
- item id and estimand;
- population (cells, publications, units, rows per seed);
- design and read budget;
- arms on identical rows;
- R², MSE and relative SSE vs B1;
- per-seed values;
- unit W/L/T and the descriptive tests;
- LOPO and the top-3 share;
- C5-BTBP in / out;
- duplicate sensitivities;
- leakage status;
- compute;
- the label, which must name a PROTOCOL §4 base label;
- two sentences: what changed scientifically, and what the next registered step is.

The writer refuses a table with a missing field or an unknown label.
