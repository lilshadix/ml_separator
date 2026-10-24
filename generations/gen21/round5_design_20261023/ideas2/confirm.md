# Confirmation with the available supply: what about 13 new laboratories can decide

Written 2026-10-23 for the round-5 design (theme: CONFIRMATION). Ideas confirm-1 to confirm-4.

> **Status: EXPLORATORY.** Every number below comes from the 229 C3 cells, the 521-cell V1 bench and the 5 dev seeds. Rounds 1–4 and probes P1–P5 have already seen all of them.
> - The replays plan the number of new units G. They confirm nothing.
> - Two analyst choices were made after looking: the winsorisation bounds and the heavy-/light-selective split. The split is the main new lead here, and it is post hoc.
> - Nothing is sealed.

---

## 0. Bottom line

1. **A pooled confirmation of B1 is out of reach.** With about 13 new laboratories, every pooled test needs 37 to more than 80 sources (P2). That includes the sealed R² leg and E-A.
2. **What the supply can decide is about the Gd term, measured without any model.** The statistic is the Gd chord: log D(Gd) minus the Eu–Tb radius interpolation (Gd lies at t = 0.500 between Eu and Tb on the −r(CN8) axis). It is computed on each new unit's own full series.
   - The chord predicts the sign of a unit's B1 − B0 gain in **95 % of the 44 corpus units**. Spearman with the cell's gain is −0.96, and 95 % of B1's gain sits on the Eu/Gd and Gd/Tb pairs.
   - So a chord test is a model-free and cheaper proxy for "does B1 beat B0 in this lab?".
3. **The one question that is both decisive and chemically new: is the Gd break of heavy-selective non-DGA systems deeper than B1's amplitude?**
   - Corpus units (exploratory, split chosen post hoc):

     | stratum | units | mean chord | sd |
     |---|---|---|---|
     | heavy-selective non-DGA | 12 | −0.217 | 0.10 |
     | light-selective non-DGA | 13 | −0.073 | 0.10 |
     | DGA | 19 | −0.051 (median −0.064) | |

     B1's single amplitude is β = −0.092.
   - A one-sample test of "mean chord < β" on new heavy-selective non-DGA units has power **0.62–0.74 at G = 6, 0.88–0.94 at G = 8 and 0.97–0.99 at G = 10**. Its size at the null is ≤ 0.019.
   - The new supply is mostly acidic organophosphorus and chelating extractants, which are heavy-selective (P4: 15 of 16 new labs on chemotypes absent from the corpus). This is the one stratum the supply fills.
4. **The B1 transport chain is only partly decidable at G ≈ 8–13.** The chain is A1 (a dip exists), then A2 (B1's term helps the average unit), then A3 (equal-weight B1 > B0).
   - A1 alone: 0.83 at G = 8 and 0.99 at G = 13 (non-DGA units).
   - The whole chain: 0.36 / 0.23 at G = 8 and 0.70 / 0.50 at G = 13 (τ = 0 / τ = 0.065).
5. **E-CT (condition transfer) can be confirmed only with at least 7 new non-trivial sibling families.**
   - SH vs B1 at 4 reads: power 0.63–0.67 at 6 families, 0.89–0.93 at 7 and 0.95–0.99 at 8.
   - The expected supply is 4–9 families, so this leg is borderline.
   - "Two reads plus your own series matches B1 with 4 reads" is **not** confirmable at any reachable G (equal-weight power ≤ 0.06; the pooled sign-flip reaches only 0.25 at G = 12).
6. **Three endpoints cannot be decided and are excluded from the package.**
   - **The window flag.** Inside a laboratory it does not separate error: MSE outside > inside in only 11 of 18 eligible publications, mean log ratio +0.03, and tail rates higher outside in 6 and lower in 5. Its power at G = 13 is ≤ 0.15. The pooled 4.9× tail ratio comes from differences between units (Simpson's paradox).
   - **Transport of B1's amplitude within ±50 % (TOST):** power ≤ 0.08.
   - **Any pooled R² claim.**

---

## 1. Checks run

All scripts are in `ideas2/scripts/`. They were run one at a time from `D:\ml_separator_gh` with `.venv\Scripts\python.exe -B -W ignore`, and each took under 3 minutes. Outputs are in `ideas2/scripts/confirm_out/`. P2's functions were imported read-only, and `-B` meant no bytecode was written.

| check | script | output |
|---|---|---|
| 1 Gd chord per unit, link to B1's gain, transport-test replay | `confirm_gd_units.py` | `gd_units_desc.csv`, `gd_units_C3_pubchem.csv`, `gd_units_summary.json`, `gd_replay.csv` |
| 2 B1 − B0 breadth (P2 machinery), window flag, winsorised Gd sizes | `confirm_breadth_window.py` | `breadth_replay.csv`, `window_summary.json`, `window_replay.csv`, `gd_winsor.csv` |
| 3 E-CT replay | `confirm_ect_replay.py`, after re-running `probes/scripts/P1_ct_build.py --out confirm_out/p1` (unchanged code; 18,020 rows, B1 identity 8.9e-16) | `ect_dev.csv`, `ect_replay.csv` |
| 4 Joint A1 → A2 → A3 chain on the same drawn units | `confirm_chain.py` | `chain_dev.csv`, `chain_replay.csv` |
| 5 Donor × selectivity-direction split of the chord | `confirm_gd_direction.py` | `gd_direction_desc.csv`, `gd_direction_replay.csv` |
| 6 Power at the family-wise level α/3 | `confirm_bonf.py` | `bonf.csv` |

### Shared conventions

- **Units.** publication × chemotype (`pubchem`) is the proxy for the declared laboratory × chemotype unit. Publication and cluster_cpg are shown as sensitivities.
- **Donor class.** DGA is defined by the round-4 SMARTS rule (`r4.donor`), not by chemotype sc009.
- **β.** −0.0923, the BPG fold mean of B1's fitted amplitude (25 folds, sd 0.019).
- **Test sidedness.** All tests are one-sided.
- **Replication.** 2,000–4,000 replicates per cell. Sign-flips are exact up to G = 13.
- **Noise.** Per-source noise is P2's τ ∈ {0, 0.065}, applied to each arm's SSE.

### 1.1 The Gd chord per unit (C3, cells with Eu, Gd and Tb; V2-clean results identical to 3 decimals)

| unit level | group | units | unit mean | unit sd | SE at G = 13 |
|---|---|---|---|---|---|
| pubchem | all | 44 | −0.092 (cell mean −0.115, n = 127; reproduces E4) | 0.157 | 0.044 |
| pubchem | non-DGA | 25 | −0.142 | 0.122 | 0.034 |
| pubchem | DGA | 19 | −0.027 (median −0.064; one +0.663 single-cell unit, pub_0a8da089ad) | 0.178 | 0.049 |
| pubchem | non-DGA without C5-BTBP and DMDPhPDA | 23 | −0.132 | 0.121 | 0.034 |
| cluster_cpg | all | 10 | −0.179 | 0.105 | 0.029 |

- The pooled within-unit sd of a single cell's chord is 0.127.
- The quadratic chord, curvature-adjusted through Sm, Eu, Tb and Dy, gives the same picture (non-DGA −0.128, DGA −0.042).
- **Correction to M33:** "between-unit sd about 0.1, SE about 0.03" holds only at cluster level. At the declared unit level the sd is 0.12–0.16 and the SE at G = 13 is 0.034–0.044.

### 1.2 The chord is a sufficient statistic for B1 vs B0

| measure | value |
|---|---|
| Spearman(cell chord, cell B1 − B0 SSE gain), 127 cells | −0.957 (BP −0.967) |
| share of B1's gain on the Eu/Gd and Gd/Tb pairs | 0.95 |
| the rule "chord < β/2 ⇒ B1 beats B0" is correct for | 92 % of cells; 95 % of the 44 units |
| cell win rate when the chord is < β/2 | 0.93 |
| cell win rate when the chord is ≥ β/2 | 0.10 |

### 1.3 Gd transport tests, unit replay

Pubchem pools; m = all of a unit's cells.

| test (H1) | non-DGA G = 8 | non-DGA G = 13 | all units G = 13 | DGA G = 13 |
|---|---|---|---|---|
| A1 GD0: mean chord < 0 | 0.84 | 0.99 | 0.72 raw / 0.85 winsorised | 0.48 |
| A2 GDHALF: mean chord < β/2 | 0.48 | 0.77 | 0.39 | 0.09 |
| GDDEEP: mean chord < β (all non-DGA) | 0.16 | 0.27 | 0.04 | 0.00 |
| TOST: \|mean chord − β\| < \|β\|/2 | 0.00 | 0.00 | 0.02 | 0.08 |

**Small new units.** Units with 1 or 2 cells cost about 0.05–0.10 of power. For example, A2 at non-DGA G = 13 is 0.66 with m = 1 and 0.70 with m = 2.

**Chemistry worlds** (non-DGA spread kept, mean moved; m = 2):

| world mean | GDDEEP at G = 8 | GDDEEP at G = 13 | A2 at G = 8 |
|---|---|---|---|
| −0.26 (HDEHP-like) | 0.92 | 1.00 | 0.99 |
| −0.175 | 0.34 | 0.58 | |

**Size.**
- Raw unit means are anti-conservative at the null boundary. One example is A1 at G = 13–20 on all units, which gives 0.043–0.050 at nominal 0.025. The cause is the right-skewed DGA tail.
- Frozen winsorisation of unit chords to [−0.40, +0.20] restores the size to ≤ 0.027 on the all-units and non-DGA pools (A1 and A2, G = 6–16).
- The DGA pool stays at 0.034–0.055. The Gd tests are therefore not run on a DGA-only stratum.

### 1.4 Donor × selectivity direction (the post hoc lead)

Direction is the sign of the least-squares slope of log D on −r over the cell's measured metals. The slope computed from the 4 DOPT_Q reads alone has the same sign in 96.6 % of the 1,145 cell × seed read sets. The stratum can therefore be frozen as a reads-only rule, or replaced by mechanism metadata.

| stratum (pubchem units) | units | winsorised mean chord | sd |
|---|---|---|---|
| non-DGA, heavy-selective | 12 (10 without C5 and DMDPhPDA) | −0.217 (−0.210) | 0.101 (0.107) |
| non-DGA, light-selective | 13 | −0.073 | 0.096 |
| DGA | 19 | −0.051 | 0.086 |

- The heavy-selective units include BTBP / BTP / PDA / phen-diamide N-donors and one dialkylphosphoric acid (−0.37).
- **C-GD** tests "heavy-selective non-DGA mean chord < β" with a one-sample t:

  | G | 5 | 6 | 8 | 10 | 13 |
  |---|---|---|---|---|---|
  | power at 0.025 | 0.44–0.56 | 0.62–0.74 | 0.88–0.94 | 0.97–0.99 | 1.00 |
  | power at α/3 | 0.16–0.24 | 0.27–0.37 | 0.58–0.71 | 0.81–0.91 | 0.97–0.99 |

  Size at the null boundary is 0.009–0.019.
- A two-sample heavy- vs light-selective contrast reaches only 0.33–0.72 for splits (6, 3) to (8, 6). It is a secondary result.
- **Chemistry anchor (web, qualitative).** In P507 (HEH[EHP]) systems the adjacent separation factors Gd/Eu, Nd/Pr and Lu/Yb are reported as "very low" ([preprints.org 202312.0986](https://www.preprints.org/manuscript/202312.0986)), consistent with a deep negative Gd chord. HDEHP's mean adjacent SF is about 2.5 ([Nature 194:84](https://www.nature.com/articles/194084a0.pdf)).
  - The planning value −0.26 is my reading of the commonly cited HDEHP pattern (Gd/Eu about 1.5, Tb/Gd about 5). It was **not** verified against a primary table. A source must fix it before freezing.

### 1.5 Equal-weight B1 − B0 breadth (P2 machinery, BPG)

Power at one-sided 0.025, EMP world, τ = 0 / 0.065.

| pool (units) | G = 8, SF_LOG | G = 13, SF_LOG | G = 13, SIGN | G = 16, SF_LOG |
|---|---|---|---|---|
| publication, non-DGA (21 units; 18 W / 2 L) | 0.54 / 0.33 | 0.73 / 0.54 | 0.85 / 0.29 | 0.84 / 0.66 |
| publication, non-DGA without C5 and DMDPhPDA | 0.49 / 0.31 | 0.68 / 0.54 | 0.83 / 0.28 | 0.80 / 0.63 |
| publication, all | 0.33 / 0.24 | 0.59 / 0.44 | 0.57 / 0.19 | 0.70 / 0.54 |
| publication, DGA | 0.20 / 0.16 | 0.39 / 0.35 | 0.24 / 0.12 | 0.53 / 0.45 |
| pubchem, non-DGA (33 units; 20 W / 9 L / 4 tied) | 0.20 / 0.17 | 0.41 / 0.32 | 0.17 / 0.13 | 0.52 / 0.39 |
| pubchem with a Eu–Gd–Tb chord, non-DGA (check 4, A3) | 0.44 / 0.32 | 0.81 / 0.56 | | 0.90 / 0.66 |

- **Pooled guard.** Adding "pooled relative SSE ≥ 2 %" to SF_LOG costs ≤ 0.05.
- **Size.** SWAP-null size is ≤ 0.034 for SIGN, SF_LOG, T_LOG, SF_POOL and WCR_POOL in every pool (2,000 replicates).
- **Reading.** Breadth needs units that carry Gd-pair information. Units without an Eu–Gd–Tb triad are mostly ties.

### 1.6 The joint chain on the same drawn units (non-DGA units that have a chord)

| G | 6 | 8 | 10 | 13 | 16 |
|---|---|---|---|---|---|
| A1 | 0.63 | 0.83–0.84 | 0.93 | 0.99 | 1.00 |
| A1 → A2 | 0.34 | 0.48 | 0.60–0.62 | 0.76 | 0.86–0.87 |
| A1 → A2 → A3 (τ 0 / 0.065) | 0.14 / 0.10 | 0.36 / 0.23 | 0.52 / 0.33 | 0.70 / 0.50 | 0.83 / 0.62 |

- At α/3, the full chain at G = 13 is 0.46 / 0.28.
- Without C5 and DMDPhPDA, the full chain at G = 13 is 0.63 / 0.45.
- A2 and A3 disagree in 16–32 % of replicates.

### 1.7 Window flag, within a unit (B1 levels rebuilt; identity with B1 8.9e-16)

**Pooled over all rows (BPG):**

| measure | inside | outside |
|---|---|---|
| coverage | 0.747 | |
| P(\|res\| > 0.5) | 0.0091 | 0.0443 (ratio 4.9; 4.6 without C5-BTBP) |
| MSE | 0.019 | 0.051 |

**Within units:**
- 18 of 39 publications have at least 3 rows on each side; they hold 95 % of the out-of-window rows.
- MSE outside > MSE inside in 11 of 18. Mean log ratio +0.03, median +0.15, sd 1.54.
- The tail rate is higher outside in 6 units and lower in 5.
- Exact sign-flip power at G = 13 is 0.05–0.09 (BPG) and 0.09–0.15 (BP).

**Verdict.** The window marks risky units or conditions, not risky rows inside a laboratory. It cannot be confirmed as a row flag with this supply.

### 1.8 E-CT (P1 PRIMARY: CLEAN, n_sib ≥ 2, no C5-BTBP; 14 pubchem units)

**Corpus contrasts:**

| contrast | W / L | mean log SSE ratio |
|---|---|---|
| SH at 4 reads vs B1 at 4 | 13 / 1 | +0.66 |
| SH at 3 reads vs B1 at 4 | 12 / 2 | +0.42 |
| SH at 2 reads vs B1 at 4 | 8 / 6 | +0.07 |
| COPY vs B1 at 4 | 4 / 10 | −0.39 |
| acid-concentration-only families, SH at 4 | 5 / 0 | +0.79 |
| n_sib = 1, SH at 4 | 8 / 1 | +1.30 (sd 2.7) |
| "trivial-looking" families (diluent / extractant concentration / temperature only), SH at 4 | 5 / 0 | |
| the same families, COPY | 0 / 5 | −1.02 |

The last row matters: for neutral solvating extractants even these changes alter the SF pattern. The near-copy expectation therefore holds only for mass-action-trivial perturbations of cation exchangers, such as pH or [HA] scans.

**Power, exact sign-flip on the equal-weight log ratio** (τ = 0 / 0.065; at G = 5, one-sided 0.025 is unreachable because the minimum p is 1/32):

| contrast | G = 5 at 0.05 | G = 6 | G = 7 | G = 8 | G = 9 |
|---|---|---|---|---|---|
| SH at 4 reads | 0.68 / 0.72 | 0.63–0.67 | 0.89–0.93 | 0.95–0.99 | 0.98–1.00 |
| SH at 3 reads | | 0.39–0.41 | 0.65–0.73 | 0.75–0.87 | 0.84–0.93 |
| SH at 2 reads | | | equal-weight ≤ 0.06 at every G ≤ 12 (pooled sign-flip ≤ 0.25) | | |

- At α/3, SH at 4 reads needs G = 8–9 (0.79–0.98).
- SWAP-null sizes are ≤ 0.033 at 0.025.
- **Campaign-fingerprint control** (SH vs the other-extractant blend from the same publication): 5 corpus units, 4 W / 1 L. Power at G = 8 is 0.70, but the pool is too small to trust. The control is descriptive only.

---

## 2. The package (confirm-1): CP-13, one frozen, one-shot confirmation

**Frozen before any new value is read.** The freeze is a hash of the package plus the analysis script; values are sealed by a separate agent.

1. **References.**
   - β_frozen: the round-4 B1 refit on all 229 C3 cells. Planning value −0.092; sensitivity at −0.072 and −0.118.
   - DOPT_Q parameters fitted on all C3 cells.
   - The PWL_R radius axis −r(CN8).
   - SH with K = 1.78 (from r = 0.36).
   - Winsorisation of unit chords to [−0.40, +0.20].
2. **Unit and eligibility.**
   - **Unit.** One value per laboratory × chemotype: the equal-weight mean over its conditions. Laboratories are identified by author-set rules (P4 conservative merge), copy groups are merged from metadata on arrival, and the gen6 chemotype map is frozen.
   - **Gd endpoints.** The unit must report Eu, Gd and Tb itself.
   - **Values excluded from the chord:** values flagged below the detection or quantification limit, and aqueous-complexant media (the M29 data-contract rule).
   - **Stratum.** "Heavy-selective non-DGA" from mechanism metadata (acidic organophosphorus, chelating cation exchangers, soft N-donors). The sensitivity is the read-slope sign, which agrees in 97 % of corpus cells.
   - **E-CT.** At least 3 conditions of the same extractant in the same laboratory. Every condition is the target once, with the others as siblings.
   - **Perturbation class from metadata:**
     - **T0:** pH, [HA] or phase ratio only, for cation exchangers. This is the validity control.
     - **T1:** aqueous medium (acid concentration or type, salting-out agent).
     - **T2:** organic phase (diluent, modifier, extractant concentration for solvating extractants).
     - **T3:** mixed.
     - The primary class is T1 ∪ T2 ∪ T3.
   - **SAFE units** (0–3) count only if they pass the same independence rule. Otherwise they form a descriptive "corpus-chemistry replication" stratum.
3. **Three separate claims**, each at one-sided α = 0.025. There is no global "success" statement, and power at α/3 is reported next to each.
   - **C-GD** (confirm-2): heavy-selective non-DGA mean winsorised chord < β_frozen (one-sample t, df G − 1). Two-sample heavy- vs light-selective is secondary.
   - **C-B1** (confirm-3): fixed sequence A1 (mean chord < 0), then A2 (< β/2), then A3 (equal-weight B1 > B0 log SSE ratio by exact sign-flip, with the pooled guard ΣΔSSE > 0). It runs on all eligible units.
   - **C-ECT** (confirm-4): SH at 4 reads vs B1 at 4 reads, exact sign-flip on per-family log SSE ratios, in the primary perturbation class. The secondary test is SH at 3 reads vs B1 at 4 reads (Holm within the claim).
4. **Minimum G.** Below these, a claim is reported as descriptive and carries no decision language.

   | claim | minimum G |
   |---|---|
   | C-GD | G_HS ≥ 6 |
   | C-B1 | A1 at G ≥ 6; A2 / A3 at G ≥ 8 |
   | C-ECT | G ≥ 6 families (7 for a power ≥ 0.89 design target) |

5. **Decision rules**, each with what a chemist would do.

   | outcome | decision | what a chemist does |
   |---|---|---|
   | C-GD passes | B1 under-corrects heavy-selective non-DGA systems | Expect Eu/Gd to be harder and Gd/Tb easier than B1 says; read Gd, or apply the stratum amplitude estimated here once it has been re-confirmed |
   | C-GD point estimate ≥ β | the corpus heavy-selective depth was N-donor specific | Keep one amplitude |
   | A1 fails (mean ≥ 0, or the 97.5 % upper bound > 0 at G ≥ 8) | withdraw B1's Gd term for new chemotypes | Use plain radius interpolation (B0) |
   | A2 futility: the 97.5 % lower bound of the mean chord > β/2 | B1 harms the average new laboratory | Use B0 for new chemotypes |
   | A1 → A2 → A3 pass | B1 is the default for a new laboratory | Use B1 |
   | C-ECT passes | the own-series template is confirmed | Measure one full series per extractant, then 3–4 metals at each new condition |
   | C-ECT kill: mean log ratio ≤ 0 | the own-series template is dead | |
   | T0 validity: COPY is not better than B1 at 4 reads in pH-only cation-exchange families | the sibling definition or the data are faulty; analysis void | |

6. **Excluded** (not decidable with this supply):
   - pooled R² / E-A (P2: 37 to > 80 sources);
   - TOST transport of β (≤ 0.08);
   - the within-unit window flag (≤ 0.15);
   - SH at 2 reads (equal-weight ≤ 0.06);
   - the sc009 Nd dip (only 2 new DGA laboratories);
   - any DGA-only Gd test (anti-conservative and ≤ 0.1 power).

### Power by supply scenario (exploratory planning; non-DGA-like units; τ 0 / 0.065 where relevant)

| scenario | G_HS (C-GD) | G_Gd (C-B1) | G_ECT | C-GD | A1 | full C-B1 chain | C-ECT (SH at 4) |
|---|---|---|---|---|---|---|---|
| verified only (7 E1 labs) | 3 | 4–5 | 2–3 | below minimum (0.44–0.56 at G = 5) | 0.50 at G = 5 | below minimum | below minimum |
| expected (≈ 13 labs) | 6 | 8 | 5–6 | 0.62–0.74 | 0.83 | 0.36 / 0.23 | 0.63–0.67 at 6 |
| all E1 + E2 convert (16–19) | 8 | 10–12 | 7–9 | 0.88–0.94 | 0.93 | 0.52–0.65 / 0.33–0.45 | 0.89–0.99 |
| extended targeted search (acidic organophosphorus and chelators; P4 §7.2 expects 20–30 labs) | 10–13 | 13–16 | 7–9 | 0.97–1.00 | 0.99 | 0.70–0.83 / 0.50–0.62 | 0.89–0.99 |

**Reading.** The supply decides C-GD first, because it is the stratum the new literature fills. The supply action follows directly: target full-series (La–Lu, at least Eu, Gd and Tb) acidic-organophosphorus and chelator papers with at least 3 conditions per extractant. This is desk work plus the user's permission for full texts.

---

## 3. The ideas, briefly (full fields in the structured output)

- **confirm-1, CP-13 package.** As in section 2. Cost: hours to freeze; data-gated.
- **confirm-2, Gd-break depth in heavy-selective non-DGA chemistry (C-GD).**
  - Hypothesis: the new units' mean chord is below β = −0.092, about −0.2.
  - Comparator: B1's frozen β. Control: light-selective units and DGA (secondary).
  - Kill: point estimate ≥ β.
  - Power 0.88–0.94 at G = 8.
- **confirm-3, B1 transport by the model-free chord (C-B1).** A1 is decisive (0.83–0.99). The full chain needs G ≥ 13 (0.50–0.70).
- **confirm-4, mechanism-aware E-CT leg (C-ECT).**
  - Power ≥ 0.89 needs at least 7 non-trivial families.
  - T0 pH-scans serve as a validity control.
  - The "2 reads" version is excluded.

## 4. Process notes

**Files.**
- **Written:** this file, six scripts (`ideas2/scripts/confirm_*.py`) and their outputs in `ideas2/scripts/confirm_out/`. That includes `p1/`, a re-run of the unchanged `P1_ct_build.py` into my own directory (2.2 MB parquet).
- **Not done:** no existing file was modified, and there were no git commands, downloads or logins.

**Process.**
- One Python process at a time. The longest run took 149 s.
- Two web searches, for chemistry facts only. No pages were fetched and nothing was cached by me.

**Earlier probe notes re-checked.**
- P2 numbers reproduce for the pools shared with P2 (publication non-DGA by sc009 vs by SMARTS differ slightly: 21 units here).
- The M33 statement "sd ≈ 0.1" is corrected in §1.1.
