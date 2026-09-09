# Audit of the "direction of selectivity is predictable from donor topology" claim

Scope: the claim quoted below, audited against `gen13_separation/analysis/stage3/s3_direction.py`,
`s3_direction_accuracy_fixed.csv`, `s3_direction_gain.csv` and `s3_direction_figure.py`.
All work is in this directory; nothing outside it was created or modified.

> "The direction of lanthanide selectivity ... is predictable from donor-set topology alone for an
> extractant whose whole chemical family was held out, at 0.77 macro accuracy under the
> publication-masked design against 0.56 for always predicting heavy-selective, a gain of +0.21 with a
> chemotype-blocked 95% interval of [0.07, 0.43] and p = 0.002; and the 39-column donor-topology family
> does this better than the full 137-column compact feature set (0.77 versus 0.72)."

**Verdict: the primary result survives. Two of its four numbers need restating, and the secondary
claim (39 beats the full set) should be dropped or heavily qualified.**

---

## 1. Reproduction

**Step 1 — independent re-derivation, written without reading `s3_direction.py`** (`v_core.py`,
`v_headline.py`). Own cohort filter, own label, own folds, own L2 logistic classifier with
chemotype-balanced weights, own chemotype-blocked bootstrap:

| features | macro acc | always-heavy | gain | 95 % chemotype-blocked CI | p |
|---|---|---|---|---|---|
| topology 39 | **0.775** | 0.5586 | +0.217 | [0.032, 0.496] | 0.017 |
| PHYSCHEM+DONORS+COORD 137 | 0.719 | 0.5586 | +0.160 | [-0.008, 0.418] | 0.065 |

**Step 2 — replication of the original estimator** (`v_runner.py`, `v_replicate.py`): reproduces the
published value **bit-for-bit**, `0.7683085207475452`, gain `0.2097560975609756`, CI `[0.0673, 0.4278]`,
p `0.003` (published `[0.0667, 0.4254]`, p `0.0022` — bootstrap Monte-Carlo noise).

Difference between my number (0.775) and theirs (0.768): estimator only. Well inside 5 %.
Across four defensible estimators the headline sits at **0.70–0.82** (RF 0.702, ExtraTrees 0.768,
logistic 0.775–0.821). 0.77 is a fair central value; the estimator spread is ±0.06.

## 2. The target — survives

* **Ridge vs plain least squares:** 2 of 289 labels change. Macro 0.7679 vs 0.7683, gain identical.
  No dependence on the ridge.
* **`n_metals` threshold:** ≥4 → 0.791 (+0.248, p 0.0002); ≥5 → 0.768 (+0.210, p 0.003);
  ≥6 → 0.768 (+0.174, p 0.008); ≥8 → 0.842 (+0.211, p 0.006); ≥14 → 0.744 (+0.144,
  **[-0.078, 0.339], p 0.20**). The ≥14 failure is a power failure, not a contradiction: only 33
  extractants / 23 chemotypes survive, and 33 of the 78 cells are lost to the script's own
  `len(tr) < 40` guard once publication masking bites.
* **Near-zero band — this one bites.** 38 % of cells have |radius coefficient| < 0.1, i.e. a whole
  La→Lu contrast under 0.32 log units (basis span 3.147; median replicate sd 0.113, pair sd 0.16).
  Refitting with a band excluded from both sides:

  | kept | macro | always-heavy | gain | CI | p |
  |---|---|---|---|---|---|
  | \|amp\| ≥ 0.05 | 0.833 | 0.592 | +0.241 | [0.084, 0.478] | 0.003 |
  | \|amp\| ≥ 0.10 | 0.890 | 0.724 | +0.166 | [0.049, 0.367] | 0.009 |
  | \|amp\| ≥ 0.20 | 0.848 | 0.787 | **+0.061** | **[-0.032, 0.187]** | **0.18** |
  | \|amp\| ≥ 0.30 | 0.847 | 0.771 | +0.076 | [-0.031, 0.221] | 0.15 |

  The failure mode the brief worried about — near-zero cells being called right by chance — is **not**
  what is happening. Stratified by margin (`v_unseen.py`) the model is near chance where the label is
  weak and excellent where it is strong: macro 0.550 (<0.05), 0.601 (0.05–0.1), 0.818 (0.1–0.2),
  0.903 (0.2–0.5), 0.917 (>0.5). What is happening is that *the baseline* is terrible in the weak band
  (always-heavy scores 0.27 at 0.05–0.1 and 0.875 above 0.5), so the **gain** is concentrated in cells
  whose direction is only 1–2 pair-noise sd across the whole series. Above 0.2 there is no room left and
  the advantage is not separated from zero.

## 3. The units — survive

* Macro is genuinely **one vote per extractant**: 82 unit rows from
  `groupby(["extractant","chemotype"]).mean()`. Confirmed.
* The bootstrap **resamples the 40 chemotypes**, taking all of a drawn chemotype's extractants.
  Confirmed in code and reproduced.
* **Dominant chemotype removed** (sc009 = 165 of 289 cells, 23 of 82 extractants): macro 0.729 against
  a 0.444 baseline, gain **+0.285 [0.100, 0.470], p 0.004**. The result strengthens.
* Fragility to disclose: **61 of 82 extractants contribute exactly one cell** (one contributes 63), and
  **29 of 40 chemotypes hold a single extractant**. Three-quarters of the macro votes are one Bernoulli
  draw.
* **The 39 topology columns take only 22 distinct values across the 82 extractants**; one vector covers
  14 chemotypes and 40 extractants, and 12 rich cells are all-NaN (imputed). **84 % of held-out cells
  have a bit-identical topology row already in training.** This is not leakage — the descriptor is
  legitimately computable for a new ligand — but "an extractant whose *whole chemical family* was held
  out" is a statement about the chemotype label, not about the feature representation, and should be
  said that way. On the 16 % with genuinely novel topology the model still gains **+0.396
  [0.078, 0.650], p 0.013**, so it is not purely a lookup.

## 4. Leakage — clean

`assert_fold_integrity` passes on all 25 BP folds, and re-checked after restricting to the ≥5-metal
subset actually used: **0** extractant, **0** ECFP-cluster, **0** chemotype and **0** publication
overlaps between train and test. Labels are used only as `fit(X[tr], y[tr])`; the median imputer is
fitted inside the pipeline on training rows only; no target token can enter a feature column
(`features._assert_allowed`). A 40-replicate extractant-level label permutation gives mean gain
-0.067, sd 0.062, **max +0.056** — the observed +0.21 is never approached, so the bootstrap p is not
anticonservative.

## 5. The comparison — the baseline is fair, but it is the wrong one to headline

Every trivial constant reduces to the published one or is worse:

| baseline | macro |
|---|---|
| always heavy | 0.5586 |
| per-fold training majority, unweighted | 0.5586 (identical rule) |
| training majority within the cell's chemotype-size tertile | 0.5586 (identical rule) |
| per-fold training majority, chemotype-weighted | 0.347 |
| always light | 0.4414 |

So "always heavy" is the strongest constant available and the claim is not gaming it.

But there are **stronger non-trivial baselines**, and this is the substantive objection:

| baseline (same ExtraTrees, same folds) | macro |
|---|---|
| DONORS census, 13 cols | **0.694** |
| COORD block minus the topology family, 75 cols | 0.691 |
| PHYSCHEM, 10 cols | 0.638 |
| `chem__dentate` alone | 0.610 |
| COND, 64 cols (negative control) | 0.421 |
| MolWt alone | 0.366 |

Topology beats the donor census by **+0.075 [0.022, 0.151], p 0.006** — real, but a quarter of the
+0.21 the claim advertises. Most of the +0.21 is "donor chemistry predicts direction", not
"donor *topology* predicts direction".

## 6. The secondary claim — does not survive as stated

* **The column count is wrong.** `lean_all` in `s3_direction.py` is `LEAN_BLOCKS` =
  COND 64 + MASSACT 8 + PHYSCHEM 10 + DONORS 13 + COORD 114 = **209 columns**, not 137.
  `s3_direction_figure.py` labels that bar "full compact set (137 columns)". 137 is
  PHYSCHEM+DONORS+COORD, which was never run in this comparison. (Coincidentally the 137-column set
  also scores ~0.72–0.73, so the plotted value is not wrong — only its label.)
* **The difference is inside its own noise.** Paired chemotype-blocked bootstrap:
  topo39 − lean209 = **+0.047 [0.0005, 0.096], p 0.049** — the interval clears zero by 0.0005.
  Against the actual 137-column set: **+0.035 [-0.015, 0.070], p 0.199** — inside noise.
  And the estimator spread on the topology arm alone (0.70–0.82) is wider than the 0.05 being claimed.
  The ordering topo > full does hold under every estimator I tried (ET +0.047, logistic +0.233), so it
  is not a fluke of sign — but "39 columns beat the full set" is not supported at the precision it is
  stated.

## 7. Restated claim

> Under the publication-masked chemotype hold-out (design BP), a 39-column donor-topology block calls
> the direction of lanthanide selectivity for **0.77** of held-out extractants (macro over 82
> extractants; **0.70–0.82** across reasonable estimators) against **0.56** for always predicting
> heavy-selective — a gain of **+0.21**, chemotype-blocked 95 % interval **[0.07, 0.42]**, **p ≈ 0.003**
> (40 label permutations give a null gain of −0.07 ± 0.06). The result is unchanged if the label comes
> from an unregularised fit, holds at 4-, 6- and 8-metal thresholds, and strengthens when the dominant
> diglycolamide chemotype is removed entirely (0.73 against a 0.44 baseline, +0.28).
>
> Two qualifications belong in the sentence. First, the honest comparator is a donor census, not a
> constant: a 13-column donor-composition block already reaches **0.69**, so the topology family adds
> **+0.075 [0.02, 0.15], p 0.006**. Second, the gain over always-heavy lives in cells whose whole-series
> contrast is below about 0.6 log units; restricted to more strongly directed cells it falls to
> **+0.06 [−0.03, 0.19]**, not distinguishable from zero.
>
> The 39-column block scores above the full **209**-column (not 137) compact set, 0.77 against 0.72, but
> the paired difference is **+0.05 [0.00, 0.10]** and should be reported as "no worse than", not as
> "better than".

## Files

| file | contents |
|---|---|
| `v_cache.py`, `bench_cache.pkl` | frozen bench snapshot (cohort, basis, coefficients, feature blocks) |
| `v_core.py`, `v_headline.py`, `v_headline.csv` | independent re-derivation (step 1) |
| `v_runner.py`, `v_replicate.py` | faithful re-implementation of the original estimator |
| `v_probe.py`, `v_diag.py` | column counts, label margin, unit structure, chemotype sizes |
| `v_lookup.py`, `v_topology_lookup.csv` | leakage checks; 22-distinct-vector / 84 %-exact-match finding |
| `v_attack.py`, `v_attack_target_units.csv` | threshold sweep, OLS label, sc009 removal, margin bands |
| `v_baselines.py`, `v_baselines.csv` | constant, stratum and feature-block baselines; estimator sweep |
| `v_unseen.py`, `v_unseen_and_margin.csv` | unseen-topology subset; accuracy by margin band |
| `v_final.py`, `v_contrasts.csv`, `v_perm_gains.npy` | paired contrasts; permutation null |
| `v_scale.py` | translation of the radius coefficient into log-D units |

## Incidental note

`s3_direction_accuracy.csv` was overwritten by another job while this audit ran; it now holds only
BQ/BR rows and no longer contains the A/B/BP rows quoted above. The A/B/BP values I audited are in
`s3_direction_accuracy_fixed.csv`, which is unchanged. Separately, the *unfixed* CSV's
`always_heavy` row was wrong (0.347) because the rule was implemented as the *chemotype-weighted*
training mean, which is below 0.5 and therefore predicted **light** in most folds; the "fixed" file's
0.5586 is correct and matches my independent value to 16 digits.
