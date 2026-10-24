# Gen21 round 4: report

**Status:** stage 1 blocks A, B and C complete (2026-10-23); fit poisoning 50/50 for every scored configuration, S11 included (`results/leakage/poison_S11.json`). The verdict is **no detected gain over the strongest reference (B1)**, and the study is **underpowered for confirmation** on the existing corpus. In block C (actinide transfer), the Ln-only multitask scaffold S11 gives no gain over B1 (equal on BPG, slightly worse on BP), and the An(III) transfer slot S13 was NOT RUN because its inner gate failed (3 of 50 fold-fits).

All model numbers are **exploratory development**:
- V1 data, C3 cohort, k = 3 (four measured metals per held-out cell), DOPT_Q read sets;
- 229 cells and 1,802 strict-adjacent pairs per seed, 5 dev seeds;
- BPG is the deciding design; BP is the robustness check.

No independent confirmation was run, because no source-independent data exist yet (DATA_REQUEST.md). Nothing has been committed or pushed.

## 1. What improves on interpolation

| model | R² BPG | R² BP | balanced MAE (BPG) | verdict |
|---|---|---|---|---|
| B0 = PWL_R (radius interpolation) | 0.4180 | 0.4179 | 0.1061 | the substantive baseline |
| **B1 = B0 + one Gd amplitude** (fold-fitted) | **0.4521** | **0.4519** | **0.1011** | **the strongest reference; nothing in round 4 beats it** |
| HIST_D7 (B1's form, fixed full-corpus −0.072) | 0.4511 | 0.4506 | 0.1011 | the same as B1 (−0.001); the learned part is one known number |
| B2 = 14 coherent potentials + read geometry | 0.4299 | 0.4246 | 0.1050 | worse than B1: a broad universal prior dilutes the Gd dip |
| S01 = B1 + donor-class Gd / Nd dips | 0.4520 | 0.4514 | | = B1 (ΔR² −0.0001 / −0.0005) |
| S02 = B1 + electron-count shapes × donor class | 0.4281 | 0.4242 | | worse than B1 (−0.024 / −0.028) |
| S05 = B1 + CatBoost correction (support summaries, donor class, RDKit descriptors) | 0.4459 | 0.4487 | 0.1022 | worse than B1 (−0.006 / −0.003); inner CV chose B1 in 16–19 of 25 folds |
| S06 = S05 + compact 3D geometry (with composition) | 0.4465 | 0.4409 | 0.1023 | worse than B1 (−0.006 / −0.011); the same on the 89% geometry-covered subset |
| S11 = B1 + Ln-only multitask scaffold (shared extractant encoder; residual and raw-log D heads) | 0.4515 | 0.4503 | 0.1011 | no gain: = B1 on BPG (−0.0005); slightly worse on BP (−0.0017 R²; −0.0020 balanced MAE, every interval excluding 0; no seed favours S11); inner CV chose B1 in 24 and 23 of 25 folds |

**B1's gain over interpolation** is +0.034 R² and +0.005 balanced MAE in 5/5 seeds on both designs. The cluster-bootstrap intervals exclude zero, but the small-cluster jackknife-t interval does not. The gain sits at Eu/Gd and Gd/Tb.

**0.50 is not reached.** B1 is 0.452, with a source-cluster interval of [0.394, 0.497].

## 2. Do f-shell, 3D or An information add incremental information?

- **f-shell / electronic structure: no evidence.**
  - S02, which conditions gen13's electron-count shapes (gd_break, tetrad e1 / e3) and the Gd term on donor class, is worse than B1.
  - S01, the D7 donor contrast of the Gd / Nd dips, equals B1.
  - The only electronic-looking effect that holds is B1's single shared Gd dip. It is a series / read-set prior. It does not show that the learner discovered half-filled-shell physics: a Gd-centred effect does not establish a mechanism.
- **3D geometry: no evidence.**
  - **What was added.** S06 adds the cell's own GFN2-xTB complex geometry to S05: Ln–donor distance, coordination number and metal charge, with composition-matched anomalies.
  - **Result.** It is worse than B1 and no better than S05, on the full cohort and on the geometry-covered subset. S06's positive-tree inner score beat both B1 and B2 in only 9 of 50 fold-fits, so the encoder slots (S07, S10) were not justified.
  - **Caveat.** The synthetic study shows the geometry route had little power at plausible signal sizes, so this is "not detected", not "refuted".
- **Actinide transfer: not detected; the transfer slot did not pass its gate.**
  - **Data.** 1,462 verified An(III) records (38 publications, 112 extractants; 88% Am) in the cohort's nitrate regime. Union exclusions by publication, copy group, extractant and Tanimoto-0.7 chemotype link are applied before any fit. After them, every fold keeps at least 557 rows, 12 copy groups and 77 extractants.
  - **Scaffold.** S11, the common Ln-only multitask scaffold, gives no gain over B1.
    - On BPG it equals B1 (ΔR² −0.0005).
    - On BP it is slightly worse: ΔR² −0.0017, and Δbalanced-MAE −0.0020 with every interval excluding 0. No seed favours S11.
    - Its inner CV chose B1 in 24 and 23 of 25 folds.
  - **Transfer gate.** The G-transfer count part passed. Its inner part failed: S13 (An(III) data through a private head) beat both S11 and B2 in only 3 of 50 inner fold-fits, and 30 are required. By the sealed route, S13 and its controls S14–S16 were NOT RUN, and their slots are not consumed.
  - **Metal-state control.** S24 was NOT RUN: the eligible An set is 83–93% Am per fold, so it carries almost no metal-state information.
  - **Caveat.** In the synthetic study, the transfer claim fired in 0 of 6 replicates even when the An targets carried the property that drives the Ln signal (the An data added at most +0.002 R² over S11). This is therefore "not detected; control power not shown", not "refuted".
  - **History.** Earlier An experiments (gen11, gen19, strain) also did not establish a lanthanide benefit.

## 3. Where it fails

- **Light and heavy ends.** La/Ce, Tm/Yb and Yb/Lu stay near R² 0.0–0.35 for every model. Tm/Yb and Yb/Lu are about 0 under DOPT_Q reads, which rarely read Lu.
- **Eu/Gd and Gd/Tb.** These are where B1's Gd term helps (0.28 → 0.42 and 0.32 → 0.55), and where S02's extra flexibility hurts.
- **Few independent sources.** One source cluster holds 65% of rows (Kish 2.2 of 13), so every interval is fragile.

## 4. Practical separation implications

- **To predict an adjacent separation factor from four measured metals:** use radius interpolation plus the Gd-dip correction (B1, or the fixed −0.072, which works equally well).
- **No tested chemistry, electronic or donor-class conditioning improves on that,** and neither does a learned extractant encoder (S11). An(III) auxiliary data did not pass the inner gate.
- **Where to spend measurements.** Adding measured metals is the lever that has been shown to work: round 3 found 0.50 needs about 6 metals by seed mean, and 7 by lower bound. Read placement (POL_ERR) helps the heavy pairs but trades error between pairs.

## 5. Remaining uncertainty and what would change it

- **Statistical power.** About 40 new independent chemotype-like sources at minimum, and possibly over 80, would be needed to confirm even a B1-sized gain under the sealed rule (POWER.md). The 0.005 balanced-MAE margin effectively requires a B1-sized gain.
- **Data quality** (PROVENANCE.md):
  - the C5-BTBP cells' attribution and floor are unverified;
  - the DMDPhPDA 5 M Pr/Nd value awaits Shimada's thesis;
  - ORNL imports lack data locations.
- **Not run:**
  - S03, S04 and S21 (block-A routes closed);
  - S12, S17 and S18 (gates failed);
  - S13, S14, S15 and S16 (G-transfer's inner part failed, 3 of 50) and S24 (its label-free count gate failed);
  - S19, S20 and S23 (no candidate to combine or rescale);
  - S07, S08, S09, S10 and S22 (block-B routes closed: S05 and S06 failed criterion 1, and the encoder gate failed).

## Evidence and reproducibility

- **Protocol and evidence:** PROTOCOL.md (§0–§12), AMENDMENTS.md (A1–A5), RUN_RECORD.md, EVIDENCE_INVENTORY.md, PROVENANCE.md, POWER.md and DATA_REQUEST.md.
- **Leakage chain**, for every scored configuration:
  - isolation tables;
  - non-training cache poisoning (50/50);
  - fit poisoning (50/50);
  - prediction intervention with rebuilt methods (max change 0.0);
  - support exactness;
  - per-fold identities.
- **Results:** `results/<ID>/` (summary, contrasts, levels, per-pair, per-publication, calibration, identity, intervention, TABLE.md); mechanical decisions in `results/blockA_decision.md`, `results/blockB_decision.md` and `results/blockC_decision.md`; the G-transfer gate in `results/gate_c.json`.
- **Rerun:** RESUME.md.
