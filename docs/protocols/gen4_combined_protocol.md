# Generation-4 proposal: combining `ml_separator` and `lanthanidestrain`

Status: **draft for discussion — not frozen**. Written 2026-08-15 after a
side-by-side audit of this repository and
[mironovb/lanthanidestrain](https://github.com/mironovb/lanthanidestrain).
Nothing here modifies the frozen gen3 protocol; gen3 should run as designed
(the torch contract bug is fixed) before any of this is adopted.

---

## 1. Which repository is better?

Different questions, so "better" splits by dimension. Summary: **lanthanidestrain
is the scientifically stronger project today**, mostly because of its data
breadth and its chemistry-level explanation of the 3D null; **ml_separator has
the stronger evaluation machinery per row of data**. The right move is not to
pick one but to merge their strengths, which is what this document specifies.

| Dimension | ml_separator | lanthanidestrain | Winner |
|---|---|---|---|
| Cohort breadth | 34 extractants, 1 family, n_eff = 4.8 | 162 extractants, 14 lanthanides, 4,746 measurements | **lanthanidestrain** |
| Target construction | antisymmetric pair target `log D_A − log D_B`, exact condition matching, structural antisymmetry | `log D` levels + derived adjacent separations; "train the contrast" arrived late as a finding | **ml_separator** |
| Primary metric discipline | equal-extractant macro MAE, `never_select_on: pooled` hard-coded in the frozen protocol | headline CatBoost R² 0.518 is a pooled row-level number; macro/micro trap present | **ml_separator** |
| Negative controls | per-block shuffled twins, 3 shuffle seeds × 5 model seeds — decisive (geometry 0/4, E3 15/15) | kill-checks are post-hoc but sharp (scale-free R², dispersion, single-scalar recovery) | **ml_separator** (prospective) |
| Reproducibility infra | fail-closed software/hash contracts; run fails on any drift | `control_guard` pins 324 artifacts by SHA-256; `--deterministic` inertness proofs | tie — adopt both |
| Pre-registration | frozen protocol JSON before experiments | per-test `*_PREREGISTRATION.md` committed before data exists | tie — adopt both |
| Scientific yield | a clean, control-backed null on Architector 3D | the *mechanism* for that null (GFN2 lanthanide contraction 2.47× under, linear-in-Z by construction) + noise-ceiling analysis | **lanthanidestrain** |
| Known leakage risk | headline run used `group_mode=extractant` against its own audit's CRITICAL finding (ECFP homolog twins, 49.8 % of rows) | duplicate-collapsing cluster bootstrap found and fixed (intervals were 12–29 % too narrow) | both had one; both documented |

The single most consequential cross-repo fact: **lanthanidestrain explains
ml_separator's geometry null.** Our structures are GFN2-xTB-derived; their
compliance experiment shows GFN2 encodes the lanthanide contraction at 0.405×
reality with only 23 % ligand-consistent response, because every Ln parameter
from Ce to Lu is a linear interpolation between two anchors. A geometry block
built on that Hamiltonian *cannot* carry pair-differencing signal — which is
exactly what our shuffle controls measured (A5/G2/G4/S4 all ≤ 2/15 vs shuffle).
The two projects independently reached the same null and one of them explains it.

## 2. Ten candidate improvements

Ranked candidates; §3 picks three. Evidence tags: [ms] = measured in this repo,
[ls] = measured in lanthanidestrain.

1. **Expand the cohort by relaxing the complete-conditions constraint.**
   Requiring all 64 condition values collapses ~190 extractants to 34
   (n_eff = 4.8) [ms]. Impute per-fold with missingness indicators (the
   pipeline already fits `SimpleImputer` inside folds) or match pairs on a
   reduced condition core. lanthanidestrain runs 162 extractants [ls]. The
   power ceiling moves more than any model change can.
2. **Make `ecfp_exact_cluster` the primary grouping.** Our own audit marks it
   CRITICAL: 49.8 % of rows have a bit-identical ECFP homolog across the fold
   boundary, inflating the A2 baseline specifically [ms]. One flag change.
3. **Run gen3 H1: antisymmetric CatBoost, group-equal weights, MAE/Huber
   loss.** Now unblocked. lanthanidestrain measured CatBoost ≫ LightGBM on the
   same features (0.518 vs 0.459 pooled) and MAE-instead-of-RMSE as its single
   largest selectivity gain (+0.107) [ls]; gen3 H1 pre-registers exactly this
   grid [ms].
4. **Run gen3 H2: the E3 electronic residual head.** E3/ELEC_PAIR is the only
   block that beat its shuffled twin (15/15) [ms]; xTB electronics is also a
   top single block in lanthanidestrain's shortlist [ls].
5. **Recompute electronic descriptors with g-xTB instead of GFN2-xTB.** GFN2's
   electronic response is linear-in-Z by construction (HOMO–LUMO residual
   0.00075 eV vs g-xTB's 0.278 eV; +0.012 eV vs +1.15 eV gadolinium break)
   [ls]. Our one real signal block (E3) is currently built on the Hamiltonian
   with almost no metal-identity information beyond Z. Highest-risk,
   highest-ceiling change; needs the g-xTB binary and a new frozen protocol.
6. **Cross-repo transfer: pretrain on lanthanidestrain's 4,746 `log D`
   levels, difference antisymmetrically, fine-tune on our pairs.** Their data
   covers 162 extractants; our gen3 H3 architecture (`g(A) − g(B)`) is exactly
   the form that can consume a level-pretrained `g` [ms+ls].
7. **Stack predictions, not representations.** Cross-fitted OOF stacking of
   the A2 tree, the E3 head and (if revived) the SNN — lanthanidestrain's SNN
   earns +0.038 by complementarity while losing alone [ls]; their embedding
   probe (fold identity recoverable at 100 %) is the reason to stack
   predictions only [ls].
8. **Estimate the noise ceiling before chasing further gains.** Their repeated
   (composition, pair) analysis gives separation reproducibility 0.1533 vs
   spread 0.2236 → R² ceiling ≈ +0.53 [ls]. We discard replicates
   (`replicate_policy: unique`) that would give us the same bound; keep them
   in a side table and compute the macro-MAE floor per extractant.
9. **Adopt their scale-free kill-checks as standing gates.** Pearson² after
   optimal rescaling, prediction-dispersion ratio, and single-scalar-recovery
   killed a pre-registered +0.0333 "win" as calibration [ls]. Cheap columns in
   our aggregate report; they close the one artifact class our shuffles miss.
10. **Extend the extractant-level bootstrap to all 74 paired comparisons**
    (only A2–A5, A2–A6, A3–A2 have intervals today [ms]) and verify the
    resampler keeps duplicate labels as independent entries — the
    interval-narrowing bug lanthanidestrain shipped and corrected [ls]. Gen3
    already words this correctly; gen2's aggregate should be re-checked.

## 3. The three to do first

1. **#1 + #2 together: re-run the gen2 ladder on the expanded cohort under
   ECFP-cluster grouping.** Everything else is interpretation-limited until
   the evaluation has more than ~5 effective groups and no homolog twins
   across folds. This one run decides whether A2's championship is real and
   gives every later comparison usable power.
2. **#3 + #4: execute gen3 as frozen.** It already encodes lanthanidestrain's
   two biggest measured wins (CatBoost + group weights, MAE-family losses) and
   targets the one control-passing block (E3). Zero new design work; the
   blocker was one version string.
3. **#6: cross-repo transfer + #7 prediction stacking.** The only route to
   "better than lanthanidestrain" rather than "equal to it": their model has
   never seen our exact-condition-matched pair target, and our model has never
   seen their 162-extractant chemical diversity. Pretrain the level model on
   their table, wrap it antisymmetrically, stack OOF predictions under our
   contracts. Pre-register before running; gate on the macro metric and a
   shuffled-pretraining control.

\#5 (g-xTB descriptors) is the most scientifically interesting but sits behind
a binary dependency and a protocol re-freeze; schedule it as generation 5 if
gen4 stalls below the noise ceiling from #8.

## 4. Metric contract for any combined run

Carried forward from the best of both:

- **Primary:** equal-extractant macro MAE (one extractant, one vote), cluster
  bootstrap at the extractant level, duplicates kept as independent entries.
- **Selection:** never on pooled MAE / pooled R² (`never_select_on`, frozen).
- **Secondary:** median and worst-quartile extractant MAE, sign accuracy,
  adjacent-Ln vs non-adjacent MAE, fraction of extractants improved.
- **Standing gates for any claimed win:** (a) beats its own shuffled twin on
  the macro metric; (b) survives Pearson²-after-rescaling; (c)
  prediction-dispersion ratio reported; (d) no single scalar on the reference
  arm recovers the gain; (e) 4/5 split seeds agree in sign.
- **Reporting:** n = number of extractant groups, never rows; n_eff (Kish)
  quoted next to every interval.

## 5. Large-artifact policy (adopted 2026-08-15)

Cluster export and GitHub both cap files at 100 MB.
`scripts/compress_run_artifacts.py` gzips any run CSV ≥ 95 MB with a
deterministic container (`mtime=0`), verifies the decompressed SHA-256 against
`artifact_hashes.json`, and writes a `transport_manifest.json` beside the
files; `--restore` reverses it byte-identically, so hash contracts survive
transport. Applied to the two local gen2 OOF files: 124.9 MB → 18.8 MB (15 %).
On the cluster, run it over the remaining run directories before `scp`/`rsync`.
