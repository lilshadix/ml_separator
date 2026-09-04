# gen11 §16 — query-design consistency of the auxiliary-trained arms

**Question.** GEN10's known limit is that its prediction for a point depends on which *other* points the user listed. Does training on the auxiliary multi-metal archive make that dependence better, worse, or unchanged?

## Correctness check

- Design-blind control (`FROZEN_MONOLITH`, gen10's frozen monolith, representation `NONE`): max shift over every perturbation = **0.0** (tolerance 1e-12); passes = **True**.
- Exact invariances (`CONTEXT`, `PERMUTE`, every arm): max shift = **8.88e-16**; passes = **True**.

Both are properties the harness cannot fail unless it is wired wrong, which is exactly why they are the check: a non-zero design-blind control would mean the benchmark was measuring its own plumbing rather than the model.

- Reproduction of gen10's own run: the gen11 anchor (`A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1`, empty auxiliary block) against gen10's `SHAPE_RECOMPOSED` on 3128 shared (seed, fold, ligand, curve, variant) units — max |Δ| = **0**, reproduces = **True**. The perturbations are drawn from `default_rng([seed, fold])` created fresh per ligand, so the two runs ask literally the same questions; this is the receipt that gen11 measures the frozen benchmark rather than a nearby one.

## What was run

- **2 split seeds** ([104729, 130363]) — a deliberate reduction from gen10's five, which took 4.6 h for three arms. Say "2 seeds" whenever these numbers are quoted.
- All 9 perturbation families were kept (REFERENCE, CONTEXT, NESTED, DENSITY, EXTEND, PERMUTE, SPARSE, SHIFT, DECOY). The seed count is the only thing cut; the perturbations never are.
- 8 measured ligands per fold, over 10 completed (seed, fold) blocks, at most 12 curves each (gen10's own cap): **49 distinct ligands, 145 curves, 16085 comparisons**. `sampling.json` lists every one.
- Arms present: FROZEN_MONOLITH, A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1, A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1, C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1, E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1.
- Auxiliary pool: 5096 cells, mean 4247 admissible per fold after the leakage filter. Cohort fingerprint `bed178ec1a7a82b0`.

## Movement of the *unchanged* points, in log units

Each row is one arm under one perturbation family. `median` is the median over comparisons of the per-comparison median shift; `p95` the median of the per-comparison p95; `p95-of-p95` the 95th percentile of those; `worst` the single largest movement seen. `level` and `shape` split the movement into the part one calibration measurement could absorb and the part it could not.

| arm | family | n | median | p95 | p95-of-p95 | worst | level | shape |
|---|---|---|---|---|---|---|---|---|
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FROZEN_MONOLITH | CONTEXT | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.0783 | 0.2508 | 0.6962 | 0.9422 | 0.0225 | 0.0818 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.0918 | 0.2446 | 0.6742 | 0.8102 | 0.0186 | 0.0850 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.1380 | 0.3421 | 0.7661 | 0.9666 | 0.0173 | 0.1432 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.1625 | 0.3411 | 0.7944 | 1.1884 | 0.0158 | 0.1707 |
| FROZEN_MONOLITH | DECOY | 178 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0105 | 0.0175 | 0.1984 | 0.9751 | 0.0105 | 0.0022 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0115 | 0.0192 | 0.2386 | 1.1075 | 0.0118 | 0.0024 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0143 | 0.0235 | 0.3664 | 1.1871 | 0.0145 | 0.0034 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0167 | 0.0258 | 0.3452 | 1.2448 | 0.0166 | 0.0031 |
| FROZEN_MONOLITH | DENSITY | 267 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0634 | 0.1768 | 0.5904 | 0.9739 | 0.0371 | 0.0616 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0610 | 0.1764 | 0.5986 | 1.0060 | 0.0396 | 0.0608 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0807 | 0.2397 | 0.6855 | 1.1994 | 0.0504 | 0.0898 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0806 | 0.2474 | 0.7109 | 1.4000 | 0.0547 | 0.0940 |
| FROZEN_MONOLITH | EXTEND | 534 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0300 | 0.0913 | 0.4146 | 0.9299 | 0.0204 | 0.0257 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0298 | 0.0876 | 0.4002 | 0.8576 | 0.0206 | 0.0253 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0357 | 0.1091 | 0.4789 | 1.1009 | 0.0235 | 0.0297 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0369 | 0.1160 | 0.5048 | 1.1249 | 0.0254 | 0.0318 |
| FROZEN_MONOLITH | NESTED | 970 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FROZEN_MONOLITH | PERMUTE | 194 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0351 | 0.0494 | 0.4897 | 1.0448 | 0.0246 | 0.0101 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0345 | 0.0472 | 0.5034 | 0.9891 | 0.0260 | 0.0095 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0351 | 0.0518 | 0.6508 | 1.2423 | 0.0279 | 0.0104 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0382 | 0.0532 | 0.6218 | 1.3114 | 0.0275 | 0.0106 |
| FROZEN_MONOLITH | SPARSE | 791 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Paired difference from the design-matched control

Baseline: `A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` — the control at the same design corner (`GENERAL` metal scheme, `ANNOTATION_SAFE` MASSACTION) every auxiliary arm is forced to. Comparing against gen10's frozen corner instead would confound the design change with the transfer. Units are paired: the same seed, fold, ligand, curve and variant asked of both arms. Positive = the auxiliary arm is *less* stable.

| arm | family | n units | baseline median | arm median | paired Δ | frac worse | sign-test p |
|---|---|---|---|---|---|---|---|
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | +0.0000 | 0.577 | 0.213 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.0918 | 0.0783 | +0.0000 | 0.552 | 0.261 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0115 | 0.0105 | +0.0000 | 0.459 | 0.264 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0610 | 0.0634 | +0.0000 | 0.627 | 4.12e-07 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0298 | 0.0300 | +0.0000 | 0.530 | 0.0857 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | +0.0000 | 0.450 | 0.315 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0345 | 0.0351 | +0.0000 | 0.499 | 0.97 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | +0.0000 | 0.562 | 0.314 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.0918 | 0.1380 | +0.0216 | 0.791 | 7.06e-12 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0115 | 0.0143 | +0.0000 | 0.576 | 0.0359 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0610 | 0.0807 | +0.0101 | 0.771 | 1.05e-28 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0298 | 0.0357 | +0.0025 | 0.647 | 9.21e-18 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | +0.0000 | 0.487 | 0.855 |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0345 | 0.0351 | +0.0000 | 0.538 | 0.0475 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | CONTEXT | 194 | 0.0000 | 0.0000 | +0.0000 | 0.558 | 0.332 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DECOY | 178 | 0.0918 | 0.1625 | +0.0400 | 0.910 | 4.28e-24 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DENSITY | 267 | 0.0115 | 0.0167 | +0.0000 | 0.615 | 0.00126 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | EXTEND | 534 | 0.0610 | 0.0806 | +0.0124 | 0.791 | 3.94e-33 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | NESTED | 970 | 0.0298 | 0.0369 | +0.0043 | 0.706 | 6.84e-34 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | PERMUTE | 194 | 0.0000 | 0.0000 | +0.0000 | 0.443 | 0.239 |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | SPARSE | 791 | 0.0345 | 0.0382 | +0.0003 | 0.560 | 0.00147 |

## The other half of the trade: accuracy

Macro MAE of the same arms on the same 2 seeds, read from the primary run's OOF parquets under `runs/gen11_transfer/arms/`. §16 is a trade — instability is only a verdict once it is set against what the arm bought.

| arm | macro MAE | shape MAE | status |
|---|---|---|---|
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | 0.9705 | 0.4866 | read |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | 0.9756 | 0.4869 | read |
| C_LN_PLUS_ACTINIDES\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | 0.9189 | 0.4843 | read |
| E_LN_PLUS_ALL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | 0.9209 | 0.4839 | read |

## The answer

Judged on `DECOY` — two candidate points two decades outside the measured window, the perturbation gen10 published its ~0.12 limit on — and calling anything under 0.02 log units unchanged:

- `A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1` (design corner only, no auxiliary rows) under `DECOY`: **UNCHANGED** — paired median shift moves +0.0000 log units (+0.0 % of the control's 0.0918), sign-test p = 0.261.
- `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` (auxiliary training) under `DECOY`: **WORSE** — paired median shift moves +0.0216 log units (+23.5 % of the control's 0.0918), sign-test p = 7.06e-12.
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` (auxiliary training) under `DECOY`: **WORSE** — paired median shift moves +0.0400 log units (+43.6 % of the control's 0.0918), sign-test p = 4.28e-24.

Read the per-family table above before generalising: a family where the paired delta is small but the control's own movement is large is still a model whose answer depends on the question, and auxiliary training was never expected to *fix* that — only not to make it worse.
