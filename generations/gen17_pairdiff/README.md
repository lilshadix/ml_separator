# gen17_pairdiff

Gen17 lead probe: within-publication pairwise difference learning (PADRE / DeepDelta), plus the
sibling anchor. It pairs well-determined cells from the same publication, so the lab effect cancels.
It then asks whether a model of delta a = a_i - a_j can recover transferable signal about the
amplitude a, the radius coefficient of the centred lanthanide log D curve. The candidate inputs are
condition differences (COND+MASSACT, 72 columns) and ligand topology or donor descriptors
(TOPO39, DONORS13). Side question: is copying a sibling cell's fitted curve a useful anchor? The
result files report an indicative null. Every model arm predicted delta a worse than predicting zero,
including on the cleanest contrast (same paper, identical conditions, different ligand).

## Status

**Unreviewed, exploratory.** The probe ran on 2026-09-09 and was already in the tree, uncommitted,
when the gen16 session started. It was committed on 2026-09-10 in `fdb1e15`, whose message says it
"has not been reviewed or re-run". `155dc6c` moved it here by pure rename; the scripts' paths were
re-pointed to `generations/` after that commit, so at `155dc6c` itself they fail when run from the
root. There is no pre-registration, decision report or confirmation. It is **not** the gen17 that
`generations/gen16_leads/DECISION_REPORT.md` section 12 proposes for pre-registration, which does not
include pairwise difference learning.

## Read first

1. `results/g17_summary.txt`: the only write-up (census, noise floor, LOPO test, clean contrast, anchor).
2. `results/g17_clean_ligand_contrast.txt` and `results/g17_sibling_anchor.txt`: raw outputs of the two small scripts.
3. `results/g17_lopo_pairs_ET400.txt` and `results/g17_lopo_pairs_ET150_ridge.txt`: sources of summary section 3.
4. The docstring of `g17_within_pub_pairs.py`: the claim, the leak rule and the falsification rule.

## Key files

| File | What it is |
|---|---|
| `g17_within_pub_pairs.py` | Main probe: ordered within-publication pairs, propagated noise floor, LOPO ExtraTrees (200 trees, min_samples_leaf 8) and Ridge (alpha 10), 400-draw publication-clustered bootstrap. Writes `results/g17_within_pub_pairs.csv`, which is not in the repository. |
| `g17_clean_ligand_contrast.py` | Same publication, identical 72-column conditions, different extractant; LOPO ET and ridge on topology and donor differences. |
| `g17_sibling_anchor.py` | Model-free anchor: copy one sibling's or the mean of siblings' coefficients; extractant-macro log SF MAE. |
| `results/` | The five text files listed above. No CSV or JSON. |

## Headline numbers (all LOPO, n_metals >= 5 subset: 289 of 521 cells, 41 publications)

- 4296 ordered pairs (2148 unordered); one publication supplies 34.5 % of them. Noise floor: |delta a| of 0.130 expected from replicate noise alone.
- Pooled pair MAE of delta a against the null of 0.1614. dCOND72 ET gives 0.1834 (+0.0220, pub-boot [+0.0083, +0.0443]). dCOND72 ridge gives 0.1745 (+0.0131, [-0.0160, +0.0639]). TOPOi+TOPOj+dCOND ET gives 0.1940 (+0.0326). Positive means worse than zero.
- Clean ligand contrast, 362 pairs from 15 publications, null 0.1830: all 8 arms are worse. The best is TOPOi|TOPOj ET at 0.1912 (+0.0082). No bootstrap.
- Sibling anchor, extractant-macro log SF MAE on 281 cells: zero curve 0.5429, copy-1 0.4258, copy-mean 0.3450. No model and no held-out fit, and BP forbids same-publication siblings.

## How to run (from the repository root; do not run casually, the main script writes into `results/`)

```bash
# 1. build the gitignored bench cache generations/gen14_direction/cache/bench.pkl (needs gen13's frozen inputs)
PYTHONPATH=generations/gen14_direction .venv/bin/python -c "from gen14 import dirbench; dirbench.load()"
# 2. the probes
.venv/bin/python -u generations/gen17_pairdiff/g17_within_pub_pairs.py
.venv/bin/python generations/gen17_pairdiff/g17_clean_ligand_contrast.py   # cwd-relative paths: root only
.venv/bin/python generations/gen17_pairdiff/g17_sibling_anchor.py          # cwd-relative paths: root only
```

On the Windows clone, use `.venv/Scripts/python.exe`, as the main script's docstring does. The `.venv`
lives in the main checkout, not in worktrees. The main script will **not** reproduce the saved
ET400 and ET150 files, which came from uncommitted variants with other estimator settings.

## Tests

None. `generations/tests/` and `generations/verify_relocation.py` do not reference this directory.

## Dependencies

- `generations/gen14_direction`: all three scripts unpickle `cache/bench.pkl`, built by `gen14/dirbench.py` `load()`.
- `generations/gen13_separation`: `gen13sep.basis.centre_rows` and `gen13sep.metals.LANTHANIDES`. Unpickling needs `gen13sep.amplitude_bench.BenchData`. `gen13sep/paths.py` puts `src/` on `sys.path`.
- Only when the cache is rebuilt, via `gen13sep/paths.py`: `dataset with 3D structures/`, `runs/gen7_architecture/cache/chemistry_map.parquet`, `runs/gen6_provenance/provenance_table.parquet` and `generations/gen12_2_eu_pred/features/coordination_descriptors.parquet`.

## Caveats

- **Not the project regime.** LOPO rather than BP with 5 seeds and the chemotype-blocked bootstrap. Each arm is a single run (random_state 0). Pooled over ordered pairs, so each unordered pair counts twice.
- **The result files do not match the committed main script.** The ET400 and ET150 files have other formats and settings (ET400's X is 4296 x 150). The CSV the script writes is absent.
- **Summary section 3 mixes runs.** The ET rows come from ET400. The ridge row and the stratified "+0.0086 (ET)" come from ET150. ET400 gives +0.0251 on the same-extractant stratum. The quoted deltas are LOPO point estimates, and the bootstrap means in the files differ (for example +0.0316 vs +0.0326).
- **Census and strata disagree.** Summary section 1 has 434 diff-extractant/identical-condition pairs (null 0.2498) and 1598 (null 0.0966). ET150 and the clean-contrast script have 362 (0.1830) and 1670 (0.1177). Section 2's signal-above-floor claim quotes the 434 figure. The directory does not explain this.
- **The clean contrast is concentrated.** 158 and 102 of its 362 pairs come from two publications, and 198 share a chemotype.
- **The anchor wording is stronger than the code.** The "SAME extractant, ANY publication" pool includes same-publication siblings, so 0.3479 is not a different-paper score, and each row scores a different cell subset. gen13's d1 atlas (`generations/gen13_separation/analysis/stage2/d1_curve_atlas/FINDINGS.md`) reports the opposite: a sibling from another paper is 0.11 worse (0.390 vs 0.286).
- The summary's "oracle direction x constant magnitude 0.436" has no generating script or result file here.
- The condition-contrast null agrees with gen15 section 8 `condfe` (within-publication conditions, Frisch-Waugh). `generations/gen16_leads/START_HERE.md` lists that as an established negative.

## Later corrections

- No later report reviews this directory. `generations/gen16_leads/STATUS.md` lists it as untouched prior-session work, and nothing in the gen16 report depends on it.
- `generations/gen15_curve/GEN15_REPORT.md` section 1a: an in-sample own-cell oracle is not a ceiling (0.1811 in sample vs 0.2736 leave-pair-out). The "ORACLE own quadratic (2-coef ceiling) 0.157" (0.1566 in `results/g17_sibling_anchor.txt`) is that kind of oracle.
- `generations/gen16_protocol/results/g16_size.log` (also unreviewed) puts the percentile bootstrap's rejection rate at 0.0705-0.0815 at nominal 0.05 (G = 40, 2000 sims). gen17's publication-clustered percentile intervals may be similarly anti-conservative. Every arm is worse than the null, so this cannot reverse any direction.
