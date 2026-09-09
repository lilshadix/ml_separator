# Gen16 environment check

Generated 2026-09-10 00:34:42 by `gen16_leads/scripts/g16_env_check.py` on PCdiasa (Windows-10-10.0.19045-SP0).  Machine-readable copy: `env_check.json`.  Total run 67.9 s.

## Verdict

- no drift detected: versions, frozen import path, fold determinism and both cheap anchors match the locked generations

## 1. Versions

Interpreter: `D:\ml_separator_gh\.venv\Scripts\python.exe` (Python 3.14.5).

| package | installed | dist-info written | gen8 doc (2026-08-20) | gen13 session memo (2026-09-07) |
|---|---|---|---|---|
| numpy | 2.5.3 | 2026-09-07 22:47 | 2.5.1 | 2.5.3 |
| pandas | 3.0.5 | 2026-09-07 22:48 | 3.0.5 | 3.0.5 |
| scikit-learn | 1.9.0 | 2026-09-07 22:48 | 1.9.0 | 1.9.0 |
| scipy | 1.18.1 | 2026-09-07 22:47 | 1.18.0 |  |
| rdkit | 2026.3.6 | 2026-09-07 22:47 | 2026.03.5 |  |
| joblib | 1.6.0 | 2026-09-07 22:48 |  |  |
| pyarrow | 25.0.1 | 2026-09-07 22:47 |  |  |
| torch | 2.14.0 | 2026-09-07 22:53 | 2.13.0 | 2.14.0+cpu |
| catboost | 1.2.10 | 2026-09-07 22:48 |  |  |
| xgboost | 3.4.1 | 2026-09-07 22:48 |  |  |
| lightgbm | 4.7.0 | 2026-09-07 22:48 |  |  |
| threadpoolctl | 3.6.0 | 2026-09-07 22:47 |  |  |
| tabpfn | 2.2.1 | 2026-09-09 15:18 |  |  |

Module strings: {'rdkit.__version__': '2026.03.6', 'torch.__version__': '2.14.0+cpu'}.

Generation commits (all after the numeric stack was installed):

- `270ae5c` 2026-09-09 07:09:54 +0500 Add gen13 separation-first generation: stage 2 diagnostics, stage 3, and the audit
- `bfad375` 2026-09-09 07:10:06 +0500 Add gen14: the separation curve as one bit and one scalar
- `5885607` 2026-09-09 16:42:59 +0500 Add gen15: the honest floor, the curvature, and the measured mode
- `dbc84e4` 2026-09-09 18:16:11 +0500 gen15 section 7: the measurement cannot pick the curve prototype, and why
- `47499b7` 2026-09-09 20:55:01 +0500 gen15 section 8, and a correction: an in-sample oracle is not a ceiling
- `df63929` 2026-09-09 23:50:39 +0500 Add gen16 working brief: six leads with a pre-registration and confirmation protocol

### Mismatches against every recorded version

| source | package | recorded | installed |
|---|---|---|---|
| docs/results/gen8_architecture_results_20260820.md:834 (2026-08-20) | numpy | 2.5.1 | 2.5.3 |
| docs/results/gen8_architecture_results_20260820.md:834 (2026-08-20) | scipy | 1.18.0 | 1.18.1 |
| docs/results/gen8_architecture_results_20260820.md:834 (2026-08-20) | torch | 2.13.0 | 2.14.0+cpu |
| docs/results/gen8_architecture_results_20260820.md:834 (2026-08-20) | rdkit | 2026.03.5 | 2026.03.6 |
| docs/results/gen7_architecture_results_20260819.md:1165 (2026-08-19) | python | 3.13 | 3.14.5 |
| docs/results/gen7_architecture_results_20260819.md:1165 (2026-08-19) | numpy | 2.5.1 | 2.5.3 |
| docs/results/gen7_architecture_results_20260819.md:1165 (2026-08-19) | torch | 2.13.0 | 2.14.0+cpu |
| docs/results/gen7_architecture_results_20260819.md:1165 (2026-08-19) | rdkit | 2026.03.5 | 2026.03.6 |
| requirements-gen3.txt (Linux cluster reference, not this machine) | python | 3.11.11 | 3.14.5 |
| requirements-gen3.txt (Linux cluster reference, not this machine) | numpy | 2.4.6 | 2.5.3 |
| requirements-gen3.txt (Linux cluster reference, not this machine) | pyarrow | 25.0.0 | 25.0.1 |
| requirements-gen3.txt (Linux cluster reference, not this machine) | scipy | 1.17.1 | 1.18.1 |
| requirements-gen3.txt (Linux cluster reference, not this machine) | joblib | 1.5.3 | 1.6.0 |
| requirements-gen3.txt (Linux cluster reference, not this machine) | torch | 2.13.0 | 2.14.0+cpu |

Reading: gen13, gen14 and gen15 recorded no version table of their own.  The only record from the gen13 session is the memory note, which matches the installed stack exactly, and every numeric-stack dist-info was written 2026-09-07 22:47-22:53, before gen13's first manifest (2026-09-07 23:36) and all six generation commits.  The gen7/gen8 docs (Aug 2026) predate that venv rebuild, so numpy 2.5.1 -> 2.5.3, scipy 1.18.0 -> 1.18.1, rdkit 2026.03.5 -> 2026.03.6, torch 2.13.0 -> 2.14.0+cpu, Python 3.13 -> 3.14.5 are differences between gen8 and gen13, not drift since gen13.  requirements-gen3.txt is the Linux cluster reference and never described this machine.  Nothing in the numeric stack has been touched since gen13 started.

### TabPFN

- `import tabpfn`: importable = **False**; ImportError: cannot import name '_is_pandas_df' from 'sklearn.utils.validation' (D:\ml_separator_gh\.venv\Lib\site-packages\sklearn\utils\validation.py)
- declared pins: ['torch<3,>=2.1', 'scikit-learn<1.7,>=1.2.0', 'typing_extensions>=4.12.0', 'scipy<2,>=1.11.1', 'pandas<3,>=1.4.0', 'einops<0.9,>=0.2.0', 'huggingface-hub<1,>=0.0.1', 'pydantic>=2.8.0', 'pydantic-settings>=2.10.1', 'eval-type-backport>=0.2.2', 'joblib>=1.2.0', 'tabpfn-common-utils[telemetry-interactive]>=0.1.8']
- scikit-learn stayed 1.9.0 (dist-info written 2026-09-07 22:48, tabpfn written 2026-09-09 15:18, between the gen14 and gen15 commits): the pins were never applied, so the 0.0014 shift START_HERE warns about did not happen here.  gen15 ran TabPFN through `exp/tabpfn/tabpfn_compat.py`, a private-name shim, on this same sklearn 1.9.0.
- frozen import path (every gen13sep module, gen14.dirbench/models, gen15.*): tabpfn in sys.modules = **False**; imported 22 modules, failed {}; third-party loaded: ['scipy', 'sklearn']; sklearn after imports 1.9.0.
- `gen15_curve/exp/decision/dec_arms.py` carries an `n_jobs=2` g13_full: True.

## 2. hash() and iteration-order audit

Scanned: gen13_separation\gen13sep, gen14_direction\gen14, gen15_curve\gen15, src\lanthanide_separation\gen6\cohorts.py, src\lanthanide_separation\gen8\inference.py (wildcluster.py excluded: untracked, not frozen).  Unreviewed hits: 0.

| file:line | class | snippet | why |
|---|---|---|---|
| `gen13_separation/gen13sep/arms_stage2.py:330` | **UNSAFE-IN-PRINCIPLE** | `return np.array([hash(row.tobytes()) for row in fp])` | hash() of bytes is salted per process.  extractant_key() uses it only as an equality key. ExtractantBalancedArm consumes it through groupby transforms (per-row weights, order-free): safe. HierarchicalCurveArm (S3_HIER / S3_EXT_LEVEL) groups by the key, so the row order of its per-extractant matrix X_ext follows the salted hash order and the ExtraTrees fit is order-invariant only up to floating-point summation order.  Not on the gen14/gen15 bench path (gen15.arms.g13_full calls tree_pipeline directly).  Do not rely on cross-process bit-identity of those two gen13 arms. |
| `gen13_separation/gen13sep/fewshot.py:35` | **SAFE** | `def stable_hash(text: str) -> int:` | blake2b of the text, not hash(). |
| `gen13_separation/gen13sep/fewshot.py:40` | **SAFE** | `rng = np.random.default_rng(stable_hash(f"{cell_id}|{repeat}|{seed}"))` | blake2b of the text, not hash(). |
| `gen13_separation/gen13sep/fewshot_stage2.py:98` | **SAFE** | `rng = np.random.default_rng(stable_hash(f"{cell_id}|{repeat}|{seed}"))` | blake2b via stable_hash(). |
| `gen15_curve/gen15/fewshot.py:336` | **SAFE** | `rng = np.random.default_rng(abs(hash((int(ci), int(seed)))) % (2 ** 32))` | tuple of ints: CPython salts str/bytes hashes only; int and tuple hashes are unsalted (verified empirically below under two PYTHONHASHSEED values).  Differs only across 32/64-bit builds. |

All other 59 matches are `set()`/`np.unique` uses that feed only membership tests, lengths, `sorted()` or set algebra; none decides a fold, a pair or a row order.  The fold plan itself is `seeded_group_kfold`: `np.unique` (sorted) + `default_rng(seed).permutation`, then publication masks by membership.  Full list in `env_check.json`.

### Two-process fold-plan comparison

`gen13sep.splits.all_folds(bench.frame, design=d)` for d in ['B', 'BR', 'BQ', 'A', 'BP'] with the default seeds [104729, 130363, 155921, 196613, 262147] (125 folds, 521 cells), SHA-256 over sorted test/train/inner indices and held-out groups, in two subprocesses with PYTHONHASHSEED=1 and =2:

- PYTHONHASHSEED=1: total `a7814c1a7bad389fbb87e1f08a6385cb4ad81b0e90fd930c22d1bc1f14da9aa2`; hash('salt-probe') = 3608921129880125405; hash((7, 104729)) = 3555083751542619629; gen15 fewshot rng seeds [2721180041, 866063850, 2848697259, 993581068, 2533411294, 2153262346]
- PYTHONHASHSEED=2: total `a7814c1a7bad389fbb87e1f08a6385cb4ad81b0e90fd930c22d1bc1f14da9aa2`; hash('salt-probe') = -6018158682598440628; hash((7, 104729)) = 3555083751542619629; gen15 fewshot rng seeds [2721180041, 866063850, 2848697259, 993581068, 2533411294, 2153262346]

Per design: {'B': '16e5f9c51835e1e0abf2d24d731c2821c9f7a1717fffeab4de94b48226aa4ee2', 'BR': 'a8627bdd9ac40107e99b6bc2da8f74ed024c1c70d2234e7a7717319f1b222372', 'BQ': '67e17b22fb53c87c72adbff9249b892e2ed30d94ca47fddc840261443e0a91e5', 'A': '68d91cf4046f2b349c29c37c8c40f802e1ed5a3fae5ec468c3e8dfa135933049', 'BP': '1c7b0ddf546cfae4dd2550d5673beb23124a2a4e9ce2c9bacf2b139bb8fd131a'}

**PASS: identical fold plans for all 5 designs x 5 seeds in two processes with different hash salts.**  str-hash salt differed: True; int-tuple hash identical: True.

## 3. xtb

- `which xtb` / `xtb.exe`: {'xtb': None, 'xtb.exe': None, 'xtb-python': None}  -> xtb_on_path = **False**
- PATH entries mentioning xtb: []; conda roots present: []
- disk search of ['C:\\Users\\Bandai', 'C:\\Program Files', 'C:\\Program Files (x86)', 'C:\\ProgramData', 'D:\\'] to depth 6 (11748 directories, 2.3 s, truncated=False): hits = []
- `import xtb` (xtb-python): {'xtb_python_importable': False, 'error': "ModuleNotFoundError: No module named 'xtb'"}

L1's reference-species energies therefore need the cluster route START_HERE plans for.

## 4. Memory and concurrency

- physical RAM 8072 MB; free before the measurements 3743 MB (load 53 %), after 3747 MB; other python.exe processes at the time: [{'pid': 16496, 'ws_mb': 3}, {'pid': 15384, 'ws_mb': 25}]; non-python baseline 4301 MB
- **cheap bench process: load cached bench + `V.score` FLAT/G14 under BP** (`--child bench`, rc 0): peak working set **229.7 MB** (polled pid 2356: 229.7 MB over 68 samples; self-reported 229.7 MB; system free dropped 216 MB from 3743 MB); after bench load {'ws_mb': 181.6, 'peak_ws_mb': 181.6, 'commit_mb': 857.2, 'peak_commit_mb': 857.2}; load 1.9 s, BP scoring 14.7 s, wall 17.0 s; BP macro MAE {'FLAT': 0.5885062528901843, 'G14': 0.5000794414203691}; contrast {'comparison': 'G14_vs_FLAT', 'point': 0.08842681146981521, 'ci95_low': 0.0010109793474018833, 'ci95_high': 0.1473098028341974, 'p_two_sided': 0.0468, 'passes_P1': 1.0}
- **heavy bench process: `dec_arms.g13_full` (400 extra trees, 209 columns, n_jobs=2) under BP** (`--child bench_et`, rc 0): peak working set **229.1 MB** (polled pid 15348: 229.0 MB over 133 samples; self-reported 229.1 MB; system free dropped 206 MB from 3747 MB); after bench load {'ws_mb': 181.9, 'peak_ws_mb': 181.9, 'commit_mb': 857.6, 'peak_commit_mb': 857.6}; load 2.0 s, BP scoring 30.4 s, wall 33.3 s; BP macro MAE {'G13_FULL': 0.5522751764799076}; contrast {}

Anchors from the cheap run (macro MAE of log SF, BP, 5 discovery seeds): {'FLAT': 0.5885062528901843, 'G14': 0.5000794414203691}; G14 - FLAT: {'comparison': 'G14_vs_FLAT', 'point': 0.08842681146981521, 'ci95_low': 0.0010109793474018833, 'ci95_high': 0.1473098028341974, 'p_two_sided': 0.0468, 'passes_P1': 1.0}.

**Recommendation: 6 concurrent bench processes.**  Budget now = free 3743 MB - 1024 MB kept free = 2719 MB; per-process peaks {'cheap_FLAT_G14': 229.7, 'extra_trees_G13_FULL_njobs2': 229.1}; RAM alone would allow 11 cheap / 11 extra-trees processes now (11 / 11 if no other python were running); the CPU cap at two threads per process on 12 logical CPUs is 6.  Keep n_jobs=2 per forest and set OMP_NUM_THREADS=2 per process so concurrent processes do not oversubscribe the 12 threads; the numbers above are per bench process, the orchestrator's own agents are not counted.

## 5. Git state

- HEAD `df63929` on `gen13-dev`; tracked under gen16_leads: ['gen16_leads/START_HERE.md'] (only START_HERE.md committed: True)
- modified tracked files (outside the brief, not ours): ['gen14_direction/results/g14_value_BP.csv', 'gen14_direction/results/g14_value_contrasts_BP.csv', 'gen14_direction/scripts/g14_value.py']
- untracked inside gen16_leads: ['gen16_leads/PRE_REGISTRATION.md', 'gen16_leads/gen16/', 'gen16_leads/results/', 'gen16_leads/scripts/', 'gen16_leads/tests/']
- untracked outside the brief: ['gen13_separation/gen13sep/wildcluster.py', 'gen13_separation/metrics/BP_all/inference_calibration.csv', 'gen13_separation/metrics/BP_all/per_extractant_wildcluster.csv', 'gen13_separation/scripts/g13_inference_audit.py', 'gen14_direction/results/g14_ave.csv', 'gen14_direction/results/g14_perm2.csv', 'gen14_direction/results/g14_straw.csv', 'gen14_direction/results/g14_straw_barcode.log', 'gen14_direction/results/g14_straw_mae.csv', 'gen14_direction/results/g14_straw_mae209.csv', 'gen14_direction/results/g14_straw_perm.log', 'gen14_direction/results/g14_value_per_extractant_BP.csv', 'gen14_direction/scripts/g14_capacity.py', 'gen14_direction/scripts/g14_perm2.py', 'gen14_direction/scripts/g14_straw.py', 'gen15_curve/scripts/g15_anchor.py', 'gen16_anchor/', 'gen16_protocol/', 'gen17_pairdiff/']
