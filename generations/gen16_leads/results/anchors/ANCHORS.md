# Gen16 anchors (phase 0)

Generated 2026-09-10T00:16:55.  Regime: 5 discovery split seeds (`gen13sep.splits.SPLIT_SEEDS`), extractant-macro metrics.  Produced by `gen16_leads/scripts/g16_anchors.py`; guarded by `gen16_leads/tests/test_anchors.py`.

| anchor | regime | expected | obtained | match |
|---|---|---|---|---|
| gen13 stage-3 headline `G13_ET_TOPO39` | macro direction accuracy, BP | `0.7683085207475452` | `0.7683085207475452` | exact |
| gen14 deployed model `G14` | macro MAE of log SF, BP | 0.5001 (`0.5000794414203691`) | `0.5000794414203691` | 4 dp and exact |
| `FLAT` (no separation) | macro MAE of log SF, BP | 0.589 (`0.5885062528901843`) | `0.5885062528901843` | 3 dp and exact |
| `G14_vs_FLAT` contrast | chemotype-blocked paired bootstrap, BP | +0.0884 [+0.0010, +0.1473] p 0.0468 | +0.0884 [+0.0010, +0.1473] p 0.0468 | 4 dp |

## G14 and FLAT under all five designs

Extractant-macro MAE of log SF, 5 discovery seeds.  GEN14_REPORT quotes G14 as 0.493 / 0.491 / 0.491 / 0.492 / 0.500.

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `G14` | 0.493208 | 0.490592 | 0.490543 | 0.492134 | 0.500079 |
| `FLAT` | 0.588506 | 0.588506 | 0.588506 | 0.588506 | 0.588506 |
| `G14_vs_FLAT` point | +0.0953 | +0.0979 | +0.0980 | +0.0964 | +0.0884 |
| `G14_vs_FLAT` 95 % CI | [+0.0046, +0.1581] | [+0.0067, +0.1607] | [+0.0081, +0.1601] | [+0.0032, +0.1624] | [+0.0010, +0.1473] |
| `G14_vs_FLAT` p | 0.0374 | 0.0318 | 0.0286 | 0.0408 | 0.0468 |
| seconds | 15 | 15 | 15 | 15 | 15 |
| G14 matches report at 3 dp | True | True | True | True | True |

## Fold-plan determinism

SHA-256 over the concatenated sorted test-index arrays of every fold, designs in order B, BR, BQ, A, BP: `7046c640676f4bf8b065393a1291464231340e7bccd33b31028010bc6d440d16`  
stricter train+test digest: `9a24b6e985c7515601e247ae04de5068f5c76974ce4992ce1bd2dcaad13c8054`  
folds per design: {'B': 25, 'BR': 25, 'BQ': 25, 'A': 25, 'BP': 25}  
2 separate subprocess runs identical to the in-session digest: **True**

## Environment

```
{
  "python": "3.14.5",
  "numpy": "2.5.3",
  "pandas": "3.0.5",
  "scipy": "1.18.1",
  "sklearn": "1.9.0",
  "joblib": "1.6.0",
  "rdkit": "2026.03.6",
  "LOKY_MAX_CPU_COUNT": "2",
  "joblib_effective_n_jobs_minus1": 2
}
```

Timings (s): {"fold_hash_subprocesses": 5.1, "bench_load": 0.0, "fold_hash_in_session": 0.2, "value_B": 14.7, "value_BR": 14.6, "value_BQ": 15.4, "value_A": 15.0, "value_BP": 15.0, "G13_ET_TOPO39_BP": 15.8}

**all_reproduced = True**
