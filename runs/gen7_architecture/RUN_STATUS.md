# Run status — COMPLETE

*All queued work finished 2026-08-19 23:20 local (`GEN7_ALL_DONE` in `final3.log`).*

Every suite in the plan ran to completion on scikit-learn 1.9.0 / pandas 3.0.5 / numpy
2.5.1, stamped into each `summary.json` and collected into `collect_provenance.json`.
The final cross-suite table is `leaderboard_all.csv` — **116 arms**, all on byte-identical
test rows, cohort fingerprint `bed178ec1a7a82b0`.

| suite | seeds | arms |
|---|---|---|
| `oracles/` | 3 | 5 |
| `finalists/` | **5** | 15 |
| `ablations/` | 3 | 17 |
| `learners/` | 3 | 20 (indicators on — the corrected sweep) |
| `representations/` | 3 | 13 |
| `indicators/` `metal/` `ligphys/` `recovered/` `pairwise/` | 3 | 6 / 6 / 7 / 5 / 3 |
| `embeddings/` `kernels/` `hnn/` | 1 | 16 / 5 / 10 |
| `level_benchmark/` | 3 | 123 configurations × 3 targets |
| `ensemble/` `kshot/` `error_analysis/` `ceiling/` | — | derived from the finalists' OOF |

The top-level report `gen7_architecture_results_20260819.md` carries the final numbers
throughout; nothing in it is provisional.

## Verifying any of it

```bash
.venv/bin/python -m pytest tests/test_gen7_harness.py tests/test_gen7_reproduction.py -q
.venv/bin/python scripts/gen7_collect.py          # rebuilds leaderboard_all.csv
```

The reproduction test asserts `MC_lig2d_ext_massaction` returns macro 1.067993 /
offset 0.936996 / shape 0.502044 on split seed 104729. It caught two silent drifts during
this session — a model-seed derivation bug and a dependency downgrade — so run it before
adding any number to a shared table.
