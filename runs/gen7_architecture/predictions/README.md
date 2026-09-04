Every out-of-fold prediction file lives beside the suite that produced it, and this
directory indexes them rather than copying: the parquet files total roughly 90 MB and two
copies would drift. `INDEX.json` lists suite, relative path, model count, seeds and rows.

Schema of every `oof_predictions.parquet`: the cohort identity columns (`row_id`,
`extractant`, `ecfp_cluster`, `tanimoto_cluster`, `condition_id`, `series_id`,
`metal_symbol`, `metal_Z`, `n_replicates`, `log_D`), then `prediction`, `fold`,
`split_seed`, `nn_train_tanimoto`, `nn_base_tanimoto`, `model`, and any per-row
diagnostics the contender emitted under an `extra__` prefix (tree dispersion, the
predicted offset, GP standard deviation, PADRE anchor spread).
