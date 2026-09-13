# gen15_curve: the honest floor, the curvature, and the measured mode

Gen15 (2026-09-09) asks three questions. First, what is the deployed gen14 zero-shot model worth
compared with predicting no separation at all? Second, where is the remaining error: in the
magnitude, the curvature coefficient, or 3D xTB energies? Third, how much do one to three measured
separation factors recover? Unless stated otherwise, every number is the extractant-macro MAE of
pairwise log SF on the frozen gen13 cohort (521 cells, 90 extractants), design BP, 5 split seeds.
Zero-shot, G14 scores 0.500 against 0.589 for FLAT, a gain of +0.088 (CI [+0.001, +0.147],
p = 0.047). MEAN_CURVE scores 0.622, worse than FLAT, so it is not a valid baseline. The curvature
holds error that nothing in the 2D structure predicts, and the xTB slope's Spearman of +0.644
tracks complex composition. One measured pair gives 0.231 on the section 4 pair set. Three
D-optimal pairs give 0.163-0.170 in all five designs on the section 5 common set. That gain comes
from the residual covariance, not from ligand chemistry. Zero-shot, the model calls the direction
of selectivity (strong-pair sign accuracy 0.812 vs 0.630), but its top-1 ligand pick is at chance.

## Status

- **Locked:** `GEN15_REPORT.md`, `gen15/`, `results/`, and the scripts and `exp/` arms the report
  cites. `../gen16_leads/START_HERE.md` calls gen13-gen15 "finished and locked". Commits: 5885607
  (sections 1-6), dbc84e4 (section 7), 47499b7 (section 8 and the section 1a correction).
- **Unreviewed:** `scripts/g15_anchor.py`, anchor regression with publication identity as the
  anchor, was committed without review in fdb1e15 (2026-09-10). The report does not mention it, and
  no `g15_anchor_*` output is committed.
- **Moved:** 155dc6c moved this directory from the repository root as pure renames. The commands
  below need the path edits that go with the move (`parents[3]`, `ROOT / "generations" / ...`).

## Read first

1. `GEN15_REPORT.md`: section 1, then **section 1a** before quoting any oracle, then sections 4-8.
2. [exp/decision/REPORT.md](exp/decision/REPORT.md): the direction call and the selection decisions.
3. [exp/embed/REPORT.md](exp/embed/REPORT.md), [exp/kernel/REPORT.md](exp/kernel/REPORT.md),
   [exp/tabpfn/tables.md](exp/tabpfn/tables.md): the section 8 null arms.
4. [../gen16_leads/DECISION_REPORT.md](../gen16_leads/DECISION_REPORT.md): later corrections, summarised below.

## Key files

- `gen15/valuebench.py`: the bench. It scores `arm(ctx) -> (n_test, 2)` under B/BR/BQ/A/BP.
  `RESULTS` points to `generations/gen15_curve/results`.
- Arms: `gen15/arms.py` (FLAT, MEAN_CURVE, G14, O_* oracles), `gen15/shape.py`, `gen15/fewshot.py`
  (BLUP, support selection, NOISE_VAR = 0.09), `gen15/mixture.py`.
- Scripts: `scripts/g15_locate.py`, `g15_shape.py`, `g15_fewshot.py`, `g15_support.py`,
  `g15_mixture.py`, `g15_fewshot_sweep.py`, `g15_uncertainty.py`. Deployed predictor:
  `scripts/g15_predict.py`.
- `exp/<arm>/`: the eight section 8 arms (embed, kernel, phys3d, tabpfn, external, condfe, labelerr,
  decision). Section 1a comes from `exp/labelerr/s4b_oracle_honesty.py`, section 3 from
  `exp/phys3d/trap_check.py`.
- `results/`: boards and contrasts. Some names match no current script output, for example
  `g15_support_board_empirical_BP.csv`, `g15_support_board_all.csv` and `g15_fewshot_board_widest.csv`.

## How to run (from the repository root; not run for this README)

Every command overwrites committed CSVs in `results/` or `exp/<arm>/`, so copy those first. On
Windows use `.venv/Scripts/python.exe`.

    .venv/bin/python generations/gen15_curve/scripts/g15_locate.py        # sections 1-2 (BP to restrict)
    .venv/bin/python generations/gen15_curve/scripts/g15_shape.py         # section 2
    .venv/bin/python generations/gen15_curve/scripts/g15_fewshot.py BP --how widest --maskpub
    .venv/bin/python generations/gen15_curve/scripts/g15_support.py       # section 5
    .venv/bin/python generations/gen15_curve/scripts/g15_mixture.py BP --how dopt
    .venv/bin/python generations/gen15_curve/scripts/g15_uncertainty.py BP --arm G14
    .venv/bin/python generations/gen15_curve/exp/labelerr/s4b_oracle_honesty.py
    .venv/bin/python generations/gen15_curve/scripts/g15_predict.py fit   # writes models/deploy_g15.joblib
    .venv/bin/python generations/gen15_curve/scripts/g15_predict.py predict \
        --smiles "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC" --measured "La/Lu=-1.31"

`generations/gen14_direction/cache/bench.pkl` is gitignored; the first bench load builds it if it is
missing. `exp/embed` downloads weights from huggingface.co. `exp/tabpfn` ran TabPFN through the
`exp/tabpfn/tabpfn_compat.py` shim on scikit-learn 1.9.0 (see `../gen16_leads/results/env/ENV.md`).

## Tests

None here. The nearest guard, not run for this README, pins the BP anchors (G14 0.5000794414203691,
FLAT 0.5885062528901843):

    PYTHONPATH=src .venv/bin/python -m pytest generations/gen16_leads/tests/test_anchors.py -q -m "not slow"

## Dependencies

- `generations/gen13_separation` (gen13sep: cohort, folds, metrics, bootstrap), `generations/gen14_direction`
  (gen14.dirbench, gen14.models), `generations/gen12_2_eu_pred` (gen122.coordination, used by `g15_predict.py`).
- `src/` (via gen13sep), `dataset with 3D structures/`, `runs/gen7_architecture/cache/chemistry_map.parquet`,
  `runs/gen6_provenance/provenance_table.parquet`. Imported by `generations/gen16_leads` and
  `generations/gen16_protocol` (unreviewed).

## Caveats and later corrections

- **Oracles are in-sample.** O_BOTH is 0.181 in sample and 0.274 with the scored pair left out
  (one pass, all cells; not the 5-seed bench). The "ceiling 0.182" in `g15_predict.py` and the
  O_BOTH "representation ceiling" in `exp/decision/REPORT.md` are in-sample values.
- **Three pair sets; do not mix them.** Section 4: G14@k1 0.231. Section 5, common set: D-optimal
  k=3 0.170. Section 7, training-fold covariance: POOLED@k3 0.192. Quote BP, never design B.
- **Section 2.** The in-sample curvature headroom is +0.073. Gen16 L2's leave-pair-out oracle gives
  +0.029 (p = 0.061); that fails P1 and closes the lead.
- **Section 3.** Rows 70/+0.195 and 71/+0.216 do not reproduce; `trap_check.csv` gives 79/+0.104
  and 81/+0.203. Gen16 L1's exact thermodynamic cycle gives -0.084 (n = 62), which closes the lead.
- **Sections 5-6.** The covariance is indefinite in 100 % of its 430 estimates, and nominal 90 %
  intervals cover 98.8 % at k=3. The intervals are too wide, not calibrated.
- **Section 7.** The 0.1865 in the table is MIX4meanPC; MIX6meanPC@k3 is 0.1852 in the CSV.
- **What still stands.** +0.088 over FLAT is still the honest zero-shot gain (gen16 added none),
  and `g15_predict.py` is still the deployed predictor.
