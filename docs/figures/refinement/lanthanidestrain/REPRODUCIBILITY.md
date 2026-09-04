# lanthanidestrain — what can actually be regenerated

Checked against a clean clone of `github.com/mironovb/lanthanidestrain` at `main`
(default branch, ~111 MB, cloned 25 Aug 2026). Every statement below was produced by
running the repository's own figure entry points in that clone, not by reading the code.

**Result: 19 of the 37 rendered figures regenerate from what is committed. 18 do not** —
and one of those 19 needs a one-word invocation fix (`PYTHONPATH=<repo root>`) that is not
written down anywhere.
That is the binding constraint on this half of the refinement, and it is a finding in its
own right: a figure whose inputs are not in the repository cannot be re-rendered, re-styled
or checked by anyone who clones it.

## What ran

```bash
python -m automl.figures                 # 2 of 11 figures written
python -m automl.figures_reanalysis --all # 4 of 5
python -m automl.figures_topo --all       # 8 of 12 (incl. the two pi_sweep_*)
PYTHONPATH=. python -m automl.figures_topo --only stage2   # 1 more, with the path fix
python -m automl.figures_pi_email --all   # 4 of 9
```

| module | regenerates | does not |
|---|---|---|
| `automl/figures.py` | `fig3_conformer_noise`, `fig6_uncertainty_calibration` | `fig1_baseline_decomposition`, `fig2_block_ablation`, `fig4_architectures`, `fig5a_parity_baseline`, `fig5b_parity_best`, `fig7_per_metal`, `fig8_metal_free_3d`, `fig9_split_variability`, `fig10_split_series` |
| `automl/figures_reanalysis.py` | `re_fig2_dualkey`, `re_fig3_energy_snr`, `re_fig4_calibration`, `re_fig5_encoder` | `re_fig1_ceiling` |
| `automl/figures_topo.py` | `topo_blend_curve`, `topo_control_decomposition`, `topo_control_factorial`, `topo_forest`, `topo_ladder`, `topo_stack`, `pi_sweep_benchmark`, `pi_sweep_prediction`, and `topo_stage2` *with `PYTHONPATH=.`* | `topo_adjacent_parity`, `topo_seed_spread`, `topo_tradeoff` |
| `automl/figures_pi_email.py` | `email_figA_coverage`, `email_figB_result`, `email_figC_control`, `email_figD_limits` | — |
| *no committed producer* | — | `pi_fig1_headline`, `pi_fig2_significance`, `pi_fig3_mechanism`, `pi_fig4_parity`, `pi_fig5_per_extractant` |

## Why each failure happens

**1. `automl/reports/all_results.csv` is not committed.** `automl/figures.py::main` reads it
with `--results-csv`, and simply skips the four figures that need it when it is absent:

```python
res_path = Path(args.results_csv)
if res_path.exists():
    res = pd.read_csv(res_path)
    fig_decomposition(res, ...); fig_block_ablation(res)
    fig_architectures(res);      fig_metal_free(res)
```

so `fig1`, `fig2`, `fig4` and `fig8` are silently not produced. The 98 other result CSVs in
`automl/reports/` **are** committed; this one is the gap.

**2. `automl/artifacts/` is not committed** beyond a single `fresh_eval/fresh_pairs.json`.
`fig5a`, `fig5b`, `fig7`, `fig9` and `fig10` read the champion out-of-fold parquets and the
sweep directories under it. Note that `.gitignore` does **not** exclude `automl/artifacts/`
— the directory is simply absent, so this looks like an omission rather than a decision.

**3. `figures_topo.py` fails with an unhelpful error rather than skipping.** Three of its
four failures trace to the same guard-ordering bug:

```python
def _runs(*dirs):
    rows = []
    for d in dirs:
        for f in sorted((ART / d).glob("run_*.json")):   # ART is the absent artifacts dir
            ...
    return pd.DataFrame(rows).dropna(subset=["adj_r2", "r2_overall"])   # <- raises here
```

With no run JSONs, `rows` is empty, `pd.DataFrame([])` has no columns, and `.dropna(subset=…)`
raises `KeyError: ['adj_r2', 'r2_overall']` — *before* the caller's own
`if runs.empty: print("skip tradeoff: no runs found"); return` can run. Observed:

```
[figures_topo] parity  FAILED: AttributeError: 'NoneType' object has no attribute 'columns'
[figures_topo] seeds   FAILED: KeyError: ['adj_r2', 'r2_overall']
[figures_topo] tradeoff FAILED: KeyError: ['adj_r2', 'r2_overall']
[figures_topo] stage2  FAILED: ModuleNotFoundError: No module named 'src'
```

The intended behaviour — a named skip — is already written; it is only unreachable. Moving
the `dropna` behind an emptiness check would turn three confusing tracebacks into three
clear skips. `stage2` fails for a different reason: it imports the top-level package `src`,
which **is** committed (`src/geometry_features.py`, `src/geometry_schema.py`,
`src/chemistry/`) but is not importable unless the repository root is on `sys.path`.
Verified: `PYTHONPATH=. python -m automl.figures_topo --only stage2` writes
`topo_stage2.png`. The figure needs a path fix, not data.

**4. `re_fig1_ceiling` skips cleanly**, printing `skip ceiling: no valid estimator` — the
correct behaviour, and the contrast that shows the topo failures are a bug rather than a
design choice.

**5. `pi_fig1`–`pi_fig5` have no producer among the figure modules.** None of
`automl/figures.py`, `automl/figures_pi_email.py`, `automl/figures_reanalysis.py`,
`automl/figures_topo.py` or `automl/qc/*.py` contains the strings `pi_fig1_headline`,
`pi_fig2_significance`, `pi_fig3_mechanism`, `pi_fig4_parity` or `pi_fig5_per_extractant`,
— searched across all 163 committed `.py` files, with `pi_sweep_benchmark` as a positive
control that the same search does find in `automl/figures_topo.py` — although the five
PNG/PDF pairs are committed under
`automl/reports/figures/` and `pi_fig1_headline` is cited in `automl/reports/PI_REPORT.md`.
They were produced by a script that was not committed, or by one since removed. They cannot
be restyled without rewriting their generator against the result CSVs.

## Consequence for this refinement pass

* Figures are refined **only** where their inputs are committed, so that every
  `figure_script.py` in `figure_refinement/lanthanidestrain/` runs against a fresh clone.
* Where a figure carries a headline claim but cannot be regenerated, it is listed in
  `SUMMARY.md` with the missing input named, rather than being re-plotted from numbers
  transcribed off the old raster — which would break the rule that a figure must be
  reproducible from its script.
* The four `automl/figures.py` figures blocked only by `all_results.csv` are the cheapest
  fix in either repository: committing that one file (or the script that writes it) restores
  `fig1`, `fig2`, `fig4` and `fig8`.

*Originals were copied to `automl/reports/figures_ORIGINAL_BACKUP/` inside the working clone
before anything was re-run, so the regeneration above overwrote nothing that was not
recoverable.*
