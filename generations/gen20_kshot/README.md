# gen20_kshot: leakage-safe k-shot learning for adjacent-lanthanide separation

**Read `IMPLEMENTATION_PLAN.md`, then `PRE_REGISTRATION.md`, then `GEN20_REPORT.md`.** The audit
(`AUDIT.md`) was written before the plan, the plan before the pre-registration, and the
pre-registration was sealed (SHA-256 `2915974a…`) before any outcome-bearing run.

**Status (2026-10-14/15):** discovery complete on all six designs (labels `discovery` = BP, B with
Setting B; `discovery_light` = BPG, BR, BQ, A; `discovery_nd_sm` = the gen13-adjacency pass);
confirmation run complete on the five withheld seeds; report written; every quoted number is in
`tables/report_numbers.csv` with its source file. Verdict in one line: measurements help, no model
beats a straight line through them by the registered margin, and the pre-registered neural arm is
significantly worse than that line.

Question: for a chemically unseen extractant (chemotype held out, its publications removed from
training), do k measured separation factors make its adjacent-lanthanide separation factors
predictable, and does any model beat a straight line through the same measurements?

## Layout

```
gen20k/      bootstrap, data (bench + manifest equality), splits (gen13 designs + BPG), design
             (fixed adjacent query set, no-closure support forest, label-free policies), cov
             (inner-OOF PSD covariance, shared-reading noise), adapters (BLUP, line, ridge line,
             rescale), arms (G14, FLAT, MEAN_CURVE, NN_TANIMOTO), nnc (Model 2: tabular contrast
             network + HEAD_FT/FULL_FT), metrics, inference (cluster bootstrap over pooled
             statistics, BCa, LOCO, wild-cluster-t, BH), setting_b (exploratory), manifest, runner,
             report
scripts/     g20_discovery.py, g20_report.py, g20_seeds.py, g20_confirm.py,
             g20_replicate_gen15.py, g20_reproduce.sh (the single command)
tests/       test_leakage.py (validity guarantees), test_leakage_injection.py (violations must raise)
manifests/   audit_reports.json, run manifests, confirmation-seed commitment
predictions/ long prediction tables (parquet, gitignored, digests in the manifests)
results/     boards, contrasts, per-pair tables, figures, the confirmation record
```

## Reproduce (from the repository root)

```bash
bash generations/gen20_kshot/scripts/g20_reproduce.sh          # anchors, tests, discovery, report, confirmation
SMOKE=1 bash generations/gen20_kshot/scripts/g20_reproduce.sh  # labelled smoke run only, never a result
.venv/bin/python -m pytest generations/gen20_kshot/tests -q -p no:cacheprovider
```

Inputs are read-only: the frozen bundle (SHA-256 `fefbefc6…`), gen13's cohort manifest
(`3f11ac4c…`), gen14's bench cache, gen19's publication copy groups. Nothing under gen13–gen19 is
modified; gen15's own code is imported for the replication of its measured mode.

## Conventions

* `y(A/B) = log10 D(A) − log10 D(B)` with A the lighter metal (gen13); heavier-first is a sign flip.
* Adjacent = strict |ΔZ| = 1 (12 pair types; promethium absent). gen13's "adjacent" also includes
  Nd–Sm and is reported as a sensitivity.
* Every number carries its design (BP selects), cohort (C3/C5/C10 or all), policy and regime
  (common query Q or remaining query R).
* A zero prediction scores 0.5 in direction accuracy; R² is never clipped and is NA with a reason
  when undefined.
