# gen21 — adjacent-lanthanide log SF on held-out experimental cells

**Status (2026-10-18): round 1 done (three ideas, dev only, final check not run).** Candidate
HB2+GP: R² 0.403 (BP) / 0.407 (BPG) at k = 3 on C3; see `GEN21_REPORT.md` §8 for what that does
and does not show.

Task: `CLAUDE_TASK.md` (Russian brief). Read `PROTOCOL.md` (written before any new method was
scored), then `ATTEMPTS.md` (every attempt: hypothesis, information contract, table, short
report), then `GEN21_REPORT.md` (the full report of this round).

Target: `logSF(A/B) = log10 D(A) − log10 D(B)`, A the lighter lanthanide, strict-adjacent pairs
only (|ΔZ| = 1; Nd/Sm is not adjacent because Pm is absent).

## Layout

```
g21/          paths, cache (per-fold context rebuilt with gen20's frozen functions), evaluate
              (measurement contract, long tables, metrics, isolation/intervention checks), methods
              (attempt 1 OBS, attempt 2 HB, attempt 3 HBG / pooled ridge), policies (SERIES_STAGGER),
              attempts (registry: hypothesis, contract, methods, baselines, contrasts)
scripts/      g21_build_cache.py, g21_reproduce.py, g21_hashes.py, g21_diag0.py (attempt-0 diagnosis),
              g21_attempt.py (one attempt, both designs, all checks), g21_diag.py (error analyses behind
              attempts 2-3 and the attempt-3 smoke look -> results/diag/), g21_show.py (print an attempt's tables)
tests/        test_g21.py (equivalence with gen20, masked reads, intervention, fold isolation,
              training-only fits for every fit route, label-free policy, SYNTHETIC recovery/pooling checks)
manifests/    inputs_sha256.json (cohort, log D matrix, fold plans, cache), final_seeds_commitment.txt
predictions/  attempt<N>_<design>.parquet: one row per (seed, fold, cell, k, query pair), one column per method
results/      reproduction_gen20_*.csv, attempt<N>/ (levels, contrasts, per-seed contrasts, per pair type,
              log D of unread metals, fitted parameters per fold, summary.json with the leakage checks)
cache/        derived, gitignored; rebuilt by g21_build_cache.py in ~2.5 min (digests in manifests/)
```

## Reproduce (from the repository root, Windows venv)

```bash
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_build_cache.py --designs BP,BPG
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_reproduce.py
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_attempt.py --attempt 1
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_attempt.py --attempt 2
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_attempt.py --attempt 3
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_attempt.py --attempt 3d
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_diag.py
.venv/Scripts/python.exe -m pytest generations/gen21/tests -q -p no:cacheprovider -W ignore
```

Inputs are read-only: gen20's package (`generations/gen20_kshot/gen20k`) and through it the frozen
gen13–gen15 code and the gen13 cohort manifest. Nothing outside `generations/gen21/` is modified.
The final-check seeds are stored outside the repository; only their SHA-256 is committed.
