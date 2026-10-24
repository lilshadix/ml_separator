# Resource plan (stage 0)

## Machine (recorded 2026-10-22)

- CPU: Intel i7-8750H, 6 cores / 12 threads.
- RAM: 7.9 GB. Keep about 1 GB free, so at most 3 parallel fold jobs.
- GPU: GTX 1050 Ti (sm_61). The installed torch 2.14 is a CPU build, so **all compute is on the CPU**.
- Disk: about 800 GB free on D:.
- Software: Python 3.14.5, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, sklearn 1.9.0 (sealed in the stage-0 manifest); catboost available.
- Thread caps: gen20k.bootstrap (OMP 4, LOKY 2, torch 4 per process).
- **No cluster, cloud or paid compute is authorised.** None will be used without the user's explicit permission.

## Stage 0 budget and measured costs

| Job | Measured / estimated wall time |
|---|---|
| Task identity (`r4_task_identity.py`) | 35 s (measured) |
| Tests (`tests/test_r4.py`, 13 tests) | 26 s (measured) |
| Reproduction of round 3's B0 / R14 / R02, per design | 5 / 6 / 6 min (measured, 3 in parallel) |
| REF (HBK + R14 + B1 + B2 fits, 25 folds), per design | about 10 min (estimate) |
| Fit poisoning `r4_poison.py REF`, 50 folds × 2 fits | about 30–40 min (estimate) |
| Non-training cache poisoning, 50 folds | about 45 s per fold; 3 shards in about 13 min (measured) |
| Seal / verify | about 1 min (digests 50 caches) |

- **Ceiling:** 3 wall-hours and about 10 CPU-hours.
- **If exceeded:** the job is stopped and the overrun is recorded in RUN_RECORD.
- **Checkpoints:** every outer fold is checkpointed in `predictions/_parts/<ID>_<design>_<seal>/`. A rerun resumes there, and the key changes whenever the stage-0 seal changes.

## Stage 1, block A (estimated before sealing; from the technical smoke and REF timings)

| Job | Estimate per design |
|---|---|
| S01 or S02 (HBK + B1 + B2 + one slot, 25 folds) | about 8–10 min |
| S03 | about 8–10 min (conditional) |
| S04 (S02 + 5 permuted draws) | about 20–30 min (conditional) |
| S21 (route + 5 shuffled draws) | about 20–30 min (conditional) |
| Fit poisoning per configuration, 50 folds × 2 | about 25 min (S01 / S02); about 60 min (S04 / S21) |
| Synthetic control study (4 worlds × 10 replicates) | about 10–15 min with 4 processes (measured: 65 s per replicate) |

- **Ceiling for block A:** 6 wall-hours and about 20 CPU-hours.
- **Power simulation:** 2 runs, about 3 min each (measured).

## Stage 1, block B (estimated; from the technical smoke)

| Job | Estimate per design |
|---|---|
| S05 or S06 (HBK + B1 + B2 + one CatBoost slot, 25 folds) | about 15–25 min (measured: 20–31 s per fold-fit, plus prediction and intervention) |
| S09 (S06 + 5 shuffled draws) | about 2.5 min per fold-fit, so about 1–1.5 h per design (conditional) |
| S22 | about 30–40 min per design (conditional) |
| Fit poisoning | S05 / S06: about 50–60 min; S09: about 4 h (100 fold-fits × 6 parts) |
| Synthetic block-B study | about 1 min per replicate per process |

- At most 3 parallel jobs (RAM).
- **Ceiling for block B:** 8 wall-hours. If S09 is triggered and its poisoning would exceed the ceiling, it runs on BPG (the deciding design) first, and the overrun is recorded.

## Stage 1, block C (estimated before sealing; synthetic runs 2-3 and the design diagnostics, single-thread torch)

One training (3 seeds x 400 epochs) takes about 4-10 s depending on load. A fold-fit trains:
- S11: 6 networks x about 4 inner blocks, plus a refit (about 25 trainings);
- each auxiliary arm: 2 positive weights x the blocks, plus a refit (about 9 trainings).

| Job | Estimate |
|---|---|
| Gate G-transfer (`r4_gate_c.py`: S11 stage + S13's two positive weights + B2, 50 fold-fits, inner only) | about 3-5 min per fold-fit; 3 shards, so about 1-1.5 h |
| S11 (HBK + B1 + B2 + S11, 25 folds per design) | about 3-4 min per fold-fit, so about 1.5 h per design |
| S13 (S11 + S13 + contrast) | about 4-5 min per fold-fit, so about 2 h per design |
| S14 or S15 (S11 + S13 + own control) | about 5-6 min per fold-fit, so about 2.5 h per design (conditional) |
| S16 (S11 + S13 + 5 permuted draws) | about 10-12 min per fold-fit, so about 4.5 h per design (conditional) |
| Fit poisoning | twice the fit cost per fold for 50 folds (CPU): S11 about 5 h, S13 about 7 h; S14 / S15 about 8-10 h each, S16 about 17-20 h (conditional). `r4_poison.py --shard i/3` splits the folds across 3 processes (wall time about 1/3), then `--combine 3` |
| Synthetic block-C study, run 3 (6 paired replicates x 3 worlds, 6 processes) | about 40-60 min per replicate |

- **Memory:** measured in synthetic run 3, each process has a working set of about 0.5 GB and about 1.1 GB private commit. Six processes left only about 0.9-1.8 GB free on the 7.9 GB machine (6 cores / 12 threads), which is at or below the 1 GB floor.
- **Parallelism cap for block C:** at most 5 synthetic processes. At most 3 r4_run / r4_poison / gate jobs, until one fold-fit of each has been measured.
- **Ceiling for block C:** 40 wall-hours.
- **If the controls are triggered** and their poisoning would exceed the ceiling, the controls run on BPG first, then on BP, and the overrun is recorded.

## Later stages

A per-slot cost estimate is written before stage 1 is sealed. The planning order, if the budget cannot cover every stage, follows the brief:
1. data provenance;
2. references;
3. residual / electronic arms and their controls;
4. then geometry / encoder;
5. then transfer.

Encoder training must fit in CPU-hours. If it does not, its slots are marked NOT RUN with the reason.
