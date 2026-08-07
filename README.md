# 3D-модели сепарации лантанидов

Репозиторий содержит два leakage-safe benchmark для прямого предсказания
сепарации соседних лантанидов:

1. tabular `Delta3D` на инвариантных статистиках координационной сферы;
2. основной исследовательский вариант — сиамская simplicial neural network
   (SNN) по готовому Vietoris–Rips asset.

Цель обеих моделей:

```text
log_SF(A/B) = log_D(A) - log_D(B)
```

`A` — более лёгкий элемент, `B` — следующий элемент по атомному номеру.
Строятся только настоящие adjacent-пары; `Nd-Sm` не является соседней парой,
поскольку между ними находится прометий.

## Почему simplicial-модель

Полный VR asset содержит 11 млн треугольников, большая часть которых описывает
внутреннее строение лиганда. Primary SNN строит из него target-independent
coordination subcomplex:

- одна вершина Ln и доноры, отмеченные генератором 3D-признаков;
- только атомы в пределах 3.10 Å от Ln;
- все VR-рёбра выбранных вершин;
- только 2-симплексы `Ln-D_i-D_j`;
- отсутствие independent edge/triangle truncation: превышение cap завершает run
  ошибкой, а не разрушает simplicial closure.

Node encoder использует атомный номер, флаг металла/донора и расстояние до Ln.
Partial charges по умолчанию выключены, чтобы доступность xTB-данных не стала
provenance shortcut. Edge messages получают filtration distance. Для каждого
2-симплекса дополнительно вычисляются три отсортированные длины сторон,
нормированная площадь, scale-free triangle quality и косинус угла
`D_i-Ln-D_j`. RBF support зафиксирован на 4.0 Å — cutoff, с которым построен
bundled VR asset; это даёт более высокое разрешение, чем шкала 6.5 Å. Все эти
признаки инвариантны к вращению, переносу и перестановке доноров.
Graph embedding объединяет состояние Ln, mean и dispersion донорной оболочки,
прямой mean по triangle states и mean по всем вершинам. Прямой triangle pooling
не заставляет many-body сигнал проходить только через node bottleneck.

Один encoder применяется к обоим комплексам пары:

```text
z_A = SNN(complex_A)
z_B = SNN(complex_B)
raw = [head(z_A-z_B, context) - head(z_B-z_A, context)] / 2
```

Это обеспечивает точную антисимметрию при перестановке металлов. Финальный
прогноз защищён inner-CV gate:

```text
prediction = prediction_tabular_Delta3D + w * (raw_SNN - prediction_tabular_Delta3D)
w in {0, 0.25, 0.50, 0.75, 1.0}
```

Ненулевой `w` сохраняется только если mean group-MAE reduction относительно
adaptive tabular Delta3D остаётся положительной после штрафа
`gate_z * standard_error`. При слабом сигнале `w=0`, и модель точно возвращается
к tabular Delta3D. Matched `2D + 2D` остаётся обязательным secondary control:
он отделяет пользу 3D от обычного усреднения моделей. Gate снижает риск negative
transfer, но не обещает улучшения на неизвестном outer fold — это может
установить только заранее заданный кластерный run.

## Leakage-защита

- exact match по `canonical_smiles` и всем `cond__*` условиям;
- `replicate-policy=unique` в primary analysis;
- обе геометрии приняты dual QC;
- exact-ECFP cluster held out целиком; поскольку каждый `canonical_smiles`
  обязан иметь ровно один ECFP, это является более строгим вариантом
  leave-extractants-out и одновременно запрещает ECFP-collision leakage;
- seeded shuffled `StratifiedGroupKFold`, stratification только по `pair_label`,
  не по target;
- outer и inner split seeds отделены от model seed;
- одинаковые inner folds для 2D, tabular Delta3D и SNN branches;
- scaling и imputation обучаются только на held-in группах; число neural epochs
  фиксируется заранее, а blend и gate выбираются только по inner OOF;
- overlap audit для extractant, ECFP cluster, source IDs, geometry keys,
  `geometry_feature_build_id` и `vr_graph_index`;
- pair-to-asset цепочка проверяется как
  `source_id -> row_geometry_map -> vr_graph_index -> VR build_ids`;
- ECFP до pairing обязан быть конечным бинарным блоком, а один SMILES не может
  иметь несколько fingerprints;
- точный `cohort_sha256`, dataset/asset/code SHA-256 и completion markers.

## Установка на кластере

PyTorch следует установить способом, соответствующим CUDA/CPU окружению вашего
кластера. Сам код не устанавливает зависимости на compute node.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[deep]'
```

## SLURM: сначала только план

Submission wrapper по умолчанию работает как dry-run и ничего не отправляет:

```bash
PYTHON_BIN=/path/to/env/bin/python \
DRY_RUN=1 \
bash slurm/submit_simplicial_multiseed.sh
```

Он покажет array из пяти заранее заданных пар `model_seed/split_seed` и
зависимый fail-closed aggregation job. В job не зашиты `partition`, `account`,
`qos` или GPU.

После проверки плана пользователь сам отправляет jobs:

```bash
PYTHON_BIN=/path/to/env/bin/python \
DRY_RUN=0 \
bash slurm/submit_simplicial_multiseed.sh
```

Эта форма использует default partition/account кластера. Если их обязательно
указывать, сначала посмотрите реальные имена через `sinfo -h -o '%P'`, затем
задайте именно их, например `PARTITION=cpu ACCOUNT=my_project`. Строки
`your_partition` и `your_account` являются placeholders и wrapper намеренно их
отклоняет до вызова `sbatch`.

Primary profile — deterministic CPU: 8 CPU, 32 GB, 24 часа, не более одной
одновременной array task. Параметры меняются через environment:

```bash
CPUS=16 MEMORY=48G WALLTIME=2-00:00:00 MAX_CONCURRENT=2 \
PYTHON_BIN=/path/to/env/bin/python DRY_RUN=0 \
bash slurm/submit_simplicial_multiseed.sh
```

CUDA — только явный opt-in. `index_add/scatter` на GPU может быть
недетерминированным, поэтому это sensitivity profile:

```bash
ACCELERATOR=cuda DETERMINISM=warn GPU_GRES=gpu:1 \
PYTHON_BIN=/path/to/env/bin/python DRY_RUN=0 \
bash slurm/submit_simplicial_multiseed.sh
```

Codex job не отправляет: submission выполняется только этой пользовательской
командой на кластере.

Незавершённые попытки сохраняются в `RUN_ROOT/attempts`, но aggregator читает
только атомарно опубликованные `RUN_ROOT/run_*` и сверяет их с неизменяемым
`RUN_ROOT/seed_plan.tsv`. Для безопасного повтора того же набора после preemption:

```bash
RUN_ROOT=/absolute/path/from/first/submission \
RESUME_RUN_ROOT=1 RESUME_COMPLETED=1 DRY_RUN=0 \
PYTHON_BIN=/path/to/env/bin/python \
bash slurm/submit_simplicial_multiseed.sh
```

Валидные seed-run проверяются по completion hash и пропускаются; только
неопубликованные seeds создают новые attempt-директории. `STAGE_VR_TO_TMP=1`
можно включить для локального scratch: сравнение runs использует SHA-256 asset,
а не временный physical path.

## Артефакты

Каждая array task создаёт отдельную директорию с:

- `summary.json` и `_SUCCESS.json`, записанными последними;
- SHA-256 каждого обязательного CSV/JSON artifact;
- exact OOF predictions и fold assignments;
- inner-CV tuning и neural training history;
- leakage audit, pair cohort и asset contract;
- per-extractant и per-pair-type metrics.

Aggregation запускается через `afterany`, поэтому даже при падении task создаёт
понятный validation report и отказывается усреднять неполный набор. При полном
успехе она выдаёт:

- распределение improvements по model/split seeds;
- `cross_seed_oof_predictions.csv`;
- отдельный paired group-bootstrap для усреднённого OOF ensemble.

Границы per-seed bootstrap CI никогда не усредняются.

## Tabular Delta3D

Tabular benchmark остаётся полезной абляцией:

```bash
.venv/bin/python scripts/run_delta3d_benchmark.py \
  --split-seed 104729 \
  --evaluation-only \
  --output-dir runs/tabular_example
```

Compact block дополнен zero-padding-safe radial quantiles и угловыми Legendre
moments. Модель использует exact antisymmetry, group-balanced training, matched
`2D + 2D` control, adaptive shrinkage и тот же seeded split protocol.

Научная мотивация и семь направлений развития находятся в
[`docs/3d_research_roadmap.md`](docs/3d_research_roadmap.md).

## Ограничение target provenance

В bundle нет общего `publication_id`/`experiment_series_id`. Поэтому точное
совпадение наблюдаемых условий ещё не доказывает, что два `log_D` измерены в
одной экспериментальной серии. Текущий `log_SF` остаётся condition-matched
proxy. Публикационный claim требует восстановления series provenance и
prospectively frozen проверки новых экстрагентов.
