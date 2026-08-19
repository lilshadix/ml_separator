# 3D-модели сепарации лантанидов

Репозиторий содержит три leakage-safe benchmark для прямого предсказания
сепарации произвольных пар лантанидов:

1. tabular `Delta3D` на инвариантных статистиках координационной сферы;
2. основной исследовательский вариант — сиамская simplicial neural network
   (SNN) по готовому Vietoris–Rips asset;
3. pre-specified `A0`–`A6` ablation, которая отделяет CONDITIONS, LN, 2D,
   global 3D и local metal-centred 3D на одних и тех же frozen folds.

Общая целевая величина:

```text
log_SF(A/B) = log_D(A) - log_D(B)
```

`A` — более лёгкий элемент, `B` — любой более тяжёлый элемент, наблюдаемый для
того же экстрагента при точно совпадающих экспериментальных условиях. Primary
scope `all` строит все неупорядоченные комбинации без зеркальных дублей; поэтому
`Nd-Sm` является допустимой парой, даже если Pm отсутствует в данных. Scope
`adjacent` оставлен только для воспроизведения исторического nearest-neighbour
benchmark и включается явно через `--pair-scope adjacent` или
`PAIR_SCOPE=adjacent` в SLURM wrapper.

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
- fold audit сохраняет число представленных типов пар и отсутствующие типы для
  каждого train/test split;
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

## Полная A0–A6 абляция 2D против 3D

Новый auditable runner отвечает на основной вопрос `A2` против `A5`: добавляет
ли local coordination 3D переносимый сигнал поверх CONDITIONS + LN + полного
2D-блока. Он также выполняет `A2` против `A6`, `A3` против `A2`, несколько
training-only geometry shuffles и непустые exploratory `D1`–`D5` blocks.

```bash
.venv/bin/python scripts/run_ablation_benchmark.py \
  --pair-scope all \
  --group-mode extractant \
  --split-seed 104729 \
  --model-seed 42 \
  --output-dir runs/ablation_example
```

На текущем immutable parquet primary scope содержит 6 699 пар 91 типа:
1 081 соседнюю и 5 618 несоседних, с атомным span от 1 до 14. Эти строки
коррелированы: панель из `n` металлов даёт `n(n-1)/2` pair contrasts, но не
столько же независимых экспериментов. Поэтому целые extractant panels остаются
в одном fold, обучение выравнивает веса групп, а вывод опирается на
per-extractant metrics и paired group bootstrap, а не на трактовку всех 6 699
строк как независимых наблюдений.

`extractant` здесь означает canonical-SMILES identity и является primary
leave-extractants-out протоколом из preregistration. Совпадение exact-ECFP между
двумя разными SMILES записывается как sensitivity warning, но не подменяет
определение extractant. Более строгий secondary run включается через
`--group-mode ecfp-exact-cluster` и уже требует нулевого ECFP overlap.

Каждый run сохраняет machine-readable `feature_registry.json`, одну frozen
outer/inner fold plan для всех arms, fold-local preprocessing, long OOF,
paired deltas, per-extractant/per-Ln tables, geometry QC и shuffle provenance.
Dipole, partial charges и другие xTB/electronic quantities явно перечисляются
как excluded non-geometric source columns и не входят в A0–A6: основной
`A2`-vs-`A5` contrast изолирует координатную coordination geometry.
Предзаданные точные coordinate-only blocks — `global_shape` и
`coordination_shape` (D4 shape/distortion и D5 donor-hull sterics).
`ligand_field` и приближённый ray-based `enclosure` требуют явного opt-in.
Полный production benchmark этой командой не запускался; выполнен только
локальный bounded smoke для проверки реального all-pairs artifact contract.

## SLURM: сначала только план

Primary A0–A6 wrapper по умолчанию работает как dry-run и ничего не отправляет:

```bash
PYTHON_BIN=/path/to/env/bin/python \
DRY_RUN=1 \
bash slurm/submit_ablation_multiseed.sh
```

Для этого проекта wrapper уже использует проверенные ISAAC defaults:
`PARTITION=campus` и `ACCOUNT=acf-utk0011`. Их не нужно повторять в команде;
литеральные placeholders `actual_*` и `your_*` отклоняются до `sbatch`.

Он использует пять model seeds и один общий `SPLIT_SEED=104729`, поэтому
outer/inner folds физически совпадают между arms и seed-runs. Помимо
`seed_plan.tsv`, wrapper до первого `sbatch` замораживает
`experiment_contract.tsv`: scope/grouping, folds, trees/grid mode, bootstrap,
descriptor blocks/profile, shuffle seeds, feature policy, а также канонические
пути и SHA-256 dataset/VR asset. Resume или aggregate при любом drift
завершается fail-closed. Реальный запуск остаётся явным действием пользователя:
`DRY_RUN=0 bash slurm/submit_ablation_multiseed.sh`.

Для отдельного simplicial benchmark безопасный план печатается аналогично:

```bash
PYTHON_BIN=/path/to/env/bin/python \
DRY_RUN=1 \
bash slurm/submit_simplicial_multiseed.sh
```

`PAIR_SCOPE=all` используется по умолчанию и входит в immutable run protocol,
completion validation и aggregation checks; смешать all-pair и adjacent runs в
одном aggregate нельзя.

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

Начальный deterministic CPU profile: 8 CPU, 32 GB, 24 часа, не более одной
одновременной array task. All-pair cohort существенно больше исторической
adjacent-когорты, поэтому walltime нужно подтвердить коротким cluster smoke-run.
Параметры меняются через environment:

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

**Обновление 2026-08-19 (gen6).** Provenance восстановлена — но не из bundle.
`safe_exp_id` раскладывается как `{stem}_SAFE:{exp_id}` и соединяется 5 992/5 992
с upstream-таблицами `lanthanide_dataset_builder/raw_data/*_SAFE.csv`, где есть
`DOI`, `entry_author`, `addition_date`. Измерено (`scripts/reconstruct_provenance.py`):
**109 публикаций**; ячейка `экстрагент × условия × металл`, которую gen5 усредняет,
пересекает границу публикации только в **0.57 %** строк, а pair-ключ — в 1.49 %;
но CV-группа режима `unseen_series` — в **27.1 %** строк. Кроме того, 287 из 313
«реплицированных» ячеек различаются экспериментальной переменной, которой нет в
bundle, так что опубликованный noise floor — это в основном missing-feature error,
а не предел воспроизводимости.

## gen6 — генерация разнообразия

`gen6` проверяет не «какая модель», а «какой информации не хватает». Основная
гипотеза: ограничивающая переменная для zero-shot предсказания на новой химии —
**химическое разнообразие обучающей выборки**, а не ёмкость модели.

* протокол (pre-registration): [`docs/gen6_diversity_protocol_20260819.md`](docs/gen6_diversity_protocol_20260819.md)
* результаты Phase 0 и Phase 1: [`docs/gen6_phase0_and_phase1_results_20260819.md`](docs/gen6_phase0_and_phase1_results_20260819.md)
* код: `src/lanthanide_separation/gen6/` (chemistry / cohorts / metrics / manifest / provenance)

Эксперимент A сравнивает BASE91 (`min_cells >= 10`, текущее правило отбора) и
EXPANDED152 (`>= 3`) **на побайтово одинаковых тестовых строках**: одна общая
когорта, folds по Tanimoto-0.7 хемотипам, различается только маска обучающих
строк. Обязательные контроли — арм с совпадающим числом строк и арм с
перемешанной целевой переменной именно у добавленных строк.

```bash
.venv/bin/python scripts/gen6_phase0.py                      # семь проверок Phase 0
.venv/bin/python scripts/run_diversity_causal.py             # эксперимент A
.venv/bin/python scripts/run_diversity_learning_curve.py     # эксперимент B
```

Всё считается локально за минуты; `slurm/submit_gen6_diversity.sh` (DRY_RUN=1 по
умолчанию) нужен только для более крупных свипов.

Phase 2 (после исхода «case A»): эксперимент C — иерархическая декомпозиция уровня
(`scripts/run_hierarchical_levels.py`, модуль `gen6/hierarchical.py`) отвечает, *какая
компонента* не переносится на новый хемотип (уровень ячейки «лиганд × условия», а не
отклик по металлам); эксперимент F — ретроспективный выбор следующего лиганда
(`scripts/run_ligand_acquisition_sim.py`, модуль `gen6/acquisition.py`). Протокол:
[`docs/gen6_phase2_protocol_20260819.md`](docs/gen6_phase2_protocol_20260819.md); результаты:
[`docs/gen6_phase2_results_20260819.md`](docs/gen6_phase2_results_20260819.md).
