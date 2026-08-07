# Дорожная карта инновационной 3D-модели

## Семь идей

| № | Направление | Новизна | Реализуемость сейчас | Leakage safety |
|---|---|---:|---:|---:|
| 1 | Counterfactual adjacent-Ln Delta3D: предсказывать `Delta logD` по изменению одной и той же координационной среды при замене Ln | 4.5/5 | 5/5 | 5/5 |
| 2 | Metal-centered many-body representation: radial Ln-donor и angular donor-Ln-donor terms, shape/strain invariants | 3.5/5 | 5/5 | 5/5 |
| 3 | Ensemble нескольких conformers, CN, nitrate denticity и stoichiometry с rank/quantile pooling | 4.5/5 | 3/5 | 5/5 |
| 4 | Preorganization/reorganization: free-ligand vs bound-ligand RMSD, strain energy и bite-angle change | 5/5 | 3.5/5 | 5/5 |
| 5 | Hydration/speciation mixture: target-free состояния с водой, nitrate и разным CN, объединённые thermodynamic pooling | 5/5 | 2/5 | 4/5 |
| 6 | Frozen OMol25/UMA metal-shell embeddings и force/relaxation diagnostics | 5/5 | 2.5/5 | 5/5 |
| 7 | Full-series multitask model: общий `logD` head плюс adjacent-difference loss | 4/5 | 4.5/5 | 5/5 |

## Выбранный вариант

Реализована идея 1 на двух уровнях представления из идеи 2:

1. compact tabular Delta3D как обязательная дешёвая абляция;
2. primary research branch — metal-centred 0/1/2-simplicial neural network.

SNN не обучается на полном молекулярном VR-комплексе из 11 млн треугольников.
Она выделяет замкнутый coordination subcomplex: Ln, отмеченные доноры в пределах
3.10 Å, все их VR-рёбра и только треугольники `Ln-D_i-D_j`. Это уменьшает
переобучение на ligand identity и напрямую кодирует many-body response первой
координационной сферы на замену соседнего металла.

Каждый 2-симплекс несёт явную инвариантную геометрию: три отсортированные
стороны, нормированную площадь, triangle quality и косинус угла
`D_i-Ln-D_j`. Таким образом, triangle message различает узкую и широкую
координационную геометрию, а не только факт существования симплекса. RBF-шкала
4.0 Å совпадает с зафиксированным cutoff исходного VR asset. Финальный graph
embedding сохраняет metal state, mean и dispersion доноров, direct triangle
pooling и all-node mean, поэтому many-body сигнал не обязан пережить только
triangle-to-node bottleneck.

Shared encoder строит `z_A` и `z_B`, после чего odd pair-head использует
`z_A-z_B`. Его прогноз антисимметризуется явно. Три независимые neural
initializations усредняются внутри каждого fold, а guarded blend поверх adaptive
tabular Delta3D включается только при положительном inner group-level signal.
Если доказательств недостаточно, вес SNN становится нулевым и прогноз точно
совпадает с tabular Delta3D. Matched `2D + 2D` сохраняется как secondary control.

Такой вариант совпадает с конечной целью сепарации, использует error
cancellation и лучше соответствует малому числу независимых экстрагентов, чем
обучение большого unrestricted molecular GNN с нуля.

Tabular block использует permutation-invariant статистики Ln-donor расстояний и
donor-Ln-donor углов, CN/cutoff response, shell gap/clearance, заряды и величину
диполя. Радиально ранжированные donor identities и отдельные indexed angles
оставлены только для sensitivity: на текущей когорте два таких редких признака
получали около 42% impurity importance и создавали shortcut через TODGA.

Для пары при одинаковых условиях:

```text
y(A,B) = log_D(A) - log_D(B)
x_3D   = phi_3D(A) - phi_3D(B)
```

Модель обучается также на переставленной паре `(B,A,-y)` и выдаёт
`(f(A,B)-f(B,A))/2`. Поэтому предсказание меняет знак при перестановке металлов,
а направление селективности не является случайным побочным свойством модели.
Отдельные baseline и full-Delta3D модели объединяются через shrinkage-вес,
выбранный только внутренним leave-extractants-out CV. Это защищает малый датасет
от ситуации, когда слабый 3D-блок ухудшает перенос на новый экстрагент.
Чтобы не принять обычное усреднение двух лесов за эффект 3D, параллельно строится
matched `2D + 2D` ensemble: второй 2D-лес получает тот же seed-поток, search
budget и правило выбора веса, что и full-Delta3D ветвь. Основной endpoint —
разность `guarded SNN hybrid - adaptive tabular Delta3D`; сравнение
`adaptive Delta3D - adaptive 2D ensemble` остаётся обязательной абляцией,
доказывающей пользу компактных 3D-признаков до neural branch.

Этот выбор согласуется с термодинамическим metal-exchange циклом: различия всего
в несколько kcal/mol определяют селективность, а вычитание похожих комплексов
частично сокращает систематическую ошибку. Configuration search при этом остаётся
критичным, поскольку неправильный минимум может иметь гораздо большую ошибку,
чем сама разность селективности ([JACS Au, 2025](https://pubs.acs.org/doi/10.1021/jacsau.4c00770)).

Metal-centered расстояния и CN физически мотивированы лантанидным сжатием:
структурный анализ CSD показывает уменьшение среднего CN примерно с 8.7 до 7.4
и первой координационной сферы примерно с 2.62 до 2.41 angstrom от La к Lu
([Scientific Reports, 2024](https://www.nature.com/articles/s41598-024-62074-3)).

## Следующие эксперименты в порядке ценности

1. Восстановить `publication_id` и настоящий `experiment_series_id`, затем
   перестроить condition-matched пары без межстатейного смешивания.
2. Добавить complex persistence-image Siamese CNN как отдельный image-native
   блок; ligand PI оставить только negative control.
3. Добавить free-ligand conformers и reorganization features. Экспериментальные
   тренды действительно меняются с preorganization и reorganization energy
   ([JACS, 2024](https://pubs.acs.org/doi/10.1021/jacs.4c07332)).
4. Генерировать несколько target-independent configurations на комплекс и
   агрегировать rank/quantile, а не доверять одной XYZ.
5. Сравнить с full-series multitask loss. Предыдущая ML-работа показала силу
   ECFP + molecular descriptors для `logD`, но не использовала этот paired 3D
   контраст ([JACS Au, 2022](https://pubs.acs.org/doi/10.1021/jacsau.2c00122)).
6. После появления настоящих series IDs проверить block-specific shrinkage или
   multi-kernel model, где inner CV может дать 3D-блоку нулевой вес.
7. Провести prospectively frozen тест: заранее зафиксировать неизвестные
   экстрагенты, затем измерить их adjacent separation factors в лаборатории.

## Критерий инновационного результата

Одного лучшего seed недостаточно. Минимальный пакет доказательств:

- одинаковые строки и folds для baseline и Delta3D;
- нулевое пересечение exact-ECFP кластеров во всех outer и inner folds;
- преимущество SNN-hybrid над adaptive tabular Delta3D на идентичных folds;
- преимущество tabular Delta3D над matched `2D + 2D`, а не только над baseline;
- paired cluster-bootstrap по экстрагентам;
- CI прироста `R2` выше нуля;
- улучшение macro-MAE и sign accuracy;
- только полные условия и `replicate-policy=unique` в основном анализе;
- повторение nested CV на заранее заданных master seeds/splits, поскольку
  bootstrap фиксированных OOF-прогнозов не учитывает training variability;
- отдельный matched `2D + 2D` comparator и нулевой вес SNN с точным fallback к
  tabular Delta3D при слабом inner-CV signal;
- identical pair-to-VR mapping и нулевой overlap `vr_graph_index` между train/test;
- cross-seed OOF ensemble строится только после выравнивания по `pair_id`, а его
  bootstrap пересчитывается с нуля;
- shuffled-complex-3D и ligand-only PI не повторяют эффект;
- финальный prospective experimental test.

Даже выполнение этих статистических пунктов на текущем bundle не превращает
proxy-target в доказанный experimental separation factor: сначала нужен
`publication_id`/`experiment_series_id`, чтобы исключить пары из разных работ.
