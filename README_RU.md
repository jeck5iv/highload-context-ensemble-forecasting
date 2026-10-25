# Прогнозирование пассажиропотока метро

[English](README.md) | [Русский](README_RU.md)

**Статус результатов:** сохранённые метрики относятся к исходным экспериментам диплома. После удаления fallback-веток, использовавших целевые значения, нужен повторный запуск на закрытых данных. Публичные тесты и агрегированные fixtures не подтверждают прежние численные улучшения на исправленной версии. [Описание исследования](docs/research.md).

Репозиторий содержит воспроизводимый исследовательский контур для краткосрочного прогнозирования входящего пассажиропотока метро в многостанционной постановке и при короткой временной истории. В одном leakage-aware протоколе сравниваются сезонные и статистические модели, бустинг, компактные sequence-модели, модели сравнения по мотивам статей и остаточные метаансамбли.

Основная точка входа выделена отдельно:

```python
from metroflow.studies.metro_short_history_forecasting.entrypoint import (
    MultiStationForecastConfig,
    run_multistation_forecasting_experiment,
    save_multistation_forecasting_result,
)
```

## Структура

```text
configs/                         конфиги экспериментов и сеток
data/                            локальная папка данных, не версионируется
docs/                            контракт данных, воспроизводимость, описание рефакторинга
notebooks/                       короткая проверка репозитория и опциональный полный запуск
scripts/                         командные точки входа
src/metroflow/                   Python-пакет
  studies/metro_short_history_forecasting/
    entrypoint.py                публичный API эксперимента
    config.py                    конфигурация
    pipeline.py                  orchestration полного запуска
    data_layer/                  mapping станций, агрегация, supervised frame
    models/                      baseline, boosting, sequence, article-inspired, meta-модели
    evaluation/                  метрики и station bootstrap
    io.py                        сохранение результатов
tests/                           быстрые проверки и regression fixtures
```

`entrypoint.py` показывает основной вход в исследование, `pipeline.py` собирает стадии, а нижние модули отвечают за данные, модели, оценку и сохранение output.

## Установка

Для быстрых проверок:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

Для полного набора моделей:

```bash
python -m pip install -e '.[full]'
python -m pip install -r requirements-full.txt
```

## Данные

Сырые операционные данные в репозиторий не включены. Ожидаемая локальная структура описана в [`docs/data_contract.md`](docs/data_contract.md):

```text
data/
  PASS_ALL_*.csv
  REF_PSG_PLACES_*.csv
  cache/
```

Погодные признаки опциональны. Их можно передать через `MultiStationForecastConfig.weather_csv` или положить в `data/` файл вида `weather_*_hourly_openmeteo_*.csv`.

## Быстрые проверки

```bash
make test
```

Эта команда не запускает полный набор моделей. Она проверяет импорты, метрики, детерминированность station-bootstrap на toy-данных, структуру пакета и компактные regression fixtures.

Проверочный ноутбук:

```bash
jupyter nbconvert --execute notebooks/00_repository_check.ipynb --to notebook --inplace
```

По умолчанию полный эксперимент в ноутбуке выключен:

```python
RUN_FULL_EXPERIMENT = False
```

В этом режиме notebook делает только быструю проверку. Если поставить `True`, он запускает настоящий forecasting run и сохраняет результаты в `outputs/notebook_forecasting_run`.

## Центральный запуск

```bash
PYTHONPATH=src python scripts/run_metro_forecasting.py \
  --data-dir ./data \
  --cache-dir ./data/cache \
  --output-dir ./outputs/metro_short_history_h060_b015 \
  --n-bootstrap 1000
```

Базовая конфигурация:

```text
bucket_minutes = 15
horizon_minutes = 60
operational_day_start_hour = 3
random_state = 3228
```

Для быстрой честной проверки локальных данных удобнее взять более крупный bucket и временно отключить bootstrap.

## Grid-запуск

```bash
PYTHONPATH=src python scripts/run_horizon_bucket_grid.py \
  --profile robustness \
  --only 60:60 \
  --output-root outputs/verify_fast \
  --with-bootstrap \
  --n-bootstrap 20 \
  --rerun
```

Режим `60:60` подходит для проверки чтения данных, station mapping, признаков, обучения моделей, метрик и сохранения output без полного grid-прогона.

## Outputs

В каждой папке эксперимента сохраняются:

```text
micro_metrics.csv
macro_metrics.csv
station_bootstrap_delta_ci.csv
station_bootstrap_metric_ci.csv
test_predictions.csv
validation_predictions.csv
stations_used.csv
stage_timings.csv
```

Сборщик prediction artifacts читает уже сохранённые outputs и строит CSV, LaTeX-таблицы и рисунки без переобучения моделей:

```bash
PYTHONPATH=src python scripts/build_prediction_artifacts.py \
  --megagrid-root outputs/multistation_megagrid_overnight_MAIN_RESULT \
  --out-dir outputs/prediction_artifacts
```

## Regression comparison

```bash
PYTHONPATH=src python scripts/compare_regression_outputs.py \
  --old-dir tests/fixtures/h060m_b015m \
  --new-dir outputs/metro_short_history_h060_b015
```

Для проверки seed лучше сравнить два свежих запуска с одинаковой конфигурацией. Сначала можно отключить bootstrap, затем отдельно включить его с малым числом resamples.

## Воспроизводимость

Основной seed равен `3228`. Операционный день задан как `03:00-02:59`; train/validation/test split строится по нему. Station bootstrap выполняется по станциям после фиксации прогнозов.

Закрытые сырые данные не публикуются. Воспроизводимость обеспечивается кодом, конфигурациями, output contract, regression fixtures и сохранёнными артефактами запусков.
