# metro_forecasting_research

Проект для экспериментов по прогнозированию пассажиропотока метро.

## Структура

```text
main.ipynb          основной ноутбук для запуска экспериментов
configs/            конфигурации экспериментов
data/               локальные данные и кэш (в репозиторий не включаются)
outputs/            артефакты запусков
scripts/            консольные точки входа
src/metroflow/      код проекта
```

## Быстрый старт

Открыть `main.ipynb`, выбрать конфигурацию и выполнить ячейки по порядку.

Минимальный пример:

```python
from metroflow.config import load_experiment_config
from metroflow.research import run_and_materialize

cfg = load_experiment_config("configs/sportivnaya_15m_60m.toml")
result = run_and_materialize(cfg)
result["summary_table"]
```

## Конфигурации

Основные параметры эксперимента задаются в TOML-файлах в папке `configs/`:

- `bucket_minutes`
- `horizon_minutes`
- `place_ids`
- набор контекстных признаков
- параметры сохранения артефактов

## Данные

Папка `data/` исключена из Git. Исходные данные и кэш хранятся локально.

## Результаты запусков

По умолчанию сохраняются:

- таблица итоговых метрик
- прогнозы на валидации и тесте
- markdown-отчет
- графики по основным метрикам

## Запуск из командной строки

```bash
python scripts/run_single.py configs/sportivnaya_15m_60m.toml
```
