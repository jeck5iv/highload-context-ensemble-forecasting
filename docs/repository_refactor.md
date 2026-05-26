# Repository refactor

The repository is organized as a study package with a visible entry point:

```text
src/metroflow/studies/metro_short_history_forecasting/entrypoint.py
```

The entry point exposes the configuration, runner, output writer, metrics, and bootstrap utilities. The implementation is split vertically:

```text
config.py        experiment configuration
pipeline.py      orchestration of data, models, evaluation, and outputs
data_layer/      station mapping, bucket aggregation, supervised frame, weather
models/          statistical, boosting, sequence, article-inspired, and meta models
evaluation/      metrics and station bootstrap
io.py            output writer
```

The refactor is deliberately conservative. The main computational logic is separated by role, but the numerical protocol, seed, output schema, and central regression fixture are preserved.
