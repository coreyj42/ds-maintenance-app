# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`ds-maintenance-app` produces forecasts of upcoming vehicle/oil maintenance events from TCU telemetry. Pipelines read features from BigQuery, preprocess time-series measurements, train XGBoost forecasters via MLForecast, then write predictions back to BigQuery. The pipelines are scheduled by Airflow in the `roadsurfer-com/data-workflows` repo — this package is consumed there as an installed dependency.

## Common commands

Poetry is required. Dependencies include private packages (`ds-package-forecast`, `ds-package-anomaly-detection`) hosted on the GCP Artifact Registry at `europe-west3-python.pkg.dev/sf-da-dwh/ds-packages` — you need the `keyrings.google-artifactregistry-auth` Poetry plugin and GCP credentials to install:

```
poetry self add keyring keyrings.google-artifactregistry-auth
poetry install
```

Make targets wrap the standard workflow:

- `make install` — `poetry install`
- `make format` — `black ds_maintenance_app tests`
- `make lint` — `flake8 ds_maintenance_app tests`
- `make test` — `pytest tests`
- `make all` — install + format + lint + test (this is what CI runs)
- `make build` — `poetry build` (artifact uploaded to GCP Artifact Registry by CD)

Single test: `poetry run pytest tests/test_components/test_preprocess.py::test_name -v`

Run the pipeline locally: `poetry run python ds_maintenance_app/run_maintenance_forecast_pipeline.py --env dev`. Requires GCP credentials with access to BigQuery datasets `ds_features_maintenance` (read) and `ds_predictions_maintenance` (write), plus `ds_monitoring_maintenance` (write, Step 6) in project `sf-da-dwh`.

## Architecture

### Pipeline structure

`pipelines/base.py` defines `BasePipeline` — an ABC that loads [config/config.yaml](ds_maintenance_app/config/config.yaml) via `load_config` (uses `importlib.resources`, so config must be packaged with the wheel), sets up a logger, and stamps a `datetime_tag`. All pipelines subclass it and implement `run_pipeline`.

`pipelines/maintenance_forecast_pipeline.py` is the only concrete pipeline today. Its `__init__` flattens every config key into a `self.*` attribute, and `run_pipeline` is a long, explicitly numbered sequence (Step 2.1, 2.2, …) that orchestrates components. **The docstring on the class is the canonical step list — keep it in sync when adding/removing/renumbering steps, since the `self.logger.info` lines in `run_pipeline` mirror it exactly.**

### Component layers (under `ds_maintenance_app/components/`)

- `data.py` — BigQuery I/O wrappers (`load_bigquery_features`, `save_table_to_bigquery`, `append_table_to_bigquery`). Non-`main` envs prefix the table name with `{env}_` so dev runs don't clobber prod tables.
- `preprocess.py` — Cleaning filters on the TCU time-series (`nullify_zeros`, `nullify_before_long_gaps`, `hampel_filter_forecasting_columns`, `nullify_decreases`, `remove_resets_from_service_columns`, `interpolate_target_columns_when_booked`, …). The Hampel filter lives in `ds_package_anomaly_detection`.
- `forecast.py` — Builds MLForecast-compatible frames (renames `id_column→unique_id`, `date_column→ds`, target→`y`), applies first-order differencing, runs backtests, and trains/executes XGBoost forecasters via `ds_package_forecast.ml_models.MLModels`.
- `postprocess.py` — Inverts differencing and applies dynamic-feature overrides (e.g. zeroing predicted distance gained while `is_blocked=1`) to both backtest and forecast outputs.
- `evaluate.py` — Backtest metrics.
- `output.py` — Converts the raw odometer/oil-percent forecasts into business outputs: next oil-change date, future service-interval distances, last serviceable date, etc.
- `monitoring.py` — Model-monitoring outputs derived from the preprocessed series. `detect_resets_in_service_columns` records every service reset (an increase in a decaying counter); `detect_zero_crossings_in_service_columns` records every time a series crosses to `≤ 0`. Both sort each series chronologically, scan per `id_column`/column, and return long tables keyed by `id_column` / `service_type` / date. Consumed in Step 6. They run on the `gmx_service_oil_interval_percent_ground_truth_state` copy (saved at Step 2.8, before reset-removal in Step 2.13) so the resets are still present, plus the two service-distance columns. Step 6 relabels the `service_type` value from `..._ground_truth_state` back to `gmx_service_oil_interval_percent` before saving, so both monitoring tables report the real column name.

### Forecast targets

Two targets are forecast in parallel through the same machinery:
- `gmx_odometer` — total distance, monotonically increasing (postprocess `trend="up"`).
- `gmx_service_oil_interval_percent` — drops over time, jumps up on service (postprocess `trend="down"`).

Service-distance columns (`gmx_service_vehicle_interval_distance`, `gmx_service_oil_interval_distance`) are not forecast directly — they're derived in Step 5 from the odometer forecast plus the last-known service distance.

### Env handling

`--env` flag controls table naming and write mode:
- `env == "main"` — production. Writes use `append_table_to_bigquery` (stamps `cycle_datetime` and `forecast_pipeline_version` from `importlib.metadata`) and no prefix on table names.
- Any other env — writes use `save_table_to_bigquery` (truncate) and prefix the table name with `{env}_`.

When adding a new output table, check whether the prod path should append (history-tracking) or overwrite — `Step 3.12`, `Step 4.7`, and `Step 5.6` follow the append-in-main pattern; `Step 2.17/2.18`, `Step 5.13/5.14` always overwrite. `Step 6.3/6.4` also always overwrite (via `save_table_to_bigquery`) but write to a **separate dataset** `ds_monitoring_maintenance` (set by `data.monitoring_dataset`), not `ds_predictions_maintenance`.

## Style and CI

- Python 3.11.9 (Poetry constraint). CI runs on 3.12.
- Black formatting and flake8 (`max-line-length=100`, ignores listed in [.flake8](.flake8)) are enforced. **CI fails if `make all` produces any git diff**, so always run `make format` before pushing.
- Branching: PRs target `dev`; only `dev → main` merges are accepted by [.github/workflows/protect_main.yaml](.github/workflows/protect_main.yaml). Pushing to `main` triggers CD which builds the wheel, uploads to GCP Artifact Registry, and dispatches the `install-airflow-packages.yml` workflow in `roadsurfer-com/data-workflows`. **Bump the version in [pyproject.toml](pyproject.toml) before merging to main**, otherwise the registry upload will conflict with the existing version.

## Config-driven behavior

Pipeline parameters (column names, filter thresholds, XGBoost hyperparameters, forecast horizon, lags, etc.) live in [ds_maintenance_app/config/config.yaml](ds_maintenance_app/config/config.yaml). The pipeline `__init__` reads every key into a `self.*` attribute; if you add a config key, wire it through `__init__` rather than reading `self.config[...]` deep in `run_pipeline`. A separate [config_tests.yaml](ds_maintenance_app/config/config_tests.yaml) is used by unit tests.
