# TFT departure delay forecasting

The `/ask` middleware recognizes every question containing `예측`, or accepts
`plan.intent="delay_forecast"`. The trigger overrides a lookup intent. An exact
forecast date is mandatory: provide one date in the plan or use `YYYY-MM-DD`,
`YYYY.MM.DD`, or `YYYY년 M월 D일` in the question. Missing or ranged dates return
`clarification_required` without calling TFT.

```text
POST /ask -> deterministic forecast routing -> TFT POST /predict
          -> Neo4j graph context + optional LightRAG documents
          -> Qwen narrative + deterministic TFT result table [F1]
```

The existing LightRAG web chat is not connected to `/ask`; use this API or a
client already integrated with it. No main LightRAG server changes are required.

## Start now, before training data exists

From the `LightRAG` directory, in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m middleware.tft_api
.\.venv\Scripts\python.exe -m middleware.query_api
```

Both services read `middleware/.env` (or `MIDDLEWARE_ENV_FILE`). The default
TFT address is `http://localhost:9623`; `/docs` exposes the prediction schema.
The base `middleware/requirements.txt` suffices for the unconfigured service.
Heavy ML dependencies are imported only when performing real inference.

Send to `http://localhost:9622/ask`:

```json
{
  "question": "호출부호 202809 선박의 2026-10-29 지연 예측을 알려줘.",
  "plan": {
    "intent": "delay_forecast",
    "call_sign": "202809",
    "start_date": "2026-10-29",
    "end_date": "2026-10-29"
  },
  "forecast_steps": 1,
  "delay_threshold_minutes": 60,
  "use_documents": true
}
```

`forecast_steps` is 1–24 **model time steps**, not necessarily hours. The default
is one step; free-text horizon lengths are not converted into this field.
Specify it explicitly. Optional plan dates constrain the returned forecast
timestamps and also the graph lookup; they do not shift a feature window.

The response includes `forecast` with one of these statuses:

| Status | Meaning |
|---|---|
| `ok` | Valid TFT quantile predictions with model version, series and cutoff |
| `model_not_ready` | No trained checkpoint configured or file missing |
| `insufficient_data` | Features missing, stale, ambiguous or incompatible with request |
| `unavailable` | Service, dependencies, checkpoint, inference or response validation failed |

Non-success states always have an empty `predictions` list and an explicit
message in the answer. No synthetic production predictions or random model
weights are substituted. The outer `status="ok"` means the query was handled;
check `forecast.status` for prediction success. Ordinary lookups retain their
existing response shape and never call TFT.

Successful predictions appear both in structured `forecast`, evidence `[F1]`,
and a deterministic table appended to the answer. P10/P50/P90 are forecast
quantiles in minutes, not observed delay or causal explanations. The threshold
probability is a linear interpolation of P10/P50/P90 capped to 10-90%, not an
empirical probability. The response also includes point-in-time AIS count,
speed, position, gross tonnage and cargo inputs when present. Graph or Qwen
narrative failure does not discard a TFT result; warnings record it.
An unplanned entity question still requires Qwen to extract its filters.

## Connect a trained model later

Install inference dependencies in the TFT service environment, matching the
versions used for training:

```powershell
python -m pip install -r middleware/requirements-tft.txt
```

Configure `middleware/.env` and restart the TFT service:

```dotenv
TFT_URL=http://localhost:9623
TFT_API_KEY=
TFT_TIMEOUT=30
TFT_CHECKPOINT=models/departure-delay.ckpt
TFT_FEATURES_PATH=data/tft/features.json
TFT_MODEL_VERSION=departure-delay-v1
TFT_MAX_FEATURE_AGE_SECONDS=3600
```

The adapter loads a trusted local PyTorch Forecasting
`TemporalFusionTransformer` checkpoint on CPU once per process and reuses its
training `dataset_parameters`, encoders and scalers. It requires a single target
in departure-delay minutes and `QuantileLoss` containing 0.1, 0.5 and 0.9.
Quantile columns are selected by the checkpoint's quantile order, following
the [library's QuantileLoss contract](https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/metrics/quantile.html).
Requests run in a thread pool with serialized model access. Use a separate
environment for the model service if ML dependencies conflict with LightRAG.

## Build AIS/Port-MIS features and train

`voyage_matching` must first produce strict `match_status=matched` rows. The
feature pipeline refuses to train on the current partial diagnostic file when it
contains no verified AIS-to-voyage matches.

Build one event row per completed departure. The default six-hour lead means
that AIS observations later than six hours before planned departure are not
used as training features. The label is
`max(0, actual departure - planned departure)` in minutes.

```powershell
.\.venv\Scripts\python.exe -m middleware.tft_features dataset `
  --portmis data\port_calls\port_calls_020_20260801_20260831_I.csv `
  --matches data\voyages\ais_portmis_matches.csv `
  --weather data\weather\weather_hourly.csv `
  --require-weather `
  --max-weather-age-hours 6 `
  --lead-hours 6 `
  --output data\tft\training.csv `
  --overwrite

.\.venv\Scripts\python.exe -m pip install -r middleware\requirements-tft.txt

.\.venv\Scripts\python.exe -m middleware.train_tft `
  --input data\tft\training.csv `
  --output models\departure-delay.ckpt `
  --encoder-length 48
```

For live inference, use a refreshed Port-MIS file containing a pending future
planned departure and AIS matches collected up to the timezone-aware cutoff.
The command includes only completed history before the cutoff and one pending
decoder row per window. It never uses a realized departure delay for that row.

```powershell
.\.venv\Scripts\python.exe -m middleware.tft_features bundle `
  --portmis data\port_calls\port_calls_current.csv `
  --matches data\voyages\ais_portmis_matches_current.csv `
  --weather data\weather\weather_hourly.csv `
  --require-weather `
  --max-weather-age-hours 6 `
  --as-of 2026-10-29T12:00:00+09:00 `
  --model-version departure-delay-v1 `
  --encoder-length 48 `
  --output data\tft\features.json `
  --overwrite
```

The training and bundle commands must use the same lead time, encoder length,
and weather feature schema. AIS is required by default. `--allow-missing-ais`
exists for data auditing, but the trainer still rejects a dataset containing no
AIS-backed rows. `--require-weather` excludes events that do not have an hourly
observation at or before the point-in-time cutoff. The weather inputs are
temperature, 10 m wind speed, hourly rain, observation availability, and age.
By default an observation older than six hours is treated as missing; use the
same `--max-weather-age-hours` value for training and live bundle generation.
Weather observations collected after the forecast cutoff are deliberately not
included because they would leak future information; a future weather forecast
may be added only with its issue time enforced.

The `.ckpt` file is the durable learned artifact. PyTorch Forecasting stores the
trained weights together with `dataset_parameters`, categorical encoders,
normalizers, and scalers in that checkpoint. `middleware.tft_api` loads it once
and creates each new input with `TimeSeriesDataSet.from_parameters(...)`, so a
new feature bundle reuses the learned preprocessing instead of fitting it
again. Keep `TFT_CHECKPOINT`, `TFT_MODEL_VERSION`, and the feature columns
unchanged when refreshing `TFT_FEATURES_PATH` with new input data. Restart the
TFT service only when replacing the checkpoint itself.

The inference input is a locally prepared JSON snapshot, reloaded per request:

```json
{
  "model_version": "departure-delay-v1",
  "windows": [{
    "series_id": "202809-at-020",
    "selectors": {"call_sign": "202809", "port_code": "020"},
    "as_of": "2026-09-27T10:00:00+09:00",
    "forecast_times": ["2026-09-27T11:00:00+09:00"],
    "rows": [
      {"time_idx": 0, "series": "202809-at-020", "delay_minutes": 10},
      {"time_idx": 1, "series": "202809-at-020", "delay_minutes": 12},
      {"time_idx": 2, "series": "202809-at-020", "delay_minutes": 0}
    ]
  }]
}
```

Column names, categorical types, group IDs,
encoder length and all known future covariates must match training. Each window
must contain exactly one series, enough historical rows and the full trained
decoder horizon. Integer time indices must be consecutive. The last horizon
rows correspond in order to `forecast_times`; preceding rows are history
available at `as_of`. Decoder target placeholders are overwritten with zero
before dataset creation and are not used as observed future targets.

The feature producer enforces point-in-time AIS availability and publishes CSV
and JSON files atomically. Do not put realized future variables into
known-future inputs. Old snapshots expire after the configured age.

Every requested entity filter must match exactly (case-insensitive); zero or
multiple matching windows return `insufficient_data`, rather than choosing an
arbitrary ship. A request may use the first N steps of the trained horizon.
Invalid, nonfinite or crossed quantiles are rejected. The current adapter does
not compute variable importance or evaluate forecast accuracy.
