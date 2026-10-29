# Regional weather dataset

Run from the repository root with the middleware dependencies installed:

```powershell
.venv/Scripts/python.exe collect_weather.py --csv data/port_calls/processed/port_calls_020_20260801_20260831_I_selected.csv
```

No Open-Meteo API key is used. The collector queries the historical archive with
`temperature_2m` (Celsius), `wind_speed_10m` (m/s), and `rain` (mm accumulated
over the preceding hour). Rain excludes snow; it is not total precipitation.
Reference: https://open-meteo.com/en/docs/historical-weather-api

Outputs under `data/weather/`:

- `weather_hourly.csv`: regional hourly values, UTC timestamps, requested and
  returned grid coordinates, model selection, collection time and provenance.
- `regions.csv`: all route codes, source names, coordinates, source and resolution status.
- `port_calls_with_weather.csv`: original rows plus departure weather columns.
- `collection_summary.json`: coverage, missing values, failed calls and join statuses.
- `raw/`: original API responses and request parameters, cached for repeat runs.

The default period is the minimum through maximum arrival date in the source.
`--start YYYY-MM-DD --end YYYY-MM-DD` overrides it. The preceding UTC day is
also collected to cover local midnight in positive time zones. These are regional
period series, not proof a ship visited a route port at any particular time.

Coordinates come from `data/weather/unlocode_source.csv`, downloaded from
https://raw.githubusercontent.com/datasets/un-locode/main/data/code-list.csv .
Missing coordinates can be supplemented by exact code matches in
`data/weather/ports_source.json`, downloaded from
https://raw.githubusercontent.com/marchah/sea-ports/master/lib/ports.json .
The second source is community maintained. Coordinates represent locations,
not verified berth positions. Unresolved/unknown codes are reported, not guessed.

Departure joins currently recognize authority code `020` as Busan (`KRPUS`).
They use the UTC hour at or immediately before the actual `tkoffDt`, never an
arrival time, planned departure or destination ETA. No date means no join;
dates outside the collected period remain missing. Other authority codes need
an explicit mapping before departure enrichment. Null weather is never zero-filled.
The original source file is not modified.

The historical API provides reanalysis, not direct station observations. These
values must not be used as future-known predictors in a departure forecast.
The collector itself does not mutate Neo4j. Import the completed files with:

```powershell
.venv/Scripts/python.exe -m middleware.import_weather
```

This creates one `WeatherObservation` per port and UTC hour, links it to its
`Port` with `OBSERVED_AT`, and links matched calls with
`HAS_DEPARTURE_WEATHER`. The importer uses `MERGE`, so it can be rerun safely.
