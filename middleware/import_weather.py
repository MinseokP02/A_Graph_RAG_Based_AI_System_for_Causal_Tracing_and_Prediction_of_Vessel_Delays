"""Import collected Open-Meteo observations and departure links into Neo4j."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .neo4j_client import Neo4jClient

CONSTRAINT = (
    "CREATE CONSTRAINT portmis_weather_observation_unique IF NOT EXISTS "
    "FOR (w:WeatherObservation) REQUIRE w.observationKey IS UNIQUE"
)
OBSERVATIONS = """
UNWIND $rows AS row
MERGE (w:WeatherObservation {observationKey: row.observationKey})
SET w += row.properties,
    w.observedAt = datetime(row.observedAt)
WITH w, row
MATCH (p:Port {code: row.portCode})
MERGE (w)-[:OBSERVED_AT]->(p)
"""
DEPARTURE_LINKS = """
UNWIND $rows AS row
MATCH (pc:PortCall {recordId: row.recordId})
MATCH (w:WeatherObservation {observationKey: row.observationKey})
SET pc.callSign = row.callSign,
    pc.vesselName = row.vesselName,
    pc.departureWeatherTime = datetime(row.observedAt),
    pc.departureTemperatureC = row.temperatureC,
    pc.departureWindSpeedMs = row.windSpeedMs,
    pc.departureRainMm = row.rainMm
MERGE (pc)-[:HAS_DEPARTURE_WEATHER]->(w)
"""
DELAYS = """
UNWIND $rows AS row
MATCH (pc:PortCall {recordId: row.recordId})
SET pc.plannedDepartureTime = datetime(row.plannedDepartureTime),
    pc.departureDelayMinutes = row.departureDelayMinutes,
    pc.isDepartureDelayed = row.isDepartureDelayed,
    pc.delayPairingMethod = 'etryptYear+etryptCo+prtAgCd+clsgn'
"""


def _batches(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def observation_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            port_code = row["port_code"].strip()
            observed_at = row["weather_time_utc"].strip()
            yield {
                "observationKey": f"{port_code}@{observed_at}",
                "portCode": port_code,
                "observedAt": observed_at,
                "properties": {
                    "portCode": port_code,
                    "temperatureC": float(row["temperature_2m_c"]),
                    "windSpeedMs": float(row["wind_speed_10m_ms"]),
                    "rainMm": float(row["rain_mm"]),
                    "latitude": float(row["grid_latitude"]),
                    "longitude": float(row["grid_longitude"]),
                    "source": row["source"],
                    "datasetType": row["dataset_type"],
                    "model": row["model"],
                },
            }


def departure_link_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            observed_at = row["departure_weather_time_utc"].strip()
            if observed_at:
                # The collector explicitly maps authority 020 to Busan/KRPUS.
                port_code = (
                    "KRPUS"
                    if row["prtAgCd"].strip() == "020"
                    else row["prtAgCd"].strip()
                )
                yield {
                    "recordId": row["record_id"].strip(),
                    "observationKey": f"{port_code}@{observed_at}",
                    "callSign": row["clsgn"].strip(),
                    "vesselName": row["vsslNm"].strip(),
                    "observedAt": observed_at,
                    "temperatureC": float(row["departure_temperature_2m_c"]),
                    "windSpeedMs": float(row["departure_wind_speed_10m_ms"]),
                    "rainMm": float(row["departure_rain_mm"]),
                }


def delay_rows(path: Path, threshold_minutes: float):
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = tuple(
                row[name].strip()
                for name in ("etryptYear", "etryptCo", "prtAgCd", "clsgn")
            )
            groups[key].append(row)
    for rows in groups.values():
        departures = [r for r in rows if r["tkoffDt"].strip()]
        planned = {r["tkoffPrrrnDt"].strip() for r in rows if r["tkoffPrrrnDt"].strip()}
        if len(departures) != 1 or len(planned) != 1:
            continue
        actual = datetime.fromisoformat(departures[0]["tkoffDt"].strip())
        planned_value = datetime.fromisoformat(next(iter(planned)))
        if planned_value.tzinfo is None:
            planned_value = planned_value.replace(tzinfo=actual.tzinfo)
        delay = (actual - planned_value).total_seconds() / 60
        yield {
            "recordId": departures[0]["record_id"].strip(),
            "plannedDepartureTime": planned_value.isoformat(),
            "departureDelayMinutes": delay,
            "isDepartureDelayed": delay >= threshold_minutes,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weather", type=Path, default=Path("data/weather/weather_hourly.csv")
    )
    parser.add_argument(
        "--port-calls",
        type=Path,
        default=Path("data/weather/port_calls_with_weather.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    config = Config.from_env(args.port_calls, args.batch_size)
    counts = {"observations": 0, "departureLinks": 0, "pairedDelays": 0}
    with (
        Neo4jClient(config) as client,
        client.driver.session(database=config.database) as session,
    ):
        session.run(CONSTRAINT).consume()
        for batch in _batches(observation_rows(args.weather), args.batch_size):
            session.execute_write(
                lambda tx, values: tx.run(OBSERVATIONS, rows=values).consume(), batch
            )
            counts["observations"] += len(batch)
        for batch in _batches(departure_link_rows(args.port_calls), args.batch_size):
            session.execute_write(
                lambda tx, values: tx.run(DEPARTURE_LINKS, rows=values).consume(), batch
            )
            counts["departureLinks"] += len(batch)
        for batch in _batches(
            delay_rows(args.port_calls, config.delay_threshold_minutes), args.batch_size
        ):
            session.execute_write(
                lambda tx, values: tx.run(DELAYS, rows=values).consume(), batch
            )
            counts["pairedDelays"] += len(batch)
    print(counts)


if __name__ == "__main__":
    main()
