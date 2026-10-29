"""Build point-in-time TFT datasets from strict AIS/Port-MIS voyage matches.

Training rows represent consecutive planned departure events at a port.  AIS
features are cut off before each event, and the target is the non-negative
difference between actual and planned departure.  Inference bundles contain
completed history plus one pending departure with a zero target placeholder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .voyage_matching import KST, UTC, normalize_identifier, parse_timestamp

TRAINING_COLUMNS = (
    "time_idx",
    "port_code",
    "call_sign",
    "vessel_name",
    "voyage_id",
    "planned_departure_utc",
    "feature_cutoff_utc",
    "delay_minutes",
    "planned_hour_sin",
    "planned_hour_cos",
    "planned_dow_sin",
    "planned_dow_cos",
    "gross_tonnage",
    "tug_used",
    "pilotage_used",
    "cargo_ton",
    "has_ais",
    "ais_observation_count",
    "ais_latitude",
    "ais_longitude",
    "ais_sog",
    "ais_sog_mean",
    "ais_cog",
    "ais_heading",
    "hours_to_arrival",
    "has_weather",
    "weather_age_hours",
    "temperature_2m_c",
    "wind_speed_10m_ms",
    "rain_mm",
)

# Port-MIS authority codes and the UN/LOCODE identifiers emitted by
# collect_weather.py. Extend this explicit mapping when another authority is
# enabled; guessing from a port name could silently attach the wrong weather.
WEATHER_PORT_CODES = {
    "020": "KRPUS",  # Busan
    "030": "KRINC",  # Incheon
}

PORT_KEY_COLUMNS = ("prtAgCd", "clsgn", "etryptYear", "etryptCo")
PORT_REQUIRED = PORT_KEY_COLUMNS + (
    "vsslNm",
    "etryptDt",
    "tkoffDt",
    "tkoffPrrrnDt",
)


class TFTDataError(ValueError):
    """Raised when safe TFT inputs cannot be constructed."""


@dataclass(frozen=True)
class PortEvent:
    voyage_id: str
    port_code: str
    call_sign: str
    vessel_name: str
    arrival: datetime | None
    planned_departure: datetime
    actual_departure: datetime | None
    gross_tonnage: float
    tug_used: float
    pilotage_used: float
    cargo_ton: float


@dataclass(frozen=True)
class AISObservation:
    timestamp: datetime
    latitude: float | None
    longitude: float | None
    sog: float | None
    cog: float | None
    heading: float | None
    hours_to_arrival: float | None


@dataclass(frozen=True)
class WeatherObservation:
    timestamp: datetime
    temperature_2m_c: float | None
    wind_speed_10m_ms: float | None
    rain_mm: float | None


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header:
        raise TFTDataError(f"CSV has no header: {path}")
    if len(header) != len(set(header)):
        raise TFTDataError(f"CSV has duplicate columns: {path}")
    return header


def _first(rows: Iterable[dict[str, str]], column: str) -> str:
    return next(
        ((row.get(column) or "").strip() for row in rows if row.get(column)), ""
    )


def _one_timestamp(rows: Iterable[dict[str, str]], column: str) -> datetime | None:
    values = {
        parse_timestamp(row[column], naive_timezone=KST)
        for row in rows
        if (row.get(column) or "").strip()
    }
    if len(values) > 1:
        raise TFTDataError(f"Conflicting {column} values in one Port-MIS voyage")
    return next(iter(values), None)


def _number(value: str | None, default: float = 0.0) -> float:
    try:
        result = float((value or "").strip())
        return result if math.isfinite(result) else default
    except ValueError:
        return default


def _flag(value: str | None) -> float:
    return 1.0 if (value or "").strip().upper() == "Y" else 0.0


def load_port_events(path: Path) -> list[PortEvent]:
    """Pair Port-MIS rows and return visits having a planned departure."""

    header = _header(path)
    missing = set(PORT_REQUIRED) - set(header)
    if missing:
        raise TFTDataError(f"Port-MIS columns missing: {', '.join(sorted(missing))}")
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            key = tuple((row.get(column) or "").strip() for column in PORT_KEY_COLUMNS)
            if not all(key):
                raise TFTDataError(f"Missing Port-MIS voyage key at row {line}")
            groups[key].append(row)

    events = []
    for (port_code, call_sign, year, count), rows in groups.items():
        planned = _one_timestamp(rows, "tkoffPrrrnDt")
        if planned is None:
            continue
        arrival = _one_timestamp(rows, "etryptDt")
        actual = _one_timestamp(rows, "tkoffDt")
        cargo = sum(
            _number(_first(rows, column))
            for column in ("ldadngTon", "trnpdtTon", "landngFrghtTon", "ldFrghtTon")
            if column in header
        )
        events.append(
            PortEvent(
                voyage_id="-".join(
                    (port_code, year, normalize_identifier(call_sign), count)
                ),
                port_code=port_code,
                call_sign=call_sign or "UNKNOWN",
                vessel_name=_first(rows, "vsslNm") or "UNKNOWN",
                arrival=arrival,
                planned_departure=planned,
                actual_departure=actual,
                gross_tonnage=_number(_first(rows, "grtg")),
                tug_used=_flag(_first(rows, "tugYn")),
                pilotage_used=_flag(_first(rows, "piltgYn")),
                cargo_ton=cargo,
            )
        )
    return sorted(
        events,
        key=lambda event: (event.port_code, event.planned_departure, event.voyage_id),
    )


def _optional_number(row: dict[str, str], *columns: str) -> float | None:
    for column in columns:
        raw = (row.get(column) or "").strip()
        if raw:
            value = _number(raw, default=math.nan)
            return value if math.isfinite(value) else None
    return None


def load_matched_ais(
    path: Path, *, accepted_statuses: Sequence[str] = ("matched",)
) -> dict[str, list[AISObservation]]:
    """Load only strict matches emitted by :mod:`voyage_matching`."""

    header = _header(path)
    missing = {"voyage_id", "match_status"} - set(header)
    if missing:
        raise TFTDataError(f"Matched AIS columns missing: {', '.join(sorted(missing))}")
    result: dict[str, list[AISObservation]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("match_status") or "").strip() not in accepted_statuses:
                continue
            voyage_id = (row.get("voyage_id") or "").strip()
            raw_time = (row.get("time_utc") or row.get("received_at_utc") or "").strip()
            if not voyage_id or not raw_time:
                continue
            try:
                timestamp = parse_timestamp(raw_time, naive_timezone=UTC)
            except ValueError:
                continue
            result[voyage_id].append(
                AISObservation(
                    timestamp=timestamp,
                    latitude=_optional_number(row, "latitude", "metadata_latitude"),
                    longitude=_optional_number(row, "longitude", "metadata_longitude"),
                    sog=_optional_number(row, "sog", "SOG"),
                    cog=_optional_number(row, "cog", "COG"),
                    heading=_optional_number(row, "true_heading", "Heading"),
                    hours_to_arrival=_optional_number(row, "hours_to_arrival"),
                )
            )
    for observations in result.values():
        observations.sort(key=lambda item: item.timestamp)
    return result


def load_weather(path: Path) -> dict[str, list[WeatherObservation]]:
    """Load hourly weather keyed by Port-MIS authority code.

    The source uses UN/LOCODE identifiers. Only explicit authority mappings are
    accepted so that an unknown port fails as missing weather instead of being
    joined to a similarly named location.
    """

    header = _header(path)
    required = {
        "port_code",
        "weather_time_utc",
        "temperature_2m_c",
        "wind_speed_10m_ms",
        "rain_mm",
    }
    missing = required - set(header)
    if missing:
        raise TFTDataError(f"Weather columns missing: {', '.join(sorted(missing))}")
    authority_by_unlocode = {value: key for key, value in WEATHER_PORT_CODES.items()}
    result: dict[str, list[WeatherObservation]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            authority = authority_by_unlocode.get((row.get("port_code") or "").strip())
            raw_time = (row.get("weather_time_utc") or "").strip()
            if not authority or not raw_time:
                continue
            try:
                timestamp = parse_timestamp(raw_time, naive_timezone=UTC)
            except ValueError:
                continue
            result[authority].append(
                WeatherObservation(
                    timestamp=timestamp,
                    temperature_2m_c=_optional_number(row, "temperature_2m_c"),
                    wind_speed_10m_ms=_optional_number(row, "wind_speed_10m_ms"),
                    rain_mm=_optional_number(row, "rain_mm"),
                )
            )
    for observations in result.values():
        observations.sort(key=lambda item: item.timestamp)
    return result


def _last_number(observations: Sequence[AISObservation], field: str) -> float:
    for observation in reversed(observations):
        value = getattr(observation, field)
        if value is not None:
            return float(value)
    return 0.0


def feature_row(
    event: PortEvent,
    observations: Sequence[AISObservation],
    cutoff: datetime,
    *,
    time_idx: int,
    target: float | None,
    weather: Sequence[WeatherObservation] = (),
    max_weather_age_hours: float = 6.0,
) -> dict[str, object]:
    """Create one event row using only information available by ``cutoff``."""

    available = [item for item in observations if item.timestamp <= cutoff]
    sog_values = [item.sog for item in available if item.sog is not None]
    if max_weather_age_hours <= 0:
        raise TFTDataError("max_weather_age_hours must be positive")
    available_weather = [
        item
        for item in weather
        if item.timestamp <= cutoff
        and (cutoff - item.timestamp).total_seconds() / 3600 <= max_weather_age_hours
    ]
    latest_weather = available_weather[-1] if available_weather else None
    weather_age = (
        max(0.0, (cutoff - latest_weather.timestamp).total_seconds() / 3600)
        if latest_weather
        else 0.0
    )
    local = event.planned_departure.astimezone(KST)
    hour_angle = 2 * math.pi * (local.hour + local.minute / 60) / 24
    dow_angle = 2 * math.pi * local.weekday() / 7
    return {
        "time_idx": time_idx,
        "port_code": event.port_code,
        "call_sign": event.call_sign,
        "vessel_name": event.vessel_name,
        "voyage_id": event.voyage_id,
        "planned_departure_utc": event.planned_departure.isoformat(),
        "feature_cutoff_utc": cutoff.astimezone(UTC).isoformat(),
        "delay_minutes": "" if target is None else round(target, 6),
        "planned_hour_sin": math.sin(hour_angle),
        "planned_hour_cos": math.cos(hour_angle),
        "planned_dow_sin": math.sin(dow_angle),
        "planned_dow_cos": math.cos(dow_angle),
        "gross_tonnage": event.gross_tonnage,
        "tug_used": event.tug_used,
        "pilotage_used": event.pilotage_used,
        "cargo_ton": event.cargo_ton,
        "has_ais": float(bool(available)),
        "ais_observation_count": float(len(available)),
        "ais_latitude": _last_number(available, "latitude"),
        "ais_longitude": _last_number(available, "longitude"),
        "ais_sog": _last_number(available, "sog"),
        "ais_sog_mean": sum(sog_values) / len(sog_values) if sog_values else 0.0,
        "ais_cog": _last_number(available, "cog"),
        "ais_heading": _last_number(available, "heading"),
        "hours_to_arrival": _last_number(available, "hours_to_arrival"),
        "has_weather": float(latest_weather is not None),
        "weather_age_hours": weather_age,
        "temperature_2m_c": (
            latest_weather.temperature_2m_c
            if latest_weather and latest_weather.temperature_2m_c is not None
            else 0.0
        ),
        "wind_speed_10m_ms": (
            latest_weather.wind_speed_10m_ms
            if latest_weather and latest_weather.wind_speed_10m_ms is not None
            else 0.0
        ),
        "rain_mm": (
            latest_weather.rain_mm
            if latest_weather and latest_weather.rain_mm is not None
            else 0.0
        ),
    }


def build_training_rows(
    events: Sequence[PortEvent],
    ais: dict[str, list[AISObservation]],
    *,
    lead_hours: float = 6.0,
    require_ais: bool = True,
    weather: dict[str, list[WeatherObservation]] | None = None,
    require_weather: bool = False,
    max_weather_age_hours: float = 6.0,
) -> list[dict[str, object]]:
    """Build completed, labeled rows with a fixed pre-departure cutoff."""

    if lead_hours < 0:
        raise TFTDataError("lead_hours must be non-negative")
    if max_weather_age_hours <= 0:
        raise TFTDataError("max_weather_age_hours must be positive")
    grouped: dict[str, list[tuple[PortEvent, datetime, float]]] = defaultdict(list)
    for event in events:
        if event.actual_departure is None:
            continue
        cutoff = event.planned_departure - timedelta(hours=lead_hours)
        available = [
            item for item in ais.get(event.voyage_id, []) if item.timestamp <= cutoff
        ]
        if require_ais and not available:
            continue
        available_weather = [
            item
            for item in (weather or {}).get(event.port_code, [])
            if item.timestamp <= cutoff
            and (cutoff - item.timestamp).total_seconds() / 3600
            <= max_weather_age_hours
        ]
        if require_weather and not available_weather:
            continue
        delay = max(
            0.0,
            (event.actual_departure - event.planned_departure).total_seconds() / 60,
        )
        grouped[event.port_code].append((event, cutoff, delay))
    rows = []
    for values in grouped.values():
        values.sort(key=lambda value: (value[0].planned_departure, value[0].voyage_id))
        for index, (event, cutoff, delay) in enumerate(values):
            rows.append(
                feature_row(
                    event,
                    ais.get(event.voyage_id, []),
                    cutoff,
                    time_idx=index,
                    target=delay,
                    weather=(weather or {}).get(event.port_code, []),
                    max_weather_age_hours=max_weather_age_hours,
                )
            )
    return rows


def build_feature_bundle(
    events: Sequence[PortEvent],
    ais: dict[str, list[AISObservation]],
    *,
    as_of: datetime,
    model_version: str,
    encoder_length: int = 48,
    lead_hours: float = 6.0,
    require_ais: bool = True,
    weather: dict[str, list[WeatherObservation]] | None = None,
    require_weather: bool = False,
    max_weather_age_hours: float = 6.0,
) -> dict[str, object]:
    """Build live windows for pending departures without realized target leakage."""

    if as_of.tzinfo is None:
        raise TFTDataError("as_of must be timezone-aware")
    if encoder_length < 1:
        raise TFTDataError("encoder_length must be positive")
    if max_weather_age_hours <= 0:
        raise TFTDataError("max_weather_age_hours must be positive")
    completed = build_training_rows(
        events,
        ais,
        lead_hours=lead_hours,
        require_ais=require_ais,
        weather=weather,
        require_weather=require_weather,
        max_weather_age_hours=max_weather_age_hours,
    )
    history_by_port: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in completed:
        if datetime.fromisoformat(str(row["planned_departure_utc"])) < as_of:
            history_by_port[str(row["port_code"])].append(row)

    windows = []
    for event in events:
        if event.actual_departure is not None or event.planned_departure <= as_of:
            continue
        target_observations = [
            item for item in ais.get(event.voyage_id, []) if item.timestamp <= as_of
        ]
        if require_ais and not target_observations:
            continue
        target_weather = [
            item
            for item in (weather or {}).get(event.port_code, [])
            if item.timestamp <= as_of
            and (as_of - item.timestamp).total_seconds() / 3600 <= max_weather_age_hours
        ]
        if require_weather and not target_weather:
            continue
        history = history_by_port[event.port_code][-encoder_length:]
        if len(history) < encoder_length:
            continue
        rows = [dict(row) for row in history]
        for index, row in enumerate(rows):
            row["time_idx"] = index
        rows.append(
            feature_row(
                event,
                ais.get(event.voyage_id, []),
                as_of,
                time_idx=len(rows),
                target=0.0,
                weather=(weather or {}).get(event.port_code, []),
                max_weather_age_hours=max_weather_age_hours,
            )
        )
        windows.append(
            {
                "series_id": event.voyage_id,
                "selectors": {
                    "call_sign": event.call_sign,
                    "vessel_name": event.vessel_name,
                    "port_code": event.port_code,
                },
                "as_of": as_of.isoformat(),
                "forecast_times": [event.planned_departure.isoformat()],
                "rows": rows,
            }
        )
    return {"model_version": model_version, "windows": windows}


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise TFTDataError(f"Output exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            delete=False,
            dir=path.parent,
            suffix=".tmp",
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=TRAINING_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        if temporary and temporary.exists():
            temporary.unlink()
        raise


def _atomic_json(path: Path, value: dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise TFTDataError(f"Output exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary and temporary.exists():
            temporary.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("dataset", "bundle"):
        command = subparsers.add_parser(name)
        command.add_argument("--portmis", type=Path, required=True)
        command.add_argument("--matches", type=Path, required=True)
        command.add_argument("--lead-hours", type=float, default=6.0)
        command.add_argument("--allow-missing-ais", action="store_true")
        command.add_argument(
            "--allow-probabilistic-matches",
            action="store_true",
            help="Include explicitly reviewed match_status=probabilistic_matched rows",
        )
        command.add_argument(
            "--weather",
            type=Path,
            help="Hourly weather CSV produced by collect_weather.py",
        )
        command.add_argument(
            "--require-weather",
            action="store_true",
            help="Exclude rows without a weather observation at or before cutoff",
        )
        command.add_argument(
            "--max-weather-age-hours",
            type=float,
            default=6.0,
            help="Maximum age of the latest weather observation (default: 6)",
        )
        command.add_argument("--overwrite", action="store_true")
    dataset = subparsers.choices["dataset"]
    dataset.add_argument("--output", type=Path, default=Path("data/tft/training.csv"))
    bundle = subparsers.choices["bundle"]
    bundle.add_argument("--output", type=Path, default=Path("data/tft/features.json"))
    bundle.add_argument("--as-of", required=True, help="Timezone-aware ISO-8601 cutoff")
    bundle.add_argument("--model-version", default="departure-delay-v1")
    bundle.add_argument("--encoder-length", type=int, default=48)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        events = load_port_events(args.portmis)
        accepted_statuses = (
            ("matched", "probabilistic_matched")
            if args.allow_probabilistic_matches
            else ("matched",)
        )
        ais = load_matched_ais(args.matches, accepted_statuses=accepted_statuses)
        if args.require_weather and not args.weather:
            raise TFTDataError("--require-weather requires --weather")
        weather = load_weather(args.weather) if args.weather else None
        if args.command == "dataset":
            rows = build_training_rows(
                events,
                ais,
                lead_hours=args.lead_hours,
                require_ais=not args.allow_missing_ais,
                weather=weather,
                require_weather=args.require_weather,
                max_weather_age_hours=args.max_weather_age_hours,
            )
            if not rows:
                raise TFTDataError(
                    "No eligible labeled voyages; strict AIS matches and completed Port-MIS departures are required"
                )
            _atomic_csv(args.output, rows, args.overwrite)
            result = {"output": str(args.output.resolve()), "rows": len(rows)}
        else:
            as_of = parse_timestamp(args.as_of, naive_timezone=KST)
            bundle = build_feature_bundle(
                events,
                ais,
                as_of=as_of,
                model_version=args.model_version,
                encoder_length=args.encoder_length,
                lead_hours=args.lead_hours,
                require_ais=not args.allow_missing_ais,
                weather=weather,
                require_weather=args.require_weather,
                max_weather_age_hours=args.max_weather_age_hours,
            )
            _atomic_json(args.output, bundle, args.overwrite)
            result = {
                "output": str(args.output.resolve()),
                "windows": len(bundle["windows"]),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, csv.Error, TFTDataError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
