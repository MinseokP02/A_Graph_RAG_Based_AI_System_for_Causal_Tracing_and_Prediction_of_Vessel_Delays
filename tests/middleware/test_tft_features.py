import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from middleware.tft_api import FeatureBundle
from middleware.tft_features import (
    TFTDataError,
    build_feature_bundle,
    build_training_rows,
    load_matched_ais,
    load_port_events,
    load_weather,
)
from middleware.train_tft import validate_training_csv

PORT_FIELDS = [
    "prtAgCd",
    "clsgn",
    "vsslNm",
    "etryptYear",
    "etryptCo",
    "etryptDt",
    "tkoffDt",
    "tkoffPrrrnDt",
    "grtg",
    "tugYn",
    "piltgYn",
    "ldadngTon",
    "trnpdtTon",
    "landngFrghtTon",
    "ldFrghtTon",
]
AIS_FIELDS = [
    "time_utc",
    "voyage_id",
    "match_status",
    "latitude",
    "longitude",
    "sog",
    "cog",
    "true_heading",
    "hours_to_arrival",
]
WEATHER_FIELDS = [
    "port_code",
    "weather_time_utc",
    "temperature_2m_c",
    "wind_speed_10m_ms",
    "rain_mm",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def port_rows(
    count: str,
    planned: str,
    actual: str = "",
) -> list[dict[str, str]]:
    base = {
        "prtAgCd": "020",
        "clsgn": f"D7T{count}",
        "vsslNm": f"SHIP {count}",
        "etryptYear": "2026",
        "etryptCo": count,
        "etryptDt": "2026-07-31T00:00:00+00:00",
        "tkoffDt": "",
        "tkoffPrrrnDt": planned,
        "grtg": "1000",
        "tugYn": "Y",
        "piltgYn": "N",
        "ldadngTon": "10",
        "trnpdtTon": "20",
        "landngFrghtTon": "30",
        "ldFrghtTon": "40",
    }
    rows = [base]
    if actual:
        departure = dict(base)
        departure["etryptDt"] = ""
        departure["tkoffDt"] = actual
        departure["tkoffPrrrnDt"] = ""
        rows.append(departure)
    return rows


def ais_row(
    voyage_id: str, timestamp: str, latitude: str, sog: str = "10"
) -> dict[str, str]:
    return {
        "time_utc": timestamp,
        "voyage_id": voyage_id,
        "match_status": "matched",
        "latitude": latitude,
        "longitude": "129.1",
        "sog": sog,
        "cog": "180",
        "true_heading": "179",
        "hours_to_arrival": "3",
    }


def fixtures(tmp_path: Path) -> tuple[Path, Path]:
    port = tmp_path / "port.csv"
    ais = tmp_path / "matched.csv"
    rows = []
    rows += port_rows("001", "2026-08-01T12:00:00+09:00", "2026-08-01T13:00:00+09:00")
    rows += port_rows("002", "2026-08-02T12:00:00+09:00", "2026-08-02T12:30:00+09:00")
    rows += port_rows("003", "2026-08-03T12:00:00+09:00", "2026-08-03T12:10:00+09:00")
    rows += port_rows("004", "2026-08-05T12:00:00+09:00")
    write_csv(port, PORT_FIELDS, rows)
    write_csv(
        ais,
        AIS_FIELDS,
        [
            ais_row("020-2026-D7T001-001", "2026-07-31T20:00:00+00:00", "35.0", "8"),
            # This observation is after the six-hour cutoff and must be excluded.
            ais_row("020-2026-D7T001-001", "2026-07-31T22:00:00+00:00", "99.0", "20"),
            ais_row("020-2026-D7T002-002", "2026-08-01T20:00:00+00:00", "35.1"),
            ais_row("020-2026-D7T003-003", "2026-08-02T20:00:00+00:00", "35.2"),
            ais_row("020-2026-D7T004-004", "2026-08-03T23:00:00+00:00", "35.3"),
            # Future relative to bundle as_of and therefore excluded.
            ais_row("020-2026-D7T004-004", "2026-08-04T01:00:00+00:00", "88.0"),
        ],
    )
    return port, ais


def test_training_rows_use_only_point_in_time_ais(tmp_path: Path) -> None:
    port, ais_path = fixtures(tmp_path)
    rows = build_training_rows(
        load_port_events(port), load_matched_ais(ais_path), lead_hours=6
    )

    assert len(rows) == 3
    assert [row["time_idx"] for row in rows] == [0, 1, 2]
    assert rows[0]["delay_minutes"] == 60
    assert rows[0]["ais_observation_count"] == 1
    assert rows[0]["ais_latitude"] == 35.0
    assert rows[0]["ais_sog_mean"] == 8.0
    assert rows[0]["cargo_ton"] == 100.0


def test_training_rows_use_latest_weather_before_cutoff(tmp_path: Path) -> None:
    port, ais_path = fixtures(tmp_path)
    weather_path = tmp_path / "weather.csv"
    write_csv(
        weather_path,
        WEATHER_FIELDS,
        [
            {
                "port_code": "KRPUS",
                "weather_time_utc": "2026-07-31T19:00:00+00:00",
                "temperature_2m_c": "27.5",
                "wind_speed_10m_ms": "4.2",
                "rain_mm": "0.3",
            },
            {
                # Six-hour cutoff is 21:00 UTC, so this future value is excluded.
                "port_code": "KRPUS",
                "weather_time_utc": "2026-07-31T22:00:00+00:00",
                "temperature_2m_c": "99",
                "wind_speed_10m_ms": "99",
                "rain_mm": "99",
            },
        ],
    )

    rows = build_training_rows(
        load_port_events(port),
        load_matched_ais(ais_path),
        lead_hours=6,
        weather=load_weather(weather_path),
    )

    assert rows[0]["has_weather"] == 1.0
    assert rows[0]["weather_age_hours"] == 2.0
    assert rows[0]["temperature_2m_c"] == 27.5
    assert rows[0]["wind_speed_10m_ms"] == 4.2
    assert rows[0]["rain_mm"] == 0.3
    assert rows[1]["has_weather"] == 0.0


def test_bundle_contains_history_and_one_unlabeled_pending_event(
    tmp_path: Path,
) -> None:
    port, ais_path = fixtures(tmp_path)
    bundle = build_feature_bundle(
        load_port_events(port),
        load_matched_ais(ais_path),
        as_of=datetime(2026, 8, 4, tzinfo=timezone.utc),
        model_version="departure-delay-v1",
        encoder_length=2,
    )

    assert bundle["model_version"] == "departure-delay-v1"
    assert len(bundle["windows"]) == 1
    window = bundle["windows"][0]
    assert window["series_id"] == "020-2026-D7T004-004"
    assert window["selectors"]["call_sign"] == "D7T004"
    assert [row["time_idx"] for row in window["rows"]] == [0, 1, 2]
    assert window["rows"][-1]["delay_minutes"] == 0.0
    assert window["rows"][-1]["ais_latitude"] == 35.3
    assert window["rows"][-1]["ais_observation_count"] == 1
    assert window["rows"][-1]["has_weather"] == 0.0
    assert (
        FeatureBundle.model_validate(bundle).windows[0].series_id == window["series_id"]
    )


def test_training_validation_rejects_dataset_without_ais(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    fields = ["delay_minutes", "has_ais"]
    write_csv(path, fields, [{"delay_minutes": "10", "has_ais": "0.0"}])

    with pytest.raises(TFTDataError, match="columns missing"):
        validate_training_csv(path, encoder_length=2)


def test_probabilistic_matches_are_explicitly_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "matches.csv"
    row = ais_row("020-2026-D7T001-001", "2026-07-31T20:00:00+00:00", "35.0")
    row["match_status"] = "probabilistic_matched"
    write_csv(path, AIS_FIELDS, [row])

    assert load_matched_ais(path) == {}
    assert load_matched_ais(
        path, accepted_statuses=("matched", "probabilistic_matched")
    )
