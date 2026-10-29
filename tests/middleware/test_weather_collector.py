"""Offline tests for weather coordinates, API data integrity and time joins."""

import pytest

from collect_weather import coordinates, enrich_departures, inventory, weather_rows


def test_coordinates_and_unknown_regions():
    assert coordinates("3508N 12903E") == pytest.approx((35.1333333333, 129.05))
    assert coordinates("3355S 01825E") == pytest.approx((-33.9166666667, 18.4166666667))
    assert coordinates("3599N 12903E") is None
    result = inventory(
        [{"prvsDpmprtNatPrtCd": "KRZZZ"}], {}, {"KRZZZ": {"coordinates": [129, 35]}}
    )
    assert result[0]["status"] == "unresolved_coordinates"


def test_departure_join_converts_timezone_and_does_not_fill_missing_dates():
    weather = [
        {
            "port_code": "KRPUS",
            "weather_time_utc": "2026-07-31T15:00+00:00",
            "temperature_2m_c": 25,
            "wind_speed_10m_ms": 2,
            "rain_mm": 0,
        }
    ]
    rows = [
        {"prtAgCd": "020", "tkoffDt": "2026-08-01T00:20:00+09:00"},
        {"prtAgCd": "020", "etryptDt": "2026-08-01T00:20:00+09:00"},
        {"prtAgCd": "020", "tkoffDt": "2026-10-01T00:20:00+09:00"},
    ]
    enriched = enrich_departures(rows, weather)
    assert enriched[0]["departure_rain_mm"] == 0
    assert enriched[0]["departure_weather_status"] == "matched"
    assert enriched[1]["departure_weather_status"] == "missing_departure_time"
    assert enriched[2]["departure_weather_status"] == "outside_collected_period"
    assert "departure_rain_mm" not in rows[0]


def test_incheon_authority_inventory_and_weather_join():
    locations = {"KRINC": {"Coordinates": "3727N 12637E", "Name": "Incheon"}}
    rows = [
        {
            "prtAgCd": "030",
            "prtAgNm": "인천",
            "prvsDpmprtNatPrtCd": "KRPUS",
            "tkoffDt": "2026-01-01T09:20:00+09:00",
        }
    ]
    regions = inventory(rows, locations, current_port_only=True)
    assert [region["port_code"] for region in regions] == ["KRINC"]
    weather = [
        {
            "port_code": "KRINC",
            "weather_time_utc": "2026-01-01T00:00+00:00",
            "temperature_2m_c": 1,
            "wind_speed_10m_ms": 3,
            "rain_mm": 0,
        }
    ]
    assert enrich_departures(rows, weather)[0]["departure_weather_status"] == "matched"


def test_api_missing_values_remain_null_and_wrong_units_fail():
    payload = {
        "hourly_units": {"temperature_2m": "°C", "wind_speed_10m": "m/s", "rain": "mm"},
        "utc_offset_seconds": 0,
        "latitude": 35,
        "longitude": 129,
        "hourly": {
            "time": ["2026-08-01T00:00"],
            "temperature_2m": [None],
            "wind_speed_10m": [2],
            "rain": [0],
        },
    }
    region = {"port_code": "KRPUS", "latitude": 35, "longitude": 129}
    assert next(weather_rows(payload, region, "now"))["temperature_2m_c"] is None
    payload["hourly_units"]["wind_speed_10m"] = "km/h"
    with pytest.raises(ValueError, match="units"):
        list(weather_rows(payload, region, "now"))
