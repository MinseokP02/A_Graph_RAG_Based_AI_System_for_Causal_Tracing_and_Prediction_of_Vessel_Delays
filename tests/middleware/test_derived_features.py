"""Derived features and timezone edge cases."""

from pathlib import Path

import pytest

from middleware.config import Config
from middleware.preprocessing import prepare_row
from middleware.utils import Stats


@pytest.mark.parametrize(
    "planned,departure,delay,deviation",
    [
        ("2026-08-01 10:00:00", "2026-08-01 11:30:00", 90, 90),
        ("2026-08-01 11:00:00", "2026-08-01 10:00:00", 0, -60),
    ],
)
def test_delay(row, planned, departure, delay, deviation):
    row.update(tkoffPrrrnDt=planned, tkoffDt=departure)
    pc = prepare_row(row, Config(Path("test")), Stats(), 1)["portCall"]
    assert pc["departureDelayMinutes"] == delay
    assert pc["departureScheduleDeviationMinutes"] == deviation
    assert pc["isDepartureDelayed"] == (delay >= 60)


def test_stay_and_cargo(row):
    row.update(
        etryptDt="2026-08-01T10:00:00+09:00",
        tkoffDt="2026-08-01 13:00:00",
        landngFrghtTon="100",
        ldFrghtTon="200",
        trnpdtTon="50",
        ldadngTon="9999",
    )
    pc = prepare_row(row, Config(Path("test")), Stats(), 1)["portCall"]
    assert pc["portStayMinutes"] == 180
    assert pc["cargoOperationTon"] == 350


def test_negative_stay_and_missing_delay(row):
    row.update(etryptDt="2026-08-01 13:00:00", tkoffDt="2026-08-01 10:00:00")
    stats = Stats()
    pc = prepare_row(row, Config(Path("test")), stats, 1)["portCall"]
    assert pc["portStayMinutes"] is None
    assert pc["departureDelayMinutes"] is None
    assert pc["isDepartureDelayed"] is None
    assert pc["cargoOperationTon"] == 0
    assert pc["loadedCargoTon"] is None
    assert stats.counts["negative_port_stay"] == 1


def test_threshold_and_offset(row):
    row.update(tkoffPrrrnDt="2026-08-01T01:00:00Z", tkoffDt="2026-08-01 11:30:00")
    pc = prepare_row(
        row, Config(Path("test"), delay_threshold_minutes=100), Stats(), 1
    )["portCall"]
    assert pc["departureDelayMinutes"] == 90
    assert pc["isDepartureDelayed"] is False
    assert pc["departureTime"].endswith("+09:00")
