"""Normalization, validation and streaming behavior."""

from pathlib import Path

import pytest

from middleware.config import Config
from middleware.preprocessing import prepare_row, records
from middleware.utils import Stats, text, vessel_key


@pytest.mark.parametrize(
    "value", [None, "", "null", "None", "NaN", float("nan"), "<NA>"]
)
def test_null(value):
    assert text(value) is None


def test_fallback_and_missing_identifier(row):
    row["clsgn"] = ""
    stats = Stats()
    record = prepare_row(row, Config(Path("test")), stats, 1)
    assert record["vessel"]["name"] == "TEST VESSEL"
    assert record["vessel"]["vesselKey"] == vessel_key(None, "test vessel")
    row["vsslNm"] = " "
    assert prepare_row(row, Config(Path("test")), stats, 2) is None
    assert stats.counts["missing_vessel_identifier"] == 1


def test_invalid_fields_do_not_abort(row):
    row.update(etryptDt="bad-date", grtg="not-number", ldadngTon="inf", tugYn="unknown")
    stats = Stats()
    record = prepare_row(row, Config(Path("test")), stats, 1)
    assert record["vessel"]["grossTonnage"] is None
    assert record["portCall"]["arrivalTime"] is None
    assert stats.counts["invalid_datetime"] == 1
    assert stats.counts["invalid_numeric"] == 2
    assert stats.counts["invalid_rows"] == 1


def test_headers_required(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("record_id\nx\n")
    with pytest.raises(ValueError, match="CSV columns"):
        list(records(Config(path), Stats()))


def test_limit_and_leading_zeros(row, csv_file):
    path = csv_file([row, row])
    stats = Stats()
    result = list(records(Config(path), stats, limit=1))
    assert len(result) == 1
    assert result[0]["ports"]["AT_PORT"]["code"] == "020"
    assert stats.counts["input_rows"] == 1
    assert list(records(Config(path), Stats(), limit=0)) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"delay_threshold_minutes": -1},
        {"delay_threshold_minutes": float("nan")},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        Config(Path("x"), **kwargs)
