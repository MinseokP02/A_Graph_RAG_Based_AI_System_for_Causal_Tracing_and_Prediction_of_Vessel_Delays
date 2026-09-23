"""Validate control-event identities and timestamp handling before writes."""

import csv

import pytest

from collect_vessel_control import CONTROL_FIELDS
from middleware.config import Config
from middleware.import_control import prepare


def test_control_preparation(tmp_path):
    path = tmp_path / "events.csv"
    row = dict.fromkeys(CONTROL_FIELDS, "")
    row.update(
        record_id="r1",
        prtAgCd="020",
        clsgn="abc",
        etryptYear="2026",
        etryptCo="006",
        cntrlOpertDt="2026-08-01T01:00:00",
        cntrlSe="03",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONTROL_FIELDS)
        writer.writeheader()
        writer.writerows([row, row])
    events = prepare(path, Config(csv_path=path))
    assert len(events) == 1
    assert events[0]["entryCount"] == "006"
    assert events[0]["callSign"] == "ABC"
    assert events[0]["eventTime"].endswith("+09:00")
    path.write_text(path.read_text().replace("2026-08-01T01:00:00", "bad-time"))
    with pytest.raises(ValueError, match="timestamp"):
        prepare(path, Config(csv_path=path))
