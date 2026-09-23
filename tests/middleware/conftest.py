"""CSV fixtures shared by middleware tests."""

import csv
from pathlib import Path

import pytest

from preprocess_port_calls import OUTPUT_COLUMNS


@pytest.fixture
def row() -> dict:
    """Return a complete synthetic row with parseable values."""
    result = dict.fromkeys(OUTPUT_COLUMNS, "")
    result.update(
        record_id="record-1",
        clsgn=" D7AB ",
        vsslNm=" Test  Vessel ",
        prtAgCd="020",
        prtAgNm="부산",
        laidupFcltyCd="OLD",
        laidupFcltySubCd="01",
        laidupFcltyNm="동명호안",
        prvsDpmprtNatPrtCd="KRYOS",
        prvsDpmprtPrtNm="여수",
    )
    return result


@pytest.fixture
def csv_file(tmp_path: Path):
    """Write fixture records without type inference."""

    def write(rows: list[dict]) -> Path:
        """Return the fixture CSV path."""
        path = tmp_path / "input.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return write
