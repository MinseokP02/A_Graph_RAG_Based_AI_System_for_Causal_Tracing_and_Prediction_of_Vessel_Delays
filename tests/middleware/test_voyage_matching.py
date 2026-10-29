import csv
from datetime import datetime, timezone
from pathlib import Path

from middleware.voyage_matching import (
    assess_coverage,
    assess_identity_coverage,
    load_voyages,
    match_ais_to_voyages,
    parse_timestamp,
)

PORT_FIELDS = [
    "prtAgCd",
    "clsgn",
    "vsslNm",
    "etryptYear",
    "etryptCo",
    "etryptDt",
    "tkoffDt",
    "tkoffPrrrnDt",
    "prvsDpmprtPrtNm",
    "nxlnptPrtNm",
    "dstnPrtNm",
]
AIS_FIELDS = [
    "time_utc",
    "received_at_utc",
    "mmsi",
    "ship_name",
    "call_sign",
    "latitude",
    "longitude",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def port_row(
    *,
    call_sign: str = "D7TEST",
    count: str = "001",
    arrival: str = "",
    departure: str = "",
) -> dict[str, str]:
    return {
        "prtAgCd": "020",
        "clsgn": call_sign,
        "vsslNm": "TEST SHIP",
        "etryptYear": "2026",
        "etryptCo": count,
        "etryptDt": arrival,
        "tkoffDt": departure,
        "tkoffPrrrnDt": "",
        "prvsDpmprtPrtNm": "TOKYO",
        "nxlnptPrtNm": "BUSAN",
        "dstnPrtNm": "BUSAN",
    }


def test_parse_aisstream_nanosecond_timestamp() -> None:
    parsed = parse_timestamp(
        "2026-08-23 08:25:47.945049250 +0000 UTC", naive_timezone=timezone.utc
    )
    assert parsed == datetime(2026, 8, 23, 8, 25, 47, 945049, tzinfo=timezone.utc)


def test_load_voyages_pairs_arrival_and_departure(tmp_path: Path) -> None:
    port = tmp_path / "port.csv"
    write_csv(
        port,
        PORT_FIELDS,
        [
            port_row(arrival="2026-08-02T09:00:00+09:00"),
            port_row(departure="2026-08-02T12:00:00+09:00"),
        ],
    )

    voyages = load_voyages(port)

    assert len(voyages) == 1
    assert voyages[0].voyage_id == "020-2026-D7TEST-001"
    assert voyages[0].arrival == datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
    assert voyages[0].departure == datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc)


def test_coverage_requires_each_day_in_requested_period(tmp_path: Path) -> None:
    port = tmp_path / "port.csv"
    ais = tmp_path / "ais.csv"
    write_csv(port, PORT_FIELDS, [port_row(arrival="2026-08-01T00:00:00+09:00")])
    write_csv(
        ais,
        AIS_FIELDS,
        [
            {
                "time_utc": "2026-08-23T08:25:00+00:00",
                "received_at_utc": "",
                "mmsi": "440000001",
                "ship_name": "TEST SHIP",
                "call_sign": "D7TEST",
                "latitude": "35.1",
                "longitude": "129.1",
            }
        ],
    )

    report = assess_coverage([ais], load_voyages(port), required_months=1)

    assert report["coverage"]["full"] is False
    assert report["coverage"]["covered_days"] == 1
    assert report["coverage"]["required_days"] == 31


def test_identity_audit_requires_common_call_sign(tmp_path: Path) -> None:
    port = tmp_path / "port.csv"
    ais = tmp_path / "ais.csv"
    write_csv(port, PORT_FIELDS, [port_row(arrival="2026-08-01T00:00:00+09:00")])
    write_csv(
        ais,
        AIS_FIELDS,
        [
            {
                "time_utc": "2026-08-01T00:00:00+00:00",
                "received_at_utc": "",
                "mmsi": "440000001",
                "ship_name": "TEST SHIP",
                "call_sign": "UNRELATED",
                "latitude": "35.1",
                "longitude": "129.1",
            }
        ],
    )

    audit = assess_identity_coverage([ais], load_voyages(port))

    assert audit["ais_unique_mmsi"] == 1
    assert audit["exact_call_sign_intersections"] == 0
    assert audit["exact_normalized_name_intersections"] == 1
    assert audit["strict_identity_linkable"] is False


def test_verified_crosswalk_matches_one_voyage(tmp_path: Path) -> None:
    port = tmp_path / "port.csv"
    ais = tmp_path / "ais.csv"
    output = tmp_path / "matched.csv"
    write_csv(
        port,
        PORT_FIELDS,
        [
            port_row(arrival="2026-08-02T09:00:00+09:00"),
            port_row(departure="2026-08-02T12:00:00+09:00"),
        ],
    )
    write_csv(
        ais,
        AIS_FIELDS,
        [
            {
                "time_utc": "2026-08-01T12:00:00+00:00",
                "received_at_utc": "",
                "mmsi": "440000001",
                "ship_name": "",
                "call_sign": "",
                "latitude": "35.1",
                "longitude": "129.1",
            }
        ],
    )

    summary = match_ais_to_voyages(
        [ais], load_voyages(port), output, crosswalk={"440000001": "D7TEST"}
    )
    with output.open(encoding="utf-8-sig", newline="") as stream:
        matched = next(csv.DictReader(stream))

    assert summary["matched_rows"] == 1
    assert matched["voyage_id"] == "020-2026-D7TEST-001"
    assert matched["match_method"] == "verified_crosswalk"
    assert matched["hours_to_arrival"] == "12.000"


def test_overlapping_voyage_windows_are_rejected_as_ambiguous(tmp_path: Path) -> None:
    port = tmp_path / "port.csv"
    ais = tmp_path / "ais.csv"
    output = tmp_path / "matched.csv"
    write_csv(
        port,
        PORT_FIELDS,
        [
            port_row(count="001", arrival="2026-08-02T09:00:00+09:00"),
            port_row(count="001", departure="2026-08-02T12:00:00+09:00"),
            port_row(count="002", arrival="2026-08-03T09:00:00+09:00"),
            port_row(count="002", departure="2026-08-03T12:00:00+09:00"),
        ],
    )
    write_csv(
        ais,
        AIS_FIELDS,
        [
            {
                "time_utc": "2026-08-01T12:00:00+00:00",
                "received_at_utc": "",
                "mmsi": "440000001",
                "ship_name": "TEST SHIP",
                "call_sign": "D7TEST",
                "latitude": "35.1",
                "longitude": "129.1",
            }
        ],
    )

    summary = match_ais_to_voyages([ais], load_voyages(port), output)
    with output.open(encoding="utf-8-sig", newline="") as stream:
        matched = next(csv.DictReader(stream))

    assert summary["matched_rows"] == 0
    assert matched["match_status"] == "ambiguous_voyage"


def test_name_matching_is_opt_in(tmp_path: Path) -> None:
    port = tmp_path / "port.csv"
    ais = tmp_path / "ais.csv"
    strict_output = tmp_path / "strict.csv"
    name_output = tmp_path / "name.csv"
    write_csv(port, PORT_FIELDS, [port_row(arrival="2026-08-02T09:00:00+09:00")])
    write_csv(
        ais,
        AIS_FIELDS,
        [
            {
                "time_utc": "2026-08-01T12:00:00+00:00",
                "received_at_utc": "",
                "mmsi": "440000001",
                "ship_name": "TEST SHIP",
                "call_sign": "",
                "latitude": "35.1",
                "longitude": "129.1",
            }
        ],
    )
    voyages = load_voyages(port)

    match_ais_to_voyages([ais], voyages, strict_output)
    match_ais_to_voyages([ais], voyages, name_output, allow_unique_name=True)
    with strict_output.open(encoding="utf-8-sig", newline="") as stream:
        strict = next(csv.DictReader(stream))
    with name_output.open(encoding="utf-8-sig", newline="") as stream:
        by_name = next(csv.DictReader(stream))

    assert strict["match_status"] == "identity_not_verified"
    assert by_name["match_status"] == "matched"
    assert by_name["match_method"] == "unique_vessel_name"
