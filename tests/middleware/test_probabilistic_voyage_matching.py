import csv
from pathlib import Path

from middleware.probabilistic_voyage_matching import (
    load_tracks,
    match_probabilistically,
    name_similarity,
)
from middleware.voyage_matching import load_voyages


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_name_similarity_normalizes_spacing_and_case() -> None:
    assert name_similarity("Test Ship 1", "TESTSHIP1") == 1.0
    assert name_similarity("P-1", "P1") == 0.0


def test_high_confidence_candidate_is_kept_separate_from_strict_match(
    tmp_path: Path,
) -> None:
    port = tmp_path / "port.csv"
    ais = tmp_path / "ais.csv"
    output = tmp_path / "matched.csv"
    candidates = tmp_path / "candidates.csv"
    write_csv(
        port,
        [
            "prtAgCd",
            "clsgn",
            "vsslNm",
            "etryptYear",
            "etryptCo",
            "etryptDt",
            "tkoffDt",
            "tkoffPrrrnDt",
        ],
        [
            {
                "prtAgCd": "020",
                "clsgn": "D7TEST",
                "vsslNm": "TEST SHIP",
                "etryptYear": "2026",
                "etryptCo": "001",
                "etryptDt": "2026-08-23T09:00:00+09:00",
                "tkoffDt": "2026-08-23T12:00:00+09:00",
                "tkoffPrrrnDt": "2026-08-23T11:00:00+09:00",
            }
        ],
    )
    write_csv(
        ais,
        ["time_utc", "mmsi", "ship_name", "latitude", "longitude", "sog"],
        [
            {
                "time_utc": "2026-08-23T01:00:00+00:00",
                "mmsi": "440000001",
                "ship_name": "test ship",
                "latitude": "35.10",
                "longitude": "129.04",
                "sog": "0.2",
            }
        ],
    )

    result = match_probabilistically(
        ais,
        load_voyages(port),
        output,
        candidates,
        port_code="020",
    )
    _, tracks, invalid = load_tracks(ais)
    with output.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))

    assert len(tracks) == 1 and not invalid
    assert result["accepted_tracks"] == 1
    assert row["match_status"] == "probabilistic_matched"
    assert row["voyage_id"] == "020-2026-D7TEST-001"
    assert float(row["match_confidence"]) == 1.0
