"""Rank AIS-to-Port-MIS voyage candidates using name, time, space and motion.

This module is deliberately separate from the strict matcher.  Its accepted
rows use ``match_status=probabilistic_matched`` and must be reviewed before
they are promoted to supervised-learning labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from .voyage_matching import (
    AIS_TIME_COLUMNS,
    KST,
    UTC,
    Voyage,
    VoyageMatchingError,
    load_voyages,
    normalize_identifier,
    parse_timestamp,
)

PORT_CENTERS = {
    "020": (35.10, 129.04),  # Busan
    "030": (37.45, 126.62),  # Incheon
}
ANNOTATION_COLUMNS = (
    "voyage_id",
    "match_status",
    "match_method",
    "match_confidence",
    "name_similarity",
    "temporal_score",
    "spatial_score",
    "motion_score",
    "candidate_margin",
    "port_call_sign",
    "port_vessel_name",
    "port_arrival_kst",
    "port_departure_kst",
    "hours_to_arrival",
)
CANDIDATE_COLUMNS = (
    "mmsi",
    "ais_vessel_name",
    "rank",
    *ANNOTATION_COLUMNS,
)


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    latitude: float
    longitude: float
    sog: float | None


@dataclass
class Track:
    mmsi: str
    vessel_name: str
    rows: list[dict[str, str]]
    observations: list[Observation]


@dataclass(frozen=True)
class Candidate:
    voyage: Voyage
    confidence: float
    name_similarity: float
    temporal_score: float
    spatial_score: float
    motion_score: float


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header:
        raise VoyageMatchingError(f"CSV has no header: {path}")
    return header


def _time_column(header: Sequence[str]) -> str:
    for column in AIS_TIME_COLUMNS:
        if column in header:
            return column
    raise VoyageMatchingError("AIS CSV has no supported timestamp column")


def _number(value: str | None) -> float | None:
    try:
        number = float((value or "").strip())
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def load_tracks(path: Path) -> tuple[list[str], list[Track], list[dict[str, str]]]:
    """Group valid position rows by MMSI and retain invalid rows for reporting."""

    header = _header(path)
    time_column = _time_column(header)
    grouped: dict[str, list[tuple[dict[str, str], Observation]]] = defaultdict(list)
    invalid = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            mmsi = normalize_identifier(row.get("mmsi") or row.get("MMSI"))
            latitude = _number(row.get("latitude") or row.get("metadata_latitude"))
            longitude = _number(row.get("longitude") or row.get("metadata_longitude"))
            try:
                timestamp = parse_timestamp(
                    (row.get(time_column) or "").strip(), naive_timezone=UTC
                )
            except ValueError:
                timestamp = None
            if not mmsi or timestamp is None or latitude is None or longitude is None:
                invalid.append(row)
                continue
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                invalid.append(row)
                continue
            grouped[mmsi].append(
                (
                    row,
                    Observation(
                        timestamp=timestamp,
                        latitude=latitude,
                        longitude=longitude,
                        sog=_number(row.get("sog") or row.get("SOG")),
                    ),
                )
            )
    tracks = []
    for mmsi, values in grouped.items():
        values.sort(key=lambda value: value[1].timestamp)
        names = Counter(
            (row.get("ship_name") or row.get("VesselName") or "").strip()
            for row, _ in values
            if (row.get("ship_name") or row.get("VesselName") or "").strip()
        )
        tracks.append(
            Track(
                mmsi=mmsi,
                vessel_name=names.most_common(1)[0][0] if names else "",
                rows=[row for row, _ in values],
                observations=[observation for _, observation in values],
            )
        )
    return header, sorted(tracks, key=lambda track: track.mmsi), invalid


def name_similarity(left: str, right: str) -> float:
    """Return a conservative normalized vessel-name similarity."""

    a, b = normalize_identifier(left), normalize_identifier(right)
    if len(a) < 4 or len(b) < 4:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _distance_hours(timestamp: datetime, voyage: Voyage) -> float | None:
    start = voyage.arrival or voyage.departure
    end = voyage.departure or voyage.arrival
    if start is None or end is None:
        return None
    if start > end:
        start, end = end, start
    if start <= timestamp <= end:
        return 0.0
    anchor = start if timestamp < start else end
    return abs((timestamp - anchor).total_seconds()) / 3600


def temporal_score(track: Track, voyage: Voyage, max_gap_hours: float) -> float:
    distances = [
        distance
        for item in track.observations
        if (distance := _distance_hours(item.timestamp, voyage)) is not None
    ]
    if not distances:
        return 0.0
    return max(0.0, 1.0 - min(distances) / max_gap_hours)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def spatial_score(track: Track, port_code: str, radius_km: float) -> float:
    center = PORT_CENTERS.get(port_code)
    if center is None:
        return 0.0
    distance = min(
        _haversine_km((item.latitude, item.longitude), center)
        for item in track.observations
    )
    return max(0.0, 1.0 - distance / radius_km)


def motion_score(track: Track) -> float:
    speeds = [item.sog for item in track.observations if item.sog is not None]
    if not speeds:
        return 0.0
    # A port-call track normally contains at least one slow approach or berth state.
    return sum(speed <= 3.0 for speed in speeds) / len(speeds)


def rank_candidates(
    track: Track,
    voyages: Sequence[Voyage],
    *,
    port_code: str,
    max_gap_hours: float = 12.0,
    port_radius_km: float = 35.0,
) -> list[Candidate]:
    """Score candidates using the current strict matcher's voyage boundaries."""

    spatial = spatial_score(track, port_code, port_radius_km)
    motion = motion_score(track)
    ranked = []
    for voyage in voyages:
        if voyage.port_authority_code != port_code:
            continue
        temporal = temporal_score(track, voyage, max_gap_hours)
        if temporal <= 0:
            continue
        similarity = name_similarity(track.vessel_name, voyage.vessel_name)
        confidence = (
            0.50 * similarity + 0.30 * temporal + 0.15 * spatial + 0.05 * motion
        )
        ranked.append(
            Candidate(
                voyage=voyage,
                confidence=confidence,
                name_similarity=similarity,
                temporal_score=temporal,
                spatial_score=spatial,
                motion_score=motion,
            )
        )
    return sorted(ranked, key=lambda candidate: candidate.confidence, reverse=True)


def _candidate_values(candidate: Candidate, margin: float) -> dict[str, str]:
    voyage = candidate.voyage
    return {
        "voyage_id": voyage.voyage_id,
        "match_method": "name_time_space_motion",
        "match_confidence": f"{candidate.confidence:.6f}",
        "name_similarity": f"{candidate.name_similarity:.6f}",
        "temporal_score": f"{candidate.temporal_score:.6f}",
        "spatial_score": f"{candidate.spatial_score:.6f}",
        "motion_score": f"{candidate.motion_score:.6f}",
        "candidate_margin": f"{margin:.6f}",
        "port_call_sign": voyage.call_sign,
        "port_vessel_name": voyage.vessel_name,
        "port_arrival_kst": voyage.arrival.astimezone(KST).isoformat()
        if voyage.arrival
        else "",
        "port_departure_kst": voyage.departure.astimezone(KST).isoformat()
        if voyage.departure
        else "",
        "hours_to_arrival": "",
    }


def match_probabilistically(
    ais_path: Path,
    voyages: Sequence[Voyage],
    output: Path,
    candidates_output: Path,
    *,
    port_code: str,
    threshold: float = 0.70,
    min_margin: float = 0.08,
    min_name_similarity: float = 0.55,
    min_spatial_score: float = 0.10,
    max_gap_hours: float = 12.0,
    port_radius_km: float = 35.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write annotated AIS rows and ranked track-level candidates atomically."""

    for path in (output, candidates_output):
        if path.exists() and not overwrite:
            raise VoyageMatchingError(f"Output exists (use --overwrite): {path}")
    header, tracks, invalid = load_tracks(ais_path)
    rankings = {
        track.mmsi: rank_candidates(
            track,
            voyages,
            port_code=port_code,
            max_gap_hours=max_gap_hours,
            port_radius_km=port_radius_km,
        )
        for track in tracks
    }
    decisions: dict[str, tuple[str, Candidate | None, float]] = {}
    for track in tracks:
        ranked = rankings[track.mmsi]
        if not track.vessel_name:
            decisions[track.mmsi] = ("missing_vessel_name", None, 0.0)
        elif not ranked:
            decisions[track.mmsi] = ("no_temporal_candidate", None, 0.0)
        else:
            best = ranked[0]
            margin = best.confidence - (ranked[1].confidence if len(ranked) > 1 else 0)
            if best.spatial_score < min_spatial_score:
                decisions[track.mmsi] = ("outside_port_radius", best, margin)
            elif (
                best.name_similarity < min_name_similarity
                or best.confidence < threshold
            ):
                decisions[track.mmsi] = ("below_threshold", best, margin)
            elif margin < min_margin:
                decisions[track.mmsi] = ("ambiguous_candidate", best, margin)
            else:
                decisions[track.mmsi] = ("probabilistic_matched", best, margin)

    # Enforce one-to-one voyage assignment. Keep only the strongest track.
    by_voyage: dict[str, list[str]] = defaultdict(list)
    for mmsi, (status, candidate, _) in decisions.items():
        if status == "probabilistic_matched" and candidate:
            by_voyage[candidate.voyage.voyage_id].append(mmsi)
    for mmsis in by_voyage.values():
        if len(mmsis) < 2:
            continue
        ordered = sorted(
            mmsis, key=lambda value: decisions[value][1].confidence, reverse=True
        )  # type: ignore[union-attr]
        for mmsi in ordered[1:]:
            _, candidate, margin = decisions[mmsi]
            decisions[mmsi] = ("voyage_collision", candidate, margin)

    output.parent.mkdir(parents=True, exist_ok=True)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    output_header = header + [name for name in ANNOTATION_COLUMNS if name not in header]
    temp_paths: list[Path] = []
    statuses: Counter[str] = Counter()
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8-sig", newline="", delete=False, dir=output.parent
        ) as stream:
            annotated_temp = Path(stream.name)
            temp_paths.append(annotated_temp)
            writer = csv.DictWriter(
                stream, fieldnames=output_header, extrasaction="ignore"
            )
            writer.writeheader()
            for track in tracks:
                status, candidate, margin = decisions[track.mmsi]
                statuses[status] += 1
                values = _candidate_values(candidate, margin) if candidate else {}
                for source in track.rows:
                    row = dict(source)
                    row.update({name: "" for name in ANNOTATION_COLUMNS})
                    row.update(values)
                    row["match_status"] = status
                    if candidate and candidate.voyage.arrival:
                        timestamp = parse_timestamp(
                            row.get("time_utc") or row.get("received_at_utc") or "",
                            naive_timezone=UTC,
                        )
                        row["hours_to_arrival"] = (
                            f"{(candidate.voyage.arrival - timestamp).total_seconds() / 3600:.3f}"
                        )
                    writer.writerow(row)
            for source in invalid:
                row = dict(source)
                row.update({name: "" for name in ANNOTATION_COLUMNS})
                row["match_status"] = "invalid_observation"
                writer.writerow(row)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            delete=False,
            dir=candidates_output.parent,
        ) as stream:
            candidates_temp = Path(stream.name)
            temp_paths.append(candidates_temp)
            writer = csv.DictWriter(stream, fieldnames=CANDIDATE_COLUMNS)
            writer.writeheader()
            for track in tracks:
                for rank, candidate in enumerate(rankings[track.mmsi][:3], 1):
                    second = (
                        rankings[track.mmsi][1].confidence
                        if len(rankings[track.mmsi]) > 1
                        else 0
                    )
                    values = _candidate_values(candidate, candidate.confidence - second)
                    writer.writerow(
                        {
                            "mmsi": track.mmsi,
                            "ais_vessel_name": track.vessel_name,
                            "rank": rank,
                            "match_status": decisions[track.mmsi][0]
                            if rank == 1
                            else "alternative",
                            **values,
                        }
                    )
        os.replace(annotated_temp, output)
        os.replace(candidates_temp, candidates_output)
    except Exception:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise
    return {
        "ais_tracks": len(tracks),
        "ais_rows": sum(len(track.rows) for track in tracks),
        "invalid_rows": len(invalid),
        "statuses": dict(sorted(statuses.items())),
        "accepted_tracks": statuses["probabilistic_matched"],
        "output": str(output.resolve()),
        "candidates_output": str(candidates_output.resolve()),
        "warning": "Probabilistic matches require review before use as TFT labels.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ais", type=Path, required=True)
    parser.add_argument("--portmis", type=Path, required=True)
    parser.add_argument("--port-code", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-name-similarity", type=float, default=0.55)
    parser.add_argument("--min-spatial-score", type=float, default=0.10)
    parser.add_argument("--max-gap-hours", type=float, default=12.0)
    parser.add_argument("--port-radius-km", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not 0 <= args.threshold <= 1 or not 0 <= args.min_margin <= 1:
            raise VoyageMatchingError(
                "threshold and min-margin must be between 0 and 1"
            )
        if not 0 <= args.min_name_similarity <= 1:
            raise VoyageMatchingError("min-name-similarity must be between 0 and 1")
        if not 0 <= args.min_spatial_score <= 1:
            raise VoyageMatchingError("min-spatial-score must be between 0 and 1")
        if args.max_gap_hours <= 0 or args.port_radius_km <= 0:
            raise VoyageMatchingError(
                "max-gap-hours and port-radius-km must be positive"
            )
        result = match_probabilistically(
            args.ais,
            load_voyages(args.portmis),
            args.output,
            args.candidates_output,
            port_code=args.port_code,
            threshold=args.threshold,
            min_margin=args.min_margin,
            min_name_similarity=args.min_name_similarity,
            min_spatial_score=args.min_spatial_score,
            max_gap_hours=args.max_gap_hours,
            port_radius_km=args.port_radius_km,
            overwrite=args.overwrite,
        )
        if args.report.exists() and not args.overwrite:
            raise VoyageMatchingError(f"Output exists (use --overwrite): {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, csv.Error, VoyageMatchingError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
