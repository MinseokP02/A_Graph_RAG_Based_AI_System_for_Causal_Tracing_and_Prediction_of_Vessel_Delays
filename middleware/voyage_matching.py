"""Match AIS observations to Port-MIS port calls without guessing.

The default command first validates temporal coverage.  Matching is stopped when
the AIS files do not cover the requested period; use ``--allow-partial-coverage``
only for an explicitly partial result.  Vessel identity is based on an exact call
sign or a user-supplied, verified MMSI-to-call-sign crosswalk.  Exact, unique
vessel-name matching is available as an opt-in fallback.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc
AIS_TIME_COLUMNS = ("time_utc", "BaseDateTime", "basedatetime", "received_at_utc")
PORT_REQUIRED_COLUMNS = (
    "prtAgCd",
    "clsgn",
    "vsslNm",
    "etryptYear",
    "etryptCo",
    "etryptDt",
    "tkoffDt",
)
MATCH_COLUMNS = (
    "voyage_id",
    "match_status",
    "match_method",
    "port_call_sign",
    "port_vessel_name",
    "port_arrival_kst",
    "port_departure_kst",
    "port_planned_departure_kst",
    "hours_to_arrival",
    "previous_port",
    "next_port",
    "destination_port",
)


class VoyageMatchingError(ValueError):
    """Raised for invalid input data or an unsafe matching request."""


@dataclass(frozen=True)
class FileRange:
    path: str
    timestamp_column: str | None
    first_utc: str | None
    last_utc: str | None
    valid_rows: int
    invalid_rows: int
    utc_days: tuple[str, ...]
    kst_days: tuple[str, ...]


@dataclass(frozen=True)
class Voyage:
    voyage_id: str
    port_authority_code: str
    call_sign: str
    call_sign_key: str
    vessel_name: str
    vessel_name_key: str
    entry_year: str
    entry_count: str
    arrival: datetime | None
    departure: datetime | None
    planned_departure: datetime | None
    previous_port: str
    next_port: str
    destination_port: str


@dataclass(frozen=True)
class Identity:
    call_sign_key: str | None
    vessel_name_key: str | None
    method: str | None


def normalize_identifier(value: str | None) -> str:
    """Normalize vessel identifiers while retaining non-Latin letters."""

    return "".join(char for char in (value or "").upper().strip() if char.isalnum())


def parse_timestamp(value: str, *, naive_timezone: timezone | ZoneInfo) -> datetime:
    """Parse ISO-8601 and the nanosecond Go timestamp emitted by AISStream."""

    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    text = re.sub(r"\s+UTC$", "", text)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Python accepts at most six fractional digits. AISStream can emit nine.
    match = re.match(r"^(.*?\.)(\d+)(\s*[+-]\d{2}:?\d{2})$", text)
    if match:
        fraction = match.group(2)[:6].ljust(6, "0")
        text = f"{match.group(1)}{fraction}{match.group(3)}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_timezone)
    return parsed.astimezone(UTC)


def add_months(value: date, months: int) -> date:
    """Add calendar months, clamping the day when necessary."""

    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header:
        raise VoyageMatchingError(f"CSV has no header: {path}")
    if len(header) != len(set(header)):
        raise VoyageMatchingError(f"CSV contains duplicate columns: {path}")
    return header


def _choose_column(header: Sequence[str], candidates: Sequence[str], path: Path) -> str:
    for candidate in candidates:
        if candidate in header:
            return candidate
    raise VoyageMatchingError(
        f"No supported timestamp column in {path}; expected one of {', '.join(candidates)}"
    )


def inspect_ais_file(path: Path) -> FileRange:
    """Return a streaming date-range and daily-coverage profile for one AIS CSV."""

    header = _read_header(path)
    column = _choose_column(header, AIS_TIME_COLUMNS, path)
    first: datetime | None = None
    last: datetime | None = None
    valid = 0
    invalid = 0
    days: set[str] = set()
    kst_days: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw = (row.get(column) or "").strip()
            if not raw:
                invalid += 1
                continue
            try:
                timestamp = parse_timestamp(raw, naive_timezone=UTC)
            except ValueError:
                invalid += 1
                continue
            valid += 1
            first = timestamp if first is None or timestamp < first else first
            last = timestamp if last is None or timestamp > last else last
            days.add(timestamp.date().isoformat())
            kst_days.add(timestamp.astimezone(KST).date().isoformat())
    return FileRange(
        path=str(path.resolve()),
        timestamp_column=column,
        first_utc=first.isoformat() if first else None,
        last_utc=last.isoformat() if last else None,
        valid_rows=valid,
        invalid_rows=invalid,
        utc_days=tuple(sorted(days)),
        kst_days=tuple(sorted(kst_days)),
    )


def load_voyages(path: Path) -> list[Voyage]:
    """Pair Port-MIS arrival/departure rows using its official visit key."""

    header = _read_header(path)
    missing = set(PORT_REQUIRED_COLUMNS) - set(header)
    if missing:
        raise VoyageMatchingError(
            f"Port-MIS CSV is missing columns: {', '.join(sorted(missing))}"
        )
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            if None in row:
                raise VoyageMatchingError(
                    f"Malformed Port-MIS CSV row {line_number}: {path}"
                )
            key = tuple(
                (row.get(column) or "").strip()
                for column in ("prtAgCd", "clsgn", "etryptYear", "etryptCo")
            )
            if not all(key):
                raise VoyageMatchingError(
                    f"Missing Port-MIS visit identifier at row {line_number}: {path}"
                )
            groups[key].append(row)

    voyages: list[Voyage] = []
    for key, rows in groups.items():
        authority, call_sign, year, count = key
        arrivals = _distinct_timestamps(rows, "etryptDt", KST)
        departures = _distinct_timestamps(rows, "tkoffDt", KST)
        planned = _distinct_timestamps(rows, "tkoffPrrrnDt", KST)
        if len(arrivals) > 1 or len(departures) > 1:
            raise VoyageMatchingError(
                f"Conflicting timestamps for Port-MIS visit {authority}/{call_sign}/{year}/{count}"
            )
        vessel_names = {
            (row.get("vsslNm") or "").strip() for row in rows if row.get("vsslNm")
        }
        if len(vessel_names) > 1:
            raise VoyageMatchingError(
                f"Conflicting vessel names for Port-MIS visit {authority}/{call_sign}/{year}/{count}"
            )
        vessel_name = next(iter(vessel_names), "")
        voyage_id = "-".join((authority, year, normalize_identifier(call_sign), count))
        voyages.append(
            Voyage(
                voyage_id=voyage_id,
                port_authority_code=authority,
                call_sign=call_sign,
                call_sign_key=normalize_identifier(call_sign),
                vessel_name=vessel_name,
                vessel_name_key=normalize_identifier(vessel_name),
                entry_year=year,
                entry_count=count,
                arrival=next(iter(arrivals), None),
                departure=next(iter(departures), None),
                planned_departure=next(iter(planned), None),
                previous_port=_first_value(rows, "prvsDpmprtPrtNm"),
                next_port=_first_value(rows, "nxlnptPrtNm"),
                destination_port=_first_value(rows, "dstnPrtNm"),
            )
        )
    return sorted(
        voyages,
        key=lambda item: (
            item.arrival or datetime.max.replace(tzinfo=UTC),
            item.voyage_id,
        ),
    )


def _distinct_timestamps(
    rows: Iterable[dict[str, str]], column: str, naive_timezone: timezone | ZoneInfo
) -> set[datetime]:
    result: set[datetime] = set()
    for row in rows:
        raw = (row.get(column) or "").strip()
        if raw:
            result.add(parse_timestamp(raw, naive_timezone=naive_timezone))
    return result


def _first_value(rows: Iterable[dict[str, str]], column: str) -> str:
    return next(
        ((row.get(column) or "").strip() for row in rows if row.get(column)), ""
    )


def assess_coverage(
    ais_files: Sequence[Path], voyages: Sequence[Voyage], required_months: int = 3
) -> dict[str, object]:
    """Assess time coverage and report whether strict matching can proceed."""

    profiles = [inspect_ais_file(path) for path in ais_files]
    arrivals = [voyage.arrival for voyage in voyages if voyage.arrival]
    if not arrivals:
        raise VoyageMatchingError("Port-MIS CSV has no valid arrival timestamps")
    port_start = min(arrivals).astimezone(KST)
    port_end = max(arrivals).astimezone(KST)
    required_start_date = port_start.date()
    required_end_exclusive = add_months(required_start_date, required_months)
    required_end_date = required_end_exclusive - timedelta(days=1)
    required_days = {
        (required_start_date + timedelta(days=offset)).isoformat()
        for offset in range((required_end_exclusive - required_start_date).days)
    }
    # Compare calendar days in KST, because Port-MIS operates in local time.
    observed_days: set[str] = set()
    starts: list[datetime] = []
    ends: list[datetime] = []
    for profile in profiles:
        if profile.first_utc:
            starts.append(datetime.fromisoformat(profile.first_utc))
        if profile.last_utc:
            ends.append(datetime.fromisoformat(profile.last_utc))
        observed_days.update(profile.kst_days)
    covered_days = required_days & observed_days
    missing_days = sorted(required_days - observed_days)
    ais_start = min(starts) if starts else None
    ais_end = max(ends) if ends else None
    range_covers = bool(
        ais_start
        and ais_end
        and ais_start
        <= datetime.combine(required_start_date, time.min, KST).astimezone(UTC)
        and ais_end
        >= datetime.combine(required_end_exclusive, time.min, KST).astimezone(UTC)
    )
    full_coverage = range_covers and not missing_days
    return {
        "port_mis": {
            "arrival_start_kst": port_start.isoformat(),
            "arrival_end_kst": port_end.isoformat(),
            "voyages": len(voyages),
        },
        "required_ais_period": {
            "start_kst": required_start_date.isoformat(),
            "end_kst": required_end_date.isoformat(),
            "calendar_months": required_months,
            "days": len(required_days),
        },
        "ais_files": [asdict(profile) for profile in profiles],
        "coverage": {
            "full": full_coverage,
            "range_covers": range_covers,
            "covered_days": len(covered_days),
            "required_days": len(required_days),
            "day_coverage_ratio": round(len(covered_days) / len(required_days), 6),
            "missing_days": missing_days,
        },
    }


def assess_identity_coverage(
    ais_files: Sequence[Path], voyages: Sequence[Voyage]
) -> dict[str, object]:
    """Measure whether AIS and Port-MIS contain common deterministic identities."""

    ais_rows = 0
    rows_with_mmsi = 0
    rows_with_call_sign = 0
    mmsi_values: set[str] = set()
    ais_call_signs: set[str] = set()
    ais_names: set[str] = set()
    for path in ais_files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                ais_rows += 1
                mmsi = normalize_identifier(row.get("mmsi") or row.get("MMSI"))
                call_sign = normalize_identifier(
                    row.get("call_sign") or row.get("CallSign")
                )
                vessel_name = normalize_identifier(
                    row.get("ship_name") or row.get("VesselName")
                )
                if mmsi:
                    rows_with_mmsi += 1
                    mmsi_values.add(mmsi)
                if call_sign:
                    rows_with_call_sign += 1
                    ais_call_signs.add(call_sign)
                if vessel_name:
                    ais_names.add(vessel_name)
    port_call_signs = {
        voyage.call_sign_key for voyage in voyages if voyage.call_sign_key
    }
    port_names = {
        voyage.vessel_name_key for voyage in voyages if voyage.vessel_name_key
    }
    return {
        "ais_rows": ais_rows,
        "ais_rows_with_mmsi": rows_with_mmsi,
        "ais_unique_mmsi": len(mmsi_values),
        "ais_rows_with_call_sign": rows_with_call_sign,
        "ais_unique_call_signs": len(ais_call_signs),
        "port_mis_unique_call_signs": len(port_call_signs),
        "exact_call_sign_intersections": len(ais_call_signs & port_call_signs),
        "exact_normalized_name_intersections": len(ais_names & port_names),
        "strict_identity_linkable": bool(ais_call_signs & port_call_signs),
        "note": "Name-only matches are excluded from strict matching unless --allow-unique-name is set.",
    }


def load_crosswalk(path: Path | None) -> dict[str, str]:
    """Load a verified crosswalk containing ``mmsi`` and ``call_sign`` columns."""

    if path is None:
        return {}
    header = _read_header(path)
    missing = {"mmsi", "call_sign"} - set(header)
    if missing:
        raise VoyageMatchingError(
            f"Crosswalk is missing columns: {', '.join(sorted(missing))}"
        )
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            mmsi = normalize_identifier(row.get("mmsi"))
            call_sign = normalize_identifier(row.get("call_sign"))
            if not mmsi or not call_sign:
                raise VoyageMatchingError(
                    f"Blank crosswalk identifier at row {line_number}: {path}"
                )
            existing = result.get(mmsi)
            if existing and existing != call_sign:
                raise VoyageMatchingError(
                    f"Conflicting call signs for MMSI {mmsi}: {path}"
                )
            result[mmsi] = call_sign
    return result


def build_ais_identity_map(ais_files: Sequence[Path]) -> dict[str, Identity]:
    """Collect unique AIS static-data identities for position-report rows."""

    call_signs: dict[str, set[str]] = defaultdict(set)
    vessel_names: dict[str, set[str]] = defaultdict(set)
    for path in ais_files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                mmsi = normalize_identifier(row.get("mmsi") or row.get("MMSI"))
                if not mmsi:
                    continue
                call_sign = normalize_identifier(
                    row.get("call_sign") or row.get("CallSign")
                )
                vessel_name = normalize_identifier(
                    row.get("ship_name") or row.get("VesselName")
                )
                if call_sign:
                    call_signs[mmsi].add(call_sign)
                if vessel_name:
                    vessel_names[mmsi].add(vessel_name)
    result: dict[str, Identity] = {}
    for mmsi in call_signs.keys() | vessel_names.keys():
        calls = call_signs[mmsi]
        names = vessel_names[mmsi]
        result[mmsi] = Identity(
            call_sign_key=next(iter(calls)) if len(calls) == 1 else None,
            vessel_name_key=next(iter(names)) if len(names) == 1 else None,
            method="ais_call_sign" if len(calls) == 1 else None,
        )
    return result


def _identity_for_row(
    row: dict[str, str],
    identities: dict[str, Identity],
    crosswalk: dict[str, str],
) -> Identity:
    mmsi = normalize_identifier(row.get("mmsi") or row.get("MMSI"))
    if mmsi in crosswalk:
        return Identity(crosswalk[mmsi], None, "verified_crosswalk")
    direct_call = normalize_identifier(row.get("call_sign") or row.get("CallSign"))
    direct_name = normalize_identifier(row.get("ship_name") or row.get("VesselName"))
    stored = identities.get(mmsi, Identity(None, None, None))
    if direct_call or stored.call_sign_key:
        return Identity(
            direct_call or stored.call_sign_key,
            direct_name or stored.vessel_name_key,
            "ais_call_sign",
        )
    return Identity(None, direct_name or stored.vessel_name_key, None)


def _voyage_window(
    voyage: Voyage, lookback: timedelta, follow: timedelta
) -> tuple[datetime, datetime] | None:
    anchor_start = voyage.arrival or voyage.departure
    anchor_end = voyage.departure or voyage.arrival
    if anchor_start is None or anchor_end is None:
        return None
    return anchor_start - lookback, anchor_end + follow


def match_ais_to_voyages(
    ais_files: Sequence[Path],
    voyages: Sequence[Voyage],
    output: Path,
    *,
    crosswalk: dict[str, str] | None = None,
    allow_unique_name: bool = False,
    arrival_lookback_hours: float = 72.0,
    departure_follow_hours: float = 6.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Stream AIS rows to a CSV containing strict voyage-match annotations."""

    if output.exists() and not overwrite:
        raise VoyageMatchingError(f"Output exists (use --overwrite): {output}")
    headers = [_read_header(path) for path in ais_files]
    if any(header != headers[0] for header in headers[1:]):
        raise VoyageMatchingError("All AIS CSV files must have identical columns")
    output_header = headers[0] + [
        column for column in MATCH_COLUMNS if column not in headers[0]
    ]
    identities = build_ais_identity_map(ais_files)
    verified = crosswalk or {}
    by_call: dict[str, list[Voyage]] = defaultdict(list)
    by_name: dict[str, list[Voyage]] = defaultdict(list)
    name_to_calls: dict[str, set[str]] = defaultdict(set)
    for voyage in voyages:
        by_call[voyage.call_sign_key].append(voyage)
        if voyage.vessel_name_key:
            by_name[voyage.vessel_name_key].append(voyage)
            name_to_calls[voyage.vessel_name_key].add(voyage.call_sign_key)
    lookback = timedelta(hours=arrival_lookback_hours)
    follow = timedelta(hours=departure_follow_hours)
    counters: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            delete=False,
            dir=output.parent,
            suffix=".tmp",
        ) as temp:
            temp_path = Path(temp.name)
            writer = csv.DictWriter(
                temp, fieldnames=output_header, extrasaction="ignore"
            )
            writer.writeheader()
            for path, header in zip(ais_files, headers):
                time_column = _choose_column(header, AIS_TIME_COLUMNS, path)
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        annotated = _match_row(
                            row,
                            time_column,
                            identities,
                            verified,
                            by_call,
                            by_name,
                            name_to_calls,
                            allow_unique_name,
                            lookback,
                            follow,
                        )
                        counters[annotated["match_status"]] += 1
                        writer.writerow(annotated)
        os.replace(temp_path, output)
    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise
    return {
        "output": str(output.resolve()),
        "rows": sum(counters.values()),
        "statuses": dict(sorted(counters.items())),
        "matched_rows": counters["matched"],
    }


def _match_row(
    row: dict[str, str],
    time_column: str,
    identities: dict[str, Identity],
    crosswalk: dict[str, str],
    by_call: dict[str, list[Voyage]],
    by_name: dict[str, list[Voyage]],
    name_to_calls: dict[str, set[str]],
    allow_unique_name: bool,
    lookback: timedelta,
    follow: timedelta,
) -> dict[str, str]:
    result = dict(row)
    result.update({column: "" for column in MATCH_COLUMNS})
    raw_time = (row.get(time_column) or "").strip()
    if not raw_time:
        result["match_status"] = "missing_timestamp"
        return result
    try:
        timestamp = parse_timestamp(raw_time, naive_timezone=UTC)
    except ValueError:
        result["match_status"] = "invalid_timestamp"
        return result
    identity = _identity_for_row(row, identities, crosswalk)
    candidates: list[Voyage]
    method = identity.method or ""
    if identity.call_sign_key:
        candidates = by_call.get(identity.call_sign_key, [])
    elif (
        allow_unique_name
        and identity.vessel_name_key
        and len(name_to_calls.get(identity.vessel_name_key, set())) == 1
    ):
        candidates = by_name.get(identity.vessel_name_key, [])
        method = "unique_vessel_name"
    else:
        result["match_status"] = "identity_not_verified"
        return result
    if not candidates:
        result["match_status"] = "identity_not_in_port_mis"
        result["match_method"] = method
        return result
    temporal = []
    for voyage in candidates:
        window = _voyage_window(voyage, lookback, follow)
        if window and window[0] <= timestamp <= window[1]:
            temporal.append(voyage)
    if not temporal:
        result["match_status"] = "outside_voyage_window"
        result["match_method"] = method
        return result
    if len(temporal) > 1:
        result["match_status"] = "ambiguous_voyage"
        result["match_method"] = method
        return result
    voyage = temporal[0]
    result.update(
        {
            "voyage_id": voyage.voyage_id,
            "match_status": "matched",
            "match_method": method,
            "port_call_sign": voyage.call_sign,
            "port_vessel_name": voyage.vessel_name,
            "port_arrival_kst": _format_kst(voyage.arrival),
            "port_departure_kst": _format_kst(voyage.departure),
            "port_planned_departure_kst": _format_kst(voyage.planned_departure),
            "hours_to_arrival": (
                f"{(voyage.arrival - timestamp).total_seconds() / 3600:.3f}"
                if voyage.arrival
                else ""
            ),
            "previous_port": voyage.previous_port,
            "next_port": voyage.next_port,
            "destination_port": voyage.destination_port,
        }
    )
    return result


def _format_kst(value: datetime | None) -> str:
    return value.astimezone(KST).isoformat() if value else ""


def _write_json(path: Path, payload: dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise VoyageMatchingError(f"Report exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
        ) as temp:
            temp_path = Path(temp.name)
            json.dump(payload, temp, ensure_ascii=False, indent=2)
            temp.write("\n")
        os.replace(temp_path, path)
    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ais", type=Path, nargs="+", required=True, help="One or more AIS CSV files"
    )
    parser.add_argument("--portmis", type=Path, required=True, help="Port-MIS CSV")
    parser.add_argument(
        "--output", type=Path, default=Path("data/voyages/ais_portmis_matches.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/voyages/voyage_matching_report.json")
    )
    parser.add_argument(
        "--crosswalk", type=Path, help="Verified CSV with mmsi,call_sign columns"
    )
    parser.add_argument("--required-months", type=int, default=3)
    parser.add_argument("--arrival-lookback-hours", type=float, default=72.0)
    parser.add_argument("--departure-follow-hours", type=float, default=6.0)
    parser.add_argument("--allow-unique-name", action="store_true")
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--assess-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.required_months < 1:
        raise SystemExit("ERROR: --required-months must be at least 1")
    try:
        voyages = load_voyages(args.portmis)
        report = assess_coverage(args.ais, voyages, args.required_months)
        report["identity_coverage"] = assess_identity_coverage(args.ais, voyages)
        if not args.assess_only and (
            report["coverage"]["full"] or args.allow_partial_coverage  # type: ignore[index]
        ):
            report["matching"] = match_ais_to_voyages(
                args.ais,
                voyages,
                args.output,
                crosswalk=load_crosswalk(args.crosswalk),
                allow_unique_name=args.allow_unique_name,
                arrival_lookback_hours=args.arrival_lookback_hours,
                departure_follow_hours=args.departure_follow_hours,
                overwrite=args.overwrite,
            )
        elif not args.assess_only:
            report["matching"] = {
                "status": "stopped_insufficient_ais_coverage",
                "remedy": "Backfill the required AIS period, then rerun. Use --allow-partial-coverage only for diagnostics.",
            }
        _write_json(args.report, report, args.overwrite)
    except (OSError, csv.Error, VoyageMatchingError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.report.resolve()}")
    if (
        not args.assess_only
        and not report["coverage"]["full"]
        and not args.allow_partial_coverage
    ):  # type: ignore[index]
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
