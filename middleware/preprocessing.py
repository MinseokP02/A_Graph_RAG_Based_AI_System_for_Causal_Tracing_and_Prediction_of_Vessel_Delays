"""Stream CSV records into typed, entity-separated graph payloads."""

import csv
import math
from collections.abc import Iterator
from datetime import datetime
from itertools import islice
from typing import Any
from zoneinfo import ZoneInfo

from preprocess_port_calls import OUTPUT_COLUMNS

from .config import Config
from .graph_schema import NUMERIC_FIELDS, PORT_FIELDS, TIME_FIELDS, VESSEL_FIELDS
from .utils import Stats, key, name, text, vessel_key


def numeric(value: Any, stats: Stats, record: str, column: str) -> float | None:
    """Coerce a finite numeric value; preserve missing values as null."""
    value = text(value)
    if value is None:
        return None
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except ValueError:
        pass
    stats.issue("invalid_numeric", record, column)
    return None


def timestamp(
    value: Any, config: Config, stats: Stats, record: str, column: str
) -> datetime | None:
    """Parse ISO date/time and localize naive values with the configured zone."""
    value = text(value)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            zone = ZoneInfo(config.source_timezone)
            # Reject DST gaps and ambiguities rather than guess an instant.
            aware = parsed.replace(tzinfo=zone)
            from datetime import timezone

            back = aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
            if back != parsed or aware.utcoffset() != aware.replace(fold=1).utcoffset():
                raise ValueError("Ambiguous or nonexistent local time")
            parsed = aware
        return parsed
    except (ValueError, OverflowError):
        stats.issue("invalid_datetime", record, column)
        return None


def boolean(value: Any, stats: Stats, record: str, column: str) -> bool | None:
    """Parse Y/N while keeping absent and invalid values null."""
    value = name(value)
    if value is None:
        return None
    if value in ("Y", "N"):
        return value == "Y"
    stats.issue("invalid_boolean", record, column)
    return None


def derived_features(props: dict, threshold: float, stats: Stats, record: str) -> dict:
    """Compute minutes and cargo features without imputing missing timestamps."""
    from datetime import timezone

    def difference(end: datetime | None, start: datetime | None) -> float | None:
        """Subtract instants in UTC, including timezone/DST transitions."""
        return (
            (
                end.astimezone(timezone.utc) - start.astimezone(timezone.utc)
            ).total_seconds()
            / 60
            if end and start
            else None
        )

    stay = difference(props["departureTime"], props["arrivalTime"])
    if stay is not None and stay < 0:
        stats.issue("negative_port_stay", record, "portStayMinutes")
        stay = None
    deviation = difference(props["departureTime"], props["plannedDepartureTime"])
    delay = max(0.0, deviation) if deviation is not None else None
    return {
        "portStayMinutes": stay,
        "departureScheduleDeviationMinutes": deviation,
        "departureDelayMinutes": delay,
        "isDepartureDelayed": delay >= threshold if delay is not None else None,
        "cargoOperationTon": sum(
            props[k] or 0.0
            for k in ("unloadedCargoTon", "loadedCargoTon", "transshipmentTon")
        ),
    }


def port(code: Any, port_name: Any, scheme: str) -> dict | None:
    """Keep authority and route codes distinct until a verified crosswalk exists."""
    code, port_name = name(code), name(port_name)
    if not code and not port_name:
        return None
    identity = (
        key("port-code", scheme, code) if code else key("port-name", scheme, port_name)
    )
    return {"portKey": identity, "code": code, "name": port_name, "codeScheme": scheme}


def prepare_row(
    row: dict, config: Config, stats: Stats, source_row: int
) -> dict | None:
    """Map one declaration row to the requested four-label graph schema."""
    before = stats.counts["invalid_fields"]
    record = text(row.get("record_id"))
    identity = vessel_key(row.get("clsgn"), row.get("vsslNm"))
    if not record or not identity:
        if not identity:
            stats.issue(
                "missing_vessel_identifier", record or str(source_row), "clsgn/vsslNm"
            )
        if not record:
            stats.issue("missing_record_id", str(source_row), "record_id")
        stats.counts["invalid_rows"] += 1
        return None
    vessel = {prop: text(row.get(col)) for prop, col in VESSEL_FIELDS.items()}
    vessel.update(
        vesselKey=identity,
        callSign=name(row.get("clsgn")),
        name=name(row.get("vsslNm")),
    )
    vessel["grossTonnage"] = numeric(row.get("grtg"), stats, record, "grtg")
    pc = {
        "recordId": record,
        "entryYear": text(row.get("etryptYear")),
        "entryCount": text(row.get("etryptCo")),
        "purpose": text(row.get("etryptPurpsNm")),
        "cargoCode": text(row.get("ldadngFrghtClCd")),
    }
    pc.update(
        {
            prop: timestamp(row.get(col), config, stats, record, col)
            for prop, col in TIME_FIELDS.items()
        }
    )
    pc.update(
        {
            prop: numeric(row.get(col), stats, record, col)
            for prop, col in NUMERIC_FIELDS.items()
        }
    )
    pc["tugUsed"] = boolean(row.get("tugYn"), stats, record, "tugYn")
    pc["pilotageUsed"] = boolean(row.get("piltgYn"), stats, record, "piltgYn")
    pc.update(derived_features(pc, config.delay_threshold_minutes, stats, record))
    for feature in ("portStayMinutes", "departureDelayMinutes"):
        if pc[feature] is not None:
            stats.counts["available_" + feature] += 1
    pc.update(
        sourceFile=str(config.csv_path.resolve()),
        sourceRow=source_row,
        sourceTimezone=config.source_timezone,
        delayThresholdMinutes=config.delay_threshold_minutes,
    )
    for prop in TIME_FIELDS:
        pc[prop] = pc[prop].isoformat() if pc[prop] else None
    ports = {
        rel: p
        for rel, (code, title, scheme) in PORT_FIELDS.items()
        if (p := port(row.get(code), row.get(title), scheme)) is not None
    }
    current = ports.get("AT_PORT")
    facility_code = text(row.get("laidupFcltyCd"))
    facility_subcode = text(row.get("laidupFcltySubCd"))
    facility_name = name(row.get("laidupFcltyNm"))
    berth = None
    if current and (facility_code or facility_name):
        berth = {
            "berthKey": key(
                "berth",
                current["portKey"],
                "code" if facility_code else "name",
                facility_code or facility_name,
                facility_subcode or "",
            ),
            "code": facility_code,
            "subCode": facility_subcode,
            "name": facility_name,
        }
    elif facility_code or facility_name:
        stats.issue("missing_berth_port", record, "prtAgCd/prtAgNm")
    if stats.counts["invalid_fields"] > before:
        stats.counts["invalid_rows"] += 1
    stats.entities["vessels"].add(identity)
    stats.entities["port_calls"].add(record)
    stats.entities["ports"].update(p["portKey"] for p in ports.values())
    if berth:
        stats.entities["berths"].add(berth["berthKey"])
    return {"vessel": vessel, "portCall": pc, "ports": ports, "berth": berth}


def records(config: Config, stats: Stats, limit: int | None = None) -> Iterator[dict]:
    """Stream rows without pandas type inference or loading the full CSV."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    with config.csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = set(OUTPUT_COLUMNS) - set(headers)
        if missing or len(set(headers)) != len(headers):
            raise ValueError(
                "Missing or duplicate CSV columns: " + ", ".join(sorted(missing))
            )
        iterator = reader if limit is None else islice(reader, limit)
        for number, row in enumerate(iterator, 1):
            stats.counts["input_rows"] += 1
            if None in row or any(v is None for v in row.values()):
                stats.issue("malformed_csv", str(number), "row")
                stats.counts["invalid_rows"] += 1
                stats.counts["skipped_rows"] += 1
                continue
            record = prepare_row(row, config, stats, number)
            if record is None:
                stats.counts["skipped_rows"] += 1
            else:
                stats.counts["valid_rows"] += 1
                yield record
